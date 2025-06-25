#include "barrier.h"

#ifndef MY_QUEUE_LENGTH
#define MY_QUEUE_LENGTH 4096
#define MY_QUEUE_FACTOR 12
/*#define MY_QUEUE_LENGTH 20*/
#endif
#ifndef WORK
#define WORK 100
#endif

#define MY_QUEUE_MASK (MY_QUEUE_LENGTH - 1)
#define MY_QUEUE_SMASK (UINT_MAX>>(MY_QUEUE_FACTOR - 1))
#define GET_TARGET(H, Q) ((((H & MY_QUEUE_MASK) % (MY_QUEUE_LENGTH/16))*16) + ((H & MY_QUEUE_MASK) / (MY_QUEUE_LENGTH/16)))
/*#define GET_TARGET(H, Q) (((H & MY_QUEUE_MASK)))*/

/*#define MY_QUEUE_LENGTH 10240*/
#define NULL_1 UINT_MAX
#define NULL_0 (UINT_MAX-1)

/*#define MAKE_PC_TEST(NAME, TYPE, ENQ, DEQ) \*/
/*kernel void NAME(__global volatile barrier_t* b,\*/
                            /*__global volatile TYPE * q,\*/
                            /*__global volatile int * input,\*/
                            /*__global volatile int * output,\*/
                            /*int num_elements)\*/
/*{\*/
    /*const unsigned int tid = (get_local_id(1)*get_local_size(0)) + get_local_id(0);\*/
    /*const unsigned int group = get_group_id(0) + (get_group_id(1) * get_num_groups(0));\*/
    /*const unsigned int groups = get_num_groups(0)*get_num_groups(1);\*/
    /*volatile __local ms_private_t prvt;\*/
    /*volatile __local unsigned test;\*/
    /*SYNCTHREADS;\*/
    /*if(tid == 0){\*/
        /*prvt.node = 2+ group;\*/
        /*prvt.value = 1+ group;\*/
        /*test = VOLATILE_READ(b->lid1);\*/
    /*}\*/
    /*SYNCTHREADS;\*/
    /*if(test != 0)\*/
        /*return;\*/
    /*if(tid ==0)\*/
        /*VOLATILE_INC(b->free);\*/
    /*volatile unsigned int item;\*/
    /*for(int i=1; i<num_elements; ++i){\*/
        /*if(tid == 0){\*/
            /*if(group == 0){\*/
                /*ENQ\*/
            /*}else{\*/
                /*DEQ\*/
                /*output[item-1] = input[item-1];\*/
            /*}\*/
            /*WAIT(&item);\*/
        /*}\*/
        /*SYNCTHREADS;\*/
    /*}\*/
    /*VOLATILE_CAS(b->lid1, 0, VOLATILE_READ(b->free));\*/
    /*if(group == 0) VOLATILE_WRITE(b->lid0, 1);\*/
    /*SYNCTHREADS;\*/
/*}*/

// inline void WAIT(volatile uint32_t * bah){
//     for(int i=0; i<WORK; i++){
//         *bah *= i + i; 
//     }
// }

typedef struct my_queue
{
    volatile uint32_t head;
    volatile uint32_t tail;
    volatile uint32_t vnull;
    volatile uint32_t done;
    union{
    volatile uint32_t items[MY_QUEUE_LENGTH];
    volatile uint32_t nodes[MY_QUEUE_LENGTH];
    };
    volatile uint32_t slots[MY_QUEUE_LENGTH];
} my_queue_t;
// typedef my_queue_t tz_queue_t;


inline int my_enqueue(__global volatile my_queue_t * q,
                unsigned int item,
                __global volatile unsigned int * done){
    unsigned int target = VOLATILE_INC(q->head) % MY_QUEUE_LENGTH;
    unsigned int fail=0;
    while(VOLATILE_CAS(q->items[target], 0, item) != 0 && TEST_FAILSAFE){
        /*if(VREAD(q->done) != 0)*/
            /*return 1;*/
        unsigned int worker=0;
        WAIT(&worker);
        fail++;
    }
    if(! (TEST_FAILSAFE)){
        VWRITE(q->done,0);
        return 1;
    }
    /*VOLATILE_XCHG(q->items[target], item);*/
    return 0;
}

inline int my_dequeue(__global volatile my_queue_t * q, volatile unsigned int * p, __global volatile unsigned int * done)
{
    unsigned int target = VOLATILE_INC(q->tail) % MY_QUEUE_LENGTH;
    unsigned int fail=0;
    while((*p = VOLATILE_XCHG(q->items[target], 0)) == 0 && TEST_FAILSAFE){
        if(VREAD(*done))
            return 1;
        unsigned int worker=0;
        WAIT(&worker);
        fail++;
    }
    if(! (TEST_FAILSAFE)){
        VWRITE(q->done,0);
        return 1;
    }
    return 0;
}


inline unsigned my_get_waiting(__global volatile my_queue_t * q){
    unsigned head = VOLATILE_READ(q->head);
    unsigned tail = VOLATILE_READ(q->tail);
    return head >= tail ? (unsigned) (head - tail) : 0;
}
void my_set_done(__global volatile my_queue_t * q){
    const unsigned int tail = VOLATILE_READ(q->tail);
    VOLATILE_WRITE(q->done, tail-1);
}

inline int my_enqueue_slot(__global volatile my_queue_t * q,
                unsigned int item){
    const unsigned int tail = VOLATILE_INC(q->tail);
    const unsigned int pass = ( tail >> MY_QUEUE_FACTOR) << 1;
    const uint32_t target = GET_TARGET(tail, q);
#ifndef NOFAILSAFE
    unsigned int fail=0;
#endif
    unsigned slot = q->slots[target];
    while(slot != pass){
        unsigned qdone = VOLATILE_READ(q->done);
        /*if(qdone != 0 && tail > qdone)*/
            /*return 1;*/
#ifndef NOFAILSAFE
        fail++;
        if(! (TEST_FAILSAFE)){
            /*VWRITE(q->done,tail);*/
            /*my_set_done(q);*/
            VWRITE(q->done,1);
            return 2;
        }
#endif
        slot = VOLATILE_READ(q->slots[target]);
    }
    /*printf("enqueued %u target=%u\n pass=%u", item, target, pass);*/
    VWRITE(q->items[target],item);
    /*VOLATILE_INC(q->slots[target]);*/
    VOLATILE_WRITE(q->slots[target], (pass+1) & MY_QUEUE_SMASK);
    /*VOLATILE_XCHG(q->items[target], item);*/
    return 0;
}

inline int my_dequeue_slot(__global volatile my_queue_t * q, volatile unsigned int * p)
{
    const unsigned int head = VOLATILE_INC(q->head);
    const unsigned int pass = ((head >> MY_QUEUE_FACTOR)<<1)+1;
    const uint32_t target = GET_TARGET(head, q);
#ifndef NOFAILSAFE
    unsigned int fail=0;
#endif
    /*printf("dequeueing target=%u pass=%u total=%u\n", target, pass, VOLATILE_INC(q->vnull));*/
    /*if(head > 1)*/
        /*return 1;*/
    unsigned slot = q->slots[target];
    while(slot != pass){
        volatile unsigned qdone = VREAD(q->done);
        /*mem_fence(CLK_LOCAL_MEM_FENCE);*/
        if(qdone != 0 && head > qdone)
            return 1;
        /*qdone++;*/
        /*if(head - VOLATILE_READ(q->tail) + 1== VOLATILE_READ(q->vnull)){*/
            /*[>my_set_done(q);<]*/
            /*VOLATILE_WRITE(q->done,VOLATILE_READ(q->tail)-1);*/
            /*return 3;*/
        /*}*/
#ifndef NOFAILSAFE
        fail++;
        if(! (TEST_FAILSAFE)){
            VWRITE(q->done,2);
            return 2;
        }
#endif
        slot = VOLATILE_READ(q->slots[target]);
    }
        /*VOLATILE_INC(q->vnull);*/

    *p = VREAD(q->items[target]);
    /*printf("dequeued %u target=%u pass=%u total=%u\n", *p, target, pass, VOLATILE_INC(q->vnull));*/
    /*VOLATILE_INC(q->slots[target]);*/
    VOLATILE_WRITE(q->slots[target], (pass+1) & MY_QUEUE_SMASK);
    return 0;
}

inline int my_enqueue_nb_slot(__global volatile my_queue_t * q,
        unsigned int item){
    volatile uint32_t tail = VOLATILE_READ(q->tail);
    uint32_t target;
    uint32_t pass;
    for(;;){
        target = GET_TARGET(tail,q); // tail % q->size (power of 2, might as well use &)
        pass = ((tail >> MY_QUEUE_FACTOR) << 1);
        /*const uint32_t pass = (tail / q->size)*2;*/ //for non power of 2
        /*fprintf(stderr, "enq pass=%u\n", pass);*/
        if(VOLATILE_READ(q->slots[target]) != pass)
            return 1;//queue is full, and may have waiting threads
        uint32_t ltail = tail;
        if((ltail = VOLATILE_CAS(q->tail, tail, tail+1)) == tail)
            break;
        tail = ltail;
    }
  /*fprintf(stderr,"%d: inserting %u\n", omp_get_thread_num(), item);*/
    VWRITE(q->items[target], item);
    /*VOLATILE_ADD(q->slots[target], q->adder);*/
    VOLATILE_WRITE(q->slots[target], (pass+1) & MY_QUEUE_SMASK);
    return 0;
}

int my_dequeue_nb_slot(__global volatile my_queue_t * q, volatile unsigned int * p)
{
    volatile uint32_t head = VOLATILE_READ(q->head);
    uint32_t target;
    uint32_t pass;
    for(;;){
        uint32_t lhead = head;
        target = GET_TARGET(head,q);
        pass = (((head >> MY_QUEUE_FACTOR)<<1) + 1);
        /*fprintf(stderr, "deq pass=%u\n", pass);*/
        /*const uint32_t pass = ((head / q->size)*2)+1;*/
        if(VOLATILE_READ(q->slots[target]) != pass)
            return 1;//queue is empty, and may have waiting threads
        if((lhead = VOLATILE_CAS(q->head, head, head+1)) == head)
            break;
            head = lhead;
    }
  /*fprintf(stderr,"%d: removing %u\n", omp_get_thread_num(), q->items[target]);*/
    *p = VREAD(q->items[target]);
    VOLATILE_WRITE(q->slots[target], (pass+1) & MY_QUEUE_SMASK);
    return 0;
}

// #include "queue_tz.cl"
// #include "queue_ms.cl"
/*#include "queue_lcrq.cl"*/
/*#include "queue_lcrq16.cl"*/
