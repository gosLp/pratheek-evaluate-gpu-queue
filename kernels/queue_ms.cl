#include "barrier.h"

typedef union {
  struct {
    volatile unsigned short count;
    volatile unsigned short ptr;
  } sep;
  volatile unsigned con;
}ms_pointer_t;

typedef struct {
  unsigned value;
  ms_pointer_t next;
  unsigned int free;
} ms_node_t;

typedef struct ms_queue {
  ms_pointer_t head;
  //unsigned foo1[31];
  ms_pointer_t tail;
  //unsigned foo2[31];
  ms_node_t nodes[MY_QUEUE_LENGTH+1];
  unsigned hazard1[1500];
  unsigned hazard2[1500];

  unsigned base_spin;
} ms_queue_t;

#define FREE_FALSE 1
#define FREE_TRUE 0
inline void ms_set_hazard(volatile __global ms_queue_t* q, uint32_t node){
    VOLATILE_WRITE(q->hazard1[WARP_ID], node);
}
inline void unms_set_hazard(volatile __global ms_queue_t* q){
    VOLATILE_WRITE(q->hazard1[WARP_ID], UINT_MAX);
}
inline void ms_set_hazard2(volatile __global ms_queue_t* q, uint32_t node){
    VOLATILE_WRITE(q->hazard2[WARP_ID], node);
}
inline void unms_set_hazard2(volatile __global ms_queue_t* q){
    VOLATILE_WRITE(q->hazard2[WARP_ID], UINT_MAX);
}

inline unsigned
new_node(volatile __global ms_queue_t * q)
{
    uint32_t fail = 0;
    uint32_t count=0;
    ms_pointer_t ptr = {.sep.count = 0};
    unsigned new_node;
    /*printf("getting new node\n");*/
    do{
        count = 0;
        unms_set_hazard(q);
        new_node = VOLATILE_INC(q->base_spin) % min((MY_QUEUE_LENGTH-1),(1<<16) - 1);
        if(new_node < 2)
            continue;
        ms_set_hazard(q, new_node);
        /*printf( "node %u free is %u\n", new_node, q->nodes[new_node].free);*/
        if(VOLATILE_READ(q->nodes[new_node].free) == FREE_FALSE){
            unms_set_hazard(q);
            continue;
        }
        VOLATILE_WRITE(q->nodes[new_node].free, FREE_FALSE);
        for(uint32_t i=0; i<GROUPS; i++){
            count += q->hazard1[i] == new_node ? 1 : 0;
            count += q->hazard2[i] == new_node ? 1 : 0;
        }
        /*if(fail++ == 100){*/
            /*fprintf(stderr, "no nodes...\n");*/
            /*exit(1);*/
        /*}*/
    }while(count != 1);
    /*printf( "allocating node %d\n", new_node);*/
    ptr.sep.ptr = (unsigned short)new_node;
    return ptr.con;
}

#define NULL 0
#define FALSE 0
#define MS_FALSE 0

inline unsigned cas(volatile __global uint32_t *X, uint32_t Y, uint32_t Z){
    return (atomic_cmpxchg(X,Y,Z) == Y);
}

inline unsigned MAKE_LONG(unsigned short node, unsigned short count){
    //   ((hi)>>sizeof(unsigned short))+(lo)

    ms_pointer_t set = {.sep.count = count,
        .sep.ptr = node
    };
    return set.con;
}

inline int
ms_enqueue(__global volatile ms_queue_t * smp, unsigned val)
{
  unsigned success;
  unsigned node;
  ms_pointer_t tail;
  ms_pointer_t next;

  next.con = new_node(smp);
  node = next.sep.ptr;
  VOLATILE_WRITE(smp->nodes[node].value,val);
  next.con = VOLATILE_READ(smp->nodes[node].next.con);
  next.sep.ptr = NULL;
  VOLATILE_WRITE(smp->nodes[node].next.con,next.con);
  /*fprintf(stderr,"%d: inserting %u\n", pid, val);*/

  for (success = FALSE; success == FALSE; ) {
    tail.con = VOLATILE_READ(smp->tail.con);
    next.con = VOLATILE_READ(smp->nodes[tail.sep.ptr].next.con);
    ms_set_hazard2(smp, tail.sep.ptr);
    if (tail.con == VREAD(smp->tail.con)) {
      if (next.sep.ptr == NULL) {
	success = cas(&smp->nodes[tail.sep.ptr].next.con,
		      next.con,
		      MAKE_LONG(node, next.sep.count+1));
      }
      if (success == FALSE) {
	cas(&smp->tail.con,
	    tail.con,
	    MAKE_LONG(smp->nodes[tail.sep.ptr].next.sep.ptr,
		      tail.sep.count+1));
      }
    }
  }
  cas(&smp->tail.con,
      tail.con,
      MAKE_LONG(node, tail.sep.count+1));
  unms_set_hazard2(smp);
  unms_set_hazard(smp);
  return 0;
}
inline unsigned
ms_dequeue(__global volatile ms_queue_t * smp, volatile unsigned *val)
{
    unsigned value;
    unsigned success;
    ms_pointer_t head;
    ms_pointer_t tail;
    ms_pointer_t next;

    do{
        for (success = FALSE; success == FALSE; ) {
            head.con = VOLATILE_READ(smp->head.con);
            tail.con = VOLATILE_READ(smp->tail.con);
            next.con = VOLATILE_READ(smp->nodes[head.sep.ptr].next.con);
            ms_set_hazard(smp, head.sep.ptr);
            ms_set_hazard2(smp, next.sep.ptr);
            if (VREAD(smp->head.con) == head.con) {
                if (head.sep.ptr == tail.sep.ptr) {
                    if (next.sep.ptr == NULL) {
                        unms_set_hazard(smp);
                        unms_set_hazard2(smp);
                        return 1;
                    }
                    cas(&smp->tail.con,
                            tail.con,
                            MAKE_LONG(next.sep.ptr, tail.sep.count+1));
                    /*WAIT();*/
                } else {
                    value = VOLATILE_READ(smp->nodes[next.sep.ptr].value);
                    success = cas(&smp->head.con,
                            head.con,
                            MAKE_LONG(next.sep.ptr, head.sep.count+1));
                    if (success == FALSE) {
                        /*WAIT();*/
                    }
                }
            }
        }
        /*fprintf(stderr,"%d: getting %u\n", pid, value);*/
    }while(value == 0);
    VOLATILE_WRITE(smp->nodes[head.sep.ptr].free, FREE_TRUE);
    unms_set_hazard(smp);
    unms_set_hazard2(smp);
    *val = value;
    return 0;
}



kernel void ms_queue_copy_test(__global volatile barrier_t* b,
                            __global volatile ms_queue_t * q,
                            __global volatile int * input,
                            __global volatile int * output,
                            int num_elements)
{
    const unsigned int tid = (get_local_id(1)*get_local_size(0)) + get_local_id(0);
    const unsigned int lparts = (get_local_size(0)*get_local_size(1));
    /*const unsigned int group = get_group_id(0) + (get_group_id(1) * get_num_groups(0));*/
    /*const unsigned int groups = get_num_groups(0)*get_num_groups(1);*/
    volatile __local unsigned int group;
    volatile __local unsigned int groups;

    /*SYNCTHREADS;*/
    /*clean_init_barr(b, &group, tid);*/
    /*SYNCTHREADS;*/
    /*clean_prime_barr(b, group, &groups, tid, num_elements);*/
    full_init(b, &group, &groups, tid, num_elements);
    SYNCTHREADS;
    if(group >= groups)
        return;
    unsigned int start = group * ((num_elements/groups));
    unsigned int end = start + ((num_elements/groups)+1);
    /*end = end > num_elements ? num_elements : end;*/
    end = group == groups - 1 ? num_elements : end;
    volatile unsigned int item;
    for(int i=start+1; i<end; ++i){
        if(tid == 0){
            /*if(my_enqueue(q, i, &(b->lid2)))*/
            while(ms_enqueue(q, i)){}
                WAIT(&item);
            /*if(my_dequeue(q, &item, &(b->lid2)))*/
            while(ms_dequeue(q, &item)){}
        }
        SYNCTHREADS;
        if(tid == 0){
            output[item-1] = input[item-1];
            WAIT(&item);
        }
    }
    SYNCTHREADS;
}
