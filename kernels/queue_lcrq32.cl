// #ifndef __OPENCL__
// #define __OPENCL__
// #endif

/*#if defined(cl_khr_int64_base_atomics) //if not, can't use this implementation at all*/

#include "lcrqueue32.h"

#define EMPTY UINT_MAX
#define CLOSED (EMPTY-1)
#undef GET_TARGET
#define GET_TARGET(H, Q) (H % Q->size)

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#define VOLATILE_CAS64(X,Y,Z) atom_cmpxchg(&(X),Y,Z)


void set_hazard(volatile __global crq32* q){
    VOLATILE_ADD(q->hazard[WARP_ID], 1);
}
void unset_hazard(volatile __global crq32* q){
    VOLATILE_SUB(q->hazard[WARP_ID], 1);
}

void init_cr_32_queue(volatile __global crq32* q, uint32_t size){
    q->head = 0; 
    q->tail.combined = 0; 
    q->next = UINT_MAX;
    q->size = CRQ_LEN;
    // Node32 stamp = {.id = {.safe = 1},
    //               .val = EMPTY};a
    Node32 stamp;
    stamp.val = EMPTY;
    SET_SAFE(stamp.id, 1);
    for(int i=0; i<CRQ_LEN; ++i){
        // stamp.id.idx = i;
        SET_IDX(stamp.id, i);
        q->ring[i].combined = stamp.combined;
    }

    for (int g = 0; g < GROUPS; ++g)
        q->hazard[g] = 0;  
}
uint32_t new_cr_32_queue(volatile __global lcrq32 * lq, uint32_t size){
    /*uint32_t count = VOLATILE_INC(new_count);*/
    /*fprintf(stderr,"allocating another crq: %u\n", count);*/
    uint32_t newcrq = 0;
    /*uint32_t fail = 0;*/
    while(1){
        newcrq = VOLATILE_INC(lq->base_spin) % 1500;
        set_hazard(&lq->base[newcrq]);
        uint32_t count=0;
        for(uint32_t i=0; i<GROUPS; i++){
            count += lq->base[newcrq].hazard[i];
        }
        if(count == 1){//SUCCESS!
            break;
        }
        /*if(fail++ == 100){*/
            /*fprintf(stderr, "no crqs...\n", fail);*/
            /*exit(1);*/
        /*}*/

        unset_hazard(&lq->base[newcrq]);
    }
    /*fprintf(stderr,"allocated another crq: %u\n", count);*/
    init_cr_32_queue(&lq->base[newcrq],size);
    return newcrq;
}

void fixState32(volatile __global crq32 * q){
    uint32_t h, t;
    while(1){
        h = VOLATILE_READ(q->head);
        t = VOLATILE_READ(q->tail.combined);

        if(VREAD(q->tail.combined) != t)
            continue;// inconsistent, repeat

        if(h<t)
            return; //nothing to do

        if(VOLATILE_CAS(q->tail.combined,t,h) == t)
            return; //success
    }
}

uint32_t cr_dequeue32(volatile __global crq32 *q, volatile uint32_t *val){
    uint32_t h;
    uint32_t target;
    volatile __global Node32 * node;
    Id32 closed_t;//boolean
    Node32 current;
    Node32 replacement;
    replacement.val = EMPTY;
    
    while(1){
        h = VOLATILE_INC(q->head);
        target = GET_TARGET(h,q);
        /*target %= CRQ_LEN;*/
        node = &(q->ring[target]);
        while(1){
            current.combined = node->combined;

            if(GET_IDX(current.id) > h) break;//go to end of outer while loop
            if(current.val != EMPTY){
                if(GET_IDX(current.id) == h){//try dequeue
                    SET_SAFE(replacement.id,GET_SAFE(current.id));
                    SET_IDX(replacement.id, h+CRQ_LEN);
                    if(VOLATILE_CAS64(node->combined,
                                        current.combined,
                                        replacement.combined) == current.combined){
                        mem_fence(CLK_GLOBAL_MEM_FENCE); 
                        val[0] = current.val;
                        return 0;
                    }
                }else{ // mark node unsafe to prevent enqueue
                    replacement.combined = current.combined;
                    SET_SAFE(replacement.id, 0);
                    current.id.combined = (current.id.combined & 0x80000000) | (h & 0x7FFFFFFF);
                    if(VOLATILE_CAS64(node->combined,
                                        current.combined,
                                        replacement.combined) == current.combined){
                        break;//go to end of outer while loop
                    }
                }
            }else{ // idx <= h and val is EMPTY, try empty transition
                SET_SAFE(replacement.id, GET_SAFE(current.id));
                SET_IDX(replacement.id,  h+CRQ_LEN);
                if(VOLATILE_CAS64(node->combined,
                            current.combined,
                            replacement.combined) == current.combined){
                    break;//go to end of outer while loop
                }
            }
        }
        //dequeue failed, test empty
        closed_t.combined = q->tail.combined;
        if(GET_T(closed_t) <= h+1){
            fixState32(q);
            return EMPTY;
        }
    }
}

uint32_t cr_dequeue32_spinopt(volatile __global crq32 *q, uint32_t *val){
    uint32_t h;
    uint32_t target;
    volatile __global Node32 * node;
    Id32 closed_t;//boolean
    Node32 current;
    Node32 replacement;
    replacement.val = EMPTY;
    
    while(1){
        h = VOLATILE_INC(q->head);
        target = GET_TARGET(h,q);
        /*target %= CRQ_LEN;*/
        node = &(q->ring[target]);
        while(1){
            current.combined = node->combined;

            if(GET_IDX(current.id) > h) break;//go to end of outer while loop
            if(current.val != EMPTY){
                if(GET_IDX(current.id) == h){//try dequeue
                    SET_SAFE(replacement.id, GET_SAFE(current.id));
                    SET_IDX(replacement.id, h+CRQ_LEN );
                    if(VOLATILE_CAS64(node->combined,
                                        current.combined,
                                        replacement.combined) == current.combined){
                        val[0] = current.val;
                        return 0;
                    }
                }else{ // mark node unsafe to prevent enqueue
                    replacement.combined = current.combined;
                    SET_SAFE(replacement.id, 0);
                    current.id.combined = (current.id.combined & 0x80000000) | (h & 0x7FFFFFFF);
                    if(VOLATILE_CAS64(node->combined,
                                        current.combined,
                                        replacement.combined) == current.combined){
                        break;//go to end of outer while loop
                    }
                }
            }else{ // idx <= h and val is EMPTY, try empty transition
                SET_SAFE(replacement.id, GET_SAFE(current.id));
                SET_IDX(replacement.id, h+CRQ_LEN);
                closed_t.combined = q->tail.combined;
                uint32_t fail = 0;
                if(GET_T(closed_t) >= h+1){
                    while(fail++ < 500){
                        if((current.val = VOLATILE_READ(node->val)) != EMPTY){
                            /*fprintf(stderr,"alloc avoided!\n");*/
                            /*goto val_ready;*///no gotos in opencl, joy
                            continue;
                        }
                        target %= fail;
                        /*usleep(1);*/
                    }
                }
                /*fprintf(stderr,"now destroying performance... t: %lu h: %lu fail: %lu\n", GET_T(closed_t), h, fail);*/
                if(VOLATILE_CAS64(node->combined,
                            current.combined,
                            replacement.combined) == current.combined){
                    break;//go to end of outer while loop
                }
            }
        }
        //dequeue failed, test empty
        closed_t.combined = q->tail.combined;
        if(GET_T(closed_t) <= h+1){
            fixState32(q);
            return EMPTY;
        }
    }
}
uint32_t cr_enqueue32(volatile __global crq32 *q, uint32_t arg){
    uint32_t h;
    uint32_t target;
    volatile __global Node32 * node;
    Id32 closed_t;//boolean
    Node32 current;
    Node32 replacement;
    replacement.val = EMPTY;
    uint32_t fail=0;
    while(1){
        closed_t.combined = VOLATILE_INC(q->tail.combined);
        if(GET_CLOSED(closed_t)){
            /*fprintf(stderr,"closed:1\n");*/
            return CLOSED;
        }
        target = GET_TARGET(GET_T(closed_t),q);
        /*target = GET_T(closed_t);*/
        /*target %= CRQ_LEN;*/
        node = &(q->ring[target]);
        current.combined = node->combined;
        if(current.val == EMPTY){//attempt enqueue
            /*fprintf(stderr, "val: %llu\tt=%llu idx: %llu safe: %llu target: %llu", arg, GET_T(closed_t), GET_IDX(current.id), GET_SAFE(current.id), target);*/
            SET_IDX(replacement.id,  GET_T(closed_t));
            SET_SAFE(replacement.id, 1);
            replacement.val = arg;
            if((GET_IDX(current.id) <= GET_T(closed_t))){
                /*fprintf(stderr, "1 ");*/
                if(GET_SAFE(current.id) == 1 || q->head <= GET_T(closed_t)){
                    /*fprintf(stderr, "2 ");*/
                    if(VOLATILE_CAS64(node->combined, 
                                        current.combined, 
                                        replacement.combined) == current.combined){
                        /*fprintf(stderr, "3\n");*/
                        mem_fence(CLK_GLOBAL_MEM_FENCE);
                        return 0;
                    }
                }
            }
            /*fprintf(stderr, "\n");*/
        }
        h = VREAD(q->head);
        if(GET_T(closed_t) - h >= CRQ_LEN || (fail++ > 10000)){//starving is a check of fail
            //TODO may need an actual test and set here, not sure
            SET_CLOSED(q->tail, 1);
            /*fprintf(stderr,"closed:2 t: %llu h:%llu len: %llu fail: %lu\n",GET_T(closed_t), h, CRQ_LEN, fail);*/
            return CLOSED;
        }
    }
}


int lcr_dequeue32(volatile __global lcrq32 * q, volatile  uint32_t *val){
    uint32_t cr;
    uint32_t v;
    /*printf("entering lcr_dequeue\n");*/
    while(1){
        while(1){
            cr = VOLATILE_READ(q->head);
            set_hazard(&q->base[cr]);
            if(cr == VOLATILE_READ(q->head))
                break;
            unset_hazard(&q->base[cr]);
        }
        uint32_t status = cr_dequeue32(&q->base[cr], &v);
        if(status != EMPTY){
            *val = (uint32_t)v;
            unset_hazard(&q->base[cr]);
            return 0;
        }
        if(q->base[cr].next == UINT_MAX){
            unset_hazard(&q->base[cr]);
            return 1;//would be empty if empty would fit
        }
        VOLATILE_CAS(q->head, cr, q->base[cr].next);
        unset_hazard(&q->base[cr]);
    }
    /*printf("exiting lcr_dequeue\n");*/
}
int lcr_dequeue32_spinopt(volatile __global lcrq32 * q,volatile uint32_t *val){
    uint32_t cr;
    uint32_t v;
    /*printf("entering lcr_dequeue\n");*/
    while(1){
        while(1){
            cr = VOLATILE_READ(q->head);
            set_hazard(&q->base[cr]);
            if(cr == VOLATILE_READ(q->head))
                break;
            unset_hazard(&q->base[cr]);
        }
        uint32_t status = cr_dequeue32_spinopt(&q->base[cr], &v);
        if(status != EMPTY){
            *val = (uint32_t)v;
            unset_hazard(&q->base[cr]);
            return 0;
        }
        if(q->base[cr].next == UINT_MAX){
            unset_hazard(&q->base[cr]);
            return 1;//would be empty if empty would fit
        }
        VOLATILE_CAS(q->head, cr, q->base[cr].next);
        unset_hazard(&q->base[cr]);
    }
    /*printf("exiting lcr_dequeue\n");*/
}

int lcr_enqueue32(volatile __global lcrq32 *q, uint32_t val){
    uint32_t cur_crq = 0, newcrq;
    /*printf("entering lcr_enqueue\n");*/
    while(1){
        while(1){
            cur_crq = VOLATILE_READ(q->tail);
            set_hazard(&q->base[cur_crq]);
            if(cur_crq == VOLATILE_READ(q->tail))
                break;
            unset_hazard(&q->base[cur_crq]);
        }

        if(q->base[cur_crq].next != UINT_MAX){
            VOLATILE_CAS(q->tail, cur_crq, q->base[cur_crq].next);
            unset_hazard(&q->base[cur_crq]);
            continue;
        }
        if(cr_enqueue32(&q->base[cur_crq], val) != CLOSED){
            unset_hazard(&q->base[cur_crq]);
            return 0;
        }
        newcrq = new_cr_32_queue(q,CRQ_LEN);//sets hazard
        /*set_hazard(&q->base[newcrq]);*/

        cr_enqueue32(&q->base[newcrq], val);
        if(VOLATILE_CAS(q->base[cur_crq].next, UINT_MAX, newcrq) == UINT_MAX){
            VOLATILE_CAS(q->tail, cur_crq, newcrq);
            unset_hazard(&q->base[cur_crq]);
            unset_hazard(&q->base[newcrq]);

            return 0;
        }
        unset_hazard(&q->base[cur_crq]);
        unset_hazard(&q->base[newcrq]);
        /*free(newcrq);*/
    }
    /*printf("exiting lcr_enqueue\n");*/
}

