#include "barrier.h"

// Generic queue test kernel that works with different queue types
kernel void generic_queue_copy_test(__global volatile barrier_t* b,
                                   __global volatile void* q,
                                   __global volatile int* input,
                                   __global volatile int* output,
                                   int num_elements)
{
    const unsigned int tid = (get_local_id(1)*get_local_size(0)) + get_local_id(0);
    volatile __local unsigned int group;
    volatile __local unsigned int groups;

    full_init(b, &group, &groups, tid, num_elements);
    SYNCTHREADS;
    
    if(group >= groups)
        return;
        
    unsigned int start = group * ((num_elements/groups));
    unsigned int end = start + ((num_elements/groups)+1);
    end = group == groups - 1 ? num_elements : end;
    
    volatile unsigned int item;
    
    for(int i = start + 1; i <= end; ++i) {
        if(tid == 0) {
            // Enqueue operation - conditional compilation based on queue type
            #ifdef USE_MS_QUEUE
                while(ms_enqueue((__global volatile ms_queue_t*)q, i)) {}
            #elif defined(USE_SFQ_QUEUE)
                while(my_enqueue_slot((__global volatile my_queue_t*)q, i)) {}
            #elif defined(USE_TZ_QUEUE)
                while(tz_enqueue((__global volatile tz_queue_t*)q, i)) {}
            #elif defined(USE_LCRQ_QUEUE)
                while(lcr_enqueue32((__global volatile lcrq32*)q, i)) {}
            #endif
            
            WAIT(&item);
            
            // Dequeue operation - conditional compilation based on queue type
            #ifdef USE_MS_QUEUE
                while(ms_dequeue((__global volatile ms_queue_t*)q, &item)) {}
            #elif defined(USE_SFQ_QUEUE)
                while(my_dequeue_slot((__global volatile my_queue_t*)q, &item)) {}
            #elif defined(USE_TZ_QUEUE)
                while(tz_dequeue((__global volatile tz_queue_t*)q, &item)) {}
            #elif defined(USE_LCRQ_QUEUE)
                while(lcr_dequeue32((__global volatile lcrq32*)q, &item)) {}
            #endif
        }
        
        SYNCTHREADS;
        
        if(tid == 0) {
            output[item - 1] = input[item - 1];
            WAIT(&item);
        }
        
        SYNCTHREADS;
    }
}