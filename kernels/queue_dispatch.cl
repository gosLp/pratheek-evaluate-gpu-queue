// kernels/queue_dispatch.cl - Fixed version
#include "queue_ms.cl"
#include "queue_sfq.cl"
#include "queue_tz.cl"
// #include "queue_lcrq32.cl"  // Commented out for now due to complexity

// Include the generic test kernel
#include "queue_test_generic.cl"

// #include "lcrqueue32.h"  // Commented out for now

// Barrier initialization kernel - works for all queue types
#include "barrier.h"

kernel void barrier_init(__global volatile barrier_t *b, 
                         uint groups_x, uint groups_y)
{
    if (get_global_id(0) == 0 && get_global_id(1) == 0) {
        // Initialize barrier fields that full_init() expects
        b->participants = 0;
        b->delay = DELAY;
        b->leader = 0;
        b->lid0 = 0;
        b->lid1 = 0;
        b->lid2 = 0;
        b->lock = 0;
        
        b->total_threads = groups_x * groups_y;
        b->total_groups = groups_x * groups_y;
        b->present = 0;
        
        b->goal = 0;
        b->free = 0;
        b->init = 0;
        
        b->even = 0;
        b->odd = 0;
    }
}

// Debug kernel for MS queue specifically
kernel void debug_ms_queue(__global volatile barrier_t* b,
                          __global volatile ms_queue_t* q,
                          __global volatile uint32_t* debug_info,
                          __global volatile uint32_t* metrics)
{
    const unsigned int tid = get_global_id(0);
    
    // Initialize debug info array
    if (tid == 0) {
        for (int i = 0; i < 100; i++) {
            debug_info[i] = 0;
        }
        
        // Report initial queue state
        debug_info[0] = q->head.con;  // head pointer
        debug_info[1] = q->tail.con;  // tail pointer
        debug_info[2] = q->base_spin; // base_spin counter
        
        // Check dummy node state
        debug_info[3] = q->nodes[1].free;    // dummy node free status
        debug_info[4] = q->nodes[1].value;   // dummy node value
        debug_info[5] = q->nodes[1].next.con; // dummy node next pointer
        
        // Count free nodes
        uint32_t free_count = 0;
        for (int i = 2; i < MY_QUEUE_LENGTH; i++) {
            if (q->nodes[i].free == FREE_TRUE) {
                free_count++;
            }
        }
        debug_info[6] = free_count;
        debug_info[99] = 9999; // completion marker
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    // Simple multi-threaded test with limited threads
    if (tid < 4) { // Only first 4 threads
        uint32_t ops_completed = 0;
        uint32_t failures = 0;
        
        for (int i = 0; i < 3; i++) { // Only 3 operations per thread
            if (tid % 2 == 0) {
                // Even threads enqueue
                int result = ms_enqueue(q, tid * 100 + i + 1); // Non-zero values
                if (result == 0) {
                    ops_completed++;
                } else {
                    failures++;
                }
            } else {
                // Odd threads dequeue
                volatile unsigned val;
                int result = ms_dequeue(q, &val);
                if (result == 0) {
                    ops_completed++;
                } else {
                    failures++;
                }
            }
            
            // Add delay between operations
            for (int delay = 0; delay < 1000; delay++) {
                volatile int dummy = delay * 2;
            }
        }
        
        // Store per-thread results
        if (tid < 4) {
            metrics[tid * 3 + 0] = ops_completed;
            metrics[tid * 3 + 1] = failures;
            metrics[tid * 3 + 2] = tid; // thread id for verification
        }
    }
}

// Simplified test kernel for basic functionality
kernel void simple_queue_test(__global volatile barrier_t* b,
                             __global volatile void* q,
                             __global volatile uint32_t* metrics,
                             __global volatile uint64_t* timing_data,
                             int pattern_type,
                             int total_operations)
{
    const unsigned int tid = get_global_id(0);
    const unsigned int total_threads = get_global_size(0);
    
    // Only use a few threads to start
    if (tid >= 8) return;
    
    volatile __local unsigned int group;
    volatile __local unsigned int groups;
    
    full_init(b, &group, &groups, tid, 10); // Very small number for testing
    SYNCTHREADS;
    
    volatile uint32_t item;
    uint32_t ops_completed = 0;
    uint32_t failures = 0;
    
    // Very limited operations to test basic functionality
    for(int i = 0; i < 3; i++) { // Just 3 operations per thread
        if (tid % 2 == 0 && tid < 4) {
            // Only 2 threads enqueue
            #ifdef USE_SFQ_QUEUE
                int result = my_enqueue_slot((__global volatile my_queue_t*)q, tid * 10 + i + 1);
                if (result == 0) ops_completed++; 
                else failures++;
            #elif defined(USE_MS_QUEUE)
                int result = ms_enqueue((__global volatile ms_queue_t*)q,  tid*10+i+1);
                if (result == 0) ops_completed++;
                else failures++;
            #elif defined(USE_TZ_QUEUE)
                int result = tz_enqueue((__global volatile tz_queue_t*)q,  tid*10+i+1);
                if (result == 0) ops_completed++;
                else failures++;
            #endif
        }else if (tid % 2 == 1 && tid < 4) {
            /* consumers (dequeue) */
            #ifdef USE_SFQ_QUEUE
                int result = my_dequeue_slot((__global volatile my_queue_t*)q, &item);
            #elif defined(USE_MS_QUEUE)
                int result = ms_dequeue((__global volatile ms_queue_t*)q, &item);
            #elif defined(USE_TZ_QUEUE)
                int result = tz_dequeue((__global volatile tz_queue_t*)q, &item);
            #endif
        }
        
        // Longer delay to prevent race conditions
        for(int delay = 0; delay < 5000; delay++) {
            volatile int dummy = delay * 2;
        }
    }
    
    if (tid < 8) {
        metrics[tid * 2] = ops_completed;
        metrics[tid * 2 + 1] = failures;
    }
}

kernel void validate_queue_logic(__global volatile void* q,
                                __global volatile uint32_t* results)
{
    const unsigned int tid = get_global_id(0);
    const unsigned int total_threads = get_global_size(0);
    
    // Only use first 10 threads: thread 0 = producer, threads 1-9 = consumers
    if (tid >= 10) return;
    
    uint32_t my_ops = 0;
    uint32_t my_failures = 0;
    volatile uint32_t item;
    
    if (tid == 0) {
        // SINGLE PRODUCER - enqueue 18 items
        for (int i = 1; i <= 18; i++) {
            int result = 1; // assume failure initially
            int attempts = 0;
            
            while (result != 0 && attempts < 1000) {
                #ifdef USE_SFQ_QUEUE
                    result = my_enqueue_slot((__global volatile my_queue_t*)q, i);
                #elif defined(USE_MS_QUEUE)
                    result = ms_enqueue((__global volatile ms_queue_t*)q, i);
                #elif defined(USE_TZ_QUEUE)
                    result = tz_enqueue((__global volatile tz_queue_t*)q, i);
                #endif
                attempts++;
                
                // Small delay
                for (int d = 0; d < 50; d++) { volatile int dummy = d; }
            }
            
            if (result == 0) {
                my_ops++;
            } else {
                my_failures++;
            }
        }
        
    } else {
        // 9 CONSUMERS - each tries to dequeue ~2 items
        for (int attempt = 0; attempt < 4; attempt++) { // Try 4 times each
            int result = 1; // assume failure
            int tries = 0;
            
            while (result != 0 && tries < 200) {
                #ifdef USE_SFQ_QUEUE
                    result = my_dequeue_slot((__global volatile my_queue_t*)q, &item);
                #elif defined(USE_MS_QUEUE)
                    result = ms_dequeue((__global volatile ms_queue_t*)q, &item);
                #elif defined(USE_TZ_QUEUE)
                    result = tz_dequeue((__global volatile tz_queue_t*)q, &item);
                #endif
                tries++;
                
                // Small delay
                for (int d = 0; d < 100; d++) { volatile int dummy = d; }
            }
            
            if (result == 0) {
                my_ops++;
            } else {
                my_failures++;
            }
        }
    }
    
    // Store results: [ops, failures, thread_id]
    results[tid * 3 + 0] = my_ops;
    results[tid * 3 + 1] = my_failures;
    results[tid * 3 + 2] = tid;
}


// Test 1: Contention Pattern Test - COMPLETE KERNEL
kernel void contention_pattern_test(__global volatile barrier_t* b,
                                   __global volatile void* q,
                                   __global volatile uint32_t* metrics,
                                   __global volatile uint64_t* timing_data,
                                   int pattern_type,
                                   int total_operations)
{
    const unsigned int tid = get_global_id(0);
    const unsigned int total_threads = get_global_size(0);
    volatile __local unsigned int group;
    volatile __local unsigned int groups;
    
    full_init(b, &group, &groups, tid, total_operations);
    SYNCTHREADS;
    
    volatile uint32_t item;
    uint32_t ops_completed = 0;
    
    switch(pattern_type) {
        case 0: // HIGH_CONTENTION: All threads hammer same operations
            // for(int i = 0; i < total_operations / total_threads; i++) {
            //     #ifdef USE_SFQ_QUEUE
            //         while(my_enqueue_slot((__global volatile my_queue_t*)q, tid + i * total_threads)) {}
            //         while(my_dequeue_slot((__global volatile my_queue_t*)q, &item)) {}
            //     #elif defined(USE_MS_QUEUE)
            //         while(ms_enqueue((__global volatile ms_queue_t*)q, tid + i * total_threads)) {}
            //         while(ms_dequeue((__global volatile ms_queue_t*)q, &item)) {}
            //     #elif defined(USE_TZ_QUEUE)
            //         while(tz_enqueue((__global volatile tz_queue_t*)q, tid + i * total_threads)) {}
            //         while(tz_dequeue((__global volatile tz_queue_t*)q, &item)) {}
            //     #endif
            //     ops_completed += 2;
            // }
            break;
            
        case 1: // PRODUCER_HEAVY: 75% producers, 25% consumers
            // if (tid < (total_threads * 3) / 4) {
            //     // Producer threads
            //     for(int i = 0; i < (total_operations * 3) / (total_threads * 4); i++) {
            //         #ifdef USE_SFQ_QUEUE
            //             while(my_enqueue_slot((__global volatile my_queue_t*)q, tid + i + 1)) {}
            //         #elif defined(USE_MS_QUEUE)
            //             while(ms_enqueue((__global volatile ms_queue_t*)q, tid + i + 1)) {}
            //         #elif defined(USE_TZ_QUEUE)
            //             while(tz_enqueue((__global volatile tz_queue_t*)q, tid + i + 1)) {}
            //         #endif
            //         ops_completed++;
            //     }
            // } else {
            //     // Consumer threads
            //     for(int i = 0; i < total_operations / (total_threads / 4); i++) {
            //         #ifdef USE_SFQ_QUEUE
            //             while(my_dequeue_slot((__global volatile my_queue_t*)q, &item)) {}
            //         #elif defined(USE_MS_QUEUE)
            //             while(ms_dequeue((__global volatile ms_queue_t*)q, &item)) {}
            //         #elif defined(USE_TZ_QUEUE)
            //             while(tz_dequeue((__global volatile tz_queue_t*)q, &item)) {}
            //         #endif
            //         ops_completed++;
            //     }
            // }
            break;
            
        case 2: // CONSUMER_HEAVY: 25% producers, 75% consumers
            if (tid < total_threads / 4) {
                // Producer threads
                for(int i = 0; i < (total_operations * 3) / (total_threads / 4); i++) {
                    #ifdef USE_SFQ_QUEUE
                        while(my_enqueue_slot((__global volatile my_queue_t*)q, tid + i + 1)) {}
                    #elif defined(USE_MS_QUEUE)
                        while(ms_enqueue((__global volatile ms_queue_t*)q, tid + i + 1)) {}
                    #elif defined(USE_TZ_QUEUE)
                        while(tz_enqueue((__global volatile tz_queue_t*)q, tid + i + 1)) {}
                    #endif
                    ops_completed++;
                }
            } else {
                // Consumer threads
                for(int i = 0; i < total_operations / (total_threads * 3 / 4); i++) {
                    #ifdef USE_SFQ_QUEUE
                        while(my_dequeue_slot((__global volatile my_queue_t*)q, &item)) {}
                    #elif defined(USE_MS_QUEUE)
                        while(ms_dequeue((__global volatile ms_queue_t*)q, &item)) {}
                    #elif defined(USE_TZ_QUEUE)
                        while(tz_dequeue((__global volatile tz_queue_t*)q, &item)) {}
                    #endif
                    ops_completed++;
                }
            }
            break;
            
        case 3: // BALANCED: 50% producers, 50% consumers
            if (tid < total_threads / 2) {
                // Producer threads
                for(int i = 0; i < total_operations / total_threads; i++) {
                    #ifdef USE_SFQ_QUEUE
                        while(my_enqueue_slot((__global volatile my_queue_t*)q, tid + i + 1)) {}
                    #elif defined(USE_MS_QUEUE)
                        while(ms_enqueue((__global volatile ms_queue_t*)q, tid + i + 1)) {}
                    #elif defined(USE_TZ_QUEUE)
                        while(tz_enqueue((__global volatile tz_queue_t*)q, tid + i + 1)) {}
                    #endif
                    ops_completed++;
                }
            } else {
                // Consumer threads
                for(int i = 0; i < total_operations / total_threads; i++) {
                    #ifdef USE_SFQ_QUEUE
                        while(my_dequeue_slot((__global volatile my_queue_t*)q, &item)) {}
                    #elif defined(USE_MS_QUEUE)
                        while(ms_dequeue((__global volatile ms_queue_t*)q, &item)) {}
                    #elif defined(USE_TZ_QUEUE)
                        while(tz_dequeue((__global volatile tz_queue_t*)q, &item)) {}
                    #endif
                    ops_completed++;
                }
            }
            break;
            
        case 4: {
            // LOW_CONTENTION: Staggered access patterns
            int wave = tid / 32; // 32 threads per wave
            for(int w = 0; w < (total_threads + 31) / 32; w++) {
                if (wave == w) {
                    for(int i = 0; i < total_operations / total_threads; i++) {
                        if (i % 2 == 0) {
                            #ifdef USE_SFQ_QUEUE
                                while(my_enqueue_slot((__global volatile my_queue_t*)q, tid + i + 1)) {}
                            #elif defined(USE_MS_QUEUE)
                                while(ms_enqueue((__global volatile ms_queue_t*)q, tid + i + 1)) {}
                            #elif defined(USE_TZ_QUEUE)
                                while(tz_enqueue((__global volatile tz_queue_t*)q, tid + i + 1)) {}
                            #endif
                        } else {
                            #ifdef USE_SFQ_QUEUE
                                while(my_dequeue_slot((__global volatile my_queue_t*)q, &item)) {}
                            #elif defined(USE_MS_QUEUE)
                                while(ms_dequeue((__global volatile ms_queue_t*)q, &item)) {}
                            #elif defined(USE_TZ_QUEUE)
                                while(tz_dequeue((__global volatile tz_queue_t*)q, &item)) {}
                            #endif
                        }
                        ops_completed++;
                    }
                }
                SYNCTHREADS;
            }
            break;
        }
    } // End of switch statement
    
    // Store results
    metrics[tid] = ops_completed;
}

// Test 2: Scheduler Simulation
kernel void scheduler_simulation(__global volatile barrier_t* b,
                                __global volatile void* q,
                                __global volatile uint32_t* task_data,
                                __global volatile uint64_t* completion_times,
                                int scheduler_type,
                                int num_tasks)
{
    const unsigned int tid = get_global_id(0);
    const unsigned int total_threads = get_global_size(0);
    volatile __local unsigned int group;
    volatile __local unsigned int groups;
    
    full_init(b, &group, &groups, tid, num_tasks);
    SYNCTHREADS;
    
    volatile uint32_t task_id;
    uint32_t tasks_processed = 0;
    
    switch(scheduler_type) {
        case 0: // WORK_STEALING: Some threads produce tasks, others steal
            if (tid < total_threads / 4) {
                // Task producers (schedulers)
                for(int i = 0; i < num_tasks / (total_threads / 4); i++) {
                    uint32_t task = tid * 1000 + i + 1;
                    #ifdef USE_SFQ_QUEUE
                        while(my_enqueue_slot((__global volatile my_queue_t*)q, task)) {}
                    #elif defined(USE_MS_QUEUE)
                        while(ms_enqueue((__global volatile ms_queue_t*)q, task)) {}
                    #elif defined(USE_TZ_QUEUE)
                        while(tz_enqueue((__global volatile tz_queue_t*)q, task)) {}
                    #endif
                    tasks_processed++;
                }
            } else {
                // Worker threads (steal tasks)
                for(int attempt = 0; attempt < num_tasks / total_threads; attempt++) {
                    #ifdef USE_SFQ_QUEUE
                        if(!my_dequeue_slot((__global volatile my_queue_t*)q, &task_id)) {
                    #elif defined(USE_MS_QUEUE)
                        if(!ms_dequeue((__global volatile ms_queue_t*)q, &task_id)) {
                    #elif defined(USE_TZ_QUEUE)
                        if(!tz_dequeue((__global volatile tz_queue_t*)q, &task_id)) {
                    #endif
                            // Simulate task processing
                            volatile uint32_t work = task_id;
                            for(int w = 0; w < 100; w++) work *= (w + 1);
                            tasks_processed++;
                        }
                }
            }
            break;
            
        case 1: // PRIORITY_QUEUE: Higher priority tasks enqueued more frequently
            for(int i = 0; i < num_tasks / total_threads; i++) {
                uint32_t priority = (i % 10 < 3) ? 1 : 0; // 30% high priority
                uint32_t task = (priority << 16) | (tid * 1000 + i + 1);
                
                if (tid % 2 == 0) {
                    // Enqueue task
                    #ifdef USE_SFQ_QUEUE
                        while(my_enqueue_slot((__global volatile my_queue_t*)q, task)) {}
                    #elif defined(USE_MS_QUEUE)
                        while(ms_enqueue((__global volatile ms_queue_t*)q, task)) {}
                    #elif defined(USE_TZ_QUEUE)
                        while(tz_enqueue((__global volatile tz_queue_t*)q, task)) {}
                    #endif
                } else {
                    // Process task
                    #ifdef USE_SFQ_QUEUE
                        while(my_dequeue_slot((__global volatile my_queue_t*)q, &task_id)) {}
                    #elif defined(USE_MS_QUEUE)
                        while(ms_dequeue((__global volatile ms_queue_t*)q, &task_id)) {}
                    #elif defined(USE_TZ_QUEUE)
                        while(tz_dequeue((__global volatile tz_queue_t*)q, &task_id)) {}
                    #endif
                    // Simulate different processing times based on priority
                    uint32_t priority_level = task_id >> 16;
                    volatile uint32_t work = task_id;
                    int work_amount = priority_level ? 50 : 200; // High priority = less work
                    for(int w = 0; w < work_amount; w++) work *= (w + 1);
                }
                tasks_processed++;
            }
            break;
    } // End of switch
    
    task_data[tid] = tasks_processed;
}

// Test 3: BFS Graph Traversal Simulation
kernel void bfs_simulation(__global volatile barrier_t* b,
                          __global volatile void* q,
                          __global volatile uint32_t* metrics,
                          __global volatile uint64_t* timing_data,
                          int pattern_id,
                          int num_nodes)
{
    const unsigned int tid = get_global_id(0);
    const unsigned int total_threads = get_global_size(0);
    volatile __local unsigned int group;
    volatile __local unsigned int groups;
    
    full_init(b, &group, &groups, tid, num_nodes);
    SYNCTHREADS;
    
    // Simple BFS simulation - each thread simulates graph traversal
    volatile uint32_t current_node;
    uint32_t nodes_processed = 0;
    
    // Initialize with starting nodes (spread across threads)
    if (tid == 0) {
        #ifdef USE_SFQ_QUEUE
            my_enqueue_slot((__global volatile my_queue_t*)q, 1);
        #elif defined(USE_MS_QUEUE)
            ms_enqueue((__global volatile ms_queue_t*)q, 1);
        #elif defined(USE_TZ_QUEUE)
            tz_enqueue((__global volatile tz_queue_t*)q, 1);
        #endif
    }
    
    SYNCTHREADS;
    
    // BFS traversal simulation
    for(int iter = 0; iter < num_nodes / total_threads; iter++) {
        #ifdef USE_SFQ_QUEUE
            if(!my_dequeue_slot((__global volatile my_queue_t*)q, &current_node)) {
        #elif defined(USE_MS_QUEUE)
            if(!ms_dequeue((__global volatile ms_queue_t*)q, &current_node)) {
        #elif defined(USE_TZ_QUEUE)
            if(!tz_dequeue((__global volatile tz_queue_t*)q, &current_node)) {
        #endif
                nodes_processed++;
                
                // Simulate adding neighbors to queue (simplified)
                uint32_t neighbor1 = (current_node % num_nodes) + 1;
                uint32_t neighbor2 = ((current_node + 1) % num_nodes) + 1;
                
                #ifdef USE_SFQ_QUEUE
                    if (neighbor1 <= num_nodes && neighbor1 != current_node) {
                        while(my_enqueue_slot((__global volatile my_queue_t*)q, neighbor1)) {}
                    }
                    if (neighbor2 <= num_nodes && neighbor2 != current_node) {
                        while(my_enqueue_slot((__global volatile my_queue_t*)q, neighbor2)) {}
                    }
                #elif defined(USE_MS_QUEUE)
                    if (neighbor1 <= num_nodes && neighbor1 != current_node) {
                        while(ms_enqueue((__global volatile ms_queue_t*)q, neighbor1)) {}
                    }
                    if (neighbor2 <= num_nodes && neighbor2 != current_node) {
                        while(ms_enqueue((__global volatile ms_queue_t*)q, neighbor2)) {}
                    }
                #elif defined(USE_TZ_QUEUE)
                    if (neighbor1 <= num_nodes && neighbor1 != current_node) {
                        while(tz_enqueue((__global volatile tz_queue_t*)q, neighbor1)) {}
                    }
                    if (neighbor2 <= num_nodes && neighbor2 != current_node) {
                        while(tz_enqueue((__global volatile tz_queue_t*)q, neighbor2)) {}
                    }
                #endif
            }
    }
    
    metrics[tid] = nodes_processed;
}

// Test 4: Burst Pattern Test
kernel void burst_pattern_test(__global volatile barrier_t* b,
                              __global volatile void* q,
                              __global volatile uint32_t* metrics,
                              __global volatile uint64_t* phase_times,
                              int pattern_type,
                              int total_operations)
{
    const unsigned int tid = get_global_id(0);
    const unsigned int total_threads = get_global_size(0);
    volatile __local unsigned int group;
    volatile __local unsigned int groups;
    
    full_init(b, &group, &groups, tid, total_operations);
    SYNCTHREADS;
    
    volatile uint32_t item;
    uint32_t ops_completed = 0;
    
    switch(pattern_type) {
        case 0: // BURST_ENQUEUE: Sudden spike in producers
            for(int phase = 0; phase < 5; phase++) {
                if (phase == 2) { // Burst phase - all threads become producers
                    for(int i = 0; i < total_operations / (total_threads * 2); i++) {
                        #ifdef USE_SFQ_QUEUE
                            while(my_enqueue_slot((__global volatile my_queue_t*)q, tid + i + 1)) {}
                        #elif defined(USE_MS_QUEUE)
                            while(ms_enqueue((__global volatile ms_queue_t*)q, tid + i + 1)) {}
                        #elif defined(USE_TZ_QUEUE)
                            while(tz_enqueue((__global volatile tz_queue_t*)q, tid + i + 1)) {}
                        #endif
                        ops_completed++;
                    }
                } else { // Normal phase - balanced
                    if (tid % 2 == 0) {
                        #ifdef USE_SFQ_QUEUE
                            while(my_enqueue_slot((__global volatile my_queue_t*)q, tid + phase + 1)) {}
                        #elif defined(USE_MS_QUEUE)
                            while(ms_enqueue((__global volatile ms_queue_t*)q, tid + phase + 1)) {}
                        #elif defined(USE_TZ_QUEUE)
                            while(tz_enqueue((__global volatile tz_queue_t*)q, tid + phase + 1)) {}
                        #endif
                    } else {
                        #ifdef USE_SFQ_QUEUE
                            while(my_dequeue_slot((__global volatile my_queue_t*)q, &item)) {}
                        #elif defined(USE_MS_QUEUE)
                            while(ms_dequeue((__global volatile ms_queue_t*)q, &item)) {}
                        #elif defined(USE_TZ_QUEUE)
                            while(tz_dequeue((__global volatile tz_queue_t*)q, &item)) {}
                        #endif
                    }
                    ops_completed++;
                }
                SYNCTHREADS;
            }
            break;
            
        case 1: {
            // PERIODIC_LOAD: Regular cycles of high/low activity
            for(int cycle = 0; cycle < 10; cycle++) {
                int activity_level = (cycle % 3 == 0) ? 3 : 1; // High activity every 3rd cycle
                
                for(int i = 0; i < activity_level; i++) {
                    if (tid < total_threads / 2) {
                        #ifdef USE_SFQ_QUEUE
                            while(my_enqueue_slot((__global volatile my_queue_t*)q, tid + cycle * 100 + i + 1)) {}
                        #elif defined(USE_MS_QUEUE)
                            while(ms_enqueue((__global volatile ms_queue_t*)q, tid + cycle * 100 + i + 1)) {}
                        #elif defined(USE_TZ_QUEUE)
                            while(tz_enqueue((__global volatile tz_queue_t*)q, tid + cycle * 100 + i + 1)) {}
                        #endif
                    } else {
                        #ifdef USE_SFQ_QUEUE
                            while(my_dequeue_slot((__global volatile my_queue_t*)q, &item)) {}
                        #elif defined(USE_MS_QUEUE)
                            while(ms_dequeue((__global volatile ms_queue_t*)q, &item)) {}
                        #elif defined(USE_TZ_QUEUE)
                            while(tz_dequeue((__global volatile tz_queue_t*)q, &item)) {}
                        #endif
                    }
                    ops_completed++;
                }
                SYNCTHREADS;
            }
            break;
        }
    } // End of switch
    
    metrics[tid] = ops_completed;
} 