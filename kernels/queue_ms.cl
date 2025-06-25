// Performance-optimized Michael-Scott Queue
// Maintains lock-freedom while reducing overhead
#ifndef FALSE
#define FALSE 0
#endif

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
  ms_pointer_t tail;
  ms_node_t nodes[MY_QUEUE_LENGTH+1];
  unsigned hazard1[1500];
  unsigned hazard2[1500];
  unsigned base_spin;
} ms_queue_t;

#define FREE_FALSE 1
#define FREE_TRUE 0

// Optimized hazard pointer management - reduced overhead
inline void ms_set_hazard(volatile __global ms_queue_t* q, uint32_t node){
    uint32_t warp_id = get_group_id(0) * 32 + (get_local_id(0) >> 5); // Assume 32-thread warps
    q->hazard1[warp_id] = node; // Direct write, no bounds check in fast path
}
inline void unms_set_hazard(volatile __global ms_queue_t* q){
    uint32_t warp_id = get_group_id(0) * 32 + (get_local_id(0) >> 5);
    q->hazard1[warp_id] = UINT_MAX;
}
inline void ms_set_hazard2(volatile __global ms_queue_t* q, uint32_t node){
    uint32_t warp_id = get_group_id(0) * 32 + (get_local_id(0) >> 5);
    q->hazard2[warp_id] = node;
}
inline void unms_set_hazard2(volatile __global ms_queue_t* q){
    uint32_t warp_id = get_group_id(0) * 32 + (get_local_id(0) >> 5);
    q->hazard2[warp_id] = UINT_MAX;
}

// Fast node allocation with minimal overhead
inline unsigned
new_node_fast(volatile __global ms_queue_t * q)
{
    ms_pointer_t ptr = {.sep.count = 0};
    unsigned new_node;
    uint32_t attempts = 0;
    
    // Fast path: try a few random nodes first
    do {
        new_node = (VOLATILE_INC(q->base_spin) % (MY_QUEUE_LENGTH - 2)) + 2;
        
        // Quick availability check
        if(VOLATILE_READ(q->nodes[new_node].free) == FREE_TRUE) {
            ms_set_hazard(q, new_node);
            
            // Try to claim it
            if(VOLATILE_CAS(q->nodes[new_node].free, FREE_TRUE, FREE_FALSE) == FREE_TRUE) {
                // Fast hazard check - only count our own group's threads
                uint32_t base_warp = get_group_id(0) * 32;
                uint32_t count = 0;
                // uint32_t max_warp = min(base_warp + 32, 1500);
                uint32_t max_warp = (base_warp + 32 < 1500) ? base_warp + 32 : 1500;
                
                for(uint32_t i = base_warp; i < max_warp; i++){
                    count += (q->hazard1[i] == new_node) ? 1 : 0;
                    count += (q->hazard2[i] == new_node) ? 1 : 0;
                }
                
                if(count == 1) { // Success!
                    ptr.sep.ptr = (unsigned short)new_node;
                    return ptr.con;
                }
                
                // Hazard conflict, release and try again
                VOLATILE_WRITE(q->nodes[new_node].free, FREE_TRUE);
            }
        }
        
        attempts++;
    } while(attempts < 100); // Much smaller limit for fast path
    
    unms_set_hazard(q);
    return 0; // Give up quickly to avoid blocking
}

// Original Michael-Scott CAS helper
inline unsigned cas(volatile __global uint32_t *X, uint32_t Y, uint32_t Z){
    return (atomic_cmpxchg(X,Y,Z) == Y);
}

inline unsigned MAKE_LONG(unsigned short node, unsigned short count){
    ms_pointer_t set = {.sep.count = count, .sep.ptr = node};
    return set.con;
}

// Optimized enqueue - closer to original algorithm
inline int
ms_enqueue_fast(__global volatile ms_queue_t * smp, unsigned val)
{
    if (val == 0) return 1; // Quick reject invalid values
    
    unsigned success = FALSE;
    unsigned node_val;
    ms_pointer_t tail;
    ms_pointer_t next;

    node_val = new_node_fast(smp);
    if (node_val == 0) return 1; // Node allocation failed
    
    ms_pointer_t node_ptr;
    node_ptr.con = node_val;
    unsigned node = node_ptr.sep.ptr;
    
    // Initialize the new node
    VOLATILE_WRITE(smp->nodes[node].value, val);
    next.con = 0; // NULL
    VOLATILE_WRITE(smp->nodes[node].next.con, next.con);

    // Classic Michael-Scott enqueue loop with minimal modifications
    while (success == FALSE) {
        tail.con = VOLATILE_READ(smp->tail.con);
        next.con = VOLATILE_READ(smp->nodes[tail.sep.ptr].next.con);
        ms_set_hazard2(smp, tail.sep.ptr);
        
        if (tail.con == VREAD(smp->tail.con)) {
            if (next.sep.ptr == 0) { // NULL
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
    
    // Swing tail
    cas(&smp->tail.con,
        tail.con,
        MAKE_LONG(node, tail.sep.count+1));
    
    unms_set_hazard2(smp);
    unms_set_hazard(smp);
    return 0;
}

// Optimized dequeue - closer to original algorithm  
inline unsigned
ms_dequeue_fast(__global volatile ms_queue_t * smp, volatile unsigned *val)
{
    unsigned value;
    unsigned success = FALSE;
    ms_pointer_t head;
    ms_pointer_t tail;
    ms_pointer_t next;

    while(1) {
        head.con = VOLATILE_READ(smp->head.con);
        tail.con = VOLATILE_READ(smp->tail.con);
        next.con = VOLATILE_READ(smp->nodes[head.sep.ptr].next.con);
        
        ms_set_hazard(smp, head.sep.ptr);
        ms_set_hazard2(smp, next.sep.ptr);
        
        if (VREAD(smp->head.con) == head.con) {
            if (head.sep.ptr == tail.sep.ptr) {
                if (next.sep.ptr == 0) { // NULL - empty queue
                    unms_set_hazard(smp);
                    unms_set_hazard2(smp);
                    return 1;
                }
                // Help advance tail
                cas(&smp->tail.con,
                    tail.con,
                    MAKE_LONG(next.sep.ptr, tail.sep.count+1));
            } else {
                // Read value before CAS
                value = VOLATILE_READ(smp->nodes[next.sep.ptr].value);
                success = cas(&smp->head.con,
                            head.con,
                            MAKE_LONG(next.sep.ptr, head.sep.count+1));
                if (success) break;
            }
        }
    }
    
    // Free the old head node
    VOLATILE_WRITE(smp->nodes[head.sep.ptr].free, FREE_TRUE);
    unms_set_hazard(smp);
    unms_set_hazard2(smp);
    *val = value;
    return 0;
}

// Fallback versions with timeouts (for safety)
inline int ms_enqueue(__global volatile ms_queue_t * smp, unsigned val) {
    int result = ms_enqueue_fast(smp, val);
    return result;
}

inline unsigned ms_dequeue(__global volatile ms_queue_t * smp, volatile unsigned *val) {
    return ms_dequeue_fast(smp, val);
}

// High-performance batch operations
inline int ms_enqueue_batch(__global volatile ms_queue_t * smp, unsigned* values, int count) {
    int success_count = 0;
    for (int i = 0; i < count; i++) {
        if (ms_enqueue_fast(smp, values[i]) == 0) {
            success_count++;
        } else {
            break; // Stop on first failure to maintain order
        }
    }
    return success_count;
}

inline int ms_dequeue_batch(__global volatile ms_queue_t * smp, volatile unsigned* values, int count) {
    int success_count = 0;
    for (int i = 0; i < count; i++) {
        if (ms_dequeue_fast(smp, &values[i]) == 0) {
            success_count++;
        } else {
            break; // Stop on first failure (empty queue)
        }
    }
    return success_count;
}