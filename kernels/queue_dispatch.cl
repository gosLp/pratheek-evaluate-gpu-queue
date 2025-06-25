// #include "queue_ms.cl"
#include "queue_sfq.cl"
// #include "queue_tz.cl"
// #include "queue_lcrq32.cl"

// Include the generic test kernel
#include "queue_test_generic.cl"

#include "lcrqueue32.h"

// LCRQ initialization kernel
#ifdef USE_LCRQ_QUEUE
kernel void lcrq_init(__global volatile lcrq32 *q)
{
    if (get_global_id(0) != 0) return;   // Only one work-item does this
    
    // Queue-level fields
    q->head      = 0;
    q->tail      = 0;
    q->base_spin = 0;
    q->crq_size  = CRQ_LEN;
    
    // Initialize the first CRQ properly
    init_cr_32_queue(&q->base[0], CRQ_LEN);
    
    // Leave other CRQs unused but valid
    for (int c = 1; c < NUM_BASE_CRQS; ++c) {
        q->base[c].next = UINT_MAX;
        q->base[c].size = CRQ_LEN;
        q->base[c].head = 0;
        q->base[c].tail.combined = 0;
    }
}
#endif