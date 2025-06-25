#include "lcrqueue32.h"

/* Single work-item initialiser -– run **once** before you start using q */
kernel void lcrq_init(__global volatile lcrq32 *q)
{
    if (get_global_id(0) != 0) return;   // safety

    /* queue-level fields */
    q->head      = 0;
    q->tail      = 0;
    q->base_spin = 0;
    q->crq_size  = CRQ_LEN;

    /* initial-seeding of the first CRQ */
    init_cr_32_queue(&q->base[0], CRQ_LEN);

    /* leave the other three CRQs unused but valid */
    for (int c = 1; c < NUM_BASE_CRQS; ++c) {
        q->base[c].next = UINT_MAX;          // disconnected
        q->base[c].size = CRQ_LEN;
        q->base[c].head = 0;
        q->base[c].tail.combined = 0;
    }
}
