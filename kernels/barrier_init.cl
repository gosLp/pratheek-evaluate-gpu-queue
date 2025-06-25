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