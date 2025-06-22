#define NULL_1 UINT_MAX
#define NULL_0 (UINT_MAX-1)

typedef struct tz_queue {
    volatile uint32_t head;
    volatile uint32_t tail;
    volatile uint32_t vnull;
    volatile uint32_t size;
    volatile uint32_t nodes[MY_QUEUE_LENGTH];
} tz_queue_t;

#ifndef __OPENCL__
tz_queue_t * new_tz_queue(uint32_t size);
void init_tz_queue(tz_queue_t * q, uint32_t size);
#endif

int tz_enqueue(volatile MEMORY_SPACE tz_queue_t * t, uint32_t newnode);
int tz_dequeue(volatile MEMORY_SPACE tz_queue_t *t, volatile uint32_t * oldnode);

uint32_t tz_queue_size(uint32_t size);
