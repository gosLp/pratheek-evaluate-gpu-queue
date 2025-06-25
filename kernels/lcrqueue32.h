#ifndef __LCRQUEUE32_H
#define __LCRQUEUE32_H

// #include "cpu_queue.h"
#include "barrier.h"

#define CRQ_LEN (1<<8)

typedef union {
    uint32_t combined __attribute__((aligned(8)));
    // struct{
    //     uint32_t idx : 31;
    //     uint32_t safe : 1;
    // };
    // struct{
    //     uint32_t t : 31;
    //     uint32_t closed : 1;
    // };
} Id32;

// Bit manipulation macros to replace bit-field access
// for idx 31 bits and safe 1 bit
#define GET_IDX(id) ((id).combined & 0x7FFFFFFF) 
#define GET_SAFE(id) (((id).combined >> 31) & 1)           // Get upper 1 bit
#define SET_IDX(id, val) ((id).combined = ((id).combined & 0x80000000) | ((val) & 0x7FFFFFFF))
#define SET_SAFE(id, val) ((id).combined = ((id).combined & 0x7FFFFFFF) | (((val) & 1) << 31))


// For t (31 bits) and closed (1 bit) - same bit layout:
#define GET_T(id) ((id).combined & 0x7FFFFFFF)             // Get lower 31 bits  
#define GET_CLOSED(id) (((id).combined >> 31) & 1)         // Get upper 1 bit
#define SET_T(id, val) ((id).combined = ((id).combined & 0x80000000) | ((val) & 0x7FFFFFFF))
#define SET_CLOSED(id, val) ((id).combined = ((id).combined & 0x7FFFFFFF) | (((val) & 1) << 31))


typedef struct {
    union{
        uint64_t combined __attribute__((aligned(8)));
        struct{
            Id32 id; // safe and idx
            uint32_t val;
        };
    };
    //uint32_t trash1[14];
    //TODO pad to cache line size?
} Node32;

typedef struct crq32{
    uint32_t head;
    uint32_t trash1[15];
    Id32 tail; // closed and t
    uint32_t trash2[15];
    uint32_t next; //init NULL
    uint32_t trash3[15];
    uint32_t size;
    uint32_t trash4[15];
    Node32 ring[CRQ_LEN];//initially node u = <1,u,empty>
    uint32_t hazard[1500];
    uint32_t free;
} crq32;

#define NUM_BASE_CRQS 4

typedef struct lcrq32{
    uint32_t head;
    uint32_t trash1[15];
    uint32_t tail;
    uint32_t trash2[15];
    uint32_t crq_size;
    uint32_t base_spin;
    crq32 base[NUM_BASE_CRQS];
}lcrq32;


// typedef union {
//     uint16_t combined;
//     uint16_t idx;
//     uint16_t t;
//     //struct{ logical now, have to use & 1<<31 to access...
//         //unsigned char safe;
//     //};
//     //struct{
//         //unsigned char closed;
//     //};
// } Id16;
// typedef struct {
//     union{
//         uint32_t combined;
//         struct{
//             Id16 id; // safe and idx
//             uint16_t val;
//         };
//     };
//     //uint32_t trash1[14];
//     //TODO pad to cache line size?
// } Node16;
// typedef struct crq16{
//     uint32_t head;
//     uint32_t trash1[15];
//     Node16 tail; // closed and t
//     uint32_t trash2[15];
//     uint32_t next; //init NULL
//     uint32_t trash3[15];
//     uint32_t size;
//     uint32_t trash4[15];
//     Node16 ring[CRQ_LEN];//initially node u = <1,u,empty>
//     uint32_t hazard[1500];
//     uint32_t data[CRQ_LEN];
// } crq16;

// typedef struct lcrq16{
//     uint32_t head;
//     uint32_t trash1[15];
//     uint32_t tail;
//     uint32_t trash2[15];
//     uint32_t crq_size;
//     uint32_t base_spin;
//     crq16 base[NUM_BASE_CRQS];
// }lcrq16;

#ifndef __OPENCL__

lcrq32 * new_lcr_32_queue(uint32_t size);
void init_lcr_32_queue(volatile lcrq32 * q);
uint32_t reset_count_crq();
#endif

int lcr_dequeue32(volatile MEMORY_SPACE lcrq32 * q, volatile uint32_t *val);
int lcr_dequeue32_spinopt(volatile MEMORY_SPACE lcrq32 * q, volatile uint32_t *val);
int lcr_enqueue32(volatile MEMORY_SPACE lcrq32 *q, uint32_t val);
uint32_t lcr_32_size(uint32_t size);




#endif //__LCRQUEUE32_H
