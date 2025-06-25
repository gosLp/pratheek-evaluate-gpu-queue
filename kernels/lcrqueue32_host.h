/* kernels/lcrqueue32_host.h – minimal host copy -------------------- */
#pragma once
#include <stdint.h>

#define CRQ_LEN        (1u<<8)   /* keep it in sync with the .clh file */
#define NUM_BASE_CRQS  4

typedef struct { uint64_t combined; } Node32_host;

typedef struct {
    uint32_t head, pad1[15];
    uint32_t tail_lo, tail_hi;           /* Id32         (2×32-bit)  */
    uint32_t pad2[15];
    uint32_t next;
    uint32_t pad3[15];
    uint32_t size;
    uint32_t pad4[15];
    Node32_host ring[CRQ_LEN];
    uint32_t hazard[1500];
    uint32_t free;
} crq32_host;

typedef struct {
    uint32_t head, pad1[15];
    uint32_t tail;
    uint32_t pad2[15];
    uint32_t crq_size;
    uint32_t base_spin;
    crq32_host base[NUM_BASE_CRQS];
} lcrq32_host;
