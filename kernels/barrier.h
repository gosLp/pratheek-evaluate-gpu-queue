//common defines
#ifndef __BARRIER_H
#define __BARRIER_H
#ifndef FAILSAFE
#define FAILSAFE 1000
#endif

#define WARP_ID get_group_id(1)
#define THREAD_ID
#ifndef GROUPS
#define GROUPS get_num_groups(1)
#endif

#ifndef uint32_t
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;
#endif

#ifndef STRIDE
#define STRIDE 1
#endif

#define PRINT //

#ifndef DELAY
#define DELAY 1000
#endif

#ifndef NOFAILSAFE
#define TEST_FAILSAFE fail < FAILSAFE
#else
#define TEST_FAILSAFE 1
#endif

//language specific
#if defined(ATOMIC_COMPUTE) || defined(ATOMIC_WRITE)
#ifdef __CUDA_ARCH__
#define VWRITE(X,Y) atomicExch(&(X),Y)
#else
#define VWRITE(X,Y) atomic_xchg(&(X),Y)
#endif
#else // ATOMIC_COMPUTE
#define VWRITE(X,Y) X=Y
#endif

#if defined(ATOMIC_COMPUTE)
#ifdef __CUDA_ARCH__
#define VREAD(X) atomicAdd(&(X),0)
#else
#define VREAD(X) atomic_add(&(X),0)
#endif
#else //defined(ATOMIC_COMPUTE) || defined(ATOMIC_WRITE)
#define VREAD(X) X
#endif

#if defined(__CUDA_ARCH__)
#define LOCAL_ID (threadIdx.x + threadIdx.y*threadDim.x)
#define VOLATILE_READ(X) atomicAdd((unsigned int *)&(X),0)
#define VOLATILE_WRITE(X,Y) atomicExch((unsigned int *)&(X),Y)
#define VOLATILE_ADD(X,Y) atomicAdd((unsigned int *)&(X),Y)
#define VOLATILE_CAS(X,Y,Z) atomicCAS((unsigned int *)&(X),Y,Z)
#else
#define VOLATILE_READ(X) atomic_add(&(X),0)
#define VOLATILE_WRITE(X,Y) atomic_xchg(&(X),Y)
#define VOLATILE_XCHG(X,Y) atomic_xchg(&(X),Y)
#define VOLATILE_ADD(X,Y) atomic_add(&(X),Y)
#define VOLATILE_SUB(X,Y) atomic_sub(&(X),Y)
#define VOLATILE_OR(X,Y) atomic_or(&(X),Y)
#define VOLATILE_INC(X) atomic_add(&(X),1)
#define VOLATILE_CAS(X,Y,Z) atomic_cmpxchg(&(X),Y,Z)
#endif

#ifdef FENCE
#ifdef __CUDA_ARCH__ //CUDA
#define THREAD_FENCE __threadfence()
#else //OpenCL
#if defined(NVIDIA)
#define THREAD_FENCE asm("membar.gl;\n\t")
#elif defined(AMD) //amd version doesn't work on GCN, need S_WAITCNT 0 added at machine code level
#define THREAD_FENCE asm("fence_memory\n\t")
#endif
#endif //__OPENCL__
#else //FENCE
#define THREAD_FENCE  //do nothing
#endif //FENCE

#ifdef __CUDA_ARCH__
#define SYNCTHREADS __syncthreads()
#else
#define SYNCTHREADS barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
#endif

inline void WAIT_LOCAL(){
    uint32_t bah = 15;
    for(int i=0; i<WORK; i++){
        bah *= i + i; 
    }
}

inline void WAIT(volatile uint32_t * bah){
    for(int i=0; i<WORK; i++){
        *bah *= i + i; 
    }
}

#define YIELD WAIT_LOCAL()
#define MEMORY_SPACE __global
#define THREAD_NUM (get_local_id(0)%WARP_SIZE)
#define MAXTHREADS 1500


//inline void thread_fence()
//{
    //// "membar.sys;\n\t"
//#ifdef FENCE
////#ifndef __CPU__
//#ifdef NVIDIA
    ////mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);//resolves to membar.cta, but enforces instruction ordering
    //asm(
        //"membar.gl;\n\t"
       //);
//#endif
//#ifdef AMD
    //mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    //asm("\n\tfence_memory\n\t");//works up until 7970, then ignored
//#endif
////#endif
//#endif //FENCE
    ////asm("S_ICACHE_INV");
//}

typedef struct mutex_t{
	volatile int ticket;
	volatile int turn;
} mutex_t;

typedef struct barrier_t{
    uint32_t participants;
    uint32_t delay;
    uint32_t leader;
    uint32_t lid0;
    uint32_t lid1;
    uint32_t lid2;
    uint32_t lock;
    
    uint32_t total_threads;
    uint32_t total_groups;
    uint32_t present;
    
    
    uint32_t goal;
    
    uint32_t free;
    uint32_t init;
    
    uint32_t even;
    uint32_t odd;
}barrier_t;

#ifndef WARP
#define WARP 32
#endif

inline void clean_init_barr(__global volatile barrier_t *b, __local volatile unsigned int* ret, unsigned int lid)
{
    if(lid == 0)
        *ret = VOLATILE_ADD((b->init), 1);
    /*SYNCTHREADS;*/
}

inline void clean_prime_barr(__global volatile barrier_t *b, uint32_t group_id, __local volatile uint32_t *ret, uint32_t lid, uint32_t request){
    if(lid==0){
        if(0 == VOLATILE_CAS((b->leader), 0, 1)){//only lock new groups out if the current last group reaches this line
            for(uint32_t i=0; i<DELAY && VOLATILE_READ(b->init) < request; ++i){
                VOLATILE_ADD(b->leader,1);
            }
            unsigned int req = VOLATILE_READ(b->init);
            if(req > request)
                req = request;
           VOLATILE_WRITE((b->participants), req);
           /*VOLATILE_WRITE((b->participants), min(VOLATILE_READ(b->init),(unsigned int)request));*/
        }
        while((*ret = VOLATILE_READ(b->participants)) == 0) { }
    }
    /*SYNCTHREADS;*/
}
inline void full_init(__global volatile barrier_t *b, __local volatile uint32_t *group_id, __local volatile uint32_t *groups, uint32_t lid, uint32_t request){
    if(lid == 0){
        if((*group_id = VOLATILE_ADD((b->init), 1))==0){
	    for(int i=0; i<DELAY && VOLATILE_READ(b->init) < request; ++i){
		VOLATILE_ADD(b->leader,1);
	    }
            unsigned int req = VOLATILE_READ(b->init);
            if(req > request)
                req = request;
           VOLATILE_WRITE((b->participants), req);
           /*VOLATILE_WRITE((b->participants), min(VOLATILE_READ(b->init),(unsigned int)request));*/
        }
        while((*groups = VOLATILE_READ(b->participants)) == 0) { }
    }
    SYNCTHREADS;
}
#endif
