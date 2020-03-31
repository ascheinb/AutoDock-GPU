#include <cstdint>
#include <cassert>
#include "defines.h"
#include "calcenergy.h"
#include "GpuData.h"

__device__ inline uint64_t llitoulli(int64_t l)
{
    uint64_t u;
    asm("mov.b64    %0, %1;" : "=l"(u) : "l"(l));
    return u;
}

__device__ inline int64_t ullitolli(uint64_t u)
{
    int64_t l;
    asm("mov.b64    %0, %1;" : "=l"(l) : "l"(u));
    return l;
}


#define WARPMINIMUMEXCHANGE(tgx, v0, k0, mask) \
    { \
        float v1    = v0; \
        int k1      = k0; \
        int otgx    = tgx ^ mask; \
        float v2    = __shfl_sync(0xffffffff, k0, otgx); \
        int k2      = __shfl_sync(0xffffffff, k0, otgx); \
        int flag    = ((k1 > k2) ^ (tgx > otgx)) && (k1 != k2); \
        k0          = flag ? k1 : k2; \
        v0          = flag ? v1 : v2; \
    }

#define WARPMINIMUM2(tgx, v0, k0) \
    WARPMINIMUMEXCHANGE(tgx, v0, k0, 1) \
    WARPMINIMUMEXCHANGE(tgx, v0, k0, 2) \
    WARPMINIMUMEXCHANGE(tgx, v0, k0, 4) \
    WARPMINIMUMEXCHANGE(tgx, v0, k0, 8) \
    WARPMINIMUMEXCHANGE(tgx, v0, k0, 16)   

#define REDUCEINTEGERSUM(value, pAccumulator) \
    if (threadIdx.x == 0) \
    { \
        *pAccumulator = 0; \
    } \
    __threadfence(); \
    __syncthreads(); \
    if (__any_sync(0xffffffff, value != 0)) \
    { \
        uint32_t tgx            = threadIdx.x & cData.warpmask; \
        value                  += __shfl_sync(0xffffffff, value, tgx ^ 1); \
        value                  += __shfl_sync(0xffffffff, value, tgx ^ 2); \
        value                  += __shfl_sync(0xffffffff, value, tgx ^ 4); \
        value                  += __shfl_sync(0xffffffff, value, tgx ^ 8); \
        value                  += __shfl_sync(0xffffffff, value, tgx ^ 16); \
        if (tgx == 0) \
        { \
            atomicAdd(pAccumulator, value); \
        } \
    } \
    __threadfence(); \
    __syncthreads(); \
    value = *pAccumulator; \
    __syncthreads();


#define ATOMICADDF32(pAccumulator, value) atomicAdd(pAccumulator, (value))
#define ATOMICSUBF32(pAccumulator, value) atomicAdd(pAccumulator, -(value))


#define RSCALE  (1ll << 30)
static const float REDUCESCALEF             = (float)RSCALE;
static const float ONEOVEREDUCESCALEF       = (float)1.0 / REDUCESCALEF;
#define REDUCEFLOATSUM(value, pAccumulator) \
    if (threadIdx.x == 0) \
    { \
        *pAccumulator = 0; \
    } \
    __threadfence(); \
    __syncthreads(); \
    if (__any_sync(0xffffffff, value != 0.0f)) \
    { \
        uint32_t tgx            = threadIdx.x & cData.warpmask; \
        value                  += __shfl_sync(0xffffffff, value, tgx ^ 1); \
        value                  += __shfl_sync(0xffffffff, value, tgx ^ 2); \
        value                  += __shfl_sync(0xffffffff, value, tgx ^ 4); \
        value                  += __shfl_sync(0xffffffff, value, tgx ^ 8); \
        value                  += __shfl_sync(0xffffffff, value, tgx ^ 16); \
        if (tgx == 0) \
        { \
            atomicAdd(pAccumulator, llitoulli(llrintf(REDUCESCALEF * value))); \
        } \
    } \
    __threadfence(); \
    __syncthreads(); \
    value = (float)(*pAccumulator) * ONEOVEREDUCESCALEF; \
    __syncthreads();




static __constant__ GpuData cData;

void SetKernelsGpuData(GpuData* pData)
{
    cudaError_t status;
    status = cudaMemcpyToSymbol(cData, pData, sizeof(GpuData));
    RTERROR(status, "SetKernelsGpuData copy to cData failed");
}

void GetKernelsGpuData(GpuData* pData)
{
    cudaError_t status;
    status = cudaMemcpyFromSymbol(pData, cData, sizeof(GpuData));
    RTERROR(status, "GetKernelsGpuData copy From cData failed");
}


// Kernel files
#include "calcenergy.cu"
#include "calcMergeEneGra.cu"
#include "auxiliary_genetic.cu"
#include "kernel1.cu"
#include "kernel2.cu"
#include "kernel4.cu"
#include "kernel_ad.cu"
