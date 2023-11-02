#ifndef _UTILS_GPU_H_
#define _UTILS_GPU_H_

#include <stdint.h>
#include <cuda_runtime.h>

__device__ inline
uint32_t ThisBlockRankInGrid() {
  return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
}

__device__ inline
uint32_t ThisGridSize() {
  return gridDim.x * gridDim.y * gridDim.z;
}

__device__ inline
uint32_t ThisThreadRankInBlock() {
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

__device__ inline
uint32_t ThisBlockSize() {
  return blockDim.x * blockDim.y * blockDim.z;
}

#endif  // _UTILS_GPU_H_