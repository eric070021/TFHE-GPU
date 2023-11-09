#ifndef _BOOTSTRAPPING_H_
#define _BOOTSTRAPPING_H_

#ifdef __CUDACC__

#include <vector>
#include <string>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "cufftdx.hpp"
#include "utils_gpu.cuh"

namespace cg = cooperative_groups;

typedef cuDoubleComplex Complex_d;

namespace lbcrypto {
// global memory
__device__ Complex_d* GINX_bootstrappingKey_CUDA;
__device__ Complex_d* monomial_CUDA;
__device__ Complex_d* twiddleTable_CUDA;
__device__ Complex_d* ct_CUDA;
__device__ Complex_d* dct_CUDA;
__device__ uint64_t* params_CUDA;

// Create CUDA streams for parallel bootstrapping
std::vector<cudaStream_t> streams;

template<class FFT, class IFFT>
__global__ void bootstrappingMultiBlock(Complex_d* acc_CUDA, Complex_d* ct_CUDA, Complex_d* dct_CUDA, uint64_t* a_CUDA, Complex_d* monomial_CUDA, Complex_d* twiddleTable_CUDA, uint64_t* params_CUDA, Complex_d* GINX_bootstrappingKey_CUDA);

template<class FFT, class IFFT>
__global__ void bootstrappingSingleBlock(Complex_d* acc_CUDA, Complex_d* ct_CUDA, Complex_d* dct_CUDA, uint64_t* a_CUDA, Complex_d* monomial_CUDA, Complex_d* twiddleTable_CUDA, uint64_t* params_CUDA, Complex_d* GINX_bootstrappingKey_CUDA);

template<class FFT>
__global__ void cuFFTDxFWD(Complex_d* data, Complex_d* twiddleTable_CUDA);

};  // namespace lbcrypto

#endif

#include <iostream>
#include <vector>
#include <string>
#include "rgsw-cryptoparameters.h"
#include "math/dftransform.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

typedef std::complex<double> Complex;

namespace lbcrypto {
struct GPUInfo {
    std::string name;
    int major;
    int minor;
    int sharedMemoryPerBlock;
    int maxBlocksPerMultiprocessor;
    int maxThreadsPerBlock;
    int maxGridX;
    int maxGridY;
    int maxGridZ;
    int maxBlockX;
    int maxBlockY;
    int maxBlockZ;
    int warpSize;
    int multiprocessorCount;
};

void GPUSetup(std::shared_ptr<std::vector<std::vector<std::vector<std::shared_ptr<std::vector<std::vector<std::vector<Complex>>>>>>>> GINX_bootstrappingKey_FFT,const std::shared_ptr<RingGSWCryptoParams> params);

template<uint32_t arch, uint32_t FFT_dimension, uint32_t FFT_num>
void GPUSetup_core(std::shared_ptr<std::vector<std::vector<std::vector<std::shared_ptr<std::vector<std::vector<std::vector<Complex>>>>>>>> GINX_bootstrappingKey_FFT,const std::shared_ptr<RingGSWCryptoParams> params);

void AddToAccCGGI_CUDA(const std::shared_ptr<RingGSWCryptoParams> params, const NativeVector& a, std::vector<std::vector<Complex>>& acc_d, std::string mode);

template<uint32_t arch, uint32_t FFT_dimension, uint32_t FFT_num>
void AddToAccCGGI_CUDA_core(const std::shared_ptr<RingGSWCryptoParams> params, const NativeVector& a, std::vector<std::vector<Complex>>& acc_d, std::string mode);

};  // namespace lbcrypto

#endif  // _BOOTSTRAPPING_H_