#ifndef _BOOTSTRAPPING_H_
#define _BOOTSTRAPPING_H_

#ifdef __CUDACC__

#include <cuComplex.h>
#include <cuda_runtime.h>
#include "cufftdx.hpp"
#include "utils_gpu.cuh"

typedef cuDoubleComplex Complex_d;

namespace lbcrypto {

using FFT_1024     = decltype(cufftdx::Block() + cufftdx::Size<1024>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::forward>() + cufftdx::ElementsPerThread<8>() +
                        cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<1>() + cufftdx::SM<890>());

using IFFT_1024     = decltype(cufftdx::Block() + cufftdx::Size<1024>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::inverse>() + cufftdx::ElementsPerThread<8>() +
                        cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<1>() + cufftdx::SM<890>());

using FFT_512      = decltype(cufftdx::Block() + cufftdx::Size<512>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::forward>() +
                        cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<1>() + cufftdx::SM<890>());

using IFFT_512     = decltype(cufftdx::Block() + cufftdx::Size<512>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::inverse>() +
                        cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<1>() + cufftdx::SM<890>());

using FFT = FFT_1024; // Default value for FFT
using IFFT = IFFT_1024; // Default value for IFFT

// global memory
__device__ Complex_d* GINX_bootstrappingKey_CUDA;
__device__ Complex_d* monomial_CUDA;
__device__ Complex_d* twiddleTable_CUDA;
__device__ Complex_d* ct_CUDA;
__device__ Complex_d* dct_CUDA;
__device__ uint64_t* params_CUDA;

template<class FFT, class IFFT>
__global__ void bootstrapping_CUDA(Complex_d* acc_CUDA, Complex_d* ct, Complex_d* dct, uint64_t* a_CUDA, Complex_d* monomial_CUDA, Complex_d* twiddleTable_CUDA, uint64_t* params_CUDA, Complex_d* GINX_bootstrappingKey_CUDA);

template<class FFT>
__global__ void cuFFTDxFWD(Complex_d* data, Complex_d* twiddleTable_CUDA);

};  // namespace lbcrypto

#endif

#include <iostream>
#include <vector>
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

void AddToAccCGGI_CUDA(const std::shared_ptr<RingGSWCryptoParams> params, const NativeVector& a, std::vector<std::vector<Complex>>& acc_d);

};  // namespace lbcrypto

#endif  // _BOOTSTRAPPING_H_