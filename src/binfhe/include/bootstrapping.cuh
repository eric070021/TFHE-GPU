#ifndef _BOOTSTRAPPING_H_
#define _BOOTSTRAPPING_H_

#ifdef __CUDACC__

#include <vector>
#include <string>
#include <map>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "cufftdx.hpp"
#include "utils_gpu.cuh"

namespace cg = cooperative_groups;

typedef cuDoubleComplex Complex_d;

namespace lbcrypto {

/* Struct used to store GPU INFO */
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

/* Vector used to store multiple GPUs INFO */
std::vector<GPUInfo> gpuInfoList;

/* Struct of these GPU pointers */
struct GPUPointer {
    Complex_d* GINX_bootstrappingKey_CUDA;
    uint64_t* keySwitchingkey_CUDA;
    Complex_d* monomial_CUDA;
    Complex_d* twiddleTable_CUDA;
    uint64_t* params_CUDA;
    Complex_d* ct_CUDA;
    Complex_d* dct_CUDA;
    Complex_d* acc_CUDA;
    uint64_t* a_CUDA;
    uint64_t* ctExt_CUDA;
    std::vector<cudaStream_t> streams; /* CUDA streams for parallel bootstrapping */
};

/* Vector used to store multiple GPUs Pointers */
std::vector<GPUPointer> GPUVec;

/* Multiple small thread blocks mode bootstrapping */
template<class FFT, class IFFT>
__global__ void bootstrappingMultiBlock(Complex_d* acc_CUDA, Complex_d* ct_CUDA, Complex_d* dct_CUDA, uint64_t* a_CUDA, 
        Complex_d* monomial_CUDA, Complex_d* twiddleTable_CUDA, Complex_d* GINX_bootstrappingKey_CUDA, uint64_t* params_CUDA);

/* Single Big thread blocks mode bootstrapping */
template<class FFT, class IFFT>
__global__ void bootstrappingSingleBlock(Complex_d* acc_CUDA, Complex_d* ct_CUDA, Complex_d* dct_CUDA, uint64_t* a_CUDA, 
        Complex_d* monomial_CUDA, Complex_d* twiddleTable_CUDA, Complex_d* GINX_bootstrappingKey_CUDA, uint64_t* params_CUDA, uint32_t syncNum);

/* cufftdx forward function to preprocess BTKey and monomial */
template<class FFT>
__global__ void cuFFTDxFWD(Complex_d* data, Complex_d* twiddleTable_CUDA);

/* Modswitch, Keyswitch, and  Modswitch kernel*/
__global__ void MKMSwitchKernel(uint64_t* ctExt_CUDA, uint64_t* keySwitchingkey_CUDA, uint64_t* params_CUDA, uint64_t fmod);

};  // namespace lbcrypto

#endif

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <chrono>
#include "rgsw-cryptoparameters.h"
#include "lwe-cryptoparameters.h"
#include "rlwe-ciphertext.h"
#include "math/dftransform.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

typedef std::complex<double> Complex;


namespace lbcrypto {

/* Struct used to query synchronizationMap */
struct syncKey {
    uint32_t cudaArch;
    uint32_t FFT_dimension;
    // Define comparison operator for the key
    bool operator<(const syncKey& other) const {
        return std::tie(cudaArch, FFT_dimension) < std::tie(other.cudaArch, other.FFT_dimension);
    }
};

/* Synchornization Map used in Single block mode */
const std::map<syncKey, uint32_t> synchronizationMap({
  // arch | dim | syncNum
    {{700, 512},    8},
    {{700, 1024},  12},
    {{700, 2048},   0},
    {{800, 512},    8},
    {{800, 1024},  12},
    {{800, 2048},   0},
    {{860, 512},    8},
    {{860, 1024},  12},
    {{860, 2048},   0},
    {{890, 512},    8},
    {{890, 1024},  12},
    {{890, 2048},   0},
    {{900, 512},    0},
    {{900, 1024},   0},
    {{900, 2048},   0},
});

/* Maximum dynamic shared memory amount of GPUs */
/* CUDA start reserve 1 KB of shared memory per thread block after arch 800 */
const std::map<uint32_t, int> sharedMemMap({
  // arch | shared memory capacity (KB)
    {700,  96}, // V100
    {800, 163}, // A100
    {860,  99}, // RTX 3090
    {890,  99}, // RTX 4090
    {900, 227}, // H100
});

/***************************************
*  Preprocessing for GPU bootstrapping
****************************************/
void GPUSetup(std::shared_ptr<std::vector<std::vector<std::vector<std::shared_ptr<std::vector<std::vector<std::vector<Complex>>>>>>>> bootstrappingKey_FFT, 
    const std::shared_ptr<RingGSWCryptoParams> RGSWParams, LWESwitchingKey keySwitchingKey, const std::shared_ptr<LWECryptoParams> LWEParams);

template<uint32_t arch, uint32_t FFT_dimension>
void GPUSetup_core(std::shared_ptr<std::vector<std::vector<std::vector<std::shared_ptr<std::vector<std::vector<std::vector<Complex>>>>>>>> bootstrappingKey_FFT, 
    const std::shared_ptr<RingGSWCryptoParams> RGSWParams, LWESwitchingKey keySwitchingKey, const std::shared_ptr<LWECryptoParams> LWEParams);

/***************************************
*  ACC that support vector of ciphertexts 
****************************************/
void AddToAccCGGI_CUDA(const std::shared_ptr<RingGSWCryptoParams> params, const std::vector<NativeVector>& a, std::shared_ptr<std::vector<RLWECiphertext>> acc, uint64_t fmod);

template<uint32_t arch, uint32_t FFT_dimension, uint32_t FFT_num>
void AddToAccCGGI_CUDA_single(const std::shared_ptr<RingGSWCryptoParams> params, const std::vector<NativeVector>& a, std::shared_ptr<std::vector<RLWECiphertext>> acc, uint64_t fmod);

template<uint32_t arch, uint32_t FFT_dimension>
void AddToAccCGGI_CUDA_multi(const std::shared_ptr<RingGSWCryptoParams> params, const std::vector<NativeVector>& a, std::shared_ptr<std::vector<RLWECiphertext>> acc, uint64_t fmod);

/***************************************
*  Modswitch, Keyswitch, and Modswitch combo
****************************************/
void MKMSwitch_CUDA(const std::shared_ptr<LWECryptoParams> params, std::shared_ptr<std::vector<LWECiphertext>> ctExt,
                         NativeInteger fmod);

};  // namespace lbcrypto

#endif  // _BOOTSTRAPPING_H_