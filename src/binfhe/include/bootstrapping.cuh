#ifndef _BOOTSTRAPPING_H_
#define _BOOTSTRAPPING_H_

#ifdef __CUDACC__
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "cufftdx.hpp"
#include "utils_gpu.cuh"
#endif
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <omp.h>
#include <chrono>
#include "binfhe-base-params.h"
#include "rgsw-cryptoparameters.h"
#include "rgsw-acckey.h"
#include "lwe-cryptoparameters.h"
#include "rlwe-ciphertext.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#ifdef __CUDACC__
namespace cg = cooperative_groups;
#endif

typedef std::complex<double> Complex;
#ifdef __CUDACC__
typedef cuDoubleComplex Complex_d;
#endif

namespace lbcrypto {

/**
 *
 * @brief GPUFFTBootstrap class is used to manage FFT-base bootstapping on GPU
 */
class GPUFFTBootstrap {
public:
    // Struct used to store GPU INFO
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

    // Struct of these GPU pointers
    #ifdef __CUDACC__
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
    #endif

    /**
     * GPU setup wrapper
     *
     * @param params a shared pointer to BinFHECryptoParams scheme parameters
     * @param BSkey Bootstrapping key
     * @param KSkey keyswitching key
     */
    static void GPUSetup(const std::shared_ptr<BinFHECryptoParams> params, RingGSWACCKey BSkey, LWESwitchingKey KSkey);

    /**
     * Clean GPU global memory
     */
    static void GPUClean();

    /**
     * EvalAcc on GPU
     *
     * @param params a shared pointer to RingGSW scheme parameters
     * @param a vector of a
     * @param acc vector of accumulator
     * @param fmod modulus used in second Modswitch (default = 0)
     */
    static void EvalAcc_CUDA(const std::shared_ptr<RingGSWCryptoParams> params, const std::vector<NativeVector>& a, std::shared_ptr<std::vector<RLWECiphertext>> acc, uint64_t fmod = 0);

    /**
     * Modswitch, Keyswitch, and Modswitch combo on GPU
     *
     * @param params a shared pointer to LWE scheme parameters
     * @param ctExt shared pointer of vector of LWE ciphertext
     * @param fmod modulus used in second Modswitch (default = 0)
     */
    static void MKMSwitch_CUDA(const std::shared_ptr<LWECryptoParams> params, std::shared_ptr<std::vector<LWECiphertext>> ctExt,
                            NativeInteger fmod);

private:
    // Maximum number of bootstrapping, to prevent overusing RAM 
    static constexpr int max_bootstapping_num = 65536;
    
    // Pre-allocated host side memory for bootstrapping
    static Complex* acc_host;
    static uint64_t* ctExt_host;

    // Synchornization Map used in Single block mode
    static const std::map<uint32_t, uint32_t> synchronizationMap;

    // Maximum dynamic shared memory amount of GPUs
    // CUDA start reserve 1 KB of shared memory per thread block after arch 800
    static const std::map<uint32_t, int> sharedMemMap;

    // Vector used to store multiple GPUs INFO
    static std::vector<GPUInfo> gpuInfoList;

    // Vector used to store multiple GPUs Pointers
    #ifdef __CUDACC__
    static std::vector<GPUPointer> GPUVec;
    #endif

    /**
     * Main accumulation function used in bootstrapping - Single Block mode
     *
     * @param params a shared pointer to RingGSW scheme parameters
     * @param a vector of a
     * @param acc vector of accumulator
     * @param fmod modulus used in second Modswitch
     */
    template<uint32_t arch, uint32_t FFT_dimension, uint32_t FFT_num>
    static void AddToAccCGGI_CUDA_single(const std::shared_ptr<RingGSWCryptoParams> params, const std::vector<NativeVector>& a, std::shared_ptr<std::vector<RLWECiphertext>> acc, uint64_t fmod);

    /**
     * Main accumulation function used in bootstrapping - Multi Blocks mode
     *
     * @param params a shared pointer to RingGSW scheme parameters
     * @param a vector of a
     * @param acc vector of accumulator
     * @param fmod modulus used in second Modswitch
     */
    template<uint32_t arch, uint32_t FFT_dimension>
    static void AddToAccCGGI_CUDA_multi(const std::shared_ptr<RingGSWCryptoParams> params, const std::vector<NativeVector>& a, std::shared_ptr<std::vector<RLWECiphertext>> acc, uint64_t fmod);

    /**
     * Core of the GPU setup wrapper
     *
     * @param params a shared pointer to BinFHECryptoParams scheme parameters
     * @param BSkey Bootstrapping key
     * @param KSkey keyswitching key
     */
    template<uint32_t arch, uint32_t FFT_dimension>
    static void GPUSetup_core(const std::shared_ptr<BinFHECryptoParams> params, RingGSWACCKey BSkey, LWESwitchingKey KSkey);

    /**
     * Copy NTT bootstrapping key to FFT format
     *
     * @param params a shared pointer to RingGSW scheme parameters
     * @param ek NTT-base Bootstrapping key
     */
    static std::shared_ptr<std::vector<std::vector<std::vector<Complex>>>> KeyCopy_FFT(const std::shared_ptr<RingGSWCryptoParams> params, RingGSWEvalKey ek);
};

// Global functions for GPU bootstrapping
#ifdef __CUDACC__
// Multiple blocks mode bootstrapping
template<class FFT, class IFFT>
static __global__ void bootstrappingMultiBlock(Complex_d* acc_CUDA, Complex_d* ct_CUDA, Complex_d* dct_CUDA, uint64_t* a_CUDA, 
        Complex_d* monomial_CUDA, Complex_d* twiddleTable_CUDA, Complex_d* GINX_bootstrappingKey_CUDA, uint64_t* keySwitchingkey_CUDA, 
            uint64_t* params_CUDA, uint64_t fmod);

// Single block mode bootstrapping
template<class FFT, class IFFT>
static __global__ void bootstrappingSingleBlock(Complex_d* acc_CUDA, Complex_d* ct_CUDA, Complex_d* dct_CUDA, uint64_t* a_CUDA, 
        Complex_d* monomial_CUDA, Complex_d* twiddleTable_CUDA, Complex_d* GINX_bootstrappingKey_CUDA, uint64_t* keySwitchingkey_CUDA, 
            uint64_t* params_CUDA, uint64_t fmod, uint32_t syncNum);

// cufftdx forward function
template<class FFT>
static __global__ void cuFFTDxFWD(Complex_d* data, Complex_d* twiddleTable_CUDA);

// Modswitch, Keyswitch, and Modswitch kernel
static __global__ void MKMSwitchKernel(uint64_t* ctExt_CUDA, uint64_t* keySwitchingkey_CUDA, uint64_t* params_CUDA, uint64_t fmod);
#endif

};  // namespace lbcrypto

#endif  // _BOOTSTRAPPING_H_