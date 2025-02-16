/*
 * Copyright (c) 2024 Inventec Corporation. All rights reserved.
 *
 * This software is licensed under the MIT License.
 *
 * MIT License:
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "bootstrapping.cuh"

#ifndef CUDA_CHECK_AND_EXIT
#    define CUDA_CHECK_AND_EXIT(error)                                                                      \
        {                                                                                                   \
            auto status = static_cast<cudaError_t>(error);                                                  \
            if (status != cudaSuccess) {                                                                    \
                std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                                          \
            }                                                                                               \
        }
#endif // CUDA_CHECK_AND_EXIT

namespace lbcrypto {

// Definition of the static member variables
Complex* GPUFFTBootstrap::acc_host;
uint64_t* GPUFFTBootstrap::ctExt_host;
std::vector<GPUFFTBootstrap::GPUInfo> GPUFFTBootstrap::gpuInfoList;
std::vector<GPUFFTBootstrap::GPUPointer> GPUFFTBootstrap::GPUVec;
const std::map<uint32_t, uint32_t> GPUFFTBootstrap::synchronizationMap ={
  // dim | syncNum
    {512,    8},
    {1024,  12},
};
const std::map<uint32_t, int> GPUFFTBootstrap::sharedMemMap = {
 // arch | shared memory capacity (KB)
    {700,  96}, // V100
    {800, 163}, // A100
    {860,  99}, // RTX 3090
    {890,  99}, // RTX 4090
    {900, 227}, // H100
};

__device__ inline void ModSubFastEq_CUDA(uint64_t &a, const uint64_t &b, const uint64_t &modulus) {
        if (a >= b) {
            a -= b;
        }
        else {
            a += (modulus - b);
        }
}

__device__ inline uint64_t RoundqQ_CUDA(const uint64_t &v, const uint64_t &q, const uint64_t &Q) {
    return static_cast<uint64_t>(floor(0.5 + static_cast<double>(v) * static_cast<double>(q) / static_cast<double>(Q))) % q;
}

__global__ void MKMSwitchKernel(uint64_t* ctExt_CUDA, uint64_t* keySwitchingkey_CUDA, uint64_t* params_CUDA, uint64_t fmod){
    /* GPU Parameters Set */
    uint32_t tid = ThisThreadRankInBlock();
    uint32_t bdim = ThisBlockSize();

    /* HE Parameters Set */
    const uint32_t n              = static_cast<uint32_t>(params_CUDA[0]);
    const uint32_t N              = static_cast<uint32_t>(params_CUDA[1]);
    const uint64_t Q              = params_CUDA[3];
    const uint64_t qKS            = params_CUDA[7];
    const uint32_t baseKS         = static_cast<uint32_t>(params_CUDA[8]);
    const uint32_t digitCountKS   = static_cast<uint32_t>(params_CUDA[9]);

    /* Shared memory */
    extern __shared__ uint64_t ct_shared[];
    for(uint32_t i = tid; i <= N; i += bdim){
        ct_shared[i] = ctExt_CUDA[i];
    }
    __syncthreads();

    /* First Modswitch */
    for (size_t i = tid; i <= N; i += bdim)
        ct_shared[i] = RoundqQ_CUDA(ct_shared[i], qKS, Q);
    __syncthreads();

    /* KeySwitch */
    uint64_t temp;
    for (uint32_t k = tid; k <= n; k += bdim){
        if (k == n) temp = ct_shared[N]; // b
        else temp = 0; // a
        for (uint32_t i = 0; i < N; ++i) {
            uint64_t atmp = ct_shared[i];
            for (uint32_t j = 0; j < digitCountKS; ++j, atmp /= baseKS) {
                uint64_t a0 = (atmp % baseKS);
                ModSubFastEq_CUDA(temp, keySwitchingkey_CUDA[i*baseKS*digitCountKS*(n + 1) + a0*digitCountKS*(n + 1) + j*(n + 1) + k], qKS);
            }
        }
        ctExt_CUDA[k] = temp;
    }
    __syncthreads();

    /* Second Modswitch */
    for (size_t i = tid; i <= n; i += bdim)
        ctExt_CUDA[i] = RoundqQ_CUDA(ctExt_CUDA[i], fmod, qKS);
    __syncthreads();
}

template<class FFT, class IFFT>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void bootstrappingMultiBlock(Complex_d* acc_CUDA, Complex_d* ct_CUDA, Complex_d* dct_CUDA, uint64_t* a_CUDA, 
        Complex_d* monomial_CUDA, Complex_d* twiddleTable_CUDA, Complex_d* GINX_bootstrappingKey_CUDA, uint64_t* keySwitchingkey_CUDA,
         uint64_t* params_CUDA, uint64_t fmod){
    
    /* GPU Parameters Set */
    cg::grid_group grid = cg::this_grid();
    uint32_t tid = ThisThreadRankInBlock(); // thread id in block
    uint32_t bid = grid.block_rank(); // block id in grid
    uint32_t gtid = grid.thread_rank(); // global thread id
    uint32_t bdim = ThisBlockSize(); // size of block
    uint32_t gdim = grid.num_threads(); // number of threads in grid

    /* HE Parameters Set */
    const uint32_t n                = static_cast<uint32_t>(params_CUDA[0]);
    const uint32_t M                = static_cast<uint32_t>(params_CUDA[1] << 1);
    const uint32_t N                = static_cast<uint32_t>(params_CUDA[1]);
    const uint32_t NHalf            = N >> 1;
    const uint64_t Q                = params_CUDA[3];
    const uint64_t QHalf            = Q >> 1;
    const uint32_t logbaseG         = static_cast<uint32_t>(params_CUDA[4]);
    const uint32_t digitsG2         = static_cast<uint32_t>(params_CUDA[5]);
    const uint32_t RGSW_size        = digitsG2 * 2 * NHalf;
    const int32_t gBits             = logbaseG;
    const int32_t gBitsMaxBits      = 64 - gBits;
    const uint32_t numDigitsToThrow = static_cast<uint32_t>(params_CUDA[6]);
    const bool modInt64             = Q > 4294967296 ? false : true; // Q > 2^32 or not

    /* cufftdx variables */
    using complex_type = typename FFT::value_type;
    const unsigned int local_fft_id = threadIdx.y;
    const unsigned int offset = cufftdx::size_of<FFT>::value * (blockIdx.x * FFT::ffts_per_block + local_fft_id);
    extern __shared__ complex_type shared_mem[];
    complex_type thread_data[FFT::storage_size];

    for(uint32_t round = 0; round < n; ++round){
        /* Approximate SignedDigitDecompose */
        for (size_t k = gtid; k < N; k += gdim) { // 0~NHalf-1: a, NHalf~N-1: b
            const uint64_t& t1 = static_cast<uint64_t>(acc_CUDA[k].x);
            const uint64_t& t2 = static_cast<uint64_t>(acc_CUDA[k].y);
            int64_t d0 = (t1 < QHalf) ? static_cast<int64_t>(t1) : (static_cast<int64_t>(t1) - static_cast<int64_t>(Q));
            int64_t d1 = (t2 < QHalf) ? static_cast<int64_t>(t2) : (static_cast<int64_t>(t2) - static_cast<int64_t>(Q));
            int64_t r0, r1;

            // throw away the least significant digits
            for(int i = 0; i < numDigitsToThrow; i++){
                r0 = (d0 << gBitsMaxBits) >> gBitsMaxBits;
                d0 = (d0 - r0) >> gBits;
                r1 = (d1 << gBitsMaxBits) >> gBitsMaxBits;
                d1 = (d1 - r1) >> gBits;
            }

            for (size_t l = 0; l < digitsG2; l += 2) {
                r0 = (d0 << gBitsMaxBits) >> gBitsMaxBits;
                d0 = (d0 - r0) >> gBits;
                if (r0 < 0)
                    r0 += static_cast<int64_t>(Q);
                if (r0 >= QHalf)
                    r0 -= static_cast<int64_t>(Q);
                dct_CUDA[l*NHalf + k].x = static_cast<double>(r0);

                r1 = (d1 << gBitsMaxBits) >> gBitsMaxBits;
                d1 = (d1 - r1) >> gBits;
                if (r1 < 0)
                    r1 += static_cast<int64_t>(Q);
                if (r1 >= QHalf)
                    r1 -= static_cast<int64_t>(Q);
                dct_CUDA[l*NHalf + k].y = static_cast<double>(r1);
            }
        }
        grid.sync();

        /* digitsG2 times Forward FFT */
        // Load data from shared memory to registers
        {
            unsigned int index = offset + threadIdx.x;
            unsigned int twist_idx = threadIdx.x;
            for (unsigned i = 0; i < FFT::elements_per_thread; i++) {
                // twisting
                dct_CUDA[index] = cuCmul(dct_CUDA[index], twiddleTable_CUDA[twist_idx]);
                thread_data[i] = complex_type {dct_CUDA[index].x, dct_CUDA[index].y};
                // FFT::stride shows how elements from a single FFT should be split between threads
                index += FFT::stride;
                twist_idx += FFT::stride;
            }
        }

        FFT().execute(thread_data, shared_mem);

        // Save results
        {
            unsigned int index = offset + threadIdx.x;
            for (unsigned i = 0; i < FFT::elements_per_thread; i++) {
                dct_CUDA[index] = make_cuDoubleComplex(thread_data[i].x, thread_data[i].y);
                // FFT::stride shows how elements from a single FFT should be split between threads
                index += FFT::stride;
            }
        }
        grid.sync();

        /* Obtain monomial */
        // First obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
        auto aNeg         = (M - a_CUDA[round]) % M;
        uint64_t indexPos = a_CUDA[round];
        uint64_t indexNeg = aNeg;
        // index is in range [0,m] - so we need to adjust the edge case when
        // index = m to index = 0
        if (indexPos == M)
            indexPos = 0;
        if (indexNeg == M)
            indexNeg = 0;
        
        /* Initialize ct_CUDA */
        for(uint32_t i = gtid; i < N; i += gdim){
            ct_CUDA[i] = make_cuDoubleComplex(0.0, 0.0);
        }
        grid.sync();

        /* ACC times Bootstrapping key and monomial */
        /* multiply with ek0 */
        // polynomial a
        for (uint32_t i = gtid; i < NHalf; i += gdim){
            thread_data[0] = complex_type {0.0, 0.0};
            for (uint32_t l = 0; l < digitsG2; ++l){
                thread_data[0].x = fma(dct_CUDA[l*NHalf + i].x, GINX_bootstrappingKey_CUDA[(round << 1)*RGSW_size + (l << 1)*NHalf + i].x, thread_data[0].x);
                thread_data[0].x = fma(-dct_CUDA[l*NHalf + i].y, GINX_bootstrappingKey_CUDA[(round << 1)*RGSW_size + (l << 1)*NHalf + i].y, thread_data[0].x);
                thread_data[0].y = fma(dct_CUDA[l*NHalf + i].x, GINX_bootstrappingKey_CUDA[(round << 1)*RGSW_size + (l << 1)*NHalf + i].y, thread_data[0].y);
                thread_data[0].y = fma(dct_CUDA[l*NHalf + i].y, GINX_bootstrappingKey_CUDA[(round << 1)*RGSW_size + (l << 1)*NHalf + i].x, thread_data[0].y);
            }
            /* multiply with postive monomial */
            ct_CUDA[i].x = fma(thread_data[0].x, monomial_CUDA[indexPos*NHalf + i].x, ct_CUDA[i].x);
            ct_CUDA[i].x = fma(-thread_data[0].y, monomial_CUDA[indexPos*NHalf + i].y, ct_CUDA[i].x);
            ct_CUDA[i].y = fma(thread_data[0].x, monomial_CUDA[indexPos*NHalf + i].y, ct_CUDA[i].y);
            ct_CUDA[i].y = fma(thread_data[0].y, monomial_CUDA[indexPos*NHalf + i].x, ct_CUDA[i].y);
        }
        // polynomial b
        for (uint32_t i = gtid; i < NHalf; i += gdim){
            thread_data[0] = complex_type {0.0, 0.0};
            for (uint32_t l = 0; l < digitsG2; ++l){
                thread_data[0].x = fma(dct_CUDA[l*NHalf + i].x, GINX_bootstrappingKey_CUDA[(round << 1)*RGSW_size + ((l << 1) + 1)*NHalf + i].x, thread_data[0].x);
                thread_data[0].x = fma(-dct_CUDA[l*NHalf + i].y, GINX_bootstrappingKey_CUDA[(round << 1)*RGSW_size + ((l << 1) + 1)*NHalf + i].y, thread_data[0].x);
                thread_data[0].y = fma(dct_CUDA[l*NHalf + i].x, GINX_bootstrappingKey_CUDA[(round << 1)*RGSW_size + ((l << 1) + 1)*NHalf + i].y, thread_data[0].y);
                thread_data[0].y = fma(dct_CUDA[l*NHalf + i].y, GINX_bootstrappingKey_CUDA[(round << 1)*RGSW_size + ((l << 1) + 1)*NHalf + i].x, thread_data[0].y);
            }
            /* multiply with postive monomial */
            ct_CUDA[NHalf + i].x = fma(thread_data[0].x, monomial_CUDA[indexPos*NHalf + i].x, ct_CUDA[NHalf + i].x);
            ct_CUDA[NHalf + i].x = fma(-thread_data[0].y, monomial_CUDA[indexPos*NHalf + i].y, ct_CUDA[NHalf + i].x);
            ct_CUDA[NHalf + i].y = fma(thread_data[0].x, monomial_CUDA[indexPos*NHalf + i].y, ct_CUDA[NHalf + i].y);
            ct_CUDA[NHalf + i].y = fma(thread_data[0].y, monomial_CUDA[indexPos*NHalf + i].x, ct_CUDA[NHalf + i].y);
        }

        /* multiply with ek1 */
        // polynomial a
        for (uint32_t i = gtid; i < NHalf; i += gdim){
            thread_data[0] = complex_type {0.0, 0.0};
            for (uint32_t l = 0; l < digitsG2; ++l){
                thread_data[0].x = fma(dct_CUDA[l*NHalf + i].x, GINX_bootstrappingKey_CUDA[((round << 1) + 1)*RGSW_size + (l << 1)*NHalf + i].x, thread_data[0].x);
                thread_data[0].x = fma(-dct_CUDA[l*NHalf + i].y, GINX_bootstrappingKey_CUDA[((round << 1) + 1)*RGSW_size + (l << 1)*NHalf + i].y, thread_data[0].x);
                thread_data[0].y = fma(dct_CUDA[l*NHalf + i].x, GINX_bootstrappingKey_CUDA[((round << 1) + 1)*RGSW_size + (l << 1)*NHalf + i].y, thread_data[0].y);
                thread_data[0].y = fma(dct_CUDA[l*NHalf + i].y, GINX_bootstrappingKey_CUDA[((round << 1) + 1)*RGSW_size + (l << 1)*NHalf + i].x, thread_data[0].y);
            }
            /* multiply with negative monomial */
            ct_CUDA[i].x = fma(thread_data[0].x, monomial_CUDA[indexNeg*NHalf + i].x, ct_CUDA[i].x);
            ct_CUDA[i].x = fma(-thread_data[0].y, monomial_CUDA[indexNeg*NHalf + i].y, ct_CUDA[i].x);
            ct_CUDA[i].y = fma(thread_data[0].x, monomial_CUDA[indexNeg*NHalf + i].y, ct_CUDA[i].y);
            ct_CUDA[i].y = fma(thread_data[0].y, monomial_CUDA[indexNeg*NHalf + i].x, ct_CUDA[i].y);
        }
        // polynomial b
        for (uint32_t i = gtid; i < NHalf; i += gdim){
            thread_data[0] = complex_type {0.0, 0.0};
            for (uint32_t l = 0; l < digitsG2; ++l){
                thread_data[0].x = fma(dct_CUDA[l*NHalf + i].x, GINX_bootstrappingKey_CUDA[((round << 1) + 1)*RGSW_size + ((l << 1) + 1)*NHalf + i].x, thread_data[0].x);
                thread_data[0].x = fma(-dct_CUDA[l*NHalf + i].y, GINX_bootstrappingKey_CUDA[((round << 1) + 1)*RGSW_size + ((l << 1) + 1)*NHalf + i].y, thread_data[0].x);
                thread_data[0].y = fma(dct_CUDA[l*NHalf + i].x, GINX_bootstrappingKey_CUDA[((round << 1) + 1)*RGSW_size + ((l << 1) + 1)*NHalf + i].y, thread_data[0].y);
                thread_data[0].y = fma(dct_CUDA[l*NHalf + i].y, GINX_bootstrappingKey_CUDA[((round << 1) + 1)*RGSW_size + ((l << 1) + 1)*NHalf + i].x, thread_data[0].y);
            }
            /* multiply with negative monomial */
            ct_CUDA[NHalf + i].x = fma(thread_data[0].x, monomial_CUDA[indexNeg*NHalf + i].x, ct_CUDA[NHalf + i].x);
            ct_CUDA[NHalf + i].x = fma(-thread_data[0].y, monomial_CUDA[indexNeg*NHalf + i].y, ct_CUDA[NHalf + i].x);
            ct_CUDA[NHalf + i].y = fma(thread_data[0].x, monomial_CUDA[indexNeg*NHalf + i].y, ct_CUDA[NHalf + i].y);
            ct_CUDA[NHalf + i].y = fma(thread_data[0].y, monomial_CUDA[indexNeg*NHalf + i].x, ct_CUDA[NHalf + i].y);
        }
        grid.sync();

        /* 2 times Inverse IFFT */
        if(bid == 0){
            // Load data from shared memory to registers
            {
                unsigned int index = offset + threadIdx.x;
                for (unsigned i = 0; i < IFFT::elements_per_thread; i++) {
                    thread_data[i] = complex_type {ct_CUDA[index].x, ct_CUDA[index].y};
                    // FFT::stride shows how elements from a single FFT should be split between threads
                    index += IFFT::stride;
                }
            }

            // Scale values
            double scale = 1.0 / cufftdx::size_of<IFFT>::value;
            for (unsigned int i = 0; i < IFFT::elements_per_thread; i++) {
                thread_data[i].x *= scale;
                thread_data[i].y *= scale;
            }

            IFFT().execute(thread_data, shared_mem);
        
            // Save results
            {
                unsigned int index = offset + threadIdx.x;
                for (unsigned i = 0; i < IFFT::elements_per_thread; i++) {
                    ct_CUDA[index].x = thread_data[i].x;
                    ct_CUDA[index].y = thread_data[i].y;
                    index += IFFT::stride;
                }
            }
        }
        grid.sync();

        /* Twisting and moding */
        // polynomial a
        for (uint32_t i = gtid; i < NHalf; i += gdim) {
            // twisting
            ct_CUDA[i] = cuCmul(ct_CUDA[i], twiddleTable_CUDA[i + NHalf]);
            // acc + round(ct)
            acc_CUDA[i].x += rint(ct_CUDA[i].x);
            acc_CUDA[i].y += rint(ct_CUDA[i].y);
            // Modulus Q
            if(modInt64){
                acc_CUDA[i].x = static_cast<double>(static_cast<int64_t>(acc_CUDA[i].x) % static_cast<int64_t>(Q));
                acc_CUDA[i].y = static_cast<double>(static_cast<int64_t>(acc_CUDA[i].y) % static_cast<int64_t>(Q));
            }
            else{
                acc_CUDA[i].x = static_cast<double>(static_cast<__int128_t>(acc_CUDA[i].x) % static_cast<__int128_t>(Q));
                acc_CUDA[i].y = static_cast<double>(static_cast<__int128_t>(acc_CUDA[i].y) % static_cast<__int128_t>(Q));
            }
            if (acc_CUDA[i].x < 0)
                acc_CUDA[i].x += static_cast<double>(Q);
            if (acc_CUDA[i].y < 0)
                acc_CUDA[i].y += static_cast<double>(Q);
        }
        // polynomial b
        for (uint32_t i = gtid + NHalf; i < N; i += gdim) {
            // twisting
            ct_CUDA[i] = cuCmul(ct_CUDA[i], twiddleTable_CUDA[i]);
            // acc + round(ct)
            acc_CUDA[i].x += rint(ct_CUDA[i].x);
            acc_CUDA[i].y += rint(ct_CUDA[i].y);
            // Modulus Q
            if(modInt64){
                acc_CUDA[i].x = static_cast<double>(static_cast<int64_t>(acc_CUDA[i].x) % static_cast<int64_t>(Q));
                acc_CUDA[i].y = static_cast<double>(static_cast<int64_t>(acc_CUDA[i].y) % static_cast<int64_t>(Q));
            }
            else{
                acc_CUDA[i].x = static_cast<double>(static_cast<__int128_t>(acc_CUDA[i].x) % static_cast<__int128_t>(Q));
                acc_CUDA[i].y = static_cast<double>(static_cast<__int128_t>(acc_CUDA[i].y) % static_cast<__int128_t>(Q));
            }
            if (acc_CUDA[i].x < 0)
                acc_CUDA[i].x += static_cast<double>(Q);
            if (acc_CUDA[i].y < 0)
                acc_CUDA[i].y += static_cast<double>(Q);
        }
        grid.sync();
    }

    /****************************************
    * the accumulator result is encrypted w.r.t. the transposed secret key
    * we can transpose "a" to get an encryption under the original secret key z
    * z = (z0, −zq/2−1, . . . , −z1)
    *****************************************/
    /* Copy acc_CUDA to ct_CUDA */
    for(uint32_t i = gtid; i < NHalf; i += gdim){
        ct_CUDA[i] = acc_CUDA[i];
    }
    grid.sync();

    for(uint32_t i = gtid+1; i < NHalf; i += gdim){
        acc_CUDA[i].x = static_cast<double>((Q - static_cast<uint64_t>(ct_CUDA[NHalf - i].y)));
        acc_CUDA[i].y = static_cast<double>((Q - static_cast<uint64_t>(ct_CUDA[NHalf - i].x)));
    }
    if(gtid == 0) acc_CUDA[0].y = static_cast<double>((Q - static_cast<uint64_t>(ct_CUDA[0].y)));
    grid.sync();
}

template<class FFT, class IFFT>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void bootstrappingSingleBlock(Complex_d* acc_CUDA, Complex_d* ct_CUDA, Complex_d* dct_CUDA, uint64_t* a_CUDA, 
        Complex_d* monomial_CUDA, Complex_d* twiddleTable_CUDA, Complex_d* GINX_bootstrappingKey_CUDA, uint64_t* keySwitchingkey_CUDA,
         uint64_t* params_CUDA, uint64_t fmod, uint32_t syncNum){
    
    /* GPU Parameters Set */
    uint32_t tid = ThisThreadRankInBlock();
    uint32_t bdim = ThisBlockSize();

    /* HE Parameters Set */
    const uint32_t n                = static_cast<uint32_t>(params_CUDA[0]);
    const uint32_t M                = static_cast<uint32_t>(params_CUDA[1] << 1);
    const uint32_t N                = static_cast<uint32_t>(params_CUDA[1]);
    const uint32_t NHalf            = N >> 1;
    const uint64_t Q                = params_CUDA[3];
    const uint64_t QHalf            = Q >> 1;
    const uint32_t logbaseG         = static_cast<uint32_t>(params_CUDA[4]);
    const uint32_t digitsG2         = static_cast<uint32_t>(params_CUDA[5]);
    const uint32_t RGSW_size        = digitsG2 * 2 * NHalf;
    const int32_t gBits             = logbaseG;
    const int32_t gBitsMaxBits      = 64 - gBits;
    const uint32_t numDigitsToThrow = static_cast<uint32_t>(params_CUDA[6]);
    const bool modInt64             = Q > 4294967296 ? false : true; // Q > 2^32 or not

    /* cufftdx variables */
    using complex_type = typename FFT::value_type;
    const unsigned int local_fft_id = threadIdx.y;
    const unsigned int offset = cufftdx::size_of<FFT>::value * (blockIdx.x * FFT::ffts_per_block + local_fft_id);
    extern __shared__ complex_type shared_mem[];
    complex_type thread_data[FFT::storage_size];     

    for(uint32_t round = 0; round < n; ++round){
        /* Approximate SignedDigitDecompose */
        for (size_t k = tid; k < N; k += bdim) { // 0~NHalf-1: a, NHalf~N-1: b
            const uint64_t& t1 = static_cast<uint64_t>(acc_CUDA[k].x);
            const uint64_t& t2 = static_cast<uint64_t>(acc_CUDA[k].y);
            int64_t d0 = (t1 < QHalf) ? static_cast<int64_t>(t1) : (static_cast<int64_t>(t1) - static_cast<int64_t>(Q));
            int64_t d1 = (t2 < QHalf) ? static_cast<int64_t>(t2) : (static_cast<int64_t>(t2) - static_cast<int64_t>(Q));
            int64_t r0, r1;

            // throw away the least significant digits
            for(int i = 0; i < numDigitsToThrow; i++){
                r0 = (d0 << gBitsMaxBits) >> gBitsMaxBits;
                d0 = (d0 - r0) >> gBits;
                r1 = (d1 << gBitsMaxBits) >> gBitsMaxBits;
                d1 = (d1 - r1) >> gBits;
            }

            for (size_t l = 0; l < digitsG2; l += 2) {
                r0 = (d0 << gBitsMaxBits) >> gBitsMaxBits;
                d0 = (d0 - r0) >> gBits;
                if (r0 < 0)
                    r0 += static_cast<int64_t>(Q);
                if (r0 >= QHalf)
                    r0 -= static_cast<int64_t>(Q);
                dct_CUDA[l*NHalf + k].x = static_cast<double>(r0);

                r1 = (d1 << gBitsMaxBits) >> gBitsMaxBits;
                d1 = (d1 - r1) >> gBits;
                if (r1 < 0)
                    r1 += static_cast<int64_t>(Q);
                if (r1 >= QHalf)
                    r1 -= static_cast<int64_t>(Q);
                dct_CUDA[l*NHalf + k].y = static_cast<double>(r1);
            }
        }
        __syncthreads();

        /* digitsG2 times Forward FFT */
        // Load data from shared memory to registers
        {
            unsigned int index = offset + threadIdx.x;
            unsigned int twist_idx = threadIdx.x;
            for (unsigned i = 0; i < FFT::elements_per_thread; i++) {
                // twisting
                dct_CUDA[index] = cuCmul(dct_CUDA[index], twiddleTable_CUDA[twist_idx]);
                thread_data[i] = complex_type {dct_CUDA[index].x, dct_CUDA[index].y};
                // FFT::stride shows how elements from a single FFT should be split between threads
                index += FFT::stride;
                twist_idx += FFT::stride;
            }
        }

        FFT().execute(thread_data, shared_mem);

        // Save results
        {
            unsigned int index = offset + threadIdx.x;
            for (unsigned i = 0; i < FFT::elements_per_thread; i++) {
                dct_CUDA[index] = make_cuDoubleComplex(thread_data[i].x, thread_data[i].y);
                // FFT::stride shows how elements from a single FFT should be split between threads
                index += FFT::stride;
            }
        }
        __syncthreads();

        /* Obtain monomial */
        // First obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
        auto aNeg         = (M - a_CUDA[round]) % M;
        uint64_t indexPos = a_CUDA[round];
        uint64_t indexNeg = aNeg;
        // index is in range [0,m] - so we need to adjust the edge case when
        // index = m to index = 0
        if (indexPos == M)
            indexPos = 0;
        if (indexNeg == M)
            indexNeg = 0;

        /* Initialize shared_mem */
        for(uint32_t i = tid; i < N; i += bdim){
            shared_mem[i] = complex_type {0.0, 0.0};
        }
        __syncthreads();

        /* ACC times Bootstrapping key and monomial */
        /* multiply with ek0 */
        // polynomial a
        for (uint32_t i = tid; i < NHalf; i += bdim){
            thread_data[0] = complex_type {0.0, 0.0};
            for (uint32_t l = 0; l < digitsG2; ++l){
                thread_data[0].x = fma(dct_CUDA[l*NHalf + i].x, GINX_bootstrappingKey_CUDA[(round << 1)*RGSW_size + (l << 1)*NHalf + i].x, thread_data[0].x);
                thread_data[0].x = fma(-dct_CUDA[l*NHalf + i].y, GINX_bootstrappingKey_CUDA[(round << 1)*RGSW_size + (l << 1)*NHalf + i].y, thread_data[0].x);
                thread_data[0].y = fma(dct_CUDA[l*NHalf + i].x, GINX_bootstrappingKey_CUDA[(round << 1)*RGSW_size + (l << 1)*NHalf + i].y, thread_data[0].y);
                thread_data[0].y = fma(dct_CUDA[l*NHalf + i].y, GINX_bootstrappingKey_CUDA[(round << 1)*RGSW_size + (l << 1)*NHalf + i].x, thread_data[0].y);
            }
            /* multiply with postive monomial */
            shared_mem[i].x = fma(thread_data[0].x, monomial_CUDA[indexPos*NHalf + i].x, shared_mem[i].x);
            shared_mem[i].x = fma(-thread_data[0].y, monomial_CUDA[indexPos*NHalf + i].y, shared_mem[i].x);
            shared_mem[i].y = fma(thread_data[0].x, monomial_CUDA[indexPos*NHalf + i].y, shared_mem[i].y);
            shared_mem[i].y = fma(thread_data[0].y, monomial_CUDA[indexPos*NHalf + i].x, shared_mem[i].y);
        }
        // polynomial b
        for (uint32_t i = tid; i < NHalf; i += bdim){
            thread_data[0] = complex_type {0.0, 0.0};
            for (uint32_t l = 0; l < digitsG2; ++l){
                thread_data[0].x = fma(dct_CUDA[l*NHalf + i].x, GINX_bootstrappingKey_CUDA[(round << 1)*RGSW_size + ((l << 1) + 1)*NHalf + i].x, thread_data[0].x);
                thread_data[0].x = fma(-dct_CUDA[l*NHalf + i].y, GINX_bootstrappingKey_CUDA[(round << 1)*RGSW_size + ((l << 1) + 1)*NHalf + i].y, thread_data[0].x);
                thread_data[0].y = fma(dct_CUDA[l*NHalf + i].x, GINX_bootstrappingKey_CUDA[(round << 1)*RGSW_size + ((l << 1) + 1)*NHalf + i].y, thread_data[0].y);
                thread_data[0].y = fma(dct_CUDA[l*NHalf + i].y, GINX_bootstrappingKey_CUDA[(round << 1)*RGSW_size + ((l << 1) + 1)*NHalf + i].x, thread_data[0].y);
            }
            /* multiply with postive monomial */
            shared_mem[NHalf + i].x = fma(thread_data[0].x, monomial_CUDA[indexPos*NHalf + i].x, shared_mem[NHalf + i].x);
            shared_mem[NHalf + i].x = fma(-thread_data[0].y, monomial_CUDA[indexPos*NHalf + i].y, shared_mem[NHalf + i].x);
            shared_mem[NHalf + i].y = fma(thread_data[0].x, monomial_CUDA[indexPos*NHalf + i].y, shared_mem[NHalf + i].y);
            shared_mem[NHalf + i].y = fma(thread_data[0].y, monomial_CUDA[indexPos*NHalf + i].x, shared_mem[NHalf + i].y);
        }

        /* multiply with ek1 */
        // polynomial a
        for (uint32_t i = tid; i < NHalf; i += bdim){
            thread_data[0] = complex_type {0.0, 0.0};
            for (uint32_t l = 0; l < digitsG2; ++l){
                thread_data[0].x = fma(dct_CUDA[l*NHalf + i].x, GINX_bootstrappingKey_CUDA[((round << 1) + 1)*RGSW_size + (l << 1)*NHalf + i].x, thread_data[0].x);
                thread_data[0].x = fma(-dct_CUDA[l*NHalf + i].y, GINX_bootstrappingKey_CUDA[((round << 1) + 1)*RGSW_size + (l << 1)*NHalf + i].y, thread_data[0].x);
                thread_data[0].y = fma(dct_CUDA[l*NHalf + i].x, GINX_bootstrappingKey_CUDA[((round << 1) + 1)*RGSW_size + (l << 1)*NHalf + i].y, thread_data[0].y);
                thread_data[0].y = fma(dct_CUDA[l*NHalf + i].y, GINX_bootstrappingKey_CUDA[((round << 1) + 1)*RGSW_size + (l << 1)*NHalf + i].x, thread_data[0].y);
            }
            /* multiply with negative monomial */
            shared_mem[i].x = fma(thread_data[0].x, monomial_CUDA[indexNeg*NHalf + i].x, shared_mem[i].x);
            shared_mem[i].x = fma(-thread_data[0].y, monomial_CUDA[indexNeg*NHalf + i].y, shared_mem[i].x);
            shared_mem[i].y = fma(thread_data[0].x, monomial_CUDA[indexNeg*NHalf + i].y, shared_mem[i].y);
            shared_mem[i].y = fma(thread_data[0].y, monomial_CUDA[indexNeg*NHalf + i].x, shared_mem[i].y);
        }
        // polynomial b
        for (uint32_t i = tid; i < NHalf; i += bdim){
            thread_data[0] = complex_type {0.0, 0.0};
            for (uint32_t l = 0; l < digitsG2; ++l){
                thread_data[0].x = fma(dct_CUDA[l*NHalf + i].x, GINX_bootstrappingKey_CUDA[((round << 1) + 1)*RGSW_size + ((l << 1) + 1)*NHalf + i].x, thread_data[0].x);
                thread_data[0].x = fma(-dct_CUDA[l*NHalf + i].y, GINX_bootstrappingKey_CUDA[((round << 1) + 1)*RGSW_size + ((l << 1) + 1)*NHalf + i].y, thread_data[0].x);
                thread_data[0].y = fma(dct_CUDA[l*NHalf + i].x, GINX_bootstrappingKey_CUDA[((round << 1) + 1)*RGSW_size + ((l << 1) + 1)*NHalf + i].y, thread_data[0].y);
                thread_data[0].y = fma(dct_CUDA[l*NHalf + i].y, GINX_bootstrappingKey_CUDA[((round << 1) + 1)*RGSW_size + ((l << 1) + 1)*NHalf + i].x, thread_data[0].y);
            }
            /* multiply with negative monomial */
            shared_mem[NHalf + i].x = fma(thread_data[0].x, monomial_CUDA[indexNeg*NHalf + i].x, shared_mem[NHalf + i].x);
            shared_mem[NHalf + i].x = fma(-thread_data[0].y, monomial_CUDA[indexNeg*NHalf + i].y, shared_mem[NHalf + i].x);
            shared_mem[NHalf + i].y = fma(thread_data[0].x, monomial_CUDA[indexNeg*NHalf + i].y, shared_mem[NHalf + i].y);
            shared_mem[NHalf + i].y = fma(thread_data[0].y, monomial_CUDA[indexNeg*NHalf + i].x, shared_mem[NHalf + i].y);
        }
        __syncthreads();

        /* 2 times Inverse IFFT */
        if(threadIdx.y < 2){
            // Load data from shared memory to registers
            {
                unsigned int index = offset + threadIdx.x;
                for (unsigned i = 0; i < IFFT::elements_per_thread; i++) {
                    thread_data[i] = shared_mem[index];
                    // FFT::stride shows how elements from a single FFT should be split between threads
                    index += IFFT::stride;
                }
            }

            // Scale values
            double scale = 1.0 / cufftdx::size_of<IFFT>::value;
            for (unsigned int i = 0; i < IFFT::elements_per_thread; i++) {
                thread_data[i].x *= scale;
                thread_data[i].y *= scale;
            }

            IFFT().execute(thread_data, shared_mem);
        
            // Save results
            {
                unsigned int index = offset + threadIdx.x;
                for (unsigned i = 0; i < IFFT::elements_per_thread; i++) {
                    shared_mem[index] = thread_data[i];
                    index += IFFT::stride;
                }
            }
        }
        else{ // must meet syncs made by IFFT
            for(uint32_t i = 0; i < syncNum; ++i)
                __syncthreads();
        }
        __syncthreads();

        /* Twisting and moding */
        // polynomial a
        for (uint32_t i = tid; i < NHalf; i += bdim) {
            // twisting
            thread_data[0].x = shared_mem[i].x* twiddleTable_CUDA[i + NHalf].x - shared_mem[i].y * twiddleTable_CUDA[i + NHalf].y;
            shared_mem[i].y = shared_mem[i].x* twiddleTable_CUDA[i + NHalf].y + shared_mem[i].y * twiddleTable_CUDA[i + NHalf].x;
            shared_mem[i].x = thread_data[0].x;
            // acc + round(ct)
            acc_CUDA[i].x += rint(shared_mem[i].x);
            acc_CUDA[i].y += rint(shared_mem[i].y);
            // Modulus Q
            if(modInt64){
                acc_CUDA[i].x = static_cast<double>(static_cast<int64_t>(acc_CUDA[i].x) % static_cast<int64_t>(Q));
                acc_CUDA[i].y = static_cast<double>(static_cast<int64_t>(acc_CUDA[i].y) % static_cast<int64_t>(Q));
            }
            else{
                acc_CUDA[i].x = static_cast<double>(static_cast<__int128_t>(acc_CUDA[i].x) % static_cast<__int128_t>(Q));
                acc_CUDA[i].y = static_cast<double>(static_cast<__int128_t>(acc_CUDA[i].y) % static_cast<__int128_t>(Q));
            }
            if (acc_CUDA[i].x < 0)
                acc_CUDA[i].x += static_cast<double>(Q);
            if (acc_CUDA[i].y < 0)
                acc_CUDA[i].y += static_cast<double>(Q);
        }
        // polynomial b
        for (uint32_t i = tid + NHalf; i < N; i += bdim) {
            // twisting
            thread_data[0].x = shared_mem[i].x* twiddleTable_CUDA[i].x - shared_mem[i].y * twiddleTable_CUDA[i].y;
            shared_mem[i].y = shared_mem[i].x* twiddleTable_CUDA[i].y + shared_mem[i].y * twiddleTable_CUDA[i].x;
            shared_mem[i].x = thread_data[0].x;
            // acc + round(ct)
            acc_CUDA[i].x += rint(shared_mem[i].x);
            acc_CUDA[i].y += rint(shared_mem[i].y);
            // Modulus Q
            if(modInt64){
                acc_CUDA[i].x = static_cast<double>(static_cast<int64_t>(acc_CUDA[i].x) % static_cast<int64_t>(Q));
                acc_CUDA[i].y = static_cast<double>(static_cast<int64_t>(acc_CUDA[i].y) % static_cast<int64_t>(Q));
            }
            else{
                acc_CUDA[i].x = static_cast<double>(static_cast<__int128_t>(acc_CUDA[i].x) % static_cast<__int128_t>(Q));
                acc_CUDA[i].y = static_cast<double>(static_cast<__int128_t>(acc_CUDA[i].y) % static_cast<__int128_t>(Q));
            }
            if (acc_CUDA[i].x < 0)
                acc_CUDA[i].x += static_cast<double>(Q);
            if (acc_CUDA[i].y < 0)
                acc_CUDA[i].y += static_cast<double>(Q);
        }
        __syncthreads();
    }

    /****************************************
    * the accumulator result is encrypted w.r.t. the transposed secret key
    * we can transpose "a" to get an encryption under the original secret key z
    * z = (z0, −zq/2−1, . . . , −z1)
    *****************************************/
    /* Copy acc_CUDA to shared_mem */
    for(uint32_t i = tid; i < NHalf; i += bdim){
        shared_mem[i] = complex_type {acc_CUDA[i].x, acc_CUDA[i].y};
    }
    __syncthreads();

    for(uint32_t i = tid+1; i < NHalf; i += bdim){
        acc_CUDA[i].x = static_cast<double>((Q - static_cast<uint64_t>(shared_mem[NHalf - i].y)));
        acc_CUDA[i].y = static_cast<double>((Q - static_cast<uint64_t>(shared_mem[NHalf - i].x)));
    }
    if(tid == 0) acc_CUDA[0].y = static_cast<double>((Q - static_cast<uint64_t>(shared_mem[0].y)));
    __syncthreads();
}

template<class FFT>
__global__ void cuFFTDxFWD(Complex_d* data, Complex_d* twiddleTable_CUDA){
    /* cufftdx variables */
    using complex_type = typename FFT::value_type;
    const unsigned int local_fft_id = threadIdx.y;
    const unsigned int offset = cufftdx::size_of<FFT>::value * (blockIdx.x + local_fft_id);
    extern __shared__ complex_type shared_mem[];
    complex_type thread_data[FFT::storage_size];     
    
    // Load data from shared memory to registers
    {
        unsigned int index = offset + threadIdx.x;
        unsigned int twist_idx = threadIdx.x;
        for (unsigned i = 0; i < FFT::elements_per_thread; i++) {
            // twisting
            data[index] = cuCmul(data[index], twiddleTable_CUDA[twist_idx]);
            thread_data[i] = complex_type {data[index].x, data[index].y};
            // FFT::stride shows how elements from a single FFT should be split between threads
            index += FFT::stride;
            twist_idx += FFT::stride;
        }
    }

    FFT().execute(thread_data, shared_mem);

    // Save results
    {
        unsigned int index = offset + threadIdx.x;
        for (unsigned i = 0; i < FFT::elements_per_thread; i++) {
            data[index] = make_cuDoubleComplex(thread_data[i].x, thread_data[i].y);
            // FFT::stride shows how elements from a single FFT should be split between threads
            index += FFT::stride;
        }
    }
}

void GPUFFTBootstrap::GPUSetup(const std::shared_ptr<BinFHECryptoParams> params, RingGSWACCKey BSkey, LWESwitchingKey KSkey, int numGPUs)
{
    /* Setting up available GPU INFO */
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return;
    }

    /* Determine the number of GPUs to use*/
    int GPUcount;
    if(numGPUs > 0 && numGPUs <= deviceCount) GPUcount = numGPUs;
    else GPUcount = deviceCount;

    for (int device = 0; device < GPUcount; ++device) {
        cudaSetDevice(device);
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, device);

        GPUInfo info;
        info.name = deviceProperties.name;
        info.major = deviceProperties.major;
        info.minor = deviceProperties.minor;
        info.sharedMemoryPerBlock = deviceProperties.sharedMemPerBlock;
        info.maxBlocksPerMultiprocessor = deviceProperties.maxThreadsPerMultiProcessor / deviceProperties.maxThreadsPerBlock;
        info.maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;
        info.maxGridX = deviceProperties.maxGridSize[0];
        info.maxGridY = deviceProperties.maxGridSize[1];
        info.maxGridZ = deviceProperties.maxGridSize[2];
        info.maxBlockX = deviceProperties.maxThreadsDim[0];
        info.maxBlockY = deviceProperties.maxThreadsDim[1];
        info.maxBlockZ = deviceProperties.maxThreadsDim[2];
        info.warpSize = deviceProperties.warpSize;
        info.multiprocessorCount = deviceProperties.multiProcessorCount;

        gpuInfoList.push_back(info);
        std::cout << "GPU " << device << ": " << info.name << " (SM " << info.major << "." << info.minor << ")" << std::endl;
    }

    /* Parameters Set */
    auto RGSWParams     = params->GetRingGSWParams();
    uint32_t NHalf      = RGSWParams->GetN() >> 1;
    uint32_t arch       = gpuInfoList[0].major * 100 + gpuInfoList[0].minor * 10;

    /* Determine template of GPUSetup_core */
    switch (arch){
        case 700: // V100
            switch (NHalf){
                case 512:
                    GPUSetup_core<700, 512>(params, BSkey, KSkey);
                    break;
                case 1024:
                    GPUSetup_core<700, 1024>(params, BSkey, KSkey);
                    break;
                case 2048:
                    GPUSetup_core<700, 2048>(params, BSkey, KSkey);
                    break;
                case 4096:
                    GPUSetup_core<700, 4096>(params, BSkey, KSkey);
                    break;
                default:
                    std::cerr << "Unsupported N, we support N = 1024, 2048, 4096, 8192\n";
                    exit(1);
            }
            break;
        case 800: // A100
            switch (NHalf){
                case 512:
                    GPUSetup_core<800, 512>(params, BSkey, KSkey);
                    break;
                case 1024:
                    GPUSetup_core<800, 1024>(params, BSkey, KSkey);
                    break;
                case 2048:
                    GPUSetup_core<800, 2048>(params, BSkey, KSkey);
                    break;
                case 4096:
                    GPUSetup_core<800, 4096>(params, BSkey, KSkey);
                    break;
                default:
                    std::cerr << "Unsupported N, we support N = 1024, 2048, 4096, 8192\n";
                    exit(1);
            }
            break;
        case 860: // RTX30 series
            switch (NHalf){
                case 512:
                    GPUSetup_core<860, 512>(params, BSkey, KSkey);
                    break;
                case 1024:
                    GPUSetup_core<860, 1024>(params, BSkey, KSkey);
                    break;
                case 2048:
                    GPUSetup_core<860, 2048>(params, BSkey, KSkey);
                    break;
                case 4096:
                    GPUSetup_core<860, 4096>(params, BSkey, KSkey);
                    break;
                default:
                    std::cerr << "Unsupported N, we support N = 1024, 2048, 4096, 8192\n";
                    exit(1);
            }
            break;
        case 890: // RTX40 series
            switch (NHalf){
                case 512:
                    GPUSetup_core<890, 512>(params, BSkey, KSkey);
                    break;
                case 1024:
                    GPUSetup_core<890, 1024>(params, BSkey, KSkey);
                    break;
                case 2048:
                    GPUSetup_core<890, 2048>(params, BSkey, KSkey);
                    break;
                case 4096:
                    GPUSetup_core<890, 4096>(params, BSkey, KSkey);
                    break;
                default:
                    std::cerr << "Unsupported N, we support N = 1024, 2048, 4096, 8192\n";
                    exit(1);
            }
            break;
        case 900: // H100
            switch (NHalf){
                case 512:
                    GPUSetup_core<900, 512>(params, BSkey, KSkey);
                    break;
                case 1024:
                    GPUSetup_core<900, 1024>(params, BSkey, KSkey);
                    break;
                case 2048:
                    GPUSetup_core<900, 2048>(params, BSkey, KSkey);
                    break;
                case 4096:
                    GPUSetup_core<900, 4096>(params, BSkey, KSkey);
                    break;
                default:
                    std::cerr << "Unsupported N, we support N = 1024, 2048, 4096, 8192\n";
                    exit(1);
            }
            break;
        default:
            std::cerr << "Unsupported GPU architecture, we support compute capability = 700, 800, 860, 890, 900\n";
            exit(1);
    }
}

template<uint32_t arch, uint32_t FFT_dimension>
void GPUFFTBootstrap::GPUSetup_core(const std::shared_ptr<BinFHECryptoParams> params, RingGSWACCKey BSkey, LWESwitchingKey KSkey)
{
    /* HE Parameters Set */
    auto RGSWParams             = params->GetRingGSWParams();
    auto LWEParams              = params->GetLWEParams();
    NativeInteger Q             = RGSWParams->GetQ();
    NativeInteger QHalf         = Q >> 1;
    int64_t Q_int               = Q.ConvertToInt();
    NativeInteger q             = LWEParams->Getq();
    int64_t q_int               = q.ConvertToInt();
    uint32_t N                  = RGSWParams->GetN();
    uint32_t NHalf              = N >> 1;
    uint32_t n                  = LWEParams->Getn();
    uint32_t numDigitsToThrow   = RGSWParams->GetNumDigitsToThrow();
    uint32_t digitsG2           = (RGSWParams->GetDigitsG() - numDigitsToThrow) << 1;
    uint32_t baseG              = RGSWParams->GetBaseG();
    uint32_t logbaseG           = static_cast<uint32_t>(std::log2(baseG));
    uint32_t RGSW_size          = digitsG2 * 2 * NHalf;
    NativeInteger qKS           = LWEParams->GetqKS();
    int64_t qKS_int             = qKS.ConvertToInt();
    uint32_t baseKS             = LWEParams->GetBaseKS();
    uint32_t digitCountKS       = (uint32_t)std::ceil(log(qKS.ConvertToDouble()) / log(static_cast<double>(baseKS)));
    uint32_t max_n_N            = n > N ? n : N;

    /* GPU settings */
    int GPU_num  = gpuInfoList.size();
    int SM_count = gpuInfoList[0].multiprocessorCount;

    /* Allocate host side memory for acc_host and ctExt_host */
    cudaMallocHost((void**)&acc_host, max_bootstapping_num * 2 * NHalf * sizeof(Complex), cudaHostAllocDefault);
    cudaMallocHost((void**)&ctExt_host, max_bootstapping_num * (max_n_N + 1) * sizeof(uint64_t), cudaHostAllocDefault);

    /* Initialize twiddle table */
    Complex *twiddleTable;
    cudaMallocHost((void**)&twiddleTable, 2 * NHalf * sizeof(Complex));
    for (size_t j = 0; j < NHalf; j++) {
        twiddleTable[j] = Complex(cos(static_cast<double>(2 * M_PI * j)/ (N << 1)), sin(static_cast<double>(2 * M_PI * j) / (N << 1)));
    }
    for (size_t j = NHalf; j < N; j++) {
        twiddleTable[j] = Complex(cos(static_cast<double>(-2 * M_PI * (j - NHalf)) / (N << 1)), sin(static_cast<double>(-2 * M_PI * (j - NHalf)) / (N << 1)));
    }

    /* Initialize params_CUDA */
    uint64_t *parameters;
    cudaMallocHost((void**)&parameters, 10 * sizeof(uint64_t));
    parameters[0] = n;
    parameters[1] = N;
    parameters[2] = q_int;
    parameters[3] = Q_int;
    parameters[4] = logbaseG;
    parameters[5] = digitsG2;
    parameters[6] = numDigitsToThrow;
    parameters[7] = qKS_int;
    parameters[8] = baseKS;
    parameters[9] = digitCountKS;
    
    /* Initialize bootstrapping key */
    // construct bootstrapping key for FFT-based accumulator
    auto bootstrappingKey_FFT = std::make_shared<std::vector<std::vector<std::vector<std::shared_ptr<std::vector<std::vector<std::vector<Complex>>>>>>>>();
    (*bootstrappingKey_FFT).resize(1);
    for (size_t i = 0; i < 1; ++i) {
        (*bootstrappingKey_FFT)[i].resize(2);
        for (size_t j = 0; j < 2; ++j) {
            (*bootstrappingKey_FFT)[i][j].resize(n);
        }
    }
    for (size_t i = 0; i < n; ++i) {
        (*bootstrappingKey_FFT)[0][0][i] = KeyCopy_FFT(RGSWParams, (*BSkey)[0][0][i]);
        (*bootstrappingKey_FFT)[0][1][i] = KeyCopy_FFT(RGSWParams, (*BSkey)[0][1][i]);
    }
    // serialize bootstrapping key into host memory
    Complex *bootstrappingKey_host;
    cudaMallocHost((void**)&bootstrappingKey_host, 2 * n * RGSW_size * sizeof(Complex)); // ternery needs two secret keys
    for(int i = 0; i < n; i++){
        for(int num_key = 0; num_key < 2; num_key++){
            for(int l = 0; l < digitsG2; l++){
                for(int m = 0; m < 2; m++){
                    std::vector<Complex> temp = (*(*bootstrappingKey_FFT)[0][num_key][i])[l][m];
                    for(int j = 0; j < NHalf; j++){
                        bootstrappingKey_host[i*2*RGSW_size + num_key*RGSW_size + l*2*NHalf + m*NHalf + j] = Complex(temp[j].real(), temp[j + NHalf].real());
                    }
                }
            }
        }
    }

    /* Initialize keySwitching key */
    uint64_t *keySwitchingkey_host;
    cudaMallocHost((void**)&keySwitchingkey_host, N * baseKS * digitCountKS * (n + 1) * sizeof(uint64_t));
    for(int i = 0; i < N; i++){
        for(int j = 0; j < baseKS; j++){
            for(int k = 0; k < digitCountKS; k++){
                for(int l = 0; l < n; l++){
                    keySwitchingkey_host[i*baseKS*digitCountKS*(n + 1) + j*digitCountKS*(n + 1) + k*(n + 1) + l] 
                        = static_cast<uint64_t>(KSkey->GetElementsA()[i][j][k][l].ConvertToInt());
                }
                keySwitchingkey_host[i*baseKS*digitCountKS*(n + 1) + j*digitCountKS*(n + 1) + k*(n + 1) + n] 
                    = static_cast<uint64_t>(KSkey->GetElementsB()[i][j][k].ConvertToInt());
            }
        }
    }

    /* Initialize monomial array */
    Complex *monomial_arr;
    cudaMallocHost((void**)&monomial_arr, 2 * N * NHalf * sizeof(Complex));
    // loop for positive values of m
    std::vector<Complex> monomial(N, Complex(0.0, 0.0));
    for (size_t m_count = 0; m_count < N; ++m_count) {
        NativePoly monomial_t    = RGSWParams->GetMonomial(m_count);
        monomial_t.SetFormat(Format::COEFFICIENT);
        for (size_t i = 0; i < N; ++i) {
            NativeInteger::SignedNativeInt d = (monomial_t[i] < QHalf) ? monomial_t[i].ConvertToInt() : (monomial_t[i].ConvertToInt() - Q_int);
            monomial[i] = Complex (static_cast<double>(d), 0);
        }
        for (size_t i = 0; i < NHalf; ++i) 
            monomial_arr[m_count*NHalf + i] = Complex(monomial[i].real(), monomial[i + NHalf].real());
    }
    // loop for negative values of m
    std::vector<Complex> monomialNeg(N, Complex(0.0, 0.0));
    for (size_t m_count = N; m_count < (N << 1); ++m_count) {   
        NativePoly monomialNeg_t = RGSWParams->GetMonomial(m_count);
        monomialNeg_t.SetFormat(Format::COEFFICIENT);
        for (size_t i = 0; i < N; ++i) {
            NativeInteger::SignedNativeInt d = (monomialNeg_t[i] < QHalf) ? monomialNeg_t[i].ConvertToInt() : (monomialNeg_t[i].ConvertToInt() - Q_int);
            monomialNeg[i] = Complex (static_cast<double>(d), 0);
        }
        for (size_t i = 0; i < NHalf; ++i) 
            monomial_arr[m_count*NHalf + i] = Complex(monomialNeg[i].real(), monomialNeg[i + NHalf].real());
    }

    /* Allocate memory and send to multiple GPUs */
    GPUVec.resize(GPU_num);
    for(int g = 0; g < GPU_num; g++){
        cudaSetDevice(g);
        
        /* Configure cuFFTDx */
        using FFT_fwd  = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::forward>() + cufftdx::ElementsPerThread<8>() +
                            cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<1>() + cufftdx::SM<arch>());

        /* Increase max shared memory */
        auto it = sharedMemMap.find(arch);
        int maxSharedMemoryAvail = it->second * 1024;
        // MKMSwitch shared memory size
        int MKMSwitch_shared_memory_size = (N + 1) * sizeof(uint64_t);
        if(MKMSwitch_shared_memory_size < maxSharedMemoryAvail){
            CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(MKMSwitchKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MKMSwitch_shared_memory_size));
        }
        // cuFFTDx Forward shared memory size
        if(FFT_fwd::shared_memory_size < maxSharedMemoryAvail){
            CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(cuFFTDxFWD<FFT_fwd>, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT_fwd::shared_memory_size));
        }

        /* Create cuda streams */
        GPUVec[g].streams.resize(SM_count);
        for (int s = 0; s < SM_count; s++) {
            cudaStreamCreate(&GPUVec[g].streams[s]);
        }

        // Bring twiddle table to GPU
        cudaMalloc(&GPUVec[g].twiddleTable_CUDA, 2 * NHalf * sizeof(Complex_d));
        cudaMemcpy(GPUVec[g].twiddleTable_CUDA, twiddleTable, 2 * NHalf * sizeof(Complex_d), cudaMemcpyHostToDevice);

        // Bring params_CUDA to GPU
        cudaMalloc(&GPUVec[g].params_CUDA, 10 * sizeof(uint64_t));
        cudaMemcpy(GPUVec[g].params_CUDA, parameters, 10 * sizeof(uint64_t), cudaMemcpyHostToDevice);

        // Bring bootstrapping key to GPU
        cudaMalloc(&GPUVec[g].GINX_bootstrappingKey_CUDA, 2 * n * RGSW_size * sizeof(Complex_d));
        cudaMemcpy(GPUVec[g].GINX_bootstrappingKey_CUDA, bootstrappingKey_host, 2 * n * RGSW_size * sizeof(Complex_d), cudaMemcpyHostToDevice);
        cuFFTDxFWD<FFT_fwd><<<2 * n * digitsG2 * 2, FFT_fwd::block_dim, FFT_fwd::shared_memory_size>>>(GPUVec[g].GINX_bootstrappingKey_CUDA, GPUVec[g].twiddleTable_CUDA);

        // Bring keySwitching key to GPU
        cudaMalloc(&GPUVec[g].keySwitchingkey_CUDA, N * baseKS * digitCountKS * (n + 1) * sizeof(uint64_t));
        cudaMemcpy(GPUVec[g].keySwitchingkey_CUDA, keySwitchingkey_host, N * baseKS * digitCountKS * (n + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice);

        // Bring monomial array to GPU
        cudaMalloc(&GPUVec[g].monomial_CUDA, 2 * N * NHalf * sizeof(Complex_d));
        cudaMemcpy(GPUVec[g].monomial_CUDA, monomial_arr, 2 * N * NHalf * sizeof(Complex_d), cudaMemcpyHostToDevice);
        cuFFTDxFWD<FFT_fwd><<<2 * N, FFT_fwd::block_dim, FFT_fwd::shared_memory_size>>>(GPUVec[g].monomial_CUDA, GPUVec[g].twiddleTable_CUDA);

        /* Allocate ct_CUDA on GPU */
        cudaMalloc(&GPUVec[g].ct_CUDA, SM_count * 2 * NHalf * sizeof(Complex_d));

        /* Allocate dct_CUDA on GPU */
        cudaMalloc(&GPUVec[g].dct_CUDA, SM_count * digitsG2 * NHalf * sizeof(Complex_d));

        /* Allocate acc_CUDA on GPU */
        cudaMalloc(&GPUVec[g].acc_CUDA, SM_count * 2 * NHalf * sizeof(Complex_d));

        /* Allocate a_CUDA on GPU */
        cudaMalloc(&GPUVec[g].a_CUDA, SM_count * n * sizeof(uint64_t));

        /* Allocate ctExt_CUDA on GPU */
        cudaMalloc(&GPUVec[g].ctExt_CUDA, SM_count * (max_n_N + 1) * sizeof(uint64_t));
    }

    /* Synchronize all GPUs */
    for(int g = 0; g < GPU_num; g++){
        cudaSetDevice(g);
        cudaDeviceSynchronize();
    }

    /* Free all host memories */
    cudaFreeHost(twiddleTable);
    cudaFreeHost(parameters);
    cudaFreeHost(bootstrappingKey_host);
    cudaFreeHost(keySwitchingkey_host);
    cudaFreeHost(monomial_arr);
}

void GPUFFTBootstrap::GPUClean(){
    int GPU_num  = gpuInfoList.size();
    for(int g = 0; g < GPU_num; g++){
        cudaSetDevice(g);
        cudaFree(GPUVec[g].GINX_bootstrappingKey_CUDA);
        cudaFree(GPUVec[g].keySwitchingkey_CUDA);
        cudaFree(GPUVec[g].monomial_CUDA);
        cudaFree(GPUVec[g].twiddleTable_CUDA);
        cudaFree(GPUVec[g].params_CUDA);
        cudaFree(GPUVec[g].ct_CUDA);
        cudaFree(GPUVec[g].dct_CUDA);
        cudaFree(GPUVec[g].acc_CUDA);
        cudaFree(GPUVec[g].a_CUDA);
        cudaFree(GPUVec[g].ctExt_CUDA);
        for (int s = 0; s < gpuInfoList[g].multiprocessorCount; s++) {
            cudaStreamDestroy(GPUVec[g].streams[s]);
        }
    }

    /* Free host memory */     
    cudaFreeHost(acc_host);
    cudaFreeHost(ctExt_host);

    gpuInfoList.clear();
    GPUVec.clear();
}

std::shared_ptr<std::vector<std::vector<std::vector<Complex>>>> GPUFFTBootstrap::KeyCopy_FFT(const std::shared_ptr<RingGSWCryptoParams> params, 
    RingGSWEvalKey ek){

    NativeInteger Q                         = params->GetQ();
    NativeInteger QHalf                     = Q >> 1;
    NativeInteger::SignedNativeInt Q_int    = Q.ConvertToInt();
    uint32_t numDigitsToThrow               = params->GetNumDigitsToThrow();
    uint32_t digitsG2                       = (params->GetDigitsG() - numDigitsToThrow) << 1;
    uint32_t N                              = params->GetN();

    auto ek_d = std::make_shared<std::vector<std::vector<std::vector<Complex>>>>
        (digitsG2, std::vector<std::vector<Complex>>(2, std::vector<Complex>(N, Complex(0.0, 0.0))));

    for (size_t j = 0; j < digitsG2; ++j) {
        for (size_t k = 0; k < 2; ++k) {
            NativePoly ek_t = (*ek)[j][k];
            ek_t.SetFormat(Format::COEFFICIENT);
            for (size_t l = 0; l < N; ++l) {
                NativeInteger::SignedNativeInt d = (ek_t[l] < QHalf) ? ek_t[l].ConvertToInt() : (ek_t[l].ConvertToInt() - Q_int);
                (*ek_d)[j][k][l].real(static_cast<double>(d));
            }
        }
    }

    return ek_d;
}

void GPUFFTBootstrap::EvalAcc_CUDA(const std::shared_ptr<RingGSWCryptoParams> params, const std::vector<NativeVector>& a, 
        std::shared_ptr<std::vector<RLWECiphertext>> acc, uint64_t fmod)
{   
    /* Parameters Set */
    uint32_t N                  = params->GetN();
    uint32_t NHalf              = N >> 1;
    uint32_t numDigitsToThrow   = params->GetNumDigitsToThrow();
    uint32_t digitsG2           = (params->GetDigitsG() - numDigitsToThrow) << 1;
    uint32_t arch               = gpuInfoList[0].major * 100 + gpuInfoList[0].minor * 10;

    /* Determine mode of EvalAcc_CUDA */
    std::string mode = "SINGLE";
    if((NHalf*digitsG2/8) > 1024){ // 1024 is the maximum number of threads per block
        mode = "MULTI";
    }
    auto sharedMem_it = sharedMemMap.find(arch);
    int maxSharedMemoryAvail = sharedMem_it->second * 1024;
    int shared_mem_size = max(NHalf*digitsG2*8, N * 16); // N*16 shared memory is used for RLWE X RGSW
    if(shared_mem_size > maxSharedMemoryAvail){ // exceed the maximum shared memory per block
        mode = "MULTI";
    }

    if(mode == "SINGLE"){
        switch (arch){
            case 700: // V100
                switch (NHalf){
                    case 512:
                        switch (digitsG2){
                            case 2:
                                AddToAccCGGI_CUDA_single<700, 512, 2>(params, a, acc, fmod);
                                break;
                            case 4:
                                AddToAccCGGI_CUDA_single<700, 512, 4>(params, a, acc, fmod);
                                break;
                            case 6:
                                AddToAccCGGI_CUDA_single<700, 512, 6>(params, a, acc, fmod);
                                break;
                            case 8:
                                AddToAccCGGI_CUDA_single<700, 512, 8>(params, a, acc, fmod);
                                break;
                            case 10:
                                AddToAccCGGI_CUDA_single<700, 512, 10>(params, a, acc, fmod);
                                break;
                            case 12:
                                AddToAccCGGI_CUDA_single<700, 512, 12>(params, a, acc, fmod);
                                break;
                            case 14:
                                AddToAccCGGI_CUDA_single<700, 512, 14>(params, a, acc, fmod);
                                break;
                            case 16:
                                AddToAccCGGI_CUDA_single<700, 512, 16>(params, a, acc, fmod);
                                break;
                            default:
                                std::cerr << "Unsupported digitsG in Single block mode\n";
                                exit(1);
                        }
                        break;
                    case 1024:
                        switch (digitsG2){
                            case 2:
                                AddToAccCGGI_CUDA_single<700, 1024, 2>(params, a, acc, fmod);
                                break;
                            case 4:
                                AddToAccCGGI_CUDA_single<700, 1024, 4>(params, a, acc, fmod);
                                break;
                            case 6:
                                AddToAccCGGI_CUDA_single<700, 1024, 6>(params, a, acc, fmod);
                                break;
                            case 8:
                                AddToAccCGGI_CUDA_single<700, 1024, 8>(params, a, acc, fmod);
                                break;
                            default:
                                std::cerr << "Unsupported digitsG in Single block mode\n";
                                exit(1);
                        }
                        break;
                    default:
                        std::cerr << "Unsupported N in Single block mode\n";
                        exit(1);
                }
                break;
            case 800: // A100
                switch (NHalf){
                    case 512:
                        switch (digitsG2){
                            case 2:
                                AddToAccCGGI_CUDA_single<800, 512, 2>(params, a, acc, fmod);
                                break;
                            case 4:
                                AddToAccCGGI_CUDA_single<800, 512, 4>(params, a, acc, fmod);
                                break;
                            case 6:
                                AddToAccCGGI_CUDA_single<800, 512, 6>(params, a, acc, fmod);
                                break;
                            case 8:
                                AddToAccCGGI_CUDA_single<800, 512, 8>(params, a, acc, fmod);
                                break;
                            case 10:
                                AddToAccCGGI_CUDA_single<800, 512, 10>(params, a, acc, fmod);
                                break;
                            case 12:
                                AddToAccCGGI_CUDA_single<800, 512, 12>(params, a, acc, fmod);
                                break;
                            case 14:
                                AddToAccCGGI_CUDA_single<800, 512, 14>(params, a, acc, fmod);
                                break;
                            case 16:
                                AddToAccCGGI_CUDA_single<800, 512, 16>(params, a, acc, fmod);
                                break;
                            default:
                                std::cerr << "Unsupported digitsG in Single block mode\n";
                                exit(1);
                        }
                        break;
                    case 1024:
                        switch (digitsG2){
                            case 2:
                                AddToAccCGGI_CUDA_single<800, 1024, 2>(params, a, acc, fmod);
                                break;
                            case 4:
                                AddToAccCGGI_CUDA_single<800, 1024, 4>(params, a, acc, fmod);
                                break;
                            case 6:
                                AddToAccCGGI_CUDA_single<800, 1024, 6>(params, a, acc, fmod);
                                break;
                            case 8:
                                AddToAccCGGI_CUDA_single<800, 1024, 8>(params, a, acc, fmod);
                                break;
                            default:
                                std::cerr << "Unsupported digitsG in Single block mode\n";
                                exit(1);
                        }
                        break;
                    default:
                        std::cerr << "Unsupported N in Single block mode\n";
                        exit(1);
                }
                break;
            case 860: // RTX30 series
                switch (NHalf){
                    case 512:
                        switch (digitsG2){
                            case 2:
                                AddToAccCGGI_CUDA_single<860, 512, 2>(params, a, acc, fmod);
                                break;
                            case 4:
                                AddToAccCGGI_CUDA_single<860, 512, 4>(params, a, acc, fmod);
                                break;
                            case 6:
                                AddToAccCGGI_CUDA_single<860, 512, 6>(params, a, acc, fmod);
                                break;
                            case 8:
                                AddToAccCGGI_CUDA_single<860, 512, 8>(params, a, acc, fmod);
                                break;
                            case 10:
                                AddToAccCGGI_CUDA_single<860, 512, 10>(params, a, acc, fmod);
                                break;
                            case 12:
                                AddToAccCGGI_CUDA_single<860, 512, 12>(params, a, acc, fmod);
                                break;
                            case 14:
                                AddToAccCGGI_CUDA_single<860, 512, 14>(params, a, acc, fmod);
                                break;
                            case 16:
                                AddToAccCGGI_CUDA_single<860, 512, 16>(params, a, acc, fmod);
                                break;
                            default:
                                std::cerr << "Unsupported digitsG in Single block mode\n";
                                exit(1);
                        }
                        break;
                    case 1024:
                        switch (digitsG2){
                            case 2:
                                AddToAccCGGI_CUDA_single<860, 1024, 2>(params, a, acc, fmod);
                                break;
                            case 4:
                                AddToAccCGGI_CUDA_single<860, 1024, 4>(params, a, acc, fmod);
                                break;
                            case 6:
                                AddToAccCGGI_CUDA_single<860, 1024, 6>(params, a, acc, fmod);
                                break;
                            case 8:
                                AddToAccCGGI_CUDA_single<860, 1024, 8>(params, a, acc, fmod);
                                break;
                            default:
                                std::cerr << "Unsupported digitsG in Single block mode\n";
                                exit(1);
                        }
                        break;
                    default:
                        std::cerr << "Unsupported N in Single block mode\n";
                        exit(1);
                }
                break;
            case 890: // RTX40 series
                switch (NHalf){
                    case 512:
                        switch (digitsG2){
                            case 2:
                                AddToAccCGGI_CUDA_single<890, 512, 2>(params, a, acc, fmod);
                                break;
                            case 4:
                                AddToAccCGGI_CUDA_single<890, 512, 4>(params, a, acc, fmod);
                                break;
                            case 6:
                                AddToAccCGGI_CUDA_single<890, 512, 6>(params, a, acc, fmod);
                                break;
                            case 8:
                                AddToAccCGGI_CUDA_single<890, 512, 8>(params, a, acc, fmod);
                                break;
                            case 10:
                                AddToAccCGGI_CUDA_single<890, 512, 10>(params, a, acc, fmod);
                                break;
                            case 12:
                                AddToAccCGGI_CUDA_single<890, 512, 12>(params, a, acc, fmod);
                                break;
                            case 14:
                                AddToAccCGGI_CUDA_single<890, 512, 14>(params, a, acc, fmod);
                                break;
                            case 16:
                                AddToAccCGGI_CUDA_single<890, 512, 16>(params, a, acc, fmod);
                                break;
                            default:
                                std::cerr << "Unsupported digitsG in Single block mode\n";
                                exit(1);
                        }
                        break;
                    case 1024:
                        switch (digitsG2){
                            case 2:
                                AddToAccCGGI_CUDA_single<890, 1024, 2>(params, a, acc, fmod);
                                break;
                            case 4:
                                AddToAccCGGI_CUDA_single<890, 1024, 4>(params, a, acc, fmod);
                                break;
                            case 6:
                                AddToAccCGGI_CUDA_single<890, 1024, 6>(params, a, acc, fmod);
                                break;
                            case 8:
                                AddToAccCGGI_CUDA_single<890, 1024, 8>(params, a, acc, fmod);
                                break;
                            default:
                                std::cerr << "Unsupported digitsG in Single block mode\n";
                                exit(1);
                        }
                        break;
                    default:
                        std::cerr << "Unsupported N in Single block mode\n";
                        exit(1);
                }
                break;
            case 900: // H100
                switch (NHalf){
                    case 512:
                        switch (digitsG2){
                            case 2:
                                AddToAccCGGI_CUDA_single<900, 512, 2>(params, a, acc, fmod);
                                break;
                            case 4:
                                AddToAccCGGI_CUDA_single<900, 512, 4>(params, a, acc, fmod);
                                break;
                            case 6:
                                AddToAccCGGI_CUDA_single<900, 512, 6>(params, a, acc, fmod);
                                break;
                            case 8:
                                AddToAccCGGI_CUDA_single<900, 512, 8>(params, a, acc, fmod);
                                break;
                            case 10:
                                AddToAccCGGI_CUDA_single<900, 512, 10>(params, a, acc, fmod);
                                break;
                            case 12:
                                AddToAccCGGI_CUDA_single<900, 512, 12>(params, a, acc, fmod);
                                break;
                            case 14:
                                AddToAccCGGI_CUDA_single<900, 512, 14>(params, a, acc, fmod);
                                break;
                            case 16:
                                AddToAccCGGI_CUDA_single<900, 512, 16>(params, a, acc, fmod);
                                break;
                            default:
                                std::cerr << "Unsupported digitsG in Single block mode\n";
                                exit(1);
                        }
                        break;
                    case 1024:
                        switch (digitsG2){
                            case 2:
                                AddToAccCGGI_CUDA_single<900, 1024, 2>(params, a, acc, fmod);
                                break;
                            case 4:
                                AddToAccCGGI_CUDA_single<900, 1024, 4>(params, a, acc, fmod);
                                break;
                            case 6:
                                AddToAccCGGI_CUDA_single<900, 1024, 6>(params, a, acc, fmod);
                                break;
                            case 8:
                                AddToAccCGGI_CUDA_single<900, 1024, 8>(params, a, acc, fmod);
                                break;
                            default:
                                std::cerr << "Unsupported digitsG in Single block mode\n";
                                exit(1);
                        }
                        break;
                    default:
                        std::cerr << "Unsupported N in Single block mode\n";
                        exit(1);
                }
                break;
            default:
                std::cerr << "Unsupported GPU architecture\n";
                exit(1);
        }
    }
    else if(mode == "MULTI"){
        switch (arch){
            case 700: // V100
                switch (NHalf){
                    case 512:
                        AddToAccCGGI_CUDA_multi<700, 512>(params, a, acc, fmod);
                        break;
                    case 1024:
                        AddToAccCGGI_CUDA_multi<700, 1024>(params, a, acc, fmod);
                        break;
                    case 2048:
                        AddToAccCGGI_CUDA_multi<700, 2048>(params, a, acc, fmod);
                        break;
                    case 4096:
                        AddToAccCGGI_CUDA_multi<700, 4096>(params, a, acc, fmod);
                        break;
                    default:
                        std::cerr << "Unsupported N, we support N = 1024, 2048, 4096, 8192\n";
                        exit(1);
                }
                break;
            case 800: // A100
                switch (NHalf){
                    case 512:
                        AddToAccCGGI_CUDA_multi<800, 512>(params, a, acc, fmod);
                        break;
                    case 1024:
                        AddToAccCGGI_CUDA_multi<800, 1024>(params, a, acc, fmod);
                        break;
                    case 2048:
                        AddToAccCGGI_CUDA_multi<800, 2048>(params, a, acc, fmod);
                        break;
                    case 4096:
                        AddToAccCGGI_CUDA_multi<800, 4096>(params, a, acc, fmod);
                        break;
                    default:
                        std::cerr << "Unsupported N, we support N = 1024, 2048, 4096, 8192\n";
                        exit(1);
                }
                break;
            case 860: // RTX30 series
                switch (NHalf){
                    case 512:
                        AddToAccCGGI_CUDA_multi<860, 512>(params, a, acc, fmod);
                        break;
                    case 1024:
                        AddToAccCGGI_CUDA_multi<860, 1024>(params, a, acc, fmod);
                        break;
                    case 2048:
                        AddToAccCGGI_CUDA_multi<860, 2048>(params, a, acc, fmod);
                        break;
                    case 4096:
                        AddToAccCGGI_CUDA_multi<860, 4096>(params, a, acc, fmod);
                        break;
                    default:
                        std::cerr << "Unsupported N, we support N = 1024, 2048, 4096, 8192\n";
                        exit(1);
                }
                break;
            case 890: // RTX40 series
                switch (NHalf){
                    case 512:
                        AddToAccCGGI_CUDA_multi<890, 512>(params, a, acc, fmod);
                        break;
                    case 1024:
                        AddToAccCGGI_CUDA_multi<890, 1024>(params, a, acc, fmod);
                        break;
                    case 2048:
                        AddToAccCGGI_CUDA_multi<890, 2048>(params, a, acc, fmod);
                        break;
                    case 4096:
                        AddToAccCGGI_CUDA_multi<890, 4096>(params, a, acc, fmod);
                        break;
                    default:
                        std::cerr << "Unsupported N, we support N = 1024, 2048, 4096, 8192\n";
                        exit(1);
                }
                break;
            case 900: // H100
                switch (NHalf){
                    case 512:
                        AddToAccCGGI_CUDA_multi<900, 512>(params, a, acc, fmod);
                        break;
                    case 1024:
                        AddToAccCGGI_CUDA_multi<900, 1024>(params, a, acc, fmod);
                        break;
                    case 2048:
                        AddToAccCGGI_CUDA_multi<900, 2048>(params, a, acc, fmod);
                        break;
                    case 4096:
                        AddToAccCGGI_CUDA_multi<900, 4096>(params, a, acc, fmod);
                        break;
                    default:
                        std::cerr << "Unsupported N, we support N = 1024, 2048, 4096, 8192\n";
                        exit(1);
                }
                break;
            default:
                std::cerr << "Unsupported GPU architecture, we support compute capability = 700, 800, 860, 890, 900\n";
                exit(1);
        }
    }
}

template<uint32_t arch, uint32_t FFT_dimension, uint32_t FFT_num>
void GPUFFTBootstrap::AddToAccCGGI_CUDA_single(const std::shared_ptr<RingGSWCryptoParams> params, const std::vector<NativeVector>& a, 
        std::shared_ptr<std::vector<RLWECiphertext>> acc, uint64_t fmod)
{   
    /* HE parameters set */
    auto mod                    = a[0].GetModulus();
    uint32_t modInt             = mod.ConvertToInt();
    auto Q                      = params->GetQ();
    NativeInteger QHalf         = Q >> 1;
    int64_t Q_int               = Q.ConvertToInt();
    uint32_t N                  = params->GetN();
    uint32_t NHalf              = N >> 1;
    uint32_t n                  =  a[0].GetLength();
    uint32_t M                  = 2 * params->GetN();
    uint32_t numDigitsToThrow   = params->GetNumDigitsToThrow();
    uint32_t digitsG2           = (params->GetDigitsG() - numDigitsToThrow) << 1;
    auto polyParams             = params->GetPolyParams();

    /* Configure cuFFTDx */
    using FFT    = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::forward>() + 
                            cufftdx::ElementsPerThread<8>() + cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<FFT_num>() + cufftdx::SM<arch>());

    using IFFT   = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::inverse>() + 
                            cufftdx::ElementsPerThread<8>() + cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<2>() + cufftdx::SM<arch>());

    /* GPU settings */
    int bootstrap_num       = acc->size();
    int GPU_num             = gpuInfoList.size();
    int SM_count            = gpuInfoList[0].multiprocessorCount;
    int shared_mem_size     = max(FFT::shared_memory_size, N * 16); // N*16 shared memory is used for RLWE X RGSW

    /* Increase max shared memory */
    auto sharedMem_it = sharedMemMap.find(arch);
    int maxSharedMemoryAvail = sharedMem_it->second * 1024;
    for(int g = 0; g < GPU_num; g++){
        cudaSetDevice(g);
        // Single block Bootstrapping shared memory size
        if(shared_mem_size < maxSharedMemoryAvail){
            cudaFuncSetAttribute(bootstrappingSingleBlock<FFT, IFFT>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
        }
    }

    /* Check whether block size exceeds cuda limitation */
    if((NHalf / FFT::elements_per_thread * digitsG2) > gpuInfoList[0].maxThreadsPerBlock){
        std::cerr << "Exceed Maximum blocks per threads (" << gpuInfoList[0].maxThreadsPerBlock << ")\n";
        std::cerr << "Using " << (NHalf / FFT::elements_per_thread * digitsG2) << " threads" << ")\n";
        std::cerr << "NHalf: " << NHalf << "FFT::elements_per_thread: " << FFT::elements_per_thread << "digitsG2: " << digitsG2 << ")\n";
        exit(1);
    }

    /* Define and allocate a_host*/
    std::vector<uint64_t> a_host(n);

    auto start = std::chrono::high_resolution_clock::now();

    /* Main Bootstrapping */
    auto sync_it = synchronizationMap.find(FFT_dimension);
    uint32_t syncNum = sync_it->second;
    int s;
    for (s = 0; s < bootstrap_num; s++) {
        int currentGPU = (s / SM_count) % GPU_num;
        if(s % SM_count == 0){
            cudaSetDevice(currentGPU);
        }
        /* Initialize a_host */
        for (size_t i = 0; i < n; ++i)
            a_host[i] = (mod.ModSub(a[s][i], mod) * (M / modInt)).ConvertToInt();
        /* Initialize acc_host */
        for(int i = 0; i < 2; i++){
            NativePoly& acc_t((*acc)[s]->GetElements()[i]);
            for(int j = 0; j < NHalf; j++){
                acc_host[(s % max_bootstapping_num)*2*NHalf + i*NHalf + j] = Complex(acc_t[j].ConvertToDouble(), acc_t[j + NHalf].ConvertToDouble());
            }
        }

        cudaMemcpyAsync(GPUVec[currentGPU].a_CUDA + (s % SM_count)*n, a_host.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice, GPUVec[currentGPU].streams[s % SM_count]);
        cudaMemcpyAsync(GPUVec[currentGPU].acc_CUDA + (s % SM_count)*2*NHalf, acc_host + (s % max_bootstapping_num)*2*NHalf, 2 * NHalf * sizeof(Complex_d), cudaMemcpyHostToDevice, GPUVec[currentGPU].streams[s % SM_count]);
        bootstrappingSingleBlock<FFT, IFFT><<<1, FFT::block_dim, shared_mem_size, GPUVec[currentGPU].streams[s % SM_count]>>>
            (GPUVec[currentGPU].acc_CUDA + (s % SM_count)*2*NHalf, GPUVec[currentGPU].ct_CUDA + (s % SM_count)*2*NHalf, GPUVec[currentGPU].dct_CUDA + (s % SM_count)*digitsG2*NHalf,
                GPUVec[currentGPU].a_CUDA + (s % SM_count)*n, GPUVec[currentGPU].monomial_CUDA, GPUVec[currentGPU].twiddleTable_CUDA, GPUVec[currentGPU].GINX_bootstrappingKey_CUDA, 
                GPUVec[currentGPU].keySwitchingkey_CUDA, GPUVec[currentGPU].params_CUDA, fmod, syncNum);
        cudaMemcpyAsync(acc_host + (s % max_bootstapping_num)*2*NHalf, GPUVec[currentGPU].acc_CUDA + (s % SM_count)*2*NHalf, 2 * NHalf * sizeof(Complex_d), cudaMemcpyDeviceToHost, GPUVec[currentGPU].streams[s % SM_count]);

        if((s % max_bootstapping_num) == max_bootstapping_num-1){
            /* Synchronize all GPUs */
            for(int g = 0; g < GPU_num; g++){
                cudaSetDevice(g);
                CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
                CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
            }

            /* cast acc_host back to NativePoly */
#pragma omp parallel for if (max_bootstapping_num > 512)
            for (int i = 0; i < max_bootstapping_num; i++) {
                NativeVector ret0(N, Q), ret1(N, Q);
                for(int j = 0; j < NHalf; j++){
                    ret0[j] = static_cast<BasicInteger>(acc_host[i*2*NHalf + j].real());
                    ret0[j + NHalf] = static_cast<BasicInteger>(acc_host[i*2*NHalf + j].imag());
                    ret1[j] = static_cast<BasicInteger>(acc_host[i*2*NHalf + NHalf + j].real());
                    ret1[j + NHalf] = static_cast<BasicInteger>(acc_host[i*2*NHalf + NHalf + j].imag());
                }
                std::vector<NativePoly> res(2);
                res[0] = NativePoly(polyParams, Format::COEFFICIENT, false);
                res[1] = NativePoly(polyParams, Format::COEFFICIENT, false);
                res[0].SetValues(std::move(ret0), Format::COEFFICIENT);
                res[1].SetValues(std::move(ret1), Format::COEFFICIENT);

                (*acc)[s-max_bootstapping_num+1+i] = std::make_shared<RLWECiphertextImpl>(std::move(res));
            }
        }
    }
    
    /* If bootstrapping number can't be divided by max bootstrapping num*/
    if((bootstrap_num % max_bootstapping_num) != 0){
        /* Synchronize all GPUs */
        for(int g = 0; g < GPU_num; g++){
            cudaSetDevice(g);
            CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
        //std::cout << bootstrap_num << "AddToAccCGGI_CUDA_single GPU time : " << elapsed.count() << " ms" << std::endl;

        /* cast acc_host back to NativePoly */
        int acc_init = bootstrap_num > max_bootstapping_num ? s - (bootstrap_num % max_bootstapping_num) : 0;
#pragma omp parallel for if ((bootstrap_num % max_bootstapping_num) > 512)
        for (int i = 0; i < (bootstrap_num % max_bootstapping_num); i++) {
            NativeVector ret0(N, Q), ret1(N, Q);
            for(int j = 0; j < NHalf; j++){
                ret0[j] = static_cast<BasicInteger>(acc_host[i*2*NHalf + j].real());
                ret0[j + NHalf] = static_cast<BasicInteger>(acc_host[i*2*NHalf + j].imag());
                ret1[j] = static_cast<BasicInteger>(acc_host[i*2*NHalf + NHalf + j].real());
                ret1[j + NHalf] = static_cast<BasicInteger>(acc_host[i*2*NHalf + NHalf + j].imag());
            }
            std::vector<NativePoly> res(2);
            res[0] = NativePoly(polyParams, Format::COEFFICIENT, false);
            res[1] = NativePoly(polyParams, Format::COEFFICIENT, false);
            res[0].SetValues(std::move(ret0), Format::COEFFICIENT);
            res[1].SetValues(std::move(ret1), Format::COEFFICIENT);

            (*acc)[i+acc_init] = std::make_shared<RLWECiphertextImpl>(std::move(res));
        }
    }
}

template<uint32_t arch, uint32_t FFT_dimension>
void GPUFFTBootstrap::AddToAccCGGI_CUDA_multi(const std::shared_ptr<RingGSWCryptoParams> params, const std::vector<NativeVector>& a, 
        std::shared_ptr<std::vector<RLWECiphertext>> acc, uint64_t fmod)
{   
    /* HE parameters set */
    auto mod                    = a[0].GetModulus();
    uint32_t modInt             = mod.ConvertToInt();
    auto Q                      = params->GetQ();
    NativeInteger QHalf         = Q >> 1;
    int64_t Q_int               = Q.ConvertToInt();
    uint32_t N                  = params->GetN();
    uint32_t NHalf              = N >> 1;
    uint32_t n                  =  a[0].GetLength();
    uint32_t M                  = 2 * params->GetN();
    uint32_t numDigitsToThrow   = params->GetNumDigitsToThrow();
    uint32_t digitsG2           = (params->GetDigitsG() - numDigitsToThrow) << 1;
    auto polyParams             = params->GetPolyParams();

    /* GPU settings */
    int bootstrap_num       = acc->size();
    int GPU_num             = gpuInfoList.size();
    int SM_count            = gpuInfoList[0].multiprocessorCount;

    /* Configure cuFFTDx */
    using FFT      = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::forward>() +
                            cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<2>() + cufftdx::SM<arch>());

    using IFFT     = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::inverse>() +
                            cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<2>() + cufftdx::SM<arch>());

    /* Increase max shared memory */
    auto sharedMem_it = sharedMemMap.find(arch);
    int maxSharedMemoryAvail = sharedMem_it->second * 1024;
    for(int g = 0; g < GPU_num; g++){
        cudaSetDevice(g);
        // Multi block Bootstrapping shared memory size
        if(FFT::shared_memory_size < maxSharedMemoryAvail){
            cudaFuncSetAttribute(bootstrappingMultiBlock<FFT, IFFT>, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size);
        }
    }

    /* Check whether block size exceeds cuda limitation */
    if((NHalf / FFT::elements_per_thread * 2) > gpuInfoList[0].maxThreadsPerBlock){
        std::cerr << "Exceed Maximum blocks per threads (" << gpuInfoList[0].maxThreadsPerBlock << ")\n";
        std::cerr << "Using " << (NHalf / FFT::elements_per_thread * digitsG2) << " threads" << ")\n";
        std::cerr << "NHalf: " << NHalf << "FFT::elements_per_thread: " << FFT::elements_per_thread << ")\n";
        exit(1);
    }

    /* Define and allocate a_host */
    std::vector<uint64_t> a_host(n);

    auto start = std::chrono::high_resolution_clock::now();

    /* Main Bootstrapping */
    Complex_d* acc_CUDA_offset, *ct_CUDA_offset, *dct_CUDA_offset;
    uint64_t* a_CUDA_offset;
    int s;
    for (s = 0; s < bootstrap_num; s++) {
        int currentGPU = (s / SM_count) % GPU_num;
        if(s % SM_count == 0){
            cudaSetDevice(currentGPU);
        }
        /* Initialize a_host */
        for (size_t i = 0; i < n; ++i)
            a_host[i] = (mod.ModSub(a[s][i], mod) * (M / modInt)).ConvertToInt();
        /* Initialize acc_host */
        for(int i = 0; i < 2; i++){
            NativePoly& acc_t((*acc)[s]->GetElements()[i]);
            for(int j = 0; j < NHalf; j++){
                acc_host[(s % max_bootstapping_num)*2*NHalf + i*NHalf + j] = Complex(acc_t[j].ConvertToDouble(), acc_t[j + NHalf].ConvertToDouble());
            }
        }

        acc_CUDA_offset = GPUVec[currentGPU].acc_CUDA + (s % SM_count)*2*NHalf;
        ct_CUDA_offset = GPUVec[currentGPU].ct_CUDA + (s % SM_count)*2*NHalf;
        dct_CUDA_offset = GPUVec[currentGPU].dct_CUDA + (s % SM_count)*digitsG2*NHalf;
        a_CUDA_offset = GPUVec[currentGPU].a_CUDA + (s % SM_count)*n;
        cudaMemcpyAsync(GPUVec[currentGPU].a_CUDA + (s % SM_count)*n, a_host.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice, GPUVec[currentGPU].streams[s % SM_count]);
        cudaMemcpyAsync(GPUVec[currentGPU].acc_CUDA + (s % SM_count)*2*NHalf, acc_host + (s % max_bootstapping_num)*2*NHalf, 2 * NHalf * sizeof(Complex_d), cudaMemcpyHostToDevice, GPUVec[currentGPU].streams[s % SM_count]);
        void *kernelArgs[] = {(void *)&acc_CUDA_offset, (void *)&ct_CUDA_offset, (void *)&dct_CUDA_offset, (void *)&a_CUDA_offset, 
            (void *)&GPUVec[currentGPU].monomial_CUDA, (void *)&GPUVec[currentGPU].twiddleTable_CUDA, (void *)&GPUVec[currentGPU].GINX_bootstrappingKey_CUDA,
                (void *)&GPUVec[currentGPU].keySwitchingkey_CUDA, (void *)&GPUVec[currentGPU].params_CUDA, (void *)&fmod};
        cudaLaunchCooperativeKernel((void*)(bootstrappingMultiBlock<FFT, IFFT>), digitsG2/2, FFT::block_dim, 
            kernelArgs, FFT::shared_memory_size, GPUVec[currentGPU].streams[s % SM_count]);
        cudaMemcpyAsync(acc_host + (s % max_bootstapping_num)*2*NHalf, GPUVec[currentGPU].acc_CUDA + (s % SM_count)*2*NHalf, 2 * NHalf * sizeof(Complex_d), cudaMemcpyDeviceToHost, GPUVec[currentGPU].streams[s % SM_count]);

        if((s % max_bootstapping_num) == max_bootstapping_num-1){
            /* Synchronize all GPUs */
            for(int g = 0; g < GPU_num; g++){
                cudaSetDevice(g);
                CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
                CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
            }

            /* cast acc_host back to NativePoly */
#pragma omp parallel for if (max_bootstapping_num > 512)
            for (int i = 0; i < max_bootstapping_num; i++) {
                NativeVector ret0(N, Q), ret1(N, Q);
                for(int j = 0; j < NHalf; j++){
                    ret0[j] = static_cast<BasicInteger>(acc_host[i*2*NHalf + j].real());
                    ret0[j + NHalf] = static_cast<BasicInteger>(acc_host[i*2*NHalf + j].imag());
                    ret1[j] = static_cast<BasicInteger>(acc_host[i*2*NHalf + NHalf + j].real());
                    ret1[j + NHalf] = static_cast<BasicInteger>(acc_host[i*2*NHalf + NHalf + j].imag());
                }
                std::vector<NativePoly> res(2);
                res[0] = NativePoly(polyParams, Format::COEFFICIENT, false);
                res[1] = NativePoly(polyParams, Format::COEFFICIENT, false);
                res[0].SetValues(std::move(ret0), Format::COEFFICIENT);
                res[1].SetValues(std::move(ret1), Format::COEFFICIENT);

                (*acc)[s-max_bootstapping_num+1+i] = std::make_shared<RLWECiphertextImpl>(std::move(res));
            }
        }
    }

    /* If bootstrapping number can't be divided by max bootstrapping num*/
    if((bootstrap_num % max_bootstapping_num) != 0){
        /* Synchronize all GPUs */
        for(int g = 0; g < GPU_num; g++){
            cudaSetDevice(g);
            CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
        //std::cout << bootstrap_num << "AddToAccCGGI_CUDA_multi GPU time : " << elapsed.count() << " ms" << std::endl;

        /* cast acc_host back to NativePoly */
        int acc_init = bootstrap_num > max_bootstapping_num ? s - (bootstrap_num % max_bootstapping_num) : 0;
#pragma omp parallel for if ((bootstrap_num % max_bootstapping_num) > 512)
        for (int i = 0; i < (bootstrap_num % max_bootstapping_num); i++) {
            NativeVector ret0(N, Q), ret1(N, Q);
            for(int j = 0; j < NHalf; j++){
                ret0[j] = static_cast<BasicInteger>(acc_host[i*2*NHalf + j].real());
                ret0[j + NHalf] = static_cast<BasicInteger>(acc_host[i*2*NHalf + j].imag());
                ret1[j] = static_cast<BasicInteger>(acc_host[i*2*NHalf + NHalf + j].real());
                ret1[j + NHalf] = static_cast<BasicInteger>(acc_host[i*2*NHalf + NHalf + j].imag());
            }
            std::vector<NativePoly> res(2);
            res[0] = NativePoly(polyParams, Format::COEFFICIENT, false);
            res[1] = NativePoly(polyParams, Format::COEFFICIENT, false);
            res[0].SetValues(std::move(ret0), Format::COEFFICIENT);
            res[1].SetValues(std::move(ret1), Format::COEFFICIENT);

            (*acc)[i+acc_init] = std::make_shared<RLWECiphertextImpl>(std::move(res));
        }
    }
}

void GPUFFTBootstrap::MKMSwitch_CUDA(const std::shared_ptr<LWECryptoParams> params, std::shared_ptr<std::vector<LWECiphertext>> ctExt, NativeInteger fmod)
{
    /* HE parameters set */
    uint32_t n              = params->Getn();
    uint32_t N              = params->GetN();
    uint32_t max_n_N        = n > N ? n : N;

    /* GPU settings */
    int bootstrap_num       = ctExt->size();
    int GPU_num             = gpuInfoList.size();
    int SM_count            = gpuInfoList[0].multiprocessorCount;

    auto start = std::chrono::high_resolution_clock::now();

    int s;
    for (s = 0; s < bootstrap_num; s++) {
        int currentGPU = (s / SM_count) % GPU_num;
        if(s % SM_count == 0){
            cudaSetDevice(currentGPU);
        }
        /* Initialize ctExt_host */
        // A
        for(int i = 0; i < N; i++)
            ctExt_host[(s % max_bootstapping_num)*(max_n_N + 1) + i] = (*ctExt)[s]->GetA()[i].ConvertToInt();
        // B
        ctExt_host[(s % max_bootstapping_num)*(max_n_N + 1) + N] = (*ctExt)[s]->GetB().ConvertToInt();

        cudaMemcpyAsync(GPUVec[currentGPU].ctExt_CUDA + (s % SM_count)*(max_n_N + 1), ctExt_host + (s % max_bootstapping_num)*(max_n_N + 1), (N + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice, GPUVec[currentGPU].streams[s % SM_count]);
        MKMSwitchKernel<<<1, 768, (N + 1) * sizeof(uint64_t), GPUVec[currentGPU].streams[s % SM_count]>>>
            (GPUVec[currentGPU].ctExt_CUDA + (s % SM_count)*(max_n_N + 1), GPUVec[currentGPU].keySwitchingkey_CUDA, GPUVec[currentGPU].params_CUDA, static_cast<uint64_t>(fmod.ConvertToInt()));
        cudaMemcpyAsync(ctExt_host + (s % max_bootstapping_num)*(max_n_N + 1), GPUVec[currentGPU].ctExt_CUDA + (s % SM_count)*(max_n_N + 1), (n + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost, GPUVec[currentGPU].streams[s % SM_count]);
        
        if((s % max_bootstapping_num) == max_bootstapping_num-1){
            /* Synchronize all GPUs */
            for(int g = 0; g < GPU_num; g++){
                cudaSetDevice(g);
                CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
                CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
            }

            /* Copy ctExt_host back to ctExt */
            for (int i = 0; i < max_bootstapping_num; i++) {
                // A
                NativeVector a(n, fmod);
                for(int j = 0; j < n; j++)
                    a[j] = ctExt_host[i*(max_n_N + 1) + j];
                // B
                NativeInteger b (ctExt_host[i*(max_n_N + 1) + n]);

                (*ctExt)[s-max_bootstapping_num+1+i] = std::make_shared<LWECiphertextImpl>(LWECiphertextImpl(std::move(a), b));
            }
        }
    }

    /* If bootstrapping number can't be divided by max bootstrapping num*/
    if((bootstrap_num % max_bootstapping_num) != 0){
        /* Synchronize all GPUs */
        for(int g = 0; g < GPU_num; g++){
            cudaSetDevice(g);
            CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
        //std::cout << bootstrap_num << "MKMSwitch_CUDA GPU time : " << elapsed.count() << " ms" << std::endl;

        /* cast acc_host back to NativePoly */
        int ct_init = bootstrap_num > max_bootstapping_num ? s - (bootstrap_num % max_bootstapping_num) : 0;
        for (int i = 0; i < (bootstrap_num % max_bootstapping_num); i++) {
            // A
            NativeVector a(n, fmod);
            for(int j = 0; j < n; j++)
                a[j] = ctExt_host[i*(max_n_N + 1) + j];
            // B
            NativeInteger b (ctExt_host[i*(max_n_N + 1) + n]);

            (*ctExt)[ct_init+i] = std::make_shared<LWECiphertextImpl>(LWECiphertextImpl(std::move(a), b));
        }
    }
}

};  // namespace lbcrypto
