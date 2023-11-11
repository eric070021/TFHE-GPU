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

/* Vector used to store multiple GPUs INFO */
std::vector<GPUInfo> gpuInfoList;

template<class FFT, class IFFT>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void bootstrappingMultiBlock(Complex_d* acc_CUDA, Complex_d* ct_CUDA, Complex_d* dct_CUDA, uint64_t* a_CUDA, 
        Complex_d* monomial_CUDA, Complex_d* twiddleTable_CUDA, uint64_t* params_CUDA, Complex_d* GINX_bootstrappingKey_CUDA){
    
    /* GPU Parameters Set */
    cg::grid_group grid = cg::this_grid();
    uint32_t tid = ThisThreadRankInBlock(); // thread id in block
    uint32_t bid = grid.block_rank(); // block id in grid
    uint32_t gtid = grid.thread_rank(); // global thread id
    uint32_t bdim = ThisBlockSize(); // size of block
    uint32_t gdim = grid.num_threads(); // number of threads in grid

    /* HE Parameters Set */
    uint64_t M            = params_CUDA[0] << 1;
    uint64_t N            = params_CUDA[0];
    uint64_t NHalf        = N >> 1;
    uint64_t n            = params_CUDA[1];
    uint64_t Q            = params_CUDA[2];
    uint64_t QHalf        = params_CUDA[2] >> 1;
    uint64_t digitsG2     = params_CUDA[3];
    uint64_t baseG        = params_CUDA[4];
    int32_t gBits = static_cast<int32_t>(log2(static_cast<double>(baseG)));
    int32_t gBitsMaxBits = 64 - gBits;
    uint32_t RGSW_size = digitsG2 * 2 * NHalf;

    /* cufftdx variables */
    using complex_type = typename FFT::value_type;
    const unsigned int local_fft_id = threadIdx.y;
    const unsigned int offset = cufftdx::size_of<FFT>::value * (blockIdx.x * FFT::ffts_per_block + local_fft_id);
    extern __shared__ complex_type shared_mem[];
    complex_type thread_data[FFT::storage_size];     
    
    /* 2 times Forward FFT */
    if(bid == 0){
        // Load data from shared memory to registers
        {
            unsigned int index = offset + threadIdx.x;
            unsigned int twist_idx = threadIdx.x;
            for (unsigned i = 0; i < FFT::elements_per_thread; i++) {
                // twisting
                acc_CUDA[index] = cuCmul(acc_CUDA[index], twiddleTable_CUDA[twist_idx]);
                thread_data[i] = complex_type {acc_CUDA[index].x, acc_CUDA[index].y};
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
                acc_CUDA[index] = make_cuDoubleComplex(thread_data[i].x, thread_data[i].y);
                // FFT::stride shows how elements from a single FFT should be split between threads
                index += FFT::stride;
            }
        }
    }
    grid.sync();
    
    for(uint32_t round = 0; round < n; ++round){
        /* Copy acc_CUDA to ct_CUDA */
        for(uint32_t i = gtid; i < N; i += gdim){
            ct_CUDA[i] = acc_CUDA[i];
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
                unsigned int twist_idx = threadIdx.x;
                for (unsigned i = 0; i < IFFT::elements_per_thread; i++) {
                    ct_CUDA[index].x = thread_data[i].x;
                    ct_CUDA[index].y = thread_data[i].y;
                    // twisting
                    ct_CUDA[index] = cuCmul(ct_CUDA[index], twiddleTable_CUDA[twist_idx + NHalf]);
                    // Round to INT128 and MOD
                    ct_CUDA[index].x = static_cast<double>(static_cast<__int128_t>(rint(ct_CUDA[index].x)) % static_cast<__int128_t>(Q));
                    if (ct_CUDA[index].x < 0)
                        ct_CUDA[index].x += static_cast<double>(Q);
                    if (ct_CUDA[index].x >= QHalf)
                        ct_CUDA[index].x -= static_cast<double>(Q);
                    ct_CUDA[index].y = static_cast<double>(static_cast<__int128_t>(rint(ct_CUDA[index].y)) % static_cast<__int128_t>(Q));
                    if (ct_CUDA[index].y < 0)
                        ct_CUDA[index].y += static_cast<double>(Q);
                    if (ct_CUDA[index].y >= QHalf)
                        ct_CUDA[index].y -= static_cast<double>(Q);
                    // IFFT::stride shows how elements from a single FFT should be split between threads
                    index += IFFT::stride;
                    twist_idx += IFFT::stride;
                }
            }
        }
        grid.sync();

        /* SignedDigitDecompose */
        // polynomial from a
        for (size_t k = gtid; k < NHalf; k += gdim) {
            int64_t d0 = static_cast<int64_t>(ct_CUDA[k].x);
            int64_t d1 = static_cast<int64_t>(ct_CUDA[k].y);

            for (size_t l = 0; l < digitsG2; l += 2) {
                int64_t r0 = (d0 << gBitsMaxBits) >> gBitsMaxBits;
                d0 = (d0 - r0) >> gBits;
                if (r0 < 0)
                    r0 += static_cast<int64_t>(Q);
                if (r0 >= QHalf)
                    r0 -= static_cast<int64_t>(Q);
                dct_CUDA[l*NHalf + k].x = static_cast<double>(r0);

                int64_t r1 = (d1 << gBitsMaxBits) >> gBitsMaxBits;
                d1 = (d1 - r1) >> gBits;
                if (r1 < 0)
                    r1 += static_cast<int64_t>(Q);
                if (r1 >= QHalf)
                    r1 -= static_cast<int64_t>(Q);
                dct_CUDA[l*NHalf + k].y = static_cast<double>(r1);
            }
        }

        // polynomial from b
        for (size_t k = gtid + NHalf; k < N; k += gdim) {
            int64_t d0 = static_cast<int64_t>(ct_CUDA[k].x);
            int64_t d1 = static_cast<int64_t>(ct_CUDA[k].y);

            for (size_t l = 0; l < digitsG2; l += 2) {
                int64_t r0 = (d0 << gBitsMaxBits) >> gBitsMaxBits;
                d0 = (d0 - r0) >> gBits;
                if (r0 < 0)
                    r0 += static_cast<int64_t>(Q);
                if (r0 >= QHalf)
                    r0 -= static_cast<int64_t>(Q);
                dct_CUDA[l*NHalf + k].x = static_cast<double>(r0);

                int64_t r1 = (d1 << gBitsMaxBits) >> gBitsMaxBits;
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
        
        /* ACC times Bootstrapping key and monomial */
        /* multiply with ek0 */
        // polynomial a
        #pragma unroll
        for (uint32_t i = gtid; i < NHalf; i += gdim){
            ct_CUDA[i] = make_cuDoubleComplex(0, 0);
            #pragma unroll
            for (uint32_t l = 0; l < digitsG2; ++l){
                ct_CUDA[i] = cuCadd(ct_CUDA[i], cuCmul(dct_CUDA[l*NHalf + i], GINX_bootstrappingKey_CUDA[round*RGSW_size + (l << 1)*NHalf + i]));
            }
        }
        // polynomial b
        #pragma unroll
        for (uint32_t i = gtid; i < NHalf; i += gdim){
            ct_CUDA[NHalf + i] = make_cuDoubleComplex(0, 0);
            #pragma unroll
            for (uint32_t l = 0; l < digitsG2; ++l){
                ct_CUDA[NHalf + i] = cuCadd(ct_CUDA[NHalf + i], cuCmul(dct_CUDA[l*NHalf + i], GINX_bootstrappingKey_CUDA[round*RGSW_size + ((l << 1) + 1)*NHalf + i]));
            }
        }
        grid.sync();

        /* multiply with postive monomial */
        // polynomial a
        #pragma unroll
        for (uint32_t i = gtid; i < NHalf; i += gdim){
            acc_CUDA[i] = cuCadd(acc_CUDA[i], cuCmul(ct_CUDA[i], monomial_CUDA[indexPos*NHalf + i]));
        }
        // polynomial b
        #pragma unroll
        for (uint32_t i = gtid; i < NHalf; i += gdim){
            acc_CUDA[NHalf + i] = cuCadd(acc_CUDA[NHalf + i], cuCmul(ct_CUDA[NHalf + i], monomial_CUDA[indexPos*NHalf + i]));
        }        
        grid.sync();

        /* multiply with ek1 */
        // polynomial a
        #pragma unroll
        for (uint32_t i = gtid; i < NHalf; i += gdim){
            ct_CUDA[i] = make_cuDoubleComplex(0, 0);
            #pragma unroll
            for (uint32_t l = 0; l < digitsG2; ++l)
                ct_CUDA[i] = cuCadd(ct_CUDA[i], cuCmul(dct_CUDA[l*NHalf + i], GINX_bootstrappingKey_CUDA[n*RGSW_size + round*RGSW_size + (l << 1)*NHalf + i]));
        }
        // polynomial b
        #pragma unroll
        for (uint32_t i = gtid; i < NHalf; i += gdim){
            ct_CUDA[NHalf + i] = make_cuDoubleComplex(0, 0);
            #pragma unroll
            for (uint32_t l = 0; l < digitsG2; ++l)
                ct_CUDA[NHalf + i] = cuCadd(ct_CUDA[NHalf + i], cuCmul(dct_CUDA[l*NHalf + i], GINX_bootstrappingKey_CUDA[n*RGSW_size + round*RGSW_size + ((l << 1) + 1)*NHalf + i]));
        }
        grid.sync();
        
        /* multiply with negative monomial */
        // polynomial a
        #pragma unroll
        for (uint32_t i = gtid; i < NHalf; i += gdim){
            acc_CUDA[i] = cuCadd(acc_CUDA[i], cuCmul(ct_CUDA[i], monomial_CUDA[indexNeg*NHalf + i]));
        }
        // polynomial b
        #pragma unroll
        for (uint32_t i = gtid; i < NHalf; i += gdim){
            acc_CUDA[NHalf + i] = cuCadd(acc_CUDA[NHalf + i], cuCmul(ct_CUDA[NHalf + i], monomial_CUDA[indexNeg*NHalf + i]));
        }        
        grid.sync();
    }

    /* 2 times Inverse IFFT */
    if(bid == 0){
        // Load data from shared memory to registers
        {
            unsigned int index = offset + threadIdx.x;
            for (unsigned i = 0; i < IFFT::elements_per_thread; i++) {
                thread_data[i] = complex_type {acc_CUDA[index].x, acc_CUDA[index].y};
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
            unsigned int twist_idx = threadIdx.x;
            for (unsigned i = 0; i < IFFT::elements_per_thread; i++) {
                acc_CUDA[index].x = thread_data[i].x;
                acc_CUDA[index].y = thread_data[i].y;
                // twisting
                acc_CUDA[index] = cuCmul(acc_CUDA[index], twiddleTable_CUDA[twist_idx + NHalf]);
                // Round to INT128 and MOD
                acc_CUDA[index].x = static_cast<double>(static_cast<__int128_t>(rint(acc_CUDA[index].x)) % static_cast<__int128_t>(Q));
                if (acc_CUDA[index].x < 0)
                    acc_CUDA[index].x += static_cast<double>(Q);
                acc_CUDA[index].y = static_cast<double>(static_cast<__int128_t>(rint(acc_CUDA[index].y)) % static_cast<__int128_t>(Q));
                if (acc_CUDA[index].y < 0)
                    acc_CUDA[index].y += static_cast<double>(Q);
                // IFFT::stride shows how elements from a single FFT should be split between threads
                index += IFFT::stride;
                twist_idx += FFT::stride;
            }
        }
    }
    grid.sync();
}

template<class FFT, class IFFT>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void bootstrappingSingleBlock(Complex_d* acc_CUDA, Complex_d* ct_CUDA, Complex_d* dct_CUDA, uint64_t* a_CUDA, 
        Complex_d* monomial_CUDA, Complex_d* twiddleTable_CUDA, uint64_t* params_CUDA, Complex_d* GINX_bootstrappingKey_CUDA){
    
    /* GPU Parameters Set */
    uint32_t tid = ThisThreadRankInBlock();
    uint32_t bdim = ThisBlockSize();

    /* HE Parameters Set */
    uint64_t M            = params_CUDA[0] << 1;
    uint64_t N            = params_CUDA[0];
    uint64_t NHalf        = N >> 1;
    uint64_t n            = params_CUDA[1];
    uint64_t Q            = params_CUDA[2];
    uint64_t QHalf        = params_CUDA[2] >> 1;
    uint64_t digitsG2     = params_CUDA[3];
    uint64_t baseG        = params_CUDA[4];
    int32_t gBits = static_cast<int32_t>(log2(static_cast<double>(baseG)));
    int32_t gBitsMaxBits = 64 - gBits;
    uint32_t RGSW_size = digitsG2 * 2 * NHalf;
    uint32_t syncNum      = static_cast<uint32_t>(params_CUDA[5]); // number of synchronization (cufftdx)

    /* cufftdx variables */
    using complex_type = typename FFT::value_type;
    const unsigned int local_fft_id = threadIdx.y;
    const unsigned int offset = cufftdx::size_of<FFT>::value * (blockIdx.x * FFT::ffts_per_block + local_fft_id);
    extern __shared__ complex_type shared_mem[];
    complex_type thread_data[FFT::storage_size];     
    
    /* 2 times Forward FFT */
    if(threadIdx.y < 2){
        // Load data from shared memory to registers
        {
            unsigned int index = offset + threadIdx.x;
            unsigned int twist_idx = threadIdx.x;
            for (unsigned i = 0; i < FFT::elements_per_thread; i++) {
                // twisting
                acc_CUDA[index] = cuCmul(acc_CUDA[index], twiddleTable_CUDA[twist_idx]);
                thread_data[i] = complex_type {acc_CUDA[index].x, acc_CUDA[index].y};
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
                acc_CUDA[index] = make_cuDoubleComplex(thread_data[i].x, thread_data[i].y);
                // FFT::stride shows how elements from a single FFT should be split between threads
                index += FFT::stride;
            }
        }
    }
    else{ // must meet syncs made by FFT
        for(uint32_t i = 0; i < syncNum; ++i)
            __syncthreads();
    }
    __syncthreads();

    for(uint32_t round = 0; round < n; ++round){
        /* Copy acc_CUDA to ct_CUDA */
        for(uint32_t i = tid; i < N; i += bdim){
            ct_CUDA[i] = acc_CUDA[i];
        }
        __syncthreads();

        /* 2 times Inverse IFFT */
        if(threadIdx.y < 2){
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
                unsigned int twist_idx = threadIdx.x;
                for (unsigned i = 0; i < IFFT::elements_per_thread; i++) {
                    ct_CUDA[index].x = thread_data[i].x;
                    ct_CUDA[index].y = thread_data[i].y;
                    // twisting
                    ct_CUDA[index] = cuCmul(ct_CUDA[index], twiddleTable_CUDA[twist_idx + NHalf]);
                    // Round to INT128 and MOD
                    ct_CUDA[index].x = static_cast<double>(static_cast<__int128_t>(rint(ct_CUDA[index].x)) % static_cast<__int128_t>(Q));
                    if (ct_CUDA[index].x < 0)
                        ct_CUDA[index].x += static_cast<double>(Q);
                    if (ct_CUDA[index].x >= QHalf)
                        ct_CUDA[index].x -= static_cast<double>(Q);
                    ct_CUDA[index].y = static_cast<double>(static_cast<__int128_t>(rint(ct_CUDA[index].y)) % static_cast<__int128_t>(Q));
                    if (ct_CUDA[index].y < 0)
                        ct_CUDA[index].y += static_cast<double>(Q);
                    if (ct_CUDA[index].y >= QHalf)
                        ct_CUDA[index].y -= static_cast<double>(Q);
                    // IFFT::stride shows how elements from a single FFT should be split between threads
                    index += IFFT::stride;
                    twist_idx += IFFT::stride;
                }
            }
        }
        else{ // must meet syncs made by IFFT
            for(uint32_t i = 0; i < syncNum; ++i)
                __syncthreads();
        }
        __syncthreads();

        /* SignedDigitDecompose */
        // polynomial from a
        for (size_t k = tid; k < NHalf; k += bdim) {
            int64_t d0 = static_cast<int64_t>(ct_CUDA[k].x);
            int64_t d1 = static_cast<int64_t>(ct_CUDA[k].y);

            for (size_t l = 0; l < digitsG2; l += 2) {
                int64_t r0 = (d0 << gBitsMaxBits) >> gBitsMaxBits;
                d0 = (d0 - r0) >> gBits;
                if (r0 < 0)
                    r0 += static_cast<int64_t>(Q);
                if (r0 >= QHalf)
                    r0 -= static_cast<int64_t>(Q);
                dct_CUDA[l*NHalf + k].x = static_cast<double>(r0);

                int64_t r1 = (d1 << gBitsMaxBits) >> gBitsMaxBits;
                d1 = (d1 - r1) >> gBits;
                if (r1 < 0)
                    r1 += static_cast<int64_t>(Q);
                if (r1 >= QHalf)
                    r1 -= static_cast<int64_t>(Q);
                dct_CUDA[l*NHalf + k].y = static_cast<double>(r1);
            }
        }

        // polynomial from b
        for (size_t k = tid + NHalf; k < N; k += bdim) {
            int64_t d0 = static_cast<int64_t>(ct_CUDA[k].x);
            int64_t d1 = static_cast<int64_t>(ct_CUDA[k].y);

            for (size_t l = 0; l < digitsG2; l += 2) {
                int64_t r0 = (d0 << gBitsMaxBits) >> gBitsMaxBits;
                d0 = (d0 - r0) >> gBits;
                if (r0 < 0)
                    r0 += static_cast<int64_t>(Q);
                if (r0 >= QHalf)
                    r0 -= static_cast<int64_t>(Q);
                dct_CUDA[l*NHalf + k].x = static_cast<double>(r0);

                int64_t r1 = (d1 << gBitsMaxBits) >> gBitsMaxBits;
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
        
        /* ACC times Bootstrapping key and monomial */
        /* multiply with ek0 */
        // polynomial a
        #pragma unroll
        for (uint32_t i = tid; i < NHalf; i += bdim){
            ct_CUDA[i] = make_cuDoubleComplex(0, 0);
            #pragma unroll
            for (uint32_t l = 0; l < digitsG2; ++l){
                ct_CUDA[i] = cuCadd(ct_CUDA[i], cuCmul(dct_CUDA[l*NHalf + i], GINX_bootstrappingKey_CUDA[round*RGSW_size + (l << 1)*NHalf + i]));
            }
        }
        // polynomial b
        #pragma unroll
        for (uint32_t i = tid; i < NHalf; i += bdim){
            ct_CUDA[NHalf + i] = make_cuDoubleComplex(0, 0);
            #pragma unroll
            for (uint32_t l = 0; l < digitsG2; ++l){
                ct_CUDA[NHalf + i] = cuCadd(ct_CUDA[NHalf + i], cuCmul(dct_CUDA[l*NHalf + i], GINX_bootstrappingKey_CUDA[round*RGSW_size + ((l << 1) + 1)*NHalf + i]));
            }
        }
        __syncthreads();

        /* multiply with postive monomial */
        // polynomial a
        #pragma unroll
        for (uint32_t i = tid; i < NHalf; i += bdim){
            acc_CUDA[i] = cuCadd(acc_CUDA[i], cuCmul(ct_CUDA[i], monomial_CUDA[indexPos*NHalf + i]));
        }
        // polynomial b
        #pragma unroll
        for (uint32_t i = tid; i < NHalf; i += bdim){
            acc_CUDA[NHalf + i] = cuCadd(acc_CUDA[NHalf + i], cuCmul(ct_CUDA[NHalf + i], monomial_CUDA[indexPos*NHalf + i]));
        }        
        __syncthreads();

        /* multiply with ek1 */
        // polynomial a
        #pragma unroll
        for (uint32_t i = tid; i < NHalf; i += bdim){
            ct_CUDA[i] = make_cuDoubleComplex(0, 0);
            #pragma unroll
            for (uint32_t l = 0; l < digitsG2; ++l)
                ct_CUDA[i] = cuCadd(ct_CUDA[i], cuCmul(dct_CUDA[l*NHalf + i], GINX_bootstrappingKey_CUDA[n*RGSW_size + round*RGSW_size + (l << 1)*NHalf + i]));
        }
        // polynomial b
        #pragma unroll
        for (uint32_t i = tid; i < NHalf; i += bdim){
            ct_CUDA[NHalf + i] = make_cuDoubleComplex(0, 0);
            #pragma unroll
            for (uint32_t l = 0; l < digitsG2; ++l)
                ct_CUDA[NHalf + i] = cuCadd(ct_CUDA[NHalf + i], cuCmul(dct_CUDA[l*NHalf + i], GINX_bootstrappingKey_CUDA[n*RGSW_size + round*RGSW_size + ((l << 1) + 1)*NHalf + i]));
        }
        __syncthreads();
        
        /* multiply with negative monomial */
        // polynomial a
        #pragma unroll
        for (uint32_t i = tid; i < NHalf; i += bdim){
            acc_CUDA[i] = cuCadd(acc_CUDA[i], cuCmul(ct_CUDA[i], monomial_CUDA[indexNeg*NHalf + i]));
        }
        // polynomial b
        #pragma unroll
        for (uint32_t i = tid; i < NHalf; i += bdim){
            acc_CUDA[NHalf + i] = cuCadd(acc_CUDA[NHalf + i], cuCmul(ct_CUDA[NHalf + i], monomial_CUDA[indexNeg*NHalf + i]));
        }        
        __syncthreads();
    }

    /* 2 times Inverse IFFT */
    if(threadIdx.y < 2){
        // Load data from shared memory to registers
        {
            unsigned int index = offset + threadIdx.x;
            for (unsigned i = 0; i < IFFT::elements_per_thread; i++) {
                thread_data[i] = complex_type {acc_CUDA[index].x, acc_CUDA[index].y};
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
            unsigned int twist_idx = threadIdx.x;
            for (unsigned i = 0; i < IFFT::elements_per_thread; i++) {
                acc_CUDA[index].x = thread_data[i].x;
                acc_CUDA[index].y = thread_data[i].y;
                // twisting
                acc_CUDA[index] = cuCmul(acc_CUDA[index], twiddleTable_CUDA[twist_idx + NHalf]);
                // Round to INT128 and MOD
                acc_CUDA[index].x = static_cast<double>(static_cast<__int128_t>(rint(acc_CUDA[index].x)) % static_cast<__int128_t>(Q));
                if (acc_CUDA[index].x < 0)
                    acc_CUDA[index].x += static_cast<double>(Q);
                acc_CUDA[index].y = static_cast<double>(static_cast<__int128_t>(rint(acc_CUDA[index].y)) % static_cast<__int128_t>(Q));
                if (acc_CUDA[index].y < 0)
                    acc_CUDA[index].y += static_cast<double>(Q);
                // IFFT::stride shows how elements from a single FFT should be split between threads
                index += IFFT::stride;
                twist_idx += FFT::stride;
            }
        }
    }
    else{ // must meet syncs made by IFFT
       for(uint32_t i = 0; i < syncNum; ++i)
            __syncthreads();
    }
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

void GPUSetup(std::shared_ptr<std::vector<std::vector<std::vector<std::shared_ptr<std::vector<std::vector<std::vector<Complex>>>>>>>> GINX_bootstrappingKey_FFT,
        const std::shared_ptr<RingGSWCryptoParams> params)
{
    std::cout << "GPU Setup Start\n";

    /* Setting up available GPU INFO */
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return;
    }

    for (int device = 0; device < deviceCount; ++device) {
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
    }

    /* Parameters Set */
    uint32_t N            = params->GetN();
    uint32_t NHalf     = N >> 1;
    uint32_t digitsG2 = params->GetDigitsG() << 1;
    uint32_t arch = gpuInfoList[0].major * 100 + gpuInfoList[0].minor * 10;

    /* Determine template of GPUSetup_core */
    switch (arch){
        case 700: // V100
            switch (NHalf){
                case 512:
                    switch (digitsG2){
                        case 2:
                            GPUSetup_core<700, 512, 2>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 4:
                            GPUSetup_core<700, 512, 4>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 6:
                            GPUSetup_core<700, 512, 6>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 8:
                            GPUSetup_core<700, 512, 8>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 10:
                            GPUSetup_core<700, 512, 10>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 12:
                            GPUSetup_core<700, 512, 12>(GINX_bootstrappingKey_FFT, params);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 1024:
                    switch (digitsG2){
                        case 2:
                            GPUSetup_core<700, 1024, 2>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 4:
                            GPUSetup_core<700, 1024, 4>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 6:
                            GPUSetup_core<700, 1024, 6>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 8:
                            GPUSetup_core<700, 1024, 8>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 10:
                            GPUSetup_core<700, 1024, 10>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 12:
                            GPUSetup_core<700, 1024, 12>(GINX_bootstrappingKey_FFT, params);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 2048:
                    switch (digitsG2){
                        case 2:
                            GPUSetup_core<700, 2048, 2>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 4:
                            GPUSetup_core<700, 2048, 4>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 6:
                            GPUSetup_core<700, 2048, 6>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 8:
                            GPUSetup_core<700, 2048, 8>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 10:
                            GPUSetup_core<700, 2048, 10>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 12:
                            GPUSetup_core<700, 2048, 12>(GINX_bootstrappingKey_FFT, params);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                default:
                    std::cerr << "Unsupported N\n";
                    exit(1);
            }
            break;
        case 800: // A100
            switch (NHalf){
                case 512:
                    switch (digitsG2){
                        case 2:
                            GPUSetup_core<800, 512, 2>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 4:
                            GPUSetup_core<800, 512, 4>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 6:
                            GPUSetup_core<800, 512, 6>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 8:
                            GPUSetup_core<800, 512, 8>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 10:
                            GPUSetup_core<800, 512, 10>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 12:
                            GPUSetup_core<800, 512, 12>(GINX_bootstrappingKey_FFT, params);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 1024:
                    switch (digitsG2){
                        case 2:
                            GPUSetup_core<800, 1024, 2>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 4:
                            GPUSetup_core<800, 1024, 4>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 6:
                            GPUSetup_core<800, 1024, 6>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 8:
                            GPUSetup_core<800, 1024, 8>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 10:
                            GPUSetup_core<800, 1024, 10>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 12:
                            GPUSetup_core<800, 1024, 12>(GINX_bootstrappingKey_FFT, params);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 2048:
                    switch (digitsG2){
                        case 2:
                            GPUSetup_core<800, 2048, 2>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 4:
                            GPUSetup_core<800, 2048, 4>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 6:
                            GPUSetup_core<800, 2048, 6>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 8:
                            GPUSetup_core<800, 2048, 8>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 10:
                            GPUSetup_core<800, 2048, 10>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 12:
                            GPUSetup_core<800, 2048, 12>(GINX_bootstrappingKey_FFT, params);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                default:
                    std::cerr << "Unsupported N\n";
                    exit(1);
            }
            break;
        case 860: // RTX30 series
            switch (NHalf){
                case 512:
                    switch (digitsG2){
                        case 2:
                            GPUSetup_core<860, 512, 2>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 4:
                            GPUSetup_core<860, 512, 4>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 6:
                            GPUSetup_core<860, 512, 6>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 8:
                            GPUSetup_core<860, 512, 8>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 10:
                            GPUSetup_core<860, 512, 10>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 12:
                            GPUSetup_core<860, 512, 12>(GINX_bootstrappingKey_FFT, params);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 1024:
                    switch (digitsG2){
                        case 2:
                            GPUSetup_core<860, 1024, 2>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 4:
                            GPUSetup_core<860, 1024, 4>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 6:
                            GPUSetup_core<860, 1024, 6>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 8:
                            GPUSetup_core<860, 1024, 8>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 10:
                            GPUSetup_core<860, 1024, 10>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 12:
                            GPUSetup_core<860, 1024, 12>(GINX_bootstrappingKey_FFT, params);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 2048:
                    switch (digitsG2){
                        case 2:
                            GPUSetup_core<860, 2048, 2>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 4:
                            GPUSetup_core<860, 2048, 4>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 6:
                            GPUSetup_core<860, 2048, 6>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 8:
                            GPUSetup_core<860, 2048, 8>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 10:
                            GPUSetup_core<860, 2048, 10>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 12:
                            GPUSetup_core<860, 2048, 12>(GINX_bootstrappingKey_FFT, params);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                default:
                    std::cerr << "Unsupported N\n";
                    exit(1);
            }
            break;
        case 890: // RTX40 series
            switch (NHalf){
                case 512:
                    switch (digitsG2){
                        case 2:
                            GPUSetup_core<890, 512, 2>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 4:
                            GPUSetup_core<890, 512, 4>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 6:
                            GPUSetup_core<890, 512, 6>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 8:
                            GPUSetup_core<890, 512, 8>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 10:
                            GPUSetup_core<890, 512, 10>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 12:
                            GPUSetup_core<890, 512, 12>(GINX_bootstrappingKey_FFT, params);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 1024:
                    switch (digitsG2){
                        case 2:
                            GPUSetup_core<890, 1024, 2>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 4:
                            GPUSetup_core<890, 1024, 4>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 6:
                            GPUSetup_core<890, 1024, 6>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 8:
                            GPUSetup_core<890, 1024, 8>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 10:
                            GPUSetup_core<890, 1024, 10>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 12:
                            GPUSetup_core<890, 1024, 12>(GINX_bootstrappingKey_FFT, params);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 2048:
                    switch (digitsG2){
                        case 2:
                            GPUSetup_core<890, 2048, 2>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 4:
                            GPUSetup_core<890, 2048, 4>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 6:
                            GPUSetup_core<890, 2048, 6>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 8:
                            GPUSetup_core<890, 2048, 8>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 10:
                            GPUSetup_core<890, 2048, 10>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 12:
                            GPUSetup_core<890, 2048, 12>(GINX_bootstrappingKey_FFT, params);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                default:
                    std::cerr << "Unsupported N\n";
                    exit(1);
            }
            break;
        case 900: // H100
            switch (NHalf){
                case 512:
                    switch (digitsG2){
                        case 2:
                            GPUSetup_core<900, 512, 2>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 4:
                            GPUSetup_core<900, 512, 4>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 6:
                            GPUSetup_core<900, 512, 6>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 8:
                            GPUSetup_core<900, 512, 8>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 10:
                            GPUSetup_core<900, 512, 10>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 12:
                            GPUSetup_core<900, 512, 12>(GINX_bootstrappingKey_FFT, params);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 1024:
                    switch (digitsG2){
                        case 2:
                            GPUSetup_core<900, 1024, 2>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 4:
                            GPUSetup_core<900, 1024, 4>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 6:
                            GPUSetup_core<900, 1024, 6>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 8:
                            GPUSetup_core<900, 1024, 8>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 10:
                            GPUSetup_core<900, 1024, 10>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 12:
                            GPUSetup_core<900, 1024, 12>(GINX_bootstrappingKey_FFT, params);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 2048:
                    switch (digitsG2){
                        case 2:
                            GPUSetup_core<900, 2048, 2>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 4:
                            GPUSetup_core<900, 2048, 4>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 6:
                            GPUSetup_core<900, 2048, 6>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 8:
                            GPUSetup_core<900, 2048, 8>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 10:
                            GPUSetup_core<900, 2048, 10>(GINX_bootstrappingKey_FFT, params);
                            break;
                        case 12:
                            GPUSetup_core<900, 2048, 12>(GINX_bootstrappingKey_FFT, params);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                default:
                    std::cerr << "Unsupported N\n";
                    exit(1);
            }
            break;
        default:
            std::cerr << "Unsupported GPU architecture\n";
            exit(1);
    }
    
    std::cout << "GPU Setup Done\n";
}

template<uint32_t arch, uint32_t FFT_dimension, uint32_t FFT_num>
void GPUSetup_core(std::shared_ptr<std::vector<std::vector<std::vector<std::shared_ptr<std::vector<std::vector<std::vector<Complex>>>>>>>> GINX_bootstrappingKey_FFT,
        const std::shared_ptr<RingGSWCryptoParams> params)
{
    /* Parameters Set */
    auto Q            = params->GetQ();
    NativeInteger QHalf = Q >> 1;
    NativeInteger::SignedNativeInt Q_int = Q.ConvertToInt();
    uint32_t N            = params->GetN();
    uint32_t NHalf     = N >> 1;
    uint32_t n = (*GINX_bootstrappingKey_FFT)[0][0].size();
    uint32_t digitsG2 = params->GetDigitsG() << 1;
    uint32_t baseG = params->GetBaseG();
    uint32_t RGSW_size = digitsG2 * 2 * NHalf;

    int SM_count = gpuInfoList[0].multiprocessorCount;

    /* Create cuda streams */
    streams.resize(gpuInfoList[0].multiprocessorCount);
    for (int s = 0; s < gpuInfoList[0].multiprocessorCount; s++) {
        cudaStreamCreate(&streams[s]);
    }

    /* Configure cuFFTDx */
    using FFT     = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::forward>() + cufftdx::ElementsPerThread<8>() +
                        cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<FFT_num>() + cufftdx::SM<arch>());

    using IFFT     = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::inverse>() + cufftdx::ElementsPerThread<8>() +
                            cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<2>() + cufftdx::SM<arch>());

    using FFT_multi      = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::forward>() +
                            cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<2>() + cufftdx::SM<arch>());

    using IFFT_multi     = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::inverse>() +
                            cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<2>() + cufftdx::SM<arch>());

    using FFT_fwd  = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::forward>() + cufftdx::ElementsPerThread<8>() +
                        cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<1>() + cufftdx::SM<arch>());

    /* Increase max shared memory */
    // Single block Bootstrapping shared memory size
    if(FFT::shared_memory_size > 65536)
        cudaFuncSetAttribute(bootstrappingSingleBlock<FFT, IFFT>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    else if(FFT::shared_memory_size > 32768)
        cudaFuncSetAttribute(bootstrappingSingleBlock<FFT, IFFT>, cudaFuncAttributePreferredSharedMemoryCarveout, 64);
    else
        cudaFuncSetAttribute(bootstrappingSingleBlock<FFT, IFFT>, cudaFuncAttributePreferredSharedMemoryCarveout, 32);
    cudaFuncSetAttribute(bootstrappingSingleBlock<FFT, IFFT>, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size);

    // Multi block Bootstrapping shared memory size
    if(FFT_multi::shared_memory_size > 65536)
        cudaFuncSetAttribute(bootstrappingMultiBlock<FFT_multi, IFFT_multi>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    else if(FFT_multi::shared_memory_size > 32768)
        cudaFuncSetAttribute(bootstrappingMultiBlock<FFT_multi, IFFT_multi>, cudaFuncAttributePreferredSharedMemoryCarveout, 64);
    else
        cudaFuncSetAttribute(bootstrappingMultiBlock<FFT_multi, IFFT_multi>, cudaFuncAttributePreferredSharedMemoryCarveout, 32);
    cudaFuncSetAttribute(bootstrappingMultiBlock<FFT_multi, IFFT_multi>, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT_multi::shared_memory_size);

    // cuFFTDx Forward shared memory size
    cudaFuncSetAttribute(cuFFTDxFWD<FFT>, cudaFuncAttributePreferredSharedMemoryCarveout, 64);
    cudaFuncSetAttribute(cuFFTDxFWD<FFT>, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT_fwd::shared_memory_size);

    /* Initialize twiddle table */
    Complex *twiddleTable;
    cudaMallocHost((void**)&twiddleTable, 2 * NHalf * sizeof(Complex));
    for (size_t j = 0; j < NHalf; j++) {
        twiddleTable[j] = Complex(cos(static_cast<double>(2 * M_PI * j)/ (N << 1)), sin(static_cast<double>(2 * M_PI * j) / (N << 1)));
    }
    for (size_t j = NHalf; j < N; j++) {
        twiddleTable[j] = Complex(cos(static_cast<double>(-2 * M_PI * (j - NHalf)) / (N << 1)), sin(static_cast<double>(-2 * M_PI * (j - NHalf)) / (N << 1)));
    }
    // Bring twiddle table to GPU
    cudaMalloc(&twiddleTable_CUDA, 2 * NHalf * sizeof(Complex_d));
    cudaMemcpy(twiddleTable_CUDA, twiddleTable, 2 * NHalf * sizeof(Complex_d), cudaMemcpyHostToDevice);
    cudaFreeHost(twiddleTable);

    /* Initialize params_CUDA */
    uint64_t *paramters;
    cudaMallocHost((void**)&paramters, 6 * sizeof(uint64_t));
    paramters[0] = N;
    paramters[1] = n;
    paramters[2] = static_cast<uint64_t>(Q_int);
    paramters[3] = digitsG2;
    paramters[4] = baseG;
    auto it = synchronizationMap.find({arch, FFT_dimension});
    if (it != synchronizationMap.end() && it->second != 0) {
        paramters[5] = static_cast<uint64_t>(it->second);
    } else {
        std::cerr << "Hasn't tested on this GPU yet, please contact r11922138@ntu.edu.tw" << std::endl;
        exit(1);
    }
    // Bring params_CUDA to GPU
    cudaMalloc(&params_CUDA, 6 * sizeof(uint64_t));
    cudaMemcpy(params_CUDA, paramters, 6 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaFreeHost(paramters);

    /* Initialize bootstrapping key */
    Complex *bootstrappingKey;
    cudaMallocHost((void**)&bootstrappingKey, 2 * n * RGSW_size * sizeof(Complex)); // ternery needs two secret keys
    for(int num_key = 0; num_key < 2; num_key++){
        for(int i = 0; i < n; i++){
            for(int l = 0; l < digitsG2; l++){
                for(int m = 0; m < 2; m++){
                    std::vector<Complex> temp = (*(*GINX_bootstrappingKey_FFT)[0][num_key][i])[l][m];
                    DiscreteFourierTransform::NegacyclicInverseTransform(temp);
                    for(int j = 0; j < NHalf; j++){
                        bootstrappingKey[num_key*n*RGSW_size + i*RGSW_size + l*2*NHalf + m*NHalf + j] = Complex(temp[j].real(), temp[j + NHalf].real());
                    }
                }
            }
        }
    }
    // Bring bootstrapping key to GPU
    cudaMalloc(&GINX_bootstrappingKey_CUDA, 2 * n * RGSW_size * sizeof(Complex_d));
    cudaMemcpy(GINX_bootstrappingKey_CUDA, bootstrappingKey, 2 * n * RGSW_size * sizeof(Complex_d), cudaMemcpyHostToDevice);
    cudaFreeHost(bootstrappingKey);
    cuFFTDxFWD<FFT_fwd><<<2 * n * digitsG2 * 2, FFT_fwd::block_dim, FFT_fwd::shared_memory_size>>>(GINX_bootstrappingKey_CUDA, twiddleTable_CUDA);
    cudaDeviceSynchronize();

    /* Initialize monomial array */
    Complex *monomial_arr;
    cudaMallocHost((void**)&monomial_arr, 2 * N * NHalf * sizeof(Complex));
    // loop for positive values of m
    std::vector<Complex> monomial(N, Complex(0.0, 0.0));
    for (size_t m_count = 0; m_count < N; ++m_count) {
        NativePoly monomial_t    = params->GetMonomial(m_count);
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
        NativePoly monomialNeg_t = params->GetMonomial(m_count);
        monomialNeg_t.SetFormat(Format::COEFFICIENT);
        for (size_t i = 0; i < N; ++i) {
            NativeInteger::SignedNativeInt d = (monomialNeg_t[i] < QHalf) ? monomialNeg_t[i].ConvertToInt() : (monomialNeg_t[i].ConvertToInt() - Q_int);
            monomialNeg[i] = Complex (static_cast<double>(d), 0);
        }
        for (size_t i = 0; i < NHalf; ++i) 
            monomial_arr[m_count*NHalf + i] = Complex(monomialNeg[i].real(), monomialNeg[i + NHalf].real());
    }
    // Bring monomial array to GPU
    cudaMalloc(&monomial_CUDA, 2 * N * NHalf * sizeof(Complex_d));
    cudaMemcpy(monomial_CUDA, monomial_arr, 2 * N * NHalf * sizeof(Complex_d), cudaMemcpyHostToDevice);
    cudaFreeHost(monomial_arr);
    cuFFTDxFWD<FFT_fwd><<<2 * N, FFT_fwd::block_dim, FFT_fwd::shared_memory_size>>>(monomial_CUDA, twiddleTable_CUDA);
    cudaDeviceSynchronize();

    /* Allocate ct_CUDA on GPU */
    cudaMalloc(&ct_CUDA, SM_count * 2 * NHalf * sizeof(Complex_d));

    /* Allocate dct_CUDA on GPU */
    cudaMalloc(&dct_CUDA, SM_count * digitsG2 * NHalf * sizeof(Complex_d));
}

void AddToAccCGGI_CUDA(const std::shared_ptr<RingGSWCryptoParams> params, const NativeVector& a, std::vector<std::vector<Complex>>& acc_d, std::string mode)
{   
    /* Parameters Set */
    uint32_t N            = params->GetN();
    uint32_t NHalf     = N >> 1;
    uint32_t digitsG2 = params->GetDigitsG() << 1;
    uint32_t arch = gpuInfoList[0].major * 100 + gpuInfoList[0].minor * 10;

    /* Determine template of AddToAccCGGI_CUDA_core */
    switch (arch){
        case 700: // V100
            switch (NHalf){
                case 512:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<700, 512, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<700, 512, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<700, 512, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<700, 512, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<700, 512, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<700, 512, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 1024:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<700, 1024, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<700, 1024, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<700, 1024, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<700, 1024, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<700, 1024, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<700, 1024, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 2048:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<700, 2048, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<700, 2048, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<700, 2048, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<700, 2048, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<700, 2048, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<700, 2048, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                default:
                    std::cerr << "Unsupported N\n";
                    exit(1);
            }
            break;
        case 800: // A100
            switch (NHalf){
                case 512:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<800, 512, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<800, 512, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<800, 512, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<800, 512, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<800, 512, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<800, 512, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 1024:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<800, 1024, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<800, 1024, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<800, 1024, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<800, 1024, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<800, 1024, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<800, 1024, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 2048:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<800, 2048, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<800, 2048, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<800, 2048, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<800, 2048, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<800, 2048, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<800, 2048, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                default:
                    std::cerr << "Unsupported N\n";
                    exit(1);
            }
            break;
        case 860: // RTX30 series
            switch (NHalf){
                case 512:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<860, 512, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<860, 512, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<860, 512, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<860, 512, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<860, 512, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<860, 512, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 1024:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<860, 1024, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<860, 1024, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<860, 1024, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<860, 1024, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<860, 1024, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<860, 1024, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 2048:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<860, 2048, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<860, 2048, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<860, 2048, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<860, 2048, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<860, 2048, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<860, 2048, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                default:
                    std::cerr << "Unsupported N\n";
                    exit(1);
            }
            break;
        case 890: // RTX40 series
            switch (NHalf){
                case 512:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<890, 512, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<890, 512, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<890, 512, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<890, 512, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<890, 512, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<890, 512, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 1024:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<890, 1024, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<890, 1024, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<890, 1024, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<890, 1024, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<890, 1024, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<890, 1024, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 2048:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<890, 2048, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<890, 2048, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<890, 2048, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<890, 2048, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<890, 2048, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<890, 2048, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                default:
                    std::cerr << "Unsupported N\n";
                    exit(1);
            }
            break;
        case 900: // H100
            switch (NHalf){
                case 512:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<900, 512, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<900, 512, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<900, 512, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<900, 512, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<900, 512, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<900, 512, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 1024:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<900, 1024, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<900, 1024, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<900, 1024, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<900, 1024, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<900, 1024, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<900, 1024, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 2048:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<900, 2048, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<900, 2048, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<900, 2048, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<900, 2048, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<900, 2048, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<900, 2048, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                default:
                    std::cerr << "Unsupported N\n";
                    exit(1);
            }
            break;
        default:
            std::cerr << "Unsupported GPU architecture\n";
            exit(1);
    }
}

template<uint32_t arch, uint32_t FFT_dimension, uint32_t FFT_num>
void AddToAccCGGI_CUDA_core(const std::shared_ptr<RingGSWCryptoParams> params, const NativeVector& a, std::vector<std::vector<Complex>>& acc_d, std::string mode)
{   
    /* parameters set */
    auto mod        = a.GetModulus();
    uint32_t modInt = mod.ConvertToInt();
    auto Q            = params->GetQ();
    NativeInteger QHalf = Q >> 1;
    NativeInteger::SignedNativeInt Q_int = Q.ConvertToInt();
    uint32_t N         = params->GetN();
    uint32_t NHalf     = N >> 1;
    uint32_t n =  a.GetLength();
    uint32_t M      = 2 * params->GetN();
    uint32_t digitsG2 = params->GetDigitsG() << 1;

    /* Configure cuFFTDx */
    using FFT     = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::forward>() + cufftdx::ElementsPerThread<8>() +
                        cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<FFT_num>() + cufftdx::SM<arch>());

    using IFFT     = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::inverse>() + cufftdx::ElementsPerThread<8>() +
                            cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<2>() + cufftdx::SM<arch>());

    using FFT_multi      = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::forward>() +
                            cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<2>() + cufftdx::SM<arch>());

    using IFFT_multi     = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::inverse>() +
                            cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<2>() + cufftdx::SM<arch>());

    /* Check whether block size exceeds cuda limitation */
    if(mode == "SINGLE"){
        if((NHalf / FFT::elements_per_thread * digitsG2) > gpuInfoList[0].maxThreadsPerBlock){
            std::cerr << "Exceed Maximum blocks per threads (" << gpuInfoList[0].maxThreadsPerBlock << ")\n";
            std::cerr << "Using " << (NHalf / FFT::elements_per_thread * digitsG2) << " threads" << ")\n";
            std::cerr << "NHalf: " << NHalf << "FFT::elements_per_thread: " << FFT::elements_per_thread << "digitsG2: " << digitsG2 << ")\n";
            exit(1);
        }
    }
    else if(mode == "MULTI"){
        if((NHalf / FFT_multi::elements_per_thread * 2) > gpuInfoList[0].maxThreadsPerBlock){
            std::cerr << "Exceed Maximum blocks per threads (" << gpuInfoList[0].maxThreadsPerBlock << ")\n";
            std::cerr << "Using " << (NHalf / FFT_multi::elements_per_thread * digitsG2) << " threads" << ")\n";
            std::cerr << "NHalf: " << NHalf << "FFT::elements_per_thread: " << FFT_multi::elements_per_thread << ")\n";
            exit(1);
        }
    }

    /* Check whether shared memory size exceeds cuda limitation */
    if(mode == "SINGLE"){
        if(FFT::shared_memory_size > gpuInfoList[0].sharedMemoryPerBlock){
            std::cerr << "Exceed Maximum sharedMemoryPerBlock ("<< gpuInfoList[0].sharedMemoryPerBlock << ")\n";
            std::cerr << "Declare "<< FFT::shared_memory_size << " now" << "\n";
            exit(1);
        }
    }
    else if(mode == "MULTI"){
        if(FFT_multi::shared_memory_size > gpuInfoList[0].sharedMemoryPerBlock){
            std::cerr << "Exceed Maximum sharedMemoryPerBlock ("<< gpuInfoList[0].sharedMemoryPerBlock << ")\n";
            std::cerr << "Declare "<< FFT_multi::shared_memory_size << " now" << "\n";
            exit(1);
        }
    }

    /* Initialize a_arr */
    uint64_t* a_arr;
    uint64_t* a_CUDA;
    cudaMallocHost((void**)&a_arr, n * sizeof(uint64_t));
    for (size_t i = 0; i < n; ++i) {
        a_arr[i] = (mod.ModSub(a[i], mod) * (M / modInt)).ConvertToInt();
    }
    // Bring a to GPU
    cudaMalloc(&a_CUDA, n * sizeof(uint64_t));
    cudaMemcpy(a_CUDA, a_arr, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaFreeHost(a_arr);

    /* Initialize acc_d_arr */
    Complex* acc_d_arr;
    Complex_d* acc_CUDA;
    cudaMallocHost((void**)&acc_d_arr, 2 * NHalf * sizeof(Complex));
    for(int i = 0; i < 2; i++)
        for(int j = 0; j < NHalf; j++)
            acc_d_arr[i*NHalf + j] = Complex(acc_d[i][j].real(), acc_d[i][j + NHalf].real());   
    // Bring acc_d to GPU
    cudaMalloc(&acc_CUDA, 2 * NHalf * sizeof(Complex_d));
    cudaMemcpy(acc_CUDA, acc_d_arr, 2 * NHalf * sizeof(Complex_d), cudaMemcpyHostToDevice);

    /* Launch boostrapping kernel */
    if(mode == "SINGLE"){
        bootstrappingSingleBlock<FFT, IFFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>
            (acc_CUDA, ct_CUDA, dct_CUDA, a_CUDA, monomial_CUDA, twiddleTable_CUDA, params_CUDA, GINX_bootstrappingKey_CUDA);
    }
    else if(mode == "MULTI"){
        void *kernelArgs[] = {(void *)&acc_CUDA, (void *)&ct_CUDA, (void *)&dct_CUDA, (void *)&a_CUDA, 
            (void *)&monomial_CUDA, (void *)&twiddleTable_CUDA, (void *)&params_CUDA, (void *)&GINX_bootstrappingKey_CUDA};
        cudaLaunchCooperativeKernel((void*)(bootstrappingMultiBlock<FFT_multi, IFFT_multi>), digitsG2/2, FFT_multi::block_dim, kernelArgs, FFT_multi::shared_memory_size);
    }
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    /* Copy acc_d_arr back to acc_d */
    cudaMemcpy(acc_d_arr, acc_CUDA, 2 * NHalf * sizeof(Complex_d), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < NHalf; j++){
            acc_d[i][j] = Complex(acc_d_arr[i*NHalf + j].real(), 0);
            acc_d[i][j + NHalf] = Complex(acc_d_arr[i*NHalf + j].imag(), 0);
        }
    }

    // /* Debugging txt files */
    // std::ofstream outputFile;
    // outputFile.open("acc.txt", std::ios::out);
    // for(uint32_t i = 0; i < 2; i++)
    //     for(uint32_t j = 0; j < (N >> 1); j++)
    //         outputFile << "(" << acc_d_arr[i*(N >> 1) + j].real() << ", " << acc_d_arr[i*(N >> 1) + j].imag() << ")" << std::endl;
    // outputFile.close();

    // // Copy the ct_CUDA back to the host
    // Complex* ct_arr;
    // cudaMallocHost((void**)&ct_arr, 2 * NHalf * sizeof(Complex));
    // cudaMemcpy(ct_arr, ct_CUDA, 2 * NHalf * sizeof(Complex_d), cudaMemcpyDeviceToHost);

    // outputFile.open("ct.txt", std::ios::out);
    // for(uint32_t i = 0; i < 2; i++)
    //     for(uint32_t j = 0; j < (N >> 1); j++)
    //         outputFile << "(" << ct_arr[i*(N >> 1) + j].real() << ", " << ct_arr[i*(N >> 1) + j].imag() << ")" << std::endl;
    // outputFile.close();

    // // Copy the dct_CUDA back to the host
    // Complex* dct_arr;
    // cudaMallocHost((void**)&dct_arr, digitsG2 * NHalf * sizeof(Complex));
    // cudaMemcpy(dct_arr, dct_CUDA, digitsG2 * NHalf * sizeof(Complex_d), cudaMemcpyDeviceToHost);

    // outputFile.open("dct.txt", std::ios::out);
    // for(uint32_t i = 0; i < digitsG2; i++)
    //     for(uint32_t j = 0; j < (N >> 1); j++)
    //         outputFile << "(" << dct_arr[i*(N >> 1) + j].real() << ", " << dct_arr[i*(N >> 1) + j].imag() << ")" << std::endl;
    // outputFile.close();
}

void AddToAccCGGI_CUDA(const std::shared_ptr<RingGSWCryptoParams> params, const std::vector<NativeVector>& a, 
        std::vector<std::vector<std::vector<Complex>>>& acc_d, std::string mode)
{   
    /* Parameters Set */
    uint32_t N            = params->GetN();
    uint32_t NHalf     = N >> 1;
    uint32_t digitsG2 = params->GetDigitsG() << 1;
    uint32_t arch = gpuInfoList[0].major * 100 + gpuInfoList[0].minor * 10;

    /* Determine template of AddToAccCGGI_CUDA_core */
    switch (arch){
        case 700: // V100
            switch (NHalf){
                case 512:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<700, 512, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<700, 512, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<700, 512, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<700, 512, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<700, 512, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<700, 512, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 1024:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<700, 1024, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<700, 1024, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<700, 1024, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<700, 1024, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<700, 1024, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<700, 1024, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 2048:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<700, 2048, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<700, 2048, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<700, 2048, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<700, 2048, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<700, 2048, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<700, 2048, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                default:
                    std::cerr << "Unsupported N\n";
                    exit(1);
            }
            break;
        case 800: // A100
            switch (NHalf){
                case 512:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<800, 512, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<800, 512, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<800, 512, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<800, 512, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<800, 512, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<800, 512, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 1024:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<800, 1024, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<800, 1024, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<800, 1024, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<800, 1024, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<800, 1024, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<800, 1024, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 2048:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<800, 2048, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<800, 2048, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<800, 2048, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<800, 2048, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<800, 2048, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<800, 2048, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                default:
                    std::cerr << "Unsupported N\n";
                    exit(1);
            }
            break;
        case 860: // RTX30 series
            switch (NHalf){
                case 512:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<860, 512, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<860, 512, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<860, 512, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<860, 512, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<860, 512, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<860, 512, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 1024:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<860, 1024, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<860, 1024, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<860, 1024, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<860, 1024, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<860, 1024, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<860, 1024, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 2048:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<860, 2048, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<860, 2048, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<860, 2048, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<860, 2048, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<860, 2048, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<860, 2048, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                default:
                    std::cerr << "Unsupported N\n";
                    exit(1);
            }
            break;
        case 890: // RTX40 series
            switch (NHalf){
                case 512:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<890, 512, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<890, 512, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<890, 512, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<890, 512, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<890, 512, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<890, 512, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 1024:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<890, 1024, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<890, 1024, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<890, 1024, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<890, 1024, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<890, 1024, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<890, 1024, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 2048:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<890, 2048, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<890, 2048, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<890, 2048, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<890, 2048, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<890, 2048, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<890, 2048, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                default:
                    std::cerr << "Unsupported N\n";
                    exit(1);
            }
            break;
        case 900: // H100
            switch (NHalf){
                case 512:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<900, 512, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<900, 512, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<900, 512, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<900, 512, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<900, 512, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<900, 512, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 1024:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<900, 1024, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<900, 1024, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<900, 1024, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<900, 1024, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<900, 1024, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<900, 1024, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                case 2048:
                    switch (digitsG2){
                        case 2:
                            AddToAccCGGI_CUDA_core<900, 2048, 2>(params, a, acc_d, mode);
                            break;
                        case 4:
                            AddToAccCGGI_CUDA_core<900, 2048, 4>(params, a, acc_d, mode);
                            break;
                        case 6:
                            AddToAccCGGI_CUDA_core<900, 2048, 6>(params, a, acc_d, mode);
                            break;
                        case 8:
                            AddToAccCGGI_CUDA_core<900, 2048, 8>(params, a, acc_d, mode);
                            break;
                        case 10:
                            AddToAccCGGI_CUDA_core<900, 2048, 10>(params, a, acc_d, mode);
                            break;
                        case 12:
                            AddToAccCGGI_CUDA_core<900, 2048, 12>(params, a, acc_d, mode);
                            break;
                        default:
                            std::cerr << "Unsupported digitsG\n";
                            exit(1);
                    }
                    break;
                default:
                    std::cerr << "Unsupported N\n";
                    exit(1);
            }
            break;
        default:
            std::cerr << "Unsupported GPU architecture\n";
            exit(1);
    }
}

template<uint32_t arch, uint32_t FFT_dimension, uint32_t FFT_num>
void AddToAccCGGI_CUDA_core(const std::shared_ptr<RingGSWCryptoParams> params, const std::vector<NativeVector>& a, 
        std::vector<std::vector<std::vector<Complex>>>& acc_d, std::string mode)
{   
    /* parameters set */
    auto mod        = a[0].GetModulus();
    uint32_t modInt = mod.ConvertToInt();
    auto Q            = params->GetQ();
    NativeInteger QHalf = Q >> 1;
    NativeInteger::SignedNativeInt Q_int = Q.ConvertToInt();
    uint32_t N         = params->GetN();
    uint32_t NHalf     = N >> 1;
    uint32_t n =  a[0].GetLength();
    uint32_t M      = 2 * params->GetN();
    uint32_t digitsG2 = params->GetDigitsG() << 1;

    int bootstrap_num = acc_d.size();
    int SM_count = gpuInfoList[0].multiprocessorCount;

    /* Configure cuFFTDx */
    using FFT     = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::forward>() + cufftdx::ElementsPerThread<8>() +
                        cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<FFT_num>() + cufftdx::SM<arch>());

    using IFFT     = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::inverse>() + cufftdx::ElementsPerThread<8>() +
                            cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<2>() + cufftdx::SM<arch>());

    using FFT_multi      = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::forward>() +
                            cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<2>() + cufftdx::SM<arch>());

    using IFFT_multi     = decltype(cufftdx::Block() + cufftdx::Size<FFT_dimension>() + cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<cufftdx::fft_direction::inverse>() +
                            cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<2>() + cufftdx::SM<arch>());

    /* Check whether block size exceeds cuda limitation */
    if(mode == "SINGLE"){
        if((NHalf / FFT::elements_per_thread * digitsG2) > gpuInfoList[0].maxThreadsPerBlock){
            std::cerr << "Exceed Maximum blocks per threads (" << gpuInfoList[0].maxThreadsPerBlock << ")\n";
            std::cerr << "Using " << (NHalf / FFT::elements_per_thread * digitsG2) << " threads" << ")\n";
            std::cerr << "NHalf: " << NHalf << "FFT::elements_per_thread: " << FFT::elements_per_thread << "digitsG2: " << digitsG2 << ")\n";
            exit(1);
        }
    }
    else if(mode == "MULTI"){
        if((NHalf / FFT_multi::elements_per_thread * 2) > gpuInfoList[0].maxThreadsPerBlock){
            std::cerr << "Exceed Maximum blocks per threads (" << gpuInfoList[0].maxThreadsPerBlock << ")\n";
            std::cerr << "Using " << (NHalf / FFT_multi::elements_per_thread * digitsG2) << " threads" << ")\n";
            std::cerr << "NHalf: " << NHalf << "FFT::elements_per_thread: " << FFT_multi::elements_per_thread << ")\n";
            exit(1);
        }
    }

    /* Check whether shared memory size exceeds cuda limitation */
    if(mode == "SINGLE"){
        if(FFT::shared_memory_size > gpuInfoList[0].sharedMemoryPerBlock){
            std::cerr << "Exceed Maximum sharedMemoryPerBlock ("<< gpuInfoList[0].sharedMemoryPerBlock << ")\n";
            std::cerr << "Declare "<< FFT::shared_memory_size << " now" << "\n";
            exit(1);
        }
    }
    else if(mode == "MULTI"){
        if(FFT_multi::shared_memory_size > gpuInfoList[0].sharedMemoryPerBlock){
            std::cerr << "Exceed Maximum sharedMemoryPerBlock ("<< gpuInfoList[0].sharedMemoryPerBlock << ")\n";
            std::cerr << "Declare "<< FFT_multi::shared_memory_size << " now" << "\n";
            exit(1);
        }
    }

    /* Initialize a_arr */
    uint64_t* a_arr;
    uint64_t* a_CUDA;
    cudaMallocHost((void**)&a_arr, bootstrap_num * n * sizeof(uint64_t));
    for (int s = 0; s < bootstrap_num; s++)
        for (size_t i = 0; i < n; ++i)
            a_arr[s*n + i] = (mod.ModSub(a[s][i], mod) * (M / modInt)).ConvertToInt();
    cudaMalloc((void**)&a_CUDA, bootstrap_num * n * sizeof(uint64_t));

    /* Initialize acc_d_arr */
    Complex* acc_d_arr;
    Complex_d* acc_CUDA;
    cudaMallocHost((void**)&acc_d_arr, bootstrap_num * 2 * NHalf * sizeof(Complex));
    for (int s = 0; s < bootstrap_num; s++)
        for(int i = 0; i < 2; i++)
            for(int j = 0; j < NHalf; j++)
                acc_d_arr[s*2*NHalf + i*NHalf + j] = Complex(acc_d[s][i][j].real(), acc_d[s][i][j + NHalf].real());
    cudaMalloc((void**)&acc_CUDA, bootstrap_num * 2 * NHalf * sizeof(Complex_d));

    if(mode == "SINGLE"){
        for (int s = 0; s < bootstrap_num; s++) {
            cudaMemcpyAsync(a_CUDA + s*n, a_arr + s*n, n * sizeof(uint64_t), cudaMemcpyHostToDevice, streams[s % SM_count]);
            cudaMemcpyAsync(acc_CUDA + s*2*NHalf, acc_d_arr + s*2*NHalf, 2 * NHalf * sizeof(Complex_d), cudaMemcpyHostToDevice, streams[s % SM_count]);
            bootstrappingSingleBlock<FFT, IFFT><<<1, FFT::block_dim, FFT::shared_memory_size, streams[s % SM_count]>>>
                (acc_CUDA + s*2*NHalf, ct_CUDA + (s % SM_count)*2*NHalf, dct_CUDA + (s % SM_count)*digitsG2*NHalf, a_CUDA + s*n, monomial_CUDA, twiddleTable_CUDA, params_CUDA, GINX_bootstrappingKey_CUDA);
            cudaMemcpyAsync(acc_d_arr + s*2*NHalf, acc_CUDA + s*2*NHalf, 2 * NHalf * sizeof(Complex_d), cudaMemcpyDeviceToHost, streams[s % SM_count]);
        }
    }
    else if(mode == "MULTI"){
        Complex_d* acc_CUDA_offset, *ct_CUDA_offset, *dct_CUDA_offset;
        uint64_t* a_CUDA_offset;
        for (int s = 0; s < bootstrap_num; s++) {
            acc_CUDA_offset = acc_CUDA + s*2*NHalf;
            ct_CUDA_offset = ct_CUDA + (s % SM_count)*2*NHalf;
            dct_CUDA_offset = dct_CUDA + (s % SM_count)*digitsG2*NHalf;
            a_CUDA_offset = a_CUDA + s*n;
            cudaMemcpyAsync(a_CUDA + s*n, a_arr + s*n, n * sizeof(uint64_t), cudaMemcpyHostToDevice, streams[s % SM_count]);
            cudaMemcpyAsync(acc_CUDA + s*2*NHalf, acc_d_arr + s*2*NHalf, 2 * NHalf * sizeof(Complex_d), cudaMemcpyHostToDevice, streams[s % SM_count]);
            void *kernelArgs[] = {(void *)&acc_CUDA_offset, (void *)&ct_CUDA_offset, (void *)&dct_CUDA_offset, (void *)&a_CUDA_offset, 
                (void *)&monomial_CUDA, (void *)&twiddleTable_CUDA, (void *)&params_CUDA, (void *)&GINX_bootstrappingKey_CUDA};
            cudaLaunchCooperativeKernel((void*)(bootstrappingMultiBlock<FFT_multi, IFFT_multi>), digitsG2/2, FFT_multi::block_dim, 
                kernelArgs, FFT_multi::shared_memory_size, streams[s % SM_count]);
            cudaMemcpyAsync(acc_d_arr + s*2*NHalf, acc_CUDA + s*2*NHalf, 2 * NHalf * sizeof(Complex_d), cudaMemcpyDeviceToHost, streams[s % SM_count]);
        }
    }
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    /* Copy acc_d_arr back to acc_d */
    for (int s = 0; s < bootstrap_num; s++) {
        for(int i = 0; i < 2; i++){
            for(int j = 0; j < NHalf; j++){
                acc_d[s][i][j] = Complex(acc_d_arr[s*2*NHalf + i*NHalf + j].real(), 0);
                acc_d[s][i][j + NHalf] = Complex(acc_d_arr[s*2*NHalf + i*NHalf + j].imag(), 0);
            }
        }
    }

    /* Free memory */     
    cudaFreeHost(a_arr);
    cudaFreeHost(acc_d_arr);
    cudaFree(a_CUDA);
    cudaFree(acc_CUDA);
}

};  // namespace lbcrypto


//     /* Measure GPU bootstrapping time */
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);

//     for (int s = 0; s < bootstrap_num; s++) {
//         // Bring input ciphertext to GPU
//         cudaMalloc(&input_dev[s], 2 * fft_size * sizeof(Complex_d));
//         cudaMemcpyAsync(input_dev[s], input[s], 2 * fft_size * sizeof(Complex_d), cudaMemcpyHostToDevice, streams[s]);

//         bootstrapping_Baseline<FFT, IFFT><<<1, FFT::block_dim, 81920, streams[s]>>>(input_dev[s], bootstrappingKey_dev, twiddleTable);

//         // Copy the result back to the host
//         cudaMemcpyAsync(input[s], input_dev[s], 2 * fft_size * sizeof(Complex_d), cudaMemcpyDeviceToHost, streams[s]);
//     }

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     std::cout << bootstrap_num << " Bootstrapping GPU time : " << milliseconds << " ms\n";
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     /* Free memory */     
//     for (int s = 0; s < bootstrap_num; s++) {
//         cudaFree(input_dev[s]);
//         cudaFreeHost(input[s]);
//         cudaStreamDestroy(streams[s]);
//     }
//     cudaFreeHost(bootstrappingKey);
// }