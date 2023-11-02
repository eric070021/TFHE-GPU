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

std::vector<GPUInfo> gpuInfoList;

template<class FFT, class IFFT>
__launch_bounds__(FFT::max_threads_per_block * 6)
__global__ void bootstrapping_CUDA(Complex_d* acc_CUDA, Complex_d* ct_CUDA, Complex_d* dct_CUDA, uint64_t* a_CUDA, 
        Complex_d* monomial_CUDA, Complex_d* twiddleTable_CUDA, uint64_t* params_CUDA, Complex_d* GINX_bootstrappingKey_CUDA){

    uint32_t tid = ThisThreadRankInBlock();
    uint32_t bdim = ThisBlockSize();

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
    else{ // must meet 8 sync made by FFT
        for(uint32_t i = 0; i < 12; ++i)
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
        else{ // must meet 8 sync made by IFFT
            for(uint32_t i = 0; i < 12; ++i)
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
    else{ // must meet 8 sync made by IFFT
       for(uint32_t i = 0; i < 12; ++i)
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

void GPUSetup(std::shared_ptr<std::vector<std::vector<std::vector<std::shared_ptr<std::vector<std::vector<std::vector<Complex>>>>>>>> GINX_bootstrappingKey_FFT,const std::shared_ptr<RingGSWCryptoParams> params){
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
    auto Q            = params->GetQ();
    NativeInteger QHalf = Q >> 1;
    NativeInteger::SignedNativeInt Q_int = Q.ConvertToInt();
    uint32_t N            = params->GetN();
    uint32_t NHalf     = N >> 1;
    uint32_t n = (*GINX_bootstrappingKey_FFT)[0][0].size();
    uint32_t digitsG2 = params->GetDigitsG() << 1;
    uint32_t baseG = params->GetBaseG();
    uint32_t RGSW_size = digitsG2 * 2 * NHalf;

    /* Determine the size of FFT */

    /* Increase max shared memory */
    // Check whether shared memory size exceeds cuda limitation
    if((FFT::shared_memory_size * digitsG2) > gpuInfoList[0].sharedMemoryPerBlock){
        std::cerr << "Exceed Maximum sharedMemoryPerBlock ("<< gpuInfoList[0].sharedMemoryPerBlock << ")\n";
        exit(1);
    }

    // Bootstrapping shared memory size
    if((FFT::shared_memory_size * digitsG2) > 65536)
        cudaFuncSetAttribute(bootstrapping_CUDA<FFT, IFFT>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    else if((FFT::shared_memory_size * digitsG2) > 32768)
        cudaFuncSetAttribute(bootstrapping_CUDA<FFT, IFFT>, cudaFuncAttributePreferredSharedMemoryCarveout, 64);
    else
        cudaFuncSetAttribute(bootstrapping_CUDA<FFT, IFFT>, cudaFuncAttributePreferredSharedMemoryCarveout, 32);
    cudaFuncSetAttribute(bootstrapping_CUDA<FFT, IFFT>, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size * digitsG2);

    // cuFFTDx Forward shared memory size
    cudaFuncSetAttribute(cuFFTDxFWD<FFT>, cudaFuncAttributePreferredSharedMemoryCarveout, 64);
    cudaFuncSetAttribute(cuFFTDxFWD<FFT>, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size);

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
    cudaMallocHost((void**)&paramters, 5 * sizeof(uint64_t));
    paramters[0] = N;
    paramters[1] = n;
    paramters[2] = static_cast<uint64_t>(Q_int);
    paramters[3] = digitsG2;
    paramters[4] = baseG;
    // Bring params_CUDA to GPU
    cudaMalloc(&params_CUDA, 5 * sizeof(uint64_t));
    cudaMemcpy(params_CUDA, paramters, 5 * sizeof(uint64_t), cudaMemcpyHostToDevice);
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
    cuFFTDxFWD<FFT><<<2 * n * digitsG2 * 2, FFT::block_dim, FFT::shared_memory_size>>>(GINX_bootstrappingKey_CUDA, twiddleTable_CUDA);
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
    cuFFTDxFWD<FFT><<<2 * N, FFT::block_dim, FFT::shared_memory_size>>>(monomial_CUDA, twiddleTable_CUDA);
    cudaDeviceSynchronize();

    /* Allocate ct_CUDA on GPU */
    cudaMalloc(&ct_CUDA, 2 * NHalf * sizeof(Complex_d));

    /* Allocate dct_CUDA on GPU */
    cudaMalloc(&dct_CUDA, digitsG2 * NHalf * sizeof(Complex_d));

    std::cout << "GPU Setup Done\n";
}

void AddToAccCGGI_CUDA(const std::shared_ptr<RingGSWCryptoParams> params, const NativeVector& a, std::vector<std::vector<Complex>>& acc_d)
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

    /* Check whether block size exceeds cuda limitation */
    if((NHalf / FFT::elements_per_thread * digitsG2) > gpuInfoList[0].maxThreadsPerBlock){
        std::cerr << "Exceed Maximum blocks per threads (" << gpuInfoList[0].maxThreadsPerBlock << ")\n";
        std::cerr << "NHalf: " << NHalf << "FFT::elements_per_thread: " << FFT::elements_per_thread << "digitsG2: " << digitsG2 << ")\n";
        exit(1);
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
    dim3 bootstrapping_block_dim(NHalf / FFT::elements_per_thread, digitsG2, 1);
    uint32_t bootstrapping_shared_mem_size = FFT::shared_memory_size * digitsG2;
    bootstrapping_CUDA<FFT, IFFT><<<1, bootstrapping_block_dim, bootstrapping_shared_mem_size>>>
        (acc_CUDA, ct_CUDA, dct_CUDA, a_CUDA, monomial_CUDA, twiddleTable_CUDA, params_CUDA, GINX_bootstrappingKey_CUDA);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    /* Copy the acc_d_arr to acc_d */
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

};  // namespace lbcrypto

// int main(){
//     /* Set number of bootstrapping */
//     const int bootstrap_num = 4096;

//     /* Initialize bootstrapping key */
//     Complex *bootstrappingKey;
//     cudaMallocHost((void**)&bootstrappingKey, 500 * RGSW_size * sizeof(Complex));
//     for(int i = 0; i < 500; i++)
//         for(int j = 0; j < digitsG2 * 2 * fft_size; j++)
//             bootstrappingKey[i*RGSW_size + j] = Complex(j, -j);

//     /* Bring bootstrapping key to GPU */
//     Complex_d* bootstrappingKey_dev;
//     cudaMalloc(&bootstrappingKey_dev, 500 * RGSW_size * sizeof(Complex_d));
//     cudaMemcpy(bootstrappingKey_dev, bootstrappingKey, 500 * RGSW_size * sizeof(Complex_d), cudaMemcpyHostToDevice);

//     /* Precompue twiddle table */
//     precomputeTable();

//     /* Increase max shared memory */
//     cudaFuncSetAttribute(bootstrapping_Baseline<FFT, IFFT>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
//     cudaFuncSetAttribute(bootstrapping_Baseline<FFT, IFFT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
//         81920);

//     // Create CUDA streams for parallel gates.
//     cudaStream_t streams[bootstrap_num];
//     for (int s = 0; s < bootstrap_num; s++) {
//         cudaStreamCreate(&streams[s]);
//     }

//     /* Initialize input ciphertext RLWE */
//     Complex *input[bootstrap_num];
//     Complex_d *input_dev[bootstrap_num];
//     for (int s = 0; s < bootstrap_num; s++) {
//         cudaMallocHost((void**)&input[s], 2 * fft_size * sizeof(Complex));
//         for(int i = 0; i < 2; i++)
//             for(int j = 0; j < fft_size; j++)
//                 input[s][i*fft_size + j] = Complex(j, j);
//     }

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

//     /* print ciphertext */
//     // for(int i = 0; i < 2; i++){
//     //     for(int j = 0; j < fft_size; j++){
//     //         std::cout << input[i*fft_size + j] << " ";
//     //     }
//     //     std::cout << std::endl;
//     // }

//     return 0;
// }


// Complex_d *input_dev;
// size_t input_pitch;
// cudaMallocPitch(&input_dev, &input_pitch, 2 * sizeof(Complex_d), N);
// cudaMemcpy2D(input_dev, input_pitch, input, 2 * sizeof(Complex_d), 2 * sizeof(Complex_d), N, cudaMemcpyHostToDevice);

// cudaMemcpy2D(input, 2 * sizeof(Complex_d), input_dev, input_pitch, 2 * sizeof(Complex_d), N, cudaMemcpyDeviceToHost);

// cudaFuncSetAttribute(bootstrapping_Baseline, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
// cudaFuncSetAttribute(bootstrapping_Baseline, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376);

// cudaFuncSetAttribute(bootstrapping_Baseline<FFT, IFFT>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
// cudaFuncSetAttribute(bootstrapping_Baseline<FFT, IFFT>, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376);