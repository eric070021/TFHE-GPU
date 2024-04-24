#include "lwe-operation.cuh"

// cuBLAS error checking macro
#define CUBLAS_ERROR_CHECK(call) \
do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

namespace lbcrypto {
    // Definition of the static member variables
    cublasXtHandle_t GPULWEOperation::handle;

    std::shared_ptr<std::vector<LWECiphertext>> GPULWEOperation::CiphertextMulMatrix_CUDA(const std::shared_ptr<BinFHECryptoParams> params, 
            const std::vector<LWECiphertext>& ct, const std::vector<std::vector<int64_t>>& matrix){
        
        /* Error check */
        if (ct.empty()) {
            std::cerr << "Input ciphertexts are empty." << std::endl;
            exit(EXIT_FAILURE);
        }
        if (matrix.empty()) {
            std::cerr << "Input matrix is empty." << std::endl;
            exit(EXIT_FAILURE);
        }
        if (ct.size() != matrix.size()) {
            std::cerr << "The number of rows of the matrix must be equal to the number of input ciphertexts." << std::endl;
            exit(EXIT_FAILURE);
        }

        /* Parameters Set */
        uint32_t M                  = params->GetLWEParams()->Getn() + 1;
        uint32_t N                  = matrix[0].size();
        uint32_t K                  = ct.size();
        uint64_t qKS                = params->GetLWEParams()->GetqKS().ConvertToInt();
        uint32_t n                  = params->GetLWEParams()->Getn();
        
        /* Allocate matrices on host */
        double *h_A = (double *)malloc(M * K * sizeof(double));
        double *h_B = (double *)malloc(K * N * sizeof(double));
        double *h_C = (double *)malloc(M * N * sizeof(double));
        
        /* Initialize matrices on host */
        for (int i = 0; i < K; ++i){
            NativeVector& ct_A = ct[i]->GetA();
            for (int j = 0; j < (M - 1); ++j){
                h_A[M*i + j] = ct_A[j].ConvertToDouble();
            }
            h_A[M*i + (M - 1)] = ct[i]->GetB().ConvertToDouble();
        }
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < N; ++j)
                h_B[N*i + j] = static_cast<double>(matrix[i][j]);

        /* Perform matrix multiplication on device */
        const double alpha = 1.0f, beta = 0.0f;
        cublasOperation_t transa = CUBLAS_OP_N, transb = CUBLAS_OP_T;
        CUBLAS_ERROR_CHECK(cublasXtDgemm(handle, transa, transb, M, N, K, &alpha, h_A, M, h_B, N, &beta, h_C, M));

        /* Serialize result matrix to ciphertexts */
        auto ct_res = std::make_shared<std::vector<LWECiphertext>> (N);
        for (int i = 0; i < N; ++i) {
            // A
            NativeVector a(n, qKS);
            for(int j = 0; j < (M-1); j++)
                a[j] = static_cast<uint64_t>(fmod(h_C[M*i + j], qKS));
            // B
            NativeInteger b (static_cast<uint64_t>(fmod(h_C[M*i + (M-1)], qKS)));

            (*ct_res)[i] = std::make_shared<LWECiphertextImpl>(LWECiphertextImpl(std::move(a), b));
        }

        /* Free host memory */
        free(h_A);
        free(h_B);
        free(h_C);
        
        return ct_res;
    }

    void GPULWEOperation::GPUSetup(int numGPUs){
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

        /* Set the device IDs to use */
        int* devices = (int *)malloc(GPUcount * sizeof(int));
        for(int i = 0; i < GPUcount; i++){
            devices[i] = i;
        }

        /* Initialize cuBLAS Xt */
        CUBLAS_ERROR_CHECK(cublasXtCreate(&handle));
        CUBLAS_ERROR_CHECK(cublasXtDeviceSelect(handle, GPUcount, devices));

        /* Free devices array */
        free(devices);
    }

    void GPULWEOperation::GPUClean(){
        CUBLAS_ERROR_CHECK(cublasXtDestroy(handle));
    }

}; // namespace lbcrypto
