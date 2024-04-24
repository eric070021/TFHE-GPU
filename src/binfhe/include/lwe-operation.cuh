#ifndef __LWE_OPERATIONS_H__
#define __LWE_OPERATIONS_H__

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublasXt.h>
#endif
#include <iostream>
#include <vector>
#include <chrono>
#include "binfhe-base-params.h"
#include "lwe-cryptoparameters.h"

namespace lbcrypto {
class GPULWEOperation {
public:
    /**
     * Function that implements the matrix multiplication of a ciphertext with a matrix (EvalDot in David's paper)
     *
     * @param params a shared pointer to BinFHECryptoParams scheme parameters
     * @param ct vector of input ciphertexts
     * @param matrix matrix to multiply with
     */
    static std::shared_ptr<std::vector<LWECiphertext>> CiphertextMulMatrix_CUDA(const std::shared_ptr<BinFHECryptoParams> params, const std::vector<LWECiphertext>& ct, const std::vector<std::vector<int64_t>>& matrix);

    /**
     * GPU setup wrapper
     *
     * @param numGPUs number of GPUs to use
     */
    static void GPUSetup(int numGPUs);

    /**
     * Clean GPU global memory
     */
    static void GPUClean();
private:
    // CUBLAS handle
    #ifdef __CUDACC__
    static cublasXtHandle_t handle;
    #endif
};

}; // namespace lbcrypto

#endif  // __LWE_OPERATIONS_H__