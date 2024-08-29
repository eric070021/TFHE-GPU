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

#ifndef __LWE_OPERATIONS_H__
#define __LWE_OPERATIONS_H__

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
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
    static std::shared_ptr<std::vector<LWECiphertext>> CiphertextMulMatrix_CUDA(const std::shared_ptr<BinFHECryptoParams> params,
         const std::vector<LWECiphertext>& ct, const std::vector<std::vector<int64_t>>& matrix, uint64_t modulus);

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
    static cublasHandle_t handle;
    #endif
};

}; // namespace lbcrypto

#endif  // __LWE_OPERATIONS_H__