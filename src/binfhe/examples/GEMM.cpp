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

#include "binfhecontext.h"

using namespace lbcrypto;

std::vector<LWECiphertext> CPUGEMM(BinFHEContext cc, std::vector<LWECiphertext> ct_vec, std::vector<std::vector<int64_t>> matrix) {
    auto LWEParams    = cc.GetParams()->GetLWEParams();
    uint64_t qKS = static_cast<uint64_t>(LWEParams->GetqKS().ConvertToInt());
    int n = LWEParams->Getn();
    int in_ct = ct_vec.size();
    int out_ct = matrix[0].size();

    std::vector<LWECiphertext> res;
    for (int i = 0; i < out_ct; i++) {
        NativeVector a(n, qKS);
        for (int j = 0; j < n; j++) {
            double temp = 0.0;
            for (int k = 0; k < in_ct; k++) {
                temp += ct_vec[k]->GetA(j).ConvertToDouble() * static_cast<double>(matrix[k][i]);
            }
            a[j] = static_cast<uint64_t>(fmod(temp, qKS));
        }
        double temp = 0.0;
        for (int k = 0; k < in_ct; k++) {
            temp += ct_vec[k]->GetB().ConvertToDouble() * static_cast<double>(matrix[k][i]);
        }
        NativeInteger b = static_cast<uint64_t>(fmod(temp, qKS));
        auto ct_res = std::make_shared<LWECiphertextImpl>(LWECiphertextImpl(std::move(a), b));
        res.push_back(ct_res);
    }
    return res;
}

void GEMMTest(){
    std::cout << "GEMM Test: " << std::endl;

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(STD128, true, 12, 0, GINX, false, 0, 1);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cc.GPUSetup();

    auto LWEParams    = cc.GetParams()->GetLWEParams();
    uint64_t qKS = static_cast<uint64_t>(LWEParams->GetqKS().ConvertToInt());
    int n = LWEParams->Getn();
    int in_ct = 1024;
    int out_ct = 1024;

    std::vector<LWECiphertext> ct_vec;
    for (int i = 0; i < in_ct; i++) {
        // A
        NativeVector a(n, qKS);
        for(int j = 0; j < n; j++)
            a[j] = rand() % qKS;
        // B
        NativeInteger b (rand() % qKS);

        auto ct1 = std::make_shared<LWECiphertextImpl>(LWECiphertextImpl(std::move(a), b));
        ct_vec.push_back(ct1);
    }
    std::vector<std::vector<int64_t>> matrix (in_ct, std::vector<int64_t>(out_ct, 1));
    for(int i = 0; i < in_ct; i++)
        for(int j = 0; j < out_ct; j++)
            matrix[i][j] = rand() % (1<<6);

    auto start_GPU = std::chrono::high_resolution_clock::now();
    auto ct_GPU_res = cc.CiphertextMulMatrix(ct_vec, matrix, qKS);
    auto end_GPU = std::chrono::high_resolution_clock::now();
    auto elapsed_GPU = std::chrono::duration_cast<std::chrono::milliseconds>(end_GPU-start_GPU);
    std::cout << "GPU CiphertextMulMatrix Time: " << (double)elapsed_GPU.count() << " ms" << std::endl;

    auto start_CPU = std::chrono::high_resolution_clock::now();
    auto ct_CPU_res = CPUGEMM(cc, ct_vec, matrix);
    auto end_CPU = std::chrono::high_resolution_clock::now();
    auto elapsed_CPU = std::chrono::duration_cast<std::chrono::milliseconds>(end_CPU-start_CPU);
    std::cout << "CPU CiphertextMulMatrix Time: " << (double)elapsed_CPU.count() << " ms" << std::endl;
    // // print ct_res
    // for (int i = 0; i < out_ct; i++) {
    //     for (int j = 0; j < n; j++) {
    //         std::cout << ct_CPU_res[i]->GetA(j).ConvertToInt() << " ";
    //     }
    //     std::cout << ct_CPU_res[i]->GetB().ConvertToInt() << " ";
    //     std::cout << std::endl;
    // }

    // compare ct_GPU_res and ct_CPU_res
    for (int i = 0; i < out_ct; i++) {
        for (int j = 0; j < n; j++) {
            if (ct_GPU_res[i]->GetA(j).ConvertToInt() != ct_CPU_res[i]->GetA(j).ConvertToInt()) {
                std::cout << "Error: ct_GPU_res[" << i << "]->GetA(" << j << ") = " << ct_GPU_res[i]->GetA(j).ConvertToInt() << ", ct_CPU_res[" << i << "]->GetA(" << j << ") = " << ct_CPU_res[i]->GetA(j).ConvertToInt() << std::endl;
            }
        }
        if (ct_GPU_res[i]->GetB().ConvertToInt() != ct_CPU_res[i]->GetB().ConvertToInt()) {
            std::cout << "Error: ct_GPU_res[" << i << "]->GetB() = " << ct_GPU_res[i]->GetB().ConvertToInt() << ", ct_CPU_res[" << i << "]->GetB() = " << ct_CPU_res[i]->GetB().ConvertToInt() << std::endl;
        }
    }

    std::cout << "--------------------------------" << std::endl;

    cc.GPUClean();
}

int main() {
    GEMMTest();

    return 0;
}