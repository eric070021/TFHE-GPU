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
#include <algorithm>

using namespace lbcrypto;

void TFHE_rs_Compare(){
    std::cout << "EvalBinGate Test: " << std::endl;

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(STD128);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cc.GPUSetup();

    std::vector<LWECiphertext> ct1_vec, ct2_vec;
    for (int i = 0; i < 256; i++) {
        auto ct1 = cc.Encrypt(sk, 1);
        auto ct2 = cc.Encrypt(sk, 1);
        ct1_vec.push_back(ct1);
        ct2_vec.push_back(ct2);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++)
        auto ctNAND = cc.EvalBinGate(AND, ct1_vec, ct2_vec);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
    std::cout << "EvalBinGate Time: " << elapsed.count() << " us" << std::endl;

    // LWEPlaintext result;
    // cc.Decrypt(sk, ctNAND[0], &result);
    // std::cout << "1 NAND 1 = " << result << std::endl;

    std::cout << "--------------------------------" << std::endl;

    cc.GPUClean();
}

int main() {
    // Sample Program: Step 1: Set CryptoContext
    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(STD128, true, 12, 0, GINX, false, 1 << 18);

    // Sample Program: Step 2: Key Generation

    // Generate the secret key
    auto sk = cc.KeyGen();

    std::cout << "Generating the bootstrapping keys..." << std::endl;

    // Generate the bootstrapping keys (refresh and switching keys)
    cc.BTKeyGen(sk);

    std::cout << "Completed the key generation." << std::endl;

    std::cout << "Setting up GPU..." << std::endl;

    // Setup GPU
    cc.GPUSetup();

    std::cout << "Completed the GPU Setup." << std::endl;

    // Sample Program: Step 3: Create the to-be-evaluated funciton and obtain its corresponding LUT
    int p = cc.GetMaxPlaintextSpace().ConvertToInt();  // Obtain the maximum plaintext space

    // Initialize Function f(x) = x^3 % p
    auto fp = [](NativeInteger m, NativeInteger p1) -> NativeInteger {
        if (m < p1)
            return (m * m * m) % p1;
        else
            return ((m - p1 / 2) * (m - p1 / 2) * (m - p1 / 2)) % p1;
    };

    // Generate LUT from function f(x)
    auto lut = cc.GenerateLUTviaFunction(fp, p);

    for(int round = 1; round <= 512; round++){
        std::vector<LWECiphertext> ct_vec;
        for (int i = 0; i < round; i++) {
            auto ct1 = cc.Encrypt(sk, i % p, FRESH, p);
            ct_vec.push_back(ct1);
        }
        // warm up
        auto ct_cube_vec = cc.EvalFunc(ct_vec, lut);
        
        // Get average of 5 runs
        std::vector<double> times;
        for(int i = 0; i < 5; i++){
            auto start = std::chrono::high_resolution_clock::now();
            ct_cube_vec = cc.EvalFunc(ct_vec, lut);
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
            times.push_back(elapsed.count());
        }
       
        std::cout << std::accumulate(times.begin(), times.end(), 0.0) / times.size() << ", " << std::flush;
    }

    cc.GPUClean();
    
    return 0;
}