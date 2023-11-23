//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2022, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

/*
  Example for the FHEW scheme small precision arbitrary function evaluation
 */

#include "binfhecontext.h"

using namespace lbcrypto;

int main() {
    // Sample Program: Step 1: Set CryptoContext
    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(STD128, true, 12);

    // Sample Program: Step 2: Key Generation

    // Generate the secret key
    auto sk = cc.KeyGen();

    std::cout << "Generating the bootstrapping keys..." << std::endl;

    // Generate the bootstrapping keys (refresh and switching keys)
    cc.BTKeyGen(sk);

    std::cout << "Completed the key generation." << std::endl;

    // Sample Program: Step 3: Create the to-be-evaluated funciton and obtain its corresponding LUT
    int p = cc.GetMaxPlaintextSpace().ConvertToInt();  // Obtain the maximum plaintext space
    //int p = 256;

    // Initialize Function f(x) = x^3 % p
    auto fp = [](NativeInteger m, NativeInteger p1) -> NativeInteger {
        if (m < p1)
            return (m * m * m) % p1;
        else
            return ((m - p1 / 2) * (m - p1 / 2) * (m - p1 / 2)) % p1;
    };
    // auto fp = [](NativeInteger m, NativeInteger p1) -> NativeInteger {
    //     return m;
    // };

    // Generate LUT from function f(x)
    auto lut = cc.GenerateLUTviaFunction(fp, p);
    std::cout << "Evaluate x^3%" << p << "." << std::endl;

    // Sample Program: Step 4: evalute f(x) homomorphically and decrypt
    // Note that we check for all the possible plaintexts.
    // auto start = std::chrono::high_resolution_clock::now();    
    // for (int i = 0; i < p; i++) {
    //     auto ct1 = cc.Encrypt(sk, i % p, FRESH, p);

    //     auto ct_cube = cc.EvalFunc(ct1, lut);

    //     LWEPlaintext result;

    //     cc.Decrypt(sk, ct_cube, &result, p);

    //     std::cout << "Input: " << i << ". Expected: " << fp(i, p) << ". Evaluated = " << result << std::endl;
    // }
    // auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    // std::cout << "Time: " << elapsed.count() << " ms" << std::endl;

    std::vector<LWECiphertext> ct_vec;
    for (int i = 0; i < p; i++) {
        auto ct1 = cc.Encrypt(sk, i % p, FRESH, p);
        ct_vec.push_back(ct1);
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto ct_cube_vec = cc.EvalFunc(ct_vec, lut);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout << "Time: " << elapsed.count() << " ms" << std::endl;
    for (int i = 0; i < p; i++) {
        LWEPlaintext result;
        cc.Decrypt(sk, ct_cube_vec[i], &result, p);
        std::cout << "Input: " << i << ". Expected: " << fp(i, p) << 
            ". Evaluated = " << result << std::endl;
    }

    // for(int round = 1; round <= 512; round++){
    //     std::vector<LWECiphertext> ct_vec;
    //     for (int i = 0; i < round; i++) {
    //         auto ct1 = cc.Encrypt(sk, i % p, FRESH, p);
    //         ct_vec.push_back(ct1);
    //     }
    //     // warm up
    //     auto ct_cube_vec = cc.EvalFunc(ct_vec, lut);
        
    //     // Get average of 10 runs
    //     std::vector<double> times;
    //     for(int i = 0; i < 10; i++){
    //         auto start = std::chrono::high_resolution_clock::now();
    //         ct_cube_vec = cc.EvalFunc(ct_vec, lut);
    //         auto end = std::chrono::high_resolution_clock::now();
    //         auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    //         times.push_back(elapsed.count());
    //     }
       
    //     std::cout << std::accumulate(times.begin(), times.end(), 0.0) / times.size() << ", ";
    // }
    
    // auto start = std::chrono::high_resolution_clock::now(); 
    // int failCount = 0;
    // for(int round = 0; round < 100; round++){
    //     failCount = 0;
    //     for (int i = 0; i < p; i++) {
    //         auto ct1 = cc.Encrypt(sk, i % p, FRESH, p);

    //         auto ct_cube = cc.EvalFunc(ct1, lut);

    //         LWEPlaintext result;

    //         cc.Decrypt(sk, ct_cube, &result, p);
    //         if(static_cast<uint64_t>(result) != fp(i, p).ConvertToInt()) {
    //             failCount++;
    //         }
    //     }
    //     std::cout << "Round: " << round + 1 << ", failCount = " << failCount << std::endl;
    // }
    // auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    // std::cout << "Time: " << elapsed.count() << " ms" << std::endl;

    return 0;
}
