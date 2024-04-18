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
#include <algorithm>

using namespace lbcrypto;

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