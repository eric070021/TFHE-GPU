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
  Example for the FHEW scheme using the default bootstrapping method (GINX)
 */

#include "binfhecontext.h"

using namespace lbcrypto;

int main() {
    // Sample Program: Step 1: Set CryptoContext

    auto cc = BinFHEContext();

    // STD128 is the security level of 128 bits of security based on LWE Estimator
    // and HE standard. Other common options are TOY, MEDIUM, STD192, and STD256.
    // MEDIUM corresponds to the level of more than 100 bits for both quantum and
    // classical computer attacks.
    cc.GenerateBinFHEContext(STD128);

    // Sample Program: Step 2: Key Generation

    // Generate the secret key
    auto sk = cc.KeyGen();

    std::cout << "Generating the bootstrapping keys..." << std::endl;

    // Generate the bootstrapping keys (refresh and switching keys)
    cc.BTKeyGen(sk);

    std::cout << "Completed the key generation." << std::endl;

    LWEPlaintext result;

    auto ct1 = cc.Encrypt(sk, 1);
    auto ct2 = cc.Encrypt(sk, 1);

    auto start = std::chrono::high_resolution_clock::now();   

    // LWECiphertext ctNAND;
    // for(int i = 0; i < 1000; ++ i)
    //   ctNAND = cc.EvalBinGate(NAND, ct1, ct2);
    // cc.Decrypt(sk, ctNAND, &result);
    // std::cout << "1 NAND 1 = " << result << std::endl;

    auto ctAND = cc.EvalBinGate(AND, ct1, ct2);
    cc.Decrypt(sk, ctAND, &result);
    std::cout << "1 AND 1 = " << result << std::endl;

    // auto ctNAND = cc.EvalBinGate(NAND, ct1, ct2);
    // cc.Decrypt(sk, ctNAND, &result);
    // std::cout << "1 NAND 1 = " << result << std::endl;

    // auto ctOR = cc.EvalBinGate(OR, ct1, ct2);
    // cc.Decrypt(sk, ctOR, &result);
    // std::cout << "1 OR 1 = " << result << std::endl;

    // auto ctNOR = cc.EvalBinGate(NOR, ct1, ct2);
    // cc.Decrypt(sk, ctNOR, &result);
    // std::cout << "1 NOR 1 = " << result << std::endl;

    // auto ctXOR = cc.EvalBinGate(XOR, ct1, ct2);
    // cc.Decrypt(sk, ctXOR, &result);
    // std::cout << "1 XOR 1 = " << result << std::endl;

    // auto ctXNOR = cc.EvalBinGate(XNOR, ct1, ct2);
    // cc.Decrypt(sk, ctXNOR, &result);
    // std::cout << "1 XNOR 1 = " << result << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout << "Time: " << elapsed.count() << " ms" << std::endl;

    return 0;
}
