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

void EvalFuncDiffLUTTest(){
    std::cout << "EvalFunc using different LUTs Test: " << std::endl;

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(STD128, true, 12, 0, GINX, false, 0, 1);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cc.GPUSetup();

    int p = cc.GetMaxPlaintextSpace().ConvertToInt();  // Obtain the maximum plaintext space
    // Initialize Function f(x) = x^3 % p
    auto fp1 = [](NativeInteger m, NativeInteger p1) -> NativeInteger {
        if (m < p1)
            return (m * m * m) % p1;
        else
            return ((m - p1 / 2) * (m - p1 / 2) * (m - p1 / 2)) % p1;
    };

    // Initialize Function f(x) = x % p
    auto fp2 = [](NativeInteger m, NativeInteger p1) -> NativeInteger {
        return m;
    };

    // Generate LUT from function f(x)
    auto lut1 = cc.GenerateLUTviaFunction(fp1, p);
    auto lut2 = cc.GenerateLUTviaFunction(fp2, p);
    std::vector<std::vector<lbcrypto::NativeInteger>> lut_vec;
    for (int i = 0; i < p/2; i++) {
        lut_vec.push_back(lut1);
    }
    for (int i = p/2; i < p; i++) {
        lut_vec.push_back(lut2);
    }

    std::vector<LWECiphertext> ct_vec;
    for (int i = 0; i < p; i++) {
        auto ct1 = cc.Encrypt(sk, i % p, FRESH, p);
        ct_vec.push_back(ct1);
    }
    
    auto ct_cube_vec = cc.EvalFunc(ct_vec, lut_vec);
  
    for (int i = 0; i < p/2; i++) {
        LWEPlaintext result;
        cc.Decrypt(sk, ct_cube_vec[i], &result, p);
        std::cout << "Input: " << i << ". Expected: " << fp1(i % p, p) << ". Evaluated = " << result << std::endl;
    }

    for (int i = p/2; i < p; i++) {
        LWEPlaintext result;
        cc.Decrypt(sk, ct_cube_vec[i], &result, p);
        std::cout << "Input: " << i << ". Expected: " << fp2(i % p, p) << ". Evaluated = " << result << std::endl;
    }

    std::cout << "--------------------------------" << std::endl;

    cc.GPUClean();
}

void EvalFuncTest(){
    std::cout << "EvalFunc Test: " << std::endl;

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(STD128, true, 12, 0, GINX, false, 0, 1);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cc.GPUSetup();

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

    std::vector<LWECiphertext> ct_vec;
    for (int i = 0; i < p; i++) {
        auto ct1 = cc.Encrypt(sk, i % p, FRESH, p);
        ct_vec.push_back(ct1);
    }
    
    auto ct_cube_vec = cc.EvalFunc(ct_vec, lut);
  
    for (int i = 0; i < p; i++) {
        LWEPlaintext result;
        cc.Decrypt(sk, ct_cube_vec[i], &result, p);
        std::cout << "Input: " << i << ". Expected: " << fp(i % p, p) << ". Evaluated = " << result << std::endl;
    }

    std::cout << "--------------------------------" << std::endl;

    cc.GPUClean();
}

void EvalBinGateTest(){
    std::cout << "EvalBinGate Test: " << std::endl;

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(STD128);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cc.GPUSetup();

    LWEPlaintext result;
    std::vector<LWECiphertext> ct1_vec, ct2_vec;
    for (int i = 0; i < 1; i++) {
        auto ct1 = cc.Encrypt(sk, 1);
        auto ct2 = cc.Encrypt(sk, 1);
        ct1_vec.push_back(ct1);
        ct2_vec.push_back(ct2);
    }

    auto ctAND = cc.EvalBinGate(AND, ct1_vec, ct2_vec);
    cc.Decrypt(sk, ctAND[0], &result);
    std::cout << "1 AND 1 = " << result << std::endl;

    auto ctNAND = cc.EvalBinGate(NAND, ct1_vec, ct2_vec);
    cc.Decrypt(sk, ctNAND[0], &result);
    std::cout << "1 NAND 1 = " << result << std::endl;

    auto ctOR = cc.EvalBinGate(OR, ct1_vec, ct2_vec);
    cc.Decrypt(sk, ctOR[0], &result);
    std::cout << "1 OR 1 = " << result << std::endl;

    auto ctNOR = cc.EvalBinGate(NOR, ct1_vec, ct2_vec);
    cc.Decrypt(sk, ctNOR[0], &result);
    std::cout << "1 NOR 1 = " << result << std::endl;

    auto ctXOR = cc.EvalBinGate(XOR, ct1_vec, ct2_vec);
    cc.Decrypt(sk, ctXOR[0], &result);
    std::cout << "1 XOR 1 = " << result << std::endl;

    auto ctXNOR = cc.EvalBinGate(XNOR, ct1_vec, ct2_vec);
    cc.Decrypt(sk, ctXNOR[0], &result);
    std::cout << "1 XNOR 1 = " << result << std::endl;

    std::cout << "--------------------------------" << std::endl;

    cc.GPUClean();
}

void EvalFloorTest(){
    std::cout << "EvalFloorTest Test: " << std::endl;

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(STD128, false, 11, 0, GINX, false, 0, 1);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cc.GPUSetup();

    int p = cc.GetMaxPlaintextSpace().ConvertToInt();

    std::vector<LWECiphertext> ct_vec;
    for (int i = 0; i < p; i++) {
        auto ct1 = cc.Encrypt(sk, i % p, FRESH, p);
        ct_vec.push_back(ct1);
    }

    uint32_t bits = 1;
    auto ctRounded_vec = cc.EvalFloor(ct_vec, bits);

    for (int i = 0; i < p; i++) {
        LWEPlaintext result;
        cc.Decrypt(sk, ctRounded_vec[i], &result, p / (1 << bits));
        std::cout << "Input: " << i % p << " >> " << bits << ". Expected: " << ((i % p) >> bits) << ". Evaluated = " << result << std::endl;
    }

    std::cout << "--------------------------------" << std::endl;

    cc.GPUClean();
}

void EvalSignTest(){
    std::cout << "EvalSignTest Test: " << std::endl;

    auto cc = BinFHEContext();
    uint32_t logQ = 17;
    cc.GenerateBinFHEContext(STD128, false, logQ, 0, GINX, false, 0, 1);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cc.GPUSetup();

    uint32_t Q = 1 << logQ;

    int q      = 4096;                                               // q
    int factor = 1 << int(logQ - log2(q));                           // Q/q
    int p      = cc.GetMaxPlaintextSpace().ConvertToInt() * factor;  // Obtain the maximum plaintext space

    std::vector<LWECiphertext> ct_vec;
    for (int i = 0; i < 8; i++) {
        auto ct1 = cc.Encrypt(sk, p / 2 + i - 3, FRESH, p, Q);
        ct_vec.push_back(ct1);
    }

    auto ctSign_vec = cc.EvalSign(ct_vec);

    for (int i = 0; i < 8; i++) {
        LWEPlaintext result;
        cc.Decrypt(sk, ctSign_vec[i], &result, 2);
        std::cout << "Input: " << i << ". Expected sign: " << (i >= 3) << ". Evaluated Sign: " << result << std::endl;
    }

    std::cout << "--------------------------------" << std::endl;

    cc.GPUClean();
}

void EvalDecompTest(){
    std::cout << "EvalDecompTest Test: " << std::endl;

    auto cc = BinFHEContext();
    uint32_t logQ = 23;
    cc.GenerateBinFHEContext(STD128, false, logQ, 0, GINX, false, 0, 1);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cc.GPUSetup();

    uint32_t Q = 1 << logQ;

    int q      = 4096;                                               // q
    int factor = 1 << int(logQ - log2(q));                           // Q/q
    uint64_t P = cc.GetMaxPlaintextSpace().ConvertToInt() * factor;  // Obtain the maximum plaintext space
    
    //uint64_t input = 8193;
    std::vector<LWECiphertext> ct_vec;
    for (int i = 0; i < 8; i++) {
        auto ct1 = cc.Encrypt(sk, i % P, FRESH, P, Q);
        ct_vec.push_back(ct1);
    }

    auto ct_decomp_vec = cc.EvalDecomp(ct_vec);

    for (int i = 0; i < 8; i++) {
        uint64_t p = cc.GetMaxPlaintextSpace().ConvertToInt();
        std::cout << "Encrypted value: " << i % P;
        std::cout << " Decomposed value: ";
        for (size_t j = 0; j < ct_decomp_vec[i].size(); j++) {
            auto ct_decomp = ct_decomp_vec[i][j];
            LWEPlaintext result;
            if (j == ct_decomp_vec[i].size() - 1) {
                // after every evalfloor, the least significant digit is dropped so the last modulus is computed as log p = (log P) mod (log GetMaxPlaintextSpace)
                auto logp = GetMSB(P - 1) % GetMSB(p - 1);
                p         = 1 << logp;
            }
            cc.Decrypt(sk, ct_decomp, &result, p);
            std::cout << "(" << result << " * " << cc.GetMaxPlaintextSpace() << "^" << j << ")";
            if (j != ct_decomp_vec[i].size() - 1) {
                std::cout << " + ";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "--------------------------------" << std::endl;

    cc.GPUClean();
}

int main() {
    EvalFuncDiffLUTTest();
    EvalFuncTest();
    EvalBinGateTest();
    EvalFloorTest();
    EvalSignTest();
    EvalDecompTest();

    return 0;
}