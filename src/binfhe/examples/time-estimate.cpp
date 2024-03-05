#include "binfhecontext.h"
#include <omp.h>

using namespace lbcrypto;

void EvalFuncTest(){
    std::cout << "EvalFunc Test: " << std::endl;

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(STD128, true, 12);
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
    for (int i = 0; i < 1024; i++) {
        auto ct1 = cc.Encrypt(sk, i % p, FRESH, p);
        ct_vec.push_back(ct1);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto ct_cube_vec = cc.EvalFunc(ct_vec, lut);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout << "EvalFunc Time: " << elapsed.count() << " ms" << std::endl;
  
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

void EvalFloorTest(){
    std::cout << "EvalFloorTest Test: " << std::endl;

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(STD128, false);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cc.GPUSetup();

    int p = cc.GetMaxPlaintextSpace().ConvertToInt();

    std::vector<LWECiphertext> ct_vec;
    for (int i = 0; i < 1024; i++) {
        auto ct1 = cc.Encrypt(sk, i % p, FRESH, p);
        ct_vec.push_back(ct1);
    }

    uint32_t bits = 1;
    auto start = std::chrono::high_resolution_clock::now();
    auto ctRounded_vec = cc.EvalFloor(ct_vec, bits);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout << "EvalFloor Time: " << elapsed.count() << " ms" << std::endl;

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
    cc.GenerateBinFHEContext(STD128, false, logQ, 0, GINX, false);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cc.GPUSetup();

    uint32_t Q = 1 << logQ;

    int q      = 4096;                                               // q
    int factor = 1 << int(logQ - log2(q));                           // Q/q
    int p      = cc.GetMaxPlaintextSpace().ConvertToInt() * factor;  // Obtain the maximum plaintext space

    std::vector<LWECiphertext> ct_vec;
    for (int i = 0; i < 1024; i++) {
        auto ct1 = cc.Encrypt(sk, p / 2 + (i % p) - 3, FRESH, p, Q);
        ct_vec.push_back(ct1);
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto ctSign_vec = cc.EvalSign(ct_vec);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout << "EvalSign Time: " << elapsed.count() << " ms" << std::endl;

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
    cc.GenerateBinFHEContext(STD128, false, logQ, 0, GINX, false);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cc.GPUSetup();

    uint32_t Q = 1 << logQ;

    int q      = 4096;                                               // q
    int factor = 1 << int(logQ - log2(q));                           // Q/q
    uint64_t P = cc.GetMaxPlaintextSpace().ConvertToInt() * factor;  // Obtain the maximum plaintext space
    
    //uint64_t input = 8193;
    std::vector<LWECiphertext> ct_vec;
    for (int i = 0; i < 1024; i++) {
        auto ct1 = cc.Encrypt(sk, i % P, FRESH, P, Q);
        ct_vec.push_back(ct1);
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto ct_decomp_vec = cc.EvalDecomp(ct_vec);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout << "EvalDecomp Time: " << elapsed.count() << " ms" << std::endl;

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
    //EvalFuncTest();
    EvalBinGateTest();
    //EvalFloorTest();
    //EvalSignTest();
    //EvalDecompTest();

    return 0;
}