#include "binfhecontext.h"
#include <omp.h>

using namespace lbcrypto;

void EvalBinGateTest(int batchSize = 16384){
    std::cout << "EvalBinGate Test: " << std::endl;

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(STD128);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cc.GPUSetup();

    std::vector<LWECiphertext> ct1_vec, ct2_vec;
    for (int i = 0; i < batchSize; i++) {
        auto ct1 = cc.Encrypt(sk, 1);
        auto ct2 = cc.Encrypt(sk, 1);
        ct1_vec.push_back(ct1);
        ct2_vec.push_back(ct2);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto ctNAND = cc.EvalBinGate(NAND, ct1_vec, ct2_vec);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout << "EvalBinGate Time: " << (double)elapsed.count() / batchSize << " ms / ctx" << std::endl;

    std::cout << "--------------------------------" << std::endl;

    cc.GPUClean();
}

void EvalFuncTest(int batchSize = 16384){
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
    for (int i = 0; i < batchSize; i++) {
        auto ct1 = cc.Encrypt(sk, i % p, FRESH, p);
        ct_vec.push_back(ct1);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto ct_cube_vec = cc.EvalFunc(ct_vec, lut);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout << "EvalFunc Time: " << (double)elapsed.count() / batchSize << " ms / ctx" << std::endl;

    std::cout << "--------------------------------" << std::endl;

    cc.GPUClean();
}

void EvalFloorTest(int batchSize = 16384){
    std::cout << "EvalFloorTest Test: " << std::endl;

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(STD128, false, 11, 0, GINX, false, 0, 1);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cc.GPUSetup();

    int p = cc.GetMaxPlaintextSpace().ConvertToInt();

    std::vector<LWECiphertext> ct_vec;
    for (int i = 0; i < batchSize; i++) {
        auto ct1 = cc.Encrypt(sk, i % p, FRESH, p);
        ct_vec.push_back(ct1);
    }

    uint32_t bits = 1;
    auto start = std::chrono::high_resolution_clock::now();
    auto ctRounded_vec = cc.EvalFloor(ct_vec, bits);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout << "EvalFloor Time: " << (double)elapsed.count() / batchSize << " ms / ctx" << std::endl;

    std::cout << "--------------------------------" << std::endl;

    cc.GPUClean();
}

void EvalSignTest(int batchSize = 16384){
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
    for (int i = 0; i < batchSize; i++) {
        auto ct1 = cc.Encrypt(sk, p / 2 + (i % p) - 3, FRESH, p, Q);
        ct_vec.push_back(ct1);
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto ctSign_vec = cc.EvalSign(ct_vec);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout << "EvalSign Time: " << (double)elapsed.count() / batchSize << " ms / ctx" << std::endl;

    std::cout << "--------------------------------" << std::endl;

    cc.GPUClean();
}

void EvalDecompTest(int batchSize = 16384){
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
    for (int i = 0; i < batchSize; i++) {
        auto ct1 = cc.Encrypt(sk, i % P, FRESH, P, Q);
        ct_vec.push_back(ct1);
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto ct_decomp_vec = cc.EvalDecomp(ct_vec);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout << "EvalDecomp Time: " << (double)elapsed.count() / batchSize << " ms / ctx" << std::endl;

    std::cout << "--------------------------------" << std::endl;

    cc.GPUClean();
}

int main() {
    EvalBinGateTest();
    EvalFuncTest();
    EvalFloorTest();
    EvalSignTest();
    EvalDecompTest();

    return 0;
}