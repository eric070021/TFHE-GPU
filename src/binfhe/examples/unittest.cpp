#include "binfhecontext.h"

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
    cc.GenerateBinFHEContext(STD128, false);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cc.GPUSetup();

    int p = cc.GetMaxPlaintextSpace().ConvertToInt();

    // Encryption
    uint32_t input = 6;
    std::vector<LWECiphertext> ct_vec;
    for (int i = 0; i < 1; i++) {
        auto ct1 = cc.Encrypt(sk, input % p, FRESH, p);
        ct_vec.push_back(ct1);
    }

    uint32_t bits;
    LWEPlaintext result;
    std::vector<LWECiphertext> ctRounded_vec;

    bits = 1;
    ctRounded_vec = cc.EvalFloor(ct_vec, bits);
    cc.Decrypt(sk, ctRounded_vec[0], &result, p / (1 << bits));
    std::cout << "Input: " << input << " >> " << bits << ". Expected: " << (input >> bits) << ". Evaluated = " << result << std::endl;

    bits = 2;
    ctRounded_vec = cc.EvalFloor(ct_vec, bits);
    cc.Decrypt(sk, ctRounded_vec[0], &result, p / (1 << bits));
    std::cout << "Input: " << input << " >> " << bits << ". Expected: " << (input >> bits) << ". Evaluated = " << result << std::endl;

    bits = 3;
    ctRounded_vec = cc.EvalFloor(ct_vec, bits);
    cc.Decrypt(sk, ctRounded_vec[0], &result, p / (1 << bits));
    std::cout << "Input: " << input << " >> " << bits << ". Expected: " << (input >> bits) << ". Evaluated = " << result << std::endl;

    std::cout << "--------------------------------" << std::endl;

    cc.GPUClean();
}

int main() {
    EvalFuncTest();
    EvalBinGateTest();
    EvalFloorTest();

    return 0;
}