#include "binfhecontext.h"

using namespace lbcrypto;

int main() {
    // EvalFunc Test
    std::cout << "EvalFunc Test: " << std::endl; 
    auto ccfunc = BinFHEContext();
    ccfunc.GenerateBinFHEContext(STD128, true, 12);
    auto skfunc = ccfunc.KeyGen();
    ccfunc.BTKeyGen(skfunc);
    ccfunc.GPUSetup();

    int p = ccfunc.GetMaxPlaintextSpace().ConvertToInt();  // Obtain the maximum plaintext space
    // Initialize Function f(x) = x^3 % p
    auto fp = [](NativeInteger m, NativeInteger p1) -> NativeInteger {
        if (m < p1)
            return (m * m * m) % p1;
        else
            return ((m - p1 / 2) * (m - p1 / 2) * (m - p1 / 2)) % p1;
    };
    // Generate LUT from function f(x)
    auto lut = ccfunc.GenerateLUTviaFunction(fp, p);

    std::vector<LWECiphertext> ct_vec;
    for (int i = 0; i < p; i++) {
        auto ct1 = ccfunc.Encrypt(skfunc, i % p, FRESH, p);
        ct_vec.push_back(ct1);
    }
    
    auto ct_cube_vec = ccfunc.EvalFunc(ct_vec, lut);
  
    for (int i = 0; i < p; i++) {
        LWEPlaintext result;
        ccfunc.Decrypt(skfunc, ct_cube_vec[i], &result, p);
        std::cout << "Input: " << i << ". Expected: " << fp(i % p, p) << ". Evaluated = " << result << std::endl;
    }

    ccfunc.GPUClean();
    
    // EvalBinGate Test
    std::cout << "\nEvalBinGate Test: " << std::endl;
    auto ccbin = BinFHEContext();
    ccbin.GenerateBinFHEContext(STD128);
    auto skbin = ccbin.KeyGen();
    ccbin.BTKeyGen(skbin);
    ccbin.GPUSetup();

    LWEPlaintext result;
    std::vector<LWECiphertext> ct1_vec, ct2_vec;
    for (int i = 0; i < 1; i++) {
        auto ct1 = ccbin.Encrypt(skbin, 1);
        auto ct2 = ccbin.Encrypt(skbin, 1);
        ct1_vec.push_back(ct1);
        ct2_vec.push_back(ct2);
    }

    auto ctAND = ccbin.EvalBinGate(AND, ct1_vec, ct2_vec);
    ccbin.Decrypt(skbin, ctAND[0], &result);
    std::cout << "1 AND 1 = " << result << std::endl;

    auto ctNAND = ccbin.EvalBinGate(NAND, ct1_vec, ct2_vec);
    ccbin.Decrypt(skbin, ctNAND[0], &result);
    std::cout << "1 NAND 1 = " << result << std::endl;

    auto ctOR = ccbin.EvalBinGate(OR, ct1_vec, ct2_vec);
    ccbin.Decrypt(skbin, ctOR[0], &result);
    std::cout << "1 OR 1 = " << result << std::endl;

    auto ctNOR = ccbin.EvalBinGate(NOR, ct1_vec, ct2_vec);
    ccbin.Decrypt(skbin, ctNOR[0], &result);
    std::cout << "1 NOR 1 = " << result << std::endl;

    auto ctXOR = ccbin.EvalBinGate(XOR, ct1_vec, ct2_vec);
    ccbin.Decrypt(skbin, ctXOR[0], &result);
    std::cout << "1 XOR 1 = " << result << std::endl;

    auto ctXNOR = ccbin.EvalBinGate(XNOR, ct1_vec, ct2_vec);
    ccbin.Decrypt(skbin, ctXNOR[0], &result);
    std::cout << "1 XNOR 1 = " << result << std::endl;

    ccbin.GPUClean();

    return 0;
}