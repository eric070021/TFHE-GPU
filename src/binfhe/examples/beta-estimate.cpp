#include "binfhecontext.h"
#include <iomanip>

using namespace lbcrypto;


long double calculateStandardDeviation(const std::vector<int64_t> data) {
    // Step 1: Calculate mean
    long double mean = 0;
    for (long double value : data) {
        mean += value;
    }
    mean /= data.size();

    // Step 2: Calculate squared differences
    std::vector<long double> squaredDifferences;
    for (long double value : data) {
        long double diff = value - mean;
        squaredDifferences.push_back(diff * diff);
    }

    // Step 3: Calculate mean of squared differences
    long double meanSquaredDiff = 0;
    for (long double value : squaredDifferences) {
        meanSquaredDiff += value;
    }
    meanSquaredDiff /= squaredDifferences.size();

    // Step 4: Take the square root
    long double standardDeviation = std::sqrt(meanSquaredDiff);

    return standardDeviation;
}

void BaseGTest(){
    std::cout << "BaseGTest for EvalFunc Test: " << std::endl;

    for(uint32_t baseG = 2 ; baseG != 1<<28; baseG = baseG << 1){
        auto cc = BinFHEContext();
        cc.GenerateBinFHEContext(STD128, true, 12, 0, GINX, false, baseG);
        auto sk = cc.KeyGen();
        cc.BTKeyGen(sk);
        cc.GPUSetup();
        auto LWEParams    = cc.GetParams()->GetLWEParams();
        int64_t q = static_cast<int64_t>(LWEParams->Getq().ConvertToInt());

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
        for (int i = 0; i < 128; i++) {
            auto ct1 = cc.Encrypt(sk, 0, FRESH, p);
            ct_vec.push_back(ct1);
        }
        
        //auto ct_cube_vec = cc.EvalFunc(ct_vec, lut);
        std::vector<lbcrypto::LWECiphertext> ct_cube_vec;
        for (int i = 0; i < 128; i++) {
            auto ct_cube = cc.EvalFunc(ct_vec[i], lut);
            ct_cube_vec.push_back(ct_cube);
        }
        
        std::vector<int64_t> beta_arr;
        for (int i = 0; i < 128; i++) {
            LWEPlaintext result;
            cc.DecryptWithoutScale(sk, ct_cube_vec[i], &result, p);
            int64_t b = result < (q/2) ? result : result - q;
            //int64_t b = std::min(result, q - result);
            beta_arr.push_back(b);
        }
        cc.GPUClean();

        // std::cout << "Mean beta: " << std::accumulate(beta_arr.begin(), beta_arr.end(), 0.0) / beta_arr.size() << std::endl;
        // std::cout << "STD beta: " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << calculateStandardDeviation(beta_arr) << std::endl;
        std::cout << "baseG: 1 << " << std::log2(baseG) << std::endl;
        std::cout << "erfc beta: " << std::log2(std::erfc(static_cast<long double>(q/p)/(2*2*calculateStandardDeviation(beta_arr)))) << std::endl;

        std::cout << "--------------------------------" << std::endl;
    }
}

void EvalFuncTest(){
    std::cout << "EvalFunc Test: " << std::endl;

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(STD128, true, 12);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cc.GPUSetup();
    auto LWEParams    = cc.GetParams()->GetLWEParams();
    int64_t q = static_cast<int64_t>(LWEParams->Getq().ConvertToInt());

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
        auto ct1 = cc.Encrypt(sk, 0, FRESH, p);
        ct_vec.push_back(ct1);
    }
    
    auto ct_cube_vec = cc.EvalFunc(ct_vec, lut);
    // std::vector<lbcrypto::LWECiphertext> ct_cube_vec;
    // for (int i = 0; i < 1024; i++) {
    //     auto ct_cube = cc.EvalFunc(ct_vec[i], lut);
    //     ct_cube_vec.push_back(ct_cube);
    // }
    
    std::vector<int64_t> beta_arr;
    for (int i = 0; i < 1024; i++) {
        LWEPlaintext result;
        cc.DecryptWithoutScale(sk, ct_cube_vec[i], &result, p);
        int64_t b = result < (q/2) ? result : result - q;
        //int64_t b = std::min(result, q - result);
        beta_arr.push_back(b);
    }
    cc.GPUClean();

    std::cout << "Mean beta: " << std::accumulate(beta_arr.begin(), beta_arr.end(), 0.0) / beta_arr.size() << std::endl;
    std::cout << "STD beta: " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << calculateStandardDeviation(beta_arr) << std::endl;
    std::cout << "erfc beta: " << std::log2(std::erfc(static_cast<long double>(q/p)/(2*2*calculateStandardDeviation(beta_arr)))) << std::endl;

    std::cout << "--------------------------------" << std::endl;
}

void EvalBinGateTest(BINFHE_PARAMSET paramset){
    std::cout << "EvalBinGate Test: " << std::endl;

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(paramset);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cc.GPUSetup();
    auto LWEParams    = cc.GetParams()->GetLWEParams();
    int q = static_cast<int64_t>(LWEParams->Getq().ConvertToInt());

    LWEPlaintext result;
    std::vector<LWECiphertext> ct1_vec, ct2_vec;
    for (int i = 0; i < 128; i++) {
        auto ct1 = cc.Encrypt(sk, 1);
        auto ct2 = cc.Encrypt(sk, 1);
        ct1_vec.push_back(ct1);
        ct2_vec.push_back(ct2);
    }

    auto ctNAND = cc.EvalBinGate(NAND, ct1_vec, ct2_vec);

    std::vector<int64_t> beta_arr;
    for (int i = 0; i < 128; i++) {
        cc.DecryptWithoutScale(sk, ctNAND[i], &result);
        int64_t b = result < (q/2) ? result : result - q;
        //int64_t b = std::min(result, q - result);
        beta_arr.push_back(b);
    }
    cc.GPUClean();

    std::cout << "Mean beta: " << std::accumulate(beta_arr.begin(), beta_arr.end(), 0.0) / beta_arr.size() << std::endl;
    std::cout << "STD beta: " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << calculateStandardDeviation(beta_arr) << std::endl;
    std::cout << "erfc beta: " << std::log2(std::erfc(static_cast<long double>(q/8)/(2*calculateStandardDeviation(beta_arr)))) << std::endl;

    std::cout << "--------------------------------" << std::endl;

    
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

void EvalSignTest(uint32_t logQ){
    std::cout << "EvalSignTest Test: " << std::endl;
                                              
    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(STD128, false, logQ);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    cc.GPUSetup();

    uint32_t Q = 1 << logQ;

    int q      = 4096;                                               // q
    int factor = 1 << int(logQ - log2(q));                           // Q/q
    int p      = cc.GetMaxPlaintextSpace().ConvertToInt() * factor;  // Obtain the maximum plaintext space

    std::cout << "logQ: " << logQ << ", p:" << cc.GetMaxPlaintextSpace().ConvertToInt()<< std::endl;

    std::vector<LWECiphertext> ct_vec;
    for (int i = 0; i < 1024; i++) {
        auto ct1 = cc.Encrypt(sk, p / 2 - 3, FRESH, p, Q);
        ct_vec.push_back(ct1);
    }

    auto ctSign_vec = cc.EvalSign(ct_vec);

    std::vector<int64_t> beta_arr;
    for (int i = 0; i < 1024; i++) {
        LWEPlaintext result;
        cc.DecryptWithoutScale(sk, ctSign_vec[i], &result, 2);
        int64_t b = result < (q/2) ? result : result - q;
        //int64_t b = std::min(result, q - result);
        beta_arr.push_back(b);
    }
    cc.GPUClean();

    std::cout << "Mean beta: " << std::accumulate(beta_arr.begin(), beta_arr.end(), 0.0) / beta_arr.size() << std::endl;
    std::cout << "STD beta: " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << calculateStandardDeviation(beta_arr) << std::endl;
    std::cout << "erfc beta: " << std::log2(std::erfc(static_cast<long double>(q/16)/(2*2*calculateStandardDeviation(beta_arr)))) << std::endl;

    std::cout << "--------------------------------" << std::endl;
}

void EvalDecompTest(){
    std::cout << "EvalDecompTest Test: " << std::endl;

    auto cc = BinFHEContext();
    uint32_t logQ = 23;
    cc.GenerateBinFHEContext(STD128, false, logQ);
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
    // BaseGTest();
    // EvalFuncTest();
    // EvalSignTest(12);
    // EvalSignTest(16);
    // EvalSignTest(20);
    // EvalSignTest(24);
    // EvalSignTest(25);
    // EvalSignTest(26);
    // EvalSignTest(28);
    // EvalSignTest(29);
    // EvalBinGateTest(MEDIUM);
    EvalBinGateTest(STD128);
    // EvalBinGateTest(STD192);
    // EvalBinGateTest(STD256);
    // EvalBinGateTest(STD128Q);
    // EvalBinGateTest(STD192Q);
    // EvalBinGateTest(STD256Q);
    // EvalFloorTest();
    // EvalDecompTest();

    return 0;
}