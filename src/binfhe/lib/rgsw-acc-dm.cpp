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

#include "rgsw-acc-dm.h"

#include <string>
#include <typeinfo>

namespace lbcrypto {

// bootstrapping key for FFT-based accumulator
std::shared_ptr<std::vector<std::vector<std::vector<std::shared_ptr<std::vector<std::vector<std::vector<std::complex<double>>>>>>>>> DM_bootstrappingKey_CUDA;

// Key generation as described in Section 4 of https://eprint.iacr.org/2014/816
RingGSWACCKey RingGSWAccumulatorDM::KeyGenAcc(const std::shared_ptr<RingGSWCryptoParams> params,
                                              const NativePoly& skNTT, ConstLWEPrivateKey LWEsk) const {
    auto sv     = LWEsk->GetElement();
    int32_t mod = sv.GetModulus().ConvertToInt();

    int32_t modHalf = mod >> 1;

    uint32_t baseR                            = params->GetBaseR();
    const std::vector<NativeInteger>& digitsR = params->GetDigitsR();
    uint32_t n                                = sv.GetLength();
    RingGSWACCKey ek                          = std::make_shared<RingGSWACCKeyImpl>(n, baseR, digitsR.size());
    DM_bootstrappingKey_CUDA                     = std::make_shared<std::vector<std::vector<std::vector<std::shared_ptr<std::vector<std::vector<std::vector<std::complex<double>>>>>>>>>();
    (*DM_bootstrappingKey_CUDA).resize(n);
    for (size_t i = 0; i < n; ++i) {
        (*DM_bootstrappingKey_CUDA)[i].resize(baseR);
        for (size_t j = 0; j < baseR; ++j) {
            (*DM_bootstrappingKey_CUDA)[i][j].resize(digitsR.size());
        }
    }

#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 1; j < baseR; ++j) {
            for (size_t k = 0; k < digitsR.size(); ++k) {
                int32_t s = (int32_t)sv[i].ConvertToInt();
                if (s > modHalf) {
                    s -= mod;
                }
                (*ek)[i][j][k] = KeyGenDM(params, skNTT, s * j * (int32_t)digitsR[k].ConvertToInt());
                (*DM_bootstrappingKey_CUDA)[i][j][k] = KeyGenDM_CUDA(params, skNTT, s * j * (int32_t)digitsR[k].ConvertToInt());
            }
        }
    }

    return ek;
}

void RingGSWAccumulatorDM::EvalAcc(const std::shared_ptr<RingGSWCryptoParams> params, const RingGSWACCKey ek,
                                   RLWECiphertext& acc, const NativeVector& a, std::string mode) const {
    uint32_t baseR = params->GetBaseR();
    auto digitsR   = params->GetDigitsR();
    auto q         = params->Getq();
    uint32_t n     = a.GetLength();
    // auto Q         = params->GetQ();
    // auto N         = params->GetN();
    // auto polyParams  = params->GetPolyParams();
    // // std::cout << "OrderIsPowerOfTwo: " << polyParams->OrderIsPowerOfTwo() << std::endl;

    // // cast acc to double
    // std::vector<std::vector<std::complex<double>>> acc_d(2, std::vector<std::complex<double>>(N, std::complex<double>(0.0, 0.0)));
    // for (size_t i = 0; i < 2; ++i) {
    //     for (size_t j = 0; j < N; ++j) {
    //         acc_d[i][j].real(acc->GetElements()[i][j].ConvertToDouble());
    //     }
    // }
    // std::cout << "RLWECiphertext Size: " << acc_d[0].size()*sizeof(acc_d[0][0])*2 << std::endl;

    // bool first = true;

    for (size_t i = 0; i < n; ++i) {
        NativeInteger aI = q.ModSub(a[i], q);
        for (size_t k = 0; k < digitsR.size(); ++k, aI /= NativeInteger(baseR)) {
            uint32_t a0 = (aI.Mod(baseR)).ConvertToInt();
            if (a0)
                AddToAccDM(params, (*ek)[i][a0][k], acc);
                // AddToAccDM_CUDA(params, (*DM_bootstrappingKey_CUDA)[i][a0][k], acc_d, first);
        }
    }

    // // Transpose
    // std::vector<std::complex<double>> temp(acc_d[0]);
    // usint m = polyParams->GetCyclotomicOrder();
    // usint k = m - 1;
    // usint logm = std::round(log2(m));
    // usint logn = std::round(log2(n));
    // for (usint j = 1; j < m; j += 2) {
    //     usint idx                         = (j * k) - (((j * k) >> logm) << logm);
    //     usint jrev                        = ReverseBits(j / 2, logn);
    //     usint idxrev                      = ReverseBits(idx / 2, logn);
    //     acc_d[0][jrev]                    = temp[idxrev];
    // }

    // // calls 2 IFFTs
    // for (size_t i = 0; i < 2; ++i)
    //     RingGSWAccumulator::NegacyclicInverseTransform(acc_d[i]);

    // // Round to INT64 and MOD
    // for (size_t i = 0; i < 2; ++i){
    //     for (size_t j = 0; j < N; ++j){
    //         int64_t temp = static_cast<int64_t>(round(acc_d[i][j].real()));
    //         acc_d[i][j].real(temp % Q.ConvertToInt());
    //     }
    // }

    // // cast acc_d to NativePoly
    // NativeVector ret0(N, Q), ret1(N, Q);
    // for (size_t i = 0; i < N; ++i) {
    //     ret0[i] = static_cast<uint64_t>(acc_d[0][i].real());
    //     ret1[i] = static_cast<uint64_t>(acc_d[1][i].real());
    // }
    // std::vector<NativePoly> res(2);
    // res[0] = NativePoly(polyParams, Format::COEFFICIENT, false);
    // res[1] = NativePoly(polyParams, Format::COEFFICIENT, false);
    // res[0].SetValues(std::move(ret0), Format::COEFFICIENT);
    // res[1].SetValues(std::move(ret1), Format::COEFFICIENT);
    // acc = std::make_shared<RLWECiphertextImpl>(std::move(res));
}

// Encryption as described in Section 5 of https://eprint.iacr.org/2014/816
// skNTT corresponds to the secret key z
RingGSWEvalKey RingGSWAccumulatorDM::KeyGenDM(const std::shared_ptr<RingGSWCryptoParams> params,
                                              const NativePoly& skNTT, const LWEPlaintext& m) const {
    NativeInteger Q   = params->GetQ();
    uint64_t q        = params->Getq().ConvertToInt();
    uint32_t N        = params->GetN();
    uint32_t digitsG  = params->GetDigitsG();
    uint32_t digitsG2 = digitsG << 1;
    auto polyParams   = params->GetPolyParams();
    auto Gpow         = params->GetGPower();
    auto result       = std::make_shared<RingGSWEvalKeyImpl>(digitsG2, 2);

    DiscreteUniformGeneratorImpl<NativeVector> dug;
    dug.SetModulus(Q);

    // Reduce mod q (dealing with negative number as well)
    int64_t mm       = (((m % q) + q) % q) * (2 * N / q);
    bool isReducedMM = false;
    if (mm >= N) {
        mm -= N;
        isReducedMM = true;
    }

    // tempA is introduced to minimize the number of NTTs
    std::vector<NativePoly> tempA(digitsG2);

    for (size_t i = 0; i < digitsG2; ++i) {
        // populate result[i][0] with uniform random a
        (*result)[i][0] = NativePoly(dug, polyParams, Format::COEFFICIENT);
        tempA[i]        = (*result)[i][0];
        // populate result[i][1] with error e
        (*result)[i][1] = NativePoly(params->GetDgg(), polyParams, Format::COEFFICIENT);
    }

    for (size_t i = 0; i < digitsG; ++i) {
        if (!isReducedMM) {
            // Add G Multiple
            (*result)[2 * i][0][mm].ModAddEq(Gpow[i], Q);
            // [a,as+e] + X^m*G
            (*result)[2 * i + 1][1][mm].ModAddEq(Gpow[i], Q);
        }
        else {
            // Subtract G Multiple
            (*result)[2 * i][0][mm].ModSubEq(Gpow[i], Q);
            // [a,as+e] - X^m*G
            (*result)[2 * i + 1][1][mm].ModSubEq(Gpow[i], Q);
        }
    }

    // 3*digitsG2 NTTs are called
    result->SetFormat(Format::EVALUATION);
    for (size_t i = 0; i < digitsG2; ++i) {
        tempA[i].SetFormat(Format::EVALUATION);
        (*result)[i][1] += tempA[i] * skNTT;
    }

    return result;
}

std::shared_ptr<std::vector<std::vector<std::vector<std::complex<double>>>>> RingGSWAccumulatorDM::KeyGenDM_CUDA(const std::shared_ptr<RingGSWCryptoParams> params,
                                              const NativePoly& skNTT, const LWEPlaintext& m) const {
    NativeInteger Q   = params->GetQ();
    uint64_t q        = params->Getq().ConvertToInt();
    uint32_t N        = params->GetN();
    uint32_t digitsG  = params->GetDigitsG();
    uint32_t digitsG2 = digitsG << 1;
    auto polyParams   = params->GetPolyParams();
    auto Gpow         = params->GetGPower();
    auto result       = std::make_unique<RingGSWEvalKeyImpl>(digitsG2, 2);

    DiscreteUniformGeneratorImpl<NativeVector> dug;
    dug.SetModulus(Q);

    // Reduce mod q (dealing with negative number as well)
    int64_t mm       = (((m % q) + q) % q) * (2 * N / q);
    bool isReducedMM = false;
    if (mm >= N) {
        mm -= N;
        isReducedMM = true;
    }

    // tempA is introduced to minimize the number of NTTs
    std::vector<NativePoly> tempA(digitsG2);

    for (size_t i = 0; i < digitsG2; ++i) {
        // populate result[i][0] with uniform random a
        (*result)[i][0] = NativePoly(dug, polyParams, Format::COEFFICIENT);
        tempA[i]        = (*result)[i][0];
        // populate result[i][1] with error e
        (*result)[i][1] = NativePoly(params->GetDgg(), polyParams, Format::COEFFICIENT);
    }

    for (size_t i = 0; i < digitsG; ++i) {
        if (!isReducedMM) {
            // Add G Multiple
            (*result)[2 * i][0][mm].ModAddEq(Gpow[i], Q);
            // [a,as+e] + X^m*G
            (*result)[2 * i + 1][1][mm].ModAddEq(Gpow[i], Q);
        }
        else {
            // Subtract G Multiple
            (*result)[2 * i][0][mm].ModSubEq(Gpow[i], Q);
            // [a,as+e] - X^m*G
            (*result)[2 * i + 1][1][mm].ModSubEq(Gpow[i], Q);
        }
    }

    // cast result to double
    auto result_d = std::make_shared<std::vector<std::vector<std::vector<std::complex<double>>>>>
        (digitsG2, std::vector<std::vector<std::complex<double>>>(2, std::vector<std::complex<double>>(N, std::complex<double>(0.0, 0.0))));
    // put result in result_d
    for (size_t i = 0; i < digitsG2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < N; ++k) {
                (*result_d)[i][j][k].real((*result)[i][j][k].ConvertToDouble());
            }
        }
    }

    // // cast tempA to double
    // std::vector<std::vector<std::complex<double>>> tempA_d(digitsG2, std::vector<std::complex<double>>(N, std::complex<double>(0.0, 0.0)));
    // // put tempA in tempA_d
    // for (size_t i = 0; i < digitsG2; ++i) {
    //     for (size_t j = 0; j < N; ++j) {
    //         tempA_d[i][j].real(tempA[i][j].ConvertToDouble());
    //     }
    // }

    // // 3*digitsG2 NTTs are called
    // result->SetFormat(Format::EVALUATION);
    // for (size_t i = 0; i < digitsG2; ++i) {
    //     tempA[i].SetFormat(Format::EVALUATION);
    //     (*result)[i][1] += tempA[i] * skNTT;
    // }

    // // Apply FFT on result_d
    // for (size_t i = 0; i < digitsG2; ++i) {
    //     for (size_t j = 0; j < 2; ++j) {
    //         DiscreteFourierTransform::NegacyclicForwardTransform((*result_d)[i][j]);
    //     }
    // }

    // // Apply FFT on tempA_d & element-wise multiplication with skNTT
    // for (size_t i = 0; i < digitsG2; ++i) {
    //     DiscreteFourierTransform::NegacyclicForwardTransform(tempA_d[i]);
    //     for (size_t j = 0; j < N; ++j) {
    //         (*result_d)[i][1][j] += tempA_d[i][j] * skNTT.GetValues()[j].ConvertToDouble();
    //     }
    // }

    return result_d;
}

// AP Accumulation as described in https://eprint.iacr.org/2020/086
void RingGSWAccumulatorDM::AddToAccDM(const std::shared_ptr<RingGSWCryptoParams> params, const RingGSWEvalKey ek,
                                      RLWECiphertext& acc) const {
    uint32_t digitsG2 = params->GetDigitsG() << 1;
    auto polyParams   = params->GetPolyParams();
    // auto q           = params->Getq();
    // auto Q            = params->GetQ();

    std::vector<NativePoly> ct = acc->GetElements();
    std::vector<NativePoly> dct(digitsG2);

    // initialize dct to zeros
    for (size_t i = 0; i < digitsG2; i++)
        dct[i] = NativePoly(polyParams, Format::COEFFICIENT, true);

    //clock_t t1, t2;
    // calls 2 NTTs
    //t1 = clock();
    for (size_t i = 0; i < 2; ++i)
        ct[i].SetFormat(Format::COEFFICIENT);
    // for (size_t i = 0; i < 2; ++i){
    //     // print element in ct
    //     for (size_t j = 0; j < ct[i].GetLength(); ++j){
    //         std::cout << ct[i][j].ConvertToInt() << " ";
    //     }
    // }
    // std::cout << std::endl;
    //t2 = clock();
    //std::cout << "NTT time: " << (t2-t1)/(double)(CLOCKS_PER_SEC) << std::endl; 

    //t1 = clock();
    SignedDigitDecompose(params, ct, dct);
    //t2 = clock();
    //std::cout << "SignedDigitDecompose time: " << (t2-t1)/(double)(CLOCKS_PER_SEC) << std::endl; 

    //t1 = clock();
    // calls digitsG2 NTTs
    for (size_t j = 0; j < digitsG2; ++j)
        dct[j].SetFormat(Format::EVALUATION);
    //t2 = clock();
    //std::cout << "iNTT time: " << (t2-t1)/(double)(CLOCKS_PER_SEC) << std::endl; 

    // acc = dct * ek (matrix product);
    // uses in-place * operators for the last call to dct[i] to gain performance
    // improvement
    const std::vector<std::vector<NativePoly>>& ev = ek->GetElements();
    // for elements[0]:
    acc->GetElements()[0].SetValuesToZero();
    for (size_t l = 1; l < digitsG2; ++l)
        acc->GetElements()[0] += (dct[l] * ev[l][0]);
    // for elements[1]:
    acc->GetElements()[1].SetValuesToZero();
    for (size_t l = 1; l < digitsG2; ++l)
        acc->GetElements()[1] += (dct[l] *= ev[l][1]);
}

void RingGSWAccumulatorDM::AddToAccDM_CUDA(const std::shared_ptr<RingGSWCryptoParams> params, const std::shared_ptr<std::vector<std::vector<std::vector<std::complex<double>>>>> ek,
                    std::vector<std::vector<std::complex<double>>>& acc, bool& first) const {
    // uint32_t digitsG2 = params->GetDigitsG() << 1;
    // auto polyParams   = params->GetPolyParams();
    // // auto q           = params->Getq();
    // auto Q            = params->GetQ();
    // auto N            = params->GetN();

    // std::vector<std::vector<std::complex<double>>> ct(acc);
    // std::vector<std::vector<std::complex<double>>> dct(digitsG2, std::vector<std::complex<double>>(N, std::complex<double>(0.0, 0.0)));
    
    // if(!first){
    //     // calls 2 IFFTs
    //     for (size_t i = 0; i < 2; ++i)
    //         DiscreteFourierTransform::NegacyclicInverseTransform(acc[i]);

    //     // Round to INT64 and MOD
    //     for (size_t i = 0; i < 2; ++i){
    //         for (size_t j = 0; j < N; ++j){
    //             int64_t temp = static_cast<int64_t>(round(acc[i][j].real()));
    //             acc[i][j].real(temp % Q.ConvertToInt());
    //         }
    //     }
    // }
    // first = false;

    // SignedDigitDecompose_FFT(params, ct, dct);
    
    // // calls digitsG2 FFTs
    // for (size_t j = 0; j < digitsG2; ++j)
    //     DiscreteFourierTransform::NegacyclicForwardTransform(dct[j]);
    
    // // acc = dct * ek (matrix product);

    // // initialize acc to zeros
    // for (size_t i = 0; i < 2; ++i){
    //     for (size_t j = 0; j < N; ++j){
    //         acc[i][j] = std::complex<double>(0.0, 0.0);
    //     }
    // }
    // // for elements[0]:
    // for (size_t i = 1; i < digitsG2; ++i){
    //     for (size_t j = 0; j < N; ++j){
    //         acc[0][j] += (dct[i][j] * (*ek)[i][0][j]);
    //     }
    // }
    // // for elements[1]:
    // for (size_t i = 1; i < digitsG2; ++i){
    //     for (size_t j = 0; j < N; ++j){
    //         acc[1][j] += (dct[i][j] * (*ek)[i][1][j]);
    //     }
    // }
}

// void RingGSWAccumulatorDM::AddToAccDM_CUDA(const std::shared_ptr<RingGSWCryptoParams> params, const std::shared_ptr<std::vector<std::vector<std::vector<std::complex<double>>>>> ek,
//                     std::vector<std::vector<std::complex<double>>>& acc) const {
//     uint32_t digitsG2 = params->GetDigitsG() << 1;
//     auto polyParams   = params->GetPolyParams();
//     // auto q           = params->Getq();
//     auto Q            = params->GetQ();
//     auto N            = params->GetN();

//     std::vector<std::vector<std::complex<double>>> ct(acc);
//     std::vector<std::vector<std::complex<double>>> dct(digitsG2, std::vector<std::complex<double>>(N, std::complex<double>(0.0, 0.0)));

//     SignedDigitDecompose_CUDA(params, ct, dct);
    
//     // calls digitsG2 FFTs
//     for (size_t j = 0; j < digitsG2; ++j)
//         RingGSWAccumulator::NegacyclicForwardTransform(dct[j]);
    
//     // acc = dct * ek (matrix product);

//     // initialize acc to zeros
//     for (size_t i = 0; i < 2; ++i){
//         for (size_t j = 0; j < N; ++j){
//             acc[i][j] = std::complex<double>(0.0, 0.0);
//         }
//     }
//     // for elements[0]:
//     for (size_t i = 1; i < digitsG2; ++i){
//         for (size_t j = 0; j < N; ++j){
//             acc[0][j] += (dct[i][j] * (*ek)[i][0][j]);
//         }
//     }
//     // for elements[1]:
//     for (size_t i = 1; i < digitsG2; ++i){
//         for (size_t j = 0; j < N; ++j){
//             acc[1][j] += (dct[i][j] * (*ek)[i][1][j]);
//         }
//     }

//     // calls 2 IFFTs
//     for (size_t i = 0; i < 2; ++i)
//          RingGSWAccumulator::NegacyclicInverseTransform(acc[i]);

//     // Round to INT64 and MOD
//     for (size_t i = 0; i < 2; ++i){
//         for (size_t j = 0; j < N; ++j){
//             int64_t temp = static_cast<int64_t>(round(acc[i][j].real()));
//             acc[i][j].real(temp % Q.ConvertToInt());
//         }
//     }
// }

};  // namespace lbcrypto
