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

#include "rgsw-acc-cggi.h"

#include <string>
#include <fstream>

namespace lbcrypto {

// bootstrapping key for FFT-based accumulator
//std::shared_ptr<std::vector<std::vector<std::vector<std::shared_ptr<std::vector<std::vector<std::vector<Complex>>>>>>>> GINX_bootstrappingKey_FFT;

// Key generation as described in Section 4 of https://eprint.iacr.org/2014/816
RingGSWACCKey RingGSWAccumulatorCGGI::KeyGenAcc(const std::shared_ptr<RingGSWCryptoParams> params,
                                                const NativePoly& skNTT, ConstLWEPrivateKey LWEsk) const {
    auto sv         = LWEsk->GetElement();
    int32_t mod     = sv.GetModulus().ConvertToInt();
    int32_t modHalf = mod >> 1;
    uint32_t n      = sv.GetLength();
    auto ek         = std::make_shared<RingGSWACCKeyImpl>(1, 2, n);

    // handles ternary secrets using signed mod 3 arithmetic; 0 -> {0,0}, 1 ->
    // {1,0}, -1 -> {0,1}
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        int32_t s = (int32_t)sv[i].ConvertToInt();
        if (s > modHalf) {
            s -= mod;
        }

        switch (s) {
            case 0:
                (*ek)[0][0][i] = KeyGenCGGI(params, skNTT, 0);
                (*ek)[0][1][i] = KeyGenCGGI(params, skNTT, 0);
                break;
            case 1:
                (*ek)[0][0][i] = KeyGenCGGI(params, skNTT, 1);
                (*ek)[0][1][i] = KeyGenCGGI(params, skNTT, 0);
                break;
            case -1:
                (*ek)[0][0][i] = KeyGenCGGI(params, skNTT, 0);
                (*ek)[0][1][i] = KeyGenCGGI(params, skNTT, 1);
                break;
            default:
                std::string errMsg = "ERROR: only ternary secret key distributions are supported.";
                OPENFHE_THROW(not_implemented_error, errMsg);
        }
    }

    // // construct bootstrapping key for FFT-based accumulator
    // GINX_bootstrappingKey_FFT = std::make_shared<std::vector<std::vector<std::vector<std::shared_ptr<std::vector<std::vector<std::vector<Complex>>>>>>>>();
    // (*GINX_bootstrappingKey_FFT).resize(1);
    // for (size_t i = 0; i < 1; ++i) {
    //     (*GINX_bootstrappingKey_FFT)[i].resize(2);
    //     for (size_t j = 0; j < 2; ++j) {
    //         (*GINX_bootstrappingKey_FFT)[i][j].resize(n);
    //     }
    // }

    // for (size_t i = 0; i < n; ++i) {
    //     (*GINX_bootstrappingKey_FFT)[0][0][i] = KeyCopyCGGI_FFT(params, (*ek)[0][0][i]);
    //     (*GINX_bootstrappingKey_FFT)[0][1][i] = KeyCopyCGGI_FFT(params, (*ek)[0][1][i]);
    // }

//     // copy ek to GINX_bootstrappingKey_FFT
//     NativeInteger Q   = params->GetQ();
//     NativeInteger QHalf = Q >> 1;
//     NativeInteger::SignedNativeInt Q_int = Q.ConvertToInt();

//     // copy value of skNTT to skFFT
//     NativePoly skNTT_t(skNTT);
//     skNTT_t.SetFormat(Format::COEFFICIENT);
//     std::vector<Complex> skFFT (skNTT_t.GetLength(), Complex(0.0, 0.0));
//     for (size_t i = 0; i < skNTT_t.GetLength(); ++i) {
//         NativeInteger::SignedNativeInt d = (skNTT_t[i] < QHalf) ? skNTT_t[i].ConvertToInt() : (skNTT_t[i].ConvertToInt() - Q_int);
//         skFFT[i].real(static_cast<BasicFloat>(d));
//     }

//     // FFT of skFFT
//     DiscreteFourierTransform::NegacyclicForwardTransform(skFFT);

// #pragma omp parallel for
//     for (size_t i = 0; i < n; ++i) {
//         int32_t s = (int32_t)sv[i].ConvertToInt();
//         if (s > modHalf) {
//             s -= mod;
//         }

//         switch (s) {
//             case 0:
//                 (*GINX_bootstrappingKey_FFT)[0][0][i] = KeyGenCGGI_FFT(params, skFFT, 0);
//                 (*GINX_bootstrappingKey_FFT)[0][1][i] = KeyGenCGGI_FFT(params, skFFT, 0);
//                 break;
//             case 1:
//                 (*GINX_bootstrappingKey_FFT)[0][0][i] = KeyGenCGGI_FFT(params, skFFT, 1);
//                 (*GINX_bootstrappingKey_FFT)[0][1][i] = KeyGenCGGI_FFT(params, skFFT, 0);
//                 break;
//             case -1:
//                 (*GINX_bootstrappingKey_FFT)[0][0][i] = KeyGenCGGI_FFT(params, skFFT, 0);
//                 (*GINX_bootstrappingKey_FFT)[0][1][i] = KeyGenCGGI_FFT(params, skFFT, 1);
//                 break;
//             default:
//                 std::string errMsg = "ERROR: only ternary secret key distributions are supported.";
//                 OPENFHE_THROW(not_implemented_error, errMsg);
//         }
//     }

    /* Bring data on GPU */
    //GPUSetup(GINX_bootstrappingKey_FFT, params);

    return ek;
}

void RingGSWAccumulatorCGGI::EvalAcc(const std::shared_ptr<RingGSWCryptoParams> params, const RingGSWACCKey ek,
                                     RLWECiphertext& acc, const NativeVector& a) const {

    // auto mod        = a.GetModulus();
    // uint32_t n      = a.GetLength();
    // uint32_t M      = 2 * params->GetN();
    // uint32_t modInt = mod.ConvertToInt();

    /* HE parameters set */
    NativeInteger Q                         = params->GetQ();
    NativeInteger QHalf                     = Q >> 1;
    NativeInteger::SignedNativeInt Q_int    = Q.ConvertToInt();
    uint32_t  N                             = params->GetN();
    auto polyParams                         = params->GetPolyParams();

    // cast acc to BasicFloat
    NativePoly acc0(acc->GetElements()[0]), acc1(acc->GetElements()[1]);
    acc0.SetFormat(Format::COEFFICIENT);
    acc1.SetFormat(Format::COEFFICIENT);
    std::vector<std::vector<Complex>> acc_d(2, std::vector<Complex>(N, Complex(0.0, 0.0)));
    for (size_t i = 0; i < N; ++i) {
        NativeInteger::SignedNativeInt d = (acc0[i] < QHalf) ? acc0[i].ConvertToInt() : (acc0[i].ConvertToInt() - Q_int);
        acc_d[0][i].real(static_cast<BasicFloat>(d));
        d = (acc1[i] < QHalf) ? acc1[i].ConvertToInt() : (acc1[i].ConvertToInt() - Q_int);
        acc_d[1][i].real(static_cast<BasicFloat>(d));
    }

    AddToAccCGGI_CUDA(params, a, acc_d, "SINGLE");
    
    // DiscreteFourierTransform::NegacyclicForwardTransform(acc_d[0]);
    // DiscreteFourierTransform::NegacyclicForwardTransform(acc_d[1]);

    // for (size_t i = 0; i < n; ++i) {
    //     // handles -a*E(1) and handles -a*E(-1) = a*E(1)
    //     AddToAccCGGI(params, (*ek)[0][0][i], (*ek)[0][1][i], mod.ModSub(a[i], mod) * (M / modInt), acc);
    //     AddToAccCGGI_FFT(params, (*GINX_bootstrappingKey_FFT)[0][0][i], (*GINX_bootstrappingKey_FFT)[0][1][i], mod.ModSub(a[i], mod) * (M / modInt), acc_d);
    // }

    // // calls 2 IFFTs
    // for (size_t i = 0; i < 2; ++i)
    //     DiscreteFourierTransform::NegacyclicInverseTransform(acc_d[i]);

    // // Round to INT64 and MOD
    // for (size_t i = 0; i < 2; ++i){
    //     for (size_t j = 0; j < N; ++j) {
    //         ModInteger temp = static_cast<ModInteger>(round(acc_d[i][j].real()));
    //         temp = temp % Q_int;
    //         if (temp < 0)
    //             temp += Q_int;
    //         acc_d[i][j].real(static_cast<BasicFloat>(temp));
    //     }
    // }

    // std::vector<NativePoly> acc_t = acc->GetElements();
    // acc_t[0].SetFormat(Format::COEFFICIENT);
    // acc_t[1].SetFormat(Format::COEFFICIENT);
    // std::ofstream outputFile;
    // outputFile.open("ntt.txt", std::ios::out);
    // for(uint32_t i = 0; i < 2; i++)
    //     for(uint32_t j = 0; j < N; j++)
    //         outputFile << acc_t[i][j] << std::endl;
    // outputFile.close();

    // std::ofstream outputFile;
    // outputFile.open("fft.txt", std::ios::out);
    // for(uint32_t i = 0; i < 2; i++)
    //     for(uint32_t j = 0; j < N ; j++)
    //         outputFile << static_cast<BasicInteger>(acc_d[i][j].real()) << std::endl;
    // outputFile.close();

    //cast acc_d to NativePoly
    NativeVector ret0(N, Q), ret1(N, Q);
    for (size_t i = 0; i < N; ++i) {
        ret0[i] = static_cast<BasicInteger>(acc_d[0][i].real());
        ret1[i] = static_cast<BasicInteger>(acc_d[1][i].real());
    }
    std::vector<NativePoly> res(2);
    res[0] = NativePoly(polyParams, Format::COEFFICIENT, false);
    res[1] = NativePoly(polyParams, Format::COEFFICIENT, false);
    res[0].SetValues(std::move(ret0), Format::COEFFICIENT);
    res[1].SetValues(std::move(ret1), Format::COEFFICIENT);
    res[0].SetFormat(Format::EVALUATION);
    res[1].SetFormat(Format::EVALUATION);
    acc = std::make_shared<RLWECiphertextImpl>(std::move(res));
}

void RingGSWAccumulatorCGGI::EvalAcc(const std::shared_ptr<RingGSWCryptoParams> params, const RingGSWACCKey ek, 
        std::shared_ptr<std::vector<RLWECiphertext>> acc, const std::vector<NativeVector>& a) const {

    /* HE parameters set */
    NativeInteger Q                         = params->GetQ();
    NativeInteger QHalf                     = Q >> 1;
    NativeInteger::SignedNativeInt Q_int    = Q.ConvertToInt();
    uint32_t  N                             = params->GetN();
    auto polyParams                         = params->GetPolyParams();

    std::vector<std::vector<std::vector<Complex>>> acc_d_vec (acc->size(), std::vector<std::vector<Complex>>(2, std::vector<Complex>(N, Complex(0.0, 0.0))));

    // cast acc to BasicFloat
    for (uint32_t count = 0; count < acc->size(); count++){
        NativePoly acc0((*acc)[count]->GetElements()[0]), acc1((*acc)[count]->GetElements()[1]);
        std::vector<std::vector<Complex>> acc_d(2, std::vector<Complex>(N, Complex(0.0, 0.0)));
        for (size_t i = 0; i < N; ++i) {
            NativeInteger::SignedNativeInt d = (acc0[i] < QHalf) ? acc0[i].ConvertToInt() : (acc0[i].ConvertToInt() - Q_int);
            acc_d_vec[count][0][i].real(static_cast<BasicFloat>(d));
            d = (acc1[i] < QHalf) ? acc1[i].ConvertToInt() : (acc1[i].ConvertToInt() - Q_int);
            acc_d_vec[count][1][i].real(static_cast<BasicFloat>(d));
        }
    }

    AddToAccCGGI_CUDA(params, a, acc_d_vec, "SINGLE");

    //cast acc_d_vec back to NativePoly
    for (uint32_t count = 0; count < acc->size(); count++){
        NativeVector ret0(N, Q), ret1(N, Q);
        for (size_t i = 0; i < N; ++i) {
            ret0[i] = static_cast<BasicInteger>(acc_d_vec[count][0][i].real());
            ret1[i] = static_cast<BasicInteger>(acc_d_vec[count][1][i].real());
        }
        std::vector<NativePoly> res(2);
        res[0] = NativePoly(polyParams, Format::COEFFICIENT, false);
        res[1] = NativePoly(polyParams, Format::COEFFICIENT, false);
        res[0].SetValues(std::move(ret0), Format::COEFFICIENT);
        res[1].SetValues(std::move(ret1), Format::COEFFICIENT);

        (*acc)[count] = std::make_shared<RLWECiphertextImpl>(std::move(res));
    }
}


void RingGSWAccumulatorCGGI::MKMSwitch(const std::shared_ptr<LWECryptoParams> params, std::shared_ptr<std::vector<LWECiphertext>> ctExt,
                         NativeInteger fmod) const{
    MKMSwitch_CUDA(params, ctExt, fmod);
}

// Encryption for the CGGI variant, as described in https://eprint.iacr.org/2020/086
RingGSWEvalKey RingGSWAccumulatorCGGI::KeyGenCGGI(const std::shared_ptr<RingGSWCryptoParams> params,
                                                  const NativePoly& skNTT, const LWEPlaintext& m) const {
    NativeInteger Q   = params->GetQ();
    uint32_t digitsG  = params->GetDigitsG();
    uint32_t digitsG2 = digitsG << 1;
    auto Gpow         = params->GetGPower();
    auto polyParams   = params->GetPolyParams();
    auto result       = std::make_shared<RingGSWEvalKeyImpl>(digitsG2, 2);

    DiscreteUniformGeneratorImpl<NativeVector> dug;
    dug.SetModulus(Q);

    // tempA is introduced to minimize the number of NTTs
    std::vector<NativePoly> tempA(digitsG2);

    for (size_t i = 0; i < digitsG2; ++i) {
        (*result)[i][0] = NativePoly(dug, polyParams, Format::COEFFICIENT);
        tempA[i]        = (*result)[i][0];
        (*result)[i][1] = NativePoly(params->GetDgg(), polyParams, Format::COEFFICIENT);
    }

    if (m > 0) {
        for (size_t i = 0; i < digitsG; ++i) {
            // Add G Multiple
            (*result)[2 * i][0][0].ModAddEq(Gpow[i], Q);
            // [a,as+e] + G
            (*result)[2 * i + 1][1][0].ModAddEq(Gpow[i], Q);
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

// CGGI Accumulation as described in https://eprint.iacr.org/2020/086
// Added ternary MUX introduced in paper https://eprint.iacr.org/2022/074.pdf section 5
// We optimize the algorithm by multiplying the monomial after the external product
// This reduces the number of polynomial multiplications which further reduces the runtime
void RingGSWAccumulatorCGGI::AddToAccCGGI(const std::shared_ptr<RingGSWCryptoParams> params, const RingGSWEvalKey ek1,
                                          const RingGSWEvalKey ek2, const NativeInteger& a, RLWECiphertext& acc) const {
    // cycltomic order
    uint64_t MInt = 2 * params->GetN();
    NativeInteger M(MInt);
    uint32_t digitsG2 = params->GetDigitsG() << 1;
    auto polyParams   = params->GetPolyParams();

    std::vector<NativePoly> ct = acc->GetElements();
    std::vector<NativePoly> dct(digitsG2);

    // initialize dct to zeros
    for (size_t i = 0; i < digitsG2; ++i)
        dct[i] = NativePoly(polyParams, Format::COEFFICIENT, true);

    // calls 2 NTTs
    for (size_t i = 0; i < 2; ++i)
        ct[i].SetFormat(Format::COEFFICIENT);

    SignedDigitDecompose(params, ct, dct);

    for (size_t i = 0; i < digitsG2; ++i)
        dct[i].SetFormat(Format::EVALUATION);

    // First obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
    auto aNeg         = M.ModSub(a, M);
    uint64_t indexPos = a.ConvertToInt();
    uint64_t indexNeg = aNeg.ConvertToInt();
    // index is in range [0,m] - so we need to adjust the edge case when
    // index = m to index = 0
    if (indexPos == MInt)
        indexPos = 0;
    if (indexNeg == MInt)
        indexNeg = 0;
    const NativePoly& monomial    = params->GetMonomial(indexPos);
    const NativePoly& monomialNeg = params->GetMonomial(indexNeg);

    // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
    // uses in-place * operators for the last call to dct[i] to gain performance
    // improvement. Needs to be done using two loops for ternary secrets.
    // TODO (dsuponit): benchmark cases with operator*() and operator*=(). Make a copy of dct?
    const std::vector<std::vector<NativePoly>>& ev1 = ek1->GetElements();
    for (size_t j = 0; j < 2; ++j) {
        NativePoly temp1(dct[0] * ev1[0][j]);
        for (size_t l = 1; l < digitsG2; ++l)
            temp1 += (dct[l] * ev1[l][j]);
        acc->GetElements()[j] += (temp1 *= monomial);
    }

    const std::vector<std::vector<NativePoly>>& ev2 = ek2->GetElements();
    // for elements[0]:
    NativePoly temp1(dct[0] * ev2[0][0]);
    for (size_t l = 1; l < digitsG2; ++l)
        temp1 += (dct[l] * ev2[l][0]);
    acc->GetElements()[0] += (temp1 *= monomialNeg);
    // for elements[1]:
    NativePoly temp2(dct[0] * ev2[0][1]);
    for (size_t l = 1; l < digitsG2; ++l)
        temp2 += (dct[l] *= ev2[l][1]);
    acc->GetElements()[1] += (temp2 *= monomialNeg);
}

/* FFT variant bootstrapping */
std::shared_ptr<std::vector<std::vector<std::vector<Complex>>>> RingGSWAccumulatorCGGI::KeyGenCGGI_FFT(const std::shared_ptr<RingGSWCryptoParams> params, 
        const std::vector<Complex>& skFFT, const LWEPlaintext& m) const{
    NativeInteger Q   = params->GetQ();
    NativeInteger QHalf = Q >> 1;
    NativeInteger::SignedNativeInt Q_int = Q.ConvertToInt();
    uint32_t N        = params->GetN();
    uint32_t digitsG  = params->GetDigitsG();
    uint32_t digitsG2 = digitsG << 1;
    auto Gpow         = params->GetGPower();
    auto polyParams   = params->GetPolyParams();
    auto result       = std::make_shared<RingGSWEvalKeyImpl>(digitsG2, 2);

    DiscreteUniformGeneratorImpl<NativeVector> dug;
    dug.SetModulus(Q);

    // tempA is introduced to minimize the number of NTTs
    std::vector<NativePoly> tempA(digitsG2);

    for (size_t i = 0; i < digitsG2; ++i) {
        (*result)[i][0] = NativePoly(dug, polyParams, Format::COEFFICIENT);
        tempA[i]        = (*result)[i][0];
        (*result)[i][1] = NativePoly(params->GetDgg(), polyParams, Format::COEFFICIENT);
    }

    if (m > 0) {
        for (size_t i = 0; i < digitsG; ++i) {
            // Add G Multiple
            (*result)[2 * i][0][0].ModAddEq(Gpow[i], Q);
            // [a,as+e] + G
            (*result)[2 * i + 1][1][0].ModAddEq(Gpow[i], Q);
        }
    }

    // cast result to BasicFloat
    auto result_d = std::make_shared<std::vector<std::vector<std::vector<Complex>>>>
        (digitsG2, std::vector<std::vector<Complex>>(2, std::vector<Complex>(N, Complex(0.0, 0.0))));
    for (size_t i = 0; i < digitsG2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < N; ++k) {
                NativeInteger::SignedNativeInt d = ((*result)[i][j][k] < QHalf) ? (*result)[i][j][k].ConvertToInt() : ((*result)[i][j][k].ConvertToInt() - Q_int);
                (*result_d)[i][j][k].real(static_cast<BasicFloat>(d));
            }
        }
    }

    // cast tempA to BasicFloat
    std::vector<std::vector<Complex>> tempA_d(digitsG2, std::vector<Complex>(N, Complex(0.0, 0.0)));
    for (size_t i = 0; i < digitsG2; ++i) {
        for (size_t j = 0; j < N; ++j) {
            NativeInteger::SignedNativeInt d = (tempA[i][j] < QHalf) ? tempA[i][j].ConvertToInt() : (tempA[i][j].ConvertToInt() - Q_int);
            tempA_d[i][j].real(static_cast<BasicFloat>(d));
        }
    }

    // 3*digitsG2 FTTs are called
    for (size_t i = 0; i < digitsG2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            DiscreteFourierTransform::NegacyclicForwardTransform((*result_d)[i][j]);
        }
    }

    for (size_t i = 0; i < digitsG2; ++i) {
        DiscreteFourierTransform::NegacyclicForwardTransform(tempA_d[i]);
        for (size_t j = 0; j < (N >> 1); ++j) {
            (*result_d)[i][1][j] += tempA_d[i][j] * skFFT[j];
        }
    }

    // transform back to coefficient domain
    for (size_t i = 0; i < digitsG2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            DiscreteFourierTransform::NegacyclicInverseTransform((*result_d)[i][j]);
        }
    }

    // Round to INT64 and MOD
    for (size_t i = 0; i < digitsG2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < N; ++k) {
                ModInteger temp = static_cast<ModInteger>(round((*result_d)[i][j][k].real()));
                temp = temp % static_cast<ModInteger>(Q_int);
                if (temp < 0)
                    temp += static_cast<ModInteger>(Q_int);
                if (temp >= static_cast<ModInteger>(QHalf.ConvertToInt()))
                    temp -= static_cast<ModInteger>(Q_int);
                (*result_d)[i][j][k].real(static_cast<BasicFloat>(temp));
            }
        }
    }

    // transform to evaluation domain
    for (size_t i = 0; i < digitsG2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            DiscreteFourierTransform::NegacyclicForwardTransform((*result_d)[i][j]);
        }
    }

    return result_d;
}

std::shared_ptr<std::vector<std::vector<std::vector<Complex>>>> RingGSWAccumulatorCGGI::KeyCopyCGGI_FFT(const std::shared_ptr<RingGSWCryptoParams> params, 
    RingGSWEvalKey ek) const{

    NativeInteger Q   = params->GetQ();
    NativeInteger QHalf = Q >> 1;
    NativeInteger::SignedNativeInt Q_int = Q.ConvertToInt();
    uint32_t digitsG  = params->GetDigitsG();
    uint32_t digitsG2 = digitsG << 1;
    uint32_t N        = params->GetN();

    auto ek_d = std::make_shared<std::vector<std::vector<std::vector<Complex>>>>
        (digitsG2, std::vector<std::vector<Complex>>(2, std::vector<Complex>(N, Complex(0.0, 0.0))));

    for (size_t j = 0; j < digitsG2; ++j) {
        for (size_t k = 0; k < 2; ++k) {
            NativePoly ek_t = (*ek)[j][k];
            ek_t.SetFormat(Format::COEFFICIENT);
            for (size_t l = 0; l < N; ++l) {
                NativeInteger::SignedNativeInt d = (ek_t[l] < QHalf) ? ek_t[l].ConvertToInt() : (ek_t[l].ConvertToInt() - Q_int);
                (*ek_d)[j][k][l].real(static_cast<BasicFloat>(d));
            }
        }
    }

    // transform to evaluation domain
    for (size_t i = 0; i < digitsG2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            DiscreteFourierTransform::NegacyclicForwardTransform((*ek_d)[i][j]);
        }
    }

    return ek_d;
}

void RingGSWAccumulatorCGGI::AddToAccCGGI_FFT(const std::shared_ptr<RingGSWCryptoParams> params, const std::shared_ptr<std::vector<std::vector<std::vector<Complex>>>> ek1,
                      const std::shared_ptr<std::vector<std::vector<std::vector<Complex>>>> ek2, const NativeInteger& a, std::vector<std::vector<Complex>>& acc) const{
    uint64_t MInt = 2 * params->GetN();
    NativeInteger M(MInt);
    uint32_t digitsG2 = params->GetDigitsG() << 1;
    auto polyParams   = params->GetPolyParams();
    auto Q            = params->GetQ();
    NativeInteger QHalf = Q >> 1;
    NativeInteger::SignedNativeInt Q_int = Q.ConvertToInt();
    auto N            = params->GetN();
    
    std::vector<std::vector<Complex>> ct(acc);
    std::vector<std::vector<Complex>> dct(digitsG2, std::vector<Complex>(N, Complex(0.0, 0.0)));

    // calls 2 Inverse FFTs
    for (size_t i = 0; i < 2; ++i)
        DiscreteFourierTransform::NegacyclicInverseTransform(ct[i]);

    // Round and MOD
    for (size_t i = 0; i < 2; ++i){
        for (size_t j = 0; j < N; ++j) {
            ModInteger temp = static_cast<ModInteger>(round(ct[i][j].real()));
            temp = temp % static_cast<ModInteger>(Q_int);
            if (temp < 0)
                temp += static_cast<ModInteger>(Q_int);
            if (temp >= static_cast<ModInteger>(QHalf.ConvertToInt()))
                temp -= static_cast<ModInteger>(Q_int);
            ct[i][j].real(static_cast<BasicFloat>(temp));
        }
    }

    SignedDigitDecompose_FFT(params, ct, dct);

    // calls digitsG2 Forward FFTs
    for (size_t i = 0; i < digitsG2; ++i)
        DiscreteFourierTransform::NegacyclicForwardTransform(dct[i]);

    // First obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
    auto aNeg         = M.ModSub(a, M);
    uint64_t indexPos = a.ConvertToInt();
    uint64_t indexNeg = aNeg.ConvertToInt();
    // index is in range [0,m] - so we need to adjust the edge case when
    // index = m to index = 0
    if (indexPos == MInt)
        indexPos = 0;
    if (indexNeg == MInt)
        indexNeg = 0;
    NativePoly monomial_t    = params->GetMonomial(indexPos);
    NativePoly monomialNeg_t = params->GetMonomial(indexNeg);
    monomial_t.SetFormat(Format::COEFFICIENT);
    monomialNeg_t.SetFormat(Format::COEFFICIENT);
    std::vector<Complex> monomial(N, Complex(0.0, 0.0));
    std::vector<Complex> monomialNeg(N, Complex(0.0, 0.0));
    for (size_t i = 0; i < N; ++i) {
        NativeInteger::SignedNativeInt d = (monomial_t[i] < QHalf) ? monomial_t[i].ConvertToInt() : (monomial_t[i].ConvertToInt() - Q_int);
        monomial[i].real(static_cast<BasicFloat>(d));
        d = (monomialNeg_t[i] < QHalf) ? monomialNeg_t[i].ConvertToInt() : (monomialNeg_t[i].ConvertToInt() - Q_int);
        monomialNeg[i].real(static_cast<BasicFloat>(d));
    }
    DiscreteFourierTransform::NegacyclicForwardTransform(monomial);
    DiscreteFourierTransform::NegacyclicForwardTransform(monomialNeg);

    // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
    // uses in-place * operators for the last call to dct[i] to gain performance
    // improvement. Needs to be done using two loops for ternary secrets.
    // TODO (dsuponit): benchmark cases with operator*() and operator*=(). Make a copy of dct?

    for (size_t l = 0; l < 2; ++l) {
        std::vector<Complex> temp((N >> 1), Complex(0.0, 0.0));
        for (size_t i = 0; i < digitsG2; ++i){
            for (size_t j = 0; j < (N >> 1); ++j){
                temp[j] += (dct[i][j] * (*ek1)[i][l][j]);
            }
        }
        for (size_t j = 0; j < (N >> 1); ++j){
            acc[l][j] += temp[j] * monomial[j];
        }
    }

    for (size_t l = 0; l < 2; ++l) {
        std::vector<Complex> temp((N >> 1), Complex(0.0, 0.0));
        for (size_t i = 0; i < digitsG2; ++i){
            for (size_t j = 0; j < (N >> 1); ++j){
                temp[j] += (dct[i][j] * (*ek2)[i][l][j]);
            }
        }
        for (size_t j = 0; j < (N >> 1); ++j){
            acc[l][j] += temp[j] * monomialNeg[j];
        }
    }
}

};  // namespace lbcrypto

// std::ofstream outputFile;
// outputFile.open("ntt.txt", std::ios::out);
// for(uint32_t i = 0; i < digitsG2; i++)
//     for(uint32_t j = 0; j < (N >> 1); j++)
//         outputFile << "(" << dct[i][j].real() << ", " << dct[i][j + (N >> 1)].real() << ")" << std::endl;
// outputFile.close();

// std::ofstream outputFile;
// outputFile.open("ntt.txt", std::ios::out);
// for(uint32_t i = 0; i < 2; i++)
//     for(uint32_t j = 0; j < (N >> 1); j++)
//         outputFile << "(" << ct[i][j].real() << ", " << ct[i][j + (N >> 1)].real() << ")" << std::endl;
// outputFile.close();

// std::ofstream outputFile;
// outputFile.open("1.txt", std::ios::out);
// for (size_t i = 0; i < digitsG2; ++i){
//     for (size_t j = 0; j < N; ++j){
//         ModInteger temp = static_cast<ModInteger>(dct[i][j].real());
//         if (temp < 0)
//             temp += Q_int;
//         outputFile << static_cast<uint32_t>(temp) << std::endl;
//     }
// }
// outputFile.close();

// std::ofstream outputFile;
// outputFile.open("2.txt", std::ios::out);
// auto N            = params->GetN();
// for (size_t i = 0; i < 2; ++i)
//     for (size_t j = 0; j < N; ++j)
//         outputFile << accVec[i][j] << std::endl;
// outputFile.close();

// uint32_t digitsG  = params->GetDigitsG();
// uint32_t digitsG2 = digitsG << 1;
// const std::vector<std::vector<NativePoly>>& ev1_t = (*ek)[0][0][i]->GetElements();
// const std::vector<std::vector<NativePoly>>& ev2_t = (*ek)[0][1][i]->GetElements();
// std::vector<std::vector<NativePoly>> ev1 (ev1_t);
// std::vector<std::vector<NativePoly>> ev2 (ev2_t);
// for (size_t i = 0; i < digitsG2; ++i) {
//     for (size_t j = 0; j < 2; ++j) {
//         ev1[i][j].SetFormat(Format::COEFFICIENT);
//         ev2[i][j].SetFormat(Format::COEFFICIENT);
//     }
// }
// auto ev1_fft = std::make_shared<std::vector<std::vector<std::vector<Complex>>>>
//     (digitsG2, std::vector<std::vector<Complex>>(2, std::vector<Complex>(N, Complex(0.0, 0.0))));
// auto ev2_fft = std::make_shared<std::vector<std::vector<std::vector<Complex>>>>
//     (digitsG2, std::vector<std::vector<Complex>>(2, std::vector<Complex>(N, Complex(0.0, 0.0))));
// for (size_t i = 0; i < digitsG2; ++i) {
//     for (size_t j = 0; j < 2; ++j) {
//         for (size_t k = 0; k < N; ++k) {
//             NativeInteger::SignedNativeInt d = (ev1[i][j][k] < QHalf) ? ev1[i][j][k].ConvertToInt() : (ev1[i][j][k].ConvertToInt() - Q_int);
//             (*ev1_fft)[i][j][k].real(static_cast<BasicFloat>(d));
//             d = (ev2[i][j][k] < QHalf) ? ev2[i][j][k].ConvertToInt() : (ev2[i][j][k].ConvertToInt() - Q_int);
//             (*ev2_fft)[i][j][k].real(static_cast<BasicFloat>(d));
//         }
//     }
// }
// for (size_t i = 0; i < digitsG2; ++i) {
//     for (size_t j = 0; j < 2; ++j) {
//         DiscreteFourierTransform::NegacyclicForwardTransform((*ev1_fft)[i][j]);
//         DiscreteFourierTransform::NegacyclicForwardTransform((*ev2_fft)[i][j]);
//     }
// }
// AddToAccCGGI_FFT(params, ev1_fft, ev2_fft, mod.ModSub(a[i], mod) * (M / modInt), acc_d);

// std::vector<NativePoly> acc_t = acc->GetElements();
// acc_t[0].SetFormat(Format::COEFFICIENT);
// acc_t[1].SetFormat(Format::COEFFICIENT);
// for (size_t i = 0; i < 2; ++i){
//     for (size_t j = 0; j < N; ++j){
//         if(acc_t[i][j] != static_cast<uint32_t>(acc_d[i][j].real()))
//             std::cout << "ERROR: acc_t[" << i << "][" << j << "] = " << acc_t[i][j] << " != acc_d[" << i << "][" << j << "] = " << static_cast<uint32_t>(acc_d[i][j].real()) << std::endl;
//     }
// }

// int N         = params->GetN();
// // compare two keys
// std::vector<std::vector<std::vector<Complex>>> bk_fft = (*(*GINX_bootstrappingKey_FFT)[0][0][0]);
// const std::vector<std::vector<NativePoly>>& ev1 = (*ek)[0][0][0]->GetElements();
// for(int i = 0; i < 8; ++i){
//     for(int j = 0; j < 2; ++j){
//         for(int k = 0; k < N; ++k){
//             if(ev1[i][j][k] != static_cast<uint32_t>(bk_fft[i][j][k].real()))
//                 std::cout << "ev1: " << ev1[i][j][k] << "bk_fft: " << static_cast<uint32_t>(bk_fft[i][j][k].real()) << std::endl;
//         }
//     }
// }