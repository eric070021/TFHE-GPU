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

double printerror(std::shared_ptr<RingGSWCryptoParams> RGSWParams, int folder, int iter){
    auto N = RGSWParams->GetN();
    auto Q = RGSWParams->GetQ();
    auto polyParams   = RGSWParams->GetPolyParams();
    uint64_t num;

    /* Secret Key */
    NativePoly skNPoly = NativePoly(polyParams, Format::EVALUATION);
    std::ifstream skFile;
    skFile.open("txt/" + std::to_string(folder) + "/sk.txt", std::ios::in);
    NativeVector m(N, Q);
    for(uint32_t i = 0; i < N; i++) {
        skFile >> num;
        m[i] = num;
        skFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    skNPoly.SetValues(std::move(m), Format::EVALUATION);
    skNPoly.SetFormat(Format::COEFFICIENT);
    skFile.close();

    /* NTT result */
    NativeVector ntt_A(N, Q);
    NativeInteger ntt_B;
    std::ifstream nttFile;
    nttFile.open("txt/" + std::to_string(folder) + "/ntt" + std::to_string(iter) + ".txt", std::ios::in);
    for(uint32_t i = 0; i < N; i++) {
        nttFile >> num;
        ntt_A[i] = num;
        nttFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    nttFile >> num;
    ntt_B = num;
    auto nttLWE = std::make_shared<LWECiphertextImpl>(std::move(ntt_A), std::move(ntt_B));
    nttFile.close();

    /* FFT result */
    NativeVector fft_A(N, Q);
    NativeInteger fft_B;
    std::ifstream fftFile;
    fftFile.open("txt/" + std::to_string(folder) + "/fft" + std::to_string(iter) + ".txt", std::ios::in);
    for(uint32_t i = 0; i < N; i++) {
        fftFile >> num;
        fft_A[i] = num;
        fftFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    fftFile >> num;
    fft_B = num;
    auto fftLWE = std::make_shared<LWECiphertextImpl>(std::move(fft_A), std::move(fft_B));
    fftFile.close();

    /* Decrypt */
    NativeVector s   = skNPoly.GetValues();
    s.SwitchModulus(Q);
    NativeInteger mu = Q.ComputeMu();
    NativeVector ntt_a   = nttLWE->GetA();
    NativeInteger ntt_inner(0);
    for (size_t i = 0; i < N; ++i) {
        ntt_inner += ntt_a[i].ModMulFast(s[i], Q, mu);
    }
    ntt_inner.ModEq(Q);
    NativeInteger ntt_r = nttLWE->GetB();
    ntt_r.ModSubFastEq(ntt_inner, Q);

    NativeVector fft_a   = fftLWE->GetA();
    NativeInteger fft_inner(0);
    for (size_t i = 0; i < N; ++i) {
        fft_inner += fft_a[i].ModMulFast(s[i], Q, mu);
    }
    fft_inner.ModEq(Q);
    NativeInteger fft_r = fftLWE->GetB();
    fft_r.ModSubFastEq(fft_inner, Q);

    /* Error analysis */
    uint64_t err = fft_r.ConvertToInt() < ntt_r.ConvertToInt() ? ntt_r.ConvertToInt() - fft_r.ConvertToInt() : fft_r.ConvertToInt() - ntt_r.ConvertToInt();
    err = err < (Q.ConvertToInt() - err) ? err : Q.ConvertToInt() - err;
    //NativeInteger err = std::min(fft_r.ModSubFast(ntt_r, Q), ntt_r.ModSubFast(fft_r, Q));
    return static_cast<double>(err) / Q.ConvertToDouble();
}

int main() {
    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(STD128, true, 12);
    auto RGSWParams   = cc.GetParams()->GetRingGSWParams();
    auto LWEParams    = cc.GetParams()->GetLWEParams();
    auto n = LWEParams->Getn();

    // /* Secret Key */
    // NativePoly skNPoly = NativePoly(polyParams, Format::EVALUATION);
    // std::ifstream skFile;
    // skFile.open("txt/sk.txt", std::ios::in);
    // NativeVector m1(N, Q);
    // for(uint32_t i = 0; i < N; i++) {
    //     skFile >> num;
    //     m1[i] = num;
    //     skFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    // }
    // skNPoly.SetValues(std::move(m1), Format::EVALUATION);
    // skNPoly.SetFormat(Format::COEFFICIENT);
    // skFile.close();

    for(uint32_t iter = 1; iter <= n; iter++){
        double sum = 0;
        for(uint32_t folder = 1; folder <= 512; folder++){
            sum += printerror(RGSWParams, folder, iter);
        }
        std::cout << std::fixed << sum / 512 << "," << std::flush;
    }

    return 0;
}