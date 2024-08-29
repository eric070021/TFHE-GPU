# openFHE-GPU

![CMake Version](https://img.shields.io/badge/CMake-%3E%3D3.18-brightgreen.svg)

A high-performance library that leverages GPU acceleration to boost the TFHE (Fully Homomorphic Encryption) bootstrapping process in the OpenFHE library.

## Licensing

This project is based on the OpenFHE library, which is licensed under the BSD 2-Clause License. The original BSD 2-Clause License can be found [here](./LICENSE-BSD).

The GPU backend and any additional contributions by Inventec Corporation are licensed under the MIT License. The MIT License can be found [here](./LICENSE-MIT).

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Building](#building)
- [Supported APIs](#supported-apis)
  - [GPU Setup](#gpu-setup)
  - [GPU Clean](#gpu-clean)
  - [GenerateBinFHEContext](#generatebinfhecontext)
  - [EvalBinGate](#evalbingate)
  - [EvalFunc](#evalfunc)
  - [EvalFloor](#evalfloor)
  - [EvalSign](#evalsign)
  - [EvalDecomp](#evaldecomp)
  - [CiphertextMulMatrix](#ciphertextmulmatrix)
- [Sample Program](#sample-program)

## Introduction

**openFHE-GPU** is a powerful library designed to enhance the TFHE bootstrapping by leveraging the computational capabilities of modern GPUs. It provides an efficient and performant way to accelerate secure computations, enabling faster execution of homomorphically encrypted operations.

## Getting Started

### Prerequisites

To successfully build and use **openFHE-GPU**, you need the following prerequisites:

- CMake version 3.18 or higher
- Supported host compiler (C++17 required)
- GCC 7+
- CUDA Compute Compatibility: Tested with compute compatibility 8.6 and 8.9
- NVCC: Version 11.0.194 or newer (CUDA Toolkit 11.0 or newer)

### Building

To build the project, follow these steps:

1. Clone the repository to your local machine:

```bash
   git clone https://github.com/eric070021/openFHE-GPU.git
   cd openFHE-GPU
```
2. Create a release build:
```bash
   make create_release
```
3. Build the project:
```bash
   make build_release
```
4. Run the example to test the library:
```bash
   ./build_release/bin/examples/binfhe/unittest
   ./build_release/bin/examples/binfhe/time-estimate
```
## Supported APIs

### GPU Setup
Make sure to call this api after generating bootstrapping key.
- **Input**:
  - numGPUs: Number of GPUs to use (default using all available GPUs)
```cpp
cc.GPUSetup(int numGPUs);
```

### GPU Clean
Call this api at the end of the program.
```cpp
cc.GPUClean();
```

### GenerateBinFHEContext
Extension of the original GenerateBinFHEContext
- **Input**:
  - set: the parameter set: TOY, MEDIUM, STD128, STD192, STD256
  - arbFunc: whether need to evaluate an arbitrary function using functional bootstrapping
  - logQ: log(input ciphertext modulus)
  - N: ring dimension for RingGSW/RLWE used in bootstrapping
  - method: the bootstrapping method (DM or CGGI)
  - timeOptimization: whether to use dynamic bootstrapping technique
  - baseG: base for RingGSW used in bootstrapping
  - numDigitsToThrow: number of digits to throw in the bootstrapping
```cpp
void GenerateBinFHEContext(BINFHE_PARAMSET set, bool arbFunc, uint32_t logQ = 11, int64_t N = 0, BINFHE_METHOD method = GINX, bool timeOptimization = false , uint32_t baseG = 0, uint32_t numDigitsToThrow = 0);
```

### EvalBinGate
- **Input**:
  - gate: Gate you want to evaluate
  - ct1: Vector of LWECiphertext 1
  - ct2: Vector of LWECiphertext 2
- **Output**:
  - Vector of LWECiphertext
```cpp
std::vector<LWECiphertext> EvalBinGate(BINGATE gate, const std::vector<LWECiphertext>& ct1, const std::vector<LWECiphertext>& ct2) const;
```

### EvalFunc
- **Input**:
  - ct: Vector of LWECiphertext
  - LUT: LookUpTable
- **Output**:
  - Vector of LWECiphertext
```cpp
std::vector<LWECiphertext> EvalFunc(const std::vector<LWECiphertext>& ct, const std::vector<NativeInteger>& LUT) const;
```
- **Input**:
  - ct: Vector of LWECiphertext
  - LUT: Vector of LookUpTables
- **Output**:
  - Vector of LWECiphertext
```cpp
std::vector<LWECiphertext> EvalFunc(const std::vector<LWECiphertext>& ct, const std::vector<std::vector<NativeInteger>>& LUT) const;
```

### EvalFloor
- **Input**:
  - ct: Vector of LWECiphertext
  - roundbits: Round bits
- **Output**:
  - Vector of LWECiphertext
```cpp
std::vector<LWECiphertext> EvalFloor(const std::vector<LWECiphertext>& ct, uint32_t roundbits = 0) const;
```

### EvalSign
- **Input**:
  - ct: Vector of LWECiphertext
- **Output**:
  - Vector of LWECiphertext
```cpp
std::vector<LWECiphertext> EvalSign(const std::vector<LWECiphertext>& ct) const;
```

### EvalDecomp
- **Input**:
  - ct: Vector of LWECiphertext
- **Output**:
  - Vector of Vector of LWECiphertext
```cpp
std::vector<std::vector<LWECiphertext>> EvalDecomp(const std::vector<LWECiphertext>& ct) const;
```

### CiphertextMulMatrix
This function performs matrix multiplication with a vector of LWECiphertext objects and a matrix represented by a vector of vectors of int64_t elements. The elements in  output ciphertexts are all mod by Qks (modulus of keyswich).
- **Input**:
  - ct: Vector of LWECiphertext
  - matrix: Matrix to be multiplied
  - modulus: Modulus of output LWECiphertext
- **Output**:
  - Vector of LWECiphertext
```cpp
std::vector<LWECiphertext> CiphertextMulMatrix(const std::vector<LWECiphertext>& ct, const std::vector<std::vector<int64_t>>& matrix, uint64_t modulus) const;
```

## Sample Program
The program below shows an example calling EvalFunc api.
```cpp
#include "binfhecontext.h"

using namespace lbcrypto;

int main() {
    // Sample Program: Step 1: Set CryptoContext
    auto cc = BinFHEContext();
    // use default baseG, throw 1 digit
    cc.GenerateBinFHEContext(STD128, true, 12, 0, GINX, false, 0, 1);

    // Sample Program: Step 2: Key Generation

    // Generate the secret key
    auto sk = cc.KeyGen();

    std::cout << "Generating the bootstrapping keys..." << std::endl;

    // Generate the bootstrapping keys (refresh and switching keys)
    cc.BTKeyGen(sk);

    std::cout << "Completed the key generation." << std::endl;

    // Sample Program: Step 3: Setup GPU
    std::cout << "Setting up GPU..." << std::endl;
    // default using all GPUs
    cc.GPUSetup();

    std::cout << "Completed the GPU Setup." << std::endl;

    // Sample Program: Step 4: Create the to-be-evaluated funciton and obtain its corresponding LUT
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
    std::cout << "Evaluate x^3%" << p << "." << std::endl;

    // Sample Program: Step 5: evalute f(x) homomorphically and decrypt
    // Note that we check for all the possible plaintexts.
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

    // Sample Program: Step 6: Clean GPU
    cc.GPUClean();
    
    return 0;
}
```
