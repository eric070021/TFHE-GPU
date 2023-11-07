# openFHE-GPU

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![CMake Version](https://img.shields.io/badge/CMake-%3E%3D3.18-brightgreen.svg)

A high-performance library that leverages GPU acceleration to boost the TFHE (Fully Homomorphic Encryption) bootstrapping process.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Building](#building)

## Introduction

**openFHE-GPU** is a powerful library designed to enhance the TFHE bootstrapping process by harnessing the computational capabilities of modern GPUs. It provides an efficient and performant way to accelerate secure computations, enabling faster execution of homomorphically encrypted operations.

## Getting Started

### Prerequisites

To successfully build and use **openFHE-GPU**, you need the following prerequisites:

- CMake version 3.18 or higher
- A compatible GPU with CUDA compatibility >= 7.0

### Building

To build the project, follow these steps:

1. Clone the repository to your local machine:

```bash
   git clone https://github.com/yourusername/openFHE-GPU.git
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
   ./build_release/bin/examples/binfhe/eval-function
```