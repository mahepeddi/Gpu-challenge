{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyMZSCaseQ95NFu5ELPNs/1B",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/mahepeddi/Gpu-challenge/blob/main/day7.cu\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pfSRNk385_BT",
        "outputId": "f403cb24-6659-4877-b1b6-5ba353907575"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "nvcc: NVIDIA (R) Cuda compiler driver\n",
            "Copyright (c) 2005-2024 NVIDIA Corporation\n",
            "Built on Thu_Jun__6_02:18:23_PDT_2024\n",
            "Cuda compilation tools, release 12.5, V12.5.82\n",
            "Build cuda_12.5.r12.5/compiler.34385749_0\n"
          ]
        }
      ],
      "source": [
        "!nvcc --version"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile 2d_convolution_with_tiling.cu\n",
        "#include <stdio.h>\n",
        "#include <iostream>\n",
        "\n",
        "// I'm assuming that the mask and the matrix to be square for simplicity\n",
        "#define Mask_width 5\n",
        "#define shared_size (32 + Mask_width - 1)\n",
        "__constant__ float M[Mask_width][Mask_width];\n",
        "\n",
        "__global__ void twod_convolution_kernel(const float* A, float* C, int n) {\n",
        "    int threadx = threadIdx.x;\n",
        "    int thready = threadIdx.y;\n",
        "    int i = blockDim.x * blockIdx.x + threadx;\n",
        "    int j = blockDim.y * blockIdx.y + thready;\n",
        "\n",
        "    __shared__ float S_A[shared_size][shared_size];\n",
        "\n",
        "    // Load main data\n",
        "    if ((i < n) && (j < n)) {\n",
        "        S_A[threadx + Mask_width/2][thready + Mask_width/2] = A[i*n+j];\n",
        "    }\n",
        "\n",
        "    // Load left halo\n",
        "    if (threadx < Mask_width/2) {\n",
        "        int left_idx = blockIdx.x * blockDim.x - (Mask_width/2) + threadx;\n",
        "        if (left_idx >= 0 && j < n) {\n",
        "            S_A[threadx][thready + Mask_width/2] = A[left_idx*n+j];\n",
        "        }\n",
        "        else {\n",
        "            S_A[threadx][thready + Mask_width/2] = 0.0f;\n",
        "        }\n",
        "    }\n",
        "\n",
        "    // Load right halo\n",
        "    if (threadx < Mask_width/2) {\n",
        "        int right_idx = blockIdx.x * blockDim.x + blockDim.x + threadx;\n",
        "        if (right_idx < n && j < n) {\n",
        "            S_A[threadx + blockDim.x + Mask_width/2][thready + Mask_width/2] = A[right_idx*n+j];\n",
        "        }\n",
        "        else {\n",
        "            S_A[threadx + blockDim.x + Mask_width/2][thready + Mask_width/2] = 0.0f;\n",
        "        }\n",
        "    }\n",
        "\n",
        "    // Load top halo\n",
        "    if (thready < Mask_width/2) {\n",
        "        int top_idy = j - (Mask_width/2) + thready;\n",
        "        if (top_idy >= 0 && i < n) {\n",
        "            S_A[threadx + Mask_width/2][thready] = A[i*n+top_idy];\n",
        "        }\n",
        "        else {\n",
        "            S_A[threadx + Mask_width/2][thready] = 0.0f;\n",
        "        }\n",
        "    }\n",
        "\n",
        "    // Load bottom halo\n",
        "    if (thready < Mask_width/2) {\n",
        "        int bottom_idy = j + blockDim.y + thready;\n",
        "        if (bottom_idy < n && i < n) {\n",
        "            S_A[threadx + Mask_width/2][thready + blockDim.y + Mask_width/2] = A[i*n+bottom_idy];\n",
        "        }\n",
        "        else {\n",
        "            S_A[threadx + Mask_width/2][thready + blockDim.y + Mask_width/2] = 0.0f;\n",
        "        }\n",
        "    }\n",
        "\n",
        "    __syncthreads();\n",
        "\n",
        "    if ((i < n) && (j < n)) {\n",
        "        float result = 0.0f;\n",
        "        for (int k = 0; k < Mask_width; k++) {\n",
        "            for (int x = 0; x < Mask_width; x++) {\n",
        "                result += S_A[threadx + k][thready + x] * M[k][x];\n",
        "            }\n",
        "        }\n",
        "        C[i*n+j] = result;\n",
        "    }\n",
        "}\n",
        "\n",
        "void checkCudaError(const char* message) {\n",
        "    cudaError_t error = cudaGetLastError();\n",
        "    if (error != cudaSuccess) {\n",
        "        fprintf(stderr, \"%s - CUDA Error: %s\\n\", message, cudaGetErrorString(error));\n",
        "        exit(EXIT_FAILURE);\n",
        "    }\n",
        "}\n",
        "\n",
        "int main() {\n",
        "    int n = 10;\n",
        "    float *h_A = (float*)malloc(n * n * sizeof(float));\n",
        "    float *h_C = (float*)malloc(n * n * sizeof(float));\n",
        "    float d_M[Mask_width][Mask_width];\n",
        "\n",
        "    for (int i = 0; i < Mask_width; i++) {\n",
        "        for (int j = 0; j < Mask_width; j++) {\n",
        "            d_M[i][j] = 5;\n",
        "        }\n",
        "    }\n",
        "\n",
        "    for (int i = 0; i < n; i++) {\n",
        "        for (int j = 0; j < n; j++) {\n",
        "            h_A[i*n + j] = 3;\n",
        "        }\n",
        "    }\n",
        "\n",
        "    float *d_a, *d_c;\n",
        "    cudaMalloc(&d_a, n*n*sizeof(float));\n",
        "    cudaMalloc(&d_c, n*n*sizeof(float));\n",
        "    cudaMemcpy(d_a, h_A, n*n*sizeof(float), cudaMemcpyHostToDevice);\n",
        "    checkCudaError(\"Failed to copy input data to device\");\n",
        "    cudaMemcpyToSymbol(M, d_M, Mask_width*Mask_width*sizeof(float));\n",
        "    checkCudaError(\"Failed to copy mask data to device\");\n",
        "\n",
        "    dim3 dimBlock(32, 32);\n",
        "    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (n + dimBlock.y - 1) / dimBlock.y);\n",
        "    twod_convolution_kernel<<<dimGrid, dimBlock>>>(d_a, d_c, n);\n",
        "    checkCudaError(\"Failed to execute the kernel\");\n",
        "\n",
        "    cudaDeviceSynchronize();\n",
        "    cudaMemcpy(h_C, d_c, n*n*sizeof(float), cudaMemcpyDeviceToHost);\n",
        "    checkCudaError(\"Failed to copy output data to host\");\n",
        "\n",
        "    // Print results\n",
        "    printf(\"Results:\\n\");\n",
        "    for (int i = 0; i < n; i++) {\n",
        "        for (int j = 0; j < n; j++) {\n",
        "            printf(\"%.2f \", h_C[i*n + j]);\n",
        "        }\n",
        "        printf(\"\\n\");\n",
        "    }\n",
        "\n",
        "    // Clean up\n",
        "    cudaFree(d_a);\n",
        "    cudaFree(d_c);\n",
        "    free(h_A);\n",
        "    free(h_C);\n",
        "\n",
        "    return 0;\n",
        "}"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lsZpM9y66GSu",
        "outputId": "92433eba-de06-4952-9193-d644bdb9f99a"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing 2d_convolution_with_tiling.cu\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!nvcc -arch=sm_75 2d_convolution_with_tiling.cu -o 2d_convolution_with_tiling\n"
      ],
      "metadata": {
        "id": "r74vHg9N63ZN"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!./2d_convolution_with_tiling\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_CzBWTyV66hU",
        "outputId": "f753de88-d1a8-43e4-fa37-6ef1e2ffc3b2"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Results:\n",
            "180.00 225.00 225.00 225.00 225.00 225.00 225.00 225.00 180.00 135.00 \n",
            "240.00 300.00 300.00 300.00 300.00 300.00 300.00 300.00 240.00 180.00 \n",
            "300.00 375.00 375.00 375.00 375.00 375.00 375.00 375.00 300.00 225.00 \n",
            "300.00 375.00 375.00 375.00 375.00 375.00 375.00 375.00 300.00 225.00 \n",
            "300.00 375.00 375.00 375.00 375.00 375.00 375.00 375.00 300.00 225.00 \n",
            "300.00 375.00 375.00 375.00 375.00 375.00 375.00 375.00 300.00 225.00 \n",
            "300.00 375.00 375.00 375.00 375.00 375.00 375.00 375.00 300.00 225.00 \n",
            "300.00 375.00 375.00 375.00 375.00 375.00 375.00 375.00 300.00 225.00 \n",
            "240.00 300.00 300.00 300.00 300.00 300.00 300.00 300.00 240.00 180.00 \n",
            "180.00 225.00 225.00 225.00 225.00 225.00 225.00 225.00 180.00 135.00 \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile one_d_convolution.cu\n",
        "#include <iostream>\n",
        "#include <cuda_runtime.h>\n",
        "#define Mask_width 5\n",
        "__constant__ float M[Mask_width];\n",
        "__global__ void oned_convolution_kernel(const float* A,float* C,int n ){\n",
        "//without tiling\n",
        "int threadId=threadIdx.x;\n",
        "int i=blockDim.x*blockIdx.x+threadId;\n",
        "\n",
        "if (i<n){\n",
        "float result=0.0f;\n",
        "for (int k=-1*Mask_width/2;k<Mask_width/2+1;k++) {\n",
        "  printf(\"%.i\",k);\n",
        "  if (i+k>=0 && i+k<n) {\n",
        "\n",
        "  result+=A[i+k]*M[k+Mask_width/2];\n",
        "\n",
        "}}\n",
        "C[i]=result;\n",
        "\n",
        "}\n",
        "}\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "// Host function to check for CUDA errors\n",
        "void checkCudaError(const char* message) {\n",
        "    cudaError_t error = cudaGetLastError();\n",
        "    if (error != cudaSuccess) {\n",
        "        std::cerr << message << \" - CUDA Error: \" << cudaGetErrorString(error) << std::endl;\n",
        "        exit(EXIT_FAILURE);\n",
        "    }\n",
        "}\n",
        "\n",
        "\n",
        "int main(){\n",
        "\n",
        "  int n=10;\n",
        "  float A[n],C[n];\n",
        "  float d_M[Mask_width];\n",
        "\n",
        "   for (int i=0; i<Mask_width;i++){\n",
        "    d_M[i]=i;\n",
        "\n",
        "  }\n",
        "  for (int i=0; i<n;i++){\n",
        "    A[i]=i;\n",
        "\n",
        "  }\n",
        "\n",
        "  float *d_a,*d_c;\n",
        "  cudaMalloc(&d_a,n*sizeof(float));\n",
        "  cudaMalloc(&d_c,n*sizeof(float));\n",
        "  cudaMemcpy(d_a,A,n*sizeof(float),cudaMemcpyHostToDevice);\n",
        "  checkCudaError(\"Failed to copy input data to device\");\n",
        "  cudaMemcpyToSymbol(M,d_M,Mask_width*sizeof(float));\n",
        "  checkCudaError(\"Failed to copy mask data to device\");\n",
        "  dim3 dimBlock(32);\n",
        "  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);\n",
        "  oned_convolution_kernel<<<dimGrid, dimBlock>>>(d_a,d_c,n);\n",
        "  checkCudaError(\"Failed to execute the kernel\");\n",
        "  cudaDeviceSynchronize();\n",
        "  cudaMemcpy(C,d_c,n*sizeof(float),cudaMemcpyDeviceToHost);\n",
        "checkCudaError(\"Failed to copy output data to host\");\n",
        "  cudaFree(d_a);\n",
        "  cudaFree(d_c);\n",
        "\n",
        "\n",
        "  //printing the results\n",
        "  printf(\"A:\\n\");\n",
        "  for (int i=0; i<n;i++){\n",
        "    printf(\"%.2f \", A[i]);\n",
        "\n",
        "  }\n",
        "  printf(\"\\n\");\n",
        "   printf(\"\\nd_m:\\n\");\n",
        "    for (int i = 0; i < Mask_width; i++) {\n",
        "\n",
        "            printf(\"%.2f \", d_M[i]);\n",
        "\n",
        "    }\n",
        "  printf(\"\\n\");\n",
        "  printf(\"\\nC:\\n\");\n",
        "    for (int i = 0; i < n; i++) {\n",
        "\n",
        "            printf(\"%.2f \", C[i]);\n",
        "\n",
        "    }\n",
        "  printf(\"\\n\");\n",
        "}"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KOKqsO_g8LpM",
        "outputId": "9681ecd9-0fb6-42ea-c488-f5cbca9b3cc2"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing one_d_convolution.cu\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!nvcc -arch=sm_75 one_d_convolution.cu -o one_d_convolution\n"
      ],
      "metadata": {
        "id": "enFiAC989mP2"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!./one_d_convolution"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CQG2Jb8997TZ",
        "outputId": "61919cdf-6c16-45e8-bb14-88243c92abb3"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "-2-2-2-2-2-2-2-2-2-2-1-1-1-1-1-1-1-1-1-111111111112222222222A:\n",
            "0.00 1.00 2.00 3.00 4.00 5.00 6.00 7.00 8.00 9.00 \n",
            "\n",
            "d_m:\n",
            "0.00 1.00 2.00 3.00 4.00 \n",
            "\n",
            "C:\n",
            "11.00 20.00 30.00 40.00 50.00 60.00 70.00 80.00 50.00 26.00 \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile one_d_convolution_with_tiling.cu\n",
        "#include <iostream>\n",
        "#include <cuda_runtime.h>\n",
        "#define Mask_width 5\n",
        "__constant__ float M[Mask_width];\n",
        "\n",
        "__global__ void oned_convolution_tiling_kernel(const float* A, float* C, int n) {\n",
        "    int threadId = threadIdx.x;\n",
        "    int i = blockDim.x * blockIdx.x + threadId;\n",
        "\n",
        "    __shared__ float S_A[32 + Mask_width - 1];\n",
        "\n",
        "    // Load main data\n",
        "    if (i < n) {\n",
        "        S_A[threadId + Mask_width/2] = A[i];\n",
        "    }\n",
        "\n",
        "    // Load left halo\n",
        "    if (threadId < Mask_width/2) {\n",
        "        int left_idx = blockIdx.x * blockDim.x - (Mask_width/2) + threadId;\n",
        "        if (left_idx >= 0) {\n",
        "            S_A[threadId] = A[left_idx];\n",
        "        }\n",
        "        else {\n",
        "            S_A[threadId] = 0.0f;\n",
        "        }\n",
        "    }\n",
        "\n",
        "    // Load right halo\n",
        "    if (threadId < Mask_width/2) {\n",
        "        int right_idx = blockIdx.x * blockDim.x + blockDim.x + threadId;\n",
        "        if (right_idx < n) {\n",
        "            S_A[threadId + blockDim.x + Mask_width/2] = A[right_idx];\n",
        "        }\n",
        "        else {\n",
        "            S_A[threadId + blockDim.x + Mask_width/2] = 0.0f;\n",
        "        }\n",
        "    }\n",
        "\n",
        "    __syncthreads();\n",
        "\n",
        "    if (i < n) {\n",
        "        float result = 0.0f;\n",
        "        for (int k = 0; k < Mask_width; k++) {\n",
        "            int idx = threadId + k;\n",
        "            if ((i + k - Mask_width/2) >= 0 && (i + k - Mask_width/2) < n) {\n",
        "                result += S_A[idx] * M[k];\n",
        "            }\n",
        "        }\n",
        "        C[i] = result;\n",
        "    }\n",
        "}\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "// Host function to check for CUDA errors\n",
        "void checkCudaError(const char* message) {\n",
        "    cudaError_t error = cudaGetLastError();\n",
        "    if (error != cudaSuccess) {\n",
        "        std::cerr << message << \" - CUDA Error: \" << cudaGetErrorString(error) << std::endl;\n",
        "        exit(EXIT_FAILURE);\n",
        "    }\n",
        "}\n",
        "\n",
        "\n",
        "int main(){\n",
        "\n",
        "  int n=10;\n",
        "  float A[n],C[n];\n",
        "  float d_M[Mask_width];\n",
        "\n",
        "   for (int i=0; i<Mask_width;i++){\n",
        "    d_M[i]=i;\n",
        "\n",
        "  }\n",
        "  for (int i=0; i<n;i++){\n",
        "    A[i]=i;\n",
        "\n",
        "  }\n",
        "\n",
        "  float *d_a,*d_c;\n",
        "  cudaMalloc(&d_a,n*sizeof(float));\n",
        "  cudaMalloc(&d_c,n*sizeof(float));\n",
        "  cudaMemcpy(d_a,A,n*sizeof(float),cudaMemcpyHostToDevice);\n",
        "  checkCudaError(\"Failed to copy input data to device\");\n",
        "  cudaMemcpyToSymbol(M,d_M,Mask_width*sizeof(float));\n",
        "  checkCudaError(\"Failed to copy mask data to device\");\n",
        "  dim3 dimBlock(32);\n",
        "  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);\n",
        "  oned_convolution_tiling_kernel<<<dimGrid, dimBlock>>>(d_a,d_c,n);\n",
        "  checkCudaError(\"Failed to execute the kernel\");\n",
        "  cudaDeviceSynchronize();\n",
        "  cudaMemcpy(C,d_c,n*sizeof(float),cudaMemcpyDeviceToHost);\n",
        "checkCudaError(\"Failed to copy output data to host\");\n",
        "  cudaFree(d_a);\n",
        "  cudaFree(d_c);\n",
        "\n",
        "\n",
        "  //printing the results\n",
        "  printf(\"A:\\n\");\n",
        "  for (int i=0; i<n;i++){\n",
        "    printf(\"%.2f \", A[i]);\n",
        "\n",
        "  }\n",
        "  printf(\"\\n\");\n",
        "   printf(\"\\nd_m:\\n\");\n",
        "    for (int i = 0; i < Mask_width; i++) {\n",
        "\n",
        "            printf(\"%.2f \", d_M[i]);\n",
        "\n",
        "    }\n",
        "  printf(\"\\n\");\n",
        "  printf(\"\\nC:\\n\");\n",
        "    for (int i = 0; i < n; i++) {\n",
        "\n",
        "            printf(\"%.2f \", C[i]);\n",
        "\n",
        "    }\n",
        "  printf(\"\\n\");\n",
        "}"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "w6xr0RNy-Bf5",
        "outputId": "19339277-5f9c-4a8a-cc08-a7809d88e41b"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing one_d_convolution_with_tiling.cu\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!nvcc -arch=sm_75 one_d_convolution_with_tiling.cu -o one_d_convolution_with_tiling\n"
      ],
      "metadata": {
        "id": "5eaW428H-P4V"
      },
      "execution_count": 21,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!./one_d_convolution_with_tiling"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iaZFH3Hr-bSb",
        "outputId": "3f3dedb6-36f7-44c3-93e8-b58bc50a8f01"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "A:\n",
            "0.00 1.00 2.00 3.00 4.00 5.00 6.00 7.00 8.00 9.00 \n",
            "\n",
            "d_m:\n",
            "0.00 1.00 2.00 3.00 4.00 \n",
            "\n",
            "C:\n",
            "11.00 20.00 30.00 40.00 50.00 60.00 70.00 80.00 50.00 26.00 \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "3Xmy6Oxd-icp"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}