// Copyright (c) OpenMMLab. All rights reserved
// -----------------------------------------------------------------------------
// Asymmetric Sigmoid Focal Loss CUDA kernel.
//
// This file is a byte-for-byte copy of
//   mmcv/ops/csrc/common/cuda/sigmoid_focal_loss_cuda_kernel.cuh
// The ONLY intentional modifications are:
//   1. Rename identifiers to the "asymmetric_*" namespace.
//   2. Split the single `gamma` parameter into `gamma_pos` (applied to the
//      positive term) and `gamma_neg` (applied to the negative term).
// Everything else (math, memory access, weight handling, numerical
// stabilisation with FLT_MIN, floating-point function selection, etc.) is
// preserved exactly as in the original upstream implementation.
// -----------------------------------------------------------------------------
#ifndef ASYMMETRIC_SIGMOID_FOCAL_LOSS_CUDA_KERNEL_CUH
#define ASYMMETRIC_SIGMOID_FOCAL_LOSS_CUDA_KERNEL_CUH

#include <float.h>

// ---- Begin excerpt from mmcv common_cuda_helper.hpp (verbatim) --------------
#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
  int optimal_block_num = (N + num_threads - 1) / num_threads;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}
// ---- End excerpt ------------------------------------------------------------

template <typename T>
__global__ void asymmetric_sigmoid_focal_loss_forward_cuda_kernel(
    const int nthreads, const T* input, const int64_t* target, const T* weight,
    T* output, const T gamma_pos, const T gamma_neg, const T alpha,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / num_classes;
    int c = index % num_classes;

    int64_t t = target[n];
    T flag_p = (t == c);
    T flag_n = (t != c);

    // p = sigmoid(x) = 1. / 1. + expf(-x)
    T p = (T)1. / ((T)1. + expf(-input[index]));

    // (1 - p)**gamma_pos * log(p)
    T term_p = pow(((T)1. - p), gamma_pos) * log(max(p, (T)FLT_MIN));
    // p**gamma_neg * log(1 - p)
    T term_n = pow(p, gamma_neg) * log(max((T)1. - p, (T)FLT_MIN));

    output[index] = (T)0.;
    output[index] += -flag_p * alpha * term_p;
    output[index] += -flag_n * ((T)1. - alpha) * term_n;
    if (weight != NULL) {
      output[index] *= weight[t];
    }
  }
}

template <typename T>
__global__ void asymmetric_sigmoid_focal_loss_backward_cuda_kernel(
    const int nthreads, const T* input, const int64_t* target, const T* weight,
    T* grad_input, const T gamma_pos, const T gamma_neg, const T alpha,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / num_classes;
    int c = index % num_classes;

    int64_t t = target[n];
    T flag_p = (t == c);
    T flag_n = (t != c);

    // p = sigmoid(x) = 1. / 1. + expf(-x)
    T p = (T)1. / ((T)1. + exp(-input[index]));

    // (1 - p)**gamma_pos * (1 - p - gamma_pos*p*log(p))
    T term_p = pow(((T)1. - p), gamma_pos) *
               ((T)1. - p - (gamma_pos * p * log(max(p, (T)FLT_MIN))));
    // p**gamma_neg * (gamma_neg * (1 - p) * log(1 - p) - p)
    T term_n = pow(p, gamma_neg) *
               (gamma_neg * ((T)1. - p) * log(max((T)1. - p, (T)FLT_MIN)) - p);

    grad_input[index] = (T)0.;
    grad_input[index] += -flag_p * alpha * term_p;
    grad_input[index] += -flag_n * ((T)1. - alpha) * term_n;
    if (weight != NULL) {
      grad_input[index] *= weight[t];
    }
  }
}

#endif  // ASYMMETRIC_SIGMOID_FOCAL_LOSS_CUDA_KERNEL_CUH
