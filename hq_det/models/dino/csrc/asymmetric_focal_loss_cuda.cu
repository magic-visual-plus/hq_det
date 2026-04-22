// Copyright (c) OpenMMLab. All rights reserved
// -----------------------------------------------------------------------------
// CUDA launcher for asymmetric sigmoid focal loss.
//
// This file is a byte-for-byte copy of
//   mmcv/ops/csrc/pytorch/cuda/focal_loss_cuda.cu
// (only the SigmoidFocalLoss{Forward,Backward}CUDAKernelLauncher sections).
// The ONLY intentional modifications are:
//   1. Rename identifiers to the "Asymmetric*" namespace.
//   2. Replace the single `gamma` parameter with `gamma_pos` / `gamma_neg`.
// Everything else (CUDA guard, dispatch macro, stream, grid configuration,
// assertion, pointer casts) is preserved exactly as in the original upstream
// implementation.
// -----------------------------------------------------------------------------
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "asymmetric_sigmoid_focal_loss_cuda_kernel.cuh"

using at::Tensor;

void AsymmetricSigmoidFocalLossForwardCUDAKernelLauncher(
    Tensor input, Tensor target, Tensor weight, Tensor output,
    const float gamma_pos, const float gamma_neg, const float alpha) {
  int output_size = output.numel();
  int num_classes = input.size(1);
  AT_ASSERTM(target.max().item<int64_t>() <= (int64_t)num_classes,
             "target label should smaller or equal than num classes");
  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(),
      "asymmetric_sigmoid_focal_loss_forward_cuda_kernel", [&] {
        asymmetric_sigmoid_focal_loss_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<scalar_t>(gamma_pos),
                static_cast<scalar_t>(gamma_neg),
                static_cast<scalar_t>(alpha), num_classes);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void AsymmetricSigmoidFocalLossBackwardCUDAKernelLauncher(
    Tensor input, Tensor target, Tensor weight, Tensor grad_input,
    const float gamma_pos, const float gamma_neg, const float alpha) {
  int output_size = grad_input.numel();
  int num_classes = input.size(1);

  at::cuda::CUDAGuard device_guard(grad_input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(),
      "asymmetric_sigmoid_focal_loss_backward_cuda_kernel", [&] {
        asymmetric_sigmoid_focal_loss_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(),
                static_cast<scalar_t>(gamma_pos),
                static_cast<scalar_t>(gamma_neg),
                static_cast<scalar_t>(alpha), num_classes);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}
