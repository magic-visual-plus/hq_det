// Copyright (c) OpenMMLab. All rights reserved
// -----------------------------------------------------------------------------
// Python binding for asymmetric sigmoid focal loss.
//
// Directly mirrors the mmcv binding
//   mmcv/ops/csrc/pytorch/focal_loss.cpp  +  pybind.cpp
// The ONLY intentional modifications are:
//   1. Rename identifiers to the "asymmetric_*" namespace.
//   2. Replace the single `gamma` argument with `gamma_pos` and `gamma_neg`.
// Signatures, argument order, types (`float`) and pybind.arg names for the
// shared arguments are preserved byte-for-byte.
// -----------------------------------------------------------------------------
#include <torch/extension.h>

#include <ATen/ATen.h>

using at::Tensor;

void AsymmetricSigmoidFocalLossForwardCUDAKernelLauncher(
    Tensor input, Tensor target, Tensor weight, Tensor output,
    float gamma_pos, float gamma_neg, float alpha);

void AsymmetricSigmoidFocalLossBackwardCUDAKernelLauncher(
    Tensor input, Tensor target, Tensor weight, Tensor grad_input,
    float gamma_pos, float gamma_neg, float alpha);

// in-place: writes into `output` (mirrors mmcv::sigmoid_focal_loss_forward)
void asymmetric_sigmoid_focal_loss_forward(Tensor input, Tensor target,
                                           Tensor weight, Tensor output,
                                           float gamma_pos, float gamma_neg,
                                           float alpha) {
  TORCH_CHECK(input.device().is_cuda(),
              "asymmetric_sigmoid_focal_loss_forward only supports CUDA");
  AsymmetricSigmoidFocalLossForwardCUDAKernelLauncher(
      input, target, weight, output, gamma_pos, gamma_neg, alpha);
}

// in-place: writes into `grad_input` (mirrors mmcv::sigmoid_focal_loss_backward)
void asymmetric_sigmoid_focal_loss_backward(Tensor input, Tensor target,
                                            Tensor weight, Tensor grad_input,
                                            float gamma_pos, float gamma_neg,
                                            float alpha) {
  TORCH_CHECK(input.device().is_cuda(),
              "asymmetric_sigmoid_focal_loss_backward only supports CUDA");
  AsymmetricSigmoidFocalLossBackwardCUDAKernelLauncher(
      input, target, weight, grad_input, gamma_pos, gamma_neg, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("asymmetric_sigmoid_focal_loss_forward",
        &asymmetric_sigmoid_focal_loss_forward,
        "asymmetric_sigmoid_focal_loss_forward", py::arg("input"),
        py::arg("target"), py::arg("weight"), py::arg("output"),
        py::arg("gamma_pos"), py::arg("gamma_neg"), py::arg("alpha"));
  m.def("asymmetric_sigmoid_focal_loss_backward",
        &asymmetric_sigmoid_focal_loss_backward,
        "asymmetric_sigmoid_focal_loss_backward", py::arg("input"),
        py::arg("target"), py::arg("weight"), py::arg("grad_input"),
        py::arg("gamma_pos"), py::arg("gamma_neg"), py::arg("alpha"));
}
