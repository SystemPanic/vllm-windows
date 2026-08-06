#pragma once

// clang-format will break include orders
// clang-format off
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>

#include "libtorch_stable/torch_utils.h"

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "libtorch_stable/core/math.hpp"
#include "libtorch_stable/cutlass_extensions/common.hpp"
// clang-format on

namespace vllm::c3x {

#if defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
template <typename GemmKernel>
__global__ __launch_bounds__(
    GemmKernel::MaxThreadsPerBlock,
    GemmKernel::
        MinBlocksPerMultiprocessor) void cutlass_kernel_indirect(typename GemmKernel::
                                                                     Params const*
                                                                         params) {
  extern __shared__ char smem[];
  GemmKernel op;
  op(*params, smem);
  cutlass::arch::synclog_print();
}
#endif

static inline cute::Shape<int, int, int, int> get_problem_shape(
    torch::stable::Tensor const& a, torch::stable::Tensor const& b) {
  int32_t m = a.size(0), n = b.size(1), k = a.size(1);
  return {m, n, k, 1};
}

template <typename GemmKernel>
void cutlass_gemm_caller(
    torch::stable::Device device, cute::Shape<int, int, int, int> prob_shape,
    typename GemmKernel::MainloopArguments mainloop_args,
    typename GemmKernel::EpilogueArguments epilogue_args,
    typename GemmKernel::TileSchedulerArguments scheduler = {}) {
  const torch::stable::accelerator::DeviceGuard device_guard(device.index());
  cutlass::KernelHardwareInfo hw_info;
  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                      prob_shape,
                                      mainloop_args,
                                      epilogue_args,
                                      hw_info,
                                      scheduler};

  // Launch the CUTLASS GEMM kernel.
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  CUTLASS_CHECK(GemmOp::can_implement(args));

#if defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
  using Params = typename GemmKernel::Params;
  constexpr size_t params_alignment = alignof(Params);
  constexpr size_t params_storage_size =
      (sizeof(Params) + params_alignment - 1) / params_alignment *
      params_alignment;
  size_t workspace_size = GemmOp::get_workspace_size(args);
  auto workspace = torch::stable::empty(params_storage_size + workspace_size,
                                        torch::headeronly::ScalarType::Byte,
                                        std::nullopt, device);
  auto* params_device = static_cast<uint8_t*>(workspace.data_ptr());
  STD_TORCH_CHECK(
      reinterpret_cast<uintptr_t>(params_device) % params_alignment == 0,
      "CUTLASS workspace does not satisfy Params alignment");
  void* gemm_workspace = params_device + params_storage_size;

  auto stream = get_current_cuda_stream(device.index());
  CUTLASS_CHECK(GemmKernel::initialize_workspace(args, gemm_workspace, stream));
  Params params = GemmKernel::to_underlying_arguments(args, gemm_workspace);

  auto cuda_status = cudaMemcpyAsync(params_device, &params, sizeof(Params),
                                     cudaMemcpyHostToDevice, stream);
  STD_TORCH_CHECK(cuda_status == cudaSuccess, "Failed to copy CUTLASS Params: ",
                  cudaGetErrorString(cuda_status));

  constexpr int smem_size = GemmKernel::SharedStorageSize;
  if constexpr (smem_size >= (48 << 10)) {
    cuda_status = cudaFuncSetAttribute(
        cutlass_kernel_indirect<GemmKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    STD_TORCH_CHECK(cuda_status == cudaSuccess,
                    "Failed to configure CUTLASS dynamic shared memory: ",
                    cudaGetErrorString(cuda_status));
  }

  auto grid = GemmKernel::get_grid_shape(params);
  auto block = GemmKernel::get_block_shape();
  cutlass_kernel_indirect<GemmKernel><<<grid, block, smem_size, stream>>>(
      reinterpret_cast<Params const*>(params_device));
  cuda_status = cudaGetLastError();
  STD_TORCH_CHECK(cuda_status == cudaSuccess,
                  "Failed to launch CUTLASS scaled GEMM: ",
                  cudaGetErrorString(cuda_status));
#else
  GemmOp gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto workspace =
      torch::stable::empty(workspace_size, torch::headeronly::ScalarType::Byte,
                           std::nullopt, device);

  cutlass::Status status = gemm_op.run(args, workspace.data_ptr(), stream);
  CUTLASS_CHECK(status);
#endif
}

template <typename Gemm, typename... EpilogueArgs>
void cutlass_gemm_caller(torch::stable::Tensor& out,
                         torch::stable::Tensor const& a,
                         torch::stable::Tensor const& b,
                         EpilogueArgs&&... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementC = typename Gemm::ElementC;
  using ElementD = typename Gemm::ElementD;
  using GemmKernel = typename Gemm::GemmKernel;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = StrideC;
  using StrideAux = StrideC;

  typename GemmKernel::ProblemShape prob_shape = get_problem_shape(a, b);
  auto [M, N, K, L] = prob_shape;

  StrideA a_stride =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
  StrideB b_stride =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
  StrideC c_stride =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
  StrideD d_stride =
      cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));
  StrideAux aux_stride = d_stride;

  auto a_ptr = static_cast<ElementAB*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB*>(b.data_ptr());
  typename GemmKernel::MainloopArguments mainloop_args{a_ptr, a_stride, b_ptr,
                                                       b_stride};

  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  // auto d_ptr = static_cast<ElementC*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(
          std::forward<EpilogueArgs>(epilogue_params)...),
      c_ptr, c_stride, c_ptr, d_stride};

  cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args,
                                  epilogue_args);
}

}  // namespace vllm::c3x