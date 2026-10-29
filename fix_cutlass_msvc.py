import os
import sys

cutlass_base_dir = sys.argv[1]

platform_h_file = os.path.join(cutlass_base_dir, "include", "cutlass", "platform", "platform.h")

if os.path.exists(platform_h_file):
    with open(platform_h_file, mode="r", encoding="utf-8") as file:
        content = "".join(file.readlines())

    if "\n#if (201703L <=__cplusplus)\n" in content:
        content = content.replace("#if (201703L <=__cplusplus)", "#if defined(_MSC_VER) || (201703L <=__cplusplus)")
        with open(platform_h_file, mode="w", encoding="utf-8") as file:
            file.write(content)

cuda_host_adapter_file = os.path.join(cutlass_base_dir, "include", "cutlass", "cuda_host_adapter.hpp")

if os.path.exists(cuda_host_adapter_file):
    with open(cuda_host_adapter_file, mode="r", encoding="utf-8") as file:
        content = "".join(file.readlines())

    if "CUTLASS_HOST_DEVICE\n  Status memsetDevice" in content:
        content = content.replace("CUTLASS_HOST_DEVICE\n  Status memsetDevice", "CUTLASS_HOST\n  Status memsetDevice")
        with open(cuda_host_adapter_file, mode="w", encoding="utf-8") as file:
            file.write(content)

cuda_tensor_fill_arm64 = os.path.join(cutlass_base_dir, "tools", "util", "include", "cutlass", "util", "reference", "host", "tensor_fill.h")

if os.path.exists(cuda_tensor_fill_arm64):
    with open(cuda_tensor_fill_arm64, mode="r", encoding="utf-8") as file:
        content = "".join(file.readlines())

    if "return Element(NAN);" in content:
        content = content.replace("return Element(NAN);", "return Element(std::numeric_limits<float>::quiet_NaN());")
        with open(cuda_tensor_fill_arm64, mode="w", encoding="utf-8") as file:
            file.write(content)