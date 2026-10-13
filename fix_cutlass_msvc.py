import os
import sys

cutlass_base_dir = sys.argv[1]

platform_h_file = os.path.join(cutlass_base_dir, "include", "cutlass", "platform", "platform.h")

if os.path.exists(platform_h_file):
    with open(platform_h_file, mode="r", encoding="utf-8") as file:
        header_content = "".join(file.readlines())

    if "\n#if (201703L <=__cplusplus)\n" in header_content:
        header_content = header_content.replace("#if (201703L <=__cplusplus)", "#if defined(_MSC_VER) || (201703L <=__cplusplus)")
        with open(platform_h_file, mode="w", encoding="utf-8") as file:
            file.write(header_content)

cuda_host_adapter_file = os.path.join(cutlass_base_dir, "include", "cutlass", "cuda_host_adapter.hpp")

if os.path.exists(cuda_host_adapter_file):
    with open(cuda_host_adapter_file, mode="r", encoding="utf-8") as file:
        header_content = "".join(file.readlines())

    if "CUTLASS_HOST_DEVICE\n  Status memsetDevice" in header_content:
        header_content = header_content.replace("CUTLASS_HOST_DEVICE\n  Status memsetDevice", "CUTLASS_HOST\n  Status memsetDevice")
        with open(cuda_host_adapter_file, mode="w", encoding="utf-8") as file:
            file.write(header_content)