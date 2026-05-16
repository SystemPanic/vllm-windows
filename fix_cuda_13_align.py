import os
from packaging.version import parse
import subprocess
tensor_map_struct = "typedef struct CUtensorMap_st {"
tensor_map_align_definition = f"""
#if defined(_MSC_VER)
  #define TENSOR_MAP_ALIGN 64
#else
  #define TENSOR_MAP_ALIGN 128
#endif

{tensor_map_struct}"""

def get_nvcc_cuda_version(cuda_home):
    nvcc_output = subprocess.check_output(
        [cuda_home + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = str(parse(output[release_idx].split(",")[0]))
    return nvcc_cuda_version

cuda_path = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", os.environ.get("CUDA_ROOT", None)))
if cuda_path is None:
    raise ValueError("No CUDA_HOME, CUDA_PATH or CUDA_ROOT is found on environment")
cuda_version = get_nvcc_cuda_version(cuda_path)
cuda_major, cuda_minor = str(cuda_version).split(".")[:2]
#CUDA 13.0 to 13.2 has a bug with MSVC when passing over-aligned types (like alignas(128)) by value as function parameters. Nvidia acknowledged it and will be fixed on CUDA 13.3, fix if lower.
if int(cuda_major) == 13 and int(cuda_minor) < 3:
    cuda_h_file = os.path.join(cuda_path, "include", "cuda.h")

    if os.path.exists(cuda_h_file):
        with open(cuda_h_file, mode="r", encoding="utf-8") as file:
            header_content = "".join(file.readlines())
        if "alignas(128)" in header_content and "_Alignas(128)" in header_content and "TENSOR_MAP_ALIGN" not in header_content:
            header_content = header_content.replace(tensor_map_struct, tensor_map_align_definition).replace("alignas(128)", "alignas(TENSOR_MAP_ALIGN)").replace("_Alignas(128)", "_Alignas(TENSOR_MAP_ALIGN)")
            with open(cuda_h_file, mode="w", encoding="utf-8") as file:
                file.write(header_content)