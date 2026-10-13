# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build DeepGEMM's `_C` pybind11 extension for <TARGET_PY>.

Driven from cmake/external_projects/deepgemm.cmake. The driver runs against
the build interpreter's torch; <TARGET_PY> is only consulted for INCLUDEPY
and SOABI, so target venvs don't need torch installed.

Usage: python build_deepgemm_C.py <DEEPGEMM_SRC_DIR> <OUTPUT_DIR> <TARGET_PY>
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils import cpp_extension

if len(sys.argv) != 4:
    sys.exit(f"usage: {sys.argv[0]} <SRC> <OUT> <TARGET_PY>")

src = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
target_py = sys.argv[3]
out.mkdir(parents=True, exist_ok=True)

IS_WINDOWS = sys.platform == "win32"

info = json.loads(
    subprocess.check_output(
        [
            target_py,
            "-c",
            "import sysconfig, json; "
            "print(json.dumps({k: sysconfig.get_config_var(k) "
            "for k in ('EXT_SUFFIX', 'INCLUDEPY')}))",
        ]
    ).decode()
)

cuda_home = cpp_extension.CUDA_HOME
if cuda_home is None:
    sys.exit("CUDA_HOME not found; cannot build DeepGEMM _C")
# CCCL lives outside the standard CUDAToolkit search (mirrors DeepGEMM's setup.py).
includes = [
    info["INCLUDEPY"],
    f"{cuda_home}/include",
    f"{cuda_home}/include/cccl",
    str(src / "csrc"),
    str(src / "deep_gemm/include"),
    str(src / "third-party/cutlass/include"),
    str(src / "third-party/cutlass/tools/util/include"),
    str(src / "third-party/deep_jit/include"),
    *cpp_extension.include_paths(device_type="cuda"),
]

if IS_WINDOWS:
    cmd = [
        os.environ.get("CXX", "cl.exe"),
        "/nologo",
        "/MD",
        "/EHsc",
        "/O2",
        "/std:c++20",
        "/bigobj",
        "/DTORCH_API_INCLUDE_EXTENSION_H",
        "/DTORCH_EXTENSION_NAME=_C",
        "/DNOMINMAX",
        "/DWIN32_LEAN_AND_MEAN",
        "/D_CRT_SECURE_NO_WARNINGS",
        "/utf-8",
        "/Zc:preprocessor",
        *(f"/I{p}" for p in includes),
        str(src / "csrc/python_api.cpp"),
        "/LD",
        f"/Fe:{str(out / f"_C{info['EXT_SUFFIX']}")}",
        f"/Fo:{out}{os.sep}",
        "/link",
        *(f"/LIBPATH:{p}" for p in cpp_extension.library_paths(device_type="cuda")),
        f"/LIBPATH:{str(Path(sys.base_prefix) / "libs")}",
        f"torch.lib",
        f"torch_python.lib",
        f"torch_cpu.lib",
        f"torch_cuda.lib",
        f"c10.lib",
        f"c10_cuda.lib",
        f"cudart.lib",
        f"nvrtc.lib",
        f"cublas.lib",
        f"cublasLt.lib"
    ]
else:
    cmd = [
        os.environ.get("CXX", "g++"),
        "-shared",
        "-fPIC",
        "-std=c++20",
        "-O3",
        "-g0",
        "-Wno-psabi",
        "-Wno-deprecated-declarations",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DTORCH_EXTENSION_NAME=_C",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
        *(f"-I{p}" for p in includes),
        str(src / "csrc/python_api.cpp"),
        *(f"-L{p}" for p in cpp_extension.library_paths(device_type="cuda")),
        f"-L{cuda_home}/lib64",
        "-ltorch",
        "-ltorch_python",
        "-ltorch_cpu",
        "-ltorch_cuda",
        "-lc10",
        "-lc10_cuda",
        "-lcudart",
        "-lnvrtc",
        "-o",
        str(out / f"_C{info['EXT_SUFFIX']}"),
    ]
print("[build_deepgemm_C] " + " ".join(cmd), flush=True)
subprocess.check_call(cmd)