### RTX Spark install / build instructions

A Visual Studio 2026 or newer is required to launch the compiler arm64 environment. The installation path is referred in the instructions as VISUAL_STUDIO_INSTALL_PATH.

**Make sure to have clang compiler installed on Visual Studio 2026**

CUDA path will be found automatically if you have the bin folder in your PATH, or have the CUDA installation path settled on well-known environment vars like CUDA_ROOT, CUDA_HOME or CUDA_PATH.

If none of these are present, make sure to set the environment variable before starting the build:
``set CUDA_ROOT=CUDA_INSTALLATION_PATH``

Some RTX Spark vLLM dependencies are tricky to install due to the recent release of the platform. Follow the steps to correctly install them the first time before using or building vLLM:

1. Open a Command Line (cmd.exe) and execute ```VISUAL_STUDIO_INSTALL_PATH\VC\Auxiliary\Build\vcvarsall.bat arm64```
2. Install python 3.14, then basic build dependencies: ``pip install ninja cmake``
3. Torch does not have official wheels for cuda windows arm64 yet, use nvidia repo: 
``pip install torch==2.14.0+cu134 torchvision==0.29.0+cu134 torchaudio==2.11.0+cu134 --extra-index-url https://pypi.nvidia.com/nvtorch_oot/``
4. After torch install, execute this: ``python -c "import torch.utils.cpp_extension as c; print(c.__file__)"``
5. open the printed cpp_extension.py path and replace all 'x64' entries with 'arm64', then save cpp_extension.py changes
6. Download and install rust at default location: https://static.rust-lang.org/rustup/dist/aarch64-pc-windows-msvc/rustup-init.exe
7. Add rust bin folder to Windows PATH env var with:
```
setx PATH %PATH%;%USERPROFILE%\.cargo\bin
set PATH=%PATH%;%USERPROFILE%\.cargo\bin
```
8. Download https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-winarm64-gpl-shared-8.1.zip and unzip it to C:\ffmpeg-8.1
9. Download [FFmpegBtbNConfig.cmake](https://github.com/SystemPanic/vllm-windows/releases/download/v0.32.0_spark/FFmpegBtbNConfig.cmake) to C:\ffmpeg-8.1\FFmpegBtbNConfig.cmake
10. Add C:\ffmpeg-8.1\bin to Windows PATH env var with:
```
setx PATH %PATH%;C:\ffmpeg-8.1\bin
set PATH=%PATH%;C:\ffmpeg-8.1\bin
```
11. Check that ``rustc --version`` and ``ffmpeg -version`` have executed correctly
12. xformers installation:
```set TORCH_CUDA_ARCH_LIST=12.1
set MAX_JOBS=18
set CMAKE_GENERATOR=Ninja
pip install --no-build-isolation -v xformers
```
13. cryptography installation:
```
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg install openssl:arm64-windows-static-md
setx OPENSSL_DIR C:\vcpkg\installed\arm64-windows-static-md
set OPENSSL_DIR=C:\vcpkg\installed\arm64-windows-static-md
pip install cryptography
```
14. outlines_core installation:
```
mkdir C:\b
set TEMP=C:\b
set TMP=C:\b
set CARGO_TARGET_DIR=C:\b\t
set AWS_LC_SYS_CMAKE_GENERATOR=Ninja
set CC=clang-cl
set CXX=clang-cl
pip install outlines_core --no-cache-dir --no-build-isolation
set CARGO_TARGET_DIR=
set TMP=
set TEMP=
set CC=
set CXX=
```
15. opencv-python-headless installation:
```
set CMAKE_ARGS=-DOPENCV_BIN_INSTALL_PATH=bin -DBUILD_WITH_STATIC_CRT=OFF -DWITH_FFMPEG=ON -DOPENCV_FFMPEG_SKIP_DOWNLOAD=ON -DVIDEOIO_PLUGIN_LIST=ffmpeg -DOPENCV_FFMPEG_USE_FIND_PACKAGE=FFmpegBtbN -DFFmpegBtbN_DIR=C:/ffmpeg-8.1
pip install opencv-python-headless==4.14.0.94
# Fix opencv dll name mismatch on arm64 (always do this after installing or reinstalling opencv-python-headless):
for /f "delims=" %i in ('python -c "import cv2,os;print(os.path.dirname(cv2.__file__))"') do copy /y "%i\opencv_videoio_ffmpeg4140_64.dll" "%i\opencv_videoio_ffmpeg4140.dll"
```

Now you are ready to install vLLM or build it.

- Install vLLM:
1. Ensure that you have the correct Python, Torch and CUDA version of the wheel. The Python, Torch and CUDA version of the wheel is specified in the release version.
2. Download the wheel from the release version of your preference (latest wheel [here](https://github.com/SystemPanic/vllm-windows/releases/latest)).
3. Install it with ``pip install DOWNLOADED_WHEEL_PATH``

- Build vLLM from source:
```
# Clone the vLLM for Windows repository from vllm-for-windows branch (NOT MAIN): 
cd C:\ & git clone --single-branch --branch vllm-for-windows https://github.com/SystemPanic/vllm-windows.git & cd C:\vllm-windows
pip install -r requirements\build\cuda.txt --no-build-isolation
pip install -r requirements\cuda.txt --no-build-isolation

# Make sure to have cuda.h patched for Windows 128 alignment limit (run python fix_cuda_align.py on Administrator console or patch it manually)

set VLLM_TARGET_DEVICE=cuda
set TORCH_CUDA_ARCH_LIST=12.1
set MAX_JOBS=10
set NVCC_THREADS=1
set USE_CUDNN=1
set SETUPTOOLS_SCM_PRETEND_VERSION=0.33.0+spark
python setup.py bdist_wheel

cd dist & pip install vllm-0.33.0+spark.cu134-cp314-cp314-win_arm64.whl
```