set "CMAKE_ARGS=-DLLAMA_CUBLAS=on CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\nvcc.exe -DCUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\ -DCUDAToolkit_INCLUDE_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\include -DCUDAToolkit_LIBRARY_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\lib -DCMAKE_CUDA_ARCHITECTURES=native"
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose


