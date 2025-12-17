@echo off
echo ====================================================
echo        NVIDIA ACO CUDA Profiling Suite
echo ====================================================
echo.

echo ====================================================
echo 1. Compiling CUDA program with optimizations
echo ====================================================
echo Compiling with optimizations...
nvcc -O3 -arch=sm_86 -std=c++17 -o rtx3060.exe aco_cuda_global_while_Nsight.cu ^
     -Xcompiler "/O2 /fp:fast /openmp /MD" ^
     -use_fast_math ^
     -lcudart ^
     -maxrregcount=32 ^
     -Xptxas "-O3,-v" ^
     --keep-device-functions ^
     --generate-line-info

if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed!
    pause
    exit /b 1
)
echo [OK] Compilation successful - rtx3060.exe created
echo.

echo ====================================================
echo 2. Basic profiling with NSYS
echo ====================================================
echo Running NSYS system-level profiling...
echo.

nsys profile ^
    --trace=cuda ^
    --cuda-memory-usage=true ^
    --stats=true ^
    --force-overwrite=true ^
    --output=nsys_profile ^
    rtx3060.exe

echo [OK] NSYS profiling completed
echo.

echo ====================================================
echo 1. Compiling CUDA program with optimizations
echo ====================================================
echo Compiling with optimizations...
nvcc -O3 -arch=sm_86 -std=c++17 -o rtx3060_kernel.exe kernel_Nsight.cu ^
     -Xcompiler "/O2 /fp:fast /openmp /MD" ^
     -use_fast_math ^
     -lcudart ^
     -maxrregcount=32 ^
     -Xptxas "-O3,-v" ^
     --keep-device-functions ^
     --generate-line-info

if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed!
    pause
    exit /b 1
)
echo [OK] Compilation successful - rtx3060_kernel.exe created
echo.

echo ====================================================
echo 2. Basic profiling with NSYS
echo ====================================================
echo Running NSYS system-level profiling...
echo.

nsys profile ^
    --trace=cuda ^
    --cuda-memory-usage=true ^
    --stats=true ^
    --force-overwrite=true ^
    --output=nsys_profile ^
    rtx3060_kernel.exe

echo [OK] NSYS profiling completed
echo.

pause