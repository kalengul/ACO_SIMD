@echo off
setlocal enabledelayedexpansion

echo === CUDA + OpenMP Hybrid Build (Windows/g++ MinGW) ===

:: Настройки
set CUDA_ARCH=sm_68
set CXX_FLAGS=-O3 -fopenmp -std=c++17 -march=native

:: Проверка компиляторов
echo Checking compilers...

where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: nvcc not found in PATH!
    echo Please install CUDA Toolkit 11.0+ and add to PATH
    echo Download from: https://developer.nvidia.com/cuda-downloads
    pause
    exit /b 1
)

where g++ >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: g++ not found in PATH!
    echo Please install MinGW-w64 with OpenMP support
    echo Download from: https://www.mingw-w64.org/
    pause
    exit /b 1
)

:: Показать версии компиляторов
echo Found compilers:
nvcc --version | findstr "release"
g++ --version | findstr "g++"

:: Очистка предыдущих сборок
echo.
echo Cleaning previous builds...
del *.o *.obj *.exe *.dll *.a *.lib *.exp 2>nul

:: Компиляция CUDA DLL
echo.
echo Step 1: Compiling CUDA DLL...
echo Command: nvcc -O3 -D BUILD_CUDA_DLL --shared -o cuda_module.dll cuda_module.cu
nvcc -O3 -D BUILD_CUDA_DLL --shared -o cuda_module.dll cuda_module.cu
if %errorlevel% neq 0 (
    echo.
    echo ERROR: CUDA compilation failed!
    echo Check your CUDA installation and GPU architecture
    pause
    exit /b %errorlevel%
)

:: Компиляция основного приложения с g++
echo.
echo Step 2: Compiling main application with g++...
echo Command: g++ %CXX_FLAGS% -o test_hybrid.exe main_omp.cpp -L. -lcuda_module
g++ %CXX_FLAGS% -o test_hybrid.exe main_omp.cpp -L. -lcuda_module
if %errorlevel% neq 0 (
    echo.
    echo ERROR: g++ compilation failed!
    echo Check your MinGW installation and OpenMP support
    pause
    exit /b %errorlevel%
)

:: Проверка результатов сборки
echo.
echo Verifying build outputs...

set BUILD_OK=1

if not exist cuda_module.dll (
    echo ERROR: cuda_module.dll not created!
    set BUILD_OK=0
) else (
    echo ✓ cuda_module.dll - OK (%~z0 bytes)
)

if not exist test_hybrid.exe (
    echo ERROR: test_hybrid.exe not created!
    set BUILD_OK=0
) else (
    echo ✓ test_hybrid.exe - OK (%~z0 bytes)
)

if !BUILD_OK! equ 0 (
    echo.
    echo BUILD FAILED!
    pause
    exit /b 1
)

echo.
echo === BUILD SUCCESSFUL ===
echo.
echo Generated files:
dir cuda_module.dll test_hybrid.exe | findstr "cuda_module.dll test_hybrid.exe"

:: Проверка зависимостей
echo.
echo Checking dependencies...
where nvoglv64.dll >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ CUDA runtime - OK
) else (
    echo ⚠ CUDA runtime may not be in PATH
)

:: Запуск приложения
echo.
echo ================================
echo Starting application test...
echo ================================
echo.
test_hybrid.exe

set EXIT_CODE=%errorlevel%

echo.
echo ================================
echo Application finished with exit code: %EXIT_CODE%
echo ================================

pause