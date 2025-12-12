@echo off
echo Building optimized versions for performance testing...

echo.
echo Compiler versions:
g++ --version
clang++ --version

echo.
echo OpenMP versions detected:
g++ -fopenmp -dM -E - 0<nul | findstr "OPENMP"
clang++ -fopenmp -dM -E - 0<nul | findstr "OPENMP"
echo.
echo Creating results directories...
if not exist "results" mkdir results
if not exist "results\gcc" mkdir results\gcc
if not exist "results\clang" mkdir results\clang
echo.
echo ========== GCC BUILD ==========
echo 1. GCC Max Optimization
g++ -O3 -fopenmp -march=native OMP_AVX_C++.cpp -o OMP_AVX_gcc.exe

echo.
echo 2. GCC Balanced:


echo.
echo 3. GCC without advanced OMP features:


echo.
echo ========== CLANG BUILD ==========

echo.
echo 4. Clang OpenMP 3.1:
clang++ -fopenmp -fopenmp-version=31 -mavx -O3 -march=native OMP_AVX_C++.cpp -o OMP_AVX_clang_31.exe

echo.
echo 5. Clang OpenMP 4.0:
clang++ -fopenmp -fopenmp-version=40 -mavx -O3 -march=native OMP_AVX_C++.cpp -o OMP_AVX_clang_40.exe

echo.
echo 6. Clang OpenMP 4.5:
clang++ -fopenmp=libomp -fopenmp-version=45 -mavx -O3 -march=native OMP_AVX_C++.cpp -o OMP_AVX_clang_45.exe

echo.
echo 7. Clang OpenMP 5.0:
clang++ -fopenmp -fopenmp-version=50 -mavx -O3 -march=native OMP_AVX_C++.cpp -o OMP_AVX_clang_50.exe

echo.
echo 8. Clang OpenMP 5.1:
clang++ -fopenmp -fopenmp-version=51 -mavx -O3 -march=native OMP_AVX_C++.cpp -o OMP_AVX_clang_51.exe

echo.
echo 9. Clang OpenMP 5.2:
clang++ -fopenmp -fopenmp-version=52 -mavx -O3 -march=native OMP_AVX_C++.cpp -o OMP_AVX_clang_52.exe

echo.
echo 10. Clang OpenMP 6.0:
clang++ -fopenmp -fopenmp-version=60 -mavx -O3 -march=native OMP_AVX_C++.cpp -o OMP_AVX_clang_60.exe

echo.
echo ========== PERFORMANCE TEST ==========
echo Running performance tests...

if exist OMP_AVX_clang_31.exe (
    echo.
    echo --- Clang OpenMP 3.1 ---
    OMP_AVX_clang_31.exe
    if not exist "results\clang\omp31" mkdir "results\clang\omp31"
    if exist "log.txt" copy "log.txt" "results\clang\omp31\log_clang_omp31.txt" >nul
    if exist "statistics.txt" copy "statistics.txt" "results\clang\omp31\statistics_clang_omp31.txt" >nul
    echo   Results saved to results\clang\omp31\
)

if exist  OMP_AVX_gcc.exe (
    echo.
    echo --- gcc OpenMP MAX ---
    
    if not exist "results\gcc\omp40" mkdir "results\gcc\omp40"
    if exist "log.txt" copy "log.txt" "results\gcc\omp40\log_gcc_omp40.txt" >nul
    if exist "statistics.txt" copy "statistics.txt" "results\gcc\omp40\statistics_gcc_omp40.txt" >nul
    echo   Results saved to results\gcc\omp40\
)

if exist OMP_AVX_clang_40.exe (
    echo.
    echo --- Clang OpenMP 4.0 ---
    OMP_AVX_clang_40.exe
    if not exist "results\clang\omp40" mkdir "results\clang\omp40"
    if exist "log.txt" copy "log.txt" "results\clang\omp40\log_clang_omp40.txt" >nul
    if exist "statistics.txt" copy "statistics.txt" "results\clang\omp40\statistics_clang_omp40.txt" >nul
    echo   Results saved to results\clang\omp40\
)

if exist OMP_AVX_clang_45.exe (
    echo.
    echo --- Clang OpenMP 4.5 ---
    OMP_AVX_clang_45.exe
    if not exist "results\clang\omp45" mkdir "results\clang\omp45"
    if exist "log.txt" copy "log.txt" "results\clang\omp45\log_clang_omp45.txt" >nul
    if exist "statistics.txt" copy "statistics.txt" "results\clang\omp45\statistics_clang_omp45.txt" >nul
    echo   Results saved to results\clang\omp45\
)

if exist OMP_AVX_clang_50.exe (
    echo.
    echo --- Clang OpenMP 5.0 ---
    OMP_AVX_clang_50.exe
    if not exist "results\clang\omp50" mkdir "results\clang\omp50"
    if exist "log.txt" copy "log.txt" "results\clang\omp50\log_clang_omp50.txt" >nul
    if exist "statistics.txt" copy "statistics.txt" "results\clang\omp50\statistics_clang_omp50.txt" >nul
    echo   Results saved to results\clang\omp50\
)

if exist OMP_AVX_clang_51.exe (
    echo.
    echo --- Clang OpenMP 5.1 ---
    OMP_AVX_clang_51.exe
    if not exist "results\clang\omp51" mkdir "results\clang\omp51"
    if exist "log.txt" copy "log.txt" "results\clang\omp51\log_clang_omp51.txt" >nul
    if exist "statistics.txt" copy "statistics.txt" "results\clang\omp51\statistics_clang_omp51.txt" >nul
    echo   Results saved to results\clang\omp51\
)

if exist OMP_AVX_clang_52.exe (
    echo.
    echo --- Clang OpenMP 5.2 ---
    OMP_AVX_clang_52.exe
    if not exist "results\clang\omp52" mkdir "results\clang\omp52"
    if exist "log.txt" copy "log.txt" "results\clang\omp52\log_clang_omp52.txt" >nul
    if exist "statistics.txt" copy "statistics.txt" "results\clang\omp52\statistics_clang_omp52.txt" >nul
    echo   Results saved to results\clang\omp52\
)

if exist OMP_AVX_clang_60.exe (
    echo.
    echo --- Clang OpenMP 6.0 ---
    OMP_AVX_clang_60.exe
    if not exist "results\clang\omp60" mkdir "results\clang\omp60"
    if exist "log.txt" copy "log.txt" "results\clang\omp60\log_clang_omp60.txt" >nul
    if exist "statistics.txt" copy "statistics.txt" "results\clang\omp60\statistics_clang_omp60.txt" >nul
    echo   Results saved to results\clang\omp60\
)
pause