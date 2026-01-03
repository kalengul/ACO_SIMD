@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo === CUDA + OpenMP Hybrid Build (Windows/Clang) ===

set EXE_NAME=test_hybrid.exe
set PARAM_UPDATER=update_params.exe
set OUTPUT_DIR=results
set LOG_DIR=%OUTPUT_DIR%\compilation_logs
set SOURCE_DIR=.

:: Установка путей к исходным файлам
set MAIN_OMP_FILE=%SOURCE_DIR%\main_omp.cpp
set CUDA_MODULE_FILE=%SOURCE_DIR%\cuda_module.cu

set REGISTER_COUNTS=32
set PARAM_SIZES=42 84 168 336 672 1344 2688 5376 10752 21504 43008 86016 172032 344064 688128 1376256

:: Создание директорий
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:: Проверка компиляторов
echo Checking compilers...
where nvcc >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvcc not found in PATH!
    echo Please install CUDA Toolkit or add to PATH
    pause
    exit /b 1
)

where clang++ >nul 2>&1
if errorlevel 1 (
    echo ERROR: clang++ not found in PATH!
    pause
    exit /b 1
)

:: Проверка программы обновления параметров
if not exist "%PARAM_UPDATER%" (
    echo ERROR: %PARAM_UPDATER% not found!
    echo Please compile update_params.c first
    pause
    exit /b 1
)

:: Получение информации о GPU
echo Checking GPU information...
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >nul 2>&1
if errorlevel 0 (
    echo GPU Information:
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    echo.
) else (
    echo WARNING: Could not query GPU information
    echo.
)

:: Определение архитектуры GPU
echo Detecting GPU architecture...
set ARCH=sm_86
for /f "tokens=1,2 delims=." %%i in ('nvidia-smi --query-gpu^=compute_capability --format^=csv ^| findstr /v "compute_capability"') do (
    if not "%%i"=="" (
        if not "%%i"=="compute_capability" (
            set CC_MAJOR=%%i
            set CC_MINOR=%%j
            echo Detected Compute Capability: !CC_MAJOR!.!CC_MINOR!
            set ARCH=sm_!CC_MAJOR!%%j
        )
    )
)

echo Using ARCH: !ARCH!
echo.

:: Проверка существования исходных файлов
echo Checking source files...
if not exist "%MAIN_OMP_FILE%" (
    echo ERROR: %MAIN_OMP_FILE% not found!
    echo Current directory: %CD%
    dir *.cpp 2>nul
    pause
    exit /b 1
)

if not exist "%CUDA_MODULE_FILE%" (
    echo ERROR: %CUDA_MODULE_FILE% not found!
    echo Current directory: %CD%
    dir *.cu 2>nul
    pause
    exit /b 1
)

echo Source files found: %MAIN_OMP_FILE%, %CUDA_MODULE_FILE%
echo.

:: Основной цикл
for %%S in (%PARAM_SIZES%) do (
    echo ===================================================
    echo Testing with PARAMETR_SIZE=%%S
    echo ===================================================
    
    :: Обновление параметров с помощью C программы
    echo Updating parameter files with PARAMETR_SIZE=%%S...
    %PARAM_UPDATER% %%S
    if errorlevel 1 (
        echo ERROR: Failed to update parameter files!
        pause
        exit /b 1
    )

    for %%R in (%REGISTER_COUNTS%) do (

        echo ===================================================
        echo Testing with PARAMETR_SIZE=%%S, MAXREGCOUNT=%%R
        echo ===================================================
        :: Получение текущего времени для имени файла
        for /f "tokens=1-3 delims=:." %%a in ("!time!") do (
            set CURRENT_TIME=%%a%%b%%c
        )
        for /f "tokens=1-3 delims=/- " %%a in ("!date!") do (
            set CURRENT_DATE=%%c%%a%%b
        )
        
        set COMPILE_LOG=%LOG_DIR%\compile_size_%%S_reg_%%R_!CURRENT_DATE!_!CURRENT_TIME!.txt
        
        echo Compiling CUDA module...
        echo Compile log: !COMPILE_LOG!
        
        :: Очистка предыдущих сборок
        del /Q *.o *.obj *.exe *.dll *.lib *.exp *.linkinfo 2>nul
        
        :: Компиляция CUDA DLL с правильными флагами для Windows
        (
            echo ========== CUDA COMPILATION START ==========
            echo Time: !date! !time!
            echo Parameters: PARAMETR_SIZE=%%S, MAXREGCOUNT=%%R, ARCH=!ARCH!
            echo Command: nvcc -O3 -arch=!ARCH! -Xcompiler "/MD" -DBUILD_CUDA_DLL --shared -maxrregcount=%%R -Xptxas "-O3,-v" -o cuda_module.dll cuda_module.cu
            echo.
        ) > "!COMPILE_LOG!"
        
        nvcc -O3 -arch=!ARCH! -Xcompiler "/MD" -DBUILD_CUDA_DLL --shared -maxrregcount=%%R -Xptxas "-O3,-v" -o cuda_module.dll cuda_module.cu 2>&1 >> "!COMPILE_LOG!"
        
        if errorlevel 1 (
            echo ERROR: CUDA compilation failed!
            echo Check log file: !COMPILE_LOG!
            pause
            exit /b 1
        )
        
        :: Проверка создания DLL
        if not exist cuda_module.dll (
            echo ERROR: cuda_module.dll not created!
            pause
            exit /b 1
        )
        
        (
            echo.
            echo ========== CUDA COMPILATION SUCCESS ==========
            echo.
        ) >> "!COMPILE_LOG!"
        
        echo Compiling main application...
        :: Компиляция с правильными флагами для Windows
        clang++ -std=c++17 -fopenmp -O3 -I. -o %EXE_NAME% main_omp.cpp -L. -lcuda_module 2>&1 >> "!COMPILE_LOG!"
        
        if errorlevel 1 (
            echo ERROR: Clang compilation failed!
            echo Check log file: !COMPILE_LOG!
            pause
            exit /b 1
        )
        
        if not exist "%EXE_NAME%" (
            echo ERROR: Executable not created!
            pause
            exit /b 1
        )
        
        (
            echo.
            echo ========== MAIN COMPILATION SUCCESS ==========
            echo Executable: %EXE_NAME%
            echo.
        ) >> "!COMPILE_LOG!"
        
        echo Running benchmark...
        echo Start time: !time!
        
        :: Запуск программы
        %EXE_NAME% 2>&1 | tee "%OUTPUT_DIR%\run_size_%%S_reg_%%R_!CURRENT_DATE!_!CURRENT_TIME!.txt"
        
        set RUN_ERROR=!errorlevel!
        echo End time: !time!
        echo Exit code: !RUN_ERROR!
        
        :: Копирование логов
        if exist "log.txt" (
            copy "log.txt" "%OUTPUT_DIR%\log_size_%%S_reg_%%R_!CURRENT_DATE!_!CURRENT_TIME!.txt" >nul
        )
        
        echo.
        echo Benchmark completed for PARAMETR_SIZE=%%S, MAXREGCOUNT=%%R
        echo Results saved in: %OUTPUT_DIR%\
        echo.
    )
)

echo All tests completed!
pause
exit /b 0