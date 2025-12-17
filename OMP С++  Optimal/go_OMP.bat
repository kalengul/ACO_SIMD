@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ===================================================
echo    OPENMP OPTIMIZED PROGRAM LAUNCHER WITH BENCHMARK
echo ===================================================

:: Настройки
set OUTPUT_DIR=results
set CONFIG_FILE=parametrs.h
set SOURCE_FILE=OMP_AVX_C++.cpp
set EXE_GCC=omp_avx_optimized_gcc.exe
set EXE_CLANG=omp_avx_optimized_clang.exe
set PARAM_SIZES="42 84 168 336 672 1344 2688 5376 10752 21504 43008 86016 172032 344064 688128 1376256"

:: Создание директорий
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%OUTPUT_DIR%\compilation_logs" mkdir "%OUTPUT_DIR%\compilation_logs"
if not exist "%OUTPUT_DIR%\benchmark_results" mkdir "%OUTPUT_DIR%\benchmark_results"

:: Проверка наличия компиляторов
echo [1/8] Checking compilers availability...

:: Проверка GCC
where g++ >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: g++ compiler not found in PATH!
    echo Please install MinGW-w64 or add to PATH
    pause
    exit /b 1
)
echo ✓ g++ compiler found

:: Проверка Clang
where clang++ >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: clang++ compiler not found in PATH!
    echo Will use only GCC compiler
    set CLANG_AVAILABLE=0
) else (
    echo ✓ clang++ compiler found
    set CLANG_AVAILABLE=1
)

echo.
echo [2/8] Setting OpenMP environment...
:: Автоматическое определение числа потоков
set /a OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%
if "%OMP_NUM_THREADS%"=="" set /a OMP_NUM_THREADS=4

set OMP_PROC_BIND=TRUE
set OMP_DYNAMIC=FALSE

echo Using %OMP_NUM_THREADS% threads
echo OMP_PROC_BIND=%OMP_PROC_BIND%
echo OMP_DYNAMIC=%OMP_DYNAMIC%

:: Функция для обновления параметра размера графа
goto :skip_functions

:update_parameter_size
    set new_size=%1
    set graph_file=Parametr_Graph/test%new_size%.txt
    
    echo Updating %CONFIG_FILE% with PARAMETR_SIZE=%new_size% and NAME_FILE_GRAPH=%graph_file%
    
    :: Создание временного файла
    set temp_file=%TEMP%\parametrs_temp_%RANDOM%.h
    
    :: Обработка исходного файла и замена значений
    (
    for /f "usebackq delims=" %%a in ("%CONFIG_FILE%") do (
        set "line=%%a"
        setlocal enabledelayedexpansion
        echo !line! | findstr /c:"#define PARAMETR_SIZE" >nul
        if not errorlevel 1 (
            echo #define PARAMETR_SIZE %new_size%
        ) else (
            echo !line! | findstr /c:"#define NAME_FILE_GRAPH" >nul
            if not errorlevel 1 (
                echo #define NAME_FILE_GRAPH "%graph_file%"
            ) else (
                echo !line!
            )
        )
        endlocal
    )
    ) > "%temp_file%"
    
    :: Копирование временного файла обратно
    copy /y "%temp_file%" "%CONFIG_FILE%" >nul
    del "%temp_file%" 2>nul
    
    :: Проверка, что изменения применены
    findstr /c:"#define PARAMETR_SIZE %new_size%" "%CONFIG_FILE%" >nul
    if errorlevel 1 (
        echo ERROR: Failed to update PARAMETR_SIZE in config file
        exit /b 1
    )
    
    findstr "%graph_file%" "%CONFIG_FILE%" >nul
    if errorlevel 1 (
        echo ERROR: Failed to update NAME_FILE_GRAPH in config file
        exit /b 1
    )
    
    echo ✓ Successfully updated %CONFIG_FILE% with PARAMETR_SIZE=%new_size%
    exit /b 0

:skip_functions

echo.
echo [3/8] Starting benchmark for different parameter sizes...
pause
:: Перебор всех размеров параметров
for %%S in (%PARAM_SIZES%) do (
    echo.
    echo ===================================================
    echo Benchmark iteration: PARAMETR_SIZE=%%S
    echo ===================================================
    pause
    :: Обновление параметра размера графа в файле parametrs.h
    echo [3.1] Updating parameter file with PARAMETR_SIZE=%%S...
    call :update_parameter_size %%S
    if errorlevel 1 (
        echo ERROR: Failed to update parameter file!
        pause
        exit /b 1
    )
 
    :: Создание метки времени для логов
    for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
    set TIMESTAMP=%datetime:~0,4%%datetime:~4,2%%datetime:~6,2%_%datetime:~8,2%%datetime:~10,2%%datetime:~12,2%
    
    :: Компиляция с GCC
    echo [3.2] Compiling with GCC (OpenMP + AVX optimization)...
    set COMPILE_LOG_GCC=%OUTPUT_DIR%\compilation_logs\compile_gcc_size_%%S_%TIMESTAMP%.txt
    
    (
        echo ========== GCC COMPILATION START ==========
        echo Timestamp: %TIMESTAMP%
        echo Parameters: PARAMETR_SIZE=%%S
        echo Source file: %SOURCE_FILE%
        echo OpenMP threads: %OMP_NUM_THREADS%
        echo Compile command: g++ -std=c++17 -fopenmp -O3 -mavx2 -mfma -march=native -o %EXE_GCC% %SOURCE_FILE%
        echo.
        g++ -std=c++17 -fopenmp -O3 -mavx2 -mfma -march=native -o %EXE_GCC% %SOURCE_FILE%
        echo.
        echo ========== GCC COMPILATION END ==========
    ) > "%COMPILE_LOG_GCC%" 2>&1
    
    if errorlevel 1 (
        echo ✗ GCC compilation failed!
        echo Compilation errors saved to: %COMPILE_LOG_GCC%
        echo Showing last 5 lines of compilation log:
        for /f %%i in ('type "%COMPILE_LOG_GCC%" ^| find /c /v ""') do set /a lines=%%i
        set /a skip=lines-5
        if !skip! lss 0 set /a skip=0
        more +!skip! "%COMPILE_LOG_GCC%"
        echo.
    ) else (
        echo ✓ GCC compilation successful
        echo Log saved to: %COMPILE_LOG_GCC%
    )
    
    :: Компиляция с Clang (если доступен)
    if "!CLANG_AVAILABLE!"=="1" (
        echo [3.3] Compiling with Clang (OpenMP + AVX optimization)...
        set COMPILE_LOG_CLANG=%OUTPUT_DIR%\compilation_logs\compile_clang_size_%%S_%TIMESTAMP%.txt
        
        (
            echo ========== CLANG COMPILATION START ==========
            echo Timestamp: %TIMESTAMP%
            echo Parameters: PARAMETR_SIZE=%%S
            echo Source file: %SOURCE_FILE%
            echo OpenMP threads: %OMP_NUM_THREADS%
            echo Compile command: clang++ -std=c++17 -fopenmp -O3 -mavx2 -mfma -march=native -o %EXE_CLANG% %SOURCE_FILE%
            echo.
            clang++ -std=c++17 -fopenmp -O3 -mavx2 -mfma -march=native -o %EXE_CLANG% %SOURCE_FILE%
            echo.
            echo ========== CLANG COMPILATION END ==========
        ) > "%COMPILE_LOG_CLANG%" 2>&1
        
        if errorlevel 1 (
            echo ✗ Clang compilation failed!
            echo Compilation errors saved to: %COMPILE_LOG_CLANG%
        ) else (
            echo ✓ Clang compilation successful
            echo Log saved to: %COMPILE_LOG_CLANG%
        )
    )
    
    echo.
    echo [3.4] Running optimized programs...
    
    :: Запуск GCC версии
    if exist "%EXE_GCC%" (
        echo --- Running GCC version ---
        echo Start time: %date% %time%
        
        for /f "tokens=1-4 delims=:.," %%a in ("%time%") do (
            set /a "start_time_gcc=(((%%a*60)+1%%b %% 100)*60)+1%%c %% 100"
            set /a "start_time_gcc=!start_time_gcc!*1000+(1%%d %% 100)*10"
        )
        
        %EXE_GCC%
        set EXIT_CODE_GCC=!errorlevel!
        
        for /f "tokens=1-4 delims=:.," %%a in ("%time%") do (
            set /a "end_time_gcc=(((%%a*60)+1%%b %% 100)*60)+1%%c %% 100"
            set /a "end_time_gcc=!end_time_gcc!*1000+(1%%d %% 100)*10"
        )
        
        :: Расчет времени выполнения
        set /a duration_gcc_ms=end_time_gcc - start_time_gcc
        set /a duration_gcc_sec=duration_gcc_ms / 1000
        set /a duration_gcc_ms_rem=duration_gcc_ms %% 1000
        
        echo End time: %date% %time%
        echo Execution time (GCC): !duration_gcc_sec!.!duration_gcc_ms_rem! seconds
        echo Exit code (GCC): !EXIT_CODE_GCC!
        
        :: Сохранение результатов GCC
        if exist "statistics.txt" (
            copy "statistics.txt" "%OUTPUT_DIR%\benchmark_results\results_gcc_size_%%S_%TIMESTAMP%.txt" >nul
            echo Results saved to: %OUTPUT_DIR%\benchmark_results\results_gcc_size_%%S_%TIMESTAMP%.txt
        )
    ) else (
        echo ✗ GCC executable not found, skipping run
    )
    
    :: Запуск Clang версии (если доступен)
    if exist "%EXE_CLANG%" (
        echo.
        echo --- Running Clang version ---
        echo Start time: %date% %time%
        
        for /f "tokens=1-4 delims=:.," %%a in ("%time%") do (
            set /a "start_time_clang=(((%%a*60)+1%%b %% 100)*60)+1%%c %% 100"
            set /a "start_time_clang=!start_time_clang!*1000+(1%%d %% 100)*10"
        )
        
        %EXE_CLANG%
        set EXIT_CODE_CLANG=!errorlevel!
        
        for /f "tokens=1-4 delims=:.," %%a in ("%time%") do (
            set /a "end_time_clang=(((%%a*60)+1%%b %% 100)*60)+1%%c %% 100"
            set /a "end_time_clang=!end_time_clang!*1000+(1%%d %% 100)*10"
        )
        
        :: Расчет времени выполнения
        set /a duration_clang_ms=end_time_clang - start_time_clang
        set /a duration_clang_sec=duration_clang_ms / 1000
        set /a duration_clang_ms_rem=duration_clang_ms %% 1000
        
        echo End time: %date% %time%
        echo Execution time (Clang): !duration_clang_sec!.!duration_clang_ms_rem! seconds
        echo Exit code (Clang): !EXIT_CODE_CLANG!
        
        :: Сохранение результатов Clang
        if exist "statistics.txt" (
            copy "statistics.txt" "%OUTPUT_DIR%\benchmark_results\results_clang_size_%%S_%TIMESTAMP%.txt" >nul
            echo Results saved to: %OUTPUT_DIR%\benchmark_results\results_clang_size_%%S_%TIMESTAMP%.txt
        )
    ) else (
        if "!CLANG_AVAILABLE!"=="1" (
            echo ✗ Clang executable not found, skipping run
        )
    )
    
    :: Создание суммарного отчета
    echo.
    echo [3.5] Creating summary report...
    set SUMMARY_FILE=%OUTPUT_DIR%\benchmark_results\summary_size_%%S_%TIMESTAMP%.txt
    
    (
        echo ========== BENCHMARK SUMMARY ==========
        echo Timestamp: %TIMESTAMP%
        echo Parameter Size: %%S
        echo OpenMP Threads: %OMP_NUM_THREADS%
        echo.
        echo --- GCC ---
        if exist "%EXE_GCC%" (
            echo Compilation: SUCCESS
            echo Execution Time: !duration_gcc_sec!.!duration_gcc_ms_rem! seconds
            echo Exit Code: !EXIT_CODE_GCC!
        ) else (
            echo Compilation: FAILED
        )
        echo.
        echo --- Clang ---
        if exist "%EXE_CLANG%" (
            echo Compilation: SUCCESS
            echo Execution Time: !duration_clang_sec!.!duration_clang_ms_rem! seconds
            echo Exit Code: !EXIT_CODE_CLANG!
        ) else (
            if "!CLANG_AVAILABLE!"=="1" (
                echo Compilation: FAILED
            ) else (
                echo Compiler: NOT AVAILABLE
            )
        )
        echo ======================================
    ) > "%SUMMARY_FILE%"
    
    echo Summary saved to: %SUMMARY_FILE%
    
    :: Очистка временных файлов
    echo [3.6] Cleaning temporary files...
    del %EXE_GCC% 2>nul
    del %EXE_CLANG% 2>nul
    del statistics.txt 2>nul
    
    echo.
    echo ===================================================
    echo Benchmark completed for PARAMETR_SIZE=%%S!
    echo ===================================================
    timeout /t 2 /nobreak >nul
)

echo.
echo [4/8] Generating final benchmark report...

:: Создание итогового отчета
set FINAL_REPORT=%OUTPUT_DIR%\final_benchmark_report_%TIMESTAMP%.txt

(
    echo ===================================================
    echo       FINAL BENCHMARK REPORT
    echo ===================================================
    echo Generated: %date% %time%
    echo Total Tests: %PARAM_SIZES: =, %
    echo OpenMP Threads: %OMP_NUM_THREADS%
    echo Compilers: GCC + !CLANG_AVAILABLE! (Clang available)
    echo ===================================================
    echo.
    echo Individual test results are in: %OUTPUT_DIR%\benchmark_results\
    echo Compilation logs are in: %OUTPUT_DIR%\compilation_logs\
    echo.
    echo To view detailed results:
    echo 1. Check benchmark_results folder for execution outputs
    echo 2. Check compilation_logs folder for compilation details
    echo 3. Each test has its own timestamped folder
    echo ===================================================
) > "%FINAL_REPORT%"

echo.
echo [5/8] Benchmark completed successfully!
echo.
echo ===================================================
echo RESULTS SUMMARY
echo ===================================================
echo Total parameter sizes tested: %PARAM_SIZES: =, %
echo Output directory: %OUTPUT_DIR%
echo - benchmark_results/ : Execution results
echo - compilation_logs/  : Compilation logs
echo Final report: %FINAL_REPORT%
echo ===================================================

pause
endlocal