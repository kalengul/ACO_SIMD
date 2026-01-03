@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ===================================================
echo    CUDA ACO Benchmark Launcher
echo ===================================================

set EXE_NAME=aco_cuda_4.exe
set OUTPUT_DIR=results
set CONFIG_FILE=parametrs.h

set REGISTER_COUNTS=16 32 64 128 256
::42 84 168 336 672 1344 2688 5376 10752 21504 43008 86016 172032 344064 688128 1376256
set PARAM_SIZES=42 84 168 336 672 1344 2688 5376 10752 21504 43008 86016 172032 344064 688128 1376256

:: Создание директории результатов
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%OUTPUT_DIR%\compilation_logs" mkdir "%OUTPUT_DIR%\compilation_logs"

:: Проверка наличия CUDA
echo Checking CUDA availability...
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: nvcc not found in PATH!
    echo Please install CUDA Toolkit or add to PATH
    pause
    exit /b 1
)
:: Проверка наличия MPI
echo Checking MPI availability...
where mpiexec >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: mpiexec not found in PATH!
    echo Please install Microsoft MPI or add to PATH
    pause
    exit /b 1
)
:: Получение информации о GPU
echo Checking GPU information...
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >nul 2>&1
if %errorlevel% equ 0 (
    echo GPU Information:
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
) else (
    echo WARNING: Could not query GPU information
)

:: Очистка предыдущих сборок
echo Cleaning previous builds...
del /Q *.o *.obj *.exe *.dll *.lib *.exp 2>nul

:: Определение архитектуры GPU
echo Detecting GPU architecture...
set ARCH=sm_86
for /f "tokens=1,2 delims=." %%i in ('nvidia-smi --query-gpu=compute_capability --format=csv ^| findstr /v "compute_capability"') do (
    if not "%%i"=="" (
        if not "%%i"=="compute_capability" (
            set CC_MAJOR=%%i
            set CC_MINOR=%%j
            set CC=%%i%%j
            echo Detected Compute Capability: !CC_MAJOR!.!CC_MINOR!
            set ARCH=sm_!CC_MAJOR!%%j
        )
    )
)

echo Using ARCH: !ARCH!

for %%S in (%PARAM_SIZES%) do (
    echo.
    echo ===================================================
    echo Testing with PARAMETR_SIZE=%%S
    echo ===================================================
    
    :: Обновление параметра в файле parametrs.h
    echo Updating parameter file with PARAMETR_SIZE=%%S...
    call :UpdateParameterSize "%%S"
    if !errorlevel! neq 0 (
        echo ERROR: Failed to update parameter file!
        pause
        exit /b 1
    )
    
    for %%R in (%REGISTER_COUNTS%) do (
        echo ===================================================
        echo Testing with PARAMETR_SIZE=%%S, MAXREGCOUNT=%%R
        echo ===================================================
        echo Compiling CUDA program with arch=%ARCH%...

        :: Создание имени файла для лога компиляции
        set COMPILE_LOG=%OUTPUT_DIR%\compilation_logs\compile_size_%%S_reg_%%R_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%.txt

        :: Расширенная компиляция с сохранением вывода в файл
        echo ========== COMPILATION START ========== > "!COMPILE_LOG!"
        echo Parameters: PARAMETR_SIZE=%%S, MAXREGCOUNT=%%R >> "!COMPILE_LOG!"
        echo Compile command: nvcc -O3 -arch=sm_70 -std=c++17 -o %EXE_NAME% aco_cuda_4.cu -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi -Xcompiler "/O2 /fp:fast /openmp /MT" -use_fast_math -lcudart -maxrregcount=%%R -Xptxas "-O3,-v" --resource-usage >> "!COMPILE_LOG!"
        echo. >> "!COMPILE_LOG!"

        nvcc -O3 -arch=!ARCH! -std=c++17 -o %EXE_NAME% aco_cuda_4.cu ^
             -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" ^
             -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" ^
             -lmsmpi ^
             -Xcompiler "/O2 /fp:fast /openmp /MT" ^
             -use_fast_math ^
             -lcudart ^
             -maxrregcount=%%R ^
             -Xptxas "-O3,-v" ^
             --resource-usage >> "!COMPILE_LOG!" 2>&1
        set COMPILE_RESULT=!errorlevel!
        
        echo ========== COMPILATION END ========== >> "!COMPILE_LOG!"
        echo Exit code: !COMPILE_RESULT! >> "!COMPILE_LOG!"

        if !COMPILE_RESULT! neq 0 (
            echo.
            echo WARNING: Compilation had warnings, but continuing...
            echo Compilation warnings/errors saved to: !COMPILE_LOG!
        ) else (
            echo Compilation successful. Log saved to: !COMPILE_LOG!
        )

        if not exist "%EXE_NAME%" (
            echo ERROR: Compilation failed - executable not created!
            pause
            exit /b 1
        )

        echo.
        echo ===================================================
        echo Running MPI ACO optimization...
        echo ===================================================
        echo.

        :: Запуск программы с измерением времени
        set START_TIME=!time!
        echo Start time: !START_TIME!
           
        mpiexec -n 1 %EXE_NAME%

        

        set END_TIME=!time!
        echo End time: !END_TIME!

        :: Расчет времени выполнения
        call :CalculateTime "!START_TIME!" "!END_TIME!" DURATION
        echo Execution time: !DURATION!

        :: Копирование логов
        if exist "log.txt" (
            copy "log.txt" "%OUTPUT_DIR%\log_size_%%S_reg_%%R_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%.txt" >nul
            echo Log file saved to %OUTPUT_DIR%
        )

        echo.
        echo ===================================================
        echo Benchmark completed for PARAMETR_SIZE=%%S, MAXREGCOUNT=%%R!
        echo Results saved in: %OUTPUT_DIR%\
        echo Execution time: !DURATION!
        echo ===================================================

        :: Очистка временных файлов
        del /Q *.linkinfo 2>nul
    )
)

pause
exit /b 0

:UpdateParameterSize
setlocal
set new_size=%~1

:: Вычисление имени файла графа на основе размера
set "graph_file=Parametr_Graph/test!new_size!.txt"

echo Updating parametrs.h with PARAMETR_SIZE=!new_size! and NAME_FILE_GRAPH=!graph_file!

:: Создание временного файла
set temp_file=%temp%\parametrs_temp.h
:: Обработка исходного файла и замена значений
(
    for /f "tokens=1,* delims=:" %%a in ('findstr /n "^" "%CONFIG_FILE%"') do (
        set "line=%%b"
        if defined line (
            :: Замена PARAMETR_SIZE
            echo !line! | findstr /c:"#define PARAMETR_SIZE " >nul
            if !errorlevel! equ 0 (
                echo #define PARAMETR_SIZE !new_size!
            ) else (
                :: Замена NAME_FILE_GRAPH
                echo !line! | findstr /c:"#define NAME_FILE_GRAPH" >nul
                if !errorlevel! equ 0 (
                    echo #define NAME_FILE_GRAPH "!graph_file!"
                ) else (
                    echo !line!
                )
            )
        ) else (
            echo.
        )
    )
) > "%temp_file%"

:: Копирование временного файла обратно
copy "%temp_file%" "%CONFIG_FILE%" >nul
del "%temp_file%"

:: Проверка, что изменения применены
findstr /c:"#define PARAMETR_SIZE !new_size!" "%CONFIG_FILE%" >nul
if !errorlevel! neq 0 (
    echo ERROR: Failed to update PARAMETR_SIZE in config file
    endlocal & exit /b 1
)

findstr /c:"!graph_file!" "%CONFIG_FILE%" >nul
if !errorlevel! neq 0 (
    echo ERROR: Failed to update NAME_FILE_GRAPH in config file
    endlocal & exit /b 1
)

echo Successfully updated parametrs.h with PARAMETR_SIZE=!new_size!
endlocal & exit /b 0

:CalculateTime
setlocal
set start=%~1
set end=%~2

set /a start_h=!start:~0,2!
set /a start_m=!start:~3,2!
set /a start_s=!start:~6,2!
set /a start_cs=!start:~9,2!

set /a end_h=!end:~0,2!
set /a end_m=!end:~3,2!
set /a end_s=!end:~6,2!
set /a end_cs=!end:~9,2!

set /a total_start=!start_h!*360000 + !start_m!*6000 + !start_s!*100 + !start_cs!
set /a total_end=!end_h!*360000 + !end_m!*6000 + !end_s!*100 + !end_cs!

set /a diff=!total_end! - !total_start!

if !diff! lss 0 (
    set /a diff=!diff! + 8640000
)

set /a hours=!diff! / 360000
set /a minutes=(!diff! %% 360000) / 6000
set /a seconds=((!diff! %% 360000) %% 6000) / 100
set /a cs=((!diff! %% 360000) %% 6000) %% 100

set "result=!hours!:!minutes!:!seconds!.!cs!"
endlocal & set %~3=%result%
goto :eof