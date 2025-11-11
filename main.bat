@echo off
echo Compiling CUDA through command line...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.35
nvcc -allow-unsupported-compiler kernel.cu -o kernel.exe
if exist kernel.exe (
    echo SUCCESS! File: kernel.exe
    kernel.exe
) else (
    echo FAILED!
)
pause