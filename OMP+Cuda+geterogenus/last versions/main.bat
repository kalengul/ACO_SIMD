@echo off
echo Compile CUDA.
nvcc -c -O3 -o ant_colony_cuda.o ant_colony_cuda.cu -arch=sm_50
echo Compile OpenMP.
g++ -c -O3 -fopenmp -o ant_colony_omp.o ant_colony_omp.cpp
echo Compile main.
g++ -c -O3 -fopenmp -o main.o main.cpp
echo Link.
g++ -o aco_hybrid.exe ant_colony_cuda.o ant_colony_omp.o main.o -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64" -lcudart -fopenmp
echo Run.
aco_hybrid.exe
pause