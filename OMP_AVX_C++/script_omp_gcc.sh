#!/bin/sh
#SBATCH -J omp_aco_gcc
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -t 1-0:0:0
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Явно загружаем модуль в скрипте
module add gcc/v9.1

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "=== Компиляция OMP программы с GCC ==="
which g++
g++ --version

g++ -fopenmp -O3 -std=c++17 -march=native -o omp_aco_gcc OMP_C++.cpp

if [ $? -eq 0 ]; then
    echo "Компиляция успешна"
    echo "=== Запуск OMP программы ==="
    echo "Потоков: $OMP_NUM_THREADS"
    echo "Начало: $(date)"
    
    ./omp_aco_gcc
    
    echo "=== Завершено: $(date) ==="
else
    echo "Ошибка компиляции!"
    exit 1
fi
