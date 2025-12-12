#!/bin/sh
#SBATCH -J omp_aco_intel
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -t 3-1:2:30
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Загружаем только Intel компилятор (OpenMP входит в него)
module add intel/v19.0.4.235

# Оптимальные настройки OpenMP для Xeon Gold 6130
export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE=dynamic

# Компилируем с оптимизацией
echo "=== Компиляция OMP программы с Intel компилятором ==="
icpc -qopenmp -O3 -std=c++17 -xHost -o omp_aco_intel OMP_C++.cpp

# Проверяем успешность компиляции
if [ $? -eq 0 ]; then
    echo "Компиляция успешна"
    echo "=== Запуск OMP программы ==="
    echo "Процессор: Intel Xeon Gold 6130"
    echo "Потоков: $OMP_NUM_THREADS"
    echo "Память: 128GB"
    echo "Начало: $(date)"
    
    ./omp_aco_intel
    
    echo "=== Завершено: $(date) ==="
else
    echo "Ошибка компиляции!"
    exit 1
fi