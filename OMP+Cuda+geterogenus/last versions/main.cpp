#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <future>
#include "ant_colony_common.h"
/*
bool load_matrix(const char* filename, std::vector<double>& parametr_value, 
                std::vector<double>& pheromon_value, std::vector<double>& kol_enter_value) {
    size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
    
    parametr_value.resize(matrix_size);
    pheromon_value.resize(matrix_size);
    kol_enter_value.resize(matrix_size);
    
    // Инициализация тестовыми данными
    for (size_t i = 0; i < matrix_size; i++) {
        parametr_value[i] = (i % MAX_VALUE_SIZE) * 0.1;
        pheromon_value[i] = 1.0;
        kol_enter_value[i] = 1.0;
    }
    
    std::cout << "Matrix loaded with " << matrix_size << " elements" << std::endl;
    return true;
}
*/
bool load_matrix(const std::string& filename, double* parametr_value, double* pheromon_value, double* kol_enter_value) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Don't open file!" << std::endl;
        return false;
    }
    for (int i = 0; i < PARAMETR_SIZE; ++i) {
        for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
            int k = MAX_VALUE_SIZE * i + j;
            if (!(infile >> parametr_value[k])) {
                std::cerr << "Error load element [" << i << "][" << j << "]" << std::endl;
                return false;
            }
            if (parametr_value[k] != -100) {
                pheromon_value[k] = 1.0;
                kol_enter_value[k] = 1.0;
            }
            else {
                pheromon_value[k] = 0.0;
                parametr_value[k] = 0.0; // Нужно ли???? - Да, чтобы избежать использования -100 в вычислениях
                kol_enter_value[k] = 0.0;
            }
        }
    }
    infile.close();
    std::cout << "Matrix successfully loaded from " << filename << std::endl;
    return true;
}

int main() {
    std::cout << "Hybrid CUDA + OpenMP Ant Colony Optimization" << std::endl;
    std::cout << "Parameters: " << PARAMETR_SIZE << " parameters, " 
              << MAX_VALUE_SIZE << " values, " << ANT_SIZE << " ants, " 
              << KOL_ITERATION << " iterations" << std::endl;
    
    // Загрузка данных
    std::vector<double> parametr_value, pheromon_value, kol_enter_value;
    if (!load_matrix("graph.txt", parametr_value, pheromon_value, kol_enter_value)) {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }
    
    // Инициализация CUDA и OpenMP
    if (!cuda_initialize(parametr_value.data(), pheromon_value.data(), kol_enter_value.data())) {
        std::cerr << "CUDA initialization failed!" << std::endl;
        return 1;
    }
    
    omp_initialize(pheromon_value.data(), kol_enter_value.data());
    
    // Буферы для данных
    std::vector<int> ant_parametr(PARAMETR_SIZE * ANT_SIZE);
    std::vector<double> antOF(ANT_SIZE);
    
    // Переменные для результатов
    double global_minOf, global_maxOf;
    int kol_hash_fail;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Асинхронный запуск первого расчета вероятностей
    std::future<void> omp_future = std::async(std::launch::async, []() {
        omp_calculate_probabilities();
    });
    
    // Основной цикл оптимизации
    for (int iteration = 0; iteration < KOL_ITERATION; iteration++) {
        auto iter_start = std::chrono::high_resolution_clock::now();
        
        // Ждем готовности данных от OpenMP
        omp_future.wait();
        
        // Получаем нормализованную матрицу вероятностей
        const double* norm_matrix = omp_get_norm_matrix_probability();
        
        // Запускаем CUDA итерацию
        cuda_run_iteration(norm_matrix, ant_parametr.data(), antOF.data(),
                          &global_minOf, &global_maxOf, &kol_hash_fail, iteration);
        
        // Асинхронно запускаем обновление феромонов для следующей итерации
        if (iteration < KOL_ITERATION - 1) {
            omp_future = std::async(std::launch::async, [&]() {
                omp_update_pheromones(ant_parametr.data(), antOF.data());
                omp_calculate_probabilities();
            });
        } else {
            // Последняя итерация - только обновление
            omp_update_pheromones(ant_parametr.data(), antOF.data());
        }
        
        // Вывод статистики
        if ((iteration + 1) % KOL_STAT_LEVEL == 0) {
            auto iter_end = std::chrono::high_resolution_clock::now();
            auto iter_duration = std::chrono::duration_cast<std::chrono::milliseconds>(iter_end - iter_start);
            
            std::cout << "Iteration " << iteration + 1 << "/" << KOL_ITERATION 
                      << " - Min: " << global_minOf << ", Max: " << global_maxOf 
                      << ", Hash fails: " << kol_hash_fail 
                      << ", Time: " << iter_duration.count() << "ms" << std::endl;
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    
    // Финальный вывод
    std::cout << "\n=== OPTIMIZATION COMPLETED ===" << std::endl;
    std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "Minimum objective: " << global_minOf << std::endl;
    std::cout << "Maximum objective: " << global_maxOf << std::endl;
    std::cout << "Hash table collisions: " << kol_hash_fail << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Очистка ресурсов
    omp_cleanup();
    cuda_cleanup();
    
    return 0;
}