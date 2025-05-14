#pragma once
#include <string>
// ----------------- Hash Table Entry Structure -----------------
struct HashEntry {
    unsigned long long key; // Unique key composed of parameters
    double value;           // Objective function value
};

int start_NON_CUDA();

bool load_matrix_non_cuda(const std::string& filename, double* parametr_value, double* pheromon_value, double* kol_enter_value);

void go_mass_probability_non_cuda(double* pheromon, double* kol_enter, double* norm_matrix_probability);
void go_all_agent_non_cuda(int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail, float& totalHashTime);
void add_pheromon_iteration_non_cuda(double* pheromon, double* kol_enter, double* agent_node, double* OF);


// Функция для цвычисления параметра х при  параметрическом графе
double go_x_non_cuda(double* parametr, int start_index, int kol_parametr);
// Функция для целевой функции Шаффера
double BenchShafferaFunction_non_cuda_2x(double* parametr);
// Функция для целевой функции Шаффера с 100 переменными
double BenchShafferaFunction_non_cuda(double* parametr);
// Функция для вычисления вероятностной формулы
double probability_formula_non_cuda(double pheromon, double kol_enter);

void initializeHashTable_non_cuda(HashEntry* hashTable, int size);
unsigned long long murmurHash64A_non_cuda(unsigned long long key, unsigned long long seed);
unsigned long long betterHashFunction_non_cuda(unsigned long long key);
unsigned long long generateKey_non_cuda(const double* agent_node, int bx);
double getCachedResultOptimized_non_cuda(HashEntry* hashTable, const double* agent_node, int bx);
void saveToCacheOptimized_non_cuda(HashEntry* hashTable, const double* agent_node, int bx, double value);