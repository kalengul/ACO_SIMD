#ifndef ANT_COLONY_COMMON_H
#define ANT_COLONY_COMMON_H

#include <vector>

#define PARAMETR_SIZE 336   // Количество параметров 21*x (6*х)
#define PARAMETR_SIZE_ONE_X 21    // Количество параметров на оди x 21 (6)
#define MAX_VALUE_SIZE 4    // Максимальное количество значений у параметров 5 (100)
#define NAME_FILE_GRAPH "Parametr_Graph/test336.txt"

#define ANT_SIZE 500      // Максимальное количество агентов 500
#define KOL_ITERATION 500   // Количество итераций ММК 500
#define KOL_STAT_LEVEL 20    // Количество этапов сбора статистики 20
#define KOL_PROGON_STATISTICS 5 //Для сбора статистики 50
#define KOL_PROGREV 1 //Количество итераций для начальнойго запуска системы 5
#define PARAMETR_Q 1.0        // Параметр ММК для усиления феромона Q 
#define MAX_PARAMETR_VALUE_TO_MIN_OPT 1        // Максимальное значение параметра чтобы выполнять разницу max-x
#define PARAMETR_RO 0.999     // Параметр ММК для испарения феромона RO
#define HASH_TABLE_SIZE 10000000 // Hash table size (10 million entries)
#define MAX_PROBES 10000 // Maximum number of probes for collision resolution
#define MAX_THREAD_CUDA 512 //1024
#define ZERO_HASH_RESULT -1.0
#define TYPE_ACO 1
#define ACOCCyN_KOL_ITERATION 50
#define PRINT_INFORMATION 0
#define CPU_RANDOM 0
#define KOL_THREAD_CPU_ANT 12
#define CONST_AVX 4 //double = 4, floaf,int = 8
#define CONST_RANDOM 100000
#define MAX_CONST 8000
#define BIN_SEARCH 0

#define GO_ALG_MINMAX 1
#define PAR_MAX_ALG_MINMAX 1000
#define PAR_MIN_ALG_MINMAX 1
#define OPTIMIZE_MAX

#define SHAFFERA 1
#define CARROM_TABLE 0
#define RASTRIGIN 0
#define ACKLEY 0
#define SPHERE 0
#define GRIEWANK 0
#define ZAKHAROV 0
#define SCHWEFEL 0
#define LEVY 0 //не работает
#define MICHAELWICZYNSKI 0
#define DELT4 0

// Структура для хэш-таблицы
struct HashEntry {
    int key[PARAMETR_SIZE];
    double value;
    bool occupied;
};

// CUDA функции
#ifdef __cplusplus
extern "C" {
#endif

    bool cuda_initialize(const double* parametr_value,
        const double* pheromon_value,
        const double* kol_enter_value);
    void cuda_run_iteration(const double* norm_matrix_probability,
        int* ant_parametr,
        double* antOF,
        double* global_minOf,
        double* global_maxOf,
        int* kol_hash_fail,
        int iteration);
    void cuda_cleanup();

    // OpenMP функции
    void omp_initialize(const double* initial_pheromon, const double* initial_kol_enter);
    void omp_calculate_probabilities();
    void omp_update_pheromones(const int* ant_parametr, const double* antOF);
    const double* omp_get_norm_matrix_probability();
    bool omp_is_data_ready();
    void omp_stop();
    void omp_cleanup();

#ifdef __cplusplus
}
#endif

// Функции для загрузки данных
bool load_matrix(const char* filename, std::vector<double>& parametr_value,
    std::vector<double>& pheromon_value, std::vector<double>& kol_enter_value);

#endif