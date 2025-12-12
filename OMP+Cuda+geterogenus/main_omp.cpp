#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <algorithm>
#include <string>
#include <fstream>
#include <iomanip>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <numeric>
#include <deque>
#include <sstream>
#include "cuda_module.h"

#include <immintrin.h>

// Исправляем конфликт имен с Windows макросами
#undef max
#undef min
// 42, 84, 168, 336, 672, 1344, 2688, 5376, 10752, 21504, 43008, 86016, 172032, 344064, 688128, 1376256
// Константы
#define MAX_VALUE_SIZE 4
#define PARAMETR_SIZE 1344
#define SET_PARAMETR_SIZE_ONE_X 21    // Количество параметров на оди x 21 (6)
#define ANT_SIZE 500
#define MAX_THREAD_CUDA 256
#define NAME_FILE_GRAPH "Parametr_Graph/test1344_4.txt"
#define KOL_ITERATION 500

#define KOL_PROGREV 0
#define KOL_PROGON_STATISTICS 50

#define PARAMETR_RO 0.999
#define PARAMETR_Q 1.0

#define PRINT_INFORMATION 0

// Оптимизационные флаги
#define OPTIMIZE_MIN_1 1
#define OPTIMIZE_MIN_2 0
#define OPTIMIZE_MAX 0
#define MAX_PARAMETR_VALUE_TO_MIN_OPT 1000.0

#define GO_HYBRID_OMP 0
#define GO_HYBRID_OMP_NON_HASH 0
#define GO_HYBRID_BALANCED_OMP 0 
#define GO_HYBRID_BALANCED_OMP_NON_HASH 0
#define GO_HYBRID_BALANCED_DYNAMIC_OMP 0
#define GO_HYBRID_BALANCED_DYNAMIC_OMP_NON_HASH 0
#define GO_HYBRID_BALANCED_DYNAMIC_OMP_PARALLEL 1
#define GO_HYBRID_BALANCED_DYNAMIC_OMP_PARALLEL_NON_HASH 0

#define BALANCED_TIME_GPU_FUNCTION 1
#define BALANCED_TIME_GPU 0
#define BALANCED_TIME_OMP_FUNCTION 1
#define BALANCED_ADD_TIME_OMP_PROBABILITY 0
#define BALANCED_ADD_TIME_OMP_ADD_AND_DECREASE 1

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

#define INITIAL_CPU_ANTS_RATIO 0.5f
#define MIN_BALANSED_ANT 10

#define HASH_TABLE_SIZE 67108864  // 2^26 (должно быть степенью двойки)
#define ZERO_HASH_RESULT -1.0
#define ZERO_HASH 100000000000
#define MAX_PROBES 10000 // Maximum number of probes for collision resolution
#define HASH_TABLE_SHARDS 16

#ifdef _OPENMP
#include <omp.h>
#endif

// Глобальные переменные
std::atomic<int> active_cuda_tasks{ 0 };
std::atomic<int> completed_iterations{ 0 };
std::atomic<int> successful_iterations{ 0 };
std::ofstream logFile;

struct IterationData {
    int iteration;
    std::vector<double> norm_matrix_probability;
    std::vector<int> ant_parametr;
    std::vector<double> antOF;
    double minOf;
    double maxOf;
    int kol_hash_fail;
    int ants_count;
    double Time_all, Time_1, Time_function;
    IterationData() : iteration(0), minOf(0), maxOf(0), kol_hash_fail(0),
        ants_count(0), Time_all(0.0), Time_1(0.0), Time_function(0.0){
    }
};

struct ThreadSafeQueue {
    std::queue<IterationData> queue;
    mutable std::mutex mtx;
    std::condition_variable cv;
    bool stopped = false;
};

struct ACOData {
    // Данные ACO
    std::vector<double> parametr_value;
    std::vector<double> current_pheromon;
    std::vector<double> current_kol_enter;
    std::vector<double> norm_matrix_probability;
    std::vector<int> ant_parametr;
    std::vector<double> antOF;

    // Управление выполнением
    std::atomic<bool> stop_requested{ false };
    std::atomic<int> current_iteration{ 0 };

    // Результаты
    double minOf = 1e9;
    double maxOf = -1e9;
    double global_minOf = 1e9;
    double global_maxOf = -1e9;
    int kol_hash_fail = 0;

    // Время выполнения
    double Time_CPU_all = 0.0, Time_CPU_prob = 0.0, Time_CPU_wait = 0.0, Time_CPU_update = 0.0;
    double Time_GPU_all = 0.0, Time_GPU = 0.0, Time_GPU_wait = 0.0, Time_GPU_function = 0.0;
    double Time_OMP_all = 0.0, Time_OMP_wait = 0.0, Time_OMP_function = 0.0;

    double cpu_ants_statistics = 0.0, gpu_ants_statistics = 0.0;

    // Очереди для межпоточного обмена
    ThreadSafeQueue gpu_to_cpu_queue;
    ThreadSafeQueue cpu_to_gpu_queue;
    ThreadSafeQueue cpu_to_OMP_queue;
    ThreadSafeQueue OMP_to_cpu_queue;

    //Очереди для обмена без блокировки
    std::queue<IterationData> gpu_to_cpu_queue_current;
    std::queue<IterationData> cpu_to_gpu_queue_current;
    std::queue<IterationData> cpu_to_OMP_queue_current;
    std::queue<IterationData> OMP_to_cpu_queue_current;

    // Для балансировки количество муравьев-агентов на итерации зависит от INITIAL_CPU_ANTS_RATIO
    std::atomic<int> gpu_ants_count;
    std::atomic<int> OMP_ants_count;
    std::mutex metrics_mutex;
};

// Структура для хэш-таблицы
typedef struct {
    unsigned long long key;
    double value;
} HashEntry;

// Глобальная хэш-таблица для OMP
HashEntry* hashTable = nullptr;

#if (MAX_VALUE_SIZE==4)
// AVX-оптимизированная функция вычисления параметра
double compute_parameter(double* params, int start, int count) noexcept {
    // Для MAX_VALUE_SIZE=4 можем использовать AVX
    if (count == 4) {
        __m256d vec = _mm256_loadu_pd(params + start);
        __m256d sum_vec = _mm256_hadd_pd(vec, vec);
        double sum = ((double*)&sum_vec)[0] + ((double*)&sum_vec)[2];
        return params[start] * sum;
    }

    double sum = 0.0;
    // Стандартная реализация для других случаев
    for (int i = 1; i < count; ++i) {
        sum += params[start + i];
    }
    return params[start] * sum;
}
#else
// Функция для вычисления параметра x
double go_x_non_cuda_omp(double* parametr, int start_index, int kol_parametr) noexcept {
    double sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
    for (int i = 1; i < kol_parametr; ++i) {
        sum += parametr[start_index + i];
    }
    return parametr[start_index] * sum;
}
#endif


#if (SHAFFERA) 
#if (MAX_VALUE_SIZE==4)
double BenchShafferaFunction_omp(double* params) noexcept {
    double sum_sq = 0.0;
    const int num_vars = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;

    // AVX-оптимизация для случая, когда SET_PARAMETR_SIZE_ONE_X=4
    if (SET_PARAMETR_SIZE_ONE_X == 4) {
        __m256d sum_sq_vec = _mm256_setzero_pd();

        for (int i = 0; i < num_vars; ++i) {
            double x = compute_parameter(params, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
            __m256d x_vec = _mm256_set1_pd(x);
            sum_sq_vec = _mm256_add_pd(sum_sq_vec, _mm256_mul_pd(x_vec, x_vec));
        }

        // Горизонтальное суммирование
        sum_sq_vec = _mm256_hadd_pd(sum_sq_vec, sum_sq_vec);
        sum_sq = ((double*)&sum_sq_vec)[0] + ((double*)&sum_sq_vec)[2];
    }
    else {
        // Стандартная реализация для других случаев
        for (int i = 0; i < num_vars; ++i) {
            double x = compute_parameter(params, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
            sum_sq += x * x;
        }
    }

    double r = std::sqrt(sum_sq);
    double sin_r = std::sin(r);
    return 0.5 - (sin_r * sin_r - 0.5) / (1.0 + 0.001 * sum_sq);
}
#else 
// Функция для целевой функции Шаффера с 100 переменными
double BenchShafferaFunction_omp(double* parametr) noexcept {
    double r_squared = 0.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:r_squared)
#endif
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
        r_squared += x * x; // Сумма квадратов
    }

    double r = sqrt(r_squared);
    double sin_r = sin(r);
    return 1.0 / 2.0 - (sin_r * sin_r - 0.5) / (1.0 + 0.001 * r_squared);
}
#endif
#endif
#if (RASTRIGIN)
#if (MAX_VALUE_SIZE==4)
double BenchShafferaFunction_omp(double* params) noexcept {
    double sum = 0.0;
    const int num_vars = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
    constexpr double two_pi = 2.0 * M_PI;

    // AVX-оптимизация для RASTRIGIN
    if (SET_PARAMETR_SIZE_ONE_X == 4) {
        __m256d sum_vec = _mm256_setzero_pd();
        __m256d ten_vec = _mm256_set1_pd(10.0);
        __m256d two_pi_vec = _mm256_set1_pd(two_pi);

        for (int i = 0; i < num_vars; ++i) {
            double x = compute_parameter(params, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
            __m256d x_vec = _mm256_set1_pd(x);
            __m256d x_sq = _mm256_mul_pd(x_vec, x_vec);

            // 10.0 * cos(2π * x)
            __m256d cos_arg = _mm256_mul_pd(two_pi_vec, x_vec);
            // Используем стандартный cos
            alignas(32) double cos_args[4];
            alignas(32) double cos_vals[4];
            _mm256_store_pd(cos_args, cos_arg);
            for (int j = 0; j < 4; ++j) {
                cos_vals[j] = std::cos(cos_args[j]);
            }
            __m256d cos_val = _mm256_load_pd(cos_vals);
            __m256d term = _mm256_mul_pd(ten_vec, cos_val);

            // x² - 10*cos(2π*x) + 10
            __m256d result = _mm256_add_pd(_mm256_sub_pd(x_sq, term), ten_vec);
            sum_vec = _mm256_add_pd(sum_vec, result);
        }

        // Горизонтальное суммирование
        sum_vec = _mm256_hadd_pd(sum_vec, sum_vec);
        sum = ((double*)&sum_vec)[0] + ((double*)&sum_vec)[2];
    }
    else {
        // Стандартная реализация
        for (int i = 0; i < num_vars; ++i) {
            double x = compute_parameter(params, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
            sum += x * x - 10.0 * std::cos(two_pi * x) + 10.0;
        }
    }
    return sum;
}
#else
// Растригин-функция
double BenchShafferaFunction_omp(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
        sum += x * x - 10 * cos(2 * M_PI * x) + 10;
    }
    return sum;
}
#endif
#endif
#if (ACKLEY)
#if (MAX_VALUE_SIZE==4)
double BenchShafferaFunction_omp(double* params) noexcept {
    const int num_vars = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
    double sum_sq = 0.0;
    double sum_cos = 0.0;

    // AVX-оптимизация для ACKLEY
    if (SET_PARAMETR_SIZE_ONE_X == 4) {
        __m256d sum_sq_vec = _mm256_setzero_pd();
        __m256d sum_cos_vec = _mm256_setzero_pd();
        __m256d two_pi_vec = _mm256_set1_pd(2.0 * M_PI);

        for (int i = 0; i < num_vars; ++i) {
            double x = compute_parameter(params, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
            __m256d x_vec = _mm256_set1_pd(x);

            // Сумма квадратов
            sum_sq_vec = _mm256_add_pd(sum_sq_vec, _mm256_mul_pd(x_vec, x_vec));

            // Сумма косинусов
            __m256d cos_arg = _mm256_mul_pd(two_pi_vec, x_vec);
            alignas(32) double cos_args[4];
            alignas(32) double cos_vals[4];
            _mm256_store_pd(cos_args, cos_arg);
            for (int j = 0; j < 4; ++j) {
                cos_vals[j] = std::cos(cos_args[j]);
            }
            __m256d cos_val = _mm256_load_pd(cos_vals);
            sum_cos_vec = _mm256_add_pd(sum_cos_vec, cos_val);
        }

        // Горизонтальное суммирование
        sum_sq_vec = _mm256_hadd_pd(sum_sq_vec, sum_sq_vec);
        sum_sq = ((double*)&sum_sq_vec)[0] + ((double*)&sum_sq_vec)[2];

        sum_cos_vec = _mm256_hadd_pd(sum_cos_vec, sum_cos_vec);
        sum_cos = ((double*)&sum_cos_vec)[0] + ((double*)&sum_cos_vec)[2];
    }
    else {
        // Стандартная реализация
        for (int i = 0; i < num_vars; ++i) {
            double x = compute_parameter(params, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
            sum_sq += x * x;
            sum_cos += std::cos(2.0 * M_PI * x);
        }
    }

    double n = static_cast<double>(num_vars);
    return -20.0 * std::exp(-0.2 * std::sqrt(sum_sq / n)) - std::exp(sum_cos / n) + 20.0 + M_E;
}
#else
// Акли-функция
double BenchShafferaFunction_omp(double* parametr) {
    double first_sum = 0.0;
    double second_sum = 0.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:first_sum, second_sum)
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
        first_sum += x * x;
        second_sum += cos(2 * M_PI * x);
    }
    double exp_term_1 = exp(-0.2 * sqrt(first_sum / num_variables));
    double exp_term_2 = exp(second_sum / num_variables);
    return -20 * exp_term_1 - exp_term_2 + M_E + 20;
}
#endif
#endif
#if (SPHERE)
#if (MAX_VALUE_SIZE==4)
double BenchShafferaFunction_omp(double* params) noexcept {
    double sum_sq = 0.0;
    const int num_vars = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;

    // AVX-оптимизация для SPHERE
    if (SET_PARAMETR_SIZE_ONE_X == 4) {
        __m256d sum_sq_vec = _mm256_setzero_pd();

        for (int i = 0; i < num_vars; ++i) {
            double x = compute_parameter(params, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
            __m256d x_vec = _mm256_set1_pd(x);
            sum_sq_vec = _mm256_add_pd(sum_sq_vec, _mm256_mul_pd(x_vec, x_vec));
        }

        // Горизонтальное суммирование
        sum_sq_vec = _mm256_hadd_pd(sum_sq_vec, sum_sq_vec);
        sum_sq = ((double*)&sum_sq_vec)[0] + ((double*)&sum_sq_vec)[2];
    }
    else {
        // Стандартная реализация
        for (int i = 0; i < num_vars; ++i) {
            double x = compute_parameter(params, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
            sum_sq += x * x;
        }
    }
    return sum_sq;
}
#else
// Сферическая функция
double BenchShafferaFunction_omp(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
        sum += x * x;
    }
    return sum;
}
#endif
#endif


#if (DELT4)
// Михаэлевич-Викинский
double BenchShafferaFunction_omp(double* parametr) {
    double r_squared = 0.0;
    double sum_if = 0.0;
    double sum = 0.0;
    double second_sum = 0.0;
    double r_cos = 1.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
        sum_if += x;
        r_squared += x * x; // Сумма квадратов
        sum += x * x - 10 * cos(2 * M_PI * x) + 10;
        second_sum += cos(2 * M_PI * x);
        r_cos *= cos(x);
    }
    if (sum_if >= -10 * num_variables && sum_if <= -5 * num_variables) {
        double r = sqrt(r_squared); //Шаффер
        double sin_r = sin(r);
        return 1.0 / 2.0 - (sin_r * sin_r - 0.5) / (1.0 + 0.001 * r_squared);
    }
    if (sum_if > -5 * num_variables && sum_if <= 0) {
        return sum; //Растрыгин
    }
    if (sum_if > 0 && sum_if <= 5 * num_variables) {
        double exp_term_1 = exp(-0.2 * sqrt(r_squared / num_variables));
        double exp_term_2 = exp(second_sum / num_variables);
        return -20 * exp_term_1 - exp_term_2 + M_E + 20; //Akley
    }
    if (sum_if > 5 * num_variables && sum_if <= 10 * num_variables) {
        double a = 1.0 - sqrt(r_squared) / M_PI; //Carrom
        double OF = r_cos * exp(fabs(a)); // Используем fabs для абсолютного значения
        return OF * OF; // Возвращаем OF в квадрате
    }
    return 0;
}
#endif
#if (CARROM_TABLE) 
// CarromTableFunction
double BenchShafferaFunction_omp(double* parametr) {
    double r_cos = 1.0;
    double r_squared = 0.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:r_squared, r_cos)
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
        r_cos *= cos(x);
        r_squared += x * x;
    }
    double a = 1.0 - sqrt(r_squared) / M_PI;
    double OF = r_cos * exp(fabs(a)); // Используем fabs для абсолютного значения
    return OF * OF; // Возвращаем OF в квадрате
}
#endif
#if (GRIEWANK)
// Гриванк-функция
double BenchShafferaFunction_omp(double* parametr) {
    double sum = 0.0;
    double prod = 1.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:sum, prod)
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
        sum += x * x;
        prod *= cos(x / sqrt(i + 1));
    }
    return sum / 4000 - prod + 1;
}
#endif
#if (ZAKHAROV)
// Захаров-функция
double BenchShafferaFunction_omp(double* parametr) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:sum1, sum2)
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
        sum1 += pow(x, 2);
        sum2 += 0.5 * i * x;
    }
    return sum1 + pow(sum2, 2) + pow(sum2, 4);
}
#endif
#if (SCHWEFEL)
// Швейфель-функция
double BenchShafferaFunction_omp(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
        sum -= x * sin(sqrt(abs(x)));
    }
    return sum;
}
#endif
#if (LEVY)
// Леви-функция
double BenchShafferaFunction_omp(double* parametr) {
    double w_first = 1 + (go_x_non_cuda_omp(parametr, 0, SET_PARAMETR_SIZE_ONE_X) - 1) / 4;
    double w_last = 1 + (go_x_non_cuda_omp(parametr, PARAMETR_SIZE - SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X) - 1) / 4;
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:sum)
    for (int i = 1; i <= num_variables - 1; ++i) {
        double wi = 1 + (go_x_non_cuda_omp(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X) - 1) / 4;
        sum += pow(wi - 1, 2) * (1 + 10 * pow(sin(M_PI * wi), 2)) +
            pow(wi - wi * w_i_prev, 2) * (1 + pow(sin(2 * M_PI * wi), 2));
    }
    return pow(sin(M_PI * w_first), 2) + sum + pow(w_last - 1, 2) * (1 + pow(sin(2 * M_PI * w_last), 2));
}
#endif
#if (MICHAELWICZYNSKI)
// Михаэлевич-Викинский
double BenchShafferaFunction_omp(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
        sum -= sin(x) * pow(sin((i + 1) * x * x / M_PI), 20);
    }
    return sum;
}
#endif

// Функция для генерации случайных чисел
inline double unified_fast_random(uint64_t& seed) {
    seed = (seed * 6364136223846793005ULL + 1442695040888963407ULL);
    return static_cast<double>(seed >> 32) / 4294967296.0;
}

// Объявления функций
void wait_completion(int max_wait_ms = 10000);
void aco_cleanup(ACOData* aco);

// Функции для вывода глобальных переменных
void print_global_variables(std::ostream& os = std::cout) {
    os << "=== GLOBAL VARIABLES ===" << std::endl;
    os << "active_cuda_tasks: " << active_cuda_tasks.load() << std::endl;
    os << "completed_iterations: " << completed_iterations.load() << std::endl;
    os << "successful_iterations: " << successful_iterations.load() << std::endl;
    os << "logFile.is_open(): " << (logFile.is_open() ? "Yes" : "No") << std::endl;
    os << std::endl;
}

// Functions for printing IterationData
void print_iteration_data(const IterationData& data, std::ostream& os = std::cout) {
    os << "=== ITERATION DATA ===" << std::endl;
    os << "Iteration: " << data.iteration << std::endl;
    os << "Minimum OF: " << data.minOf << std::endl;
    os << "Maximum OF: " << data.maxOf << std::endl;
    os << "Hash failures: " << data.kol_hash_fail << std::endl;
    os << "Ants count: " << data.ants_count << std::endl;
    os << "Time_all: " << data.Time_all << std::endl;
    os << "Time_1: " << data.Time_1 << std::endl;
    os << "Time_function: " << data.Time_function << std::endl;
    os << "norm_matrix_probability: " << data.norm_matrix_probability.size() << " elements" << std::endl;
    os << "ant_parametr: " << data.ant_parametr.size() << " elements" << std::endl;
    os << "antOF: " << data.antOF.size() << " elements" << std::endl;
    std::cout << "norm_matrix_probability (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
    for (int i = 0; i < PARAMETR_SIZE; ++i) {
        for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
            std::cout << data.norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ", ";
        }
        std::cout << std::endl;
    }

    if (data.ant_parametr.size()!=0){
    std::cout << "ANT (" << data.ants_count << "):" << std::endl;
    for (int i = 0; i < data.ants_count; ++i) {
        std::cout << "ANT #" << i << " ";
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            std::cout << data.ant_parametr[i * PARAMETR_SIZE + j] << " ";
        }
        std::cout << "-> " << data.antOF[i] << std::endl;
    }
    }

    os << std::endl;
}

// Function for printing antOF vector statistics
void print_antof_statistics(const std::vector<double>& antof, std::ostream& os = std::cout) {
    if (antof.empty()) {
        os << "antOF: vector is empty" << std::endl;
        return;
    }

    double min_val = antof[0];
    double max_val = antof[0];
    double sum = 0.0;

    for (double val : antof) {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }

    double avg = sum / antof.size();

    os << "antOF Statistics:" << std::endl;
    os << "  Minimum: " << std::fixed << std::setprecision(6) << min_val << std::endl;
    os << "  Maximum: " << std::fixed << std::setprecision(6) << max_val << std::endl;
    os << "  Average: " << std::fixed << std::setprecision(6) << avg << std::endl;
    os << "  Count: " << antof.size() << std::endl;
}

// Functions for printing ThreadSafeQueue
void print_thread_safe_queue(const ThreadSafeQueue& queue, std::ostream& os = std::cout, const std::string& queue_name = "Queue") {
    std::lock_guard<std::mutex> lock(queue.mtx);

    os << "=== THREAD SAFE QUEUE: " << queue_name << " ===" << std::endl;
    os << "Queue size: " << queue.queue.size() << std::endl;
    os << "Stopped status: " << (queue.stopped ? "yes" : "no") << std::endl;

    if (!queue.queue.empty()) {
        os << "Queue contents:" << std::endl;
        std::queue<IterationData> temp_queue = queue.queue;
        int index = 0;

        while (!temp_queue.empty()) {
            const IterationData& data = temp_queue.front();
            os << "  [" << index << "] Iteration: " << data.iteration
                << ", ants: " << data.ants_count
                << ", minOf: " << data.minOf
                << ", maxOf: " << data.maxOf << std::endl;
            temp_queue.pop();
            index++;
        }
    }
    else {
        os << "Queue is empty" << std::endl;
    }
    os << std::endl;
}

// Functions for printing ACOData
void print_acodata(const ACOData& aco, std::ostream& os = std::cout, bool print_vectors = true) {
    os << "=== ACO DATA ===" << std::endl;

    // Basic data
    os << "--- Execution Control ---" << std::endl;
    os << "stop_requested: " << (aco.stop_requested.load() ? "yes" : "no") << std::endl;
    os << "current_iteration: " << aco.current_iteration.load() << std::endl;

    // Results
    os << "--- Results ---" << std::endl;
    os << "minOf: " << aco.minOf << std::endl;
    os << "maxOf: " << aco.maxOf << std::endl;
    os << "global_minOf: " << aco.global_minOf << std::endl;
    os << "global_maxOf: " << aco.global_maxOf << std::endl;
    os << "hash_failures: " << aco.kol_hash_fail << std::endl;

    // Execution time
    os << "--- Execution Time (ms) ---" << std::endl;
    os << "CPU_all: " << aco.Time_CPU_all << std::endl;
    os << "CPU_prob: " << aco.Time_CPU_prob << std::endl;
    os << "CPU_wait: " << aco.Time_CPU_wait << std::endl;
    os << "CPU_update: " << aco.Time_CPU_update << std::endl;
    os << "GPU_all: " << aco.Time_GPU_all << std::endl;
    os << "GPU: " << aco.Time_GPU << std::endl;
    os << "GPU_function: " << aco.Time_GPU_function << std::endl;
    os << "OMP_all: " << aco.Time_OMP_all << std::endl;
    os << "OMP_function: " << aco.Time_OMP_function << std::endl;

    // Load balancing
    os << "--- Load Balancing ---" << std::endl;
    os << "gpu_ants_count: " << aco.gpu_ants_count.load() << std::endl;
    os << "OMP_ants_count: " << aco.OMP_ants_count.load() << std::endl;

    // Data vectors
    os << "--- Data Vectors ---" << std::endl;
    os << "parametr_value: " << aco.parametr_value.size() << " elements" << std::endl;
    os << "current_pheromon: " << aco.current_pheromon.size() << " elements" << std::endl;
    os << "current_kol_enter: " << aco.current_kol_enter.size() << " elements" << std::endl;
    os << "norm_matrix_probability: " << aco.norm_matrix_probability.size() << " elements" << std::endl;
    os << "ant_parametr: " << aco.ant_parametr.size() << " elements" << std::endl;
    os << "antOF: " << aco.antOF.size() << " elements" << std::endl;

    std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
    for (int i = 0; i < PARAMETR_SIZE; ++i) {
        for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
            std::cout << aco.parametr_value[i * MAX_VALUE_SIZE + j] << "("
                << aco.current_pheromon[i * MAX_VALUE_SIZE + j] << ", "
                << aco.current_kol_enter[i * MAX_VALUE_SIZE + j] << "-> "
                << aco.norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") ";
        }
        std::cout << std::endl;
    }

    std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
    for (int i = 0; i < ANT_SIZE; ++i) {
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            std::cout << aco.ant_parametr[i * PARAMETR_SIZE + j] << " ";
        }
        std::cout << "-> " << aco.antOF[i] << std::endl;
    }

    // Queues
    os << "--- Queues ---" << std::endl;
    print_thread_safe_queue(aco.gpu_to_cpu_queue, os, "GPU -> CPU");
    print_thread_safe_queue(aco.cpu_to_gpu_queue, os, "CPU -> GPU");
    print_thread_safe_queue(aco.cpu_to_OMP_queue, os, "CPU -> OMP");
    print_thread_safe_queue(aco.OMP_to_cpu_queue, os, "OMP -> CPU");

    os << std::endl;
}

// Function for brief ACOData output (key metrics only)
void print_acodata_summary(const ACOData& aco, std::ostream& os = std::cout) {
    os << "=== ACO DATA SUMMARY ===" << std::endl;
    os << "Iteration: " << aco.current_iteration.load()
        << " | Global Min: " << std::fixed << std::setprecision(6) << aco.global_minOf
        << " | Global Max: " << std::fixed << std::setprecision(6) << aco.global_maxOf
        << " | Hash fails: " << aco.kol_hash_fail
        << " | Stop: " << (aco.stop_requested.load() ? "Y" : "N") << std::endl;

    os << "Time CPU: " << std::fixed << std::setprecision(2) << aco.Time_CPU_all << "ms"
        << " | GPU: " << aco.Time_GPU_all << "ms"
        << " | OMP: " << aco.Time_OMP_all << "ms" << std::endl;

    os << "Ants GPU: " << aco.gpu_ants_count.load()
        << " | OMP: " << aco.OMP_ants_count.load() << std::endl;
}

// Function for printing execution progress
void print_progress(const ACOData& aco, int total_iterations = KOL_ITERATION) {
    int current_iter = aco.current_iteration.load();
    double progress = (static_cast<double>(current_iter) / total_iterations) * 100.0;

    std::cout << "\rProgress: " << current_iter << "/" << total_iterations
        << " (" << std::fixed << std::setprecision(1) << progress << "%)"
        << " | Min: " << std::setprecision(6) << aco.global_minOf
        << " | Max: " << aco.global_maxOf
        << " | Active CUDA: " << active_cuda_tasks.load()
        << std::flush;
}

// Function for printing performance metrics
void print_performance_breakdown(const ACOData& aco, std::ostream& os = std::cout) {
    double total_time = aco.Time_CPU_all + aco.Time_GPU_all + aco.Time_OMP_all;

    os << "=== PERFORMANCE BREAKDOWN ===" << std::endl;
    os << "Total time: " << std::fixed << std::setprecision(2) << total_time << " ms" << std::endl;
    os << "CPU: " << aco.Time_CPU_all << " ms (" << (aco.Time_CPU_all / total_time) * 100 << "%)" << std::endl;
    os << "  - Probability calc: " << aco.Time_CPU_prob << " ms" << std::endl;
    os << "  - Waiting: " << aco.Time_CPU_wait << " ms" << std::endl;
    os << "  - Update: " << aco.Time_CPU_update << " ms" << std::endl;
    os << "GPU: " << aco.Time_GPU_all << " ms (" << (aco.Time_GPU_all / total_time) * 100 << "%)" << std::endl;
    os << "  - Function: " << aco.Time_GPU_function << " ms" << std::endl;
    os << "OMP: " << aco.Time_OMP_all << " ms (" << (aco.Time_OMP_all / total_time) * 100 << "%)" << std::endl;
    os << "  - Function: " << aco.Time_OMP_function << " ms" << std::endl;
}

// ----------------- Быстрая хэш-функция -----------------
inline unsigned long long fastHashFunction(unsigned long long key) {
    // Упрощенная и более эффективная хэш-функция
    key = (~key) + (key << 18);
    key = key ^ (key >> 31);
    key = key * 21;
    key = key ^ (key >> 11);
    key = key + (key << 6);
    key = key ^ (key >> 22);

    return key & (HASH_TABLE_SIZE - 1);
}

// ----------------- Генерация ключа из пути агента -----------------
inline unsigned long long generateKey(const int* agent_path) {
    // Полиномиальное хэширование для лучшего распределения
    const unsigned long long prime = 1099511628211ULL;
    unsigned long long key = 14695981039346656037ULL;

    for (int i = 0; i < PARAMETR_SIZE; i++) {
        key ^= static_cast<unsigned long long>(agent_path[i]);
        key *= prime;
    }

    return key;
}

// ----------------- Альтернативная генерация ключа -----------------
inline unsigned long long generateKeySimple(const int* agent_path) {
    unsigned long long key = 0;

    for (int i = 0; i < PARAMETR_SIZE; i++) {
        key = key * MAX_VALUE_SIZE + agent_path[i];
    }

    return key;
}

// Структура для атомарной хэш-таблицы
typedef struct {
    std::atomic<unsigned long long> key;
    std::atomic<double> value;
    std::atomic<int> timestamp;
} AtomicHashEntry;

// Глобальная атомарная хэш-таблица для OMP
AtomicHashEntry* atomicHashTable = nullptr;

// Структура для шардированной хэш-таблицы
struct ShardedHashTable {
    AtomicHashEntry** tables;
    int num_shards;
    int size_per_shard;

    ShardedHashTable(int shards, int size_per_shard) : num_shards(shards), size_per_shard(size_per_shard) {
        tables = new AtomicHashEntry * [shards];
        for (int i = 0; i < shards; i++) {
            tables[i] = new AtomicHashEntry[size_per_shard];
        }
    }

    ~ShardedHashTable() {
        for (int i = 0; i < num_shards; i++) {
            delete[] tables[i];
        }
        delete[] tables;
    }
};

// Глобальная шардированная хэш-таблица
ShardedHashTable* shardedHashTable = nullptr;

// ----------------- Инициализация атомарной хэш-таблицы -----------------
void initializeAtomicHashTable(AtomicHashEntry* hashTable, int size) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        hashTable[i].key.store(ZERO_HASH, std::memory_order_relaxed);
        hashTable[i].value.store(0.0, std::memory_order_relaxed);
        hashTable[i].timestamp.store(0, std::memory_order_relaxed);
    }
}

// ----------------- Инициализация шардированной хэш-таблицы -----------------
void initializeShardedHashTable(ShardedHashTable* table) {
#pragma omp parallel for schedule(static)
    for (int shard = 0; shard < table->num_shards; shard++) {
        for (int i = 0; i < table->size_per_shard; i++) {
            table->tables[shard][i].key.store(ZERO_HASH, std::memory_order_relaxed);
            table->tables[shard][i].value.store(0.0, std::memory_order_relaxed);
            table->tables[shard][i].timestamp.store(0, std::memory_order_relaxed);
        }
    }
}

// ----------------- Функция для определения шарда -----------------
inline int get_shard(const int* agent_path) {
    // Простая хэш-функция для определения шарда
    unsigned long long hash = 0;
    for (int i = 0; i < PARAMETR_SIZE; i++) {
        hash = hash * 31 + agent_path[i];
    }
    return hash % HASH_TABLE_SHARDS;
}

// ----------------- Atomic hash lookup -----------------
double atomic_hash_lookup(AtomicHashEntry* hashTable, const int* agent_path, int iteration) {
    unsigned long long key = generateKey(agent_path);
    unsigned long long idx = fastHashFunction(key);
    const unsigned long long mask = HASH_TABLE_SIZE - 1;

    // Atomic read with acquire semantics
    for (int i = 0; i < std::min(MAX_PROBES, 8); ++i) {
        unsigned long long current_idx = (idx + i) & mask;
        unsigned long long current_key = hashTable[current_idx].key.load(std::memory_order_acquire);

        if (current_key == ZERO_HASH) {
            return ZERO_HASH_RESULT;
        }
        if (current_key == key) {
            // Update timestamp - this is safe as it's just metadata
            hashTable[current_idx].timestamp.store(iteration, std::memory_order_relaxed);
            double value = hashTable[current_idx].value.load(std::memory_order_relaxed);

            // Ensure we read the value after confirming the key
            std::atomic_thread_fence(std::memory_order_acquire);

            return value;
        }
    }

    return ZERO_HASH_RESULT;
}

// ----------------- Sharded hash lookup -----------------
double sharded_hash_lookup(ShardedHashTable& table, const int* agent_path, int iteration) {
    int shard = get_shard(agent_path);
    unsigned long long key = generateKey(agent_path);
    unsigned long long idx = fastHashFunction(key) % table.size_per_shard;
    AtomicHashEntry* hash_table = table.tables[shard];

    // Atomic read with acquire semantics
    for (int i = 0; i < std::min(MAX_PROBES, 8); ++i) {
        unsigned long long current_idx = (idx + i) % table.size_per_shard;
        unsigned long long current_key = hash_table[current_idx].key.load(std::memory_order_acquire);

        if (current_key == ZERO_HASH) {
            return ZERO_HASH_RESULT;
        }
        if (current_key == key) {
            // Update timestamp
            hash_table[current_idx].timestamp.store(iteration, std::memory_order_relaxed);
            double value = hash_table[current_idx].value.load(std::memory_order_relaxed);

            std::atomic_thread_fence(std::memory_order_acquire);
            return value;
        }
    }

    return ZERO_HASH_RESULT;
}

// ----------------- Thread-safe hash store -----------------
bool atomic_hash_store(AtomicHashEntry* hashTable, const int* agent_path, double value, int iteration) {
    unsigned long long key = generateKey(agent_path);
    unsigned long long idx = fastHashFunction(key);
    const unsigned long long mask = HASH_TABLE_SIZE - 1;

    // Double-check pattern to avoid race conditions
    for (int i = 0; i < std::min(MAX_PROBES, 4); ++i) {
        unsigned long long current_idx = (idx + i) & mask;

        // First, check if the key already exists (fast path)
        unsigned long long current_key = hashTable[current_idx].key.load(std::memory_order_acquire);
        if (current_key == key) {
            // Key exists, just update the value and timestamp
            hashTable[current_idx].value.store(value, std::memory_order_relaxed);
            hashTable[current_idx].timestamp.store(iteration, std::memory_order_release);
            return true;
        }

        // Try to acquire the slot with CAS
        unsigned long long expected = ZERO_HASH;
        if (hashTable[current_idx].key.compare_exchange_strong(expected, key,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
            // Successfully acquired the slot
            hashTable[current_idx].value.store(value, std::memory_order_relaxed);
            hashTable[current_idx].timestamp.store(iteration, std::memory_order_release);

            // Ensure all writes are visible to other threads
            std::atomic_thread_fence(std::memory_order_release);
            return true;
        }

        // If CAS failed but the key matches, update the existing entry
        if (expected == key) {
            hashTable[current_idx].value.store(value, std::memory_order_relaxed);
            hashTable[current_idx].timestamp.store(iteration, std::memory_order_release);
            return true;
        }
    }

    return false;
}

// ----------------- Sharded hash store -----------------
bool sharded_hash_store(ShardedHashTable& table, const int* agent_path, double value, int iteration) {
    int shard = get_shard(agent_path);
    unsigned long long key = generateKey(agent_path);
    unsigned long long idx = fastHashFunction(key) % table.size_per_shard;
    AtomicHashEntry* hash_table = table.tables[shard];

    // Double-check pattern to avoid race conditions
    for (int i = 0; i < std::min(MAX_PROBES, 4); ++i) {
        unsigned long long current_idx = (idx + i) % table.size_per_shard;

        // First, check if the key already exists (fast path)
        unsigned long long current_key = hash_table[current_idx].key.load(std::memory_order_acquire);
        if (current_key == key) {
            // Key exists, just update the value and timestamp
            hash_table[current_idx].value.store(value, std::memory_order_relaxed);
            hash_table[current_idx].timestamp.store(iteration, std::memory_order_release);
            return true;
        }

        // Try to acquire the slot with CAS
        unsigned long long expected = ZERO_HASH;
        if (hash_table[current_idx].key.compare_exchange_strong(expected, key,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
            // Successfully acquired the slot
            hash_table[current_idx].value.store(value, std::memory_order_relaxed);
            hash_table[current_idx].timestamp.store(iteration, std::memory_order_release);

            std::atomic_thread_fence(std::memory_order_release);
            return true;
        }

        // If CAS failed but the key matches, update the existing entry
        if (expected == key) {
            hash_table[current_idx].value.store(value, std::memory_order_relaxed);
            hash_table[current_idx].timestamp.store(iteration, std::memory_order_release);
            return true;
        }
    }

    return false;
}

// ----------------- Обновленные функции для OMP -----------------
double getCachedResultAtomic(AtomicHashEntry* hashTable, const int* agent_path, int iteration) {
    return atomic_hash_lookup(hashTable, agent_path, iteration);
}

bool saveToCacheAtomic(AtomicHashEntry* hashTable, const int* agent_path, double value, int iteration) {
    return atomic_hash_store(hashTable, agent_path, value, iteration);
}

double getCachedResultSharded(ShardedHashTable& table, const int* agent_path, int iteration) {
    return sharded_hash_lookup(table, agent_path, iteration);
}

bool saveToCacheSharded(ShardedHashTable& table, const int* agent_path, double value, int iteration) {
    return sharded_hash_store(table, agent_path, value, iteration);
}

//Функции для взаимодействия с OMP хэш-таблицей
// ----------------- Инициализация хэш-таблицы -----------------
void initializeHashTable_non_cuda(HashEntry* hashTable, int size) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        hashTable[i].key = ZERO_HASH;
        hashTable[i].value = 0.0;
    }
}

// ----------------- Очистка хэш-таблицы -----------------
void clearHashTable(HashEntry* hashTable, int size) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        hashTable[i].key = ZERO_HASH;
        hashTable[i].value = 0.0;
    }
}

// ----------------- Поиск в хэш-таблице -----------------
double getCachedResultOptimized_non_cuda(HashEntry* __restrict hashTable, const int* __restrict agent_path, int bx) {
    unsigned long long key = generateKey(agent_path);
    unsigned long long idx = fastHashFunction(key);
    const unsigned long long mask = HASH_TABLE_SIZE - 1;

    // Поиск с линейным probing (более эффективно для кэша)
    for (int i = 0; i < MAX_PROBES; i++) {
        unsigned long long new_idx = (idx + i) & mask;
        if (hashTable[new_idx].key == key) {
            return hashTable[new_idx].value; // Найдено
        }
        if (hashTable[new_idx].key == ZERO_HASH) {
            return ZERO_HASH_RESULT; // Не найдено - пустой слот
        }
    }

    return ZERO_HASH_RESULT; // Не найдено после всех проб
}

// ----------------- Сохранение в хэш-таблицу -----------------
bool saveToCacheOptimized_non_cuda(HashEntry* __restrict hashTable, const int* __restrict agent_path, int bx, double value) {
    unsigned long long key = generateKey(agent_path);
    unsigned long long idx = fastHashFunction(key);
    const unsigned long long mask = HASH_TABLE_SIZE - 1;

    // Поиск пустого слота или обновление существующего с линейным probing
    for (int i = 0; i < MAX_PROBES; i++) {
        unsigned long long new_idx = (idx + i) & mask;

        if (hashTable[new_idx].key == ZERO_HASH || hashTable[new_idx].key == key) {
            // Найден пустой слот или существующий ключ
            hashTable[new_idx].key = key;
            hashTable[new_idx].value = value;
            return true;
        }
    }

    // Не удалось найти слот
    return false;
}

// ----------------- Статистика хэш-таблицы -----------------
void printHashTableStats(const HashEntry* hashTable, int size) {
    int used_slots = 0;

#pragma omp parallel for reduction(+:used_slots) schedule(static)
    for (int i = 0; i < size; i++) {
        if (hashTable[i].key != ZERO_HASH) {
            used_slots++;
        }
    }

    double load_factor = static_cast<double>(used_slots) / size;

    std::cout << "=== Hash Table Statistics ===" << std::endl;
    std::cout << "Size: " << size << std::endl;
    std::cout << "Used slots: " << used_slots << std::endl;
    std::cout << "Load factor: " << (load_factor * 100.0) << "%" << std::endl;
    std::cout << "Max probes: " << MAX_PROBES << std::endl;
    std::cout << "=============================" << std::endl;
}

// ----------------- Коэффициент заполнения -----------------
double getHashTableLoadFactor(const HashEntry* hashTable, int size) {
    int used_slots = 0;

#pragma omp parallel for reduction(+:used_slots) schedule(static)
    for (int i = 0; i < size; i++) {
        if (hashTable[i].key != ZERO_HASH) {
            used_slots++;
        }
    }

    return static_cast<double>(used_slots) / size;
}


// Функции для работы с ThreadSafeQueue
void queue_push(ThreadSafeQueue* queue, const IterationData& data) {
    std::lock_guard<std::mutex> lock(queue->mtx);
    queue->queue.push(data);
    queue->cv.notify_one();
}

bool queue_pop(ThreadSafeQueue* queue, IterationData& data) {
    std::unique_lock<std::mutex> lock(queue->mtx);
    queue->cv.wait(lock, [queue]() { return !queue->queue.empty() || queue->stopped; });

    if (queue->stopped && queue->queue.empty()) return false;

    data = std::move(queue->queue.front());
    queue->queue.pop();
    return true;
}

void queue_stop(ThreadSafeQueue* queue) {
    std::lock_guard<std::mutex> lock(queue->mtx);
    queue->stopped = true;
    queue->cv.notify_all();
}

size_t queue_size(const ThreadSafeQueue* queue) {
    std::lock_guard<std::mutex> lock(queue->mtx);
    return queue->queue.size();
}


// Вспомогательные функции
inline double probability_formula_non_cuda(double pheromon, double kol_enter) {
    return (kol_enter != 0.0 && pheromon != 0.0) ? (1.0 / kol_enter + pheromon) : 0.0;
}

// Callback функция для CUDA
void cuda_completion_callback(double* results, int size, int iteration) {
    double min_val = results[0];
    double max_val = results[0];
    double sum = 0.0;

    for (int i = 0; i < size; i++) {
        if (results[i] < min_val) min_val = results[i];
        if (results[i] > max_val) max_val = results[i];
        sum += results[i];
    }

    int iter_num = completed_iterations.fetch_add(1) + 1;
    successful_iterations.fetch_add(1);

#if (PRINT_INFORMATION)
    if (iteration % KOL_PROGON_STATISTICS == 0) {
        std::cout << "[Callback] Iteration " << iteration << " (Total: " << iter_num
            << "): Min=" << min_val << ", Max=" << max_val
            << ", Avg=" << (sum / size) << std::endl;
    }
#endif

    active_cuda_tasks.fetch_sub(1);
}

// Функция загрузки параметров из файла
bool load_matrix(const std::string& filename, double* parametr_value, double* pheromon_value, double* kol_enter_value) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Don't open file!" << std::endl;
        return false;
    }

#if (PRINT_INFORMATION)
    std::cout << "[Main] Loading parameters from file: " << filename << std::endl;
#endif

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
                parametr_value[k] = 0.0;
                kol_enter_value[k] = 0.0;
            }
        }
    }
    infile.close();

#if (PRINT_INFORMATION)
    std::cout << "[Main] Parameters loaded successfully!" << std::endl;
#endif

    return true;
}

// Функции для работы с ACO
void aco_init(ACOData* aco) {
    aco->norm_matrix_probability.resize(MAX_VALUE_SIZE * PARAMETR_SIZE);
    aco->ant_parametr.resize(PARAMETR_SIZE * ANT_SIZE);
    aco->antOF.resize(ANT_SIZE);
    aco->OMP_ants_count.store(static_cast<int>(ANT_SIZE * INITIAL_CPU_ANTS_RATIO));
    aco->gpu_ants_count.store(ANT_SIZE - aco->OMP_ants_count.load());
}

bool aco_initialize_non_hash(ACOData* aco, const std::vector<double>& parametr_value_new, const std::vector<double>& pheromon_value, const std::vector<double>& kol_enter_value) {
#if (PRINT_INFORMATION)
    std::cout << "[Main] Initializing ACO" << std::endl;
#endif

    aco->parametr_value = parametr_value_new;
    aco->current_pheromon = pheromon_value;
    aco->current_kol_enter = kol_enter_value;
    aco->norm_matrix_probability.resize(MAX_VALUE_SIZE * PARAMETR_SIZE);
    aco->ant_parametr.resize(PARAMETR_SIZE * ANT_SIZE);
    aco->antOF.resize(ANT_SIZE);

    aco->global_minOf = 1e9;
    aco->global_maxOf = -1e9;
    aco->kol_hash_fail = 0;

#if (PRINT_INFORMATION)
    std::cout << "[Main] Initializing CUDA..." << std::endl;
#endif

    if (!cuda_initialize_non_hash(aco->parametr_value.data(), aco->current_pheromon.data(), aco->current_kol_enter.data())) {
        std::cerr << "[Main] CUDA initialization failed!" << std::endl;
        return false;
    }

#if (PRINT_INFORMATION)
    const char* version = cuda_get_version();
    std::cout << "[Main] " << version << std::endl;
    std::cout << "[Main] ACO initialized successfully!" << std::endl;
    std::cout << "[Main] Parameters: " << PARAMETR_SIZE << " params, "
        << MAX_VALUE_SIZE << " values, " << ANT_SIZE << " ants" << std::endl;
#endif

    return true;
}

bool aco_initialize_from_file_non_hash(ACOData* aco, const std::string& filename) {
    const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
    std::vector<double> parametr_value(matrix_size);
    std::vector<double> pheromon_value(matrix_size);
    std::vector<double> kol_enter_value(matrix_size);

    if (!load_matrix(filename, parametr_value.data(), pheromon_value.data(), kol_enter_value.data())) {
        std::cerr << "[Main] Failed to load parameters from file: " << filename << std::endl;
        return false;
    }

    return aco_initialize_non_hash(aco, parametr_value, pheromon_value, kol_enter_value);
}

bool aco_initialize(ACOData* aco, const std::vector<double>& parametr_value_new, const std::vector<double>& pheromon_value, const std::vector<double>& kol_enter_value) {
#if (PRINT_INFORMATION)
    std::cout << "[Main] Initializing ACO" << std::endl;
#endif

    aco->parametr_value = parametr_value_new;
    aco->current_pheromon = pheromon_value;
    aco->current_kol_enter = kol_enter_value;
    aco->norm_matrix_probability.resize(MAX_VALUE_SIZE * PARAMETR_SIZE);
    aco->ant_parametr.resize(PARAMETR_SIZE * ANT_SIZE);
    aco->antOF.resize(ANT_SIZE);

    aco->global_minOf = 1e9;
    aco->global_maxOf = -1e9;
    aco->kol_hash_fail = 0;

#if (PRINT_INFORMATION)
    std::cout << "[Main] Initializing CUDA..." << std::endl;
#endif

    if (!cuda_initialize(aco->parametr_value.data(), aco->current_pheromon.data(), aco->current_kol_enter.data())) {
        std::cerr << "[Main] CUDA initialization failed!" << std::endl;
        return false;
    }

#if (PRINT_INFORMATION)
    const char* version = cuda_get_version();
    std::cout << "[Main] " << version << std::endl;
    std::cout << "[Main] ACO initialized successfully!" << std::endl;
    std::cout << "[Main] Parameters: " << PARAMETR_SIZE << " params, "
        << MAX_VALUE_SIZE << " values, " << ANT_SIZE << " ants" << std::endl;
#endif

    return true;
}

bool aco_initialize_from_file(ACOData* aco, const std::string& filename) {
    const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
    std::vector<double> parametr_value(matrix_size);
    std::vector<double> pheromon_value(matrix_size);
    std::vector<double> kol_enter_value(matrix_size);

    if (!load_matrix(filename, parametr_value.data(), pheromon_value.data(), kol_enter_value.data())) {
        std::cerr << "[Main] Failed to load parameters from file: " << filename << std::endl;
        return false;
    }

    return aco_initialize(aco, parametr_value, pheromon_value, kol_enter_value);
}

#if (MAX_VALUE_SIZE==4)
// AVX2-оптимизированное вычисление матрицы вероятностей для MAX_VALUE_SIZE=4
void calculate_probabilities(ACOData* aco) {
    const int total_params = PARAMETR_SIZE;
#pragma omp parallel for schedule(static)
    for (int param = 0; param < total_params; ++param) {
        const int base = param * MAX_VALUE_SIZE;

        // Загружаем 4 значения феромонов в AVX-регистр
        __m256d pheromone_vec = _mm256_loadu_pd(&aco->current_pheromon[base]);
        __m256d visits_vec = _mm256_loadu_pd(&aco->current_kol_enter[base]);

        // Вычисляем сумму феромонов (горизонтальное суммирование)
        __m256d sum_pheromone_vec = _mm256_hadd_pd(pheromone_vec, pheromone_vec);
        double total_pheromone = ((double*)&sum_pheromone_vec)[0] + ((double*)&sum_pheromone_vec)[2];

        if (total_pheromone <= 0.0) {
            // Равномерное распределение
            __m256d uniform_probs = _mm256_set_pd(1.0, 0.75, 0.5, 0.25);
            _mm256_storeu_pd(&aco->norm_matrix_probability[base], uniform_probs);
            continue;
        }

        // Вычисление вероятностей с использованием AVX
        __m256d inv_total_pheromone = _mm256_set1_pd(1.0 / total_pheromone);
        __m256d norm_pheromone = _mm256_mul_pd(pheromone_vec, inv_total_pheromone);

        // Вычисляем вероятности: 1.0/visits + norm_pheromone
        __m256d inv_visits = _mm256_div_pd(_mm256_set1_pd(1.0), visits_vec);
        __m256d temp_probs = _mm256_add_pd(inv_visits, norm_pheromone);

        // Заменяем NaN/Inf на 0.0
        __m256d zero_vec = _mm256_setzero_pd();
        __m256d valid_mask = _mm256_and_pd(
            _mm256_cmp_pd(visits_vec, zero_vec, _CMP_GT_OQ),
            _mm256_cmp_pd(pheromone_vec, zero_vec, _CMP_GT_OQ)
        );
        temp_probs = _mm256_blendv_pd(zero_vec, temp_probs, valid_mask);

        // Суммируем вероятности
        __m256d sum_probs_vec = _mm256_hadd_pd(temp_probs, temp_probs);
        double sum_probs = ((double*)&sum_probs_vec)[0] + ((double*)&sum_probs_vec)[2];

        if (sum_probs > 0.0) {
            // Нормализуем и вычисляем кумулятивные вероятности
            __m256d inv_sum_probs = _mm256_set1_pd(1.0 / sum_probs);
            __m256d norm_probs = _mm256_mul_pd(temp_probs, inv_sum_probs);

            // Вычисляем кумулятивную сумму
            double cumulative = 0.0;
            double probs[4];
            _mm256_storeu_pd(probs, norm_probs);

            for (int i = 0; i < 4; ++i) {
                cumulative += probs[i];
                aco->norm_matrix_probability[base + i] = cumulative;
            }
            aco->norm_matrix_probability[base + 3] = 1.0; // Гарантируем, что последнее значение = 1.0
        }
        else {
            // Равномерное распределение при нулевых вероятностях
            __m256d uniform_probs = _mm256_set_pd(1.0, 0.75, 0.5, 0.25);
            _mm256_storeu_pd(&aco->norm_matrix_probability[base], uniform_probs);
        }
    }
}
#else
void calculate_probabilities(ACOData* aco) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

#ifdef _OPENMP
#pragma omp simd reduction(+:sumVector)
#endif
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += aco->current_pheromon[MAX_VALUE_SIZE * tx + i];
        }

#ifdef _OPENMP
#pragma omp simd
#endif
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = aco->current_pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

#ifdef _OPENMP
#pragma omp simd reduction(+:sumVector)
#endif
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], aco->current_kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }

        aco->norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
        for (int i = 1; i < MAX_VALUE_SIZE; i++) {
            aco->norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + aco->norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
        }
    }
}
#endif
#if (MAX_VALUE_SIZE==4)
// AVX2-оптимизированное обновление феромонов (оптимизированное по памяти)
void update_pheromones_async(ACOData* aco, int iteration) {
    const int total_cells = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int num_agents = ANT_SIZE;

    // Испарение феромонов с AVX
    __m256d ro_vec = _mm256_set1_pd(PARAMETR_RO);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < total_cells; i += 4) {
        if (i + 3 < total_cells) {
            __m256d pheromone_vec = _mm256_loadu_pd(&aco->current_pheromon[i]);
            __m256d evaporated = _mm256_mul_pd(pheromone_vec, ro_vec);
            _mm256_storeu_pd(&aco->current_pheromon[i], evaporated);
        }
        else {
            // Обработка оставшихся элементов
            for (int j = i; j < total_cells; ++j) {
                aco->current_pheromon[j] *= PARAMETR_RO;
            }
        }
    }

    // Обновление посещений и феромонов с оптимизацией памяти
#pragma omp parallel
    {
        // Используем фиксированные массивы вместо векторов для избежания динамического выделения
        const int local_size = total_cells;
        double* local_pheromone = new double[local_size]();
        int* local_visits = new int[local_size]();

#pragma omp for nowait
        for (int agent = 0; agent < num_agents; ++agent) {
            double score = aco->antOF[agent];
            double add_value = 0.0;

#if OPTIMIZE_MIN_1
            add_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT > score) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - score) : 0.0;
#elif OPTIMIZE_MIN_2
            add_value = (score == 0) ? (PARAMETR_Q / 0.0000001) : (PARAMETR_Q / score);
#elif OPTIMIZE_MAX
            add_value = PARAMETR_Q * score;
#endif

            const int* path = &aco->ant_parametr[agent * PARAMETR_SIZE];

            for (int param = 0; param < PARAMETR_SIZE; ++param) {
                int choice = path[param];
                int idx = param * MAX_VALUE_SIZE + choice;
                local_visits[idx]++;

                if (add_value > 0.0) {
                    local_pheromone[idx] += add_value;
                }
            }
        }

        // Слияние локальных данных
#pragma omp critical
        {
            for (int i = 0; i < total_cells; ++i) {
                aco->current_kol_enter[i] += local_visits[i];
                aco->current_pheromon[i] += local_pheromone[i];
            }
        }

        // Освобождаем память
        delete[] local_pheromone;
        delete[] local_visits;
    }
}
#else
void update_pheromones_async(ACOData* aco, int iteration) {
    const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;

#ifdef _OPENMP
#pragma omp parallel for simd
#endif
    for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
        aco->current_pheromon[idx] *= PARAMETR_RO;
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<double> local_pheromon_add(TOTAL_CELLS, 0.0);
        std::vector<int> local_kol_enter_add(TOTAL_CELLS, 0);

#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int i = 0; i < ANT_SIZE; ++i) {
            double agent_of = aco->antOF[i];
#if OPTIMIZE_MIN_2
            double agent_of_reciprocal = (agent_of == 0) ? (PARAMETR_Q / 0.0000001) : (PARAMETR_Q / agent_of);
#endif

            const int* agent_path = &aco->ant_parametr[i * PARAMETR_SIZE];
            for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                int k = agent_path[tx];
                int idx = MAX_VALUE_SIZE * tx + k;

                local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_2
                local_pheromon_add[idx] += agent_of_reciprocal;
#else
                local_pheromon_add[idx] += PARAMETR_Q * agent_of;
#endif
            }
        }

#ifdef _OPENMP
#pragma omp critical
#endif
        {
#ifdef _OPENMP
#pragma omp simd
#endif
            for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
                aco->current_kol_enter[idx] += local_kol_enter_add[idx];
                aco->current_pheromon[idx] += local_pheromon_add[idx];
            }
        }
    }
}
#endif
// ========================= GO_HYBRID_OMP РЕЖИМ =========================

// Параллельная обработка муравьев на CPU с использованием OpenMP
#if (MAX_VALUE_SIZE==4)
// AVX2-оптимизированная генерация агентов (оптимизированная по памяти)
void calculate_ant_paths_omp_atomic(ACOData* aco, int start_ant, int num_ants, std::vector<int>& ant_paths, std::vector<double>& antOF, double& minOf, double& maxOf, int& kol_hash_fail) {
    auto start_time = std::chrono::high_resolution_clock::now();
    minOf = 1e9;
    maxOf = -1e9;
    int local_kol_hash_fail = 0;

#pragma omp parallel reduction(min:minOf) reduction(max:maxOf) reduction(+:local_kol_hash_fail)
    {
        std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count() + omp_get_thread_num());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

#pragma omp for schedule(static)
        for (int ant_index = 0; ant_index < num_ants; ant_index++) {
            int global_ant_index = start_ant + ant_index;
            double agent[PARAMETR_SIZE] = { 0 };
            int value_index = 0; // Значение по умолчанию (первый элемент)
            // AVX-оптимизированная генерация пути муравья
            for (int param_index = 0; param_index < PARAMETR_SIZE; param_index++) {
                if (value_index != 3) {
                    double random_value = dist(rng);
                    const int prob_base = param_index * MAX_VALUE_SIZE;

                    // Загружаем 4 вероятности в AVX-регистр
                    __m256d prob_vec = _mm256_loadu_pd(&aco->norm_matrix_probability[prob_base]);

                    // Сравниваем случайное значение с вероятностями
                    __m256d rand_vec = _mm256_set1_pd(random_value);
                    __m256d cmp_result = _mm256_cmp_pd(rand_vec, prob_vec, _CMP_LE_OQ);

                    int mask = _mm256_movemask_pd(cmp_result);


                    // Находим первый установленный бит
                    if (mask != 0) {
                        value_index = __builtin_ctz(mask);
                    }
                }
                else {
                    value_index = 0;
                }

                int path_index = global_ant_index * PARAMETR_SIZE + param_index;
                ant_paths[path_index] = value_index;
                agent[param_index] = aco->parametr_value[param_index * MAX_VALUE_SIZE + value_index];
            }

            // Проверка кэша и вычисление целевой функции
            double cached_result = getCachedResultOptimized_non_cuda(hashTable,
                &ant_paths[global_ant_index * PARAMETR_SIZE], global_ant_index);

            if (cached_result < 0) {
                // Вычисляем новое значение
                antOF[global_ant_index] = BenchShafferaFunction_omp(agent);

                // Сохраняем в кэш
#pragma omp critical(hash_write)
                {
                    saveToCacheOptimized_non_cuda(hashTable,
                        &ant_paths[global_ant_index * PARAMETR_SIZE], global_ant_index, antOF[global_ant_index]);
                }
            }
            else {
                // Используем кэшированное значение
                local_kol_hash_fail++;
                antOF[global_ant_index] = cached_result;
            }
                // Обновление минимумов/максимумов
                if (antOF[global_ant_index] < minOf) {
                    minOf = antOF[global_ant_index];
                }
                if (antOF[global_ant_index] > maxOf) {
                    maxOf = antOF[global_ant_index];
                }
        }
    }

    kol_hash_fail += local_kol_hash_fail;
    auto end_time = std::chrono::high_resolution_clock::now();
    aco->Time_OMP_function += std::chrono::duration<double, std::milli>(end_time - start_time).count();
}
#else
void calculate_ant_paths_omp_atomic(ACOData* aco, int start_ant, int num_ants, std::vector<int>& ant_paths, std::vector<double>& antOF, double& minOf, double& maxOf, int& kol_hash_fail) {
    auto start_time = std::chrono::high_resolution_clock::now();

    minOf = 1e9;
    maxOf = -1e9;
    int local_kol_hash_fail = 0;

#pragma omp parallel reduction(min:minOf) reduction(max:maxOf) reduction(+:local_kol_hash_fail)
    {
        uint64_t seed = std::chrono::steady_clock::now().time_since_epoch().count() + omp_get_thread_num();

#pragma omp for  schedule(static)
        for (int ant_index = 0; ant_index < num_ants; ant_index++) {
            int global_ant_index = start_ant + ant_index;
            double agent[PARAMETR_SIZE] = { 0 };
            bool valid_solution = true;

            // Генерация пути муравья
            for (int param_index = 0; param_index < PARAMETR_SIZE; param_index++) {
                double randomValue = unified_fast_random(seed);
                int value_index = 0;

                while (valid_solution && value_index < MAX_VALUE_SIZE &&
                    randomValue > aco->norm_matrix_probability[MAX_VALUE_SIZE * param_index + value_index]) {
                    value_index++;
                }

                int path_index = global_ant_index * PARAMETR_SIZE + param_index;
                ant_paths[path_index] = value_index;
                agent[param_index] = aco->parametr_value[param_index * MAX_VALUE_SIZE + value_index];
                valid_solution = (value_index != MAX_VALUE_SIZE - 1);
            }
            double cachedResult = -1.0;
            
            cachedResult = getCachedResultOptimized_non_cuda(hashTable, &ant_paths[global_ant_index * PARAMETR_SIZE], global_ant_index);

            //std::cout << cachedResult << ", ";
            if (cachedResult < 0) {
                antOF[global_ant_index] = BenchShafferaFunction_omp(agent);

#pragma omp critical(hash_write)
                {
                    saveToCacheOptimized_non_cuda(hashTable, &ant_paths[global_ant_index * PARAMETR_SIZE], global_ant_index, antOF[global_ant_index]);
                }

            }
            else {
                local_kol_hash_fail++;
                antOF[global_ant_index] = cachedResult;
            }
            // Обновление минимумов/максимумов
            if (antOF[global_ant_index] < minOf) {
                minOf = antOF[global_ant_index];
            }
            if (antOF[global_ant_index] > maxOf) {
                maxOf = antOF[global_ant_index];
            }
        }
    }

    kol_hash_fail += local_kol_hash_fail;
    auto end_time = std::chrono::high_resolution_clock::now();
    aco->Time_OMP_function += std::chrono::duration<double, std::milli>(end_time - start_time).count();
}
#endif
#if (MAX_VALUE_SIZE==4)
// AVX2-оптимизированная генерация агентов (оптимизированная по памяти)
void calculate_ant_paths_omp_atomic_non_hash(ACOData* aco, int start_ant, int num_ants, std::vector<int>& ant_paths, std::vector<double>& antOF, double& minOf, double& maxOf) {
    auto start_time = std::chrono::high_resolution_clock::now();
    minOf = 1e9;
    maxOf = -1e9;

#pragma omp parallel reduction(min:minOf) reduction(max:maxOf)
    {
        std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count() + omp_get_thread_num());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

#pragma omp for schedule(static)
        for (int ant_index = 0; ant_index < num_ants; ant_index++) {
            int global_ant_index = start_ant + ant_index;
            double agent[PARAMETR_SIZE] = { 0 };
            int value_index = 0; // Значение по умолчанию (первый элемент)
            // AVX-оптимизированная генерация пути муравья
            for (int param_index = 0; param_index < PARAMETR_SIZE; param_index++) {
                if (value_index != 3) {
                    double random_value = dist(rng);
                    const int prob_base = param_index * MAX_VALUE_SIZE;

                    // Загружаем 4 вероятности в AVX-регистр
                    __m256d prob_vec = _mm256_loadu_pd(&aco->norm_matrix_probability[prob_base]);

                    // Сравниваем случайное значение с вероятностями
                    __m256d rand_vec = _mm256_set1_pd(random_value);
                    __m256d cmp_result = _mm256_cmp_pd(rand_vec, prob_vec, _CMP_LE_OQ);

                    int mask = _mm256_movemask_pd(cmp_result);


                    // Находим первый установленный бит
                    if (mask != 0) {
                        value_index = __builtin_ctz(mask);
                    }
                }
                else {
                    value_index = 0;
                }

                int path_index = global_ant_index * PARAMETR_SIZE + param_index;
                ant_paths[path_index] = value_index;
                agent[param_index] = aco->parametr_value[param_index * MAX_VALUE_SIZE + value_index];
            }

            // Вычисляем новое значение
            antOF[global_ant_index] = BenchShafferaFunction_omp(agent);
                // Обновление минимумов/максимумов
                if (antOF[global_ant_index] < minOf) {
                    minOf = antOF[global_ant_index];
                }
                if (antOF[global_ant_index] > maxOf) {
                    maxOf = antOF[global_ant_index];
                }

        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    aco->Time_OMP_function += std::chrono::duration<double, std::milli>(end_time - start_time).count();
}
#else
void calculate_ant_paths_omp_atomic_non_hash(ACOData* aco, int start_ant, int num_ants, std::vector<int>& ant_paths, std::vector<double>& antOF, double& minOf, double& maxOf) {
    auto start_time = std::chrono::high_resolution_clock::now();

    minOf = 1e9;
    maxOf = -1e9;

#pragma omp parallel reduction(min:minOf) reduction(max:maxOf)
    {
        uint64_t seed = std::chrono::steady_clock::now().time_since_epoch().count() + omp_get_thread_num();

#pragma omp for  schedule(static)
        for (int ant_index = 0; ant_index < num_ants; ant_index++) {
            int global_ant_index = start_ant + ant_index;
            double agent[PARAMETR_SIZE] = { 0 };
            bool valid_solution = true;

            // Генерация пути муравья
            for (int param_index = 0; param_index < PARAMETR_SIZE; param_index++) {
                double randomValue = unified_fast_random(seed);
                int value_index = 0;

                while (valid_solution && value_index < MAX_VALUE_SIZE &&
                    randomValue > aco->norm_matrix_probability[MAX_VALUE_SIZE * param_index + value_index]) {
                    value_index++;
                }

                int path_index = global_ant_index * PARAMETR_SIZE + param_index;
                ant_paths[path_index] = value_index;
                agent[param_index] = aco->parametr_value[param_index * MAX_VALUE_SIZE + value_index];
                valid_solution = (value_index != MAX_VALUE_SIZE - 1);
            }
            antOF[global_ant_index] = BenchShafferaFunction_omp(agent);

                // Обновление минимумов/максимумов
                if (antOF[global_ant_index] < minOf) {
                    minOf = antOF[global_ant_index];
                }
                if (antOF[global_ant_index] > maxOf) {
                    maxOf = antOF[global_ant_index];
                }

        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    aco->Time_OMP_function += std::chrono::duration<double, std::milli>(end_time - start_time).count();
}
#endif
void calculate_ant_paths_atomic(ACOData* aco, int start_ant, int num_ants, std::vector<int>& ant_paths, std::vector<double>& antOF, double& minOf, double& maxOf, int& kol_hash_fail) {
    auto start_time = std::chrono::high_resolution_clock::now();

    minOf = 1e9;
    maxOf = -1e9;
    int local_kol_hash_fail = 0;

        uint64_t seed = std::chrono::steady_clock::now().time_since_epoch().count() + omp_get_thread_num();

        for (int ant_index = 0; ant_index < num_ants; ant_index++) {
            int global_ant_index = start_ant + ant_index;
            double agent[PARAMETR_SIZE] = { 0 };
            bool valid_solution = true;

            // Генерация пути муравья
            for (int param_index = 0; param_index < PARAMETR_SIZE; param_index++) {
                double randomValue = unified_fast_random(seed);
                int value_index = 0;

                while (valid_solution && value_index < MAX_VALUE_SIZE &&
                    randomValue > aco->norm_matrix_probability[MAX_VALUE_SIZE * param_index + value_index]) {
                    value_index++;
                }

                int path_index = global_ant_index * PARAMETR_SIZE + param_index;
                ant_paths[path_index] = value_index;
                agent[param_index] = aco->parametr_value[param_index * MAX_VALUE_SIZE + value_index];
                valid_solution = (value_index != MAX_VALUE_SIZE - 1);
            }
            double cachedResult = -1.0;

            cachedResult = getCachedResultOptimized_non_cuda(hashTable, &ant_paths[global_ant_index * PARAMETR_SIZE], global_ant_index);
            //std::cout << cachedResult << ", ";
            if (cachedResult < 0) {
                antOF[global_ant_index] = BenchShafferaFunction_omp(agent);
                saveToCacheOptimized_non_cuda(hashTable, &ant_paths[global_ant_index * PARAMETR_SIZE], global_ant_index, antOF[global_ant_index]);
            }
            else {
                local_kol_hash_fail++;
                antOF[global_ant_index] = cachedResult;
            }

            // Обновление минимумов/максимумов
            if (antOF[global_ant_index] < minOf) {
                minOf = antOF[global_ant_index];
            }
            if (antOF[global_ant_index] > maxOf) {
                maxOf = antOF[global_ant_index];
            }
        }
    kol_hash_fail += local_kol_hash_fail;
    auto end_time = std::chrono::high_resolution_clock::now();
    aco->Time_OMP_function += std::chrono::duration<double, std::milli>(end_time - start_time).count();
}
void calculate_ant_paths_atomic_non_hash(ACOData* aco, int start_ant, int num_ants, std::vector<int>& ant_paths, std::vector<double>& antOF, double& minOf, double& maxOf) {
    auto start_time = std::chrono::high_resolution_clock::now();
    minOf = 1e9;
    maxOf = -1e9;
    int local_kol_hash_fail = 0;
        uint64_t seed = std::chrono::steady_clock::now().time_since_epoch().count() + omp_get_thread_num();
        for (int ant_index = 0; ant_index < num_ants; ant_index++) {
            int global_ant_index = start_ant + ant_index;
            double agent[PARAMETR_SIZE] = { 0 };
            bool valid_solution = true;

            // Генерация пути муравья
            for (int param_index = 0; param_index < PARAMETR_SIZE; param_index++) {
                double randomValue = unified_fast_random(seed);
                int value_index = 0;

                while (valid_solution && value_index < MAX_VALUE_SIZE &&
                    randomValue > aco->norm_matrix_probability[MAX_VALUE_SIZE * param_index + value_index]) {
                    value_index++;
                }

                int path_index = global_ant_index * PARAMETR_SIZE + param_index;
                ant_paths[path_index] = value_index;
                agent[param_index] = aco->parametr_value[param_index * MAX_VALUE_SIZE + value_index];
                valid_solution = (value_index != MAX_VALUE_SIZE - 1);
            }
            antOF[global_ant_index] = BenchShafferaFunction_omp(agent);

            // Обновление минимумов/максимумов
            if (antOF[global_ant_index] < minOf) {
                minOf = antOF[global_ant_index];
            }
            if (antOF[global_ant_index] > maxOf) {
                maxOf = antOF[global_ant_index];
            }
        }
    auto end_time = std::chrono::high_resolution_clock::now();
    aco->Time_OMP_function += std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

// ----------------- Обновленные функции расчета путей муравьев -----------------
void calculate_ant_paths_omp_atomic_new(ACOData* aco, int start_ant, int num_ants, std::vector<int>& ant_paths,  std::vector<double>& antOF, double& minOf, double& maxOf, int& kol_hash_fail) {
    auto start_time = std::chrono::high_resolution_clock::now();

    minOf = 1e9;
    maxOf = -1e9;
    int local_kol_hash_fail = 0;

#pragma omp parallel reduction(min:minOf) reduction(max:maxOf) reduction(+:local_kol_hash_fail)
    {
        uint64_t seed = std::chrono::steady_clock::now().time_since_epoch().count() + omp_get_thread_num();

#pragma omp for schedule(static)
        for (int ant_index = 0; ant_index < num_ants; ant_index++) {
            int global_ant_index = start_ant + ant_index;
            double agent[PARAMETR_SIZE] = { 0 };
            bool valid_solution = true;

            // Генерация пути муравья
            for (int param_index = 0; param_index < PARAMETR_SIZE; param_index++) {
                double randomValue = unified_fast_random(seed);
                int value_index = 0;

                while (valid_solution && value_index < MAX_VALUE_SIZE &&
                    randomValue > aco->norm_matrix_probability[MAX_VALUE_SIZE * param_index + value_index]) {
                    value_index++;
                }

                int path_index = global_ant_index * PARAMETR_SIZE + param_index;
                ant_paths[path_index] = value_index;
                agent[param_index] = aco->parametr_value[param_index * MAX_VALUE_SIZE + value_index];
                valid_solution = (value_index != MAX_VALUE_SIZE - 1);
            }

            double cachedResult = getCachedResultAtomic(atomicHashTable, &ant_paths[global_ant_index * PARAMETR_SIZE], aco->current_iteration.load());

            if (cachedResult == ZERO_HASH_RESULT) {
                antOF[global_ant_index] = BenchShafferaFunction_omp(agent);
                saveToCacheAtomic(atomicHashTable, &ant_paths[global_ant_index * PARAMETR_SIZE],
                    antOF[global_ant_index], aco->current_iteration.load());
            }
            else {
                local_kol_hash_fail++;
                antOF[global_ant_index] = cachedResult;
            }

            // Обновление минимумов/максимумов
            if (antOF[global_ant_index] < minOf) {
                minOf = antOF[global_ant_index];
            }
            if (antOF[global_ant_index] > maxOf) {
                maxOf = antOF[global_ant_index];
            }
        }
    }

    kol_hash_fail += local_kol_hash_fail;
    auto end_time = std::chrono::high_resolution_clock::now();
    aco->Time_OMP_function += std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

// ----------------- Обновленная функция для шардированной таблицы -----------------
void calculate_ant_paths_omp_sharded(ACOData* aco, int start_ant, int num_ants, std::vector<int>& ant_paths, std::vector<double>& antOF, double& minOf, double& maxOf, int& kol_hash_fail) {
    auto start_time = std::chrono::high_resolution_clock::now();

    minOf = 1e9;
    maxOf = -1e9;
    int local_kol_hash_fail = 0;

#pragma omp parallel reduction(min:minOf) reduction(max:maxOf) reduction(+:local_kol_hash_fail)
    {
        uint64_t seed = std::chrono::steady_clock::now().time_since_epoch().count() + omp_get_thread_num();

#pragma omp for schedule(static)
        for (int ant_index = 0; ant_index < num_ants; ant_index++) {
            int global_ant_index = start_ant + ant_index;
            double agent[PARAMETR_SIZE] = { 0 };
            bool valid_solution = true;

            // Генерация пути муравья
            for (int param_index = 0; param_index < PARAMETR_SIZE; param_index++) {
                double randomValue = unified_fast_random(seed);
                int value_index = 0;

                while (valid_solution && value_index < MAX_VALUE_SIZE &&
                    randomValue > aco->norm_matrix_probability[MAX_VALUE_SIZE * param_index + value_index]) {
                    value_index++;
                }

                int path_index = global_ant_index * PARAMETR_SIZE + param_index;
                ant_paths[path_index] = value_index;
                agent[param_index] = aco->parametr_value[param_index * MAX_VALUE_SIZE + value_index];
                valid_solution = (value_index != MAX_VALUE_SIZE - 1);
            }

            double cachedResult = getCachedResultSharded(*shardedHashTable, &ant_paths[global_ant_index * PARAMETR_SIZE],
                aco->current_iteration.load());

            if (cachedResult == ZERO_HASH_RESULT) {
                antOF[global_ant_index] = BenchShafferaFunction_omp(agent);
                saveToCacheSharded(*shardedHashTable, &ant_paths[global_ant_index * PARAMETR_SIZE],
                    antOF[global_ant_index], aco->current_iteration.load());
            }
            else {
                local_kol_hash_fail++;
                antOF[global_ant_index] = cachedResult;
            }

            // Обновление минимумов/максимумов
            if (antOF[global_ant_index] < minOf) {
                minOf = antOF[global_ant_index];
            }
            if (antOF[global_ant_index] > maxOf) {
                maxOf = antOF[global_ant_index];
            }
        }
    }

    kol_hash_fail += local_kol_hash_fail;
    auto end_time = std::chrono::high_resolution_clock::now();
    aco->Time_OMP_function += std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

void OMP_thread_function(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "   [OMP ant Thread] Started" << std::endl;
#endif

    auto start_time = std::chrono::high_resolution_clock::now();
    auto start_time_wait = std::chrono::high_resolution_clock::now();
    while (!aco->stop_requested.load()) {
        IterationData cpu_data;
        if (!queue_pop(&aco->cpu_to_OMP_queue, cpu_data)) {
            break;
        }
        auto end_time_wait = std::chrono::high_resolution_clock::now();
        aco->Time_OMP_wait += std::chrono::duration<double, std::milli>(end_time_wait - start_time_wait).count();
#if (PRINT_INFORMATION)
        std::cout << "   [OMP ant Thread] queue_pop(&aco->cpu_to_OMP_queue, cpu_data)" << std::endl;
#endif
        
        // Выделяем память для результатов
        std::vector<int> omp_ant_paths(cpu_data.ants_count * PARAMETR_SIZE);
        std::vector<double> omp_antOF(cpu_data.ants_count);

        double minOf = 1e9;
        double maxOf = -1e9;
        int kol_hash_fail = 0;
        // Параллельная обработка муравьев на CPU
        calculate_ant_paths_omp_atomic(aco, 0, cpu_data.ants_count, omp_ant_paths, omp_antOF, minOf, maxOf, kol_hash_fail);
#if (PRINT_INFORMATION)
        std::cout << "   [OMP ant Thread] END calculate_ant_paths_omp_atomic" << std::endl;
#endif
        // Отправляем результаты обратно
        IterationData omp_data;
        omp_data.iteration = cpu_data.iteration;
        omp_data.norm_matrix_probability = cpu_data.norm_matrix_probability; // Сохраняем входные данные
        omp_data.ant_parametr = std::move(omp_ant_paths);
        omp_data.antOF = std::move(omp_antOF);
        omp_data.minOf = minOf;
        omp_data.maxOf = maxOf;
        omp_data.kol_hash_fail = kol_hash_fail;
        omp_data.ants_count = cpu_data.ants_count;

        start_time_wait = std::chrono::high_resolution_clock::now();
        queue_push(&aco->OMP_to_cpu_queue, omp_data);
#if (PRINT_INFORMATION)
        std::cout << "   [OMP ant Thread] queue_push(&aco->OMP_to_cpu_queue, omp_data)" << std::endl;
#endif
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    aco->Time_OMP_all = std::chrono::duration<double, std::milli>(end_time - start_time).count();

#if (PRINT_INFORMATION)
    std::cout << "   [OMP Optimized Thread] Finished" << std::endl;
#endif
}
void OMP_thread_function_non_hash(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "[OMP ant Thread] Started" << std::endl;
#endif

    auto start_time = std::chrono::high_resolution_clock::now();
    auto start_time_wait = std::chrono::high_resolution_clock::now();
    while (!aco->stop_requested.load()) {
        IterationData cpu_data;
        if (!queue_pop(&aco->cpu_to_OMP_queue, cpu_data)) {
            break;
        }
        auto end_time_wait = std::chrono::high_resolution_clock::now();
        aco->Time_OMP_wait += std::chrono::duration<double, std::milli>(end_time_wait - start_time_wait).count();
#if (PRINT_INFORMATION)
        std::cout << "[OMP ant Thread] queue_pop(&aco->cpu_to_OMP_queue, cpu_data)" << std::endl;
#endif

        // Выделяем память для результатов
        std::vector<int> omp_ant_paths(cpu_data.ants_count * PARAMETR_SIZE);
        std::vector<double> omp_antOF(cpu_data.ants_count);

        double minOf = 1e9;
        double maxOf = -1e9;

        // Параллельная обработка муравьев на CPU
        calculate_ant_paths_omp_atomic_non_hash(aco, 0, cpu_data.ants_count, omp_ant_paths, omp_antOF, minOf, maxOf);
#if (PRINT_INFORMATION)
        std::cout << "[OMP ant Thread] END calculate_ant_paths_omp_atomic" << std::endl;
#endif
        // Отправляем результаты обратно
        IterationData omp_data;
        omp_data.iteration = cpu_data.iteration;
        omp_data.norm_matrix_probability = cpu_data.norm_matrix_probability; // Сохраняем входные данные
        omp_data.ant_parametr = std::move(omp_ant_paths);
        omp_data.antOF = std::move(omp_antOF);
        omp_data.minOf = minOf;
        omp_data.maxOf = maxOf;
        omp_data.kol_hash_fail = 0;
        omp_data.ants_count = cpu_data.ants_count;
        start_time_wait = std::chrono::high_resolution_clock::now();

        queue_push(&aco->OMP_to_cpu_queue, omp_data);
#if (PRINT_INFORMATION)
        std::cout << "[OMP ant Thread] queue_push(&aco->OMP_to_cpu_queue, omp_data)" << std::endl;
#endif
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    aco->Time_OMP_all = std::chrono::duration<double, std::milli>(end_time - start_time).count();

#if (PRINT_INFORMATION)
    std::cout << "[OMP Optimized Thread] Finished" << std::endl;
#endif
}

void OMP_thread_function_parallel(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "   [OMP ant Thread] Started" << std::endl;
#endif
    auto start_time = std::chrono::high_resolution_clock::now();
    auto start_time_wait = std::chrono::high_resolution_clock::now();
    while (!aco->stop_requested.load()) {
        IterationData cpu_data;
        if (!aco->cpu_to_OMP_queue_current.empty()) {
            cpu_data = std::move(aco->cpu_to_OMP_queue_current.front());
            aco->cpu_to_OMP_queue_current.pop();
            auto end_time_wait = std::chrono::high_resolution_clock::now();
            aco->Time_OMP_wait += std::chrono::duration<double, std::milli>(end_time_wait - start_time_wait).count();
#if (PRINT_INFORMATION)
            std::cout << "   [OMP ant Thread] queue_pop(&aco->cpu_to_OMP_queue, cpu_data)" << std::endl;
#endif

            // Выделяем память для результатов
            std::vector<int> omp_ant_paths(cpu_data.ants_count * PARAMETR_SIZE);
            std::vector<double> omp_antOF(cpu_data.ants_count);

            double minOf = 1e9;
            double maxOf = -1e9;
            int kol_hash_fail = 0;
            // Параллельная обработка муравьев на CPU
            calculate_ant_paths_omp_atomic(aco, 0, cpu_data.ants_count, omp_ant_paths, omp_antOF, minOf, maxOf, kol_hash_fail);
#if (PRINT_INFORMATION)
            std::cout << "   [OMP ant Thread] END calculate_ant_paths_omp_atomic" << std::endl;
#endif
            // Отправляем результаты обратно
            IterationData omp_data;
            omp_data.iteration = cpu_data.iteration;
            omp_data.norm_matrix_probability = cpu_data.norm_matrix_probability; // Сохраняем входные данные
            omp_data.ant_parametr = std::move(omp_ant_paths);
            omp_data.antOF = std::move(omp_antOF);
            omp_data.minOf = minOf;
            omp_data.maxOf = maxOf;
            omp_data.kol_hash_fail = kol_hash_fail;
            omp_data.ants_count = cpu_data.ants_count;

            start_time_wait = std::chrono::high_resolution_clock::now();
            aco->OMP_to_cpu_queue_current.push(omp_data);

#if (PRINT_INFORMATION)
            std::cout << "   [OMP ant Thread] queue_push(&aco->OMP_to_cpu_queue, omp_data)" << std::endl;
#endif
        }

    }

    auto end_time = std::chrono::high_resolution_clock::now();
    aco->Time_OMP_all = std::chrono::duration<double, std::milli>(end_time - start_time).count();

#if (PRINT_INFORMATION)
    std::cout << "[OMP Optimized Thread] Finished" << std::endl;
#endif
}
void OMP_thread_function_parallel_non_hash(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "   [OMP ant Thread] Started" << std::endl;
#endif
    auto start_time = std::chrono::high_resolution_clock::now();
    auto start_time_wait = std::chrono::high_resolution_clock::now();
    while (!aco->stop_requested.load()) {
        IterationData cpu_data;
        if (!aco->cpu_to_OMP_queue_current.empty()) {
            cpu_data = std::move(aco->cpu_to_OMP_queue_current.front());
            aco->cpu_to_OMP_queue_current.pop();
            auto end_time_wait = std::chrono::high_resolution_clock::now();
            aco->Time_OMP_wait += std::chrono::duration<double, std::milli>(end_time_wait - start_time_wait).count();
#if (PRINT_INFORMATION)
            std::cout << "   [OMP ant Thread] queue_pop(&aco->cpu_to_OMP_queue, cpu_data)" << std::endl;
#endif

            // Выделяем память для результатов
            std::vector<int> omp_ant_paths(cpu_data.ants_count * PARAMETR_SIZE);
            std::vector<double> omp_antOF(cpu_data.ants_count);

            double minOf = 1e9;
            double maxOf = -1e9;
            // Параллельная обработка муравьев на CPU
            calculate_ant_paths_omp_atomic_non_hash(aco, 0, cpu_data.ants_count, omp_ant_paths, omp_antOF, minOf, maxOf);
#if (PRINT_INFORMATION)
            std::cout << "   [OMP ant Thread] END calculate_ant_paths_omp_atomic" << std::endl;
#endif
            // Отправляем результаты обратно
            IterationData omp_data;
            omp_data.iteration = cpu_data.iteration;
            omp_data.norm_matrix_probability = cpu_data.norm_matrix_probability; // Сохраняем входные данные
            omp_data.ant_parametr = std::move(omp_ant_paths);
            omp_data.antOF = std::move(omp_antOF);
            omp_data.minOf = minOf;
            omp_data.maxOf = maxOf;
            omp_data.kol_hash_fail = 0;
            omp_data.ants_count = cpu_data.ants_count;

            start_time_wait = std::chrono::high_resolution_clock::now();
            aco->OMP_to_cpu_queue_current.push(omp_data);

#if (PRINT_INFORMATION)
            std::cout << "   [OMP ant Thread] queue_push(&aco->OMP_to_cpu_queue, omp_data)" << std::endl;
#endif
        }

    }

    auto end_time = std::chrono::high_resolution_clock::now();
    aco->Time_OMP_all = std::chrono::duration<double, std::milli>(end_time - start_time).count();

#if (PRINT_INFORMATION)
    std::cout << "[OMP Optimized Thread] Finished" << std::endl;
#endif
}

void cpu_thread_function_OMP(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "[CPU Thread] Started" << std::endl;
#endif

    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = std::chrono::high_resolution_clock::now();

    for (int iteration = 0; iteration < KOL_ITERATION && !aco->stop_requested.load(); iteration++) {
        auto start_time_prob = std::chrono::high_resolution_clock::now();
        calculate_probabilities(aco);

        end_time = std::chrono::high_resolution_clock::now();
        aco->Time_CPU_prob += std::chrono::duration<double, std::milli>(end_time - start_time_prob).count();

        // Подготовка данных для GPU и OMP
        IterationData cpu_data_gpu;
        cpu_data_gpu.iteration = iteration;
        cpu_data_gpu.norm_matrix_probability = aco->norm_matrix_probability;
        cpu_data_gpu.ants_count = aco->gpu_ants_count.load();

        IterationData cpu_data_omp;
        cpu_data_omp.iteration = iteration;
        cpu_data_omp.norm_matrix_probability = aco->norm_matrix_probability;
        cpu_data_omp.ants_count = aco->OMP_ants_count.load();

#if (PRINT_INFORMATION)
        std::cout << "[CPU Hybrid] Iteration " << iteration
            << " - GPU ants: " << cpu_data_gpu.ants_count
            << ", OMP ants: " << cpu_data_omp.ants_count << std::endl;
#endif

        
        queue_push(&aco->cpu_to_gpu_queue, cpu_data_gpu);
        queue_push(&aco->cpu_to_OMP_queue, cpu_data_omp);

        IterationData gpu_data;
        IterationData OMP_data;

        bool gpu_received = false;
        bool omp_received = false;
        const int MAX_WAIT_MS = 30000; // 30 секунд максимум

        auto wait_start = std::chrono::steady_clock::now();
        auto start_time_wait = std::chrono::high_resolution_clock::now();
        // Ожидаем результаты от обоих устройств с таймаутом
        while ((!gpu_received || !omp_received) && !aco->stop_requested.load()) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - wait_start);

            // Проверяем таймаут
            if (elapsed.count() > MAX_WAIT_MS) {
                std::cerr << "[ERROR] Timeout waiting for GPU/OMP results in iteration " << iteration << std::endl;

                // Форсируем остановку
                aco->stop_requested.store(true);
                queue_stop(&aco->cpu_to_gpu_queue);
                queue_stop(&aco->cpu_to_OMP_queue);
                queue_stop(&aco->gpu_to_cpu_queue);
                queue_stop(&aco->OMP_to_cpu_queue);
                break;
            }

            // Пытаемся получить данные (оригинальный queue_pop с 2 параметрами)
            if (!gpu_received) {
                // Используем неблокирующую проверку с помощью queue_size
                if (queue_size(&aco->gpu_to_cpu_queue) > 0) {
                    if (queue_pop(&aco->gpu_to_cpu_queue, gpu_data)) {
                        gpu_received = true;
#if (PRINT_INFORMATION)
                        std::cout << "[CPU Hybrid] Received GPU results for iteration " << gpu_data.iteration << std::endl;
#endif
                    }
                }
            }

            if (!omp_received) {
                // Используем неблокирующую проверку с помощью queue_size
                if (queue_size(&aco->OMP_to_cpu_queue) > 0) {
                    if (queue_pop(&aco->OMP_to_cpu_queue, OMP_data)) {
                        omp_received = true;
#if (PRINT_INFORMATION)
                        std::cout << "[CPU Hybrid] Received OMP results for iteration " << OMP_data.iteration << std::endl;
#endif
                    }
                }
            }
            /*
            // Если оба результата не получены, небольшая пауза
            if (!gpu_received || !omp_received) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            */
        }

        // Если данные не получены из-за таймаута, пропускаем итерацию
        if (!gpu_received || !omp_received) {
            std::cerr << "[ERROR] Skipping iteration " << iteration << " due to missing data" << std::endl;

            // Диагностика
            std::cout << "[DIAGNOSTIC] Queue status after timeout:" << std::endl;
            std::cout << "  CPU->GPU: " << queue_size(&aco->cpu_to_gpu_queue) << " items" << std::endl;
            std::cout << "  GPU->CPU: " << queue_size(&aco->gpu_to_cpu_queue) << " items" << std::endl;
            std::cout << "  CPU->OMP: " << queue_size(&aco->cpu_to_OMP_queue) << " items" << std::endl;
            std::cout << "  OMP->CPU: " << queue_size(&aco->OMP_to_cpu_queue) << " items" << std::endl;
            std::cout << "  Active CUDA tasks: " << active_cuda_tasks.load() << std::endl;

            continue;
        }

        end_time = std::chrono::high_resolution_clock::now();
        aco->Time_CPU_wait += std::chrono::duration<double, std::milli>(end_time - start_time_wait).count();

        // Проверяем корректность данных
        if (gpu_data.ants_count + OMP_data.ants_count != ANT_SIZE) {
            std::cerr << "[CPU Hybrid] ERROR: Total ants count mismatch! GPU: " << gpu_data.ants_count << ", OMP: " << OMP_data.ants_count << ", Expected: " << ANT_SIZE << std::endl;
        }

        // Объединяем результаты в общие массивы
        if (gpu_data.ants_count > 0) {
            std::copy(gpu_data.ant_parametr.begin(), gpu_data.ant_parametr.end(),
                aco->ant_parametr.begin());
            std::copy(gpu_data.antOF.begin(), gpu_data.antOF.end(),
                aco->antOF.begin());
        }

        if (OMP_data.ants_count > 0) {
            int omp_start_index = gpu_data.ants_count * PARAMETR_SIZE;
            std::copy(OMP_data.ant_parametr.begin(), OMP_data.ant_parametr.end(),
                aco->ant_parametr.begin() + omp_start_index);

            int omp_of_start_index = gpu_data.ants_count;
            std::copy(OMP_data.antOF.begin(), OMP_data.antOF.end(),
                aco->antOF.begin() + omp_of_start_index);
        }

        if (gpu_data.minOf < aco->global_minOf) { aco->global_minOf = gpu_data.minOf; }
        if (OMP_data.minOf < aco->global_minOf) { aco->global_minOf = OMP_data.minOf; }
        if (gpu_data.maxOf > aco->global_maxOf) { aco->global_maxOf = gpu_data.maxOf; }
        if (OMP_data.maxOf > aco->global_maxOf) { aco->global_maxOf = OMP_data.maxOf; }
        aco->kol_hash_fail += gpu_data.kol_hash_fail + OMP_data.kol_hash_fail;

        auto start_time_update = std::chrono::high_resolution_clock::now();
        update_pheromones_async(aco, gpu_data.iteration);
        end_time = std::chrono::high_resolution_clock::now();
        aco->Time_CPU_update += std::chrono::duration<double, std::milli>(end_time - start_time_update).count();

#if (PRINT_INFORMATION)
        if (iteration % KOL_PROGON_STATISTICS == 0) {
            std::cout << "[CPU Hybrid] Iteration " << iteration
                << " completed. Min: " << aco->global_minOf
                << ", Max: " << aco->global_maxOf
                << ", Hash fails: " << aco->kol_hash_fail
                << ", GPU: " << gpu_data.ants_count << " ants"
                << ", OMP: " << OMP_data.ants_count << " ants" << std::endl;
        }
#endif
    }

    // Останавливаем все очереди
    queue_stop(&aco->cpu_to_gpu_queue);
    queue_stop(&aco->gpu_to_cpu_queue);
    queue_stop(&aco->cpu_to_OMP_queue);
    queue_stop(&aco->OMP_to_cpu_queue);

    end_time = std::chrono::high_resolution_clock::now();
    aco->Time_CPU_all = std::chrono::duration<double, std::milli>(end_time - start_time).count();

#if (PRINT_INFORMATION)
    std::cout << "[CPU Hybrid Thread] Finished" << std::endl;
    std::cout << "[CPU Hybrid] Final stats - Min: " << aco->global_minOf
        << ", Max: " << aco->global_maxOf
        << ", Total hash fails: " << aco->kol_hash_fail << std::endl;
#endif
}
void dynamic_cpu_thread_function_OMP(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "[CPU Thread] Started" << std::endl;
#endif

    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = std::chrono::high_resolution_clock::now();
    double Last_Time_GPU_function = 0;
    double Last_Time_OMP_function = 0;

    for (int iteration = 0; iteration < KOL_ITERATION && !aco->stop_requested.load(); iteration++) {
        auto start_time_prob = std::chrono::high_resolution_clock::now();
        calculate_probabilities(aco);

        end_time = std::chrono::high_resolution_clock::now();
        aco->Time_CPU_prob += std::chrono::duration<double, std::milli>(end_time - start_time_prob).count();

        // Подготовка данных для GPU и OMP
        IterationData cpu_data_gpu;
        cpu_data_gpu.iteration = iteration;
        cpu_data_gpu.norm_matrix_probability = aco->norm_matrix_probability;
        cpu_data_gpu.ants_count = aco->gpu_ants_count.load();

        IterationData cpu_data_omp;
        cpu_data_omp.iteration = iteration;
        cpu_data_omp.norm_matrix_probability = aco->norm_matrix_probability;
        cpu_data_omp.ants_count = aco->OMP_ants_count.load();

#if (PRINT_INFORMATION)
        std::cout << "[CPU Hybrid] Iteration " << iteration
            << " - GPU ants: " << cpu_data_gpu.ants_count
            << ", OMP ants: " << cpu_data_omp.ants_count << std::endl;
#endif
        aco->cpu_ants_statistics += cpu_data_omp.ants_count;
        aco->gpu_ants_statistics += cpu_data_gpu.ants_count;
        
        queue_push(&aco->cpu_to_gpu_queue, cpu_data_gpu);
        queue_push(&aco->cpu_to_OMP_queue, cpu_data_omp);

        IterationData gpu_data;
        IterationData OMP_data;

        bool gpu_received = false;
        bool omp_received = false;
        const int MAX_WAIT_MS = 30000; // 30 секунд максимум

        auto wait_start = std::chrono::steady_clock::now();
        auto start_time_wait = std::chrono::high_resolution_clock::now();
        // Ожидаем результаты от обоих устройств с таймаутом
        while ((!gpu_received || !omp_received) && !aco->stop_requested.load()) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - wait_start);

            // Проверяем таймаут
            if (elapsed.count() > MAX_WAIT_MS) {
                std::cerr << "[ERROR] Timeout waiting for GPU/OMP results in iteration " << iteration << std::endl;

                // Форсируем остановку
                aco->stop_requested.store(true);
                queue_stop(&aco->cpu_to_gpu_queue);
                queue_stop(&aco->cpu_to_OMP_queue);
                queue_stop(&aco->gpu_to_cpu_queue);
                queue_stop(&aco->OMP_to_cpu_queue);
                break;
            }

            // Пытаемся получить данные (оригинальный queue_pop с 2 параметрами)
            if (!gpu_received) {
                // Используем неблокирующую проверку с помощью queue_size
                if (queue_size(&aco->gpu_to_cpu_queue) > 0) {
                    if (queue_pop(&aco->gpu_to_cpu_queue, gpu_data)) {
                        gpu_received = true;
#if (PRINT_INFORMATION)
                        std::cout << "[CPU Hybrid] Received GPU results for iteration " << gpu_data.iteration << std::endl;
#endif
                    }
                }
            }

            if (!omp_received) {
                // Используем неблокирующую проверку с помощью queue_size
                if (queue_size(&aco->OMP_to_cpu_queue) > 0) {
                    if (queue_pop(&aco->OMP_to_cpu_queue, OMP_data)) {
                        omp_received = true;
#if (PRINT_INFORMATION)
                        std::cout << "[CPU Hybrid] Received OMP results for iteration " << OMP_data.iteration << std::endl;
#endif
                    }
                }
            }

        }

        // Если данные не получены из-за таймаута, пропускаем итерацию
        if (!gpu_received || !omp_received) {
            std::cerr << "[ERROR] Skipping iteration " << iteration << " due to missing data" << std::endl;

            // Диагностика
            std::cout << "[DIAGNOSTIC] Queue status after timeout:" << std::endl;
            std::cout << "  CPU->GPU: " << queue_size(&aco->cpu_to_gpu_queue) << " items" << std::endl;
            std::cout << "  GPU->CPU: " << queue_size(&aco->gpu_to_cpu_queue) << " items" << std::endl;
            std::cout << "  CPU->OMP: " << queue_size(&aco->cpu_to_OMP_queue) << " items" << std::endl;
            std::cout << "  OMP->CPU: " << queue_size(&aco->OMP_to_cpu_queue) << " items" << std::endl;
            std::cout << "  Active CUDA tasks: " << active_cuda_tasks.load() << std::endl;

            continue;
        }

        end_time = std::chrono::high_resolution_clock::now();
        aco->Time_CPU_wait += std::chrono::duration<double, std::milli>(end_time - start_time_wait).count();

        // Проверяем корректность данных
        if (gpu_data.ants_count + OMP_data.ants_count != ANT_SIZE) {
            std::cerr << "[CPU Hybrid] ERROR: Total ants count mismatch! GPU: " << gpu_data.ants_count << ", OMP: " << OMP_data.ants_count << ", Expected: " << ANT_SIZE << std::endl;
        }

        // Объединяем результаты в общие массивы
        if (gpu_data.ants_count > 0) {
            std::copy(gpu_data.ant_parametr.begin(), gpu_data.ant_parametr.end(),
                aco->ant_parametr.begin());
            std::copy(gpu_data.antOF.begin(), gpu_data.antOF.end(),
                aco->antOF.begin());
        }

        if (OMP_data.ants_count > 0) {
            int omp_start_index = gpu_data.ants_count * PARAMETR_SIZE;
            std::copy(OMP_data.ant_parametr.begin(), OMP_data.ant_parametr.end(),
                aco->ant_parametr.begin() + omp_start_index);

            int omp_of_start_index = gpu_data.ants_count;
            std::copy(OMP_data.antOF.begin(), OMP_data.antOF.end(),
                aco->antOF.begin() + omp_of_start_index);
        }

        if (gpu_data.minOf < aco->global_minOf) { aco->global_minOf = gpu_data.minOf; }
        if (OMP_data.minOf < aco->global_minOf) { aco->global_minOf = OMP_data.minOf; }
        if (gpu_data.maxOf > aco->global_maxOf) { aco->global_maxOf = gpu_data.maxOf; }
        if (OMP_data.maxOf > aco->global_maxOf) { aco->global_maxOf = OMP_data.maxOf; }
        aco->kol_hash_fail += gpu_data.kol_hash_fail + OMP_data.kol_hash_fail;
        double gpu_time_balanced = 0;
        double omp_time_balanced = 0;
#if (BALANCED_TIME_GPU_FUNCTION)
        gpu_time_balanced += aco->Time_GPU_function;
#elif (BALANCED_TIME_GPU)
        gpu_time_balanced += aco->Time_GPU;
#endif
#if (BALANCED_TIME_OMP_FUNCTION)
        omp_time_balanced += aco->Time_OMP_function;
#endif
#if (BALANCED_ADD_TIME_OMP_PROBABILITY)
        omp_time_balanced += aco->Time_CPU_prob;
#endif
#if (BALANCED_ADD_TIME_OMP_ADD_AND_DECREASE)
        omp_time_balanced += aco->Time_CPU_update;
#endif
        gpu_time_balanced = gpu_time_balanced - Last_Time_GPU_function;
        omp_time_balanced = omp_time_balanced - Last_Time_OMP_function;

        if (gpu_time_balanced <= 0 || omp_time_balanced <= 0) {
            aco->OMP_ants_count.store(static_cast<int>(ANT_SIZE * INITIAL_CPU_ANTS_RATIO));
            aco->gpu_ants_count.store(ANT_SIZE - aco->OMP_ants_count.load());
            return;
        }

        // Вычисляем текущее соотношение времени на одного муравья
        double current_omp_ants = static_cast<double>(aco->OMP_ants_count.load());
        double current_gpu_ants = static_cast<double>(aco->gpu_ants_count.load());

        double time_per_ant_omp = omp_time_balanced / current_omp_ants;
        double time_per_ant_gpu = gpu_time_balanced / current_gpu_ants;

        double new_omp_ants_double = (time_per_ant_gpu * ANT_SIZE) / (time_per_ant_omp + time_per_ant_gpu);
        int new_omp_ants = static_cast<int>(new_omp_ants_double);

        // Ограничиваем минимальное и максимальное количество
        new_omp_ants = std::max(MIN_BALANSED_ANT, std::min(ANT_SIZE - MIN_BALANSED_ANT, new_omp_ants));

        aco->OMP_ants_count.store(new_omp_ants);
        aco->gpu_ants_count.store(ANT_SIZE - new_omp_ants);

        // Прогнозируемое время выполнения
        double predicted_omp_time = new_omp_ants * time_per_ant_omp;
        double predicted_gpu_time = (ANT_SIZE - new_omp_ants) * time_per_ant_gpu;

        //std::cout << "Balance: GPU=" << gpu_time_balanced << "ms (" << current_gpu_ants << " ants)," << " OMP=" << omp_time_balanced << "ms (" << current_omp_ants << " ants)" << " Time per ant: GPU=" << time_per_ant_gpu << "ms, OMP=" << time_per_ant_omp << "ms" << " New distribution: OMP_ants=" << new_omp_ants << ", GPU_ants=" << (ANT_SIZE - new_omp_ants) << " Predicted time: GPU=" << predicted_gpu_time << "ms, OMP=" << predicted_omp_time << "ms" << std::endl;
        Last_Time_GPU_function = 0;
        Last_Time_OMP_function = 0;
#if (BALANCED_TIME_GPU_FUNCTION)
        Last_Time_GPU_function += aco->Time_GPU_function;
#elif (BALANCED_TIME_GPU)
        Last_Time_GPU_function += aco->Time_GPU;
#endif
#if (BALANCED_TIME_OMP_FUNCTION)
        Last_Time_OMP_function += aco->Time_OMP_function;
#endif
#if (BALANCED_ADD_TIME_OMP_PROBABILITY)
        Last_Time_OMP_function += aco->Time_CPU_prob;
#endif
#if (BALANCED_ADD_TIME_OMP_ADD_AND_DECREASE)
        Last_Time_OMP_function += aco->Time_CPU_update;
#endif
        auto start_time_update = std::chrono::high_resolution_clock::now();
        update_pheromones_async(aco, gpu_data.iteration);
        end_time = std::chrono::high_resolution_clock::now();
        aco->Time_CPU_update += std::chrono::duration<double, std::milli>(end_time - start_time_update).count();

#if (PRINT_INFORMATION)
        if (iteration % KOL_PROGON_STATISTICS == 0) {
            std::cout << "[CPU Hybrid] Iteration " << iteration
                << " completed. Min: " << aco->global_minOf
                << ", Max: " << aco->global_maxOf
                << ", Hash fails: " << aco->kol_hash_fail
                << ", GPU: " << gpu_data.ants_count << " ants"
                << ", OMP: " << OMP_data.ants_count << " ants" << std::endl;
        }
#endif
    }

    // Останавливаем все очереди
    queue_stop(&aco->cpu_to_gpu_queue);
    queue_stop(&aco->gpu_to_cpu_queue);
    queue_stop(&aco->cpu_to_OMP_queue);
    queue_stop(&aco->OMP_to_cpu_queue);

    end_time = std::chrono::high_resolution_clock::now();
    aco->Time_CPU_all = std::chrono::duration<double, std::milli>(end_time - start_time).count();

#if (PRINT_INFORMATION)
    std::cout << "[CPU Hybrid Thread] Finished" << std::endl;
    std::cout << "[CPU Hybrid] Final stats - Min: " << aco->global_minOf
        << ", Max: " << aco->global_maxOf
        << ", Total hash fails: " << aco->kol_hash_fail << std::endl;
#endif
}
void dynamic_cpu_thread_function_OMP_parallel(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "[CPU Thread] Started" << std::endl;
#endif

    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = std::chrono::high_resolution_clock::now();
    double Last_Time_GPU_function = 0;
    double Last_Time_OMP_function = 0;
    int iteration = 0;
    bool go_push_OMP = true, go_push_gpu = true, go_push_OMP_balansed = false, go_push_gpu_balansed = false;

    while (iteration < KOL_ITERATION*2 && !aco->stop_requested) {
        auto start_time_prob = std::chrono::high_resolution_clock::now();
        calculate_probabilities(aco);
        end_time = std::chrono::high_resolution_clock::now();
        aco->Time_CPU_prob += std::chrono::duration<double, std::milli>(end_time - start_time_prob).count();

        // Подготовка данных для GPU и OMP
        IterationData cpu_data_gpu;
        cpu_data_gpu.iteration = iteration;
        cpu_data_gpu.norm_matrix_probability = aco->norm_matrix_probability;
        cpu_data_gpu.ants_count = aco->gpu_ants_count;

        IterationData cpu_data_omp;
        cpu_data_omp.iteration = iteration;
        cpu_data_omp.norm_matrix_probability = aco->norm_matrix_probability;
        cpu_data_omp.ants_count = aco->OMP_ants_count;

#if (PRINT_INFORMATION)
        std::cout << "[CPU Hybrid] Iteration " << iteration << " - GPU ants: " << cpu_data_gpu.ants_count << ", OMP ants: " << cpu_data_omp.ants_count << std::endl;
#endif

        auto start_time_wait = std::chrono::high_resolution_clock::now();

        if (go_push_OMP) {
            aco->cpu_to_OMP_queue_current.push(cpu_data_omp); 
            go_push_OMP = false;
        }
        if (go_push_gpu){ 
            aco->cpu_to_gpu_queue_current.push(cpu_data_gpu); 
            go_push_gpu = false;
        }
        aco->cpu_ants_statistics += cpu_data_omp.ants_count;
        aco->gpu_ants_statistics += cpu_data_gpu.ants_count;

        IterationData gpu_data;
        IterationData OMP_data;

        if (!aco->OMP_to_cpu_queue_current.empty()) {
            end_time = std::chrono::high_resolution_clock::now();
            aco->Time_CPU_wait += std::chrono::duration<double, std::milli>(end_time - start_time_wait).count();
            OMP_data = std::move(aco->OMP_to_cpu_queue_current.front());
            aco->OMP_to_cpu_queue_current.pop();
            if (OMP_data.ants_count > 0) {
                std::copy(OMP_data.ant_parametr.begin(), OMP_data.ant_parametr.end(), aco->ant_parametr.begin());
                std::copy(OMP_data.antOF.begin(), OMP_data.antOF.end(), aco->antOF.begin());
                auto start_time_update = std::chrono::high_resolution_clock::now();
                update_pheromones_async(aco, iteration);
                end_time = std::chrono::high_resolution_clock::now();
                aco->Time_CPU_update += std::chrono::duration<double, std::milli>(end_time - start_time_update).count();
                if (OMP_data.minOf < aco->global_minOf) { aco->global_minOf = OMP_data.minOf; }
                if (OMP_data.maxOf > aco->global_maxOf) { aco->global_maxOf = OMP_data.maxOf; }
                aco->kol_hash_fail += OMP_data.kol_hash_fail;
                iteration++;
                go_push_OMP = true;
                go_push_OMP_balansed = true;
            }
#if (PRINT_INFORMATION)
            std::cout << "[CPU Hybrid] OMP_to_cpu_queue_current.pop() " << OMP_data.ants_count << " ants go_push_OMP=" << go_push_OMP << ", " << go_push_OMP_balansed << std::endl;
#endif
            start_time_wait = std::chrono::high_resolution_clock::now();
        }

        if (!aco->gpu_to_cpu_queue_current.empty()) {
            end_time = std::chrono::high_resolution_clock::now();
            aco->Time_CPU_wait += std::chrono::duration<double, std::milli>(end_time - start_time_wait).count();
            gpu_data = std::move(aco->gpu_to_cpu_queue_current.front());
            aco->gpu_to_cpu_queue_current.pop();
            if (gpu_data.ants_count > 0) {
                std::copy(gpu_data.ant_parametr.begin(), gpu_data.ant_parametr.end(), aco->ant_parametr.begin());
                std::copy(gpu_data.antOF.begin(), gpu_data.antOF.end(), aco->antOF.begin());
                auto start_time_update = std::chrono::high_resolution_clock::now();
                update_pheromones_async(aco, iteration);
                end_time = std::chrono::high_resolution_clock::now();
                aco->Time_CPU_update += std::chrono::duration<double, std::milli>(end_time - start_time_update).count();
                if (gpu_data.minOf < aco->global_minOf) { aco->global_minOf = gpu_data.minOf; }
                if (gpu_data.maxOf > aco->global_maxOf) { aco->global_maxOf = gpu_data.maxOf; }
                aco->kol_hash_fail += gpu_data.kol_hash_fail;
                iteration++;
                go_push_gpu = true;
                go_push_gpu_balansed = true;
            }
#if (PRINT_INFORMATION)
            std::cout << "[CPU Hybrid] gpu_to_cpu_queue_current.pop() " << gpu_data.ants_count << " ants go_push_gpu=" << go_push_gpu << ", " << go_push_gpu_balansed << std::endl;
#endif
            start_time_wait = std::chrono::high_resolution_clock::now();
        }

#if (PRINT_INFORMATION)
        std::cout << "[CPU Hybrid] Iteration END " << iteration << " - GPU ants: " << aco->gpu_ants_count << ", OMP ants: " << aco->OMP_ants_count << " go_push_gpu=" << go_push_gpu<< ", " << go_push_gpu_balansed << " go_push_OMP=" << go_push_OMP << ", " << go_push_OMP_balansed  << std::endl;
#endif
        if (go_push_OMP_balansed && go_push_gpu_balansed){
            go_push_OMP_balansed = false;
            go_push_gpu_balansed = false;
            end_time = std::chrono::high_resolution_clock::now();
            aco->Time_CPU_wait += std::chrono::duration<double, std::milli>(end_time - start_time_wait).count();
            
#if (PRINT_INFORMATION)
            std::cout << "Balance: aco->Time_GPU_function=" << aco->Time_GPU_function << " aco->Time_OMP_function=" << aco->Time_OMP_function << std::endl;
#endif
            if (aco->Time_GPU_function != Last_Time_GPU_function && aco->Time_OMP_function != Last_Time_OMP_function) {
                double gpu_time_balanced = 0;
                double omp_time_balanced = 0;
#if (BALANCED_TIME_GPU_FUNCTION)
                gpu_time_balanced += aco->Time_GPU_function;
#elif (BALANCED_TIME_GPU)
                gpu_time_balanced += aco->Time_GPU;
#endif
#if (BALANCED_TIME_OMP_FUNCTION)
                omp_time_balanced += aco->Time_OMP_function;
#endif
#if (BALANCED_ADD_TIME_OMP_PROBABILITY)
                omp_time_balanced += aco->Time_CPU_prob;
#endif
#if (BALANCED_ADD_TIME_OMP_ADD_AND_DECREASE)
                omp_time_balanced += aco->Time_CPU_update;
#endif
                gpu_time_balanced = gpu_time_balanced - Last_Time_GPU_function;
                omp_time_balanced = omp_time_balanced - Last_Time_OMP_function;
#if (PRINT_INFORMATION)
                std::cout << "Balance: gpu_time_balanced=" << gpu_time_balanced << " omp_time_balanced=" << omp_time_balanced << std::endl;
#endif
                if (gpu_time_balanced <= 0 || omp_time_balanced <= 0) {
                    aco->OMP_ants_count = static_cast<int>(ANT_SIZE * INITIAL_CPU_ANTS_RATIO);
                    aco->gpu_ants_count = ANT_SIZE - aco->OMP_ants_count;
                    return;
                }

                // Вычисляем текущее соотношение времени на одного муравья
                double current_omp_ants = static_cast<double>(aco->OMP_ants_count);
                double current_gpu_ants = static_cast<double>(aco->gpu_ants_count);
#if (PRINT_INFORMATION)
                std::cout << "Balance: current_gpu_ants=" << current_gpu_ants << " current_omp_ants=" << current_omp_ants << std::endl;
#endif
                double time_per_ant_omp = omp_time_balanced / current_omp_ants;
                double time_per_ant_gpu = gpu_time_balanced / current_gpu_ants;
#if (PRINT_INFORMATION)
                std::cout << "Balance: time_per_ant_gpu=" << time_per_ant_gpu << " time_per_ant_omp=" << time_per_ant_omp << std::endl;
#endif
                double new_omp_ants_double = (time_per_ant_gpu * ANT_SIZE) / (time_per_ant_omp + time_per_ant_gpu);
                int new_omp_ants = static_cast<int>(new_omp_ants_double);
#if (PRINT_INFORMATION)
                std::cout << "Balance: new_omp_ants_double=" << new_omp_ants_double << " new_omp_ants=" << new_omp_ants << std::endl;
#endif
                // Ограничиваем минимальное и максимальное количество
                new_omp_ants = std::max(MIN_BALANSED_ANT, std::min(ANT_SIZE - MIN_BALANSED_ANT, new_omp_ants));

                aco->OMP_ants_count = new_omp_ants;
                aco->gpu_ants_count = ANT_SIZE - new_omp_ants;
#if (PRINT_INFORMATION)
                std::cout << "Balance: aco->gpu_ants_count=" << aco->gpu_ants_count << " aco->OMP_ants_count=" << aco->OMP_ants_count << std::endl;
#endif
                // Прогнозируемое время выполнения
                double predicted_omp_time = new_omp_ants * time_per_ant_omp;
                double predicted_gpu_time = (ANT_SIZE - new_omp_ants) * time_per_ant_gpu;
#if (PRINT_INFORMATION)
                std::cout << "Balance: GPU=" << gpu_time_balanced << "ms (" << current_gpu_ants << " ants)," << " OMP=" << omp_time_balanced << "ms (" << current_omp_ants << " ants)" << " Time per ant: GPU=" << time_per_ant_gpu << "ms, OMP=" << time_per_ant_omp << "ms" << " New distribution: OMP_ants=" << new_omp_ants << ", GPU_ants=" << (ANT_SIZE - new_omp_ants) << " Predicted time: GPU=" << predicted_gpu_time << "ms, OMP=" << predicted_omp_time << "ms" << std::endl;
#endif
                Last_Time_GPU_function = 0;
                Last_Time_OMP_function = 0;
#if (BALANCED_TIME_GPU_FUNCTION)
                Last_Time_GPU_function += aco->Time_GPU_function;
#elif (BALANCED_TIME_GPU)
                Last_Time_GPU_function += aco->Time_GPU;
#endif
#if (BALANCED_TIME_OMP_FUNCTION)
                Last_Time_OMP_function += aco->Time_OMP_function;
#endif
#if (BALANCED_ADD_TIME_OMP_PROBABILITY)
                Last_Time_OMP_function += aco->Time_CPU_prob;
#endif
#if (BALANCED_ADD_TIME_OMP_ADD_AND_DECREASE)
                Last_Time_OMP_function += aco->Time_CPU_update;
#endif
#if (PRINT_INFORMATION)
                std::cout << "Balance: Last_Time_GPU_function=" << Last_Time_GPU_function << " Last_Time_OMP_function=" << Last_Time_OMP_function << std::endl;
#endif
                start_time_wait = std::chrono::high_resolution_clock::now();
            }
        }
        
#if (PRINT_INFORMATION)
        std::cout << "[CPU Hybrid] Iteration END BALANCED " << iteration << " - GPU ants: " << aco->gpu_ants_count << ", OMP ants: " << aco->OMP_ants_count << " go_push_gpu=" << go_push_gpu << ", " << go_push_gpu_balansed << " go_push_OMP=" << go_push_OMP << ", " << go_push_OMP_balansed << std::endl;
#endif
    }

    aco->stop_requested = true;
    end_time = std::chrono::high_resolution_clock::now();
    aco->Time_CPU_all = std::chrono::duration<double, std::milli>(end_time - start_time).count();

#if (PRINT_INFORMATION)
    std::cout << "[CPU Hybrid Thread] Finished" << std::endl;
    std::cout << "[CPU Hybrid] Final stats - Min: " << aco->global_minOf << ", Max: " << aco->global_maxOf  << ", Total hash fails: " << aco->kol_hash_fail << std::endl;
#endif
}

void cpu_thread_function(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "[CPU Thread] Started" << std::endl;
#endif

    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = std::chrono::high_resolution_clock::now();
    

    IterationData gpu_data;

    for (int iteration = 0; iteration < KOL_ITERATION && !aco->stop_requested.load(); iteration++) {
        auto start_time_prob = std::chrono::high_resolution_clock::now();
        calculate_probabilities(aco);
        end_time = std::chrono::high_resolution_clock::now();
        aco->Time_CPU_prob += std::chrono::duration<double, std::milli>(end_time - start_time_prob).count();

        IterationData cpu_data;
        cpu_data.ants_count = ANT_SIZE;
        cpu_data.iteration = iteration;
        cpu_data.norm_matrix_probability = aco->norm_matrix_probability;
#if (PRINT_INFORMATION)
        std::cout << "[CPU Thread] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
#endif  
        auto start_time_wait = std::chrono::high_resolution_clock::now();
        queue_push(&aco->cpu_to_gpu_queue, cpu_data);

        if (!queue_pop(&aco->gpu_to_cpu_queue, gpu_data)) {
            break;
        }
#if (PRINT_INFORMATION)
        std::cout << "[CPU Thread] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
#endif    

        end_time = std::chrono::high_resolution_clock::now();
        aco->Time_CPU_wait += std::chrono::duration<double, std::milli>(end_time - start_time_wait).count();

        aco->ant_parametr = gpu_data.ant_parametr; 
        aco->antOF = gpu_data.antOF; 

        if (gpu_data.minOf < aco->global_minOf) { aco->global_minOf = gpu_data.minOf; }
        if (gpu_data.maxOf > aco->global_maxOf) { aco->global_maxOf = gpu_data.maxOf; }
        aco->kol_hash_fail += gpu_data.kol_hash_fail;

        auto start_time_update = std::chrono::high_resolution_clock::now();
        update_pheromones_async(aco, gpu_data.iteration);
        end_time = std::chrono::high_resolution_clock::now();
        aco->Time_CPU_update += std::chrono::duration<double, std::milli>(end_time - start_time_update).count();

#if (PRINT_INFORMATION)
        if (gpu_data.iteration % KOL_PROGON_STATISTICS == 0) {
            std::cout << "[CPU Thread] Iteration " << gpu_data.iteration
                << " completed. Min: " << aco->global_minOf
                << ", Max: " << aco->global_maxOf << std::endl;
        }
#endif
    }

    queue_stop(&aco->cpu_to_gpu_queue);
    queue_stop(&aco->gpu_to_cpu_queue);
    end_time = std::chrono::high_resolution_clock::now();
    aco->Time_CPU_all = std::chrono::duration<double, std::milli>(end_time - start_time).count();

#if (PRINT_INFORMATION)
    std::cout << "[CPU Thread] Finished" << std::endl;
#endif
}

void gpu_thread_function(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "[GPU Thread] Started" << std::endl;
#endif

    auto start_time = std::chrono::high_resolution_clock::now();
    auto start_time_wait = std::chrono::high_resolution_clock::now();
    while (!aco->stop_requested.load()) {
        IterationData cpu_data;
        if (!queue_pop(&aco->cpu_to_gpu_queue, cpu_data)) {
            break;
        }
        auto end_time_wait = std::chrono::high_resolution_clock::now();
        aco->Time_GPU_wait += std::chrono::duration<double, std::milli>(end_time_wait - start_time_wait).count();
#if (PRINT_INFORMATION)
        std::cout << "[GPU Thread] queue_pop(&aco->cpu_to_gpu_queue, cpu_data)" << std::endl;
#endif
        active_cuda_tasks.fetch_add(1);

        // ВХОДНЫЕ ДАННЫЕ ДЛЯ GPU - только norm_matrix_probability
        // ВЫХОДНЫЕ ДАННЫЕ ОТ GPU - ant_parametr и antOF

        // ВЫДЕЛЯЕМ ПАМЯТЬ ДЛЯ РЕЗУЛЬТАТОВ (GPU заполнит их)
        std::vector<int> ant_parametr_result(PARAMETR_SIZE * cpu_data.ants_count);
        std::vector<double> antOF_result(cpu_data.ants_count);
        double minOf_result = 1e9;
        double maxOf_result = -1e9;
        int kol_hash_fail_result = 0;
        double time_1_result = 0.0;
        double time_function_result = 0.0;

#if (PRINT_INFORMATION)
        std::cout << "[GPU Thread] Processing iteration " << cpu_data.iteration << " with " << cpu_data.ants_count << " ants " << ant_parametr_result.size()<< " "<< std::endl;
#endif

        // ВЫЗЫВАЕМ CUDA - передаем буферы для результатов
        cuda_run_iteration(
            cpu_data.norm_matrix_probability.data(), // Вход: вероятности
            ant_parametr_result.data(),              // Выход: параметры муравьев
            antOF_result.data(),                     // Выход: значения функции
            cpu_data.ants_count,
            &minOf_result,
            &maxOf_result,
            &kol_hash_fail_result,
            &time_1_result,
            &time_function_result,
            cpu_data.iteration,
            cuda_completion_callback
        );

        // СОЗДАЕМ РЕЗУЛЬТАТ ДЛЯ CPU
        IterationData gpu_data;
        gpu_data.iteration = cpu_data.iteration;
        gpu_data.norm_matrix_probability = cpu_data.norm_matrix_probability; // Копируем входные данные
        gpu_data.ant_parametr = std::move(ant_parametr_result); // Перемещаем РЕЗУЛЬТАТЫ
        gpu_data.antOF = std::move(antOF_result);               // Перемещаем РЕЗУЛЬТАТЫ  
        gpu_data.minOf = minOf_result;
        gpu_data.maxOf = maxOf_result;
        gpu_data.kol_hash_fail = kol_hash_fail_result;
        gpu_data.ants_count = cpu_data.ants_count;
        gpu_data.Time_1 = time_1_result;
        gpu_data.Time_function = time_function_result;
        aco->Time_GPU += gpu_data.Time_1;
        aco->Time_GPU_function += gpu_data.Time_function;

#if (PRINT_INFORMATION)
        std::cout << "[GPU Thread] Iteration " << gpu_data.iteration
            << " completed. Min: " << gpu_data.minOf
            << ", Max: " << gpu_data.maxOf
            << ", Hash fails: " << gpu_data.kol_hash_fail << std::endl;
#endif
        start_time_wait = std::chrono::high_resolution_clock::now();
        queue_push(&aco->gpu_to_cpu_queue, gpu_data);
#if (PRINT_INFORMATION)
        std::cout << "[GPU Thread] queue_push(&aco->gpu_to_cpu_queue, gpu_data)" << std::endl;
#endif
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    aco->Time_GPU_all = std::chrono::duration<double, std::milli>(end_time - start_time).count();


#if (PRINT_INFORMATION)
    std::cout << "[GPU Thread] Finished" << std::endl;
#endif
}
void gpu_thread_function_non_hash(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "[GPU Thread] Started" << std::endl;
#endif

    auto start_time = std::chrono::high_resolution_clock::now();
    auto start_time_wait = std::chrono::high_resolution_clock::now();
    while (!aco->stop_requested.load()) {
        IterationData cpu_data;
        if (!queue_pop(&aco->cpu_to_gpu_queue, cpu_data)) {
            break;
        }
        auto end_time_wait = std::chrono::high_resolution_clock::now();
        aco->Time_GPU_wait += std::chrono::duration<double, std::milli>(end_time_wait - start_time_wait).count();
        active_cuda_tasks.fetch_add(1);

        // ВХОДНЫЕ ДАННЫЕ ДЛЯ GPU - только norm_matrix_probability
        // ВЫХОДНЫЕ ДАННЫЕ ОТ GPU - ant_parametr и antOF

        // ВЫДЕЛЯЕМ ПАМЯТЬ ДЛЯ РЕЗУЛЬТАТОВ (GPU заполнит их)
        std::vector<int> ant_parametr_result(PARAMETR_SIZE * cpu_data.ants_count);
        std::vector<double> antOF_result(cpu_data.ants_count);
        double minOf_result = 1e9;
        double maxOf_result = -1e9;
        int kol_hash_fail_result = 0;
        double time_1_result = 0.0;
        double time_function_result = 0.0;

#if (PRINT_INFORMATION)
        std::cout << "[GPU Thread] Processing iteration " << cpu_data.iteration << " with " << cpu_data.ants_count << " ants " << ant_parametr_result.size() << " " << std::endl;
#endif

        // ВЫЗЫВАЕМ CUDA - передаем буферы для результатов
        cuda_run_iteration_non_hash(
            cpu_data.norm_matrix_probability.data(), // Вход: вероятности
            ant_parametr_result.data(),              // Выход: параметры муравьев
            antOF_result.data(),                     // Выход: значения функции
            cpu_data.ants_count,
            &minOf_result,
            &maxOf_result,
            &time_1_result,
            &time_function_result,
            cpu_data.iteration,
            cuda_completion_callback
        );

        // СОЗДАЕМ РЕЗУЛЬТАТ ДЛЯ CPU
        IterationData gpu_data;
        gpu_data.iteration = cpu_data.iteration;
        gpu_data.norm_matrix_probability = cpu_data.norm_matrix_probability; // Копируем входные данные
        gpu_data.ant_parametr = std::move(ant_parametr_result); // Перемещаем РЕЗУЛЬТАТЫ
        gpu_data.antOF = std::move(antOF_result);               // Перемещаем РЕЗУЛЬТАТЫ  
        gpu_data.minOf = minOf_result;
        gpu_data.maxOf = maxOf_result;
        gpu_data.kol_hash_fail = 0;
        gpu_data.ants_count = cpu_data.ants_count;
        gpu_data.Time_1 = time_1_result;
        gpu_data.Time_function = time_function_result;
        aco->Time_GPU += gpu_data.Time_1;
        aco->Time_GPU_function += gpu_data.Time_function;

#if (PRINT_INFORMATION)
        std::cout << "[GPU Thread] Iteration " << gpu_data.iteration << " completed. Min: " << gpu_data.minOf << ", Max: " << gpu_data.maxOf << ", Hash fails: " << gpu_data.kol_hash_fail << std::endl;
#endif
        start_time_wait = std::chrono::high_resolution_clock::now();
        queue_push(&aco->gpu_to_cpu_queue, gpu_data);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    aco->Time_GPU_all = std::chrono::duration<double, std::milli>(end_time - start_time).count();


#if (PRINT_INFORMATION)
    std::cout << "[GPU Thread] Finished" << std::endl;
#endif
}

void gpu_thread_function_parallel(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "   [GPU ant Thread] Started" << std::endl;
#endif

    auto start_time = std::chrono::high_resolution_clock::now();
    auto start_time_wait = std::chrono::high_resolution_clock::now();
    while (!aco->stop_requested.load()) {
        IterationData cpu_data;
        if (!aco->cpu_to_gpu_queue_current.empty()) {
            cpu_data = std::move(aco->cpu_to_gpu_queue_current.front());
            aco->cpu_to_gpu_queue_current.pop();
            auto end_time_wait = std::chrono::high_resolution_clock::now();
            aco->Time_GPU_wait += std::chrono::duration<double, std::milli>(end_time_wait - start_time_wait).count();
#if (PRINT_INFORMATION)
            std::cout << "   [GPU ant Thread] queue_pop(&aco->cpu_to_gpu_queue, cpu_data)" << std::endl;
#endif
            active_cuda_tasks.fetch_add(1);
            // ВЫДЕЛЯЕМ ПАМЯТЬ ДЛЯ РЕЗУЛЬТАТОВ (GPU заполнит их)
            std::vector<int> ant_parametr_result(PARAMETR_SIZE * cpu_data.ants_count);
            std::vector<double> antOF_result(cpu_data.ants_count);
            double minOf_result = 1e9;
            double maxOf_result = -1e9;
            int kol_hash_fail_result = 0;
            double time_1_result = 0.0;
            double time_function_result = 0.0;
#if (PRINT_INFORMATION)
            std::cout << "   [GPU ant Thread] Processing iteration " << cpu_data.iteration << " with " << cpu_data.ants_count << " ants " << ant_parametr_result.size() << " " << std::endl;
#endif
            // ВЫЗЫВАЕМ CUDA - передаем буферы для результатов
            cuda_run_iteration(
                cpu_data.norm_matrix_probability.data(),
                ant_parametr_result.data(),              
                antOF_result.data(),                   
                cpu_data.ants_count,
                &minOf_result,
                &maxOf_result,
                &kol_hash_fail_result,
                &time_1_result,
                &time_function_result,
                cpu_data.iteration,
                cuda_completion_callback
            );

            // СОЗДАЕМ РЕЗУЛЬТАТ ДЛЯ CPU
            IterationData gpu_data;
            gpu_data.iteration = cpu_data.iteration;
            gpu_data.norm_matrix_probability = cpu_data.norm_matrix_probability; // Копируем входные данные
            gpu_data.ant_parametr = std::move(ant_parametr_result); // Перемещаем РЕЗУЛЬТАТЫ
            gpu_data.antOF = std::move(antOF_result);               // Перемещаем РЕЗУЛЬТАТЫ  
            gpu_data.minOf = minOf_result;
            gpu_data.maxOf = maxOf_result;
            gpu_data.kol_hash_fail = kol_hash_fail_result;
            gpu_data.ants_count = cpu_data.ants_count;
            gpu_data.Time_1 = time_1_result;
            gpu_data.Time_function = time_function_result;
            aco->Time_GPU += gpu_data.Time_1;
            aco->Time_GPU_function += gpu_data.Time_function;

#if (PRINT_INFORMATION)
            std::cout << "   [GPU ant Thread] Iteration " << gpu_data.iteration
                << " completed. Min: " << gpu_data.minOf
                << ", Max: " << gpu_data.maxOf
                << ", Hash fails: " << gpu_data.kol_hash_fail << std::endl;
#endif
            start_time_wait = std::chrono::high_resolution_clock::now();
            aco->gpu_to_cpu_queue_current.push(gpu_data);
#if (PRINT_INFORMATION)
                std::cout << "   [GPU ant Thread] queue_push(&aco->gpu_to_cpu_queue, gpu_data)" << std::endl;
#endif
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    aco->Time_GPU_all = std::chrono::duration<double, std::milli>(end_time - start_time).count();


#if (PRINT_INFORMATION)
    std::cout << "[GPU Thread] Finished" << std::endl;
#endif
}
void gpu_thread_function_parallel_non_hash(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "   [GPU ant Thread] Started" << std::endl;
#endif

    auto start_time = std::chrono::high_resolution_clock::now();
    auto start_time_wait = std::chrono::high_resolution_clock::now();
    while (!aco->stop_requested.load()) {
        IterationData cpu_data;
        if (!aco->cpu_to_gpu_queue_current.empty()) {
            cpu_data = std::move(aco->cpu_to_gpu_queue_current.front());
            aco->cpu_to_gpu_queue_current.pop();
            auto end_time_wait = std::chrono::high_resolution_clock::now();
            aco->Time_GPU_wait += std::chrono::duration<double, std::milli>(end_time_wait - start_time_wait).count();
#if (PRINT_INFORMATION)
            std::cout << "   [GPU ant Thread] queue_pop(&aco->cpu_to_gpu_queue, cpu_data)" << std::endl;
#endif
            active_cuda_tasks.fetch_add(1);
            // ВЫДЕЛЯЕМ ПАМЯТЬ ДЛЯ РЕЗУЛЬТАТОВ (GPU заполнит их)
            std::vector<int> ant_parametr_result(PARAMETR_SIZE * cpu_data.ants_count);
            std::vector<double> antOF_result(cpu_data.ants_count);
            double minOf_result = 1e9;
            double maxOf_result = -1e9;

            double time_1_result = 0.0;
            double time_function_result = 0.0;
#if (PRINT_INFORMATION)
            std::cout << "   [GPU ant Thread] Processing iteration " << cpu_data.iteration << " with " << cpu_data.ants_count << " ants " << ant_parametr_result.size() << " " << std::endl;
#endif
            // ВЫЗЫВАЕМ CUDA - передаем буферы для результатов
            cuda_run_iteration_non_hash(
                cpu_data.norm_matrix_probability.data(), // Вход: вероятности
                ant_parametr_result.data(),              // Выход: параметры муравьев
                antOF_result.data(),                     // Выход: значения функции
                cpu_data.ants_count,
                &minOf_result,
                &maxOf_result,
                &time_1_result,
                &time_function_result,
                cpu_data.iteration,
                cuda_completion_callback
            );

            // СОЗДАЕМ РЕЗУЛЬТАТ ДЛЯ CPU
            IterationData gpu_data;
            gpu_data.iteration = cpu_data.iteration;
            gpu_data.norm_matrix_probability = cpu_data.norm_matrix_probability; // Копируем входные данные
            gpu_data.ant_parametr = std::move(ant_parametr_result); // Перемещаем РЕЗУЛЬТАТЫ
            gpu_data.antOF = std::move(antOF_result);               // Перемещаем РЕЗУЛЬТАТЫ  
            gpu_data.minOf = minOf_result;
            gpu_data.maxOf = maxOf_result;
            gpu_data.kol_hash_fail = 0;
            gpu_data.ants_count = cpu_data.ants_count;
            gpu_data.Time_1 = time_1_result;
            gpu_data.Time_function = time_function_result;
            aco->Time_GPU += gpu_data.Time_1;
            aco->Time_GPU_function += gpu_data.Time_function;

#if (PRINT_INFORMATION)
            std::cout << "   [GPU ant Thread] Iteration " << gpu_data.iteration
                << " completed. Min: " << gpu_data.minOf
                << ", Max: " << gpu_data.maxOf
                << ", Hash fails: " << gpu_data.kol_hash_fail << std::endl;
#endif
            aco->gpu_to_cpu_queue_current.push(gpu_data);
#if (PRINT_INFORMATION)
            std::cout << "   [GPU ant Thread] queue_push(&aco->gpu_to_cpu_queue, gpu_data)" << std::endl;
#endif
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    start_time_wait = std::chrono::high_resolution_clock::now();
    aco->Time_GPU_all = std::chrono::duration<double, std::milli>(end_time - start_time).count();


#if (PRINT_INFORMATION)
    std::cout << "[GPU Thread] Finished" << std::endl;
#endif
}

void run_pipeline(ACOData* aco) {
    std::thread cpu_thread(cpu_thread_function, aco);
    std::thread gpu_thread(gpu_thread_function, aco);

    cpu_thread.join();
    gpu_thread.join();

    wait_completion();
}

void run_pipeline_non_hash(ACOData* aco) {
    std::thread cpu_thread(cpu_thread_function, aco);
    std::thread gpu_thread(gpu_thread_function_non_hash, aco);

    cpu_thread.join();
    gpu_thread.join();

    wait_completion();
}

int start_hybrid() {
    auto start_time = std::chrono::high_resolution_clock::now();
    ACOData aco;
    aco_init(&aco);

    aco.Time_GPU = 0.0;
    aco.Time_GPU_function = 0.0;
    aco.Time_GPU_all = 0.0;
    aco.Time_CPU_all = 0.0;
    aco.Time_CPU_prob = 0.0;
    aco.Time_CPU_wait = 0.0;
    aco.Time_CPU_update = 0.0;

    std::string filename = NAME_FILE_GRAPH;
    if (!aco_initialize_from_file(&aco, filename)) {
        std::cerr << "[Main] Trying to use default parameters..." << std::endl;

        const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
        std::vector<double> parametr_value(matrix_size);
        std::vector<double> pheromon_value(matrix_size, 1.0);
        std::vector<double> kol_enter_value(matrix_size, 1.0);

        for (int i = 0; i < PARAMETR_SIZE; i++) {
            for (int j = 0; j < MAX_VALUE_SIZE; j++) {
                parametr_value[i * MAX_VALUE_SIZE + j] = (i * MAX_VALUE_SIZE + j) * 0.01;
            }
        }

        if (!aco_initialize(&aco, parametr_value, pheromon_value, kol_enter_value)) {
            std::cerr << "Failed to initialize ACO!" << std::endl;
            return 1;
        }
    }

    auto start_time2 = std::chrono::high_resolution_clock::now();
    run_pipeline(&aco);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    auto duration_iteration = std::chrono::duration<double, std::milli>(end_time - start_time2).count();

#if (PRINT_INFORMATION)
    std::cout << "\n=== PIPELINE COMPLETED ===" << std::endl;
    std::cout << "Total time: " << duration << " ms" << std::endl;
    std::cout << "Completed iterations: " << completed_iterations.load() << std::endl;
    std::cout << "Active tasks remaining: " << active_cuda_tasks.load() << std::endl;
    std::cout << "Global Min: " << aco.global_minOf << std::endl;
    std::cout << "Global Max: " << aco.global_maxOf << std::endl;
    std::cout << "Hash fails: " << aco.kol_hash_fail << std::endl;
#endif

    std::cout << "Time Hybrid OMP;" << duration << "; " << duration_iteration << "; "
        << aco.Time_GPU_all << "; " << aco.Time_GPU << "; " << aco.Time_GPU_function << "; " << aco.Time_GPU_wait << "; "
        << aco.Time_CPU_all << "; " << aco.Time_CPU_prob << "; " << aco.Time_CPU_wait << "; " << aco.Time_CPU_update << "; "
        << aco.Time_OMP_all << "; " << aco.Time_OMP_function << "; " << aco.Time_OMP_wait << "; "
        << aco.global_minOf << "; " << aco.global_maxOf << "; "
        << aco.kol_hash_fail << "; " << std::endl;
    logFile << "Time Hybrid OMP;" << duration << "; " << duration_iteration << "; "
        << aco.Time_GPU_all << "; " << aco.Time_GPU << "; " << aco.Time_GPU_function << "; " << aco.Time_GPU_wait << "; "
        << aco.Time_CPU_all << "; " << aco.Time_CPU_prob << "; " << aco.Time_CPU_wait << "; " << aco.Time_CPU_update << "; "
        << aco.Time_OMP_all << "; " << aco.Time_OMP_function << "; " << aco.Time_OMP_wait << "; "
        << aco.global_minOf << "; " << aco.global_maxOf << "; "
        << aco.kol_hash_fail << "; " << std::endl;

    aco_cleanup(&aco);
    return 0;
}

int start_hybrid_non_hash() {
    auto start_time = std::chrono::high_resolution_clock::now();
    ACOData aco;
    aco_init(&aco);

    aco.Time_GPU = 0.0;
    aco.Time_GPU_function = 0.0;
    aco.Time_GPU_all = 0.0;
    aco.Time_CPU_all = 0.0;
    aco.Time_CPU_prob = 0.0;
    aco.Time_CPU_wait = 0.0;
    aco.Time_CPU_update = 0.0;

    std::string filename = NAME_FILE_GRAPH;
    if (!aco_initialize_from_file_non_hash(&aco, filename)) {
        std::cerr << "[Main] Trying to use default parameters..." << std::endl;

        const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
        std::vector<double> parametr_value(matrix_size);
        std::vector<double> pheromon_value(matrix_size, 1.0);
        std::vector<double> kol_enter_value(matrix_size, 1.0);

        for (int i = 0; i < PARAMETR_SIZE; i++) {
            for (int j = 0; j < MAX_VALUE_SIZE; j++) {
                parametr_value[i * MAX_VALUE_SIZE + j] = (i * MAX_VALUE_SIZE + j) * 0.01;
            }
        }

        if (!aco_initialize_non_hash(&aco, parametr_value, pheromon_value, kol_enter_value)) {
            std::cerr << "Failed to initialize ACO!" << std::endl;
            return 1;
        }
    }

    auto start_time2 = std::chrono::high_resolution_clock::now();
    run_pipeline_non_hash(&aco);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    auto duration_iteration = std::chrono::duration<double, std::milli>(end_time - start_time2).count();

#if (PRINT_INFORMATION)
    std::cout << "\n=== PIPELINE COMPLETED ===" << std::endl;
    std::cout << "Total time: " << duration << " ms" << std::endl;
    std::cout << "Completed iterations: " << completed_iterations.load() << std::endl;
    std::cout << "Active tasks remaining: " << active_cuda_tasks.load() << std::endl;
    std::cout << "Global Min: " << aco.global_minOf << std::endl;
    std::cout << "Global Max: " << aco.global_maxOf << std::endl;
    std::cout << "Hash fails: " << aco.kol_hash_fail << std::endl;
#endif

    std::cout << "Time Hybrid OMP non hash;" << duration << "; " << duration_iteration << "; "
        << aco.Time_GPU_all << "; " << aco.Time_GPU << "; " << aco.Time_GPU_function << "; " << aco.Time_GPU_wait << "; "
        << aco.Time_CPU_all << "; " << aco.Time_CPU_prob << "; " << aco.Time_CPU_wait << "; " << aco.Time_CPU_update << "; "
        << aco.Time_OMP_all << "; " << aco.Time_OMP_function << "; " << aco.Time_OMP_wait << "; "
        << aco.global_minOf << "; " << aco.global_maxOf << "; "
        << aco.kol_hash_fail << "; " << std::endl;
    logFile << "Time Hybrid OMP non hash;" << duration << "; " << duration_iteration << "; "
        << aco.Time_GPU_all << "; " << aco.Time_GPU << "; " << aco.Time_GPU_function << "; " << aco.Time_GPU_wait << "; "
        << aco.Time_CPU_all << "; " << aco.Time_CPU_prob << "; " << aco.Time_CPU_wait << "; " << aco.Time_CPU_update << "; "
        << aco.Time_OMP_all << "; " << aco.Time_OMP_function << "; " << aco.Time_OMP_wait << "; "
        << aco.global_minOf << "; " << aco.global_maxOf << "; "
        << aco.kol_hash_fail << "; " << std::endl;

    aco_cleanup(&aco);
    return 0;
}

// ========================= GO_HYBRID_BALANCED_OMP РЕЖИМ =========================

void run_balanced_pipeline(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "[Balanced Pipeline] Starting balanced execution..." << std::endl;
#endif

    std::thread cpu_thread(cpu_thread_function_OMP, aco);
    std::thread gpu_thread(gpu_thread_function, aco);
    std::thread OMP_thread(OMP_thread_function, aco);


    cpu_thread.join();
    gpu_thread.join();
    OMP_thread.join();


    wait_completion();
}
void run_balanced_pipeline_non_hash(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "[Balanced Pipeline] Starting balanced execution..." << std::endl;
#endif

    std::thread cpu_thread(cpu_thread_function_OMP, aco);
    std::thread gpu_thread(gpu_thread_function_non_hash, aco);
    std::thread OMP_thread(OMP_thread_function_non_hash, aco);


    cpu_thread.join();
    gpu_thread.join();
    OMP_thread.join();


    wait_completion();
}

void run_balanced_dynamic_pipeline(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "[Balanced Pipeline] Starting dynamic balanced execution..." << std::endl;
#endif

    std::thread cpu_thread(dynamic_cpu_thread_function_OMP, aco);
    std::thread gpu_thread(gpu_thread_function, aco);
    std::thread OMP_thread(OMP_thread_function, aco);


    cpu_thread.join();
    gpu_thread.join();
    OMP_thread.join();


    wait_completion();
}
void run_balanced_dynamic_pipeline_non_hash(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "[Balanced Pipeline] Starting dynamic balanced execution..." << std::endl;
#endif

    std::thread cpu_thread(dynamic_cpu_thread_function_OMP, aco);
    std::thread gpu_thread(gpu_thread_function_non_hash, aco);
    std::thread OMP_thread(OMP_thread_function_non_hash, aco);


    cpu_thread.join();
    gpu_thread.join();
    OMP_thread.join();


    wait_completion();
}

void run_balanced_dynamic_pipeline_parallel(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "[Balanced Pipeline] Starting dynamic parallel balanced execution..." << std::endl;
#endif

    std::thread cpu_thread(dynamic_cpu_thread_function_OMP_parallel, aco);
    std::thread gpu_thread(gpu_thread_function_parallel, aco);
    std::thread OMP_thread(OMP_thread_function_parallel, aco);


    cpu_thread.join();
    gpu_thread.join();
    OMP_thread.join();


    wait_completion();
}
void run_balanced_dynamic_pipeline_parallel_non_hash(ACOData* aco) {
#if (PRINT_INFORMATION)
    std::cout << "[Balanced Pipeline] Starting dynamic parallel balanced execution..." << std::endl;
#endif

    std::thread cpu_thread(dynamic_cpu_thread_function_OMP_parallel, aco);
    std::thread gpu_thread(gpu_thread_function_parallel_non_hash, aco);
    std::thread OMP_thread(OMP_thread_function_parallel_non_hash, aco);


    cpu_thread.join();
    gpu_thread.join();
    OMP_thread.join();


    wait_completion();
}

int start_balanced_hybrid() {
#if (PRINT_INFORMATION)
    std::cout << "=== BALANCED HYBRID START ===" << std::endl;
#endif
    auto start_time = std::chrono::high_resolution_clock::now();

    ACOData aco;
    aco_init(&aco);

    // Инициализация хэш-таблицы
    hashTable = new HashEntry[HASH_TABLE_SIZE];
    if (hashTable == nullptr) {
        std::cerr << "[Main] Hash Error!" << std::endl;
        return 1;
    }
    initializeHashTable_non_cuda(hashTable, HASH_TABLE_SIZE);

    std::string filename = NAME_FILE_GRAPH;
    if (!aco_initialize_from_file(&aco, filename)) {
        std::cerr << "[Main] Trying to use default parameters..." << std::endl;

        const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
        std::vector<double> parametr_value(matrix_size);
        std::vector<double> pheromon_value(matrix_size, 1.0);
        std::vector<double> kol_enter_value(matrix_size, 1.0);

        for (int i = 0; i < PARAMETR_SIZE; i++) {
            for (int j = 0; j < MAX_VALUE_SIZE; j++) {
                parametr_value[i * MAX_VALUE_SIZE + j] = (i * MAX_VALUE_SIZE + j) * 0.01;
            }
        }

        if (!aco_initialize(&aco, parametr_value, pheromon_value, kol_enter_value)) {
            std::cerr << "Failed to initialize ACO!" << std::endl;
            return 1;
        }
    }

#if (PRINT_INFORMATION)
    std::cout << "=== run_balanced_pipeline() START ===" << std::endl;
#endif
    auto start_time2 = std::chrono::high_resolution_clock::now();
    run_balanced_pipeline(&aco);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    auto duration_iteration = std::chrono::duration<double, std::milli>(end_time - start_time2).count();
#if (PRINT_INFORMATION)
    std::cout << "=== BALANCED HYBRID COMPLETED ===" << std::endl;
    std::cout << "Total time: " << duration << " ms" << std::endl;
#endif
    aco.cpu_ants_statistics = aco.cpu_ants_statistics / (ANT_SIZE * KOL_ITERATION);
    aco.gpu_ants_statistics = aco.gpu_ants_statistics / (ANT_SIZE * KOL_ITERATION);
    std::cout << "Time Hybrid OMP parallel 1/2;" << duration << "; " << duration_iteration << "; "
        << aco.Time_GPU_all << "; " << aco.Time_GPU << "; " << aco.Time_GPU_function << "; " << aco.Time_GPU_wait << "; "
        << aco.Time_CPU_all << "; " << aco.Time_CPU_prob << "; " << aco.Time_CPU_wait << "; " << aco.Time_CPU_update << "; "
        << aco.Time_OMP_all << "; " << aco.Time_OMP_function << "; " << aco.Time_OMP_wait << "; "
        << aco.global_minOf << "; " << aco.global_maxOf << "; "
        << aco.cpu_ants_statistics << "; " << aco.gpu_ants_statistics << "; "
        << aco.kol_hash_fail << "; " << std::endl;
    logFile << "Time Hybrid OMP parallel 1/2;" << duration << "; " << duration_iteration << "; "
        << aco.Time_GPU_all << "; " << aco.Time_GPU << "; " << aco.Time_GPU_function << "; " << aco.Time_GPU_wait << "; "
        << aco.Time_CPU_all << "; " << aco.Time_CPU_prob << "; " << aco.Time_CPU_wait << "; " << aco.Time_CPU_update << "; "
        << aco.Time_OMP_all << "; " << aco.Time_OMP_function << "; " << aco.Time_OMP_wait << "; "
        << aco.global_minOf << "; " << aco.global_maxOf << "; "
        << aco.cpu_ants_statistics << "; " << aco.gpu_ants_statistics << "; "
        << aco.kol_hash_fail << "; " << std::endl;


    aco_cleanup(&aco);
    // Освобождение памяти хэш-таблицы на CPU
    if (hashTable != nullptr) {
        delete[] hashTable;
        hashTable = nullptr;
    }
    return 0;
}
int start_balanced_hybrid_non_hash() {
#if (PRINT_INFORMATION)
    std::cout << "=== BALANCED HYBRID START ===" << std::endl;
#endif
    auto start_time = std::chrono::high_resolution_clock::now();

    ACOData aco;
    aco_init(&aco);

    std::string filename = NAME_FILE_GRAPH;
    if (!aco_initialize_from_file_non_hash(&aco, filename)) {
        std::cerr << "[Main] Trying to use default parameters..." << std::endl;

        const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
        std::vector<double> parametr_value(matrix_size);
        std::vector<double> pheromon_value(matrix_size, 1.0);
        std::vector<double> kol_enter_value(matrix_size, 1.0);

        for (int i = 0; i < PARAMETR_SIZE; i++) {
            for (int j = 0; j < MAX_VALUE_SIZE; j++) {
                parametr_value[i * MAX_VALUE_SIZE + j] = (i * MAX_VALUE_SIZE + j) * 0.01;
            }
        }

        if (!aco_initialize_non_hash(&aco, parametr_value, pheromon_value, kol_enter_value)) {
            std::cerr << "Failed to initialize ACO!" << std::endl;
            return 1;
        }
    }

#if (PRINT_INFORMATION)
    std::cout << "=== run_balanced_pipeline() START ===" << std::endl;
#endif
    auto start_time2 = std::chrono::high_resolution_clock::now();
    run_balanced_pipeline_non_hash(&aco);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    auto duration_iteration = std::chrono::duration<double, std::milli>(end_time - start_time2).count();
#if (PRINT_INFORMATION)
    std::cout << "=== BALANCED HYBRID COMPLETED ===" << std::endl;
    std::cout << "Total time: " << duration << " ms" << std::endl;
#endif
    aco.cpu_ants_statistics = aco.cpu_ants_statistics / (ANT_SIZE * KOL_ITERATION);
    aco.gpu_ants_statistics = aco.gpu_ants_statistics / (ANT_SIZE * KOL_ITERATION);
    std::cout << "Time Hybrid OMP parallel 1/2 non hash;" << duration << "; " << duration_iteration << "; "
        << aco.Time_GPU_all << "; " << aco.Time_GPU << "; " << aco.Time_GPU_function << "; " << aco.Time_GPU_wait << "; "
        << aco.Time_CPU_all << "; " << aco.Time_CPU_prob << "; " << aco.Time_CPU_wait << "; " << aco.Time_CPU_update << "; "
        << aco.Time_OMP_all << "; " << aco.Time_OMP_function << "; " << aco.Time_OMP_wait << "; "
        << aco.global_minOf << "; " << aco.global_maxOf << "; "
        << aco.cpu_ants_statistics << "; " << aco.gpu_ants_statistics << "; "
        << aco.kol_hash_fail << "; " << std::endl;
    logFile << "Time Hybrid OMP parallel 1/2 non hash;" << duration << "; " << duration_iteration << "; "
        << aco.Time_GPU_all << "; " << aco.Time_GPU << "; " << aco.Time_GPU_function << "; " << aco.Time_GPU_wait << "; "
        << aco.Time_CPU_all << "; " << aco.Time_CPU_prob << "; " << aco.Time_CPU_wait << "; " << aco.Time_CPU_update << "; "
        << aco.Time_OMP_all << "; " << aco.Time_OMP_function << "; " << aco.Time_OMP_wait << "; "
        << aco.global_minOf << "; " << aco.global_maxOf << "; "
        << aco.cpu_ants_statistics << "; " << aco.gpu_ants_statistics << "; "
        << aco.kol_hash_fail << "; " << std::endl;

    aco_cleanup(&aco);

    return 0;
}

int start_balanced_hybrid_dynamic() {
#if (PRINT_INFORMATION)
    std::cout << "=== BALANCED HYBRID START ===" << std::endl;
#endif
    auto start_time = std::chrono::high_resolution_clock::now();

    ACOData aco;
    aco_init(&aco);

    // Инициализация хэш-таблицы
    hashTable = new HashEntry[HASH_TABLE_SIZE];
    if (hashTable == nullptr) {
        std::cerr << "[Main] Hash Error!" << std::endl;
        return 1;
    }
    initializeHashTable_non_cuda(hashTable, HASH_TABLE_SIZE);

    std::string filename = NAME_FILE_GRAPH;
    if (!aco_initialize_from_file(&aco, filename)) {
        std::cerr << "[Main] Trying to use default parameters..." << std::endl;

        const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
        std::vector<double> parametr_value(matrix_size);
        std::vector<double> pheromon_value(matrix_size, 1.0);
        std::vector<double> kol_enter_value(matrix_size, 1.0);

        for (int i = 0; i < PARAMETR_SIZE; i++) {
            for (int j = 0; j < MAX_VALUE_SIZE; j++) {
                parametr_value[i * MAX_VALUE_SIZE + j] = (i * MAX_VALUE_SIZE + j) * 0.01;
            }
        }

        if (!aco_initialize(&aco, parametr_value, pheromon_value, kol_enter_value)) {
            std::cerr << "Failed to initialize ACO!" << std::endl;
            return 1;
        }
    }

#if (PRINT_INFORMATION)
    std::cout << "=== run_balanced_pipeline() START ===" << std::endl;
#endif
    auto start_time2 = std::chrono::high_resolution_clock::now();
    run_balanced_dynamic_pipeline(&aco);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    auto duration_iteration = std::chrono::duration<double, std::milli>(end_time - start_time2).count();
#if (PRINT_INFORMATION)
    std::cout << "=== BALANCED HYBRID COMPLETED ===" << std::endl;
    std::cout << "Total time: " << duration << " ms" << std::endl;
#endif
    aco.cpu_ants_statistics = aco.cpu_ants_statistics / (ANT_SIZE * KOL_ITERATION);
    aco.gpu_ants_statistics = aco.gpu_ants_statistics / (ANT_SIZE * KOL_ITERATION);
    std::cout << "Time Hybrid OMP parallel dynamic;" << duration << "; " << duration_iteration << "; "
        << aco.Time_GPU_all << "; " << aco.Time_GPU << "; " << aco.Time_GPU_function << "; " << aco.Time_GPU_wait << "; "
        << aco.Time_CPU_all << "; " << aco.Time_CPU_prob << "; " << aco.Time_CPU_wait << "; " << aco.Time_CPU_update << "; "
        << aco.Time_OMP_all << "; " << aco.Time_OMP_function << "; " << aco.Time_OMP_wait << "; "
        << aco.global_minOf << "; " << aco.global_maxOf << "; "
        << aco.cpu_ants_statistics << "; " << aco.gpu_ants_statistics << "; "
        << aco.kol_hash_fail << "; " << std::endl;
    logFile << "Time Hybrid OMP parallel dynamic;" << duration << "; " << duration_iteration << "; "
        << aco.Time_GPU_all << "; " << aco.Time_GPU << "; " << aco.Time_GPU_function << "; " << aco.Time_GPU_wait << "; "
        << aco.Time_CPU_all << "; " << aco.Time_CPU_prob << "; " << aco.Time_CPU_wait << "; " << aco.Time_CPU_update << "; "
        << aco.Time_OMP_all << "; " << aco.Time_OMP_function << "; " << aco.Time_OMP_wait << "; "
        << aco.global_minOf << "; " << aco.global_maxOf << "; "
        << aco.cpu_ants_statistics << "; " << aco.gpu_ants_statistics << "; "
        << aco.kol_hash_fail << "; " << std::endl;


    aco_cleanup(&aco);
    // Освобождение памяти хэш-таблицы на CPU
    if (hashTable != nullptr) {
        delete[] hashTable;
        hashTable = nullptr;
    }
    return 0;
}
int start_balanced_hybrid_dynamic_non_hash() {
#if (PRINT_INFORMATION)
    std::cout << "=== BALANCED HYBRID START ===" << std::endl;
#endif
    auto start_time = std::chrono::high_resolution_clock::now();

    ACOData aco;
    aco_init(&aco);

    std::string filename = NAME_FILE_GRAPH;
    if (!aco_initialize_from_file_non_hash(&aco, filename)) {
        std::cerr << "[Main] Trying to use default parameters..." << std::endl;

        const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
        std::vector<double> parametr_value(matrix_size);
        std::vector<double> pheromon_value(matrix_size, 1.0);
        std::vector<double> kol_enter_value(matrix_size, 1.0);

        for (int i = 0; i < PARAMETR_SIZE; i++) {
            for (int j = 0; j < MAX_VALUE_SIZE; j++) {
                parametr_value[i * MAX_VALUE_SIZE + j] = (i * MAX_VALUE_SIZE + j) * 0.01;
            }
        }

        if (!aco_initialize_non_hash(&aco, parametr_value, pheromon_value, kol_enter_value)) {
            std::cerr << "Failed to initialize ACO!" << std::endl;
            return 1;
        }
    }

#if (PRINT_INFORMATION)
    std::cout << "=== run_balanced_pipeline() START ===" << std::endl;
#endif
    auto start_time2 = std::chrono::high_resolution_clock::now();
    run_balanced_dynamic_pipeline_non_hash(&aco);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    auto duration_iteration = std::chrono::duration<double, std::milli>(end_time - start_time2).count();
#if (PRINT_INFORMATION)
    std::cout << "=== BALANCED HYBRID COMPLETED ===" << std::endl;
    std::cout << "Total time: " << duration << " ms" << std::endl;
#endif
    aco.cpu_ants_statistics = aco.cpu_ants_statistics / (ANT_SIZE * KOL_ITERATION);
    aco.gpu_ants_statistics = aco.gpu_ants_statistics / (ANT_SIZE * KOL_ITERATION);
    std::cout << "Time Hybrid OMP parallel dynamic non hash;" << duration << "; " << duration_iteration << "; "
        << aco.Time_GPU_all << "; " << aco.Time_GPU << "; " << aco.Time_GPU_function << "; " << aco.Time_GPU_wait << "; "
        << aco.Time_CPU_all << "; " << aco.Time_CPU_prob << "; " << aco.Time_CPU_wait << "; " << aco.Time_CPU_update << "; "
        << aco.Time_OMP_all << "; " << aco.Time_OMP_function << "; " << aco.Time_OMP_wait << "; "
        << aco.global_minOf << "; " << aco.global_maxOf << "; "
        << aco.cpu_ants_statistics << "; " << aco.gpu_ants_statistics << "; "
        << aco.kol_hash_fail << "; " << std::endl;
    logFile << "Time Hybrid OMP parallel dynamic non hash;" << duration << "; " << duration_iteration << "; "
        << aco.Time_GPU_all << "; " << aco.Time_GPU << "; " << aco.Time_GPU_function << "; " << aco.Time_GPU_wait << "; "
        << aco.Time_CPU_all << "; " << aco.Time_CPU_prob << "; " << aco.Time_CPU_wait << "; " << aco.Time_CPU_update << "; "
        << aco.Time_OMP_all << "; " << aco.Time_OMP_function << "; " << aco.Time_OMP_wait << "; "
        << aco.global_minOf << "; " << aco.global_maxOf << "; "
        << aco.cpu_ants_statistics << "; " << aco.gpu_ants_statistics << "; "
        << aco.kol_hash_fail << "; " << std::endl;

    aco_cleanup(&aco);

    return 0;
}

int start_balanced_hybrid_dynamic_parallel() {
#if (PRINT_INFORMATION)
    std::cout << "=== BALANCED HYBRID START ===" << std::endl;
#endif
    auto start_time = std::chrono::high_resolution_clock::now();

    ACOData aco;
    aco_init(&aco);

    // Инициализация хэш-таблицы
    hashTable = new HashEntry[HASH_TABLE_SIZE];
    if (hashTable == nullptr) {
        std::cerr << "[Main] Hash Error!" << std::endl;
        return 1;
    }
    initializeHashTable_non_cuda(hashTable, HASH_TABLE_SIZE);

    std::string filename = NAME_FILE_GRAPH;
    if (!aco_initialize_from_file(&aco, filename)) {
        std::cerr << "[Main] Trying to use default parameters..." << std::endl;

        const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
        std::vector<double> parametr_value(matrix_size);
        std::vector<double> pheromon_value(matrix_size, 1.0);
        std::vector<double> kol_enter_value(matrix_size, 1.0);

        for (int i = 0; i < PARAMETR_SIZE; i++) {
            for (int j = 0; j < MAX_VALUE_SIZE; j++) {
                parametr_value[i * MAX_VALUE_SIZE + j] = (i * MAX_VALUE_SIZE + j) * 0.01;
            }
        }

        if (!aco_initialize(&aco, parametr_value, pheromon_value, kol_enter_value)) {
            std::cerr << "Failed to initialize ACO!" << std::endl;
            return 1;
        }
    }

#if (PRINT_INFORMATION)
    std::cout << "=== run_balanced_dynamic_pipeline_parallel() START ===" << std::endl;
#endif
    auto start_time2 = std::chrono::high_resolution_clock::now();
    run_balanced_dynamic_pipeline_parallel(&aco);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    auto duration_iteration = std::chrono::duration<double, std::milli>(end_time - start_time2).count();
#if (PRINT_INFORMATION)
    std::cout << "=== BALANCED PARALLEL HYBRID COMPLETED ===" << std::endl;
    std::cout << "Total time: " << duration << " ms" << std::endl;
#endif
    aco.cpu_ants_statistics = aco.cpu_ants_statistics / (ANT_SIZE * KOL_ITERATION);
    aco.gpu_ants_statistics = aco.gpu_ants_statistics / (ANT_SIZE * KOL_ITERATION);
    
    std::cout << "Time Hybrid OMP parallel thread non blocked dynamic;" << duration << "; " << duration_iteration << "; "
        << aco.Time_GPU_all << "; " << aco.Time_GPU << "; " << aco.Time_GPU_function << "; " << aco.Time_GPU_wait << "; "
        << aco.Time_CPU_all << "; " << aco.Time_CPU_prob << "; " << aco.Time_CPU_wait << "; " << aco.Time_CPU_update << "; "
        << aco.Time_OMP_all << "; " << aco.Time_OMP_function << "; " << aco.Time_OMP_wait << "; "
        << aco.global_minOf << "; " << aco.global_maxOf << "; "
        << aco.cpu_ants_statistics << "; " << aco.gpu_ants_statistics << "; "
        << aco.kol_hash_fail << "; " << std::endl;
    logFile << "Time Hybrid OMP parallel thread non blocked dynamic;" << duration << "; " << duration_iteration << "; "
        << aco.Time_GPU_all << "; " << aco.Time_GPU << "; " << aco.Time_GPU_function << "; " << aco.Time_GPU_wait << "; "
        << aco.Time_CPU_all << "; " << aco.Time_CPU_prob << "; " << aco.Time_CPU_wait << "; " << aco.Time_CPU_update << "; "
        << aco.Time_OMP_all << "; " << aco.Time_OMP_function << "; " << aco.Time_OMP_wait << "; "
        << aco.global_minOf << "; " << aco.global_maxOf << "; "
        << aco.cpu_ants_statistics << "; " << aco.gpu_ants_statistics << "; "
        << aco.kol_hash_fail << "; " << std::endl;


    aco_cleanup(&aco);
    // Освобождение памяти хэш-таблицы на CPU
    if (hashTable != nullptr) {
        delete[] hashTable;
        hashTable = nullptr;
    }
    return 0;
}
int start_balanced_hybrid_dynamic_parallel_non_hash() {
#if (PRINT_INFORMATION)
    std::cout << "=== BALANCED HYBRID START ===" << std::endl;
#endif
    auto start_time = std::chrono::high_resolution_clock::now();

    ACOData aco;
    aco_init(&aco);

    std::string filename = NAME_FILE_GRAPH;
    if (!aco_initialize_from_file_non_hash(&aco, filename)) {
        std::cerr << "[Main] Trying to use default parameters..." << std::endl;

        const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
        std::vector<double> parametr_value(matrix_size);
        std::vector<double> pheromon_value(matrix_size, 1.0);
        std::vector<double> kol_enter_value(matrix_size, 1.0);

        for (int i = 0; i < PARAMETR_SIZE; i++) {
            for (int j = 0; j < MAX_VALUE_SIZE; j++) {
                parametr_value[i * MAX_VALUE_SIZE + j] = (i * MAX_VALUE_SIZE + j) * 0.01;
            }
        }

        if (!aco_initialize_non_hash(&aco, parametr_value, pheromon_value, kol_enter_value)) {
            std::cerr << "Failed to initialize ACO!" << std::endl;
            return 1;
        }
    }

#if (PRINT_INFORMATION)
    std::cout << "=== run_balanced_dynamic_pipeline_parallel() START ===" << std::endl;
#endif
    auto start_time2 = std::chrono::high_resolution_clock::now();
    run_balanced_dynamic_pipeline_parallel_non_hash(&aco);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    auto duration_iteration = std::chrono::duration<double, std::milli>(end_time - start_time2).count();
#if (PRINT_INFORMATION)
    std::cout << "=== BALANCED PARALLEL HYBRID COMPLETED ===" << std::endl;
    std::cout << "Total time: " << duration << " ms" << std::endl;
#endif
    aco.cpu_ants_statistics = aco.cpu_ants_statistics / (ANT_SIZE * KOL_ITERATION);
    aco.gpu_ants_statistics = aco.gpu_ants_statistics / (ANT_SIZE * KOL_ITERATION);
    std::cout << "Time Hybrid OMP parallel thread non blocked dynamic non hash;" << duration << "; " << duration_iteration << "; "
        << aco.Time_GPU_all << "; " << aco.Time_GPU << "; " << aco.Time_GPU_function << "; " << aco.Time_GPU_wait << "; "
        << aco.Time_CPU_all << "; " << aco.Time_CPU_prob << "; " << aco.Time_CPU_wait << "; " << aco.Time_CPU_update << "; "
        << aco.Time_OMP_all << "; " << aco.Time_OMP_function << "; " << aco.Time_OMP_wait << "; "
        << aco.global_minOf << "; " << aco.global_maxOf << "; "
        << aco.cpu_ants_statistics << "; " << aco.gpu_ants_statistics << "; "
        << aco.kol_hash_fail << "; " << std::endl;
    logFile << "Time Hybrid OMP parallel thread non blocked dynamic non hash;" << duration << "; " << duration_iteration << "; "
        << aco.Time_GPU_all << "; " << aco.Time_GPU << "; " << aco.Time_GPU_function << "; " << aco.Time_GPU_wait << "; "
        << aco.Time_CPU_all << "; " << aco.Time_CPU_prob << "; " << aco.Time_CPU_wait << "; " << aco.Time_CPU_update << "; "
        << aco.Time_OMP_all << "; " << aco.Time_OMP_function << "; " << aco.Time_OMP_wait << "; "
        << aco.global_minOf << "; " << aco.global_maxOf << "; "
        << aco.cpu_ants_statistics << "; " << aco.gpu_ants_statistics << "; "
        << aco.kol_hash_fail << "; " << std::endl;

    aco_cleanup(&aco);
    return 0;
}
// Общие функции

void wait_completion(int max_wait_ms) {
    auto start = std::chrono::steady_clock::now();

    while (active_cuda_tasks.load() > 0) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);

        if (elapsed.count() > max_wait_ms) {
            std::cerr << "[Main] Timeout waiting for CUDA completion!" << std::endl;
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    cuda_synchronize();
}

void aco_cleanup(ACOData* aco) {
    aco->stop_requested.store(true);
    queue_stop(&aco->cpu_to_gpu_queue);
    queue_stop(&aco->gpu_to_cpu_queue);
    wait_completion(10000);
    cuda_cleanup();

#if (PRINT_INFORMATION)
    std::cout << "[Main] Cleanup completed. Successful iterations: "
        << successful_iterations.load() << std::endl;
    std::cout << "[Main] Final results - Min: " << aco->global_minOf
        << ", Max: " << aco->global_maxOf
        << ", Hash fails: " << aco->kol_hash_fail << std::endl;
#endif
}

// Функция main
int main() {
    std::cout << __cplusplus << std::endl;

    logFile.open("log.txt");
    if (!logFile.is_open()) {
        std::cerr << "Ошибка открытия лог-файла!" << std::endl;
        return 1;
    }

    // Вывод информации о версиях
    std::cout << "Max threads OMP : " << omp_get_max_threads() << " ";
    std::cout << "OpenMP version: " << _OPENMP << " :";
#if _OPENMP >= 202411 
    std::cout << "OpenMP 6.0 (2026) plane" << std::endl;
#elif _OPENMP >= 202111 
    std::cout << "OpenMP 5.2 (2023) active" << std::endl;
#elif _OPENMP >= 202011 
    std::cout << "OpenMP 5.1 (2021) active" << std::endl;
#elif _OPENMP >= 201811 
    std::cout << "OpenMP 5.0 (2018) active" << std::endl;
#elif _OPENMP >= 201511 
    std::cout << "OpenMP 4.5 (2015) optimal" << std::endl;
#elif _OPENMP >= 201307 
    std::cout << "OpenMP 4.0 (2013) active" << std::endl;
#elif _OPENMP >= 201107 
    std::cout << "OpenMP 3.1 (2011) supported" << std::endl;
#elif _OPENMP >= 200805 
    std::cout << "OpenMP 3.0 (2008) supported" << std::endl;
#elif _OPENMP >= 200505 
    std::cout << "OpenMP 2.5 (2005) outdated" << std::endl;
#elif _OPENMP >= 200203 
    std::cout << "OpenMP 2.0 (2002) outdated" << std::endl;
#elif _OPENMP >= 199710 
    std::cout << "OpenMP 1.0 (1999) outdated" << std::endl;
#else 
    std::cout << "Older OpenMP version" << std::endl;
#endif
    logFile << "Max threads OMP : " << omp_get_max_threads() << " ";
    logFile << "OpenMP version: " << _OPENMP << " :";
#if _OPENMP >= 202411 
    logFile << "OpenMP 6.0 (2026) plane" << std::endl;
#elif _OPENMP >= 202111 
    logFile << "OpenMP 5.2 (2023) active" << std::endl;
#elif _OPENMP >= 202011 
    logFile << "OpenMP 5.1 (2021) active" << std::endl;
#elif _OPENMP >= 201811 
    logFile << "OpenMP 5.0 (2018) active" << std::endl;
#elif _OPENMP >= 201511 
    logFile << "OpenMP 4.5 (2015) optimal" << std::endl;
#elif _OPENMP >= 201307 
    logFile << "OpenMP 4.0 (2013) active" << std::endl;
#elif _OPENMP >= 201107 
    logFile << "OpenMP 3.1 (2011) supported" << std::endl;
#elif _OPENMP >= 200805 
    logFile << "OpenMP 3.0 (2008) supported" << std::endl;
#elif _OPENMP >= 200505 
    logFile << "OpenMP 2.5 (2005) outdated" << std::endl;
#elif _OPENMP >= 200203 
    logFile << "OpenMP 2.0 (2002) outdated" << std::endl;
#elif _OPENMP >= 199710 
    logFile << "OpenMP 1.0 (1999) outdated" << std::endl;
#else 
    logFile << "Older OpenMP version" << std::endl;
#endif
    
    // Вывод информации о константах
    std::cout << "PARAMETR_SIZE: " << PARAMETR_SIZE << "; "
        << "MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << "; "
        << "SET_PARAMETR_SIZE_ONE_X: " << SET_PARAMETR_SIZE_ONE_X << "; "
        << "ANT_SIZE: " << ANT_SIZE << "; "
        << "MAX_THREAD_CUDA: " << MAX_THREAD_CUDA << "; "
        << "NAME_FILE_GRAPH: " << NAME_FILE_GRAPH << "; "
        << "KOL_ITERATION: " << KOL_ITERATION << "; "
        << "KOL_PROGREV: " << KOL_PROGREV << "; "
        << "KOL_PROGON_STATISTICS: " << KOL_PROGON_STATISTICS << "; "
        << "PARAMETR_RO: " << PARAMETR_RO << "; "
        << "PARAMETR_Q: " << PARAMETR_Q << "; "
        << "PRINT_INFORMATION: " << PRINT_INFORMATION << "; "
        << "MAX_PARAMETR_VALUE_TO_MIN_OPT: " << MAX_PARAMETR_VALUE_TO_MIN_OPT << "; "
        << "OPTIMIZE: " << (OPTIMIZE_MIN_1 ? "OPTIMIZE_MIN_1 " : "") << (OPTIMIZE_MIN_2 ? "OPTIMIZE_MIN_2 " : "") << (OPTIMIZE_MAX ? "OPTIMIZE_MAX " : "") << "; "
        << "FUNCTION: " << (SHAFFERA ? "SHAFFERA " : "") << (CARROM_TABLE ? "CARROM_TABLE " : "") << (RASTRIGIN ? "RASTRIGIN " : "") << (ACKLEY ? "ACKLEY " : "") << (SPHERE ? "SPHERE " : "") << (GRIEWANK ? "GRIEWANK " : "") << (ZAKHAROV ? "ZAKHAROV " : "") << (SCHWEFEL ? "SCHWEFEL " : "") << (LEVY ? "LEVY " : "") << (MICHAELWICZYNSKI ? "MICHAELWICZYNSKI " : "")
        << "BALANCED: " << (BALANCED_TIME_GPU_FUNCTION ? "BALANCED_TIME_GPU_FUNCTION " : "") << (BALANCED_TIME_GPU ? "BALANCED_TIME_GPU " : "") << (BALANCED_TIME_OMP_FUNCTION ? "BALANCED_TIME_OMP_FUNCTION " : "") << (BALANCED_ADD_TIME_OMP_PROBABILITY ? "BALANCED_ADD_TIME_OMP_PROBABILITY " : "") << (BALANCED_ADD_TIME_OMP_ADD_AND_DECREASE ? "BALANCED_ADD_TIME_OMP_ADD_AND_DECREASE " : "")
        << "INITIAL_CPU_ANTS_RATIO: " << INITIAL_CPU_ANTS_RATIO << "; "
        << "MIN_BALANSED_ANT: " << MIN_BALANSED_ANT << "; "
        << "HASH_TABLE_SIZE: " << HASH_TABLE_SIZE << "; "
        << "ZERO_HASH_RESULT: " << ZERO_HASH_RESULT << "; "
        << "ZERO_HASH: " << ZERO_HASH << "; "
        << "MAX_PROBES: " << MAX_PROBES
        << std::endl;
    logFile << "PARAMETR_SIZE: " << PARAMETR_SIZE << "; "
        << "MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << "; "
        << "SET_PARAMETR_SIZE_ONE_X: " << SET_PARAMETR_SIZE_ONE_X << "; "
        << "ANT_SIZE: " << ANT_SIZE << "; "
        << "MAX_THREAD_CUDA: " << MAX_THREAD_CUDA << "; "
        << "NAME_FILE_GRAPH: " << NAME_FILE_GRAPH << "; "
        << "KOL_ITERATION: " << KOL_ITERATION << "; "
        << "KOL_PROGREV: " << KOL_PROGREV << "; "
        << "KOL_PROGON_STATISTICS: " << KOL_PROGON_STATISTICS << "; "
        << "PARAMETR_RO: " << PARAMETR_RO << "; "
        << "PARAMETR_Q: " << PARAMETR_Q << "; "
        << "PRINT_INFORMATION: " << PRINT_INFORMATION << "; "
        << "MAX_PARAMETR_VALUE_TO_MIN_OPT: " << MAX_PARAMETR_VALUE_TO_MIN_OPT << "; "
        << "OPTIMIZE: " << (OPTIMIZE_MIN_1 ? "OPTIMIZE_MIN_1 " : "") << (OPTIMIZE_MIN_2 ? "OPTIMIZE_MIN_2 " : "") << (OPTIMIZE_MAX ? "OPTIMIZE_MAX " : "") << "; "
        << "FUNCTION: " << (SHAFFERA ? "SHAFFERA " : "") << (CARROM_TABLE ? "CARROM_TABLE " : "") << (RASTRIGIN ? "RASTRIGIN " : "") << (ACKLEY ? "ACKLEY " : "") << (SPHERE ? "SPHERE " : "") << (GRIEWANK ? "GRIEWANK " : "") << (ZAKHAROV ? "ZAKHAROV " : "") << (SCHWEFEL ? "SCHWEFEL " : "") << (LEVY ? "LEVY " : "") << (MICHAELWICZYNSKI ? "MICHAELWICZYNSKI " : "")
        << "BALANCED: " << (BALANCED_TIME_GPU_FUNCTION ? "BALANCED_TIME_GPU_FUNCTION " : "") << (BALANCED_TIME_GPU ? "BALANCED_TIME_GPU " : "") << (BALANCED_TIME_OMP_FUNCTION ? "BALANCED_TIME_OMP_FUNCTION " : "") << (BALANCED_ADD_TIME_OMP_PROBABILITY ? "BALANCED_ADD_TIME_OMP_PROBABILITY " : "") << (BALANCED_ADD_TIME_OMP_ADD_AND_DECREASE ? "BALANCED_ADD_TIME_OMP_ADD_AND_DECREASE " : "")
        << "INITIAL_CPU_ANTS_RATIO: " << INITIAL_CPU_ANTS_RATIO << "; "
        << "MIN_BALANSED_ANT: " << MIN_BALANSED_ANT << "; "
        << "HASH_TABLE_SIZE: " << HASH_TABLE_SIZE << "; "
        << "ZERO_HASH_RESULT: " << ZERO_HASH_RESULT << "; "
        << "ZERO_HASH: " << ZERO_HASH << "; "
        << "MAX_PROBES: " << MAX_PROBES
        << std::endl;
    std::cout << "=== CUDA + OpenMP Hybrid Pipeline Test ===" << std::endl;

    if (GO_HYBRID_OMP) {
        int j = 0;
        while (j < KOL_PROGREV) {
            std::cout << "PROGREV " << j << " ";
            start_hybrid();
            j = j + 1;
        }

        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS) {
            std::cout << i << " ";
            start_hybrid();
            i = i + 1;
        }

        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time Hybrid OMP:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl;
    }
    if (GO_HYBRID_OMP_NON_HASH) {
        int j = 0;
        while (j < KOL_PROGREV) {
            std::cout << "PROGREV " << j << " ";
            start_hybrid_non_hash();
            j = j + 1;
        }

        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS) {
            std::cout << i << " ";
            start_hybrid_non_hash();
            i = i + 1;
        }

        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time Hybrid OMP non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl;
    }
    if (GO_HYBRID_BALANCED_OMP) {
        int j = 0;
        while (j < KOL_PROGREV) {
            std::cout << "PROGREV " << j << " ";
            start_balanced_hybrid();
            j = j + 1;
        }

        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS) {
            std::cout << i << " ";
            start_balanced_hybrid();
            i = i + 1;
        }

        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time Hybrid OMP parallel 1/2:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl;
    }
    if (GO_HYBRID_BALANCED_OMP_NON_HASH) {
        int j = 0;
        while (j < KOL_PROGREV) {
            std::cout << "PROGREV " << j << " ";
            start_balanced_hybrid_non_hash();
            j = j + 1;
        }

        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS) {
            std::cout << i << " ";
            start_balanced_hybrid_non_hash();
            i = i + 1;
        }

        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time Hybrid OMP parallel 1/2 non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl;
    }
    if (GO_HYBRID_BALANCED_DYNAMIC_OMP) {
        int j = 0;
        while (j < KOL_PROGREV) {
            std::cout << "PROGREV " << j << " ";
            start_balanced_hybrid_dynamic();
            j = j + 1;
        }

        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS) {
            std::cout << i << " ";
            start_balanced_hybrid_dynamic();
            i = i + 1;
        }

        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time Hybrid OMP parallel dynamic:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl;
    }
    if (GO_HYBRID_BALANCED_DYNAMIC_OMP_NON_HASH) {
        int j = 0;
        while (j < KOL_PROGREV) {
            std::cout << "PROGREV " << j << " ";
            start_balanced_hybrid_dynamic_non_hash();
            j = j + 1;
        }

        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS) {
            std::cout << i << " ";
            start_balanced_hybrid_dynamic_non_hash();
            i = i + 1;
        }

        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time Hybrid OMP parallel dynamic non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl;
    }
    if (GO_HYBRID_BALANCED_DYNAMIC_OMP_PARALLEL) {
        int j = 0;
        while (j < KOL_PROGREV) {
            std::cout << "PROGREV " << j << " ";
            start_balanced_hybrid_dynamic_parallel();
            j = j + 1;
        }

        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS) {
            std::cout << i << " ";
            start_balanced_hybrid_dynamic_parallel();
            i = i + 1;
        }

        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time Hybrid OMP parallel thread non blocked dynamic:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl;
    }
    if (GO_HYBRID_BALANCED_DYNAMIC_OMP_PARALLEL_NON_HASH) {
        int j = 0;
        while (j < KOL_PROGREV) {
            std::cout << "PROGREV " << j << " ";
            start_balanced_hybrid_dynamic_parallel_non_hash();
            j = j + 1;
        }

        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS) {
            std::cout << i << " ";
            start_balanced_hybrid_dynamic_parallel_non_hash();
            i = i + 1;
        }

        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time Hybrid OMP parallel thread non blocked dynamic:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl;
    }

    logFile.close();
    return 0;
}