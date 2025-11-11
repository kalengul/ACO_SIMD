#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <chrono>
#include <thread>
#include <future>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Константы и параметры
#define PARAMETR_SIZE 42   // Количество параметров 21*x (6*х)
#define PARAMETR_SIZE_ONE_X 21    // Количество параметров на оди x 21 (6)
#define MAX_VALUE_SIZE 4    // Максимальное количество значений у параметров 5 (100)
#define ZERO_HASH_RESULT -1.0
#define OPTIMIZE_MAX
#define NAME_FILE_GRAPH "Parametr_Graph/test42_4.txt"

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
#define TYPE_ACO 2
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

#define OPTIMIZE_MIN_1 0
#define OPTIMIZE_MIN_2 0



#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::ofstream logFile; // Глобальная переменная для лог-файла
std::ofstream outfile("statistics.txt"); // Глобальная переменная для файла статистики

// CUDA проверка ошибок
#define CUDA_CHECK(call) {\
    cudaError_t err = call;\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA error in " << #call << " at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl;\
        exit(1);\
    }\
}

// Структура для хэш-таблицы
struct HashEntry {
    int key[PARAMETR_SIZE];
    double value;
    bool occupied;
};

// Глобальные переменные для CPU состояния
static std::vector<double> current_pheromon;
static std::vector<double> current_kol_enter;
static std::vector<double> norm_matrix_probability;
static std::atomic<bool> data_ready{ false };
static std::atomic<bool> stop_requested{ false };

// Глобальные переменные для CUDA ресурсов
static double* parametr_value_dev = nullptr;
static double* pheromon_value_dev = nullptr;
static double* kol_enter_value_dev = nullptr;
static double* norm_matrix_probability_dev = nullptr;
static double* antOFdev = nullptr;
static int* ant_parametr_dev = nullptr;
static double* maxOf_dev = nullptr;
static double* minOf_dev = nullptr;
static int* kol_hash_fail_dev = nullptr;
static HashEntry* hashTable_dev = nullptr;
static cudaStream_t cuda_stream = nullptr;
static bool cuda_initialized = false;

// ==================== ПРОТОТИПЫ ФУНКЦИЙ ====================

// CUDA функции
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

// Функция загрузки матрицы
bool load_matrix(const std::string& filename, double* parametr_value, double* pheromon_value, double* kol_enter_value);

// ==================== CUDA ЯДРА ====================

__global__ void initializeHashTable(HashEntry* hashTable, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        hashTable[idx].occupied = false;
        for (int i = 0; i < PARAMETR_SIZE; i++) {
            hashTable[idx].key[i] = 0;
        }
        hashTable[idx].value = 0.0;
    }
}

__device__ double getCachedResultOptimized(HashEntry* hashTable, int* agent_node, int bx) {
    unsigned int hash = 0;
    for (int i = 0; i < PARAMETR_SIZE; i++) {
        hash = hash * 31 + agent_node[bx * PARAMETR_SIZE + i];
    }
    int index = hash % HASH_TABLE_SIZE;

    if (hashTable[index].occupied) {
        bool match = true;
        for (int i = 0; i < PARAMETR_SIZE; i++) {
            if (hashTable[index].key[i] != agent_node[bx * PARAMETR_SIZE + i]) {
                match = false;
                break;
            }
        }
        if (match) {
            return hashTable[index].value;
        }
    }
    return -1.0;
}

__device__ void saveToCacheOptimized(HashEntry* hashTable, int* agent_node, int bx, double result) {
    unsigned int hash = 0;
    for (int i = 0; i < PARAMETR_SIZE; i++) {
        hash = hash * 31 + agent_node[bx * PARAMETR_SIZE + i];
    }
    int index = hash % HASH_TABLE_SIZE;

    for (int i = 0; i < PARAMETR_SIZE; i++) {
        hashTable[index].key[i] = agent_node[bx * PARAMETR_SIZE + i];
    }
    hashTable[index].value = result;
    hashTable[index].occupied = true;
}

// Функция для цвычисления параметра х при  параметрическом графе
__device__ double go_x(double* parametr, int start_index, int kol_parametr) {
    double sum = 0.0;
    for (int i = 1; i < kol_parametr; ++i) {
        sum += parametr[start_index + i];
    }
    return parametr[start_index] * sum; // Умножаем на первый параметр в диапазоне
}

#if (SHAFFERA) 
// Функция для целевой функции Шаффера
__device__ double BenchShafferaFunction(double* parametr) {
    double r_squared = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        r_squared += x * x; // Сумма квадратов
    }
    double r = sqrt(r_squared);
    double sin_r = sin(r);
    return 1.0 / 2.0 - (sin_r * sin_r - 0.5) / (1.0 + 0.001 * r_squared);
}
#endif
#if (DELT4)
// Михаэлевич-Викинский
__device__ double BenchShafferaFunction(double* parametr) {
    double r_squared = 0.0;
    double sum_if = 0.0;
    double sum = 0.0;
    double second_sum = 0.0;
    double r_cos = 1.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
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
__device__ double BenchShafferaFunction(double* parametr) {
    double r_cos = 1.0;
    double r_squared = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        r_cos *= cos(x);
        r_squared += x * x;
    }
    double a = 1.0 - sqrt(r_squared) / M_PI;
    double OF = r_cos * exp(fabs(a)); // Используем fabs для абсолютного значения
    return OF * OF; // Возвращаем OF в квадрате
}
#endif
#if (RASTRIGIN)
// Растригин-функция
__device__ double BenchShafferaFunction(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum += x * x - 10 * cos(2 * M_PI * x) + 10;
    }
    return sum;
}
#endif
#if (ACKLEY)
// Акли-функция
__device__ double BenchShafferaFunction(double* parametr) {
    double first_sum = 0.0;
    double second_sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        first_sum += x * x;
        second_sum += cos(2 * M_PI * x);
    }
    double exp_term_1 = exp(-0.2 * sqrt(first_sum / num_variables));
    double exp_term_2 = exp(second_sum / num_variables);
    return -20 * exp_term_1 - exp_term_2 + M_E + 20;
}
#endif
#if (SPHERE)
// Сферическая функция
__device__ double BenchShafferaFunction(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum += x * x;
    }
    return sum;
}
#endif
#if (GRIEWANK)
// Гриванк-функция
__device__ double BenchShafferaFunction(double* parametr) {
    double sum = 0.0;
    double prod = 1.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum += x * x;
        prod *= cos(x / sqrt(i + 1));
    }
    return sum / 4000 - prod + 1;
}
#endif
#if (ZAKHAROV)
// Захаров-функция
__device__ double BenchShafferaFunction(double* parametr) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum1 += pow(x, 2);
        sum2 += 0.5 * i * x;
    }
    return sum1 + pow(sum2, 2) + pow(sum2, 4);
}
#endif
#if (SCHWEFEL)
// Швейфель-функция
__device__ double BenchShafferaFunction(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum -= x * sin(sqrt(abs(x)));
    }
    return sum;
}
#endif
#if (LEVY)
// Леви-функция
__device__ double BenchShafferaFunction(double* parametr) {
    double w_first = 1 + (go_x(parametr, 0, PARAMETR_SIZE_ONE_X) - 1) / 4;
    double w_last = 1 + (go_x(parametr, PARAMETR_SIZE - PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X) - 1) / 4;
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 1; i <= num_variables - 1; ++i) {
        double wi = 1 + (go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X) - 1) / 4;
        sum += pow(wi - 1, 2) * (1 + 10 * pow(sin(M_PI * wi), 2)) +
            pow(wi - wi * w_i_prev, 2) * (1 + pow(sin(2 * M_PI * wi), 2));
    }
    return pow(sin(M_PI * w_first), 2) + sum + pow(w_last - 1, 2) * (1 + pow(sin(2 * M_PI * w_last), 2));
}
#endif
#if (MICHAELWICZYNSKI)
// Михаэлевич-Викинский
__device__ double BenchShafferaFunction(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum -= sin(x) * pow(sin((i + 1) * x * x / M_PI), 20);
    }
    return sum;
}
#endif

__global__ void go_all_agent_only_4_optimized(double* parametr, double* norm_matrix_probability,
    int* agent_node, double* OF, HashEntry* hashTable,
    double* maxOf_dev, double* minOf_dev, int* kol_hash_fail,
    int iteration) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;
    if (bx < ANT_SIZE) {
        // Упрощенная инициализация генератора случайных чисел
        unsigned long long seed = (blockIdx.x * blockDim.x + threadIdx.x) + iteration * 12345;
        curandState state;
        curand_init(seed, 0, 0, &state);

        double agent[PARAMETR_SIZE] = { 0 };
        bool valid_solution = true;

        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            double randomValue = curand_uniform(&state);
            int k = 0;

            while (valid_solution && k < MAX_VALUE_SIZE &&
                randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                k++;
            }

            agent_node[bx * PARAMETR_SIZE + tx] = k;
            agent[tx] = parametr[tx * MAX_VALUE_SIZE + k];
            valid_solution = (k != MAX_VALUE_SIZE - 1);
        }

        // ИСПРАВЛЕНО: HashTable -> hashTable
        double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);

        if (cachedResult == -1.0) {
            OF[bx] = BenchShafferaFunction(agent);
            saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            OF[bx] = cachedResult;
            atomicAdd(kol_hash_fail, 1);
        }

        // Атомарные операции для double
        if (OF[bx] != ZERO_HASH_RESULT) {
            unsigned long long* address_as_ull = (unsigned long long*)maxOf_dev;
            unsigned long long old = *address_as_ull, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(fmax(OF[bx], __longlong_as_double(assumed))));
            } while (assumed != old);

            address_as_ull = (unsigned long long*)minOf_dev;
            old = *address_as_ull, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(fmin(OF[bx], __longlong_as_double(assumed))));
            } while (assumed != old);
        }
    }
}

// ==================== CUDA ФУНКЦИИ ====================

bool cuda_initialize(const double* parametr_value,
    const double* pheromon_value,
    const double* kol_enter_value) {
    if (cuda_initialized) {
        cuda_cleanup();
    }

    // Проверка доступности CUDA
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        std::cerr << "CUDA is not available: " << cudaGetErrorString(error_id) << std::endl;
        return false;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return false;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;

    const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;  // ИСПРАВЛЕНО: убрано size_t
    const int ant_matrix_size = PARAMETR_SIZE * ANT_SIZE;    // ИСПРАВЛЕНО: убрано size_t

    // Создание CUDA stream
    CUDA_CHECK(cudaStreamCreate(&cuda_stream));

    // Выделение памяти на устройстве
    CUDA_CHECK(cudaMalloc(&parametr_value_dev, matrix_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&pheromon_value_dev, matrix_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&kol_enter_value_dev, matrix_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&norm_matrix_probability_dev, matrix_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ant_parametr_dev, ant_matrix_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&antOFdev, ANT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&kol_hash_fail_dev, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));

    // Инициализация хэш-таблицы
    const int threadsPerBlock = MAX_THREAD_CUDA;
    const int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    initializeHashTable << <blocks_init_hash, threadsPerBlock, 0, cuda_stream >> > (hashTable_dev, HASH_TABLE_SIZE);

    // Копирование параметров (они не меняются)
    CUDA_CHECK(cudaMemcpyAsync(parametr_value_dev, parametr_value,
        matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(pheromon_value_dev, pheromon_value,
        matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(kol_enter_value_dev, kol_enter_value,
        matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_stream));

    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    cuda_initialized = true;

    std::cout << "CUDA initialized successfully" << std::endl;
    return true;
}

void cuda_run_iteration(const double* norm_matrix_probability,
    int* ant_parametr,
    double* antOF,
    double* global_minOf,
    double* global_maxOf,
    int* kol_hash_fail,
    int iteration) {
    if (!cuda_initialized) {
        std::cerr << "CUDA not initialized!" << std::endl;
        return;
    }

    const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;  // ИСПРАВЛЕНО: убрано size_t
    const int ant_matrix_size = PARAMETR_SIZE * ANT_SIZE;    // ИСПРАВЛЕНО: убрано size_t

    // Сброс статистики
    double min_init = 1e9, max_init = -1e9;
    CUDA_CHECK(cudaMemcpyAsync(maxOf_dev, &min_init, sizeof(double), cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(minOf_dev, &max_init, sizeof(double), cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CHECK(cudaMemsetAsync(kol_hash_fail_dev, 0, sizeof(int), cuda_stream));

    // Копирование нормализованной матрицы вероятностей
    CUDA_CHECK(cudaMemcpyAsync(norm_matrix_probability_dev, norm_matrix_probability,
        matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_stream));

    // Запуск ядра
    const int threadsPerBlock = MAX_THREAD_CUDA;
    const int numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    go_all_agent_only_4_optimized << <numBlocks, threadsPerBlock, 0, cuda_stream >> > (
        parametr_value_dev, norm_matrix_probability_dev, ant_parametr_dev,
        antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail_dev, iteration);

    // Копирование результатов обратно
    CUDA_CHECK(cudaMemcpyAsync(ant_parametr, ant_parametr_dev,
        ant_matrix_size * sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(antOF, antOFdev,
        ANT_SIZE * sizeof(double), cudaMemcpyDeviceToHost, cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(global_minOf, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost, cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(global_maxOf, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost, cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(kol_hash_fail, kol_hash_fail_dev, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));

    // Синхронизация stream
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
}

void cuda_cleanup() {
    if (cuda_initialized) {
        if (parametr_value_dev) cudaFree(parametr_value_dev);
        if (pheromon_value_dev) cudaFree(pheromon_value_dev);
        if (kol_enter_value_dev) cudaFree(kol_enter_value_dev);
        if (norm_matrix_probability_dev) cudaFree(norm_matrix_probability_dev);
        if (ant_parametr_dev) cudaFree(ant_parametr_dev);
        if (antOFdev) cudaFree(antOFdev);
        if (hashTable_dev) cudaFree(hashTable_dev);
        if (maxOf_dev) cudaFree(maxOf_dev);
        if (minOf_dev) cudaFree(minOf_dev);
        if (kol_hash_fail_dev) cudaFree(kol_hash_fail_dev);
        if (cuda_stream) cudaStreamDestroy(cuda_stream);

        parametr_value_dev = nullptr;
        pheromon_value_dev = nullptr;
        kol_enter_value_dev = nullptr;
        norm_matrix_probability_dev = nullptr;
        ant_parametr_dev = nullptr;
        antOFdev = nullptr;
        hashTable_dev = nullptr;
        maxOf_dev = nullptr;
        minOf_dev = nullptr;
        kol_hash_fail_dev = nullptr;
        cuda_stream = nullptr;

        cuda_initialized = false;
        std::cout << "CUDA resources cleaned up" << std::endl;
    }
}

// ==================== CPU ФУНКЦИИ (OpenMP) ====================

void omp_initialize(const double* initial_pheromon, const double* initial_kol_enter) {
    const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;  // ИСПРАВЛЕНО: убрано size_t

    current_pheromon.resize(matrix_size);
    current_kol_enter.resize(matrix_size);
    norm_matrix_probability.resize(matrix_size);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < matrix_size; i++) {  // ИСПРАВЛЕНО: int вместо size_t
        current_pheromon[i] = initial_pheromon[i];
        current_kol_enter[i] = initial_kol_enter[i];
    }

#ifdef _OPENMP
    omp_set_num_threads(omp_get_max_threads());
    std::cout << "OpenMP initialized with " << omp_get_max_threads() << " threads" << std::endl;
#else
    std::cout << "Running in sequential mode" << std::endl;
#endif

    data_ready.store(false);
    stop_requested.store(false);
}

void omp_calculate_probabilities() {
    if (stop_requested.load()) return;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumPheromon = 0.0;
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumPheromon += current_pheromon[MAX_VALUE_SIZE * tx + i];
        }

        double svertkaArray[MAX_VALUE_SIZE] = { 0 };
        double sumSvertka = 0.0;

        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            if (current_kol_enter[MAX_VALUE_SIZE * tx + i] != 0 && sumPheromon != 0) {
                double pheromonNorm = current_pheromon[MAX_VALUE_SIZE * tx + i] / sumPheromon;
                svertkaArray[i] = (1.0 / current_kol_enter[MAX_VALUE_SIZE * tx + i]) + pheromonNorm;
                sumSvertka += svertkaArray[i];
            }
        }

        if (sumSvertka > 0) {
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                cumulative += svertkaArray[i] / sumSvertka;
                norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = cumulative;
            }
        }
        else {
            // Равномерное распределение если все нули
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (i + 1.0) / MAX_VALUE_SIZE;
            }
        }
    }

    data_ready.store(true);
}

void omp_update_pheromones(const int* ant_parametr, const double* antOF) {
    if (stop_requested.load()) return;

    while (!data_ready.load() && !stop_requested.load()) {
        std::this_thread::yield();
    }

    if (stop_requested.load()) return;

    const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;  // ИСПРАВЛЕНО: убрано size_t

    // Испарение феромонов
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < matrix_size; i++) {  // ИСПРАВЛЕНО: int вместо size_t
        current_pheromon[i] *= PARAMETR_RO;
    }

    // Добавление нового феромона (убрано collapse для совместимости)
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        for (int i = 0; i < ANT_SIZE; i++) {
            int k = ant_parametr[i * PARAMETR_SIZE + tx];
            if (k >= 0 && k < MAX_VALUE_SIZE) {
#ifdef _OPENMP
#pragma omp atomic
#endif
                current_kol_enter[MAX_VALUE_SIZE * tx + k]++;

#ifdef OPTIMIZE_MAX
#ifdef _OPENMP
#pragma omp atomic
#endif
                current_pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * antOF[i];
#endif
            }
        }
    }

    data_ready.store(false);
}

const double* omp_get_norm_matrix_probability() {
    return norm_matrix_probability.data();
}

bool omp_is_data_ready() {
    return data_ready.load();
}

void omp_stop() {
    stop_requested.store(true);
}

void omp_cleanup() {
    omp_stop();
    current_pheromon.clear();
    current_kol_enter.clear();
    norm_matrix_probability.clear();
}

// ==================== ФУНКЦИЯ ЗАГРУЗКИ МАТРИЦЫ ====================

bool load_matrix(const std::string& filename, double* parametr_value, double* pheromon_value, double* kol_enter_value) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Cannot open file: " << filename << std::endl;

        // Создание тестовой матрицы если файл не существует
        std::cout << "Creating test matrix..." << std::endl;
        for (int i = 0; i < PARAMETR_SIZE; ++i) {
            for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                int k = MAX_VALUE_SIZE * i + j;
                parametr_value[k] = (i * MAX_VALUE_SIZE + j) * 0.1;
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
        return true;
    }

    for (int i = 0; i < PARAMETR_SIZE; ++i) {
        for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
            int k = MAX_VALUE_SIZE * i + j;
            if (!(infile >> parametr_value[k])) {
                std::cerr << "Error loading element [" << i << "][" << j << "]" << std::endl;
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
    std::cout << "Matrix successfully loaded from " << filename << std::endl;
    return true;
}

// ==================== MAIN ФУНКЦИЯ ====================

int main() {
    std::cout << __cplusplus << std::endl;
    logFile.open("log.txt");
    if (!logFile.is_open()) {
        std::cerr << "Ошибка открытия лог-файла!" << std::endl;
        return 1; // Возврат с ошибкой
    }
    //Создание векторов для статистики 
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "Don't have CUDA device." << std::endl;
        return 1;
    }
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        std::cout << "Device " << device << ": " << prop.name << std::endl;
        logFile << "Device " << device << ": " << prop.name << "; ";
        std::cout << "Max thread in blocks: " << prop.maxThreadsPerBlock << std::endl;
        logFile << "Max thread in blocks: " << prop.maxThreadsPerBlock << " ";
        std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;
        logFile << "Total Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB ";
        std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
        logFile << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes ";
        std::cout << "Constant Memory: " << prop.totalConstMem << " bytes" << std::endl;
        logFile << "Constant Memory: " << prop.totalConstMem << " bytes ";
        std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
        logFile << "Registers per Block: " << prop.regsPerBlock << " ";
        std::cout << "Warp Size: " << prop.warpSize << std::endl;
        logFile << "Warp Size: " << prop.warpSize << " ";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        logFile << "Compute Capability: " << prop.major << "." << prop.minor << " ";
        std::cout << "Can Map Host Memory: " << (prop.canMapHostMemory ? "Yes" : "No") << std::endl;
        logFile << "Can Map Host Memory: " << (prop.canMapHostMemory ? "Yes" : "No") << " ";
        std::cout << "Clock Rate: " << prop.clockRate / 1000.0f << " MHz" << std::endl;
        logFile << "Clock Rate: " << prop.clockRate / 1000.0f << " MHz ";
        std::cout << "L2 Cache Size: " << (prop.l2CacheSize == 0 ? 0 : prop.l2CacheSize) << " bytes" << std::endl;
        logFile << "L2 Cache Size: " << (prop.l2CacheSize == 0 ? 0 : prop.l2CacheSize) << " bytes ";
        std::cout << "Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
        logFile << "Multiprocessor Count: " << prop.multiProcessorCount << "; ";
        std::cout << "Max thread in blocks by axis: ("
            << prop.maxThreadsDim[0] << ", "
            << prop.maxThreadsDim[1] << ", "
            << prop.maxThreadsDim[2] << ")" << std::endl;
        logFile << "Max thread in blocks by axis: ("
            << prop.maxThreadsDim[0] << ", "
            << prop.maxThreadsDim[1] << ", "
            << prop.maxThreadsDim[2] << "); ";
        std::cout << "Max blocks by axis: ("
            << prop.maxGridSize[0] << ", "
            << prop.maxGridSize[1] << ", "
            << prop.maxGridSize[2] << ")" << std::endl;
        logFile << "Max blocks by axis: ("
            << prop.maxGridSize[0] << ", "
            << prop.maxGridSize[1] << ", "
            << prop.maxGridSize[2] << ");";
    }

    std::cout << "PARAMETR_SIZE: " << PARAMETR_SIZE << "; "
        << "PARAMETR_SIZE_ONE_X: " << PARAMETR_SIZE_ONE_X << "; "
        << "MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << "; "
        << "ANT_SIZE: " << ANT_SIZE << "; "
        << "NAME_FILE_GRAPH: " << NAME_FILE_GRAPH << "; "
        << "KOL_ITERATION: " << KOL_ITERATION << "; "
        << "KOL_PROGON_STATISTICS: " << KOL_PROGON_STATISTICS << "; "
        << "PARAMETR_Q: " << PARAMETR_Q << "; "
        << "PARAMETR_RO: " << PARAMETR_RO << "; "
        << "TYPE_ACO: " << TYPE_ACO << "; "
        << "ACOCCyN_KOL_ITERATION: " << ACOCCyN_KOL_ITERATION << ", " << "MAX_PARAMETR_VALUE_TO_MIN_OPT: " << MAX_PARAMETR_VALUE_TO_MIN_OPT << ", " << "HASH_TABLE_SIZE: " << HASH_TABLE_SIZE << ", " << "MAX_PROBES: " << MAX_PROBES << ", " << "MAX_THREAD_CUDA: " << MAX_THREAD_CUDA << ", " << "CPU_RANDOM: " << CPU_RANDOM << ", " << "KOL_THREAD_CPU_ANT: " << KOL_THREAD_CPU_ANT << ", " << "CONST_RANDOM: " << CONST_RANDOM << ", " << "MAX_CONST: " << MAX_CONST << ", "
        << "OPTIMIZE: " << (OPTIMIZE_MIN_1 ? "OPTIMIZE_MIN_1 " : "") << (OPTIMIZE_MIN_2 ? "OPTIMIZE_MIN_2 " : "") << (OPTIMIZE_MAX ? "OPTIMIZE_MAX " : "") << "; "
        << "FUNCTION: " << (SHAFFERA ? "SHAFFERA " : "") << (CARROM_TABLE ? "CARROM_TABLE " : "") << (RASTRIGIN ? "RASTRIGIN " : "") << (ACKLEY ? "ACKLEY " : "") << (SPHERE ? "SPHERE " : "") << (GRIEWANK ? "GRIEWANK " : "") << (ZAKHAROV ? "ZAKHAROV " : "") << (SCHWEFEL ? "SCHWEFEL " : "") << (LEVY ? "LEVY " : "") << (MICHAELWICZYNSKI ? "MICHAELWICZYNSKI " : "")
        << "GO_ALG_MINMAX: " << GO_ALG_MINMAX << "(" << PAR_MIN_ALG_MINMAX << ", " << PAR_MAX_ALG_MINMAX << ")" << "; "
        << std::endl;
    logFile << "PARAMETR_SIZE: " << PARAMETR_SIZE << "; "
        << "PARAMETR_SIZE_ONE_X: " << PARAMETR_SIZE_ONE_X << "; "
        << "MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << "; "
        << "NAME_FILE_GRAPH: " << NAME_FILE_GRAPH << "; "
        << "ANT_SIZE: " << ANT_SIZE << "; "
        << "KOL_ITERATION: " << KOL_ITERATION << "; "
        << "KOL_PROGON_STATISTICS: " << KOL_PROGON_STATISTICS << "; "
        << "PARAMETR_Q: " << PARAMETR_Q << "; "
        << "PARAMETR_RO: " << PARAMETR_RO << "; "
        << "TYPE_ACO: " << TYPE_ACO << "; "
        << "ACOCCyN_KOL_ITERATION: " << ACOCCyN_KOL_ITERATION << ", " << "MAX_PARAMETR_VALUE_TO_MIN_OPT: " << MAX_PARAMETR_VALUE_TO_MIN_OPT << ", " << "HASH_TABLE_SIZE: " << HASH_TABLE_SIZE << ", " << "MAX_PROBES: " << MAX_PROBES << ", " << "MAX_THREAD_CUDA: " << MAX_THREAD_CUDA << ", " << "CPU_RANDOM: " << CPU_RANDOM << ", " << "KOL_THREAD_CPU_ANT: " << KOL_THREAD_CPU_ANT << ", " << "CONST_RANDOM: " << CONST_RANDOM << ", " << "MAX_CONST: " << MAX_CONST << ", "
        << "OPTIMIZE: " << (OPTIMIZE_MIN_1 ? "OPTIMIZE_MIN_1 " : "") << (OPTIMIZE_MIN_2 ? "OPTIMIZE_MIN_2 " : "") << (OPTIMIZE_MAX ? "OPTIMIZE_MAX " : "") << "; "
        << "FUNCTION: " << (SHAFFERA ? "SHAFFERA " : "") << (CARROM_TABLE ? "CARROM_TABLE " : "") << (RASTRIGIN ? "RASTRIGIN " : "") << (ACKLEY ? "ACKLEY " : "") << (SPHERE ? "SPHERE " : "") << (GRIEWANK ? "GRIEWANK " : "") << (ZAKHAROV ? "ZAKHAROV " : "") << (SCHWEFEL ? "SCHWEFEL " : "") << (LEVY ? "LEVY " : "") << (MICHAELWICZYNSKI ? "MICHAELWICZYNSKI " : "")
        << "GO_ALG_MINMAX: " << GO_ALG_MINMAX << "(" << PAR_MIN_ALG_MINMAX << ", " << PAR_MAX_ALG_MINMAX << ")" << "; "
        << std::endl;
    std::cout << "Hybrid CUDA + OpenMP Ant Colony Optimization" << std::endl;
    std::cout << "Parameters: " << PARAMETR_SIZE << " parameters, "
        << MAX_VALUE_SIZE << " values, " << ANT_SIZE << " ants, "
        << KOL_ITERATION << " iterations" << std::endl;

    // Загрузка данных
    std::vector<double> parametr_value(MAX_VALUE_SIZE * PARAMETR_SIZE);
    std::vector<double> pheromon_value(MAX_VALUE_SIZE * PARAMETR_SIZE);
    std::vector<double> kol_enter_value(MAX_VALUE_SIZE * PARAMETR_SIZE);

    if (!load_matrix(NAME_FILE_GRAPH, parametr_value.data(), pheromon_value.data(), kol_enter_value.data())) {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Инициализация CUDA и OpenMP
    if (!cuda_initialize(parametr_value.data(), pheromon_value.data(), kol_enter_value.data())) {
        std::cerr << "CUDA initialization failed! Running CPU-only version..." << std::endl;
    }

    omp_initialize(pheromon_value.data(), kol_enter_value.data());

    // Буферы для данных
    std::vector<int> ant_parametr(PARAMETR_SIZE * ANT_SIZE);
    std::vector<double> antOF(ANT_SIZE);

    // Переменные для результатов
    double global_minOf = 0, global_maxOf = 0;
    int kol_hash_fail = 0;

    auto total_start = std::chrono::high_resolution_clock::now();

    // Основной цикл оптимизации
    for (int iteration = 0; iteration < KOL_ITERATION; iteration++) {
        auto iter_start = std::chrono::high_resolution_clock::now();

        // 1. Расчет вероятностей на CPU (OpenMP)
        omp_calculate_probabilities();

        // 2. Получаем нормализованную матрицу вероятностей
        const double* norm_matrix = omp_get_norm_matrix_probability();

        // 3. Запуск муравьев на GPU (CUDA)
        cuda_run_iteration(norm_matrix, ant_parametr.data(), antOF.data(),
            &global_minOf, &global_maxOf, &kol_hash_fail, iteration);

        // 4. Обновление феромонов на CPU (OpenMP)
        omp_update_pheromones(ant_parametr.data(), antOF.data());

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