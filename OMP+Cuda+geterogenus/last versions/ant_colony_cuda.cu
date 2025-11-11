#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include "ant_colony_common.h"

#define CUDA_CHECK(call) {cudaError_t err = call; if (err != cudaSuccess) {std::cerr << "CUDA error in " << #call << " at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; exit(1); }}


// Глобальные переменные для CUDA ресурсов
static double *parametr_value_dev = nullptr;
static double *pheromon_value_dev = nullptr;
static double *kol_enter_value_dev = nullptr;
static double *norm_matrix_probability_dev = nullptr;
static double *antOFdev = nullptr;
static int *ant_parametr_dev = nullptr;
static double *maxOf_dev = nullptr;
static double *minOf_dev = nullptr;
static int *kol_hash_fail_dev = nullptr;
static HashEntry* hashTable_dev = nullptr;
static cudaStream_t cuda_stream = nullptr;
static bool cuda_initialized = false;

// CUDA ядра
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

// Функция для вычисления параметра х при  параметрическом графе
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

__global__ void go_all_agent_only_4_optimized(double* parametr, double* norm_matrix_probability, int* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail, int iteration) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;
    if (bx < ANT_SIZE) {
        curandState state;
        curand_init(clock64() + bx + iteration * 1000, 0, 0, &state);
        
        double agent[PARAMETR_SIZE] = {0};
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
        
        double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
        
        if (cachedResult == -1.0) {
            OF[bx] = BenchShafferaFunction(agent);
            saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
        } else {
            OF[bx] = cachedResult;
            atomicAdd(kol_hash_fail, 1);
        }
        
        if (OF[bx] != ZERO_HASH_RESULT) {
            atomicMax((int64_t*)maxOf_dev, __double_as_longlong(OF[bx]));
            atomicMin((int64_t*)minOf_dev, __double_as_longlong(OF[bx]));
        }
    }
}

// Функции управления CUDA
bool cuda_initialize(const double* parametr_value, const double* pheromon_value, const double* kol_enter_value) {
    if (cuda_initialized) {
        cuda_cleanup();
    }
    size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
    size_t ant_matrix_size = PARAMETR_SIZE * ANT_SIZE;
    
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
    initializeHashTable<<<blocks_init_hash, threadsPerBlock, 0, cuda_stream>>>(hashTable_dev, HASH_TABLE_SIZE);
    
    // Копирование параметров
    CUDA_CHECK(cudaMemcpyAsync(parametr_value_dev, parametr_value, matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(pheromon_value_dev, pheromon_value, matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(kol_enter_value_dev, kol_enter_value, matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    cuda_initialized = true;
    
    std::cout << "CUDA initialized successfully" << std::endl;
    return true;
}

void cuda_run_iteration(const double* norm_matrix_probability, int* ant_parametr, double* antOF, double* global_minOf, double* global_maxOf, int* kol_hash_fail, int iteration) {
    if (!cuda_initialized) {
        std::cerr << "CUDA not initialized!" << std::endl;
        return;
    }
    
    size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
    size_t ant_matrix_size = PARAMETR_SIZE * ANT_SIZE;
    
    // Сброс статистики
    double min_init = 1e9, max_init = -1e9;
    CUDA_CHECK(cudaMemcpyAsync(maxOf_dev, &min_init, sizeof(double), cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(minOf_dev, &max_init, sizeof(double), cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CHECK(cudaMemsetAsync(kol_hash_fail_dev, 0, sizeof(int), cuda_stream));
    
    // Копирование нормализованной матрицы вероятностей
    CUDA_CHECK(cudaMemcpyAsync(norm_matrix_probability_dev, norm_matrix_probability, matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_stream));
    
    // Запуск ядра
    const int threadsPerBlock = MAX_THREAD_CUDA;
    const int numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    
    go_all_agent_only_4_optimized<<<numBlocks, threadsPerBlock, 0, cuda_stream>>>(parametr_value_dev, norm_matrix_probability_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail_dev, iteration);
    
    // Копирование результатов обратно
    CUDA_CHECK(cudaMemcpyAsync(ant_parametr, ant_parametr_dev, ant_matrix_size * sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(antOF, antOFdev, ANT_SIZE * sizeof(double), cudaMemcpyDeviceToHost, cuda_stream));
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