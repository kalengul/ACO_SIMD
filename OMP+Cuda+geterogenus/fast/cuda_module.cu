#include "cuda_module.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <thread>

#define MAX_VALUE_SIZE 4
#define PARAMETR_SIZE 42
#define PARAMETR_SIZE_ONE_X 21    // Количество параметров на оди x 21 (6)
#define ANT_SIZE 500

#define MAX_THREAD_CUDA 256
#define HASH_TABLE_SIZE 10000000 // Hash table size (10 million entries)
#define ZERO_HASH_RESULT -1.0

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

// Структура для данных CUDA
struct CudaData {
    double* parametr_value_dev;
    double* pheromon_value_dev;
    double* kol_enter_value_dev;
    double* norm_matrix_probability_dev;
    double* antOFdev;
    int* ant_parametr_dev;
    double* maxOf_dev;
    double* minOf_dev;
    int* kol_hash_fail_dev;
    HashEntry* hashTable_dev;
    cudaStream_t stream;
    bool cuda_initialized;
    bool initialized;
};

static CudaData cuda_data = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, false, false };

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
    double x = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
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

// Добавьте эту функцию в CUDA ядро для отладки
__device__ void debug_print_ant_parameters(double* parametr, int* agent_node, double* OF, int iteration) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;

    // Выводим параметры только для первых 5 муравьев и первой итерации
    if (bx < 5 && iteration == 0) {
        printf("[CUDA Debug] Ant %d: [", bx);
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            int k = agent_node[bx * PARAMETR_SIZE + tx];
            double value = parametr[tx * MAX_VALUE_SIZE + k];
            printf("%.3f", value);
            if (tx < PARAMETR_SIZE - 1) printf(", ");
        }
        printf("] -> OF = %.6f\n", OF[bx]);
    }
}

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
        //OF[bx] = -bx;
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
    /*
    // Добавьте в конец ядра:
    if (bx == 0 && iteration == 0) {
        debug_print_ant_parameters(parametr, agent_node, OF, iteration);
    }
    */
}

// Простое CUDA ядро
__global__ void simple_ant_kernel(double* parametr, double* probabilities, int* ant_params, double* results, int iteration) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < ANT_SIZE) {
        unsigned long long seed = (blockIdx.x * blockDim.x + threadIdx.x) + iteration * 12345;
        curandState state;
        curand_init(seed, 0, 0, &state);

        double sum = 0.0;
        for (int i = 0; i < PARAMETR_SIZE; i++) {
            int param_idx = ant_params[idx * PARAMETR_SIZE + i];
            double value = parametr[i * MAX_VALUE_SIZE + param_idx];
            sum += value * value;
        }

        double noise = curand_uniform(&state) * 0.1;
        results[idx] = sum + noise;
    }
}

// Функции управления CUDA
void cuda_cleanup() {
    if (cuda_data.initialized) {
        if (cuda_data.parametr_value_dev) cudaFree(cuda_data.parametr_value_dev);
        if (cuda_data.pheromon_value_dev) cudaFree(cuda_data.pheromon_value_dev);
        if (cuda_data.kol_enter_value_dev) cudaFree(cuda_data.kol_enter_value_dev);
        if (cuda_data.norm_matrix_probability_dev) cudaFree(cuda_data.norm_matrix_probability_dev);
        if (cuda_data.ant_parametr_dev) cudaFree(cuda_data.ant_parametr_dev);
        if (cuda_data.antOFdev) cudaFree(cuda_data.antOFdev);
        if (cuda_data.hashTable_dev) cudaFree(cuda_data.hashTable_dev);
        if (cuda_data.maxOf_dev) cudaFree(cuda_data.maxOf_dev);
        if (cuda_data.minOf_dev) cudaFree(cuda_data.minOf_dev);
        if (cuda_data.kol_hash_fail_dev) cudaFree(cuda_data.kol_hash_fail_dev);
        if (cuda_data.stream) cudaStreamDestroy(cuda_data.stream);

        cuda_data.parametr_value_dev = nullptr;
        cuda_data.pheromon_value_dev = nullptr;
        cuda_data.kol_enter_value_dev = nullptr;
        cuda_data.norm_matrix_probability_dev = nullptr;
        cuda_data.ant_parametr_dev = nullptr;
        cuda_data.antOFdev = nullptr;
        cuda_data.hashTable_dev = nullptr;
        cuda_data.maxOf_dev = nullptr;
        cuda_data.minOf_dev = nullptr;
        cuda_data.kol_hash_fail_dev = nullptr;
        cuda_data.stream = nullptr;

        cuda_data.initialized = false;

    }
}

bool cuda_initialize(const double* parametr_value, const double* pheromon_value, const double* kol_enter_value) {
    if (cuda_data.initialized) {
        cuda_cleanup();
    }

    size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
    size_t ant_matrix_size = PARAMETR_SIZE * ANT_SIZE;
    const int threadsPerBlock = MAX_THREAD_CUDA;
    const int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "[CUDA] No CUDA devices found!" << std::endl;
        return false;
    }
    err = cudaStreamCreate(&cuda_data.stream);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Failed to create stream: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    // Выделение памяти на устройстве
    err = cudaMalloc(&cuda_data.parametr_value_dev, matrix_size * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&cuda_data.pheromon_value_dev, matrix_size * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&cuda_data.kol_enter_value_dev, matrix_size * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&cuda_data.norm_matrix_probability_dev, matrix_size * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&cuda_data.ant_parametr_dev, ant_matrix_size * sizeof(int));
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&cuda_data.antOFdev, ANT_SIZE * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&cuda_data.maxOf_dev, sizeof(double));
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&cuda_data.minOf_dev, sizeof(double));
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&cuda_data.kol_hash_fail_dev, sizeof(int));
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMalloc(&cuda_data.hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry));
    if (err != cudaSuccess) goto cuda_error;
    
    // Инициализация хэш-таблицы
    initializeHashTable<<<blocks_init_hash, threadsPerBlock, 0, cuda_data.stream>>>(cuda_data.hashTable_dev, HASH_TABLE_SIZE);
    
    // Копирование параметров
    err = cudaMemcpyAsync(cuda_data.parametr_value_dev, parametr_value, matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_data.stream);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMemcpyAsync(cuda_data.pheromon_value_dev, pheromon_value, matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_data.stream);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMemcpyAsync(cuda_data.kol_enter_value_dev, kol_enter_value, matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_data.stream);
    if (err != cudaSuccess) goto cuda_error;
    cudaStreamSynchronize(cuda_data.stream);
    cuda_data.initialized = true;

    return true;

cuda_error:
    std::cerr << "[CUDA] Error: " << cudaGetErrorString(err) << std::endl;
    cuda_cleanup();
    return false;
}

void cuda_run_iteration(const double* norm_matrix_probability, int* ant_parametr, double* antOF, double* global_minOf, double* global_maxOf, int* kol_hash_fail, double* time_all, double* time_function, int iteration, void (*completion_callback)(double*, int, int)) {
    if (!cuda_data.initialized) {
        std::cerr << "[CUDA] CUDA not initialized!" << std::endl;
        return;
    }
    size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
    size_t ant_matrix_size = PARAMETR_SIZE * ANT_SIZE;
    
    // Сброс статистики
    double min_init = 1e9, max_init = -1e9;
    cudaMemcpyAsync(cuda_data.maxOf_dev, &max_init, sizeof(double), cudaMemcpyHostToDevice, cuda_data.stream);
    cudaMemcpyAsync(cuda_data.minOf_dev, &min_init, sizeof(double), cudaMemcpyHostToDevice, cuda_data.stream);
    cudaMemsetAsync(cuda_data.kol_hash_fail_dev, 0, sizeof(int), cuda_data.stream);
    
    // Копирование нормализованной матрицы вероятностей
    cudaMemcpyAsync(cuda_data.norm_matrix_probability_dev, norm_matrix_probability, matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_data.stream);
    
    // Запуск ядра
    const int threadsPerBlock = MAX_THREAD_CUDA;
    const int numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    go_all_agent_only_4_optimized<<<numBlocks, threadsPerBlock, 0, cuda_data.stream >>>(cuda_data.parametr_value_dev, cuda_data.norm_matrix_probability_dev, cuda_data.ant_parametr_dev, cuda_data.antOFdev, cuda_data.hashTable_dev, cuda_data.maxOf_dev, cuda_data.minOf_dev, cuda_data.kol_hash_fail_dev, iteration);
   
    // Копирование результатов обратно
    cudaMemcpyAsync(ant_parametr, cuda_data.ant_parametr_dev, ant_matrix_size * sizeof(int), cudaMemcpyDeviceToHost, cuda_data.stream);
    cudaMemcpyAsync(antOF, cuda_data.antOFdev, ANT_SIZE * sizeof(double), cudaMemcpyDeviceToHost, cuda_data.stream);
    cudaMemcpyAsync(global_minOf, cuda_data.minOf_dev, sizeof(double), cudaMemcpyDeviceToHost, cuda_data.stream);
    cudaMemcpyAsync(global_maxOf, cuda_data.maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost, cuda_data.stream);
    cudaMemcpyAsync(kol_hash_fail, cuda_data.kol_hash_fail_dev, sizeof(int), cudaMemcpyDeviceToHost, cuda_data.stream);
    
    // Синхронизация stream
    cudaEvent_t completion_event;
    cudaEventCreate(&completion_event);
    cudaEventRecord(completion_event, cuda_data.stream);

    std::thread([antOF, completion_event, completion_callback, iteration]() {
        cudaEventSynchronize(completion_event);
        cudaEventDestroy(completion_event);

        completion_callback(antOF, ANT_SIZE, iteration);
        }).detach();

}
/*
// Реализации функций
bool cuda_initialize(const double* parametr_value) {
    if (cuda_data.initialized) {
        cuda_cleanup();
    }

    std::cout << "[CUDA] Initializing..." << std::endl;

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "[CUDA] No CUDA devices found!" << std::endl;
        return false;
    }

    err = cudaStreamCreate(&cuda_data.stream);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Failed to create stream: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
    
    err = cudaMalloc(&cuda_data.parametr_value_dev, matrix_size * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;
    
    err = cudaMalloc(&cuda_data.norm_matrix_probability_dev, matrix_size * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;
    
    err = cudaMalloc(&cuda_data.ant_parametr_dev, PARAMETR_SIZE * ANT_SIZE * sizeof(int));
    if (err != cudaSuccess) goto cuda_error;
    
    err = cudaMalloc(&cuda_data.antOFdev, ANT_SIZE * sizeof(double));
    if (err != cudaSuccess) goto cuda_error;

    err = cudaMemcpyAsync(cuda_data.parametr_value_dev, parametr_value, 
                         matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_data.stream);
    if (err != cudaSuccess) goto cuda_error;

    cudaStreamSynchronize(cuda_data.stream);
    cuda_data.initialized = true;

    std::cout << "[CUDA] Initialized successfully!" << std::endl;
    return true;

cuda_error:
    std::cerr << "[CUDA] Error: " << cudaGetErrorString(err) << std::endl;
    cuda_cleanup();
    return false;
}
*/
void cuda_run_async(const double* norm_matrix_probability,
                   const int* ant_parametr,
                   double* antOF,
                   int iteration,
                   void (*completion_callback)(double*, int, int)) {
    if (!cuda_data.initialized) {
        std::cerr << "[CUDA] Not initialized!" << std::endl;
        return;
    }

    size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
    cudaMemcpyAsync(cuda_data.norm_matrix_probability_dev, norm_matrix_probability,
                   matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_data.stream);

    cudaMemcpyAsync(cuda_data.ant_parametr_dev, ant_parametr,
                   PARAMETR_SIZE * ANT_SIZE * sizeof(int), cudaMemcpyHostToDevice, cuda_data.stream);

    int threadsPerBlock = MAX_THREAD_CUDA;
    int numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    simple_ant_kernel<<<numBlocks, threadsPerBlock, 0, cuda_data.stream>>>(
        cuda_data.parametr_value_dev,
        cuda_data.norm_matrix_probability_dev,
        cuda_data.ant_parametr_dev,
        cuda_data.antOFdev,
        iteration
    );

    cudaMemcpyAsync(antOF, cuda_data.antOFdev, 
                   ANT_SIZE * sizeof(double), cudaMemcpyDeviceToHost, cuda_data.stream);

    cudaEvent_t completion_event;
    cudaEventCreate(&completion_event);
    cudaEventRecord(completion_event, cuda_data.stream);

    std::thread([antOF, completion_event, completion_callback, iteration]() {
        cudaEventSynchronize(completion_event);
        cudaEventDestroy(completion_event);
        
        completion_callback(antOF, ANT_SIZE, iteration);
    }).detach();
}

void cuda_synchronize() {
    if (cuda_data.initialized && cuda_data.stream) {
        cudaStreamSynchronize(cuda_data.stream);
    }
}

const char* cuda_get_version() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        return "Don't have CUDA device.";
    }
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        std::cout << "Device " << device << ": " << prop.name << std::endl;
        //logFile << "Device " << device << ": " << prop.name << "; ";
        std::cout << "Max thread in blocks: " << prop.maxThreadsPerBlock << std::endl;
        //logFile << "Max thread in blocks: " << prop.maxThreadsPerBlock << " ";
        std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;
        //logFile << "Total Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB ";
        std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
        //logFile << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes ";
        std::cout << "Constant Memory: " << prop.totalConstMem << " bytes" << std::endl;
        //logFile << "Constant Memory: " << prop.totalConstMem << " bytes ";
        std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
        //logFile << "Registers per Block: " << prop.regsPerBlock << " ";
        std::cout << "Warp Size: " << prop.warpSize << std::endl;
        //logFile << "Warp Size: " << prop.warpSize << " ";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        //logFile << "Compute Capability: " << prop.major << "." << prop.minor << " ";
        std::cout << "Can Map Host Memory: " << (prop.canMapHostMemory ? "Yes" : "No") << std::endl;
        //logFile << "Can Map Host Memory: " << (prop.canMapHostMemory ? "Yes" : "No") << " ";
        std::cout << "Clock Rate: " << prop.clockRate / 1000.0f << " MHz" << std::endl;
        //logFile << "Clock Rate: " << prop.clockRate / 1000.0f << " MHz ";
        std::cout << "L2 Cache Size: " << (prop.l2CacheSize == 0 ? 0 : prop.l2CacheSize) << " bytes" << std::endl;
        //logFile << "L2 Cache Size: " << (prop.l2CacheSize == 0 ? 0 : prop.l2CacheSize) << " bytes ";
        std::cout << "Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
        //logFile << "Multiprocessor Count: " << prop.multiProcessorCount << "; ";
        std::cout << "Max thread in blocks by axis: ("
            << prop.maxThreadsDim[0] << ", "
            << prop.maxThreadsDim[1] << ", "
            << prop.maxThreadsDim[2] << ")" << std::endl;
        /*
        logFile << "Max thread in blocks by axis: ("
            << prop.maxThreadsDim[0] << ", "
            << prop.maxThreadsDim[1] << ", "
            << prop.maxThreadsDim[2] << "); ";
        */
        std::cout << "Max blocks by axis: ("
            << prop.maxGridSize[0] << ", "
            << prop.maxGridSize[1] << ", "
            << prop.maxGridSize[2] << ")" << std::endl;
        /*
        logFile << "Max blocks by axis: ("
            << prop.maxGridSize[0] << ", "
            << prop.maxGridSize[1] << ", "
            << prop.maxGridSize[2] << ");";
        */
    }
    return "CUDA Module v1.0 (Clang/Windows)";
}