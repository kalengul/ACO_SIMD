#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "parametrs.h"

#define CUDA_CHECK(call) do {cudaError_t err = call; if (err != cudaSuccess) {std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl;  exit(EXIT_FAILURE); } } while(0)

// Глобальный лог-файл
std::ofstream logFile("log_v100.txt");

struct alignas(32) HashEntry {
    unsigned long long key;
    double value;
    unsigned long long padding;
};

struct PerformanceMetrics {
    float total_time_ms, kernel_time_ms, memory_time_ms;
    float computeProbabilities_time_ms, antColonyOptimization_time_ms, updatePheromones_time_ms;
    float occupancy, memory_throughput_gbs;
    double min_fitness, max_fitness;
    int hash_hits, hash_misses;
    int iterations_completed;
    float warp_execution_efficiency;
    float l2_cache_hit_rate;
};

// ==================== VOLTA V100 SPECIFIC OPTIMIZATIONS ====================

// Оптимизированные примитивы для V100 (только для глобальной памяти!)
__device__ __forceinline__ double ldg_global(const double* ptr) {
    return __ldg(ptr);  // Только для глобальной памяти!
}

__device__ __forceinline__ int ldg_global(const int* ptr) {
    return __ldg(ptr);  // Только для глобальной памяти!
}

// Простое чтение для shared memory
__device__ __forceinline__ double load_shared(const double* ptr) {
    return *ptr;  // Простое чтение для shared memory
}

__device__ __forceinline__ int load_shared(const int* ptr) {
    return *ptr;  // Простое чтение для shared memory
}

__device__ __forceinline__ double4 safe_load_double4(const double* ptr) {
    double4 result;
    result.x = ptr[0];
    result.y = ptr[1];
    result.z = ptr[2];
    result.w = ptr[3];
    return result;
}

// Быстрая функция go_x с оптимизациями для V100
__device__ __forceinline__ double go_x_v100(const double* __restrict__ parametr, int start_index) {
    double sum = 0.0;

    // Развертывание цикла для лучшей ILP на V100
#pragma unroll 8
    for (int i = 1; i < SET_PARAMETR_SIZE_ONE_X; ++i) {
        sum += ldg_global(&parametr[start_index + i]);  // Глобальная память
    }
    return ldg_global(&parametr[start_index]) * sum;  // Глобальная память
}

// ==================== TEST FUNCTIONS FOR V100 ====================

#if (SHAFFERA)
__device__ double BenchShafferaFunction_V100(const double* __restrict__ parametr) {
    double r_squared = 0.0;
    const int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;

    // Максимальное развертывание для V100
#pragma unroll 8
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_v100(parametr, i * SET_PARAMETR_SIZE_ONE_X);
        r_squared += x * x;
    }

    // Используем быстрые математические функции V100
    double r = sqrt(r_squared);
    double sin_r = sin(r);
    return 0.5 - (sin_r * sin_r - 0.5) / (1.0 + 0.001 * r_squared);
}
#endif
#if (SPHERE)
__device__ double BenchShafferaFunction_V100(const double* __restrict__ parametr) {
    double sum = 0.0;
    const int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;

#pragma unroll 8
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_v100(parametr, i * SET_PARAMETR_SIZE_ONE_X);
        sum += x * x;
    }
    return sum;
}
#endif
#if (RASTRIGIN)
__device__ double BenchShafferaFunction_V100(const double* __restrict__ parametr) {
    double sum = 0.0;
    const int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;

#pragma unroll 8
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_v100(parametr, i * SET_PARAMETR_SIZE_ONE_X);
        sum += x * x - 10.0 * cos(2.0 * M_PI * x) + 10.0;
    }
    return sum;
}
#endif
#if (ACKLEY)
__device__ double BenchShafferaFunction_V100(const double* __restrict__ parametr) {
    double first_sum = 0.0;
    double second_sum = 0.0;
    const int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;

#pragma unroll 8
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_v100(parametr, i * SET_PARAMETR_SIZE_ONE_X);
        first_sum += x * x;
        second_sum += cos(2.0 * M_PI * x);
    }

    double exp_term_1 = exp(-0.2 * sqrt(first_sum / num_variables));
    double exp_term_2 = exp(second_sum / num_variables);
    return -20.0 * exp_term_1 - exp_term_2 + M_E + 20.0;
}
#endif

// ==================== HASH TABLE OPERATIONS ====================

__device__ __forceinline__ unsigned long long generateKey_V100(const int* __restrict__ agent_node, int agent_id) {
    unsigned long long key = 0;

    // Простой цикл без векторизации
    for (int i = 0; i < PARAMETR_SIZE; i++) {
        int val = agent_node[agent_id * PARAMETR_SIZE + i];
        key = key * MAX_VALUE_SIZE + (unsigned long long)val;
    }
    return key;
}
__device__ __forceinline__ unsigned int fast_modulo(unsigned long long value, unsigned int divisor) {
    // Быстрый способ вычисления остатка для степени двойки
    if ((divisor & (divisor - 1)) == 0) { // Проверка на степень двойки
        return value & (divisor - 1);
    }
    // Для не-степеней двойки используем умножение
    return value % divisor;
}
__device__ __forceinline__ double getCachedResult_V100(HashEntry* __restrict__ hashTable,
    const int* __restrict__ agent_node,
    int agent_id) {
#if GO_HASH
    unsigned long long key = generateKey_V100(agent_node, agent_id);

    // Безопасное вычисление индекса
    unsigned int idx;
    if ((HASH_TABLE_SIZE & (HASH_TABLE_SIZE - 1)) == 0) {
        // Если HASH_TABLE_SIZE - степень двойки, используем битовую маску
        idx = key & (HASH_TABLE_SIZE - 1);
    }
    else {
        // Иначе используем простое вычисление
        idx = (unsigned int)(key % HASH_TABLE_SIZE);
    }

    // Линейное probing вместо квадратичного
    for (int i = 0; i < MAX_PROBES; i++) {
        unsigned long long entry_key = hashTable[idx].key;
        if (entry_key == key) return hashTable[idx].value;
        if (entry_key == ZERO_HASH_RESULT) return -1.0;

        // Простое инкрементирование индекса
        idx++;
        if (idx >= HASH_TABLE_SIZE) idx = 0;
    }
#endif
    return -1.0;
}
__device__ __forceinline__ void saveToCache_V100(HashEntry* __restrict__ hashTable,
    const int* __restrict__ agent_node,
    int agent_id,
    double value) {
#if GO_HASH
    unsigned long long key = generateKey_V100(agent_node, agent_id);

    // Безопасное вычисление индекса
    unsigned int idx;
    if ((HASH_TABLE_SIZE & (HASH_TABLE_SIZE - 1)) == 0) {
        idx = key & (HASH_TABLE_SIZE - 1);
    }
    else {
        idx = (unsigned int)(key % HASH_TABLE_SIZE);
    }

    for (int i = 0; i < MAX_PROBES; i++) {
        unsigned long long old = atomicCAS(&hashTable[idx].key, ZERO_HASH_RESULT, key);
        if (old == ZERO_HASH_RESULT || old == key) {
            hashTable[idx].value = value;
            return;
        }

        // Простое инкрементирование индекса
        idx++;
        if (idx >= HASH_TABLE_SIZE) idx = 0;
    }
#endif
}
// ==================== ATOMIC OPERATIONS FOR V100 ====================

__device__ __forceinline__ void atomicMax_V100(double* address, double value) {
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long old = *addr_as_ull, assumed;

    do {
        assumed = old;
        if (value <= __longlong_as_double(assumed)) break;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(value));
    } while (assumed != old);
}
__device__ __forceinline__ void atomicMin_V100(double* address, double value) {
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long old = *addr_as_ull, assumed;

    do {
        assumed = old;
        if (value >= __longlong_as_double(assumed)) break;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(value));
    } while (assumed != old);
}

// ==================== KERNELS OPTIMIZED FOR V100 ====================

// Ядро испарения феромонов с векторизацией для HBM2 памяти V100
__global__ void evaporatePheromones_V100(double* __restrict__ pheromon) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    const double ro = PARAMETR_RO;

    // Обрабатываем по 8 элементов за раз для лучшего использования пропускной способности HBM2
    for (unsigned int i = tid * 8; i < PARAMETR_SIZE * MAX_VALUE_SIZE; i += stride * 8) {
        // Развертывание для лучшей производительности на V100
#pragma unroll
        for (int j = 0; j < 8 && (i + j) < PARAMETR_SIZE * MAX_VALUE_SIZE; j++) {
            pheromon[i + j] *= ro;
        }
    }
}

// Ядро вычисления вероятностей с shared memory для V100
__global__ void computeProbabilities_V100(const double* __restrict__ pheromon, const double* __restrict__ kol_enter, double* __restrict__ norm_matrix_probability) {

    // Используем динамический shared memory
    extern __shared__ double shared_data[];
    double* shared_pheromon = shared_data;
    double* shared_kol = &shared_data[MAX_VALUE_SIZE];

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int tid = bx * blockDim.x + tx;

    if (tid >= PARAMETR_SIZE) return;

    // Загружаем данные в shared memory из глобальной памяти
    if (tx < MAX_VALUE_SIZE) {
        // Загрузка из ГЛОБАЛЬНОЙ памяти - используем ldg_global
        shared_pheromon[tx] = ldg_global(&pheromon[tid * MAX_VALUE_SIZE + tx]);
        shared_kol[tx] = ldg_global(&kol_enter[tid * MAX_VALUE_SIZE + tx]);
    }
    __syncthreads();

    // Вычисление суммы феромонов с развертыванием
    double sum_pheromon = 0.0;
#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        // Чтение из SHARED памяти - используем простое чтение
        sum_pheromon += shared_pheromon[i];
    }

    const double inv_sum_pheromon = (sum_pheromon > 1e-10) ? 1.0 / sum_pheromon : 0.0;

    // Вычисление вероятностей
    double probabilities[MAX_VALUE_SIZE];
    double sum_prob = 0.0;

#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        // Чтение из SHARED памяти - используем простое чтение
        const double pheromon_norm = shared_pheromon[i] * inv_sum_pheromon;
        const double kol_val = shared_kol[i];
        probabilities[i] = (kol_val > 0.0 && pheromon_norm > 0.0) ?
            (1.0 / kol_val) + pheromon_norm : 0.0;
        sum_prob += probabilities[i];
    }

    // Нормализация
    const double inv_sum_prob = (sum_prob > 1e-10) ? 1.0 / sum_prob : 0.0;
    double cumulative = 0.0;

#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        cumulative += probabilities[i] * inv_sum_prob;
        norm_matrix_probability[tid * MAX_VALUE_SIZE + i] = cumulative;
    }

    // Последний элемент всегда 1.0
    if (tx == 0) {
        norm_matrix_probability[tid * MAX_VALUE_SIZE + MAX_VALUE_SIZE - 1] = 1.0;
    }
}

// Ядро оптимизации муравьев для V100 с оптимизированным RNG
__global__ void antColonyOptimization_V100(const double* __restrict__ dev_parametr_value,
    const double* __restrict__ norm_matrix_probability,
    int* __restrict__ agent_node,
    double* __restrict__ OF,
    HashEntry* __restrict__ hashTable,
    double* __restrict__ maxOf_dev,
    double* __restrict__ minOf_dev,
    int* __restrict__ kol_hash_fail) {

    const int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_id >= ANT_SIZE) return;

    // Используем Philox RNG который лучше работает на V100
    curandStatePhilox4_32_10_t state;
    curand_init(clock64(), ant_id, 0, &state);

    // Выбор путей для всех параметров
    for (int param_idx = 0; param_idx < PARAMETR_SIZE; param_idx++) {
        const float4 randoms = curand_uniform4(&state);
        const double randomValue = randoms.x;
        int selected_index = 0;

        // ПРОСТОЙ линейный поиск без векторизации и проверок выравнивания
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            if (randomValue <= norm_matrix_probability[param_idx * MAX_VALUE_SIZE + i]) {
                selected_index = i;
                break;
            }
        }

        agent_node[ant_id * PARAMETR_SIZE + param_idx] = selected_index;
    }

    // Проверка кэша
    double cached = -1.0;

#if GO_HASH
    cached = getCachedResult_V100(hashTable, agent_node, ant_id);
#endif

    if (cached < 0.0) {
        // Создаем временный массив параметров
        double temp_params[PARAMETR_SIZE];

        // Загрузка параметров без векторизации
        for (int i = 0; i < PARAMETR_SIZE; i++) {
            const int idx = agent_node[ant_id * PARAMETR_SIZE + i];
            temp_params[i] = dev_parametr_value[i * MAX_VALUE_SIZE + idx];
        }

        // Вычисление функции приспособленности
        OF[ant_id] = BenchShafferaFunction_V100(temp_params);

#if GO_HASH
        saveToCache_V100(hashTable, agent_node, ant_id, OF[ant_id]);
#endif
    }
    else {
        OF[ant_id] = cached;
        atomicAdd(kol_hash_fail, 1);
    }

    // Обновление глобальных min/max значений
    const double fitness = OF[ant_id];
    if (fitness != ZERO_HASH_RESULT && !isnan(fitness) && !isinf(fitness)) {
        atomicMax_V100(maxOf_dev, fitness);
        atomicMin_V100(minOf_dev, fitness);
    }
}
// Ядро обновления феромонов с warp-level оптимизациями для V100
__global__ void updatePheromones_V100(const double* __restrict__ OF,
    const int* __restrict__ agent_node,
    double* __restrict__ pheromon,
    double* __restrict__ kol_enter) {

    // Самый простой вариант - без shared memory
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Каждый поток обрабатывает свой диапазон агентов
    const int agents_per_thread = (ANT_SIZE + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
    const int start_agent = tid * agents_per_thread;
    const int end_agent = min(start_agent + agents_per_thread, ANT_SIZE);

    // Прямое обновление глобальной памяти
    for (int ant_id = start_agent; ant_id < end_agent; ant_id++) {
        const double fitness = OF[ant_id];
        if (fitness != ZERO_HASH_RESULT) {
            // Обновляем все параметры для этого агента
            for (int param_idx = 0; param_idx < PARAMETR_SIZE; param_idx++) {
                const int k = agent_node[ant_id * PARAMETR_SIZE + param_idx];

                atomicAdd(&kol_enter[param_idx * MAX_VALUE_SIZE + k], 1.0);

#if OPTIMIZE_MIN_1
                const double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - fitness;
                if (delta > 0.0) {
                    atomicAdd(&pheromon[param_idx * MAX_VALUE_SIZE + k], PARAMETR_Q * delta);
                }
#endif
            }
        }
    }
}
// Инициализация хэш-таблицы
__global__ void initializeHashTable_V100(HashEntry* hashTable) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = tid; i < HASH_TABLE_SIZE; i += stride) {
        hashTable[i].key = ZERO_HASH_RESULT;
        hashTable[i].value = 0.0;
    }
}

// ==================== GLOBAL VARIABLES ====================

static double* dev_parametr_value = nullptr;
static double* dev_pheromon = nullptr;
static double* dev_kol_enter = nullptr;
static double* dev_norm_matrix = nullptr;
static double* dev_OF = nullptr;
static double* dev_max = nullptr;
static double* dev_min = nullptr;
static int* dev_agent_node = nullptr;
static int* dev_hash_fail = nullptr;
static HashEntry* dev_hashTable = nullptr;

static cudaStream_t compute_stream = nullptr;
static cudaEvent_t kernel_start, kernel_stop;

// ==================== HOST FUNCTIONS ====================

bool initialize_cuda_resources_V100(const double* params, const double* pheromon, const double* kol_enter) {
    std::cout << "Initializing CUDA resources for Tesla V100..." << std::endl;

    // Проверка архитектуры GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    if (prop.major < 7) {
        std::cerr << "ERROR: This code requires Volta architecture (V100) or higher!" << std::endl;
        std::cerr << "Detected GPU: " << prop.name << " with Compute Capability "
            << prop.major << "." << prop.minor << std::endl;
        return false;
    }

    std::cout << "GPU: " << prop.name << " (Compute Capability "
        << prop.major << "." << prop.minor << ")" << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024.0) << " GB" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;

    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));

    const size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    const size_t agent_node_size = PARAMETR_SIZE * ANT_SIZE * sizeof(int);

    std::cout << "Matrix size: " << matrix_size << " bytes" << std::endl;
    std::cout << "Agent node size: " << agent_node_size << " bytes" << std::endl;

    // Выделение памяти
    CUDA_CHECK(cudaMalloc(&dev_parametr_value, matrix_size));
    CUDA_CHECK(cudaMalloc(&dev_pheromon, matrix_size));
    CUDA_CHECK(cudaMalloc(&dev_kol_enter, matrix_size));
    CUDA_CHECK(cudaMalloc(&dev_norm_matrix, matrix_size));
    CUDA_CHECK(cudaMalloc(&dev_agent_node, agent_node_size));
    CUDA_CHECK(cudaMalloc(&dev_OF, ANT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev_max, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev_min, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev_hash_fail, sizeof(int)));

    size_t total_allocated = matrix_size * 3 + agent_node_size +
        ANT_SIZE * sizeof(double) +
        sizeof(double) * 2 + sizeof(int);

    std::cout << "Allocated " << total_allocated / (1024.0 * 1024.0)
        << " MB on GPU" << std::endl;

    // Копирование данных на устройство
    CUDA_CHECK(cudaMemcpy(dev_parametr_value, params, matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_pheromon, pheromon, matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_kol_enter, kol_enter, matrix_size, cudaMemcpyHostToDevice));

    // Инициализация хэш-таблицы
#if GO_HASH
    CUDA_CHECK(cudaMalloc(&dev_hashTable, HASH_TABLE_SIZE * sizeof(HashEntry)));

    int threads = 256;
    int blocks = (HASH_TABLE_SIZE + threads - 1) / threads;
    initializeHashTable_V100 << <blocks, threads, 0, compute_stream >> > (dev_hashTable);
#endif

    // Инициализация данных
    double init_max = -1e9;
    double init_min = 1e9;
    int init_fail = 0;

    CUDA_CHECK(cudaMemcpy(dev_max, &init_max, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_min, &init_min, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_hash_fail, &init_fail, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dev_OF, 0, ANT_SIZE * sizeof(double)));

    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    CUDA_CHECK(cudaGetLastError());

    std::cout << "CUDA resources initialized successfully!" << std::endl;
    return true;
}

void check_configuration() {
    std::cout << "\n=== Configuration Check ===" << std::endl;
    std::cout << "PARAMETR_SIZE: " << PARAMETR_SIZE << std::endl;
    std::cout << "MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << std::endl;
    std::cout << "ANT_SIZE: " << ANT_SIZE << std::endl;
    std::cout << "V100_BLOCK_SIZE: " << V100_BLOCK_SIZE << std::endl;

    // Проверка размеров
    size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
    std::cout << "Matrix elements: " << matrix_size << std::endl;
    std::cout << "Total memory for matrices: "
        << (matrix_size * 3 * sizeof(double)) / (1024.0 * 1024.0)
        << " MB" << std::endl;

    // Проверка GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "\n=== GPU Limits ===" << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;

    if (V100_BLOCK_SIZE > prop.maxThreadsPerBlock) {
        std::cerr << "ERROR: V100_BLOCK_SIZE exceeds GPU limit!" << std::endl;
    }
}

PerformanceMetrics run_aco_iterations_V100(int num_iterations) {
    PerformanceMetrics metrics = { 0 };
    auto total_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t start, start_ant, start_update, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&start_ant));
    CUDA_CHECK(cudaEventCreate(&start_update));
    CUDA_CHECK(cudaEventCreate(&stop));
    float kernel_time = 0.0, computeProbabilities_time = 0.0, antColonyOptimization_time = 0.0, updatePheromones_time = 0.0;

    // Конфигурация для V100 (80 SM, 512 потоков на блок)
    const int threads_per_block = V100_BLOCK_SIZE;
    const int blocks_per_grid = (PARAMETR_SIZE + threads_per_block - 1) / threads_per_block;

    const int ant_threads_per_block = std::min(V100_BLOCK_SIZE, ANT_SIZE);
    const int ant_blocks_per_grid = (ANT_SIZE + ant_threads_per_block - 1) / ant_threads_per_block;

    // Инициализация статистики
    double max_init = -1e9;
    double min_init = 1e9;
    int fail_init = 0;

    CUDA_CHECK(cudaMemcpyAsync(dev_max, &max_init, sizeof(double), cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_min, &min_init, sizeof(double), cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_hash_fail, &fail_init, sizeof(int), cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    // Основной цикл ACO
    for (int iter = 0; iter < num_iterations; ++iter) {
        float iter_time;
        CUDA_CHECK(cudaEventRecord(start, compute_stream));

        // Исправленный вызов ядра: правильный размер shared memory
        size_t shared_mem_size = 2 * MAX_VALUE_SIZE * sizeof(double);
        computeProbabilities_V100 << <blocks_per_grid, threads_per_block, shared_mem_size, compute_stream >> > (dev_pheromon, dev_kol_enter, dev_norm_matrix);

        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        computeProbabilities_time += iter_time;
        CUDA_CHECK(cudaEventRecord(start_ant, compute_stream));

        antColonyOptimization_V100 << <ant_blocks_per_grid, ant_threads_per_block, 0, compute_stream >> > (dev_parametr_value, dev_norm_matrix, dev_agent_node, dev_OF, dev_hashTable, dev_max, dev_min, dev_hash_fail);

        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start_ant, stop));
        antColonyOptimization_time += iter_time;
        CUDA_CHECK(cudaEventRecord(start_update, compute_stream));

        evaporatePheromones_V100 << <blocks_per_grid, threads_per_block, 0, compute_stream >> > (dev_pheromon);
        const size_t shared_mem_update = 2 * threads_per_block * MAX_VALUE_SIZE * sizeof(double);
        updatePheromones_V100 << <blocks_per_grid, threads_per_block, shared_mem_update, compute_stream >> > (dev_OF, dev_agent_node, dev_pheromon, dev_kol_enter);

        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start_update, stop));
        updatePheromones_time += iter_time;
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        kernel_time += iter_time;
    }
    // Сбор результатов
    double best_fitness, worst_fitness;
    int hash_fails;

    CUDA_CHECK(cudaMemcpy(&best_fitness, dev_min, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&worst_fitness, dev_max, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hash_fails, dev_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    metrics.min_fitness = best_fitness;
    metrics.max_fitness = worst_fitness;
    metrics.hash_misses = hash_fails;
    metrics.hash_hits = num_iterations * ANT_SIZE - hash_fails;

    // Получение метрик производительности
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int warp_size = prop.warpSize;
    int warps_per_block = (threads_per_block + warp_size - 1) / warp_size;
    int max_warps_per_sm = max_threads_per_sm / warp_size;
    metrics.occupancy = std::min(100.0f, (warps_per_block * 32.0f / max_warps_per_sm) * 100.0f);


    cudaEventDestroy(start);
    cudaEventDestroy(start_ant);
    cudaEventDestroy(start_update);
    cudaEventDestroy(stop);

    auto total_end = std::chrono::high_resolution_clock::now();
    metrics.total_time_ms = std::chrono::duration<float, std::milli>(total_end - total_start).count();
    metrics.kernel_time_ms = kernel_time;
    metrics.computeProbabilities_time_ms = computeProbabilities_time;
    metrics.antColonyOptimization_time_ms = antColonyOptimization_time;
    metrics.updatePheromones_time_ms = updatePheromones_time;
    metrics.memory_time_ms = metrics.total_time_ms - kernel_time;
    // Расчет пропускной способности памяти
    size_t total_data_transferred = (MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double)) * 3 * num_iterations + (PARAMETR_SIZE * ANT_SIZE * sizeof(int)) * num_iterations + (ANT_SIZE * sizeof(double)) * num_iterations;
    metrics.memory_throughput_gbs = (total_data_transferred / (1024.0 * 1024.0 * 1024.0)) / (metrics.total_time_ms / 1000.0);

    return metrics;
}

bool load_matrix_data(const std::string& filename, std::vector<double>& params,
    std::vector<double>& pheromones, std::vector<double>& visits) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    const size_t total_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
    params.resize(total_size);
    pheromones.resize(total_size);
    visits.resize(total_size);

    for (size_t i = 0; i < total_size; ++i) {
        if (!(file >> params[i])) {
            std::cerr << "Error reading file: " << filename << " at element " << i << std::endl;
            return false;
        }

        if (params[i] != -100.0) {
            pheromones[i] = 1.0;
            visits[i] = 1.0;
        }
        else {
            params[i] = pheromones[i] = visits[i] = 0.0;
        }
    }

    file.close();
    std::cout << "Loaded " << total_size << " elements from " << filename << std::endl;
    return true;
}

void print_gpu_info_V100() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;

    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        std::cout << "\n=== Device " << device << ": " << prop.name << " ===" << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global Memory:      " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared Memory/Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  L2 Cache Size:      " << prop.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "  SM Count:           " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Threads/Block:  " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Warp Size:          " << prop.warpSize << std::endl;
        std::cout << "  Memory Clock Rate:  " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Bus Width:   " << prop.memoryBusWidth << " bits" << std::endl;

        if (prop.major < 7) {
            std::cout << "  WARNING: This device is not a Volta V100!" << std::endl;
        }
    }

    // Выбор устройства 0
    CUDA_CHECK(cudaSetDevice(0));
}

void print_metrics(const PerformanceMetrics& metrics, const char* str, int run_id) {
    const double hit_rate = (metrics.hash_hits + metrics.hash_misses > 0) ?
        100.0 * metrics.hash_hits / (metrics.hash_hits + metrics.hash_misses) : 0.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Run " << str << run_id << ": "
        << "Total=" << metrics.total_time_ms << "ms "
        << "Kernel=" << metrics.kernel_time_ms << "ms "
        << "(Prob=" << metrics.computeProbabilities_time_ms << "ms "
        << "Ant=" << metrics.antColonyOptimization_time_ms << "ms "
        << "Update=" << metrics.updatePheromones_time_ms << "ms) "
        << "Memory=" << metrics.memory_time_ms << "ms "
        << "HitRate=" << hit_rate << "% "
        << "Occupancy=" << metrics.occupancy << "% "
        << "Throughput=" << metrics.memory_throughput_gbs << " GB/s "
        << "MIN=" << std::setprecision(6) << metrics.min_fitness << " "
        << "MAX=" << metrics.max_fitness
        << std::endl;

    logFile << std::fixed << std::setprecision(2);
    logFile << "Run " << str << run_id << "; "
        << metrics.total_time_ms << "; "
        << metrics.kernel_time_ms << "; "
        << metrics.computeProbabilities_time_ms << "; "
        << metrics.antColonyOptimization_time_ms << "; "
        << metrics.updatePheromones_time_ms << "; "
        << metrics.memory_time_ms << "; "
        << hit_rate << "; "
        << metrics.occupancy << "; "
        << metrics.memory_throughput_gbs << "; "
        << std::setprecision(6) << metrics.min_fitness << "; "
        << metrics.max_fitness << std::endl;
}

void cleanup_cuda_resources_V100() {
    std::cout << "Cleaning up CUDA resources..." << std::endl;

    if (compute_stream) {
        cudaStreamDestroy(compute_stream);
        compute_stream = nullptr;
    }

    if (kernel_start) cudaEventDestroy(kernel_start);
    if (kernel_stop) cudaEventDestroy(kernel_stop);

    if (dev_parametr_value) cudaFree(dev_parametr_value);
    if (dev_pheromon) cudaFree(dev_pheromon);
    if (dev_kol_enter) cudaFree(dev_kol_enter);
    if (dev_norm_matrix) cudaFree(dev_norm_matrix);
    if (dev_agent_node) cudaFree(dev_agent_node);
    if (dev_OF) cudaFree(dev_OF);
    if (dev_max) cudaFree(dev_max);
    if (dev_min) cudaFree(dev_min);
    if (dev_hash_fail) cudaFree(dev_hash_fail);

#if GO_HASH
    if (dev_hashTable) cudaFree(dev_hashTable);
#endif

    // Сброс указателей
    dev_parametr_value = dev_pheromon = dev_kol_enter = dev_norm_matrix = nullptr;
    dev_agent_node = nullptr;
    dev_OF = dev_max = dev_min = nullptr;
    dev_hash_fail = nullptr;
#if GO_HASH
    dev_hashTable = nullptr;
#endif

    cudaDeviceReset();
    std::cout << "CUDA resources cleaned up successfully!" << std::endl;
}

// Функция для тестирования производительности отдельных ядер
void benchmark_kernels_V100() {
    std::cout << "\n=== Benchmarking V100 Kernels ===" << std::endl;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Тест ядра испарения
    CUDA_CHECK(cudaEventRecord(start, compute_stream));
    evaporatePheromones_V100 << <256, 512, 0, compute_stream >> > (dev_pheromon);
    CUDA_CHECK(cudaEventRecord(stop, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    float time;
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));

    // Тест ядра вычисления вероятностей
    CUDA_CHECK(cudaEventRecord(start, compute_stream));
    size_t shared_mem_size = 2 * MAX_VALUE_SIZE * sizeof(double);
    computeProbabilities_V100 << <256, 512, shared_mem_size, compute_stream >> > (dev_pheromon, dev_kol_enter, dev_norm_matrix);
    CUDA_CHECK(cudaEventRecord(stop, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::cout << "==================================================================" << std::endl;
    std::cout << "  CUDA ACO Global Optimization - Tesla V100 Optimized Version" << std::endl;
    std::cout << "  Compiled for Compute Capability 7.0+ (Volta Architecture)" << std::endl;
    std::cout << "==================================================================" << std::endl;

    PerformanceMetrics metrics;

    // Вывод информации о GPU
    print_gpu_info_V100();

    // Вывод информации о конфигурации
    std::cout << "\n=== Configuration Parameters ===" << std::endl;
    std::cout << "  PARAMETR_SIZE:     " << PARAMETR_SIZE << std::endl;
    std::cout << "  MAX_VALUE_SIZE:    " << MAX_VALUE_SIZE << std::endl;
    std::cout << "  SET_PARAMETR_SIZE_ONE_X: " << SET_PARAMETR_SIZE_ONE_X << std::endl;
    std::cout << "  ANT_SIZE:          " << ANT_SIZE << std::endl;
    std::cout << "  KOL_ITERATION:     " << KOL_ITERATION << std::endl;
    std::cout << "  V100_BLOCK_SIZE:   " << V100_BLOCK_SIZE << std::endl;
    std::cout << "  GO_HASH:           " << (GO_HASH ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  HASH_TABLE_SIZE:   " << HASH_TABLE_SIZE << std::endl;

    // Загрузка данных
    std::cout << "\n=== Loading Data ===" << std::endl;
    std::cout << "File: " << NAME_FILE_GRAPH << std::endl;

    std::vector<double> params, pheromones, visits;
    if (!load_matrix_data(NAME_FILE_GRAPH, params, pheromones, visits)) {
        std::cerr << "Failed to load matrix data!" << std::endl;
        return 1;
    }

    // Инициализация CUDA ресурсов
    std::cout << "\n=== Initializing CUDA Resources ===" << std::endl;
    if (!initialize_cuda_resources_V100(params.data(), pheromones.data(), visits.data())) {
        std::cerr << "Failed to initialize CUDA resources!" << std::endl;
        return 1;
    }
    check_configuration();
    // Прогрев GPU
    std::cout << "\n=== Performing Warmup Runs ===" << std::endl;
    for (int i = 0; i < KOL_PROGREV; ++i) {
        std::cout << "Warmup " << (i + 1) << "/" << KOL_PROGREV << std::endl;
        run_aco_iterations_V100(KOL_ITERATION);
    }

    // Бенчмарк ядер (опционально)
    if (BENCHMARK_KERNELS) {
        benchmark_kernels_V100();
    }

    // Основной цикл оптимизации
    std::cout << "\n=== Starting ACO Optimization ===" << std::endl;
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < KOL_PROGON_STATISTICS; ++i) {
        std::cout << "\n--- Run " << (i + 1) << "/" << KOL_PROGON_STATISTICS << " ---" << std::endl;
        metrics = run_aco_iterations_V100(KOL_ITERATION);
        print_metrics(metrics, "V100", i + 1);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);

    // Итоговые результаты
    std::cout << "\n==================================================================" << std::endl;
    std::cout << "  BENCHMARK RESULTS - TESLA V100" << std::endl;
    std::cout << "==================================================================" << std::endl;

    std::cout << "\nPerformance Summary:" << std::endl;
    std::cout << "  Total Execution Time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "  Average Time per Run: " << total_duration.count() / KOL_PROGON_STATISTICS << " ms" << std::endl;
    std::cout << "  Kernel Efficiency: "
        << (metrics.kernel_time_ms / metrics.total_time_ms * 100.0) << "%" << std::endl;

    std::cout << "\nKernel Breakdown:" << std::endl;
    std::cout << "  ComputeProbabilities: " << metrics.computeProbabilities_time_ms << " ms ("
        << (metrics.computeProbabilities_time_ms / metrics.kernel_time_ms * 100.0) << "%)" << std::endl;
    std::cout << "  AntColonyOptimization: " << metrics.antColonyOptimization_time_ms << " ms ("
        << (metrics.antColonyOptimization_time_ms / metrics.kernel_time_ms * 100.0) << "%)" << std::endl;
    std::cout << "  UpdatePheromones: " << metrics.updatePheromones_time_ms << " ms ("
        << (metrics.updatePheromones_time_ms / metrics.kernel_time_ms * 100.0) << "%)" << std::endl;

    std::cout << "\nMemory Performance:" << std::endl;
    std::cout << "  Memory Throughput: " << metrics.memory_throughput_gbs << " GB/s" << std::endl;
    std::cout << "  Memory Time: " << metrics.memory_time_ms << " ms ("
        << (metrics.memory_time_ms / metrics.total_time_ms * 100.0) << "%)" << std::endl;

    std::cout << "\nOptimization Results:" << std::endl;
    std::cout << "  Best Fitness:  " << std::scientific << std::setprecision(6)
        << metrics.min_fitness << std::endl;
    std::cout << "  Worst Fitness: " << metrics.max_fitness << std::endl;
    std::cout << "  Cache Hit Rate: " << std::fixed << std::setprecision(2)
        << (100.0 * metrics.hash_hits / std::max(1, metrics.hash_hits + metrics.hash_misses))
        << "%" << std::endl;

    std::cout << "\nGPU Utilization:" << std::endl;
    std::cout << "  Theoretical Occupancy: " << metrics.occupancy << "%" << std::endl;

    std::cout << "\n==================================================================" << std::endl;
    std::cout << "  ACO Optimization Completed Successfully!" << std::endl;
    std::cout << "==================================================================" << std::endl;

    // Очистка ресурсов
    cleanup_cuda_resources_V100();

    logFile.close();
    return 0;
}