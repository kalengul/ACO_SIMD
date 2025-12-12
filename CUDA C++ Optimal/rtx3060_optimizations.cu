#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "parametrs.h"

#define CUDA_CHECK(call) do {cudaError_t err = call; if (err != cudaSuccess) {std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl;  exit(EXIT_FAILURE); } } while(0)

// Глобальный лог-файл
std::ofstream logFile("log.txt");

struct alignas(32) HashEntry {
    unsigned long long key;
    double value;
    unsigned long long padding; // Для выравнивания 32 байта
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

// Векторизованный доступ к памяти с использованием ldg
template<typename T>
__device__ __forceinline__ T ldg(const T* ptr) {
#if __CUDA_ARCH__ >= 320
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

__device__ __forceinline__ double go_x(const double* __restrict__ parametr, int start_index) {
    double sum = 0.0;
#pragma unroll
    for (int i = 1; i < SET_PARAMETR_SIZE_ONE_X; ++i) {
        sum += parametr[start_index + i];
    }
    return parametr[start_index] * sum;
}

// ==================== TEST FUNCTIONS OPTIMIZED ====================

#if (SHAFFERA)
__device__ double BenchShafferaFunction(const double* __restrict__ parametr) {
    double r_squared = 0.0;
    const int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;

#pragma unroll 4
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X);
        r_squared += x * x;
    }

    double r = sqrt(r_squared);
    double sin_r = sin(r);
    return 0.5 - (sin_r * sin_r - 0.5) / (1.0 + 0.001 * r_squared);
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
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma unroll 4
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X);
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
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma unroll 4
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X);
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
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma unroll 4
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X);
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
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma unroll 4
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X);
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
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma unroll 4
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X);
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
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma unroll 4
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X);
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
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma unroll 4
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X);
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
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma unroll 4
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X);
        sum -= x * sin(sqrt(fabs(x)));
    }
    return sum;
}
#endif
#if (LEVY)
// Леви-функция
__device__ double BenchShafferaFunction(double* parametr) {
    double w_first = 1 + (go_x(parametr, 0) - 1) / 4;
    double w_last = 1 + (go_x(parametr, (num_variables - 1) * SET_PARAMETR_SIZE_ONE_X) - 1) / 4;
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma unroll 4
    for (int i = 1; i <= num_variables - 1; ++i) {
        double wi = 1 + (go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X) - 1) / 4;
        double wi_prev = 1 + (go_x(parametr, (i - 1) * SET_PARAMETR_SIZE_ONE_X) - 1) / 4;
        sum += pow(wi - 1, 2) * (1 + 10 * pow(sin(M_PI * wi), 2)) +
            pow(wi - wi_prev, 2) * (1 + pow(sin(2 * M_PI * wi), 2));
    }
    return pow(sin(M_PI * w_first), 2) + sum + pow(w_last - 1, 2) * (1 + pow(sin(2 * M_PI * w_last), 2));
}
#endif
#if (MICHAELWICZYNSKI)
// Михаэлевич-Викинский
__device__ double BenchShafferaFunction(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
#pragma unroll 4
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X);
        sum -= sin(x) * pow(sin((i + 1) * x * x / M_PI), 20);
    }
    return sum;
}
#endif

__device__ __forceinline__ unsigned long long generateKey_Ampere(const int* __restrict__ agent_node, int agent_id) {
    unsigned long long key = 0;

    for (int i = 0; i < PARAMETR_SIZE; i++) {
        key = key * MAX_VALUE_SIZE + agent_node[agent_id * PARAMETR_SIZE + i];
    }

    return key;
}
__device__ __forceinline__ double getCachedResult_Ampere(HashEntry* __restrict__ hashTable, const int* __restrict__ agent_node, int agent_id) {
#if GO_HASH
    unsigned long long key = generateKey_Ampere(agent_node, agent_id);
    unsigned int idx = key % HASH_TABLE_SIZE;
    for (int i = 0; i < MAX_PROBES; i++) {
        if (hashTable[idx].key == key) return hashTable[idx].value;
        if (hashTable[idx].key == ZERO_HASH_RESULT) return -1.0;
        idx = (idx + (i + 1) * (i + 1)) % HASH_TABLE_SIZE;
    }
#endif
    return -1.0;
}
__device__ __forceinline__ void saveToCache(HashEntry* __restrict__ hashTable, const int* __restrict__ agent_node, int agent_id, double value) {
#if GO_HASH
    unsigned long long key = generateKey_Ampere(agent_node, agent_id);
    unsigned long long idx = key % HASH_TABLE_SIZE;

    for (int i = 0; i < MAX_PROBES; i++) {
        unsigned long long old = atomicCAS(&hashTable[idx].key, ZERO_HASH_RESULT, key);
        if (old == ZERO_HASH_RESULT || old == key) {
            hashTable[idx].value = value;
            return;
        }
        idx = (idx + (i + 1) * (i + 1)) % HASH_TABLE_SIZE;
    }
#endif
}

__device__ __forceinline__ void atomicMax_Ampere(double* address, double value) {
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long old = *addr_as_ull, assumed;

    do {
        assumed = old;
        if (value <= __longlong_as_double(assumed)) break;

        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(value));
    } while (assumed != old);
}
__device__ __forceinline__ void atomicMin(double* address, double value) {
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long old = *addr_as_ull, assumed;
    do {
        assumed = old;
        if (value >= __longlong_as_double(assumed)) break;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(value));
    } while (assumed != old);
}

// Ядро испарения феромонов 
__global__ void evaporatePheromones_Ampere(double* __restrict__ pheromon) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Предзагрузка константы в регистр
    const double ro = PARAMETR_RO;

    // Векторизованный доступ для лучшей пропускной способности памяти
    for (unsigned int i = tid * 4; i < PARAMETR_SIZE * MAX_VALUE_SIZE; i += stride * 4) {
        // Обрабатываем 4 элемента за раз
        for (int j = 0; j < 4 && (i + j) < PARAMETR_SIZE * MAX_VALUE_SIZE; j++) {
            pheromon[i + j] *= ro;
        }
    }
}

// Ядро вычисления вероятностей
__global__ void computeProbabilities_Ampere(const double* __restrict__ pheromon, const double* __restrict__ kol_enter, double* __restrict__ norm_matrix_probability) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = bx * blockDim.x + tx;

    if (tid >= PARAMETR_SIZE) return;

    // Вычисление суммы феромонов
    double sum_pheromon = 0.0;
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        sum_pheromon += pheromon[tid * MAX_VALUE_SIZE + i];
    }

    double inv_sum_pheromon = (sum_pheromon > 1e-10) ? 1.0 / sum_pheromon : 0.0;

    // Вычисление вероятностей
    double probabilities[MAX_VALUE_SIZE];
    double sum_prob = 0.0;

    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        double pheromon_norm = pheromon[tid * MAX_VALUE_SIZE + i] * inv_sum_pheromon;
        double kol_val = kol_enter[tid * MAX_VALUE_SIZE + i];
        probabilities[i] = (kol_val > 0.0 && pheromon_norm > 0.0) ?
            (1.0 / kol_val) + pheromon_norm : 0.0;
        sum_prob += probabilities[i];
    }

    // Нормализация
    double inv_sum_prob = (sum_prob > 1e-10) ? 1.0 / sum_prob : 0.0;
    double cumulative = 0.0;

    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        cumulative += probabilities[i] * inv_sum_prob;
        norm_matrix_probability[tid * MAX_VALUE_SIZE + i] = cumulative;
    }

    // Последний элемент всегда 1.0
    if (threadIdx.x == 0 && tid < PARAMETR_SIZE) {
        norm_matrix_probability[tid * MAX_VALUE_SIZE + MAX_VALUE_SIZE - 1] = 1.0;
    }
}

// Ядро оптимизации муравьев
__global__ void antColonyOptimization_Ampere(const double* __restrict__ dev_parametr_value, const double* __restrict__ norm_matrix_probability, int* __restrict__ agent_node, double* __restrict__ OF, HashEntry* __restrict__ hashTable, double* __restrict__ maxOf_dev, double* __restrict__ minOf_dev, int* __restrict__ kol_hash_fail) {

    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_id >= ANT_SIZE) return;

    // Каждый поток обрабатывает одного агента
    curandState state;
    curand_init(clock64() + ant_id, 0, 0, &state);

    // Выбор путей для всех параметров
    for (int param_idx = 0; param_idx < PARAMETR_SIZE; param_idx++) {
        double randomValue = curand_uniform(&state);
        int selected_index = 0;

        // Поиск выбранного значения
        while (selected_index < MAX_VALUE_SIZE - 1 &&
            randomValue > norm_matrix_probability[param_idx * MAX_VALUE_SIZE + selected_index]) {
            selected_index++;
        }

        agent_node[ant_id * PARAMETR_SIZE + param_idx] = selected_index;
    }

    // Вычисление целевой функции
    double cached = getCachedResult_Ampere(hashTable, agent_node, ant_id);

    if (cached < 0.0) {
        // СОЗДАЕМ ВРЕМЕННЫЙ МАССИВ ПАРАМЕТРОВ ДЛЯ ВЫЧИСЛЕНИЯ ФУНКЦИИ
        double temp_params[PARAMETR_SIZE];

        for (int i = 0; i < PARAMETR_SIZE; i++) {
            int selected_idx = agent_node[ant_id * PARAMETR_SIZE + i];
            temp_params[i] = dev_parametr_value[i * MAX_VALUE_SIZE + selected_idx];
        }

        // ВЫЗЫВАЕМ ФУНКЦИЮ БЕНЧМАРКА
        OF[ant_id] = BenchShafferaFunction(temp_params);
        saveToCache(hashTable, agent_node, ant_id, OF[ant_id]);
    }
    else {
        OF[ant_id] = cached;
        atomicAdd(kol_hash_fail, 1);
    }

    // Обновление глобальных значений (ИСПРАВЛЕНО)
    double fitness = OF[ant_id];
    if (fitness != ZERO_HASH_RESULT && !isnan(fitness) && !isinf(fitness)) {
        atomicMax_Ampere(maxOf_dev, fitness);
        atomicMin(minOf_dev, fitness);
    }
}
// Ядро обновления феромонов
__global__ void updatePheromones_Ampere(const double* __restrict__ OF, const int* __restrict__ agent_node, double* __restrict__ pheromon, double* __restrict__ kol_enter) {

    // Используем shared memory для агрегации дельт
    extern __shared__ double shared_data[];
    double* pheromon_delta = shared_data;
    double* kol_delta = &shared_data[blockDim.x * MAX_VALUE_SIZE];

    int thread_id = threadIdx.x;
    int param_idx = blockIdx.x * blockDim.x + thread_id;

    // Инициализация shared memory
    for (int k = 0; k < MAX_VALUE_SIZE; k++) {
        if (thread_id < MAX_VALUE_SIZE) {
            int idx = thread_id * blockDim.x + (threadIdx.x % blockDim.x);
            if (idx < blockDim.x * MAX_VALUE_SIZE) {
                pheromon_delta[idx] = 0.0;
                kol_delta[idx] = 0.0;
            }
        }
    }
    __syncthreads();

    if (param_idx >= PARAMETR_SIZE) return;

    // Каждый поток обрабатывает несколько агентов для лучшей загрузки
    const int agents_per_thread = (ANT_SIZE + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
    int start_agent = thread_id * agents_per_thread;
    int end_agent = min(start_agent + agents_per_thread, ANT_SIZE);

    // Накопление дельт в shared memory
    for (int ant_id = start_agent; ant_id < end_agent; ant_id++) {
        if (OF[ant_id] != ZERO_HASH_RESULT) {
            int k = agent_node[ant_id * PARAMETR_SIZE + param_idx];

            // Атомарное добавление в shared memory
            atomicAdd(&kol_delta[thread_id * MAX_VALUE_SIZE + k], 1.0);

#if OPTIMIZE_MIN_1
            double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[ant_id];
            if (delta > 0.0) {
                atomicAdd(&pheromon_delta[thread_id * MAX_VALUE_SIZE + k], PARAMETR_Q * delta);
            }
#endif
        }
    }
    __syncthreads();

    // Редукция в shared memory (один поток обрабатывает одну колонку)
    for (int k = 0; k < MAX_VALUE_SIZE; k++) {
        double total_pheromon = 0.0;
        double total_kol = 0.0;

        // Суммируем по всем потокам блока
        for (int t = 0; t < blockDim.x; t++) {
            total_pheromon += pheromon_delta[t * MAX_VALUE_SIZE + k];
            total_kol += kol_delta[t * MAX_VALUE_SIZE + k];
        }

        // Один поток обновляет глобальную память для этой колонки
        if (thread_id == (k % blockDim.x)) {
            if (total_kol > 0.0) {
                atomicAdd(&kol_enter[param_idx * MAX_VALUE_SIZE + k], total_kol);
            }
            if (total_pheromon > 0.0) {
                atomicAdd(&pheromon[param_idx * MAX_VALUE_SIZE + k], total_pheromon);
            }
        }
    }
}

// Инициализация хэш-таблицы
__global__ void initializeHashTable(HashEntry* hashTable) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = tid; i < HASH_TABLE_SIZE; i += stride) {
        hashTable[i].key = ZERO_HASH_RESULT;
        hashTable[i].value = 0.0;
    }
}

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

bool initialize_cuda_resources_Ampere(const double* params, const double* pheromon, const double* kol_enter) {
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));

    const size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    const size_t agent_node_size = PARAMETR_SIZE * ANT_SIZE * sizeof(int);

    // Выделение device memory
    CUDA_CHECK(cudaMalloc(&dev_parametr_value, matrix_size));
    CUDA_CHECK(cudaMalloc(&dev_pheromon, matrix_size));
    CUDA_CHECK(cudaMalloc(&dev_kol_enter, matrix_size));
    CUDA_CHECK(cudaMalloc(&dev_norm_matrix, matrix_size));
    CUDA_CHECK(cudaMalloc(&dev_agent_node, agent_node_size));
    CUDA_CHECK(cudaMalloc(&dev_OF, ANT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev_max, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev_min, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev_hash_fail, sizeof(int)));

    // Копирование данных на устройство
    CUDA_CHECK(cudaMemcpy(dev_parametr_value, params, matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_pheromon, pheromon, matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_kol_enter, kol_enter, matrix_size, cudaMemcpyHostToDevice));

    // Инициализация хэш-таблицы
#if GO_HASH
    CUDA_CHECK(cudaMalloc(&dev_hashTable, HASH_TABLE_SIZE * sizeof(HashEntry)));

    int threads = 256;
    int blocks = (HASH_TABLE_SIZE + threads - 1) / threads;
    initializeHashTable << <blocks, threads >> > (dev_hashTable);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // Инициализация данных
    CUDA_CHECK(cudaMemset(dev_OF, 0, ANT_SIZE * sizeof(double)));

    // Prefetch данных в L2 cache
#if USE_L2_PREFETCH
    cudaMemPrefetchAsync(dev_parametr_value, matrix_size, 0);
    cudaMemPrefetchAsync(dev_pheromon, matrix_size, 0);
    cudaMemPrefetchAsync(dev_kol_enter, matrix_size, 0);
#endif

    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    return true;
}

PerformanceMetrics run_aco_iterations_Ampere(int num_iterations) {
    PerformanceMetrics metrics = { 0 };
    auto total_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t start, start_ant, start_update, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&start_ant));
    CUDA_CHECK(cudaEventCreate(&start_update));
    CUDA_CHECK(cudaEventCreate(&stop));
    float kernel_time = 0.0, computeProbabilities_time = 0.0, antColonyOptimization_time = 0.0, updatePheromones_time = 0.0;

    // Конфигурация для Ampere
    int threads_per_block = BLOCK_SIZE;
    int blocks_per_grid = (PARAMETR_SIZE + threads_per_block - 1) / threads_per_block;

    int ant_threads_per_block = std::min(ANT_SIZE, BLOCK_SIZE);
    int ant_blocks_per_grid = (ANT_SIZE + ant_threads_per_block - 1) / ant_threads_per_block;

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

        computeProbabilities_Ampere << <blocks_per_grid, threads_per_block, 0, compute_stream >> > (dev_pheromon, dev_kol_enter, dev_norm_matrix);

        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        computeProbabilities_time += iter_time;
        CUDA_CHECK(cudaEventRecord(start_ant, compute_stream));

        antColonyOptimization_Ampere << <ant_blocks_per_grid, ant_threads_per_block, 0, compute_stream >> > (dev_parametr_value, dev_norm_matrix, dev_agent_node, dev_OF, dev_hashTable, dev_max, dev_min, dev_hash_fail);

        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start_ant, stop));
        antColonyOptimization_time += iter_time;
        CUDA_CHECK(cudaEventRecord(start_update, compute_stream));

        evaporatePheromones_Ampere << <blocks_per_grid, threads_per_block, 0, compute_stream >> > (dev_pheromon);
        size_t shared_mem_size = 2 * threads_per_block * MAX_VALUE_SIZE * sizeof(double);
        updatePheromones_Ampere << <blocks_per_grid, threads_per_block, shared_mem_size, compute_stream >> >(dev_OF, dev_agent_node, dev_pheromon, dev_kol_enter);

        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        kernel_time += iter_time;
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start_update, stop));
        updatePheromones_time += iter_time;
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

bool load_matrix_data(const std::string& filename, std::vector<double>& params, std::vector<double>& pheromones, std::vector<double>& visits) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    size_t total_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
    params.resize(total_size);
    pheromones.resize(total_size);
    visits.resize(total_size);

    for (size_t i = 0; i < total_size; ++i) {
        if (!(file >> params[i])) {
            std::cerr << "Error reading file: " << filename << std::endl;
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
    return true;
}

void print_gpu_info_Ampere() {
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
    }

    // Выбор устройства
    CUDA_CHECK(cudaSetDevice(0));
}
void print_metrics(const PerformanceMetrics& metrics, const char* str, int run_id) {
    std::cout << "Run " << str << run_id << ": "
        << "Time=" << metrics.total_time_ms << "ms "
        << "Kernel=" << metrics.kernel_time_ms << "ms "
        << "Probabilities=" << metrics.computeProbabilities_time_ms << "ms "
        << "Ant=" << metrics.antColonyOptimization_time_ms << "ms "
        << "Update=" << metrics.updatePheromones_time_ms << "ms "
        << "Memory=" << metrics.memory_time_ms << "ms "
        << "HitRate=" << (100.0 * metrics.hash_hits / (metrics.hash_hits + metrics.hash_misses)) << "% "
        << "Occupancy=" << metrics.occupancy << "% "
        << "memory_throughput_gbs=" << metrics.memory_throughput_gbs << " "
        << "MIN=" << metrics.min_fitness << " "
        << "MAX=" << metrics.max_fitness << " "
        << std::endl;
    logFile << "Run " << str << run_id << "; "
        << metrics.total_time_ms << "; "
        << metrics.kernel_time_ms << "; "
        << metrics.computeProbabilities_time_ms << "; "
        << metrics.antColonyOptimization_time_ms << "; "
        << metrics.updatePheromones_time_ms << "; "
        << metrics.memory_time_ms << "; "
        << (100.0 * metrics.hash_hits / (metrics.hash_hits + metrics.hash_misses)) << "; "
        << metrics.occupancy << "; "
        << metrics.memory_throughput_gbs << "; "
        << metrics.min_fitness << "; "
        << metrics.max_fitness << "; "
        << std::endl;
}
void cleanup_cuda_resources_Ampere() {
    if (compute_stream) {
        cudaStreamDestroy(compute_stream);
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
}

int main() {
    std::cout << "==================================================================" << std::endl;
    std::cout << "  CUDA ACO Global Optimization - RTX 3060 Optimized Version" << std::endl;
    std::cout << "==================================================================" << std::endl;
    PerformanceMetrics metrics;
    // Вывод информации о GPU
    print_gpu_info_Ampere();

    // Вывод информации о конфигурации
    std::cout << "\n=== Configuration Parameters ===" << std::endl;
    std::cout << "  PARAMETR_SIZE:     " << PARAMETR_SIZE << std::endl;
    std::cout << "  MAX_VALUE_SIZE:    " << MAX_VALUE_SIZE << std::endl;
    std::cout << "  SET_PARAMETR_SIZE_ONE_X: " << SET_PARAMETR_SIZE_ONE_X << std::endl;
    std::cout << "  ANT_SIZE:          " << ANT_SIZE << std::endl;
    std::cout << "  KOL_ITERATION:     " << KOL_ITERATION << std::endl;
    std::cout << "  BLOCK_SIZE:        " << BLOCK_SIZE << std::endl;
    std::cout << "  GO_HASH:           " << (GO_HASH ? "Enabled" : "Disabled") << std::endl;

    // Загрузка данных
    std::cout << "\n=== Loading Data ===" << std::endl;
    std::cout << "File: " << NAME_FILE_GRAPH << std::endl;

    std::vector<double> params, pheromones, visits;
    if (!load_matrix_data(NAME_FILE_GRAPH, params, pheromones, visits)) {
        std::cerr << "Failed to load matrix data!" << std::endl;
        return 1;
    }

    std::cout << "Data loaded: " << params.size() << " elements" << std::endl;

    // Инициализация CUDA ресурсов с оптимизациями Ampere
    std::cout << "\n=== Initializing CUDA Resources ===" << std::endl;
    if (!initialize_cuda_resources_Ampere(params.data(), pheromones.data(), visits.data())) {
        std::cerr << "Failed to initialize CUDA resources!" << std::endl;
        return 1;
    }

    std::cout << "Performing warmup runs..." << std::endl;
    for (int i = 0; i < KOL_PROGREV; ++i) {
        std::cout << "Warmup " << i << std::endl;
        run_aco_iterations_Ampere(KOL_ITERATION);
    }

    std::cout << "\n=== Starting function ACO ===" << std::endl;
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < KOL_PROGON_STATISTICS; ++i) {
        metrics = run_aco_iterations_Ampere(KOL_ITERATION);
        print_metrics(metrics, "ACO 3060 optimize", i);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    std::cout << "4function ACO total execution time: " << total_duration.count() << " ms" << std::endl;

    // Вывод результатов
    std::cout << "\n==================================================================" << std::endl;
    std::cout << "  BENCHMARK RESULTS - RTX 3060 OPTIMIZED" << std::endl;
    std::cout << "==================================================================" << std::endl;

    std::cout << "\nTime Metrics:" << std::endl;
    std::cout << "  Total Time:        " << metrics.total_time_ms << " ms" << std::endl;
    std::cout << "  Kernel Time:       " << metrics.kernel_time_ms << " ms" << std::endl;

    std::cout << "\nFitness Results:" << std::endl;
    std::cout << "  Best Fitness:      " << metrics.min_fitness << std::endl;
    std::cout << "  Worst Fitness:     " << metrics.max_fitness << std::endl;

#if GO_HASH
    std::cout << "\nCache Statistics:" << std::endl;
    std::cout << "  Cache Hits:        " << metrics.hash_hits << std::endl;
    std::cout << "  Cache Misses:      " << metrics.hash_misses << std::endl;
#endif

    std::cout << "\nConfiguration Summary:" << std::endl;
    std::cout << "  PARAMETR_SIZE:     " << PARAMETR_SIZE << std::endl;
    std::cout << "  ANT_SIZE:          " << ANT_SIZE << std::endl;
    std::cout << "  Iterations:        " << metrics.iterations_completed << std::endl;

    std::cout << "\nTotal Execution Time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "==================================================================" << std::endl;

    // Очистка ресурсов
    std::cout << "\n=== Cleaning up resources ===" << std::endl;
    cleanup_cuda_resources_Ampere();

    return 0;
}