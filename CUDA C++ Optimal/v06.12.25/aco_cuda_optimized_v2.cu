#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "parametrs.h"

//0.00123005

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(err) << std::endl; exit(EXIT_FAILURE); } } while(0)

std::ofstream logFile("log.txt"); // Глобальная переменная для файла статистики
// ==================== КОНСТАНТНАЯ ПАМЯТЬ ====================
#if (MAX_VALUE_SIZE * PARAMETR_SIZE < MAX_CONST)
__constant__ double dev_parametr_value[MAX_VALUE_SIZE * PARAMETR_SIZE];
#endif
// ==================== СТРУКТУРЫ ====================
struct alignas(16) HashEntry {
    unsigned long long key;
    double value;
};

struct PerformanceMetrics {
    float total_time_ms, kernel_time_ms, memory_time_ms, computeProbabilities_time_ms, antColonyOptimization_time_ms, updatePheromones_time_ms;
    float occupancy, memory_throughput_gbs;
    double min_fitness, max_fitness;
    int hash_hits, hash_misses;
};

// ==================== DEVICE ФУНКЦИИ ====================
__device__ __forceinline__ double go_x(const double* __restrict__ parametr, int start_index) {
    double sum = 0.0;
#pragma unroll
    for (int i = 1; i < PARAMETR_SIZE_ONE_X; ++i) {
        sum += parametr[start_index + i];
    }
    return parametr[start_index] * sum;
}

#if (SHAFFERA)
__device__ double BenchShafferaFunction(const double* __restrict__ parametr) {
    double r_squared = 0.0;
    const int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;

#pragma unroll 4
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X);
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
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
#pragma unroll 4
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
#pragma unroll 4
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
#pragma unroll 4
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
#pragma unroll 4
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
#pragma unroll 4
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
#pragma unroll 4
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
#pragma unroll 4
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
#pragma unroll 4
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
#pragma unroll 4
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
#pragma unroll 4
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum -= sin(x) * pow(sin((i + 1) * x * x / M_PI), 20);
    }
    return sum;
}
#endif

__device__ __forceinline__ unsigned long long generateKey(const int* __restrict__ agent_node, int bx) {
    unsigned long long key = 0;
#pragma unroll 8
    for (int i = 0; i < PARAMETR_SIZE; i++) {
        key = key * MAX_VALUE_SIZE + agent_node[bx * PARAMETR_SIZE + i];
    }
    return key;
}
__device__ __forceinline__ double getCachedResult(HashEntry* __restrict__ hashTable, const int* __restrict__ agent_node, int bx) {
    unsigned long long key = generateKey(agent_node, bx);
    unsigned long long idx = key % HASH_TABLE_SIZE;

#pragma unroll 4
    for (int i = 0; i < MAX_PROBES; i++) {
        if (hashTable[idx].key == key) return hashTable[idx].value;
        if (hashTable[idx].key == ZERO_HASH_RESULT) return -1.0;
        idx = (idx + (i + 1) * (i + 1)) % HASH_TABLE_SIZE;
    }
    return -1.0;
}
__device__ __forceinline__ void saveToCache(HashEntry* __restrict__ hashTable,  const int* __restrict__ agent_node,  int bx, double value) {
    unsigned long long key = generateKey(agent_node, bx);
    unsigned long long idx = key % HASH_TABLE_SIZE;

#pragma unroll 4
    for (int i = 0; i < MAX_PROBES; i++) {
        unsigned long long old = atomicCAS(&hashTable[idx].key, ZERO_HASH_RESULT, key);
        if (old == ZERO_HASH_RESULT || old == key) {
            hashTable[idx].value = value;
            return;
        }
        idx = (idx + (i + 1) * (i + 1)) % HASH_TABLE_SIZE;
    }
}
__device__ __forceinline__ void atomicMax(double* address, double value) {
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long old = *addr_as_ull, assumed;
    do {
        assumed = old;
        if (value > __longlong_as_double(assumed))
            old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(value));
    } while (assumed != old);
}
__device__ __forceinline__ void atomicMin(double* address, double value) {
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long old = *addr_as_ull, assumed;
    do {
        assumed = old;
        if (value < __longlong_as_double(assumed))
            old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(value));
    } while (assumed != old);
}

// ==================== ОПТИМИЗИРОВАННЫЕ ЯДРА ====================
__global__ void initializeHashTable(HashEntry* hashTable) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = tid; i < HASH_TABLE_SIZE; i += stride) {
        hashTable[i].key = ZERO_HASH_RESULT;
        hashTable[i].value = 0.0;
    }
}

__global__ void computeProbabilities(const double* __restrict__ pheromon, const double* __restrict__ kol_enter, double* __restrict__ norm_matrix_probability) {
    __shared__ double s_pheromon[BLOCK_SIZE * MAX_VALUE_SIZE];
    __shared__ double s_prob[BLOCK_SIZE * MAX_VALUE_SIZE];

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = tx + bx * blockDim.x;

    if (tid >= PARAMETR_SIZE) return;

    // Загрузка данных в shared memory
#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        s_pheromon[tx * MAX_VALUE_SIZE + i] = pheromon[tid * MAX_VALUE_SIZE + i];
    }
    __syncthreads();

    // Вычисление вероятностей
    const int start_idx = tx * MAX_VALUE_SIZE;
    double sum = 0.0, prob_sum = 0.0;

    // Нормализация
#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        sum += s_pheromon[start_idx + i];
    }
    double inv_sum = (sum != 0.0) ? 1.0 / sum : 1.0;

#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        double pheromon_norm = s_pheromon[start_idx + i] * inv_sum;
        double kol_val = kol_enter[tid * MAX_VALUE_SIZE + i];
        double prob = (kol_val != 0.0 && pheromon_norm != 0.0) ? 1.0 / kol_val + pheromon_norm : 0.0;
        s_prob[start_idx + i] = prob;
        prob_sum += prob;
    }

    // Нормализация
    double inv_prob_sum = (prob_sum != 0.0) ? 1.0 / prob_sum : 1.0;
    double cumulative = 0.0;

#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        cumulative += s_prob[start_idx + i] * inv_prob_sum;
        norm_matrix_probability[tid * MAX_VALUE_SIZE + i] = cumulative;
    }

    if (tx == 0) {
        norm_matrix_probability[tid * MAX_VALUE_SIZE + MAX_VALUE_SIZE - 1] = 1.0;
    }
}
#if (PARAMETR_SIZE < MAX_SHARED)
__global__ void antColonyOptimization(
#if (MAX_VALUE_SIZE * PARAMETR_SIZE > MAX_CONST)
    double* __restrict__ dev_parametr_value,
#endif
    double* __restrict__ norm_matrix_probability, int* __restrict__ agent_node, double* __restrict__ OF, HashEntry* __restrict__ hashTable, double* __restrict__ maxOf_dev, double* __restrict__ minOf_dev, int* __restrict__ kol_hash_fail) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ant_id = bx;

    if (ant_id >= ANT_SIZE) return;

    // Динамическая shared memory для параметров одного муравья
    extern __shared__ double s_agent[];

    curandState state;
    curand_init(clock64() + ant_id * blockDim.x + tx, 0, 0, &state);

    // Выбор пути муравья
#if (BIN_SEARCH)
#pragma unroll
    for (int param_idx = tx; param_idx < PARAMETR_SIZE; param_idx += blockDim.x) {
        double randomValue = curand_uniform(&state);
        int low = 0, high = MAX_VALUE_SIZE - 1;

        while (low <= high) {
            int mid = (low + high) >> 1;
            if (randomValue > norm_matrix_probability[param_idx * MAX_VALUE_SIZE + mid])
                low = mid + 1;
            else
                high = mid - 1;
        }
        agent_node[ant_id * PARAMETR_SIZE + param_idx] = low;
        s_agent[param_idx] = dev_parametr_value[param_idx * MAX_VALUE_SIZE + low];
    }
#else
#pragma unroll
    for (int param_idx = tx; param_idx < PARAMETR_SIZE; param_idx += blockDim.x) {
        double randomValue = curand_uniform(&state);
        int k = 0;
        while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[param_idx * MAX_VALUE_SIZE + k]) {
            k++;
        }
        agent_node[ant_id * PARAMETR_SIZE + param_idx] = k;
        s_agent[param_idx] = dev_parametr_value[param_idx * MAX_VALUE_SIZE + k];
    }
#endif

    __syncthreads();

    if (tx == 0) {
        double cached = getCachedResult(hashTable, agent_node, ant_id);

        if (cached < 0.0) {
            OF[ant_id] = BenchShafferaFunction(s_agent);
            saveToCache(hashTable, agent_node, ant_id, OF[ant_id]);
        }
        else {
            OF[ant_id] = cached;
            atomicAdd(kol_hash_fail, 1);
        }

        if (OF[ant_id] != ZERO_HASH_RESULT) {
            atomicMax(maxOf_dev, OF[ant_id]);
            atomicMin(minOf_dev, OF[ant_id]);
        }
    }
}
#else
// Более эффективная версия с pre-allocated буфером
__global__ void antColonyOptimization( double* __restrict__ dev_parametr_value, double* __restrict__ norm_matrix_probability, int* __restrict__ agent_node, double* __restrict__ OF, HashEntry* __restrict__ hashTable, double* __restrict__ maxOf_dev, double* __restrict__ minOf_dev, int* __restrict__ kol_hash_fail, double* __restrict__ global_params_buffer) {  // Pre-allocated buffer

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ant_id = bx;

    if (ant_id >= ANT_SIZE) return;

    curandState state;
    curand_init(clock64() + ant_id * blockDim.x + tx, 0, 0, &state);

    // Указатель на параметры этого агента в буфере
    double* agent_params = &global_params_buffer[ant_id * PARAMETR_SIZE];

    // Фаза 1: Выбор путей и непосредственное вычисление
    for (int param_idx = tx; param_idx < PARAMETR_SIZE; param_idx += blockDim.x) {
        double randomValue = curand_uniform(&state);
        int selected_index = 0;

#if (BIN_SEARCH)
        int low = 0, high = MAX_VALUE_SIZE - 1;
        while (low <= high) {
            int mid = (low + high) >> 1;
            if (randomValue > norm_matrix_probability[param_idx * MAX_VALUE_SIZE + mid])
                low = mid + 1;
            else
                high = mid - 1;
        }
        selected_index = low;
#else
        int k = 0;
        while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[param_idx * MAX_VALUE_SIZE + k]) {
            k++;
        }
        selected_index = k;
#endif

        agent_node[ant_id * PARAMETR_SIZE + param_idx] = selected_index;

        // Сохраняем выбранное значение параметра
        agent_params[param_idx] = dev_parametr_value[param_idx * MAX_VALUE_SIZE + selected_index];
    }

    __syncthreads();

    // Фаза 2: Вычисление фитнес-функции
    if (tx == 0) {
        double cached = getCachedResult(hashTable, agent_node, ant_id);

        if (cached < 0.0) {
            OF[ant_id] = BenchShafferaFunction(agent_params);
            saveToCache(hashTable, agent_node, ant_id, OF[ant_id]);
        }
        else {
            OF[ant_id] = cached;
            atomicAdd(kol_hash_fail, 1);
        }

        if (OF[ant_id] != ZERO_HASH_RESULT) {
            atomicMax(maxOf_dev, OF[ant_id]);
            atomicMin(minOf_dev, OF[ant_id]);
        }
    }
}
#endif

__global__ void updatePheromones(const double* __restrict__ OF, const int* __restrict__ agent_node, double* __restrict__ pheromon, double* __restrict__ kol_enter) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int param_idx = bx * blockDim.x + tx;

    if (param_idx >= PARAMETR_SIZE) return;

    // Испарение феромонов
#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        pheromon[param_idx * MAX_VALUE_SIZE + i] *= PARAMETR_RO;
    }

    // Добавление нового феромона
#pragma unroll 4
    for (int ant_id = 0; ant_id < ANT_SIZE; ant_id++) {
        if (OF[ant_id] != ZERO_HASH_RESULT) {
            int k = agent_node[ant_id * PARAMETR_SIZE + param_idx];
            int idx = param_idx * MAX_VALUE_SIZE + k;

            atomicAdd(&kol_enter[idx], 1.0);

#if (OPTIMIZE_MIN_1)
            double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[ant_id];
            if (delta > 0) atomicAdd(&pheromon[idx], PARAMETR_Q * delta);
#elif (OPTIMIZE_MIN_2)
            double of_val = (OF[ant_id] == 0) ? 1e-7 : OF[ant_id];
            atomicAdd(&pheromon[idx], PARAMETR_Q / of_val);
#elif (OPTIMIZE_MAX)
            atomicAdd(&pheromon[idx], PARAMETR_Q * OF[ant_id]);
#endif
        }
    }
}

// ==================== ОБЪЕДИНЕННОЕ ЯДРО ====================
__global__ void updateAndComputePheromones(const double* __restrict__ OF, const int* __restrict__ agent_node, double* __restrict__ pheromon, double* __restrict__ kol_enter, double* __restrict__ norm_matrix_probability) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int param_idx = bx * blockDim.x + tx;

    if (param_idx >= PARAMETR_SIZE) return;

    // Фаза 1: Обновление феромонов (updatePheromones)

    // Испарение феромонов
#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        pheromon[param_idx * MAX_VALUE_SIZE + i] *= PARAMETR_RO;
    }

    // Добавление нового феромона
#pragma unroll 4
        for (int ant_id = 0; ant_id < ANT_SIZE; ant_id++) {
            if (OF[ant_id] != ZERO_HASH_RESULT) {
                int k = agent_node[ant_id * PARAMETR_SIZE + param_idx];
                int idx = param_idx * MAX_VALUE_SIZE + k;

                atomicAdd(&kol_enter[idx], 1.0);

#if (OPTIMIZE_MIN_1)
                double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[ant_id];
                if (delta > 0) atomicAdd(&pheromon[idx], PARAMETR_Q * delta);
#elif (OPTIMIZE_MIN_2)
                double of_val = (OF[ant_id] == 0) ? 1e-7 : OF[ant_id];
                atomicAdd(&pheromon[idx], PARAMETR_Q / of_val);
#elif (OPTIMIZE_MAX)
                atomicAdd(&pheromon[idx], PARAMETR_Q * OF[ant_id]);
#endif
            }
        }
    __syncthreads();

    // Фаза 2: Вычисление вероятностей (computeProbabilities)

    // Используем shared memory для оптимизации
    __shared__ double s_pheromon[BLOCK_SIZE * MAX_VALUE_SIZE];
    __shared__ double s_prob[BLOCK_SIZE * MAX_VALUE_SIZE];

    // Загрузка данных в shared memory
#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        s_pheromon[tx * MAX_VALUE_SIZE + i] = pheromon[param_idx * MAX_VALUE_SIZE + i];
    }
    __syncthreads();

    // Вычисление вероятностей
    const int start_idx = tx * MAX_VALUE_SIZE;
    double sum = 0.0, prob_sum = 0.0;

    // Нормализация феромонов
#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        sum += s_pheromon[start_idx + i];
    }
    double inv_sum = (sum != 0.0) ? 1.0 / sum : 1.0;

    // Вычисление вероятностей с учетом количества посещений
#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        double pheromon_norm = s_pheromon[start_idx + i] * inv_sum;
        double kol_val = kol_enter[param_idx * MAX_VALUE_SIZE + i];
        double prob = (kol_val != 0.0 && pheromon_norm != 0.0) ? 1.0 / kol_val + pheromon_norm : 0.0;
        s_prob[start_idx + i] = prob;
        prob_sum += prob;
    }

    // Нормализация вероятностей
    double inv_prob_sum = (prob_sum != 0.0) ? 1.0 / prob_sum : 1.0;
    double cumulative = 0.0;

#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        cumulative += s_prob[start_idx + i] * inv_prob_sum;
        norm_matrix_probability[param_idx * MAX_VALUE_SIZE + i] = cumulative;
    }

    if (tx == 0) {
        norm_matrix_probability[param_idx * MAX_VALUE_SIZE + MAX_VALUE_SIZE - 1] = 1.0;
    }
}

// ==================== ПАРАЛЛЕЛЬНОЕ ЯДРО ====================
__global__ void parallelACOIteration(const double* __restrict__ OF, const int* __restrict__ agent_node, double* __restrict__ pheromon, double* __restrict__ kol_enter, double* __restrict__ norm_matrix_probability,
    // Параметры для antColonyOptimization
#if (MAX_VALUE_SIZE * PARAMETR_SIZE > MAX_CONST)
    double* __restrict__ dev_parametr_value,
#endif
    double* __restrict__ norm_matrix_probability_ant, int* __restrict__ agent_node_ant, double* __restrict__ OF_ant, HashEntry* __restrict__ hashTable, double* __restrict__ maxOf_dev, double* __restrict__ minOf_dev, int* __restrict__ kol_hash_fail
#if (PARAMETR_SIZE > MAX_SHARED)
    , double* __restrict__ dev_agent_params
#endif
) {

    // Определяем, какая часть блока выполняет какую задачу
    int total_threads = blockDim.x;
    int update_threads = total_threads / 2;  // Половина потоков для обновления феромонов
    int ant_threads = total_threads - update_threads;  // Оставшиеся для оптимизации муравьев

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    // ========== ЧАСТЬ 1: ОБНОВЛЕНИЕ ФЕРОМОНОВ И ВЕРОЯТНОСТЕЙ ==========
    if (tx < update_threads) {
        int update_tx = tx;
        int update_bx = bx;
        int param_idx = update_bx * update_threads + update_tx;

        if (param_idx >= PARAMETR_SIZE) return;

        // Испарение феромонов
#pragma unroll
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon[param_idx * MAX_VALUE_SIZE + i] *= PARAMETR_RO;
        }

        // Нормальное обновление с реальными агентами
#pragma unroll 4
        for (int ant_id = 0; ant_id < ANT_SIZE; ant_id++) {
            if (OF[ant_id] != ZERO_HASH_RESULT) {
                int k = agent_node[ant_id * PARAMETR_SIZE + param_idx];
                int idx = param_idx * MAX_VALUE_SIZE + k;
                atomicAdd(&kol_enter[idx], 1.0);
#if (OPTIMIZE_MIN_1)
                double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[ant_id];
                if (delta > 0) atomicAdd(&pheromon[idx], PARAMETR_Q * delta);
#elif (OPTIMIZE_MIN_2)
                double of_val = (OF[ant_id] == 0) ? 1e-7 : OF[ant_id];
                atomicAdd(&pheromon[idx], PARAMETR_Q / of_val);
#elif (OPTIMIZE_MAX)
                atomicAdd(&pheromon[idx], PARAMETR_Q * OF[ant_id]);
#endif
            }
        }

        __syncthreads();

        // Вычисление вероятностей
        __shared__ double s_pheromon[BLOCK_SIZE * MAX_VALUE_SIZE];
        __shared__ double s_prob[BLOCK_SIZE * MAX_VALUE_SIZE];

        // Загрузка данных в shared memory
#pragma unroll
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            s_pheromon[update_tx * MAX_VALUE_SIZE + i] = pheromon[param_idx * MAX_VALUE_SIZE + i];
        }
        __syncthreads();

        const int start_idx = update_tx * MAX_VALUE_SIZE;
        double sum = 0.0, prob_sum = 0.0;

        // Нормализация
#pragma unroll
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sum += s_pheromon[start_idx + i];
        }
        double inv_sum = (sum != 0.0) ? 1.0 / sum : 1.0;

#pragma unroll
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            double pheromon_norm = s_pheromon[start_idx + i] * inv_sum;
            double kol_val = kol_enter[param_idx * MAX_VALUE_SIZE + i];
            double prob = (kol_val != 0.0 && pheromon_norm != 0.0) ? 1.0 / kol_val + pheromon_norm : 0.0;
            s_prob[start_idx + i] = prob;
            prob_sum += prob;
        }

        // Нормализация вероятностей
        double inv_prob_sum = (prob_sum != 0.0) ? 1.0 / prob_sum : 1.0;
        double cumulative = 0.0;

#pragma unroll
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            cumulative += s_prob[start_idx + i] * inv_prob_sum;
            norm_matrix_probability[param_idx * MAX_VALUE_SIZE + i] = cumulative;
        }

        if (update_tx == 0) {
            norm_matrix_probability[param_idx * MAX_VALUE_SIZE + MAX_VALUE_SIZE - 1] = 1.0;
        }
    }

    // ========== ЧАСТЬ 2: ОПТИМИЗАЦИЯ МУРАВЬЕВ ==========
    else {
        int ant_tx = tx - update_threads;
        int ant_bx = bx;
        int ant_id = ant_bx;

        if (ant_id >= ANT_SIZE) return;

        // Динамическая shared memory для параметров муравья
        extern __shared__ double s_agent[];
        double* ant_params = &s_agent[ant_tx * PARAMETR_SIZE];

        curandState state;
        curand_init(clock64() + ant_id * ant_threads + ant_tx, 0, 0, &state);

        // Выбор пути муравья
#if (BIN_SEARCH)
#pragma unroll
        for (int param_idx = ant_tx; param_idx < PARAMETR_SIZE; param_idx += ant_threads) {
            double randomValue = curand_uniform(&state);
            int low = 0, high = MAX_VALUE_SIZE - 1;
            while (low <= high) {
                int mid = (low + high) >> 1;
                if (randomValue > norm_matrix_probability_ant[param_idx * MAX_VALUE_SIZE + mid])
                    low = mid + 1;
                else
                    high = mid - 1;
            }
            agent_node_ant[ant_id * PARAMETR_SIZE + param_idx] = low;
            ant_params[param_idx] = dev_parametr_value[param_idx * MAX_VALUE_SIZE + low];
        }
#else
#pragma unroll
        for (int param_idx = ant_tx; param_idx < PARAMETR_SIZE; param_idx += ant_threads) {
            double randomValue = curand_uniform(&state);
            int k = 0;
            while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability_ant[param_idx * MAX_VALUE_SIZE + k]) {
                k++;
            }
            agent_node_ant[ant_id * PARAMETR_SIZE + param_idx] = k;
            ant_params[param_idx] = dev_parametr_value[param_idx * MAX_VALUE_SIZE + k];
        }
#endif

        __syncthreads();

        // Вычисление функции приспособленности
        if (ant_tx == 0) {
            double cached = getCachedResult(hashTable, agent_node_ant, ant_id);

            if (cached < 0.0) {
                OF_ant[ant_id] = BenchShafferaFunction(ant_params);
                saveToCache(hashTable, agent_node_ant, ant_id, OF_ant[ant_id]);
            }
            else {
                OF_ant[ant_id] = cached;
                atomicAdd(kol_hash_fail, 1);
            }

            if (OF_ant[ant_id] != ZERO_HASH_RESULT) {
                atomicMax(maxOf_dev, OF_ant[ant_id]);
                atomicMin(minOf_dev, OF_ant[ant_id]);
            }
        }
    }
}

// ==================== GLOBAL CUDA РЕСУРСЫ ====================
static double* dev_pheromon = nullptr, * dev_kol_enter = nullptr, * dev_norm_matrix = nullptr;
static double* dev_OF = nullptr, * dev_max = nullptr, * dev_min = nullptr;
static int* dev_agent_node = nullptr, * dev_hash_fail = nullptr;
static HashEntry* dev_hashTable = nullptr;
static cudaStream_t compute_stream = nullptr;
#if (MAX_VALUE_SIZE * PARAMETR_SIZE > MAX_CONST)
static double* dev_parametr_value = nullptr;
#endif
#if (PARAMETR_SIZE > MAX_SHARED)
static double* dev_agent_params = nullptr;
#endif

// ==================== ФУНКЦИИ УПРАВЛЕНИЯ ПАМЯТЬЮ ====================
bool initialize_cuda_resources(const double* params, const double* pheromon, const double* kol_enter) {
    CUDA_CHECK(cudaStreamCreate(&compute_stream));

    const size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    const size_t ant_matrix_size = PARAMETR_SIZE * ANT_SIZE * sizeof(int);

    // Выделение памяти на устройстве
#if (MAX_VALUE_SIZE * PARAMETR_SIZE > MAX_CONST)
    CUDA_CHECK(cudaMallocAsync(&dev_parametr_value, matrix_size, compute_stream));
#endif
#if (PARAMETR_SIZE > MAX_SHARED)
    CUDA_CHECK(cudaMallocAsync(&dev_agent_params, PARAMETR_SIZE * ANT_SIZE * sizeof(double), compute_stream));
#endif
    CUDA_CHECK(cudaMallocAsync(&dev_pheromon, matrix_size, compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_kol_enter, matrix_size, compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_norm_matrix, matrix_size, compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_agent_node, ant_matrix_size, compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_OF, ANT_SIZE * sizeof(double), compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_max, sizeof(double), compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_min, sizeof(double), compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_hash_fail, sizeof(int), compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_hashTable, HASH_TABLE_SIZE * sizeof(HashEntry), compute_stream));

    // Копирование данных в постоянную память и на устройство
#if (MAX_VALUE_SIZE * PARAMETR_SIZE < MAX_CONST)
    CUDA_CHECK(cudaMemcpyToSymbolAsync(dev_parametr_value, params, matrix_size, 0, cudaMemcpyHostToDevice, compute_stream));
#else
    CUDA_CHECK(cudaMemcpyAsync(dev_parametr_value, params, matrix_size, cudaMemcpyHostToDevice, compute_stream));
#endif
    CUDA_CHECK(cudaMemcpyAsync(dev_pheromon, pheromon, matrix_size, cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_kol_enter, kol_enter, matrix_size, cudaMemcpyHostToDevice, compute_stream));

    // Инициализация хэш-таблицы
    int threads = BLOCK_SIZE;
    int blocks = (HASH_TABLE_SIZE + threads - 1) / threads;
    initializeHashTable << <blocks, threads, 0, compute_stream >> > (dev_hashTable);

    return cudaStreamSynchronize(compute_stream) == cudaSuccess;
}

void cleanup_cuda_resources() {
    if (compute_stream) {
        cudaStreamSynchronize(compute_stream);
        cudaStreamDestroy(compute_stream);
        compute_stream = nullptr;
    }

    if (dev_pheromon) cudaFree(dev_pheromon);
    if (dev_kol_enter) cudaFree(dev_kol_enter);
    if (dev_norm_matrix) cudaFree(dev_norm_matrix);
    if (dev_agent_node) cudaFree(dev_agent_node);
    if (dev_OF) cudaFree(dev_OF);
    if (dev_max) cudaFree(dev_max);
    if (dev_min) cudaFree(dev_min);
    if (dev_hash_fail) cudaFree(dev_hash_fail);
    if (dev_hashTable) cudaFree(dev_hashTable);
#if (MAX_VALUE_SIZE * PARAMETR_SIZE > MAX_CONST)
    if (dev_parametr_value) cudaFree(dev_parametr_value);
#endif
#if (PARAMETR_SIZE > MAX_SHARED)
    if (dev_agent_params) cudaFree(dev_agent_params);
#endif

    dev_pheromon = dev_kol_enter = dev_norm_matrix = dev_OF = dev_max = dev_min = nullptr;
    dev_agent_node = nullptr;
    dev_hash_fail = nullptr;
    dev_hashTable = nullptr;
#if (MAX_VALUE_SIZE * PARAMETR_SIZE > MAX_CONST)
    dev_parametr_value = nullptr;
#endif
#if (PARAMETR_SIZE > MAX_SHARED)
    dev_agent_params = nullptr;
#endif
}

// ==================== ОСНОВНАЯ ФУНКЦИЯ ВЫПОЛНЕНИЯ ====================
PerformanceMetrics run_aco_iterations(int num_iterations) {
    PerformanceMetrics metrics = { 0 };
    auto total_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t start, start_ant, start_update, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&start_ant));
    CUDA_CHECK(cudaEventCreate(&start_update));
    CUDA_CHECK(cudaEventCreate(&stop));

    float kernel_time = 0.0, computeProbabilities_time = 0.0, antColonyOptimization_time = 0.0, updatePheromones_time = 0.0;
    int threads_per_block = std::min(PARAMETR_SIZE, BLOCK_SIZE);
    int ant_blocks = ANT_SIZE;
    size_t sharedev_mem = PARAMETR_SIZE * sizeof(double);

    // Инициализация статистики
    double max_init = -1e9, min_init = 1e9;
    int fail_init = 0;

    CUDA_CHECK(cudaMemcpyAsync(dev_max, &max_init, sizeof(double), cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_min, &min_init, sizeof(double), cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_hash_fail, &fail_init, sizeof(int), cudaMemcpyHostToDevice, compute_stream));
    /*
    double* parametr_value = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    double* pheromon_value = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    double* kol_enter_value = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    double* norm_matrix_probability = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    int* ant_parametr = new int[PARAMETR_SIZE * ANT_SIZE];
    double* antOF = new double[ANT_SIZE];
    //std::cout << "ant_blocks=" << ant_blocks << " threads_per_block=" << threads_per_block << std::endl;
    */
    // Основной цикл итераций
    for (int iter = 0; iter < num_iterations; ++iter) {
        float iter_time;
        CUDA_CHECK(cudaEventRecord(start, compute_stream));

        computeProbabilities << <(PARAMETR_SIZE + (BLOCK_SIZE-1)) / BLOCK_SIZE, BLOCK_SIZE, 0, compute_stream >> > (dev_pheromon, dev_kol_enter, dev_norm_matrix); // 1. Вычисление вероятностей
        /*
        CUDA_CHECK(cudaMemcpy(norm_matrix_probability, dev_norm_matrix, MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pheromon_value, dev_pheromon, MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_enter_value, dev_kol_enter, MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
        for (int i = 0; i < PARAMETR_SIZE; ++i) {
            for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                std::cout << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
            }
            std::cout << std::endl; // Переход на новую строку
        }
        */

        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        computeProbabilities_time += iter_time;
        CUDA_CHECK(cudaEventRecord(start_ant, compute_stream));
#if (MAX_VALUE_SIZE * PARAMETR_SIZE < MAX_CONST)
        antColonyOptimization << <ant_blocks, threads_per_block, sharedev_mem, compute_stream >> > (dev_norm_matrix, dev_agent_node, dev_OF, dev_hashTable, dev_max, dev_min, dev_hash_fail); // 2. Оптимизация муравьями
#elif (PARAMETR_SIZE < MAX_SHARED)
        antColonyOptimization << <ant_blocks, threads_per_block, sharedev_mem, compute_stream >> > (dev_parametr_value, dev_norm_matrix, dev_agent_node, dev_OF, dev_hashTable, dev_max, dev_min, dev_hash_fail); // 2. Оптимизация муравьями
#else
        antColonyOptimization << <ant_blocks, threads_per_block, 0, compute_stream >> > (dev_parametr_value, dev_norm_matrix, dev_agent_node, dev_OF, dev_hashTable, dev_max, dev_min, dev_hash_fail, dev_agent_params);
#endif
        /*
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        
        CUDA_CHECK(cudaMemcpy(ant_parametr, dev_agent_node, PARAMETR_SIZE * ANT_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(antOF, dev_OF, ANT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
        for (int i = 0; i < ANT_SIZE; ++i) {
            for (int j = 0; j < PARAMETR_SIZE; ++j) {
                std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "; "; 
            }
            std::cout << "-> " << antOF[i] << std::endl;

        }
        */
        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start_ant, stop));
        antColonyOptimization_time += iter_time;
        CUDA_CHECK(cudaEventRecord(start_update, compute_stream));
        
        updatePheromones << <(PARAMETR_SIZE + (BLOCK_SIZE-1)) / BLOCK_SIZE, BLOCK_SIZE, 0, compute_stream >> > (dev_OF, dev_agent_node, dev_pheromon, dev_kol_enter); // 3. Обновление феромонов

        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        kernel_time += iter_time;
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start_update, stop));
        updatePheromones_time += iter_time;
    }

    // Сбор результатов


    double best_fitness;
    double low_fitness;
    int hash_fails;
    CUDA_CHECK(cudaMemcpyAsync(&best_fitness, dev_min, sizeof(double), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(&low_fitness, dev_max, sizeof(double), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(&hash_fails, dev_hash_fail, sizeof(int), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    metrics.min_fitness = best_fitness;
    metrics.max_fitness = low_fitness;
    metrics.hash_misses = hash_fails;
    metrics.hash_hits = num_iterations * ANT_SIZE - hash_fails;

    // Расчет дополнительных метрик
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
// ==================== ОБЪЕДИНЕННАЯ ФУНКЦИЯ ВЫПОЛНЕНИЯ ====================
PerformanceMetrics run_combined_iterations(int num_iterations, bool use_dummy_first_iteration = true) {
    PerformanceMetrics metrics = { 0 };
    auto total_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t start, start_ant, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&start_ant));
    CUDA_CHECK(cudaEventCreate(&stop));

    float kernel_time = 0.0, combined_time = 0.0, antColonyOptimization_time = 0.0;
    int threads_per_block = std::min(PARAMETR_SIZE, BLOCK_SIZE);
    int ant_blocks = ANT_SIZE;

    // Инициализация статистики
    double max_init = -1e9, min_init = 1e9;
    int fail_init = 0;

    CUDA_CHECK(cudaMemcpyAsync(dev_max, &max_init, sizeof(double), cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_min, &min_init, sizeof(double), cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_hash_fail, &fail_init, sizeof(int), cudaMemcpyHostToDevice, compute_stream));

    int* ant_parametr = new int[PARAMETR_SIZE * ANT_SIZE];
    double* antOF = new double[ANT_SIZE];
    //Заполнение начальными значениями ant_parametr_dev, antOFdev
    for (int i = 0; i < ANT_SIZE; ++i) {
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            ant_parametr[i * PARAMETR_SIZE + j] = 0;
        }
        antOF[i] = 1;
    }
    CUDA_CHECK(cudaMemcpy(dev_agent_node, ant_parametr, PARAMETR_SIZE * ANT_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_OF, antOF, ANT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    delete[] ant_parametr;
    delete[] antOF;

    // Основной цикл итераций
    for (int iter = 0; iter < num_iterations; ++iter) {
        float iter_time;
        CUDA_CHECK(cudaEventRecord(start, compute_stream));

        // Запуск объединенного ядра
        updateAndComputePheromones << <(PARAMETR_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, compute_stream >> > (dev_OF, dev_agent_node, dev_pheromon, dev_kol_enter, dev_norm_matrix );

        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        combined_time += iter_time;

        // Оптимизация муравьями (остается отдельно)
        CUDA_CHECK(cudaEventRecord(start_ant, compute_stream));

#if (MAX_VALUE_SIZE * PARAMETR_SIZE < MAX_CONST)
        antColonyOptimization << <ant_blocks, threads_per_block, PARAMETR_SIZE * sizeof(double), compute_stream >> > (dev_norm_matrix, dev_agent_node, dev_OF, dev_hashTable, dev_max, dev_min, dev_hash_fail);
#elif (PARAMETR_SIZE < MAX_SHARED)
        antColonyOptimization << <ant_blocks, threads_per_block, PARAMETR_SIZE * sizeof(double), compute_stream >> > (dev_parametr_value, dev_norm_matrix, dev_agent_node, dev_OF, dev_hashTable, dev_max, dev_min, dev_hash_fail);
#else
        antColonyOptimization << <ant_blocks, threads_per_block, 0, compute_stream >> > ( dev_parametr_value, dev_norm_matrix, dev_agent_node, dev_OF, dev_hashTable, dev_max, dev_min, dev_hash_fail, dev_agent_params);
#endif

        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start_ant, stop));
        antColonyOptimization_time += iter_time;
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        kernel_time += iter_time;
    }

    // Сбор результатов
    double best_fitness, low_fitness;
    int hash_fails;
    CUDA_CHECK(cudaMemcpyAsync(&best_fitness, dev_min, sizeof(double), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(&low_fitness, dev_max, sizeof(double), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(&hash_fails, dev_hash_fail, sizeof(int), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    metrics.min_fitness = best_fitness;
    metrics.max_fitness = low_fitness;
    metrics.hash_misses = hash_fails;
    metrics.hash_hits = num_iterations * ANT_SIZE - hash_fails;

    // Расчет дополнительных метрик
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int warp_size = prop.warpSize;
    int warps_per_block = (threads_per_block + warp_size - 1) / warp_size;
    int max_warps_per_sm = max_threads_per_sm / warp_size;
    metrics.occupancy = std::min(100.0f, (warps_per_block * 32.0f / max_warps_per_sm) * 100.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(start_ant);
    cudaEventDestroy(stop);

    auto total_end = std::chrono::high_resolution_clock::now();
    metrics.total_time_ms = std::chrono::duration<float, std::milli>(total_end - total_start).count();
    metrics.kernel_time_ms = kernel_time;
    metrics.computeProbabilities_time_ms = combined_time; // Объединенное время
    metrics.updatePheromones_time_ms = 0.0;     // Объединенное время
    metrics.antColonyOptimization_time_ms = antColonyOptimization_time;
    metrics.memory_time_ms = metrics.total_time_ms - kernel_time;

    // Расчет пропускной способности памяти
    size_t total_data_transferred = (MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double)) * 4 * num_iterations +
        (PARAMETR_SIZE * ANT_SIZE * sizeof(int)) * num_iterations +
        (ANT_SIZE * sizeof(double)) * num_iterations;
    metrics.memory_throughput_gbs = (total_data_transferred / (1024.0 * 1024.0 * 1024.0)) / (metrics.total_time_ms / 1000.0);

    return metrics;
}
// ==================== ФУНКЦИЯ ДЛЯ ПАРАЛЛЕЛЬНОГО ЗАПУСКА ====================
/*
PerformanceMetrics run_parallel_aco_iterations(int num_iterations, bool use_dummy_first_iteration = true) {
    PerformanceMetrics metrics = { 0 };
    auto total_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float parallel_time = 0.0;

    // Создаем временные буферы только если нужно
    double* dev_norm_matrix_ant = nullptr;
    int* dev_agent_node_ant = nullptr;
    double* dev_OF_ant = nullptr;
    double* dev_global_ant_params = nullptr;

    CUDA_CHECK(cudaMallocAsync(&dev_norm_matrix_ant, MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double), resources.compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_agent_node_ant, PARAMETR_SIZE * ANT_SIZE * sizeof(int), resources.compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_OF_ant, ANT_SIZE * sizeof(double), resources.compute_stream));

    // Выделяем global memory для параметров муравьев только если shared memory недостаточно
#if (PARAMETR_SIZE * int(BLOCK_SIZE / 2) * sizeof(double) > MAX_SHARED)
    CUDA_CHECK(cudaMallocAsync(&dev_global_ant_params, PARAMETR_SIZE * ANT_SIZE * sizeof(double), resources.compute_stream));
#endif

    // Копируем начальные данные
    CUDA_CHECK(cudaMemcpyAsync(dev_norm_matrix_ant, resources.dev_norm_matrix, MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double), cudaMemcpyDeviceToDevice, resources.compute_stream));

    // Основной цикл итераций
    for (int iter = 0; iter < num_iterations; ++iter) {
        float iter_time;
        CUDA_CHECK(cudaEventRecord(start, resources.compute_stream));

        bool use_dummy = (use_dummy_first_iteration && iter == 0);

        // Вычисляем конфигурацию запуска
        int update_blocks = (PARAMETR_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int ant_blocks = ANT_SIZE;
        int total_blocks = max(update_blocks, ant_blocks);

        // Увеличиваем количество потоков для параллельного выполнения
        int parallel_threads = BLOCK_SIZE * 2;

        // Вычисляем размер shared memory динамически
        size_t shared_mem_size = 0;
#if (BLOCK_SIZE * MAX_VALUE_SIZE * 2 * sizeof(double) <= MAX_SHARED)
        shared_mem_size += BLOCK_SIZE * MAX_VALUE_SIZE * 2 * sizeof(double); // Для феромонов
#endif
#if (PARAMETR_SIZE * (parallel_threads / 2) * sizeof(double) <= MAX_SHARED)
        shared_mem_size += PARAMETR_SIZE * (parallel_threads / 2) * sizeof(double); // Для параметров муравьев
#endif

        // Запуск параллельного ядра
        parallelACOIteration << <total_blocks, parallel_threads, shared_mem_size, resources.compute_stream >> > (
            // Основные параметры
            resources.dev_OF,
            resources.dev_agent_node,
            resources.dev_pheromon,
            resources.dev_kol_enter,
            resources.dev_norm_matrix,
            dev_norm_matrix_ant,
            dev_agent_node_ant,
            dev_OF_ant,
            resources.dev_hashTable,
            resources.dev_max,
            resources.dev_min,
            resources.dev_hash_fail,
            use_dummy,

            // Условные параметры
#if (MAX_VALUE_SIZE * PARAMETR_SIZE > MAX_CONST)
            resources.dev_parametr_value,
#endif
#if (PARAMETR_SIZE * (parallel_threads / 2) * sizeof(double) > MAX_SHARED)
            dev_global_ant_params
#endif
            );

        CUDA_CHECK(cudaEventRecord(stop, resources.compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(resources.compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        parallel_time += iter_time;

        // Копируем результаты для следующей итерации
        if (iter < num_iterations - 1) {
            CUDA_CHECK(cudaMemcpyAsync(dev_norm_matrix_ant, resources.dev_norm_matrix,
                MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double),
                cudaMemcpyDeviceToDevice, resources.compute_stream));
        }
    }

    // Очистка временных буферов
    auto safeFree = [&](auto& ptr) {
        if (ptr) {
            cudaFreeAsync(ptr, resources.compute_stream);
            ptr = nullptr;
        }
        };

    safeFree(dev_norm_matrix_ant);
    safeFree(dev_agent_node_ant);
    safeFree(dev_OF_ant);
    safeFree(dev_global_ant_params);

    // ... остальная часть функции (сбор метрик) без изменений
    return metrics;
}

*/
// ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
bool loadev_matrix_data(const std::string& filename, std::vector<double>& params, std::vector<double>& pheromones, std::vector<double>& visits) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    size_t total_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
    params.resize(total_size);
    pheromones.resize(total_size);
    visits.resize(total_size);

    for (size_t i = 0; i < total_size; ++i) {
        if (!(file >> params[i])) {
            std::cerr << "Error reading element " << i << std::endl;
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

// ==================== ФУНКЦИЯ ОЦЕНКИ ПАМЯТИ GPU ====================
double calculate_gpu_memory_usage() {
    double total_memory = 0.0;

    // Основные буферы (всегда выделяются)
    total_memory += MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double); // dev_pheromon
    total_memory += MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double); // dev_kol_enter
    total_memory += MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double); // dev_norm_matrix
    total_memory += PARAMETR_SIZE * ANT_SIZE * sizeof(int);          // dev_agent_node
    total_memory += ANT_SIZE * sizeof(double);                       // dev_OF
    total_memory += sizeof(double);                                  // dev_max
    total_memory += sizeof(double);                                  // dev_min
    total_memory += sizeof(int);                                     // dev_hash_fail
//    total_memory += HASH_TABLE_SIZE * sizeof(HashEntry);             // dev_hashTable

    // Условные буферы
#if (MAX_VALUE_SIZE * PARAMETR_SIZE > MAX_CONST)
    total_memory += MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double); // dev_parametr_value
#endif

#if (PARAMETR_SIZE > MAX_SHARED)
    total_memory += PARAMETR_SIZE * ANT_SIZE * sizeof(double);       // dev_agent_params
#endif

    return total_memory;
}
double calculate_max_parametr_size_in_gpu_memory() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        double max_parametr = 0.0;
        double GlobalMem = prop.totalGlobalMem;
        GlobalMem = GlobalMem - sizeof(double) * 2 - sizeof(int) - ANT_SIZE * sizeof(double) - HASH_TABLE_SIZE * sizeof(HashEntry);
#if (MAX_VALUE_SIZE * PARAMETR_SIZE < MAX_CONST)
        return GlobalMem / (MAX_VALUE_SIZE * sizeof(double) * 3 + ANT_SIZE * sizeof(int));
#elif (PARAMETR_SIZE < MAX_SHARED)
        return GlobalMem / (MAX_VALUE_SIZE * sizeof(double) * 4 + ANT_SIZE * sizeof(int));
#else
        return GlobalMem / (MAX_VALUE_SIZE * sizeof(double) * 4 + ANT_SIZE * sizeof(int) + ANT_SIZE * sizeof(double));
#endif
    }
}
void print_metrics(const PerformanceMetrics& metrics, char* str, int run_id) {
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
int print_information() {
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
        std::cout << "GPU Global Memory Usage: " << calculate_gpu_memory_usage() / (1024.0 * 1024.0) << " MB" << std::endl;
        logFile << "GPU Global Memory Usage: " << calculate_gpu_memory_usage() / (1024.0 * 1024.0) << " MB ";
        std::cout << "GPU Global Memory Hash Table: " << HASH_TABLE_SIZE * sizeof(HashEntry) / (1024.0 * 1024.0) << " MB" << std::endl;
        logFile << "GPU Global Memory Hash Table: " << HASH_TABLE_SIZE * sizeof(HashEntry) / (1024.0 * 1024.0) << " MB ";
        std::cout << "Max PARAMETR_SIZE in GPU Global Memory: " << int(calculate_max_parametr_size_in_gpu_memory()) << " " << std::endl;
        logFile << "Max PARAMETR_SIZE in GPU Global Memory: " << int(calculate_max_parametr_size_in_gpu_memory()) << " ";
        std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes (" << prop.sharedMemPerBlock / 1024 << " KB)" << std::endl;
        logFile << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes (" << prop.sharedMemPerBlock / 1024 << " KB) ";
        std::cout << "Shared memory per multiprocessor: " << prop.sharedMemPerMultiprocessor << " bytes (" << prop.sharedMemPerMultiprocessor / 1024 << " KB)" << std::endl;
        logFile << "Shared memory per multiprocessor: " << prop.sharedMemPerMultiprocessor << " bytes (" << prop.sharedMemPerMultiprocessor / 1024 << " KB) ";
        std::cout << "Data size in Shared memory per block: " << PARAMETR_SIZE * sizeof(double) << " bytes (" << PARAMETR_SIZE * sizeof(double) / 1024.0 << " KB)" << std::endl;
        logFile << "Data size in Shared memory per block: " << PARAMETR_SIZE * sizeof(double) << " bytes (" << PARAMETR_SIZE * sizeof(double) / 1024.0 << " KB) ";
        std::cout << "Constant Memory: " << prop.totalConstMem << " bytes" << std::endl;
        logFile << "Constant Memory: " << prop.totalConstMem << " bytes ";
        std::cout << "Parameter data size (in Constant Memory): " << MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double) << " bytes (" << MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double) / 1024.0 << " KB)" << std::endl;
        logFile << "Parameter data size (in Constant Memory): " << MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double) << " bytes (" << MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double) / 1024.0 << " KB) ";
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
    // Вывод информации о константах
    std::cout << "PARAMETR_SIZE: " << PARAMETR_SIZE << "; "
        << "MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << "; "
        << "PARAMETR_SIZE_ONE_X: " << PARAMETR_SIZE_ONE_X << "; "
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
        << "HASH_TABLE_SIZE: " << HASH_TABLE_SIZE << "; "
        << "ZERO_HASH_RESULT: " << ZERO_HASH_RESULT << "; "
        << "ZERO_HASH: " << ZERO_HASH << "; "
        << "MAX_PROBES: " << MAX_PROBES
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
    return 0;
}
void calculate_memory_limits() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    size_t shared_mem_per_block = prop.sharedMemPerBlock;
    size_t shared_mem_per_multiprocessor = prop.sharedMemPerMultiprocessor;
    int max_threads_per_block = prop.maxThreadsPerBlock;

    std::cout << "=== GPU Memory Limits ===" << std::endl;

    if (MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double) > 65536) {
        std::cout << "Using global memory for parameters (exceeds constant memory limit)" << std::endl;
    }
    // Расчет максимального PARAMETR_SIZE для shared memory
    std::cout << "Max PARAMETR_SIZE for shared memory: " << prop.sharedMemPerBlock / (sizeof(double)* MAX_VALUE_SIZE)
        << " (with " << prop.sharedMemPerBlock  << " bytes)" << std::endl;

    // Расчет с учетом других данных в shared memory
    size_t conservative_limit = (shared_mem_per_block * 0.8) / sizeof(double); // 80% от доступной памяти
    std::cout << "Conservative PARAMETR_SIZE limit: " << conservative_limit << std::endl;

    // Проверка текущих параметров
    size_t required_shared_mem = PARAMETR_SIZE * sizeof(double);
    std::cout << "Required shared memory for current PARAMETR_SIZE: "
        << required_shared_mem << " bytes" << std::endl;

    if (required_shared_mem > shared_mem_per_block) {
        std::cout << "WARNING: PARAMETR_SIZE too large for shared memory!" << std::endl;
        std::cout << "Consider reducing PARAMETR_SIZE or using global memory implementation" << std::endl;
    }
    else {
        std::cout << "Current PARAMETR_SIZE fits in shared memory" << std::endl;
    }
    std::cout << "=========================" << std::endl;
}
// ==================== MAIN ФУНКЦИЯ ====================
int main() {
    std::cout << "Initializing CUDA ACO Optimizer..." << std::endl;
    print_information();
    calculate_memory_limits();
    // Загрузка данных
    std::vector<double> params, pheromones, visits;
    if (!loadev_matrix_data(NAME_FILE_GRAPH, params, pheromones, visits)) {
        std::cerr << "Failed to load matrix data" << std::endl;
        return 1;
    }

    // Инициализация CUDA
    if (!initialize_cuda_resources(params.data(), pheromones.data(), visits.data())) {
        std::cerr << "CUDA resources initialization failed" << std::endl;
        return 1;
    }

    // Автоматическая очистка при выходе
    auto cleanup_guard = []() { cleanup_cuda_resources(); };
    // Прогрев
    std::cout << "Performing warmup runs..." << std::endl;
    for (int i = 0; i < KOL_PROGREV; ++i) {
        std::cout << "Warmup " << i << std::endl;
        run_aco_iterations(KOL_ITERATION);
    }

    // Основные запуски
    std::cout << "Starting main runs..." << std::endl;
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < KOL_PROGON_STATISTICS; ++i) {
        auto metrics = run_aco_iterations(KOL_ITERATION);
        print_metrics(metrics, "Original ACO", i);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    std::cout << "Original ACO total execution time: " << total_duration.count() << " ms" << std::endl;

    // Дополнительные запуски - объединенная версия
    std::cout << "\n=== Starting combined ACO runs ===" << std::endl;
    total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < KOL_PROGON_STATISTICS; ++i) {
        auto metrics = run_combined_iterations(KOL_ITERATION, true); // true - использовать фиктивных агентов на первой итерации
        print_metrics(metrics, "Combined Run", i);
    }

    total_end = std::chrono::high_resolution_clock::now();
    total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    std::cout << "Combined ACO total execution time: " << total_duration.count() << " ms" << std::endl;

    // Очистка ресурсов
    cleanup_guard();
    logFile.close();

    return 0;
}