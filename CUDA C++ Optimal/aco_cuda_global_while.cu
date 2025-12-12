#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "parametrs.h"

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(err) << std::endl; exit(EXIT_FAILURE); } } while(0)

std::ofstream logFile("log.txt"); // Глобальная переменная для файла статистики


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


__device__ __forceinline__ double go_x(const double* __restrict__ parametr, int start_index) {
    double sum = 0.0;
#pragma unroll
    for (int i = 1; i < SET_PARAMETR_SIZE_ONE_X; ++i) {
        sum += parametr[start_index + i];
    }
    return parametr[start_index] * sum;
}

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

__device__ __forceinline__ unsigned long long generateKey(const int* __restrict__ agent_node, int bx) {
    unsigned long long key = 0;
#pragma unroll 8
    for (int i = 0; i < PARAMETR_SIZE; i++) {
        key = key * MAX_VALUE_SIZE + agent_node[bx * PARAMETR_SIZE + i];
    }
    return key;
}
__device__ __forceinline__ double getCachedResult(HashEntry* __restrict__ hashTable, const int* __restrict__ agent_node, int bx) {
#if GO_HASH
    unsigned long long key = generateKey(agent_node, bx);
    unsigned long long idx = key % HASH_TABLE_SIZE;

#pragma unroll 4
    for (int i = 0; i < MAX_PROBES; i++) {
        if (hashTable[idx].key == key) return hashTable[idx].value;
        if (hashTable[idx].key == ZERO_HASH_RESULT) return -1.0;
        idx = (idx + (i + 1) * (i + 1)) % HASH_TABLE_SIZE;
    }
#endif
    return -1.0;
}
__device__ __forceinline__ void saveToCache(HashEntry* __restrict__ hashTable,  const int* __restrict__ agent_node,  int bx, double value) {
#if GO_HASH
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
#endif
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

// Испарение феромонов
__device__ void evaporatePheromones_dev(double* __restrict__ pheromon) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Обрабатываем несколько параметров на поток для лучшего использования кэша
    const int params_per_thread = 4; // Экспериментируйте с этим значением
    const int start_idx = tid * params_per_thread;

    for (int p = 0; p < params_per_thread; p++) {
        const int param_idx = start_idx + p;
        if (param_idx >= PARAMETR_SIZE) break;

        const int base_idx = param_idx * MAX_VALUE_SIZE;

        // Предварительная загрузка в L2 кэш
        __builtin_assume_aligned(&pheromon[base_idx], 32);

        // Явное развертывание для лучшей ILP (Instruction Level Parallelism)
        double v0 = pheromon[base_idx] * PARAMETR_RO;
        double v1 = pheromon[base_idx + 1] * PARAMETR_RO;
        double v2 = pheromon[base_idx + 2] * PARAMETR_RO;
        double v3 = pheromon[base_idx + 3] * PARAMETR_RO;
        double v4 = pheromon[base_idx + 4] * PARAMETR_RO;

        // Независимые операции - могут выполняться параллельно
        pheromon[base_idx] = v0;
        pheromon[base_idx + 1] = v1;
        pheromon[base_idx + 2] = v2;
        pheromon[base_idx + 3] = v3;
        pheromon[base_idx + 4] = v4;
    }
}
// Добавление феромонов
__device__ void depositPheromones_dev(const double* __restrict__ OF, const int* __restrict__ agent_node, double* __restrict__ pheromon, double* __restrict__ kol_enter) {
    const int param_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (param_idx >= PARAMETR_SIZE) return;

    const int warp_id = threadIdx.x / WARP_SIZE; //номер warp внутри блока (0-7) если блок 256
    const int lane_id = threadIdx.x % WARP_SIZE; //номер потока внутри warp (0-31) если блок 256
    const int warps_per_block = blockDim.x / WARP_SIZE; //сколько warps помещается в одном блоке

    // Каждый warp обрабатывает подмножество муравьев
    // Всего 500 муравьев / 32 = ~16 муравьев на поток в worst case
    // Но лучше разделить между warps в блоке

    // Накопление в регистрах для каждого возможного значения k (0-4)
    double pheromon_delta[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    double kol_enter_delta[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };

    // Разделение работы: каждый поток обрабатывает stride муравьев
    const int total_threads = gridDim.x * blockDim.x;
    const int threads_per_param = max(1, total_threads / PARAMETR_SIZE);
    const int thread_rank = (blockIdx.x * blockDim.x + threadIdx.x) % threads_per_param;

    // Обработка назначенных муравьев
    for (int ant_id = thread_rank; ant_id < ANT_SIZE; ant_id += threads_per_param) {
        if (OF[ant_id] != ZERO_HASH_RESULT) {
            const int k = agent_node[ant_id * PARAMETR_SIZE + param_idx];

            kol_enter_delta[k] += 1.0;

#if OPTIMIZE_MIN_1
            const double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[ant_id];
            if (delta > 0.0) {
                pheromon_delta[k] += PARAMETR_Q * delta;
            }
#elif OPTIMIZE_MIN_2
            const double of_val = fmax(OF[ant_id], 1e-7);
            pheromon_delta[k] += PARAMETR_Q / of_val;
#elif OPTIMIZE_MAX
            pheromon_delta[k] += PARAMETR_Q * OF[ant_id];
#endif
        }
    }

    // Редукция внутри warp для каждого k
    for (int k = 0; k < MAX_VALUE_SIZE; k++) {
        // Редукция с помощью warp shuffle
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            kol_enter_delta[k] += __shfl_down_sync(0xFFFFFFFF, kol_enter_delta[k], offset);
            pheromon_delta[k] += __shfl_down_sync(0xFFFFFFFF, pheromon_delta[k], offset);
        }
    }

    // Только первый поток в warp записывает результаты
    if (lane_id == 0) {
        const int base_idx = param_idx * MAX_VALUE_SIZE;

        for (int k = 0; k < MAX_VALUE_SIZE; k++) {
            const int idx = base_idx + k;

            if (kol_enter_delta[k] > 0.0) {
                atomicAdd(&kol_enter[idx], kol_enter_delta[k]);
            }

            if (pheromon_delta[k] > 0.0) {
                atomicAdd(&pheromon[idx], pheromon_delta[k]);
            }
        }
    }
}
__device__ void depositPheromones_dev_min_parametrs( const double* __restrict__ OF, const int* __restrict__ agent_node, double* __restrict__ pheromon, double* __restrict__ kol_enter) {
    const int param_idx = blockIdx.x;
    if (param_idx >= PARAMETR_SIZE) return;

    int tx = threadIdx.x;
    int total_threads = blockDim.x;  // Используем весь блок для одного параметра!

    const int lane_id = tx % WARP_SIZE;

    // Накопление в регистрах
    double pheromon_delta[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    double kol_enter_delta[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };

    // Все потоки блока обрабатывают этот параметр
    for (int ant_id = tx; ant_id < ANT_SIZE; ant_id += total_threads) {
        if (OF[ant_id] != ZERO_HASH_RESULT) {
            const int k = agent_node[ant_id * PARAMETR_SIZE + param_idx];

            kol_enter_delta[k] += 1.0;

#if OPTIMIZE_MIN_1
            const double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[ant_id];
            if (delta > 0.0) pheromon_delta[k] += PARAMETR_Q * delta;
#elif OPTIMIZE_MIN_2
            const double of_val = fmax(OF[ant_id], 1e-7);
            pheromon_delta[k] += PARAMETR_Q / of_val;
#elif OPTIMIZE_MAX
            pheromon_delta[k] += PARAMETR_Q * OF[ant_id];
#endif
        }
    }

    // Редукция ВО ВСЕМ БЛОКЕ (а не только в warp'е)
    // Используем shared memory для редукции всего блока

    __shared__ double shared_kol[WARP_SIZE][MAX_VALUE_SIZE];  // [warp_id][k]
    __shared__ double shared_pheromon[WARP_SIZE][MAX_VALUE_SIZE];

    // Каждый warp записывает свои накопления в shared memory
    if (lane_id < MAX_VALUE_SIZE) {
        int warp_id = tx / WARP_SIZE;
        shared_kol[warp_id][lane_id] = kol_enter_delta[lane_id];
        shared_pheromon[warp_id][lane_id] = pheromon_delta[lane_id];
    }
    __syncthreads();

    // Редукция между warp'ами (только первые 5 потоков)
    if (tx < MAX_VALUE_SIZE) {
        double total_kol = 0.0;
        double total_pheromon = 0.0;

        for (int warp = 0; warp < blockDim.x / WARP_SIZE; warp++) {
            total_kol += shared_kol[warp][tx];
            total_pheromon += shared_pheromon[warp][tx];
        }

        // Запись результатов
        const int idx = param_idx * MAX_VALUE_SIZE + tx;
        if (total_kol > 0.0) atomicAdd(&kol_enter[idx], total_kol);
        if (total_pheromon > 0.0) atomicAdd(&pheromon[idx], total_pheromon);
    }
}
__device__ void depositPheromones_dev_transposed(const double* __restrict__ OF, const int* __restrict__ agent_node, double* __restrict__ pheromon_transposed, double* __restrict__ kol_enter_transposed){
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    // Каждый блок обрабатывает определенное количество параметров
    const int PARAMS_PER_BLOCK = 8; // Экспериментируйте с этим значением
    const int start_param = bx * PARAMS_PER_BLOCK;
    const int end_param = min(start_param + PARAMS_PER_BLOCK, PARAMETR_SIZE);

    if (start_param >= PARAMETR_SIZE) return;

    // Каждый поток в блоке обрабатывает определенный value_idx
    const int value_idx = tx % MAX_VALUE_SIZE;
    const int thread_group = tx / MAX_VALUE_SIZE;
    const int total_threads_in_group = blockDim.x / MAX_VALUE_SIZE;

    // Разделяем работу по агентам между потоками в группе
    const int AGENTS_PER_ITERATION = 32;

    // Накопление в регистрах
    double pheromon_delta = 0.0;
    double kol_enter_delta = 0.0;

    // Обрабатываем все параметры, назначенные этому блоку
    for (int param_idx = start_param + thread_group;
        param_idx < end_param;
        param_idx += total_threads_in_group) {

        if (param_idx >= end_param) continue;

        // ОБРАБАТЫВАЕМ АГЕНТОВ БЛОКАМИ - ВСТАВЬТЕ ЗДЕСЬ
        for (int ant_base = 0; ant_base < ANT_SIZE; ant_base += AGENTS_PER_ITERATION) {
            int ant_end = min(ant_base + AGENTS_PER_ITERATION, ANT_SIZE);

#pragma unroll
            for (int ant_id = ant_base; ant_id < ant_end; ant_id++) {
                if (OF[ant_id] == ZERO_HASH_RESULT) continue;

                // Прямой доступ - теперь это коалесцированный!
                if (agent_node[ant_id * PARAMETR_SIZE + param_idx] == value_idx) {
                    kol_enter_delta += 1.0;

#if OPTIMIZE_MIN_1
                    const double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[ant_id];
                    if (delta > 0.0) pheromon_delta += PARAMETR_Q * delta;
#elif OPTIMIZE_MIN_2
                    const double of_val = fmax(OF[ant_id], 1e-7);
                    pheromon_delta += PARAMETR_Q / of_val;
#elif OPTIMIZE_MAX
                    pheromon_delta += PARAMETR_Q * OF[ant_id];
#endif
                }
            }
        }

        // Atomic операции для этого параметра
        if (kol_enter_delta > 0.0) {
            int kol_idx = value_idx * PARAMETR_SIZE + param_idx;
            atomicAdd(&kol_enter_transposed[kol_idx], kol_enter_delta);
        }

        if (pheromon_delta > 0.0) {
            int pheromon_idx = value_idx * PARAMETR_SIZE + param_idx;
            atomicAdd(&pheromon_transposed[pheromon_idx], pheromon_delta);
        }

        // Сброс накоплений для следующего параметра
        pheromon_delta = 0.0;
        kol_enter_delta = 0.0;
    }
}
// Нормализация матриц
__device__ void computeProbabilities_dev(const double* __restrict__ pheromon, const double* __restrict__ kol_enter, double* __restrict__ norm_matrix_probability) {
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
__device__ void computeProbabilities_dev_transposed(const double* __restrict__ pheromon_transposed, const double* __restrict__ kol_enter_transposed, double* __restrict__ norm_matrix_probability_transposed) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int parametr_id = tx + bx * blockDim.x;
    if (parametr_id >= PARAMETR_SIZE) return;

    __shared__ double s_pheromon[MAX_VALUE_SIZE][BLOCK_SIZE];
    __shared__ double s_prob[MAX_VALUE_SIZE][BLOCK_SIZE];

    // Каждый поток загружает ВСЕ 5 значений для своего параметра
#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        // Все потоки блока читают pheromon_transposed[i][parametr_id]. При разных parametr_id, но одинаковом i -> coalesced доступ!
        s_pheromon[i][tx] = pheromon_transposed[i * PARAMETR_SIZE + parametr_id];
    }
    __syncthreads();

    // 2. Вычисление суммы феромонов
    double sum = 0.0;
#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        sum += s_pheromon[i][tx];
    }
    double inv_sum = (sum != 0.0) ? 1.0 / sum : 1.0;

    // 3. Вычисление вероятностей
    double prob_sum = 0.0;
#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        double pheromon_norm = s_pheromon[i][tx] * inv_sum;
        double kol_val = kol_enter_transposed[i * PARAMETR_SIZE + parametr_id];
        double prob = (kol_val != 0.0 && pheromon_norm != 0.0) ? 1.0 / kol_val + pheromon_norm : 0.0;
        s_prob[i][tx] = prob;
        prob_sum += prob;
    }

    // 4. Нормализация и кумулятивная сумма
    double inv_prob_sum = (prob_sum != 0.0) ? 1.0 / prob_sum : 1.0;
    double cumulative = 0.0;

#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        cumulative += s_prob[i][tx] * inv_prob_sum;

        // Запись в транспонированную матрицу. Все потоки записывают norm_matrix_probability_transposed[i][parametr_id]
        norm_matrix_probability_transposed[i * PARAMETR_SIZE + parametr_id] = cumulative;
    }

    // 5. Последний элемент = 1.0
    if (tx == 0) {
        norm_matrix_probability_transposed[(MAX_VALUE_SIZE - 1) * PARAMETR_SIZE + parametr_id] = 1.0;
    }
}

// Вычисление путей муравьев-агентов
__device__ void antColonyOptimization_dev(double* __restrict__ dev_parametr_value, double* __restrict__ norm_matrix_probability, int* __restrict__ agent_node, double* __restrict__ OF, HashEntry* __restrict__ hashTable, double* __restrict__ maxOf_dev, double* __restrict__ minOf_dev, int* __restrict__ kol_hash_fail, double* __restrict__ global_params_buffer) {
     //СТАРАЯ НЕ ТРАНСПОНИРОВАННАЯ ВЕРСИЯ
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

        int k = 0;
        while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[param_idx * MAX_VALUE_SIZE + k]) {
            k++;
        }
        selected_index = k;

        agent_node[ant_id * PARAMETR_SIZE + param_idx] = selected_index;

        // Сохраняем выбранное значение параметра
        agent_params[param_idx] = dev_parametr_value[param_idx * MAX_VALUE_SIZE + selected_index];
    }

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
__device__ void antColonyOptimization_dev_transposed(double* __restrict__ dev_parametr_value_transposed, double* __restrict__ norm_matrix_probability_transposed, int* __restrict__ agent_node, double* __restrict__ OF, HashEntry* __restrict__ hashTable, double* __restrict__ maxOf_dev, double* __restrict__ minOf_dev, int* __restrict__ kol_hash_fail, double* __restrict__ global_params_buffer) {
    // ВСЕГДА используем 1 поток = 1 агент для простоты и надежности
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= ANT_SIZE) return;

    // Инициализация генератора случайных чисел
    curandState state;
    curand_init(clock64() + tid, 0, 0, &state);

    // Указатели на данные агента
    int* agent_node_ptr = &agent_node[tid * PARAMETR_SIZE];
    double* agent_params = &global_params_buffer[tid * PARAMETR_SIZE];

    // ============================================================================
    // АВТОМАТИЧЕСКИЙ ВЫБОР СТРАТЕГИИ НА ОСНОВЕ PARAMETR_SIZE
    // ============================================================================

    if (PARAMETR_SIZE <= 1024) {
        // ========================================================================
        // ДЛЯ МАЛЫХ PARAMETR_SIZE: прямой доступ
        // ========================================================================

        for (int i = 0; i < PARAMETR_SIZE; i++) {
            double randomValue = curand_uniform(&state);

            double p0 = norm_matrix_probability_transposed[0 * PARAMETR_SIZE + i];
            double p1 = norm_matrix_probability_transposed[1 * PARAMETR_SIZE + i];
            double p2 = norm_matrix_probability_transposed[2 * PARAMETR_SIZE + i];
            double p3 = norm_matrix_probability_transposed[3 * PARAMETR_SIZE + i];
            double p4 = norm_matrix_probability_transposed[4 * PARAMETR_SIZE + i];

            int selected_index = 0;
            if (randomValue > p0) {
                if (randomValue > p2) {
                    if (randomValue > p3) {
                        selected_index = 4;
                    }
                    else {
                        selected_index = 3;
                    }
                }
                else {
                    if (randomValue > p1) {
                        selected_index = 2;
                    }
                    else {
                        selected_index = 1;
                    }
                }
            }

            agent_node_ptr[i] = selected_index;
            agent_params[i] = dev_parametr_value_transposed[selected_index * PARAMETR_SIZE + i];
        }

    }
    else {
        // ========================================================================
        // ДЛЯ БОЛЬШИХ PARAMETR_SIZE: тайлинг внутри одного потока
        // ========================================================================

        for (int tile_start = 0; tile_start < PARAMETR_SIZE; tile_start += TILE_SIZE) {
            int tile_end = tile_start + TILE_SIZE;
            if (tile_end > PARAMETR_SIZE) tile_end = PARAMETR_SIZE;

            // Обрабатываем тайл
            for (int i = tile_start; i < tile_end; i++) {
                double randomValue = curand_uniform(&state);

                double p0 = norm_matrix_probability_transposed[0 * PARAMETR_SIZE + i];
                double p1 = norm_matrix_probability_transposed[1 * PARAMETR_SIZE + i];
                double p2 = norm_matrix_probability_transposed[2 * PARAMETR_SIZE + i];
                double p3 = norm_matrix_probability_transposed[3 * PARAMETR_SIZE + i];
                double p4 = norm_matrix_probability_transposed[4 * PARAMETR_SIZE + i];

                int selected_index = 0;
                if (randomValue > p0) {
                    if (randomValue > p2) {
                        if (randomValue > p3) {
                            selected_index = 4;
                        }
                        else {
                            selected_index = 3;
                        }
                    }
                    else {
                        if (randomValue > p1) {
                            selected_index = 2;
                        }
                        else {
                            selected_index = 1;
                        }
                    }
                }

                agent_node_ptr[i] = selected_index;
                agent_params[i] = dev_parametr_value_transposed[selected_index * PARAMETR_SIZE + i];
            }
        }
    }

    // ============================================================================
    // ВЫЧИСЛЕНИЕ ЦЕЛЕВОЙ ФУНКЦИИ
    // ============================================================================

    double cached = getCachedResult(hashTable, agent_node_ptr, tid);

    if (cached < 0.0) {
        OF[tid] = BenchShafferaFunction(agent_params);
        saveToCache(hashTable, agent_node_ptr, tid, OF[tid]);
    }
    else {
        OF[tid] = cached;
        atomicAdd(kol_hash_fail, 1);
    }

    if (OF[tid] != ZERO_HASH_RESULT) {
        atomicMax(maxOf_dev, OF[tid]);
        atomicMin(minOf_dev, OF[tid]);
    }
}



__global__ void antColonyOptimization(const bool go_transposed, double* __restrict__ dev_parametr_value, double* __restrict__ norm_matrix_probability, int* __restrict__ agent_node, double* __restrict__ OF,  HashEntry* __restrict__ hashTable,  double* __restrict__ maxOf_dev, double* __restrict__ minOf_dev, int* __restrict__ kol_hash_fail, double* __restrict__ global_params_buffer) {  // Pre-allocated buffer
    if (go_transposed) {
        antColonyOptimization_dev_transposed(dev_parametr_value, norm_matrix_probability, agent_node, OF, hashTable, maxOf_dev, minOf_dev, kol_hash_fail, global_params_buffer);
    }
    else {
        antColonyOptimization_dev(dev_parametr_value, norm_matrix_probability, agent_node, OF, hashTable, maxOf_dev, minOf_dev, kol_hash_fail, global_params_buffer);
    }
}
__global__ void computeProbabilities(const bool go_transposed, const double* __restrict__ pheromon, const double* __restrict__ kol_enter, double* __restrict__ norm_matrix_probability) {
    if (go_transposed) {
        computeProbabilities_dev_transposed(pheromon, kol_enter, norm_matrix_probability);
    }
    else {
        computeProbabilities_dev(pheromon, kol_enter, norm_matrix_probability);
    }
}
__global__ void updatePheromones(const bool go_min_parametrs, const bool go_transposed, const double* __restrict__ OF, const int* __restrict__ agent_node, double* __restrict__ pheromon, double* __restrict__ kol_enter) {
    evaporatePheromones_dev(pheromon);
    if (go_transposed) {
        depositPheromones_dev_transposed(OF, agent_node, pheromon, kol_enter);
    }
    else {
        if (go_min_parametrs) {
            depositPheromones_dev_min_parametrs(OF, agent_node, pheromon, kol_enter);
        }
        else
        {
            depositPheromones_dev(OF, agent_node, pheromon, kol_enter);
        }
    }
}
__global__ void evaporatePheromones(double* __restrict__ pheromon) {
    evaporatePheromones_dev(pheromon);
}
__global__ void depositPheromones(const bool go_min_parametrs, const bool go_transposed, const double* __restrict__ OF, const int* __restrict__ agent_node, double* __restrict__ pheromon, double* __restrict__ kol_enter) {
    if (go_transposed) {
        depositPheromones_dev_transposed(OF, agent_node, pheromon, kol_enter);
    }
    else {
        if (go_min_parametrs) {
            depositPheromones_dev_min_parametrs(OF, agent_node, pheromon, kol_enter);
        }
        else
        {
            depositPheromones_dev(OF, agent_node, pheromon, kol_enter);
        }
    }
}

// ==================== ОБЪЕДИНЕННОЕ ЯДРО ====================
// Фаза 1+3: Работа с матрицами
__global__ void updateAndComputePheromones(const bool go_transposed, const double* __restrict__ OF, const int* __restrict__ agent_node, double* __restrict__ pheromon, double* __restrict__ kol_enter, double* __restrict__ norm_matrix_probability) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int param_idx = bx * blockDim.x + tx;

    if (param_idx >= PARAMETR_SIZE) return;

    // Испарение феромонов
    // Обрабатываем несколько параметров на поток для лучшего использования кэша 32 параметра × 5 значений = 160 операций умножения на поток
    const int params_per_thread = WARP_SIZE;
    const int start_idx = param_idx * params_per_thread;

    for (int p = 0; p < params_per_thread; p++) {
        const int current_param_idx = start_idx + p;
        if (current_param_idx >= PARAMETR_SIZE) break;
        const int base_idx = current_param_idx * MAX_VALUE_SIZE;

        // Сообщение компилятору о выравненных значениях для более удобной выгрузки результатов (пока разница не заметна)
        //__builtin_assume_aligned(&pheromon[base_idx], WARP_SIZE);

        // Явное развертывание для лучшей ILP (Instruction Level Parallelism)
        double v0 = pheromon[base_idx] * PARAMETR_RO;
        double v1 = pheromon[base_idx + 1] * PARAMETR_RO;
        double v2 = pheromon[base_idx + 2] * PARAMETR_RO;
        double v3 = pheromon[base_idx + 3] * PARAMETR_RO;
        double v4 = pheromon[base_idx + 4] * PARAMETR_RO;

        pheromon[base_idx] = v0;
        pheromon[base_idx + 1] = v1;
        pheromon[base_idx + 2] = v2;
        pheromon[base_idx + 3] = v3;
        pheromon[base_idx + 4] = v4;
    }

    // Добавление нового феромона
    const int warp_id = threadIdx.x / WARP_SIZE; // Номер warp в блоке (0-7 для блока 256)
    const int lane_id = threadIdx.x % WARP_SIZE; // Номер потока в warp (0-31)
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int total_threads = gridDim.x * blockDim.x;
    const int threads_per_param = max(1, total_threads / PARAMETR_SIZE);
    const int thread_rank = (blockIdx.x * blockDim.x + threadIdx.x) % threads_per_param;

    // Каждый warp обрабатывает подмножество муравьев. Всего 500 муравьев / 32 = ~16 муравьев на поток в worst case

    double pheromon_delta[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    double kol_enter_delta[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };

    // Обработка назначенных муравьев Поток 0: муравьи 0, X, 2X, 3X, Поток 1: муравьи 1, X + 1, 2X + 1, 3X + 1, ...
    for (int ant_id = thread_rank; ant_id < ANT_SIZE; ant_id += threads_per_param) {
        if (OF[ant_id] != ZERO_HASH_RESULT) {
            const int k = agent_node[ant_id * PARAMETR_SIZE + param_idx];

            kol_enter_delta[k] += 1.0;

#if OPTIMIZE_MIN_1
            const double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[ant_id];
            if (delta > 0.0) {
                pheromon_delta[k] += PARAMETR_Q * delta;
            }
#elif OPTIMIZE_MIN_2
            const double of_val = fmax(OF[ant_id], 1e-7);
            pheromon_delta[k] += PARAMETR_Q / of_val;
#elif OPTIMIZE_MAX
            pheromon_delta[k] += PARAMETR_Q * OF[ant_id];
#endif
        }
    }

    // Редукция внутри warp для каждого k
    for (int k = 0; k < MAX_VALUE_SIZE; k++) {
        // Редукция с помощью warp shuffle для kol_enter_delta и pheromon_delta:
        // Шаг 1 (offset=16): Поток 0: свои_данные + данные_потока_16 Поток 1 : свои_данные + данные_потока_17
        // Шаг 2 (offset = 8) : Поток 0 : (0 + 16) + (8 + 24)
        // Итог : Поток 0 содержит сумму всех 32 потоков!
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            kol_enter_delta[k] += __shfl_down_sync(0xFFFFFFFF, kol_enter_delta[k], offset);
            pheromon_delta[k] += __shfl_down_sync(0xFFFFFFFF, pheromon_delta[k], offset);
        }
    }
    // Только первый поток в warp записывает результаты
    if (lane_id == 0) {
        const int base_idx = param_idx * MAX_VALUE_SIZE;

        for (int k = 0; k < MAX_VALUE_SIZE; k++) {
            const int idx = base_idx + k;

            if (kol_enter_delta[k] > 0.0) {
                atomicAdd(&kol_enter[idx], kol_enter_delta[k]);
            }

            if (pheromon_delta[k] > 0.0) {
                atomicAdd(&pheromon[idx], pheromon_delta[k]);
            }
        }
    }

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
    const int start_idx_old = tx * MAX_VALUE_SIZE;
    double sum = 0.0, prob_sum = 0.0;

    // Нормализация феромонов
#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        sum += s_pheromon[start_idx_old + i];
    }
    double inv_sum = (sum != 0.0) ? 1.0 / sum : 1.0;

    // Вычисление вероятностей с учетом количества посещений
#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        double pheromon_norm = s_pheromon[start_idx_old + i] * inv_sum;
        double kol_val = kol_enter[param_idx * MAX_VALUE_SIZE + i];
        double prob = (kol_val != 0.0 && pheromon_norm != 0.0) ? 1.0 / kol_val + pheromon_norm : 0.0;
        s_prob[start_idx_old + i] = prob;
        prob_sum += prob;
    }

    // Нормализация вероятностей
    double inv_prob_sum = (prob_sum != 0.0) ? 1.0 / prob_sum : 1.0;
    double cumulative = 0.0;

#pragma unroll
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        cumulative += s_prob[start_idx_old + i] * inv_prob_sum;
        norm_matrix_probability[param_idx * MAX_VALUE_SIZE + i] = cumulative;
    }

    if (tx == 0) {
        norm_matrix_probability[param_idx * MAX_VALUE_SIZE + MAX_VALUE_SIZE - 1] = 1.0;
    }
}

// ==================== ПАРАЛЛЕЛЬНОЕ ЯДРО ====================
__global__ void parallelACOIteration(const double* __restrict__ OF, const int* __restrict__ agent_node, double* __restrict__ pheromon, double* __restrict__ kol_enter, double* __restrict__ norm_matrix_probability, double* __restrict__ dev_parametr_value, double* __restrict__ norm_matrix_probability_ant, int* __restrict__ agent_node_ant, double* __restrict__ OF_ant,  HashEntry* __restrict__ hashTable,  double* __restrict__ maxOf_dev, double* __restrict__ minOf_dev, int* __restrict__ kol_hash_fail, double* __restrict__ dev_agent_params) {

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
#if (GO_HASH)
            double cached = getCachedResult(hashTable, agent_node_ant, ant_id);

            if (cached < 0.0) {
                OF_ant[ant_id] = BenchShafferaFunction(ant_params);
                saveToCache(hashTable, agent_node_ant, ant_id, OF_ant[ant_id]);
            }
            else {
                OF_ant[ant_id] = cached;
                atomicAdd(kol_hash_fail, 1);
            }
#else
            OF_ant[ant_id] = BenchShafferaFunction(ant_params);
#endif
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
static double* dev_parametr_value = nullptr;
static double* dev_agent_params = nullptr;

// ==================== ФУНКЦИИ УПРАВЛЕНИЯ ПАМЯТЬЮ ====================
bool initialize_cuda_resources(const double* params, const double* pheromon, const double* kol_enter) {
    CUDA_CHECK(cudaStreamCreate(&compute_stream));

    const size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    const size_t ant_matrix_size = PARAMETR_SIZE * ANT_SIZE * sizeof(int);

    // Выделение памяти на устройстве
    CUDA_CHECK(cudaMallocAsync(&dev_parametr_value, matrix_size, compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_agent_params, PARAMETR_SIZE * ANT_SIZE * sizeof(double), compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_pheromon, matrix_size, compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_kol_enter, matrix_size, compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_norm_matrix, matrix_size, compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_agent_node, ant_matrix_size, compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_OF, ANT_SIZE * sizeof(double), compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_max, sizeof(double), compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_min, sizeof(double), compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_hash_fail, sizeof(int), compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_hashTable, HASH_TABLE_SIZE * sizeof(HashEntry), compute_stream));

    CUDA_CHECK(cudaMemcpyAsync(dev_parametr_value, params, matrix_size, cudaMemcpyHostToDevice, compute_stream));
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
    if (dev_parametr_value) cudaFree(dev_parametr_value);
    if (dev_agent_params) cudaFree(dev_agent_params);

    dev_pheromon = dev_kol_enter = dev_norm_matrix = dev_OF = dev_max = dev_min = nullptr;
    dev_agent_node = nullptr;
    dev_hash_fail = nullptr;
    dev_hashTable = nullptr;
    dev_parametr_value = nullptr;
    dev_agent_params = nullptr;
}
// ==================== 4-Х ЭТАПНАЯ ФУНКЦИЯ ВЫПОЛНЕНИЯ ====================
PerformanceMetrics run_aco_iterations_4function(const bool go_min_parametrs, const bool go_transposed, int num_iterations) {
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
#if (GO_HASH)
    CUDA_CHECK(cudaMemcpyAsync(dev_hash_fail, &fail_init, sizeof(int), cudaMemcpyHostToDevice, compute_stream));
#endif // (GO_HASH)
    // Основной цикл итераций
    for (int iter = 0; iter < num_iterations; ++iter) {
        float iter_time;
        CUDA_CHECK(cudaEventRecord(start, compute_stream));

        computeProbabilities << <(PARAMETR_SIZE + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE, 0, compute_stream >> > (go_transposed, dev_pheromon, dev_kol_enter, dev_norm_matrix); // 1. Вычисление вероятностей

        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        computeProbabilities_time += iter_time;
        CUDA_CHECK(cudaEventRecord(start_ant, compute_stream));

        antColonyOptimization << <ant_blocks, threads_per_block, 0, compute_stream >> > (go_transposed,dev_parametr_value, dev_norm_matrix, dev_agent_node, dev_OF, dev_hashTable, dev_max, dev_min, dev_hash_fail, dev_agent_params);
        
        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start_ant, stop));
        antColonyOptimization_time += iter_time;
        CUDA_CHECK(cudaEventRecord(start_update, compute_stream));

        evaporatePheromones << <(PARAMETR_SIZE + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE, 0, compute_stream >> > (dev_pheromon);
        depositPheromones << <(PARAMETR_SIZE + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE, 0, compute_stream >> > (go_min_parametrs, go_transposed, dev_OF, dev_agent_node, dev_pheromon, dev_kol_enter);

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
#if (GO_HASH)
    CUDA_CHECK(cudaMemcpyAsync(&hash_fails, dev_hash_fail, sizeof(int), cudaMemcpyDeviceToHost, compute_stream));
#endif
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
// ==================== ОСНОВНАЯ ФУНКЦИЯ ВЫПОЛНЕНИЯ 3 ЭТАПА ====================
PerformanceMetrics run_aco_iterations(const bool go_min_parametrs, const bool go_transposed, int num_iterations) {
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
#if (GO_HASH)
    CUDA_CHECK(cudaMemcpyAsync(dev_hash_fail, &fail_init, sizeof(int), cudaMemcpyHostToDevice, compute_stream));
#endif // (GO_HASH)

    double* parametr_value = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    double* pheromon_value = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    double* kol_enter_value = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    double* norm_matrix_probability = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    int* ant_parametr = new int[PARAMETR_SIZE * ANT_SIZE];
    double* antOF = new double[ANT_SIZE];
    //std::cout << "ant_blocks=" << ant_blocks << " threads_per_block=" << threads_per_block << std::endl;

    // Основной цикл итераций
    for (int iter = 0; iter < num_iterations; ++iter) {
        float iter_time;
        CUDA_CHECK(cudaEventRecord(start, compute_stream));

        computeProbabilities << <(PARAMETR_SIZE + (BLOCK_SIZE-1)) / BLOCK_SIZE, BLOCK_SIZE, 0, compute_stream >> > (go_transposed, dev_pheromon, dev_kol_enter, dev_norm_matrix); // 1. Вычисление вероятностей
        /*
        CUDA_CHECK(cudaMemcpy(norm_matrix_probability, dev_norm_matrix, MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pheromon_value, dev_pheromon, MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_enter_value, dev_kol_enter, MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
                std::cout << "(" << pheromon_value[i * PARAMETR_SIZE + j] << ", " << kol_enter_value[i * PARAMETR_SIZE + j] << "-> " << norm_matrix_probability[i * PARAMETR_SIZE + j] << ") "; // Индексируем элементы
            }
            std::cout << std::endl; // Переход на новую строку
        }
        */

        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        computeProbabilities_time += iter_time;
        CUDA_CHECK(cudaEventRecord(start_ant, compute_stream));

        antColonyOptimization << <ant_blocks, threads_per_block, 0, compute_stream >> > (go_transposed, dev_parametr_value, dev_norm_matrix, dev_agent_node, dev_OF, dev_hashTable, dev_max, dev_min, dev_hash_fail, dev_agent_params);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        /*
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
        
        updatePheromones << <(PARAMETR_SIZE + (BLOCK_SIZE-1)) / BLOCK_SIZE, BLOCK_SIZE, 0, compute_stream >> > (go_min_parametrs, go_transposed, dev_OF, dev_agent_node, dev_pheromon, dev_kol_enter); // 3. Обновление феромонов

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
#if (GO_HASH)
    CUDA_CHECK(cudaMemcpyAsync(&hash_fails, dev_hash_fail, sizeof(int), cudaMemcpyDeviceToHost, compute_stream));
#endif
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
// ==================== ОБЪЕДИНЕННАЯ ФУНКЦИЯ ВЫПОЛНЕНИЯ 2 ЭТАПА ====================
PerformanceMetrics run_combined_iterations(const bool go_min_parametrs, const bool go_transposed, int num_iterations, bool use_dummy_first_iteration = true) {
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
#if (GO_HASH)
    CUDA_CHECK(cudaMemcpyAsync(dev_hash_fail, &fail_init, sizeof(int), cudaMemcpyHostToDevice, compute_stream));
#endif

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
        updateAndComputePheromones << <(PARAMETR_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, compute_stream >> > (go_transposed, dev_OF, dev_agent_node, dev_pheromon, dev_kol_enter, dev_norm_matrix );

        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        combined_time += iter_time;

        // Оптимизация муравьями (остается отдельно)
        CUDA_CHECK(cudaEventRecord(start_ant, compute_stream));

        antColonyOptimization << <ant_blocks, threads_per_block, 0, compute_stream >> > (go_transposed, dev_parametr_value, dev_norm_matrix, dev_agent_node, dev_OF, dev_hashTable, dev_max, dev_min, dev_hash_fail, dev_agent_params);

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
#if (GO_HASH)
    CUDA_CHECK(cudaMemcpyAsync(&hash_fails, dev_hash_fail, sizeof(int), cudaMemcpyDeviceToHost, compute_stream));
#endif
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
// ==================== ФУНКЦИЯ ДЛЯ ЗАПУСКА В 1 ЭТАП ====================

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
// Функция транспонирования матрицы
void transposeMatrix(const std::vector<double>& source, std::vector<double>& dest,  size_t rows, size_t cols) {
    dest.resize(rows * cols);

    for (size_t i = 0; i < cols; ++i) {
        for (size_t j = 0; j < rows; ++j) {
            dest[i * rows + j] = source[j * cols + i];
        }
    }
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

    total_memory += MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double); // dev_parametr_value

    total_memory += PARAMETR_SIZE * ANT_SIZE * sizeof(double);       // dev_agent_params


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
    return 0.0;
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
    std::cout << "KOL_GPU: " << KOL_GPU << "; "
        << "PARAMETR_SIZE: " << PARAMETR_SIZE << "; "
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
        << "GO_HASH: " << GO_HASH << "; "
        << "HASH_TABLE_SIZE: " << HASH_TABLE_SIZE << "; "
        << "ZERO_HASH_RESULT: " << ZERO_HASH_RESULT << "; "
        << "ZERO_HASH: " << ZERO_HASH << "; "
        << "MAX_PROBES: " << MAX_PROBES << "; "
        << "MAX_CONST: " << MAX_CONST << "; "
        << "MAX_SHARED: " << MAX_SHARED << "; "
        << "BLOCK_SIZE: " << BLOCK_SIZE << "; "
        << "BIN_SEARCH: " << BIN_SEARCH << "; "
        << "NON_WHILE_ANT: " << NON_WHILE_ANT << "; "
        << std::endl;
    if (MAX_VALUE_SIZE * PARAMETR_SIZE < MAX_CONST) { std::cout<< "USE CONST MEMORY" << std::endl; }
    if (PARAMETR_SIZE < MAX_SHARED) { std::cout << "USE SHARED MEMORY" << std::endl; }
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
        << "GO_HASH: " << GO_HASH << "; "
        << "HASH_TABLE_SIZE: " << HASH_TABLE_SIZE << "; "
        << "ZERO_HASH_RESULT: " << ZERO_HASH_RESULT << "; "
        << "ZERO_HASH: " << ZERO_HASH << "; "
        << "MAX_PROBES: " << MAX_PROBES << "; "
        << "MAX_CONST: " << MAX_CONST << "; "
        << "MAX_SHARED: " << MAX_SHARED << "; "
        << "BLOCK_SIZE: " << BLOCK_SIZE << "; "
        << "BIN_SEARCH: " << BIN_SEARCH << "; "
        << "NON_WHILE_ANT: " << NON_WHILE_ANT << "; "
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

    bool go_min_parametrs = false;
    bool go_transposed = false;

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

    std::cout << "Memory bounds check:" << std::endl;
    std::cout << "  PARAMETR_SIZE: " << PARAMETR_SIZE << std::endl;
    std::cout << "  MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << std::endl;
    std::cout << "  ANT_SIZE: " << ANT_SIZE << std::endl;
    std::cout << "  Total matrix elements: " << PARAMETR_SIZE * MAX_VALUE_SIZE << std::endl;
    std::cout << "  Total agent nodes: " << ANT_SIZE * PARAMETR_SIZE << std::endl;

    // Проверка выделенной памяти
    size_t expected_matrix_size = PARAMETR_SIZE * MAX_VALUE_SIZE * sizeof(double);
    size_t expected_agent_size = ANT_SIZE * PARAMETR_SIZE * sizeof(int);

    std::cout << "Expected matrix memory: " << expected_matrix_size << " bytes" << std::endl;
    std::cout << "Expected agent memory: " << expected_agent_size << " bytes" << std::endl;
    // Автоматическая очистка при выходе
    auto cleanup_guard = []() { cleanup_cuda_resources(); };
    // Прогрев
    std::cout << "Performing warmup runs..." << std::endl;
    for (int i = 0; i < KOL_PROGREV; ++i) {
        std::cout << "Warmup " << i << std::endl;
        run_aco_iterations(go_min_parametrs, go_transposed, KOL_ITERATION);
    }

    std::cout << "\n=== Starting 4 function ACO runs ===" << std::endl;
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < KOL_PROGON_STATISTICS; ++i) {
        auto metrics = run_aco_iterations_4function(go_min_parametrs, go_transposed, KOL_ITERATION);
        print_metrics(metrics, "4 function ACO 4start", i);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    std::cout << "4function ACO total execution time: " << total_duration.count() << " ms" << std::endl;

    std::cout << "\n=== Starting main ACO runs ===" << std::endl;
    total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < KOL_PROGON_STATISTICS; ++i) {
        auto metrics = run_aco_iterations(go_min_parametrs, go_transposed, KOL_ITERATION);
        print_metrics(metrics, "Original ACO 3start", i);
    }

    total_end = std::chrono::high_resolution_clock::now();
    total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    std::cout << "Original ACO total execution time: " << total_duration.count() << " ms" << std::endl;

    // Дополнительные запуски - объединенная версия
    std::cout << "\n=== Starting combined ACO runs ===" << std::endl;
    total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < KOL_PROGON_STATISTICS; ++i) {
        auto metrics = run_combined_iterations(go_min_parametrs, go_transposed, KOL_ITERATION, true);
        print_metrics(metrics, "Combined Run 2start", i);
    }

    total_end = std::chrono::high_resolution_clock::now();
    total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    std::cout << "Combined ACO total execution time: " << total_duration.count() << " ms" << std::endl;
 
    go_transposed = true;
    if (go_transposed) {
        std::vector<double> params_transposed, pheromones_transposed, visits_transposed;
        transposeMatrix(params, params_transposed, PARAMETR_SIZE, MAX_VALUE_SIZE);
        transposeMatrix(pheromones, pheromones_transposed, PARAMETR_SIZE, MAX_VALUE_SIZE);
        transposeMatrix(visits, visits_transposed, PARAMETR_SIZE, MAX_VALUE_SIZE);
        if (!initialize_cuda_resources(params_transposed.data(), pheromones_transposed.data(), visits_transposed.data())) {
            std::cerr << "CUDA resources initialization failed" << std::endl;
            return 1;
        }

        std::cout << "\n=== Starting 4 function ACO runs transposed ===" << std::endl;
        total_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < KOL_PROGON_STATISTICS; ++i) {
            auto metrics = run_aco_iterations_4function(go_min_parametrs, go_transposed, KOL_ITERATION);
            print_metrics(metrics, "4 function ACO 4start transposed", i);
        }

        total_end = std::chrono::high_resolution_clock::now();
        total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
        std::cout << "4function ACO total execution time transposed: " << total_duration.count() << " ms" << std::endl;
    }
    // Очистка ресурсов
    cleanup_guard();
    logFile.close();

    return 0;
}