#include "cuda_module.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <thread>


// 42, 84, 168, 336, 672, 1344, 2688, 5376, 10752, 21504, 43008, 86016, 172032, 344064, 688128, 1376256
#define MAX_VALUE_SIZE 4
#define PARAMETR_SIZE 42
#define SET_PARAMETR_SIZE_ONE_X 21    // Количество параметров на оди x 21 (6)
#define ANT_SIZE 500

#define MAX_THREAD_CUDA 256
#define THREADS_PER_AGENT 32
#define HASH_TABLE_SIZE 10000000 // Hash table size (10 million entries)
#define ZERO_HASH_RESULT 1234567891
#define ZERO_HASH 0
#define MAX_PROBES 10000 // Maximum number of probes for collision resolution

#define PRINT_INFORMATION 0

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

// ----------------- Hash Table Entry Structure -----------------
struct alignas(16) HashEntry {
    unsigned long long key;
    float value;  // Используем float для экономии памяти
};
// Структура для данных CUDA
struct CudaData {
    double* parametr_value_dev;        // MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double)
    double* pheromon_value_dev;        // MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double)
    double* kol_enter_value_dev;       // MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double)
    double* norm_matrix_probability_dev; // MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double)
    double* antOFdev;                  // ANT_SIZE * sizeof(double)
    int* ant_parametr_dev;             // PARAMETR_SIZE * ANT_SIZE * sizeof(int)
    double* maxOf_dev;                 // sizeof(double)
    double* minOf_dev;                 // sizeof(double)
    int* kol_hash_fail_dev;            // sizeof(int)
    HashEntry* hashTable_dev;          // HASH_TABLE_SIZE * sizeof(HashEntry)
    cudaStream_t stream;
    bool cuda_initialized;
    bool initialized;
};

static CudaData cuda_data = {
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, 0, false, false
};

// ----------------- Atomic Operations для double -----------------
__device__ __forceinline__ void atomicMaxDouble(double* address, double value) {
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long old = *addr_as_ull, assumed;
    do {
        assumed = old;
        if (value > __longlong_as_double(assumed))
            old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(value));
    } while (assumed != old);
}
__device__ __forceinline__ void atomicMinDouble(double* address, double value) {
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long old = *addr_as_ull, assumed;
    do {
        assumed = old;
        if (value < __longlong_as_double(assumed))
            old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(value));
    } while (assumed != old);
}

// ----------------- Kernel: Initializing Hash Table -----------------
__global__ void initializeHashTable(HashEntry* hashTable, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        hashTable[i].key = ZERO_HASH_RESULT;
        hashTable[i].value = 0.0f;
    }
}
// ----------------- MurmurHash64A Implementation -----------------
__device__ unsigned long long murmurHash64A(unsigned long long key, unsigned long long seed = 0xDEADBEEFDEADBEEF) {
    unsigned long long m = 0xc6a4a7935bd1e995ULL;
    int r = 47;
    unsigned long long h = seed ^ (8 * m);

    unsigned long long k = key;
    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}
// ----------------- Improved Hash Function Using MurmurHash -----------------
__device__ unsigned long long betterHashFunction(unsigned long long key) {
    return murmurHash64A(key) % HASH_TABLE_SIZE;
}
// ----------------- Key Generation Function -----------------
__device__ unsigned long long generateKey(const int* agent_node, int bx) {
    unsigned long long key = 0;
    unsigned long long factor = 1;
    for (int i = 0; i < PARAMETR_SIZE; i++) {
        int val = agent_node[bx * PARAMETR_SIZE + i];
        key += val * factor;
        factor *= MAX_VALUE_SIZE;
    }
    return key;
}
// ----------------- Hash Table Search with Quadratic Probing -----------------
__device__ double getCachedResultOptimized(HashEntry* hashTable, const int* agent_node, int bx) {
    unsigned long long key = generateKey(agent_node, bx);
    unsigned long long idx = betterHashFunction(key);

    for (int i = 0; i < MAX_PROBES; i++) {
        unsigned long long probe_idx = (idx + i * i) % HASH_TABLE_SIZE;

        if (hashTable[probe_idx].key == key) {
            return hashTable[probe_idx].value;
        }
        if (hashTable[probe_idx].key == ZERO_HASH_RESULT) {
            return -1.0;
        }
    }
    return -1.0;
}
// ----------------- Hash Table Insertion with Quadratic Probing -----------------
__device__ void saveToCacheOptimized(HashEntry* hashTable, const int* agent_node, int bx, double value) {
    unsigned long long key = generateKey(agent_node, bx);
    unsigned long long idx = betterHashFunction(key);

    for (int i = 0; i < MAX_PROBES; i++) {
        unsigned long long probe_idx = (idx + i * i) % HASH_TABLE_SIZE;
        unsigned long long expected = ZERO_HASH_RESULT;
        unsigned long long desired = key;

        unsigned long long old = atomicCAS(&(hashTable[probe_idx].key), expected, desired);
        if (old == expected || old == key) {
            hashTable[probe_idx].value = (float)value;
            return;
        }
    }
}

// Функция для вычисления параметра х при  параметрическом графе
__device__ double go_x(double* parametr, int start_index, int kol_parametr) {
    double sum = 0.0;
    for (int i = 1; i < kol_parametr; ++i) {
        sum += parametr[start_index + i];
    }
    return parametr[start_index] * sum; // Умножаем на первый параметр в диапазоне
}


#if (DELT4)
// Михаэлевич-Викинский
__device__ double BenchShafferaFunction(double* parametr) {
    double r_squared = 0.0;
    double sum_if = 0.0;
    double sum = 0.0;
    double second_sum = 0.0;
    double r_cos = 1.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
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
#if (RASTRIGIN)
// Растригин-функция
__device__ double BenchShafferaFunction(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
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
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
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
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
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
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
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
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
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
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
        sum -= x * sin(sqrt(abs(x)));
    }
    return sum;
}
#endif
#if (LEVY)
// Леви-функция - ИСПРАВЛЕННАЯ ВЕРСИЯ
__device__ double BenchShafferaFunction(double* parametr) {
    double w_first = 1 + (go_x(parametr, 0, SET_PARAMETR_SIZE_ONE_X) - 1) / 4;
    double w_last = 1 + (go_x(parametr, (PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X - 1) * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X) - 1) / 4;
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;

    for (int i = 1; i <= num_variables - 1; ++i) {
        double wi = 1 + (go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X) - 1) / 4;
        double w_prev = 1 + (go_x(parametr, (i - 1) * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X) - 1) / 4;
        sum += pow(wi - 1, 2) * (1 + 10 * pow(sin(M_PI * wi), 2)) +
            pow(wi - w_prev, 2) * (1 + pow(sin(2 * M_PI * wi), 2));
    }
    return pow(sin(M_PI * w_first), 2) + sum + pow(w_last - 1, 2) * (1 + pow(sin(2 * M_PI * w_last), 2));
}
#endif
#if (MICHAELWICZYNSKI)
// Михаэлевич-Викинский
__device__ double BenchShafferaFunction(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
        sum -= sin(x) * pow(sin((i + 1) * x * x / M_PI), 20);
    }
    return sum;
}
#endif

#if (SHAFFERA)
__device__ double BenchShafferaFunction(double* __restrict__ parametr) {
    double r_squared = 0.0;
    const int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;

    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
        r_squared += x * x;
    }

    double r = sqrt(r_squared);
    double sin_r = sin(r);
    return 0.5 - (sin_r * sin_r - 0.5) / (1.0 + 0.001 * r_squared);
}
#endif
#if (CARROM_TABLE)
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
    double OF = r_cos * exp(fabs(a));
    return OF * OF;
}
#endif

// ==================== ОПТИМИЗИРОВАННЫЕ ФУНКЦИИ ХЭШ-ТАБЛИЦЫ ====================
__device__ __forceinline__ unsigned long long generateKey4(const int* __restrict__ agent_node, int ant_id) {
    // Битовый пак для 4 значений (2 бита на значение)
    unsigned long long key = 0;
    const int bits_per_value = 2;

#pragma unroll 8
    for (int i = 0; i < PARAMETR_SIZE; i++) {
        int value = agent_node[ant_id * PARAMETR_SIZE + i] & 0x3;
        key = (key << bits_per_value) | value;
    }
    return key;
}
__device__ __forceinline__ float getCachedResult4(HashEntry* __restrict__ hashTable, const int* __restrict__ agent_node, int ant_id) {
#if GO_HASH
    unsigned long long key = generateKey4(agent_node, ant_id);
    unsigned long long idx = key % HASH_TABLE_SIZE;

#pragma unroll 4
    for (int i = 0; i < 3; i++) {
        if (hashTable[idx].key == key) return hashTable[idx].value;
        if (hashTable[idx].key == ZERO_HASH_RESULT) return -1.0f;
        idx = (idx + 1) % HASH_TABLE_SIZE;
    }
#endif
    return -1.0f;
}
__device__ __forceinline__ void saveToCache4(HashEntry* __restrict__ hashTable, const int* __restrict__ agent_node, int ant_id, float value) {
#if GO_HASH
    unsigned long long key = generateKey4(agent_node, ant_id);
    unsigned long long idx = key % HASH_TABLE_SIZE;
#pragma unroll 4
    for (int i = 0; i < 3; i++) {
        unsigned long long old = atomicCAS(&hashTable[idx].key, ZERO_HASH_RESULT, key);
        if (old == ZERO_HASH_RESULT || old == key) {
            hashTable[idx].value = value;
            return;
        }
        idx = (idx + 1) % HASH_TABLE_SIZE;
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

#if (MAX_VALUE_SIZE==4)
// Оптимизированная версия с использованием шаблонов и лучшей организации потоков
__global__ void go_all_agent_optimized_v2( double* __restrict__ parametr, double* __restrict__ norm_matrix_probability, int* __restrict__ agent_node, double* __restrict__ OF, HashEntry* __restrict__ hashTable, double* __restrict__ maxOf_dev, double* __restrict__ minOf_dev, int* __restrict__ kol_hash_fail, int iteration, int ant_size) {

    // ==================== ОПТИМИЗАЦИЯ 1: ОРГАНИЗАЦИЯ ПАМЯТИ ====================
    __shared__ curandState shared_states[MAX_THREAD_CUDA];  // Shared memory для генераторов
    __shared__ double probs[PARAMETR_SIZE][MAX_VALUE_SIZE - 1]; // Предзагружаем вероятности (для MAX_VALUE_SIZE=4)
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int block_agents = blockDim.x;  // Количество агентов в блоке
    int global_agent_id = bx * block_agents + tx; // Глобальный ID агента
    if (global_agent_id > ant_size) { return; }

#pragma unroll
        for (int param = 0; param < PARAMETR_SIZE; param++) {
            int base_idx = param * MAX_VALUE_SIZE;
            probs[param][0] = norm_matrix_probability[base_idx];
            probs[param][1] = norm_matrix_probability[base_idx + 1];
            probs[param][2] = norm_matrix_probability[base_idx + 2];
        }
    __syncthreads();
    curand_init(clock64() + global_agent_id * 100 + iteration * 7919, tx, 0, &shared_states[tx]);

    // ==================== ОПТИМИЗАЦИЯ 4: УСТРАНЕНИЕ WARP DIVERGENCE ====================
    // Только первый поток в группе обрабатывает агента
    if (tx % THREADS_PER_AGENT == 0) {
        curandState* state = &shared_states[tx];
        double agent[PARAMETR_SIZE];
        int nodes[PARAMETR_SIZE];
        bool valid_solution = true;

        // ==================== ОПТИМИЗАЦИЯ 5: РАЗВЕРНУТЫЙ ЦИКЛ И БИНАРНЫЙ ПОИСК ====================
#pragma unroll
        for (int param = 0; param < PARAMETR_SIZE; param++) {
            double randomValue = curand_uniform(state);
            int selected_idx = 0;
            // Развернутый цикл для избежания Warp Divergence
            if (valid_solution) {
                if (randomValue <= probs[param][0]) {
                    selected_idx = 0;
                }
                else if (randomValue <= probs[param][1]) {
                    selected_idx = 1;
                }
                else if (randomValue <= probs[param][2]) {
                    selected_idx = 2;
                }
                else {
                    selected_idx = 3;
                }
            }
            nodes[param] = selected_idx;

            // Быстрая загрузка параметров с предвычисленными индексами
            int param_idx = param * MAX_VALUE_SIZE + selected_idx;
            agent[param] = parametr[param_idx];
            // Проверка выбора 4-го значения (9) в слое 0,3,6,9
            valid_solution = (selected_idx != (MAX_VALUE_SIZE - 1));
        }
        double fitness;
#if (GO_HASH)
            // Используем оптимизированный кэш
            double cachedResult = getCachedResultOptimized(hashTable, nodes, global_agent_id);
            if (cachedResult < 0.0) {
                fitness = BenchShafferaFunction(agent);
                saveToCacheOptimized(hashTable, nodes, global_agent_id, fitness);
            }
            else {
                fitness = cachedResult;
                atomicAdd(kol_hash_fail, 1);
            }
#else
        fitness = BenchShafferaFunction(agent);
#endif
        // Сохраняем узлы в память
        int agent_base = global_agent_id * PARAMETR_SIZE;
#pragma unroll 4
        for (int i = 0; i < PARAMETR_SIZE; i++) {
            agent_node[agent_base + i] = nodes[i];
        }

        // Сохраняем значение фитнес-функции
        if (valid_solution && fabs(fitness - ZERO_HASH_RESULT) > 1e-10) {
            atomicMaxDouble(maxOf_dev, fitness);
            atomicMinDouble(minOf_dev, fitness);
        }
        else {
            OF[global_agent_id] = ZERO_HASH_RESULT;
        }
    }
}
__global__ void go_all_agent_optimized(double* parametr, double* norm_matrix_probability, int* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail, int iteration, int ant_size) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;
    if (bx < ant_size) {
        curandState state;
        curand_init(clock64() + bx + iteration * 1000, 0, 0, &state);

        double agent[PARAMETR_SIZE] = { 0 };
        bool valid_solution = true;
#pragma unroll
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            double randomValue = curand_uniform(&state);
            int k = 0;
            if (valid_solution) {
                if (randomValue <= norm_matrix_probability[MAX_VALUE_SIZE * tx + 0]) {
                    k = 0;
                }
                else if (randomValue <= norm_matrix_probability[MAX_VALUE_SIZE * tx + 1]) {
                    k = 1;
                }
                else if (randomValue <= norm_matrix_probability[MAX_VALUE_SIZE * tx + 2]) {
                    k = 2;
                }
                else {
                    k = 3;
                }
            }
            agent_node[bx * PARAMETR_SIZE + tx] = k;
            agent[tx] = parametr[tx * MAX_VALUE_SIZE + k];
            valid_solution = (k != MAX_VALUE_SIZE - 1);
        }
        //OF[bx] = -bx;
        //OF[bx] = BenchShafferaFunction(agent);


        double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);

        if (cachedResult < 0) {
            OF[bx] = BenchShafferaFunction(agent);
            saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            OF[bx] = cachedResult;
            atomicAdd(kol_hash_fail, 1);
        }

        if (OF[bx] != ZERO_HASH_RESULT) {
            atomicMaxDouble(maxOf_dev, OF[bx]);
            atomicMinDouble(minOf_dev, OF[bx]);
        }
    }
}

#else
__global__ void go_all_agent_optimized(double* parametr, double* norm_matrix_probability, int* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail, int iteration, int ant_size) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;
    if (bx < ant_size) {
        curandState state;
        curand_init(clock64() + bx + iteration * 1000, 0, 0, &state);

        double agent[PARAMETR_SIZE] = { 0 };

        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            double randomValue = curand_uniform(&state);
            int k = 0;

            while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                k++;
            }
            agent_node[bx * PARAMETR_SIZE + tx] = k;
            agent[tx] = parametr[tx * MAX_VALUE_SIZE + k];
        }
        //OF[bx] = -bx;
        //OF[bx] = BenchShafferaFunction(agent);


        double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);

        if (cachedResult < 0) {
            OF[bx] = BenchShafferaFunction(agent);
            saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            OF[bx] = cachedResult;
            atomicAdd(kol_hash_fail, 1);
        }

        if (OF[bx] != ZERO_HASH_RESULT) {
            atomicMaxDouble(maxOf_dev, OF[bx]);
            atomicMinDouble(minOf_dev, OF[bx]);
        }
    }
}
#endif

#if (MAX_VALUE_SIZE==4)
__global__ void go_all_agent_optimized_non_hash(double* parametr, double* norm_matrix_probability, int* agent_node, double* OF, double* maxOf_dev, double* minOf_dev, int iteration, int ant_size) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;
    if (bx < ant_size) {
        curandState state;
        curand_init(clock64() + bx + iteration * 1000, 0, 0, &state);

        double agent[PARAMETR_SIZE] = { 0 };
        bool valid_solution = true;
#pragma unroll
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            double randomValue = curand_uniform(&state);
            int k = 0;
            if (valid_solution) {
                if (randomValue <= norm_matrix_probability[MAX_VALUE_SIZE * tx + 0]) {
                    k = 0;
                }
                else if (randomValue <= norm_matrix_probability[MAX_VALUE_SIZE * tx + 1]) {
                    k = 1;
                }
                else if (randomValue <= norm_matrix_probability[MAX_VALUE_SIZE * tx + 2]) {
                    k = 2;
                }
                else {
                    k = 3;
                }
            }
            agent_node[bx * PARAMETR_SIZE + tx] = k;
            agent[tx] = parametr[tx * MAX_VALUE_SIZE + k];
            valid_solution = (k != MAX_VALUE_SIZE - 1);
        }

        OF[bx] = BenchShafferaFunction(agent);
        atomicMaxDouble(maxOf_dev, OF[bx]);
        atomicMinDouble(minOf_dev, OF[bx]);
    }
}

#else
__global__ void go_all_agent_optimized_non_hash(double* parametr, double* norm_matrix_probability, int* agent_node, double* OF, double* maxOf_dev, double* minOf_dev, int iteration, int ant_size) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;
    if (bx < ant_size) {
        curandState state;
        curand_init(clock64() + bx + iteration * 1000, 0, 0, &state);

        double agent[PARAMETR_SIZE] = { 0 };

        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            double randomValue = curand_uniform(&state);
            int k = 0;

            while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                k++;
            }
            agent_node[bx * PARAMETR_SIZE + tx] = k;
            agent[tx] = parametr[tx * MAX_VALUE_SIZE + k];
        }

        OF[bx] = BenchShafferaFunction(agent);
        atomicMaxDouble(maxOf_dev, OF[bx]);
        atomicMinDouble(minOf_dev, OF[bx]);
    }
}
#endif

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
        if (cuda_data.parametr_value_dev) { cudaFree(cuda_data.parametr_value_dev); }
        if (cuda_data.pheromon_value_dev) { cudaFree(cuda_data.pheromon_value_dev); }
        if (cuda_data.kol_enter_value_dev) { cudaFree(cuda_data.kol_enter_value_dev); }
        if (cuda_data.norm_matrix_probability_dev) { cudaFree(cuda_data.norm_matrix_probability_dev); }
        if (cuda_data.ant_parametr_dev) { cudaFree(cuda_data.ant_parametr_dev); }
        if (cuda_data.antOFdev) { cudaFree(cuda_data.antOFdev); }
        if (cuda_data.hashTable_dev) { cudaFree(cuda_data.hashTable_dev); }
        if (cuda_data.maxOf_dev) { cudaFree(cuda_data.maxOf_dev); }
        if (cuda_data.minOf_dev) { cudaFree(cuda_data.minOf_dev); }
        if (cuda_data.kol_hash_fail_dev) { cudaFree(cuda_data.kol_hash_fail_dev); }
        if (cuda_data.stream) { cudaStreamDestroy(cuda_data.stream); }

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
#if (PRINT_INFORMATION)
        std::cout << "[CUDA] CUDA resources cleaned up" << std::endl;
#endif // (PRINT_INFORMATION)
    }
}

bool cuda_initialize_non_hash(const double* parametr_value, const double* pheromon_value, const double* kol_enter_value) {
    if (cuda_data.initialized) {
        cuda_cleanup();
    }
#if (PRINT_INFORMATION)
    std::cout << "[CUDA] Initializing..." << std::endl;
#endif // (PRINT_INFORMATION)
    size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
    size_t ant_matrix_size = PARAMETR_SIZE * ANT_SIZE;
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "[CUDA] No CUDA devices found!" << std::endl;
        return false;
    }
    // Установка флагов для RTX 3060
    cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost);
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
    // Копирование параметров
    err = cudaMemcpyAsync(cuda_data.parametr_value_dev, parametr_value, matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_data.stream);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMemcpyAsync(cuda_data.pheromon_value_dev, pheromon_value, matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_data.stream);
    if (err != cudaSuccess) goto cuda_error;
    err = cudaMemcpyAsync(cuda_data.kol_enter_value_dev, kol_enter_value, matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_data.stream);
    if (err != cudaSuccess) goto cuda_error;
    cudaStreamSynchronize(cuda_data.stream);
    cuda_data.initialized = true;
#if (PRINT_INFORMATION)
    std::cout << "[CUDA] Initialized successfully!" << std::endl;
#endif // PRINT_INFORMATION==1
    return true;

cuda_error:
    std::cerr << "[CUDA] Error: " << cudaGetErrorString(err) << std::endl;
    cuda_cleanup();
    return false;
}
bool cuda_initialize(const double* parametr_value, const double* pheromon_value, const double* kol_enter_value) {
    if (cuda_data.initialized) {
        cuda_cleanup();
    }
#if (PRINT_INFORMATION)
    std::cout << "[CUDA] Initializing..." << std::endl;
#endif // (PRINT_INFORMATION)
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
    // Установка флагов для RTX 3060
    cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost);
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
#if (PRINT_INFORMATION)
    std::cout << "[CUDA] Initialized successfully!" << std::endl;
#endif // PRINT_INFORMATION==1
    return true;

cuda_error:
    std::cerr << "[CUDA] Error: " << cudaGetErrorString(err) << std::endl;
    cuda_cleanup();
    return false;
}

void cuda_run_iteration(const double* norm_matrix_probability, int* ant_parametr, double* antOF, int ant_size, double* global_minOf, double* global_maxOf, int* kol_hash_fail, double* time_all, double* time_function, int iteration, void (*completion_callback)(double*, int, int)) {
    if (!cuda_data.initialized) {
        std::cerr << "[CUDA] CUDA not initialized!" << std::endl;
        return;
    }
    cudaEvent_t start_time_event, start_time_event_function, stop_event;
    float gpuTime = 0.0f; float gpuTime_function = 0.0f;
    cudaEventCreate(&start_time_event);
    cudaEventRecord(start_time_event, cuda_data.stream);
    cudaEventCreate(&start_time_event_function);
    cudaEventCreate(&stop_event);
    size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
    size_t ant_matrix_size = PARAMETR_SIZE * ant_size;
    
    // Сброс статистики
    double min_init = 1e9, max_init = -1e9;
    int zero_hash_fail = 0;
    cudaMemcpyAsync(cuda_data.maxOf_dev, &max_init, sizeof(double), cudaMemcpyHostToDevice, cuda_data.stream);
    cudaMemcpyAsync(cuda_data.minOf_dev, &min_init, sizeof(double), cudaMemcpyHostToDevice, cuda_data.stream);
    cudaMemcpyAsync(cuda_data.kol_hash_fail_dev, &zero_hash_fail, sizeof(int), cudaMemcpyHostToDevice, cuda_data.stream);

    // Копирование нормализованной матрицы вероятностей
    cudaMemcpyAsync(cuda_data.norm_matrix_probability_dev, norm_matrix_probability, matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_data.stream);
    cudaEventRecord(start_time_event_function, cuda_data.stream);
    // Запуск ядра
    const int threadsPerBlock = MAX_THREAD_CUDA;
    const int numBlocks = (ant_size + threadsPerBlock - 1) / threadsPerBlock;

    go_all_agent_optimized_v2 <<<numBlocks, threadsPerBlock, 0, cuda_data.stream >>>(cuda_data.parametr_value_dev, cuda_data.norm_matrix_probability_dev, cuda_data.ant_parametr_dev, cuda_data.antOFdev, cuda_data.hashTable_dev,cuda_data.maxOf_dev, cuda_data.minOf_dev, cuda_data.kol_hash_fail_dev, iteration, ant_size);


    cudaEventRecord(stop_event, cuda_data.stream);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&gpuTime_function, start_time_event_function, stop_event);
    *time_function = *time_function + gpuTime_function;
    // Копирование результатов обратно
    cudaMemcpyAsync(ant_parametr, cuda_data.ant_parametr_dev, ant_matrix_size * sizeof(int), cudaMemcpyDeviceToHost, cuda_data.stream);
    cudaMemcpyAsync(antOF, cuda_data.antOFdev, ant_size * sizeof(double), cudaMemcpyDeviceToHost, cuda_data.stream);
    cudaMemcpyAsync(global_minOf, cuda_data.minOf_dev, sizeof(double), cudaMemcpyDeviceToHost, cuda_data.stream);
    cudaMemcpyAsync(global_maxOf, cuda_data.maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost, cuda_data.stream);
    cudaMemcpyAsync(kol_hash_fail, cuda_data.kol_hash_fail_dev, sizeof(int), cudaMemcpyDeviceToHost, cuda_data.stream);
    
    // Синхронизация и callback
    cudaStreamSynchronize(cuda_data.stream);
    cudaEventRecord(stop_event, cuda_data.stream);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&gpuTime, start_time_event, stop_event);
    *time_all = *time_all + gpuTime;

    // Вызов callback синхронно (упрощенная версия)
    if (completion_callback) {
        completion_callback(antOF, ant_size, iteration);
    }

    cudaEventDestroy(start_time_event);
    cudaEventDestroy(start_time_event_function);
    cudaEventDestroy(stop_event);
}
void cuda_run_iteration_non_hash(const double* norm_matrix_probability, int* ant_parametr, double* antOF, int ant_size, double* global_minOf, double* global_maxOf, double* time_all, double* time_function, int iteration, void (*completion_callback)(double*, int, int)) {
    if (!cuda_data.initialized) {
        std::cerr << "[CUDA] CUDA not initialized!" << std::endl;
        return;
    }
    cudaEvent_t start_time_event, start_time_event_function, stop_event;
    float gpuTime = 0.0f; float gpuTime_function = 0.0f;
    cudaEventCreate(&start_time_event);
    cudaEventRecord(start_time_event, cuda_data.stream);
    cudaEventCreate(&start_time_event_function);
    cudaEventCreate(&stop_event);
    size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
    size_t ant_matrix_size = PARAMETR_SIZE * ant_size;

    // Сброс статистики
    double min_init = 1e9, max_init = -1e9;
    cudaMemcpyAsync(cuda_data.maxOf_dev, &max_init, sizeof(double), cudaMemcpyHostToDevice, cuda_data.stream);
    cudaMemcpyAsync(cuda_data.minOf_dev, &min_init, sizeof(double), cudaMemcpyHostToDevice, cuda_data.stream);

    // Копирование нормализованной матрицы вероятностей
    cudaMemcpyAsync(cuda_data.norm_matrix_probability_dev, norm_matrix_probability, matrix_size * sizeof(double), cudaMemcpyHostToDevice, cuda_data.stream);
    cudaEventRecord(start_time_event_function, cuda_data.stream);
    // Запуск ядра
    const int threadsPerBlock = MAX_THREAD_CUDA;
    const int numBlocks = (ant_size + threadsPerBlock - 1) / threadsPerBlock;

    go_all_agent_optimized_non_hash << <numBlocks, threadsPerBlock, 0, cuda_data.stream >> > (cuda_data.parametr_value_dev, cuda_data.norm_matrix_probability_dev, cuda_data.ant_parametr_dev, cuda_data.antOFdev, cuda_data.maxOf_dev, cuda_data.minOf_dev, iteration, ant_size);

    cudaEventRecord(stop_event, cuda_data.stream);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&gpuTime_function, start_time_event_function, stop_event);
    *time_function = *time_function + gpuTime_function;
    // Копирование результатов обратно
    cudaMemcpyAsync(ant_parametr, cuda_data.ant_parametr_dev, ant_matrix_size * sizeof(int), cudaMemcpyDeviceToHost, cuda_data.stream);
    cudaMemcpyAsync(antOF, cuda_data.antOFdev, ant_size * sizeof(double), cudaMemcpyDeviceToHost, cuda_data.stream);
    cudaMemcpyAsync(global_minOf, cuda_data.minOf_dev, sizeof(double), cudaMemcpyDeviceToHost, cuda_data.stream);
    cudaMemcpyAsync(global_maxOf, cuda_data.maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost, cuda_data.stream);

    // Синхронизация и callback
    cudaStreamSynchronize(cuda_data.stream);
    cudaEventRecord(stop_event, cuda_data.stream);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&gpuTime, start_time_event, stop_event);
    *time_all = *time_all + gpuTime;

    // Вызов callback синхронно (упрощенная версия)
    if (completion_callback) {
        completion_callback(antOF, ant_size, iteration);
    }

    cudaEventDestroy(start_time_event);
    cudaEventDestroy(start_time_event_function);
    cudaEventDestroy(stop_event);
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
void cuda_run_async(const double* norm_matrix_probability, const int* ant_parametr, double* antOF, int iteration, void (*completion_callback)(double*, int, int)) {
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