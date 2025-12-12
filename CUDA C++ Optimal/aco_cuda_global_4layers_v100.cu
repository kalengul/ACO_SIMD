#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "parametrs4.h"

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(err) << std::endl; exit(EXIT_FAILURE); } } while(0)

std::ofstream logFile("log.txt");

// ==================== ОПТИМИЗИРОВАННЫЕ СТРУКТУРЫ ДАННЫХ ====================
struct alignas(16) HashEntry {
    unsigned long long key;
    float value;
};

struct PerformanceMetrics {
    float total_time_ms, kernel_time_ms, memory_time_ms;
    float computeProbabilities_time_ms, antColonyOptimization_time_ms, updatePheromones_time_ms;
    float occupancy, memory_throughput_gbs;
    double min_fitness, max_fitness;
    int hash_hits, hash_misses;
};

// ==================== КОНСТАНТЫ ДЛЯ TESLA V100 ====================
const int V100_SM_COUNT = 80;
const int V100_WARP_SIZE = 32;
const int V100_MAX_THREADS_PER_SM = 2048;
const int V100_MAX_THREADS_PER_BLOCK = 1024;
const int V100_SHARED_MEM_PER_BLOCK = 96 * 1024;
//const int V100_MAX_REGISTERS_PER_THREAD = 255;

// Оптимальные размеры блоков для V100
const int OPTIMAL_BLOCK_SIZE = 256;
const int OPTIMAL_ANT_BLOCK_SIZE = 256;
const int OPTIMAL_UPDATE_BLOCK_SIZE = 256;

// Векторные типы для оптимизации памяти
struct alignas(16) Float4_v100 {
    float x, y, z, w;

    __device__ __host__ Float4_v100 operator*(float scalar) const {
        return { x * scalar, y * scalar, z * scalar, w * scalar };
    }

    __device__ __host__ Float4_v100 operator+(const Float4_v100& other) const {
        return { x + other.x, y + other.y, z + other.z, w + other.w };
    }

    __device__ __host__ Float4_v100 operator+=(const Float4_v100& other) {
        x += other.x; y += other.y; z += other.z; w += other.w;
        return *this;
    }
};

// Структура для локальных данных потока
struct ThreadData_v100 {
    float pheromon_delta[4];
    float kol_enter_delta[4];
    Float4_v100 pheromon_current;
    Float4_v100 kol_enter_current;

    __device__ ThreadData_v100() {
        pheromon_delta[0] = pheromon_delta[1] = pheromon_delta[2] = pheromon_delta[3] = 0.0f;
        kol_enter_delta[0] = kol_enter_delta[1] = kol_enter_delta[2] = kol_enter_delta[3] = 0.0f;
    }
};

// ==================== ОПТИМИЗИРОВАННЫЕ МАТЕМАТИЧЕСКИЕ ФУНКЦИИ ====================
__device__ __forceinline__ double go_x_v100(const double* __restrict__ parametr, int start_index) {
    double sum = 0.0;

    // Развертка для лучшей производительности
    int i = 1;
    const int step = 4;
    const int limit = SET_PARAMETR_SIZE_ONE_X - step + 1;

    for (; i < limit; i += step) {
        sum += parametr[start_index + i] + parametr[start_index + i + 1] +
            parametr[start_index + i + 2] + parametr[start_index + i + 3];
    }

    for (; i < SET_PARAMETR_SIZE_ONE_X; ++i) {
        sum += parametr[start_index + i];
    }

    return parametr[start_index] * sum;
}

// ==================== БЕНЧМАРК ФУНКЦИИ ====================
#if (SHAFFERA)
__device__ double BenchShafferaFunction_v100(const double* __restrict__ parametr) {
    double r_squared = 0.0;
    const int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;

    // Оптимизированный цикл с разверткой для V100
#pragma unroll 8
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_v100(parametr, i * SET_PARAMETR_SIZE_ONE_X);
        r_squared += x * x;
    }

    double r = sqrt(r_squared);
    double sin_r = sin(r);
    return 0.5 - (sin_r * sin_r - 0.5) / (1.0 + 0.001 * r_squared);
}
#endif

#if (CARROM_TABLE)
__device__ double BenchShafferaFunction_v100(double* parametr) {
    double r_cos = 1.0;
    double r_squared = 0.0;
    int num_variables = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;

#pragma unroll 4
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_v100(parametr, i * SET_PARAMETR_SIZE_ONE_X);
        r_cos *= cos(x);
        r_squared += x * x;
    }
    double a = 1.0 - sqrt(r_squared) / M_PI;
    double OF = r_cos * exp(fabs(a));
    return OF * OF;
}
#endif

// ==================== ХЭШ-ТАБЛИЦА ДЛЯ V100 ====================
__device__ __forceinline__ unsigned long long generateKey_v100(const int* __restrict__ agent_node, int ant_id) {
    unsigned long long key = 0;
    const int bits_per_value = 2;

    // Оптимизированная версия с группировкой операций
    int base_idx = ant_id * PARAMETR_SIZE;

    // Обрабатываем по 16 значений за итерацию
    for (int i = 0; i < PARAMETR_SIZE; i += 16) {
        unsigned long long chunk = 0;
        int limit = min(PARAMETR_SIZE - i, 16);

        for (int j = 0; j < limit; ++j) {
            int value = agent_node[base_idx + i + j] & 0x3;
            chunk = (chunk << bits_per_value) | value;
        }

        if (limit < 16) {
            chunk <<= (bits_per_value * (16 - limit));
        }

        key ^= chunk;
        key = (key << 13) | (key >> (64 - 13)); // Поворот для лучшего распределения
    }

    return key;
}
__device__ __forceinline__ float getCachedResult_v100(HashEntry* __restrict__ hashTable, const int* __restrict__ agent_node, int ant_id) {
#if GO_HASH
    unsigned long long key = generateKey_v100(agent_node, ant_id);
    unsigned long long idx = key % HASH_TABLE_SIZE;

    // Используем prefetch для лучшей производительности
    __builtin_prefetch(&hashTable[idx], 0, 3);

    // Линейный поиск с ограниченным числом проб
    for (int i = 0; i < 4; i++) {
        if (hashTable[idx].key == key) return hashTable[idx].value;
        if (hashTable[idx].key == ZERO_HASH_RESULT) return -1.0f;
        idx = (idx + 1) % HASH_TABLE_SIZE;
    }
#endif
    return -1.0f;
}

__device__ __forceinline__ void saveToCache_v100(HashEntry* __restrict__ hashTable, const int* __restrict__ agent_node, int ant_id, float value) {
#if GO_HASH
    unsigned long long key = generateKey_v100(agent_node, ant_id);
    unsigned long long idx = key % HASH_TABLE_SIZE;

    for (int i = 0; i < 4; i++) {
        unsigned long long old = atomicCAS(&hashTable[idx].key, ZERO_HASH_RESULT, key);
        if (old == ZERO_HASH_RESULT || old == key) {
            hashTable[idx].value = value;
            return;
        }
        idx = (idx + 1) % HASH_TABLE_SIZE;
    }
#endif
}

// ==================== АТОМАРНЫЕ ОПЕРАЦИИ ====================
__device__ __forceinline__ void atomicMax_v100(float* address, float value) {
    int* addr_as_int = (int*)address;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        if (value > __int_as_float(assumed))
            old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}

__device__ __forceinline__ void atomicMin_v100(float* address, float value) {
    int* addr_as_int = (int*)address;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        if (value < __int_as_float(assumed))
            old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}

// ==================== ЯДРА ИНИЦИАЛИЗАЦИИ ====================
__global__ void initializeHashTable_v100(HashEntry* hashTable) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = tid; i < HASH_TABLE_SIZE; i += stride) {
        hashTable[i].key = ZERO_HASH_RESULT;
        hashTable[i].value = 0.0f;
    }
}

// ==================== ОПТИМИЗИРОВАННЫЕ ЯДРА ДЛЯ V100 ====================
__device__ void evaporatePheromones_v100_dev(float* __restrict__ pheromon) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= PARAMETR_SIZE) return;

    const int base_idx = tid * 4;

    Float4_v100* pheromon_vec = reinterpret_cast<Float4_v100*>(&pheromon[base_idx]);
    Float4_v100 current = *pheromon_vec;

    current = current * PARAMETR_RO;

    *pheromon_vec = current;
}

__device__ void depositPheromones_v100_dev(const float* __restrict__ OF,
    const int* __restrict__ agent_node,
    float* __restrict__ pheromon,
    float* __restrict__ kol_enter) {
    const int param_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (param_idx >= PARAMETR_SIZE) return;

    const int base_idx = param_idx * 4;

    // Используем shared memory для редукции
    __shared__ float s_pheromon[4][32];
    __shared__ float s_kol_enter[4][32];

    int warp_id = threadIdx.x / V100_WARP_SIZE;
    int lane_id = threadIdx.x % V100_WARP_SIZE;

    // Инициализация shared memory
    if (lane_id < 4) {
        s_pheromon[lane_id][warp_id] = 0.0f;
        s_kol_enter[lane_id][warp_id] = 0.0f;
    }
    __syncthreads();

    // Каждый поток обрабатывает свою часть агентов
    const int threads_total = gridDim.x * blockDim.x;
    const int agents_per_thread = (ANT_SIZE + threads_total - 1) / threads_total;
    const int start_agent = param_idx * agents_per_thread;
    const int end_agent = min(start_agent + agents_per_thread, ANT_SIZE);

    float local_pheromon[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float local_kol_enter[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (int ant_id = start_agent; ant_id < end_agent; ant_id++) {
        if (OF[ant_id] != ZERO_HASH_RESULT) {
            const int k = agent_node[ant_id * PARAMETR_SIZE + param_idx];

            if (k >= 0 && k < 4) {
                local_kol_enter[k] += 1.0f;

#if OPTIMIZE_MIN_1
                float delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[ant_id];
                if (delta > 0.0f) local_pheromon[k] += PARAMETR_Q * delta;
#elif OPTIMIZE_MIN_2
                float of_val = fmaxf(OF[ant_id], 1e-7f);
                local_pheromon[k] += PARAMETR_Q / of_val;
#elif OPTIMIZE_MAX
                local_pheromon[k] += PARAMETR_Q * OF[ant_id];
#endif
            }
        }
    }

    // Редукция внутри warp
    for (int k = 0; k < 4; k++) {
        for (int offset = 16; offset > 0; offset /= 2) {
            local_pheromon[k] += __shfl_down_sync(0xFFFFFFFF, local_pheromon[k], offset);
            local_kol_enter[k] += __shfl_down_sync(0xFFFFFFFF, local_kol_enter[k], offset);
        }

        if (lane_id == 0) {
            s_pheromon[k][warp_id] = local_pheromon[k];
            s_kol_enter[k][warp_id] = local_kol_enter[k];
        }
    }

    __syncthreads();

    // Запись результатов
    if (threadIdx.x < 4) {
        float total_pheromon = 0.0f;
        float total_kol_enter = 0.0f;

        for (int w = 0; w < blockDim.x / V100_WARP_SIZE; w++) {
            total_pheromon += s_pheromon[threadIdx.x][w];
            total_kol_enter += s_kol_enter[threadIdx.x][w];
        }

        if (total_kol_enter > 0.0f) {
            atomicAdd(&kol_enter[base_idx + threadIdx.x], total_kol_enter);
        }

        if (total_pheromon > 0.0f) {
            atomicAdd(&pheromon[base_idx + threadIdx.x], total_pheromon);
        }
    }
}

__device__ void computeProbabilities_v100_dev(const float* __restrict__ pheromon,
    const float* __restrict__ kol_enter,
    float* __restrict__ norm_matrix_probability) {
    const int param_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (param_idx >= PARAMETR_SIZE) return;

    const int base_idx = param_idx * 4;

    // Используем shared memory для быстрого доступа
    __shared__ Float4_v100 s_pheromon[256];
    __shared__ Float4_v100 s_kol_enter[256];

    int local_idx = threadIdx.x;

    // Загрузка данных в shared memory
    if (local_idx < 256 && param_idx + local_idx < PARAMETR_SIZE) {
        int load_idx = (param_idx + local_idx) * 4;
        s_pheromon[local_idx] = *reinterpret_cast<const Float4_v100*>(&pheromon[load_idx]);
        s_kol_enter[local_idx] = *reinterpret_cast<const Float4_v100*>(&kol_enter[load_idx]);
    }
    __syncthreads();

    // Каждый поток обрабатывает свой параметр
    if (param_idx < PARAMETR_SIZE) {
        Float4_v100 p = s_pheromon[threadIdx.x];
        Float4_v100 k = s_kol_enter[threadIdx.x];

        // Вычисление суммы феромонов
        float sum = p.x + p.y + p.z + p.w;
        float inv_sum = (sum > 1e-10f) ? 1.0f / sum : 0.0f;

        // Нормализация феромонов
        p.x *= inv_sum;
        p.y *= inv_sum;
        p.z *= inv_sum;
        p.w *= inv_sum;

        // Вычисление вероятностей
        float prob[4];
        prob[0] = (k.x > 0.0f && p.x > 0.0f) ? 1.0f / k.x + p.x : 0.0f;
        prob[1] = (k.y > 0.0f && p.y > 0.0f) ? 1.0f / k.y + p.y : 0.0f;
        prob[2] = (k.z > 0.0f && p.z > 0.0f) ? 1.0f / k.z + p.z : 0.0f;
        prob[3] = (k.w > 0.0f && p.w > 0.0f) ? 1.0f / k.w + p.w : 0.0f;

        // Нормализация
        float prob_sum = prob[0] + prob[1] + prob[2] + prob[3];
        float inv_prob_sum = (prob_sum > 1e-10f) ? 1.0f / prob_sum : 0.0f;

        // Кумулятивная сумма
        float cumulative = 0.0f;
        cumulative += prob[0] * inv_prob_sum;
        norm_matrix_probability[base_idx] = cumulative;

        cumulative += prob[1] * inv_prob_sum;
        norm_matrix_probability[base_idx + 1] = cumulative;

        cumulative += prob[2] * inv_prob_sum;
        norm_matrix_probability[base_idx + 2] = cumulative;

        norm_matrix_probability[base_idx + 3] = 1.0f;
    }
}

// ==================== ОБЪЕДИНЕННОЕ ЯДРО ДЛЯ V100 ====================
__device__ void updatePheromonesAndProbabilities_v100(const float* __restrict__ OF,
    const int* __restrict__ agent_node,
    float* __restrict__ pheromon,
    float* __restrict__ kol_enter,
    float* __restrict__ norm_matrix_probability) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = threadIdx.x / V100_WARP_SIZE;
    const int lane_id = threadIdx.x % V100_WARP_SIZE;

    if (tid >= PARAMETR_SIZE) return;

    const int base_idx = tid * 4;

    // 1. Загрузка и испарение феромонов
    Float4_v100 pheromon_vec = *reinterpret_cast<Float4_v100*>(&pheromon[base_idx]);
    Float4_v100 kol_enter_vec = *reinterpret_cast<Float4_v100*>(&kol_enter[base_idx]);

    pheromon_vec = pheromon_vec * PARAMETR_RO;
    *reinterpret_cast<Float4_v100*>(&pheromon[base_idx]) = pheromon_vec;

    // 2. Вычисление дельт
    ThreadData_v100 thread_data;
    thread_data.pheromon_current = pheromon_vec;
    thread_data.kol_enter_current = kol_enter_vec;

    const int total_threads = gridDim.x * blockDim.x;
    const int agents_per_thread = (ANT_SIZE + total_threads - 1) / total_threads;
    const int start_agent = tid * agents_per_thread;
    const int end_agent = min(start_agent + agents_per_thread, ANT_SIZE);

    for (int ant_id = start_agent; ant_id < end_agent; ant_id++) {
        if (OF[ant_id] != ZERO_HASH_RESULT) {
            const int k = agent_node[ant_id * PARAMETR_SIZE + tid];

            if (k >= 0 && k < 4) {
                thread_data.kol_enter_delta[k] += 1.0f;

#if OPTIMIZE_MIN_1
                float delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[ant_id];
                if (delta > 0.0f) thread_data.pheromon_delta[k] += PARAMETR_Q * delta;
#elif OPTIMIZE_MIN_2
                float of_val = fmaxf(OF[ant_id], 1e-7f);
                thread_data.pheromon_delta[k] += PARAMETR_Q / of_val;
#elif OPTIMIZE_MAX
                thread_data.pheromon_delta[k] += PARAMETR_Q * OF[ant_id];
#endif
            }
        }
    }

    // 3. Редукция внутри warp
    __shared__ float s_pheromon_delta[4][32];
    __shared__ float s_kol_enter_delta[4][32];

    if (lane_id < 4) {
        s_pheromon_delta[lane_id][warp_id] = 0.0f;
        s_kol_enter_delta[lane_id][warp_id] = 0.0f;
    }

    for (int k = 0; k < 4; k++) {
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_data.pheromon_delta[k] += __shfl_down_sync(0xFFFFFFFF, thread_data.pheromon_delta[k], offset);
            thread_data.kol_enter_delta[k] += __shfl_down_sync(0xFFFFFFFF, thread_data.kol_enter_delta[k], offset);
        }

        if (lane_id == 0) {
            s_pheromon_delta[k][warp_id] = thread_data.pheromon_delta[k];
            s_kol_enter_delta[k][warp_id] = thread_data.kol_enter_delta[k];
        }
    }

    __syncthreads();

    // 4. Глобальное обновление
    if (threadIdx.x < 4) {
        float total_pheromon_delta = 0.0f;
        float total_kol_enter_delta = 0.0f;

        for (int w = 0; w < blockDim.x / V100_WARP_SIZE; w++) {
            total_pheromon_delta += s_pheromon_delta[threadIdx.x][w];
            total_kol_enter_delta += s_kol_enter_delta[threadIdx.x][w];
        }

        if (total_kol_enter_delta > 0.0f) {
            atomicAdd(&kol_enter[base_idx + threadIdx.x], total_kol_enter_delta);
        }

        if (total_pheromon_delta > 0.0f) {
            atomicAdd(&pheromon[base_idx + threadIdx.x], total_pheromon_delta);
        }
    }

    __syncthreads();

    // 5. Вычисление вероятностей
    Float4_v100 final_pheromon = *reinterpret_cast<Float4_v100*>(&pheromon[base_idx]);
    Float4_v100 final_kol_enter = *reinterpret_cast<Float4_v100*>(&kol_enter[base_idx]);

    float sum = final_pheromon.x + final_pheromon.y + final_pheromon.z + final_pheromon.w;

    if (sum > 1e-10f) {
        float inv_sum = 1.0f / sum;
        final_pheromon.x *= inv_sum;
        final_pheromon.y *= inv_sum;
        final_pheromon.z *= inv_sum;
        final_pheromon.w *= inv_sum;
    }

    float prob[4];
    prob[0] = (final_kol_enter.x > 0.0f && final_pheromon.x > 0.0f) ? 1.0f / final_kol_enter.x + final_pheromon.x : 0.0f;
    prob[1] = (final_kol_enter.y > 0.0f && final_pheromon.y > 0.0f) ? 1.0f / final_kol_enter.y + final_pheromon.y : 0.0f;
    prob[2] = (final_kol_enter.z > 0.0f && final_pheromon.z > 0.0f) ? 1.0f / final_kol_enter.z + final_pheromon.z : 0.0f;
    prob[3] = (final_kol_enter.w > 0.0f && final_pheromon.w > 0.0f) ? 1.0f / final_kol_enter.w + final_pheromon.w : 0.0f;

    float prob_sum = prob[0] + prob[1] + prob[2] + prob[3];

    if (prob_sum > 1e-10f) {
        float inv_prob_sum = 1.0f / prob_sum;
        float cumulative = 0.0f;

        cumulative += prob[0] * inv_prob_sum;
        norm_matrix_probability[base_idx] = cumulative;

        cumulative += prob[1] * inv_prob_sum;
        norm_matrix_probability[base_idx + 1] = cumulative;

        cumulative += prob[2] * inv_prob_sum;
        norm_matrix_probability[base_idx + 2] = cumulative;

        norm_matrix_probability[base_idx + 3] = 1.0f;
    }
    else {
        norm_matrix_probability[base_idx] = 0.25f;
        norm_matrix_probability[base_idx + 1] = 0.5f;
        norm_matrix_probability[base_idx + 2] = 0.75f;
        norm_matrix_probability[base_idx + 3] = 1.0f;
    }
}

// ==================== ЯДРО ОПТИМИЗАЦИИ МУРАВЬЕВ (С ДИНАМИЧЕСКОЙ SHARED MEMORY) ====================
__device__ void antColonyOptimization_v100_dev(const float* __restrict__ dev_parametr_value,
    const float* __restrict__ norm_matrix_probability,
    int* __restrict__ agent_node,
    float* __restrict__ OF,
    HashEntry* __restrict__ hashTable,
    float* __restrict__ maxOf_dev,
    float* __restrict__ minOf_dev,
    int* __restrict__ kol_hash_fail,
    float* __restrict__ global_params_buffer) {

    extern __shared__ char shared_memory[];

    const int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_id >= ANT_SIZE) return;

    // Вычисляем, сколько shared memory нужно для одного потока
    const int doubles_per_thread = PARAMETR_SIZE;
    const int thread_offset = threadIdx.x * doubles_per_thread;

    // Проверяем, что у нас достаточно shared memory
    if (threadIdx.x * doubles_per_thread + doubles_per_thread > blockDim.x * doubles_per_thread) {
        return; // Выходим, если не хватает памяти
    }

    double* agent_params_local = reinterpret_cast<double*>(&shared_memory[thread_offset * sizeof(double)]);

    curandState state;
    curand_init(clock64() + ant_id, 0, 0, &state);

    int* agent_node_ptr = &agent_node[ant_id * PARAMETR_SIZE];

    // Генерация пути муравья
    for (int param_idx = 0; param_idx < PARAMETR_SIZE; param_idx++) {
        float randomValue = curand_uniform(&state);
        const int base_prob_idx = param_idx * 4;

        int selected_index = 3;  // По умолчанию последнее значение
        float p0 = norm_matrix_probability[base_prob_idx];
        if (randomValue <= p0) {
            selected_index = 0;
        }
        else {
            float p1 = norm_matrix_probability[base_prob_idx + 1];
            if (randomValue <= p1) {
                selected_index = 1;
            }
            else {
                float p2 = norm_matrix_probability[base_prob_idx + 2];
                if (randomValue <= p2) {
                    selected_index = 2;
                }
            }
        }
        agent_node_ptr[param_idx] = selected_index;

        float param_value = dev_parametr_value[base_prob_idx + selected_index];
        global_params_buffer[ant_id * PARAMETR_SIZE + param_idx] = param_value;

        // Сохраняем в shared memory для вычислений
        agent_params_local[param_idx] = (double)param_value;
    }

    // Вычисление целевой функции
    float cached = getCachedResult_v100(hashTable, agent_node, ant_id);

    if (cached < 0.0f) {
        float result = (float)BenchShafferaFunction_v100(agent_params_local);
        OF[ant_id] = result;
        saveToCache_v100(hashTable, agent_node, ant_id, result);
    }
    else {
        OF[ant_id] = cached;
        atomicAdd(kol_hash_fail, 1);
    }

    if (OF[ant_id] != ZERO_HASH_RESULT) {
        atomicMax_v100(maxOf_dev, OF[ant_id]);
        atomicMin_v100(minOf_dev, OF[ant_id]);
    }
}

// ==================== ГЛОБАЛЬНЫЕ ЯДРА ====================
__global__ void evaporatePheromones_v100(float* __restrict__ pheromon) {
    evaporatePheromones_v100_dev(pheromon);
}

__global__ void depositPheromones_v100(const float* __restrict__ OF,
    const int* __restrict__ agent_node,
    float* __restrict__ pheromon,
    float* __restrict__ kol_enter) {
    depositPheromones_v100_dev(OF, agent_node, pheromon, kol_enter);
}

__global__ void updatePheromones_v100(const float* __restrict__ OF,
    const int* __restrict__ agent_node,
    float* __restrict__ pheromon,
    float* __restrict__ kol_enter) {
    evaporatePheromones_v100_dev(pheromon);
    depositPheromones_v100_dev(OF, agent_node, pheromon, kol_enter);
}

__global__ void computeProbabilities_v100(const float* __restrict__ pheromon,
    const float* __restrict__ kol_enter,
    float* __restrict__ norm_matrix_probability) {
    computeProbabilities_v100_dev(pheromon, kol_enter, norm_matrix_probability);
}

__global__ void antColonyOptimization_v100(const float* __restrict__ dev_parametr_value,
    const float* __restrict__ norm_matrix_probability,
    int* __restrict__ agent_node,
    float* __restrict__ OF,
    HashEntry* __restrict__ hashTable,
    float* __restrict__ maxOf_dev,
    float* __restrict__ minOf_dev,
    int* __restrict__ kol_hash_fail,
    float* __restrict__ global_params_buffer) {
    antColonyOptimization_v100_dev(dev_parametr_value, norm_matrix_probability, agent_node,
        OF, hashTable, maxOf_dev, minOf_dev, kol_hash_fail,
        global_params_buffer);
}

__global__ void combinedUpdateKernel_v100(const float* __restrict__ OF,
    const int* __restrict__ agent_node,
    float* __restrict__ pheromon,
    float* __restrict__ kol_enter,
    float* __restrict__ norm_matrix_probability) {
    updatePheromonesAndProbabilities_v100(OF, agent_node, pheromon, kol_enter, norm_matrix_probability);
}

// ==================== ГЛОБАЛЬНЫЕ РЕСУРСЫ ====================
static float* dev_pheromon = nullptr, * dev_kol_enter = nullptr, * dev_norm_matrix = nullptr;
static float* dev_OF = nullptr, * dev_max = nullptr, * dev_min = nullptr;
static float* dev_parametr_value = nullptr, * dev_agent_params = nullptr;
static int* dev_agent_node = nullptr, * dev_hash_fail = nullptr;
static HashEntry* dev_hashTable = nullptr;
static cudaStream_t compute_stream = nullptr;

// ==================== УПРАВЛЕНИЕ ПАМЯТЬЮ ====================
bool initialize_cuda_resources_v100(const float* params, const float* pheromon, const float* kol_enter) {
    CUDA_CHECK(cudaStreamCreate(&compute_stream));

    // Оптимизируем настройки для V100
    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 128));

    const size_t matrix_size = 4 * PARAMETR_SIZE * sizeof(float);
    const size_t ant_matrix_size = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    const size_t agent_params_size = PARAMETR_SIZE * ANT_SIZE * sizeof(float);

    // Выделяем память с выравниванием
    CUDA_CHECK(cudaMallocAsync(&dev_parametr_value, matrix_size, compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_agent_params, agent_params_size, compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_pheromon, matrix_size, compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_kol_enter, matrix_size, compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_norm_matrix, matrix_size, compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_agent_node, ant_matrix_size, compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_OF, ANT_SIZE * sizeof(float), compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_max, sizeof(float), compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_min, sizeof(float), compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_hash_fail, sizeof(int), compute_stream));
    CUDA_CHECK(cudaMallocAsync(&dev_hashTable, HASH_TABLE_SIZE * sizeof(HashEntry), compute_stream));

    // Используем pinned memory для быстрого копирования
    CUDA_CHECK(cudaMemcpyAsync(dev_parametr_value, params, matrix_size,
        cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_pheromon, pheromon, matrix_size,
        cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_kol_enter, kol_enter, matrix_size,
        cudaMemcpyHostToDevice, compute_stream));

    // Инициализация хэш-таблицы
    int threads = min(1024, HASH_TABLE_SIZE);
    int blocks = (HASH_TABLE_SIZE + threads - 1) / threads;
    initializeHashTable_v100 << <blocks, threads, 0, compute_stream >> > (dev_hashTable);

    return cudaStreamSynchronize(compute_stream) == cudaSuccess;
}

void cleanup_cuda_resources_v100() {
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

// ==================== ФУНКЦИИ ВЫПОЛНЕНИЯ ИТЕРАЦИЙ ====================
PerformanceMetrics run_aco_iterations4_v100(int num_iterations) {
    PerformanceMetrics metrics = { 0 };
    auto total_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Инициализация статистики
    float max_init = -1e9f, min_init = 1e9f;
    int fail_init = 0;

    CUDA_CHECK(cudaMemcpyAsync(dev_max, &max_init, sizeof(float), cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_min, &min_init, sizeof(float), cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_hash_fail, &fail_init, sizeof(int), cudaMemcpyHostToDevice, compute_stream));

    float kernel_time = 0.0f;
    float compute_time = 0.0f;
    float ant_time = 0.0f;
    float update_time = 0.0f;

    // Оптимальные размеры сетки для V100
    int blocks_prob = (PARAMETR_SIZE + OPTIMAL_BLOCK_SIZE - 1) / OPTIMAL_BLOCK_SIZE;
    int blocks_ant = (ANT_SIZE + OPTIMAL_ANT_BLOCK_SIZE - 1) / OPTIMAL_ANT_BLOCK_SIZE;
    int blocks_update = (PARAMETR_SIZE + OPTIMAL_UPDATE_BLOCK_SIZE - 1) / OPTIMAL_UPDATE_BLOCK_SIZE;

    // Ограничение блоков для лучшего использования SMs
    blocks_prob = min(blocks_prob, V100_SM_COUNT * 4);
    blocks_ant = min(blocks_ant, V100_SM_COUNT * 2);
    blocks_update = min(blocks_update, V100_SM_COUNT * 4);

    for (int iter = 0; iter < num_iterations; ++iter) {
        float iter_time;

        // Этап 1: Вычисление вероятностей
        CUDA_CHECK(cudaEventRecord(start, compute_stream));
        computeProbabilities_v100 << <blocks_prob, OPTIMAL_BLOCK_SIZE, 0, compute_stream >> > (
            dev_pheromon, dev_kol_enter, dev_norm_matrix);
        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        compute_time += iter_time;

        size_t shared_mem_size = OPTIMAL_ANT_BLOCK_SIZE * PARAMETR_SIZE * sizeof(double);
        // Этап 2: Оптимизация муравьями
        CUDA_CHECK(cudaEventRecord(start, compute_stream));
        antColonyOptimization_v100 << <blocks_ant, OPTIMAL_ANT_BLOCK_SIZE, shared_mem_size, compute_stream >> > (
            dev_parametr_value, dev_norm_matrix, dev_agent_node, dev_OF, dev_hashTable,
            dev_max, dev_min, dev_hash_fail, dev_agent_params);
        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        ant_time += iter_time;

        // Этап 3: Обновление феромонов
        CUDA_CHECK(cudaEventRecord(start, compute_stream));
        evaporatePheromones_v100 << <blocks_update, OPTIMAL_UPDATE_BLOCK_SIZE, 0, compute_stream >> > (
            dev_pheromon);
        depositPheromones_v100 << <blocks_update, OPTIMAL_UPDATE_BLOCK_SIZE, 0, compute_stream >> > (
            dev_OF, dev_agent_node, dev_pheromon, dev_kol_enter);
        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        update_time += iter_time;
        kernel_time += iter_time;
    }

    // Сбор результатов
    float best_fitness, low_fitness;
    int hash_fails;
    CUDA_CHECK(cudaMemcpyAsync(&best_fitness, dev_min, sizeof(float), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(&low_fitness, dev_max, sizeof(float), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(&hash_fails, dev_hash_fail, sizeof(int), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    metrics.min_fitness = best_fitness;
    metrics.max_fitness = low_fitness;
    metrics.hash_misses = hash_fails;
    metrics.hash_hits = num_iterations * ANT_SIZE - hash_fails;

    // Расчет occupancy
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int warps_per_block = (OPTIMAL_BLOCK_SIZE + prop.warpSize - 1) / prop.warpSize;
    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    metrics.occupancy = std::min(100.0f, (warps_per_block * 32.0f / max_warps_per_sm) * 100.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    auto total_end = std::chrono::high_resolution_clock::now();
    metrics.total_time_ms = std::chrono::duration<float, std::milli>(total_end - total_start).count();
    metrics.kernel_time_ms = kernel_time;
    metrics.computeProbabilities_time_ms = compute_time;
    metrics.antColonyOptimization_time_ms = ant_time;
    metrics.updatePheromones_time_ms = update_time;
    metrics.memory_time_ms = metrics.total_time_ms - kernel_time;

    // Расчет пропускной способности памяти
    size_t total_data_transferred = (4 * PARAMETR_SIZE * sizeof(float)) * 4 * num_iterations +
        (PARAMETR_SIZE * ANT_SIZE * sizeof(int)) * num_iterations +
        (ANT_SIZE * sizeof(float)) * num_iterations;
    metrics.memory_throughput_gbs = (total_data_transferred / (1024.0 * 1024.0 * 1024.0)) /
        (metrics.total_time_ms / 1000.0);

    return metrics;
}

PerformanceMetrics run_aco_iterations3_v100(int num_iterations) {
    PerformanceMetrics metrics = { 0 };
    auto total_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float max_init = -1e9f, min_init = 1e9f;
    int fail_init = 0;

    CUDA_CHECK(cudaMemcpyAsync(dev_max, &max_init, sizeof(float), cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_min, &min_init, sizeof(float), cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_hash_fail, &fail_init, sizeof(int), cudaMemcpyHostToDevice, compute_stream));

    float kernel_time = 0.0f;
    float compute_time = 0.0f;
    float ant_time = 0.0f;
    float update_time = 0.0f;

    int blocks_prob = (PARAMETR_SIZE + OPTIMAL_BLOCK_SIZE - 1) / OPTIMAL_BLOCK_SIZE;
    int blocks_ant = (ANT_SIZE + OPTIMAL_ANT_BLOCK_SIZE - 1) / OPTIMAL_ANT_BLOCK_SIZE;
    int blocks_update = (PARAMETR_SIZE + OPTIMAL_UPDATE_BLOCK_SIZE - 1) / OPTIMAL_UPDATE_BLOCK_SIZE;

    blocks_prob = min(blocks_prob, V100_SM_COUNT * 4);
    blocks_ant = min(blocks_ant, V100_SM_COUNT * 2);
    blocks_update = min(blocks_update, V100_SM_COUNT * 4);

    for (int iter = 0; iter < num_iterations; ++iter) {
        float iter_time;

        CUDA_CHECK(cudaEventRecord(start, compute_stream));
        computeProbabilities_v100 << <blocks_prob, OPTIMAL_BLOCK_SIZE, 0, compute_stream >> > (
            dev_pheromon, dev_kol_enter, dev_norm_matrix);
        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        compute_time += iter_time;

        CUDA_CHECK(cudaEventRecord(start, compute_stream));
        antColonyOptimization_v100 << <blocks_ant, OPTIMAL_ANT_BLOCK_SIZE, 0, compute_stream >> > (
            dev_parametr_value, dev_norm_matrix, dev_agent_node, dev_OF, dev_hashTable,
            dev_max, dev_min, dev_hash_fail, dev_agent_params);
        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        ant_time += iter_time;

        CUDA_CHECK(cudaEventRecord(start, compute_stream));
        updatePheromones_v100 << <blocks_update, OPTIMAL_UPDATE_BLOCK_SIZE, 0, compute_stream >> > (
            dev_OF, dev_agent_node, dev_pheromon, dev_kol_enter);
        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        update_time += iter_time;
        kernel_time += iter_time;
    }

    float best_fitness, low_fitness;
    int hash_fails;
    CUDA_CHECK(cudaMemcpyAsync(&best_fitness, dev_min, sizeof(float), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(&low_fitness, dev_max, sizeof(float), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(&hash_fails, dev_hash_fail, sizeof(int), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    metrics.min_fitness = best_fitness;
    metrics.max_fitness = low_fitness;
    metrics.hash_misses = hash_fails;
    metrics.hash_hits = num_iterations * ANT_SIZE - hash_fails;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int warps_per_block = (OPTIMAL_BLOCK_SIZE + prop.warpSize - 1) / prop.warpSize;
    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    metrics.occupancy = std::min(100.0f, (warps_per_block * 32.0f / max_warps_per_sm) * 100.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    auto total_end = std::chrono::high_resolution_clock::now();
    metrics.total_time_ms = std::chrono::duration<float, std::milli>(total_end - total_start).count();
    metrics.kernel_time_ms = kernel_time;
    metrics.computeProbabilities_time_ms = compute_time;
    metrics.antColonyOptimization_time_ms = ant_time;
    metrics.updatePheromones_time_ms = update_time;
    metrics.memory_time_ms = metrics.total_time_ms - kernel_time;

    size_t total_data_transferred = (4 * PARAMETR_SIZE * sizeof(float)) * 4 * num_iterations +
        (PARAMETR_SIZE * ANT_SIZE * sizeof(int)) * num_iterations +
        (ANT_SIZE * sizeof(float)) * num_iterations;
    metrics.memory_throughput_gbs = (total_data_transferred / (1024.0 * 1024.0 * 1024.0)) /
        (metrics.total_time_ms / 1000.0);

    return metrics;
}

PerformanceMetrics run_aco_iterations2_v100(int num_iterations) {
    PerformanceMetrics metrics = { 0 };
    auto total_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float max_init = -1e9f, min_init = 1e9f;
    int fail_init = 0;

    CUDA_CHECK(cudaMemcpyAsync(dev_max, &max_init, sizeof(float), cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_min, &min_init, sizeof(float), cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_hash_fail, &fail_init, sizeof(int), cudaMemcpyHostToDevice, compute_stream));

    float kernel_time = 0.0f;
    float compute_time = 0.0f;
    float ant_time = 0.0f;
    float update_time = 0.0f;

    int blocks_prob = (PARAMETR_SIZE + OPTIMAL_BLOCK_SIZE - 1) / OPTIMAL_BLOCK_SIZE;
    int blocks_ant = (ANT_SIZE + OPTIMAL_ANT_BLOCK_SIZE - 1) / OPTIMAL_ANT_BLOCK_SIZE;
    int blocks_update = (PARAMETR_SIZE + OPTIMAL_UPDATE_BLOCK_SIZE - 1) / OPTIMAL_UPDATE_BLOCK_SIZE;

    blocks_prob = min(blocks_prob, V100_SM_COUNT * 4);
    blocks_ant = min(blocks_ant, V100_SM_COUNT * 2);
    blocks_update = min(blocks_update, V100_SM_COUNT * 4);

    for (int iter = 0; iter < num_iterations; ++iter) {
        float iter_time;

        CUDA_CHECK(cudaEventRecord(start, compute_stream));
        combinedUpdateKernel_v100 << <blocks_update, OPTIMAL_UPDATE_BLOCK_SIZE, 0, compute_stream >> > (
            dev_OF, dev_agent_node, dev_pheromon, dev_kol_enter, dev_norm_matrix);
        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        compute_time += iter_time;

        CUDA_CHECK(cudaEventRecord(start, compute_stream));
        antColonyOptimization_v100 << <blocks_ant, OPTIMAL_ANT_BLOCK_SIZE, 0, compute_stream >> > (
            dev_parametr_value, dev_norm_matrix, dev_agent_node, dev_OF, dev_hashTable,
            dev_max, dev_min, dev_hash_fail, dev_agent_params);
        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        ant_time += iter_time;
        kernel_time += iter_time;
    }

    float best_fitness, low_fitness;
    int hash_fails;
    CUDA_CHECK(cudaMemcpyAsync(&best_fitness, dev_min, sizeof(float), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(&low_fitness, dev_max, sizeof(float), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(&hash_fails, dev_hash_fail, sizeof(int), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    metrics.min_fitness = best_fitness;
    metrics.max_fitness = low_fitness;
    metrics.hash_misses = hash_fails;
    metrics.hash_hits = num_iterations * ANT_SIZE - hash_fails;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int warps_per_block = (OPTIMAL_BLOCK_SIZE + prop.warpSize - 1) / prop.warpSize;
    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    metrics.occupancy = std::min(100.0f, (warps_per_block * 32.0f / max_warps_per_sm) * 100.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    auto total_end = std::chrono::high_resolution_clock::now();
    metrics.total_time_ms = std::chrono::duration<float, std::milli>(total_end - total_start).count();
    metrics.kernel_time_ms = kernel_time;
    metrics.computeProbabilities_time_ms = compute_time;
    metrics.antColonyOptimization_time_ms = ant_time;
    metrics.updatePheromones_time_ms = update_time;
    metrics.memory_time_ms = metrics.total_time_ms - kernel_time;

    size_t total_data_transferred = (4 * PARAMETR_SIZE * sizeof(float)) * 4 * num_iterations +
        (PARAMETR_SIZE * ANT_SIZE * sizeof(int)) * num_iterations +
        (ANT_SIZE * sizeof(float)) * num_iterations;
    metrics.memory_throughput_gbs = (total_data_transferred / (1024.0 * 1024.0 * 1024.0)) /
        (metrics.total_time_ms / 1000.0);

    return metrics;
}

void launchCombinedUpdateV100(cudaStream_t stream, const float* d_OF, const int* d_agent_node,
    float* d_pheromon, float* d_kol_enter, float* d_norm_matrix) {

    int block_size = 256;
    int grid_size = (PARAMETR_SIZE + block_size - 1) / block_size;
    grid_size = min(grid_size, V100_SM_COUNT * 4);

    combinedUpdateKernel_v100 << <grid_size, block_size, 0, stream >> > (
        d_OF, d_agent_node, d_pheromon, d_kol_enter, d_norm_matrix);

    CUDA_CHECK(cudaGetLastError());
}

// ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
bool load_matrix_data_v100(const std::string& filename, std::vector<float>& params,
    std::vector<float>& pheromones, std::vector<float>& visits) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    size_t total_size = 4 * PARAMETR_SIZE;
    params.resize(total_size);
    pheromones.resize(total_size);
    visits.resize(total_size);

    for (size_t i = 0; i < total_size; ++i) {
        double value;
        if (!(file >> value)) {
            std::cerr << "Error reading element " << i << std::endl;
            return false;
        }

        params[i] = static_cast<float>(value);
        if (value != -100.0) {
            pheromones[i] = 1.0f;
            visits[i] = 1.0f;
        }
        else {
            params[i] = pheromones[i] = visits[i] = 0.0f;
        }
    }

    file.close();
    return true;
}

void print_v100_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        std::cout << "\n=== TESLA V100 GPU Information ===" << std::endl;
        std::cout << "Device " << device << ": " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)
            << " GB" << std::endl;
        std::cout << "SMs: " << prop.multiProcessorCount << std::endl;
        std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "Warp Size: " << prop.warpSize << std::endl;
        std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000.0 << " MHz" << std::endl;
        std::cout << "===================================" << std::endl;

        logFile << "\nTESLA V100 GPU Information:" << std::endl;
        logFile << "Device: " << prop.name << std::endl;
        logFile << "Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
        logFile << "SMs: " << prop.multiProcessorCount << std::endl;
        logFile << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    }
}

void print_optimization_info_v100() {
    std::cout << "\n=== Optimization Settings for TESLA V100 ===" << std::endl;
    std::cout << "MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << " (optimized for 4)" << std::endl;
    std::cout << "PARAMETR_SIZE: " << PARAMETR_SIZE << std::endl;
    std::cout << "ANT_SIZE: " << ANT_SIZE << std::endl;
    std::cout << "OPTIMAL_BLOCK_SIZE: " << OPTIMAL_BLOCK_SIZE << std::endl;
    std::cout << "OPTIMAL_ANT_BLOCK_SIZE: " << OPTIMAL_ANT_BLOCK_SIZE << std::endl;
    std::cout << "Using vectorized operations for 4 values" << std::endl;
    std::cout << "Using shared memory optimizations for V100" << std::endl;
    std::cout << "Using warp-level reductions" << std::endl;
    std::cout << "Using prefetch instructions for better cache utilization" << std::endl;
    std::cout << "=============================================" << std::endl;

    logFile << "\nOptimization Settings for V100:" << std::endl;
    logFile << "MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << std::endl;
    logFile << "PARAMETR_SIZE: " << PARAMETR_SIZE << std::endl;
    logFile << "ANT_SIZE: " << ANT_SIZE << std::endl;
}

void print_metrics_v100(const PerformanceMetrics& metrics, const char* str, int run_id) {
    std::cout << "V100 Run " << str << run_id << ": "
        << "Time=" << metrics.total_time_ms << "ms "
        << "Kernel=" << metrics.kernel_time_ms << "ms "
        << "Prob=" << metrics.computeProbabilities_time_ms << "ms "
        << "Ant=" << metrics.antColonyOptimization_time_ms << "ms "
        << "Update=" << metrics.updatePheromones_time_ms << "ms "
        << "Memory=" << metrics.memory_time_ms << "ms "
        << "HitRate=" << (100.0 * metrics.hash_hits / (metrics.hash_hits + metrics.hash_misses)) << "% "
        << "Occupancy=" << metrics.occupancy << "% "
        << "Throughput=" << metrics.memory_throughput_gbs << " GB/s "
        << "MIN=" << metrics.min_fitness << " "
        << "MAX=" << metrics.max_fitness << " "
        << std::endl;

    logFile << "V100 Run " << str << run_id << "; "
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

// ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
int main() {
    std::cout << "Initializing CUDA ACO Optimizer for TESLA V100..." << std::endl;

    print_v100_info();
    print_optimization_info_v100();

    // Загрузка данных
    std::vector<float> params, pheromones, visits;
    if (!load_matrix_data_v100(NAME_FILE_GRAPH, params, pheromones, visits)) {
        std::cerr << "Failed to load matrix data" << std::endl;
        return 1;
    }

    // Проверка выделения памяти
    size_t required_memory = (4 * PARAMETR_SIZE * sizeof(float) * 4) +
        (PARAMETR_SIZE * ANT_SIZE * sizeof(int)) +
        (PARAMETR_SIZE * ANT_SIZE * sizeof(float)) +
        (ANT_SIZE * sizeof(float) * 2) +
        (HASH_TABLE_SIZE * sizeof(HashEntry));

    std::cout << "\n=== Memory Requirements ===" << std::endl;
    std::cout << "Required memory: " << required_memory / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    std::cout << "TESLA V100 has: 16 GB HBM2" << std::endl;
    std::cout << "Usage: " << (required_memory / (16.0 * 1024.0 * 1024.0 * 1024.0)) * 100.0 << "%" << std::endl;

    // Инициализация CUDA
    if (!initialize_cuda_resources_v100(params.data(), pheromones.data(), visits.data())) {
        std::cerr << "CUDA resources initialization failed" << std::endl;
        return 1;
    }

    // Прогрев GPU
    std::cout << "\n=== Warming up GPU ===" << std::endl;
    for (int i = 0; i < KOL_PROGREV; ++i) {
        std::cout << "Warmup iteration " << i + 1 << "/" << KOL_PROGREV << std::endl;
        auto warmup_metrics = run_aco_iterations4_v100(KOL_ITERATION);
    }
    std::cout << "Warmup completed" << std::endl;

    // Основные запуски
    std::cout << "\n=== Starting Optimized ACO Runs for V100 ===" << std::endl;

    auto total_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < KOL_PROGON_STATISTICS; ++i) {
        auto metrics = run_aco_iterations4_v100(KOL_ITERATION);
        print_metrics_v100(metrics, "V100_Optimized_4", i);
    }
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    std::cout << "Total time for 4-phase runs: " << total_duration.count() << " ms" << std::endl;

    std::cout << "\n=== Starting 3-phase ACO Runs ===" << std::endl;
    total_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < KOL_PROGON_STATISTICS; ++i) {
        auto metrics = run_aco_iterations3_v100(KOL_ITERATION);
        print_metrics_v100(metrics, "V100_Optimized_3", i);
    }
    total_end = std::chrono::high_resolution_clock::now();
    total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    std::cout << "Total time for 3-phase runs: " << total_duration.count() << " ms" << std::endl;

    std::cout << "\n=== Starting Combined ACO Runs ===" << std::endl;
    total_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < KOL_PROGON_STATISTICS; ++i) {
        auto metrics = run_aco_iterations2_v100(KOL_ITERATION);
        print_metrics_v100(metrics, "V100_Optimized_2", i);
    }
    total_end = std::chrono::high_resolution_clock::now();
    total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    std::cout << "Total time for combined runs: " << total_duration.count() << " ms" << std::endl;

    // Очистка
    cleanup_cuda_resources_v100();
    logFile.close();

    std::cout << "\nOptimization completed successfully for TESLA V100!" << std::endl;
    return 0;
}