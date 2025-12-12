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
    float value;  // Используем float для экономии памяти
};

struct PerformanceMetrics {
    float total_time_ms, kernel_time_ms, memory_time_ms;
    float computeProbabilities_time_ms, antColonyOptimization_time_ms, updatePheromones_time_ms;
    float occupancy, memory_throughput_gbs;
    double min_fitness, max_fitness;
    int hash_hits, hash_misses;
};

// ==================== КОНСТАНТЫ ДЛЯ RTX 3060 ====================
// RTX 3060: 28 SMs, 128 CUDA cores per SM, 3584 total cores
const int RTX3060_SM_COUNT = 28;
const int RTX3060_WARP_SIZE = 32;
const int RTX3060_MAX_THREADS_PER_SM = 1536;
const int RTX3060_MAX_THREADS_PER_BLOCK = 1024;
const int RTX3060_SHARED_MEM_PER_BLOCK = 48 * 1024;  // 48KB

// Оптимальные размеры блоков для RTX 3060

const int OPTIMAL_BLOCK_SIZE = 256;  // Лучший occupancy для 4 значений
const int OPTIMAL_ANT_BLOCK_SIZE = 128;  // Для работы с агентами

// ==================== ОПТИМИЗИРОВАННЫЕ ФУНКЦИИ ДЛЯ 4 ЗНАЧЕНИЙ ====================

// Используем векторные типы для 4 значений
struct alignas(32) Float4 {
    float x, y, z, w;

    __device__ __host__ Float4 operator*(float scalar) const {
        return { x * scalar, y * scalar, z * scalar, w * scalar };
    }

    __device__ __host__ Float4 operator+(const Float4& other) const {
        return { x + other.x, y + other.y, z + other.z, w + other.w };
    }
};

// Структура для хранения промежуточных данных в регистрах
struct ThreadData {
    float pheromon_delta[4];
    float kol_enter_delta[4];
    Float4 pheromon_current;
    Float4 kol_enter_current;

    __device__ ThreadData() {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            pheromon_delta[i] = 0.0f;
            kol_enter_delta[i] = 0.0f;
        }
    }
};

__device__ __forceinline__ double go_x(const double* __restrict__ parametr, int start_index) {
    double sum = 0.0;
#pragma unroll
    for (int i = 1; i < SET_PARAMETR_SIZE_ONE_X; ++i) {
        sum += parametr[start_index + i];
    }
    return parametr[start_index] * sum;
}

// ОПТИМИЗИРОВАННЫЕ ВЕРСИИ ФУНКЦИЙ ДЛЯ RTX 3060
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
__device__ __forceinline__ float getCachedResult4(HashEntry* __restrict__ hashTable, const int* __restrict__ agent_node,  int ant_id) {
#if GO_HASH
    unsigned long long key = generateKey4(agent_node, ant_id);
    unsigned long long idx = key % HASH_TABLE_SIZE;

    // Быстрый линейный поиск с малым числом проб
    for (int i = 0; i < 3; i++) {
        if (hashTable[idx].key == key) return hashTable[idx].value;
        if (hashTable[idx].key == ZERO_HASH_RESULT) return -1.0f;
        idx = (idx + 1) % HASH_TABLE_SIZE;
    }
#endif
    return -1.0f;
}
__device__ __forceinline__ void saveToCache4(HashEntry* __restrict__ hashTable, const int* __restrict__ agent_node,  int ant_id, float value) {
#if GO_HASH
    unsigned long long key = generateKey4(agent_node, ant_id);
    unsigned long long idx = key % HASH_TABLE_SIZE;

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

// ==================== АТОМАРНЫЕ ОПЕРАЦИИ ====================
__device__ __forceinline__ void atomicMax(float* address, float value) {
    int* addr_as_int = (int*)address;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        if (value > __int_as_float(assumed))
            old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}
__device__ __forceinline__ void atomicMin(float* address, float value) {
    int* addr_as_int = (int*)address;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        if (value < __int_as_float(assumed))
            old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}

// ==================== ОПТИМИЗИРОВАННЫЕ ЯДРА ДЛЯ RTX 3060 ====================
__global__ void initializeHashTable(HashEntry* hashTable) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = tid; i < HASH_TABLE_SIZE; i += stride) {
        hashTable[i].key = ZERO_HASH_RESULT;
        hashTable[i].value = 0.0f;
    }
}

// ОПТИМИЗИРОВАННОЕ ИСПАРЕНИЕ ФЕРОМОНОВ ДЛЯ 4 ЗНАЧЕНИЙ
__device__ void evaporatePheromones_optimized_dev(float* __restrict__ pheromon) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= PARAMETR_SIZE) return;

    const int base_idx = tid * 4;

    // Векторизованная загрузка и умножение
    Float4* pheromon_vec = reinterpret_cast<Float4*>(&pheromon[base_idx]);
    Float4 current = *pheromon_vec;

    // Умножение на коэффициент испарения
    current = current * PARAMETR_RO;

    // Сохранение обратно
    *pheromon_vec = current;
}

// ОПТИМИЗИРОВАННОЕ ДОБАВЛЕНИЕ ФЕРОМОНОВ ДЛЯ 4 ЗНАЧЕНИЙ
__device__ void depositPheromones_optimized_dev(const float* __restrict__ OF, const int* __restrict__ agent_node, float* __restrict__ pheromon, float* __restrict__ kol_enter) {
    const int param_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (param_idx >= PARAMETR_SIZE) return;

    const int base_idx = param_idx * 4;

    // Используем shared memory для редукции внутри блока
    __shared__ float s_pheromon[4][32];  // [значение][поток в warp]
    __shared__ float s_kol_enter[4][32];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Инициализация shared memory
    if (lane_id < 4) {
        s_pheromon[lane_id][warp_id] = 0.0f;
        s_kol_enter[lane_id][warp_id] = 0.0f;
    }
    __syncthreads();

    // Каждый поток обрабатывает часть агентов
    const int threads_total = gridDim.x * blockDim.x;
    const int agents_per_thread = (ANT_SIZE + threads_total - 1) / threads_total;
    const int start_agent = param_idx * agents_per_thread;
    const int end_agent = min(start_agent + agents_per_thread, ANT_SIZE);

    // Локальное накопление
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

        // Сохранение в shared memory
        if (lane_id == 0) {
            s_pheromon[k][warp_id] = local_pheromon[k];
            s_kol_enter[k][warp_id] = local_kol_enter[k];
        }
    }

    __syncthreads();

    // Запись результатов (только первые 4 потока)
    if (threadIdx.x < 4) {
        float total_pheromon = 0.0f;
        float total_kol_enter = 0.0f;

        // Суммирование по всем warp'ам
        for (int w = 0; w < blockDim.x / 32; w++) {
            total_pheromon += s_pheromon[threadIdx.x][w];
            total_kol_enter += s_kol_enter[threadIdx.x][w];
        }

        // Атомарное добавление
        if (total_kol_enter > 0.0f) {
            atomicAdd(&kol_enter[base_idx + threadIdx.x], total_kol_enter);
        }

        if (total_pheromon > 0.0f) {
            atomicAdd(&pheromon[base_idx + threadIdx.x], total_pheromon);
        }
    }
}

// ОПТИМИЗИРОВАННОЕ ВЫЧИСЛЕНИЕ ВЕРОЯТНОСТЕЙ ДЛЯ 4 ЗНАЧЕНИЙ
__device__ void computeProbabilities_optimized_dev(const float* __restrict__ pheromon, const float* __restrict__ kol_enter, float* __restrict__ norm_matrix_probability) {
    const int param_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (param_idx >= PARAMETR_SIZE) return;

    const int base_idx = param_idx * 4;

    // Используем shared memory для быстрого доступа
    __shared__ Float4 s_pheromon[32];  // 32 параметра на блок
    __shared__ Float4 s_kol_enter[32];
    __shared__ float s_prob[32][4];

    int local_idx = threadIdx.x;

    // Загрузка данных в shared memory
    if (local_idx < 32 && param_idx + local_idx < PARAMETR_SIZE) {
        int load_idx = (param_idx + local_idx) * 4;
        s_pheromon[local_idx] = *reinterpret_cast<const Float4*>(&pheromon[load_idx]);
        s_kol_enter[local_idx] = *reinterpret_cast<const Float4*>(&kol_enter[load_idx]);
    }
    __syncthreads();

    // Каждый поток обрабатывает свой параметр
    if (param_idx < PARAMETR_SIZE) {
        Float4 p = s_pheromon[threadIdx.x];
        Float4 k = s_kol_enter[threadIdx.x];

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

        // Сохранение в shared memory для нормализации
#pragma unroll
        for (int i = 0; i < 4; i++) {
            s_prob[threadIdx.x][i] = prob[i];
        }
        __syncthreads();

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

        // Последнее значение всегда 1.0
        norm_matrix_probability[base_idx + 3] = 1.0f;
    }
}

// ОБЪЕДИНЕННОЕ ЯДРО ДЛЯ RTX 3060
__device__ void updatePheromonesAndProbabilities_rtx3060(const float* __restrict__ OF, const int* __restrict__ agent_node, float* __restrict__ pheromon, float* __restrict__ kol_enter, float* __restrict__ norm_matrix_probability) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    if (tid >= PARAMETR_SIZE) return;

    const int base_idx = tid * 4;

    // ========== ЭТАП 1: ЗАГРУЗКА И ИСПАРЕНИЕ ФЕРОМОНОВ ==========

    // Векторизованная загрузка феромонов
    Float4 pheromon_vec = *reinterpret_cast<Float4*>(&pheromon[base_idx]);
    Float4 kol_enter_vec = *reinterpret_cast<Float4*>(&kol_enter[base_idx]);

    // Испарение феромонов (умножение на коэффициент)
    pheromon_vec = pheromon_vec * PARAMETR_RO;

    // Сохраняем испаренные феромоны сразу обратно
    *reinterpret_cast<Float4*>(&pheromon[base_idx]) = pheromon_vec;

    // ========== ЭТАП 2: ВЫЧИСЛЕНИЕ ДЕЛЬТ ДЛЯ ДОБАВЛЕНИЯ ==========

    ThreadData thread_data;
    thread_data.pheromon_current = pheromon_vec;
    thread_data.kol_enter_current = kol_enter_vec;

    // Оптимизация для RTX 3060: каждый поток обрабатывает свою часть агентов
    // RTX 3060 имеет 28 SM, 1536 потоков на SM, оптимально загружаем все SM

    const int total_threads = gridDim.x * blockDim.x;
    const int agents_per_thread = (ANT_SIZE + total_threads - 1) / total_threads;
    const int start_agent = tid * agents_per_thread;
    const int end_agent = min(start_agent + agents_per_thread, ANT_SIZE);

    // Обработка назначенных агентов
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

    // ========== ЭТАП 3: РЕДУКЦИЯ ВНУТРИ WARP ==========

    // Используем shared memory для редукции внутри блока
    __shared__ float s_pheromon_delta[4][32];  // [значение][warp_id]
    __shared__ float s_kol_enter_delta[4][32];

    // Инициализация shared memory
    if (lane_id < 4) {
        s_pheromon_delta[lane_id][warp_id] = 0.0f;
        s_kol_enter_delta[lane_id][warp_id] = 0.0f;
    }

    // Редукция внутри warp с использованием warp shuffle
#pragma unroll
    for (int k = 0; k < 4; k++) {
        // Редукция внутри warp (32 потока)
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_data.pheromon_delta[k] += __shfl_down_sync(0xFFFFFFFF, thread_data.pheromon_delta[k], offset);
            thread_data.kol_enter_delta[k] += __shfl_down_sync(0xFFFFFFFF, thread_data.kol_enter_delta[k], offset);
        }

        // Сохраняем результат каждого warp'а в shared memory
        if (lane_id == 0) {
            s_pheromon_delta[k][warp_id] = thread_data.pheromon_delta[k];
            s_kol_enter_delta[k][warp_id] = thread_data.kol_enter_delta[k];
        }
    }

    __syncthreads();

    // ========== ЭТАП 4: ГЛОБАЛЬНОЕ ОБНОВЛЕНИЕ ==========

    // Только первые 4 потока в блоке обновляют глобальную память
    if (threadIdx.x < 4) {
        float total_pheromon_delta = 0.0f;
        float total_kol_enter_delta = 0.0f;

        // Суммируем результаты всех warp'ов в блоке
        for (int w = 0; w < blockDim.x / 32; w++) {
            total_pheromon_delta += s_pheromon_delta[threadIdx.x][w];
            total_kol_enter_delta += s_kol_enter_delta[threadIdx.x][w];
        }

        // Атомарное обновление глобальной памяти
        if (total_kol_enter_delta > 0.0f) {
            atomicAdd(&kol_enter[base_idx + threadIdx.x], total_kol_enter_delta);
        }

        if (total_pheromon_delta > 0.0f) {
            atomicAdd(&pheromon[base_idx + threadIdx.x], total_pheromon_delta);
        }
    }

    __syncthreads();

    // ========== ЭТАП 5: ПОВТОРНАЯ ЗАГРУЗКА ОБНОВЛЕННЫХ ДАННЫХ ==========

    // Загружаем обновленные данные после атомарных операций
    if (threadIdx.x == 0) {
        // Только первый поток в блоке загружает данные для всех параметров блока
        for (int i = 0; i < min(blockDim.x, PARAMETR_SIZE - blockIdx.x * blockDim.x); i++) {
            int param_idx = blockIdx.x * blockDim.x + i;
            int idx = param_idx * 4;

            Float4* p_vec = reinterpret_cast<Float4*>(&pheromon[idx]);
            Float4* k_vec = reinterpret_cast<Float4*>(&kol_enter[idx]);

            // Сохраняем в shared memory для быстрого доступа
            __shared__ Float4 s_pheromon_updated[256];
            __shared__ Float4 s_kol_enter_updated[256];

            if (i < 256) {
                s_pheromon_updated[i] = *p_vec;
                s_kol_enter_updated[i] = *k_vec;
            }
        }
    }

    __syncthreads();

    // ========== ЭТАП 6: ВЫЧИСЛЕНИЕ ВЕРОЯТНОСТЕЙ ==========

    // Загружаем обновленные данные из shared memory
    __shared__ Float4 s_pheromon_block[256];
    __shared__ Float4 s_kol_enter_block[256];
    __shared__ float s_prob_block[256][4];

    if (threadIdx.x < min(blockDim.x, PARAMETR_SIZE - blockIdx.x * blockDim.x)) {
        int local_idx = threadIdx.x;
        s_pheromon_block[local_idx] = *reinterpret_cast<Float4*>(&pheromon[base_idx]);
        s_kol_enter_block[local_idx] = *reinterpret_cast<Float4*>(&kol_enter[base_idx]);
    }

    __syncthreads();

    // Каждый поток вычисляет вероятности для своего параметра
    if (tid < PARAMETR_SIZE) {
        Float4 p_final = s_pheromon_block[threadIdx.x];
        Float4 k_final = s_kol_enter_block[threadIdx.x];

        // Вычисление суммы феромонов
        float sum = p_final.x + p_final.y + p_final.z + p_final.w;
        float inv_sum = (sum > 1e-10f) ? 1.0f / sum : 0.0f;

        // Нормализация феромонов
        p_final.x *= inv_sum;
        p_final.y *= inv_sum;
        p_final.z *= inv_sum;
        p_final.w *= inv_sum;

        // Вычисление вероятностей
        float prob[4];
        prob[0] = (k_final.x > 0.0f && p_final.x > 0.0f) ? 1.0f / k_final.x + p_final.x : 0.0f;
        prob[1] = (k_final.y > 0.0f && p_final.y > 0.0f) ? 1.0f / k_final.y + p_final.y : 0.0f;
        prob[2] = (k_final.z > 0.0f && p_final.z > 0.0f) ? 1.0f / k_final.z + p_final.z : 0.0f;
        prob[3] = (k_final.w > 0.0f && p_final.w > 0.0f) ? 1.0f / k_final.w + p_final.w : 0.0f;

        // Сохраняем в shared memory для нормализации
#pragma unroll
        for (int i = 0; i < 4; i++) {
            s_prob_block[threadIdx.x][i] = prob[i];
        }

        __syncthreads();

        // Нормализация вероятностей
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

        // Последнее значение всегда 1.0
        norm_matrix_probability[base_idx + 3] = 1.0f;
    }
}

// ==================== АЛЬТЕРНАТИВНАЯ ВЕРСИЯ С МЕНЬШИМ ИСПОЛЬЗОВАНИЕМ SHARED MEMORY ====================

__device__ void updatePheromonesAndProbabilities_rtx3060_compact( const float* __restrict__ OF, const int* __restrict__ agent_node, float* __restrict__ pheromon, float* __restrict__ kol_enter, float* __restrict__ norm_matrix_probability) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= PARAMETR_SIZE) return;

    const int base_idx = tid * 4;

    // 1. Испарение феромонов
    Float4 pheromon_vec = *reinterpret_cast<Float4*>(&pheromon[base_idx]);
    pheromon_vec = pheromon_vec * PARAMETR_RO;
    *reinterpret_cast<Float4*>(&pheromon[base_idx]) = pheromon_vec;

    // 2. Вычисление дельт для добавления
    float pheromon_delta[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float kol_enter_delta[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    // Обработка всех агентов для этого параметра
    // Для RTX 3060 можно позволить себе полный перебор, так как ANT_SIZE обычно небольшой
    for (int ant_id = 0; ant_id < ANT_SIZE; ant_id++) {
        if (OF[ant_id] != ZERO_HASH_RESULT) {
            const int k = agent_node[ant_id * PARAMETR_SIZE + tid];

            if (k >= 0 && k < 4) {
                kol_enter_delta[k] += 1.0f;

#if OPTIMIZE_MIN_1
                float delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[ant_id];
                if (delta > 0.0f) pheromon_delta[k] += PARAMETR_Q * delta;
#elif OPTIMIZE_MIN_2
                float of_val = fmaxf(OF[ant_id], 1e-7f);
                pheromon_delta[k] += PARAMETR_Q / of_val;
#elif OPTIMIZE_MAX
                pheromon_delta[k] += PARAMETR_Q * OF[ant_id];
#endif
            }
        }
    }

    // 3. Атомарное обновление глобальной памяти
    for (int k = 0; k < 4; k++) {
        if (kol_enter_delta[k] > 0.0f) {
            atomicAdd(&kol_enter[base_idx + k], kol_enter_delta[k]);
        }
        if (pheromon_delta[k] > 0.0f) {
            atomicAdd(&pheromon[base_idx + k], pheromon_delta[k]);
        }
    }

    // 4. Загружаем обновленные данные
    Float4 final_pheromon = *reinterpret_cast<Float4*>(&pheromon[base_idx]);
    Float4 final_kol_enter = *reinterpret_cast<Float4*>(&kol_enter[base_idx]);

    // 5. Вычисление вероятностей
    // Сумма феромонов
    float sum = final_pheromon.x + final_pheromon.y + final_pheromon.z + final_pheromon.w;

    if (sum > 1e-10f) {
        float inv_sum = 1.0f / sum;
        final_pheromon.x *= inv_sum;
        final_pheromon.y *= inv_sum;
        final_pheromon.z *= inv_sum;
        final_pheromon.w *= inv_sum;
    }

    // Вероятности
    float prob[4];
    prob[0] = (final_kol_enter.x > 0.0f && final_pheromon.x > 0.0f) ?
        1.0f / final_kol_enter.x + final_pheromon.x : 0.0f;
    prob[1] = (final_kol_enter.y > 0.0f && final_pheromon.y > 0.0f) ?
        1.0f / final_kol_enter.y + final_pheromon.y : 0.0f;
    prob[2] = (final_kol_enter.z > 0.0f && final_pheromon.z > 0.0f) ?
        1.0f / final_kol_enter.z + final_pheromon.z : 0.0f;
    prob[3] = (final_kol_enter.w > 0.0f && final_pheromon.w > 0.0f) ?
        1.0f / final_kol_enter.w + final_pheromon.w : 0.0f;

    // Нормализация вероятностей
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
        // Равномерное распределение если все вероятности нулевые
        norm_matrix_probability[base_idx] = 0.25f;
        norm_matrix_probability[base_idx + 1] = 0.5f;
        norm_matrix_probability[base_idx + 2] = 0.75f;
        norm_matrix_probability[base_idx + 3] = 1.0f;
    }
}

// ==================== ВЕРСИЯ С ОПТИМИЗАЦИЕЙ ДЛЯ МАЛОГО ЧИСЛА АГЕНТОВ ====================

__device__ void updatePheromonesAndProbabilities_rtx3060_small_batch( const float* __restrict__ OF, const int* __restrict__ agent_node, float* __restrict__ pheromon, float* __restrict__ kol_enter, float* __restrict__ norm_matrix_probability) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Обрабатываем несколько параметров на поток для лучшего использования RTX 3060
    const int params_per_thread = 2;  // 2 параметра на поток для баланса нагрузки

    for (int p = 0; p < params_per_thread; p++) {
        int param_idx = tid * params_per_thread + p;
        if (param_idx >= PARAMETR_SIZE) continue;

        const int base_idx = param_idx * 4;

        // 1. Загрузка и испарение феромонов
        Float4 pheromon_vec = *reinterpret_cast<Float4*>(&pheromon[base_idx]);
        pheromon_vec = pheromon_vec * PARAMETR_RO;
        *reinterpret_cast<Float4*>(&pheromon[base_idx]) = pheromon_vec;

        // 2. Вычисление дельт с warp-level оптимизацией
        float pheromon_delta[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        float kol_enter_delta[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        // Разделяем агентов между потоками в warp'е
        for (int ant_id = lane_id; ant_id < ANT_SIZE; ant_id += 32) {
            if (OF[ant_id] != ZERO_HASH_RESULT) {
                const int k = agent_node[ant_id * PARAMETR_SIZE + param_idx];

                if (k >= 0 && k < 4) {
                    kol_enter_delta[k] += 1.0f;

#if OPTIMIZE_MIN_1
                    float delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[ant_id];
                    if (delta > 0.0f) pheromon_delta[k] += PARAMETR_Q * delta;
#elif OPTIMIZE_MIN_2
                    float of_val = fmaxf(OF[ant_id], 1e-7f);
                    pheromon_delta[k] += PARAMETR_Q / of_val;
#elif OPTIMIZE_MAX
                    pheromon_delta[k] += PARAMETR_Q * OF[ant_id];
#endif
                }
            }
        }

        // Редукция внутри warp
        for (int k = 0; k < 4; k++) {
            for (int offset = 16; offset > 0; offset /= 2) {
                pheromon_delta[k] += __shfl_down_sync(0xFFFFFFFF, pheromon_delta[k], offset);
                kol_enter_delta[k] += __shfl_down_sync(0xFFFFFFFF, kol_enter_delta[k], offset);
            }

            // Только первый поток в warp'е делает atomic операции
            if (lane_id == 0) {
                if (kol_enter_delta[k] > 0.0f) {
                    atomicAdd(&kol_enter[base_idx + k], kol_enter_delta[k]);
                }
                if (pheromon_delta[k] > 0.0f) {
                    atomicAdd(&pheromon[base_idx + k], pheromon_delta[k]);
                }
            }
        }

        __syncthreads();

        // 3. Вычисление вероятностей
        Float4 final_pheromon = *reinterpret_cast<Float4*>(&pheromon[base_idx]);
        Float4 final_kol_enter = *reinterpret_cast<Float4*>(&kol_enter[base_idx]);

        // Сумма феромонов
        float sum = final_pheromon.x + final_pheromon.y + final_pheromon.z + final_pheromon.w;

        if (sum > 1e-10f) {
            float inv_sum = 1.0f / sum;
            final_pheromon.x *= inv_sum;
            final_pheromon.y *= inv_sum;
            final_pheromon.z *= inv_sum;
            final_pheromon.w *= inv_sum;
        }

        // Вероятности
        float prob[4];
        prob[0] = (final_kol_enter.x > 0.0f && final_pheromon.x > 0.0f) ?
            1.0f / final_kol_enter.x + final_pheromon.x : 0.0f;
        prob[1] = (final_kol_enter.y > 0.0f && final_pheromon.y > 0.0f) ?
            1.0f / final_kol_enter.y + final_pheromon.y : 0.0f;
        prob[2] = (final_kol_enter.z > 0.0f && final_pheromon.z > 0.0f) ?
            1.0f / final_kol_enter.z + final_pheromon.z : 0.0f;
        prob[3] = (final_kol_enter.w > 0.0f && final_pheromon.w > 0.0f) ?
            1.0f / final_kol_enter.w + final_pheromon.w : 0.0f;

        // Нормализация
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
    }
}

// ОПТИМИЗИРОВАННАЯ ОПТИМИЗАЦИЯ МУРАВЬЕВ ДЛЯ 4 ЗНАЧЕНИЙ
__device__ void antColonyOptimization_optimized_dev(const float* __restrict__ dev_parametr_value, const float* __restrict__ norm_matrix_probability, int* __restrict__ agent_node,float* __restrict__ OF, HashEntry* __restrict__ hashTable, float* __restrict__ maxOf_dev, float* __restrict__ minOf_dev, int* __restrict__ kol_hash_fail, float* __restrict__ global_params_buffer) {
    const int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_id >= ANT_SIZE) return;

    // Локальная shared memory для параметров агента
    //__shared__ double s_agent_params[128][16];  // [поток][параметр]

    curandState state;
    curand_init(clock64() + ant_id, 0, 0, &state);

    int* agent_node_ptr = &agent_node[ant_id * PARAMETR_SIZE];
    bool go_selected_index = true;
    // Выбор путей - оптимизировано для 4 значений
    for (int param_idx = 0; param_idx < PARAMETR_SIZE; param_idx++) {
        float randomValue = curand_uniform(&state);
        const int base_prob_idx = param_idx * 4;

        // Быстрый выбор без цикла
        int selected_index = 0;  // По умолчанию последнее значение
        if (go_selected_index) {
            selected_index = 3;  // По умолчанию последнее значение
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
        }
        agent_node_ptr[param_idx] = selected_index;
        go_selected_index = (selected_index != 3);

        // Сохранение значения параметра
        float param_value = dev_parametr_value[base_prob_idx + selected_index];
        global_params_buffer[ant_id * PARAMETR_SIZE + param_idx] = param_value;
        //s_agent_params[threadIdx.x][param_idx % 16] = param_value;
    }

    // Вычисление целевой функции
    float cached = getCachedResult4(hashTable, agent_node, ant_id);

    if (cached < 0.0f) {
        // Используем shared memory для быстрого доступа к параметрам
        double agent_params_local[PARAMETR_SIZE];
        for (int i = 0; i < PARAMETR_SIZE; i++) {
            agent_params_local[i] = global_params_buffer[ant_id * PARAMETR_SIZE + i];
        }

        float result = (float)BenchShafferaFunction(agent_params_local);
        OF[ant_id] = result;
        saveToCache4(hashTable, agent_node, ant_id, result);
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

__global__ void evaporatePheromones_optimized(float* __restrict__ pheromon) {
    evaporatePheromones_optimized_dev(pheromon);
}
__global__ void depositPheromones_optimized(const float* __restrict__ OF, const int* __restrict__ agent_node, float* __restrict__ pheromon, float* __restrict__ kol_enter) {
    depositPheromones_optimized_dev(OF, agent_node, pheromon, kol_enter);
}
__global__ void updatePheromones_optimized(const float* __restrict__ OF, const int* __restrict__ agent_node, float* __restrict__ pheromon, float* __restrict__ kol_enter) {
    evaporatePheromones_optimized_dev(pheromon);
    depositPheromones_optimized_dev(OF, agent_node, pheromon, kol_enter);
}
__global__ void computeProbabilities_optimized(const float* __restrict__ pheromon, const float* __restrict__ kol_enter, float* __restrict__ norm_matrix_probability) {
    computeProbabilities_optimized_dev(pheromon, kol_enter, norm_matrix_probability);
}
__global__ void antColonyOptimization_optimized(const float* __restrict__ dev_parametr_value, const float* __restrict__ norm_matrix_probability, int* __restrict__ agent_node, float* __restrict__ OF, HashEntry* __restrict__ hashTable, float* __restrict__ maxOf_dev, float* __restrict__ minOf_dev, int* __restrict__ kol_hash_fail, float* __restrict__ global_params_buffer) {
    antColonyOptimization_optimized_dev(dev_parametr_value, norm_matrix_probability, agent_node, OF, hashTable, maxOf_dev, minOf_dev, kol_hash_fail, global_params_buffer);
}
__global__ void combinedUpdateKernel_rtx3060(const float* __restrict__ OF, const int* __restrict__ agent_node, float* __restrict__ pheromon, float* __restrict__ kol_enter, float* __restrict__ norm_matrix_probability, int version = 0) {

    switch (version) {
    case 0:
        // Полная версия с shared memory оптимизациями
        updatePheromonesAndProbabilities_rtx3060(OF, agent_node, pheromon, kol_enter, norm_matrix_probability);
        break;
    case 1:
        // Компактная версия с меньшим использованием shared memory
        updatePheromonesAndProbabilities_rtx3060_compact(OF, agent_node, pheromon, kol_enter, norm_matrix_probability);
        break;
    case 2:
        // Версия для малого числа агентов с batch обработкой
        updatePheromonesAndProbabilities_rtx3060_small_batch(OF, agent_node, pheromon, kol_enter, norm_matrix_probability);
        break;
    default:
        // Версия по умолчанию
        updatePheromonesAndProbabilities_rtx3060(OF, agent_node, pheromon, kol_enter, norm_matrix_probability);
        break;
    }
}

// ==================== ГЛОБАЛЬНЫЕ РЕСУРСЫ ====================
static float* dev_pheromon = nullptr, * dev_kol_enter = nullptr, * dev_norm_matrix = nullptr;
static float* dev_OF = nullptr, * dev_max = nullptr, * dev_min = nullptr;
static float* dev_parametr_value = nullptr, * dev_agent_params = nullptr;
static int* dev_agent_node = nullptr, * dev_hash_fail = nullptr;
static HashEntry* dev_hashTable = nullptr;
static cudaStream_t compute_stream = nullptr;

// ==================== УПРАВЛЕНИЕ ПАМЯТЬЮ ====================
bool initialize_cuda_resources(const float* params, const float* pheromon, const float* kol_enter) {
    CUDA_CHECK(cudaStreamCreate(&compute_stream));

    const size_t matrix_size = 4 * PARAMETR_SIZE * sizeof(float);  // MAX_VALUE_SIZE = 4
    const size_t ant_matrix_size = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    const size_t agent_params_size = PARAMETR_SIZE * ANT_SIZE * sizeof(float);

    // Выделение памяти с выравниванием для RTX 3060
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

    // Копирование данных с использованием pinned memory для лучшей производительности
    CUDA_CHECK(cudaMemcpyAsync(dev_parametr_value, params, matrix_size,
        cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_pheromon, pheromon, matrix_size,
        cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_kol_enter, kol_enter, matrix_size,
        cudaMemcpyHostToDevice, compute_stream));

    // Инициализация хэш-таблицы
    int threads = min(1024, HASH_TABLE_SIZE);
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

// ==================== ОПТИМИЗИРОВАННАЯ ФУНКЦИЯ ВЫПОЛНЕНИЯ ====================
PerformanceMetrics run_aco_iterations4_optimized(int num_iterations) {
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

    // Оптимальные размеры сетки для RTX 3060
    int blocks_prob = (PARAMETR_SIZE + OPTIMAL_BLOCK_SIZE - 1) / OPTIMAL_BLOCK_SIZE;
    int blocks_ant = (ANT_SIZE + OPTIMAL_ANT_BLOCK_SIZE - 1) / OPTIMAL_ANT_BLOCK_SIZE;
    int blocks_update = (PARAMETR_SIZE + OPTIMAL_BLOCK_SIZE - 1) / OPTIMAL_BLOCK_SIZE;

    // Ограничение блоков для лучшего использования SMs
    blocks_prob = min(blocks_prob, RTX3060_SM_COUNT * 4);
    blocks_ant = min(blocks_ant, RTX3060_SM_COUNT * 2);
    blocks_update = min(blocks_update, RTX3060_SM_COUNT * 4);

    for (int iter = 0; iter < num_iterations; ++iter) {
        float iter_time;

        // Этап 1: Вычисление вероятностей
        CUDA_CHECK(cudaEventRecord(start, compute_stream));
        computeProbabilities_optimized << <blocks_prob, OPTIMAL_BLOCK_SIZE, 0, compute_stream >> > ( dev_pheromon, dev_kol_enter, dev_norm_matrix);
        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        compute_time += iter_time;

        // Этап 2: Оптимизация муравьями
        CUDA_CHECK(cudaEventRecord(start, compute_stream));
        antColonyOptimization_optimized << <blocks_ant, OPTIMAL_ANT_BLOCK_SIZE, 0, compute_stream >> > ( dev_parametr_value, dev_norm_matrix, dev_agent_node, dev_OF, dev_hashTable, dev_max, dev_min, dev_hash_fail, dev_agent_params);
        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        ant_time += iter_time;

        // Этап 3: Обновление феромонов
        CUDA_CHECK(cudaEventRecord(start, compute_stream));
        evaporatePheromones_optimized << <blocks_update, OPTIMAL_BLOCK_SIZE, 0, compute_stream >> > ( dev_pheromon);
        depositPheromones_optimized << <blocks_update, OPTIMAL_BLOCK_SIZE, 0, compute_stream >> > ( dev_OF, dev_agent_node, dev_pheromon, dev_kol_enter);
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

    // Расчет occupancy для RTX 3060
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

    // Расчет пропускной способности памяти для RTX 3060
    size_t total_data_transferred = (4 * PARAMETR_SIZE * sizeof(float)) * 4 * num_iterations + (PARAMETR_SIZE * ANT_SIZE * sizeof(int)) * num_iterations + (ANT_SIZE * sizeof(float)) * num_iterations;
    metrics.memory_throughput_gbs = (total_data_transferred / (1024.0 * 1024.0 * 1024.0)) / (metrics.total_time_ms / 1000.0);

    return metrics;
}
PerformanceMetrics run_aco_iterations3_optimized(int num_iterations) {
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

    // Оптимальные размеры сетки для RTX 3060
    int blocks_prob = (PARAMETR_SIZE + OPTIMAL_BLOCK_SIZE - 1) / OPTIMAL_BLOCK_SIZE;
    int blocks_ant = (ANT_SIZE + OPTIMAL_ANT_BLOCK_SIZE - 1) / OPTIMAL_ANT_BLOCK_SIZE;
    int blocks_update = (PARAMETR_SIZE + OPTIMAL_BLOCK_SIZE - 1) / OPTIMAL_BLOCK_SIZE;

    // Ограничение блоков для лучшего использования SMs
    blocks_prob = min(blocks_prob, RTX3060_SM_COUNT * 4);
    blocks_ant = min(blocks_ant, RTX3060_SM_COUNT * 2);
    blocks_update = min(blocks_update, RTX3060_SM_COUNT * 4);

    for (int iter = 0; iter < num_iterations; ++iter) {
        float iter_time;

        // Этап 1: Вычисление вероятностей
        CUDA_CHECK(cudaEventRecord(start, compute_stream));
        computeProbabilities_optimized << <blocks_prob, OPTIMAL_BLOCK_SIZE, 0, compute_stream >> > (dev_pheromon, dev_kol_enter, dev_norm_matrix);
        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        compute_time += iter_time;

        // Этап 2: Оптимизация муравьями
        CUDA_CHECK(cudaEventRecord(start, compute_stream));
        antColonyOptimization_optimized << <blocks_ant, OPTIMAL_ANT_BLOCK_SIZE, 0, compute_stream >> > (dev_parametr_value, dev_norm_matrix, dev_agent_node, dev_OF, dev_hashTable, dev_max, dev_min, dev_hash_fail, dev_agent_params);
        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        ant_time += iter_time;

        // Этап 3: Обновление феромонов
        CUDA_CHECK(cudaEventRecord(start, compute_stream));
        updatePheromones_optimized << <blocks_update, OPTIMAL_BLOCK_SIZE, 0, compute_stream >> > (dev_OF, dev_agent_node, dev_pheromon, dev_kol_enter);
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

    // Расчет occupancy для RTX 3060
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

    // Расчет пропускной способности памяти для RTX 3060
    size_t total_data_transferred = (4 * PARAMETR_SIZE * sizeof(float)) * 4 * num_iterations + (PARAMETR_SIZE * ANT_SIZE * sizeof(int)) * num_iterations + (ANT_SIZE * sizeof(float)) * num_iterations;
    metrics.memory_throughput_gbs = (total_data_transferred / (1024.0 * 1024.0 * 1024.0)) / (metrics.total_time_ms / 1000.0);

    return metrics;
}

void launchCombinedUpdateRTX3060(cudaStream_t stream, const float* d_OF, const int* d_agent_node, float* d_pheromon, float* d_kol_enter, float* d_norm_matrix) {

    // Определяем оптимальную версию на основе параметров
    int version = 0;

    if (PARAMETR_SIZE <= 1024 && ANT_SIZE <= 256) {
        // Для небольших данных используем компактную версию
        version = 1;
    }
    else if (PARAMETR_SIZE >= 4096) {
        // Для больших данных используем batch версию
        version = 2;
    }
    // Иначе используем версию по умолчанию (0)

    // Оптимальные размеры для RTX 3060
    const int block_size = 256;  // Оптимально для RTX 3060
    int grid_size;

    if (version == 2) {
        // Для batch версии меньше блоков
        grid_size = (PARAMETR_SIZE + block_size * 2 - 1) / (block_size * 2);
    }
    else {
        grid_size = (PARAMETR_SIZE + block_size - 1) / block_size;
    }

    // Ограничиваем grid size для лучшего использования 28 SM RTX 3060
    grid_size = min(grid_size, 28 * 4);  // 4 блока на SM

    // Запускаем ядро
    combinedUpdateKernel_rtx3060 << <grid_size, block_size, 0, stream >> > (d_OF, d_agent_node, d_pheromon, d_kol_enter, d_norm_matrix, version);

    CUDA_CHECK(cudaGetLastError());
}
PerformanceMetrics run_aco_iterations2_optimized(int num_iterations) {
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

    // Оптимальные размеры сетки для RTX 3060
    int blocks_prob = (PARAMETR_SIZE + OPTIMAL_BLOCK_SIZE - 1) / OPTIMAL_BLOCK_SIZE;
    int blocks_ant = (ANT_SIZE + OPTIMAL_ANT_BLOCK_SIZE - 1) / OPTIMAL_ANT_BLOCK_SIZE;
    int blocks_update = (PARAMETR_SIZE + OPTIMAL_BLOCK_SIZE - 1) / OPTIMAL_BLOCK_SIZE;

    // Ограничение блоков для лучшего использования SMs
    blocks_prob = min(blocks_prob, RTX3060_SM_COUNT * 4);
    blocks_ant = min(blocks_ant, RTX3060_SM_COUNT * 2);
    blocks_update = min(blocks_update, RTX3060_SM_COUNT * 4);

    for (int iter = 0; iter < num_iterations; ++iter) {
        float iter_time;

        // Этап 1: Вычисление вероятностей
        CUDA_CHECK(cudaEventRecord(start, compute_stream));
        launchCombinedUpdateRTX3060(compute_stream, dev_OF, dev_agent_node, dev_pheromon, dev_kol_enter, dev_norm_matrix);
        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        compute_time += iter_time;

        // Этап 2: Оптимизация муравьями
        CUDA_CHECK(cudaEventRecord(start, compute_stream));
        antColonyOptimization_optimized << <blocks_ant, OPTIMAL_ANT_BLOCK_SIZE, 0, compute_stream >> > (dev_parametr_value, dev_norm_matrix, dev_agent_node, dev_OF, dev_hashTable, dev_max, dev_min, dev_hash_fail, dev_agent_params);
        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        ant_time += iter_time;

        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
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

    // Расчет occupancy для RTX 3060
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

    // Расчет пропускной способности памяти для RTX 3060
    size_t total_data_transferred = (4 * PARAMETR_SIZE * sizeof(float)) * 4 * num_iterations + (PARAMETR_SIZE * ANT_SIZE * sizeof(int)) * num_iterations + (ANT_SIZE * sizeof(float)) * num_iterations;
    metrics.memory_throughput_gbs = (total_data_transferred / (1024.0 * 1024.0 * 1024.0)) / (metrics.total_time_ms / 1000.0);

    return metrics;
}


// ==================== ЗАГРУЗКА ДАННЫХ ====================
bool load_matrix_data_optimized(const std::string& filename, std::vector<float>& params, std::vector<float>& pheromones, std::vector<float>& visits) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    size_t total_size = 4 * PARAMETR_SIZE;  // MAX_VALUE_SIZE = 4
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

// ==================== ИНФОРМАЦИЯ О GPU ====================
void print_rtx3060_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        std::cout << "\n=== RTX 3060 GPU Information ===" << std::endl;
        std::cout << "Device " << device << ": " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
        std::cout << "Warp Size: " << prop.warpSize << std::endl;
        std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
        std::cout << "L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000.0 << " MHz" << std::endl;
        std::cout << "GPU Overlap: " << (prop.deviceOverlap ? "Yes" : "No") << std::endl;
        std::cout << "Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
        std::cout << "ECC Enabled: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
        std::cout << "=================================" << std::endl;

        logFile << "\nRTX 3060 GPU Information:" << std::endl;
        logFile << "Device: " << prop.name << std::endl;
        logFile << "Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;
        logFile << "SMs: " << prop.multiProcessorCount << std::endl;
        logFile << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    }
}
void print_optimization_info() {
    std::cout << "\n=== Optimization Settings for RTX 3060 ===" << std::endl;
    std::cout << "MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << " (optimized for 4)" << std::endl;
    std::cout << "PARAMETR_SIZE: " << PARAMETR_SIZE << std::endl;
    std::cout << "ANT_SIZE: " << ANT_SIZE << std::endl;
    std::cout << "OPTIMAL_BLOCK_SIZE: " << OPTIMAL_BLOCK_SIZE << std::endl;
    std::cout << "OPTIMAL_ANT_BLOCK_SIZE: " << OPTIMAL_ANT_BLOCK_SIZE << std::endl;
    std::cout << "Using vectorized operations for 4 values" << std::endl;
    std::cout << "Using shared memory optimizations" << std::endl;
    std::cout << "Using warp-level reductions" << std::endl;
    std::cout << "==========================================" << std::endl;

    logFile << "\nOptimization Settings:" << std::endl;
    logFile << "MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << std::endl;
    logFile << "PARAMETR_SIZE: " << PARAMETR_SIZE << std::endl;
    logFile << "ANT_SIZE: " << ANT_SIZE << std::endl;
}
void print_metrics(const PerformanceMetrics& metrics, const char* str, int run_id) {
    std::cout << "Run " << str << run_id << ": "
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

// ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
int main() {
    std::cout << "Initializing CUDA ACO Optimizer for RTX 3060..." << std::endl;

    // Вывод информации о GPU
    print_rtx3060_info();
    print_optimization_info();

    // Загрузка данных
    std::vector<float> params, pheromones, visits;
    if (!load_matrix_data_optimized(NAME_FILE_GRAPH, params, pheromones, visits)) {
        std::cerr << "Failed to load matrix data" << std::endl;
        return 1;
    }

    // Проверка выделения памяти
    size_t required_memory = (4 * PARAMETR_SIZE * sizeof(float) * 4) +  // 4 матрицы
        (PARAMETR_SIZE * ANT_SIZE * sizeof(int)) +
        (PARAMETR_SIZE * ANT_SIZE * sizeof(float)) +
        (ANT_SIZE * sizeof(float) * 2) +
        (HASH_TABLE_SIZE * sizeof(HashEntry));

    std::cout << "\n=== Memory Requirements ===" << std::endl;
    std::cout << "Required memory: " << required_memory / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "RTX 3060 has: 12,288 MB" << std::endl;
    std::cout << "Usage: " << (required_memory / (12.0 * 1024.0 * 1024.0)) * 100.0 << "%" << std::endl;

    if (required_memory > 12 * 1024 * 1024 * 1024ULL) {
        std::cerr << "ERROR: Not enough GPU memory!" << std::endl;
        return 1;
    }

    // Инициализация CUDA
    if (!initialize_cuda_resources(params.data(), pheromones.data(), visits.data())) {
        std::cerr << "CUDA resources initialization failed" << std::endl;
        return 1;
    }

    // Прогрев GPU
    std::cout << "\n=== Warming up GPU ===" << std::endl;
    for (int i = 0; i < KOL_PROGREV; ++i) {
        std::cout << "Warmup iteration " << i + 1 << "/" << KOL_PROGREV << std::endl;
        auto warmup_metrics = run_aco_iterations4_optimized(KOL_ITERATION);
    }
    std::cout << "Warmup completed" << std::endl;

    // Основные запуски
    std::cout << "\n=== Starting Optimized ACO Runs ===" << std::endl;
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < KOL_PROGON_STATISTICS; ++i) {
        auto metrics = run_aco_iterations4_optimized(KOL_ITERATION);
        print_metrics(metrics, "Optimized_4", i);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);

    std::cout << "\n=== Starting Optimized ACO Runs ===" << std::endl;
    total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < KOL_PROGON_STATISTICS; ++i) {
        auto metrics = run_aco_iterations3_optimized(KOL_ITERATION);
        print_metrics(metrics, "Optimized_3", i);
    }

    total_end = std::chrono::high_resolution_clock::now();
    total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);

    std::cout << "\n=== Starting Optimized ACO Runs ===" << std::endl;
    total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < KOL_PROGON_STATISTICS; ++i) {
        auto metrics = run_aco_iterations2_optimized(KOL_ITERATION);
        print_metrics(metrics, "Optimized_2", i);
    }

    total_end = std::chrono::high_resolution_clock::now();
    total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);

    // Очистка
    cleanup_cuda_resources();
    logFile.close();

    std::cout << "\nOptimization completed successfully!" << std::endl;
    return 0;
}