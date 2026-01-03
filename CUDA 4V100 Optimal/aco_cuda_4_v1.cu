#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <mpi.h>
#include "parametrs.h"

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(err) << std::endl; exit(EXIT_FAILURE); } } while(0)
#define MPI_CHECK(call) do { int err = call; if (err != MPI_SUCCESS) { std::cerr << "MPI error " << __FILE__ << ":" << __LINE__ << std::endl; exit(EXIT_FAILURE); } } while(0)

std::ofstream* logFilePtr = nullptr; // Указатель вместо глобальной переменной


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
// ==================== MPI ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ====================
int mpi_world_rank = 0;
int mpi_world_size = 1;
int mpi_local_rank = 0;
bool mpi_is_master = true;
// ==================== CUDA РЕСУРСЫ ДЛЯ КАЖДОГО ПРОЦЕССА ====================
struct CUDAResources {
    double* dev_pheromon = nullptr;
    double* dev_kol_enter = nullptr;
    double* dev_norm_matrix = nullptr;
    double* dev_OF = nullptr;
    double* dev_max = nullptr;
    double* dev_min = nullptr;
    int* dev_agent_node = nullptr;
    int* dev_hash_fail = nullptr;
    HashEntry* dev_hashTable = nullptr;
    cudaStream_t compute_stream = nullptr;
    double* dev_parametr_value = nullptr;
    double* dev_agent_params = nullptr;

    double* host_pheromon_buffer = nullptr;
    double* host_kol_enter_buffer = nullptr;
    double* host_norm_matrix_buffer = nullptr;
};
CUDAResources g_resources;

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

//MPI
bool mpi_initialize(int* argc, char*** argv) {
    MPI_CHECK(MPI_Init(argc, argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size));

    // Получаем локальный ранг для GPU
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
        MPI_INFO_NULL, &local_comm));

    int local_size = 0;
    MPI_CHECK(MPI_Comm_rank(local_comm, &mpi_local_rank));
    MPI_CHECK(MPI_Comm_size(local_comm, &local_size));  // ФИКС: был &mpi_local_rank

    MPI_CHECK(MPI_Comm_free(&local_comm));

    // Проверяем доступные GPU
    int num_gpus = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&num_gpus);

    if (cuda_status != cudaSuccess) {
        std::cerr << "Process " << mpi_world_rank
            << ": Failed to get GPU count! Error: "
            << cudaGetErrorString(cuda_status) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (num_gpus == 0) {
        std::cerr << "Process " << mpi_world_rank
            << ": No CUDA devices available!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Выбираем GPU: используем остаток от деления мирового ранга на количество GPU
    int gpu_id = mpi_world_rank % num_gpus;

    // Альтернативно можно использовать локальный ранг:
    // int gpu_id = mpi_local_rank % num_gpus;

    cuda_status = cudaSetDevice(gpu_id);

    if (cuda_status != cudaSuccess) {
        std::cerr << "Process " << mpi_world_rank
            << ": Failed to set CUDA device " << gpu_id
            << "! Error: " << cudaGetErrorString(cuda_status) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Проверяем устройство
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);

    std::cout << "Process " << mpi_world_rank << "/" << mpi_world_size
        << " using GPU " << gpu_id << ": " << prop.name
        << " (Compute " << prop.major << "." << prop.minor << ")"
        << std::endl;

    mpi_is_master = (mpi_world_rank == 0);

    return true;
}
void mpi_finalize() {
    MPI_CHECK(MPI_Finalize());
}
void mpi_synchronize() {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
}
void mpi_reduce_pheromones(double* local_pheromon, double* global_pheromon, size_t size) {
    MPI_CHECK(MPI_Reduce(local_pheromon, global_pheromon, size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
}
void mpi_reduce_visits(double* local_kol_enter, double* global_kol_enter, size_t size) {
    MPI_CHECK(MPI_Reduce(local_kol_enter, global_kol_enter, size, MPI_DOUBLE,  MPI_SUM, 0, MPI_COMM_WORLD));
}
void mpi_broadcast_norm_matrix(double* norm_matrix, size_t size) {
    MPI_CHECK(MPI_Bcast(norm_matrix, size, MPI_DOUBLE, 0, MPI_COMM_WORLD));
}
void mpi_gather_statistics(double* local_min, double* local_max, double* global_min, double* global_max) {
    MPI_CHECK(MPI_Reduce(local_min, global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(local_max, global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
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
// Вычисление путей муравьев-агентов
__device__ void antColonyOptimization_and_deposit_warp_optimized_global_dev(double* __restrict__ dev_parametr_value, double* __restrict__ norm_matrix_probability, HashEntry* __restrict__ hashTable, double* __restrict__ maxOf_dev, double* __restrict__ minOf_dev, int* __restrict__ kol_hash_fail, double* __restrict__ pheromon, double* __restrict__ kol_enter, int* __restrict__ agent_node, double* __restrict__ OF, double* __restrict__ global_params_buffer) {
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int warp_id = tx / WARP_SIZE;
    const int lane_id = tx % WARP_SIZE;
    const int ant_id = bx;

    if (ant_id >= ANT_SIZE) return;

    // Указатели на данные конкретного агента в глобальной памяти
    int* agent_node_ptr = &agent_node[ant_id * PARAMETR_SIZE];
    double* agent_params = &global_params_buffer[ant_id * PARAMETR_SIZE];

    // Генератор случайных чисел
    curandState state;
    curand_init(clock64() + ant_id * blockDim.x + tx * 7919, 0, 0, &state);

    // Shared memory только для критических данных (не зависящих от PARAMETR_SIZE)
    __shared__ double shared_OF;
    __shared__ double shared_pheromone_delta;

    // Инициализация shared memory
    if (tx == 0) {
        shared_OF = ZERO_HASH_RESULT;
        shared_pheromone_delta = 0.0;
    }
    __syncthreads();

    // 1. ГЕНЕРАЦИЯ ПУТИ АГЕНТА (все потоки участвуют)
    const int params_per_thread = (PARAMETR_SIZE + blockDim.x - 1) / blockDim.x;

    // Оптимизация: используем tile-based доступ для лучшей локализации
    const int num_tiles = (PARAMETR_SIZE + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = warp_id; tile < num_tiles; tile += blockDim.x / WARP_SIZE) {
        const int tile_start = tile * TILE_SIZE;
        const int tile_end = min(tile_start + TILE_SIZE, PARAMETR_SIZE);

        // Обрабатываем параметры в текущем тайле
        for (int i = lane_id; i < (tile_end - tile_start); i += WARP_SIZE) {
            const int param_idx = tile_start + i;

            double randomValue = curand_uniform(&state);
            int selected_index = 0;

            // Быстрый поиск с использованием указателей
            const double* prob_ptr = &norm_matrix_probability[param_idx * MAX_VALUE_SIZE];

            // Оптимизация: развернутый поиск для MAX_VALUE_SIZE=5
#if MAX_VALUE_SIZE == 5
            selected_index = (randomValue > prob_ptr[0]) ?
                ((randomValue > prob_ptr[2]) ?
                    ((randomValue > prob_ptr[3]) ? 4 : 3) :
                    ((randomValue > prob_ptr[1]) ? 2 : 1)) : 0;
#else
// Общий случай с развертыванием
#pragma unroll
            for (int k = 0; k < MAX_VALUE_SIZE - 1; k++) {
                if (randomValue > prob_ptr[k]) {
                    selected_index = k + 1;
                }
                else {
                    break;
                }
            }
            if (selected_index >= MAX_VALUE_SIZE) selected_index = MAX_VALUE_SIZE - 1;
#endif

            // Сохраняем в глобальную память
            agent_node_ptr[param_idx] = selected_index;
            agent_params[param_idx] = dev_parametr_value[param_idx * MAX_VALUE_SIZE + selected_index];
        }
    }

    __syncthreads();

    // 2. ВЫЧИСЛЕНИЕ ЦЕЛЕВОЙ ФУНКЦИИ (один поток на блок)
    if (tx == 0) {
        double cached = getCachedResult(hashTable, agent_node_ptr, ant_id);

        if (cached < 0.0) {
            shared_OF = BenchShafferaFunction(agent_params);
            saveToCache(hashTable, agent_node_ptr, ant_id, shared_OF);
        }
        else {
            shared_OF = cached;
            atomicAdd(kol_hash_fail, 1);
        }

        // Сохраняем OF в глобальную память
        OF[ant_id] = shared_OF;

        if (shared_OF != ZERO_HASH_RESULT) {
            atomicMax(maxOf_dev, shared_OF);
            atomicMin(minOf_dev, shared_OF);
        }

        // Вычисляем дельту феромона (если нужно)
#if OPTIMIZE_MIN_1
        const double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - shared_OF;
        if (delta > 0.0) shared_pheromone_delta = PARAMETR_Q * delta;
#elif OPTIMIZE_MIN_2
        const double of_val = fmax(shared_OF, 1e-7);
        shared_pheromone_delta = PARAMETR_Q / of_val;
#elif OPTIMIZE_MAX
        shared_pheromone_delta = PARAMETR_Q * shared_OF;
#endif
    }

    __syncthreads();

    // 3. ДЕПОЗИТ ФЕРОМОНОВ (все потоки участвуют)
    double current_OF = shared_OF;
    double pheromone_delta = shared_pheromone_delta;

    if (current_OF != ZERO_HASH_RESULT) {
        // Распространяем данные по warp'у
        unsigned warp_mask = __ballot_sync(0xFFFFFFFF, true);
        current_OF = __shfl_sync(warp_mask, current_OF, 0);
        pheromone_delta = __shfl_sync(warp_mask, pheromone_delta, 0);

        // Обновление феромонов с tile-based доступом
        for (int tile = warp_id; tile < num_tiles; tile += blockDim.x / WARP_SIZE) {
            const int tile_start = tile * TILE_SIZE;
            const int tile_end = min(tile_start + TILE_SIZE, PARAMETR_SIZE);

            // Предзагружаем selected indices для тайла
            int tile_selected[TILE_SIZE];
            if (lane_id < TILE_SIZE && tile_start + lane_id < tile_end) {
                tile_selected[lane_id] = agent_node_ptr[tile_start + lane_id];
            }

            // Обновляем феромоны в тайле
            for (int i = lane_id; i < (tile_end - tile_start); i += WARP_SIZE) {
                const int param_idx = tile_start + i;
                const int selected_idx = tile_selected[i];
                const int pheromon_idx = param_idx * MAX_VALUE_SIZE + selected_idx;

                // Атомарное обновление
                atomicAdd(&kol_enter[pheromon_idx], 1.0);

                if (pheromone_delta > 0.0) {
                    atomicAdd(&pheromon[pheromon_idx], pheromone_delta);
                }
            }
        }
    }
}
__device__ void antColonyOptimization_and_deposit_warp_optimized_dev(double* __restrict__ dev_parametr_value, double* __restrict__ norm_matrix_probability, HashEntry* __restrict__ hashTable, double* __restrict__ maxOf_dev, double* __restrict__ minOf_dev, int* __restrict__ kol_hash_fail, double* __restrict__ pheromon, double* __restrict__ kol_enter) {
    __shared__ int shared_selected[PARAMETR_SIZE];
    __shared__ double shared_agent_params[PARAMETR_SIZE];
    __shared__ double shared_OF;


    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int warp_id = tx / WARP_SIZE;
    const int lane_id = tx % WARP_SIZE;
    const int ant_id = bx;

    if (ant_id >= ANT_SIZE) return;

    // Инициализация shared memory (один поток на блок)
    if (tx == 0) {
        shared_OF = ZERO_HASH_RESULT;
    }
    __syncthreads();

    // Оптимизация 2: Векторизованная генерация случайных чисел
    curandState state;
    curand_init(clock64() + ant_id * blockDim.x + tx * 7919, 0, 0, &state);

    // 1. ГЕНЕРАЦИЯ ПУТИ АГЕНТА с оптимизациями
    const int params_per_thread = (PARAMETR_SIZE + blockDim.x - 1) / blockDim.x;

    // Оптимизация 3: Предзагрузка вероятностей в регистры для часто используемых параметров
    if (PARAMETR_SIZE <= 128) {
        // Для малых PARAMETR_SIZE можно предзагрузить все вероятности
        double local_probs[PARAMETR_SIZE][MAX_VALUE_SIZE];
        for (int p = 0; p < params_per_thread; p++) {
            const int param_idx = tx + p * blockDim.x;
            if (param_idx >= PARAMETR_SIZE) break;

#pragma unroll
            for (int k = 0; k < MAX_VALUE_SIZE; k++) {
                local_probs[p][k] = norm_matrix_probability[param_idx * MAX_VALUE_SIZE + k];
            }
        }

        for (int p = 0; p < params_per_thread; p++) {
            const int param_idx = tx + p * blockDim.x;
            if (param_idx >= PARAMETR_SIZE) break;

            double randomValue = curand_uniform(&state);
            int selected_index = 0;

            // Оптимизация 4: Развернутый цикл для MAX_VALUE_SIZE=5
#if MAX_VALUE_SIZE == 5
            selected_index = (randomValue > local_probs[p][0]) ?
                ((randomValue > local_probs[p][2]) ?
                    ((randomValue > local_probs[p][3]) ? 4 : 3) :
                    ((randomValue > local_probs[p][1]) ? 2 : 1)) : 0;
#else
// Общий случай с развертыванием
#pragma unroll
            for (int k = 0; k < MAX_VALUE_SIZE - 1; k++) {
                selected_index += (randomValue > local_probs[p][k]);
            }
#endif

            shared_selected[param_idx] = selected_index;
            shared_agent_params[param_idx] = dev_parametr_value[param_idx * MAX_VALUE_SIZE + selected_index];
        }
    }
    else {
        // Для больших PARAMETR_SIZE используем стандартный подход
        for (int p = 0; p < params_per_thread; p++) {
            const int param_idx = tx + p * blockDim.x;
            if (param_idx >= PARAMETR_SIZE) break;

            double randomValue = curand_uniform(&state);
            int selected_index = 0;

            // Оптимизация 5: Используем указатели для лучшей локализации памяти
            const double* prob_ptr = &norm_matrix_probability[param_idx * MAX_VALUE_SIZE];

#if MAX_VALUE_SIZE == 5
            // Ручное развертывание для 5 значений
            selected_index = (randomValue > prob_ptr[0]) ?
                ((randomValue > prob_ptr[2]) ?
                    ((randomValue > prob_ptr[3]) ? 4 : 3) :
                    ((randomValue > prob_ptr[1]) ? 2 : 1)) : 0;
#else
            // Автоматическое развертывание
#pragma unroll
            for (int k = 0; k < MAX_VALUE_SIZE - 1; k++) {
                if (randomValue > prob_ptr[k]) {
                    selected_index = k + 1;
                }
                else {
                    break;
                }
            }
#endif

            shared_selected[param_idx] = selected_index;
            shared_agent_params[param_idx] = dev_parametr_value[param_idx * MAX_VALUE_SIZE + selected_index];
        }
    }

    __syncthreads();

    // 2. ВЫЧИСЛЕНИЕ ЦЕЛЕВОЙ ФУНКЦИИ с warp-оптимизацией
    if (warp_id == 0) {
        // Только первый warp вычисляет OF
        int temp_nodes[PARAMETR_SIZE];

        // Оптимизация 6: Копирование с coalesced доступом
        if (PARAMETR_SIZE <= 1024 && lane_id < PARAMETR_SIZE) {
#pragma unroll 4
            for (int i = lane_id; i < PARAMETR_SIZE; i += WARP_SIZE) {
                temp_nodes[i] = shared_selected[i];
            }
        }

        // Оптимизация 7: Используем ballot для синхронизации внутри warp
        unsigned active_mask = __ballot_sync(0xFFFFFFFF, lane_id < WARP_SIZE);

        if (lane_id == 0) {
            double cached = getCachedResult(hashTable, temp_nodes, ant_id);

            if (cached < 0.0) {
                shared_OF = BenchShafferaFunction(shared_agent_params);
                saveToCache(hashTable, temp_nodes, ant_id, shared_OF);
            }
            else {
                shared_OF = cached;
                atomicAdd(kol_hash_fail, 1);
            }

            if (shared_OF != ZERO_HASH_RESULT) {
                atomicMax(maxOf_dev, shared_OF);
                atomicMin(minOf_dev, shared_OF);
            }
        }

        // Распространяем shared_OF по warp'у
        shared_OF = __shfl_sync(active_mask, shared_OF, 0);
    }

    __syncthreads();

    // 3. ДЕПОЗИТ ФЕРОМОНОВ с warp-сокращением
    if (shared_OF != ZERO_HASH_RESULT) {
        // Оптимизация 8: Вычисляем дельту феромона с warp-редукцией
        double pheromone_delta = 0.0;

        if (lane_id == 0) {
#if OPTIMIZE_MIN_1
            const double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - shared_OF;
            if (delta > 0.0) pheromone_delta = PARAMETR_Q * delta;
#elif OPTIMIZE_MIN_2
            const double of_val = fmax(shared_OF, 1e-7);
            pheromone_delta = PARAMETR_Q / of_val;
#elif OPTIMIZE_MAX
            pheromone_delta = PARAMETR_Q * shared_OF;
#endif
        }

        // Распространяем дельту по warp'у
        unsigned warp_mask = __ballot_sync(0xFFFFFFFF, true);
        pheromone_delta = __shfl_sync(warp_mask, pheromone_delta, 0);

        // Оптимизация 9: Используем tile-based доступ для лучшей локализации
        const int num_tiles = (PARAMETR_SIZE + TILE_SIZE - 1) / TILE_SIZE;

        for (int tile = warp_id; tile < num_tiles; tile += blockDim.x / WARP_SIZE) {
            const int tile_start = tile * TILE_SIZE;
            const int tile_end = min(tile_start + TILE_SIZE, PARAMETR_SIZE);

            // Оптимизация 10: Предзагрузка selected indices для тайла
            int tile_selected[TILE_SIZE];
            if (lane_id < TILE_SIZE && tile_start + lane_id < tile_end) {
                tile_selected[lane_id] = shared_selected[tile_start + lane_id];
            }

            // Обновляем феромоны в тайле
            for (int i = lane_id; i < (tile_end - tile_start); i += WARP_SIZE) {
                const int param_idx = tile_start + i;
                const int selected_idx = tile_selected[i];
                const int pheromon_idx = param_idx * MAX_VALUE_SIZE + selected_idx;

                // Оптимизация 11: Объединяем атомарные операции
                if (pheromone_delta > 0.0) {
                    // Атомарное обновление kol_enter и pheromon
                    atomicAdd(&kol_enter[pheromon_idx], 1.0);
                    atomicAdd(&pheromon[pheromon_idx], pheromone_delta);
                }
                else {
                    atomicAdd(&kol_enter[pheromon_idx], 1.0);
                }
            }
        }
    }
}

__global__ void antColonyOptimizationAndDeposit(double* __restrict__ dev_parametr_value, double* __restrict__ norm_matrix_probability, HashEntry* __restrict__ hashTable, double* __restrict__ maxOf_dev, double* __restrict__ minOf_dev, int* __restrict__ kol_hash_fail, double* __restrict__ pheromon, double* __restrict__ kol_enter, int* __restrict__ agent_node, double* __restrict__ OF, double* __restrict__ global_params_buffer) {
#if (PARAMETR_SIZE < 4095) 
    antColonyOptimization_and_deposit_warp_optimized_dev(dev_parametr_value, norm_matrix_probability, hashTable, maxOf_dev, minOf_dev, kol_hash_fail, pheromon, kol_enter);
#else
    antColonyOptimization_and_deposit_warp_optimized_global_dev(dev_parametr_value, norm_matrix_probability, hashTable, maxOf_dev, minOf_dev, kol_hash_fail, pheromon, kol_enter, agent_node, OF, global_params_buffer);
#endif
}

__global__ void computeProbabilitiesAndEvaporete(const bool go_transposed, double* __restrict__ pheromon, double* __restrict__ kol_enter, double* __restrict__ norm_matrix_probability) {
    evaporatePheromones_dev(pheromon);
    computeProbabilities_dev(pheromon, kol_enter, norm_matrix_probability);
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

    // Вычисление вероятностей (computeProbabilities)

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
    /*
    double* parametr_value = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    double* pheromon_value = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    double* kol_enter_value = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    CUDA_CHECK(cudaMemcpyAsync(parametr_value, dev_parametr_value, MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(pheromon_value, dev_pheromon, MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(kol_enter_value, dev_kol_enter, MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
    for (int j = 0; j < PARAMETR_SIZE; ++j) {
        for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
            std::cout << "(" << pheromon_value[j * MAX_VALUE_SIZE + i] << ", " << kol_enter_value[j * MAX_VALUE_SIZE + i] << "-> " << parametr_value[j * MAX_VALUE_SIZE + i] << ") "; // Индексируем элементы
        }
        std::cout << std::endl; // Переход на новую строку
    }
    */
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
// ==================== ФУНКЦИИ УПРАВЛЕНИЯ ПАМЯТЬЮ С MPI ====================
// Объявление функции для частичной очистки
void cleanup_cuda_resources_mpi_partial() {
    // Очищаем только то, что было выделено
    if (g_resources.compute_stream) {
        cudaStreamDestroy(g_resources.compute_stream);
        g_resources.compute_stream = nullptr;
    }

    if (g_resources.dev_parametr_value) {
        cudaFree(g_resources.dev_parametr_value);
        g_resources.dev_parametr_value = nullptr;
    }
    if (g_resources.dev_agent_params) {
        cudaFree(g_resources.dev_agent_params);
        g_resources.dev_agent_params = nullptr;
    }
    if (g_resources.dev_pheromon) {
        cudaFree(g_resources.dev_pheromon);
        g_resources.dev_pheromon = nullptr;
    }
    if (g_resources.dev_kol_enter) {
        cudaFree(g_resources.dev_kol_enter);
        g_resources.dev_kol_enter = nullptr;
    }
    if (g_resources.dev_norm_matrix) {
        cudaFree(g_resources.dev_norm_matrix);
        g_resources.dev_norm_matrix = nullptr;
    }
    if (g_resources.dev_agent_node) {
        cudaFree(g_resources.dev_agent_node);
        g_resources.dev_agent_node = nullptr;
    }
    if (g_resources.dev_OF) {
        cudaFree(g_resources.dev_OF);
        g_resources.dev_OF = nullptr;
    }
    if (g_resources.dev_max) {
        cudaFree(g_resources.dev_max);
        g_resources.dev_max = nullptr;
    }
    if (g_resources.dev_min) {
        cudaFree(g_resources.dev_min);
        g_resources.dev_min = nullptr;
    }
    if (g_resources.dev_hash_fail) {
        cudaFree(g_resources.dev_hash_fail);
        g_resources.dev_hash_fail = nullptr;
    }
    if (g_resources.dev_hashTable) {
        cudaFree(g_resources.dev_hashTable);
        g_resources.dev_hashTable = nullptr;
    }

    delete[] g_resources.host_pheromon_buffer;
    delete[] g_resources.host_kol_enter_buffer;
    delete[] g_resources.host_norm_matrix_buffer;

    // Обнуляем указатели
    g_resources.host_pheromon_buffer = nullptr;
    g_resources.host_kol_enter_buffer = nullptr;
    g_resources.host_norm_matrix_buffer = nullptr;
}

void cleanup_cuda_resources_mpi() {
    // Сначала синхронизируем все операции
    if (g_resources.compute_stream) {
        cudaStreamSynchronize(g_resources.compute_stream);
        cudaStreamDestroy(g_resources.compute_stream);
        g_resources.compute_stream = nullptr;
    }

    // Освобождаем GPU ресурсы
    if (g_resources.dev_parametr_value) {
        cudaFree(g_resources.dev_parametr_value);
        g_resources.dev_parametr_value = nullptr;
    }
    if (g_resources.dev_agent_params) {
        cudaFree(g_resources.dev_agent_params);
        g_resources.dev_agent_params = nullptr;
    }
    if (g_resources.dev_pheromon) {
        cudaFree(g_resources.dev_pheromon);
        g_resources.dev_pheromon = nullptr;
    }
    if (g_resources.dev_kol_enter) {
        cudaFree(g_resources.dev_kol_enter);
        g_resources.dev_kol_enter = nullptr;
    }
    if (g_resources.dev_norm_matrix) {
        cudaFree(g_resources.dev_norm_matrix);
        g_resources.dev_norm_matrix = nullptr;
    }
    if (g_resources.dev_agent_node) {
        cudaFree(g_resources.dev_agent_node);
        g_resources.dev_agent_node = nullptr;
    }
    if (g_resources.dev_OF) {
        cudaFree(g_resources.dev_OF);
        g_resources.dev_OF = nullptr;
    }
    if (g_resources.dev_max) {
        cudaFree(g_resources.dev_max);
        g_resources.dev_max = nullptr;
    }
    if (g_resources.dev_min) {
        cudaFree(g_resources.dev_min);
        g_resources.dev_min = nullptr;
    }
    if (g_resources.dev_hash_fail) {
        cudaFree(g_resources.dev_hash_fail);
        g_resources.dev_hash_fail = nullptr;
    }
    if (g_resources.dev_hashTable) {
        cudaFree(g_resources.dev_hashTable);
        g_resources.dev_hashTable = nullptr;
    }

    // Освобождаем host буферы
    delete[] g_resources.host_pheromon_buffer;
    delete[] g_resources.host_kol_enter_buffer;
    delete[] g_resources.host_norm_matrix_buffer;

    // Обнуляем все указатели
    g_resources.host_pheromon_buffer = nullptr;
    g_resources.host_kol_enter_buffer = nullptr;
    g_resources.host_norm_matrix_buffer = nullptr;
}

bool initialize_cuda_resources_mpi(const double* params, const double* pheromon, const double* kol_enter) {
    // Проверяем, что предыдущие ресурсы были очищены
    if (g_resources.compute_stream != nullptr ||
        g_resources.dev_parametr_value != nullptr) {
        std::cerr << "Process " << mpi_world_rank
            << ": CUDA resources already initialized! Cleaning up first." << std::endl;
        cleanup_cuda_resources_mpi();
    }

    // Сначала проверяем доступность CUDA устройства
    cudaError_t cuda_status;

    // Проверяем, что устройство установлено правильно
    int current_device;
    cuda_status = cudaGetDevice(&current_device);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Process " << mpi_world_rank << ": Failed to get current device. Error: "
            << cudaGetErrorString(cuda_status) << std::endl;
        return false;
    }

    // Создаем stream с явной проверкой
    cuda_status = cudaStreamCreate(&g_resources.compute_stream);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Process " << mpi_world_rank << ": Failed to create CUDA stream. Error: "
            << cudaGetErrorString(cuda_status) << std::endl;
        return false;
    }

    const size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    const size_t ant_matrix_size = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    const size_t matrix_elements = MAX_VALUE_SIZE * PARAMETR_SIZE;

    // Объявляем переменные здесь, чтобы избежать warnings о bypass initialization
    int threads = 0;
    int blocks = 0;

    // Выделение памяти на устройстве с проверкой ошибок
#define ALLOCATE_AND_CHECK(ptr, size, name) \
        cuda_status = cudaMallocAsync(&ptr, size, g_resources.compute_stream); \
        if (cuda_status != cudaSuccess) { \
            std::cerr << "Process " << mpi_world_rank << ": Failed to allocate " << name << ". Error: " \
                      << cudaGetErrorString(cuda_status) << std::endl; \
            goto cleanup_error; \
        }

    ALLOCATE_AND_CHECK(g_resources.dev_parametr_value, matrix_size, "dev_parametr_value");
    ALLOCATE_AND_CHECK(g_resources.dev_agent_params, PARAMETR_SIZE * ANT_SIZE * sizeof(double), "dev_agent_params");
    ALLOCATE_AND_CHECK(g_resources.dev_pheromon, matrix_size, "dev_pheromon");
    ALLOCATE_AND_CHECK(g_resources.dev_kol_enter, matrix_size, "dev_kol_enter");
    ALLOCATE_AND_CHECK(g_resources.dev_norm_matrix, matrix_size, "dev_norm_matrix");
    ALLOCATE_AND_CHECK(g_resources.dev_agent_node, ant_matrix_size, "dev_agent_node");
    ALLOCATE_AND_CHECK(g_resources.dev_OF, ANT_SIZE * sizeof(double), "dev_OF");
    ALLOCATE_AND_CHECK(g_resources.dev_max, sizeof(double), "dev_max");
    ALLOCATE_AND_CHECK(g_resources.dev_min, sizeof(double), "dev_min");
    ALLOCATE_AND_CHECK(g_resources.dev_hash_fail, sizeof(int), "dev_hash_fail");
    ALLOCATE_AND_CHECK(g_resources.dev_hashTable, HASH_TABLE_SIZE * sizeof(HashEntry), "dev_hashTable");

#undef ALLOCATE_AND_CHECK

    // Выделение host буферов
    try {
        g_resources.host_pheromon_buffer = new double[matrix_elements];
        g_resources.host_kol_enter_buffer = new double[matrix_elements];
        g_resources.host_norm_matrix_buffer = new double[matrix_elements];
    }
    catch (const std::bad_alloc& e) {
        std::cerr << "Process " << mpi_world_rank << ": Failed to allocate host buffers: " << e.what() << std::endl;
        goto cleanup_error;
    }

    // Инициализация буферов нулями
    memset(g_resources.host_pheromon_buffer, 0, matrix_elements * sizeof(double));
    memset(g_resources.host_kol_enter_buffer, 0, matrix_elements * sizeof(double));
    memset(g_resources.host_norm_matrix_buffer, 0, matrix_elements * sizeof(double));

    // Копируем общие данные (параметры) на GPU
    cuda_status = cudaMemcpyAsync(g_resources.dev_parametr_value, params, matrix_size,
        cudaMemcpyHostToDevice, g_resources.compute_stream);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Process " << mpi_world_rank << ": Failed to copy dev_parametr_value. Error: "
            << cudaGetErrorString(cuda_status) << std::endl;
        goto cleanup_error;
    }

    cuda_status = cudaMemcpyAsync(g_resources.dev_pheromon, pheromon, matrix_size,
        cudaMemcpyHostToDevice, g_resources.compute_stream);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Process " << mpi_world_rank << ": Failed to copy dev_pheromon. Error: "
            << cudaGetErrorString(cuda_status) << std::endl;
        goto cleanup_error;
    }

    cuda_status = cudaMemcpyAsync(g_resources.dev_kol_enter, kol_enter, matrix_size,
        cudaMemcpyHostToDevice, g_resources.compute_stream);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Process " << mpi_world_rank << ": Failed to copy dev_kol_enter. Error: "
            << cudaGetErrorString(cuda_status) << std::endl;
        goto cleanup_error;
    }

    // Инициализация хэш-таблицы
    threads = BLOCK_SIZE;
    blocks = (HASH_TABLE_SIZE + threads - 1) / threads;

    // Запускаем ядро и проверяем ошибки
    initializeHashTable << <blocks, threads, 0, g_resources.compute_stream >> > (g_resources.dev_hashTable);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        std::cerr << "Process " << mpi_world_rank << ": Kernel launch failed: "
            << cudaGetErrorString(cuda_status) << std::endl;
        goto cleanup_error;
    }

    // Синхронизируем stream
    cuda_status = cudaStreamSynchronize(g_resources.compute_stream);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Process " << mpi_world_rank << ": Stream synchronization failed: "
            << cudaGetErrorString(cuda_status) << std::endl;
        goto cleanup_error;
    }

    // Явная синхронизация устройства
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        std::cerr << "Process " << mpi_world_rank << ": Device synchronization failed: "
            << cudaGetErrorString(cuda_status) << std::endl;
        goto cleanup_error;
    }
    return true;

cleanup_error:
    std::cerr << "Process " << mpi_world_rank << ": CUDA error during initialization: "
        << cudaGetErrorString(cuda_status) << std::endl;

    // Частичная очистка при ошибке
    cleanup_cuda_resources_mpi_partial();
    return false;
}
void print_matrix() {
    double* parametr_value = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    double* pheromon_value = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    double* kol_enter_value = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    double* norm_matrix = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    CUDA_CHECK(cudaMemcpyAsync(parametr_value, g_resources.dev_parametr_value, MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(norm_matrix, g_resources.dev_norm_matrix, MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(pheromon_value, g_resources.dev_pheromon, MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(kol_enter_value, g_resources.dev_kol_enter, MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
    for (int j = 0; j < PARAMETR_SIZE; ++j) {
        for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
            std::cout << "(" << pheromon_value[j * MAX_VALUE_SIZE + i] << ", " << kol_enter_value[j * MAX_VALUE_SIZE + i] << "-> " << norm_matrix[j * MAX_VALUE_SIZE + i] << ") "; // Индексируем элементы
        }
        std::cout << std::endl; // Переход на новую строку
    }
}

// ==================== ОСНОВНАЯ ФУНКЦИЯ ВЫПОЛНЕНИЯ ИТЕРАЦИЙ С MPI ====================
PerformanceMetrics run_combined_iterations_mpi(const bool go_min_parametrs, const bool go_transposed, int num_iterations) {
    PerformanceMetrics metrics = { 0 };
    auto total_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t start, start_ant, stop;
    cudaError_t cuda_status;

    cuda_status = cudaEventCreate(&start);
    cuda_status = cudaEventCreate(&start_ant);
    cuda_status = cudaEventCreate(&stop);

    float kernel_time = 0.0, combined_time = 0.0, antColonyOptimization_time = 0.0;
    int threads_per_block = std::min(PARAMETR_SIZE, BLOCK_SIZE);
    int ant_blocks = ANT_SIZE;

    const size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
    const size_t matrix_bytes = matrix_size * sizeof(double);

    // Инициализация статистики
    double max_init = -1e9, min_init = 1e9;
    int fail_init = 0;
    double* global_pheromon = nullptr;
    double* global_kol_enter = nullptr;
    double* global_pheromon_reduction = nullptr;
    double* global_kol_enter_reduction = nullptr;

    if (mpi_is_master) {
        // Буферы для редукции на главном процессе
        try {
            global_pheromon = new double[matrix_size];
            global_kol_enter = new double[matrix_size];
            global_pheromon_reduction = new double[matrix_size];
            global_kol_enter_reduction = new double[matrix_size];

            // Инициализируем глобальные феромоны и посещения
            for (size_t i = 0; i < matrix_size; i++) {
                global_pheromon[i] = 1.0;  // Начальное значение феромона
                global_kol_enter[i] = 1.0; // Начальное значение посещений
            }
        }
        catch (const std::bad_alloc& e) {
            std::cerr << "Process " << mpi_world_rank << ": Failed to allocate global buffers: " << e.what() << std::endl;
            cudaEventDestroy(start);
            cudaEventDestroy(start_ant);
            cudaEventDestroy(stop);
            return metrics;
        }
    }
    else {
        // Для не-мастер процессов выделяем временные буферы для редукции
        try {
            global_pheromon_reduction = new double[matrix_size];
            global_kol_enter_reduction = new double[matrix_size];
        }
        catch (const std::bad_alloc& e) {
            std::cerr << "Process " << mpi_world_rank << ": Failed to allocate reduction buffers: " << e.what() << std::endl;
            cudaEventDestroy(start);
            cudaEventDestroy(start_ant);
            cudaEventDestroy(stop);
            return metrics;
        }
    }

    // Копируем начальные значения на GPU
    cuda_status = cudaMemcpyAsync(g_resources.dev_max, &max_init, sizeof(double),cudaMemcpyHostToDevice, g_resources.compute_stream);
    cuda_status = cudaMemcpyAsync(g_resources.dev_min, &min_init, sizeof(double), cudaMemcpyHostToDevice, g_resources.compute_stream);
#if (GO_HASH)
    cuda_status = cudaMemcpyAsync(g_resources.dev_hash_fail, &fail_init, sizeof(int), cudaMemcpyHostToDevice, g_resources.compute_stream);
#endif
    // Инициализация матриц вероятностей на всех процессах
    {
        // Создаем начальную матрицу вероятностей
        std::vector<double> init_norm_matrix(matrix_size, 0.0);

        // Равномерное распределение вероятностей
        for (size_t i = 0; i < matrix_size; i += MAX_VALUE_SIZE) {
            double cumulative = 0.0;
            double step = 1.0 / MAX_VALUE_SIZE;
            for (int j = 0; j < MAX_VALUE_SIZE; j++) {
                cumulative += step;
                init_norm_matrix[i + j] = cumulative;
            }
            init_norm_matrix[i + MAX_VALUE_SIZE - 1] = 1.0; // Обеспечиваем точное значение 1.0
        }

        // Копируем на GPU
        cuda_status = cudaMemcpyAsync(g_resources.dev_norm_matrix, init_norm_matrix.data(), matrix_bytes, cudaMemcpyHostToDevice, g_resources.compute_stream);
        // Копируем в host буфер для broadcast
        memcpy(g_resources.host_norm_matrix_buffer, init_norm_matrix.data(), matrix_bytes);
        // На главном процессе также инициализируем феромоны и посещения
        if (mpi_is_master) {
            cuda_status = cudaMemcpyAsync(g_resources.dev_pheromon, global_pheromon, matrix_bytes, cudaMemcpyHostToDevice, g_resources.compute_stream);
            cuda_status = cudaMemcpyAsync(g_resources.dev_kol_enter, global_kol_enter, matrix_bytes, cudaMemcpyHostToDevice, g_resources.compute_stream);
        }
        else {
            // На не-мастер процессах инициализируем нулями
            std::vector<double> zeros(matrix_size, 0.0);
            cuda_status = cudaMemcpyAsync(g_resources.dev_pheromon, zeros.data(), matrix_bytes, cudaMemcpyHostToDevice, g_resources.compute_stream);
            cuda_status = cudaMemcpyAsync(g_resources.dev_kol_enter, zeros.data(), matrix_bytes, cudaMemcpyHostToDevice, g_resources.compute_stream);
        }
        cuda_status = cudaStreamSynchronize(g_resources.compute_stream);
    }

    // Рассылаем матрицу вероятностей всем процессам (уже инициализирована, но для надежности)
    mpi_synchronize();
    mpi_broadcast_norm_matrix(g_resources.host_norm_matrix_buffer, matrix_size);
    mpi_synchronize();
    // Основной цикл итераций
    for (int iter = 0; iter < num_iterations; ++iter) {
        float iter_time;

        // Каждый процесс выполняет оптимизацию со своими муравьями
        cuda_status = cudaEventRecord(start_ant, g_resources.compute_stream);

        // Запуск оптимизации муравьев
        antColonyOptimizationAndDeposit << <ant_blocks, threads_per_block, 0, g_resources.compute_stream >> > (g_resources.dev_parametr_value, g_resources.dev_norm_matrix, g_resources.dev_hashTable, g_resources.dev_max, g_resources.dev_min, g_resources.dev_hash_fail, g_resources.dev_pheromon, g_resources.dev_kol_enter, g_resources.dev_agent_node, g_resources.dev_OF, g_resources.dev_agent_params);

        cuda_status = cudaGetLastError();
        cuda_status = cudaEventRecord(stop, g_resources.compute_stream);
        cuda_status = cudaStreamSynchronize(g_resources.compute_stream);
        cuda_status = cudaEventElapsedTime(&iter_time, start_ant, stop);
        antColonyOptimization_time += iter_time;

        // Копируем локальные данные на хост
        cuda_status = cudaMemcpyAsync(g_resources.host_pheromon_buffer, g_resources.dev_pheromon, matrix_bytes, cudaMemcpyDeviceToHost, g_resources.compute_stream);
        cuda_status = cudaMemcpyAsync(g_resources.host_kol_enter_buffer, g_resources.dev_kol_enter, matrix_bytes, cudaMemcpyDeviceToHost, g_resources.compute_stream);

        cuda_status = cudaStreamSynchronize(g_resources.compute_stream);
        // Собираем локальные феромоны и посещения на главном процессе через редукцию
        mpi_synchronize();
        // Обнуляем буферы редукции
        memset(global_pheromon_reduction, 0, matrix_bytes);
        memset(global_kol_enter_reduction, 0, matrix_bytes);

        mpi_reduce_pheromones(g_resources.host_pheromon_buffer, global_pheromon_reduction, matrix_size);
        mpi_reduce_visits(g_resources.host_kol_enter_buffer, global_kol_enter_reduction, matrix_size);

        mpi_synchronize();

        // Главный процесс обновляет глобальные феромоны
        if (mpi_is_master) {
            cuda_status = cudaEventRecord(start, g_resources.compute_stream);
            // Добавляем результаты редукции к общему количеству феромона и посещений
            for (size_t nom_matrix = 0; nom_matrix < matrix_size; ++nom_matrix) {
                global_pheromon[nom_matrix] += global_pheromon_reduction[nom_matrix];
                global_kol_enter[nom_matrix] += global_kol_enter_reduction[nom_matrix];
            }

            // Копируем редуцированные данные на GPU главного процесса
            cuda_status = cudaMemcpyAsync(g_resources.dev_pheromon, global_pheromon, matrix_bytes, cudaMemcpyHostToDevice, g_resources.compute_stream);
            cuda_status = cudaMemcpyAsync(g_resources.dev_kol_enter, global_kol_enter, matrix_bytes, cudaMemcpyHostToDevice, g_resources.compute_stream);
            cuda_status = cudaStreamSynchronize(g_resources.compute_stream);

            // Вычисляем новую матрицу вероятностей
            int blocks = (PARAMETR_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
            computeProbabilitiesAndEvaporete << <blocks, BLOCK_SIZE, 0, g_resources.compute_stream >> > (go_transposed, g_resources.dev_pheromon, g_resources.dev_kol_enter, g_resources.dev_norm_matrix);

            cuda_status = cudaGetLastError();
            // Копируем для рассылки
            cuda_status = cudaMemcpyAsync(g_resources.host_norm_matrix_buffer, g_resources.dev_norm_matrix, matrix_bytes, cudaMemcpyDeviceToHost, g_resources.compute_stream);
            cuda_status = cudaEventRecord(stop, g_resources.compute_stream);
            cuda_status = cudaStreamSynchronize(g_resources.compute_stream);
            cuda_status = cudaEventElapsedTime(&iter_time, start, stop);
            combined_time += iter_time;
        }

        // Все процессы копируют новую матрицу на свои GPU
        mpi_synchronize();
        mpi_broadcast_norm_matrix(g_resources.host_norm_matrix_buffer, matrix_size);
        mpi_synchronize();

        cuda_status = cudaMemcpyAsync(g_resources.dev_norm_matrix, g_resources.host_norm_matrix_buffer,matrix_bytes, cudaMemcpyHostToDevice, g_resources.compute_stream);
        cuda_status = cudaStreamSynchronize(g_resources.compute_stream);

        // Сбрасываем локальные феромоны и посещения к нулю для следующей итерации
        std::vector<double> zeros(matrix_size, 0.0);
        cuda_status = cudaMemcpyAsync(g_resources.dev_pheromon, zeros.data(),matrix_bytes, cudaMemcpyHostToDevice, g_resources.compute_stream);
        cuda_status = cudaMemcpyAsync(g_resources.dev_kol_enter, zeros.data(), matrix_bytes, cudaMemcpyHostToDevice, g_resources.compute_stream);
        cuda_status = cudaStreamSynchronize(g_resources.compute_stream);

        mpi_synchronize();
    }

cleanup:
    mpi_synchronize();

    // Освобождаем память
    if (mpi_is_master) {
        delete[] global_pheromon;
        delete[] global_kol_enter;
        delete[] global_pheromon_reduction;
        delete[] global_kol_enter_reduction;
    }
    else {
        delete[] global_pheromon_reduction;
        delete[] global_kol_enter_reduction;
    }

    // Сбор статистики со всех процессов
    double local_best_fitness = 0.0, local_low_fitness = 0.0;
    int local_hash_fails = 0;

    cuda_status = cudaMemcpyAsync(&local_best_fitness, g_resources.dev_min, sizeof(double), cudaMemcpyDeviceToHost, g_resources.compute_stream);
    cuda_status = cudaMemcpyAsync(&local_low_fitness, g_resources.dev_max, sizeof(double), cudaMemcpyDeviceToHost, g_resources.compute_stream);

#if (GO_HASH)
        cuda_status = cudaMemcpyAsync(&local_hash_fails, g_resources.dev_hash_fail, sizeof(int), cudaMemcpyDeviceToHost, g_resources.compute_stream);
#endif
        cudaStreamSynchronize(g_resources.compute_stream);

    double global_best_fitness = 0.0, global_low_fitness = 0.0;
    mpi_gather_statistics(&local_best_fitness, &local_low_fitness, &global_best_fitness, &global_low_fitness);

    if (mpi_is_master) {
        metrics.min_fitness = global_best_fitness;
        metrics.max_fitness = global_low_fitness;

#if (GO_HASH)
        int global_hash_fails = 0;
        MPI_CHECK(MPI_Reduce(&local_hash_fails, &global_hash_fails, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));

        metrics.hash_misses = global_hash_fails;
        metrics.hash_hits = num_iterations * ANT_SIZE * mpi_world_size - global_hash_fails;
#endif

        // Расчет дополнительных метрик
        cudaDeviceProp prop;
        cuda_status = cudaGetDeviceProperties(&prop, 0);
        if (cuda_status == cudaSuccess) {
            int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
            int warp_size = prop.warpSize;
            int warps_per_block = (threads_per_block + warp_size - 1) / warp_size;
            int max_warps_per_sm = max_threads_per_sm / warp_size;
            metrics.occupancy = std::min(100.0f, (warps_per_block * 32.0f / max_warps_per_sm) * 100.0f);
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        metrics.total_time_ms = std::chrono::duration<float, std::milli>(total_end - total_start).count();
        metrics.kernel_time_ms = kernel_time;
        metrics.computeProbabilities_time_ms = combined_time;
        metrics.updatePheromones_time_ms = 0.0;
        metrics.antColonyOptimization_time_ms = antColonyOptimization_time;
        metrics.memory_time_ms = metrics.total_time_ms - kernel_time;

        // Расчет пропускной способности памяти
        size_t total_data_transferred = (matrix_bytes * 4 * num_iterations * mpi_world_size) + (PARAMETR_SIZE * ANT_SIZE * sizeof(int)) * num_iterations * mpi_world_size + (ANT_SIZE * sizeof(double)) * num_iterations * mpi_world_size;
        metrics.memory_throughput_gbs = (total_data_transferred / (1024.0 * 1024.0 * 1024.0)) / (metrics.total_time_ms / 1000.0);
    }

    // Освобождаем события
    cudaEventDestroy(start);
    cudaEventDestroy(start_ant);
    cudaEventDestroy(stop);
    return metrics;
}
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
    if (!mpi_is_master) return; // Только мастер выводит
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
    if (logFilePtr && logFilePtr->is_open()) {
        *logFilePtr << "Run " << str << run_id << "; "
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
}
int print_information() {
    if (!mpi_is_master) return 0; // Только мастер выводит

    // Создание векторов для статистики 
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "Don't have CUDA device." << std::endl;
        return 1;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        // Вывод в консоль
        std::cout << "Device " << device << ": " << prop.name << std::endl;
        std::cout << "Max thread in blocks: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "GPU Global Memory Usage: " << calculate_gpu_memory_usage() / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "GPU Global Memory Hash Table: " << HASH_TABLE_SIZE * sizeof(HashEntry) / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Max PARAMETR_SIZE in GPU Global Memory: " << int(calculate_max_parametr_size_in_gpu_memory()) << " " << std::endl;
        std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes (" << prop.sharedMemPerBlock / 1024 << " KB)" << std::endl;
        std::cout << "Shared memory per multiprocessor: " << prop.sharedMemPerMultiprocessor << " bytes (" << prop.sharedMemPerMultiprocessor / 1024 << " KB)" << std::endl;
        std::cout << "Data size in Shared memory per block: " << PARAMETR_SIZE * sizeof(double) << " bytes (" << PARAMETR_SIZE * sizeof(double) / 1024.0 << " KB)" << std::endl;
        std::cout << "Constant Memory: " << prop.totalConstMem << " bytes" << std::endl;
        std::cout << "Parameter data size (in Constant Memory): " << MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double) << " bytes (" << MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double) / 1024.0 << " KB)" << std::endl;
        std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
        std::cout << "Warp Size: " << prop.warpSize << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Can Map Host Memory: " << (prop.canMapHostMemory ? "Yes" : "No") << std::endl;
        std::cout << "Clock Rate: " << prop.clockRate / 1000.0f << " MHz" << std::endl;
        std::cout << "L2 Cache Size: " << (prop.l2CacheSize == 0 ? 0 : prop.l2CacheSize) << " bytes" << std::endl;
        std::cout << "Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "Max thread in blocks by axis: ("
            << prop.maxThreadsDim[0] << ", "
            << prop.maxThreadsDim[1] << ", "
            << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "Max blocks by axis: ("
            << prop.maxGridSize[0] << ", "
            << prop.maxGridSize[1] << ", "
            << prop.maxGridSize[2] << ")" << std::endl;

        // Вывод в файл
        if (logFilePtr && logFilePtr->is_open()) {
            *logFilePtr
                << "Device " << device << ": " << prop.name << "; "
                << "Max thread in blocks: " << prop.maxThreadsPerBlock << " "
                << "Total Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB "
                << "GPU Global Memory Usage: " << calculate_gpu_memory_usage() / (1024.0 * 1024.0) << " MB "
                << "GPU Global Memory Hash Table: " << HASH_TABLE_SIZE * sizeof(HashEntry) / (1024.0 * 1024.0) << " MB "
                << "Max PARAMETR_SIZE in GPU Global Memory: " << int(calculate_max_parametr_size_in_gpu_memory()) << " "
                << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes (" << prop.sharedMemPerBlock / 1024 << " KB) "
                << "Shared memory per multiprocessor: " << prop.sharedMemPerMultiprocessor << " bytes (" << prop.sharedMemPerMultiprocessor / 1024 << " KB) "
                << "Data size in Shared memory per block: " << PARAMETR_SIZE * sizeof(double) << " bytes (" << PARAMETR_SIZE * sizeof(double) / 1024.0 << " KB) "
                << "Constant Memory: " << prop.totalConstMem << " bytes "
                << "Parameter data size (in Constant Memory): " << MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double) << " bytes (" << MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double) / 1024.0 << " KB) "
                << "Registers per Block: " << prop.regsPerBlock << " "
                << "Warp Size: " << prop.warpSize << " "
                << "Compute Capability: " << prop.major << "." << prop.minor << " "
                << "Can Map Host Memory: " << (prop.canMapHostMemory ? "Yes" : "No") << " "
                << "Clock Rate: " << prop.clockRate / 1000.0f << " MHz "
                << "L2 Cache Size: " << (prop.l2CacheSize == 0 ? 0 : prop.l2CacheSize) << " bytes "
                << "Multiprocessor Count: " << prop.multiProcessorCount << "; "
                << "Max thread in blocks by axis: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "); "
                << "Max blocks by axis: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ");";
        }
    }

    // Формируем строки для флагов оптимизации
    std::string optimize_str;
    if (OPTIMIZE_MIN_1) optimize_str += "OPTIMIZE_MIN_1 ";
    if (OPTIMIZE_MIN_2) optimize_str += "OPTIMIZE_MIN_2 ";
    if (OPTIMIZE_MAX) optimize_str += "OPTIMIZE_MAX ";

    // Формируем строки для функций
    std::string function_str;
    if (SHAFFERA) function_str += "SHAFFERA ";
    if (CARROM_TABLE) function_str += "CARROM_TABLE ";
    if (RASTRIGIN) function_str += "RASTRIGIN ";
    if (ACKLEY) function_str += "ACKLEY ";
    if (SPHERE) function_str += "SPHERE ";
    if (GRIEWANK) function_str += "GRIEWANK ";
    if (ZAKHAROV) function_str += "ZAKHAROV ";
    if (SCHWEFEL) function_str += "SCHWEFEL ";
    if (LEVY) function_str += "LEVY ";
    if (MICHAELWICZYNSKI) function_str += "MICHAELWICZYNSKI ";

    // Вывод информации о константах в консоль
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
        << "OPTIMIZE: " << optimize_str << "; "
        << "FUNCTION: " << function_str
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

    if (MAX_VALUE_SIZE * PARAMETR_SIZE < MAX_CONST) {
        std::cout << "USE CONST MEMORY" << std::endl;
    }
    if (PARAMETR_SIZE < MAX_SHARED) {
        std::cout << "USE SHARED MEMORY" << std::endl;
    }

    // Вывод информации о константах в файл
    if (logFilePtr && logFilePtr->is_open()) {
        *logFilePtr
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
            << "OPTIMIZE: " << optimize_str << "; "
            << "FUNCTION: " << function_str
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

        if (MAX_VALUE_SIZE * PARAMETR_SIZE < MAX_CONST) {
            *logFilePtr << "USE CONST MEMORY" << std::endl;
        }
        if (PARAMETR_SIZE < MAX_SHARED) {
            *logFilePtr << "USE SHARED MEMORY" << std::endl;
        }
    }

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
// ==================== MAIN ФУНКЦИЯ С MPI ====================
int main(int argc, char** argv) {
    bool go_min_parametrs = false;
    bool go_transposed = false;

    // Инициализация MPI
    if (!mpi_initialize(&argc, &argv)) {
        std::cerr << "MPI initialization failed" << std::endl;
        return 1;
    }
    // Только мастер открывает файл лога
    if (mpi_is_master) {
        logFilePtr = new std::ofstream("log.txt");
        if (!logFilePtr->is_open()) {
            std::cerr << "Failed to open log file" << std::endl;
            delete logFilePtr;
            logFilePtr = nullptr;
        }
    }

    try {
        // Загрузка данных (только главным процессом)
        std::vector<double> params, pheromones, visits;
        if (mpi_is_master) {
            if (!loadev_matrix_data(NAME_FILE_GRAPH, params, pheromones, visits)) {
                std::cerr << "Failed to load matrix data " << NAME_FILE_GRAPH << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        // Рассылаем размер данных всем процессам
        size_t param_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
        if (mpi_is_master) {
            MPI_CHECK(MPI_Bcast(&param_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD));
        }
        else {
            MPI_CHECK(MPI_Bcast(&param_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD));
            params.resize(param_size);
            pheromones.resize(param_size);
            visits.resize(param_size);
        }

        // Рассылаем данные параметров всем процессам
        MPI_CHECK(MPI_Bcast(params.data(), param_size, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Bcast(pheromones.data(), param_size, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Bcast(visits.data(), param_size, MPI_DOUBLE, 0, MPI_COMM_WORLD));

        // Синхронизация перед инициализацией CUDA
        mpi_synchronize();

        if (mpi_is_master) {
            std::cout << "Memory bounds check:" << std::endl;
            std::cout << "  PARAMETR_SIZE: " << PARAMETR_SIZE << std::endl;
            std::cout << "  MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << std::endl;
            std::cout << "  ANT_SIZE per process: " << ANT_SIZE << std::endl;
            std::cout << "  Total ants across all processes: " << ANT_SIZE * mpi_world_size << std::endl;
            std::cout << "  Total matrix elements: " << PARAMETR_SIZE * MAX_VALUE_SIZE << std::endl;
            print_information();
        }

        // Warmup runs
        for (int i = 0; i < KOL_PROGREV; ++i) {
            if (mpi_is_master) {
                std::cout << "Warmup " << i << std::endl;
            }

            // Инициализация CUDA с проверкой
            bool init_success = initialize_cuda_resources_mpi(params.data(), pheromones.data(), visits.data());

            if (!init_success) {
                std::cerr << "Process " << mpi_world_rank << ": Failed to initialize CUDA resources" << std::endl;
                cleanup_cuda_resources_mpi();
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Глобальная проверка успешности инициализации
            int all_success = 0;
            int local_success = init_success ? 1 : 0;
            MPI_CHECK(MPI_Allreduce(&local_success, &all_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));

            if (all_success != 1) {
                if (mpi_is_master) {
                    std::cerr << "ERROR: CUDA initialization failed in one or more processes!" << std::endl;
                }
                cleanup_cuda_resources_mpi();
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            mpi_synchronize();

            // Запуск оптимизации
            try {
                run_combined_iterations_mpi(go_min_parametrs, go_transposed, KOL_ITERATION);
            }
            catch (const std::exception& e) {
                std::cerr << "Process " << mpi_world_rank << ": Error in run_combined_iterations_mpi: "  << e.what() << std::endl;
                cleanup_cuda_resources_mpi();
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Очистка ресурсов
            cleanup_cuda_resources_mpi();

            // Проверяем, что все процессы завершили очистку
            mpi_synchronize();
        }

        for (int i = 0; i < KOL_PROGON_STATISTICS; ++i) {

            // Инициализация CUDA с проверкой
            bool init_success = initialize_cuda_resources_mpi(params.data(), pheromones.data(), visits.data());

            if (!init_success) {
                std::cerr << "Process " << mpi_world_rank << ": Failed to initialize CUDA resources" << std::endl;
                cleanup_cuda_resources_mpi();
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Глобальная проверка успешности инициализации
            int all_success = 0;
            int local_success = init_success ? 1 : 0;
            MPI_CHECK(MPI_Allreduce(&local_success, &all_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));

            if (all_success != 1) {
                if (mpi_is_master) {
                    std::cerr << "ERROR: CUDA initialization failed in one or more processes!" << std::endl;
                }
                cleanup_cuda_resources_mpi();
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            mpi_synchronize();

            // Запуск оптимизации
            try {
                PerformanceMetrics metrics =  run_combined_iterations_mpi(go_min_parametrs, go_transposed, KOL_ITERATION);
                if (mpi_is_master) {
                    print_metrics(metrics, "MPI Combined Run", i);
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Process " << mpi_world_rank << ": Error in run_combined_iterations_mpi: "  << e.what() << std::endl;
                cleanup_cuda_resources_mpi();
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Очистка ресурсов
            cleanup_cuda_resources_mpi();

            // Проверяем, что все процессы завершили очистку
            mpi_synchronize();
        }
        /*
        // Основные запуски
        if (GO_CUDA_2_STEP_AGENT && mpi_is_master) {
            std::cout << "\n=== Starting MPI ACO and DEPOSIT runs ===" << std::endl;
            auto total_start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < KOL_PROGON_STATISTICS; ++i) {
                if (mpi_is_master) {
                    std::cout << "Run " << i + 1 << "/" << KOL_PROGON_STATISTICS << std::endl;
                }

                // Инициализация CUDA ресурсов
                bool init_success = initialize_cuda_resources_mpi(params.data(), pheromones.data(), visits.data());

                if (!init_success) {
                    std::cerr << "Process " << mpi_world_rank << ": Failed to initialize CUDA resources for run " << i << std::endl;
                    cleanup_cuda_resources_mpi();
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                // Глобальная проверка
                int all_success = 0;
                int local_success = init_success ? 1 : 0;
                MPI_CHECK(MPI_Allreduce(&local_success, &all_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));

                if (all_success != 1) {
                    if (mpi_is_master) {
                        std::cerr << "ERROR: CUDA initialization failed for run " << i << std::endl;
                    }
                    cleanup_cuda_resources_mpi();
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                mpi_synchronize();

                try {
                    PerformanceMetrics metrics = run_combined_iterations_mpi(go_min_parametrs, go_transposed, KOL_ITERATION);

                    if (mpi_is_master) {
                        print_metrics(metrics, "MPI Combined Run", i);
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "Process " << mpi_world_rank << ": Error in run_combined_iterations_mpi for run " << i << ": " << e.what() << std::endl;
                }

                // Очистка ресурсов
                cleanup_cuda_resources_mpi();

                // Пауза между запусками для стабильности
                mpi_synchronize();

            }

            if (mpi_is_master) {
                auto total_end = std::chrono::high_resolution_clock::now();
                auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    total_end - total_start);
                std::cout << "MPI ACO total execution time: " << total_duration.count() << " ms" << std::endl;
            }
        }
        */
    }
    catch (const std::exception& e) {
        std::cerr << "Process " << mpi_world_rank << ": Unhandled exception: " << e.what() << std::endl;
        cleanup_cuda_resources_mpi();
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Очистка ресурсов
    cleanup_cuda_resources_mpi();

    if (mpi_is_master && logFilePtr) {
        logFilePtr->close();
        delete logFilePtr;
        logFilePtr = nullptr;
    }

    mpi_finalize();

    return 0;
}