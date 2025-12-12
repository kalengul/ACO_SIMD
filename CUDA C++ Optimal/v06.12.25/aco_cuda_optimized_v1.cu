#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <iomanip>
#include <vector>
#include <random>
#include <ctime>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <omp.h>
#include <mutex>
#include <thread>
#include "parametrs.h"

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

// ==================== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ====================
std::ofstream logFile;
std::ofstream outfile("statistics.txt");
std::mutex mtx;

// Константная память CUDA
#if (MAX_VALUE_SIZE * PARAMETR_SIZE < MAX_CONST)
__constant__ double parametr_value_dev_const[MAX_VALUE_SIZE * PARAMETR_SIZE];
#else
__constant__ double parametr_value_dev_const[100];
#endif
__constant__ int gpuTime_const;

// ==================== СТРУКТУРЫ ДАННЫХ ====================
struct HashEntry {
    unsigned long long key;
    double value;
};

struct PerformanceMetrics {
    float total_time_ms;
    float kernel_time_ms;
    float memory_time_ms;
    float occupancy;
    float memory_throughput_gbs;
    double best_fitness;
    int hash_hits;
    int hash_misses;
};

// ==================== ФУНКЦИИ БЕНЧМАРКОВ ====================
__device__ double go_x(double* parametr, int start_index, int kol_parametr) {
    double sum = 0.0;
    for (int i = 1; i < kol_parametr; ++i) {
        sum += parametr[start_index + i];
    }
    return parametr[start_index] * sum;
}

#if (SHAFFERA) 
__device__ double BenchShafferaFunction(double* parametr) {
    double r_squared = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        r_squared += x * x;
    }
    double r = sqrt(r_squared);
    double sin_r = sin(r);
    return 1.0 / 2.0 - (sin_r * sin_r - 0.5) / (1.0 + 0.001 * r_squared);
}
#endif
#if (DELT4)
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
        r_squared += x * x;
        sum += x * x - 10 * cos(2 * M_PI * x) + 10;
        second_sum += cos(2 * M_PI * x);
        r_cos *= cos(x);
    }
    if (sum_if >= -10 * num_variables && sum_if <= -5 * num_variables) {
        double r = sqrt(r_squared);
        double sin_r = sin(r);
        return 1.0 / 2.0 - (sin_r * sin_r - 0.5) / (1.0 + 0.001 * r_squared);
    }
    if (sum_if > -5 * num_variables && sum_if <= 0) {
        return sum;
    }
    if (sum_if > 0 && sum_if <= 5 * num_variables) {
        double exp_term_1 = exp(-0.2 * sqrt(r_squared / num_variables));
        double exp_term_2 = exp(second_sum / num_variables);
        return -20 * exp_term_1 - exp_term_2 + M_E + 20;
    }
    if (sum_if > 5 * num_variables && sum_if <= 10 * num_variables) {
        double a = 1.0 - sqrt(r_squared) / M_PI;
        double OF = r_cos * exp(fabs(a));
        return OF * OF;
    }
    return 0;
}
#endif
#if (CARROM_TABLE) 
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
    double OF = r_cos * exp(fabs(a));
    return OF * OF;
}
#endif
#if (RASTRIGIN)
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
__device__ double BenchShafferaFunction(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum -= x * sin(sqrt(fabs(x)));
    }
    return sum;
}
#endif
#if (LEVY)
__device__ double BenchShafferaFunction(double* parametr) {
    double w_first = 1 + (go_x(parametr, 0, PARAMETR_SIZE_ONE_X) - 1) / 4;
    double w_last = 1 + (go_x(parametr, PARAMETR_SIZE - PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X) - 1) / 4;
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 1; i <= num_variables - 1; ++i) {
        double wi = 1 + (go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X) - 1) / 4;
        sum += pow(wi - 1, 2) * (1 + 10 * pow(sin(M_PI * wi), 2)) +
               pow(wi - 1, 2) * (1 + pow(sin(2 * M_PI * wi), 2));
    }
    return pow(sin(M_PI * w_first), 2) + sum + pow(w_last - 1, 2) * (1 + pow(sin(2 * M_PI * w_last), 2));
}
#endif
#if (MICHAELWICZYNSKI)
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

// ==================== ФУНКЦИИ ХЭШИРОВАНИЯ ====================
__device__ unsigned long long murmurHash64A(unsigned long long key, unsigned long long seed = 0xDEADBEEFDEADBEEF) {
    unsigned long long m = 0xc6a4a7935bd1e995;
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

__device__ unsigned long long betterHashFunction(unsigned long long key) {
    return murmurHash64A(key) % HASH_TABLE_SIZE;
}

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

__device__ double getCachedResultOptimized(HashEntry* hashTable, const int* agent_node, int bx) {
    unsigned long long key = generateKey(agent_node, bx);
    unsigned long long idx = betterHashFunction(key);
    int i = 1;
    while (i <= MAX_PROBES) {
        if (hashTable[idx].key == key) {
            return hashTable[idx].value;
        }
        if (hashTable[idx].key == ZERO_HASH_RESULT) {
            return -1.0;
        }
        unsigned long long new_idx = idx + static_cast<unsigned long long>(i * i); 
        if (new_idx >= HASH_TABLE_SIZE) { 
            new_idx %= HASH_TABLE_SIZE; 
        }
        idx = new_idx;
        i++;
    }
    return -1.0;
}

__device__ void saveToCacheOptimized(HashEntry* hashTable, const int* agent_node, int bx, double value) {
    unsigned long long key = generateKey(agent_node, bx);
    unsigned long long idx = betterHashFunction(key);
    int i = 1;

    while (i <= MAX_PROBES) {
        unsigned long long expected = ZERO_HASH_RESULT;
        unsigned long long desired = key;
        unsigned long long old = atomicCAS(&(hashTable[idx].key), expected, desired);
        if (old == expected || old == key) {
            hashTable[idx].value = value;
            return;
        }
        unsigned long long new_idx = idx + static_cast<unsigned long long>(i * i); 
        if (new_idx >= HASH_TABLE_SIZE) { 
            new_idx %= HASH_TABLE_SIZE; 
        }
        idx = new_idx;
        i++;
    }
}

// ==================== АТОМАРНЫЕ ОПЕРАЦИИ ====================
__device__ void atomicMax(double* address, double value) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;

    do {
        assumed = old;
        if (value > __longlong_as_double(old)) {
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value));
        }
    } while (old != assumed);
}

__device__ void atomicMin(double* address, double value) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;

    do {
        assumed = old;
        if (value < __longlong_as_double(old)) {
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value));
        }
    } while (old != assumed);
}

// ==================== ОСНОВНЫЕ ЯДРА CUDA ====================
__global__ void initializeHashTable(HashEntry* hashTable, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        hashTable[i].key = ZERO_HASH_RESULT;
        hashTable[i].value = 0.0;
    }
}

__device__ double probability_formula(double pheromon, double kol_enter) {
    double res = 0;
    if ((kol_enter != 0) && (pheromon != 0)) {
        res = 1.0 / kol_enter + pheromon;
    }
    return res;
}

__device__ void go_mass_probability_optimized(int tx, double* __restrict__ pheromon, double* __restrict__ kol_enter, double* __restrict__ norm_matrix_probability) {
    __shared__ double shared_sums[BLOCK_SIZE];

    const int start_idx = MAX_VALUE_SIZE * tx;
    double sumVector = 0.0;
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        sumVector += pheromon[start_idx + i];
    }

    const double inv_sumVector = (sumVector != 0.0) ? 1.0 / sumVector : 1.0;
    double svertka_sum = 0.0;

    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        const double pheromon_norm = pheromon[start_idx + i] * inv_sumVector;
        const double svertka_val = probability_formula(pheromon_norm, kol_enter[start_idx + i]);
        norm_matrix_probability[start_idx + i] = svertka_val;
        svertka_sum += svertka_val;
    }

    const double inv_svertka_sum = (svertka_sum != 0.0) ? 1.0 / svertka_sum : 1.0;

    double cumulative = 0.0;
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        cumulative += norm_matrix_probability[start_idx + i] * inv_svertka_sum;
        norm_matrix_probability[start_idx + i] = cumulative;
    }
    
    if (tx == 0) {
        norm_matrix_probability[start_idx + MAX_VALUE_SIZE - 1] = 1.0;
    }
}

__global__ void go_mass_probability_thread(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    int tx = threadIdx.x;
    go_mass_probability_optimized(tx, pheromon, kol_enter, norm_matrix_probability);
}

__device__ void go_ant_path(int tx, int bx, curandState* state, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node) {
    double randomValue = curand_uniform(state);

#if (BIN_SEARCH)
    int low = 0, high = MAX_VALUE_SIZE - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + mid])
            low = mid + 1;
        else
            high = mid - 1;
    }
    int k = low;
#endif
#if (!BIN_SEARCH)
    int k = 0;
    while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
        k++;
    }
    __syncthreads();
#endif

    agent_node[bx * PARAMETR_SIZE + tx] = k;
    agent[tx] = parametr[tx * MAX_VALUE_SIZE + k];
}

__global__ void go_all_agent(double* parametr, double* norm_matrix_probability, int* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    double agent[PARAMETR_SIZE];
    curandState state;
    curand_init((bx * blockDim.x + tx) * clock64() + gpuTime_const, 0, 0, &state);
    
    go_ant_path(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);

    double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
    int nom_iteration = 1;
    
    if (cachedResult == -1.0) {
        OF[bx] = BenchShafferaFunction(agent);
        saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
    } else {
        switch (TYPE_ACO) {
        case 0:
            OF[bx] = cachedResult;
            atomicAdd(&kol_hash_fail[0], 1);
            break;
        case 1:
            OF[bx] = ZERO_HASH_RESULT;
            atomicAdd(&kol_hash_fail[0], 1);
            break;
        case 2:
            while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION)) {
                go_ant_path(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
                cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                nom_iteration = nom_iteration + 1;
                atomicAdd(&kol_hash_fail[0], 1);
            }
            OF[bx] = BenchShafferaFunction(agent);
            if (OF[bx] != ZERO_HASH_RESULT) { 
                saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]); 
            }
            break;
        default:
            OF[bx] = cachedResult;
            atomicAdd(&kol_hash_fail[0], 1);
            break;
        }
    }
    
    __syncthreads();
    
    if (OF[bx] != ZERO_HASH_RESULT) {
        atomicMax(maxOf_dev, OF[bx]);
        atomicMin(minOf_dev, OF[bx]);
    }
}

__device__ void add_pheromon_iteration(int tx, double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
        pheromon[MAX_VALUE_SIZE * tx + i] = pheromon[MAX_VALUE_SIZE * tx + i] * PARAMETR_RO;
    }
    
    for (int i = 0; i < ANT_SIZE; ++i) {
        if (OF[i] != ZERO_HASH_RESULT) {
            int k = agent_node[i * PARAMETR_SIZE + tx];
            kol_enter[MAX_VALUE_SIZE * tx + k]++;
#if (OPTIMIZE_MIN_1)
            if (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i] > 0) {
                pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i]);
            }
#endif
#if (OPTIMIZE_MIN_2)
            if (OF[i] == 0) { OF[i] = 0.0000001; }
            pheromon[MAX_VALUE_SIZE * tx + k] = pheromon[MAX_VALUE_SIZE * tx + k] + PARAMETR_Q / OF[i];
#endif
#if (OPTIMIZE_MAX)
            pheromon[MAX_VALUE_SIZE * tx + k] = pheromon[MAX_VALUE_SIZE * tx + k] + PARAMETR_Q * OF[i];
#endif
        }
    }
}

__global__ void add_pheromon_iteration_thread(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    int tx = threadIdx.x;
    add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
}

// ==================== ФУНКЦИИ ХОСТА ====================
bool load_matrix(const std::string & filename, double* parametr_value, double* pheromon_value, double* kol_enter_value) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Don't open file!" << std::endl;
        return false;
    }
    
    for (int i = 0; i < PARAMETR_SIZE; ++i) {
        for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
            int k = MAX_VALUE_SIZE * i + j;
            if (!(infile >> parametr_value[k])) {
                std::cerr << "Error load element [" << i << "][" << j << "]" << std::endl;
                return false;
            }
            if (parametr_value[k] != -100) {
                pheromon_value[k] = 1.0;
                kol_enter_value[k] = 1.0;
            } else {
                pheromon_value[k] = 0.0;
                parametr_value[k] = 0.0;
                kol_enter_value[k] = 0.0;
            }
        }
    }
    
    infile.close();
    return true;
}

void print_performance_metrics(const PerformanceMetrics& metrics) {
    std::cout << "\n=== PERFORMANCE METRICS ===" << std::endl;
    std::cout << "Total Time: " << metrics.total_time_ms << " ms" << std::endl;
    std::cout << "Kernel Time: " << metrics.kernel_time_ms << " ms" << std::endl;
    std::cout << "Memory Time: " << metrics.memory_time_ms << " ms" << std::endl;
    std::cout << "Occupancy: " << metrics.occupancy << "%" << std::endl;
    std::cout << "Memory Throughput: " << metrics.memory_throughput_gbs << " GB/s" << std::endl;
    std::cout << "Best Fitness: " << metrics.best_fitness << std::endl;
    std::cout << "Hash Hits: " << metrics.hash_hits << std::endl;
    std::cout << "Hash Misses: " << metrics.hash_misses << std::endl;
    std::cout << "Hash Hit Rate: " << (100.0 * metrics.hash_hits / (metrics.hash_hits + metrics.hash_misses)) << "%" << std::endl;
}

void save_metrics_to_csv(const PerformanceMetrics& metrics, const std::string& filename) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) return;
    
    file << std::fixed << std::setprecision(6);
    file << metrics.total_time_ms << ","
         << metrics.kernel_time_ms << ","
         << metrics.memory_time_ms << ","
         << metrics.occupancy << ","
         << metrics.memory_throughput_gbs << ","
         << metrics.best_fitness << ","
         << metrics.hash_hits << ","
         << metrics.hash_misses << ","
         << static_cast<double>(metrics.hash_hits) / (metrics.hash_hits + metrics.hash_misses) * 100.0
         << std::endl;
    
    file.close();
}

void create_csv_header(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "TotalTime(ms),KernelTime(ms),MemoryTime(ms),Occupancy(%),"
         << "MemoryThroughput(GB/s),BestFitness,HashHits,HashMisses,HashHitRate(%)" << std::endl;
    
    file.close();
}

PerformanceMetrics start_CUDA_Time_with_metrics() {
    auto total_start = std::chrono::high_resolution_clock::now();
    PerformanceMetrics metrics = {0};
    
    cudaEvent_t startAll, startAll1, start1, start2, start3, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&start3));
    CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float gpuTime3 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int i_gpuTime = 0;
    
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); 
    long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    
    double global_maxOf = -std::numeric_limits<double>::max();
    double global_minOf = std::numeric_limits<double>::max();

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    if (!load_matrix(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value)) {
        std::cerr << "Error loading matrix!" << std::endl;
        return metrics;
    }

    // Выделение памяти на устройстве
    double* parametr_value_dev = nullptr;
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    HashEntry* hashTable_dev = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));

    // Инициализация хэш-таблицы
    int threads_init_hash = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threads_init_hash - 1) / threads_init_hash;
    initializeHashTable << <blocks_init_hash, threads_init_hash >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError());

    dim3 kol_parametr(PARAMETR_SIZE);
    dim3 kol_ant(ANT_SIZE);

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    
    if (PARAMETR_SIZE < MAX_THREAD_CUDA) {
        for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
            CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int)));
            
            CUDA_CHECK(cudaEventRecord(start1, 0));
            go_mass_probability_thread << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
            CUDA_CHECK(cudaGetLastError());
            
            CUDA_CHECK(cudaEventRecord(start2, 0));
            go_all_agent << <kol_ant, kol_parametr >> > (parametr_value_dev, norm_matrix_probability_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaEventRecord(start3, 0));
            add_pheromon_iteration_thread << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
            CUDA_CHECK(cudaGetLastError());
            
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
            SumgpuTime1 = SumgpuTime1 + gpuTime1;
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
            SumgpuTime2 = SumgpuTime2 + gpuTime2;
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime3, start3, stop));
            SumgpuTime3 = SumgpuTime3 + gpuTime3;
            
            i_gpuTime = int(int(gpuTime * 1000) % 10000000);

            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            
            if ((nom_iter + 1) % kol_shag_stat == 0) {
                int NomStatistics = nom_iter / kol_shag_stat;
                if (PRINT_INFORMATION) { 
                    std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; 
                }
            }
        }
    }
    
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Расчет метрик
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    
    metrics.total_time_ms = total_duration.count();
    metrics.kernel_time_ms = SumgpuTime1 + SumgpuTime2 + SumgpuTime3;
    metrics.best_fitness = *global_minOf_in_device;
    metrics.hash_hits = KOL_ITERATION * ANT_SIZE - *kol_hash_fail_in_device;
    metrics.hash_misses = *kol_hash_fail_in_device;
    
    // Расчет пропускной способности памяти
    size_t total_data_transferred = numBytes_matrix_graph * 3 + // params, pheromones, probabilities
                                   numBytes_matrix_ant +        // agent data
                                   numBytes_ant;               // fitness data
    metrics.memory_throughput_gbs = (total_data_transferred / (1024.0 * 1024.0 * 1024.0)) / (metrics.total_time_ms / 1000.0);
    
    // Оценка occupancy
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int warp_size = prop.warpSize;
    int warps_per_block = (PARAMETR_SIZE + warp_size - 1) / warp_size;
    int max_warps_per_sm = max_threads_per_sm / warp_size;
    metrics.occupancy = std::min(100.0f, (warps_per_block * 32.0f / max_warps_per_sm) * 100.0f);

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(parametr_value_dev));
    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] global_maxOf_in_device;
    delete[] global_minOf_in_device;
    delete[] kol_hash_fail_in_device;

    return metrics;
}

int main(int argc, char* argv[]) {
    logFile.open("log.txt");
    if (!logFile.is_open()) {
        std::cerr << "Ошибка открытия лог-файла!" << std::endl;
        return 1;
    }

    // Создание директории для результатов
    system("mkdir results 2>nul");
    
    // Создание CSV файла с заголовком
    create_csv_header("results/performance_metrics.csv");

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
    }

    std::cout << "PARAMETR_SIZE: " << PARAMETR_SIZE << "; "
              << "MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << "; "
              << "ANT_SIZE: " << ANT_SIZE << "; "
              << "NAME_FILE_GRAPH: " << NAME_FILE_GRAPH << "; "
              << "KOL_ITERATION: " << KOL_ITERATION << std::endl;

    if (GO_CUDA_TIME) {
        // Прогрев
        for (int j = 0; j < KOL_PROGREV; j++) {
            std::cout << "PROGREV " << j << " ";
            auto metrics = start_CUDA_Time_with_metrics();
        }

        // Основные запуски с сбором метрик
        auto total_start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < KOL_PROGON_STATISTICS; i++) {
            std::cout << "Run " << i << " ";
            auto metrics = start_CUDA_Time_with_metrics();
            
            print_performance_metrics(metrics);
            save_metrics_to_csv(metrics, "results/performance_metrics.csv");
        }
        
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
        
        std::cout << "Total execution time: " << total_duration.count() << " ms" << std::endl;
        logFile << "Total execution time: " << total_duration.count() << " ms" << std::endl;
    }

    logFile.close();
    outfile.close();
    return 0;
}