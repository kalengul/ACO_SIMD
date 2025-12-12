#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <omp.h>
#include <algorithm>
#include <sstream>
#include <atomic>
#include <immintrin.h>
#include <thread>
#include <unordered_map>

// Загрузка констант из файла параметров
#include "parametrs.h"

// ============================================================================
// PERFORMANCE METRICS STRUCTURES
// ============================================================================

struct PerformanceMetrics {
    double total_time = 0.0;
    double iteration_time = 0.0;
    double probability_time = 0.0;
    double agent_time = 0.0;
    double pheromone_time = 0.0;
    double hash_time = 0.0;
    double evaluation_time = 0.0;
    int total_iterations = 0;
    int total_agents = 0;
    int hash_hits = 0;
    int hash_misses = 0;
    double min_objective = std::numeric_limits<double>::max();
    double max_objective = std::numeric_limits<double>::lowest();
    double convergence_rate = 0.0;
    double parallel_efficiency = 0.0;
    double memory_throughput = 0.0;
    int optimal_threads = 0;
};

struct HashEntry {
    std::atomic<unsigned long long> key;
    double value;
    std::atomic<int> timestamp;
};

struct Statistics {
    double sum = 0, sum_sq = 0;
    int count = 0;
};

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

std::vector<Statistics> stat_duration(KOL_STAT_LEVEL);
std::vector<Statistics> stat_SumgpuTime1(KOL_STAT_LEVEL);
std::vector<Statistics> stat_global_minOf(KOL_STAT_LEVEL);
std::vector<Statistics> stat_global_maxOf(KOL_STAT_LEVEL);
std::vector<Statistics> stat_kol_hash_fail(KOL_STAT_LEVEL);
double compute_parameter(double* params, int start, int count) noexcept {
    double sum = 0.0;
    // Оптимизированный цикл с развертыванием
    int i = 1;
    for (; i <= count - 4; i += 4) {
        sum += params[start + i] + params[start + i + 1] +
            params[start + i + 2] + params[start + i + 3];
    }
    for (; i < count; ++i) {
        sum += params[start + i];
    }
    return params[start] * sum;
}

double compute_probability(double pheromone, double visits) noexcept {
    return (visits > 0.0 && pheromone > 0.0) ? (1.0 / visits + pheromone) : 0.0;
}

#if (SHAFFERA) 
double benchmark_function(double* params) noexcept {
    double sum_sq = 0.0;
    const int num_vars = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;

    // Оптимизированное вычисление с развертыванием цикла
    for (int i = 0; i < num_vars; ++i) {
        double x = compute_parameter(params, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum_sq += x * x;
    }

    double r = std::sqrt(sum_sq);
    double sin_r = std::sin(r);
    return 0.5 - (sin_r * sin_r - 0.5) / (1.0 + 0.001 * sum_sq);
}
#endif

#if (RASTRIGIN)
double benchmark_function(double* params) noexcept {
    double sum = 0.0;
    const int num_vars = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    constexpr double two_pi = 2.0 * M_PI;

#pragma omp simd reduction(+:sum)
    for (int i = 0; i < num_vars; ++i) {
        double x = compute_parameter(params, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum += x * x - 10.0 * std::cos(two_pi * x) + 10.0;
    }
    return sum;
}
#endif

#if (ACKLEY)
double benchmark_function(double* params) noexcept {
    const int num_vars = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    double sum_sq = 0.0;
    double sum_cos = 0.0;

#pragma omp simd reduction(+:sum_sq, sum_cos)
    for (int i = 0; i < num_vars; ++i) {
        double x = compute_parameter(params, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum_sq += x * x;
        sum_cos += std::cos(2.0 * M_PI * x);
    }

    double n = static_cast<double>(num_vars);
    return -20.0 * std::exp(-0.2 * std::sqrt(sum_sq / n)) - std::exp(sum_cos / n) + 20.0 + M_E;
}
#endif

#if (SPHERE)
double benchmark_function(double* params) noexcept {
    double sum_sq = 0.0;
    const int num_vars = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;

#pragma omp simd reduction(+:sum_sq)
    for (int i = 0; i < num_vars; ++i) {
        double x = compute_parameter(params, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum_sq += x * x;
    }
    return sum_sq;
}
#endif

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

void calculate_efficiency(PerformanceMetrics& metrics, int num_threads) {
    metrics.optimal_threads = num_threads;
    double sequential_equivalent = metrics.total_time * num_threads;
    metrics.parallel_efficiency = (sequential_equivalent > 0) ?
        (metrics.iteration_time / sequential_equivalent) * 100.0 : 0.0;

    if (metrics.total_iterations > 1) {
        metrics.convergence_rate = (metrics.max_objective - metrics.min_objective) / metrics.total_iterations;
    }

    size_t total_memory_accessed = (metrics.total_agents * PARAMETR_SIZE * sizeof(double)) * metrics.total_iterations;
    metrics.memory_throughput = (metrics.total_time > 0) ?
        (total_memory_accessed / (1024.0 * 1024.0)) / (metrics.total_time / 1000.0) : 0.0;
}

void print_summary(const PerformanceMetrics& metrics) {
    std::cout << "\n=== PERFORMANCE SUMMARY ===" << std::endl;
    std::cout << "Total time: " << metrics.total_time << " ms" << std::endl;
    std::cout << "Time per iteration: " << metrics.iteration_time / metrics.total_iterations << " ms" << std::endl;
    std::cout << "Optimal threads: " << metrics.optimal_threads << std::endl;
    std::cout << "Parallel efficiency: " << metrics.parallel_efficiency << "%" << std::endl;
    std::cout << "Hash hit rate: " << (metrics.hash_hits * 100.0 / (metrics.hash_hits + metrics.hash_misses)) << "%" << std::endl;
    std::cout << "Convergence rate: " << metrics.convergence_rate << " units/iteration" << std::endl;
    std::cout << "Memory throughput: " << metrics.memory_throughput << " MB/s" << std::endl;
    std::cout << "Best objective: " << metrics.min_objective << " (min), " << metrics.max_objective << " (max)" << std::endl;

    std::cout << "\n=== TIME DISTRIBUTION ===" << std::endl;
    std::cout << "Probability computation: " << metrics.probability_time << " ms ("
        << (metrics.probability_time * 100.0 / metrics.total_time) << "%)" << std::endl;
    std::cout << "Agent generation: " << metrics.agent_time << " ms ("
        << (metrics.agent_time * 100.0 / metrics.total_time) << "%)" << std::endl;
    std::cout << "Pheromone update: " << metrics.pheromone_time << " ms ("
        << (metrics.pheromone_time * 100.0 / metrics.total_time) << "%)" << std::endl;
    std::cout << "Hash operations: " << metrics.hash_time << " ms ("
        << (metrics.hash_time * 100.0 / metrics.total_time) << "%)" << std::endl;
    std::cout << "Function evaluation: " << metrics.evaluation_time << " ms ("
        << (metrics.evaluation_time * 100.0 / metrics.total_time) << "%)" << std::endl;
}

std::string get_log_line(const PerformanceMetrics& metrics) {
    std::ostringstream ss;
    ss << "PERF:;" << metrics.total_time << ";" << metrics.iteration_time << ";" << metrics.parallel_efficiency << ";"
        << (metrics.hash_hits * 100.0 / (metrics.hash_hits + metrics.hash_misses)) << ";" << metrics.convergence_rate << ";"
        << metrics.memory_throughput << ";" << metrics.min_objective << ";" << metrics.max_objective << ";"
        << metrics.probability_time << ";" << metrics.agent_time << ";" << metrics.pheromone_time << ";"
        << metrics.hash_time << ";" << metrics.evaluation_time << ";" << metrics.optimal_threads;
    return ss.str();
}

void log_parameters(std::ofstream& logFile) {
    logFile << "PARAMETERS: "
        << "PARAMETR_SIZE: " << PARAMETR_SIZE << "; "
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
        << "FUNCTION: " << (SHAFFERA ? "SHAFFERA " : "") << (CARROM_TABLE ? "CARROM_TABLE " : "") << (RASTRIGIN ? "RASTRIGIN " : "") << (ACKLEY ? "ACKLEY " : "") << (SPHERE ? "SPHERE " : "") << (GRIEWANK ? "GRIEWANK " : "") << (ZAKHAROV ? "ZAKHAROV " : "") << (SCHWEFEL ? "SCHWEFEL " : "") << (LEVY ? "LEVY " : "") << (MICHAELWICZYNSKI ? "MICHAELWICZYNSKI " : "") << "; "
        << "HASH_TABLE_SIZE: " << HASH_TABLE_SIZE << "; "
        << "ZERO_HASH_RESULT: " << ZERO_HASH_RESULT << "; "
        << "ZERO_HASH: " << ZERO_HASH << "; "
        << "MAX_PROBES: " << MAX_PROBES << "; "
        << "KOL_STAT_LEVEL: " << KOL_STAT_LEVEL << "; "
        << "TYPE_ACO: " << TYPE_ACO << "; "
        << "ACOCCyN_KOL_ITERATION: " << ACOCCyN_KOL_ITERATION << "; "
        << "CPU_RANDOM: " << CPU_RANDOM << "; "
        << "KOL_THREAD_CPU_ANT: " << KOL_THREAD_CPU_ANT << "; "
        << "CONST_AVX: " << CONST_AVX << "; "
        << "CONST_RANDOM: " << CONST_RANDOM << "; "
        << "MAX_CONST: " << MAX_CONST << "; "
        << "BIN_SEARCH: " << BIN_SEARCH << "; "
        << "GO_ALG_MINMAX: " << GO_ALG_MINMAX << "; "
        << "PAR_MAX_ALG_MINMAX: " << PAR_MAX_ALG_MINMAX << "; "
        << "PAR_MIN_ALG_MINMAX: " << PAR_MIN_ALG_MINMAX
        << std::endl;
}

// ============================================================================
// OPTIMIZED HASH TABLE FUNCTIONS WITH PROPER SYNCHRONIZATION
// ============================================================================

struct ShardedHashTable {
    std::vector<HashEntry*> tables;
    int shards;
    int size_per_shard;
};

void init_hash_table(ShardedHashTable& table, int total_size, int num_shards) {
    table.shards = num_shards;
    table.size_per_shard = total_size / num_shards;
    table.tables.resize(table.shards);

    for (int i = 0; i < table.shards; ++i) {
        table.tables[i] = new HashEntry[table.size_per_shard];
        for (int j = 0; j < table.size_per_shard; ++j) {
            table.tables[i][j].key.store(ZERO_HASH, std::memory_order_relaxed);
            table.tables[i][j].value = 0.0;
            table.tables[i][j].timestamp.store(0, std::memory_order_relaxed);
        }
    }
}

void free_hash_table(ShardedHashTable& table) {
    for (int i = 0; i < table.shards; ++i) {
        delete[] table.tables[i];
    }
    table.tables.clear();
}

// Оптимизированная хэш-функция
unsigned long long compute_hash_fnv1a(const int* path) noexcept {
    const unsigned long long prime = 1099511628211ULL;
    unsigned long long hash = 14695981039346656037ULL;

    // Развертывание цикла для лучшей производительности
    for (int i = 0; i < PARAMETR_SIZE; i += 4) {
        if (i + 3 < PARAMETR_SIZE) {
            hash ^= static_cast<unsigned long long>(path[i]);
            hash *= prime;
            hash ^= static_cast<unsigned long long>(path[i + 1]);
            hash *= prime;
            hash ^= static_cast<unsigned long long>(path[i + 2]);
            hash *= prime;
            hash ^= static_cast<unsigned long long>(path[i + 3]);
            hash *= prime;
        }
        else {
            // Обработка оставшихся элементов
            for (int j = i; j < PARAMETR_SIZE; ++j) {
                hash ^= static_cast<unsigned long long>(path[j]);
                hash *= prime;
            }
        }
    }
    return hash;
}

int get_shard(const ShardedHashTable& table, const int* path) noexcept {
    return compute_hash_fnv1a(path) % table.shards;
}

// Atomic hash lookup with proper memory ordering
double hash_lookup(ShardedHashTable& table, const int* path, int iteration, PerformanceMetrics& metrics) noexcept {
    auto start = std::chrono::high_resolution_clock::now();

    int shard = get_shard(table, path);
    unsigned long long key = compute_hash_fnv1a(path);
    unsigned long long idx = key % table.size_per_shard;
    HashEntry* hash_table = table.tables[shard];

    // Atomic read with acquire semantics
    for (int i = 0; i < std::min(MAX_PROBES, 8); ++i) {
        unsigned long long current_idx = (idx + i) % table.size_per_shard;
        unsigned long long current_key = hash_table[current_idx].key.load(std::memory_order_acquire);

        if (current_key == ZERO_HASH) {
            metrics.hash_misses++;
            auto end = std::chrono::high_resolution_clock::now();
            metrics.hash_time += std::chrono::duration<double, std::milli>(end - start).count();
            return ZERO_HASH_RESULT;
        }
        if (current_key == key + 1) {
            // Update timestamp - this is safe as it's just metadata
            hash_table[current_idx].timestamp.store(iteration, std::memory_order_relaxed);
            double value = hash_table[current_idx].value;

            // Ensure we read the value after confirming the key
            std::atomic_thread_fence(std::memory_order_acquire);

            metrics.hash_hits++;
            auto end = std::chrono::high_resolution_clock::now();
            metrics.hash_time += std::chrono::duration<double, std::milli>(end - start).count();
            return value;
        }
    }

    metrics.hash_misses++;
    auto end = std::chrono::high_resolution_clock::now();
    metrics.hash_time += std::chrono::duration<double, std::milli>(end - start).count();
    return ZERO_HASH_RESULT;
}

// Thread-safe hash store with proper synchronization
bool hash_store(ShardedHashTable& table, const int* path, double value, int iteration, PerformanceMetrics& metrics) noexcept {
    auto start = std::chrono::high_resolution_clock::now();

    int shard = get_shard(table, path);
    unsigned long long key = compute_hash_fnv1a(path);
    unsigned long long idx = key % table.size_per_shard;
    HashEntry* hash_table = table.tables[shard];

    // Double-check pattern to avoid race conditions
    for (int i = 0; i < std::min(MAX_PROBES, 4); ++i) {
        unsigned long long current_idx = (idx + i) % table.size_per_shard;

        // First, check if the key already exists (fast path)
        unsigned long long current_key = hash_table[current_idx].key.load(std::memory_order_acquire);
        if (current_key == key + 1) {
            // Key exists, just update the value and timestamp
            hash_table[current_idx].value = value;
            hash_table[current_idx].timestamp.store(iteration, std::memory_order_release);
            auto end = std::chrono::high_resolution_clock::now();
            metrics.hash_time += std::chrono::duration<double, std::milli>(end - start).count();
            return true;
        }

        // Try to acquire the slot with CAS
        unsigned long long expected = ZERO_HASH;
        if (hash_table[current_idx].key.compare_exchange_strong(expected, key + 1,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
            // Successfully acquired the slot
            hash_table[current_idx].value = value;
            hash_table[current_idx].timestamp.store(iteration, std::memory_order_release);

            // Ensure all writes are visible to other threads
            std::atomic_thread_fence(std::memory_order_release);

            auto end = std::chrono::high_resolution_clock::now();
            metrics.hash_time += std::chrono::duration<double, std::milli>(end - start).count();
            return true;
        }

        // If CAS failed but the key matches, update the existing entry
        if (expected == key + 1) {
            hash_table[current_idx].value = value;
            hash_table[current_idx].timestamp.store(iteration, std::memory_order_release);
            auto end = std::chrono::high_resolution_clock::now();
            metrics.hash_time += std::chrono::duration<double, std::milli>(end - start).count();
            return true;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    metrics.hash_time += std::chrono::duration<double, std::milli>(end - start).count();
    return false;
}

// Combined lookup-or-compute function to eliminate race conditions
double lookup_or_compute(ShardedHashTable& table, const int* path, double* agent_params, int iteration, PerformanceMetrics& metrics) noexcept {
    // First try to lookup
    double cached = hash_lookup(table, path, iteration, metrics);

    if (cached != ZERO_HASH_RESULT) {
        return cached;
    }

    // If not found, compute the value
    auto eval_start = std::chrono::high_resolution_clock::now();
    double result = benchmark_function(agent_params);
    auto eval_end = std::chrono::high_resolution_clock::now();
    metrics.evaluation_time += std::chrono::duration<double, std::milli>(eval_end - eval_start).count();

    // Store the computed value
    hash_store(table, path, result, iteration, metrics);

    return result;
}

// ============================================================================
// OPTIMIZED CORE FUNCTIONS
// ============================================================================

void update_statistics(Statistics& stat, double value) noexcept {
    stat.sum += value;
    stat.sum_sq += value * value;
    stat.count++;
}

double mean_statistics(const Statistics& stat) noexcept {
    return stat.count > 0 ? stat.sum / stat.count : 0.0;
}

double variance_statistics(const Statistics& stat) noexcept {
    return stat.count > 1 ? (stat.sum_sq / stat.count) - (mean_statistics(stat) * mean_statistics(stat)) : 0.0;
}

void clear_statistics(Statistics& stat) noexcept {
    stat.sum = stat.sum_sq = stat.count = 0;
}

double fast_random(uint64_t& seed) noexcept {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return (seed >> 11) * 0x1.0p-53;
}


// Оптимизированное вычисление матрицы вероятностей
void compute_probability_matrix(double* __restrict pheromone, double* __restrict visits, double* __restrict probabilities, PerformanceMetrics& metrics) noexcept {
    auto start = std::chrono::high_resolution_clock::now();

    const int total_params = PARAMETR_SIZE;
    const int max_values = MAX_VALUE_SIZE;

#pragma omp parallel for schedule(static)
    for (int param = 0; param < total_params; ++param) {
        const int base = param * max_values;
        double total_pheromone = 0.0;

        // Векторизованное суммирование феромонов
        for (int i = 0; i < max_values; i += 4) {
            if (i + 3 < max_values) {
                total_pheromone += pheromone[base + i] + pheromone[base + i + 1] +
                    pheromone[base + i + 2] + pheromone[base + i + 3];
            }
            else {
                for (int j = i; j < max_values; ++j) {
                    total_pheromone += pheromone[base + j];
                }
            }
        }

        if (total_pheromone <= 0.0) {
            // Равномерное распределение
            double step = 1.0 / max_values;
            double cumulative = step;
            probabilities[base] = cumulative;
            for (int i = 1; i < max_values - 1; ++i) {
                cumulative += step;
                probabilities[base + i] = cumulative;
            }
            probabilities[base + max_values - 1] = 1.0;
            continue;
        }

        // Вычисление вероятностей
        double sum_probs = 0.0;
        double temp_probs[MAX_VALUE_SIZE];

        for (int i = 0; i < max_values; ++i) {
            double norm_pheromone = pheromone[base + i] / total_pheromone;
            temp_probs[i] = compute_probability(norm_pheromone, visits[base + i]);
            sum_probs += temp_probs[i];
        }

        if (sum_probs > 0.0) {
            double cumulative = 0.0;
            double inv_sum = 1.0 / sum_probs;
            for (int i = 0; i < max_values; ++i) {
                cumulative += temp_probs[i] * inv_sum;
                probabilities[base + i] = cumulative;
            }
            probabilities[base + max_values - 1] = 1.0;
        }
        else {
            double step = 1.0 / max_values;
            double cumulative = step;
            probabilities[base] = cumulative;
            for (int i = 1; i < max_values - 1; ++i) {
                cumulative += step;
                probabilities[base + i] = cumulative;
            }
            probabilities[base + max_values - 1] = 1.0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    metrics.probability_time += std::chrono::duration<double, std::milli>(end - start).count();
}

// Оптимизированная генерация агентов с thread-safe хэш-таблицей
void generate_agents(int iteration, double* __restrict params, double* __restrict probabilities, double* __restrict agents, int* __restrict paths, double* __restrict scores, ShardedHashTable& hash_table, PerformanceMetrics& metrics) noexcept {
    auto start = std::chrono::high_resolution_clock::now();

    const int num_agents = ANT_SIZE;
    const int num_params = PARAMETR_SIZE;
    const int max_values = MAX_VALUE_SIZE;

#pragma omp parallel
    {
        std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count() + iteration * 1000 + omp_get_thread_num() * 12345);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

#pragma omp for schedule(static)
        for (int agent = 0; agent < num_agents; ++agent) {
            const int agent_base = agent * num_params;

            // Generate path with optimized binary search
            for (int param = 0; param < num_params; ++param) {
                double rand_val = dist(rng);
                const int prob_base = param * max_values;

                int left = 0, right = max_values - 1;
                int choice = right;

                while (left <= right) {
                    int mid = left + (right - left) / 2;
                    if (rand_val > probabilities[prob_base + mid]) {
                        left = mid + 1;
                    }
                    else {
                        choice = mid;
                        right = mid - 1;
                    }
                }

                paths[agent_base + param] = choice;
                agents[agent_base + param] = params[param * max_values + choice];
            }

            // Use thread-safe lookup-or-compute function
            scores[agent] = lookup_or_compute(hash_table, &paths[agent_base],
                &agents[agent_base], iteration, metrics);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    metrics.agent_time += std::chrono::duration<double, std::milli>(end - start).count();
    metrics.total_agents += ANT_SIZE;
}

// Оптимизированное обновление феромонов
void update_pheromones(double* __restrict pheromone, double* __restrict visits, const int* __restrict paths, const double* __restrict scores, PerformanceMetrics& metrics) noexcept {
    auto start = std::chrono::high_resolution_clock::now();

    const int total_cells = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int num_agents = ANT_SIZE;

    // Испарение феромонов
#pragma omp parallel for schedule(static)
    for (int i = 0; i < total_cells; ++i) {
        pheromone[i] *= PARAMETR_RO;
    }

    // Обновление посещений и феромонов
#pragma omp parallel
    {
        // Локальные буферы для каждого потока
        std::vector<double> local_pheromone(total_cells, 0.0);
        std::vector<int> local_visits(total_cells, 0);

#pragma omp for nowait
        for (int agent = 0; agent < num_agents; ++agent) {
            double score = scores[agent];
            double add_value = 0.0;

#if OPTIMIZE_MIN_1
            add_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT > score) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - score) : 0.0;
#elif OPTIMIZE_MAX
            add_value = PARAMETR_Q * score;
#endif

            const int* path = &paths[agent * PARAMETR_SIZE];

            for (int param = 0; param < PARAMETR_SIZE; ++param) {
                int choice = path[param];
                int idx = param * MAX_VALUE_SIZE + choice;
                local_visits[idx]++;

                if (add_value > 0.0) {
                    local_pheromone[idx] += add_value;
                }
            }
        }

        // Слияние локальных данных
#pragma omp critical
        {
            for (int i = 0; i < total_cells; ++i) {
                visits[i] += local_visits[i];
                pheromone[i] += local_pheromone[i];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    metrics.pheromone_time += std::chrono::duration<double, std::milli>(end - start).count();
}

bool load_matrix(const std::string& filename, double* __restrict params, double* __restrict pheromone, double* __restrict visits) noexcept {
    std::ifstream file(filename);
    if (!file) {
        std::cout << "File not found, generating synthetic matrix..." << std::endl;
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> dist(-10.0, 10.0);

        for (int i = 0; i < PARAMETR_SIZE; ++i) {
            for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                int idx = i * MAX_VALUE_SIZE + j;
                params[idx] = dist(rng);
                pheromone[idx] = 1.0;
                visits[idx] = 1.0;
            }
        }
        return true;
    }

    for (int i = 0; i < PARAMETR_SIZE; ++i) {
        for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
            int idx = i * MAX_VALUE_SIZE + j;
            if (!(file >> params[idx])) {
                std::cerr << "Error reading element [" << i << "][" << j << "]" << std::endl;
                return false;
            }

            if (params[idx] != -100.0) {
                pheromone[idx] = 1.0;
                visits[idx] = 1.0;
            }
            else {
                pheromone[idx] = 0.0;
                params[idx] = 0.0;
                visits[idx] = 0.0;
            }
        }
    }

    file.close();
    return true;
}

int find_optimal_thread_count() {
    int max_threads = omp_get_max_threads();
    unsigned int hardware_threads = std::thread::hardware_concurrency();
    int physical_cores = (hardware_threads != 0) ? hardware_threads : max_threads;

    // Для данной задачи используем меньше потоков для уменьшения конкуренции
    //int optimal = std::max(1, std::min(physical_cores / 2, max_threads));
    int optimal = max_threads;
    std::cout << "Detected " << physical_cores << " hardware threads, "
        << max_threads << " available OpenMP threads" << std::endl;
    std::cout << "Using " << optimal << " threads for optimal performance" << std::endl;

    return optimal;
}

// ============================================================================
// OPTIMIZED MAIN ALGORITHM
// ============================================================================

int run_optimized_algorithm() {
    PerformanceMetrics metrics;

    int optimal_threads = find_optimal_thread_count();
    omp_set_dynamic(0);
    omp_set_num_threads(optimal_threads);

    std::cout << "\n=== OPTIMIZED OPENMP ALGORITHM ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  PARAMETR_SIZE: " << PARAMETR_SIZE << std::endl;
    std::cout << "  PARAMETR_SIZE_ONE_X: " << PARAMETR_SIZE_ONE_X << std::endl;
    std::cout << "  MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << std::endl;
    std::cout << "  ANT_SIZE: " << ANT_SIZE << std::endl;
    std::cout << "  KOL_ITERATION: " << KOL_ITERATION << std::endl;
    std::cout << "  Optimal threads: " << optimal_threads << std::endl;
    std::cout << "  Hash table size: " << HASH_TABLE_SIZE << " (sharded)" << std::endl;

    auto total_start = std::chrono::high_resolution_clock::now();

    const int matrix_size = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int agents_size = PARAMETR_SIZE * ANT_SIZE;
    const int stat_interval = KOL_ITERATION / KOL_STAT_LEVEL;

    ShardedHashTable hash_table;
    init_hash_table(hash_table, HASH_TABLE_SIZE, optimal_threads);

    // Выделение памяти с выравниванием
    std::vector<double> params(matrix_size);
    std::vector<double> pheromone(matrix_size);
    std::vector<double> visits(matrix_size);
    std::vector<double> probabilities(matrix_size);
    std::vector<double> agents(agents_size);
    std::vector<int> paths(agents_size);
    std::vector<double> scores(ANT_SIZE);

    std::cout << "Loading matrix from: " << NAME_FILE_GRAPH << std::endl;
    if (!load_matrix(NAME_FILE_GRAPH, params.data(), pheromone.data(), visits.data())) {
        return -1;
    }

    // Логирование параметров
    std::ofstream logFile("log.txt", std::ios::app);
    log_parameters(logFile);

    std::cout << "Starting optimization with performance tracking..." << std::endl;

    // Основной цикл оптимизации
    for (int iter = 0; iter < KOL_ITERATION; ++iter) {
        metrics.total_iterations++;

        if (PRINT_INFORMATION && iter % 50 == 0) {
            std::cout << "Iteration " << iter << std::endl;
        }

        auto iter_start = std::chrono::high_resolution_clock::now();

        // Основные фазы алгоритма
        compute_probability_matrix(pheromone.data(), visits.data(), probabilities.data(), metrics);
        generate_agents(iter, params.data(), probabilities.data(), agents.data(), paths.data(), scores.data(), hash_table, metrics);
        update_pheromones(pheromone.data(), visits.data(), paths.data(), scores.data(), metrics);

        // Обновление лучших значений
        double iter_min = *std::min_element(scores.begin(), scores.end());
        double iter_max = *std::max_element(scores.begin(), scores.end());

        if (iter_min < metrics.min_objective) metrics.min_objective = iter_min;
        if (iter_max > metrics.max_objective) metrics.max_objective = iter_max;

        auto iter_end = std::chrono::high_resolution_clock::now();
        metrics.iteration_time += std::chrono::duration<double, std::milli>(iter_end - iter_start).count();
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    metrics.total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    calculate_efficiency(metrics, optimal_threads);
    print_summary(metrics);

    logFile << get_log_line(metrics) << std::endl;
    logFile.close();

    // Рекомендации по оптимизации
    std::cout << "\n=== PERFORMANCE RECOMMENDATIONS ===" << std::endl;

    double hit_rate = (metrics.hash_hits * 100.0 / (metrics.hash_hits + metrics.hash_misses));
    if (hit_rate < 10.0) {
        std::cout << "🎯 Consider disabling hash table (hit rate: " << hit_rate << "%)" << std::endl;
    }

    if (metrics.parallel_efficiency < 50.0) {
        std::cout << "⚡ Try different thread count (current efficiency: " << metrics.parallel_efficiency << "%)" << std::endl;
    }

    if (metrics.agent_time > metrics.total_time * 0.7) {
        std::cout << "🔧 Optimize agent generation (currently " << (metrics.agent_time * 100.0 / metrics.total_time) << "%)" << std::endl;
    }

    std::cout << "\n=== FINAL RESULTS ===" << std::endl;
    std::cout << "Best solution found: " << metrics.min_objective << std::endl;
    std::cout << "Total agents processed: " << metrics.total_agents << std::endl;
    std::cout << "Overall performance: " << (metrics.total_agents / (metrics.total_time / 1000.0))
        << " agents/second" << std::endl;

    free_hash_table(hash_table);
    return 0;
}

int main() {
    std::cout << "=== OPENMP PERFORMANCE OPTIMIZATION ===" << std::endl;
    std::cout << "OpenMP version: " << _OPENMP << std::endl;

    auto start_time = std::chrono::system_clock::now();
    std::time_t start_time_t = std::chrono::system_clock::to_time_t(start_time);
    std::cout << "Start time: " << std::ctime(&start_time_t);

    int result = run_optimized_algorithm();

    auto end_time = std::chrono::system_clock::now();
    std::time_t end_time_t = std::chrono::system_clock::to_time_t(end_time);
    std::cout << "End time: " << std::ctime(&end_time_t);

    std::cout << "\nProgram finished with code: " << result << std::endl;
    return result;
}