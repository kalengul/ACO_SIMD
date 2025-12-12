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
#include <numeric>

// Загрузка констант из файла параметров
#include "parametrs.h"

std::ofstream outfile("statistics.txt"); // Глобальная переменная для файла статистики
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
    double speedup = 0.0;
    double scalability = 0.0;
};

struct ThreadExperimentResult {
    int thread_count;
    PerformanceMetrics metrics, square_metrics;
    double average_time;
    double std_dev_time;
    double best_objective;
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

    for (int i = 0; i < num_vars; ++i) {
        double x = compute_parameter(params, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum_sq += x * x;
    }
    return sum_sq;
}
#endif

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
unsigned long long compute_hash_fnv1a(const int* path) noexcept {
    const unsigned long long prime = 1099511628211ULL;
    unsigned long long hash = 14695981039346656037ULL;

    for (int i = 0; i < PARAMETR_SIZE; ++i) {
        hash ^= static_cast<unsigned long long>(path[i]);
        hash *= prime;
    }
    return hash;
}
int get_shard(const ShardedHashTable& table, const int* path) noexcept {
    return compute_hash_fnv1a(path) % table.shards;
}
double hash_lookup(ShardedHashTable& table, const int* path, int iteration, PerformanceMetrics& metrics) noexcept {
    auto start = std::chrono::high_resolution_clock::now();

    int shard = get_shard(table, path);
    unsigned long long key = compute_hash_fnv1a(path);
    unsigned long long idx = key % table.size_per_shard;
    HashEntry* hash_table = table.tables[shard];

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
            hash_table[current_idx].timestamp.store(iteration, std::memory_order_relaxed);
            double value = hash_table[current_idx].value;

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
bool hash_store(ShardedHashTable& table, const int* path, double value, int iteration, PerformanceMetrics& metrics) noexcept {
    auto start = std::chrono::high_resolution_clock::now();

    int shard = get_shard(table, path);
    unsigned long long key = compute_hash_fnv1a(path);
    unsigned long long idx = key % table.size_per_shard;
    HashEntry* hash_table = table.tables[shard];

    for (int i = 0; i < std::min(MAX_PROBES, 4); ++i) {
        unsigned long long current_idx = (idx + i) % table.size_per_shard;
        unsigned long long current_key = hash_table[current_idx].key.load(std::memory_order_acquire);
        
        if (current_key == key + 1) {
            hash_table[current_idx].value = value;
            hash_table[current_idx].timestamp.store(iteration, std::memory_order_release);
            auto end = std::chrono::high_resolution_clock::now();
            metrics.hash_time += std::chrono::duration<double, std::milli>(end - start).count();
            return true;
        }

        unsigned long long expected = ZERO_HASH;
        if (hash_table[current_idx].key.compare_exchange_strong(expected, key + 1,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
            hash_table[current_idx].value = value;
            hash_table[current_idx].timestamp.store(iteration, std::memory_order_release);
            std::atomic_thread_fence(std::memory_order_release);

            auto end = std::chrono::high_resolution_clock::now();
            metrics.hash_time += std::chrono::duration<double, std::milli>(end - start).count();
            return true;
        }

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
double lookup_or_compute(ShardedHashTable& table, const int* path, double* agent_params, int iteration, PerformanceMetrics& metrics) noexcept {
    double cached = hash_lookup(table, path, iteration, metrics);

    if (cached != ZERO_HASH_RESULT) {
        return cached;
    }

    auto eval_start = std::chrono::high_resolution_clock::now();
    double result = benchmark_function(agent_params);
    auto eval_end = std::chrono::high_resolution_clock::now();
    metrics.evaluation_time += std::chrono::duration<double, std::milli>(eval_end - eval_start).count();

    hash_store(table, path, result, iteration, metrics);

    return result;
}


void compute_probability_matrix(double* __restrict pheromone, double* __restrict visits, double* __restrict probabilities, PerformanceMetrics& metrics) noexcept {
    auto start = std::chrono::high_resolution_clock::now();

    const int total_params = PARAMETR_SIZE;
    const int max_values = MAX_VALUE_SIZE;

#pragma omp parallel for schedule(static)
    for (int param = 0; param < total_params; ++param) {
        const int base = param * max_values;
        double total_pheromone = 0.0;

        for (int i = 0; i < max_values; i += 4) {
            if (i + 3 < max_values) {
                total_pheromone += pheromone[base + i] + pheromone[base + i + 1] +
                    pheromone[base + i + 2] + pheromone[base + i + 3];
            } else {
                for (int j = i; j < max_values; ++j) {
                    total_pheromone += pheromone[base + j];
                }
            }
        }

        if (total_pheromone <= 0.0) {
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
        } else {
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

            for (int param = 0; param < num_params; ++param) {
                double rand_val = dist(rng);
                const int prob_base = param * max_values;

                int left = 0, right = max_values - 1;
                int choice = right;

                while (left <= right) {
                    int mid = left + (right - left) / 2;
                    if (rand_val > probabilities[prob_base + mid]) {
                        left = mid + 1;
                    } else {
                        choice = mid;
                        right = mid - 1;
                    }
                }

                paths[agent_base + param] = choice;
                agents[agent_base + param] = params[param * max_values + choice];
            }

            scores[agent] = lookup_or_compute(hash_table, &paths[agent_base], &agents[agent_base], iteration, metrics);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    metrics.agent_time += std::chrono::duration<double, std::milli>(end - start).count();
    metrics.total_agents += ANT_SIZE;
}
void update_pheromones(double* __restrict pheromone, double* __restrict visits, const int* __restrict paths, const double* __restrict scores, PerformanceMetrics& metrics) noexcept {
    auto start = std::chrono::high_resolution_clock::now();

    const int total_cells = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int num_agents = ANT_SIZE;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < total_cells; ++i) {
        pheromone[i] *= PARAMETR_RO;
    }

#pragma omp parallel
    {
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
            } else {
                pheromone[idx] = 0.0;
                params[idx] = 0.0;
                visits[idx] = 0.0;
            }
        }
    }

    file.close();
    return true;
}

//Оценка эффективности
void calculate_efficiency(PerformanceMetrics& metrics, int num_threads, double single_thread_time = 0.0) {
    metrics.optimal_threads = num_threads;

    if (single_thread_time > 0) {
        metrics.speedup = single_thread_time / metrics.total_time;
        metrics.scalability = (metrics.speedup / num_threads) * 100.0;
    }

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
    std::cout << "Speedup: " << metrics.speedup << "x" << std::endl;
    std::cout << "Scalability: " << metrics.scalability << "%" << std::endl;
    
    if (metrics.hash_hits + metrics.hash_misses > 0) {
        std::cout << "Hash hit rate: " << (metrics.hash_hits * 100.0 / (metrics.hash_hits + metrics.hash_misses)) << "%" << std::endl;
    }
    
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
void print_comparison_table(const std::vector<ThreadExperimentResult>& results) {
    if (results.empty()) return;
    std::cout << "Threads ; iteration_time (ms) ; probability_time ; agent_time ; pheromone_time ; hash_time ; evaluation_time ; speedup ; parallel_efficiency ; hit_rate ; min_objective ;" << std::endl;
    for (const auto& result : results) {
        double hit_rate = (result.metrics.hash_hits * 100.0 / (result.metrics.hash_hits + result.metrics.hash_misses));
        std::cout << result.thread_count << "; " << result.metrics.iteration_time << "; " << result.metrics.probability_time << "; " << result.metrics.agent_time << "; " << result.metrics.pheromone_time << "; " << result.metrics.hash_time << "; " << result.metrics.evaluation_time << "; " << result.metrics.speedup << "; " << result.metrics.parallel_efficiency << "; " << hit_rate << "; " << result.metrics.min_objective << "; ";
        outfile << result.thread_count << "; " << result.metrics.iteration_time << "; " << result.metrics.probability_time << "; " << result.metrics.agent_time << "; " << result.metrics.pheromone_time << "; " << result.metrics.hash_time << "; " << result.metrics.evaluation_time << "; " << result.metrics.speedup << "; " << result.metrics.parallel_efficiency << "; " << hit_rate << "; " << result.metrics.min_objective << ";  ;";
        outfile << result.square_metrics.iteration_time << "; " << result.square_metrics.probability_time << "; " << result.square_metrics.agent_time << "; " << result.square_metrics.pheromone_time << "; " << result.square_metrics.hash_time << "; " << result.square_metrics.evaluation_time << "; " << std::endl;
    }
}
void print_optimal_configuration(const std::vector<ThreadExperimentResult>& results) {
    if (results.empty()) return;

    auto best_result = std::min_element(results.begin(), results.end(),
        [](const ThreadExperimentResult& a, const ThreadExperimentResult& b) {
            return a.metrics.total_time < b.metrics.total_time;
        });

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "OPTIMAL CONFIGURATION" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Recommended thread count: " << best_result->thread_count << std::endl;
    std::cout << "Execution time: " << best_result->metrics.total_time << " ms" << std::endl;
    std::cout << "Speedup: " << best_result->metrics.speedup << "x" << std::endl;
    std::cout << "Parallel efficiency: " << best_result->metrics.parallel_efficiency << "%" << std::endl;
    std::cout << "Best objective value: " << best_result->metrics.min_objective << std::endl;
    
    outfile << "\n=== OPTIMAL CONFIGURATION ===" << std::endl;
    outfile << "Threads: " << best_result->thread_count << std::endl;
    outfile << "Time: " << best_result->metrics.total_time << " ms" << std::endl;
    outfile << "Speedup: " << best_result->metrics.speedup << "x" << std::endl;
    outfile << "Efficiency: " << best_result->metrics.parallel_efficiency << "%" << std::endl;
    outfile << "Best Objective: " << best_result->metrics.min_objective << std::endl;
}

PerformanceMetrics run_single_experiment(int num_threads, int iteration_count = KOL_ITERATION) {
    PerformanceMetrics metrics;

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    const int matrix_size = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int agents_size = PARAMETR_SIZE * ANT_SIZE;

    ShardedHashTable hash_table;
    init_hash_table(hash_table, HASH_TABLE_SIZE, std::min(num_threads, 8));

    std::vector<double> params(matrix_size);
    std::vector<double> pheromone(matrix_size);
    std::vector<double> visits(matrix_size);
    std::vector<double> probabilities(matrix_size);
    std::vector<double> agents(agents_size);
    std::vector<int> paths(agents_size);
    std::vector<double> scores(ANT_SIZE);

    if (!load_matrix(NAME_FILE_GRAPH, params.data(), pheromone.data(), visits.data())) {
        metrics.total_time = -1;
        return metrics;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iteration_count; ++iter) {
        metrics.total_iterations++;

        auto iter_start = std::chrono::high_resolution_clock::now();

        compute_probability_matrix(pheromone.data(), visits.data(), probabilities.data(), metrics);
        generate_agents(iter, params.data(), probabilities.data(), agents.data(), paths.data(), scores.data(), hash_table, metrics);
        update_pheromones(pheromone.data(), visits.data(), paths.data(), scores.data(), metrics);

        double iter_min = *std::min_element(scores.begin(), scores.end());
        double iter_max = *std::max_element(scores.begin(), scores.end());

        //std::cout << "     iter_min: " << iter_min << "     iter_max: " << iter_max << std::endl;

        if (iter_min < metrics.min_objective) metrics.min_objective = iter_min;
        if (iter_max > metrics.max_objective) metrics.max_objective = iter_max;

        auto iter_end = std::chrono::high_resolution_clock::now();
        metrics.iteration_time += std::chrono::duration<double, std::milli>(iter_end - iter_start).count();
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    metrics.total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    free_hash_table(hash_table);
    return metrics;
}
std::vector<ThreadExperimentResult> run_thread_scaling_experiment() {
    std::vector<ThreadExperimentResult> results;

    int max_threads = omp_get_max_threads();
    unsigned int hardware_threads = std::thread::hardware_concurrency();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "THREAD SCALING EXPERIMENT" << std::endl;
    std::cout << "Hardware threads: " << hardware_threads << std::endl;
    std::cout << "Max OpenMP threads: " << max_threads << std::endl;
    std::cout << "Runs per configuration: " << KOL_PROGON_STATISTICS << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::vector<int> thread_counts;
    if (max_threads <= 32) {
        for (int i = 1; i <= max_threads; ++i) {
            thread_counts.push_back(i);
        }
    } else {
        thread_counts = {1, 2, 4, 6, 8, 12, 16};
        // Добавляем физические ядра если они в разумном диапазоне
        if (hardware_threads > 0 && hardware_threads <= 32) {
            if (std::find(thread_counts.begin(), thread_counts.end(), hardware_threads) == thread_counts.end()) {
                thread_counts.push_back(hardware_threads);
            }
        }
        // Добавляем максимальное количество потоков если оно разумное
        if (max_threads <= 32 && std::find(thread_counts.begin(), thread_counts.end(), max_threads) == thread_counts.end()) {
            thread_counts.push_back(max_threads);
        }
    }

    std::sort(thread_counts.begin(), thread_counts.end());

    std::cout << "Testing thread counts: ";
    for (int threads : thread_counts) {
        std::cout << threads << " ";
    }
    std::cout << std::endl << std::endl;

    double single_thread_time = 0.0;

    for (int thread_count : thread_counts) {
        std::cout << "Testing " << thread_count << " threads..." << std::endl;
        
        std::vector<double> execution_times;
        PerformanceMetrics average_metrics;
        PerformanceMetrics square_metrics;
        double best_objective = std::numeric_limits<double>::max();
        int successful_runs = 0;

        for (int run = 0; run < KOL_PROGON_STATISTICS; ++run) {
            std::cout << "  Run " << (run + 1) << "/" << KOL_PROGON_STATISTICS << "...";

            auto metrics = run_single_experiment(thread_count);

            if (metrics.total_time > 0) {
                execution_times.push_back(metrics.total_time);
                
                // Суммируем метрики для усреднения
                average_metrics.total_time += metrics.total_time;
                average_metrics.iteration_time += metrics.iteration_time;
                average_metrics.probability_time += metrics.probability_time;
                average_metrics.agent_time += metrics.agent_time;
                average_metrics.pheromone_time += metrics.pheromone_time;
                average_metrics.hash_time += metrics.hash_time;
                average_metrics.evaluation_time += metrics.evaluation_time;
                average_metrics.hash_hits += metrics.hash_hits;
                average_metrics.hash_misses += metrics.hash_misses;
                average_metrics.total_agents += metrics.total_agents;
                average_metrics.total_iterations += metrics.total_iterations;

                square_metrics.total_time += metrics.total_time * metrics.total_time;
                square_metrics.iteration_time += metrics.iteration_time * metrics.iteration_time;
                square_metrics.probability_time += metrics.probability_time * metrics.probability_time;
                square_metrics.agent_time += metrics.agent_time * metrics.agent_time;
                square_metrics.pheromone_time += metrics.pheromone_time * metrics.pheromone_time;
                square_metrics.hash_time += metrics.hash_time * metrics.hash_time;
                square_metrics.evaluation_time += metrics.evaluation_time * metrics.evaluation_time;
                square_metrics.hash_hits += metrics.hash_hits;
                square_metrics.hash_misses += metrics.hash_misses;

                if (metrics.min_objective < best_objective) {
                    best_objective = metrics.min_objective;
                }
                if (metrics.min_objective < average_metrics.min_objective) {
                    average_metrics.min_objective = metrics.min_objective;
                }
                if (metrics.max_objective > average_metrics.max_objective) {
                    average_metrics.max_objective = metrics.max_objective;
                }

                successful_runs++;
                std::cout << " OK (" << metrics.total_time << " ms) min=" << metrics.min_objective << std::endl;
            } else {
                std::cout << " FAILED" << std::endl;
            }
        }

        if (successful_runs > 0) {
            // Усредняем метрики
            double scale = 1.0 / successful_runs;
            average_metrics.total_time *= scale;
            average_metrics.iteration_time *= scale;
            average_metrics.probability_time *= scale;
            average_metrics.agent_time *= scale;
            average_metrics.pheromone_time *= scale;
            average_metrics.hash_time *= scale;
            average_metrics.evaluation_time *= scale;
            average_metrics.hash_hits = static_cast<int>(average_metrics.hash_hits * scale);
            average_metrics.hash_misses = static_cast<int>(average_metrics.hash_misses * scale);
            average_metrics.total_agents = static_cast<int>(average_metrics.total_agents * scale);
            average_metrics.total_iterations = static_cast<int>(average_metrics.total_iterations * scale);

            calculate_efficiency(average_metrics, thread_count, single_thread_time);

            ThreadExperimentResult result;
            result.thread_count = thread_count;
            result.metrics = average_metrics;
            result.square_metrics = square_metrics;
            result.best_objective = best_objective;
            
            results.push_back(result);

            std::cout << "  ✓ Completed " << successful_runs << "/" << KOL_PROGON_STATISTICS
                << " runs, avg: " << average_metrics.total_time << " ms)" << std::endl;
        }
    }
    return results;
}

void log_parameters() {
    outfile << "OpenMP version: " << _OPENMP << "; "
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

int main() {
    std::cout << "=== OPENMP PERFORMANCE OPTIMIZATION WITH THREAD SCALING ===" << std::endl;
    std::cout << "OpenMP version: " << _OPENMP << std::endl;



    // Логируем параметры
    log_parameters();
    auto start_time = std::chrono::system_clock::now();
    std::time_t start_time_t = std::chrono::system_clock::to_time_t(start_time);
    std::cout << "Start time: " << std::ctime(&start_time_t);

    try {
        auto results = run_thread_scaling_experiment();
        
        if (!results.empty()) {
            print_comparison_table(results);
            print_optimal_configuration(results);
            
            std::cout << "\n=== EXPERIMENT COMPLETED SUCCESSFULLY ===" << std::endl;
            std::cout << "Results saved to statistics.txt" << std::endl;
            std::cout << "Parameters logged to log.txt" << std::endl;
        } else {
            std::cerr << "No successful runs completed!" << std::endl;
            return -1;
        }
    }
    catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        std::cerr << "Try reducing KOL_PROGON_STATISTICS, ANT_SIZE, or PARAMETR_SIZE" << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    auto end_time = std::chrono::system_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
    std::cout << "Total experiment duration: " << total_duration.count() << " minutes" << std::endl;

    outfile.close();
    return 0;
}