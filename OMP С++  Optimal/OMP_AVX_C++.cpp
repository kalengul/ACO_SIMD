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
 // 42, 84, 168, 336, 672, 1344, 2688, 5376, 10752, 21504, 43008, 86016, 172032, 344064, 688128, 1376256
#define PARAMETR_SIZE 21504   // Количество параметров 21*x (6*х)
#define MAX_VALUE_SIZE 4    // Максимальное количество значений у параметров 5 (100)
#define NAME_FILE_GRAPH "Parametr_Graph/test21504_4.txt"

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

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

std::vector<Statistics> stat_duration(KOL_STAT_LEVEL);
std::vector<Statistics> stat_SumgpuTime1(KOL_STAT_LEVEL);
std::vector<Statistics> stat_global_minOf(KOL_STAT_LEVEL);
std::vector<Statistics> stat_global_maxOf(KOL_STAT_LEVEL);
std::vector<Statistics> stat_kol_hash_fail(KOL_STAT_LEVEL);

// ============================================================================
// CORE COMPUTATION FUNCTIONS
// ============================================================================

// AVX-оптимизированная функция вычисления параметра
double compute_parameter(double* params, int start, int count) noexcept {
    // Для MAX_VALUE_SIZE=4 можем использовать AVX
    if (count == 4) {
        __m256d vec = _mm256_loadu_pd(params + start);
        __m256d sum_vec = _mm256_hadd_pd(vec, vec);
        double sum = ((double*)&sum_vec)[0] + ((double*)&sum_vec)[2];
        return params[start] * sum;
    }

    double sum = 0.0;
    // Стандартная реализация для других случаев
    for (int i = 1; i < count; ++i) {
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
    const int num_vars = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;

    // AVX-оптимизация для случая, когда SET_PARAMETR_SIZE_ONE_X=4
    if (SET_PARAMETR_SIZE_ONE_X == 4) {
        __m256d sum_sq_vec = _mm256_setzero_pd();

        for (int i = 0; i < num_vars; ++i) {
            double x = compute_parameter(params, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
            __m256d x_vec = _mm256_set1_pd(x);
            sum_sq_vec = _mm256_add_pd(sum_sq_vec, _mm256_mul_pd(x_vec, x_vec));
        }

        // Горизонтальное суммирование
        sum_sq_vec = _mm256_hadd_pd(sum_sq_vec, sum_sq_vec);
        sum_sq = ((double*)&sum_sq_vec)[0] + ((double*)&sum_sq_vec)[2];
    }
    else {
        // Стандартная реализация для других случаев
        for (int i = 0; i < num_vars; ++i) {
            double x = compute_parameter(params, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
            sum_sq += x * x;
        }
    }

    double r = std::sqrt(sum_sq);
    double sin_r = std::sin(r);
    return 0.5 - (sin_r * sin_r - 0.5) / (1.0 + 0.001 * sum_sq);
}
#endif
#if (RASTRIGIN)
double benchmark_function(double* params) noexcept {
    double sum = 0.0;
    const int num_vars = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
    constexpr double two_pi = 2.0 * M_PI;

    // AVX-оптимизация для RASTRIGIN
    if (SET_PARAMETR_SIZE_ONE_X == 4) {
        __m256d sum_vec = _mm256_setzero_pd();
        __m256d ten_vec = _mm256_set1_pd(10.0);
        __m256d two_pi_vec = _mm256_set1_pd(two_pi);

        for (int i = 0; i < num_vars; ++i) {
            double x = compute_parameter(params, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
            __m256d x_vec = _mm256_set1_pd(x);
            __m256d x_sq = _mm256_mul_pd(x_vec, x_vec);

            // 10.0 * cos(2π * x)
            __m256d cos_arg = _mm256_mul_pd(two_pi_vec, x_vec);
            // Используем стандартный cos
            alignas(32) double cos_args[4];
            alignas(32) double cos_vals[4];
            _mm256_store_pd(cos_args, cos_arg);
            for (int j = 0; j < 4; ++j) {
                cos_vals[j] = std::cos(cos_args[j]);
            }
            __m256d cos_val = _mm256_load_pd(cos_vals);
            __m256d term = _mm256_mul_pd(ten_vec, cos_val);

            // x² - 10*cos(2π*x) + 10
            __m256d result = _mm256_add_pd(_mm256_sub_pd(x_sq, term), ten_vec);
            sum_vec = _mm256_add_pd(sum_vec, result);
        }

        // Горизонтальное суммирование
        sum_vec = _mm256_hadd_pd(sum_vec, sum_vec);
        sum = ((double*)&sum_vec)[0] + ((double*)&sum_vec)[2];
    }
    else {
        // Стандартная реализация
        for (int i = 0; i < num_vars; ++i) {
            double x = compute_parameter(params, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
            sum += x * x - 10.0 * std::cos(two_pi * x) + 10.0;
        }
    }
    return sum;
}
#endif
#if (ACKLEY)
double benchmark_function(double* params) noexcept {
    const int num_vars = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;
    double sum_sq = 0.0;
    double sum_cos = 0.0;

    // AVX-оптимизация для ACKLEY
    if (SET_PARAMETR_SIZE_ONE_X == 4) {
        __m256d sum_sq_vec = _mm256_setzero_pd();
        __m256d sum_cos_vec = _mm256_setzero_pd();
        __m256d two_pi_vec = _mm256_set1_pd(2.0 * M_PI);

        for (int i = 0; i < num_vars; ++i) {
            double x = compute_parameter(params, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
            __m256d x_vec = _mm256_set1_pd(x);

            // Сумма квадратов
            sum_sq_vec = _mm256_add_pd(sum_sq_vec, _mm256_mul_pd(x_vec, x_vec));

            // Сумма косинусов
            __m256d cos_arg = _mm256_mul_pd(two_pi_vec, x_vec);
            alignas(32) double cos_args[4];
            alignas(32) double cos_vals[4];
            _mm256_store_pd(cos_args, cos_arg);
            for (int j = 0; j < 4; ++j) {
                cos_vals[j] = std::cos(cos_args[j]);
            }
            __m256d cos_val = _mm256_load_pd(cos_vals);
            sum_cos_vec = _mm256_add_pd(sum_cos_vec, cos_val);
        }

        // Горизонтальное суммирование
        sum_sq_vec = _mm256_hadd_pd(sum_sq_vec, sum_sq_vec);
        sum_sq = ((double*)&sum_sq_vec)[0] + ((double*)&sum_sq_vec)[2];

        sum_cos_vec = _mm256_hadd_pd(sum_cos_vec, sum_cos_vec);
        sum_cos = ((double*)&sum_cos_vec)[0] + ((double*)&sum_cos_vec)[2];
    }
    else {
        // Стандартная реализация
        for (int i = 0; i < num_vars; ++i) {
            double x = compute_parameter(params, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
            sum_sq += x * x;
            sum_cos += std::cos(2.0 * M_PI * x);
        }
    }

    double n = static_cast<double>(num_vars);
    return -20.0 * std::exp(-0.2 * std::sqrt(sum_sq / n)) - std::exp(sum_cos / n) + 20.0 + M_E;
}
#endif
#if (SPHERE)
double benchmark_function(double* params) noexcept {
    double sum_sq = 0.0;
    const int num_vars = PARAMETR_SIZE / SET_PARAMETR_SIZE_ONE_X;

    // AVX-оптимизация для SPHERE
    if (SET_PARAMETR_SIZE_ONE_X == 4) {
        __m256d sum_sq_vec = _mm256_setzero_pd();

        for (int i = 0; i < num_vars; ++i) {
            double x = compute_parameter(params, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
            __m256d x_vec = _mm256_set1_pd(x);
            sum_sq_vec = _mm256_add_pd(sum_sq_vec, _mm256_mul_pd(x_vec, x_vec));
        }

        // Горизонтальное суммирование
        sum_sq_vec = _mm256_hadd_pd(sum_sq_vec, sum_sq_vec);
        sum_sq = ((double*)&sum_sq_vec)[0] + ((double*)&sum_sq_vec)[2];
    }
    else {
        // Стандартная реализация
        for (int i = 0; i < num_vars; ++i) {
            double x = compute_parameter(params, i * SET_PARAMETR_SIZE_ONE_X, SET_PARAMETR_SIZE_ONE_X);
            sum_sq += x * x;
        }
    }
    return sum_sq;
}
#endif

struct ShardedHashTable {
    std::vector<HashEntry*> tables;
    int shards;
    int size_per_shard;

    // Конструктор для инициализации
    ShardedHashTable(int total_size, int num_shards) {
        shards = num_shards;
        size_per_shard = total_size / num_shards;
        tables.resize(shards, nullptr);

        for (int i = 0; i < shards; ++i) {
            tables[i] = new HashEntry[size_per_shard];
            for (int j = 0; j < size_per_shard; ++j) {
                tables[i][j].key.store(ZERO_HASH, std::memory_order_relaxed);
                tables[i][j].value = 0.0;
                tables[i][j].timestamp.store(0, std::memory_order_relaxed);
            }
        }
    }

    // Деструктор для очистки памяти
    ~ShardedHashTable() {
        for (int i = 0; i < shards; ++i) {
            if (tables[i] != nullptr) {
                delete[] tables[i];
                tables[i] = nullptr;
            }
        }
        tables.clear();
    }
};
// Оптимизированная хэш-функция
unsigned long long compute_hash_fnv1a(const int* path) noexcept {
    const unsigned long long prime = 1099511628211ULL;
    unsigned long long hash = 14695981039346656037ULL;

    // Обрабатываем элементы напрямую
    for (int i = 0; i < PARAMETR_SIZE; ++i) {
        hash ^= static_cast<unsigned long long>(path[i]);
        hash *= prime;
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

// AVX2-оптимизированное вычисление матрицы вероятностей для MAX_VALUE_SIZE=4
void compute_probability_matrix(double* __restrict pheromone, double* __restrict visits, double* __restrict probabilities, PerformanceMetrics& metrics) noexcept {
    auto start = std::chrono::high_resolution_clock::now();

    const int total_params = PARAMETR_SIZE;

#pragma omp parallel for schedule(static)
    for (int param = 0; param < total_params; ++param) {
        const int base = param * MAX_VALUE_SIZE;

        // Загружаем 4 значения феромонов в AVX-регистр
        __m256d pheromone_vec = _mm256_loadu_pd(&pheromone[base]);
        __m256d visits_vec = _mm256_loadu_pd(&visits[base]);

        // Вычисляем сумму феромонов (горизонтальное суммирование)
        __m256d sum_pheromone_vec = _mm256_hadd_pd(pheromone_vec, pheromone_vec);
        double total_pheromone = ((double*)&sum_pheromone_vec)[0] + ((double*)&sum_pheromone_vec)[2];

        if (total_pheromone <= 0.0) {
            // Равномерное распределение
            __m256d uniform_probs = _mm256_set_pd(1.0, 0.75, 0.5, 0.25);
            _mm256_storeu_pd(&probabilities[base], uniform_probs);
            continue;
        }

        // Вычисление вероятностей с использованием AVX
        __m256d inv_total_pheromone = _mm256_set1_pd(1.0 / total_pheromone);
        __m256d norm_pheromone = _mm256_mul_pd(pheromone_vec, inv_total_pheromone);

        // Вычисляем вероятности: 1.0/visits + norm_pheromone
        __m256d inv_visits = _mm256_div_pd(_mm256_set1_pd(1.0), visits_vec);
        __m256d temp_probs = _mm256_add_pd(inv_visits, norm_pheromone);

        // Заменяем NaN/Inf на 0.0
        __m256d zero_vec = _mm256_setzero_pd();
        __m256d valid_mask = _mm256_and_pd(
            _mm256_cmp_pd(visits_vec, zero_vec, _CMP_GT_OQ),
            _mm256_cmp_pd(pheromone_vec, zero_vec, _CMP_GT_OQ)
        );
        temp_probs = _mm256_blendv_pd(zero_vec, temp_probs, valid_mask);

        // Суммируем вероятности
        __m256d sum_probs_vec = _mm256_hadd_pd(temp_probs, temp_probs);
        double sum_probs = ((double*)&sum_probs_vec)[0] + ((double*)&sum_probs_vec)[2];

        if (sum_probs > 0.0) {
            // Нормализуем и вычисляем кумулятивные вероятности
            __m256d inv_sum_probs = _mm256_set1_pd(1.0 / sum_probs);
            __m256d norm_probs = _mm256_mul_pd(temp_probs, inv_sum_probs);

            // Вычисляем кумулятивную сумму
            double cumulative = 0.0;
            double probs[4];
            _mm256_storeu_pd(probs, norm_probs);

            for (int i = 0; i < 4; ++i) {
                cumulative += probs[i];
                probabilities[base + i] = cumulative;
            }
            probabilities[base + 3] = 1.0; // Гарантируем, что последнее значение = 1.0
        }
        else {
            // Равномерное распределение при нулевых вероятностях
            __m256d uniform_probs = _mm256_set_pd(1.0, 0.75, 0.5, 0.25);
            _mm256_storeu_pd(&probabilities[base], uniform_probs);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    metrics.probability_time += std::chrono::duration<double, std::milli>(end - start).count();
}
// AVX2-оптимизированная генерация агентов (оптимизированная по памяти)
void generate_agents(int iteration, double* __restrict params, double* __restrict probabilities, double* __restrict agents, int* __restrict paths, double* __restrict scores, ShardedHashTable& hash_table, PerformanceMetrics& metrics) noexcept {
    auto start = std::chrono::high_resolution_clock::now();

    const int num_agents = ANT_SIZE;
    const int num_params = PARAMETR_SIZE;

#pragma omp parallel
    {
        std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count() + iteration * 1000 + omp_get_thread_num() * 12345);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

#pragma omp for schedule(static)
        for (int agent = 0; agent < num_agents; ++agent) {
            const int agent_base = agent * num_params;
            int choice = 0; // Значение по умолчанию (первый элемент)
            // Генерируем путь с оптимизированным бинарным поиском
            for (int param = 0; param < num_params; ++param) {
                if (choice !=3 ){
                    double rand_val = dist(rng);
                    const int prob_base = param * MAX_VALUE_SIZE;

                    // Загружаем 4 вероятности в AVX-регистр
                    __m256d prob_vec = _mm256_loadu_pd(&probabilities[prob_base]);

                    // Сравниваем случайное значение с вероятностями
                    __m256d rand_vec = _mm256_set1_pd(rand_val);
                    __m256d cmp_result = _mm256_cmp_pd(rand_vec, prob_vec, _CMP_LE_OQ);

                    int mask = _mm256_movemask_pd(cmp_result);
                

                    // Находим первый установленный бит (первую вероятность, которая >= rand_val)
                    if (mask != 0) {
                        choice = __builtin_ctz(mask);
                    }
                }
                else {
                    choice = 0;
                }

                paths[agent_base + param] = choice;
                agents[agent_base + param] = params[param * MAX_VALUE_SIZE + choice];
            }

            // Используем thread-safe lookup-or-compute функцию
            scores[agent] = lookup_or_compute(hash_table, &paths[agent_base],&agents[agent_base], iteration, metrics);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    metrics.agent_time += std::chrono::duration<double, std::milli>(end - start).count();
    metrics.total_agents += ANT_SIZE;
}
// AVX2-оптимизированное обновление феромонов (оптимизированное по памяти)
void update_pheromones(double* __restrict pheromone, double* __restrict visits, const int* __restrict paths, const double* __restrict scores, PerformanceMetrics& metrics) noexcept {
    auto start = std::chrono::high_resolution_clock::now();

    const int total_cells = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int num_agents = ANT_SIZE;

    // Испарение феромонов с AVX
    __m256d ro_vec = _mm256_set1_pd(PARAMETR_RO);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < total_cells; i += 4) {
        if (i + 3 < total_cells) {
            __m256d pheromone_vec = _mm256_loadu_pd(&pheromone[i]);
            __m256d evaporated = _mm256_mul_pd(pheromone_vec, ro_vec);
            _mm256_storeu_pd(&pheromone[i], evaporated);
        }
        else {
            // Обработка оставшихся элементов
            for (int j = i; j < total_cells; ++j) {
                pheromone[j] *= PARAMETR_RO;
            }
        }
    }

    // Обновление посещений и феромонов с оптимизацией памяти
#pragma omp parallel
    {
        // Используем фиксированные массивы вместо векторов для избежания динамического выделения
        const int local_size = total_cells;
        double* local_pheromone = new double[local_size]();
        int* local_visits = new int[local_size]();

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

        // Освобождаем память
        delete[] local_pheromone;
        delete[] local_visits;
    }

    auto end = std::chrono::high_resolution_clock::now();
    metrics.pheromone_time += std::chrono::duration<double, std::milli>(end - start).count();
}

// Загрузка матрицы
bool load_matrix(const std::string& filename, double* __restrict params, double* __restrict pheromone, double* __restrict visits) noexcept {
    std::ifstream file(filename);
    if (!file) {
        std::cout << "File not found, generating synthetic matrix..." << std::endl;
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> dist(-10.0, 10.0);

        // Оптимизированная инициализация без AVX для простоты
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
            double score_a = (0.6 * (1.0 / a.average_time)) + (0.4 * a.metrics.parallel_efficiency);
            double score_b = (0.6 * (1.0 / b.average_time)) + (0.4 * b.metrics.parallel_efficiency);
            return score_a > score_b;
        });

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "OPTIMAL CONFIGURATION" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Recommended thread count: " << best_result->thread_count << std::endl;
    std::cout << "Average execution time: " << best_result->average_time << " ms" << std::endl;
    std::cout << "Parallel efficiency: " << best_result->metrics.parallel_efficiency << "%" << std::endl;
    std::cout << "Speedup: " << best_result->metrics.speedup << "x" << std::endl;
    std::cout << "Best objective value: " << best_result->best_objective << std::endl;
}
PerformanceMetrics run_single_experiment(int num_threads, int iteration_count = KOL_ITERATION) {
    PerformanceMetrics metrics;

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    // Вычисляем размеры массивов
    const int matrix_size = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int agents_size = PARAMETR_SIZE * ANT_SIZE;

    // Создаем хэш-таблицу с умным управлением памятью
    ShardedHashTable hash_table(HASH_TABLE_SIZE, std::min(num_threads, 8));

    // Выделяем память для основных массивов
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

    // Основной цикл оптимизации
    for (int iter = 0; iter < iteration_count; ++iter) {
        metrics.total_iterations++;

        auto iter_start = std::chrono::high_resolution_clock::now();

        // Основные фазы алгоритма
        compute_probability_matrix(pheromone.data(), visits.data(), probabilities.data(), metrics);
        generate_agents(iter, params.data(), probabilities.data(), agents.data(), paths.data(), scores.data(), hash_table, metrics);
        update_pheromones(pheromone.data(), visits.data(), paths.data(), scores.data(), metrics);

        // Обновление лучших значений
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

    // Память автоматически освобождается при выходе из функции
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

    std::vector<int> thread_counts = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

    // Добавляем количество физических ядер если оно не в списке
    if (hardware_threads > 0 && hardware_threads <= max_threads) {
        if (std::find(thread_counts.begin(), thread_counts.end(), hardware_threads) == thread_counts.end()) {
            thread_counts.push_back(hardware_threads);
        }
    }

    // Добавляем максимальное количество потоков если оно разумное
    if (max_threads <= 16 && std::find(thread_counts.begin(), thread_counts.end(), max_threads) == thread_counts.end()) {
        thread_counts.push_back(max_threads);
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
        PerformanceMetrics average_metrics;
        PerformanceMetrics square_metrics;
        int successful_runs = 0;

        for (int run = 0; run < KOL_PROGON_STATISTICS; ++run) {
            std::cout << "  Run " << (run + 1) << "/" << KOL_PROGON_STATISTICS << "...";

            auto metrics = run_single_experiment(thread_count);

            if (metrics.total_time > 0) {
                // Суммируем только основные метрики для экономии памяти
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

                square_metrics.total_time += metrics.total_time* metrics.total_time;
                square_metrics.iteration_time += metrics.iteration_time*metrics.iteration_time;
                square_metrics.probability_time += metrics.probability_time* metrics.probability_time;
                square_metrics.agent_time += metrics.agent_time* metrics.agent_time;
                square_metrics.pheromone_time += metrics.pheromone_time* metrics.pheromone_time;
                square_metrics.hash_time += metrics.hash_time* metrics.hash_time;
                square_metrics.evaluation_time += metrics.evaluation_time*metrics.evaluation_time;
                square_metrics.hash_hits += metrics.hash_hits;
                square_metrics.hash_misses += metrics.hash_misses;

                if (metrics.min_objective < average_metrics.min_objective) {
                    average_metrics.min_objective = metrics.min_objective;
                }
                if (metrics.max_objective > average_metrics.max_objective) {
                    average_metrics.max_objective = metrics.max_objective;
                }

                successful_runs++;
                std::cout << " OK (" << metrics.total_time << " ms) min=" << metrics.min_objective << std::endl;
            }
            else {
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

            // Вычисляем статистику

            calculate_efficiency(average_metrics, thread_count, single_thread_time);

            ThreadExperimentResult result;
            result.thread_count = thread_count;
            result.metrics = average_metrics;
            result.square_metrics = square_metrics;
            results.push_back(result);

            std::cout << "  ✓ Completed " << successful_runs << "/" << KOL_PROGON_STATISTICS
                << " runs, avg: " << average_metrics.total_time << " ms" << std::endl;
        }
    }

    return results;
}

void log_parameters() {
    outfile << "OpenMP version: " << _OPENMP << "; AVX2 optimization enabled; "
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
    log_parameters();
    std::cout << "=== MEMORY-OPTIMIZED THREAD SCALING ANALYSIS ===" << std::endl;
    std::cout << "OpenMP version: " << _OPENMP << std::endl;
    std::cout << "AVX2 optimization enabled" << std::endl;
    std::cout << "\n=== OPTIMIZED OPENMP ALGORITHM WITH AVX2 ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  PARAMETR_SIZE: " << PARAMETR_SIZE << std::endl;
    std::cout << "  SET_PARAMETR_SIZE_ONE_X: " << SET_PARAMETR_SIZE_ONE_X << std::endl;
    std::cout << "  MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << " (AVX2 optimized)" << std::endl;
    std::cout << "  ANT_SIZE: " << ANT_SIZE << std::endl;
    std::cout << "  KOL_ITERATION: " << KOL_ITERATION << std::endl;
    std::cout << "  Hash table size: " << HASH_TABLE_SIZE << " (sharded)" << std::endl;

    auto start_time = std::chrono::system_clock::now();
    std::time_t start_time_t = std::chrono::system_clock::to_time_t(start_time);
    std::cout << "Start time: " << std::ctime(&start_time_t);

    try {
        auto results = run_thread_scaling_experiment();
        print_comparison_table(results);
        print_optimal_configuration(results);

        std::cout << "\nExperiment completed successfully!" << std::endl;
    }
    catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        std::cerr << "Try reducing KOL_PROGON_STATISTICS or ANT_SIZE" << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    auto end_time = std::chrono::system_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
    std::cout << "Total duration: " << total_duration.count() << " minutes" << std::endl;
}