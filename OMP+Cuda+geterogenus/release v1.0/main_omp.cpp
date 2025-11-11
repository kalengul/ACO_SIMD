#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <algorithm>
#include <string>
#include <fstream>
#include <windows.h>
#include <iomanip>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <numeric>
#include <deque>
#include "cuda_module.h"


// Исправляем конфликт имен с Windows макросами
#undef max
#undef min

// 4 - 42, 84, 168, 336, 672, 1344, 2688,

// Добавляем константы напрямую
#define MAX_VALUE_SIZE 4
#define PARAMETR_SIZE 1344
#define ANT_SIZE 500
#define MAX_THREAD_CUDA 256
#define NAME_FILE_GRAPH "Parametr_Graph/test1344_4.txt"
#define KOL_ITERATION 500

#define KOL_PROGREV 5   //5
#define KOL_PROGON_STATISTICS 50   //50


#define PARAMETR_RO 0.999
#define PARAMETR_Q 1.0

#define PRINT_INFORMATION 0

// Оптимизационные флаги
#define OPTIMIZE_MIN_1 1
#define OPTIMIZE_MIN_2 0
#define OPTIMIZE_MAX 0
#define MAX_PARAMETR_VALUE_TO_MIN_OPT 1000.0

#define GO_HYBRID_OMP 1
#define GO_HYBRID_BALANCED_OMP 0

#ifdef _OPENMP
#include <omp.h>
#endif

// Новые константы для балансировки
#define BALANCE_UPDATE_INTERVAL 10
#define INITIAL_CPU_ANTS_RATIO 0.3f
#define MIN_ANTS_PER_DEVICE 50
#define MAX_BALANCE_HISTORY 20

// Структура для метрик производительности
struct PerformanceMetrics {
    double cpu_execution_time;
    double gpu_execution_time;
    int cpu_ants_processed;
    int gpu_ants_processed;
    double cpu_throughput;  // муравьи/миллисекунду
    double gpu_throughput;
    std::chrono::steady_clock::time_point timestamp;
};

// Структура для передачи данных между потоками
struct IterationData {
    int iteration;
    std::vector<double> norm_matrix_probability;
    std::vector<int> ant_parametr;
    std::vector<double> antOF;
    double minOf;
    double maxOf;
    int kol_hash_fail;
    int ants_count;        // Количество муравьев для обработки
    int ants_processed;    // Сколько фактически обработано

    IterationData() : iteration(0), minOf(0), maxOf(0), kol_hash_fail(0),
        ants_count(0), ants_processed(0) {}
};

class ThreadSafeQueue {
private:
    std::queue<IterationData> queue;
    mutable std::mutex mtx; 
    std::condition_variable cv;
    bool stopped = false;

public:
    void push(const IterationData& data) {
        std::lock_guard<std::mutex> lock(mtx);
        queue.push(data);
        cv.notify_one();
    }

    bool pop(IterationData& data) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]() { return !queue.empty() || stopped; });

        if (stopped && queue.empty()) return false;

        data = std::move(queue.front());
        queue.pop();
        return true;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mtx);
        stopped = true;
        cv.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }

    void printQueueInfo(const std::string& prefix = "") const {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << prefix << "ThreadSafeQueue Info:" << std::endl;
        std::cout << prefix << "  Queue Size: " << queue.size() << std::endl;
        std::cout << prefix << "  Stopped: " << (stopped ? "Yes" : "No") << std::endl;
    }
};

// Глобальные переменные для синхронизации
std::atomic<int> active_cuda_tasks{ 0 };
std::atomic<int> completed_iterations{ 0 };
std::atomic<int> successful_iterations{ 0 };

std::ofstream logFile; // Глобальная переменная для лог-файла

// Функция для вывода PerformanceMetrics
void printPerformanceMetrics(const PerformanceMetrics& metrics, const std::string& prefix = "") {
    std::cout << prefix << "Performance Metrics:" << std::endl;
    std::cout << prefix << "  CPU Execution Time: " << metrics.cpu_execution_time << " ms" << std::endl;
    std::cout << prefix << "  GPU Execution Time: " << metrics.gpu_execution_time << " ms" << std::endl;
    std::cout << prefix << "  CPU Ants Processed: " << metrics.cpu_ants_processed << std::endl;
    std::cout << prefix << "  GPU Ants Processed: " << metrics.gpu_ants_processed << std::endl;
    std::cout << prefix << "  CPU Throughput: " << metrics.cpu_throughput << " ants/ms" << std::endl;
    std::cout << prefix << "  GPU Throughput: " << metrics.gpu_throughput << " ants/ms" << std::endl;
    std::cout << prefix << "  Timestamp: " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(
            metrics.timestamp.time_since_epoch()).count() << " ms" << std::endl;
}
// Функция для вывода IterationData
void printIterationData(const IterationData& data, const std::string& prefix = "") {
    std::cout << prefix << "Iteration Data:" << std::endl;
    std::cout << prefix << "  Iteration: " << data.iteration << std::endl;
    std::cout << prefix << "  MinOf: " << data.minOf << std::endl;
    std::cout << prefix << "  MaxOf: " << data.maxOf << std::endl;
    std::cout << prefix << "  Kol Hash Fail: " << data.kol_hash_fail << std::endl;
    std::cout << prefix << "  Ants Count: " << data.ants_count << std::endl;
    std::cout << prefix << "  Ants Processed: " << data.ants_processed << std::endl;

    // Вывод размеров векторов
    std::cout << prefix << "  Norm Matrix Probability Size: "
        << data.norm_matrix_probability.size() << std::endl;
    std::cout << prefix << "  Ant Parametr Size: "
        << data.ant_parametr.size() << std::endl;
    std::cout << prefix << "  AntOF Size: "
        << data.antOF.size() << std::endl;

    std::cout << "norm_matrix_probability (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
    for (int i = 0; i < PARAMETR_SIZE; ++i) {
        for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
            std::cout << data.norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ", ";
        }
        std::cout << std::endl; // Переход на новую строку
    }

    if (!data.antOF.empty()) {
        std::cout << prefix << "  First 5 AntOF Values: ";
        for (int i = 0; i < data.antOF.size(); ++i) {
            std::cout << data.antOF[i] << " ";
        }
        std::cout << std::endl;
    }
}

// Callback функция для CUDA
void cuda_completion_callback(double* results, int size, int iteration) {
    double min_val = results[0];
    double max_val = results[0];
    double sum = 0.0;

    for (int i = 0; i < size; i++) {
        if (results[i] < min_val) min_val = results[i];
        if (results[i] > max_val) max_val = results[i];
        sum += results[i];
    }

    int iter_num = completed_iterations.fetch_add(1) + 1;
    successful_iterations.fetch_add(1);
#if (PRINT_INFORMATION)
    if (iteration % KOL_PROGON_STATISTICS == 0) {
        std::cout << "[Callback] Iteration " << iteration << " (Total: " << iter_num
            << "): Min=" << min_val << ", Max=" << max_val
            << ", Avg=" << (sum / size) << std::endl;
    }
#endif // PRINT_INFORMATION==1
    

    active_cuda_tasks.fetch_sub(1);
}

// Функция загрузки параметров из файла
bool load_matrix(const std::string& filename, double* parametr_value, double* pheromon_value, double* kol_enter_value) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Don't open file!" << std::endl;
        return false;
    }
#if (PRINT_INFORMATION)
    std::cout << "[Main] Loading parameters from file: " << filename << std::endl;
#endif // PRINT_INFORMATION==1

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
            }
            else {
                pheromon_value[k] = 0.0;
                parametr_value[k] = 0.0;
                kol_enter_value[k] = 0.0;
            }
        }
    }
    infile.close();
#if (PRINT_INFORMATION)
    std::cout << "[Main] Parameters loaded successfully!" << std::endl;
#endif // PRINT_INFORMATION==1

    return true;
}

// Функция для вычисления вероятностной формулы
inline double probability_formula_non_cuda(double pheromon, double kol_enter) {
    return (kol_enter != 0.0 && pheromon != 0.0) ? (1.0 / kol_enter + pheromon) : 0.0;
}

class LoadBalancer {
private:
    std::deque<PerformanceMetrics> metrics_history;
    std::mutex metrics_mutex;
    float current_cpu_ratio;
    int total_ants;

public:
    LoadBalancer(int total_ants_count)
        : current_cpu_ratio(INITIAL_CPU_ANTS_RATIO)
        , total_ants(total_ants_count) { }

    void initialize() {
        current_cpu_ratio = INITIAL_CPU_ANTS_RATIO;
        total_ants = ANT_SIZE;
#if (PRINT_INFORMATION)
        std::cout << "[Main] LoadBalancer INITIALIZATION " << std::endl;
#endif // PRINT_INFORMATION==1
    }

    void printLoadBalancerInfo(const std::string& prefix = "") const {
        std::cout << prefix << "LoadBalancer Info:" << std::endl;
        std::cout << prefix << "  Current CPU Ratio: " << current_cpu_ratio << std::endl;
        std::cout << prefix << "  Total Ants: " << total_ants << std::endl;
        std::cout << prefix << "  Metrics History Size: " << metrics_history.size() << std::endl;
        std::cout << prefix << "  MIN_ANTS_PER_DEVICE: " << MIN_ANTS_PER_DEVICE << std::endl;
        std::cout << prefix << "  MAX_BALANCE_HISTORY: " << MAX_BALANCE_HISTORY << std::endl;

        // Вывод распределения муравьев
        int cpu_ants, gpu_ants;
        const_cast<LoadBalancer*>(this)->get_ant_distribution(cpu_ants, gpu_ants);
        std::cout << prefix << "  Current Distribution - CPU: " << cpu_ants << ", GPU: " << gpu_ants << std::endl;

        // Вывод последних метрик
        if (!metrics_history.empty()) {
            std::cout << prefix << "  Last Metrics:" << std::endl;
            printPerformanceMetrics(metrics_history.back(), prefix + "    ");
        }
    }

    // Добавление новых метрик
    void add_metrics(const PerformanceMetrics& metrics) {
        std::lock_guard<std::mutex> lock(metrics_mutex);

        metrics_history.push_back(metrics);
        if (metrics_history.size() > MAX_BALANCE_HISTORY) {
            metrics_history.pop_front();
        }

        update_balance_ratio();
    }

    // Получение текущего распределения муравьев
    void get_ant_distribution(int& cpu_ants, int& gpu_ants) {
        std::lock_guard<std::mutex> lock(metrics_mutex);

        cpu_ants = (std::max)(MIN_ANTS_PER_DEVICE,
            static_cast<int>(total_ants * current_cpu_ratio));
        gpu_ants = total_ants - cpu_ants;

        // Гарантируем минимальное количество на каждом устройстве
        if (cpu_ants < MIN_ANTS_PER_DEVICE) {
            cpu_ants = MIN_ANTS_PER_DEVICE;
            gpu_ants = total_ants - cpu_ants;
        }
        if (gpu_ants < MIN_ANTS_PER_DEVICE) {
            gpu_ants = MIN_ANTS_PER_DEVICE;
            cpu_ants = total_ants - gpu_ants;
        }
    }

    float get_current_cpu_ratio() const {
        return current_cpu_ratio;
    }

private:
    void update_balance_ratio() {
        if (metrics_history.size() < 3) return;

        double avg_cpu_throughput = 0.0;
        double avg_gpu_throughput = 0.0;
        int count = 0;

        for (const auto& metric : metrics_history) {
            avg_cpu_throughput += metric.cpu_throughput;
            avg_gpu_throughput += metric.gpu_throughput;
            count++;
        }

        avg_cpu_throughput /= count;
        avg_gpu_throughput /= count;

        if (avg_cpu_throughput + avg_gpu_throughput > 0) {
            // Вычисляем новое соотношение на основе производительности
            float new_ratio = avg_cpu_throughput / (avg_cpu_throughput + avg_gpu_throughput);

            // Плавное изменение (чтобы избежать резких скачков)
            current_cpu_ratio = 0.7f * current_cpu_ratio + 0.3f * new_ratio;

            // Ограничиваем диапазон
            current_cpu_ratio = (std::max)(0.1f, (std::min)(0.9f, current_cpu_ratio));

#if (PRINT_INFORMATION)
            std::cout << "[LoadBalancer] CPU ratio: " << current_cpu_ratio
                << " (CPU throughput: " << avg_cpu_throughput
                << ", GPU throughput: " << avg_gpu_throughput << ")" << std::endl;
#endif
        }
    }
};

class BalancedHybridACO {
private:
    // Данные ACO
    std::vector<double> parametr_value;
    std::vector<double> current_pheromon;
    std::vector<double> current_kol_enter;
    std::vector<double> norm_matrix_probability;
    std::vector<int> ant_parametr;
    std::vector<double> antOF;

    // Управление выполнением
    std::atomic<bool> stop_requested{ false };
    std::atomic<int> current_iteration{ 0 };

    // Результаты
    double minOf = 1e9;
    double maxOf = -1e9;
    double global_minOf = 1e9;
    double global_maxOf = -1e9;
    int kol_hash_fail = 0;

    // Время выполнения
    double Time_CPU_all = 0.0, Time_CPU_prob = 0.0, Time_CPU_wait = 0.0, Time_CPU_update = 0.0;
    double Time_GPU_all = 0.0, Time_GPU = 0.0, Time_GPU_function = 0.0;

    // Очереди для межпоточного обмена
    ThreadSafeQueue gpu_to_cpu_queue;
    ThreadSafeQueue cpu_to_gpu_queue;

    // Система балансировки
    LoadBalancer load_balancer;

    // Для балансировки
    std::atomic<int> cpu_ants_count;
    std::atomic<int> gpu_ants_count;
    std::vector<PerformanceMetrics> recent_metrics;
    std::mutex metrics_mutex;

public:
    BalancedHybridACO()
        : load_balancer(ANT_SIZE)
        , cpu_ants_count(static_cast<int>(ANT_SIZE* INITIAL_CPU_ANTS_RATIO))
        , gpu_ants_count(ANT_SIZE - cpu_ants_count.load()) {
        norm_matrix_probability.resize(MAX_VALUE_SIZE * PARAMETR_SIZE);
        ant_parametr.resize(PARAMETR_SIZE * ANT_SIZE);
        antOF.resize(ANT_SIZE);
        std::cout << "[Main] BalancedHybridACO INITIALIZATION" << std::endl;
    }

    bool initialize(const std::vector<double>& parametr_value_new, const std::vector<double>& pheromon_value, const std::vector<double>& kol_enter_value) {
#if (PRINT_INFORMATION)
        std::cout << "[Main] Initializing BalancedHybridACO" << std::endl;
#endif // PRINT_INFORMATION==1
        parametr_value = parametr_value_new;
        current_pheromon = pheromon_value;
        current_kol_enter = kol_enter_value;
        norm_matrix_probability.resize(MAX_VALUE_SIZE * PARAMETR_SIZE);

        ant_parametr.resize(PARAMETR_SIZE * ANT_SIZE);
        antOF.resize(ANT_SIZE);

        // Инициализация переменных результатов
        global_minOf = 1e9;
        global_maxOf = -1e9;
        kol_hash_fail = 0;

        load_balancer.initialize();

#if (PRINT_INFORMATION)
        std::cout << "[Main] Initializing CUDA..." << std::endl;
#endif // PRINT_INFORMATION==1

        // Инициализация CUDA
        if (!cuda_initialize(parametr_value.data(), pheromon_value.data(), kol_enter_value.data())) {
            std::cerr << "[Main] CUDA initialization failed!" << std::endl;
            return false;
        }
#if (PRINT_INFORMATION)
        const char* version = cuda_get_version();
        std::cout << "[Main] " << version << std::endl;
        std::cout << "[Main] Balanced Hybrid ACO initialized successfully!" << std::endl;
        std::cout << "[Main] Parameters: " << PARAMETR_SIZE << " params, "
            << MAX_VALUE_SIZE << " values, " << ANT_SIZE << " ants" << std::endl;
#endif // PRINT_INFORMATION==1
        return true;
    }

    bool initialize_from_file(const std::string& filename) {
        const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
        std::vector<double> parametr_value(matrix_size);
        std::vector<double> pheromon_value(matrix_size);
        std::vector<double> kol_enter_value(matrix_size);

        if (!load_matrix(filename, parametr_value.data(), pheromon_value.data(), kol_enter_value.data())) {
            std::cerr << "[Main] Failed to load parameters from file: " << filename << std::endl;
            return false;
        }

        return initialize(parametr_value, pheromon_value, kol_enter_value);
    }

    void printBalancedHybridACOInfo(const std::string& prefix = "") const {
        std::cout << prefix << "BalancedHybridACO Info:" << std::endl;
        std::cout << prefix << "  Stop Requested: " << (stop_requested.load() ? "Yes" : "No") << std::endl;
        std::cout << prefix << "  Current Iteration: " << current_iteration.load() << std::endl;
        std::cout << prefix << "  MinOf: " << minOf << std::endl;
        std::cout << prefix << "  MaxOf: " << maxOf << std::endl;
        std::cout << prefix << "  Global MinOf: " << global_minOf << std::endl;
        std::cout << prefix << "  Global MaxOf: " << global_maxOf << std::endl;
        std::cout << prefix << "  Kol Hash Fail: " << kol_hash_fail << std::endl;

        std::cout << prefix << "  Parametr Value Size: " << parametr_value.size() << std::endl;
        std::cout << prefix << "  Current Pheromon Size: " << current_pheromon.size() << std::endl;
        std::cout << prefix << "  Current Kol Enter Size: " << current_kol_enter.size() << std::endl;
        std::cout << prefix << "  Norm Matrix Probability Size: " << norm_matrix_probability.size() << std::endl;
        std::cout << prefix << "  Ant Parametr Size: " << ant_parametr.size() << std::endl;
        std::cout << prefix << "  AntOF Size: " << antOF.size() << std::endl;

        std::cout << prefix << "  CPU Ants Count: " << cpu_ants_count.load() << std::endl;
        std::cout << prefix << "  GPU Ants Count: " << gpu_ants_count.load() << std::endl;
        std::cout << prefix << "  Recent Metrics Size: " << recent_metrics.size() << std::endl;

        // Время выполнения
        std::cout << prefix << "  Time CPU All: " << Time_CPU_all << " ms" << std::endl;
        std::cout << prefix << "  Time CPU Prob: " << Time_CPU_prob << " ms" << std::endl;
        std::cout << prefix << "  Time CPU Wait: " << Time_CPU_wait << " ms" << std::endl;
        std::cout << prefix << "  Time CPU Update: " << Time_CPU_update << " ms" << std::endl;
        std::cout << prefix << "  Time GPU All: " << Time_GPU_all << " ms" << std::endl;
        std::cout << prefix << "  Time GPU: " << Time_GPU << " ms" << std::endl;
        std::cout << prefix << "  Time GPU Function: " << Time_GPU_function << " ms" << std::endl;

        std::cout << "MATRIX (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
        for (int i = 0; i < PARAMETR_SIZE; ++i) {
            for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                std::cout << " (" << current_pheromon[i * MAX_VALUE_SIZE + j] << " -> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << " + " << current_kol_enter[i * MAX_VALUE_SIZE + j] << ") ";
            }
            std::cout << std::endl; // Переход на новую строку
        }

        if (!antOF.empty()) {
            std::cout << prefix << "  AntOF Values: ";
            for (int i = 0; i < antOF.size(); ++i) {
                std::cout << antOF[i] << " ";
            }
            std::cout << std::endl;
        }

        // Информация о очередях
        cpu_to_gpu_queue.printQueueInfo(prefix + "  CPU->GPU ");
        gpu_to_cpu_queue.printQueueInfo(prefix + "  GPU->CPU ");

        // Информация о балансировщике
        load_balancer.printLoadBalancerInfo(prefix + "  ");
    }

    // Модифицированная функция запуска с балансировкой
    void run_balanced_pipeline() {
#if (PRINT_INFORMATION)
        printBalancedHybridACOInfo();
        std::cout << "[Balanced Pipeline] Starting balanced execution..." << std::endl;
#endif

        std::thread cpu_thread(&BalancedHybridACO::cpu_balanced_thread_function, this);
        std::thread gpu_thread(&BalancedHybridACO::gpu_balanced_thread_function, this);
        std::thread balance_thread(&BalancedHybridACO::balance_monitor_thread, this);

        cpu_thread.join();
        gpu_thread.join();
        balance_thread.join();

        wait_completion();
    }

    void wait_completion(int max_wait_ms = 10000) {
        auto start = std::chrono::steady_clock::now();

        while (active_cuda_tasks.load() > 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);

            if (elapsed.count() > max_wait_ms) {
                std::cerr << "[Main] Timeout waiting for CUDA completion!" << std::endl;
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        cuda_synchronize();
    }

    void cleanup() {
        stop_requested.store(true);
        cpu_to_gpu_queue.stop();
        gpu_to_cpu_queue.stop();
        wait_completion();
        cuda_cleanup();
#if (PRINT_INFORMATION)
        std::cout << "[Main] Cleanup completed. Successful iterations: "
            << successful_iterations.load() << std::endl;
        std::cout << "[Main] Final results - Min: " << global_minOf
            << ", Max: " << global_maxOf
            << ", Hash fails: " << kol_hash_fail << std::endl;
        std::cout << "[Main] Final CPU ratio: " << load_balancer.get_current_cpu_ratio() << std::endl;
#endif // PRINT_INFORMATION==1
    }

    // Геттеры
    int get_active_tasks() const { return active_cuda_tasks.load(); }
    int get_completed_iterations() const { return completed_iterations.load(); }
    double get_global_min() const { return global_minOf; }
    double get_global_max() const { return global_maxOf; }
    int get_hash_fails() const { return kol_hash_fail; }
    double get_time_cpu_all() const { return Time_CPU_all; }
    double get_time_cpu_prob() const { return Time_CPU_prob; }
    double get_time_cpu_wait() const { return Time_CPU_wait; }
    double get_time_cpu_update() const { return Time_CPU_update; }
    double get_time_gpu_all() const { return Time_GPU_all; }
    double get_time_gpu() const { return Time_GPU; }
    double get_time_gpu_function() const { return Time_GPU_function; }
    float get_current_cpu_ratio() const { return load_balancer.get_current_cpu_ratio(); }

private:
    // Методы CPUAntCalculator, интегрированные в BalancedHybridACO

    void set_probability_matrix(const std::vector<double>& prob_matrix) {
        norm_matrix_probability = prob_matrix;
    }

    // Вычисление пути для одного муравья на CPU
    void calculate_ant_path(int ant_index, std::vector<int>& ant_path, std::vector<double>& antOF) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        double agent[PARAMETR_SIZE] = { 0 };
        bool valid_solution = true;

        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            double randomValue = dis(gen);
            int k = 0;

            while (valid_solution && k < MAX_VALUE_SIZE &&
                randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                k++;
            }
            ant_path[ant_index * PARAMETR_SIZE + tx] = k;
            agent[tx] = parametr_value[tx * MAX_VALUE_SIZE + k];
            valid_solution = (k != MAX_VALUE_SIZE - 1);
        }
        // Вычисление целевой функции
        antOF[ant_index] = calculate_objective_function(agent);
    }

    // Пакетное вычисление путей для группы муравьев
    PerformanceMetrics calculate_ant_batch(int start_ant, int num_ants, std::vector<int>& ant_path, std::vector<double>& antOF) {
        auto start_time = std::chrono::high_resolution_clock::now();

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < num_ants; i++) {
            calculate_ant_path(start_ant + i, ant_path, antOF);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        PerformanceMetrics metrics;
        metrics.cpu_execution_time = execution_time;
        metrics.cpu_ants_processed = num_ants;
        metrics.cpu_throughput = (execution_time > 0) ? num_ants / execution_time : 0;
        metrics.timestamp = std::chrono::steady_clock::now();

        return metrics;
    }

    double calculate_objective_function(double agent[PARAMETR_SIZE]) {
        // Используем ту же функцию, что и в CUDA ядре
        // В реальной реализации нужно дублировать логику BenchShafferaFunction
        double sum = 0.0;
        for (int i = 0; i < PARAMETR_SIZE; i++) {
            sum += agent[i] * agent[i];
        }
        return sum;
    }

    // Поток CPU с балансировкой
    void cpu_balanced_thread_function() {
#if (PRINT_INFORMATION)
        std::cout << "[Balanced CPU Thread] Started" << std::endl;
#endif
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int iteration = 0; iteration < KOL_ITERATION && !stop_requested.load(); iteration++) {
            // Вычисляем матрицу вероятностей для текущего состояния
            std::cout << "[Balanced CPU Thread] Started" << iteration << std::endl;
            auto start_time_prob = std::chrono::high_resolution_clock::now();
            calculate_probabilities();
            auto end_time = std::chrono::high_resolution_clock::now();
            Time_CPU_prob += std::chrono::duration<double, std::milli>(end_time - start_time_prob).count();

            set_probability_matrix(norm_matrix_probability);

            // Получаем текущее распределение муравьев
            int current_cpu_ants, current_gpu_ants;
            load_balancer.get_ant_distribution(current_cpu_ants, current_gpu_ants);
            cpu_ants_count.store(current_cpu_ants);
            gpu_ants_count.store(current_gpu_ants);

            if (current_cpu_ants > 0) {
                // Вычисляем муравьев на CPU
                std::vector<int> cpu_ant_path(current_cpu_ants * PARAMETR_SIZE);
                std::vector<double> cpu_antOF(current_cpu_ants);

                auto cpu_metrics = calculate_ant_batch(
                    0, current_cpu_ants, cpu_ant_path, cpu_antOF);

                // Обновляем общие результаты
                update_results_from_cpu(cpu_ant_path, cpu_antOF, iteration);

                // Сохраняем метрики
                {
                    std::lock_guard<std::mutex> lock(metrics_mutex);
                    recent_metrics.push_back(cpu_metrics);
                }
            }

            // Отправляем оставшихся муравьев на GPU
            if (current_gpu_ants > 0) {
                IterationData gpu_data;
                gpu_data.iteration = iteration;
                gpu_data.norm_matrix_probability = norm_matrix_probability;
                gpu_data.ants_count = current_gpu_ants;

                auto start_time_wait = std::chrono::high_resolution_clock::now();
                cpu_to_gpu_queue.push(gpu_data);

                // Ждем результаты от GPU
                IterationData received_gpu_data;
                if (!gpu_to_cpu_queue.pop(received_gpu_data)) {
                    break;
                }
                end_time = std::chrono::high_resolution_clock::now();
                Time_CPU_wait += std::chrono::duration<double, std::milli>(end_time - start_time_wait).count();

                // Обновляем данные на основе результатов GPU
                process_gpu_results(received_gpu_data, iteration);
            }

            // Обновляем феромоны на основе всех результатов
            auto start_time_update = std::chrono::high_resolution_clock::now();
            update_pheromones_async(iteration);
            end_time = std::chrono::high_resolution_clock::now();
            Time_CPU_update += std::chrono::duration<double, std::milli>(end_time - start_time_update).count();

#if (PRINT_INFORMATION)
            if (iteration % KOL_PROGON_STATISTICS == 0) {
                std::cout << "[CPU Thread] Iteration " << iteration
                    << " completed. Min: " << global_minOf
                    << ", Max: " << global_maxOf
                    << " (CPU: " << current_cpu_ants << ", GPU: " << current_gpu_ants << ")" << std::endl;
            }
#endif // PRINT_INFORMATION==1
        }

        // Сигнализируем о завершении
        cpu_to_gpu_queue.stop();
        gpu_to_cpu_queue.stop();
        auto end_time = std::chrono::high_resolution_clock::now();
        Time_CPU_all = std::chrono::duration<double, std::milli>(end_time - start_time).count();
#if (PRINT_INFORMATION)
        std::cout << "[CPU Thread] Finished" << std::endl;
#endif
    }

    // Модифицированная GPU функция
    void gpu_balanced_thread_function() {
#if (PRINT_INFORMATION)
        std::cout << "[Balanced GPU Thread] Started" << std::endl;
#endif
        auto start_time = std::chrono::high_resolution_clock::now();

        while (!stop_requested.load()) {
            IterationData cpu_data;
            if (!cpu_to_gpu_queue.pop(cpu_data)) {
                break;
            }

            // Запускаем вычисления только для назначенного количества муравьев
            process_gpu_ants(cpu_data);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        Time_GPU_all = std::chrono::duration<double, std::milli>(end_time - start_time).count();
#if (PRINT_INFORMATION)
        std::cout << "[GPU Thread] Finished" << std::endl;
#endif
    }

    // Поток мониторинга и балансировки
    void balance_monitor_thread() {
#if (PRINT_INFORMATION)
        std::cout << "[Balance Monitor Thread] Started" << std::endl;
#endif
        int iteration_count = 0;

        while (!stop_requested.load() && iteration_count < KOL_ITERATION) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            // Каждые BALANCE_UPDATE_INTERVAL итераций обновляем балансировку
            if (iteration_count % BALANCE_UPDATE_INTERVAL == 0 && !recent_metrics.empty()) {
                PerformanceMetrics avg_metrics = calculate_average_metrics();
                load_balancer.add_metrics(avg_metrics);
            }
            iteration_count++;
        }
#if (PRINT_INFORMATION)
        std::cout << "[Balance Monitor Thread] Finished" << std::endl;
#endif
    }

    void process_gpu_ants(const IterationData& cpu_data) {
        // Запуск вычислений на GPU с текущей матрицей вероятностей
        active_cuda_tasks.fetch_add(1);

        // Используем существующую функцию
        cuda_run_iteration(cpu_data.norm_matrix_probability.data(),
            ant_parametr.data(),
            antOF.data(),
            &minOf,
            &maxOf,
            &kol_hash_fail,
            &Time_GPU,
            &Time_GPU_function,
            cpu_data.iteration,
            cuda_completion_callback);

        // Отправляем результаты обратно в CPU поток
        IterationData gpu_data;
        gpu_data.iteration = cpu_data.iteration;
        gpu_data.ant_parametr = ant_parametr;
        gpu_data.antOF = antOF;
        gpu_data.minOf = minOf;
        gpu_data.maxOf = maxOf;
        gpu_data.kol_hash_fail = kol_hash_fail;
        gpu_data.ants_processed = cpu_data.ants_count;

        gpu_to_cpu_queue.push(gpu_data);
    }

    void update_results_from_cpu(const std::vector<int>& cpu_ant_path,
        const std::vector<double>& cpu_antOF,
        int iteration) {
        // Обновляем общие массивы с результатами CPU вычислений
        int current_cpu_ants = cpu_antOF.size();

        for (int i = 0; i < current_cpu_ants; i++) {
            // Копируем путь муравья
            std::copy(cpu_ant_path.begin() + i * PARAMETR_SIZE,
                cpu_ant_path.begin() + (i + 1) * PARAMETR_SIZE,
                ant_parametr.begin() + i * PARAMETR_SIZE);

            // Копируем значение целевой функции
            antOF[i] = cpu_antOF[i];

            // Обновляем минимум и максимум
            if (cpu_antOF[i] < global_minOf) global_minOf = cpu_antOF[i];
            if (cpu_antOF[i] > global_maxOf) global_maxOf = cpu_antOF[i];
        }

        // Частичное обновление феромонов на основе CPU результатов
        update_pheromones_partial(cpu_ant_path, cpu_antOF, iteration);
    }

    void process_gpu_results(const IterationData& gpu_data, int iteration) {
        // Обновляем данные на основе результатов GPU
        // Учитываем что GPU обработал только часть муравьев
        int gpu_start_index = cpu_ants_count.load();
        int gpu_ants_processed = std::min(gpu_data.ants_processed, ANT_SIZE - gpu_start_index);

        for (int i = 0; i < gpu_ants_processed; i++) {
            int global_index = gpu_start_index + i;
            if (global_index < ANT_SIZE) {
                antOF[global_index] = gpu_data.antOF[i];

                if (gpu_data.antOF[i] < global_minOf) global_minOf = gpu_data.antOF[i];
                if (gpu_data.antOF[i] > global_maxOf) global_maxOf = gpu_data.antOF[i];
            }
        }
    }

    void calculate_probabilities() {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            double sumVector = 0;
            double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

#ifdef _OPENMP
#pragma omp simd reduction(+:sumVector)
#endif
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                sumVector += current_pheromon[MAX_VALUE_SIZE * tx + i];
            }

#ifdef _OPENMP
#pragma omp simd
#endif
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                pheromon_norm[i] = current_pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
            }

            sumVector = 0;
            double svertka[MAX_VALUE_SIZE] = { 0 };

#ifdef _OPENMP
#pragma omp simd reduction(+:sumVector)
#endif
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                svertka[i] = probability_formula_non_cuda(pheromon_norm[i], current_kol_enter[MAX_VALUE_SIZE * tx + i]);
                sumVector += svertka[i];
            }

            norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
            }
        }
    }

    void update_pheromones_async(int iteration) {
        const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;

#ifdef _OPENMP
#pragma omp parallel for simd
#endif
        for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
            current_pheromon[idx] *= PARAMETR_RO;
        }

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::vector<double> local_pheromon_add(TOTAL_CELLS, 0.0);
            std::vector<int> local_kol_enter_add(TOTAL_CELLS, 0);

#ifdef _OPENMP
#pragma omp for nowait
#endif
            for (int i = 0; i < ANT_SIZE; ++i) {
                double agent_of = antOF[i];
#if OPTIMIZE_MIN_2
                double agent_of_reciprocal = (agent_of == 0) ? (PARAMETR_Q / 0.0000001) : (PARAMETR_Q / agent_of);
#endif

                const int* agent_path = &ant_parametr[i * PARAMETR_SIZE];
                for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                    int k = agent_path[tx];
                    int idx = MAX_VALUE_SIZE * tx + k;

                    local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_2
                    local_pheromon_add[idx] += agent_of_reciprocal;
#else
                    local_pheromon_add[idx] += PARAMETR_Q * agent_of;
#endif
                }
            }

#ifdef _OPENMP
#pragma omp critical
#endif
            {
#ifdef _OPENMP
#pragma omp simd
#endif
                for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
                    current_kol_enter[idx] += local_kol_enter_add[idx];
                    current_pheromon[idx] += local_pheromon_add[idx];
                }
            }
        }
    }

    void update_pheromones_partial(const std::vector<int>& ant_path,
        const std::vector<double>& antOF,
        int iteration) {
        // Упрощенная версия обновления феромонов только для CPU муравьев
        const int num_ants = antOF.size();
        const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;

        // Испарение феромонов
        for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
            current_pheromon[idx] *= PARAMETR_RO;
        }

        // Добавление новых феромонов
        for (int i = 0; i < num_ants; ++i) {
            double agent_of = antOF[i];
            const int* agent_path = &ant_path[i * PARAMETR_SIZE];

            for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                int k = agent_path[tx];
                int idx = MAX_VALUE_SIZE * tx + k;

                current_kol_enter[idx]++;
                current_pheromon[idx] += PARAMETR_Q * agent_of;
            }
        }
    }

    PerformanceMetrics calculate_average_metrics() {
        std::lock_guard<std::mutex> lock(metrics_mutex);

        PerformanceMetrics avg;
        int count = recent_metrics.size();

        for (const auto& metric : recent_metrics) {
            avg.cpu_execution_time += metric.cpu_execution_time;
            avg.cpu_ants_processed += metric.cpu_ants_processed;
            avg.cpu_throughput += metric.cpu_throughput;
        }

        if (count > 0) {
            avg.cpu_execution_time /= count;
            avg.cpu_ants_processed /= count;
            avg.cpu_throughput /= count;
        }

        // Очищаем историю после использования
        recent_metrics.clear();

        return avg;
    }
};

class HybridACO {
private:
    std::vector<double> current_pheromon;
    std::vector<double> current_kol_enter;
    std::vector<double> norm_matrix_probability;

    std::vector<int> ant_parametr;
    std::vector<double> antOF;

    std::atomic<bool> stop_requested{ false };
    std::atomic<int> current_iteration{ 0 };

    double minOf = 1e9;
    double maxOf = -1e9;

    // Переменные для хранения результатов CUDA
    double global_minOf = 1e9;
    double global_maxOf = -1e9;
    int kol_hash_fail = 0;

    double Time_CPU_all, Time_CPU_prob, Time_CPU_wait, Time_CPU_update, Time_GPU_all, Time_GPU, Time_GPU_function;

    // Очереди для межпоточного обмена
    ThreadSafeQueue gpu_to_cpu_queue;
    ThreadSafeQueue cpu_to_gpu_queue;

public:
    bool initialize(const std::vector<double>& parametr_value, const std::vector<double>& pheromon_value, const std::vector<double>& kol_enter_value) {
        // Инициализация данных
        current_pheromon = pheromon_value;
        current_kol_enter = kol_enter_value;
        norm_matrix_probability.resize(MAX_VALUE_SIZE * PARAMETR_SIZE);

        ant_parametr.resize(PARAMETR_SIZE * ANT_SIZE);
        antOF.resize(ANT_SIZE);

        // Инициализация переменных результатов
        global_minOf = 1e9;
        global_maxOf = -1e9;
        kol_hash_fail = 0;
#if (PRINT_INFORMATION)
        std::cout << "[Main] Initializing CUDA..." << std::endl;
#endif // PRINT_INFORMATION==1

        // Инициализация CUDA
        if (!cuda_initialize(parametr_value.data(), pheromon_value.data(), kol_enter_value.data())) {
            std::cerr << "[Main] CUDA initialization failed!" << std::endl;
            return false;
        }
#if (PRINT_INFORMATION)
        const char* version = cuda_get_version();
        std::cout << "[Main] " << version << std::endl;
        std::cout << "[Main] Hybrid ACO initialized successfully!" << std::endl;
        std::cout << "[Main] Parameters: " << PARAMETR_SIZE << " params, "
            << MAX_VALUE_SIZE << " values, " << ANT_SIZE << " ants" << std::endl;
#endif // PRINT_INFORMATION==1
        return true;
    }

    bool initialize_from_file(const std::string& filename) {
        const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
        std::vector<double> parametr_value(matrix_size);
        std::vector<double> pheromon_value(matrix_size);
        std::vector<double> kol_enter_value(matrix_size);

        if (!load_matrix(filename, parametr_value.data(), pheromon_value.data(), kol_enter_value.data())) {
            std::cerr << "[Main] Failed to load parameters from file: " << filename << std::endl;
            return false;
        }

        return initialize(parametr_value, pheromon_value, kol_enter_value);
    }

    void printHybridACOInfo(const std::string& prefix = "") const {
        std::cout << prefix << "HybridACO Info:" << std::endl;
        std::cout << prefix << "  Stop Requested: " << (stop_requested.load() ? "Yes" : "No") << std::endl;
        std::cout << prefix << "  Current Iteration: " << current_iteration.load() << std::endl;
        std::cout << prefix << "  MinOf: " << minOf << std::endl;
        std::cout << prefix << "  MaxOf: " << maxOf << std::endl;
        std::cout << prefix << "  Global MinOf: " << global_minOf << std::endl;
        std::cout << prefix << "  Global MaxOf: " << global_maxOf << std::endl;
        std::cout << prefix << "  Kol Hash Fail: " << kol_hash_fail << std::endl;

        std::cout << prefix << "  Current Pheromon Size: " << current_pheromon.size() << std::endl;
        std::cout << prefix << "  Current Kol Enter Size: " << current_kol_enter.size() << std::endl;
        std::cout << prefix << "  Norm Matrix Probability Size: " << norm_matrix_probability.size() << std::endl;
        std::cout << prefix << "  Ant Parametr Size: " << ant_parametr.size() << std::endl;
        std::cout << prefix << "  AntOF Size: " << antOF.size() << std::endl;

        // Время выполнения
        std::cout << prefix << "  Time CPU All: " << Time_CPU_all << " ms" << std::endl;
        std::cout << prefix << "  Time CPU Prob: " << Time_CPU_prob << " ms" << std::endl;
        std::cout << prefix << "  Time CPU Wait: " << Time_CPU_wait << " ms" << std::endl;
        std::cout << prefix << "  Time CPU Update: " << Time_CPU_update << " ms" << std::endl;
        std::cout << prefix << "  Time GPU All: " << Time_GPU_all << " ms" << std::endl;
        std::cout << prefix << "  Time GPU: " << Time_GPU << " ms" << std::endl;
        std::cout << prefix << "  Time GPU Function: " << Time_GPU_function << " ms" << std::endl;

        // Вывод первых нескольких значений для примера
        if (!current_pheromon.empty()) {
            std::cout << prefix << "  First 5 Pheromon Values: ";
            for (int i = 0; i < std::min(5, (int)current_pheromon.size()); ++i) {
                std::cout << current_pheromon[i] << " ";
            }
            std::cout << std::endl;
        }

        if (!antOF.empty()) {
            std::cout << prefix << "  First 5 AntOF Values: ";
            for (int i = 0; i < std::min(5, (int)antOF.size()); ++i) {
                std::cout << antOF[i] << " ";
            }
            std::cout << std::endl;
        }

        // Информация о очередях
        cpu_to_gpu_queue.printQueueInfo(prefix + "  CPU->GPU ");
        gpu_to_cpu_queue.printQueueInfo(prefix + "  GPU->CPU ");
    }

    // Поток GPU - вычисляет пути по текущей матрице
    void gpu_thread_function() {
#if (PRINT_INFORMATION)
        std::cout << "[GPU Thread] Started" << std::endl;
#endif // PRINT_INFORMATION==1
        auto start_time = std::chrono::high_resolution_clock::now();
        //Time_CPU

        while (!stop_requested.load()) {
            IterationData cpu_data;
            if (!cpu_to_gpu_queue.pop(cpu_data)) {
                break;
            }

            // Запуск вычислений на GPU с текущей матрицей вероятностей
            active_cuda_tasks.fetch_add(1);

            cuda_run_iteration(cpu_data.norm_matrix_probability.data(),
                ant_parametr.data(),
                antOF.data(),
                &minOf,
                &maxOf,
                &kol_hash_fail,
                &Time_GPU,
                &Time_GPU_function,
                cpu_data.iteration,
                cuda_completion_callback);

            // Отправляем результаты обратно в CPU поток
            IterationData gpu_data;
            gpu_data.iteration = cpu_data.iteration;
            gpu_data.ant_parametr = ant_parametr;
            gpu_data.antOF = antOF;
            gpu_data.minOf = minOf;
            gpu_data.maxOf = maxOf;
            gpu_data.kol_hash_fail = kol_hash_fail;

            gpu_to_cpu_queue.push(gpu_data);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        Time_GPU_all = std::chrono::duration<double, std::milli>(end_time - start_time).count();
#if (PRINT_INFORMATION)
        std::cout << "[GPU Thread] Finished" << std::endl;
#endif // PRINT_INFORMATION==1

    }

    // Поток CPU - обновляет феромоны и вычисляет новую матрицу
    void cpu_thread_function() {
#if (PRINT_INFORMATION)
        std::cout << "[CPU Thread] Started" << std::endl;
#endif // PRINT_INFORMATION==1
        auto end_time = std::chrono::high_resolution_clock::now();
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int iteration = 0; iteration < KOL_ITERATION && !stop_requested.load(); iteration++) {
            // Вычисляем матрицу вероятностей для текущего состояния
            auto start_time_prob = std::chrono::high_resolution_clock::now();
            calculate_probabilities();
            end_time = std::chrono::high_resolution_clock::now();
            Time_CPU_prob += std::chrono::duration<double, std::milli>(end_time - start_time_prob).count();

            // Отправляем данные в GPU поток
            IterationData cpu_data;
            cpu_data.iteration = iteration;
            cpu_data.norm_matrix_probability = norm_matrix_probability;
            auto start_time_wait = std::chrono::high_resolution_clock::now();
            cpu_to_gpu_queue.push(cpu_data);

            // Ждем результаты от GPU
            IterationData gpu_data;
            if (!gpu_to_cpu_queue.pop(gpu_data)) {
                break;
            }
            end_time = std::chrono::high_resolution_clock::now();
            Time_CPU_wait += std::chrono::duration<double, std::milli>(end_time - start_time_wait).count();

            // Обновляем данные на основе результатов GPU
            ant_parametr = std::move(gpu_data.ant_parametr);
            antOF = std::move(gpu_data.antOF);

            if (gpu_data.minOf < global_minOf) { global_minOf = gpu_data.minOf; }
            if (gpu_data.maxOf > global_maxOf) { global_maxOf = gpu_data.maxOf; }
            kol_hash_fail += gpu_data.kol_hash_fail;

            // Обновляем феромоны на основе результатов
            auto start_time_update = std::chrono::high_resolution_clock::now();
            update_pheromones_async(gpu_data.iteration);
            end_time = std::chrono::high_resolution_clock::now();
            Time_CPU_update += std::chrono::duration<double, std::milli>(end_time - start_time_update).count();
#if (PRINT_INFORMATION)
            if (gpu_data.iteration % KOL_PROGON_STATISTICS == 0) {
                std::cout << "[CPU Thread] Iteration " << gpu_data.iteration
                    << " completed. Min: " << global_minOf
                    << ", Max: " << global_maxOf << std::endl;
            }
#endif // PRINT_INFORMATION==1

        }

        // Сигнализируем о завершении
        cpu_to_gpu_queue.stop();
        gpu_to_cpu_queue.stop();
        end_time = std::chrono::high_resolution_clock::now();
        Time_CPU_all = std::chrono::duration<double, std::milli>(end_time - start_time).count();
#if (PRINT_INFORMATION)
        std::cout << "[CPU Thread] Finished" << std::endl;
#endif // PRINT_INFORMATION==1

    }

    void run_pipeline() {
        // Запускаем два потока
        std::thread cpu_thread(&HybridACO::cpu_thread_function, this);
        std::thread gpu_thread(&HybridACO::gpu_thread_function, this);

        // Ждем завершения потоков
        cpu_thread.join();
        gpu_thread.join();

        // Ждем завершения всех CUDA задач
        wait_completion();
    }

    void wait_completion(int max_wait_ms = 10000) {
        auto start = std::chrono::steady_clock::now();

        while (active_cuda_tasks.load() > 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);

            if (elapsed.count() > max_wait_ms) {
                std::cerr << "[Main] Timeout waiting for CUDA completion!" << std::endl;
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        cuda_synchronize();
    }

    void cleanup() {
        stop_requested.store(true);
        cpu_to_gpu_queue.stop();
        gpu_to_cpu_queue.stop();
        wait_completion();
        cuda_cleanup();
#if (PRINT_INFORMATION)
        std::cout << "[Main] Cleanup completed. Successful iterations: "
            << successful_iterations.load() << std::endl;
        std::cout << "[Main] Final results - Min: " << global_minOf
            << ", Max: " << global_maxOf
            << ", Hash fails: " << kol_hash_fail << std::endl;
#endif // PRINT_INFORMATION==1

    }

    int get_active_tasks() const { return active_cuda_tasks.load(); }
    int get_completed_iterations() const { return completed_iterations.load(); }

    double get_global_min() const { return global_minOf; }
    double get_global_max() const { return global_maxOf; }
    int get_hash_fails() const { return kol_hash_fail; }
    double get_time_cpu_all() const { return Time_CPU_all; }
    double get_time_cpu_prob() const { return Time_CPU_prob; }
    double get_time_cpu_wait() const { return Time_CPU_wait; }
    double get_time_cpu_update() const { return Time_CPU_update; }
    double get_time_gpu_all() const { return Time_GPU_all; }
    double get_time_gpu() const { return Time_GPU; }
    double get_time_gpu_function() const { return Time_GPU_function; }

private:
    void calculate_probabilities() {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            double sumVector = 0;
            double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

#ifdef _OPENMP
#pragma omp simd reduction(+:sumVector)
#endif
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                sumVector += current_pheromon[MAX_VALUE_SIZE * tx + i];
            }

#ifdef _OPENMP
#pragma omp simd
#endif
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                pheromon_norm[i] = current_pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
            }

            sumVector = 0;
            double svertka[MAX_VALUE_SIZE] = { 0 };

#ifdef _OPENMP
#pragma omp simd reduction(+:sumVector)
#endif
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                svertka[i] = probability_formula_non_cuda(pheromon_norm[i], current_kol_enter[MAX_VALUE_SIZE * tx + i]);
                sumVector += svertka[i];
            }

            norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
            }
        }
    }

    void update_pheromones_async(int iteration) {
        const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;

#ifdef _OPENMP
#pragma omp parallel for simd
#endif
        for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
            current_pheromon[idx] *= PARAMETR_RO;
        }

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::vector<double> local_pheromon_add(TOTAL_CELLS, 0.0);
            std::vector<int> local_kol_enter_add(TOTAL_CELLS, 0);

#ifdef _OPENMP
#pragma omp for nowait
#endif
            for (int i = 0; i < ANT_SIZE; ++i) {
                double agent_of = antOF[i];
#if OPTIMIZE_MIN_2
                double agent_of_reciprocal = (agent_of == 0) ? (PARAMETR_Q / 0.0000001) : (PARAMETR_Q / agent_of);
#endif

                const int* agent_path = &ant_parametr[i * PARAMETR_SIZE];
                for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                    int k = agent_path[tx];
                    int idx = MAX_VALUE_SIZE * tx + k;

                    local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_2
                    local_pheromon_add[idx] += agent_of_reciprocal;
#else
                    local_pheromon_add[idx] += PARAMETR_Q * agent_of;
#endif
                }
            }

#ifdef _OPENMP
#pragma omp critical
#endif
            {
#ifdef _OPENMP
#pragma omp simd
#endif
                for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
                    current_kol_enter[idx] += local_kol_enter_add[idx];
                    current_pheromon[idx] += local_pheromon_add[idx];
                }
            }
        }
    }
};

int start_hybrid() {
    auto start_time = std::chrono::high_resolution_clock::now();
    HybridACO aco;
    std::string filename = NAME_FILE_GRAPH;
    if (!aco.initialize_from_file(filename)) {
        std::cerr << "[Main] Trying to use default parameters..." << std::endl;

        const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
        std::vector<double> parametr_value(matrix_size);
        std::vector<double> pheromon_value(matrix_size, 1.0);
        std::vector<double> kol_enter_value(matrix_size, 1.0);

        for (int i = 0; i < PARAMETR_SIZE; i++) {
            for (int j = 0; j < MAX_VALUE_SIZE; j++) {
                parametr_value[i * MAX_VALUE_SIZE + j] = (i * MAX_VALUE_SIZE + j) * 0.01;
            }
        }

        if (!aco.initialize(parametr_value, pheromon_value, kol_enter_value)) {
            std::cerr << "Failed to initialize ACO!" << std::endl;
            return 1;
        }
    }

    auto start_time2 = std::chrono::high_resolution_clock::now();
    // Запускаем конвейерную обработку
    aco.run_pipeline();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    auto duration_iteration = std::chrono::duration<double, std::milli>(end_time - start_time2).count();
    // duration += std::chrono::duration<double, std::milli>(end_all - start).count();
#if (PRINT_INFORMATION)
    std::cout << "\n=== PIPELINE COMPLETED ===" << std::endl;
    std::cout << "Total time: " << duration << " ms" << std::endl;
    std::cout << "Completed iterations: " << aco.get_completed_iterations() << std::endl;
    std::cout << "Active tasks remaining: " << aco.get_active_tasks() << std::endl;
    std::cout << "Global Min: " << aco.get_global_min() << std::endl;
    std::cout << "Global Max: " << aco.get_global_max() << std::endl;
    std::cout << "Hash fails: " << aco.get_hash_fails() << std::endl;

#endif // PRINT_INFORMATION==1
    std::cout << "Time Hybrid OMP;" << duration << "; " << duration_iteration << "; " << aco.get_time_gpu_all() << "; " << aco.get_time_gpu() << "; " << aco.get_time_gpu_function() << "; " << aco.get_time_cpu_all() << "; " << aco.get_time_cpu_prob() << "; " << aco.get_time_cpu_wait() << "; " << aco.get_time_cpu_update() << "; " << aco.get_global_min() << "; " << aco.get_global_max() << "; " << aco.get_hash_fails() << "; " << std::endl;
    logFile   << "Time Hybrid OMP;" << duration << "; " << duration_iteration << "; " << aco.get_time_gpu_all() << "; " << aco.get_time_gpu() << "; " << aco.get_time_gpu_function() << "; " << aco.get_time_cpu_all() << "; " << aco.get_time_cpu_prob() << "; " << aco.get_time_cpu_wait() << "; " << aco.get_time_cpu_update() << "; " << aco.get_global_min() << "; " << aco.get_global_max() << "; " << aco.get_hash_fails() << "; " << std::endl;
    aco.cleanup();
    return 0;
}

int start_balanced_hybrid() {
    std::cout << "=== BALANCED HYBRID START ===" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    // Инициализация сбалансированной версии
    BalancedHybridACO balanced_aco;

    std::string filename = NAME_FILE_GRAPH;
    if (!balanced_aco.initialize_from_file(filename)) {
        std::cerr << "[Main] Trying to use default parameters..." << std::endl;

        const int matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;
        std::vector<double> parametr_value(matrix_size);
        std::vector<double> pheromon_value(matrix_size, 1.0);
        std::vector<double> kol_enter_value(matrix_size, 1.0);

        for (int i = 0; i < PARAMETR_SIZE; i++) {
            for (int j = 0; j < MAX_VALUE_SIZE; j++) {
                parametr_value[i * MAX_VALUE_SIZE + j] = (i * MAX_VALUE_SIZE + j) * 0.01;
            }
        }

        if (!balanced_aco.initialize(parametr_value, pheromon_value, kol_enter_value)) {
            std::cerr << "Failed to initialize ACO!" << std::endl;
            return 1;
        }
    }
    std::cout << "=== balanced_aco.run_balanced_pipeline() START ===" << std::endl;
    balanced_aco.run_balanced_pipeline();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    std::cout << "=== BALANCED HYBRID COMPLETED ===" << std::endl;
    std::cout << "Total time: " << duration << " ms" << std::endl;
    std::cout << "Final CPU ratio: " << balanced_aco.get_current_cpu_ratio() << std::endl;
}

int main() {

    std::cout << __cplusplus << std::endl;
    logFile.open("log.txt");
    if (!logFile.is_open()) {
        std::cerr << "Ошибка открытия лог-файла!" << std::endl;
        return 1; // Возврат с ошибкой
    }
    std::cout << "Max threads OMP : " << omp_get_max_threads() << " ";
    std::cout << "OpenMP version: " << _OPENMP << " :";
#if _OPENMP >= 202411 
    std::cout << "OpenMP 6.0 (2026) plane" << std::endl;
#elif _OPENMP >= 202111 
    std::cout << "OpenMP 5.2 (2023) active" << std::endl;
#elif _OPENMP >= 202011 
    std::cout << "OpenMP 5.1 (2021) active" << std::endl;
#elif _OPENMP >= 201811 
    std::cout << "OpenMP 5.0 (2018) active" << std::endl;
#elif _OPENMP >= 201511 
    std::cout << "OpenMP 4.5 (2015) optimal" << std::endl;
#elif _OPENMP >= 201307 
    std::cout << "OpenMP 4.0 (2013) active" << std::endl;
#elif _OPENMP >= 201107 
    std::cout << "OpenMP 3.1 (2011) supported" << std::endl;
#elif _OPENMP >= 200805 
    std::cout << "OpenMP 3.0 (2008) supported" << std::endl;
#elif _OPENMP >= 200505 
    std::cout << "OpenMP 2.5 (2005) outdated" << std::endl;
#elif _OPENMP >= 200203 
    std::cout << "OpenMP 2.0 (2002) outdated" << std::endl;
#elif _OPENMP >= 199710 
    std::cout << "OpenMP 1.0 (1999) outdated" << std::endl;
#else 
    std::cout << "Older OpenMP version" << std::endl;
#endif
    logFile << "Max threads OMP : " << omp_get_max_threads() << " ";
    logFile << "OpenMP version: " << _OPENMP << " :";
#if _OPENMP >= 202611 
    logFile << "OpenMP 6.0 (2026) plane" << std::endl;
#elif _OPENMP >= 202311 
    logFile << "OpenMP 5.2 (2023) active" << std::endl;
#elif _OPENMP >= 202111 
    logFile << "OpenMP 5.1 (2021) active" << std::endl;
#elif _OPENMP >= 201811 
    logFile << "OpenMP 5.0 (2018) active" << std::endl;
#elif _OPENMP >= 201511 
    logFile << "OpenMP 4.5 (2015) optimal" << std::endl;
#elif _OPENMP >= 201307 
    logFile << "OpenMP 4.0 (2013) active" << std::endl;
#elif _OPENMP >= 201107 
    logFile << "OpenMP 3.1 (2011) supported" << std::endl;
#elif _OPENMP >= 200805 
    logFile << "OpenMP 3.0 (2008) supported" << std::endl;
#elif _OPENMP >= 200505 
    logFile << "OpenMP 2.5 (2005) outdated" << std::endl;
#elif _OPENMP >= 200203 
    logFile << "OpenMP 2.0 (2002) outdated" << std::endl;
#elif _OPENMP >= 199910 
    logFile << "OpenMP 1.0 (1999) outdated" << std::endl;
#else 
    logFile << "Older OpenMP version" << std::endl;
#endif
    std::cout << "PARAMETR_SIZE: " << PARAMETR_SIZE << "; "
        << "MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << "; "
        << "ANT_SIZE: " << ANT_SIZE << "; "
        << "NAME_FILE_GRAPH: " << NAME_FILE_GRAPH << "; "
        << "KOL_ITERATION: " << KOL_ITERATION << "; "
        << "KOL_PROGON_STATISTICS: " << KOL_PROGON_STATISTICS << "; "
        << "PARAMETR_Q: " << PARAMETR_Q << "; "
        << "PARAMETR_RO: " << PARAMETR_RO << "; "
        << "MAX_PARAMETR_VALUE_TO_MIN_OPT: " << MAX_PARAMETR_VALUE_TO_MIN_OPT << ", " << "MAX_THREAD_CUDA: " << MAX_THREAD_CUDA << ", " 
        << "OPTIMIZE: " << (OPTIMIZE_MIN_1 ? "OPTIMIZE_MIN_1 " : "") << (OPTIMIZE_MIN_2 ? "OPTIMIZE_MIN_2 " : "") << (OPTIMIZE_MAX ? "OPTIMIZE_MAX " : "") << "; "
        << std::endl;
    logFile << "PARAMETR_SIZE: " << PARAMETR_SIZE << "; "
        << "MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << "; "
        << "NAME_FILE_GRAPH: " << NAME_FILE_GRAPH << "; "
        << "ANT_SIZE: " << ANT_SIZE << "; "
        << "KOL_ITERATION: " << KOL_ITERATION << "; "
        << "KOL_PROGON_STATISTICS: " << KOL_PROGON_STATISTICS << "; "
        << "PARAMETR_Q: " << PARAMETR_Q << "; "
        << "PARAMETR_RO: " << PARAMETR_RO << "; "
        << "MAX_PARAMETR_VALUE_TO_MIN_OPT: " << MAX_PARAMETR_VALUE_TO_MIN_OPT << ", "  << "MAX_THREAD_CUDA: " << MAX_THREAD_CUDA << ", " 
        << "OPTIMIZE: " << (OPTIMIZE_MIN_1 ? "OPTIMIZE_MIN_1 " : "") << (OPTIMIZE_MIN_2 ? "OPTIMIZE_MIN_2 " : "") << (OPTIMIZE_MAX ? "OPTIMIZE_MAX " : "") << "; "
        << std::endl;

    std::cout << "=== CUDA + OpenMP Hybrid Pipeline Test ===" << std::endl;
    if (GO_HYBRID_OMP) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_hybrid();
            j = j + 1;
        }
        // Запуск таймера
        //clear_all_stat();
        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_hybrid();
            i = i + 1;
        }
        // Остановка таймера
        auto end2 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time Hybrid OMP:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        logFile.close();
    }
    if (GO_HYBRID_BALANCED_OMP) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_balanced_hybrid();
            j = j + 1;
        }
        // Запуск таймера
        //clear_all_stat();
        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_balanced_hybrid();
            i = i + 1;
        }
        // Остановка таймера
        auto end2 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time Balanced Hybrid OMP:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        logFile.close();
    }
    return 0;
}