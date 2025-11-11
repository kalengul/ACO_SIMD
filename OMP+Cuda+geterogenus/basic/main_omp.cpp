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
#include "cuda_module.h"

// Добавляем константы напрямую
#define MAX_VALUE_SIZE 4
#define PARAMETR_SIZE 42
#define ANT_SIZE 500
#define MAX_THREAD_CUDA 256
#define NAME_FILE_GRAPH "Parametr_Graph/test42_4.txt"
#define KOL_ITERATION 500
#define KOL_PROGON_STATISTICS 50

#define PARAMETR_RO 0.99
#define PARAMETR_Q 1.0

// Оптимизационные флаги
#define OPTIMIZE_MIN_1 0
#define OPTIMIZE_MIN_2 1
#define OPTIMIZE_MAX 0
#define MAX_PARAMETR_VALUE_TO_MIN_OPT 1000.0

#ifdef _OPENMP
#include <omp.h>
#endif

// Структура для передачи данных между потоками
struct IterationData {
    int iteration;
    std::vector<double> norm_matrix_probability;
    std::vector<int> ant_parametr;
    std::vector<double> antOF;
    double minOf;
    double maxOf;
    int kol_hash_fail;
};

class ThreadSafeQueue {
private:
    std::queue<IterationData> queue;
    mutable std::mutex mtx; // Убрали const
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
};

// Глобальные переменные для синхронизации
std::atomic<int> active_cuda_tasks{ 0 };
std::atomic<int> completed_iterations{ 0 };
std::atomic<int> successful_iterations{ 0 };

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

    if (iteration % KOL_PROGON_STATISTICS == 0) {
        std::cout << "[Callback] Iteration " << iteration << " (Total: " << iter_num
            << "): Min=" << min_val << ", Max=" << max_val
            << ", Avg=" << (sum / size) << std::endl;
    }

    active_cuda_tasks.fetch_sub(1);
}

// Функция загрузки параметров из файла
bool load_matrix(const std::string& filename, double* parametr_value, double* pheromon_value, double* kol_enter_value) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Don't open file!" << std::endl;
        return false;
    }

    std::cout << "[Main] Loading parameters from file: " << filename << std::endl;

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

    std::cout << "[Main] Parameters loaded successfully!" << std::endl;
    return true;
}

// Функция для вычисления вероятностной формулы
inline double probability_formula_non_cuda(double pheromon, double kol_enter) {
    return (kol_enter != 0.0 && pheromon != 0.0) ? (1.0 / kol_enter + pheromon) : 0.0;
}

class HybridACO {
private:
    std::vector<double> current_pheromon;
    std::vector<double> current_kol_enter;
    std::vector<double> norm_matrix_probability;

    std::vector<int> ant_parametr;
    std::vector<double> antOF;

    std::atomic<bool> stop_requested{ false };
    std::atomic<int> current_iteration{ 0 };

    double minOf = 0.0;
    double maxOf = 0.0;

    // Переменные для хранения результатов CUDA
    double global_minOf = 1e9;
    double global_maxOf = -1e9;
    int kol_hash_fail = 0;

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

        std::cout << "[Main] Initializing CUDA..." << std::endl;

        // Инициализация CUDA
        if (!cuda_initialize(parametr_value.data(), pheromon_value.data(), kol_enter_value.data())) {
            std::cerr << "[Main] CUDA initialization failed!" << std::endl;
            return false;
        }

        const char* version = cuda_get_version();
        std::cout << "[Main] " << version << std::endl;

        std::cout << "[Main] Hybrid ACO initialized successfully!" << std::endl;
        std::cout << "[Main] Parameters: " << PARAMETR_SIZE << " params, "
            << MAX_VALUE_SIZE << " values, " << ANT_SIZE << " ants" << std::endl;

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

    // Поток GPU - вычисляет пути по текущей матрице
    void gpu_thread_function() {
        std::cout << "[GPU Thread] Started" << std::endl;

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

        std::cout << "[GPU Thread] Finished" << std::endl;
    }

    // Поток CPU - обновляет феромоны и вычисляет новую матрицу
    void cpu_thread_function() {
        std::cout << "[CPU Thread] Started" << std::endl;

        for (int iteration = 0; iteration < KOL_ITERATION && !stop_requested.load(); iteration++) {
            // Вычисляем матрицу вероятностей для текущего состояния
            calculate_probabilities();

            // Отправляем данные в GPU поток
            IterationData cpu_data;
            cpu_data.iteration = iteration;
            cpu_data.norm_matrix_probability = norm_matrix_probability;
            cpu_to_gpu_queue.push(cpu_data);

            // Ждем результаты от GPU
            IterationData gpu_data;
            if (!gpu_to_cpu_queue.pop(gpu_data)) {
                break;
            }

            // Обновляем данные на основе результатов GPU
            ant_parametr = std::move(gpu_data.ant_parametr);
            antOF = std::move(gpu_data.antOF);

            if (gpu_data.minOf < global_minOf) { global_minOf = gpu_data.minOf; }
            if (gpu_data.maxOf > global_maxOf) { global_maxOf = gpu_data.maxOf; }
            kol_hash_fail += gpu_data.kol_hash_fail;

            // Обновляем феромоны на основе результатов
            update_pheromones_async(gpu_data.iteration);

            if (gpu_data.iteration % KOL_PROGON_STATISTICS == 0) {
                std::cout << "[CPU Thread] Iteration " << gpu_data.iteration
                    << " completed. Min: " << global_minOf
                    << ", Max: " << global_maxOf << std::endl;
            }
        }

        // Сигнализируем о завершении
        cpu_to_gpu_queue.stop();
        gpu_to_cpu_queue.stop();

        std::cout << "[CPU Thread] Finished" << std::endl;
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

        std::cout << "[Main] Cleanup completed. Successful iterations: "
            << successful_iterations.load() << std::endl;
        std::cout << "[Main] Final results - Min: " << global_minOf
            << ", Max: " << global_maxOf
            << ", Hash fails: " << kol_hash_fail << std::endl;
    }

    int get_active_tasks() const { return active_cuda_tasks.load(); }
    int get_completed_iterations() const { return completed_iterations.load(); }

    double get_global_min() const { return global_minOf; }
    double get_global_max() const { return global_maxOf; }
    int get_hash_fails() const { return kol_hash_fail; }

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

int main() {
    std::cout << "=== CUDA + OpenMP Hybrid Pipeline Test ===" << std::endl;

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

    auto start_time = std::chrono::high_resolution_clock::now();

    // Запускаем конвейерную обработку
    aco.run_pipeline();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\n=== PIPELINE COMPLETED ===" << std::endl;
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;
    std::cout << "Completed iterations: " << aco.get_completed_iterations() << std::endl;
    std::cout << "Active tasks remaining: " << aco.get_active_tasks() << std::endl;
    std::cout << "Global Min: " << aco.get_global_min() << std::endl;
    std::cout << "Global Max: " << aco.get_global_max() << std::endl;
    std::cout << "Hash fails: " << aco.get_hash_fails() << std::endl;

    aco.cleanup();

    std::cout << "\nPress Enter to exit..." << std::endl;
    std::cin.get();

    return 0;
}