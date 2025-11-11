#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "ant_colony_common.h"

// Глобальные переменные для OpenMP состояния
static std::vector<double> current_pheromon;
static std::vector<double> current_kol_enter;
static std::vector<double> norm_matrix_probability;
static std::atomic<bool> data_ready{ false };
static std::atomic<bool> stop_requested{ false };

// Функции OpenMP обработки
void omp_initialize(const double* initial_pheromon, const double* initial_kol_enter) {
    size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;

    current_pheromon.resize(matrix_size);
    current_kol_enter.resize(matrix_size);
    norm_matrix_probability.resize(matrix_size);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < matrix_size; i++) {
        current_pheromon[i] = initial_pheromon[i];
        current_kol_enter[i] = initial_kol_enter[i];
    }

#ifdef _OPENMP
    omp_set_num_threads(omp_get_max_threads());
    std::cout << "OpenMP initialized with " << omp_get_max_threads() << " threads" << std::endl;
#else
    std::cout << "OpenMP not available, running sequentially" << std::endl;
#endif

    data_ready.store(false);
    stop_requested.store(false);
}

void omp_calculate_probabilities() {
    if (stop_requested.load()) return;

    const size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        // Упрощенная версия без AVX для совместимости
        double sumPheromon = 0.0;
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumPheromon += current_pheromon[MAX_VALUE_SIZE * tx + i];
        }

        double svertkaArray[MAX_VALUE_SIZE] = { 0 };
        double sumSvertka = 0.0;

        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            if (current_kol_enter[MAX_VALUE_SIZE * tx + i] != 0 && sumPheromon != 0) {
                double pheromonNorm = current_pheromon[MAX_VALUE_SIZE * tx + i] / sumPheromon;
                svertkaArray[i] = (1.0 / current_kol_enter[MAX_VALUE_SIZE * tx + i]) + pheromonNorm;
                sumSvertka += svertkaArray[i];
            }
        }

        if (sumSvertka > 0) {
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                cumulative += svertkaArray[i] / sumSvertka;
                norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = cumulative;
            }
        }
        else {
            // Равномерное распределение если все нули
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (i + 1.0) / MAX_VALUE_SIZE;
            }
        }
    }

    data_ready.store(true);
}

void omp_update_pheromones(const int* ant_parametr, const double* antOF) {
    if (stop_requested.load()) return;

    // Ждем пока данные будут готовы для обновления
    while (!data_ready.load() && !stop_requested.load()) {
        std::this_thread::yield();
    }

    if (stop_requested.load()) return;

    const size_t matrix_size = MAX_VALUE_SIZE * PARAMETR_SIZE;

    // Испарение феромонов
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < matrix_size; i++) {
        current_pheromon[i] *= PARAMETR_RO;
    }

    // Добавление нового феромона
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        for (int i = 0; i < ANT_SIZE; i++) {
            int k = ant_parametr[i * PARAMETR_SIZE + tx];
            if (k >= 0 && k < MAX_VALUE_SIZE) {
#ifdef _OPENMP
#pragma omp atomic
#endif
                current_kol_enter[MAX_VALUE_SIZE * tx + k]++;

#ifdef OPTIMIZE_MAX
#ifdef _OPENMP
#pragma omp atomic
#endif
                current_pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * antOF[i];
#endif
            }
        }
    }

    data_ready.store(false);
}

const double* omp_get_norm_matrix_probability() {
    return norm_matrix_probability.data();
}

bool omp_is_data_ready() {
    return data_ready.load();
}

void omp_stop() {
    stop_requested.store(true);
}

void omp_cleanup() {
    omp_stop();
    current_pheromon.clear();
    current_kol_enter.clear();
    norm_matrix_probability.clear();
}