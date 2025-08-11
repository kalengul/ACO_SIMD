#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <limits.h>
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

#define CUDA_CHECK(call)  {cudaError_t err = call;if (err != cudaSuccess) {std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; exit(EXIT_FAILURE); } }

std::ofstream logFile; // Глобальная переменная для лог-файла
std::ofstream outfile("statistics.txt"); // Глобальная переменная для файла статистики
std::mutex mtx; // Мьютекс для защиты доступа к общим данным

//Структура для сбора статистики 
class Statistics {
public:
    double sum;
    double sum_sq;
    int count;

    Statistics() : sum(0), sum_sq(0), count(0) {}
    void updateStatistics(double value) {
        sum += value;
        sum_sq += value * value;
        count++;
    }
    double mean() const {
        return count > 0 ? sum / count : 0.0;
    }
    double variance() const {
        if (count > 1) {
            double mean_val = mean();
            return (sum_sq / count) - (mean_val * mean_val);
        }
        return 0.0;
    }
    void clear() {
        sum_sq = 0;
        sum = 0;
        count = 0;
    }
};

//Глобальные переменные для сбора статистики
std::vector<Statistics> stat_duration(KOL_STAT_LEVEL);
std::vector<Statistics> stat_duration_iteration(KOL_STAT_LEVEL);
std::vector<Statistics> stat_SumgpuTime1(KOL_STAT_LEVEL);
std::vector<Statistics> stat_SumgpuTime2(KOL_STAT_LEVEL);
std::vector<Statistics> stat_SumgpuTime3(KOL_STAT_LEVEL);
std::vector<Statistics> stat_SumgpuTime4(KOL_STAT_LEVEL);
std::vector<Statistics> stat_SumgpuTime5(KOL_STAT_LEVEL);
std::vector<Statistics> stat_SumgpuTime6(KOL_STAT_LEVEL);
std::vector<Statistics> stat_SumgpuTime7(KOL_STAT_LEVEL);
std::vector<Statistics> stat_SumgpuTime8(KOL_STAT_LEVEL);
std::vector<Statistics> stat_global_minOf(KOL_STAT_LEVEL);
std::vector<Statistics> stat_global_maxOf(KOL_STAT_LEVEL);
std::vector<Statistics> stat_kol_hash_fail(KOL_STAT_LEVEL);

#if (MAX_VALUE_SIZE * PARAMETR_SIZE < MAX_CONST)
__constant__ double parametr_value_dev_const[MAX_VALUE_SIZE * PARAMETR_SIZE]; // на 2688 выдает ошибку по максимуму константной памяти
#endif
#if (MAX_VALUE_SIZE * PARAMETR_SIZE >= MAX_CONST)
__constant__ double parametr_value_dev_const[10 * 10]; // на 2688 выдает ошибку по максимуму константной памяти
#endif
__constant__ int gpuTime_const; // Объявление константной памяти для gpuTime

// Функция для цвычисления параметра х при  параметрическом графе
__device__ double go_x(double* parametr, int start_index, int kol_parametr) {
    double sum = 0.0;
    for (int i = 1; i < kol_parametr; ++i) {
        sum += parametr[start_index + i];
    }
    return parametr[start_index] * sum; // Умножаем на первый параметр в диапазоне
}

#if (SHAFFERA) 
// Функция для целевой функции Шаффера
__device__ double BenchShafferaFunction(double* parametr) {
    double r_squared = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        r_squared += x * x; // Сумма квадратов
    }
    double r = sqrt(r_squared);
    double sin_r = sin(r);
    return 1.0 / 2.0 - (sin_r * sin_r - 0.5) / (1.0 + 0.001 * r_squared);
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
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
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
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
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
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
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
// Сферическая функция
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
// Гриванк-функция
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
// Захаров-функция
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
// Швейфель-функция
__device__ double BenchShafferaFunction(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum -= x * sin(sqrt(abs(x)));
    }
    return sum;
}
#endif
#if (LEVY)
// Леви-функция
__device__ double BenchShafferaFunction(double* parametr) {
    double w_first = 1 + (go_x(parametr, 0, PARAMETR_SIZE_ONE_X) - 1) / 4;
    double w_last = 1 + (go_x(parametr, PARAMETR_SIZE - PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X) - 1) / 4;
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 1; i <= num_variables - 1; ++i) {
        double wi = 1 + (go_x(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X) - 1) / 4;
        sum += pow(wi - 1, 2) * (1 + 10 * pow(sin(M_PI * wi), 2)) +
            pow(wi - wi * w_i_prev, 2) * (1 + pow(sin(2 * M_PI * wi), 2));
    }
    return pow(sin(M_PI * w_first), 2) + sum + pow(w_last - 1, 2) * (1 + pow(sin(2 * M_PI * w_last), 2));
}
#endif
#if (MICHAELWICZYNSKI)
// Михаэлевич-Викинский
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

// Функция для вычисления параметра x при параметрическом графе
double go_x_non_cuda_omp(double* parametr, int start_index, int kol_parametr) {
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (int i = 1; i < kol_parametr; ++i) {
        sum += parametr[start_index + i];
    }
    return parametr[start_index] * sum; // Умножаем на первый параметр в диапазоне
}
// Функция для цвычисления параметра х при  параметрическом графе
double go_x_non_cuda(double* parametr, int start_index, int kol_parametr) {
    double sum = 0.0;
    for (int i = 1; i < kol_parametr; ++i) {
        sum += parametr[start_index + i];
    }
    return parametr[start_index] * sum; // Умножаем на первый параметр в диапазоне
}

#if (SHAFFERA) 
// Функция для целевой функции Шаффера с 100 переменными
double BenchShafferaFunction_omp(double* parametr) {
    double r_squared = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;

#pragma omp parallel for reduction(+:r_squared)
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        r_squared += x * x; // Сумма квадратов
    }

    double r = sqrt(r_squared);
    double sin_r = sin(r);
    return 1.0 / 2.0 - (sin_r * sin_r - 0.5) / (1.0 + 0.001 * r_squared);
}
#endif
#if (DELT4)
// Михаэлевич-Викинский
double BenchShafferaFunction_omp(double* parametr) {
    double r_squared = 0.0;
    double sum_if = 0.0;
    double sum = 0.0;
    double second_sum = 0.0;
    double r_cos = 1.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
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
double BenchShafferaFunction_omp(double* parametr) {
    double r_cos = 1.0;
    double r_squared = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:r_squared, r_cos)
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
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
double BenchShafferaFunction_omp(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum += x * x - 10 * cos(2 * M_PI * x) + 10;
    }
    return sum;
}
#endif
#if (ACKLEY)
// Акли-функция
double BenchShafferaFunction_omp(double* parametr) {
    double first_sum = 0.0;
    double second_sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:first_sum, second_sum)
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
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
double BenchShafferaFunction_omp(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum += x * x;
    }
    return sum;
}
#endif
#if (GRIEWANK)
// Гриванк-функция
double BenchShafferaFunction_omp(double* parametr) {
    double sum = 0.0;
    double prod = 1.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:sum, prod)
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum += x * x;
        prod *= cos(x / sqrt(i + 1));
    }
    return sum / 4000 - prod + 1;
}
#endif
#if (ZAKHAROV)
// Захаров-функция
double BenchShafferaFunction_omp(double* parametr) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:sum1, sum2)
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum1 += pow(x, 2);
        sum2 += 0.5 * i * x;
    }
    return sum1 + pow(sum2, 2) + pow(sum2, 4);
}
#endif
#if (SCHWEFEL)
// Швейфель-функция
double BenchShafferaFunction_omp(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum -= x * sin(sqrt(abs(x)));
    }
    return sum;
}
#endif
#if (LEVY)
// Леви-функция
double BenchShafferaFunction_omp(double* parametr) {
    double w_first = 1 + (go_x_non_cuda_omp(parametr, 0, PARAMETR_SIZE_ONE_X) - 1) / 4;
    double w_last = 1 + (go_x_non_cuda_omp(parametr, PARAMETR_SIZE - PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X) - 1) / 4;
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:sum)
    for (int i = 1; i <= num_variables - 1; ++i) {
        double wi = 1 + (go_x_non_cuda_omp(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X) - 1) / 4;
        sum += pow(wi - 1, 2) * (1 + 10 * pow(sin(M_PI * wi), 2)) +
            pow(wi - wi * w_i_prev, 2) * (1 + pow(sin(2 * M_PI * wi), 2));
    }
    return pow(sin(M_PI * w_first), 2) + sum + pow(w_last - 1, 2) * (1 + pow(sin(2 * M_PI * w_last), 2));
}
#endif
#if (MICHAELWICZYNSKI)
// Михаэлевич-Викинский
double BenchShafferaFunction_omp(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda_omp(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum -= sin(x) * pow(sin((i + 1) * x * x / M_PI), 20);
    }
    return sum;
}
#endif


// Функция для non_CUDA
#if (SHAFFERA) 
double BenchShafferaFunction_non_cuda(double* parametr) {
    double r_squared = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        r_squared += x * x; // Сумма квадратов
    }
    double r = sqrt(r_squared);
    double sin_r = sin(r);
    return 1.0 / 2.0 - (sin_r * sin_r - 0.5) / (1.0 + 0.001 * r_squared);
}
#endif
#if (DELT4)
// Михаэлевич-Викинский
double BenchShafferaFunction_non_cuda(double* parametr) {
    double r_squared = 0.0;
    double sum_if = 0.0;
    double sum = 0.0;
    double second_sum = 0.0;
    double r_cos = 1.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
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
double BenchShafferaFunction_non_cuda(double* parametr) {
    double r_cos = 1.0;
    double r_squared = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
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
double BenchShafferaFunction_non_cuda(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum += x * x - 10 * cos(2 * M_PI * x) + 10;
    }
    return sum;
}
#endif
#if (ACKLEY)
// Акли-функция
double BenchShafferaFunction_non_cuda(double* parametr) {
    double first_sum = 0.0;
    double second_sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
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
double BenchShafferaFunction_non_cuda(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum += x * x;
    }
    return sum;
}
#endif
#if (GRIEWANK)
// Гриванк-функция
double BenchShafferaFunction_non_cuda(double* parametr) {
    double sum = 0.0;
    double prod = 1.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum += x * x;
        prod *= cos(x / sqrt(i + 1));
    }
    return sum / 4000 - prod + 1;
}
#endif
#if (ZAKHAROV)
// Захаров-функция
double BenchShafferaFunction_non_cuda(double* parametr) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum1 += pow(x, 2);
        sum2 += 0.5 * i * x;
    }
    return sum1 + pow(sum2, 2) + pow(sum2, 4);
}
#endif
#if (SCHWEFEL)
// Швейфель-функция
double BenchShafferaFunction_non_cuda(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum -= x * sin(sqrt(abs(x)));
    }
    return sum;
}
#endif
#if (LEVY)
// Леви-функция
double BenchShafferaFunction_non_cuda(double* parametr) {
    double w_first = 1 + (go_x_non_cuda(parametr, 0, PARAMETR_SIZE_ONE_X) - 1) / 4;
    double w_last = 1 + (go_x_non_cuda(parametr, PARAMETR_SIZE - PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X) - 1) / 4;
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 1; i <= num_variables - 1; ++i) {
        double wi = 1 + (go_x_non_cuda(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X) - 1) / 4;
        sum += pow(wi - 1, 2) * (1 + 10 * pow(sin(M_PI * wi), 2)) +
            pow(wi - wi * w_i_prev, 2) * (1 + pow(sin(2 * M_PI * wi), 2));
    }
    return pow(sin(M_PI * w_first), 2) + sum + pow(w_last - 1, 2) * (1 + pow(sin(2 * M_PI * w_last), 2));
}
#endif
#if (MICHAELWICZYNSKI)
// Михаэлевич-Викинский
double BenchShafferaFunction_non_cuda(double* parametr) {
    double sum = 0.0;
    int num_variables = PARAMETR_SIZE / PARAMETR_SIZE_ONE_X;
    for (int i = 0; i < num_variables; ++i) {
        double x = go_x_non_cuda(parametr, i * PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
        sum -= sin(x) * pow(sin((i + 1) * x * x / M_PI), 20);
    }
    return sum;
}
#endif

void update_all_Stat(int i, double duration, double duration_iteration, double gpuTime1, double gpuTime2, double gpuTime3, double gpuTime4, double gpuTime5, double gpuTime6, double gpuTime7, double gpuTime8, double minOf, double maxOf, int hash_fail_count) {

    stat_duration[i].updateStatistics(duration);
    stat_duration_iteration[i].updateStatistics(duration_iteration);
    stat_SumgpuTime1[i].updateStatistics(gpuTime1);
    stat_SumgpuTime2[i].updateStatistics(gpuTime2);
    stat_SumgpuTime3[i].updateStatistics(gpuTime3);
    stat_SumgpuTime4[i].updateStatistics(gpuTime4);
    stat_SumgpuTime5[i].updateStatistics(gpuTime5);
    stat_SumgpuTime6[i].updateStatistics(gpuTime6);
    stat_SumgpuTime7[i].updateStatistics(gpuTime7);
    stat_SumgpuTime8[i].updateStatistics(gpuTime8);
    stat_global_minOf[i].updateStatistics(minOf);
    stat_global_maxOf[i].updateStatistics(maxOf);
    stat_kol_hash_fail[i].updateStatistics(double(hash_fail_count));
    if (PRINT_INFORMATION) {
        // Вывод информации на экран
        std::cout << "Updating statistics for index: " << i << " " << stat_SumgpuTime1.size() << std::endl;
        std::cout << "Duration: " << stat_duration[i].sum << std::endl;
        std::cout << "Duration per iteration: " << stat_duration_iteration[i].sum << std::endl;
        std::cout << "GPU Time 1: " << stat_SumgpuTime1[i].sum << std::endl;
        std::cout << "GPU Time 2: " << stat_SumgpuTime2[i].sum << std::endl;
        std::cout << "GPU Time 3: " << stat_SumgpuTime3[i].sum << std::endl;
        std::cout << "GPU Time 4: " << stat_SumgpuTime4[i].sum << std::endl;
        std::cout << "GPU Time 5: " << stat_SumgpuTime5[i].sum << std::endl;
        std::cout << "GPU Time 6: " << stat_SumgpuTime6[i].sum << std::endl;
        std::cout << "GPU Time 7: " << stat_SumgpuTime7[i].sum << std::endl;
        std::cout << "GPU Time 8: " << stat_SumgpuTime8[i].sum << std::endl;
        std::cout << "Min of: " << stat_global_minOf[i].sum << std::endl;
        std::cout << "Max of: " << stat_global_maxOf[i].sum << std::endl;
        std::cout << "Hash fail count: " << stat_kol_hash_fail[i].sum << std::endl;
    }
}

static void clear_all_stat() {
    for (int i = 0; i < stat_duration.size(); i++)
    {
        if (PRINT_INFORMATION) { std::cout << "Clear statistics for index: " << i << " " << stat_duration.size() << std::endl; }
        stat_duration[i].clear();
        stat_duration_iteration[i].clear();
        stat_SumgpuTime1[i].clear();
        stat_SumgpuTime2[i].clear();
        stat_SumgpuTime3[i].clear();
        stat_SumgpuTime4[i].clear();
        stat_SumgpuTime5[i].clear();
        stat_SumgpuTime6[i].clear();
        stat_SumgpuTime7[i].clear();
        stat_SumgpuTime8[i].clear();
        stat_global_minOf[i].clear();
        stat_global_maxOf[i].clear();
        stat_kol_hash_fail[i].clear();
    }

}

void save_all_stat_text_file(const std::string& name_model) {

    outfile << std::fixed << std::setprecision(6); // Устанавливаем формат вывода

    for (int i = 0; i < KOL_STAT_LEVEL; ++i) {
        outfile << name_model << "; " << (i + 1) << "; " << int(KOL_ITERATION / KOL_STAT_LEVEL * (i + 1)) << "; ";
        outfile << "Duration Mean:; " << stat_duration[i].mean() << "; " << stat_duration[i].variance() << "; ";
        outfile << "GPU Time 1 Mean:; " << stat_SumgpuTime1[i].mean() << "; " << stat_SumgpuTime1[i].variance() << "; ";
        outfile << "GPU Time 2 Mean:; " << stat_SumgpuTime2[i].mean() << "; " << stat_SumgpuTime2[i].variance() << "; ";
        outfile << "GPU Time 3 Mean:; " << stat_SumgpuTime3[i].mean() << "; " << stat_SumgpuTime3[i].variance() << "; ";
        outfile << "GPU Time 4 Mean:; " << stat_SumgpuTime4[i].mean() << "; " << stat_SumgpuTime4[i].variance() << "; ";
        outfile << "GPU Time 5 Mean:; " << stat_SumgpuTime5[i].mean() << "; " << stat_SumgpuTime5[i].variance() << "; ";
        outfile << "GPU Time 6 Mean:; " << stat_SumgpuTime6[i].mean() << "; " << stat_SumgpuTime6[i].variance() << "; ";
        outfile << "GPU Time 7 Mean:; " << stat_SumgpuTime7[i].mean() << "; " << stat_SumgpuTime7[i].variance() << "; ";
        outfile << "GPU Time 8 Mean:; " << stat_SumgpuTime8[i].mean() << "; " << stat_SumgpuTime8[i].variance() << "; ";
        outfile << "Global Min Mean:; " << stat_global_minOf[i].mean() << "; " << stat_global_minOf[i].variance() << "; ";
        outfile << "Global Max Mean:; " << stat_global_maxOf[i].mean() << "; " << stat_global_maxOf[i].variance() << "; ";
        outfile << "Hash Fail Count Mean:; " << stat_kol_hash_fail[i].mean() << "; " << stat_kol_hash_fail[i].variance() << ";\n";
    }


}

// ----------------- Hash Table Entry Structure -----------------
struct HashEntry {
    unsigned long long key; // Unique key composed of parameters
    double value;           // Objective function value
};

// ----------------- Kernel: Initializing Hash Table -----------------
__global__ void initializeHashTable(HashEntry* hashTable, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        hashTable[i].key = ZERO_HASH_RESULT; // Установить ключ в максимальное значение
        hashTable[i].value = 0.0; // Установить значение как NaN
    }
}

// ----------------- MurmurHash64A Implementation -----------------
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

// ----------------- Improved Hash Function Using MurmurHash -----------------
__device__ unsigned long long betterHashFunction(unsigned long long key) {
    return murmurHash64A(key) % HASH_TABLE_SIZE;
}

// ----------------- Key Generation Function -----------------
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

// ----------------- Hash Table Search with Quadratic Probing -----------------
__device__ double getCachedResultOptimized(HashEntry* hashTable, const int* agent_node, int bx) {
    unsigned long long key = generateKey(agent_node, bx);
    unsigned long long idx = betterHashFunction(key);
    int i = 1;
    while (i <= MAX_PROBES) {
        if (hashTable[idx].key == key) {
            return hashTable[idx].value; // Found
        }
        if (hashTable[idx].key == ZERO_HASH_RESULT) {
            return -1.0; // Not found and slot is empty
        }
        unsigned long long new_idx = idx + static_cast<unsigned long long>(i * i); if (new_idx >= HASH_TABLE_SIZE) { new_idx %= HASH_TABLE_SIZE; }idx = new_idx;
        i++;
    }
    return -1.0; // Not found after maximum probes
}

// ----------------- Key Generation Function -----------------
__device__ unsigned long long generateKey_one(const int* agent_node) {
    unsigned long long key = 0;
    unsigned long long factor = 1;
    for (int i = 0; i < PARAMETR_SIZE; i++) {
        int val = agent_node[i];
        key += val * factor;
        factor *= MAX_VALUE_SIZE;
    }
    return key;
}

// ----------------- Hash Table Search with Quadratic Probing -----------------
__device__ double getCachedResultOptimized_one(HashEntry* hashTable, const int* agent_node) {
    unsigned long long key = generateKey_one(agent_node);
    unsigned long long idx = betterHashFunction(key);
    int i = 1;
    while (i <= MAX_PROBES) {
        if (hashTable[idx].key == key) {
            return hashTable[idx].value; // Found
        }
        if (hashTable[idx].key == ZERO_HASH_RESULT) {
            return -1.0; // Not found and slot is empty
        }
        unsigned long long new_idx = idx + static_cast<unsigned long long>(i * i); if (new_idx >= HASH_TABLE_SIZE) { new_idx %= HASH_TABLE_SIZE; }idx = new_idx;
        i++;
    }
    return -1.0; // Not found after maximum probes
}

// ----------------- Hash Table Insertion with Quadratic Probing -----------------
__device__ void saveToCacheOptimized(HashEntry* hashTable, const int* agent_node, int bx, double value) {
    unsigned long long key = generateKey(agent_node, bx);
    unsigned long long idx = betterHashFunction(key);
    int i = 1;

    while (i <= MAX_PROBES) {
        unsigned long long expected = ZERO_HASH_RESULT;
        unsigned long long desired = key;
        unsigned long long old = atomicCAS(&(hashTable[idx].key), expected, desired);
        if (old == expected || old == key) {
            // Successfully inserted or key already exists
            hashTable[idx].value = value;
            return;
        }
        unsigned long long new_idx = idx + static_cast<unsigned long long>(i * i); if (new_idx >= HASH_TABLE_SIZE) { new_idx %= HASH_TABLE_SIZE; }idx = new_idx;
        i++;
    }
    // If the table is full, handle the error or ignore
}

// ----------------- Hash Table Insertion with Quadratic Probing -----------------
__device__ void saveToCacheOptimized_one(HashEntry* hashTable, const int* agent_node, double value) {
    unsigned long long key = generateKey_one(agent_node);
    unsigned long long idx = betterHashFunction(key);
    int i = 1;
    while (i <= MAX_PROBES) {
        unsigned long long expected = ZERO_HASH_RESULT;
        unsigned long long desired = key;
        unsigned long long old = atomicCAS(&(hashTable[idx].key), expected, desired);
        if (old == expected || old == key) {
            // Successfully inserted or key already exists
            hashTable[idx].value = value;
            return;
        }
        unsigned long long new_idx = idx + static_cast<unsigned long long>(i * i); if (new_idx >= HASH_TABLE_SIZE) { new_idx %= HASH_TABLE_SIZE; }idx = new_idx;
        i++;
    }
    // If the table is full, handle the error or ignore
}

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

// Функция для вычисления вероятностной формулы
// Входные данные - значение феромона pheromon и количества посещений вершины kol_enter
__device__ double probability_formula(double pheromon, double kol_enter) {
    double res = 0;
    if ((kol_enter != 0) && (pheromon != 0)) {
        res = 1.0 / kol_enter + pheromon;
    }
    return res;
}

//Подготовка массива для вероятностного поиска
//pheromon,kol_enter - матрицы слоев, если надо больше придется менять
//norm_matrix_probability -  итоговая отнормированная матрица
//probability_formula - функция для вычисления вероятностной формулы
__device__ void go_mass_probability(int tx, double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    //Нормализация слоя с феромоном
    double sumVector = 0;
    double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
    }
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
    }
    sumVector = 0;
    double svertka[MAX_VALUE_SIZE] = { 0 };

    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        svertka[i] = probability_formula(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
        sumVector += svertka[i];
    }

    norm_matrix_probability[MAX_VALUE_SIZE * tx] = (svertka[0]) / sumVector;
    for (int i = 1; i < MAX_VALUE_SIZE; i++) {
        norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i]) / sumVector + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1]; //Нормаирование значений матрицы с накоплением
    }
}

__global__ void go_mass_probability_thread(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    int tx = threadIdx.x; // индекс потока (столбца)
    go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);
}
__global__ void go_mass_probability_block(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    int tx = blockIdx.x; // индекс потока (столбца)
    go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);
}
__global__ void go_mass_probability_only(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (параметра) 
    if (tx < PARAMETR_SIZE) {
        go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);
    }
}

__device__ void go_mass_probability_not_f(int tx, double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    //Нормализация слоя с феромоном
    double sumVector = 0;
    double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
    }
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
    }
    sumVector = 0;
    double svertka[MAX_VALUE_SIZE] = { 0 };

    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        svertka[i] = probability_formula(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
        sumVector += svertka[i];
    }

    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = svertka[i] / sumVector;
    }
}

__global__ void go_mass_probability_not_f_thread(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    int tx = threadIdx.x; // индекс потока (столбца)
    go_mass_probability_not_f(tx, pheromon, kol_enter, norm_matrix_probability);
}
__global__ void go_mass_probability_not_f_block(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    int tx = blockIdx.x; // индекс потока (столбца)
    go_mass_probability_not_f(tx, pheromon, kol_enter, norm_matrix_probability);
}
__global__ void go_mass_probability_not_f_only(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (параметра) 
    if (tx < PARAMETR_SIZE) {
        go_mass_probability_not_f(tx, pheromon, kol_enter, norm_matrix_probability);
    }
}

__device__ void go_mass_probability_sort(int tx, double* pheromon, double* kol_enter, double* norm_matrix_probability, int* indices) {
    //Нормализация слоя с феромоном
    double sumVector = 0;
    double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
    }
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
    }
    sumVector = 0;
    double svertka[MAX_VALUE_SIZE] = { 0 };

    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        svertka[i] = probability_formula(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
        sumVector += svertka[i];
    }
    // Заполняем массив индексов начальными значениями
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        indices[MAX_VALUE_SIZE * tx + i] = i;
    }
    // Ручная сортировка методом вставки с сохранением индексов
    for (int j = 1; j < MAX_VALUE_SIZE; ++j) {
        double key = svertka[j];               // Значение текущего элемента
        int idx_key = indices[MAX_VALUE_SIZE * tx + j];              // Индекс текущего элемента
        int i = j - 1;                         // Начинаем проверку с предыдущего элемента

        while (i >= 0 && svertka[i] < key) {
            svertka[i + 1] = svertka[i];         // Сдвигаем больше элементы вправо
            indices[MAX_VALUE_SIZE * tx + i + 1] = indices[MAX_VALUE_SIZE * tx + i];         // Обновляем соответствующий индекс
            i--;
        }
        svertka[i + 1] = key;                   // Кладём ключ на новое место
        indices[MAX_VALUE_SIZE * tx + i + 1] = idx_key;                // Сохраняем индекс ключа
    }
    norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
    for (int i = 1; i < MAX_VALUE_SIZE; i++) {
        norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1]; // Нормирование значений матрицы с накоплением
    }
}

__global__ void go_mass_probability_sort_thread(double* pheromon, double* kol_enter, double* norm_matrix_probability, int* indices) {
    int tx = threadIdx.x; // индекс потока (столбца)
    go_mass_probability_sort(tx, pheromon, kol_enter, norm_matrix_probability, indices);
}
__global__ void go_mass_probability_sort_block(double* pheromon, double* kol_enter, double* norm_matrix_probability, int* indices) {
    int tx = blockIdx.x; // индекс потока (столбца)
    go_mass_probability_sort(tx, pheromon, kol_enter, norm_matrix_probability, indices);
}
__global__ void go_mass_probability_sort_only(double* pheromon, double* kol_enter, double* norm_matrix_probability, int* indices) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (параметра) 
    if (tx < PARAMETR_SIZE) {
        go_mass_probability_sort(tx, pheromon, kol_enter, norm_matrix_probability, indices);
    }
}

__device__ void go_ant_path(int tx, int bx, curandState* state, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node) {
    double randomValue = curand_uniform(state);
    //Определение номера значения
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
    // Запись подматрицы блока в глобальную память каждый поток записывает один элемент
    agent_node[bx * PARAMETR_SIZE + tx] = k;
    agent[tx] = parametr[tx * MAX_VALUE_SIZE + k];
}
__device__ void go_ant_path_one_agent(int tx, int bx, curandState* state, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node) {
    double randomValue = curand_uniform(state);
    //Определение номера значения
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
    // Запись подматрицы блока в глобальную память каждый поток записывает один элемент
    agent_node[tx] = k;
    agent[tx] = parametr[tx * MAX_VALUE_SIZE + k];
}
__device__ void go_ant_path_matrix(int tx, int bx, curandState* state, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node) {
    double randomValue = curand_uniform(state);
    //Определение номера значения
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
    agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
}
__device__ void go_ant_path_random_values(int tx, int bx, double* random_values, int nom_iteration, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node) {
    int k = 0;
    while (k < MAX_VALUE_SIZE && random_values[(bx * PARAMETR_SIZE + tx) * (nom_iteration + 1)] > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) { //
        k++;
    }
    __syncthreads();
    // Запись подматрицы блока в глобальную память каждый поток записывает один элемент
    agent_node[bx * PARAMETR_SIZE + tx] = k;
    agent[tx] = parametr[tx * MAX_VALUE_SIZE + k];
}
__device__ void go_ant_path_const(int tx, int bx, curandState* state, double* norm_matrix_probability, double* agent, int* agent_node) {

    double randomValue = curand_uniform(state);
    //Определение номера значения
    int k = 0;
    while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
        k++;
    }
    __syncthreads();
    // Запись подматрицы блока в глобальную память каждый поток записывает один элемент
    agent_node[bx * PARAMETR_SIZE + tx] = k;
    agent[tx] = parametr_value_dev_const[tx * MAX_VALUE_SIZE + k];
}
__device__ void go_ant_path_random_values_const(int tx, int bx, double* random_values, int nom_iteration, double* norm_matrix_probability, double* agent, int* agent_node) {
    int k = 0;
    while (k < MAX_VALUE_SIZE && random_values[(bx * PARAMETR_SIZE + tx) * (nom_iteration + 1)] > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) { //
        k++;
    }
    __syncthreads();
    // Запись подматрицы блока в глобальную память каждый поток записывает один элемент
    agent_node[bx * PARAMETR_SIZE + tx] = k;
    agent[tx] = parametr_value_dev_const[tx * MAX_VALUE_SIZE + k];
}
__device__ void go_ant_path_not_f(int tx, int bx, curandState* state, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node) {
    double randomValue = 0;
    double max = 0;
    int kmax = 0;
    for (int k = 0; k < MAX_VALUE_SIZE; k++) {
        randomValue = norm_matrix_probability[MAX_VALUE_SIZE * tx + k] * curand_uniform(state);
        if (max < randomValue) {
            max = randomValue;
            kmax = k;
        }
    }
    // Запись подматрицы блока в глобальную память каждый поток записывает один элемент
    agent_node[bx * PARAMETR_SIZE + tx] = kmax;
    agent[tx] = parametr[tx * MAX_VALUE_SIZE + kmax];
}
__device__ void go_ant_path_sort(int tx, int bx, curandState* state, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, int* indices) {
    double randomValue = curand_uniform(state);
    //Определение номера значения
    int k = 0;
    while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
        k++;
    }
    __syncthreads();
    // Запись подматрицы блока в глобальную память каждый поток записывает один элемент
    agent_node[bx * PARAMETR_SIZE + tx] = indices[k];
    agent[tx] = parametr[tx * MAX_VALUE_SIZE + indices[k]];
}

__global__ void go_all_agent_only(double* parametr, double* norm_matrix_probability, double* random_values, int* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (агента) 
    if (bx < ANT_SIZE) {
        //int tx = threadIdx.x;  
        curandState state;
        curand_init(clock64() * bx + gpuTime_const, 0, 0, &state); // Инициализация состояния генератора случайных чисел
        double agent[PARAMETR_SIZE];
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            if (CPU_RANDOM) {
                go_ant_path_random_values(tx, bx, random_values, 0, parametr, norm_matrix_probability, agent, agent_node);
            }
            else
            {
                go_ant_path(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
            }
        }
        // Проверка наличия решения в Хэш-таблице
        double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
        int nom_iteration = 1;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            //OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
            OF[bx] = BenchShafferaFunction(agent);
            saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            //Если значение в Хэш-найдено, то агент "нулевой"
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 1: // ACOCNI
                OF[bx] = ZERO_HASH_RESULT;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 2: // ACOCCyN
                nom_iteration = 0;
                while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                {
                    for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                        if (CPU_RANDOM) {
                            go_ant_path_random_values(tx, bx, random_values, nom_iteration, parametr, norm_matrix_probability, agent, agent_node);
                        }
                        else
                        {
                            go_ant_path(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
                        }
                    }
                    // Проверка наличия решения в Хэш-таблице
                    cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                    nom_iteration++;
                    atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                }
                OF[bx] = BenchShafferaFunction(agent);
                if (OF[bx] != ZERO_HASH_RESULT) { saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]); }
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            }
        }
        __syncthreads();
        if (OF[bx] != ZERO_HASH_RESULT)
        {
            // Обновление максимального и минимального значений с использованием атомарных операций
            atomicMax(maxOf_dev, OF[bx]);
            atomicMin(minOf_dev, OF[bx]);
        }
    }
}
__global__ void go_all_agent_only_global(double* parametr, double* norm_matrix_probability, double* random_values, double* agent, int* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (агента) 
    if (bx < ANT_SIZE) {
        //int tx = threadIdx.x;  
        curandState state;
        curand_init(clock64() * bx + gpuTime_const, 0, 0, &state); // Инициализация состояния генератора случайных чисел
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            if (CPU_RANDOM) {
                go_ant_path_random_values(tx, bx, random_values, 0, parametr, norm_matrix_probability, &agent[bx * PARAMETR_SIZE], agent_node);
            }
            else
            {
                go_ant_path(tx, bx, &state, parametr, norm_matrix_probability, &agent[bx * PARAMETR_SIZE], agent_node);
            }
        }
        // Проверка наличия решения в Хэш-таблице
        double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
        int nom_iteration = 1;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
            //OF[bx] = BenchShafferaFunction(agent);
            saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            //Если значение в Хэш-найдено, то агент "нулевой"
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 1: // ACOCNI
                OF[bx] = ZERO_HASH_RESULT;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 2: // ACOCCyN
                while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                {
                    for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                        if (CPU_RANDOM) {
                            go_ant_path_random_values(tx, bx, random_values, nom_iteration, parametr, norm_matrix_probability, &agent[bx * PARAMETR_SIZE], agent_node);
                        }
                        else
                        {
                            go_ant_path(tx, bx, &state, parametr, norm_matrix_probability, &agent[bx * PARAMETR_SIZE], agent_node);
                        }
                    }
                    // Проверка наличия решения в Хэш-таблице
                    cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                    nom_iteration = nom_iteration + 1;
                    atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                }
                OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
                if (OF[bx] != ZERO_HASH_RESULT) { saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]); }
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            }
        }
        __syncthreads();
        if (OF[bx] != ZERO_HASH_RESULT)
        {
            // Обновление максимального и минимального значений с использованием атомарных операций
            atomicMax(maxOf_dev, OF[bx]);
            atomicMin(minOf_dev, OF[bx]);
        }
    }
}
__global__ void go_all_agent_only_const(double* norm_matrix_probability, double* random_values, int* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (агента) 
    if (bx < ANT_SIZE) {
        //int tx = threadIdx.x;  
        curandState state;
        curand_init(clock64() * bx + gpuTime_const, 0, 0, &state); // Инициализация состояния генератора случайных чисел
        double agent[PARAMETR_SIZE];
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            if (CPU_RANDOM) { go_ant_path_random_values_const(tx, bx, random_values, 0, norm_matrix_probability, agent, agent_node); }
            else { go_ant_path_const(tx, bx, &state, norm_matrix_probability, agent, agent_node); }
        }
        // Проверка наличия решения в Хэш-таблице
        double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
        int nom_iteration = 1;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            //OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
            OF[bx] = BenchShafferaFunction(agent);
            saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            //Если значение в Хэш-найдено, то агент "нулевой"
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 1: // ACOCNI
                OF[bx] = ZERO_HASH_RESULT;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 2: // ACOCCyN
                while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                {
                    for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                        if (CPU_RANDOM) { go_ant_path_random_values_const(tx, bx, random_values, 0, norm_matrix_probability, agent, agent_node); }
                        else { go_ant_path_const(tx, bx, &state, norm_matrix_probability, agent, agent_node); }
                    }
                    // Проверка наличия решения в Хэш-таблице
                    cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                    nom_iteration = nom_iteration + 1;
                    atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                }
                OF[bx] = BenchShafferaFunction(agent);
                if (OF[bx] != ZERO_HASH_RESULT) { saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]); }
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            }
        }
        __syncthreads();
        if (OF[bx] != ZERO_HASH_RESULT)
        {
            // Обновление максимального и минимального значений с использованием атомарных операций
            atomicMax(maxOf_dev, OF[bx]);
            atomicMin(minOf_dev, OF[bx]);
        }
    }
}
__global__ void go_all_agent_only_non_hash(double* parametr, double* norm_matrix_probability, double* random_values, int* agent_node, double* OF, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (агента) 
    if (bx < ANT_SIZE) {
        //int tx = threadIdx.x;  
        double agent[PARAMETR_SIZE];
        curandState state;
        curand_init(clock64() * bx + gpuTime_const, 0, 0, &state); // Инициализация состояния генератора случайных чисел
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            if (CPU_RANDOM) {
                go_ant_path_random_values(tx, bx, random_values, 0, parametr, norm_matrix_probability, agent, agent_node);
            }
            else
            {
                go_ant_path(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
            }
        }
        OF[bx] = BenchShafferaFunction(agent);

        // Обновление максимального и минимального значений с использованием атомарных операций
        atomicMax(maxOf_dev, OF[bx]);
        atomicMin(minOf_dev, OF[bx]);
    }
}
__global__ void go_all_agent(double* parametr, double* norm_matrix_probability, double* random_values, int* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = blockIdx.x;  // Параллелизм по муравьям
    int tx = threadIdx.x; // Параллелизм по параметрам
    double agent[PARAMETR_SIZE];
    curandState state;
    curand_init((bx * blockDim.x + tx) * clock64() + gpuTime_const, 0, 0, &state); // Инициализация состояния генератора случайных чисел
    if (CPU_RANDOM) {
        go_ant_path_random_values(tx, bx, random_values, 0, parametr, norm_matrix_probability, agent, agent_node);
    }
    else
    {
        // Генерация случайного числа с использованием curand

        go_ant_path(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
    }
    double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
    int nom_iteration = 1;
    if (cachedResult == -1.0) {
        // Если значение не найденов ХЭШ, то заносим новое значение
        OF[bx] = BenchShafferaFunction(agent);
        saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
    }
    else {
        //Если значение в Хэш-найдено, то агент "нулевой"
        //Поиск алгоритма для нулевого агента
        switch (TYPE_ACO) {
        case 0: // ACOCN
            OF[bx] = cachedResult;
            atomicAdd(&kol_hash_fail[0], 1);
            break;
        case 1: // ACOCNI
            OF[bx] = ZERO_HASH_RESULT;
            atomicAdd(&kol_hash_fail[0], 1);
            break;
        case 2: // ACOCCyN
            while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
            {
                if (CPU_RANDOM) {
                    go_ant_path_random_values(tx, bx, random_values, nom_iteration, parametr, norm_matrix_probability, agent, agent_node);
                }
                else
                {
                    go_ant_path(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
                }
                // Проверка наличия решения в Хэш-таблице
                cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                nom_iteration = nom_iteration + 1;
                atomicAdd(&kol_hash_fail[0], 1);
            }
            OF[bx] = BenchShafferaFunction(agent);
            if (OF[bx] != ZERO_HASH_RESULT) { saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]); }
            break;
        default:
            OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
            atomicAdd(&kol_hash_fail[0], 1);
            break;
        }


    }
    __syncthreads();
    if (OF[bx] != ZERO_HASH_RESULT)
    {
        // Обновление максимального и минимального значений с использованием атомарных операций
        atomicMax(maxOf_dev, OF[bx]);
        atomicMin(minOf_dev, OF[bx]);
    }
}
__global__ void go_all_agent_const(double* norm_matrix_probability, double* random_values, int* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = blockIdx.x;  // Параллелизм по муравьям
    int tx = threadIdx.x; // Параллелизм по параметрам
    double agent[PARAMETR_SIZE];
    curandState state;
    curand_init((bx * blockDim.x + tx) * clock64() + gpuTime_const, 0, 0, &state); // Инициализация состояния генератора случайных чисел
    if (CPU_RANDOM) { go_ant_path_random_values_const(tx, bx, random_values, 0, norm_matrix_probability, agent, agent_node); }
    else { go_ant_path_const(tx, bx, &state, norm_matrix_probability, agent, agent_node); }
    double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
    int nom_iteration = 1;
    if (cachedResult == -1.0) {
        // Если значение не найденов ХЭШ, то заносим новое значение
        OF[bx] = BenchShafferaFunction(agent);
        saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
    }
    else {
        //Если значение в Хэш-найдено, то агент "нулевой"
        //Поиск алгоритма для нулевого агента
        switch (TYPE_ACO) {
        case 0: // ACOCN
            OF[bx] = cachedResult;
            atomicAdd(&kol_hash_fail[0], 1);
            break;
        case 1: // ACOCNI
            OF[bx] = ZERO_HASH_RESULT;
            atomicAdd(&kol_hash_fail[0], 1);
            break;
        case 2: // ACOCCyN
            while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
            {
                if (CPU_RANDOM) { go_ant_path_random_values_const(tx, bx, random_values, 0, norm_matrix_probability, agent, agent_node); }
                else { go_ant_path_const(tx, bx, &state, norm_matrix_probability, agent, agent_node); }
                // Проверка наличия решения в Хэш-таблице
                cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                nom_iteration = nom_iteration + 1;
                atomicAdd(&kol_hash_fail[0], 1);
            }
            OF[bx] = BenchShafferaFunction(agent);
            if (OF[bx] != ZERO_HASH_RESULT) { saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]); }
            break;
        default:
            OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
            atomicAdd(&kol_hash_fail[0], 1);
            break;
        }


    }
    __syncthreads();
    if (OF[bx] != ZERO_HASH_RESULT)
    {
        // Обновление максимального и минимального значений с использованием атомарных операций
        atomicMax(maxOf_dev, OF[bx]);
        atomicMin(minOf_dev, OF[bx]);
    }
}
__global__ void go_all_agent_only_block(double* parametr, double* norm_matrix_probability, double* random_values, int* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = blockIdx.x;  // Параллелизм по муравьям
    //int tx = threadIdx.x; // Параллелизм по параметрам 
    double agent[PARAMETR_SIZE];
    curandState state;
    curand_init((bx * blockDim.x) * clock64() + gpuTime_const, 0, 0, &state); // Инициализация состояния генератора случайных чисел
    for (int tx = 0; tx < PARAMETR_SIZE; tx++)
    {
        if (CPU_RANDOM) {
            go_ant_path_random_values(tx, bx, random_values, 0, parametr, norm_matrix_probability, agent, agent_node);
        }
        else
        {
            // Генерация случайного числа с использованием curand

            go_ant_path(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
        }
        double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
        int nom_iteration = 1;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            OF[bx] = BenchShafferaFunction(agent);
            saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            //Если значение в Хэш-найдено, то агент "нулевой"
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                atomicAdd(&kol_hash_fail[0], 1);
                break;
            case 1: // ACOCNI
                OF[bx] = ZERO_HASH_RESULT;
                atomicAdd(&kol_hash_fail[0], 1);
                break;
            case 2: // ACOCCyN
                while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                {
                    if (CPU_RANDOM) {
                        go_ant_path_random_values(tx, bx, random_values, nom_iteration, parametr, norm_matrix_probability, agent, agent_node);
                    }
                    else
                    {
                        go_ant_path(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
                    }
                    // Проверка наличия решения в Хэш-таблице
                    cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                    nom_iteration = nom_iteration + 1;
                    atomicAdd(&kol_hash_fail[0], 1);
                }
                OF[bx] = BenchShafferaFunction(agent);
                if (OF[bx] != ZERO_HASH_RESULT) { saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]); }
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                atomicAdd(&kol_hash_fail[0], 1);
                break;
            }


        }
    }
    if (OF[bx] != ZERO_HASH_RESULT)
    {
        // Обновление максимального и минимального значений с использованием атомарных операций
        atomicMax(maxOf_dev, OF[bx]);
        atomicMin(minOf_dev, OF[bx]);
    }

}
__global__ void go_all_agent_non_hash(double* parametr, double* norm_matrix_probability, double* random_values, int* agent_node, double* OF, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = blockIdx.x;  // индекс  (столбца)
    int tx = threadIdx.x; // индекс  (агента) 
    double agent[PARAMETR_SIZE];
    curandState state;
    curand_init((bx * blockDim.x + tx) * clock64() + gpuTime_const, 0, 0, &state); // Инициализация состояния генератора случайных чисел
    if (CPU_RANDOM) {
        go_ant_path_random_values(tx, bx, random_values, 0, parametr, norm_matrix_probability, agent, agent_node);
    }
    else
    {
        // Генерация случайного числа с использованием curand
        go_ant_path(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
    }
    //__syncthreads();
    OF[bx] = BenchShafferaFunction(agent);
    // Обновление максимального и минимального значений с использованием атомарных операций
    atomicMax(maxOf_dev, OF[bx]);
    atomicMin(minOf_dev, OF[bx]);
}

__global__ void go_all_agent_not_f_only(double* parametr, double* norm_matrix_probability, double* random_values, int* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (агента) 
    if (bx < ANT_SIZE) {
        //int tx = threadIdx.x;  
        curandState state;
        curand_init(clock64() * bx + gpuTime_const, 0, 0, &state); // Инициализация состояния генератора случайных чисел
        double agent[PARAMETR_SIZE];
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            if (CPU_RANDOM) {
                go_ant_path_random_values(tx, bx, random_values, 0, parametr, norm_matrix_probability, agent, agent_node);
            }
            else
            {
                go_ant_path_not_f(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
            }
        }
        // Проверка наличия решения в Хэш-таблице
        double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
        int nom_iteration = 1;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            //OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
            OF[bx] = BenchShafferaFunction(agent);
            saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            //Если значение в Хэш-найдено, то агент "нулевой"
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 1: // ACOCNI
                OF[bx] = ZERO_HASH_RESULT;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 2: // ACOCCyN
                nom_iteration = 0;
                while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                {
                    for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                        if (CPU_RANDOM) {
                            go_ant_path_random_values(tx, bx, random_values, nom_iteration, parametr, norm_matrix_probability, agent, agent_node);
                        }
                        else
                        {
                            go_ant_path_not_f(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
                        }
                    }
                    // Проверка наличия решения в Хэш-таблице
                    cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                    nom_iteration++;
                    atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                }
                OF[bx] = BenchShafferaFunction(agent);
                if (OF[bx] != ZERO_HASH_RESULT) { saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]); }
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            }
        }
        __syncthreads();
        if (OF[bx] != ZERO_HASH_RESULT)
        {
            // Обновление максимального и минимального значений с использованием атомарных операций
            atomicMax(maxOf_dev, OF[bx]);
            atomicMin(minOf_dev, OF[bx]);
        }
    }
}
__global__ void go_all_agent_not_sort_only(double* parametr, double* norm_matrix_probability, double* random_values, int* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail, int* indices) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (агента) 
    if (bx < ANT_SIZE) {
        //int tx = threadIdx.x;  
        curandState state;
        curand_init(clock64() * bx + gpuTime_const, 0, 0, &state); // Инициализация состояния генератора случайных чисел
        double agent[PARAMETR_SIZE];
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            if (CPU_RANDOM) {
                go_ant_path_random_values(tx, bx, random_values, 0, parametr, norm_matrix_probability, agent, agent_node);
            }
            else
            {
                go_ant_path_sort(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node, indices);
            }
        }
        // Проверка наличия решения в Хэш-таблице
        double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
        int nom_iteration = 1;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            //OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
            OF[bx] = BenchShafferaFunction(agent);
            saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            //Если значение в Хэш-найдено, то агент "нулевой"
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 1: // ACOCNI
                OF[bx] = ZERO_HASH_RESULT;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 2: // ACOCCyN
                nom_iteration = 0;
                while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                {
                    for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                        if (CPU_RANDOM) {
                            go_ant_path_random_values(tx, bx, random_values, nom_iteration, parametr, norm_matrix_probability, agent, agent_node);
                        }
                        else
                        {
                            go_ant_path_sort(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node, indices);
                        }
                    }
                    // Проверка наличия решения в Хэш-таблице
                    cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                    nom_iteration++;
                    atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                }
                OF[bx] = BenchShafferaFunction(agent);
                if (OF[bx] != ZERO_HASH_RESULT) { saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]); }
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            }
        }
        __syncthreads();
        if (OF[bx] != ZERO_HASH_RESULT)
        {
            // Обновление максимального и минимального значений с использованием атомарных операций
            atomicMax(maxOf_dev, OF[bx]);
            atomicMin(minOf_dev, OF[bx]);
        }
    }
}

__device__ void add_pheromon_iteration(int tx, double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    //Испарение весов-феромона
    for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
        pheromon[MAX_VALUE_SIZE * tx + i] = pheromon[MAX_VALUE_SIZE * tx + i] * PARAMETR_RO;
    }
    //Добавление весов-феромона
    for (int i = 0; i < ANT_SIZE; ++i) {
        if (OF[i] != ZERO_HASH_RESULT) {
            int k = agent_node[i * PARAMETR_SIZE + tx];
            kol_enter[MAX_VALUE_SIZE * tx + k]++;
#if (OPTIMIZE_MIN_1)
            if (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i] > 0) {
                pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i]);
            }
#endif // (OPTIMIZE_MIN_1)
#if (OPTIMIZE_MIN_2)
            if (OF[i] == 0) { OF[i] = 0.0000001; }
            pheromon[MAX_VALUE_SIZE * tx + k] = pheromon[MAX_VALUE_SIZE * tx + k] + PARAMETR_Q / OF[i];
#endif // (OPTIMIZE_MIN_2)
#if (OPTIMIZE_MAX)
            pheromon[MAX_VALUE_SIZE * tx + k] = pheromon[MAX_VALUE_SIZE * tx + k] + PARAMETR_Q * OF[i];
#endif // (OPTIMIZE_MAX)
        }
    }
}

__global__ void add_pheromon_iteration_thread(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    //    int bx = blockIdx.x; // индекс блока (не требуется)
    int tx = threadIdx.x; // индекс потока (параметра)
    add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
}
__global__ void add_pheromon_iteration_block(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    int tx = blockIdx.x; // индекс блока (не требуется)
    //int tx = threadIdx.x; // индекс потока (параметра)
    add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
}
__global__ void add_pheromon_iteration_only(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (параметра) 
    if (tx < PARAMETR_SIZE) {
        add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
    }
}

__global__ void go_mass_probability_and_add_pheromon_iteration(double* pheromon, double* kol_enter, double* norm_matrix_probability, int* agent_node, double* OF) {
    //    int bx = blockIdx.x; // индекс блока (не требуется)
    int tx = threadIdx.x; // индекс потока (столбца)
    //Испарение весов-феромона
    add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
    //Нормализация слоя с феромоном
    go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);
}
__global__ void go_mass_probability_and_add_pheromon_iteration_block(double* pheromon, double* kol_enter, double* norm_matrix_probability, int* agent_node, double* OF) {
    int tx = blockIdx.x; // индекс блока (не требуется)
    //int tx = threadIdx.x; // индекс потока (столбца)
    //Испарение весов-феромона
    add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
    //Нормализация слоя с феромоном
    go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);
}
__global__ void go_mass_probability_and_add_pheromon_iteration_only(double* pheromon, double* kol_enter, double* norm_matrix_probability, int* agent_node, double* OF) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (параметра) 
    if (tx < PARAMETR_SIZE) {
        //Испарение весов-феромона
        add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
        //Нормализация слоя с феромоном
        go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);
    }
}

__global__ void go_mass_probability_not_f_and_add_pheromon_iteration(double* pheromon, double* kol_enter, double* norm_matrix_probability, int* agent_node, double* OF) {
    //    int bx = blockIdx.x; // индекс блока (не требуется)
    int tx = threadIdx.x; // индекс потока (столбца)
    //Испарение весов-феромона
    add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
    //Нормализация слоя с феромоном
    go_mass_probability_not_f(tx, pheromon, kol_enter, norm_matrix_probability);
}
__global__ void go_mass_probability_not_f_and_add_pheromon_iteration_block(double* pheromon, double* kol_enter, double* norm_matrix_probability, int* agent_node, double* OF) {
    int tx = blockIdx.x; // индекс блока (не требуется)
    //int tx = threadIdx.x; // индекс потока (столбца)
    //Испарение весов-феромона
    add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
    //Нормализация слоя с феромоном
    go_mass_probability_not_f(tx, pheromon, kol_enter, norm_matrix_probability);
}
__global__ void go_mass_probability_not_f_and_add_pheromon_iteration_only(double* pheromon, double* kol_enter, double* norm_matrix_probability, int* agent_node, double* OF) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (параметра) 
    if (tx < PARAMETR_SIZE) {
        //Испарение весов-феромона
        add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
        //Нормализация слоя с феромоном
        go_mass_probability_not_f(tx, pheromon, kol_enter, norm_matrix_probability);
    }
}

__global__ void go_mass_probability_sort_and_add_pheromon_iteration(double* pheromon, double* kol_enter, double* norm_matrix_probability, int* agent_node, double* OF, int* indices) {
    //    int bx = blockIdx.x; // индекс блока (не требуется)
    int tx = threadIdx.x; // индекс потока (столбца)
    //Испарение весов-феромона
    add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
    //Нормализация слоя с феромоном
    go_mass_probability_sort(tx, pheromon, kol_enter, norm_matrix_probability, indices);
}
__global__ void go_mass_probability_sort_and_add_pheromon_iteration_block(double* pheromon, double* kol_enter, double* norm_matrix_probability, int* agent_node, double* OF, int* indices) {
    int tx = blockIdx.x; // индекс блока (не требуется)
    //int tx = threadIdx.x; // индекс потока (столбца)
    //Испарение весов-феромона
    add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
    //Нормализация слоя с феромоном
    go_mass_probability_sort(tx, pheromon, kol_enter, norm_matrix_probability, indices);
}
__global__ void go_mass_probability_sort_and_add_pheromon_iteration_only(double* pheromon, double* kol_enter, double* norm_matrix_probability, int* agent_node, double* OF, int* indices) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (параметра) 
    if (tx < PARAMETR_SIZE) {
        //Испарение весов-феромона
        add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
        //Нормализация слоя с феромоном
        go_mass_probability_sort(tx, pheromon, kol_enter, norm_matrix_probability, indices);
    }
}

__global__ void decrease_pheromon_iteration(double* pheromon) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (параметра) 
    if (tx < PARAMETR_SIZE * MAX_VALUE_SIZE) {
        pheromon[tx] = pheromon[tx] * PARAMETR_RO;
    }
}

__device__ void only_add_pheromon_iteration_par(int tx, double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    //tx - номер параметра
    //Добавление весов-феромона
    for (int i = 0; i < ANT_SIZE; ++i) {
        if (OF[i] != ZERO_HASH_RESULT) {
            int k = agent_node[i * PARAMETR_SIZE + tx];
            kol_enter[MAX_VALUE_SIZE * tx + k]++;
#if (OPTIMIZE_MIN_1)
            if (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i] > 0) {
                pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i]);
            }
#endif // (OPTIMIZE_MIN_1)
#if (OPTIMIZE_MIN_2)
            if (OF[i] == 0) { OF[i] = 0.0000001; }
            pheromon[MAX_VALUE_SIZE * tx + k] = pheromon[MAX_VALUE_SIZE * tx + k] + PARAMETR_Q / OF[i];
#endif // (OPTIMIZE_MIN_2)
#if (OPTIMIZE_MAX)
            pheromon[MAX_VALUE_SIZE * tx + k] = pheromon[MAX_VALUE_SIZE * tx + k] + PARAMETR_Q * OF[i];
#endif // (OPTIMIZE_MAX)
        }
    }
}
__device__ void only_add_pheromon_iteration_ant(int tx, double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    //tx - номер агента
    //Добавление весов-феромона
    if (OF[tx] != ZERO_HASH_RESULT) {
        for (int nom_par = 0; nom_par < PARAMETR_SIZE; ++nom_par) {
            kol_enter[MAX_VALUE_SIZE * nom_par + agent_node[tx * PARAMETR_SIZE + nom_par]]++;
            if (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[tx] > 0) {
                pheromon[MAX_VALUE_SIZE * nom_par + agent_node[tx * PARAMETR_SIZE + nom_par]] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[tx]); // MIN
            }
        }
    }
}
__device__ void only_add_pheromon_iteration_ant_par(int ant, int par, double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    //tx - номер агента
    //bx - номер параметра 
    if (OF[ant] != ZERO_HASH_RESULT) {
        kol_enter[MAX_VALUE_SIZE * par + agent_node[ant * PARAMETR_SIZE + par]]++;
        if (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[ant] > 0) {
            pheromon[MAX_VALUE_SIZE * par + agent_node[ant * PARAMETR_SIZE + par]] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[ant]); // MIN
        }
    }

}
__global__ void only_add_pheromon_iteration_par_block_only(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    //int tx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (параметра) 
    int tx = blockIdx.x;
    only_add_pheromon_iteration_par(tx, pheromon, kol_enter, agent_node, OF);
}
__global__ void only_add_pheromon_iteration_par_only(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (параметра) 
    only_add_pheromon_iteration_par(tx, pheromon, kol_enter, agent_node, OF);
}
__global__ void only_add_pheromon_iteration_ant_block_only(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    //int tx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (параметра) 
    int tx = blockIdx.x;
    only_add_pheromon_iteration_ant(tx, pheromon, kol_enter, agent_node, OF);
}
__global__ void only_add_pheromon_iteration_ant_only(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (параметра) 
    only_add_pheromon_iteration_ant(tx, pheromon, kol_enter, agent_node, OF);
}
__global__ void only_add_pheromon_iteration_ant_par_thread(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    if (tx < PARAMETR_SIZE && bx < ANT_SIZE) {
        only_add_pheromon_iteration_ant_par(bx, tx, pheromon, kol_enter, agent_node, OF);
    }
}
__global__ void only_add_pheromon_iteration_ant_par_thread_transp(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    if (bx < PARAMETR_SIZE && tx < ANT_SIZE) {
        only_add_pheromon_iteration_ant_par(tx, bx, pheromon, kol_enter, agent_node, OF);
    }
}

__global__ void go_all_agent_opt(double* pheromon, double* kol_enter, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = blockIdx.x; // индекс блока (агента)
    int tx = threadIdx.x; // индекс потока (параметра)
    curandState state;
    kol_hash_fail[0] = 0;
    curand_init(static_cast<unsigned long long>(bx * blockDim.x + tx), 0, 0, &state); // Инициализация состояния генератора случайных чисел
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);
        go_ant_path_matrix(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);

        double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
        int nom_iteration = 1;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
            saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            //Если значение в Хэш-найдено, то агент "нулевой"
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 1: // ACOCNI
                OF[bx] = ZERO_HASH_RESULT;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 2: // ACOCCyN
                while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                {
                    go_ant_path(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
                    // Проверка наличия решения в Хэш-таблице
                    cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                    nom_iteration = nom_iteration + 1;
                    atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                }
                OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
                if (OF[bx] != ZERO_HASH_RESULT) { saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]); }
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            }
        }
        // Обновление максимального и минимального значений с использованием атомарных операций
        __syncthreads();
        if (OF[bx] != ZERO_HASH_RESULT)
        {
            // Обновление максимального и минимального значений с использованием атомарных операций
            atomicMax(maxOf_dev, OF[bx]);
            atomicMin(minOf_dev, OF[bx]);
        }
        //Испарение весов-феромона
        add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
    }
}
__global__ void go_all_agent_opt_local(double* pheromon, double* kol_enter, double* parametr, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = blockIdx.x; // индекс блока (агента)
    int tx = threadIdx.x; // индекс потока (параметра)
    double norm_matrix_probability[MAX_VALUE_SIZE * PARAMETR_SIZE];
    double agent[PARAMETR_SIZE * ANT_SIZE] = { 0 };
    int agent_node[PARAMETR_SIZE * ANT_SIZE] = { 0 };
    double OF[ANT_SIZE] = { 0 };
    curandState state;
    curand_init(static_cast<unsigned long long>(bx * blockDim.x + tx), 0, 0, &state); // Инициализация состояния генератора случайных чисел
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);
        go_ant_path_matrix(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);

        double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
        int nom_iteration = 1;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
            saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            //Если значение в Хэш-найдено, то агент "нулевой"
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 1: // ACOCNI
                OF[bx] = ZERO_HASH_RESULT;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 2: // ACOCCyN
                while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                {
                    go_ant_path(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
                    // Проверка наличия решения в Хэш-таблице
                    cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                    nom_iteration = nom_iteration + 1;
                    atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                }
                OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
                if (OF[bx] != ZERO_HASH_RESULT) { saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]); }
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            }
        }
        // Обновление максимального и минимального значений с использованием атомарных операций
        __syncthreads();
        if (OF[bx] != ZERO_HASH_RESULT)
        {
            // Обновление максимального и минимального значений с использованием атомарных операций
            atomicMax(maxOf_dev, OF[bx]);
            atomicMin(minOf_dev, OF[bx]);
        }
        //Испарение весов-феромона
        add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
    }
}
__global__ void go_all_agent_opt_non_hash(double* pheromon, double* kol_enter, int* gpuTime, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = blockIdx.x; // индекс блока (агента)
    int tx = threadIdx.x; // индекс потока (столбца)
    unsigned long long seed = 1230 + bx * ANT_SIZE + tx * PARAMETR_SIZE + clock64();
    curandState state;
    curand_init(bx * blockDim.x + tx + clock64(), 0, 0, &state); // Инициализация состояния генератора случайных чисел
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);
        go_ant_path(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
        OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
        // Обновление максимального и минимального значений с использованием атомарных операций
        atomicMax(maxOf_dev, OF[bx]);
        atomicMin(minOf_dev, OF[bx]);
        //Испарение весов-феромона
        add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
    }
}
__global__ void go_all_agent_opt_only(double* pheromon, double* kol_enter, int* gpuTime, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (агента) 
    //double agent[PARAMETR_SIZE * ANT_SIZE];
    if (bx < ANT_SIZE) {
        //int tx = threadIdx.x; // индекс потока (столбца)
        curandState state;
        curand_init(static_cast<unsigned long long>(bx) * 10, 0, 0, &state); // Инициализация состояния генератора случайных чисел
        for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                //Нормализация слоя с феромоном
                go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);
            }
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                go_ant_path_matrix(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
            }

            // Проверка наличия решения в Хэш-таблице
            double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
            int nom_iteration = 1;
            if (cachedResult == -1.0) {
                // Если значение не найденов ХЭШ, то заносим новое значение
                OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
                saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
            }
            else {
                //Если значение в Хэш-найдено, то агент "нулевой"
                //Поиск алгоритма для нулевого агента
                switch (TYPE_ACO) {
                case 0: // ACOCN
                    OF[bx] = cachedResult;
                    atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                    break;
                case 1: // ACOCNI
                    OF[bx] = ZERO_HASH_RESULT;
                    atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                    break;
                case 2: // ACOCCyN
                    while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                    {
                        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                            go_ant_path_matrix(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
                        }
                        // Проверка наличия решения в Хэш-таблице
                        cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                        nom_iteration = nom_iteration + 1;
                        atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                    }
                    OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
                    if (OF[bx] != ZERO_HASH_RESULT) { saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]); }
                    break;

                default:
                    OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                    atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                    break;
                }
            }
            __syncthreads();
            if (OF[bx] != ZERO_HASH_RESULT)
            {
                // Обновление максимального и минимального значений с использованием атомарных операций
                atomicMax(maxOf_dev, OF[bx]);
                atomicMin(minOf_dev, OF[bx]);
            }

            for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                //Испарение весов-феромона
                add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
            }

        }
    }
}
__global__ void go_all_agent_opt_only_local(double* pheromon, double* kol_enter, double* parametr, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (агента) 
    //__shared__ 
    double agent[PARAMETR_SIZE] = { 0 };
    double norm_matrix_probability[MAX_VALUE_SIZE * PARAMETR_SIZE] = { 0 };
    int agent_node[PARAMETR_SIZE] = { 0 };
    double OF[ANT_SIZE] = { 0 };
    if (bx < ANT_SIZE) {
        //int tx = threadIdx.x; // индекс потока (столбца)
        curandState state;
        curand_init(static_cast<unsigned long long>(bx) * 10, 0, 0, &state); // Инициализация состояния генератора случайных чисел
        for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                //Нормализация слоя с феромоном
                go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);
            }
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                go_ant_path_one_agent(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
            }

            // Проверка наличия решения в Хэш-таблице
            double cachedResult = getCachedResultOptimized_one(hashTable, agent_node);
            int nom_iteration = 1;
            if (cachedResult == -1.0) {
                // Если значение не найденов ХЭШ, то заносим новое значение
                OF[bx] = BenchShafferaFunction(agent);
                saveToCacheOptimized_one(hashTable, agent_node, OF[bx]);
            }
            else {
                //Если значение в Хэш-найдено, то агент "нулевой"
                //Поиск алгоритма для нулевого агента
                switch (TYPE_ACO) {
                case 0: // ACOCN
                    OF[bx] = cachedResult;
                    atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                    break;
                case 1: // ACOCNI
                    OF[bx] = ZERO_HASH_RESULT;
                    atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                    break;
                case 2: // ACOCCyN
                    while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                    {
                        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                            go_ant_path_one_agent(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
                        }
                        // Проверка наличия решения в Хэш-таблице
                        cachedResult = getCachedResultOptimized_one(hashTable, agent_node);
                        nom_iteration = nom_iteration + 1;
                        atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                    }
                    OF[bx] = BenchShafferaFunction(agent);
                    if (OF[bx] != ZERO_HASH_RESULT) { saveToCacheOptimized_one(hashTable, agent_node, OF[bx]); }
                    break;

                default:
                    OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                    atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                    break;
                }
            }
            __syncthreads();
            if (OF[bx] != ZERO_HASH_RESULT)
            {
                // Обновление максимального и минимального значений с использованием атомарных операций
                atomicMax(maxOf_dev, OF[bx]);
                atomicMin(minOf_dev, OF[bx]);
            }
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                if (OF[bx] != ZERO_HASH_RESULT) {
                    int k = agent_node[tx];
                    kol_enter[MAX_VALUE_SIZE * tx + k]++;
                    //pheromon[MAX_VALUE_SIZE * tx + k] = pheromon[MAX_VALUE_SIZE * tx + k] + PARAMETR_Q * OF[i]; //MAX
                    //if (OF[i] == 0) { OF[i] = 0.0000001; }
                    //pheromon[MAX_VALUE_SIZE * tx + k] = pheromon[MAX_VALUE_SIZE * tx + k] + PARAMETR_Q / OF[i]; //MIN
                    if (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[bx] > 0) {
                        pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[bx]); // MIN
                    }
                }
                //Испарение весов-феромона
                for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
                    pheromon[MAX_VALUE_SIZE * tx + i] = pheromon[MAX_VALUE_SIZE * tx + i] * PARAMETR_RO;
                }
            }
        }
    }
}
__global__ void go_all_agent_opt_only_non_hash(double* pheromon, double* kol_enter, int* gpuTime, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (агента) 
    if (bx < ANT_SIZE) {
        //int tx = threadIdx.x; // индекс потока (столбца)
        unsigned long long seed = 1230 + bx * ANT_SIZE + threadIdx.x * PARAMETR_SIZE + clock64();
        curandState state;
        curand_init(bx + clock64(), 0, 0, &state); // Инициализация состояния генератора случайных чисел
        for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                //Нормализация слоя с феромоном
                go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);
            }
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                go_ant_path(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
            }
            OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
            // Обновление максимального и минимального значений с использованием атомарных операций
            atomicMax(maxOf_dev, OF[bx]);
            atomicMin(minOf_dev, OF[bx]);
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                //Испарение весов-феромона
                add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
            }
        }
    }
}

__device__ void go_ant_path_transp(int tx, int bx, curandState* state, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node) {
    double randomValue = curand_uniform(state);
    int k = 0;
    while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[tx + k * PARAMETR_SIZE]) {
        k++;
    }
    __syncthreads();
    agent_node[bx * PARAMETR_SIZE + tx] = k;
    agent[tx] = parametr[tx + k * PARAMETR_SIZE];
}
__device__ void go_ant_path_transp_random_values(int tx, int bx, double* random_values, int nom_iteration, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node) {
    int k = 0;
    while (k < MAX_VALUE_SIZE && random_values[(bx * PARAMETR_SIZE + tx) * (nom_iteration + 1)] > norm_matrix_probability[tx + k * PARAMETR_SIZE]) { //
        k++;
    }
    __syncthreads();
    agent_node[bx * PARAMETR_SIZE + tx] = k;
    agent[tx] = parametr[tx + k * PARAMETR_SIZE];
}

__global__ void go_all_agent_only_transp(double* parametr, double* norm_matrix_probability, double* random_values, int* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (агента) 
    if (bx < ANT_SIZE) {
        //int tx = threadIdx.x;  
        curandState state;
        curand_init(clock64() * bx + gpuTime_const, 0, 0, &state); // Инициализация состояния генератора случайных чисел
        double agent[PARAMETR_SIZE];
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            if (CPU_RANDOM) {
                go_ant_path_transp_random_values(tx, bx, random_values, 0, parametr, norm_matrix_probability, agent, agent_node);
            }
            else
            {
                go_ant_path_transp(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
            }
        }
        // Проверка наличия решения в Хэш-таблице
        double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
        int nom_iteration = 1;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            //OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
            OF[bx] = BenchShafferaFunction(agent);
            saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            //Если значение в Хэш-найдено, то агент "нулевой"
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 1: // ACOCNI
                OF[bx] = ZERO_HASH_RESULT;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 2: // ACOCCyN
                while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                {
                    for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                        if (CPU_RANDOM) {
                            go_ant_path_transp_random_values(tx, bx, random_values, nom_iteration, parametr, norm_matrix_probability, agent, agent_node);
                        }
                        else
                        {
                            go_ant_path_transp(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
                        }
                    }
                    // Проверка наличия решения в Хэш-таблице
                    cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                    nom_iteration = nom_iteration + 1;
                    atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                }
                OF[bx] = BenchShafferaFunction(agent);
                if (OF[bx] != ZERO_HASH_RESULT) { saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]); }
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            }
        }
        __syncthreads();
        if (OF[bx] != ZERO_HASH_RESULT)
        {
            // Обновление максимального и минимального значений с использованием атомарных операций
            atomicMax(maxOf_dev, OF[bx]);
            atomicMin(minOf_dev, OF[bx]);
        }
    }
}
__global__ void go_all_agent_only_transp_non_hash(double* parametr, double* norm_matrix_probability, double* random_values, int* agent_node, double* OF, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (агента) 
    if (bx < ANT_SIZE) {
        double agent[PARAMETR_SIZE];
        curandState state;
        curand_init(clock64() * bx + gpuTime_const, 0, 0, &state); // Инициализация состояния генератора случайных чисел
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            if (CPU_RANDOM) {
                go_ant_path_transp_random_values(tx, bx, random_values, 0, parametr, norm_matrix_probability, agent, agent_node);
            }
            else
            {
                go_ant_path_transp(tx, bx, &state, parametr, norm_matrix_probability, agent, agent_node);
            }
        }
        OF[bx] = BenchShafferaFunction(agent);
        // Обновление максимального и минимального значений с использованием атомарных операций
        atomicMax(maxOf_dev, OF[bx]);
        atomicMin(minOf_dev, OF[bx]);
    }
}

__global__ void go_min_max(double* pheromon) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (параметра) 
    if (tx < PARAMETR_SIZE * MAX_VALUE_SIZE) {
        if (pheromon[tx] > PAR_MAX_ALG_MINMAX) { pheromon[tx] = PAR_MAX_ALG_MINMAX; }
        if (pheromon[tx] < PAR_MIN_ALG_MINMAX) { pheromon[tx] = PAR_MIN_ALG_MINMAX; }
    }
}

// Функция для загрузки матрицы из файла
bool load_matrix(const std::string& filename, double* parametr_value, double* pheromon_value, double* kol_enter_value)
{
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Don't open file!" << std::endl;
        return false;
    }
    for (int i = 0; i < PARAMETR_SIZE; ++i) {
        for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
            int k = MAX_VALUE_SIZE * i + j;
            if (!(infile >> parametr_value[k])) { // Чтение элемента в массив a
                std::cerr << "Error load element [" << i << "][" << j << "]" << std::endl;
                return false;
            }
            if (parametr_value[k] != -100) {
                pheromon_value[k] = 1.0; // Присваиваем значение pheromon_value
                kol_enter_value[k] = 1.0;
            }
            else {
                pheromon_value[k] = 0.0; // Присваиваем значение pheromon_value
                parametr_value[k] = 0.0; //Нужно ли????
                kol_enter_value[k] = 0.0;
            }
        }
    }
    infile.close();
    return true;
}
bool load_matrix_transp(const std::string& filename, double* parametr_value, double* pheromon_value, double* kol_enter_value)
{
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Don't open file!" << std::endl;
        return false;
    }
    for (int i = 0; i < PARAMETR_SIZE; ++i) {
        for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
            int k = i + j * PARAMETR_SIZE;
            if (!(infile >> parametr_value[k])) { // Чтение элемента в массив a
                std::cerr << "Error load element [" << i << "][" << j << "]" << std::endl;
                return false;
            }

            if (parametr_value[k] != -100) {
                pheromon_value[k] = 1.0; // Присваиваем значение pheromon_value
                kol_enter_value[k] = 1.0;
            }
            else {
                pheromon_value[k] = 0.0; // Присваиваем значение pheromon_value
                parametr_value[k] = 0.0; //Нужно ли????
                kol_enter_value[k] = 0.0;
            }
        }
    }
    infile.close();
    return true;
}

static int start_CUDA() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t stop, start, startAll;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&start));//IZE * sizeof(double);

    float gpuTime = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    float AllgpuTime = 0.0f, AllgpuTime1 = 0.0f, SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumgpuTime4 = 0.0f, SumgpuTime5 = 0.0f, SumgpuTime6 = 0.0f, SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;

    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
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

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    int threads_init_hash = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threads_init_hash - 1) / threads_init_hash;
    initializeHashTable << <blocks_init_hash, threads_init_hash >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    dim3 kol_ant(ANT_SIZE);
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    if (PARAMETR_SIZE < MAX_THREAD_CUDA) {
        for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
            CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
            go_mass_probability_thread << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (CPU_RANDOM) {
                //Создание множества случайных чисел на итерации
                for (int i = 0; i < kolBytes_random_value; ++i) {
                    random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
                }
                if (PRINT_INFORMATION) {
                    std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                    for (int i = 0; i < ANT_SIZE; ++i) {
                        for (int j = 0; j < PARAMETR_SIZE; ++j) {
                            std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                        }
                        std::cout << std::endl;

                    }
                }
                CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
            }
            go_all_agent << <kol_ant, kol_parametr >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            add_pheromon_iteration_thread << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
            i_gpuTime = int(int(gpuTime * 1000) % 10000000);

            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            if ((nom_iter + 1) % kol_shag_stat == 0) {
                int NomStatistics = nom_iter / kol_shag_stat;
                update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, *global_minOf_in_device, *global_maxOf_in_device, *kol_hash_fail_in_device / PARAMETR_SIZE);
            }
        }
    }
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

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
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));
    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    std::cout << "Time CUDA:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumgpuTime4 << ";" << SumgpuTime5 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumgpuTime4 << ";" << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_Time() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
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
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
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

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    int threads_init_hash = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threads_init_hash - 1) / threads_init_hash;
    initializeHashTable << <blocks_init_hash, threads_init_hash >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
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
            CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
            CUDA_CHECK(cudaEventRecord(start1));
            go_mass_probability_thread << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (PRINT_INFORMATION) {
                CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
                for (int i = 0; i < PARAMETR_SIZE; ++i) {
                    for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                        std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                    }
                    std::cout << std::endl; // Переход на новую строку
                }
            }
            CUDA_CHECK(cudaEventRecord(start2));
            if (CPU_RANDOM) {
                //Создание множества случайных чисел на итерации
                for (int i = 0; i < kolBytes_random_value; ++i) {
                    random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
                }
                if (PRINT_INFORMATION) {
                    std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                    for (int i = 0; i < ANT_SIZE; ++i) {
                        for (int j = 0; j < PARAMETR_SIZE; ++j) {
                            std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                        }
                        std::cout << std::endl;

                    }
                }
                CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
            }
            go_all_agent << <kol_ant, kol_parametr >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (PRINT_INFORMATION) {
                CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
                std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                        //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                    }
                    std::cout << "-> " << antOF[i] << std::endl;

                }
            }

            CUDA_CHECK(cudaEventRecord(start3));
            add_pheromon_iteration_thread << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
            SumgpuTime1 = SumgpuTime1 + gpuTime1;
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
            SumgpuTime2 = SumgpuTime2 + gpuTime2;
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime3, start3, stop));
            SumgpuTime3 = SumgpuTime3 + gpuTime3;
            i_gpuTime = int(int(gpuTime * 1000) % 10000000);
            double maxOf = -INT16_MAX;
            double minOf = INT16_MAX;

            if (PRINT_INFORMATION) {
                std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                    }
                    std::cout << "-> " << antOF[i] << std::endl;
                    */
                    if (antOF[i] > maxOf) {
                        maxOf = antOF[i];
                    }
                    if (antOF[i] < minOf) {
                        minOf = antOF[i];
                    }
                }
                if (minOf < global_minOf) {
                    global_minOf = minOf;
                }
                if (maxOf > global_maxOf) {
                    global_maxOf = maxOf;
                }
                std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            if ((nom_iter + 1) % kol_shag_stat == 0) {
                int NomStatistics = nom_iter / kol_shag_stat;
                if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
                update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, *global_minOf_in_device, *global_maxOf_in_device, *kol_hash_fail_in_device / PARAMETR_SIZE);
            }
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));

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
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA Time:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumgpuTime4 << ";" << SumgpuTime5 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA Time:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumgpuTime4 << ";" << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_Const() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
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
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    int threads_init_hash = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threads_init_hash - 1) / threads_init_hash;
    initializeHashTable << <blocks_init_hash, threads_init_hash >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    dim3 kol_ant(ANT_SIZE);

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    //CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(parametr_value_dev_const, parametr_value, numBytes_matrix_graph));   // Копирование в константную память
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    if (PARAMETR_SIZE < MAX_THREAD_CUDA) {
        for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
            CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
            CUDA_CHECK(cudaEventRecord(start1));
            go_mass_probability_thread << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (PRINT_INFORMATION) {
                CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
                for (int i = 0; i < PARAMETR_SIZE; ++i) {
                    for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                        std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                    }
                    std::cout << std::endl; // Переход на новую строку
                }
            }
            CUDA_CHECK(cudaEventRecord(start2));
            if (CPU_RANDOM) {
                //Создание множества случайных чисел на итерации
                for (int i = 0; i < kolBytes_random_value; ++i) {
                    random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
                }
                if (PRINT_INFORMATION) {
                    std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                    for (int i = 0; i < ANT_SIZE; ++i) {
                        for (int j = 0; j < PARAMETR_SIZE; ++j) {
                            std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                        }
                        std::cout << std::endl;

                    }
                }
                CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
            }
            go_all_agent_const << <kol_ant, kol_parametr >> > (norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (PRINT_INFORMATION) {
                CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
                std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                        //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                    }
                    std::cout << "-> " << antOF[i] << std::endl;

                }
            }

            CUDA_CHECK(cudaEventRecord(start3));
            add_pheromon_iteration_thread << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
            SumgpuTime1 = SumgpuTime1 + gpuTime1;
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
            SumgpuTime2 = SumgpuTime2 + gpuTime2;
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime3, start3, stop));
            SumgpuTime3 = SumgpuTime3 + gpuTime3;
            i_gpuTime = int(int(gpuTime * 1000) % 10000000);
            double maxOf = -INT16_MAX;
            double minOf = INT16_MAX;

            if (PRINT_INFORMATION) {
                std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                    }
                    std::cout << "-> " << antOF[i] << std::endl;
                    */
                    if (antOF[i] > maxOf) {
                        maxOf = antOF[i];
                    }
                    if (antOF[i] < minOf) {
                        minOf = antOF[i];
                    }
                }
                if (minOf < global_minOf) {
                    global_minOf = minOf;
                }
                if (maxOf > global_maxOf) {
                    global_maxOf = maxOf;
                }
                std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            if ((nom_iter + 1) % kol_shag_stat == 0) {
                int NomStatistics = nom_iter / kol_shag_stat;
                if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
                update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, *global_minOf_in_device, *global_maxOf_in_device, *kol_hash_fail_in_device / PARAMETR_SIZE);
            }
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));

    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA Const:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumgpuTime4 << ";" << SumgpuTime5 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA Const:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumgpuTime4 << ";" << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_only_block_Time() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
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
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
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

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    int threads_init_hash = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threads_init_hash - 1) / threads_init_hash;
    initializeHashTable << <blocks_init_hash, threads_init_hash >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    dim3 kol_ant(ANT_SIZE);

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1));
        go_mass_probability_block << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        go_all_agent_only_block << <kol_ant, 1 >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        CUDA_CHECK(cudaEventRecord(start3));
        add_pheromon_iteration_block << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime3, start3, stop));
        SumgpuTime3 = SumgpuTime3 + gpuTime3;
        i_gpuTime = int(int(gpuTime * 1000) % 10000000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << "-> " << antOF[i] << std::endl;
                */
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
            if (minOf < global_minOf) {
                global_minOf = minOf;
            }
            if (maxOf > global_maxOf) {
                global_maxOf = maxOf;
            }
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, *global_minOf_in_device, *global_maxOf_in_device, *kol_hash_fail_in_device / PARAMETR_SIZE);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));

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
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA Time only block:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumgpuTime4 << ";" << SumgpuTime5 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA Time only block:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumgpuTime4 << ";" << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_non_hash() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
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
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;

    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
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

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));
    //CUDA_CHECK(cudaMalloc((void**)&cache_dev, TABLE_SIZE * sizeof(HashEntry)));

    // Установка конфигурации запуска ядра
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
            CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
            CUDA_CHECK(cudaEventRecord(start1, 0));
            go_mass_probability_thread << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (PRINT_INFORMATION) {
                CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
                for (int i = 0; i < PARAMETR_SIZE; ++i) {
                    for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                        std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                    }
                    std::cout << std::endl; // Переход на новую строку
                }
            }
            CUDA_CHECK(cudaEventRecord(start2, 0));

            go_all_agent_non_hash << <kol_ant, kol_parametr >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, maxOf_dev, minOf_dev, kol_hash_fail);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (PRINT_INFORMATION) {
                CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
                std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                        //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                    }
                    std::cout << "-> " << antOF[i] << std::endl;

                }
            }

            CUDA_CHECK(cudaEventRecord(start3, 0));
            add_pheromon_iteration_thread << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
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
            double maxOf = -INT16_MAX;
            double minOf = INT16_MAX;

            if (PRINT_INFORMATION) {
                std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                    }
                    std::cout << "-> " << antOF[i] << std::endl;
                    */
                    if (antOF[i] > maxOf) {
                        maxOf = antOF[i];
                    }
                    if (antOF[i] < minOf) {
                        minOf = antOF[i];
                    }
                }
                if (minOf < global_minOf) {
                    global_minOf = minOf;
                }
                if (maxOf > global_maxOf) {
                    global_maxOf = maxOf;
                }
                CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
                std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            if ((nom_iter + 1) % kol_shag_stat == 0) {
                int NomStatistics = nom_iter / kol_shag_stat;
                if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
                update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device / PARAMETR_SIZE);
            }
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));

    CUDA_CHECK(cudaFree(parametr_value_dev));
    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_ant() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
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
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;

    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
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
    int numBlocks = 0;
    int numThreads = 0;

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_block << <kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        go_all_agent_only << <numBlocks, numThreads >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j] << " ";
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        CUDA_CHECK(cudaEventRecord(start3, 0));
        add_pheromon_iteration_block << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime3, start3, stop));
        SumgpuTime3 = SumgpuTime3 + gpuTime3;
        i_gpuTime = int(gpuTime * 1000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(gpuTime * 1000) << "x" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << "-> " << antOF[i] << std::endl;
                */
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
            if (minOf < global_minOf) {
                global_minOf = minOf;
            }
            if (maxOf > global_maxOf) {
                global_maxOf = maxOf;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));

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
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_ant_Const() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
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
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;

    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int numBlocks = 0;
    int numThreads = 0;

    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(parametr_value_dev_const, parametr_value, numBytes_matrix_graph));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_block << <kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        go_all_agent_only_const << <numBlocks, numThreads >> > (norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }
        CUDA_CHECK(cudaEventRecord(start3, 0));
        add_pheromon_iteration_block << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime3, start3, stop));
        SumgpuTime3 = SumgpuTime3 + gpuTime3;
        i_gpuTime = int(gpuTime * 1000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(gpuTime * 1000) << "x" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << "-> " << antOF[i] << std::endl;
                */
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
            if (minOf < global_minOf) {
                global_minOf = minOf;
            }
            if (maxOf > global_maxOf) {
                global_maxOf = maxOf;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));

    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant const:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant const:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_ant_decrease_par_Const() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, start4, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&startAll1)); CUDA_CHECK(cudaEventCreate(&start1)); CUDA_CHECK(cudaEventCreate(&start2)); CUDA_CHECK(cudaEventCreate(&start3)); CUDA_CHECK(cudaEventCreate(&start4)); CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f; int i_gpuTime = 0; float AllgpuTime = 0.0f; float AllgpuTime1 = 0.0f; float gpuTime1 = 0.0f; float gpuTime2 = 0.0f; float gpuTime3 = 0.0f; float gpuTime4 = 0.0f; float SumgpuTime1 = 0.0f;  float SumgpuTime2 = 0.0f; float SumgpuTime3 = 0.0f; float SumgpuTime4 = 0.0f; float SumgpuTime5 = 0.0f; float SumgpuTime6 = 0.0f; float SumgpuTime7 = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;

    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int numBlocks = 0;
    int numThreads = 0;
    int numBlocksDecrease = 0;
    int numThreadsDecrease = 0;

    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    if (threadsPerBlock < PARAMETR_SIZE) {
        numBlocksDecrease = (PARAMETR_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreadsDecrease = MAX_THREAD_CUDA;
    }
    else {
        numBlocksDecrease = 1;
        numThreadsDecrease = PARAMETR_SIZE;
    }

    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(parametr_value_dev_const, parametr_value, numBytes_matrix_graph));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_block << <kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        go_all_agent_only_const << <numBlocks, numThreads >> > (norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }
        CUDA_CHECK(cudaEventRecord(start3, 0));
        // add_pheromon_iteration_block << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        decrease_pheromon_iteration << < numBlocksDecrease, numThreadsDecrease >> > (pheromon_value_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        CUDA_CHECK(cudaEventRecord(start4, 0));
        only_add_pheromon_iteration_par_only << < numBlocksDecrease, numThreadsDecrease >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        //add_pheromon_iteration_block << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime3, start3, stop));
        SumgpuTime3 = SumgpuTime3 + gpuTime3;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime4, start4, stop));
        SumgpuTime4 = SumgpuTime4 + gpuTime4;
        i_gpuTime = int(gpuTime * 1000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(gpuTime * 1000) << "x" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << "-> " << antOF[i] << std::endl;
                */
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
            if (minOf < global_minOf) {
                global_minOf = minOf;
            }
            if (maxOf > global_maxOf) {
                global_maxOf = maxOf;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));
    CUDA_CHECK(cudaEventDestroy(start4));

    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant const delt par:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumgpuTime4 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant const delt par:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumgpuTime4 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}
static int start_CUDA_ant_decrease_par_block_Const() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, start4, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&startAll1)); CUDA_CHECK(cudaEventCreate(&start1)); CUDA_CHECK(cudaEventCreate(&start2)); CUDA_CHECK(cudaEventCreate(&start3)); CUDA_CHECK(cudaEventCreate(&start4)); CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f; int i_gpuTime = 0; float AllgpuTime = 0.0f; float AllgpuTime1 = 0.0f; float gpuTime1 = 0.0f; float gpuTime2 = 0.0f; float gpuTime3 = 0.0f; float gpuTime4 = 0.0f; float SumgpuTime1 = 0.0f;  float SumgpuTime2 = 0.0f; float SumgpuTime3 = 0.0f; float SumgpuTime4 = 0.0f; float SumgpuTime5 = 0.0f; float SumgpuTime6 = 0.0f; float SumgpuTime7 = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;

    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int numBlocks = 0;
    int numThreads = 0;
    int numBlocksDecrease = 0;
    int numThreadsDecrease = 0;

    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    if (threadsPerBlock < PARAMETR_SIZE * MAX_VALUE_SIZE) {
        numBlocksDecrease = (PARAMETR_SIZE * MAX_VALUE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreadsDecrease = MAX_THREAD_CUDA;
    }
    else {
        numBlocksDecrease = 1;
        numThreadsDecrease = PARAMETR_SIZE * MAX_VALUE_SIZE;
    }

    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(parametr_value_dev_const, parametr_value, numBytes_matrix_graph));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_block << <kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        go_all_agent_only_const << <numBlocks, numThreads >> > (norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }
        CUDA_CHECK(cudaEventRecord(start3, 0));
        // add_pheromon_iteration_block << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        decrease_pheromon_iteration << < numBlocksDecrease, numThreadsDecrease >> > (pheromon_value_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        CUDA_CHECK(cudaEventRecord(start4, 0));
        only_add_pheromon_iteration_par_block_only << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        //add_pheromon_iteration_block << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime3, start3, stop));
        SumgpuTime3 = SumgpuTime3 + gpuTime3;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime4, start4, stop));
        SumgpuTime4 = SumgpuTime4 + gpuTime4;
        i_gpuTime = int(gpuTime * 1000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(gpuTime * 1000) << "x" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << "-> " << antOF[i] << std::endl;
                */
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
            if (minOf < global_minOf) {
                global_minOf = minOf;
            }
            if (maxOf > global_maxOf) {
                global_maxOf = maxOf;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));
    CUDA_CHECK(cudaEventDestroy(start4));

    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant const delt par block:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumgpuTime4 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant const delt par block:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumgpuTime4 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}
static int start_CUDA_ant_decrease_ant_Const() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, start4, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&startAll1)); CUDA_CHECK(cudaEventCreate(&start1)); CUDA_CHECK(cudaEventCreate(&start2)); CUDA_CHECK(cudaEventCreate(&start3)); CUDA_CHECK(cudaEventCreate(&start4)); CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f; int i_gpuTime = 0; float AllgpuTime = 0.0f; float AllgpuTime1 = 0.0f; float gpuTime1 = 0.0f; float gpuTime2 = 0.0f; float gpuTime3 = 0.0f; float gpuTime4 = 0.0f; float SumgpuTime1 = 0.0f;  float SumgpuTime2 = 0.0f; float SumgpuTime3 = 0.0f; float SumgpuTime4 = 0.0f; float SumgpuTime5 = 0.0f; float SumgpuTime6 = 0.0f; float SumgpuTime7 = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;

    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int numBlocks = 0;
    int numThreads = 0;
    int numBlocksDecrease = 0;
    int numThreadsDecrease = 0;

    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    if (threadsPerBlock < PARAMETR_SIZE * MAX_VALUE_SIZE) {
        numBlocksDecrease = (PARAMETR_SIZE * MAX_VALUE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreadsDecrease = MAX_THREAD_CUDA;
    }
    else {
        numBlocksDecrease = 1;
        numThreadsDecrease = PARAMETR_SIZE * MAX_VALUE_SIZE;
    }


    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(parametr_value_dev_const, parametr_value, numBytes_matrix_graph));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_block << <kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        go_all_agent_only_const << <numBlocks, numThreads >> > (norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }
        CUDA_CHECK(cudaEventRecord(start3, 0));
        // add_pheromon_iteration_block << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        decrease_pheromon_iteration << < numBlocksDecrease, numThreadsDecrease >> > (pheromon_value_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        CUDA_CHECK(cudaEventRecord(start4, 0));
        //only_add_pheromon_iteration_par_only << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev); //Плохо из-за необходимости случайного обращения к глобальной памяти 
        only_add_pheromon_iteration_ant_only << < numBlocks, numThreads >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        //add_pheromon_iteration_block << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime3, start3, stop));
        SumgpuTime3 = SumgpuTime3 + gpuTime3;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime4, start4, stop));
        SumgpuTime4 = SumgpuTime4 + gpuTime4;
        i_gpuTime = int(gpuTime * 1000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(gpuTime * 1000) << "x" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << "-> " << antOF[i] << std::endl;
                */
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
            if (minOf < global_minOf) {
                global_minOf = minOf;
            }
            if (maxOf > global_maxOf) {
                global_maxOf = maxOf;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));
    CUDA_CHECK(cudaEventDestroy(start4));

    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant const delt ant:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumgpuTime4 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant const delt ant:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumgpuTime4 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}
static int start_CUDA_ant_decrease_ant_block_Const() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, start4, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&startAll1)); CUDA_CHECK(cudaEventCreate(&start1)); CUDA_CHECK(cudaEventCreate(&start2)); CUDA_CHECK(cudaEventCreate(&start3)); CUDA_CHECK(cudaEventCreate(&start4)); CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f; int i_gpuTime = 0; float AllgpuTime = 0.0f; float AllgpuTime1 = 0.0f; float gpuTime1 = 0.0f; float gpuTime2 = 0.0f; float gpuTime3 = 0.0f; float gpuTime4 = 0.0f; float SumgpuTime1 = 0.0f;  float SumgpuTime2 = 0.0f; float SumgpuTime3 = 0.0f; float SumgpuTime4 = 0.0f; float SumgpuTime5 = 0.0f; float SumgpuTime6 = 0.0f; float SumgpuTime7 = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;

    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int numBlocks = 0;
    int numThreads = 0;
    int numBlocksDecrease = 0;
    int numThreadsDecrease = 0;

    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    if (threadsPerBlock < PARAMETR_SIZE * MAX_VALUE_SIZE) {
        numBlocksDecrease = (PARAMETR_SIZE * MAX_VALUE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreadsDecrease = MAX_THREAD_CUDA;
    }
    else {
        numBlocksDecrease = 1;
        numThreadsDecrease = PARAMETR_SIZE * MAX_VALUE_SIZE;
    }

    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    dim3 kol_ant(ANT_SIZE);

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(parametr_value_dev_const, parametr_value, numBytes_matrix_graph));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_block << <kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        go_all_agent_only_const << <numBlocks, numThreads >> > (norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }
        CUDA_CHECK(cudaEventRecord(start3, 0));
        // add_pheromon_iteration_block << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        decrease_pheromon_iteration << < numBlocksDecrease, numThreadsDecrease >> > (pheromon_value_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        CUDA_CHECK(cudaEventRecord(start4, 0));
        //only_add_pheromon_iteration_par_only << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev); //Плохо из-за необходимости случайного обращения к глобальной памяти 
        only_add_pheromon_iteration_ant_block_only << < kol_ant, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        //add_pheromon_iteration_block << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime3, start3, stop));
        SumgpuTime3 = SumgpuTime3 + gpuTime3;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime4, start4, stop));
        SumgpuTime4 = SumgpuTime4 + gpuTime4;
        i_gpuTime = int(gpuTime * 1000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(gpuTime * 1000) << "x" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << "-> " << antOF[i] << std::endl;
                */
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
            if (minOf < global_minOf) {
                global_minOf = minOf;
            }
            if (maxOf > global_maxOf) {
                global_maxOf = maxOf;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));
    CUDA_CHECK(cudaEventDestroy(start4));

    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant const delt ant block:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumgpuTime4 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant const delt ant block:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumgpuTime4 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}
static int start_CUDA_ant_decrease_ant_par_Const() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, start4, start5, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&startAll1)); CUDA_CHECK(cudaEventCreate(&start1)); CUDA_CHECK(cudaEventCreate(&start2)); CUDA_CHECK(cudaEventCreate(&start3)); CUDA_CHECK(cudaEventCreate(&start4)); CUDA_CHECK(cudaEventCreate(&start5)); CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f; int i_gpuTime = 0; float AllgpuTime = 0.0f; float AllgpuTime1 = 0.0f; float gpuTime1 = 0.0f; float gpuTime2 = 0.0f; float gpuTime3 = 0.0f; float gpuTime4 = 0.0f; float gpuTime5 = 0.0f; float SumgpuTime1 = 0.0f;  float SumgpuTime2 = 0.0f; float SumgpuTime3 = 0.0f; float SumgpuTime4 = 0.0f; float SumgpuTime5 = 0.0f; float SumgpuTime6 = 0.0f; float SumgpuTime7 = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;

    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int numBlocks = 0;
    int numThreads = 0;
    int numBlocksDecrease = 0;
    int numThreadsDecrease = 0;

    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    if (threadsPerBlock < PARAMETR_SIZE * MAX_VALUE_SIZE) {
        numBlocksDecrease = (PARAMETR_SIZE * MAX_VALUE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreadsDecrease = MAX_THREAD_CUDA;
    }
    else {
        numBlocksDecrease = 1;
        numThreadsDecrease = PARAMETR_SIZE * MAX_VALUE_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    dim3 kol_ant(ANT_SIZE);

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(parametr_value_dev_const, parametr_value, numBytes_matrix_graph));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    if (PARAMETR_SIZE < MAX_THREAD_CUDA) {
        for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
            CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
            CUDA_CHECK(cudaEventRecord(start1, 0));
            go_mass_probability_block << <kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (PRINT_INFORMATION) {
                CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
                for (int i = 0; i < PARAMETR_SIZE; ++i) {
                    for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                        std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                    }
                    std::cout << std::endl; // Переход на новую строку
                }
            }
            CUDA_CHECK(cudaEventRecord(start2, 0));
            if (CPU_RANDOM) {
                //Создание множества случайных чисел на итерации
                for (int i = 0; i < kolBytes_random_value; ++i) {
                    random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
                }
                if (PRINT_INFORMATION) {
                    std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                    for (int i = 0; i < ANT_SIZE; ++i) {
                        for (int j = 0; j < PARAMETR_SIZE; ++j) {
                            std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                        }
                        std::cout << std::endl;

                    }
                }
                CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
            }
            go_all_agent_only_const << <numBlocks, numThreads >> > (norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (PRINT_INFORMATION) {
                CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
                std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                        //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                    }
                    std::cout << "-> " << antOF[i] << std::endl;

                }
            }
            CUDA_CHECK(cudaEventRecord(start3, 0));
            // add_pheromon_iteration_block << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
            decrease_pheromon_iteration << < numBlocksDecrease, numThreadsDecrease >> > (pheromon_value_dev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            CUDA_CHECK(cudaEventRecord(start4, 0));
            //only_add_pheromon_iteration_par_only << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev); //Плохо из-за необходимости случайного обращения к глобальной памяти 
            only_add_pheromon_iteration_ant_par_thread << < kol_ant, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
            //add_pheromon_iteration_block << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            CUDA_CHECK(cudaEventRecord(start5, 0));
            if (GO_ALG_MINMAX) { go_min_max << < numBlocksDecrease, numThreadsDecrease >> > (pheromon_value_dev); }
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
            SumgpuTime1 = SumgpuTime1 + gpuTime1;
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
            SumgpuTime2 = SumgpuTime2 + gpuTime2;
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime3, start3, stop));
            SumgpuTime3 = SumgpuTime3 + gpuTime3;
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime4, start4, stop));
            SumgpuTime4 = SumgpuTime4 + gpuTime4;
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime5, start5, stop));
            SumgpuTime5 = SumgpuTime5 + gpuTime5;
            i_gpuTime = int(gpuTime * 1000);
            double maxOf = -INT16_MAX;
            double minOf = INT16_MAX;

            if (PRINT_INFORMATION) {
                std::cout << "h_seeds (" << int(gpuTime * 1000) << "x" << ANT_SIZE << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                    }
                    std::cout << "-> " << antOF[i] << std::endl;
                    */
                    if (antOF[i] > maxOf) {
                        maxOf = antOF[i];
                    }
                    if (antOF[i] < minOf) {
                        minOf = antOF[i];
                    }
                }
                if (minOf < global_minOf) {
                    global_minOf = minOf;
                }
                if (maxOf > global_maxOf) {
                    global_maxOf = maxOf;
                }
                CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
                std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            if ((nom_iter + 1) % kol_shag_stat == 0) {
                int NomStatistics = nom_iter / kol_shag_stat;
                if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
                update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
            }
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));
    CUDA_CHECK(cudaEventDestroy(start4));
    CUDA_CHECK(cudaEventDestroy(start5));

    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant const delt ant par:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumgpuTime4 << "; " << SumgpuTime5 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant const delt  ant par:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumgpuTime4 << "; " << SumgpuTime5 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}
static int start_CUDA_ant_decrease_ant_par_transp_Const() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, start4, start5, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&startAll1)); CUDA_CHECK(cudaEventCreate(&start1)); CUDA_CHECK(cudaEventCreate(&start2)); CUDA_CHECK(cudaEventCreate(&start3)); CUDA_CHECK(cudaEventCreate(&start4)); CUDA_CHECK(cudaEventCreate(&start5)); CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f; int i_gpuTime = 0; float AllgpuTime = 0.0f; float AllgpuTime1 = 0.0f; float gpuTime1 = 0.0f; float gpuTime2 = 0.0f; float gpuTime3 = 0.0f; float gpuTime4 = 0.0f; float gpuTime5 = 0.0f; float SumgpuTime1 = 0.0f;  float SumgpuTime2 = 0.0f; float SumgpuTime3 = 0.0f; float SumgpuTime4 = 0.0f; float SumgpuTime5 = 0.0f; float SumgpuTime6 = 0.0f; float SumgpuTime7 = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;

    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int numBlocks = 0;
    int numThreads = 0;
    int numBlocksDecrease = 0;
    int numThreadsDecrease = 0;

    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    if (threadsPerBlock < PARAMETR_SIZE * MAX_VALUE_SIZE) {
        numBlocksDecrease = (PARAMETR_SIZE * MAX_VALUE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreadsDecrease = MAX_THREAD_CUDA;
    }
    else {
        numBlocksDecrease = 1;
        numThreadsDecrease = PARAMETR_SIZE * MAX_VALUE_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    dim3 kol_ant(ANT_SIZE);

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(parametr_value_dev_const, parametr_value, numBytes_matrix_graph));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    if (ANT_SIZE < MAX_THREAD_CUDA) {
        for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
            CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
            CUDA_CHECK(cudaEventRecord(start1, 0));
            go_mass_probability_block << <kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (PRINT_INFORMATION) {
                CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
                for (int i = 0; i < PARAMETR_SIZE; ++i) {
                    for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                        std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                    }
                    std::cout << std::endl; // Переход на новую строку
                }
            }
            CUDA_CHECK(cudaEventRecord(start2, 0));
            if (CPU_RANDOM) {
                //Создание множества случайных чисел на итерации
                for (int i = 0; i < kolBytes_random_value; ++i) {
                    random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
                }
                if (PRINT_INFORMATION) {
                    std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                    for (int i = 0; i < ANT_SIZE; ++i) {
                        for (int j = 0; j < PARAMETR_SIZE; ++j) {
                            std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                        }
                        std::cout << std::endl;

                    }
                }
                CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
            }
            go_all_agent_only_const << <numBlocks, numThreads >> > (norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (PRINT_INFORMATION) {
                CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
                std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                        //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                    }
                    std::cout << "-> " << antOF[i] << std::endl;

                }
            }
            CUDA_CHECK(cudaEventRecord(start3, 0));
            // add_pheromon_iteration_block << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
            decrease_pheromon_iteration << < numBlocksDecrease, numThreadsDecrease >> > (pheromon_value_dev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            CUDA_CHECK(cudaEventRecord(start4, 0));
            only_add_pheromon_iteration_ant_par_thread_transp << <  kol_parametr, kol_ant >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            CUDA_CHECK(cudaEventRecord(start5, 0));
            if (GO_ALG_MINMAX) { go_min_max << < numBlocksDecrease, numThreadsDecrease >> > (pheromon_value_dev); }
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
            SumgpuTime1 = SumgpuTime1 + gpuTime1;
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
            SumgpuTime2 = SumgpuTime2 + gpuTime2;
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime3, start3, stop));
            SumgpuTime3 = SumgpuTime3 + gpuTime3;
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime4, start4, stop));
            SumgpuTime4 = SumgpuTime4 + gpuTime4;
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime5, start5, stop));
            SumgpuTime5 = SumgpuTime5 + gpuTime5;
            i_gpuTime = int(gpuTime * 1000);
            double maxOf = -INT16_MAX;
            double minOf = INT16_MAX;

            if (PRINT_INFORMATION) {
                std::cout << "h_seeds (" << int(gpuTime * 1000) << "x" << ANT_SIZE << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                    }
                    std::cout << "-> " << antOF[i] << std::endl;
                    */
                    if (antOF[i] > maxOf) {
                        maxOf = antOF[i];
                    }
                    if (antOF[i] < minOf) {
                        minOf = antOF[i];
                    }
                }
                if (minOf < global_minOf) {
                    global_minOf = minOf;
                }
                if (maxOf > global_maxOf) {
                    global_maxOf = maxOf;
                }
                CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
                std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            if ((nom_iter + 1) % kol_shag_stat == 0) {
                int NomStatistics = nom_iter / kol_shag_stat;
                if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
                update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
            }
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));
    CUDA_CHECK(cudaEventDestroy(start4));
    CUDA_CHECK(cudaEventDestroy(start5));

    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant const delt ant par transp:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumgpuTime4 << "; " << SumgpuTime5 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant const delt  ant par transp:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumgpuTime4 << "; " << SumgpuTime5 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

//объявление функций CPU
void go_mass_probability_non_cuda(double* pheromon, double* kol_enter, double* norm_matrix_probability);
void add_pheromon_iteration_non_cuda(double* pheromon, double* kol_enter, int* agent_node, double* OF);

static int start_CUDA_ant_add_CPU_Time() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start2, start5, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&start5));
    CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float gpuTime5 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;

    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
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
    int numBlocks = 0;
    int numThreads = 0;

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        auto start1 = std::chrono::high_resolution_clock::now();
        //CUDA_CHECK(cudaEventRecord(start1, 0);
        //go_mass_probability_block << <kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        go_mass_probability_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);
        auto end_go_mass_probability = std::chrono::high_resolution_clock::now();
        SumgpuTime4 += std::chrono::duration<float, std::milli>(end_go_mass_probability - start1).count();
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaMemcpy(norm_matrix_probability_dev, norm_matrix_probability, numBytes_matrix_graph, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(start5, 0));
        auto start6 = std::chrono::high_resolution_clock::now();
        go_all_agent_only << <numBlocks, numThreads >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        auto end_iter_6 = std::chrono::high_resolution_clock::now();
        SumgpuTime6 += std::chrono::duration<float, std::milli>(end_iter_6 - start6).count();
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&gpuTime5, start5, stop);
        SumgpuTime5 = SumgpuTime5 + gpuTime5;
        CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        auto start3 = std::chrono::high_resolution_clock::now();
        // Обновление весов-феромонов
        add_pheromon_iteration_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);
        auto end_iter = std::chrono::high_resolution_clock::now();
        SumgpuTime1 += std::chrono::duration<float, std::milli>(end_iter - start1).count();
        SumgpuTime3 += std::chrono::duration<float, std::milli>(end_iter - start3).count();
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        //CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        //SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        //CUDA_CHECK(cudaEventElapsedTime(&gpuTime3, start3, stop));
        //SumgpuTime3 = SumgpuTime3 + gpuTime3;
        i_gpuTime = int(int(gpuTime * 1000) % 10000000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << "-> " << antOF[i] << std::endl;
                */
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }

            if (minOf < global_minOf) {
                global_minOf = minOf;
            }
            if (maxOf > global_maxOf) {
                global_maxOf = maxOf;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, *global_minOf_in_device, *global_maxOf_in_device, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start2));
    cudaEventDestroy(start5);

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
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant add CPU Time:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant add CPU Time:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_ant_add_CPU_Time_global() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start2, start5, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&start5));
    CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float gpuTime5 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;

    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* parametr_value_dev = nullptr;
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* ant_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int numBlocks = 0;
    int numThreads = 0;

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&ant_dev, numBytes_matrix_ant));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        auto start1 = std::chrono::high_resolution_clock::now();
        //CUDA_CHECK(cudaEventRecord(start1, 0);
        //go_mass_probability_block << <kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        go_mass_probability_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);
        auto end_go_mass_probability = std::chrono::high_resolution_clock::now();
        SumgpuTime4 += std::chrono::duration<float, std::milli>(end_go_mass_probability - start1).count();

        CUDA_CHECK(cudaEventRecord(start2, 0));

        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaMemcpy(norm_matrix_probability_dev, norm_matrix_probability, numBytes_matrix_graph, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(start5, 0));
        auto start6 = std::chrono::high_resolution_clock::now();
        go_all_agent_only_global << <numBlocks, numThreads >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        auto end_iter_6 = std::chrono::high_resolution_clock::now();
        SumgpuTime6 += std::chrono::duration<float, std::milli>(end_iter_6 - start6).count();
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&gpuTime5, start5, stop);
        SumgpuTime5 = SumgpuTime5 + gpuTime5;
        CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));

        auto start3 = std::chrono::high_resolution_clock::now();
        // Обновление весов-феромонов
        add_pheromon_iteration_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);
        auto end_iter = std::chrono::high_resolution_clock::now();
        SumgpuTime1 += std::chrono::duration<float, std::milli>(end_iter - start1).count();
        SumgpuTime3 += std::chrono::duration<float, std::milli>(end_iter - start3).count();
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        //CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        //SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        //CUDA_CHECK(cudaEventElapsedTime(&gpuTime3, start3, stop));
        //SumgpuTime3 = SumgpuTime3 + gpuTime3;
        i_gpuTime = int(int(gpuTime * 1000) % 10000000);
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, *global_minOf_in_device, *global_maxOf_in_device, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start2));
    cudaEventDestroy(start5);

    CUDA_CHECK(cudaFree(parametr_value_dev));
    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(ant_dev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant add CPU Time global:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant add CPU Time global:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_ant_add_CPU_Const() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start2, start5, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&start5));
    CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float gpuTime5 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;

    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int numBlocks = 0;
    int numThreads = 0;

    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(parametr_value_dev_const, parametr_value, numBytes_matrix_graph));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        auto start1 = std::chrono::high_resolution_clock::now();
        //CUDA_CHECK(cudaEventRecord(start1, 0);
        //go_mass_probability_block << <kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        go_mass_probability_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);
        auto end_go_mass_probability = std::chrono::high_resolution_clock::now();
        SumgpuTime4 += std::chrono::duration<float, std::milli>(end_go_mass_probability - start1).count();
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaMemcpy(norm_matrix_probability_dev, norm_matrix_probability, numBytes_matrix_graph, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(start5, 0));
        auto start6 = std::chrono::high_resolution_clock::now();
        go_all_agent_only_const << <numBlocks, numThreads >> > (norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        auto end_iter_6 = std::chrono::high_resolution_clock::now();
        SumgpuTime6 += std::chrono::duration<float, std::milli>(end_iter_6 - start6).count();
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&gpuTime5, start5, stop);
        SumgpuTime5 = SumgpuTime5 + gpuTime5;
        CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        auto start3 = std::chrono::high_resolution_clock::now();
        // Обновление весов-феромонов
        add_pheromon_iteration_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);
        auto end_iter = std::chrono::high_resolution_clock::now();
        SumgpuTime1 += std::chrono::duration<float, std::milli>(end_iter - start1).count();
        SumgpuTime3 += std::chrono::duration<float, std::milli>(end_iter - start3).count();
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        //CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        //SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        //CUDA_CHECK(cudaEventElapsedTime(&gpuTime3, start3, stop));
        //SumgpuTime3 = SumgpuTime3 + gpuTime3;
        i_gpuTime = int(int(gpuTime * 1000) % 10000000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << "-> " << antOF[i] << std::endl;
                */
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }

            if (minOf < global_minOf) {
                global_minOf = minOf;
            }
            if (maxOf > global_maxOf) {
                global_maxOf = maxOf;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, *global_minOf_in_device, *global_maxOf_in_device, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start2));
    cudaEventDestroy(start5);

    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant add CPU Const:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant add CPU Const:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_ant_add_CPU() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start2, start5, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&start5));
    CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int i_gpuTime = 0;

    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    //int numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); 
    int numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));
    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
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
    int numBlocks = 0;
    int numThreads = 0;


    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));

    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        go_mass_probability_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaMemcpy(norm_matrix_probability_dev, norm_matrix_probability, numBytes_matrix_graph, cudaMemcpyHostToDevice));

        go_all_agent_only << <numBlocks, numThreads >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));

        // Обновление весов-феромонов
        add_pheromon_iteration_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        i_gpuTime = int(int(gpuTime * 1000) % 10000000);

    }

    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start2));
    cudaEventDestroy(start5);

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
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant add CPU:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant add CPU:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_ant_add_CPU_non_hash() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start2, start5, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&start5));
    CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float gpuTime5 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;

    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
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
    int numBlocks = 0;
    int numThreads = 0;
    if (MAX_THREAD_CUDA < ANT_SIZE) {
        numBlocks = (ANT_SIZE + MAX_THREAD_CUDA - 1) / MAX_THREAD_CUDA;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        auto start1 = std::chrono::high_resolution_clock::now();
        //CUDA_CHECK(cudaEventRecord(start1, 0);
        //go_mass_probability_block << <kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        go_mass_probability_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);
        auto end_go_mass_probability = std::chrono::high_resolution_clock::now();
        SumgpuTime4 += std::chrono::duration<float, std::milli>(end_go_mass_probability - start1).count();
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaMemcpy(norm_matrix_probability_dev, norm_matrix_probability, numBytes_matrix_graph, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(start5, 0));
        go_all_agent_only_non_hash << <numBlocks, numThreads >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&gpuTime5, start5, stop);
        SumgpuTime5 = SumgpuTime5 + gpuTime5;
        CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        //CUDA_CHECK(cudaEventRecord(start3, 0));
        //add_pheromon_iteration_block << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);

        auto start3 = std::chrono::high_resolution_clock::now();
        // Обновление весов-феромонов
        add_pheromon_iteration_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);
        auto end_iter = std::chrono::high_resolution_clock::now();
        SumgpuTime1 += std::chrono::duration<float, std::milli>(end_iter - start1).count();
        SumgpuTime3 += std::chrono::duration<float, std::milli>(end_iter - start3).count();
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        //CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        //SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        //CUDA_CHECK(cudaEventElapsedTime(&gpuTime3, start3, stop));
        //SumgpuTime3 = SumgpuTime3 + gpuTime3;
        i_gpuTime = int(int(gpuTime * 1000) % 10000000);

        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, *global_minOf_in_device, *global_maxOf_in_device, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start2));
    cudaEventDestroy(start5);

    CUDA_CHECK(cudaFree(parametr_value_dev));
    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant add CPU non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant add CPU non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

//объявление функций CPU
void go_all_agent_non_cuda(int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail);
void initializeHashTable_non_cuda(HashEntry* hashTable, int size);

static int start_CUDA_ant_add_CPU2_Time() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
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
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int kol_hash_fail = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    //int numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); 
    int numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }
    // Выделение памяти для хэш-таблицы на CPU
    HashEntry* hashTable = new HashEntry[HASH_TABLE_SIZE];
    // Вызов функции инициализации
    initializeHashTable_non_cuda(hashTable, HASH_TABLE_SIZE);

    // Выделение памяти на устройстве
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;

    int numBlocksParametr = 0;
    int numThreadsParametr = 0;

    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    const int threadsPerBlock = MAX_THREAD_CUDA;
    if (threadsPerBlock < PARAMETR_SIZE) {
        numBlocksParametr = (PARAMETR_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreadsParametr = MAX_THREAD_CUDA;
    }
    else {
        numBlocksParametr = 1;
        numThreadsParametr = PARAMETR_SIZE;
    }

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_only << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));

        // Вычисление пути агентов
        CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
        auto end_temp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
        auto start_CPU_2 = std::chrono::high_resolution_clock::now();
        go_all_agent_non_cuda(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail);
        auto end_CPU_4 = std::chrono::high_resolution_clock::now();
        SumgpuTime4 += std::chrono::duration<float, std::milli>(end_CPU_4 - start_CPU_2).count();
        //go_all_agent_only << <numBlocks, numThreads >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaMemcpy(ant_parametr_dev, ant_parametr, numBytesInt_matrix_ant, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(antOFdev, antOF, numBytes_ant, cudaMemcpyHostToDevice));


        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        CUDA_CHECK(cudaEventRecord(start3, 0));
        add_pheromon_iteration_only << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        double maxOf = -std::numeric_limits<double>::max();
        double minOf = std::numeric_limits<double>::max();
        for (int i = 0; i < ANT_SIZE; ++i) {
            if (antOF[i] != ZERO_HASH_RESULT) {
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
        }
        // Обновление глобальных максимумов и минимумов
        if (minOf < global_minOf) {
            global_minOf = minOf;
        }
        if (maxOf > global_maxOf) {
            global_maxOf = maxOf;
        }

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime3, start3, stop));
        SumgpuTime3 = SumgpuTime3 + gpuTime3;
        auto end_CPU_5 = std::chrono::high_resolution_clock::now();
        SumgpuTime5 += std::chrono::duration<float, std::milli>(end_CPU_5 - start_CPU_2).count();

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << "-> " << antOF[i] << std::endl;
                */
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
            if (minOf < global_minOf) {
                global_minOf = minOf;
            }
            if (maxOf > global_maxOf) {
                global_maxOf = maxOf;
            }
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }

        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, kol_hash_fail);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));

    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] hashTable;               // Освобождение памяти для хэш-таблицы

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant add CPU12 Time:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << global_minOf << "; " << global_maxOf << ";" << kol_hash_fail << ";" << std::endl;
    logFile << "Time CUDA ant add CPU12 Time:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << global_minOf << "; " << global_maxOf << ";" << kol_hash_fail << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}
/*

// Выделение памяти на устройстве (константная память)
__constant__ double parametr_value_dev_const[MAX_VALUE_SIZE * PARAMETR_SIZE];
__constant__ double pheromon_value_dev_const[MAX_VALUE_SIZE * PARAMETR_SIZE];
__constant__ double kol_enter_value_dev_const[MAX_VALUE_SIZE * PARAMETR_SIZE];
__constant__ double norm_matrix_probability_dev_const[MAX_VALUE_SIZE * PARAMETR_SIZE];

__device__ void go_ant_path_optMem(int tx, int bx, curandState* state, double* agent, int* agent_node) {

    double randomValue = curand_uniform(state);
    //Определение номера значения
    int k = 0;
    while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability_dev_const[MAX_VALUE_SIZE * tx + k]) {
        k++;
    }
    __syncthreads();
    // Запись подматрицы блока в глобальную память
    // каждый поток записывает один элемент
    agent_node[bx * PARAMETR_SIZE + tx] = k;
    agent[bx * PARAMETR_SIZE + tx] = parametr_value_dev_const[tx * MAX_VALUE_SIZE + k];
}

__global__ void go_all_agent_only_optMem(int* agent_node, double* OF, HashEntry* hashTable, int* kol_hash_fail) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (агента)
    if (bx < ANT_SIZE) {
        double agent[PARAMETR_SIZE * ANT_SIZE];
        //int tx = threadIdx.x;
        curandState state;
        curand_init(clock64() * bx + gpuTime_const, 0, 0, &state); // Инициализация состояния генератора случайных чисел
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            go_ant_path_optMem(tx, bx, &state, agent, agent_node);
        }
        // Проверка наличия решения в Хэш-таблице
        double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
        int nom_iteration = 1;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
            saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            //Если значение в Хэш-найдено, то агент "нулевой"
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 1: // ACOCNI
                OF[bx] = ZERO_HASH_RESULT;
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            case 2: // ACOCCyN
                while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                {

                    for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                        go_ant_path_optMem(tx, bx, &state, agent, agent_node);
                    }
                    // Проверка наличия решения в Хэш-таблице
                    cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                    nom_iteration = nom_iteration + 1;
                    atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                }
                OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
                break;
            }
        }
        __syncthreads();

    }
}

static int start_CUDA_ant_add_CPU_optMem() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1,  start2,  stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start2));
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
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;

    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве

    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int numBlocks = 0;
    int numThreads = 0;

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    //dim3 kol_ant(ANT_SIZE);

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(parametr_value_dev_const, parametr_value, numBytes_matrix_graph));
    CUDA_CHECK(cudaMemcpyToSymbol(pheromon_value_dev_const, pheromon_value, numBytes_matrix_graph));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        auto start1 = std::chrono::high_resolution_clock::now();
        //CUDA_CHECK(cudaEventRecord(start1, 0));
        //go_mass_probability_block << <kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        go_mass_probability_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

        CUDA_CHECK(cudaEventRecord(start2, 0));

        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaMemcpyToSymbol(norm_matrix_probability_dev_const, norm_matrix_probability, numBytes_matrix_graph));

        go_all_agent_only_optMem << <numBlocks, numThreads >> > (ant_parametr_dev, antOFdev, hashTable_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        auto start3 = std::chrono::high_resolution_clock::now();
        // Обновление весов-феромонов
        CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
        add_pheromon_iteration_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

        // Поиск максимума и минимума
        double maxOf = -std::numeric_limits<double>::max();
        double minOf = std::numeric_limits<double>::max();
        for (int i = 0; i < ANT_SIZE; ++i) {
            if (antOF[i] != ZERO_HASH_RESULT) {
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
        }
        // Обновление глобальных максимумов и минимумов
        if (minOf < global_minOf) {
            global_minOf = minOf;
        }
        if (maxOf > global_maxOf) {
            global_maxOf = maxOf;
        }
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        i_gpuTime = int(int(gpuTime * 1000) % 10000000);

        auto end_iter = std::chrono::high_resolution_clock::now();
        SumgpuTime1 += std::chrono::duration<float, std::milli>(end_iter - start1).count();
        SumgpuTime3 += std::chrono::duration<float, std::milli>(end_iter - start3).count();

        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start2));

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
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant add CPU optMem:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << global_minOf << "; " << global_maxOf << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant add CPU optMem:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << global_minOf << "; " << global_maxOf << ";" << *kol_hash_fail_in_device << ";" << std::endl;
        delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}
*/
static int start_CUDA_ant_non_hash() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
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
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;

    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
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
    int numBlocks = 0;
    int numThreads = 0;
    if (MAX_THREAD_CUDA < ANT_SIZE) {
        numBlocks = (ANT_SIZE + MAX_THREAD_CUDA - 1) / MAX_THREAD_CUDA;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_block << <kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        go_all_agent_only_non_hash << <numBlocks, numThreads >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        CUDA_CHECK(cudaEventRecord(start3, 0));
        add_pheromon_iteration_block << < kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
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
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << "-> " << antOF[i] << std::endl;
                */
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
            if (minOf < global_minOf) {
                global_minOf = minOf;
            }
            if (maxOf > global_maxOf) {
                global_maxOf = maxOf;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));

    CUDA_CHECK(cudaFree(parametr_value_dev));
    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_ant_par() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
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
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
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
    int numBlocks = 0;
    int numThreads = 0;
    int numBlocksParametr = 0;
    int numThreadsParametr = 0;


    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    if (threadsPerBlock < PARAMETR_SIZE) {
        numBlocksParametr = (PARAMETR_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreadsParametr = MAX_THREAD_CUDA;
    }
    else {
        numBlocksParametr = 1;
        numThreadsParametr = PARAMETR_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_only << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        go_all_agent_only << <numBlocks, numThreads >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        CUDA_CHECK(cudaEventRecord(start3, 0));
        add_pheromon_iteration_only << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
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
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << "-> " << antOF[i] << std::endl;
                */
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }

            if (minOf < global_minOf) {
                global_minOf = minOf;
            }
            if (maxOf > global_maxOf) {
                global_maxOf = maxOf;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));

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
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant par:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant par:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_ant_par_not_f() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
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
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
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
    int numBlocks = 0;
    int numThreads = 0;
    int numBlocksParametr = 0;
    int numThreadsParametr = 0;


    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    if (threadsPerBlock < PARAMETR_SIZE) {
        numBlocksParametr = (PARAMETR_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreadsParametr = MAX_THREAD_CUDA;
    }
    else {
        numBlocksParametr = 1;
        numThreadsParametr = PARAMETR_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_not_f_only << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        go_all_agent_not_f_only << <numBlocks, numThreads >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            //CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    //std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j] << " ";
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }
        CUDA_CHECK(cudaEventRecord(start3, 0));
        add_pheromon_iteration_only << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
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
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;
        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << "-> " << antOF[i] << std::endl;
                */
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }

            if (minOf < global_minOf) {
                global_minOf = minOf;
            }
            if (maxOf > global_maxOf) {
                global_maxOf = maxOf;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));

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
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant par not f:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant par not f:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
} //Не работает 

static int start_CUDA_ant_par_sort() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
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
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int numBytesInt_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(int);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    int* indices = new int[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* parametr_value_dev = nullptr;
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    int* indices_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int numBlocks = 0;
    int numThreads = 0;
    int numBlocksParametr = 0;
    int numThreadsParametr = 0;


    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&indices_dev, numBytesInt_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    if (threadsPerBlock < PARAMETR_SIZE) {
        numBlocksParametr = (PARAMETR_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreadsParametr = MAX_THREAD_CUDA;
    }
    else {
        numBlocksParametr = 1;
        numThreadsParametr = PARAMETR_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_sort_only << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev, indices_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(indices, indices_dev, numBytesInt_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + indices[i * MAX_VALUE_SIZE + j]] << "(" << pheromon_value[i * MAX_VALUE_SIZE + indices[i * MAX_VALUE_SIZE + j]] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + indices[i * MAX_VALUE_SIZE + j]] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + indices[i * MAX_VALUE_SIZE + j]] << "):" << j << " "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        go_all_agent_not_sort_only << <numBlocks, numThreads >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail, indices_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            //CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    //std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j] /* << "(" << ant[i * PARAMETR_SIZE + j] */ << " ";
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        CUDA_CHECK(cudaEventRecord(start3, 0));
        add_pheromon_iteration_only << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
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
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << "-> " << antOF[i] << std::endl;
                */
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }

            if (minOf < global_minOf) {
                global_minOf = minOf;
            }
            if (maxOf > global_maxOf) {
                global_maxOf = maxOf;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));

    CUDA_CHECK(cudaFree(parametr_value_dev));
    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(indices_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant par sort:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant par sort:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_ant_par_global() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
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
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* parametr_value_dev = nullptr;
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* ant_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int numBlocks = 0;
    int numThreads = 0;
    int numBlocksParametr = 0;
    int numThreadsParametr = 0;


    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));
    CUDA_CHECK(cudaMalloc((void**)&ant_dev, numBytes_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    if (threadsPerBlock < PARAMETR_SIZE) {
        numBlocksParametr = (PARAMETR_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreadsParametr = MAX_THREAD_CUDA;
    }
    else {
        numBlocksParametr = 1;
        numThreadsParametr = PARAMETR_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_only << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра

        CUDA_CHECK(cudaEventRecord(start2, 0));
        go_all_agent_only_global << <numBlocks, numThreads >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра

        CUDA_CHECK(cudaEventRecord(start3, 0));
        add_pheromon_iteration_only << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
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
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));

    CUDA_CHECK(cudaFree(parametr_value_dev));
    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(ant_dev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant par global:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant par global:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_ant_par_Const() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
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
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int numBlocks = 0;
    int numThreads = 0;
    int numBlocksParametr = 0;
    int numThreadsParametr = 0;

    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    if (threadsPerBlock < PARAMETR_SIZE) {
        numBlocksParametr = (PARAMETR_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreadsParametr = MAX_THREAD_CUDA;
    }
    else {
        numBlocksParametr = 1;
        numThreadsParametr = PARAMETR_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(parametr_value_dev_const, parametr_value, numBytes_matrix_graph));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_only << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        go_all_agent_only_const << <numBlocks, numThreads >> > (norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        CUDA_CHECK(cudaEventRecord(start3, 0));
        add_pheromon_iteration_only << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
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
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << "-> " << antOF[i] << std::endl;
                */
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }

            if (minOf < global_minOf) {
                global_minOf = minOf;
            }
            if (maxOf > global_maxOf) {
                global_maxOf = maxOf;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(start3));

    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA ant par const:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA ant par const:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_opt() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t startAll, start, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;


    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
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
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    int threads_init_hash = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threads_init_hash - 1) / threads_init_hash;
    initializeHashTable << <blocks_init_hash, threads_init_hash >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    dim3 kol_ant(ANT_SIZE);

    //Заполнение начальными значениями ant_parametr_dev, antOFdev
    for (int i = 0; i < ANT_SIZE; ++i) {
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            ant_parametr[i * PARAMETR_SIZE + j] = 0;
        }
        antOF[i] = 1;
    }

    CUDA_CHECK(cudaMemcpy(ant_parametr_dev, ant_parametr, numBytesInt_matrix_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(antOFdev, antOF, numBytes_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    if (PARAMETR_SIZE < MAX_THREAD_CUDA) {
        for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

            CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
            go_mass_probability_and_add_pheromon_iteration << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev, ant_parametr_dev, antOFdev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (CPU_RANDOM) {
                //Создание множества случайных чисел на итерации
                for (int i = 0; i < kolBytes_random_value; ++i) {
                    random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
                }
                if (PRINT_INFORMATION) {
                    std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                    for (int i = 0; i < ANT_SIZE; ++i) {
                        for (int j = 0; j < PARAMETR_SIZE; ++j) {
                            std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                        }
                        std::cout << std::endl;

                    }
                }
                CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
            }
            go_all_agent << <kol_ant, kol_parametr >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
            i_gpuTime = int(gpuTime * 1000);

            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            if ((nom_iter + 1) % kol_shag_stat == 0) {
                int NomStatistics = nom_iter / kol_shag_stat;
                update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device / PARAMETR_SIZE);
            }
        }
    }
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
    // Освобождение ресурсов
    // Освобождение ресурсов

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
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    std::cout << "Time CUDA opt:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA opt:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_opt_Time() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;


    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
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

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    int threads_init_hash = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threads_init_hash - 1) / threads_init_hash;
    initializeHashTable << <blocks_init_hash, threads_init_hash >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    dim3 kol_ant(ANT_SIZE);

    //Заполнение начальными значениями ant_parametr_dev, antOFdev
    for (int i = 0; i < ANT_SIZE; ++i) {
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            ant_parametr[i * PARAMETR_SIZE + j] = 0;
        }
        antOF[i] = 1;
    }

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(ant_parametr_dev, ant_parametr, numBytesInt_matrix_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(antOFdev, antOF, numBytes_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    if (PARAMETR_SIZE < MAX_THREAD_CUDA) {
        for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

            CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
            CUDA_CHECK(cudaEventRecord(start1, 0));
            go_mass_probability_and_add_pheromon_iteration << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev, ant_parametr_dev, antOFdev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (PRINT_INFORMATION) {
                CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
                for (int i = 0; i < PARAMETR_SIZE; ++i) {
                    for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                        std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                    }
                    std::cout << std::endl; // Переход на новую строку
                }
            }
            CUDA_CHECK(cudaEventRecord(start2, 0));
            if (CPU_RANDOM) {
                //Создание множества случайных чисел на итерации
                for (int i = 0; i < kolBytes_random_value; ++i) {
                    random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
                }
                if (PRINT_INFORMATION) {
                    std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                    for (int i = 0; i < ANT_SIZE; ++i) {
                        for (int j = 0; j < PARAMETR_SIZE; ++j) {
                            std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                        }
                        std::cout << std::endl;

                    }
                }
                CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
            }
            go_all_agent << <kol_ant, kol_parametr >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (PRINT_INFORMATION) {
                CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
                std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                        //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                    }
                    std::cout << "-> " << antOF[i] << std::endl;

                }
            }

            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
            SumgpuTime1 = SumgpuTime1 + gpuTime1;
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
            SumgpuTime2 = SumgpuTime2 + gpuTime2;

            i_gpuTime = int(int(gpuTime * 1000) % 10000000);
            double maxOf = -INT16_MAX;
            double minOf = INT16_MAX;

            if (PRINT_INFORMATION) {
                std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
                CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
                std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            if ((nom_iter + 1) % kol_shag_stat == 0) {
                int NomStatistics = nom_iter / kol_shag_stat;
                if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
                update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device / PARAMETR_SIZE);
            }
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));

    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));

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
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA opt Time:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA opt Time:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_opt_Const() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;


    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    int threads_init_hash = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threads_init_hash - 1) / threads_init_hash;
    initializeHashTable << <blocks_init_hash, threads_init_hash >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    dim3 kol_ant(ANT_SIZE);

    //Заполнение начальными значениями ant_parametr_dev, antOFdev
    for (int i = 0; i < ANT_SIZE; ++i) {
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            ant_parametr[i * PARAMETR_SIZE + j] = 0;
        }
        antOF[i] = 1;

    }

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(ant_parametr_dev, ant_parametr, numBytesInt_matrix_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(antOFdev, antOF, numBytes_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(parametr_value_dev_const, parametr_value, numBytes_matrix_graph));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    if (PARAMETR_SIZE < MAX_THREAD_CUDA) {
        for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

            CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
            CUDA_CHECK(cudaEventRecord(start1, 0));
            go_mass_probability_and_add_pheromon_iteration << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev, ant_parametr_dev, antOFdev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (PRINT_INFORMATION) {
                CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
                for (int i = 0; i < PARAMETR_SIZE; ++i) {
                    for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                        std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                    }
                    std::cout << std::endl; // Переход на новую строку
                }
            }
            CUDA_CHECK(cudaEventRecord(start2, 0));
            if (CPU_RANDOM) {
                //Создание множества случайных чисел на итерации
                for (int i = 0; i < kolBytes_random_value; ++i) {
                    random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
                }
                if (PRINT_INFORMATION) {
                    std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                    for (int i = 0; i < ANT_SIZE; ++i) {
                        for (int j = 0; j < PARAMETR_SIZE; ++j) {
                            std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                        }
                        std::cout << std::endl;

                    }
                }
                CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
            }
            go_all_agent_const << <kol_ant, kol_parametr >> > (norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (PRINT_INFORMATION) {
                CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
                std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                        //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                    }
                    std::cout << "-> " << antOF[i] << std::endl;

                }
            }

            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
            SumgpuTime1 = SumgpuTime1 + gpuTime1;
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
            SumgpuTime2 = SumgpuTime2 + gpuTime2;

            i_gpuTime = int(int(gpuTime * 1000) % 10000000);
            double maxOf = -INT16_MAX;
            double minOf = INT16_MAX;

            if (PRINT_INFORMATION) {
                std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
                CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
                std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            if ((nom_iter + 1) % kol_shag_stat == 0) {
                int NomStatistics = nom_iter / kol_shag_stat;
                if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
                update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device / PARAMETR_SIZE);
            }
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));

    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));

    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA opt Const:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA opt Const:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_opt_non_hash() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
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

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    dim3 kol_ant(ANT_SIZE);

    //Заполнение начальными значениями ant_parametr_dev, antOFdev
    for (int i = 0; i < ANT_SIZE; ++i) {
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            ant_parametr[i * PARAMETR_SIZE + j] = 0;
        }
        antOF[i] = 1;
    }

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(ant_parametr_dev, ant_parametr, numBytesInt_matrix_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(antOFdev, antOF, numBytes_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    if (PARAMETR_SIZE < MAX_THREAD_CUDA) {
        for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

            CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
            CUDA_CHECK(cudaEventRecord(start1, 0));
            go_mass_probability_and_add_pheromon_iteration << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev, ant_parametr_dev, antOFdev);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (PRINT_INFORMATION) {
                CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
                std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
                for (int i = 0; i < PARAMETR_SIZE; ++i) {
                    for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                        std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                    }
                    std::cout << std::endl; // Переход на новую строку
                }
            }
            CUDA_CHECK(cudaEventRecord(start2, 0));
            if (CPU_RANDOM) {
                //Создание множества случайных чисел на итерации
                for (int i = 0; i < kolBytes_random_value; ++i) {
                    random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
                }
                if (PRINT_INFORMATION) {
                    std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                    for (int i = 0; i < ANT_SIZE; ++i) {
                        for (int j = 0; j < PARAMETR_SIZE; ++j) {
                            std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                        }
                        std::cout << std::endl;

                    }
                }
                CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
            }
            go_all_agent_non_hash << <kol_ant, kol_parametr >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, maxOf_dev, minOf_dev, kol_hash_fail);
            CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
            if (PRINT_INFORMATION) {
                CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
                std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                        //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                    }
                    std::cout << "-> " << antOF[i] << std::endl;

                }
            }

            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
            SumgpuTime1 = SumgpuTime1 + gpuTime1;
            CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
            SumgpuTime2 = SumgpuTime2 + gpuTime2;

            i_gpuTime = int(int(gpuTime * 1000) % 10000000);
            double maxOf = -INT16_MAX;
            double minOf = INT16_MAX;

            if (PRINT_INFORMATION) {
                std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
                CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
                std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
            }
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            if ((nom_iter + 1) % kol_shag_stat == 0) {
                int NomStatistics = nom_iter / kol_shag_stat;
                if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
                update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device / PARAMETR_SIZE);
            }
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));

    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));

    CUDA_CHECK(cudaFree(parametr_value_dev));
    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA opt non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA opt non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_opt_ant() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
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
    int numBlocks = 0;
    int numThreads = 0;


    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);

    //Заполнение начальными значениями ant_parametr_dev, antOFdev
    for (int i = 0; i < ANT_SIZE; ++i) {
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            ant_parametr[i * PARAMETR_SIZE + j] = 0;
        }
        antOF[i] = 1;
    }

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(ant_parametr_dev, ant_parametr, numBytesInt_matrix_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(antOFdev, antOF, numBytes_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_and_add_pheromon_iteration_block << <kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        go_all_agent_only << <numBlocks, numThreads >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;

        i_gpuTime = int(int(gpuTime * 1000) % 10000000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));

    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));

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
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA opt ant:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA opt ant:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_opt_ant_Const() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int numBlocks = 0;
    int numThreads = 0;

    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);

    //Заполнение начальными значениями ant_parametr_dev, antOFdev
    for (int i = 0; i < ANT_SIZE; ++i) {
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            ant_parametr[i * PARAMETR_SIZE + j] = 0;
        }
        antOF[i] = 1;
    }

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(ant_parametr_dev, ant_parametr, numBytesInt_matrix_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(antOFdev, antOF, numBytes_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(parametr_value_dev_const, parametr_value, numBytes_matrix_graph));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_and_add_pheromon_iteration_block << <kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;
                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        go_all_agent_only_const << <numBlocks, numThreads >> > (norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;
            }
        }
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;

        i_gpuTime = int(int(gpuTime * 1000) % 10000000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));

    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));

    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA opt ant const:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA opt ant const:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_opt_ant_non_hash() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
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
    int numBlocks = 0;
    int numThreads = 0;
    if (MAX_THREAD_CUDA < ANT_SIZE) {
        numBlocks = (ANT_SIZE + MAX_THREAD_CUDA - 1) / MAX_THREAD_CUDA;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);

    //Заполнение начальными значениями ant_parametr_dev, antOFdev
    for (int i = 0; i < ANT_SIZE; ++i) {
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            ant_parametr[i * PARAMETR_SIZE + j] = 0;
        }
        antOF[i] = 1;
    }

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(ant_parametr_dev, ant_parametr, numBytesInt_matrix_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(antOFdev, antOF, numBytes_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_and_add_pheromon_iteration_block << <kol_parametr, 1 >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        go_all_agent_only_non_hash << <numBlocks, numThreads >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;

        i_gpuTime = int(int(gpuTime * 1000) % 10000000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));

    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));

    CUDA_CHECK(cudaFree(parametr_value_dev));
    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA opt ant non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA opt ant non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_opt_ant_par() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
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
    int numBlocks = 0;
    int numThreads = 0;
    int numBlocksParametr = 0;
    int numThreadsParametr = 0;


    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    if (threadsPerBlock < PARAMETR_SIZE) {
        numBlocksParametr = (PARAMETR_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreadsParametr = MAX_THREAD_CUDA;
    }
    else {
        numBlocksParametr = 1;
        numThreadsParametr = PARAMETR_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра

    //Заполнение начальными значениями ant_parametr_dev, antOFdev
    for (int i = 0; i < ANT_SIZE; ++i) {
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            ant_parametr[i * PARAMETR_SIZE + j] = 0;
        }
        antOF[i] = 1;
    }

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(ant_parametr_dev, ant_parametr, numBytesInt_matrix_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(antOFdev, antOF, numBytes_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_and_add_pheromon_iteration_only << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;

                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        go_all_agent_only << <numBlocks, numThreads >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;

        i_gpuTime = int(int(gpuTime * 1000) % 10000000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));

    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));

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
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA opt ant par:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA opt ant par:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_opt_ant_par_global() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* parametr_value_dev = nullptr;
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    double* ant_dev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int numBlocks = 0;
    int numThreads = 0;
    int numBlocksParametr = 0;
    int numThreadsParametr = 0;


    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&ant_dev, numBytes_matrix_ant));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    if (threadsPerBlock < PARAMETR_SIZE) {
        numBlocksParametr = (PARAMETR_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreadsParametr = MAX_THREAD_CUDA;
    }
    else {
        numBlocksParametr = 1;
        numThreadsParametr = PARAMETR_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра

    //Заполнение начальными значениями ant_parametr_dev, antOFdev
    for (int i = 0; i < ANT_SIZE; ++i) {
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            ant_parametr[i * PARAMETR_SIZE + j] = 0;
        }
        antOF[i] = 1;
    }

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(ant_parametr_dev, ant_parametr, numBytesInt_matrix_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(antOFdev, antOF, numBytes_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_and_add_pheromon_iteration_only << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        CUDA_CHECK(cudaEventRecord(start2, 0));

        go_all_agent_only_global << <numBlocks, numThreads >> > (parametr_value_dev, norm_matrix_probability_dev, random_values_dev, ant_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;

        i_gpuTime = int(int(gpuTime * 1000) % 10000000);
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));

    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));

    CUDA_CHECK(cudaFree(parametr_value_dev));
    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(ant_dev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA opt ant par global:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA opt ant par global:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_opt_ant_par_Const() {
    auto start_temp = std::chrono::high_resolution_clock::now();
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop));

    float gpuTime = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    float SumgpuTime4 = 0.0f;
    float SumgpuTime5 = 0.0f;
    float SumgpuTime6 = 0.0f;
    float SumgpuTime7 = 0.0f;
    int i_gpuTime = 0;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    //Генератор на CPU для GPU
    int numBytes_random_value = 0;
    int kolBytes_random_value = 0;
    if (CPU_RANDOM) {
        numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
        kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE;
        if (TYPE_ACO >= 2 && CPU_RANDOM)
        {
            numBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION * sizeof(double);
            kolBytes_random_value = PARAMETR_SIZE * ANT_SIZE * ACOCCyN_KOL_ITERATION;
        }
    }
    double* random_values = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    double* random_values_print = new double[kolBytes_random_value]; //Для хранения массива случайных чисел
    // Генератор случайных чисел
    auto end_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> current_time = end_temp - start_temp;
    std::default_random_engine generator(123 + int(current_time.count() * CONST_RANDOM)); // Используем gpuTime как начальное значение + current_time.count()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double* random_values_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&random_values_dev, numBytes_random_value));

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int numBlocks = 0;
    int numThreads = 0;
    int numBlocksParametr = 0;
    int numThreadsParametr = 0;

    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    if (threadsPerBlock < PARAMETR_SIZE) {
        numBlocksParametr = (PARAMETR_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreadsParametr = MAX_THREAD_CUDA;
    }
    else {
        numBlocksParametr = 1;
        numThreadsParametr = PARAMETR_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра

    //Заполнение начальными значениями ant_parametr_dev, antOFdev
    for (int i = 0; i < ANT_SIZE; ++i) {
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            ant_parametr[i * PARAMETR_SIZE + j] = 0;
        }
        antOF[i] = 1;
    }
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(ant_parametr_dev, ant_parametr, numBytesInt_matrix_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(antOFdev, antOF, numBytes_ant, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(parametr_value_dev_const, parametr_value, numBytes_matrix_graph));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        CUDA_CHECK(cudaMemcpyToSymbol(gpuTime_const, &i_gpuTime, sizeof(int))); // Копирование значения в константную память
        CUDA_CHECK(cudaEventRecord(start1, 0));
        go_mass_probability_and_add_pheromon_iteration_only << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev, ant_parametr_dev, antOFdev);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        CUDA_CHECK(cudaEventRecord(start2, 0));
        if (CPU_RANDOM) {
            //Создание множества случайных чисел на итерации
            for (int i = 0; i < kolBytes_random_value; ++i) {
                random_values[i] = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            }
            if (PRINT_INFORMATION) {
                std::cout << "random_values (" << kolBytes_random_value << "):" << std::endl;
                for (int i = 0; i < ANT_SIZE; ++i) {
                    for (int j = 0; j < PARAMETR_SIZE; ++j) {
                        std::cout << random_values[i * PARAMETR_SIZE + j] << " ";
                    }
                    std::cout << std::endl;
                }
            }
            CUDA_CHECK(cudaMemcpy(random_values_dev, random_values, numBytes_matrix_ant, cudaMemcpyHostToDevice));//Запись множества в память GPU  
        }
        go_all_agent_only_const << <numBlocks, numThreads >> > (norm_matrix_probability_dev, random_values_dev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
        if (PRINT_INFORMATION) {
            CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(random_values_print, random_values_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime1, start1, stop));
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTime2, start2, stop));
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        i_gpuTime = int(int(gpuTime * 1000) % 10000000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(int(gpuTime * 1000) % 10000000) << "x" << ANT_SIZE << "):" << std::endl;
            CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, *kol_hash_fail_in_device);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));

    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));

    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
    CUDA_CHECK(cudaFree(hashTable_dev));
    CUDA_CHECK(cudaFree(maxOf_dev));
    CUDA_CHECK(cudaFree(minOf_dev));
    CUDA_CHECK(cudaFree(kol_hash_fail));
    CUDA_CHECK(cudaFree(random_values_dev));

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] antSumOF;
    delete[] ant_hash_add;
    delete[] random_values;
    delete[] random_values_print;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA opt ant par const:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA opt ant par const:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_opt_one_GPU() {
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&stop));

    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* parametr_value_dev = nullptr;
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* antdev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antdev, numBytes_matrix_ant));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    int threads_init_hash = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threads_init_hash - 1) / threads_init_hash;
    initializeHashTable << <blocks_init_hash, threads_init_hash >> > (hashTable_dev, HASH_TABLE_SIZE);

    // Установка конфигурации запуска ядра
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
        go_all_agent_opt << <kol_ant, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, parametr_value_dev, norm_matrix_probability_dev, antdev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
    }
    if (PRINT_INFORMATION) {
        CUDA_CHECK(cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost));
        std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
        for (int i = 0; i < PARAMETR_SIZE; ++i) {
            for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
            }
            std::cout << std::endl;
        }
        CUDA_CHECK(cudaMemcpy(ant_parametr, ant_parametr_dev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ant, antdev, numBytesInt_matrix_ant, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
        std::cout << "ANT (" << ANT_SIZE << "):" << *kol_hash_fail_in_device / PARAMETR_SIZE << std::endl;
        for (int i = 0; i < ANT_SIZE; ++i) {
            for (int j = 0; j < PARAMETR_SIZE; ++j) {
                //std::cout << ant[i * PARAMETR_SIZE + j] << " + " << random_values_print[i * PARAMETR_SIZE + j] << " ";
                std::cout << ant_parametr[i * PARAMETR_SIZE + j] << " ";// << "(" << ant[i * PARAMETR_SIZE + j] << ") ";
            }
            std::cout << "-> " << antOF[i] << std::endl;

        }
    }
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));

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
    delete[] antSumOF;
    delete[] ant_hash_add;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA opt one:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA opt one:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_opt_one_GPU_local() {
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&stop));

    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* parametr_value_dev = nullptr;
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));


    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    int threads_init_hash = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threads_init_hash - 1) / threads_init_hash;
    initializeHashTable << <blocks_init_hash, threads_init_hash >> > (hashTable_dev, HASH_TABLE_SIZE);

    // Установка конфигурации запуска ядра
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
        go_all_agent_opt_local << <kol_ant, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, parametr_value_dev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
    }
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));

    CUDA_CHECK(cudaFree(parametr_value_dev));
    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
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
    delete[] antSumOF;
    delete[] ant_hash_add;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA opt one local:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA opt one local:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_opt_one_GPU_non_hash() {
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&stop));

    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* parametr_value_dev = nullptr;
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* antdev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int* gpuTime_dev = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antdev, numBytes_matrix_ant));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));
    CUDA_CHECK(cudaMalloc((void**)&gpuTime_dev, sizeof(int)));

    // Установка конфигурации запуска ядра
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
        go_all_agent_opt_non_hash << <kol_ant, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, gpuTime_dev, parametr_value_dev, norm_matrix_probability_dev, antdev, ant_parametr_dev, antOFdev, maxOf_dev, minOf_dev, kol_hash_fail);
    }
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));

    CUDA_CHECK(cudaFree(parametr_value_dev));
    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
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
    delete[] antSumOF;
    delete[] ant_hash_add;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA opt one non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA opt one non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_opt_one_GPU_ant() {
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop));

    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;
    int numBlocks = 0;
    int numThreads = 0;

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* parametr_value_dev = nullptr;
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* antdev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int* gpuTime_dev = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antdev, numBytes_matrix_ant));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));
    CUDA_CHECK(cudaMalloc((void**)&gpuTime_dev, sizeof(int)));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1));
    go_all_agent_opt_only << <numBlocks, numThreads >> > (pheromon_value_dev, kol_enter_value_dev, gpuTime_dev, parametr_value_dev, norm_matrix_probability_dev, antdev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));

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
    delete[] antSumOF;
    delete[] ant_hash_add;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA opt one ant:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA opt one ant:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_opt_one_GPU_ant_local() {
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop));

    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;
    int numBlocks = 0;
    int numThreads = 0;

    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* parametr_value_dev = nullptr;
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* antdev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int* gpuTime_dev = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antdev, numBytes_matrix_ant));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));
    CUDA_CHECK(cudaMalloc((void**)&gpuTime_dev, sizeof(int)));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry)));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock < ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1));
    go_all_agent_opt_only_local << <numBlocks, numThreads >> > (pheromon_value_dev, kol_enter_value_dev, parametr_value_dev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));

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
    delete[] antSumOF;
    delete[] ant_hash_add;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA opt one ant local:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA opt one ant local:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

static int start_CUDA_opt_one_GPU_ant_non_hash() {
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, stop;
    CUDA_CHECK(cudaEventCreate(&startAll));
    CUDA_CHECK(cudaEventRecord(startAll, 0));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&startAll1));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop));

    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    long long numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double); long long numBytesInt_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(int);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;


    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];
    double* ant_hash_add = new double[ANT_SIZE];
    double* global_maxOf_in_device = new double;
    double* global_minOf_in_device = new double;
    int* kol_hash_fail_in_device = new int;
    int numBlocks = 0;
    int numThreads = 0;
    if (MAX_THREAD_CUDA < ANT_SIZE) {
        numBlocks = (ANT_SIZE + MAX_THREAD_CUDA - 1) / MAX_THREAD_CUDA;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 1;
        numThreads = ANT_SIZE;
    }
    if (!load_matrix(NAME_FILE_GRAPH,
        parametr_value,
        pheromon_value,
        kol_enter_value))
    {
        std::cerr << "Error loading matrix!" << std::endl;
        return 1;
    }

    // Выделение памяти на устройстве
    double* parametr_value_dev = nullptr;
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* antdev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    int* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int* gpuTime_dev = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph));
    CUDA_CHECK(cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph));

    CUDA_CHECK(cudaMalloc((void**)&antdev, numBytes_matrix_ant));
    CUDA_CHECK(cudaMalloc((void**)&antOFdev, numBytes_ant));
    CUDA_CHECK(cudaMalloc((void**)&maxOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&minOf_dev, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&kol_hash_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ant_parametr_dev, numBytesInt_matrix_ant));
    CUDA_CHECK(cudaMalloc((void**)&gpuTime_dev, sizeof(int)));

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(startAll1, 0));
    go_all_agent_opt_only_non_hash << <numBlocks, numThreads >> > (pheromon_value_dev, kol_enter_value_dev, gpuTime_dev, parametr_value_dev, norm_matrix_probability_dev, antdev, ant_parametr_dev, antOFdev, maxOf_dev, minOf_dev, kol_hash_fail);
    CUDA_CHECK(cudaGetLastError()); // Проверка на ошибки после запуска ядра
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime1, startAll1, stop));
    CUDA_CHECK(cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost));
    // Освобождение ресурсов
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(startAll1));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(start2));

    CUDA_CHECK(cudaFree(parametr_value_dev));
    CUDA_CHECK(cudaFree(pheromon_value_dev));
    CUDA_CHECK(cudaFree(kol_enter_value_dev));
    CUDA_CHECK(cudaFree(norm_matrix_probability_dev));
    CUDA_CHECK(cudaFree(ant_parametr_dev));
    CUDA_CHECK(cudaFree(antOFdev));
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
    delete[] antSumOF;
    delete[] ant_hash_add;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&AllgpuTime, startAll, stop));

    CUDA_CHECK(cudaEventDestroy(startAll));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::cout << "Time CUDA opt one ant non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    logFile << "Time CUDA opt one ant non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device << ";" << std::endl;
    delete global_maxOf_in_device;
    delete global_minOf_in_device;
    delete kol_hash_fail_in_device;
    return 0;
}

// ----------------- Kernel: Initializing Hash Table -----------------
void initializeHashTable_omp(HashEntry* hashTable, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        hashTable[i].key = ZERO_HASH_RESULT;
        hashTable[i].value = 0.0;
    }
}

// ----------------- Kernel: Initializing Hash Table -----------------
void initializeHashTable_non_cuda(HashEntry* hashTable, int size) {
    for (int i = 0; i < size; i++) {
        hashTable[i].key = ZERO_HASH_RESULT;
        hashTable[i].value = 0.0;
    }
}

// ----------------- MurmurHash64A Implementation -----------------
unsigned long long murmurHash64A_non_cuda(unsigned long long key, unsigned long long seed = 0xDEADBEEFDEADBEEF) {
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

// ----------------- Improved Hash Function Using MurmurHash -----------------
unsigned long long betterHashFunction_non_cuda(unsigned long long key) {
    return murmurHash64A_non_cuda(key) % HASH_TABLE_SIZE;
}

// ----------------- Key Generation Function -----------------
unsigned long long generateKey_non_cuda(const int* agent_node, int bx) {
    unsigned long long key = 0;
    unsigned long long factor = 1;
    for (int i = 0; i < PARAMETR_SIZE; i++) {
        int val = agent_node[bx * PARAMETR_SIZE + i];
        //        std::cout << val << " ";
        key += val * factor;
        factor *= MAX_VALUE_SIZE;
    }
    //    std::cout <<" key=" << key;
    //    std::cout << std::endl;
    return key;
}

// ---------------- - Hash Table Search with Quadratic Probing---------------- -
double getCachedResultOptimized_omp(HashEntry* hashTable, const int* agent_node, int bx) {
    unsigned long long key = generateKey_non_cuda(agent_node, bx);
    unsigned long long idx = betterHashFunction_non_cuda(key);

#pragma omp parallel for
    for (int i = 1; i <= MAX_PROBES; i++) {
        unsigned long long new_idx = idx + static_cast<unsigned long long>(i * i); if (new_idx >= HASH_TABLE_SIZE) { new_idx %= HASH_TABLE_SIZE; }idx = new_idx;

        // Используем критическую секцию для безопасного доступа к хэш-таблице
#pragma omp critical
        {
            if (hashTable[idx].key == key) {
                return hashTable[idx].value; // Found
            }
            if (hashTable[idx].key == ZERO_HASH_RESULT) {
                return -1.0; // Not found and slot is empty
            }
        }
    }
    return -1.0; // Not found after maximum probes
}

// ----------------- Hash Table Search with Quadratic Probing -----------------
double getCachedResultOptimized_non_cuda(HashEntry* hashTable, const int* agent_node, int bx) {
    unsigned long long key = generateKey_non_cuda(agent_node, bx);
    unsigned long long idx = betterHashFunction_non_cuda(key);
    int i = 1;

    while (i <= MAX_PROBES) {
        if (hashTable[idx].key == key) {
            return hashTable[idx].value; // Found
        }
        if (hashTable[idx].key == ZERO_HASH_RESULT) {
            return -1.0; // Not found and slot is empty
        }
        unsigned long long new_idx = idx + static_cast<unsigned long long>(i * i); if (new_idx >= HASH_TABLE_SIZE) { new_idx %= HASH_TABLE_SIZE; }idx = new_idx;
        i++;
    }
    return -1.0; // Not found after maximum probes
}

// ----------------- Hash Table Insertion with Quadratic Probing -----------------
void saveToCacheOptimized_omp(HashEntry* hashTable, const int* agent_node, int bx, double value) {
    unsigned long long key = generateKey_non_cuda(agent_node, bx);
    unsigned long long idx = betterHashFunction_non_cuda(key);

#pragma omp parallel for
    for (int i = 1; i <= MAX_PROBES; i++) {
        unsigned long long new_idx = idx + static_cast<unsigned long long>(i * i); if (new_idx >= HASH_TABLE_SIZE) { new_idx %= HASH_TABLE_SIZE; }idx = new_idx;
        unsigned long long expected = ZERO_HASH_RESULT;
        unsigned long long old_key;

        // Используем критическую секцию для безопасного доступа к хэш-таблице
#pragma omp critical
        {
            old_key = hashTable[idx].key;

            if (old_key == expected) {
                // Successfully inserted
                hashTable[idx].key = key;
                hashTable[idx].value = value;
                return;
            }
            else if (old_key == key) {
                // Key already exists
                hashTable[idx].value = value; // Update value
                return;
            }
        }
    }
    // Если таблица полна, обработайте ошибку или игнорируйте
}

// ----------------- Hash Table Insertion with Quadratic Probing -----------------
void saveToCacheOptimized_non_cuda(HashEntry* hashTable, const int* agent_node, int bx, double value) {
    unsigned long long key = generateKey_non_cuda(agent_node, bx);
    unsigned long long idx = betterHashFunction_non_cuda(key);
    int i = 1;

    while (i <= MAX_PROBES) {
        unsigned long long expected = ZERO_HASH_RESULT;
        unsigned long long old_key = hashTable[idx].key;

        if (old_key == expected) {
            // Successfully inserted
            hashTable[idx].key = key;
            hashTable[idx].value = value;
            return;
        }
        else if (old_key == key) {
            // Key already exists
            hashTable[idx].value = value; // Update value
            return;
        }

        unsigned long long new_idx = idx + static_cast<unsigned long long>(i * i); if (new_idx >= HASH_TABLE_SIZE) { new_idx %= HASH_TABLE_SIZE; }idx = new_idx;
        i++;
    }
    // If the table is full, handle the error or ignore
}

// Функция для вычисления вероятностной формулы
double probability_formula_non_cuda(double pheromon, double kol_enter) {
    double res = 0;
    if ((kol_enter != 0) && (pheromon != 0)) {
        res = 1.0 / kol_enter + pheromon;
    }
    return res;
}

// Подготовка массива для вероятностного поиска
void go_mass_probability_omp(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    // Нормализация слоя с феромоном
#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Суммируем значения феромонов
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        // Вычисляем вероятностные значения
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }

        norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
        for (int i = 1; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1]; // Нормирование значений матрицы с накоплением
        }
    }
}
void go_mass_probability_non_cuda_not_f_omp(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    // Нормализация слоя с феромоном
#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Суммируем значения феромонов
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }

        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = svertka[i] / sumVector;
        }
    }
}
void go_mass_probability_non_cuda_sort_omp(double* pheromon, double* kol_enter, double* norm_matrix_probability, int* indices) {
    // Нормализация слоя с феромоном
#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Суммируем значения феромонов
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }
        // Заполняем массив индексов начальными значениями
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            indices[i] = i;
        }
        //Может посмотреть встроенные сортировки
        // Ручная сортировка методом вставки с сохранением индексов
        for (int j = 1; j < MAX_VALUE_SIZE; ++j) {
            double key = svertka[j];               // Значение текущего элемента
            int idx_key = indices[j];              // Индекс текущего элемента
            int i = j - 1;                         // Начинаем проверку с предыдущего элемента

            while (i >= 0 && svertka[i] > key) {
                svertka[i + 1] = svertka[i];         // Сдвигаем больше элементы вправо
                indices[i + 1] = indices[i];         // Обновляем соответствующий индекс
                i--;
            }
            svertka[i + 1] = key;                   // Кладём ключ на новое место
            indices[i + 1] = idx_key;                // Сохраняем индекс ключа
        }
        norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
        for (int i = 1; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1]; // Нормирование значений матрицы с накоплением
        }
    }
}
// Подготовка массива для вероятностного поиска
void go_mass_probability_non_cuda(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    // Нормализация слоя с феромоном
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Суммируем значения феромонов
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        // Вычисляем вероятностные значения
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }

        norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
        for (int i = 1; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1]; // Нормирование значений матрицы с накоплением
        }
    }
}
void go_mass_probability_non_cuda_not_f(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    // Нормализация слоя с феромоном
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Суммируем значения феромонов
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }

        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = svertka[i] / sumVector;
        }
    }
}
void go_mass_probability_non_cuda_sort(double* pheromon, double* kol_enter, double* norm_matrix_probability, int* indices) {
    // Нормализация слоя с феромоном
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Суммируем значения феромонов
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }
        // Заполняем массив индексов начальными значениями
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            indices[i] = i;
        }
        //Может посмотреть встроенные сортировки
        // Ручная сортировка методом вставки с сохранением индексов
        for (int j = 1; j < MAX_VALUE_SIZE; ++j) {
            double key = svertka[j];               // Значение текущего элемента
            int idx_key = indices[j];              // Индекс текущего элемента
            int i = j - 1;                         // Начинаем проверку с предыдущего элемента

            while (i >= 0 && svertka[i] > key) {
                svertka[i + 1] = svertka[i];         // Сдвигаем больше элементы вправо
                indices[i + 1] = indices[i];         // Обновляем соответствующий индекс
                i--;
            }
            svertka[i + 1] = key;                   // Кладём ключ на новое место
            indices[i + 1] = idx_key;                // Сохраняем индекс ключа
        }
        norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
        for (int i = 1; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1]; // Нормирование значений матрицы с накоплением
        }
    }
}

// Функция для вычисления пути агентов на CPU с использованием OpenMP
void go_all_agent_omp(int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail, double& totalHashTime, double& totalOFTime) {
    // Генератор случайных чисел

    {
        std::default_random_engine generator(123 + gpuTime); // Используем gpuTime и номер потока как начальное значение
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
#pragma omp parallel
#pragma omp for
        for (int bx = 0; bx < ANT_SIZE; bx++) { // Проходим по всем агентам
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
                // Определение номера значения
                int k = 0;
                while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                    k++;
                }

                // Запись подматрицы блока в глобальную память
                agent_node[bx * PARAMETR_SIZE + tx] = k;
                agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
            }
            auto start = std::chrono::high_resolution_clock::now();
            // Проверка наличия решения в Хэш-таблице
            double cachedResult = getCachedResultOptimized_omp(hashTable, agent_node, bx);
            int nom_iteration = 0;
            if (cachedResult == -1.0) {
                // Если значение не найдено в ХЭШ, то заносим новое значение
                auto start_OF = std::chrono::high_resolution_clock::now();
                OF[bx] = BenchShafferaFunction_omp(&agent[bx * PARAMETR_SIZE]);
                auto end_OF = std::chrono::high_resolution_clock::now();
                totalOFTime += std::chrono::duration<double, std::milli>(end_OF - start_OF).count();
                saveToCacheOptimized_omp(hashTable, agent_node, bx, OF[bx]);
            }
            else {
                // Если значение в Хэш-найдено, то агент "нулевой"
#pragma omp atomic
                kol_hash_fail++;

                // Поиск алгоритма для нулевого агента
                switch (TYPE_ACO) {
                case 0: // ACOCN
                    OF[bx] = cachedResult;
                    break;
                case 1: // ACOCNI
                    OF[bx] = ZERO_HASH_RESULT;
                    break;
                case 2: // ACOCCyN
                    while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION)) {
                        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) { // Проходим по всем параметрам
                            double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]

                            // Определение номера значения
                            int k = 0;
                            while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                                k++;
                            }

                            // Запись подматрицы блока в глобальную память
                            agent_node[bx * PARAMETR_SIZE + tx] = k;
                            agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
                        }

                        // Проверка наличия решения в Хэш-таблице
                        cachedResult = getCachedResultOptimized_omp(hashTable, agent_node, bx);
                        nom_iteration++;
                        kol_hash_fail++;
                    }
                    OF[bx] = BenchShafferaFunction_omp(&agent[bx * PARAMETR_SIZE]);
                    saveToCacheOptimized_omp(hashTable, agent_node, bx, OF[bx]);
                    break;
                default:
                    OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                    break;
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
#pragma omp atomic
            totalHashTime += std::chrono::duration<double, std::milli>(end - start).count();
        }
    }
}

void go_all_agent_omp_non_hash(int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, int& kol_hash_fail, double& totalHashTime, double& totalOFTime) {
    // Генератор случайных чисел

    {
        std::default_random_engine generator(123 + gpuTime); // Используем gpuTime и номер потока как начальное значение
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
#pragma omp parallel
#pragma omp for
        for (int bx = 0; bx < ANT_SIZE; bx++) { // Проходим по всем агентам
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
                // Определение номера значения
                int k = 0;
                while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                    k++;
                }

                // Запись подматрицы блока в глобальную память
                agent_node[bx * PARAMETR_SIZE + tx] = k;
                agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
            }
            auto start_OF = std::chrono::high_resolution_clock::now();
            OF[bx] = BenchShafferaFunction_omp(&agent[bx * PARAMETR_SIZE]);
            auto end_OF = std::chrono::high_resolution_clock::now();
            totalOFTime += std::chrono::duration<double, std::milli>(end_OF - start_OF).count();
        }
    }
}

// Функция для вычисления пути агентов на CPU
void go_all_agent_non_cuda_time(int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail, double& totalHashTime, double& totalOFTime, double& HashTimeSave, double& HashTimeSearch, double& SumTimeSearch) {
    // Генератор случайных чисел
    std::default_random_engine generator(123 + gpuTime); // Используем gpuTime как начальное значение
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int bx = 0; bx < ANT_SIZE; bx++) { // Проходим по всем агентам
        auto start_ant = std::chrono::high_resolution_clock::now();
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            // Определение номера значения

            int k = 0;
            while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                k++;
            }
            // Запись подматрицы блока в глобальную память
            agent_node[bx * PARAMETR_SIZE + tx] = k;
            agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
        }
        auto end_ant = std::chrono::high_resolution_clock::now();
        SumTimeSearch += std::chrono::duration<double, std::milli>(end_ant - start_ant).count();
        auto start = std::chrono::high_resolution_clock::now();
        // Проверка наличия решения в Хэш-таблице
        double cachedResult = getCachedResultOptimized_non_cuda(hashTable, agent_node, bx);
        auto end_OF = std::chrono::high_resolution_clock::now();
        HashTimeSearch += std::chrono::duration<double, std::milli>(end_OF - start).count();
        /*
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            std::cout << agent[bx * PARAMETR_SIZE + j] << " ";
        }
        std::cout << "-> " << cachedResult << std::endl;*/
        int nom_iteration = 0;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            auto start_OF = std::chrono::high_resolution_clock::now();
            OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
            auto end_OF = std::chrono::high_resolution_clock::now();
            totalOFTime += std::chrono::duration<double, std::milli>(end_OF - start_OF).count();
            auto start_SaveHash = std::chrono::high_resolution_clock::now();
            saveToCacheOptimized_non_cuda(hashTable, agent_node, bx, OF[bx]);
            auto end_SaveHash = std::chrono::high_resolution_clock::now();
            HashTimeSave += std::chrono::duration<double, std::milli>(end_SaveHash - start_SaveHash).count();
        }
        else {
            auto start_OF_2 = std::chrono::high_resolution_clock::now();
            auto end_OF_2 = std::chrono::high_resolution_clock::now();
            //Если значение в Хэш-найдено, то агент "нулевой"
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                kol_hash_fail = kol_hash_fail + 1;
                break;
            case 1: // ACOCNI
                OF[bx] = ZERO_HASH_RESULT;
                kol_hash_fail = kol_hash_fail + 1;
                break;
            case 2: // ACOCCyN
                while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                {
                    start_OF_2 = std::chrono::high_resolution_clock::now();
                    for (int tx = 0; tx < PARAMETR_SIZE; ++tx) { // Проходим по всем параметрам
                        double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]

                        // Определение номера значения
                        int k = 0;
                        while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                            k++;
                        }

                        // Запись подматрицы блока в глобальную память
                        agent_node[bx * PARAMETR_SIZE + tx] = k;
                        agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
                    }
                    end_OF_2 = std::chrono::high_resolution_clock::now();
                    SumTimeSearch += std::chrono::duration<double, std::milli>(end_OF_2 - start_OF_2).count();
                    // Проверка наличия решения в Хэш-таблице
                    start_OF_2 = std::chrono::high_resolution_clock::now();
                    cachedResult = getCachedResultOptimized_non_cuda(hashTable, agent_node, bx);
                    end_OF_2 = std::chrono::high_resolution_clock::now();
                    HashTimeSearch += std::chrono::duration<double, std::milli>(end_OF_2 - start_OF_2).count();
                    nom_iteration = nom_iteration + 1;
                    kol_hash_fail = kol_hash_fail + 1;
                }

                start_OF_2 = std::chrono::high_resolution_clock::now();
                OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
                end_OF_2 = std::chrono::high_resolution_clock::now();
                totalOFTime += std::chrono::duration<double, std::milli>(end_OF_2 - start_OF_2).count();
                start_OF_2 = std::chrono::high_resolution_clock::now();
                saveToCacheOptimized_non_cuda(hashTable, agent_node, bx, OF[bx]);
                end_OF_2 = std::chrono::high_resolution_clock::now();
                HashTimeSave += std::chrono::duration<double, std::milli>(end_OF_2 - start_OF_2).count();
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                kol_hash_fail = kol_hash_fail + 1;
                break;
            }


        }
        //std::cout << bx << "bx " << kol_hash_fail << " " << OF[bx] << " ";
        auto end = std::chrono::high_resolution_clock::now();
        totalHashTime += std::chrono::duration<double, std::milli>(end - start).count();
    }
}

// Функция для вычисления пути агентов на CPU
void go_all_agent_non_cuda(int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail) {
    // Генератор случайных чисел
    std::default_random_engine generator(123 + gpuTime); // Используем gpuTime как начальное значение
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int bx = 0; bx < ANT_SIZE; bx++) { // Проходим по всем агентам
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            // Определение номера значения
            int k = 0;
            while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                k++;
            }

            // Запись подматрицы блока в глобальную память
            agent_node[bx * PARAMETR_SIZE + tx] = k;
            agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
        }
        // Проверка наличия решения в Хэш-таблице
        double cachedResult = getCachedResultOptimized_non_cuda(hashTable, agent_node, bx);
        int nom_iteration = 0;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
            saveToCacheOptimized_non_cuda(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            //Если значение в Хэш-найдено, то агент "нулевой"
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                kol_hash_fail = kol_hash_fail + 1;
                break;
            case 1: // ACOCNI
                OF[bx] = ZERO_HASH_RESULT;
                kol_hash_fail = kol_hash_fail + 1;
                break;
            case 2: // ACOCCyN
                while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                {
                    for (int tx = 0; tx < PARAMETR_SIZE; ++tx) { // Проходим по всем параметрам
                        double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]

                        // Определение номера значения
                        int k = 0;
                        while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                            k++;
                        }

                        // Запись подматрицы блока в глобальную память
                        agent_node[bx * PARAMETR_SIZE + tx] = k;
                        agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
                    }
                    // Проверка наличия решения в Хэш-таблице
                    cachedResult = getCachedResultOptimized_non_cuda(hashTable, agent_node, bx);
                    nom_iteration = nom_iteration + 1;
                    kol_hash_fail = kol_hash_fail + 1;
                }

                OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
                saveToCacheOptimized_non_cuda(hashTable, agent_node, bx, OF[bx]);
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                kol_hash_fail = kol_hash_fail + 1;
                break;
            }


        }
        //std::cout << bx << "bx " << kol_hash_fail << " " << OF[bx] << " ";
    }
}

void process_agent(int bx, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail) {
    std::default_random_engine generator(rand()); // Генератор случайных чисел
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
        double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
        // Определение номера значения
        int k = 0;
        while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
            k++;
        }

        // Запись подматрицы блока в глобальную память
        agent_node[bx * PARAMETR_SIZE + tx] = k;
        agent[bx * PARAMETR_SIZE + k] = parametr[tx * MAX_VALUE_SIZE + k];
    }

    // Проверка наличия решения в Хэш-таблице
    double cachedResult = getCachedResultOptimized_non_cuda(hashTable, agent_node, bx);
    int nom_iteration = 0;

    if (cachedResult == -1.0) {
        // Если значение не найдено в ХЭШ, то заносим новое значение
        OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
        saveToCacheOptimized_non_cuda(hashTable, agent_node, bx, OF[bx]);
    }
    else {
        // Если значение в Хэш-найдено, то агент "нулевой"
        switch (TYPE_ACO) {
        case 0: // ACOCN
            OF[bx] = cachedResult;
            mtx.lock();
            kol_hash_fail++;
            mtx.unlock();
            break;
        case 1: // ACOCNI
            OF[bx] = ZERO_HASH_RESULT;
            mtx.lock();
            kol_hash_fail++;
            mtx.unlock();
            break;
        case 2: // ACOCCyN
            while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION)) {
                for (int tx = 0; tx < PARAMETR_SIZE; ++tx) { // Проходим по всем параметрам
                    double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]

                    // Определение номера значения
                    int k = 0;
                    while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                        k++;
                    }

                    // Запись подматрицы блока в глобальную память
                    agent_node[bx * PARAMETR_SIZE + tx] = k;
                    agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
                }
                // Проверка наличия решения в Хэш-таблице
                cachedResult = getCachedResultOptimized_non_cuda(hashTable, agent_node, bx);
                nom_iteration++;
                mtx.lock();
                kol_hash_fail++;
                mtx.unlock();
            }

            OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
            saveToCacheOptimized_non_cuda(hashTable, agent_node, bx, OF[bx]);
            break;
        default:
            OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
            mtx.lock();
            kol_hash_fail++;
            mtx.unlock();
            break;
        }
    }
}

void go_all_agent_non_cuda_thread(double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail, int num_threads) {
    std::vector<std::thread> threads;

    for (int bx = 0; bx < ANT_SIZE; bx++) {
        if (threads.size() >= num_threads) {
            for (auto& thread : threads) {
                thread.join(); // Ожидаем завершения потока
            }
            threads.clear(); // Очищаем вектор потоков для следующей порции
        }
        threads.emplace_back(process_agent, bx, parametr, norm_matrix_probability, agent, agent_node, OF, hashTable, std::ref(kol_hash_fail));
    }
    // Ожидание завершения оставшихся потоков
    for (auto& thread : threads) {
        thread.join();
    }
}

void go_all_agent_non_cuda_non_hash(int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, int& kol_hash_fail, double& totalOFTime) {
    // Генератор случайных чисел
    std::default_random_engine generator(123 + gpuTime); // Используем gpuTime как начальное значение
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int bx = 0; bx < ANT_SIZE; bx++) { // Проходим по всем агентам
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            // Определение номера значения
            int k = 0;
            while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                k++;
            }

            // Запись подматрицы блока в глобальную память
            agent_node[bx * PARAMETR_SIZE + tx] = k;
            agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
        }
        auto start_OF = std::chrono::high_resolution_clock::now();
        OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
        auto end_OF = std::chrono::high_resolution_clock::now();
        totalOFTime += std::chrono::duration<double, std::milli>(end_OF - start_OF).count();
    }
}

void add_pheromon_iteration_omp(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    // Испарение весов-феромона
#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
        for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
            pheromon[MAX_VALUE_SIZE * tx + i] *= PARAMETR_RO;
        }
    }

#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
        // Добавление весов-феромона
        for (int i = 0; i < ANT_SIZE; ++i) {
            int k = agent_node[i * PARAMETR_SIZE + tx];
#pragma omp atomic
            kol_enter[MAX_VALUE_SIZE * tx + k]++;
#if (OPTIMIZE_MIN_1)
            if (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i] > 0) {
                pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i]);
            }
#endif // (OPTIMIZE_MIN_1)
#if (OPTIMIZE_MIN_2)
            if (OF[i] == 0) { OF[i] = 0.0000001; }
            pheromon[MAX_VALUE_SIZE * tx + k] = pheromon[MAX_VALUE_SIZE * tx + k] + PARAMETR_Q / OF[i];
#endif // (OPTIMIZE_MIN_2)
#if (OPTIMIZE_MAX)
            pheromon[MAX_VALUE_SIZE * tx + k] = pheromon[MAX_VALUE_SIZE * tx + k] + PARAMETR_Q * OF[i];
#endif // (OPTIMIZE_MAX)
        }
    }
}

// Обновление слоев графа
void add_pheromon_iteration_non_cuda(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    // Испарение весов-феромона
    for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
        for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
            pheromon[MAX_VALUE_SIZE * tx + i] *= PARAMETR_RO;
        }
    }
    for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
        // Добавление весов-феромона
        for (int i = 0; i < ANT_SIZE; ++i) {
            int k = agent_node[i * PARAMETR_SIZE + tx];
            if (k >= 0 && k < MAX_VALUE_SIZE) { // Проверка на выход за пределы массива kol_enter
                kol_enter[MAX_VALUE_SIZE * tx + k]++;
#if (OPTIMIZE_MIN_1)
                if (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i] > 0) {
                    pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i]);
                }
#endif // (OPTIMIZE_MIN_1)
#if (OPTIMIZE_MIN_2)
                if (OF[i] == 0) { OF[i] = 0.0000001; }
                pheromon[MAX_VALUE_SIZE * tx + k] = pheromon[MAX_VALUE_SIZE * tx + k] + PARAMETR_Q / OF[i];
#endif // (OPTIMIZE_MIN_2)
#if (OPTIMIZE_MAX)
                pheromon[MAX_VALUE_SIZE * tx + k] = pheromon[MAX_VALUE_SIZE * tx + k] + PARAMETR_Q * OF[i];
#endif // (OPTIMIZE_MAX)
            }
        }
    }
}

// Функция для загрузки матрицы из файла
bool load_matrix_non_cuda(const std::string& filename, double* parametr_value, double* pheromon_value, double* kol_enter_value) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Don't open file!" << std::endl;
        return false;
    }

    for (int i = 0; i < PARAMETR_SIZE; ++i) {
        for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
            int k = MAX_VALUE_SIZE * i + j;
            if (!(infile >> parametr_value[k])) { // Чтение элемента в массив a
                std::cerr << "Error load element [" << i << "][" << j << "]" << std::endl;
                return false;
            }

            if (parametr_value[k] != -100) {
                pheromon_value[k] = 1.0; // Присваиваем значение pheromon_value
                kol_enter_value[k] = 1.0;
            }
            else {
                pheromon_value[k] = 0.0; // Присваиваем значение pheromon_value
                parametr_value[k] = 0.0; //Нужно ли????
                kol_enter_value[k] = 0.0;
            }


        }
    }
    infile.close();
    return true;
}

bool load_matrix_transp_non_cuda(const std::string& filename, double* parametr_value, double* pheromon_value, double* kol_enter_value) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Don't open file!" << std::endl;
        return false;
    }

    for (int i = 0; i < PARAMETR_SIZE; ++i) {
        for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
            int k = i + j * PARAMETR_SIZE;
            if (!(infile >> parametr_value[k])) { // Чтение элемента в массив a
                std::cerr << "Error load element [" << i << "][" << j << "]" << std::endl;
                return false;
            }

            if (parametr_value[k] != -100) {
                pheromon_value[k] = 1.0; // Присваиваем значение pheromon_value
                kol_enter_value[k] = 1.0;
            }
            else {
                pheromon_value[k] = 0.0; // Присваиваем значение pheromon_value
                parametr_value[k] = 0.0; //Нужно ли????
                kol_enter_value[k] = 0.0;
            }


        }
    }
    infile.close();
    return true;
}

// Подготовка массива для вероятностного поиска
void go_mass_probability_transp_non_cuda(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    //Сумма Тi для Tnorm
    double sumVectorT[PARAMETR_SIZE] = { 0 };
    double sumVectorZ[PARAMETR_SIZE] = { 0 };
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        sumVectorT[tx] = 0.0;
        sumVectorZ[tx] = 0.0;
    }
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            sumVectorT[tx] += pheromon[tx + i * PARAMETR_SIZE];
        }
    }
    //Вычисление Tnorm
    double* pheromon_norm = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            pheromon_norm[tx + i * PARAMETR_SIZE] = pheromon[tx + i * PARAMETR_SIZE] / sumVectorT[tx];
        }
    }
    //Вычисление Z и P
    double* svertka = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            if ((kol_enter[tx + i * PARAMETR_SIZE] != 0) && (pheromon_norm[tx + i * PARAMETR_SIZE] != 0)) {
                svertka[tx + i * PARAMETR_SIZE] = 1.0 / kol_enter[tx + i * PARAMETR_SIZE] + pheromon_norm[tx + i * PARAMETR_SIZE];
            }
            else
            {
                svertka[tx + i * PARAMETR_SIZE] = 0.0;
            }
            sumVectorZ[tx] += svertka[tx + i * PARAMETR_SIZE];
        }
    }
    //Вычисление F
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        if (i == 0) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                norm_matrix_probability[tx + i * PARAMETR_SIZE] = (svertka[tx + i * PARAMETR_SIZE] / sumVectorZ[tx]);
            }
        }
        else
        {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                norm_matrix_probability[tx + i * PARAMETR_SIZE] = (svertka[tx + i * PARAMETR_SIZE] / sumVectorZ[tx]) + norm_matrix_probability[tx + (i - 1) * PARAMETR_SIZE];
            }
        }
    }
    delete[] pheromon_norm;
    delete[] svertka;
}
// Функция для вычисления пути агентов на CPU
void go_all_agent_transp_non_cuda_time(int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail, double& totalHashTime, double& totalOFTime, double& HashTimeSave, double& HashTimeSearch, double& SumTimeSearch) {
    // Генератор случайных чисел
    std::default_random_engine generator(123 + gpuTime); // Используем gpuTime как начальное значение
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int bx = 0; bx < ANT_SIZE; bx++) { // Проходим по всем агентам
        auto start_ant = std::chrono::high_resolution_clock::now();
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            // Определение номера значения

            int k = 0;
            while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[tx + k * PARAMETR_SIZE]) {
                k++;
            }
            // Запись подматрицы блока в глобальную память
            agent_node[bx * PARAMETR_SIZE + tx] = k;
            agent[bx * PARAMETR_SIZE + tx] = parametr[tx + k * PARAMETR_SIZE];
        }
        auto end_ant = std::chrono::high_resolution_clock::now();
        SumTimeSearch += std::chrono::duration<double, std::milli>(end_ant - start_ant).count();
        auto start = std::chrono::high_resolution_clock::now();
        // Проверка наличия решения в Хэш-таблице
        double cachedResult = getCachedResultOptimized_non_cuda(hashTable, agent_node, bx);
        auto end_OF = std::chrono::high_resolution_clock::now();
        HashTimeSearch += std::chrono::duration<double, std::milli>(end_OF - start).count();
        /*
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            std::cout << agent[bx * PARAMETR_SIZE + j] << " ";
        }
        std::cout << "-> " << cachedResult << std::endl;*/
        int nom_iteration = 0;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            auto start_OF = std::chrono::high_resolution_clock::now();
            OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
            auto end_OF = std::chrono::high_resolution_clock::now();
            totalOFTime += std::chrono::duration<double, std::milli>(end_OF - start_OF).count();
            auto start_SaveHash = std::chrono::high_resolution_clock::now();
            saveToCacheOptimized_non_cuda(hashTable, agent_node, bx, OF[bx]);
            auto end_SaveHash = std::chrono::high_resolution_clock::now();
            HashTimeSave += std::chrono::duration<double, std::milli>(end_SaveHash - start_SaveHash).count();
        }
        else {
            auto start_OF_2 = std::chrono::high_resolution_clock::now();
            auto end_OF_2 = std::chrono::high_resolution_clock::now();
            //Если значение в Хэш-найдено, то агент "нулевой"
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                kol_hash_fail = kol_hash_fail + 1;
                break;
            case 1: // ACOCNI
                OF[bx] = ZERO_HASH_RESULT;
                kol_hash_fail = kol_hash_fail + 1;
                break;
            case 2: // ACOCCyN
                while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                {
                    start_OF_2 = std::chrono::high_resolution_clock::now();
                    for (int tx = 0; tx < PARAMETR_SIZE; ++tx) { // Проходим по всем параметрам
                        double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]

                        // Определение номера значения
                        int k = 0;
                        while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[tx + k * PARAMETR_SIZE]) {
                            k++;
                        }

                        // Запись подматрицы блока в глобальную память
                        agent_node[bx * PARAMETR_SIZE + tx] = k;
                        agent[bx * PARAMETR_SIZE + tx] = parametr[tx + k * PARAMETR_SIZE];
                    }
                    end_OF_2 = std::chrono::high_resolution_clock::now();
                    SumTimeSearch += std::chrono::duration<double, std::milli>(end_OF_2 - start_OF_2).count();
                    // Проверка наличия решения в Хэш-таблице
                    start_OF_2 = std::chrono::high_resolution_clock::now();
                    cachedResult = getCachedResultOptimized_non_cuda(hashTable, agent_node, bx);
                    end_OF_2 = std::chrono::high_resolution_clock::now();
                    HashTimeSearch += std::chrono::duration<double, std::milli>(end_OF_2 - start_OF_2).count();
                    nom_iteration = nom_iteration + 1;
                    kol_hash_fail = kol_hash_fail + 1;
                }

                start_OF_2 = std::chrono::high_resolution_clock::now();
                OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
                end_OF_2 = std::chrono::high_resolution_clock::now();
                totalOFTime += std::chrono::duration<double, std::milli>(end_OF_2 - start_OF_2).count();
                start_OF_2 = std::chrono::high_resolution_clock::now();
                saveToCacheOptimized_non_cuda(hashTable, agent_node, bx, OF[bx]);
                end_OF_2 = std::chrono::high_resolution_clock::now();
                HashTimeSave += std::chrono::duration<double, std::milli>(end_OF_2 - start_OF_2).count();
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                kol_hash_fail = kol_hash_fail + 1;
                break;
            }


        }
        //std::cout << bx << "bx " << kol_hash_fail << " " << OF[bx] << " ";
        auto end = std::chrono::high_resolution_clock::now();
        totalHashTime += std::chrono::duration<double, std::milli>(end - start).count();
    }
}
void go_all_agent_transp_non_cuda(int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail) {
    // Генератор случайных чисел
    std::default_random_engine generator(123 + gpuTime); // Используем gpuTime как начальное значение
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int bx = 0; bx < ANT_SIZE; bx++) { // Проходим по всем агентам
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            // Определение номера значения
            int k = 0;
            while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[tx + k * PARAMETR_SIZE]) {
                k++;
            }
            // Запись подматрицы блока в глобальную память
            agent_node[bx * PARAMETR_SIZE + tx] = k;
            agent[bx * PARAMETR_SIZE + tx] = parametr[tx + k * PARAMETR_SIZE];
        }
        // Проверка наличия решения в Хэш-таблице
        double cachedResult = getCachedResultOptimized_non_cuda(hashTable, agent_node, bx);
        int nom_iteration = 0;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
            saveToCacheOptimized_non_cuda(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            //Если значение в Хэш-найдено, то агент "нулевой"
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                kol_hash_fail = kol_hash_fail + 1;
                break;
            case 1: // ACOCNI
                OF[bx] = ZERO_HASH_RESULT;
                kol_hash_fail = kol_hash_fail + 1;
                break;
            case 2: // ACOCCyN
                while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                {
                    for (int tx = 0; tx < PARAMETR_SIZE; ++tx) { // Проходим по всем параметрам
                        double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
                        // Определение номера значения
                        int k = 0;
                        while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[tx + k * PARAMETR_SIZE]) {
                            k++;
                        }
                        // Запись подматрицы блока в глобальную память
                        agent_node[bx * PARAMETR_SIZE + tx] = k;
                        agent[bx * PARAMETR_SIZE + tx] = parametr[tx + k * PARAMETR_SIZE];
                    }
                    // Проверка наличия решения в Хэш-таблице
                    cachedResult = getCachedResultOptimized_non_cuda(hashTable, agent_node, bx);
                    nom_iteration = nom_iteration + 1;
                    kol_hash_fail = kol_hash_fail + 1;
                }
                OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
                saveToCacheOptimized_non_cuda(hashTable, agent_node, bx, OF[bx]);
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                kol_hash_fail = kol_hash_fail + 1;
                break;
            }
        }
        //std::cout << bx << "bx " << kol_hash_fail << " " << OF[bx] << " ";
    }
}
void go_all_agent_transp_non_cuda_non_hash(int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, int& kol_hash_fail, double& totalOFTime) {
    // Генератор случайных чисел
    std::default_random_engine generator(123 + gpuTime); // Используем gpuTime как начальное значение
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int bx = 0; bx < ANT_SIZE; bx++) { // Проходим по всем агентам
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            // Определение номера значения
            int k = 0;
            while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[tx + k * PARAMETR_SIZE]) {
                k++;
            }

            // Запись подматрицы блока в глобальную память
            agent_node[bx * PARAMETR_SIZE + tx] = k;
            agent[bx * PARAMETR_SIZE + tx] = parametr[tx + k * PARAMETR_SIZE];
        }
        auto start_OF = std::chrono::high_resolution_clock::now();
        OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
        auto end_OF = std::chrono::high_resolution_clock::now();
        totalOFTime += std::chrono::duration<double, std::milli>(end_OF - start_OF).count();
    }
}
// Обновление слоев графа
void add_pheromon_iteration_transp_non_cuda(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    // Испарение весов-феромона
    for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            pheromon[tx + i * PARAMETR_SIZE] *= PARAMETR_RO;
        }
    }
    for (int i = 0; i < ANT_SIZE; ++i) {
        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            int k = agent_node[tx + i * PARAMETR_SIZE];
            kol_enter[tx + k * PARAMETR_SIZE]++;
            //            pheromon[tx + k * PARAMETR_SIZE] += PARAMETR_Q * OF[i]; // MAX
            //            pheromon[tx + k * PARAMETR_SIZE] += PARAMETR_Q / OF[i]; // MIN
            if (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i] > 0) {
                pheromon[tx + k * PARAMETR_SIZE] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i]); // MIN
            }
        }
    }
}

int start_omp() {
    auto start = std::chrono::high_resolution_clock::now();
    double SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumgpuTime4 = 0.0f, SumgpuTime5 = 0.0f, SumgpuTime6 = 0.0f, SumgpuTime7 = 0.0f;
    double duration = 0.0f, duration_iteration = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    int kol_hash_fail = 0;
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;

    // Выделение памяти для хэш-таблицы на CPU
    HashEntry* hashTable = new HashEntry[HASH_TABLE_SIZE];
    // Вызов функции инициализации
    initializeHashTable_omp(hashTable, HASH_TABLE_SIZE);

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

    // Загрузка матрицы из файла
    load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);

    auto start_iteration = std::chrono::high_resolution_clock::now();

    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        auto start1 = std::chrono::high_resolution_clock::now();

        // Расчет нормализованной вероятности
        go_mass_probability_omp(pheromon_value, kol_enter_value, norm_matrix_probability);

        if (PRINT_INFORMATION) {
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "("
                        << pheromon_value[i * MAX_VALUE_SIZE + j] << ", "
                        << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> "
                        << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") ";
                }
                std::cout << std::endl;
            }
        }

        // Вычисление пути агентов
        auto start2 = std::chrono::high_resolution_clock::now();
        auto end_temp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> current_time = end_temp - start;
        go_all_agent_omp(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail, SumgpuTime4, SumgpuTime5);

        if (PRINT_INFORMATION) {
            std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                }
                std::cout << "-> " << antOF[i] << std::endl;
            }
        }

        auto start3 = std::chrono::high_resolution_clock::now();

        // Обновление весов-феромонов
        add_pheromon_iteration_omp(pheromon_value, kol_enter_value, ant_parametr, antOF);

        // Поиск максимума и минимума
        double maxOf = -std::numeric_limits<double>::max();
        double minOf = std::numeric_limits<double>::max();

#pragma omp parallel for reduction(max:maxOf) reduction(min:minOf)
        for (int i = 0; i < ANT_SIZE; ++i) {
            if (antOF[i] != ZERO_HASH_RESULT) {
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
        }

        // Обновление глобальных максимумов и минимумов
#pragma omp atomic
        global_maxOf = std::max(global_maxOf, maxOf);
#pragma omp atomic
        global_minOf = std::min(global_minOf, minOf);

        auto end_iter = std::chrono::high_resolution_clock::now();
        SumgpuTime1 += std::chrono::duration<double, std::milli>(end_iter - start1).count();
        SumgpuTime2 += std::chrono::duration<double, std::milli>(end_iter - start2).count();
        SumgpuTime3 += std::chrono::duration<double, std::milli>(end_iter - start3).count();
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, kol_hash_fail);
        }
    }
    auto end_iteration = std::chrono::high_resolution_clock::now();
    duration_iteration += std::chrono::duration<double, std::milli>(end_iteration - start_iteration).count();

    // Освобождение памяти в конце программы
    delete[] hashTable;               // Освобождение памяти для хэш-таблицы
    delete[] parametr_value;          // Освобождение памяти для параметров
    delete[] pheromon_value;          // Освобождение памяти для феромонов
    delete[] kol_enter_value;         // Освобождение памяти для количества входов
    delete[] norm_matrix_probability; // Освобождение памяти для нормализованной матрицы вероятностей
    delete[] ant;                     // Освобождение памяти для муравьев
    delete[] ant_parametr;            // Освобождение памяти для параметров муравьев
    delete[] antOF;                   // Освобождение памяти для результата муравьев

    auto end = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Time omp:;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumgpuTime4 << "; " << SumgpuTime5 << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time omp:;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumgpuTime4 << "; " << SumgpuTime5 << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    // Возвращаем результат или выводим информацию о времени
    return 0; // или другой результат по необходимости
}

int start_omp_non_hash() {
    auto start = std::chrono::high_resolution_clock::now();
    double SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumgpuTime4 = 0.0f, SumgpuTime5 = 0.0f, SumgpuTime6 = 0.0f, SumgpuTime7 = 0.0f;
    double duration = 0.0f, duration_iteration = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    int kol_hash_fail = 0;
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;

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

    // Загрузка матрицы из файла
    load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);

    auto start_iteration = std::chrono::high_resolution_clock::now();

    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        auto start1 = std::chrono::high_resolution_clock::now();

        // Расчет нормализованной вероятности
        go_mass_probability_omp(pheromon_value, kol_enter_value, norm_matrix_probability);

        if (PRINT_INFORMATION) {
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "("
                        << pheromon_value[i * MAX_VALUE_SIZE + j] << ", "
                        << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> "
                        << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") ";
                }
                std::cout << std::endl;
            }
        }

        // Вычисление пути агентов
        auto start2 = std::chrono::high_resolution_clock::now();
        auto end_temp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> current_time = end_temp - start;
        go_all_agent_omp_non_hash(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, kol_hash_fail, SumgpuTime4, SumgpuTime5);

        if (PRINT_INFORMATION) {
            std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                }
                std::cout << "-> " << antOF[i] << std::endl;
            }
        }

        auto start3 = std::chrono::high_resolution_clock::now();

        // Обновление весов-феромонов
        add_pheromon_iteration_omp(pheromon_value, kol_enter_value, ant_parametr, antOF);

        // Поиск максимума и минимума
        double maxOf = -std::numeric_limits<double>::max();
        double minOf = std::numeric_limits<double>::max();

#pragma omp parallel for reduction(max:maxOf) reduction(min:minOf)
        for (int i = 0; i < ANT_SIZE; ++i) {
            if (antOF[i] != ZERO_HASH_RESULT) {
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
        }

        // Обновление глобальных максимумов и минимумов
#pragma omp atomic
        global_maxOf = std::max(global_maxOf, maxOf);
#pragma omp atomic
        global_minOf = std::min(global_minOf, minOf);

        auto end_iter = std::chrono::high_resolution_clock::now();
        SumgpuTime1 += std::chrono::duration<double, std::milli>(end_iter - start1).count();
        SumgpuTime2 += std::chrono::duration<double, std::milli>(end_iter - start2).count();
        SumgpuTime3 += std::chrono::duration<double, std::milli>(end_iter - start3).count();
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, kol_hash_fail);
        }
    }
    auto end_iteration = std::chrono::high_resolution_clock::now();
    duration_iteration += std::chrono::duration<double, std::milli>(end_iteration - start_iteration).count();

    // Освобождение памяти в конце программы
    delete[] parametr_value;          // Освобождение памяти для параметров
    delete[] pheromon_value;          // Освобождение памяти для феромонов
    delete[] kol_enter_value;         // Освобождение памяти для количества входов
    delete[] norm_matrix_probability; // Освобождение памяти для нормализованной матрицы вероятностей
    delete[] ant;                     // Освобождение памяти для муравьев
    delete[] ant_parametr;            // Освобождение памяти для параметров муравьев
    delete[] antOF;                   // Освобождение памяти для результата муравьев

    auto end = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Time omp non hash:;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumgpuTime4 << "; " << SumgpuTime5 << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time omp non hash:;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumgpuTime4 << "; " << SumgpuTime5 << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    // Возвращаем результат или выводим информацию о времени
    return 0; // или другой результат по необходимости
}

int start_NON_CUDA_time() {
    auto start = std::chrono::high_resolution_clock::now();
    double SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumTimeHashTotal = 0.0f, SumTimeOF = 0.0f, SumTimeHashSearch = 0.0f, SumTimeHashSave = 0.0f, SumTimeSearchAgent = 0.0f;
    double duration = 0.0f, duration_iteration = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    int kol_hash_fail = 0;
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    // Выделение памяти для хэш-таблицы на CPU
    HashEntry* hashTable = new HashEntry[HASH_TABLE_SIZE];
    // Вызов функции инициализации
    initializeHashTable_non_cuda(hashTable, HASH_TABLE_SIZE);

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

    // Загрузка матрицы из файла
    load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);

    auto start_iteration = std::chrono::high_resolution_clock::now();
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        auto start1 = std::chrono::high_resolution_clock::now();
        // Расчет нормализованной вероятности
        go_mass_probability_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

        if (PRINT_INFORMATION) {
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }

        // Вычисление пути агентов

        auto start2 = std::chrono::high_resolution_clock::now();
        auto end_temp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> current_time = end_temp - start;
        go_all_agent_non_cuda_time(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail, SumTimeHashTotal, SumTimeOF, SumTimeHashSearch, SumTimeHashSave, SumTimeSearchAgent);

        if (PRINT_INFORMATION) {
            std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";

                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        auto start3 = std::chrono::high_resolution_clock::now();
        // Обновление весов-феромонов
        add_pheromon_iteration_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

        // Поиск максимума и минимума
        double maxOf = -std::numeric_limits<double>::max();
        double minOf = std::numeric_limits<double>::max();
        for (int i = 0; i < ANT_SIZE; ++i) {
            if (antOF[i] != ZERO_HASH_RESULT) {
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
        }

        // Обновление глобальных максимумов и минимумов
        if (minOf < global_minOf) {
            global_minOf = minOf;
        }
        if (maxOf > global_maxOf) {
            global_maxOf = maxOf;
        }
        auto end_iter = std::chrono::high_resolution_clock::now();
        SumgpuTime1 += std::chrono::duration<double, std::milli>(end_iter - start1).count();
        SumgpuTime2 += std::chrono::duration<double, std::milli>(end_iter - start2).count();
        SumgpuTime3 += std::chrono::duration<double, std::milli>(end_iter - start3).count();
        if (PRINT_INFORMATION) {
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << std::endl;
        }
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumTimeHashTotal, SumTimeOF, SumTimeHashSearch, SumTimeHashSave, SumTimeSearchAgent, global_minOf, global_maxOf, kol_hash_fail);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    duration_iteration += std::chrono::duration<double, std::milli>(end - start_iteration).count();

    // Освобождение памяти в конце программы
    delete[] hashTable;               // Освобождение памяти для хэш-таблицы
    delete[] parametr_value;          // Освобождение памяти для параметров
    delete[] pheromon_value;          // Освобождение памяти для феромонов
    delete[] kol_enter_value;         // Освобождение памяти для количества входов
    delete[] norm_matrix_probability; // Освобождение памяти для нормализованной матрицы вероятностей
    delete[] ant;                     // Освобождение памяти для муравьев
    delete[] ant_parametr;            // Освобождение памяти для параметров муравьев
    delete[] antOF;                   // Освобождение памяти для результата муравьев

    auto end_all = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double, std::milli>(end_all - start).count();
    std::cout << "Time non CUDA_time;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA_time;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}

int start_NON_CUDA_thread() {
    auto start = std::chrono::high_resolution_clock::now();
    float SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumTimeHashTotal = 0.0f, SumTimeOF = 0.0f, SumTimeHashSearch = 0.0f, SumTimeHashSave = 0.0f, SumTimeSearchAgent = 0.0f;
    double duration = 0.0f, duration_iteration = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    int kol_hash_fail = 0;
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    // Выделение памяти для хэш-таблицы на CPU
    HashEntry* hashTable = new HashEntry[HASH_TABLE_SIZE];
    // Вызов функции инициализации
    initializeHashTable_non_cuda(hashTable, HASH_TABLE_SIZE);

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

    // Загрузка матрицы из файла
    load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        // Расчет нормализованной вероятности
        go_mass_probability_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

        if (PRINT_INFORMATION) {
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }

        // Вычисление пути агентов
        go_all_agent_non_cuda_thread(parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail, KOL_THREAD_CPU_ANT);

        if (PRINT_INFORMATION) {
            std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";

                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }
        // Обновление весов-феромонов
        add_pheromon_iteration_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

        // Поиск максимума и минимума
        double maxOf = -std::numeric_limits<double>::max();
        double minOf = std::numeric_limits<double>::max();
        for (int i = 0; i < ANT_SIZE; ++i) {
            if (antOF[i] != ZERO_HASH_RESULT) {
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
        }

        // Обновление глобальных максимумов и минимумов
        if (minOf < global_minOf) {
            global_minOf = minOf;
        }
        if (maxOf > global_maxOf) {
            global_maxOf = maxOf;
        }
        if (PRINT_INFORMATION) {
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << std::endl;
        }
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumTimeHashTotal, SumTimeOF, SumTimeHashSearch, SumTimeHashSave, SumTimeSearchAgent, global_minOf, global_maxOf, kol_hash_fail);
        }
    }

    // Освобождение памяти в конце программы
    delete[] hashTable;               // Освобождение памяти для хэш-таблицы
    delete[] parametr_value;          // Освобождение памяти для параметров
    delete[] pheromon_value;          // Освобождение памяти для феромонов
    delete[] kol_enter_value;         // Освобождение памяти для количества входов
    delete[] norm_matrix_probability; // Освобождение памяти для нормализованной матрицы вероятностей
    delete[] ant;                     // Освобождение памяти для муравьев
    delete[] ant_parametr;            // Освобождение памяти для параметров муравьев
    delete[] antOF;                   // Освобождение памяти для результата муравьев

    auto end_all = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double, std::milli>(end_all - start).count();
    std::cout << "Time non CUDA thread;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA thread;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}

int start_NON_CUDA() {
    auto start = std::chrono::high_resolution_clock::now();
    float SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumTimeHashTotal = 0.0f, SumTimeOF = 0.0f, SumTimeHashSearch = 0.0f, SumTimeHashSave = 0.0f;
    double duration = 0.0f, duration_iteration = 0.0f;
    int kol_hash_fail = 0;
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    // Выделение памяти для хэш-таблицы на CPU
    HashEntry* hashTable = new HashEntry[HASH_TABLE_SIZE];
    // Вызов функции инициализации
    initializeHashTable_non_cuda(hashTable, HASH_TABLE_SIZE);

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

    // Загрузка матрицы из файла
    load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        // Расчет нормализованной вероятности
        go_mass_probability_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

        // Вычисление пути агентов
        auto end_temp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> current_time = end_temp - start;
        go_all_agent_non_cuda(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail);

        // Обновление весов-феромонов
        add_pheromon_iteration_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

        // Поиск максимума и минимума
        double maxOf = -std::numeric_limits<double>::max();
        double minOf = std::numeric_limits<double>::max();
        for (int i = 0; i < ANT_SIZE; ++i) {
            if (antOF[i] != ZERO_HASH_RESULT) {
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
        }
        // Обновление глобальных максимумов и минимумов
        if (minOf < global_minOf) {
            global_minOf = minOf;
        }
        if (maxOf > global_maxOf) {
            global_maxOf = maxOf;
        }
    }

    // Освобождение памяти в конце программы
    delete[] hashTable;               // Освобождение памяти для хэш-таблицы
    delete[] parametr_value;          // Освобождение памяти для параметров
    delete[] pheromon_value;          // Освобождение памяти для феромонов
    delete[] kol_enter_value;         // Освобождение памяти для количества входов
    delete[] norm_matrix_probability; // Освобождение памяти для нормализованной матрицы вероятностей
    delete[] ant;                     // Освобождение памяти для муравьев
    delete[] ant_parametr;            // Освобождение памяти для параметров муравьев
    delete[] antOF;                   // Освобождение памяти для результата муравьев

    auto end_all = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double, std::milli>(end_all - start).count();
    std::cout << "Time non CUDA;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}

int start_NON_CUDA_non_hash() {
    auto start = std::chrono::high_resolution_clock::now();
    double SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumgpuTime5 = 0.0f, SumgpuTime6 = 0.0f, SumgpuTime7 = 0.0f;
    double duration = 0.0f, duration_iteration = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    int kol_hash_fail = 0;
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;

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

    // Загрузка матрицы из файла
    load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);

    auto start_iteration = std::chrono::high_resolution_clock::now();
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        auto start1 = std::chrono::high_resolution_clock::now();
        // Расчет нормализованной вероятности
        go_mass_probability_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

        if (PRINT_INFORMATION) {
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }

        // Вычисление пути агентов

        auto start2 = std::chrono::high_resolution_clock::now();
        auto end_temp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> current_time = end_temp - start;
        go_all_agent_non_cuda_non_hash(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, kol_hash_fail, SumgpuTime5);

        if (PRINT_INFORMATION) {
            std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        auto start3 = std::chrono::high_resolution_clock::now();
        // Обновление весов-феромонов
        add_pheromon_iteration_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

        // Поиск максимума и минимума
        double maxOf = -std::numeric_limits<double>::max();
        double minOf = std::numeric_limits<double>::max();
        for (int i = 0; i < ANT_SIZE; ++i) {
            if (antOF[i] != ZERO_HASH_RESULT) {
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
        }

        // Обновление глобальных максимумов и минимумов
        if (minOf < global_minOf) {
            global_minOf = minOf;
        }
        if (maxOf > global_maxOf) {
            global_maxOf = maxOf;
        }
        auto end_iter = std::chrono::high_resolution_clock::now();
        SumgpuTime1 += std::chrono::duration<double, std::milli>(end_iter - start1).count();
        SumgpuTime2 += std::chrono::duration<double, std::milli>(end_iter - start2).count();
        SumgpuTime3 += std::chrono::duration<double, std::milli>(end_iter - start3).count();
        if (PRINT_INFORMATION) {
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << std::endl;
        }
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, 0, SumgpuTime5, SumgpuTime6, SumgpuTime7, global_minOf, 0, global_maxOf, kol_hash_fail);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    duration_iteration += std::chrono::duration<double, std::milli>(end - start_iteration).count();

    // Освобождение памяти в конце программы
    delete[] parametr_value;          // Освобождение памяти для параметров
    delete[] pheromon_value;          // Освобождение памяти для феромонов
    delete[] kol_enter_value;         // Освобождение памяти для количества входов
    delete[] norm_matrix_probability; // Освобождение памяти для нормализованной матрицы вероятностей
    delete[] ant;                     // Освобождение памяти для муравьев
    delete[] ant_parametr;            // Освобождение памяти для параметров муравьев
    delete[] antOF;                   // Освобождение памяти для результата муравьев

    auto end_all = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double, std::milli>(end_all - start).count();
    std::cout << "Time non CUDA non hash;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << "0" << "; " << SumgpuTime5 << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA non hash:;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << "0" << "; " << SumgpuTime5 << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}

int start_NON_CUDA_transp_time() {
    auto start = std::chrono::high_resolution_clock::now();
    double SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumTimeHashTotal = 0.0f, SumTimeOF = 0.0f, SumTimeHashSearch = 0.0f, SumTimeHashSave = 0.0f, SumTimeSearchAgent = 0.0f;
    double duration = 0.0f, duration_iteration = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    int kol_hash_fail = 0;
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    // Выделение памяти для хэш-таблицы на CPU
    HashEntry* hashTable = new HashEntry[HASH_TABLE_SIZE];
    // Вызов функции инициализации
    initializeHashTable_non_cuda(hashTable, HASH_TABLE_SIZE);

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

    // Загрузка матрицы из файла
    load_matrix_transp_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);

    auto start_iteration = std::chrono::high_resolution_clock::now();
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        auto start1 = std::chrono::high_resolution_clock::now();
        // Расчет нормализованной вероятности
        go_mass_probability_transp_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

        if (PRINT_INFORMATION) {
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i + j * PARAMETR_SIZE] << "(" << pheromon_value[i + j * PARAMETR_SIZE] << ", " << kol_enter_value[i + j * PARAMETR_SIZE] << "-> " << norm_matrix_probability[i + j * PARAMETR_SIZE] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }

        // Вычисление пути агентов

        auto start2 = std::chrono::high_resolution_clock::now();
        auto end_temp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> current_time = end_temp - start;
        go_all_agent_transp_non_cuda_time(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail, SumTimeHashTotal, SumTimeOF, SumTimeHashSearch, SumTimeHashSave, SumTimeSearchAgent);

        if (PRINT_INFORMATION) {
            std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";

                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        auto start3 = std::chrono::high_resolution_clock::now();
        // Обновление весов-феромонов
        add_pheromon_iteration_transp_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

        // Поиск максимума и минимума
        double maxOf = -std::numeric_limits<double>::max();
        double minOf = std::numeric_limits<double>::max();
        for (int i = 0; i < ANT_SIZE; ++i) {
            if (antOF[i] != ZERO_HASH_RESULT) {
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
        }

        // Обновление глобальных максимумов и минимумов
        if (minOf < global_minOf) {
            global_minOf = minOf;
        }
        if (maxOf > global_maxOf) {
            global_maxOf = maxOf;
        }
        auto end_iter = std::chrono::high_resolution_clock::now();
        SumgpuTime1 += std::chrono::duration<double, std::milli>(end_iter - start1).count();
        SumgpuTime2 += std::chrono::duration<double, std::milli>(end_iter - start2).count();
        SumgpuTime3 += std::chrono::duration<double, std::milli>(end_iter - start3).count();
        if (PRINT_INFORMATION) {
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << std::endl;
        }
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumTimeHashTotal, SumTimeOF, SumTimeHashSearch, SumTimeHashSave, SumTimeSearchAgent, global_minOf, global_maxOf, kol_hash_fail);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    duration_iteration += std::chrono::duration<double, std::milli>(end - start_iteration).count();

    // Освобождение памяти в конце программы
    delete[] hashTable;               // Освобождение памяти для хэш-таблицы
    delete[] parametr_value;          // Освобождение памяти для параметров
    delete[] pheromon_value;          // Освобождение памяти для феромонов
    delete[] kol_enter_value;         // Освобождение памяти для количества входов
    delete[] norm_matrix_probability; // Освобождение памяти для нормализованной матрицы вероятностей
    delete[] ant;                     // Освобождение памяти для муравьев
    delete[] ant_parametr;            // Освобождение памяти для параметров муравьев
    delete[] antOF;                   // Освобождение памяти для результата муравьев

    auto end_all = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double, std::milli>(end_all - start).count();
    std::cout << "Time non CUDA_transp_time;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA_transp_time;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}

int start_NON_CUDA_transp() {
    auto start = std::chrono::high_resolution_clock::now();
    float SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumTimeHashTotal = 0.0f, SumTimeOF = 0.0f, SumTimeHashSearch = 0.0f, SumTimeHashSave = 0.0f;
    double duration = 0.0f, duration_iteration = 0.0f;
    int kol_hash_fail = 0;
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    // Выделение памяти для хэш-таблицы на CPU
    HashEntry* hashTable = new HashEntry[HASH_TABLE_SIZE];
    // Вызов функции инициализации
    initializeHashTable_non_cuda(hashTable, HASH_TABLE_SIZE);

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

    // Загрузка матрицы из файла
    load_matrix_transp_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        // Расчет нормализованной вероятности
        go_mass_probability_transp_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

        // Вычисление пути агентов
        auto end_temp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> current_time = end_temp - start;
        go_all_agent_transp_non_cuda(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail);

        // Обновление весов-феромонов
        add_pheromon_iteration_transp_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

        // Поиск максимума и минимума
        double maxOf = -std::numeric_limits<double>::max();
        double minOf = std::numeric_limits<double>::max();
        for (int i = 0; i < ANT_SIZE; ++i) {
            if (antOF[i] != ZERO_HASH_RESULT) {
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
        }

        // Обновление глобальных максимумов и минимумов
        if (minOf < global_minOf) {
            global_minOf = minOf;
        }
        if (maxOf > global_maxOf) {
            global_maxOf = maxOf;
        }
    }

    // Освобождение памяти в конце программы
    delete[] hashTable;               // Освобождение памяти для хэш-таблицы
    delete[] parametr_value;          // Освобождение памяти для параметров
    delete[] pheromon_value;          // Освобождение памяти для феромонов
    delete[] kol_enter_value;         // Освобождение памяти для количества входов
    delete[] norm_matrix_probability; // Освобождение памяти для нормализованной матрицы вероятностей
    delete[] ant;                     // Освобождение памяти для муравьев
    delete[] ant_parametr;            // Освобождение памяти для параметров муравьев
    delete[] antOF;                   // Освобождение памяти для результата муравьев

    auto end_all = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double, std::milli>(end_all - start).count();
    std::cout << "Time non CUDA_transp;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA_transp;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}

int start_NON_CUDA_transp_non_hash() {
    auto start = std::chrono::high_resolution_clock::now();
    double SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumgpuTime5 = 0.0f, SumgpuTime6 = 0.0f, SumgpuTime7 = 0.0f;
    double duration = 0.0f, duration_iteration = 0.0f;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    int kol_hash_fail = 0;
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;

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

    // Загрузка матрицы из файла
    load_matrix_transp_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);

    auto start_iteration = std::chrono::high_resolution_clock::now();
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        auto start1 = std::chrono::high_resolution_clock::now();
        // Расчет нормализованной вероятности
        go_mass_probability_transp_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

        if (PRINT_INFORMATION) {
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i + j * PARAMETR_SIZE] << "(" << pheromon_value[i + j * PARAMETR_SIZE] << ", " << kol_enter_value[i + j * PARAMETR_SIZE] << "-> " << norm_matrix_probability[i + j * PARAMETR_SIZE] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }

        // Вычисление пути агентов

        auto start2 = std::chrono::high_resolution_clock::now();
        auto end_temp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> current_time = end_temp - start;
        go_all_agent_transp_non_cuda_non_hash(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, kol_hash_fail, SumgpuTime5);

        if (PRINT_INFORMATION) {
            std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        auto start3 = std::chrono::high_resolution_clock::now();
        // Обновление весов-феромонов
        add_pheromon_iteration_transp_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

        // Поиск максимума и минимума
        double maxOf = -std::numeric_limits<double>::max();
        double minOf = std::numeric_limits<double>::max();
        for (int i = 0; i < ANT_SIZE; ++i) {
            if (antOF[i] != ZERO_HASH_RESULT) {
                if (antOF[i] > maxOf) {
                    maxOf = antOF[i];
                }
                if (antOF[i] < minOf) {
                    minOf = antOF[i];
                }
            }
        }

        // Обновление глобальных максимумов и минимумов
        if (minOf < global_minOf) {
            global_minOf = minOf;
        }
        if (maxOf > global_maxOf) {
            global_maxOf = maxOf;
        }
        auto end_iter = std::chrono::high_resolution_clock::now();
        SumgpuTime1 += std::chrono::duration<double, std::milli>(end_iter - start1).count();
        SumgpuTime2 += std::chrono::duration<double, std::milli>(end_iter - start2).count();
        SumgpuTime3 += std::chrono::duration<double, std::milli>(end_iter - start3).count();
        if (PRINT_INFORMATION) {
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << std::endl;
        }
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, 0, SumgpuTime5, SumgpuTime6, SumgpuTime7, global_minOf, 0, global_maxOf, kol_hash_fail);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    duration_iteration += std::chrono::duration<double, std::milli>(end - start_iteration).count();

    // Освобождение памяти в конце программы
    delete[] parametr_value;          // Освобождение памяти для параметров
    delete[] pheromon_value;          // Освобождение памяти для феромонов
    delete[] kol_enter_value;         // Освобождение памяти для количества входов
    delete[] norm_matrix_probability; // Освобождение памяти для нормализованной матрицы вероятностей
    delete[] ant;                     // Освобождение памяти для муравьев
    delete[] ant_parametr;            // Освобождение памяти для параметров муравьев
    delete[] antOF;                   // Освобождение памяти для результата муравьев
    auto end_all = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double, std::milli>(end_all - start).count();
    std::cout << "Time non CUDA_transp non hash;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << "0" << "; " << SumgpuTime5 << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA_transp non hash:;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << "0" << "; " << SumgpuTime5 << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    return 0;
}

class Node {  // Узел графа
public:
    // Параметры узла
    double pheromon;
    int KolSolution;
    double pheromonNorm;
    int KolSolutionNorm;
    int KolSolutionAll;

    // Конструктор для инициализации узла
    void init(double value) {
        clear(1);
        val = value;
    }

    // Метод очистки узла
    void clear(int allClear) {
        pheromon = 1;
        KolSolution = 0;
        pheromonNorm = 1;
        KolSolutionNorm = 1;

        if (allClear == 1) {
            KolSolutionAll = 0;
        }
    }

    // Метод уменьшения феромонов
    void DecreasePheromon(double par) {
        pheromon *= par;
    }

    double val; // Значение узла
};

class Parametr {
public:
    // Инициализация параметра
    void init(const std::string& value) {
        name = value;
        node.clear();  // Очищаем вектор узлов перед добавлением
    }

    // Очистка всех узлов
    void ClearAllNode(int allClear) {
        for (size_t NomEl = 0; NomEl < node.size(); ++NomEl) {
            node[NomEl].clear(allClear);
        }
    }

    // Уменьшение феромонов во всех узлах
    void DecreasePheromon(double par) {
        for (size_t NomEl = 0; NomEl < node.size(); ++NomEl) {
            node[NomEl].DecreasePheromon(par);
        }
    }

    std::string name;  // Имя параметра
    std::vector<Node> node;  // Вектор узлов
};

// Функция загрузки графа из файла
bool LoadGraph(const std::string& filename, std::vector<Parametr>& parameters) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Не удалось открыть файл: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double value;
        Parametr param;

        // Инициализация параметра (можно задать имя, если нужно)
        param.init("Parameter"); // Замените "Parameter" на нужное имя

        while (iss >> value) {
            if (value != -100.0) { // Проверяем, что значение не -100.0
                Node node{};
                node.init(value); // Инициализируем узел
                param.node.push_back(node); // Добавляем узел в параметр
            }
        }

        if (!param.node.empty()) { // Добавляем параметр только если есть узлы
            parameters.push_back(param);
        }
    }

    file.close();
    return true;
}

class PG {
public:
    static double alf1; // 1
    static double alf2; // 1
    static double alf3; // 1
    static double koef1; // 1
    static double koef2; // 1
    static double koef3; // 0
    static int AllSolution;

    std::vector<Parametr> ParametricGraph; // Массив параметров

    double ProbabilityNode(const Node& node) {
        double kolSolution = node.KolSolution;
        if (kolSolution == 0) {
            kolSolution = 0.5;
        }
        double Probability = koef1 * std::pow(node.pheromonNorm, alf1) +
            koef2 * std::pow(1 / kolSolution, alf2); // + koef3 * std::pow(node.KolSolutionAll / AllSolution, alf3);

        if (Probability == 0) {
            Probability = 0.00000001; // Минимальное значение вероятности
        }
        return Probability;
    }

    int GoAntNextNode(const std::vector<Node>& arrayNode) {
        std::vector<double> probability(arrayNode.size());
        double sum = 0.0;

        for (size_t i = 0; i < arrayNode.size(); ++i) {
            sum += ProbabilityNode(arrayNode[i]);
            probability[i] = sum;
        }

        double rnd = static_cast<double>(rand()) / RAND_MAX; // Генерация случайного числа от 0 до 1
        int i = 0;

        while (i < probability.size() && rnd > probability[i] / sum) {
            i++;
        }

        return i < probability.size() ? i : -1; // Возвращаем индекс выбранного узла или -1, если не найден
    }

    std::vector<int> next() { // Предполагается, что возвращаемый тип - вектор целых чисел
        std::vector<int> way;
        int NomParametr = 0;

        // Окончание движения агента
        while (NomParametr < ParametricGraph.size()) {
            // Получение вершины из слоя
            int nextNodeIndex = GoAntNextNode(ParametricGraph[NomParametr].node);
            if (nextNodeIndex != -1) { // Проверяем, что индекс валиден
                way.push_back(nextNodeIndex);
            }
            // Выбор следующего слоя
            NomParametr++;
        }

        return way;
    }

    void NormPheromon() {
        for (size_t NomPar = 0; NomPar < ParametricGraph.size(); ++NomPar) {
            double MaxP = 0;
            int MaxK = 0;

            // Проходим по узлам для нахождения максимальных значений
            for (size_t NomEl = 0; NomEl < ParametricGraph[NomPar].node.size(); ++NomEl) {
                Node& currentNode = ParametricGraph[NomPar].node[NomEl];

                if (currentNode.pheromon == 0) {
                    currentNode.pheromon = 1e-8; // Устанавливаем минимальное значение
                }
                MaxP = std::max(MaxP, currentNode.pheromon);
                MaxK = std::max(MaxK, currentNode.KolSolution);
            }

            // Нормализация значений
            for (size_t NomEl = 0; NomEl < ParametricGraph[NomPar].node.size(); ++NomEl) {
                Node& currentNode = ParametricGraph[NomPar].node[NomEl];
                if (MaxP != 0) {
                    currentNode.pheromonNorm = currentNode.pheromon / MaxP;
                }
                if (MaxK != 0) {
                    currentNode.KolSolutionNorm = static_cast<int>(static_cast<double>(currentNode.KolSolution) / MaxK);
                }
            }
        }
    }

    void ClearPheromon(int allClear) {
        for (auto& param : ParametricGraph) {
            param.ClearAllNode(allClear);
        }
    }

    void DecreasePheromon(double par) {
        for (auto& param : ParametricGraph) {
            param.DecreasePheromon(par);
        }
    }

    void updatePheromone(const std::vector<int>& path, double objectiveValue) {
        for (size_t i = 0; i < path.size(); ++i) {

            ParametricGraph[i].node[path[i]].KolSolution += 1;
            ParametricGraph[i].node[path[i]].pheromon += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - objectiveValue); // Пример обновления феромонов

        }
    }
};

class Ant {
public:
    Ant() : pg(nullptr), objectiveValue(0.0) {} // Конструктор по умолчанию
    Ant(PG& g) : pg(&g), objectiveValue(0.0) {}

    // Метод для поиска пути
    std::vector<int> findPath() {
        path = pg->next();
        return path;
    }

    // Метод для оценки целевой функции на основе path
    double evaluateObjectiveFunction() {
        // Преобразуем path в параметры
        double parameters[PARAMETR_SIZE];
        convertPathToParameters(path, parameters);
        objectiveValue = BenchShafferaFunction_non_cuda(parameters);
        return objectiveValue;
    }

    void set_objectiveValue(double set_objective_Value) {
        objectiveValue = set_objective_Value;
    }

    // Метод для обновления феромонов
    void updatePheromone() {
        pg->updatePheromone(path, objectiveValue);
    }

    // Метод для вывода информации о муравье
    void printInfo() const {
        /* std::cout << "Path: ";
         for (int node : path) {
             std::cout << node << " ";
         }
         std::cout << " -> " << objectiveValue << "\n";
         */
         // Вывод параметров
        double parameters[PARAMETR_SIZE];
        convertPathToParameters(path, parameters);
        std::cout << "Parameters: ";
        for (int i = 0; i < PARAMETR_SIZE; ++i) { // Предполагается, что размер параметров соответствует размеру пути
            std::cout << parameters[i] << " ";
        }
        std::cout << " -> " << objectiveValue << "\n";
    }

private:
    PG* pg; // Указатель на граф
    std::vector<int> path; // Хранение пути
    double objectiveValue; // Хранение значения целевой функции

    // Метод для преобразования пути в параметры
    void convertPathToParameters(const std::vector<int>& path, double* parameters) const {
        // Здесь вы должны реализовать логику преобразования пути в параметры
        for (size_t i = 0; i < path.size(); ++i) {
            parameters[i] = static_cast<double>(pg->ParametricGraph[i].node[path[i]].val); // Пример преобразования
        }
        // Дополнительная логика может быть добавлена здесь в зависимости от вашей задачи
    }
};

struct HashEntry_classic {
    std::vector<int> key; // Ключ (вектор пути)
    double value; // Значение
};

// Функция инициализации хэш-таблицы
void initializeHashTable_classic(HashEntry_classic* hashTable, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        hashTable[i] = HashEntry_classic(); // Инициализируем каждый элемент
    }
}

// ----------------- Key Generation Function -----------------
unsigned long long generateKey_classic(const std::vector<int>& path) {
    unsigned long long key = 0;
    unsigned long long factor = 1;
    for (int val : path) {
        key += val * factor;
        factor *= MAX_VALUE_SIZE; // MAX_VALUE_SIZE - максимальное значение, которое может принимать элемент пути
    }
    return key;
}

// ----------------- Hash Function for Path -----------------
unsigned long long hashFunction_classic(const std::vector<int>& path) {
    unsigned long long key = generateKey_classic(path);
    return betterHashFunction_non_cuda(key);
}

double getCachedResultOptimized_classic_ant(HashEntry_classic* hashTable, const std::vector<int>& path) {
    unsigned long long key = hashFunction_classic(path);
    unsigned long long idx = key;
    int i = 1;

    while (i <= MAX_PROBES) {
        if (hashTable[idx].key == path) {
            return hashTable[idx].value; // Найдено
        }
        if (hashTable[idx].key.empty()) {
            return -1.0; // Не найдено и слот пуст
        }
        unsigned long long new_idx = idx + static_cast<unsigned long long>(i * i); if (new_idx >= HASH_TABLE_SIZE) { new_idx %= HASH_TABLE_SIZE; }idx = new_idx;
        i++;
    }
    return -1.0; // Не найдено после максимального количества проб
}

void saveToCacheOptimized_classic_ant(HashEntry_classic* hashTable, const std::vector<int>& path, double value) {
    unsigned long long key = hashFunction_classic(path);
    unsigned long long idx = key;
    int i = 1;

    while (i <= MAX_PROBES) {
        if (hashTable[idx].key.empty()) {
            // Успешно вставлено
            hashTable[idx].key = path;
            hashTable[idx].value = value;
            return;
        }
        else if (hashTable[idx].key == path) {
            // Ключ уже существует
            hashTable[idx].value = value; // Обновление значения
            return;
        }
        unsigned long long new_idx = idx + static_cast<unsigned long long>(i * i); if (new_idx >= HASH_TABLE_SIZE) { new_idx %= HASH_TABLE_SIZE; }idx = new_idx;
        i++;
    }
    // Если таблица полна, обработайте ошибку или проигнорируйте
}

// Определение статических членов класса PG
double PG::alf1 = 1;
double PG::alf2 = 1;
double PG::alf3 = 1;
double PG::koef1 = 1;
double PG::koef2 = 1;
double PG::koef3 = 0;
int PG::AllSolution = 0;
/*
void processAnt(int nom_ant, std::vector<Ant>& ants, PG& pg, double& maxOf, double& minOf, int& kol_hash_fail) {
    double objValue = 0;
    ants[nom_ant] = Ant(pg); // Инициализация каждого муравья с графом
    std::vector<int> path = ants[nom_ant].findPath(); // Поиск пути

    auto start4 = std::chrono::high_resolution_clock::now();
    double cachedResult = getCachedResultOptimized_classic_ant();

    int nom_iteration = 0;

    if (cachedResult == -1.0) {
        auto start_OF = std::chrono::high_resolution_clock::now();
        objValue = ants[nom_ant].evaluateObjectiveFunction(); // Оценка целевой функции
        auto end_OF = std::chrono::high_resolution_clock::now();

        saveToCacheOptimized_classic_ant();
    }
    else {
        kol_hash_fail++;
        switch (TYPE_ACO) {
        case 0: // ACOCN
            objValue = cachedResult;
            break;
        case 1: // ACOCNI
            objValue = ZERO_HASH_RESULT;
            break;
        case 2: // ACOCCyN
            while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION)) {
                nom_iteration++;
                path = ants[nom_ant].findPath(); // Поиск пути
                cachedResult = getCachedResultOptimized_classic_ant();
            }
            objValue = ants[nom_ant].evaluateObjectiveFunction(); // Оценка целевой функции
            saveToCacheOptimized_classic_ant();
            break;
        default:
            objValue = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
            break;
        }
    }

    auto end4 = std::chrono::high_resolution_clock::now();

    ants[nom_ant].set_objectiveValue(objValue);

    std::lock_guard<std::mutex> lock(mtx); // Защита доступа к общим переменным
    if (objValue != ZERO_HASH_RESULT) {
        if (objValue > maxOf) {
            maxOf = objValue;
        }
        if (objValue < minOf) {
            minOf = objValue;
        }
    }

    if () {
        ants[nom_ant].printInfo(); // Вывод информации о муравье
    }
}
*/
void start_ant_classic() {
    auto start = std::chrono::high_resolution_clock::now();
    double global_maxOf = -std::numeric_limits<double>::max();
    double global_minOf = std::numeric_limits<double>::max();
    float SumTime1 = 0.0f;
    float SumTime2 = 0.0f;
    float SumTime3 = 0.0f;
    float SumTime4 = 0.0f;
    double SumTime5 = 0.0f;
    float SumTime6 = 0.0f;
    float SumTime7 = 0.0f;
    float duration_iteration = 0.0f;
    double duration = 0.0f;
    int kol_hash_fail = 0;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    //std::cout << " Go: ";
    // Выделение памяти для хэш-таблицы на CPU
    HashEntry_classic* hashTable = new HashEntry_classic[HASH_TABLE_SIZE];
    // Вызов функции инициализации
    initializeHashTable_classic(hashTable, HASH_TABLE_SIZE);


    PG pg;

    if (LoadGraph(NAME_FILE_GRAPH, pg.ParametricGraph)) {

        auto start_iteration = std::chrono::high_resolution_clock::now();
        for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
            auto start1 = std::chrono::high_resolution_clock::now();

            double maxOf = -std::numeric_limits<double>::max();
            double minOf = std::numeric_limits<double>::max();
            pg.NormPheromon();

            if (PRINT_INFORMATION) {
                // Вывод информации о загруженных параметрах и узлах
                for (const auto& param : pg.ParametricGraph) {
                    std::cout << "Name Parametr " << param.name << " " << param.node.size();
                    for (const auto& node : param.node) {
                        std::cout << ", Node: " << node.val << "(" << node.pheromon << " -> " << node.pheromonNorm << " ," << node.KolSolution << ")";
                    }
                    std::cout << std::endl;
                }
            }

            auto start2 = std::chrono::high_resolution_clock::now();
            Ant ants[ANT_SIZE]; // Массив муравьев
            for (int nom_ant = 0; nom_ant < ANT_SIZE; nom_ant++) { // Проходим по всем агентам
                //std::cout << " nom_ant:" << nom_ant;
                double objValue = 0;
                ants[nom_ant] = Ant(pg); // Инициализация каждого муравья с графом
                std::vector<int> path = ants[nom_ant].findPath(); // Поиск пути
                //Проверка Хэш функции
                // Проверка наличия решения в Хэш-таблице
                auto start4 = std::chrono::high_resolution_clock::now();
                double cachedResult = getCachedResultOptimized_classic_ant(hashTable, path);
                //std::cout << " cachedResult:" << cachedResult;
                int nom_iteration = 0;

                if (cachedResult == -1.0) {
                    // Если значение не найденов ХЭШ, то заносим новое значение
                    auto start_OF = std::chrono::high_resolution_clock::now();
                    objValue = ants[nom_ant].evaluateObjectiveFunction(); // Оценка целевой функции
                    auto end_OF = std::chrono::high_resolution_clock::now();
                    SumTime5 += std::chrono::duration<double, std::milli>(end_OF - start_OF).count();
                    saveToCacheOptimized_classic_ant(hashTable, path, objValue);
                }
                else {
                    //Если значение в Хэш-найдено, то агент "нулевой"
                    kol_hash_fail = kol_hash_fail + 1;
                    //Поиск алгоритма для нулевого агента
                    switch (TYPE_ACO) {
                    case 0: // ACOCN
                        objValue = cachedResult;
                        break;
                    case 1: // ACOCNI
                        objValue = ZERO_HASH_RESULT;
                        break;
                    case 2: // ACOCCyN
                        while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                        {
                            nom_iteration = nom_iteration + 1;
                            std::vector<int> path = ants[nom_ant].findPath(); // Поиск пути
                            cachedResult = getCachedResultOptimized_classic_ant(hashTable, path);
                        }
                        objValue = ants[nom_ant].evaluateObjectiveFunction(); // Оценка целевой функции
                        saveToCacheOptimized_classic_ant(hashTable, path, objValue);
                        break;
                    default:
                        objValue = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                        break;
                    }
                }
                auto end4 = std::chrono::high_resolution_clock::now();
                SumTime4 += std::chrono::duration<float, std::milli>(end4 - start4).count();
                ants[nom_ant].set_objectiveValue(objValue);
                if (objValue != ZERO_HASH_RESULT) {
                    if (objValue > maxOf) {
                        maxOf = objValue;
                    }
                    if (objValue < minOf) {
                        minOf = objValue;
                    }
                }
                if (PRINT_INFORMATION) {
                    ants[nom_ant].printInfo(); // Вывод информации о муравье
                }
            }
            //std::cout << std::endl;
            auto start3 = std::chrono::high_resolution_clock::now();
            //Обновление феромона
            pg.DecreasePheromon(PARAMETR_RO);
            for (int nom_ant = 0; nom_ant < ANT_SIZE; nom_ant++) {
                ants[nom_ant].updatePheromone(); // Обновляем феромоны
            }
            if (minOf < global_minOf) {
                global_minOf = minOf;
            }
            if (maxOf > global_maxOf) {
                global_maxOf = maxOf;
            }
            auto end1 = std::chrono::high_resolution_clock::now();
            SumTime1 += std::chrono::duration<float, std::milli>(end1 - start1).count();
            SumTime2 += std::chrono::duration<float, std::milli>(end1 - start2).count();
            SumTime3 += std::chrono::duration<float, std::milli>(end1 - start3).count();
            if ((nom_iter + 1) % kol_shag_stat == 0) {
                int NomStatistics = nom_iter / kol_shag_stat;
                if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
                update_all_Stat(NomStatistics, 0, 0, SumTime1, SumTime2, SumTime3, SumTime4, SumTime5, SumTime6, SumTime7, 0, global_minOf, global_maxOf, kol_hash_fail);
            }
        }
        auto end_iteration = std::chrono::high_resolution_clock::now();
        duration_iteration += std::chrono::duration<float, std::milli>(end_iteration - start_iteration).count();

    }
    else {
        std::cerr << "Graph Load Error" << std::endl;
    }
    delete[] hashTable; // Освобождаем память
    auto end_all = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double, std::milli>(end_all - start).count();
    std::cout << "Time classical ACO;" << duration << "; " << duration_iteration << "; " << SumTime1 << "; " << SumTime2 << "; " << SumTime3 << "; " << SumTime4 << "; " << SumTime5 << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time classical ACO;" << duration << "; " << duration_iteration << "; " << SumTime1 << "; " << SumTime2 << "; " << SumTime3 << "; " << SumTime4 << "; " << SumTime5 << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

}

void start_ant_classic_non_hash() {
    auto start = std::chrono::high_resolution_clock::now();
    double global_maxOf = -std::numeric_limits<double>::max();
    double global_minOf = std::numeric_limits<double>::max();
    float SumTime1 = 0.0f;
    float SumTime2 = 0.0f;
    float SumTime3 = 0.0f;
    float SumTime4 = 0.0f;
    double SumTime5 = 0.0f;
    float SumTime6 = 0.0f;
    float SumTime7 = 0.0f;
    float duration_iteration = 0.0f;
    double duration = 0.0f;
    int kol_hash_fail = 0;
    int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    //std::cout << " Go: ";

    PG pg;

    if (LoadGraph(NAME_FILE_GRAPH, pg.ParametricGraph)) {

        auto start_iteration = std::chrono::high_resolution_clock::now();
        for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
            auto start1 = std::chrono::high_resolution_clock::now();

            double maxOf = -std::numeric_limits<double>::max();
            double minOf = std::numeric_limits<double>::max();
            pg.NormPheromon();

            if (PRINT_INFORMATION) {
                // Вывод информации о загруженных параметрах и узлах
                for (const auto& param : pg.ParametricGraph) {
                    std::cout << "Name Parametr " << param.name << " " << param.node.size();
                    for (const auto& node : param.node) {
                        std::cout << ", Node: " << node.val << "(" << node.pheromon << " -> " << node.pheromonNorm << " ," << node.KolSolution << ")";
                    }
                    std::cout << std::endl;
                }
            }
            auto start2 = std::chrono::high_resolution_clock::now();
            Ant ants[ANT_SIZE]; // Массив муравьев
            for (int nom_ant = 0; nom_ant < ANT_SIZE; nom_ant++) { // Проходим по всем агентам
                //std::cout << " nom_ant:" << nom_ant;
                double objValue = 0;
                ants[nom_ant] = Ant(pg); // Инициализация каждого муравья с графом
                std::vector<int> path = ants[nom_ant].findPath(); // Поиск пути
                //Проверка Хэш функции
                auto start_OF = std::chrono::high_resolution_clock::now();
                objValue = ants[nom_ant].evaluateObjectiveFunction(); // Оценка целевой функции
                auto end_OF = std::chrono::high_resolution_clock::now();
                SumTime5 += std::chrono::duration<double, std::milli>(end_OF - start_OF).count();
                ants[nom_ant].set_objectiveValue(objValue);
                if (objValue > maxOf) {
                    maxOf = objValue;
                }
                if (objValue < minOf) {
                    minOf = objValue;
                }
                if (PRINT_INFORMATION) {
                    ants[nom_ant].printInfo(); // Вывод информации о муравье
                }
            }
            auto start3 = std::chrono::high_resolution_clock::now();
            //std::cout << std::endl;
            //Обновление феромона
            pg.DecreasePheromon(PARAMETR_RO);
            for (int nom_ant = 0; nom_ant < ANT_SIZE; nom_ant++) {
                ants[nom_ant].updatePheromone(); // Обновляем феромоны
            }
            if (minOf < global_minOf) {
                global_minOf = minOf;
            }
            if (maxOf > global_maxOf) {
                global_maxOf = maxOf;
            }
            auto end1 = std::chrono::high_resolution_clock::now();
            SumTime1 += std::chrono::duration<float, std::milli>(end1 - start1).count();
            SumTime2 += std::chrono::duration<float, std::milli>(end1 - start2).count();
            SumTime3 += std::chrono::duration<float, std::milli>(end1 - start3).count();
            if ((nom_iter + 1) % kol_shag_stat == 0) {
                int NomStatistics = nom_iter / kol_shag_stat;
                if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
                update_all_Stat(NomStatistics, 0, 0, SumTime1, SumTime2, SumTime3, 0, SumTime5, SumTime6, SumTime7, 0, global_minOf, global_maxOf, kol_hash_fail);
            }
        }
        auto end_iteration = std::chrono::high_resolution_clock::now();
        duration_iteration += std::chrono::duration<float, std::milli>(end_iteration - start_iteration).count();

    }
    else {
        std::cerr << "Graph Load Error" << std::endl;
    }
    auto end_all = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double, std::milli>(end_all - start).count();
    std::cout << "Time classical ACO non hash;" << duration << "; " << duration_iteration << "; " << SumTime1 << "; " << SumTime2 << "; " << SumTime3 << "; " << SumTime4 << "; " << SumTime5 << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time classical ACO non hash;" << duration << "; " << duration_iteration << "; " << SumTime1 << "; " << SumTime2 << "; " << SumTime3 << "; " << SumTime4 << "; " << SumTime5 << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

}


int main(int argc, char* argv[]) {
    // Открытие лог-файла
    //_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    std::cout << __cplusplus << std::endl;
    logFile.open("log.txt");
    if (!logFile.is_open()) {
        std::cerr << "Ошибка открытия лог-файла!" << std::endl;
        return 1; // Возврат с ошибкой
    }
    //Создание векторов для статистики 
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
        std::cout << "Max thread in blocks: " << prop.maxThreadsPerBlock << std::endl;
        logFile << "Max thread in blocks: " << prop.maxThreadsPerBlock << " ";
        std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;
        logFile << "Total Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB ";
        std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
        logFile << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes ";
        std::cout << "Constant Memory: " << prop.totalConstMem << " bytes" << std::endl;
        logFile << "Constant Memory: " << prop.totalConstMem << " bytes ";
        std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
        logFile << "Registers per Block: " << prop.regsPerBlock << " ";
        std::cout << "Warp Size: " << prop.warpSize << std::endl;
        logFile << "Warp Size: " << prop.warpSize << " ";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        logFile << "Compute Capability: " << prop.major << "." << prop.minor << " ";
        std::cout << "Can Map Host Memory: " << (prop.canMapHostMemory ? "Yes" : "No") << std::endl;
        logFile << "Can Map Host Memory: " << (prop.canMapHostMemory ? "Yes" : "No") << " ";
        std::cout << "Clock Rate: " << prop.clockRate / 1000.0f << " MHz" << std::endl;
        logFile << "Clock Rate: " << prop.clockRate / 1000.0f << " MHz ";
        std::cout << "L2 Cache Size: " << (prop.l2CacheSize == 0 ? 0 : prop.l2CacheSize) << " bytes" << std::endl;
        logFile << "L2 Cache Size: " << (prop.l2CacheSize == 0 ? 0 : prop.l2CacheSize) << " bytes ";
        std::cout << "Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
        logFile << "Multiprocessor Count: " << prop.multiProcessorCount << "; ";
        std::cout << "Max thread in blocks by axis: ("
            << prop.maxThreadsDim[0] << ", "
            << prop.maxThreadsDim[1] << ", "
            << prop.maxThreadsDim[2] << ")" << std::endl;
        logFile << "Max thread in blocks by axis: ("
            << prop.maxThreadsDim[0] << ", "
            << prop.maxThreadsDim[1] << ", "
            << prop.maxThreadsDim[2] << "); ";
        std::cout << "Max blocks by axis: ("
            << prop.maxGridSize[0] << ", "
            << prop.maxGridSize[1] << ", "
            << prop.maxGridSize[2] << ")" << std::endl;
        logFile << "Max blocks by axis: ("
            << prop.maxGridSize[0] << ", "
            << prop.maxGridSize[1] << ", "
            << prop.maxGridSize[2] << ");";
    }

    std::cout << "PARAMETR_SIZE: " << PARAMETR_SIZE << "; "
        << "PARAMETR_SIZE_ONE_X: " << PARAMETR_SIZE_ONE_X << "; "
        << "MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << "; "
        << "ANT_SIZE: " << ANT_SIZE << "; "
        << "NAME_FILE_GRAPH: " << NAME_FILE_GRAPH << "; "
        << "KOL_ITERATION: " << KOL_ITERATION << "; "
        << "KOL_PROGON_STATISTICS: " << KOL_PROGON_STATISTICS << "; "
        << "PARAMETR_Q: " << PARAMETR_Q << "; "
        << "PARAMETR_RO: " << PARAMETR_RO << "; "
        << "TYPE_ACO: " << TYPE_ACO << "; "
        << "ACOCCyN_KOL_ITERATION: " << ACOCCyN_KOL_ITERATION << ", " << "MAX_PARAMETR_VALUE_TO_MIN_OPT: " << MAX_PARAMETR_VALUE_TO_MIN_OPT << ", " << "HASH_TABLE_SIZE: " << HASH_TABLE_SIZE << ", " << "MAX_PROBES: " << MAX_PROBES << ", " << "MAX_THREAD_CUDA: " << MAX_THREAD_CUDA << ", " << "CPU_RANDOM: " << CPU_RANDOM << ", " << "KOL_THREAD_CPU_ANT: " << KOL_THREAD_CPU_ANT << ", " << "CONST_RANDOM: " << CONST_RANDOM << ", " << "MAX_CONST: " << MAX_CONST << ", "
        << "OPTIMIZE: " << (OPTIMIZE_MIN_1 ? "OPTIMIZE_MIN_1 " : "") << (OPTIMIZE_MIN_2 ? "OPTIMIZE_MIN_2 " : "") << (OPTIMIZE_MAX ? "OPTIMIZE_MAX " : "") << "; "
        << "FUNCTION: " << (SHAFFERA ? "SHAFFERA " : "") << (CARROM_TABLE ? "CARROM_TABLE " : "") << (RASTRIGIN ? "RASTRIGIN " : "") << (ACKLEY ? "ACKLEY " : "") << (SPHERE ? "SPHERE " : "") << (GRIEWANK ? "GRIEWANK " : "") << (ZAKHAROV ? "ZAKHAROV " : "") << (SCHWEFEL ? "SCHWEFEL " : "") << (LEVY ? "LEVY " : "") << (MICHAELWICZYNSKI ? "MICHAELWICZYNSKI " : "")
        << "GO_ALG_MINMAX: " << GO_ALG_MINMAX << "(" << PAR_MIN_ALG_MINMAX << ", " << PAR_MAX_ALG_MINMAX << ")" << "; "
        << std::endl;
    logFile << "PARAMETR_SIZE: " << PARAMETR_SIZE << "; "
        << "PARAMETR_SIZE_ONE_X: " << PARAMETR_SIZE_ONE_X << "; "
        << "MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << "; "
        << "NAME_FILE_GRAPH: " << NAME_FILE_GRAPH << "; "
        << "ANT_SIZE: " << ANT_SIZE << "; "
        << "KOL_ITERATION: " << KOL_ITERATION << "; "
        << "KOL_PROGON_STATISTICS: " << KOL_PROGON_STATISTICS << "; "
        << "PARAMETR_Q: " << PARAMETR_Q << "; "
        << "PARAMETR_RO: " << PARAMETR_RO << "; "
        << "TYPE_ACO: " << TYPE_ACO << "; "
        << "ACOCCyN_KOL_ITERATION: " << ACOCCyN_KOL_ITERATION << ", " << "MAX_PARAMETR_VALUE_TO_MIN_OPT: " << MAX_PARAMETR_VALUE_TO_MIN_OPT << ", " << "HASH_TABLE_SIZE: " << HASH_TABLE_SIZE << ", " << "MAX_PROBES: " << MAX_PROBES << ", " << "MAX_THREAD_CUDA: " << MAX_THREAD_CUDA << ", " << "CPU_RANDOM: " << CPU_RANDOM << ", " << "KOL_THREAD_CPU_ANT: " << KOL_THREAD_CPU_ANT << ", " << "CONST_RANDOM: " << CONST_RANDOM << ", " << "MAX_CONST: " << MAX_CONST << ", "
        << "OPTIMIZE: " << (OPTIMIZE_MIN_1 ? "OPTIMIZE_MIN_1 " : "") << (OPTIMIZE_MIN_2 ? "OPTIMIZE_MIN_2 " : "") << (OPTIMIZE_MAX ? "OPTIMIZE_MAX " : "") << "; "
        << "FUNCTION: " << (SHAFFERA ? "SHAFFERA " : "") << (CARROM_TABLE ? "CARROM_TABLE " : "") << (RASTRIGIN ? "RASTRIGIN " : "") << (ACKLEY ? "ACKLEY " : "") << (SPHERE ? "SPHERE " : "") << (GRIEWANK ? "GRIEWANK " : "") << (ZAKHAROV ? "ZAKHAROV " : "") << (SCHWEFEL ? "SCHWEFEL " : "") << (LEVY ? "LEVY " : "") << (MICHAELWICZYNSKI ? "MICHAELWICZYNSKI " : "")
        << "GO_ALG_MINMAX: " << GO_ALG_MINMAX << "(" << PAR_MIN_ALG_MINMAX << ", " << PAR_MAX_ALG_MINMAX << ")" << "; "
        << std::endl;
    //matrix_ACO2_non_hash();
    if (GO_CUDA_TIME) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_Time();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_Time();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        // Вывод информации на экран и в лог-файл
        std::string message = "Time CUDA time:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA Time");
    }
    if (GO_CUDA) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        // Вывод информации на экран и в лог-файл
        std::string message = "Time CUDA:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA");
    }
    if (GO_CUDA_NON_HASH) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_non_hash();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_non_hash();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        // Вывод информации на экран и в лог-файл
        std::string message = "Time CUDA non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA non hash");
    }
    if (GO_CUDA_BLOCK_TIME) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_only_block_Time();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_only_block_Time();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        // Вывод информации на экран и в лог-файл
        std::string message = "Time CUDA only block:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA only block");
    }
    if (GO_CUDA_ANT) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_ant();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_ant();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        std::string message = "Time CUDA ant:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA ant");
    }
    if (GO_CUDA_ANT_PAR) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_ant_par();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_ant_par();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        std::string message = "Time CUDA ant par:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA ant par");
    }
    if (GO_CUDA_ANT_PAR_GLOBAL) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_ant_par_global();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_ant_par_global();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        std::string message = "Time CUDA ant par global:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA ant par global");
    }
    if (GO_CUDA_ANT_NON_HASH) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_ant_non_hash();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_ant_non_hash();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        std::string message = "Time CUDA ant non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA ant non hash");
    }
    if (GO_CUDA_ANT_ADD_CPU_TIME) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_ant_add_CPU_Time();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_ant_add_CPU_Time();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        std::string message = "Time CUDA ant add CPU Time:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA ant add CPU Time");
    }
    if (GO_CUDA_ANT_ADD_CPU_TIME_GLOBAL) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_ant_add_CPU_Time_global();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_ant_add_CPU_Time_global();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        std::string message = "Time CUDA ant add CPU Time global:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA ant add CPU Time global");
    }
    if (GO_CUDA_ANT_ADD_CPU) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_ant_add_CPU();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_ant_add_CPU();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        std::string message = "Time CUDA ant add CPU:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA ant add CPU");
    }
    if (GO_CUDA_ANT_ADD_CPU_NON_HASH) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_ant_add_CPU_non_hash();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_ant_add_CPU_non_hash();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        std::string message = "Time CUDA ant add CPU non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA ant add CPU non hash");
    }
    /*
    if (GO_CUDA_ANT_ADD_CPU_OPTMEM) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_ant_add_CPU_optMem();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_ant_add_CPU_optMem();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        std::string message = "Time CUDA ant add CPU OptMem:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA ant add CPU OptMem");
    }
    */
    if (GO_CUDA_ANT_ADD_CPU2_TIME) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_ant_add_CPU2_Time();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_ant_add_CPU2_Time();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        std::string message = "Time CUDA ant add CPU12:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA ant add CPU12");
    }
    if (GO_CUDA_OPT) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_opt();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_opt();
            i = i + 1;
        }
        // Остановка таймера
        auto end = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end - start;
        std::string message = "Time CUDA opt:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA opt");
    }
    if (GO_CUDA_OPT_TIME) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_opt_Time();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_opt_Time();
            i = i + 1;
        }
        // Остановка таймера
        auto end = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end - start;
        std::string message = "Time CUDA opt:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA opt Time");
    }
    if (GO_CUDA_OPT_NON_HASH) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_opt_non_hash();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_opt_non_hash();
            i = i + 1;
        }
        // Остановка таймера
        auto end = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end - start;
        std::string message = "Time CUDA opt non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA opt non hash");
    }
    if (GO_CUDA_OPT_ANT) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_opt_ant();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_opt_ant();
            i = i + 1;
        }
        // Остановка таймера
        auto end = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end - start;
        std::string message = "Time CUDA opt ant:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA opt ant");
    }
    if (GO_CUDA_OPT_ANT_NON_HASH) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_opt_ant_non_hash();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_opt_ant_non_hash();
            i = i + 1;
        }
        // Остановка таймера
        auto end = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end - start;
        std::string message = "Time CUDA opt ant non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA opt ant non hash");
    }
    if (GO_CUDA_OPT_ANT_PAR) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_opt_ant_par();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_opt_ant_par();
            i = i + 1;
        }
        // Остановка таймера
        auto end = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end - start;
        std::string message = "Time CUDA opt ant par:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA opt ant par");
    }
    if (GO_CUDA_OPT_ANT_PAR_GLOBAL) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_opt_ant_par_global();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_opt_ant_par_global();
            i = i + 1;
        }
        // Остановка таймера
        auto end = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end - start;
        std::string message = "Time CUDA opt ant par global:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA opt ant par global");
    }
    if (GO_CUDA_ONE_OPT) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_opt_one_GPU();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start0 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_opt_one_GPU();
            i = i + 1;
        }
        // Остановка таймера
        auto end0 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end0 - start0;
        std::string message = "Time CUDA opt_one_GPU:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA opt_one_GPU");
    }
    if (GO_CUDA_ONE_OPT_LOCAL) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_opt_one_GPU_local();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start0 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_opt_one_GPU_local();
            i = i + 1;
        }
        // Остановка таймера
        auto end0 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end0 - start0;
        std::string message = "Time CUDA opt_one_GPU Local:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA opt_one_GPU Local");
    }
    if (GO_CUDA_ONE_OPT_NON_HASH) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_opt_one_GPU_non_hash();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_opt_one_GPU_non_hash();
            i = i + 1;
        }
        // Остановка таймера
        auto end = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end - start;
        std::string message = "Time CUDA opt_one_GPU non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA opt_one_GPU non hash");
    }
    if (GO_CUDA_ONE_OPT_ANT) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_opt_one_GPU_ant();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start0 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_opt_one_GPU_ant();
            i = i + 1;
        }
        // Остановка таймера
        auto end0 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end0 - start0;
        std::string message = "Time CUDA opt_one_GPU ant:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA opt_one_GPU ant");
    }
    if (GO_CUDA_ONE_OPT_ANT_LOCAL) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_opt_one_GPU_ant_local();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start0 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_opt_one_GPU_ant_local();
            i = i + 1;
        }
        // Остановка таймера
        auto end0 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end0 - start0;
        std::string message = "Time CUDA opt_one_GPU ant local:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA opt_one_GPU ant local");
    }
    if (GO_CUDA_ONE_OPT_ANT_NON_HASH) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_opt_one_GPU_ant_non_hash();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start0 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_opt_one_GPU_ant_non_hash();
            i = i + 1;
        }
        // Остановка таймера
        auto end0 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end0 - start0;
        std::string message = "Time CUDA opt_one_GPU ant non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA opt_one_GPU ant non hash");
    }
    if (GO_NON_CUDA_TIME) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_NON_CUDA_time();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_NON_CUDA_time();
            i = i + 1;
        }
        // Остановка таймера
        auto end2 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time non CUDA_time:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("non CUDA Time");
    }
    if (GO_NON_CUDA) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_NON_CUDA();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_NON_CUDA();
            i = i + 1;
        }
        // Остановка таймера
        auto end2 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time non CUDA:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("non CUDA");
    }
    if (GO_NON_CUDA_THREAD) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_NON_CUDA_thread();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_NON_CUDA_thread();
            i = i + 1;
        }
        // Остановка таймера
        auto end2 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time non CUDA thread:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("non CUDA Thread");
    }
    if (GO_NON_CUDA_NON_HASH) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_NON_CUDA_non_hash();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_NON_CUDA_non_hash();
            i = i + 1;
        }
        // Остановка таймера
        auto end2 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time non CUDA non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("non CUDA non hash");
    }
    if (GO_CLASSIC_ACO) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_ant_classic();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start3 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_ant_classic();
            i = i + 1;

        }
        // Остановка таймера
        auto end3 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end3 - start3;
        std::string message = "Time Classic ACO:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("Classic ACO");
    }
    if (GO_CLASSIC_ACO_NON_HASH) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_ant_classic_non_hash();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start3 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS) {
            std::cout << i << " ";
            start_ant_classic_non_hash();
            i = i + 1;
        }
        // Остановка таймера
        auto end3 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end3 - start3;
        std::string message = "Time Classic ACO non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("Classic ACO non hash");
    }
    if (GO_OMP) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_omp();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_omp();
            i = i + 1;
        }
        // Остановка таймера
        auto end2 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time omp:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("omp");
    }
    if (GO_OMP_NON_HASH) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_omp_non_hash();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_omp_non_hash();
            i = i + 1;
        }
        // Остановка таймера
        auto end2 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time omp non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("omp non hash");
    }
    if (GO_NON_CUDA_TRANSP_TIME) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_NON_CUDA_transp_time();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_NON_CUDA_transp_time();
            i = i + 1;
        }
        // Остановка таймера
        auto end2 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time non CUDA_transp_time:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("non CUDA transp Time");
    }
    if (GO_NON_CUDA_TRANSP) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_NON_CUDA_transp();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_NON_CUDA_transp();
            i = i + 1;
        }
        // Остановка таймера
        auto end2 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time non CUDA transp:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("non CUDA transp");
    }
    if (GO_NON_CUDA_TRANSP_NON_HASH) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_NON_CUDA_transp_non_hash();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_NON_CUDA_transp_non_hash();
            i = i + 1;
        }
        // Остановка таймера
        auto end2 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time non CUDA transp non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("non CUDA transp non hash");
    }
    if (MAX_VALUE_SIZE * PARAMETR_SIZE < MAX_CONST) {
        if (GO_CUDA_CONST) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_CUDA_Const();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start1 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_CUDA_Const();
                i = i + 1;
            }
            // Остановка таймера
            auto end1 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end1 - start1;
            // Вывод информации на экран и в лог-файл
            std::string message = "Time CUDA Const:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("CUDA Const");
        }
        if (GO_CUDA_ANT_CONST) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_CUDA_ant_Const();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start1 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_CUDA_ant_Const();
                i = i + 1;
            }
            // Остановка таймера
            auto end1 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end1 - start1;
            std::string message = "Time CUDA ant Const:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("CUDA ant Const");
        }
        if (GO_CUDA_ANT_DECREASE_PAR_CONST) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_CUDA_ant_decrease_par_Const();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start1 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_CUDA_ant_decrease_par_Const();
                i = i + 1;
            }
            // Остановка таймера
            auto end1 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end1 - start1;
            std::string message = "Time CUDA ant Const decrease par:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("CUDA ant Const decrease par");
        }
        if (GO_CUDA_ANT_DECREASE_PAR_BLOCK_CONST) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_CUDA_ant_decrease_par_block_Const();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start1 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_CUDA_ant_decrease_par_block_Const();
                i = i + 1;
            }
            // Остановка таймера
            auto end1 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end1 - start1;
            std::string message = "Time CUDA ant Const decrease block par:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("CUDA ant Const decrease block par");
        }
        if (GO_CUDA_ANT_DECREASE_ANT_CONST) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_CUDA_ant_decrease_ant_Const();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start1 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_CUDA_ant_decrease_ant_Const();
                i = i + 1;
            }
            // Остановка таймера
            auto end1 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end1 - start1;
            std::string message = "Time CUDA ant Const decrease ant:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("CUDA ant Const decrease ant");
        }
        if (GO_CUDA_ANT_DECREASE_ANT_BLOCK_CONST) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_CUDA_ant_decrease_ant_block_Const();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start1 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_CUDA_ant_decrease_ant_block_Const();
                i = i + 1;
            }
            // Остановка таймера
            auto end1 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end1 - start1;
            std::string message = "Time CUDA ant Const decrease block ant:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("CUDA ant Const decrease block ant");
        }
        if (GO_CUDA_ANT_DECREASE_ANT_PAR_CONST) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_CUDA_ant_decrease_ant_par_Const();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start1 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_CUDA_ant_decrease_ant_par_Const();
                i = i + 1;
            }
            // Остановка таймера
            auto end1 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end1 - start1;
            std::string message = "Time CUDA ant Const decrease ant par:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("Time CUDA ant Const decrease ant par");
        }
        if (GO_CUDA_ANT_DECREASE_ANT_PAR_TRANSP_CONST) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_CUDA_ant_decrease_ant_par_transp_Const();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start1 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_CUDA_ant_decrease_ant_par_transp_Const();
                i = i + 1;
            }
            // Остановка таймера
            auto end1 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end1 - start1;
            std::string message = "Time CUDA ant Const decrease ant par transp:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("Time CUDA ant Const decrease ant par transp");
        }
        if (GO_CUDA_ANT_PAR_CONST) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_CUDA_ant_par_Const();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start1 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_CUDA_ant_par_Const();
                i = i + 1;
            }
            // Остановка таймера
            auto end1 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end1 - start1;
            std::string message = "Time CUDA ant par Const:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("CUDA ant par Const");
        }
        if (GO_CUDA_ANT_ADD_CPU_CONST) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_CUDA_ant_add_CPU_Const();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start1 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_CUDA_ant_add_CPU_Const();
                i = i + 1;
            }
            // Остановка таймера
            auto end1 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end1 - start1;
            std::string message = "Time CUDA ant add CPU Const:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("CUDA ant add CPU Const");
        }
        if (GO_CUDA_OPT_CONST) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_CUDA_opt_Const();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_CUDA_opt_Const();
                i = i + 1;
            }
            // Остановка таймера
            auto end = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end - start;
            std::string message = "Time CUDA opt Const:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("CUDA opt Const");
        }
        if (GO_CUDA_OPT_ANT_CONST) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_CUDA_opt_ant_Const();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_CUDA_opt_ant_Const();
                i = i + 1;
            }
            // Остановка таймера
            auto end = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end - start;
            std::string message = "Time CUDA opt ant Const:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("CUDA opt ant Const");
        }
        if (GO_CUDA_OPT_ANT_PAR_CONST) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_CUDA_opt_ant_par_Const();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_CUDA_opt_ant_par_Const();
                i = i + 1;
            }
            // Остановка таймера
            auto end = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end - start;
            std::string message = "Time CUDA opt ant par Const:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("CUDA opt ant par Const");
        }
    }
    if (GO_CUDA_ANT_PAR_NOT_F) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_ant_par_not_f();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_ant_par_not_f();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        std::string message = "Time CUDA ant par not f:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA ant par not f");
    }
    if (GO_CUDA_ANT_PAR_SORT) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_CUDA_ant_par_sort();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_CUDA_ant_par_sort();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        std::string message = "Time CUDA ant par sort:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("CUDA ant par sort");
    }
    // Закрытие лог-файла
    logFile.close();
    outfile.close();
}
