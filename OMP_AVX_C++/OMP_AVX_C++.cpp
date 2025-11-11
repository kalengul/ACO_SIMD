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
#include <omp.h>
#include <mutex>
#include <thread>
#include "parametrs.h" 

std::ofstream logFile; // Глобальная переменная для лог-файла
std::ofstream outfile("statistics.txt"); // Глобальная переменная для файла статистики
std::mutex mtx; // Мьютекс для защиты доступа к общим данным

#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#include <cstdlib>
#define ALIGNED_ALLOC(alignment, size) aligned_alloc(alignment, size)
#define ALIGNED_FREE(ptr) free(ptr)
#endif


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

// ----------------- Инициализация хэш-таблицы -----------------
void initializeHashTable_non_cuda(HashEntry* hashTable, int size) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        hashTable[i].key = ZERO_HASH;
        hashTable[i].value = 0.0;
    }
}

// ----------------- Очистка хэш-таблицы -----------------
void clearHashTable(HashEntry* hashTable, int size) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        hashTable[i].key = ZERO_HASH;
        hashTable[i].value = 0.0;
    }
}

// ----------------- Быстрая хэш-функция -----------------
inline unsigned long long fastHashFunction(unsigned long long key) {
    // Оптимизированная хэш-функция для MSVC
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);

    // Быстрый модуль через битовую маску
    return key & (HASH_TABLE_SIZE - 1);
}

// ----------------- Генерация ключа из пути агента -----------------
inline unsigned long long generateKey(const int* agent_path) {
    // Полиномиальное хэширование для лучшего распределения
    const unsigned long long prime = 1099511628211ULL;
    unsigned long long key = 14695981039346656037ULL;

    for (int i = 0; i < PARAMETR_SIZE; i++) {
        key ^= static_cast<unsigned long long>(agent_path[i]);
        key *= prime;
    }

    return key;
}

// ----------------- Альтернативная генерация ключа -----------------
inline unsigned long long generateKeySimple(const int* agent_path) {
    unsigned long long key = 0;

    for (int i = 0; i < PARAMETR_SIZE; i++) {
        key = key * MAX_VALUE_SIZE + agent_path[i];
    }

    return key;
}

// ----------------- Поиск в хэш-таблице -----------------
double getCachedResultOptimized_non_cuda(HashEntry* __restrict hashTable, const int* __restrict agent_path, int bx) {
    unsigned long long key = generateKey(agent_path);
    unsigned long long idx = fastHashFunction(key);
    const unsigned long long mask = HASH_TABLE_SIZE - 1;

    // Поиск с квадратичным probing
    for (int i = 0; i < MAX_PROBES; i++) {
        unsigned long long new_idx = (idx + static_cast<unsigned long long>(i * i)) & mask;

        // Используем совместимый макрос для предсказания ветвлений
        if (hashTable[new_idx].key == key) {
            return hashTable[new_idx].value; // Найдено
        }
        if (hashTable[new_idx].key == ZERO_HASH) {
            return ZERO_HASH_RESULT; // Не найдено - пустой слот
        }
    }

    return ZERO_HASH_RESULT; // Не найдено после всех проб
}

// ----------------- Сохранение в хэш-таблицу -----------------
bool saveToCacheOptimized_non_cuda(HashEntry* __restrict hashTable, const int* __restrict agent_path, int bx, double value) {
    unsigned long long key = generateKey(agent_path);
    unsigned long long idx = fastHashFunction(key);
    const unsigned long long mask = HASH_TABLE_SIZE - 1;

    // Поиск пустого слота или обновление существующего
    for (int i = 0; i < MAX_PROBES; i++) {
        unsigned long long new_idx = (idx + static_cast<unsigned long long>(i * i)) & mask;

        if (hashTable[new_idx].key == ZERO_HASH || hashTable[new_idx].key == key) {
            // Найден пустой слот или существующий ключ
            hashTable[new_idx].key = key;
            hashTable[new_idx].value = value;
            return true;
        }
    }

    // Не удалось найти слот
    std::cerr << "Warning: Hash table full, could not insert key" << std::endl;
    return false;
}

// ----------------- Потокобезопасная версия поиска -----------------
double getCachedResultOptimized_OMP_non_cuda(HashEntry* __restrict hashTable, const int* __restrict agent_path, int bx) {
    unsigned long long key = generateKey(agent_path);
    unsigned long long idx = fastHashFunction(key);
    const unsigned long long mask = HASH_TABLE_SIZE - 1;

    double result = ZERO_HASH_RESULT;

    // Критическая секция для потокобезопасного доступа
#pragma omp critical(hash_lookup)
    {
        for (int i = 0; i < MAX_PROBES; i++) {
            unsigned long long new_idx = (idx + static_cast<unsigned long long>(i * i)) & mask;

            if (hashTable[new_idx].key == key) {
                result = hashTable[new_idx].value;
                break;
            }
            if (hashTable[new_idx].key == ZERO_HASH) {
                result = ZERO_HASH_RESULT;
                break;
            }
        }
    }

    return result;
}

// ----------------- Потокобезопасная версия сохранения -----------------
bool saveToCacheOptimized_OMP_non_cuda(HashEntry* __restrict hashTable, const int* __restrict agent_path, int bx, double value) {
    unsigned long long key = generateKey(agent_path);
    unsigned long long idx = fastHashFunction(key);
    const unsigned long long mask = HASH_TABLE_SIZE - 1;

    bool success = false;

#pragma omp critical(hash_save)
    {
        for (int i = 0; i < MAX_PROBES; i++) {
            unsigned long long new_idx = (idx + static_cast<unsigned long long>(i * i)) & mask;

            if (hashTable[new_idx].key == ZERO_HASH || hashTable[new_idx].key == key) {
                hashTable[new_idx].key = key;
                hashTable[new_idx].value = value;
                success = true;
                break;
            }
        }
    }

    return success;
}

// ----------------- Статистика хэш-таблицы -----------------
void printHashTableStats(const HashEntry* hashTable, int size) {
    int used_slots = 0;

#pragma omp parallel for reduction(+:used_slots) schedule(static)
    for (int i = 0; i < size; i++) {
        if (hashTable[i].key != ZERO_HASH) {
            used_slots++;
        }
    }

    double load_factor = static_cast<double>(used_slots) / size;

    std::cout << "=== Hash Table Statistics ===" << std::endl;
    std::cout << "Size: " << size << std::endl;
    std::cout << "Used slots: " << used_slots << std::endl;
    std::cout << "Load factor: " << (load_factor * 100.0) << "%" << std::endl;
    std::cout << "Max probes: " << MAX_PROBES << std::endl;
    std::cout << "=============================" << std::endl;
}

// ----------------- Коэффициент заполнения -----------------
double getHashTableLoadFactor(const HashEntry* hashTable, int size) {
    int used_slots = 0;

#pragma omp parallel for reduction(+:used_slots) schedule(static)
    for (int i = 0; i < size; i++) {
        if (hashTable[i].key != ZERO_HASH) {
            used_slots++;
        }
    }

    return static_cast<double>(used_slots) / size;
}

// Функция для вычисления вероятностной формулы
inline double probability_formula_non_cuda(double pheromon, double kol_enter) {
    return (kol_enter != 0.0 && pheromon != 0.0) ? (1.0 / kol_enter + pheromon) : 0.0;
}

// Подготовка массива для вероятностного поиска

// Базовая версия OpenMP 2.0/3.0/3.1 - только CPU параллелизм
void go_mass_probability_omp_2_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 2.0-3.1 version (CPU parallel for)\n");

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
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
        }
    }
}
#if _OPENMP >= 201307  // OpenMP 4.0+
void go_mass_probability_omp_4_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 4.0 version (SIMD vectorization)\n");

    // OpenMP 4.0: separate simd directive
#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Суммируем значения феромонов
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        // Вычисляем вероятностные значения
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }

        norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
        for (int i = 1; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
        }
    }
}
#endif
#if _OPENMP >= 201511  // OpenMP 4.5+
void go_mass_probability_omp_4_5(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 4.5 version (if clause and loop nesting)\n");

    // OpenMP 4.5: if clause для условного выполнения
#if defined(__clang__)
#pragma omp parallel for
#else
#pragma omp parallel for // if(PARAMETR_SIZE > 100)
#endif
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Автоматическая векторизация внутренних циклов
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        // Вычисляем вероятностные значения
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }

        norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
        for (int i = 1; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
        }
    }
}
#endif
#if _OPENMP >= 201811  // OpenMP 5.0+
void go_mass_probability_omp_5_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 5.0 version (loop transformation)\n");

    // OpenMP 5.0: tile directive для оптимизации доступа к памяти
#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // OpenMP 5.0: scan directive для редукций (если поддерживается)
#ifdef __clang__
#pragma omp simd reduction(+:sumVector)
#else
#pragma omp simd reduction(inscan,+:sumVector)
#endif
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
#ifndef __clang__
#pragma omp scan inclusive(sumVector)
#endif
        }

        // Нормализуем значения феромонов
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        // Вычисляем вероятностные значения
#ifdef __clang__
#pragma omp simd reduction(+:sumVector)
#else
#pragma omp simd reduction(inscan,+:sumVector)
#endif
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
#ifndef __clang__
#pragma omp scan inclusive(sumVector)
#endif
        }

        norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
        for (int i = 1; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
        }
    }
}
#endif
#if _OPENMP >= 202011  // OpenMP 5.1+
void go_mass_probability_omp_5_1(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 5.1 version (error recovery and loop features)\n");

    // OpenMP 5.1: order(concurrent) для неупорядоченного выполнения
#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // OpenMP 5.1: неблокирующие редукции
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        // Вычисляем вероятностные значения
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }

        norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
        for (int i = 1; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
        }
    }
}
#endif
#if _OPENMP >= 202111  // OpenMP 5.2+
void go_mass_probability_omp_5_2(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 5.2 version (latest features)\n");

    // OpenMP 5.2: assume clauses для оптимизатора
#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // OpenMP 5.2: улучшенные редукции
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        // Вычисляем вероятностные значения
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }

        norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
        for (int i = 1; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
        }
    }
}
#endif
void go_mass_probability_omp(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("OpenMP version detected: %d\n", _OPENMP);

#if _OPENMP >= 202111  // OpenMP 5.2+
    go_mass_probability_omp_5_2(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 202011  // OpenMP 5.1+
    go_mass_probability_omp_5_1(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201811  // OpenMP 5.0+
    go_mass_probability_omp_5_0(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201511  // OpenMP 4.5+
    go_mass_probability_omp_4_5(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201307  // OpenMP 4.0+
    go_mass_probability_omp_4_0(pheromon, kol_enter, norm_matrix_probability);
#else  // OpenMP 2.0/3.0/3.1
    go_mass_probability_omp_2_0(pheromon, kol_enter, norm_matrix_probability);
#endif
}
// Базовая версия OpenMP 2.0/3.0/3.1 - только CPU параллелизм
void go_mass_probability_non_cuda_not_f_omp_2_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 2.0-3.1 version (CPU parallel for)\n");

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

        // Записываем нормализованные вероятности
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = svertka[i] / sumVector;
        }
    }
}
#if _OPENMP >= 201307  // OpenMP 4.0+
void go_mass_probability_non_cuda_not_f_omp_4_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 4.0 version (SIMD vectorization)\n");

#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Суммируем значения феромонов
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        // Вычисляем вероятностные значения
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }

        // Записываем нормализованные вероятности
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = svertka[i] / sumVector;
        }
    }
}
#endif
#if _OPENMP >= 201511  // OpenMP 4.5+
void go_mass_probability_non_cuda_not_f_omp_4_5(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 4.5 version (if clause and loop nesting)\n");
#if defined(__clang__)
#pragma omp parallel for 
#else
#pragma omp parallel for // if(PARAMETR_SIZE > 100)
#endif
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Суммируем значения феромонов
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        // Вычисляем вероятностные значения
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }

        // Записываем нормализованные вероятности
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = svertka[i] / sumVector;
        }
    }
}
#endif
#if _OPENMP >= 201811  // OpenMP 5.0+
void go_mass_probability_non_cuda_not_f_omp_5_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 5.0 version (loop transformation)\n");

#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Суммируем значения феромонов
#ifdef __clang__
#pragma omp simd reduction(+:sumVector)
#else
#pragma omp simd reduction(inscan,+:sumVector)
#endif
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
#ifndef __clang__
#pragma omp scan inclusive(sumVector)
#endif
        }

        // Нормализуем значения феромонов
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        // Вычисляем вероятностные значения
#ifdef __clang__
#pragma omp simd reduction(+:sumVector)
#else
#pragma omp simd reduction(inscan,+:sumVector)
#endif
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
#ifndef __clang__
#pragma omp scan inclusive(sumVector)
#endif
        }

        // Записываем нормализованные вероятности
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = svertka[i] / sumVector;
        }
    }
}
#endif
#if _OPENMP >= 202011  // OpenMP 5.1+
void go_mass_probability_non_cuda_not_f_omp_5_1(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 5.1 version (error recovery and loop features)\n");

#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Суммируем значения феромонов
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        // Вычисляем вероятностные значения
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }

        // Записываем нормализованные вероятности
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = svertka[i] / sumVector;
        }
    }
}
#endif
#if _OPENMP >= 202111  // OpenMP 5.2+
void go_mass_probability_non_cuda_not_f_omp_5_2(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 5.2 version (latest features)\n");

#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Суммируем значения феромонов
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        // Вычисляем вероятностные значения
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }

        // Записываем нормализованные вероятности
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = svertka[i] / sumVector;
        }
    }
}
#endif
void go_mass_probability_non_cuda_not_f_omp(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
#if _OPENMP >= 202111  // OpenMP 5.2+
    go_mass_probability_non_cuda_not_f_omp_5_2(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 202011  // OpenMP 5.1+
    go_mass_probability_non_cuda_not_f_omp_5_1(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201811  // OpenMP 5.0+
    go_mass_probability_non_cuda_not_f_omp_5_0(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201511  // OpenMP 4.5+
    go_mass_probability_non_cuda_not_f_omp_4_5(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201307  // OpenMP 4.0+
    go_mass_probability_non_cuda_not_f_omp_4_0(pheromon, kol_enter, norm_matrix_probability);
#else  // OpenMP 2.0/3.0/3.1
    go_mass_probability_non_cuda_not_f_omp_2_0(pheromon, kol_enter, norm_matrix_probability);
#endif
}
// Базовая версия OpenMP 2.0/3.0/3.1 - только CPU параллелизм
void go_mass_probability_non_cuda_sort_omp_2_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability, int* __restrict indices) {
    //printf("Using OpenMP 2.0-3.1 version (CPU parallel for)\n");

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

        // Заполняем массив индексов начальными значениями
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            indices[MAX_VALUE_SIZE * tx + i] = i;
        }

        // Ручная сортировка методом вставки с сохранением индексов
        for (int j = 1; j < MAX_VALUE_SIZE; ++j) {
            double key = svertka[j];               // Значение текущего элемента
            int idx_key = indices[MAX_VALUE_SIZE * tx + j]; // Индекс текущего элемента
            int i = j - 1;                         // Начинаем проверку с предыдущего элемента

            while (i >= 0 && svertka[i] > key) {
                svertka[i + 1] = svertka[i];         // Сдвигаем больше элементы вправо
                indices[MAX_VALUE_SIZE * tx + i + 1] = indices[MAX_VALUE_SIZE * tx + i]; // Обновляем соответствующий индекс
                i--;
            }
            svertka[i + 1] = key;                   // Кладём ключ на новое место
            indices[MAX_VALUE_SIZE * tx + i + 1] = idx_key; // Сохраняем индекс ключа
        }

        // Нормирование значений матрицы с накоплением
        norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
        for (int i = 1; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
        }
    }
}
#if _OPENMP >= 201307  // OpenMP 4.0+
void go_mass_probability_non_cuda_sort_omp_4_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability, int* __restrict indices) {
    //printf("Using OpenMP 4.0 version (SIMD vectorization)\n");

#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Суммируем значения феромонов
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        // Вычисляем вероятностные значения
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }

        // Заполняем массив индексов начальными значениями
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            indices[MAX_VALUE_SIZE * tx + i] = i;
        }

        // Ручная сортировка методом вставки с сохранением индексов
        for (int j = 1; j < MAX_VALUE_SIZE; ++j) {
            double key = svertka[j];               // Значение текущего элемента
            int idx_key = indices[MAX_VALUE_SIZE * tx + j]; // Индекс текущего элемента
            int i = j - 1;                         // Начинаем проверку с предыдущего элемента

            while (i >= 0 && svertka[i] > key) {
                svertka[i + 1] = svertka[i];         // Сдвигаем больше элементы вправо
                indices[MAX_VALUE_SIZE * tx + i + 1] = indices[MAX_VALUE_SIZE * tx + i]; // Обновляем соответствующий индекс
                i--;
            }
            svertka[i + 1] = key;                   // Кладём ключ на новое место
            indices[MAX_VALUE_SIZE * tx + i + 1] = idx_key; // Сохраняем индекс ключа
        }

        // Нормирование значений матрицы с накоплением
        norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
        for (int i = 1; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
        }
    }
}
#endif
#if _OPENMP >= 201511  // OpenMP 4.5+
void go_mass_probability_non_cuda_sort_omp_4_5(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability, int* __restrict indices) {
    //printf("Using OpenMP 4.5 version (if clause and loop nesting)\n");
#if defined(__clang__)
#pragma omp parallel for 
#else
#pragma omp parallel for // if(PARAMETR_SIZE > 100)
#endif

    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Суммируем значения феромонов
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        // Вычисляем вероятностные значения
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }

        // Заполняем массив индексов начальными значениями
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            indices[MAX_VALUE_SIZE * tx + i] = i;
        }

        // Ручная сортировка методом вставки с сохранением индексов
        for (int j = 1; j < MAX_VALUE_SIZE; ++j) {
            double key = svertka[j];               // Значение текущего элемента
            int idx_key = indices[MAX_VALUE_SIZE * tx + j]; // Индекс текущего элемента
            int i = j - 1;                         // Начинаем проверку с предыдущего элемента

            while (i >= 0 && svertka[i] > key) {
                svertka[i + 1] = svertka[i];         // Сдвигаем больше элементы вправо
                indices[MAX_VALUE_SIZE * tx + i + 1] = indices[MAX_VALUE_SIZE * tx + i]; // Обновляем соответствующий индекс
                i--;
            }
            svertka[i + 1] = key;                   // Кладём ключ на новое место
            indices[MAX_VALUE_SIZE * tx + i + 1] = idx_key; // Сохраняем индекс ключа
        }

        // Нормирование значений матрицы с накоплением
        norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
        for (int i = 1; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
        }
    }
}
#endif
#if _OPENMP >= 201811  // OpenMP 5.0+
void go_mass_probability_non_cuda_sort_omp_5_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability, int* __restrict indices) {
    //printf("Using OpenMP 5.0 version (loop transformation)\n");

#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Суммируем значения феромонов
#ifdef __clang__
#pragma omp simd reduction(+:sumVector)
#else
#pragma omp simd reduction(inscan,+:sumVector)
#endif
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
#ifndef __clang__
#pragma omp scan inclusive(sumVector)
#endif
        }

        // Нормализуем значения феромонов
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        // Вычисляем вероятностные значения
#ifdef __clang__
#pragma omp simd reduction(+:sumVector)
#else
#pragma omp simd reduction(inscan,+:sumVector)
#endif
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
#ifndef __clang__
#pragma omp scan inclusive(sumVector)
#endif
        }

        // Заполняем массив индексов начальными значениями
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            indices[MAX_VALUE_SIZE * tx + i] = i;
        }

        // Ручная сортировка методом вставки с сохранением индексов
        for (int j = 1; j < MAX_VALUE_SIZE; ++j) {
            double key = svertka[j];               // Значение текущего элемента
            int idx_key = indices[MAX_VALUE_SIZE * tx + j]; // Индекс текущего элемента
            int i = j - 1;                         // Начинаем проверку с предыдущего элемента

            while (i >= 0 && svertka[i] > key) {
                svertka[i + 1] = svertka[i];         // Сдвигаем больше элементы вправо
                indices[MAX_VALUE_SIZE * tx + i + 1] = indices[MAX_VALUE_SIZE * tx + i]; // Обновляем соответствующий индекс
                i--;
            }
            svertka[i + 1] = key;                   // Кладём ключ на новое место
            indices[MAX_VALUE_SIZE * tx + i + 1] = idx_key; // Сохраняем индекс ключа
        }

        // Нормирование значений матрицы с накоплением
        norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
        for (int i = 1; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
        }
    }
}
#endif
#if _OPENMP >= 202011  // OpenMP 5.1+
void go_mass_probability_non_cuda_sort_omp_5_1(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability, int* __restrict indices) {
    //printf("Using OpenMP 5.1 version (error recovery and loop features)\n");

#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Суммируем значения феромонов
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        // Вычисляем вероятностные значения
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }

        // Заполняем массив индексов начальными значениями
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            indices[MAX_VALUE_SIZE * tx + i] = i;
        }

        // Ручная сортировка методом вставки с сохранением индексов
        for (int j = 1; j < MAX_VALUE_SIZE; ++j) {
            double key = svertka[j];               // Значение текущего элемента
            int idx_key = indices[MAX_VALUE_SIZE * tx + j]; // Индекс текущего элемента
            int i = j - 1;                         // Начинаем проверку с предыдущего элемента

            while (i >= 0 && svertka[i] > key) {
                svertka[i + 1] = svertka[i];         // Сдвигаем больше элементы вправо
                indices[MAX_VALUE_SIZE * tx + i + 1] = indices[MAX_VALUE_SIZE * tx + i]; // Обновляем соответствующий индекс
                i--;
            }
            svertka[i + 1] = key;                   // Кладём ключ на новое место
            indices[MAX_VALUE_SIZE * tx + i + 1] = idx_key; // Сохраняем индекс ключа
        }

        // Нормирование значений матрицы с накоплением
        norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
        for (int i = 1; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
        }
    }
}
#endif
#if _OPENMP >= 202111  // OpenMP 5.2+
void go_mass_probability_non_cuda_sort_omp_5_2(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability, int* __restrict indices) {
    //printf("Using OpenMP 5.2 version (latest features)\n");

#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

        // Суммируем значения феромонов
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        // Вычисляем вероятностные значения
#pragma omp simd reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            svertka[i] = probability_formula_non_cuda(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
            sumVector += svertka[i];
        }

        // Заполняем массив индексов начальными значениями
#pragma omp simd
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            indices[MAX_VALUE_SIZE * tx + i] = i;
        }

        // Ручная сортировка методом вставки с сохранением индексов
        for (int j = 1; j < MAX_VALUE_SIZE; ++j) {
            double key = svertka[j];               // Значение текущего элемента
            int idx_key = indices[MAX_VALUE_SIZE * tx + j]; // Индекс текущего элемента
            int i = j - 1;                         // Начинаем проверку с предыдущего элемента

            while (i >= 0 && svertka[i] > key) {
                svertka[i + 1] = svertka[i];         // Сдвигаем больше элементы вправо
                indices[MAX_VALUE_SIZE * tx + i + 1] = indices[MAX_VALUE_SIZE * tx + i]; // Обновляем соответствующий индекс
                i--;
            }
            svertka[i + 1] = key;                   // Кладём ключ на новое место
            indices[MAX_VALUE_SIZE * tx + i + 1] = idx_key; // Сохраняем индекс ключа
        }

        // Нормирование значений матрицы с накоплением
        norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
        for (int i = 1; i < MAX_VALUE_SIZE; i++) {
            norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
        }
    }
}
#endif
void go_mass_probability_non_cuda_sort_omp(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability, int* __restrict indices) {
#if _OPENMP >= 202111  // OpenMP 5.2+
    go_mass_probability_non_cuda_sort_omp_5_2(pheromon, kol_enter, norm_matrix_probability, indices);
#elif _OPENMP >= 202011  // OpenMP 5.1+
    go_mass_probability_non_cuda_sort_omp_5_1(pheromon, kol_enter, norm_matrix_probability, indices);
#elif _OPENMP >= 201811  // OpenMP 5.0+
    go_mass_probability_non_cuda_sort_omp_5_0(pheromon, kol_enter, norm_matrix_probability, indices);
#elif _OPENMP >= 201511  // OpenMP 4.5+
    go_mass_probability_non_cuda_sort_omp_4_5(pheromon, kol_enter, norm_matrix_probability, indices);
#elif _OPENMP >= 201307  // OpenMP 4.0+
    go_mass_probability_non_cuda_sort_omp_4_0(pheromon, kol_enter, norm_matrix_probability, indices);
#else  // OpenMP 2.0/3.0/3.1
    go_mass_probability_non_cuda_sort_omp_2_0(pheromon, kol_enter, norm_matrix_probability, indices);
#endif
}
// Базовая версия OpenMP 2.0/3.0/3.1 - только CPU параллелизм
void go_opt_mass_probability_omp_2_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 2.0-3.1 version (CPU parallel for)\n");

#pragma omp parallel for schedule(static)
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        const int base_idx = MAX_VALUE_SIZE * tx;

        double sumPheromon = 0.0;
        // Суммируем значения феромонов
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumPheromon += pheromon[base_idx + i];
        }

        // Обработка случая нулевой суммы
        if (sumPheromon == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[base_idx] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[base_idx + i] = norm_matrix_probability[base_idx + i - 1] + uniform_prob;
            }
            continue;
        }

        // Вычисляем svertka и их сумму
        double sumSvertka = 0.0;
        double svertka_values[MAX_VALUE_SIZE];

        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            double pheromon_norm = pheromon[base_idx + i] / sumPheromon;
            svertka_values[i] = probability_formula_non_cuda(pheromon_norm, kol_enter[base_idx + i]);
            sumSvertka += svertka_values[i];
        }

        // Обработка случая нулевой суммы svertka
        if (sumSvertka == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[base_idx] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[base_idx + i] = norm_matrix_probability[base_idx + i - 1] + uniform_prob;
            }
        }
        else {
            // Вычисляем кумулятивные вероятности
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                cumulative += svertka_values[i] / sumSvertka;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            // Гарантируем, что последнее значение равно 1.0 (из-за ошибок округления)
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
    }
}
#if _OPENMP >= 201307  // OpenMP 4.0+
void go_opt_mass_probability_omp_4_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 4.0 version (SIMD vectorization)\n");

#pragma omp parallel for schedule(static)
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        const int base_idx = MAX_VALUE_SIZE * tx;

        double sumPheromon = 0.0;
        // Суммируем значения феромонов
#pragma omp simd reduction(+:sumPheromon)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumPheromon += pheromon[base_idx + i];
        }

        // Обработка случая нулевой суммы
        if (sumPheromon == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[base_idx] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[base_idx + i] = norm_matrix_probability[base_idx + i - 1] + uniform_prob;
            }
            continue;
        }

        // Вычисляем svertka и их сумму
        double sumSvertka = 0.0;
        double svertka_values[MAX_VALUE_SIZE];

#pragma omp simd reduction(+:sumSvertka)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            double pheromon_norm = pheromon[base_idx + i] / sumPheromon;
            svertka_values[i] = probability_formula_non_cuda(pheromon_norm, kol_enter[base_idx + i]);
            sumSvertka += svertka_values[i];
        }

        // Обработка случая нулевой суммы svertka
        if (sumSvertka == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[base_idx] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[base_idx + i] = norm_matrix_probability[base_idx + i - 1] + uniform_prob;
            }
        }
        else {
            // Вычисляем кумулятивные вероятности
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                cumulative += svertka_values[i] / sumSvertka;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            // Гарантируем, что последнее значение равно 1.0 (из-за ошибок округления)
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
    }
}
#endif
#if _OPENMP >= 201511  // OpenMP 4.5+
void go_opt_mass_probability_omp_4_5(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 4.5 version (if clause and loop nesting)\n");

#if defined(__clang__)
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(static) // if(PARAMETR_SIZE > 100)
#endif
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        const int base_idx = MAX_VALUE_SIZE * tx;

        double sumPheromon = 0.0;
        // Суммируем значения феромонов
#pragma omp simd reduction(+:sumPheromon)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumPheromon += pheromon[base_idx + i];
        }

        // Обработка случая нулевой суммы
        if (sumPheromon == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[base_idx] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[base_idx + i] = norm_matrix_probability[base_idx + i - 1] + uniform_prob;
            }
            continue;
        }

        // Вычисляем svertka и их сумму
        double sumSvertka = 0.0;
        double svertka_values[MAX_VALUE_SIZE];

#pragma omp simd reduction(+:sumSvertka)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            double pheromon_norm = pheromon[base_idx + i] / sumPheromon;
            svertka_values[i] = probability_formula_non_cuda(pheromon_norm, kol_enter[base_idx + i]);
            sumSvertka += svertka_values[i];
        }

        // Обработка случая нулевой суммы svertka
        if (sumSvertka == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[base_idx] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[base_idx + i] = norm_matrix_probability[base_idx + i - 1] + uniform_prob;
            }
        }
        else {
            // Вычисляем кумулятивные вероятности
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                cumulative += svertka_values[i] / sumSvertka;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            // Гарантируем, что последнее значение равно 1.0 (из-за ошибок округления)
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
    }
}
#endif
#if _OPENMP >= 201811  // OpenMP 5.0+
void go_opt_mass_probability_omp_5_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 5.0 version (loop transformation)\n");

#pragma omp parallel for schedule(static)
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        const int base_idx = MAX_VALUE_SIZE * tx;

        double sumPheromon = 0.0;
        // Суммируем значения феромонов
#ifdef __clang__
#pragma omp simd reduction(+:sumPheromon)
#else
#pragma omp simd reduction(inscan,+:sumPheromon)
#endif
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumPheromon += pheromon[base_idx + i];
#ifndef __clang__
#pragma omp scan inclusive(sumPheromon)
#endif
        }

        // Обработка случая нулевой суммы
        if (sumPheromon == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[base_idx] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[base_idx + i] = norm_matrix_probability[base_idx + i - 1] + uniform_prob;
            }
            continue;
        }

        // Вычисляем svertka и их сумму
        double sumSvertka = 0.0;
        double svertka_values[MAX_VALUE_SIZE];

#ifdef __clang__
#pragma omp simd reduction(+:sumSvertka)
#else
#pragma omp simd reduction(inscan,+:sumSvertka)
#endif
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            double pheromon_norm = pheromon[base_idx + i] / sumPheromon;
            svertka_values[i] = probability_formula_non_cuda(pheromon_norm, kol_enter[base_idx + i]);
            sumSvertka += svertka_values[i];
#ifndef __clang__
#pragma omp scan inclusive(sumSvertka)
#endif
        }

        // Обработка случая нулевой суммы svertka
        if (sumSvertka == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[base_idx] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[base_idx + i] = norm_matrix_probability[base_idx + i - 1] + uniform_prob;
            }
        }
        else {
            // Вычисляем кумулятивные вероятности
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                cumulative += svertka_values[i] / sumSvertka;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            // Гарантируем, что последнее значение равно 1.0 (из-за ошибок округления)
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
    }
}
#endif
#if _OPENMP >= 202011  // OpenMP 5.1+
void go_opt_mass_probability_omp_5_1(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 5.1 version (error recovery and loop features)\n");

#pragma omp parallel for schedule(static)
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        const int base_idx = MAX_VALUE_SIZE * tx;

        double sumPheromon = 0.0;
        // Суммируем значения феромонов
#pragma omp simd reduction(+:sumPheromon)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumPheromon += pheromon[base_idx + i];
        }

        // Обработка случая нулевой суммы
        if (sumPheromon == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[base_idx] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[base_idx + i] = norm_matrix_probability[base_idx + i - 1] + uniform_prob;
            }
            continue;
        }

        // Вычисляем svertka и их сумму
        double sumSvertka = 0.0;
        double svertka_values[MAX_VALUE_SIZE];

#pragma omp simd reduction(+:sumSvertka)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            double pheromon_norm = pheromon[base_idx + i] / sumPheromon;
            svertka_values[i] = probability_formula_non_cuda(pheromon_norm, kol_enter[base_idx + i]);
            sumSvertka += svertka_values[i];
        }

        // Обработка случая нулевой суммы svertka
        if (sumSvertka == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[base_idx] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[base_idx + i] = norm_matrix_probability[base_idx + i - 1] + uniform_prob;
            }
        }
        else {
            // Вычисляем кумулятивные вероятности
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                cumulative += svertka_values[i] / sumSvertka;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            // Гарантируем, что последнее значение равно 1.0 (из-за ошибок округления)
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
    }
}
#endif
#if _OPENMP >= 202111  // OpenMP 5.2+
void go_opt_mass_probability_omp_5_2(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("Using OpenMP 5.2 version (latest features)\n");

#pragma omp parallel for schedule(static)
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        const int base_idx = MAX_VALUE_SIZE * tx;

        double sumPheromon = 0.0;
        // Суммируем значения феромонов
#pragma omp simd reduction(+:sumPheromon)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumPheromon += pheromon[base_idx + i];
        }

        // Обработка случая нулевой суммы
        if (sumPheromon == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[base_idx] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[base_idx + i] = norm_matrix_probability[base_idx + i - 1] + uniform_prob;
            }
            continue;
        }

        // Вычисляем svertka и их сумму
        double sumSvertka = 0.0;
        double svertka_values[MAX_VALUE_SIZE];

#pragma omp simd reduction(+:sumSvertka)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            double pheromon_norm = pheromon[base_idx + i] / sumPheromon;
            svertka_values[i] = probability_formula_non_cuda(pheromon_norm, kol_enter[base_idx + i]);
            sumSvertka += svertka_values[i];
        }

        // Обработка случая нулевой суммы svertka
        if (sumSvertka == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[base_idx] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[base_idx + i] = norm_matrix_probability[base_idx + i - 1] + uniform_prob;
            }
        }
        else {
            // Вычисляем кумулятивные вероятности
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                cumulative += svertka_values[i] / sumSvertka;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            // Гарантируем, что последнее значение равно 1.0 (из-за ошибок округления)
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
    }
}
#endif
void go_opt_mass_probability_omp(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //printf("OpenMP version detected: %d\n", _OPENMP);

#if _OPENMP >= 202111  // OpenMP 5.2+
    go_opt_mass_probability_omp_5_2(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 202011  // OpenMP 5.1+
    go_opt_mass_probability_omp_5_1(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201811  // OpenMP 5.0+
    go_opt_mass_probability_omp_5_0(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201511  // OpenMP 4.5+
    go_opt_mass_probability_omp_4_5(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201307  // OpenMP 4.0+
    go_opt_mass_probability_omp_4_0(pheromon, kol_enter, norm_matrix_probability);
#else  // OpenMP 2.0/3.0/3.1
    go_opt_mass_probability_omp_2_0(pheromon, kol_enter, norm_matrix_probability);
#endif
}

void go_mass_probability_non_cuda(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
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
void go_mass_probability_non_cuda_not_f(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
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
void go_mass_probability_non_cuda_sort(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability, int* __restrict indices) {
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
//Общие версии оптимизированные под любые версии OMP
inline double unified_fast_random(uint64_t& seed) {  
    // Xorshift64 - хороший баланс скорость/качество
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    double result = (seed >> 11) / 9007199254740992.0;  // [0, 1)
    return result;
}
void go_all_agent_omp(int gpuTime, double* __restrict parametr, double* __restrict norm_matrix_probability, double* __restrict agent, int* __restrict agent_node, double* __restrict OF, HashEntry* __restrict hashTable, int& kol_hash_fail, double& totalHashTime, double& totalOFTime) {

    int local_kol_hash_fail = 0;
    double local_totalHashTime = 0.0;
    double local_totalOFTime = 0.0;

    // Универсальная параллельная секция с условными директивами
#if _OPENMP >= 201511 && !defined(__clang__)
#pragma omp parallel reduction(+:local_kol_hash_fail, local_totalHashTime, local_totalOFTime) // if(ANT_SIZE > 100)
#else
#pragma omp parallel reduction(+:local_kol_hash_fail, local_totalHashTime, local_totalOFTime)
#endif
    {
        uint64_t seed = 123 + gpuTime + omp_get_thread_num();

        // Условное распределение работы в зависимости от версии OpenMP
#if defined(__clang__)
        // Clang - только базовые возможности
#pragma omp for schedule(static)
#else
        // Другие компиляторы - условные возможности  
#if _OPENMP >= 201511
#pragma omp for schedule(dynamic, 16)
#elif _OPENMP >= 201307
#pragma omp for schedule(guided)  
#else
#pragma omp for schedule(static)
#endif
#endif
        for(int bx = 0; bx < ANT_SIZE; bx++) {
            // Оптимизированная генерация пути с учетом малого MAX_VALUE_SIZE
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                double randomValue = unified_fast_random(seed);

                // Линейный поиск с предположением о малом размере
                int k = 0;
                while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                    k++;
                }
                agent_node[bx * PARAMETR_SIZE + tx] = k;
                agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
            }

            auto start = std::chrono::high_resolution_clock::now();
            double cachedResult = -1.0;

#pragma omp critical (hash_lookup)
            {
                cachedResult = getCachedResultOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx);
            }

            if (cachedResult == -1.0) {
                auto start_OF = std::chrono::high_resolution_clock::now();
                OF[bx] = BenchShafferaFunction_omp(&agent[bx * PARAMETR_SIZE]);
                auto end_OF = std::chrono::high_resolution_clock::now();
                local_totalOFTime += std::chrono::duration<double, std::milli>(end_OF - start_OF).count();

#pragma omp critical(hash_write)
                {
                    saveToCacheOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx, OF[bx]);
                }
            }
            else {
                local_kol_hash_fail++;

                // Обработка в зависимости от типа алгоритма
                switch (TYPE_ACO) {
                case 0: // ACOCN
                    OF[bx] = cachedResult;
                    break;

                case 1: // ACOCNI
                    OF[bx] = ZERO_HASH_RESULT;
                    break;
                case 2: // ACOCCyN
                {
                    int nom_iteration = 0;
                    double currentCachedResult = cachedResult;

                    // Пытаемся найти уникальный путь
                    while (currentCachedResult != -1.0 && nom_iteration < ACOCCyN_KOL_ITERATION) {
                        // Генерируем новый путь с тем же генератором
                        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                            double randomValue = unified_fast_random(seed);

                            int k = 0;
                            while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                                k++;
                            }
                            agent_node[bx * PARAMETR_SIZE + tx] = k;
                            agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
                        }

#pragma omp critical (hash_lookup)
                        {
                            currentCachedResult = getCachedResultOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx);
                        }
                        nom_iteration++;
                        local_kol_hash_fail++;
                    }

                    // Если нашли уникальный путь или превысили лимит итераций
                    if (currentCachedResult == -1.0) {
                        auto start_OF = std::chrono::high_resolution_clock::now();
                        OF[bx] = BenchShafferaFunction_omp(&agent[bx * PARAMETR_SIZE]);
                        auto end_OF = std::chrono::high_resolution_clock::now();
                        local_totalOFTime += std::chrono::duration<double, std::milli>(end_OF - start_OF).count();

#pragma omp critical(hash_write)
                        {
                            saveToCacheOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx, OF[bx]);
                        }
                    }
                    else {
                        // Используем последнее найденное кэшированное значение
                        OF[bx] = currentCachedResult;
                    }
                }
                break;

                default:
                    OF[bx] = cachedResult;
                    break;
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            local_totalHashTime += std::chrono::duration<double, std::milli>(end - start).count();
        }
    }
    // Обновление глобальных переменных
    kol_hash_fail += local_kol_hash_fail;
    totalHashTime += local_totalHashTime;
    totalOFTime += local_totalOFTime;
}
void go_all_agent_omp_non_hash(int gpuTime, double* __restrict parametr, double* __restrict norm_matrix_probability, double* __restrict agent, int* __restrict agent_node, double* __restrict OF, int& kol_hash_fail, double& totalHashTime, double& totalOFTime) {

    double local_totalOFTime = 0.0;

    // Условный параллелизм для OpenMP 4.5+
#if _OPENMP >= 201511 && !defined(__clang__)
#pragma omp parallel reduction(+:local_totalOFTime) // if(ANT_SIZE > 100)
#else
#pragma omp parallel reduction(+:local_totalOFTime)
#endif
    {
        uint64_t seed = 123 + gpuTime + omp_get_thread_num();
        if (seed == 0) seed = 1;

        // Условное распределение работы в зависимости от версии OpenMP
#if defined(__clang__)
        // Clang - только базовые возможности
#pragma omp for schedule(static)
#else
        // Другие компиляторы - условные возможности  
#if _OPENMP >= 201511
#pragma omp for schedule(dynamic, 16)
#elif _OPENMP >= 201307
#pragma omp for schedule(guided)  
#else
#pragma omp for schedule(static)
#endif
#endif
        for (int bx = 0; bx < ANT_SIZE; bx++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                double randomValue = unified_fast_random(seed);

                int k = 0;
                // Эффективный линейный поиск
                while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                    k++;
                }
                agent_node[bx * PARAMETR_SIZE + tx] = k;
                agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
            }

            auto start_OF = std::chrono::high_resolution_clock::now();
            OF[bx] = BenchShafferaFunction_omp(&agent[bx * PARAMETR_SIZE]);
            auto end_OF = std::chrono::high_resolution_clock::now();
            local_totalOFTime += std::chrono::duration<double, std::milli>(end_OF - start_OF).count();
        }
    }
    totalOFTime += local_totalOFTime;
}
void go_all_agent_omp_binary_non_hash_unified(int gpuTime, double* __restrict parametr, double* __restrict norm_matrix_probability, double* __restrict agent, int* __restrict agent_node, double* __restrict OF, int& kol_hash_fail, double& totalHashTime, double& totalOFTime) {
    double local_totalOFTime = 0.0;
    // Условный параллелизм для OpenMP 4.5+
#if _OPENMP >= 201511 && !defined(__clang__)
#pragma omp parallel reduction(+:local_totalOFTime) // if(ANT_SIZE > 100)
#else
#pragma omp parallel reduction(+:local_totalOFTime)
#endif
    {
        // Более эффективный генератор случайных чисел
        std::mt19937_64 generator(123 + gpuTime + omp_get_thread_num());
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        // Условное распределение работы в зависимости от версии OpenMP
#if defined(__clang__)
        // Clang - только базовые возможности
#pragma omp for schedule(static)
#else
        // Другие компиляторы - условные возможности  
#if _OPENMP >= 201511
#pragma omp for schedule(dynamic, 16)
#elif _OPENMP >= 201307
#pragma omp for schedule(guided)  
#else
#pragma omp for schedule(static)
#endif
#endif
        for (int bx = 0; bx < ANT_SIZE; bx++) {
            const int agent_base_idx = bx * PARAMETR_SIZE;
            // Генерация пути агента с бинарным поиском
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                double randomValue = distribution(generator);
                const int norm_base_idx = MAX_VALUE_SIZE * tx;
                const int param_base_idx = tx * MAX_VALUE_SIZE;

                // Бинарный поиск в cumulative distribution
                int left = 0;
                int right = MAX_VALUE_SIZE - 1;
                int k = right; // По умолчанию последний элемент

                while (left <= right) {
                    int mid = left + (right - left) / 2;
                    if (randomValue > norm_matrix_probability[norm_base_idx + mid]) {
                        left = mid + 1;
                    }
                    else {
                        k = mid;
                        right = mid - 1;
                    }
                }

                agent_node[agent_base_idx + tx] = k;
                agent[agent_base_idx + tx] = parametr[param_base_idx + k];
            }

            // Измерение времени вычисления целевой функции
            auto start_OF = std::chrono::high_resolution_clock::now();
            OF[bx] = BenchShafferaFunction_omp(&agent[agent_base_idx]);
            auto end_OF = std::chrono::high_resolution_clock::now();

            local_totalOFTime += std::chrono::duration<double, std::milli>(end_OF - start_OF).count();
        }
    }
    totalOFTime += local_totalOFTime;
}

// Функция для вычисления пути агентов на CPU
void go_all_agent_non_cuda_time(int gpuTime, double* __restrict parametr, double* __restrict norm_matrix_probability, double* __restrict agent, int* __restrict agent_node, double* __restrict OF, HashEntry* __restrict hashTable, int& kol_hash_fail, double& totalHashTime, double& totalOFTime, double& HashTimeSave, double& HashTimeSearch, double& SumTimeSearch) {
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
        double cachedResult = getCachedResultOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx);
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
            saveToCacheOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx, OF[bx]);
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
                    cachedResult = getCachedResultOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx);
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
                saveToCacheOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx, OF[bx]);
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
void go_all_agent_non_cuda(int gpuTime, double* __restrict parametr, double* __restrict norm_matrix_probability, double* __restrict agent, int* __restrict agent_node, double* __restrict OF, HashEntry* __restrict hashTable, int& kol_hash_fail) {
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
        double cachedResult = getCachedResultOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx);
        int nom_iteration = 0;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
            saveToCacheOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx, OF[bx]);
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
                    cachedResult = getCachedResultOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx);
                    nom_iteration = nom_iteration + 1;
                    kol_hash_fail = kol_hash_fail + 1;
                }

                OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
                saveToCacheOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx, OF[bx]);
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

void process_agent(int bx, double* __restrict parametr, double* __restrict norm_matrix_probability, double* __restrict agent, int* __restrict agent_node, double* __restrict OF, HashEntry* __restrict hashTable, int& kol_hash_fail) {
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
    double cachedResult = getCachedResultOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx);
    int nom_iteration = 0;

    if (cachedResult == -1.0) {
        // Если значение не найдено в ХЭШ, то заносим новое значение
        OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
        saveToCacheOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx, OF[bx]);
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
                cachedResult = getCachedResultOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx);
                nom_iteration++;
                mtx.lock();
                kol_hash_fail++;
                mtx.unlock();
            }

            OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
            saveToCacheOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx, OF[bx]);
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

void go_all_agent_non_cuda_thread(double* __restrict parametr, double* __restrict norm_matrix_probability, double* __restrict agent, int* __restrict agent_node, double* __restrict OF, HashEntry* __restrict hashTable, int& kol_hash_fail, int num_threads) {
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
void go_all_agent_non_cuda_non_hash(int gpuTime, double* __restrict parametr, double* __restrict norm_matrix_probability, double* __restrict agent, int* __restrict agent_node, double* __restrict OF, int& kol_hash_fail, double& totalOFTime) {
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

// Базовая версия OpenMP 2.0/3.0/3.1 - с локальными буферами
void add_pheromon_iteration_omp_2_0(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // Phase 1: Evaporation
#pragma omp parallel for
    for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
        pheromon[idx] *= PARAMETR_RO;
    }

    // Phase 2: Accumulation with thread-local buffers
#pragma omp parallel
    {
        // Thread-local accumulation buffers
        double* local_pheromon_add = static_cast<double*>(calloc(TOTAL_CELLS, sizeof(double)));
        int* local_kol_enter_add = static_cast<int*>(calloc(TOTAL_CELLS, sizeof(int)));

#pragma omp for nowait
        for (int i = 0; i < ANT_SIZE; ++i) {
            double agent_of = OF[i];
#if OPTIMIZE_MIN_2
            double agent_of_reciprocal = (agent_of == 0) ? (PARAMETR_Q / 0.0000001) : (PARAMETR_Q / agent_of);
#elif OPTIMIZE_MAX
            double agent_of_scaled = PARAMETR_Q * agent_of;
#endif

            for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                int k = agent_node[i * PARAMETR_SIZE + tx];
                int idx = MAX_VALUE_SIZE * tx + k;

                local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_1
                double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of;
                if (delta > 0) {
                    local_pheromon_add[idx] += PARAMETR_Q * delta;
                }
#elif OPTIMIZE_MIN_2
                local_pheromon_add[idx] += agent_of_reciprocal;
#elif OPTIMIZE_MAX
                local_pheromon_add[idx] += agent_of_scaled;
#endif
            }
        }

        // Merge thread-local results
#pragma omp critical
        {
            for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
                kol_enter[idx] += local_kol_enter_add[idx];
                pheromon[idx] += local_pheromon_add[idx];
            }
        }

        free(local_pheromon_add);
        free(local_kol_enter_add);
    }
}
#if _OPENMP >= 201307  // OpenMP 4.0+
void add_pheromon_iteration_omp_4_0(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // Phase 1: Evaporation with SIMD
#pragma omp parallel for simd
    for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
        pheromon[idx] *= PARAMETR_RO;
    }

    // Phase 2: Accumulation with optimized thread-local buffers
#pragma omp parallel
    {
        // Thread-local accumulation buffers
        double* local_pheromon_add = static_cast<double*>(calloc(TOTAL_CELLS, sizeof(double)));
        int* local_kol_enter_add = static_cast<int*>(calloc(TOTAL_CELLS, sizeof(int)));

#pragma omp for nowait
        for (int i = 0; i < ANT_SIZE; ++i) {
            double agent_of = OF[i];
#if OPTIMIZE_MIN_2
            double agent_of_reciprocal = (agent_of == 0) ? (PARAMETR_Q / 0.0000001) : (PARAMETR_Q / agent_of);
#elif OPTIMIZE_MAX
            double agent_of_scaled = PARAMETR_Q * agent_of;
#endif

            // Cache-friendly access pattern
            const int* agent_path = &agent_node[i * PARAMETR_SIZE];
            for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                int k = agent_path[tx];
                int idx = MAX_VALUE_SIZE * tx + k;

                local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_1
                double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of;
                if (delta > 0) {
                    local_pheromon_add[idx] += PARAMETR_Q * delta;
                }
#elif OPTIMIZE_MIN_2
                local_pheromon_add[idx] += agent_of_reciprocal;
#elif OPTIMIZE_MAX
                local_pheromon_add[idx] += agent_of_scaled;
#endif
            }
        }

        // Merge thread-local results with SIMD
#pragma omp critical
        {
#pragma omp simd
            for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
                kol_enter[idx] += local_kol_enter_add[idx];
                pheromon[idx] += local_pheromon_add[idx];
            }
        }

        free(local_pheromon_add);
        free(local_kol_enter_add);
    }
}
#endif
#if _OPENMP >= 201511  // OpenMP 4.5+
void add_pheromon_iteration_omp_4_5(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // Phase 1: Evaporation with if clause
#ifdef __clang__
#pragma omp parallel for simd
#else
#pragma omp parallel for simd // if(TOTAL_CELLS > 1000)
#endif
    for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
        pheromon[idx] *= PARAMETR_RO;
    }

    // Phase 2: Accumulation with improved scheduling
#pragma omp parallel
    {
        // Thread-local accumulation buffers
        double* local_pheromon_add = static_cast<double*>(calloc(TOTAL_CELLS, sizeof(double)));
        int* local_kol_enter_add = static_cast<int*>(calloc(TOTAL_CELLS, sizeof(int)));

#pragma omp for nowait
        for (int i = 0; i < ANT_SIZE; ++i) {
            double agent_of = OF[i];
#if OPTIMIZE_MIN_2
            double agent_of_reciprocal = (agent_of == 0) ? (PARAMETR_Q / 0.0000001) : (PARAMETR_Q / agent_of);
#elif OPTIMIZE_MAX
            double agent_of_scaled = PARAMETR_Q * agent_of;
#endif

            // Optimized memory access
            const int* agent_path = &agent_node[i * PARAMETR_SIZE];
            for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                int k = agent_path[tx];
                int idx = MAX_VALUE_SIZE * tx + k;

                local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_1
                double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of;
                if (delta > 0) {
                    local_pheromon_add[idx] += PARAMETR_Q * delta;
                }
#elif OPTIMIZE_MIN_2
                local_pheromon_add[idx] += agent_of_reciprocal;
#elif OPTIMIZE_MAX
                local_pheromon_add[idx] += agent_of_scaled;
#endif
            }
        }

        // Efficient merging
#pragma omp critical
        {
            for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
#pragma omp simd
                for (int k = 0; k < MAX_VALUE_SIZE; ++k) {
                    int idx = MAX_VALUE_SIZE * tx + k;
                    kol_enter[idx] += local_kol_enter_add[idx];
                    pheromon[idx] += local_pheromon_add[idx];
                }
            }
        }

        free(local_pheromon_add);
        free(local_kol_enter_add);
    }
}
#endif
#if _OPENMP >= 201811  // OpenMP 5.0+
void add_pheromon_iteration_omp_5_0(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // Phase 1: Evaporation
#pragma omp parallel for simd
    for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
        pheromon[idx] *= PARAMETR_RO;
    }

    // Phase 2: Accumulation with loop transformation hints
#pragma omp parallel
    {
        // Thread-local accumulation buffers
        double* local_pheromon_add = static_cast<double*>(calloc(TOTAL_CELLS, sizeof(double)));
        int* local_kol_enter_add = static_cast<int*>(calloc(TOTAL_CELLS, sizeof(int)));

        // Initialize buffers to zero (для безопасности)
        for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
            local_pheromon_add[idx] = 0.0;
            local_kol_enter_add[idx] = 0;
        }

#pragma omp for nowait
        for (int i = 0; i < ANT_SIZE; ++i) {
            double agent_of = OF[i];
#if OPTIMIZE_MIN_2
            double agent_of_reciprocal = (agent_of == 0) ? (PARAMETR_Q / 0.0000001) : (PARAMETR_Q / agent_of);
#elif OPTIMIZE_MAX
            double agent_of_scaled = PARAMETR_Q * agent_of;
#endif

            // Cache-friendly access pattern
            const int* agent_path = &agent_node[i * PARAMETR_SIZE];
            for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                int k = agent_path[tx];
                int idx = MAX_VALUE_SIZE * tx + k;

                local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_1
                double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of;
                if (delta > 0) {
                    local_pheromon_add[idx] += PARAMETR_Q * delta;
                }
#elif OPTIMIZE_MIN_2
                local_pheromon_add[idx] += agent_of_reciprocal;
#elif OPTIMIZE_MAX
                local_pheromon_add[idx] += agent_of_scaled;
#endif
            }
        }

        // Merge with efficient memory access
#pragma omp critical
        {
#ifdef __clang__
            for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
                kol_enter[idx] += local_kol_enter_add[idx];
                pheromon[idx] += local_pheromon_add[idx];
            }
#else
#pragma omp simd
            for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
                kol_enter[idx] += local_kol_enter_add[idx];
                pheromon[idx] += local_pheromon_add[idx];
            }
#endif
        }

        free(local_pheromon_add);
        free(local_kol_enter_add);
    }
}
#endif
#if _OPENMP >= 202011  // OpenMP 5.1+
void add_pheromon_iteration_omp_5_1(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // Phase 1: Evaporation
#pragma omp parallel for simd
    for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
        pheromon[idx] *= PARAMETR_RO;
    }

    // Phase 2: Accumulation with non-blocking operations
#pragma omp parallel
    {
        // Thread-local accumulation buffers
        double* local_pheromon_add = static_cast<double*>(calloc(TOTAL_CELLS, sizeof(double)));
        int* local_kol_enter_add = static_cast<int*>(calloc(TOTAL_CELLS, sizeof(int)));

        // Initialize to zero
        for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
            local_pheromon_add[idx] = 0.0;
            local_kol_enter_add[idx] = 0;
        }

#if defined(__clang__)
        // Clang - только базовые возможности
#pragma omp for schedule(static)
#else
        // Другие компиляторы - условные возможности  
#if _OPENMP >= 201511
#pragma omp for schedule(dynamic, 16)
#elif _OPENMP >= 201307
#pragma omp for schedule(guided)  
#else
#pragma omp for schedule(static)
#endif
#endif
        for (int i = 0; i < ANT_SIZE; ++i) {
            double agent_of = OF[i];
#if OPTIMIZE_MIN_2
            double agent_of_reciprocal = (agent_of == 0) ? (PARAMETR_Q / 0.0000001) : (PARAMETR_Q / agent_of);
#elif OPTIMIZE_MAX
            double agent_of_scaled = PARAMETR_Q * agent_of;
#endif

            // Optimized memory access
            const int* agent_path = &agent_node[i * PARAMETR_SIZE];
            for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                int matrix_idx = MAX_VALUE_SIZE * tx + agent_path[tx];

                local_kol_enter_add[matrix_idx]++;

#if OPTIMIZE_MIN_1
                double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of;
                if (delta > 0) {
                    local_pheromon_add[matrix_idx] += PARAMETR_Q * delta;
                }
#elif OPTIMIZE_MIN_2
                local_pheromon_add[matrix_idx] += agent_of_reciprocal;
#elif OPTIMIZE_MAX
                local_pheromon_add[matrix_idx] += agent_of_scaled;
#endif
            }
        }

        // Efficient reduction
#pragma omp critical
        {
#pragma omp simd
            for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
                kol_enter[idx] += local_kol_enter_add[idx];
                pheromon[idx] += local_pheromon_add[idx];
            }
        }

        free(local_pheromon_add);
        free(local_kol_enter_add);
    }
}
#endif
#if _OPENMP >= 202111  // OpenMP 5.2+
void add_pheromon_iteration_omp_5_2(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // Phase 1: Evaporation
#pragma omp parallel for simd
    for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
        pheromon[idx] *= PARAMETR_RO;
    }

    // Phase 2: Accumulation with latest optimizations
#pragma omp parallel
    {
        // Thread-local accumulation buffers with explicit initialization
        double* local_pheromon_add = static_cast<double*>(calloc(TOTAL_CELLS, sizeof(double)));
        int* local_kol_enter_add = static_cast<int*>(calloc(TOTAL_CELLS, sizeof(int)));

        // Ensure initialization to zero
        if (local_pheromon_add && local_kol_enter_add) {
            for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
                local_pheromon_add[idx] = 0.0;
                local_kol_enter_add[idx] = 0;
            }

#pragma omp for nowait
            for (int i = 0; i < ANT_SIZE; ++i) {
                double agent_of = OF[i];
#if OPTIMIZE_MIN_2
                double agent_of_reciprocal = (agent_of == 0) ? (PARAMETR_Q / 0.0000001) : (PARAMETR_Q / agent_of);
#elif OPTIMIZE_MAX
                double agent_of_scaled = PARAMETR_Q * agent_of;
#endif

                for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                    int k = agent_node[i * PARAMETR_SIZE + tx];
                    int idx = MAX_VALUE_SIZE * tx + k;

                    local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_1
                    double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of;
                    if (delta > 0) {
                        local_pheromon_add[idx] += PARAMETR_Q * delta;
                    }
#elif OPTIMIZE_MIN_2
                    local_pheromon_add[idx] += agent_of_reciprocal;
#elif OPTIMIZE_MAX
                    local_pheromon_add[idx] += agent_of_scaled;
#endif
                }
            }

            // Merge results
#pragma omp critical
            {
                for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
                    kol_enter[idx] += local_kol_enter_add[idx];
                    pheromon[idx] += local_pheromon_add[idx];
                }
            }
        }

        if (local_pheromon_add) free(local_pheromon_add);
        if (local_kol_enter_add) free(local_kol_enter_add);
    }
}
#endif
void add_pheromon_iteration_omp(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
#if _OPENMP >= 202111  // OpenMP 5.2+
    add_pheromon_iteration_omp_5_2(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 202011  // OpenMP 5.1+
    add_pheromon_iteration_omp_5_1(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 201811  // OpenMP 5.0+
    add_pheromon_iteration_omp_5_0(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 201511  // OpenMP 4.5+
    add_pheromon_iteration_omp_4_5(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 201307  // OpenMP 4.0+
    add_pheromon_iteration_omp_4_0(pheromon, kol_enter, agent_node, OF);
#else  // OpenMP 2.0/3.0/3.1
    add_pheromon_iteration_omp_2_0(pheromon, kol_enter, agent_node, OF);
#endif
}
void add_pheromon_iteration_non_cuda(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // Испарение: последовательный доступ
    for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
        pheromon[idx] *= PARAMETR_RO;
    }

    // Накопление: предварительные вычисления и кэш-дружественный доступ
    for (int i = 0; i < ANT_SIZE; ++i) {
        const double agent_of = OF[i];

        // Предварительные вычисления вне внутреннего цикла
#if OPTIMIZE_MIN_1
        const double min1_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of > 0) ?
            PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
        const double min2_value = PARAMETR_Q / ((agent_of == 0) ? 0.0000001 : agent_of);
#elif OPTIMIZE_MAX
        const double max_value = PARAMETR_Q * agent_of;
#endif

        const int* agent_path = &agent_node[i * PARAMETR_SIZE];

        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            int k = agent_path[tx];
            if (static_cast<unsigned>(k) < static_cast<unsigned>(MAX_VALUE_SIZE)) {
                int idx = MAX_VALUE_SIZE * tx + k;
                kol_enter[idx]++;

#if OPTIMIZE_MIN_1
                if (min1_value > 0) {
                    pheromon[idx] += min1_value;
                }
#elif OPTIMIZE_MIN_2
                pheromon[idx] += min2_value;
#elif OPTIMIZE_MAX
                pheromon[idx] += max_value;
#endif
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

// Подготовка массива для вероятностного поиска (транспонированная версия)
void go_mass_probability_transp_non_cuda(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    // Обрабатываем каждый параметр отдельно
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVectorT = 0.0;
        // Суммируем значения феромонов для текущего параметра
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVectorT += pheromon[tx + i * PARAMETR_SIZE];
        }
        // Нормализуем значения феромонов
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
        if (sumVectorT != 0.0) {
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                pheromon_norm[i] = pheromon[tx + i * PARAMETR_SIZE] / sumVectorT;
            }
        }
        // Вычисляем svertka и их сумму
        double sumVectorZ = 0.0;
        double svertka[MAX_VALUE_SIZE] = { 0 };
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            if (kol_enter[tx + i * PARAMETR_SIZE] != 0.0 && pheromon_norm[i] != 0.0) {
                svertka[i] = 1.0 / kol_enter[tx + i * PARAMETR_SIZE] + pheromon_norm[i];
            }
            else {
                svertka[i] = 0.0;
            }
            sumVectorZ += svertka[i];
        }

        // Вычисляем кумулятивные вероятности
        if (sumVectorZ != 0.0) {
            norm_matrix_probability[tx + 0 * PARAMETR_SIZE] = svertka[0] / sumVectorZ;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[tx + i * PARAMETR_SIZE] = (svertka[i] / sumVectorZ) + norm_matrix_probability[tx + (i - 1) * PARAMETR_SIZE];
            }
        }
        else {
            // Если сумма нулевая, устанавливаем равномерное распределение
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[tx + 0 * PARAMETR_SIZE] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[tx + i * PARAMETR_SIZE] = uniform_prob + norm_matrix_probability[tx + (i - 1) * PARAMETR_SIZE];
            }
        }
    }
}
// Базовая версия OpenMP 2.0/3.0/3.1
void go_mass_probability_transp_OMP_non_cuda_2_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVectorT = 0.0;

        // Суммируем значения феромонов для текущего параметра
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVectorT += pheromon[tx + i * PARAMETR_SIZE];
        }

        // Нормализуем значения феромонов
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
        if (sumVectorT != 0.0) {
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                pheromon_norm[i] = pheromon[tx + i * PARAMETR_SIZE] / sumVectorT;
            }
        }

        // Вычисляем svertka и их сумму
        double sumVectorZ = 0.0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            if (kol_enter[tx + i * PARAMETR_SIZE] != 0.0 && pheromon_norm[i] != 0.0) {
                svertka[i] = 1.0 / kol_enter[tx + i * PARAMETR_SIZE] + pheromon_norm[i];
            }
            else {
                svertka[i] = 0.0;
            }
            sumVectorZ += svertka[i];
        }

        // Вычисляем кумулятивные вероятности
        if (sumVectorZ != 0.0) {
            norm_matrix_probability[tx + 0 * PARAMETR_SIZE] = svertka[0] / sumVectorZ;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[tx + i * PARAMETR_SIZE] =
                    (svertka[i] / sumVectorZ) + norm_matrix_probability[tx + (i - 1) * PARAMETR_SIZE];
            }
        }
        else {
            // Если сумма нулевая, устанавливаем равномерное распределение
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[tx + 0 * PARAMETR_SIZE] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[tx + i * PARAMETR_SIZE] =
                    uniform_prob + norm_matrix_probability[tx + (i - 1) * PARAMETR_SIZE];
            }
        }
    }
}
#if _OPENMP >= 201307  // OpenMP 4.0+
void go_mass_probability_transp_OMP_non_cuda_4_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    // OpenMP 4.0: separate simd directive
#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVectorT = 0.0;

        // Суммируем значения феромонов с SIMD
#pragma omp simd reduction(+:sumVectorT)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVectorT += pheromon[tx + i * PARAMETR_SIZE];
        }

        // Нормализуем значения феромонов
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
        if (sumVectorT != 0.0) {
#pragma omp simd
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                pheromon_norm[i] = pheromon[tx + i * PARAMETR_SIZE] / sumVectorT;
            }
        }

        // Вычисляем svertka и их сумму
        double sumVectorZ = 0.0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

#pragma omp simd reduction(+:sumVectorZ)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            if (kol_enter[tx + i * PARAMETR_SIZE] != 0.0 && pheromon_norm[i] != 0.0) {
                svertka[i] = 1.0 / kol_enter[tx + i * PARAMETR_SIZE] + pheromon_norm[i];
            }
            else {
                svertka[i] = 0.0;
            }
            sumVectorZ += svertka[i];
        }

        // Вычисляем кумулятивные вероятности
        if (sumVectorZ != 0.0) {
            norm_matrix_probability[tx + 0 * PARAMETR_SIZE] = svertka[0] / sumVectorZ;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[tx + i * PARAMETR_SIZE] =
                    (svertka[i] / sumVectorZ) + norm_matrix_probability[tx + (i - 1) * PARAMETR_SIZE];
            }
        }
        else {
            // Если сумма нулевая, устанавливаем равномерное распределение
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[tx + 0 * PARAMETR_SIZE] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[tx + i * PARAMETR_SIZE] =
                    uniform_prob + norm_matrix_probability[tx + (i - 1) * PARAMETR_SIZE];
            }
        }
    }
}
#endif
#if _OPENMP >= 201511  // OpenMP 4.5+
void go_mass_probability_transp_OMP_non_cuda_4_5(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    // OpenMP 4.5: if clause для условного выполнения
#if defined(__clang__)
#pragma omp parallel for
#else
#pragma omp parallel for // if(PARAMETR_SIZE > 100)
#endif
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVectorT = 0.0;

        // Автоматическая векторизация внутренних циклов
#pragma omp simd reduction(+:sumVectorT)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVectorT += pheromon[tx + i * PARAMETR_SIZE];
        }

        // Нормализуем значения феромонов
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
        if (sumVectorT != 0.0) {
#pragma omp simd
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                pheromon_norm[i] = pheromon[tx + i * PARAMETR_SIZE] / sumVectorT;
            }
        }

        // Вычисляем svertka и их сумму
        double sumVectorZ = 0.0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

#pragma omp simd reduction(+:sumVectorZ)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            if (kol_enter[tx + i * PARAMETR_SIZE] != 0.0 && pheromon_norm[i] != 0.0) {
                svertka[i] = 1.0 / kol_enter[tx + i * PARAMETR_SIZE] + pheromon_norm[i];
            }
            else {
                svertka[i] = 0.0;
            }
            sumVectorZ += svertka[i];
        }

        // Вычисляем кумулятивные вероятности
        if (sumVectorZ != 0.0) {
            norm_matrix_probability[tx + 0 * PARAMETR_SIZE] = svertka[0] / sumVectorZ;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[tx + i * PARAMETR_SIZE] =
                    (svertka[i] / sumVectorZ) + norm_matrix_probability[tx + (i - 1) * PARAMETR_SIZE];
            }
        }
        else {
            // Если сумма нулевая, устанавливаем равномерное распределение
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[tx + 0 * PARAMETR_SIZE] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[tx + i * PARAMETR_SIZE] =
                    uniform_prob + norm_matrix_probability[tx + (i - 1) * PARAMETR_SIZE];
            }
        }
    }
}
#endif
#if _OPENMP >= 201811  // OpenMP 5.0+
void go_mass_probability_transp_OMP_non_cuda_5_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    // OpenMP 5.0: tile directive для оптимизации доступа к памяти
#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVectorT = 0.0;

        // OpenMP 5.0: scan directive для редукций (если поддерживается)
#ifdef __clang__
#pragma omp simd reduction(+:sumVectorT)
#else
#pragma omp simd reduction(inscan,+:sumVectorT)
#endif
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVectorT += pheromon[tx + i * PARAMETR_SIZE];
#ifndef __clang__
#pragma omp scan inclusive(sumVectorT)
#endif
        }

        // Нормализуем значения феромонов
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
        if (sumVectorT != 0.0) {
#pragma omp simd
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                pheromon_norm[i] = pheromon[tx + i * PARAMETR_SIZE] / sumVectorT;
            }
        }

        // Вычисляем svertka и их сумму
        double sumVectorZ = 0.0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

#ifdef __clang__
#pragma omp simd reduction(+:sumVectorZ)
#else
#pragma omp simd reduction(inscan,+:sumVectorZ)
#endif
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            if (kol_enter[tx + i * PARAMETR_SIZE] != 0.0 && pheromon_norm[i] != 0.0) {
                svertka[i] = 1.0 / kol_enter[tx + i * PARAMETR_SIZE] + pheromon_norm[i];
            }
            else {
                svertka[i] = 0.0;
            }
            sumVectorZ += svertka[i];
#ifndef __clang__
#pragma omp scan inclusive(sumVectorZ)
#endif
        }

        // Вычисляем кумулятивные вероятности
        if (sumVectorZ != 0.0) {
            norm_matrix_probability[tx + 0 * PARAMETR_SIZE] = svertka[0] / sumVectorZ;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[tx + i * PARAMETR_SIZE] =
                    (svertka[i] / sumVectorZ) + norm_matrix_probability[tx + (i - 1) * PARAMETR_SIZE];
            }
        }
        else {
            // Если сумма нулевая, устанавливаем равномерное распределение
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[tx + 0 * PARAMETR_SIZE] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[tx + i * PARAMETR_SIZE] =
                    uniform_prob + norm_matrix_probability[tx + (i - 1) * PARAMETR_SIZE];
            }
        }
    }
}
#endif
#if _OPENMP >= 202011  // OpenMP 5.1+
void go_mass_probability_transp_OMP_non_cuda_5_1(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    // OpenMP 5.1: error recovery and loop features
#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVectorT = 0.0;

        // OpenMP 5.1: неблокирующие редукции
#pragma omp simd reduction(+:sumVectorT)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVectorT += pheromon[tx + i * PARAMETR_SIZE];
        }

        // Нормализуем значения феромонов
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
        if (sumVectorT != 0.0) {
#pragma omp simd
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                pheromon_norm[i] = pheromon[tx + i * PARAMETR_SIZE] / sumVectorT;
            }
        }

        // Вычисляем svertka и их сумму
        double sumVectorZ = 0.0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

#pragma omp simd reduction(+:sumVectorZ)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            if (kol_enter[tx + i * PARAMETR_SIZE] != 0.0 && pheromon_norm[i] != 0.0) {
                svertka[i] = 1.0 / kol_enter[tx + i * PARAMETR_SIZE] + pheromon_norm[i];
            }
            else {
                svertka[i] = 0.0;
            }
            sumVectorZ += svertka[i];
        }

        // Вычисляем кумулятивные вероятности
        if (sumVectorZ != 0.0) {
            norm_matrix_probability[tx + 0 * PARAMETR_SIZE] = svertka[0] / sumVectorZ;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[tx + i * PARAMETR_SIZE] =
                    (svertka[i] / sumVectorZ) + norm_matrix_probability[tx + (i - 1) * PARAMETR_SIZE];
            }
        }
        else {
            // Если сумма нулевая, устанавливаем равномерное распределение
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[tx + 0 * PARAMETR_SIZE] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[tx + i * PARAMETR_SIZE] =
                    uniform_prob + norm_matrix_probability[tx + (i - 1) * PARAMETR_SIZE];
            }
        }
    }
}
#endif
#if _OPENMP >= 202111  // OpenMP 5.2+
void go_mass_probability_transp_OMP_non_cuda_5_2(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    // OpenMP 5.2: assume clauses для оптимизатора
#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVectorT = 0.0;

        // OpenMP 5.2: улучшенные редукции
#pragma omp simd reduction(+:sumVectorT)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVectorT += pheromon[tx + i * PARAMETR_SIZE];
        }

        // Нормализуем значения феромонов
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
        if (sumVectorT != 0.0) {
#pragma omp simd
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                pheromon_norm[i] = pheromon[tx + i * PARAMETR_SIZE] / sumVectorT;
            }
        }

        // Вычисляем svertka и их сумму
        double sumVectorZ = 0.0;
        double svertka[MAX_VALUE_SIZE] = { 0 };

#pragma omp simd reduction(+:sumVectorZ)
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            if (kol_enter[tx + i * PARAMETR_SIZE] != 0.0 && pheromon_norm[i] != 0.0) {
                svertka[i] = 1.0 / kol_enter[tx + i * PARAMETR_SIZE] + pheromon_norm[i];
            }
            else {
                svertka[i] = 0.0;
            }
            sumVectorZ += svertka[i];
        }

        // Вычисляем кумулятивные вероятности
        if (sumVectorZ != 0.0) {
            norm_matrix_probability[tx + 0 * PARAMETR_SIZE] = svertka[0] / sumVectorZ;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[tx + i * PARAMETR_SIZE] =
                    (svertka[i] / sumVectorZ) + norm_matrix_probability[tx + (i - 1) * PARAMETR_SIZE];
            }
        }
        else {
            // Если сумма нулевая, устанавливаем равномерное распределение
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            norm_matrix_probability[tx + 0 * PARAMETR_SIZE] = uniform_prob;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[tx + i * PARAMETR_SIZE] =
                    uniform_prob + norm_matrix_probability[tx + (i - 1) * PARAMETR_SIZE];
            }
        }
    }
}
#endif
void go_mass_probability_transp_OMP_non_cuda(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
#if _OPENMP >= 202111  // OpenMP 5.2+
    go_mass_probability_transp_OMP_non_cuda_5_2(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 202011  // OpenMP 5.1+
    go_mass_probability_transp_OMP_non_cuda_5_1(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201811  // OpenMP 5.0+
    go_mass_probability_transp_OMP_non_cuda_5_0(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201511  // OpenMP 4.5+
    go_mass_probability_transp_OMP_non_cuda_4_5(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201307  // OpenMP 4.0+
    go_mass_probability_transp_OMP_non_cuda_4_0(pheromon, kol_enter, norm_matrix_probability);
#else  // OpenMP 2.0/3.0/3.1
    go_mass_probability_transp_OMP_non_cuda_2_0(pheromon, kol_enter, norm_matrix_probability);
#endif
}
// Функция для вычисления пути агентов на CPU
void go_all_agent_transp_non_cuda_time(int gpuTime, double* __restrict parametr, double* __restrict norm_matrix_probability, double* __restrict agent, int* __restrict agent_node, double* __restrict OF, HashEntry* __restrict hashTable, int& kol_hash_fail, double& totalHashTime, double& totalOFTime, double& HashTimeSave, double& HashTimeSearch, double& SumTimeSearch) {
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
        double cachedResult = getCachedResultOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx);
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
            saveToCacheOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx, OF[bx]);
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
                    cachedResult = getCachedResultOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx);
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
                saveToCacheOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx, OF[bx]);
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
void go_all_agent_transp_non_cuda(int gpuTime, double* __restrict parametr, double* __restrict norm_matrix_probability, double* __restrict agent, int* __restrict agent_node, double* __restrict OF, HashEntry* __restrict hashTable, int& kol_hash_fail) {
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
        double cachedResult = getCachedResultOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx);
        int nom_iteration = 0;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
            saveToCacheOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx, OF[bx]);
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
                    cachedResult = getCachedResultOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx);
                    nom_iteration = nom_iteration + 1;
                    kol_hash_fail = kol_hash_fail + 1;
                }
                OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
                saveToCacheOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx, OF[bx]);
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
void go_all_agent_transp_non_cuda_non_hash(int gpuTime, double* __restrict parametr, double* __restrict norm_matrix_probability, double* __restrict agent, int* __restrict agent_node, double* __restrict OF, int& kol_hash_fail, double& totalOFTime) {
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
void go_all_agent_transp_non_cuda_non_hash_OMP_optimized(int gpuTime, double* __restrict parametr, double* __restrict norm_matrix_probability, double* __restrict agent, int* __restrict agent_node, double* __restrict OF, int& kol_hash_fail, double& totalOFTime) {
    //printf("Using OpenMP 2.0-3.1 version (CPU parallel for)\n");
    std::default_random_engine generator(123 + gpuTime);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double local_totalOFTime = 0.0;

    // Генерируем все случайные числа заранее
    std::vector<double> randomValues(ANT_SIZE * PARAMETR_SIZE);
    for (int i = 0; i < ANT_SIZE * PARAMETR_SIZE; i++) {
        randomValues[i] = distribution(generator);
    }

    // Универсальная параллельная секция с условными директивами
#if _OPENMP >= 201511 && !defined(__clang__)
#pragma omp parallel reduction(+:local_totalOFTime) // if(ANT_SIZE > 100)
#else
#pragma omp parallel reduction(+:local_totalOFTime)
#endif
    {

        // Условное распределение работы в зависимости от версии OpenMP
#if defined(__clang__)
// Clang - только базовые возможности
#pragma omp for schedule(static)
#else
// Другие компиляторы - условные возможности  
#if _OPENMP >= 201511
#pragma omp for schedule(dynamic, 16)
#elif _OPENMP >= 201307
#pragma omp for schedule(guided)  
#else
#pragma omp for schedule(static)
#endif
#endif
        for (int bx = 0; bx < ANT_SIZE; bx++) {
            double* current_agent = &agent[bx * PARAMETR_SIZE];
            int* current_agent_node = &agent_node[bx * PARAMETR_SIZE];
            double* current_random = &randomValues[bx * PARAMETR_SIZE];

            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                double randomValue = current_random[tx];
                int k = 0;
                // Оптимизированный линейный поиск с предсказанием ветвления
                for (; k < MAX_VALUE_SIZE - 4; k += 4) {
                    if (randomValue <= norm_matrix_probability[tx + k * PARAMETR_SIZE]) break;
                    if (randomValue <= norm_matrix_probability[tx + (k + 1) * PARAMETR_SIZE]) { k += 1; break; }
                    if (randomValue <= norm_matrix_probability[tx + (k + 2) * PARAMETR_SIZE]) { k += 2; break; }
                    if (randomValue <= norm_matrix_probability[tx + (k + 3) * PARAMETR_SIZE]) { k += 3; break; }
                }
                // Обработка оставшихся элементов
                for (; k < MAX_VALUE_SIZE; k++) {
                    if (randomValue <= norm_matrix_probability[tx + k * PARAMETR_SIZE]) {
                        break;
                    }
                }

                if (k >= MAX_VALUE_SIZE) k = MAX_VALUE_SIZE - 1;

                current_agent_node[tx] = k;
                current_agent[tx] = parametr[tx + k * PARAMETR_SIZE];
            }
            auto start_OF = std::chrono::high_resolution_clock::now();
            OF[bx] = BenchShafferaFunction_non_cuda(current_agent);
            auto end_OF = std::chrono::high_resolution_clock::now();
            local_totalOFTime += std::chrono::duration<double, std::milli>(end_OF - start_OF).count();
        }
    }
    totalOFTime += local_totalOFTime;
}
void go_all_agent_transp_OMP_non_cuda_time(int gpuTime, double* __restrict parametr, double* __restrict norm_matrix_probability, double* __restrict agent, int* __restrict agent_node, double* __restrict OF, HashEntry* __restrict hashTable, int& kol_hash_fail, double& totalHashTime, double& totalOFTime, double& HashTimeSave, double& HashTimeSearch, double& SumTimeSearch) {
    int local_kol_hash_fail = 0;
    double local_totalHashTime = 0.0;
    double local_totalOFTime = 0.0;
    double local_HashTimeSave = 0.0;
    double local_HashTimeSearch = 0.0;
    double local_SumTimeSearch = 0.0;

    // Универсальная параллельная секция с условными директивами
#if _OPENMP >= 201511 && !defined(__clang__)
#pragma omp parallel reduction(+:local_kol_hash_fail, local_totalHashTime, local_totalOFTime, local_HashTimeSave, local_HashTimeSearch, local_SumTimeSearch) // if(ANT_SIZE > 100)
#else
#pragma omp parallel reduction(+:local_kol_hash_fail, local_totalHashTime, local_totalOFTime, local_HashTimeSave, local_HashTimeSearch, local_SumTimeSearch)
#endif
    {
        uint64_t seed = 123 + gpuTime + omp_get_thread_num();

        // Условное распределение работы в зависимости от версии OpenMP
#if defined(__clang__)
// Clang - только базовые возможности
#pragma omp for schedule(static)
#else
// Другие компиляторы - условные возможности  
#if _OPENMP >= 201511
#pragma omp for schedule(dynamic, 16)
#elif _OPENMP >= 201307
#pragma omp for schedule(guided)  
#else
#pragma omp for schedule(static)
#endif
#endif
        for (int bx = 0; bx < ANT_SIZE; bx++) {
            auto start_ant = std::chrono::high_resolution_clock::now();
            // Генерация агента
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                double randomValue = unified_fast_random(seed);
                int k = 0;
                while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[tx + k * PARAMETR_SIZE]) {
                    k++;
                }
                agent_node[bx * PARAMETR_SIZE + tx] = k;
                agent[bx * PARAMETR_SIZE + tx] = parametr[tx + k * PARAMETR_SIZE];
            }

            auto end_ant = std::chrono::high_resolution_clock::now();
            local_SumTimeSearch += std::chrono::duration<double, std::milli>(end_ant - start_ant).count();
            auto start = std::chrono::high_resolution_clock::now();
            double cachedResult = -1.0;

#pragma omp critical(hash_table_read)
            {
                cachedResult = getCachedResultOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx);
            }

            auto end_OF = std::chrono::high_resolution_clock::now();
            local_HashTimeSearch += std::chrono::duration<double, std::milli>(end_OF - start).count();

            int nom_iteration = 0;
            if (cachedResult == -1.0) {
                auto start_OF = std::chrono::high_resolution_clock::now();
                OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
                auto end_OF = std::chrono::high_resolution_clock::now();
                local_totalOFTime += std::chrono::duration<double, std::milli>(end_OF - start_OF).count();

                auto start_SaveHash = std::chrono::high_resolution_clock::now();
#pragma omp critical(hash_table_write)
                {
                    saveToCacheOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx, OF[bx]);
                }
                auto end_SaveHash = std::chrono::high_resolution_clock::now();
                local_HashTimeSave += std::chrono::duration<double, std::milli>(end_SaveHash - start_SaveHash).count();
            }
            else {
                auto start_OF_2 = std::chrono::high_resolution_clock::now();
                auto end_OF_2 = std::chrono::high_resolution_clock::now();

                switch (TYPE_ACO) {
                case 0: // ACOCN
                    OF[bx] = cachedResult;
                    local_kol_hash_fail++;
                    break;
                case 1: // ACOCNI
                    OF[bx] = ZERO_HASH_RESULT;
                    local_kol_hash_fail++;
                    break;
                case 2: // ACOCCyN
                    while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION)) {
                        start_OF_2 = std::chrono::high_resolution_clock::now();
                        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                            double randomValue = unified_fast_random(seed);
                            int k = 0;
                            while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[tx + k * PARAMETR_SIZE]) {
                                k++;
                            }
                            agent_node[bx * PARAMETR_SIZE + tx] = k;
                            agent[bx * PARAMETR_SIZE + tx] = parametr[tx + k * PARAMETR_SIZE];
                        }

                        end_OF_2 = std::chrono::high_resolution_clock::now();
                        local_SumTimeSearch += std::chrono::duration<double, std::milli>(end_OF_2 - start_OF_2).count();
                        start_OF_2 = std::chrono::high_resolution_clock::now();
#pragma omp critical(hash_table_read)
                        {
                            cachedResult = getCachedResultOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx);
                        }
                        end_OF_2 = std::chrono::high_resolution_clock::now();
                        local_HashTimeSearch += std::chrono::duration<double, std::milli>(end_OF_2 - start_OF_2).count();
                        nom_iteration++;
                        local_kol_hash_fail++;
                    }

                    start_OF_2 = std::chrono::high_resolution_clock::now();
                    OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
                    end_OF_2 = std::chrono::high_resolution_clock::now();
                    local_totalOFTime += std::chrono::duration<double, std::milli>(end_OF_2 - start_OF_2).count();

                    start_OF_2 = std::chrono::high_resolution_clock::now();
#pragma omp critical(hash_table_write)
                    {
                        saveToCacheOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx, OF[bx]);
                    }
                    end_OF_2 = std::chrono::high_resolution_clock::now();
                    local_HashTimeSave += std::chrono::duration<double, std::milli>(end_OF_2 - start_OF_2).count();
                    break;
                default:
                    OF[bx] = cachedResult;
                    local_kol_hash_fail++;
                    break;
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            local_totalHashTime += std::chrono::duration<double, std::milli>(end - start).count();
        }
    }

    kol_hash_fail += local_kol_hash_fail;
    totalHashTime += local_totalHashTime;
    totalOFTime += local_totalOFTime;
    HashTimeSave += local_HashTimeSave;
    HashTimeSearch += local_HashTimeSearch;
    SumTimeSearch += local_SumTimeSearch;
}
// Обновление слоев графа
void add_pheromon_iteration_transp_non_cuda(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;
    // Испарение весов-феромона
    for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
        pheromon[idx] *= PARAMETR_RO;
    }

    // Накопление с учетом транспонированного хранения
    for (int i = 0; i < ANT_SIZE; ++i) {
        const double agent_of = OF[i];

        // Предварительные вычисления вне внутреннего цикла
#if OPTIMIZE_MIN_1
        const double min1_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of > 0) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
        const double min2_value = PARAMETR_Q / ((agent_of == 0) ? 0.0000001 : agent_of);
#elif OPTIMIZE_MAX
        const double max_value = PARAMETR_Q * agent_of;
#endif

        const int* agent_path = &agent_node[i * PARAMETR_SIZE];

        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            int k = agent_path[tx];
            if (k >= 0 && k < MAX_VALUE_SIZE) {
                // Транспонированный индекс: tx + k * PARAMETR_SIZE
                int idx = tx + k * PARAMETR_SIZE;
                kol_enter[idx]++;

#if OPTIMIZE_MIN_1
                if (min1_value > 0) {
                    pheromon[idx] += min1_value;
                }
#elif OPTIMIZE_MIN_2
                pheromon[idx] += min2_value;
#elif OPTIMIZE_MAX
                pheromon[idx] += max_value;
#endif
            }
        }
    }
}
// Базовая версия OpenMP 2.0/3.0/3.1
void add_pheromon_iteration_transp_OMP_non_cuda_2_0(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // Испарение феромона
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
        pheromon[idx] *= PARAMETR_RO;
    }

    // Накопление с thread-local буферами
#pragma omp parallel
    {
        double* local_pheromon_add = (double*)calloc(TOTAL_CELLS, sizeof(double));
        int* local_kol_enter_add = (int*)calloc(TOTAL_CELLS, sizeof(int));

#pragma omp for nowait
        for (int i = 0; i < ANT_SIZE; ++i) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            const double min1_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of > 0) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            const double min2_value = PARAMETR_Q / ((agent_of == 0) ? 0.0000001 : agent_of);
#elif OPTIMIZE_MAX
            const double max_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];
            for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                int k = agent_path[tx];
                if (static_cast<unsigned>(k) < static_cast<unsigned>(MAX_VALUE_SIZE)) {
                    int idx = tx + k * PARAMETR_SIZE;
                    local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_1
                    if (min1_value > 0) {
                        local_pheromon_add[idx] += min1_value;
                    }
#elif OPTIMIZE_MIN_2
                    local_pheromon_add[idx] += min2_value;
#elif OPTIMIZE_MAX
                    local_pheromon_add[idx] += max_value;
#endif
                }
            }
        }

#pragma omp critical
        {
            for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
                kol_enter[idx] += local_kol_enter_add[idx];
                pheromon[idx] += local_pheromon_add[idx];
            }
        }

        free(local_pheromon_add);
        free(local_kol_enter_add);
    }
}
#if _OPENMP >= 201307  // OpenMP 4.0+
void add_pheromon_iteration_transp_OMP_non_cuda_4_0(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // Испарение феромона с SIMD
#pragma omp parallel for simd schedule(static)
    for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
        pheromon[idx] *= PARAMETR_RO;
    }

    // Накопление с оптимизированными thread-local буферами
#pragma omp parallel
    {
        double* local_pheromon_add = (double*)calloc(TOTAL_CELLS, sizeof(double));
        int* local_kol_enter_add = (int*)calloc(TOTAL_CELLS, sizeof(int));

#pragma omp for nowait
        for (int i = 0; i < ANT_SIZE; ++i) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            const double min1_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of > 0) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            const double min2_value = PARAMETR_Q / ((agent_of == 0) ? 0.0000001 : agent_of);
#elif OPTIMIZE_MAX
            const double max_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];
            for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                int k = agent_path[tx];
                if (static_cast<unsigned>(k) < static_cast<unsigned>(MAX_VALUE_SIZE)) {
                    int idx = tx + k * PARAMETR_SIZE;
                    local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_1
                    if (min1_value > 0) {
                        local_pheromon_add[idx] += min1_value;
                    }
#elif OPTIMIZE_MIN_2
                    local_pheromon_add[idx] += min2_value;
#elif OPTIMIZE_MAX
                    local_pheromon_add[idx] += max_value;
#endif
                }
            }
        }

        // Слияние с SIMD
#pragma omp critical
        {
#pragma omp simd
            for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
                kol_enter[idx] += local_kol_enter_add[idx];
                pheromon[idx] += local_pheromon_add[idx];
            }
        }

        free(local_pheromon_add);
        free(local_kol_enter_add);
    }
}
#endif
#if _OPENMP >= 201511  // OpenMP 4.5+
void add_pheromon_iteration_transp_OMP_non_cuda_4_5(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // Испарение феромона с if clause
#ifdef __clang__
#pragma omp parallel for simd schedule(static)
#else
#pragma omp parallel for simd schedule(static) // if(TOTAL_CELLS > 1000)
#endif
    for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
        pheromon[idx] *= PARAMETR_RO;
    }

    // Накопление с улучшенным планированием
#pragma omp parallel
    {
        double* local_pheromon_add = (double*)calloc(TOTAL_CELLS, sizeof(double));
        int* local_kol_enter_add = (int*)calloc(TOTAL_CELLS, sizeof(int));

        // OpenMP 4.5: улучшенное планирование
#pragma omp for schedule(static) nowait
        for (int i = 0; i < ANT_SIZE; ++i) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            const double min1_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of > 0) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            const double min2_value = PARAMETR_Q / ((agent_of == 0) ? 0.0000001 : agent_of);
#elif OPTIMIZE_MAX
            const double max_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];
            for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                int k = agent_path[tx];
                if (static_cast<unsigned>(k) < static_cast<unsigned>(MAX_VALUE_SIZE)) {
                    int idx = tx + k * PARAMETR_SIZE;
                    local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_1
                    if (min1_value > 0) {
                        local_pheromon_add[idx] += min1_value;
                    }
#elif OPTIMIZE_MIN_2
                    local_pheromon_add[idx] += min2_value;
#elif OPTIMIZE_MAX
                    local_pheromon_add[idx] += max_value;
#endif
                }
            }
        }

        // Эффективное слияние
#pragma omp critical
        {
            for (int k = 0; k < MAX_VALUE_SIZE; ++k) {
#pragma omp simd
                for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                    int idx = tx + k * PARAMETR_SIZE;
                    kol_enter[idx] += local_kol_enter_add[idx];
                    pheromon[idx] += local_pheromon_add[idx];
                }
            }
        }

        free(local_pheromon_add);
        free(local_kol_enter_add);
    }
}
#endif
#if _OPENMP >= 201811  // OpenMP 5.0+
void add_pheromon_iteration_transp_OMP_non_cuda_5_0(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // Испарение феромона с loop трансформацией
#pragma omp parallel for simd schedule(static)
    for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
        pheromon[idx] *= PARAMETR_RO;
    }

    // Накопление с nonmonotonic scheduling
#pragma omp parallel
    {
        double* local_pheromon_add = (double*)calloc(TOTAL_CELLS, sizeof(double));
        int* local_kol_enter_add = (int*)calloc(TOTAL_CELLS, sizeof(int));

        // OpenMP 5.0: nonmonotonic scheduling
#pragma omp for //schedule(nonmonotonic:static) nowait
        for (int i = 0; i < ANT_SIZE; ++i) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            const double min1_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of > 0) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            const double min2_value = PARAMETR_Q / ((agent_of == 0) ? 0.0000001 : agent_of);
#elif OPTIMIZE_MAX
            const double max_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];
            for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                int k = agent_path[tx];
                if (static_cast<unsigned>(k) < static_cast<unsigned>(MAX_VALUE_SIZE)) {
                    int idx = tx + k * PARAMETR_SIZE;
                    local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_1
                    if (min1_value > 0) {
                        local_pheromon_add[idx] += min1_value;
                    }
#elif OPTIMIZE_MIN_2
                    local_pheromon_add[idx] += min2_value;
#elif OPTIMIZE_MAX
                    local_pheromon_add[idx] += max_value;
#endif
                }
            }
        }

        // Слияние с улучшенной векторизацией
#pragma omp critical
        {
#ifdef __clang__
            for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
                kol_enter[idx] += local_kol_enter_add[idx];
                pheromon[idx] += local_pheromon_add[idx];
            }
#else
#pragma omp simd
            for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
                kol_enter[idx] += local_kol_enter_add[idx];
                pheromon[idx] += local_pheromon_add[idx];
            }
#endif
        }

        free(local_pheromon_add);
        free(local_kol_enter_add);
    }
}
#endif
#if _OPENMP >= 202011  // OpenMP 5.1+
void add_pheromon_iteration_transp_OMP_non_cuda_5_1(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // Испарение феромона с неблокирующими операциями
#pragma omp parallel for simd schedule(static)
    for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
        pheromon[idx] *= PARAMETR_RO;
    }

    // Накопление с error recovery features
#pragma omp parallel
    {
        double* local_pheromon_add = (double*)calloc(TOTAL_CELLS, sizeof(double));
        int* local_kol_enter_add = (int*)calloc(TOTAL_CELLS, sizeof(int));

        // OpenMP 5.1: улучшенное планирование
#pragma omp for //schedule(nonmonotonic:static) nowait
        for (int i = 0; i < ANT_SIZE; ++i) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            const double min1_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of > 0) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            const double min2_value = PARAMETR_Q / ((agent_of == 0) ? 0.0000001 : agent_of);
#elif OPTIMIZE_MAX
            const double max_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];
            for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                int k = agent_path[tx];
                if (static_cast<unsigned>(k) < static_cast<unsigned>(MAX_VALUE_SIZE)) {
                    int idx = tx + k * PARAMETR_SIZE;
                    local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_1
                    if (min1_value > 0) {
                        local_pheromon_add[idx] += min1_value;
                    }
#elif OPTIMIZE_MIN_2
                    local_pheromon_add[idx] += min2_value;
#elif OPTIMIZE_MAX
                    local_pheromon_add[idx] += max_value;
#endif
                }
            }
        }

        // Эффективное слияние с выравниванием памяти
#pragma omp critical
        {
#pragma omp simd aligned(pheromon, kol_enter, local_pheromon_add, local_kol_enter_add:64)
            for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
                kol_enter[idx] += local_kol_enter_add[idx];
                pheromon[idx] += local_pheromon_add[idx];
            }
        }

        free(local_pheromon_add);
        free(local_kol_enter_add);
    }
}
#endif
#if _OPENMP >= 202111  // OpenMP 5.2+
void add_pheromon_iteration_transp_OMP_non_cuda_5_2(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int TOTAL_CELLS = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // Испарение феромона с assume clauses
#pragma omp parallel for simd schedule(static)
    for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
        pheromon[idx] *= PARAMETR_RO;
    }

    // Накопление с latest OpenMP 5.2 features
#pragma omp parallel
    {
        // OpenMP 5.2: aligned allocation для лучшей векторизации
        double* local_pheromon_add = (double*)ALIGNED_ALLOC(64, TOTAL_CELLS * sizeof(double));
        int* local_kol_enter_add = (int*)ALIGNED_ALLOC(64, TOTAL_CELLS * sizeof(int));

        // Инициализация буферов
        for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
            local_pheromon_add[idx] = 0.0;
            local_kol_enter_add[idx] = 0;
        }

        // OpenMP 5.2: assume clauses для оптимизатора
#pragma omp for schedule(static) nowait
        for (int i = 0; i < ANT_SIZE; ++i) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            const double min1_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of > 0) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            const double min2_value = PARAMETR_Q / ((agent_of == 0) ? 0.0000001 : agent_of);
#elif OPTIMIZE_MAX
            const double max_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];

            // OpenMP 5.2: assume для лучшей оптимизации
#if !defined(__clang__)
#pragma omp assume PARAMETR_SIZE <= 100
#endif
            for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                int k = agent_path[tx];
                if (static_cast<unsigned>(k) < static_cast<unsigned>(MAX_VALUE_SIZE)) {
                    int idx = tx + k * PARAMETR_SIZE;
                    local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_1
                    if (min1_value > 0) {
                        local_pheromon_add[idx] += min1_value;
                    }
#elif OPTIMIZE_MIN_2
                    local_pheromon_add[idx] += min2_value;
#elif OPTIMIZE_MAX
                    local_pheromon_add[idx] += max_value;
#endif
                }
            }
        }

        // OpenMP 5.2: улучшенное слияние
#pragma omp critical
        {
#pragma omp simd aligned(pheromon, kol_enter, local_pheromon_add, local_kol_enter_add:64)
            for (int idx = 0; idx < TOTAL_CELLS; ++idx) {
                kol_enter[idx] += local_kol_enter_add[idx];
                pheromon[idx] += local_pheromon_add[idx];
            }
        }

        ALIGNED_FREE(local_pheromon_add);
        ALIGNED_FREE(local_kol_enter_add);
    }
}
#endif
void add_pheromon_iteration_transp_OMP_non_cuda(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
#if _OPENMP >= 202111  // OpenMP 5.2+
    add_pheromon_iteration_transp_OMP_non_cuda_5_2(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 202011  // OpenMP 5.1+
    add_pheromon_iteration_transp_OMP_non_cuda_5_1(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 201811  // OpenMP 5.0+
    add_pheromon_iteration_transp_OMP_non_cuda_5_0(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 201511  // OpenMP 4.5+
    add_pheromon_iteration_transp_OMP_non_cuda_4_5(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 201307  // OpenMP 4.0+
    add_pheromon_iteration_transp_OMP_non_cuda_4_0(pheromon, kol_enter, agent_node, OF);
#else  // OpenMP 2.0/3.0/3.1
    add_pheromon_iteration_transp_OMP_non_cuda_2_0(pheromon, kol_enter, agent_node, OF);
#endif
}

int start_omp() {
    auto start = std::chrono::high_resolution_clock::now();
    double SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumgpuTime4 = 0.0f, SumgpuTime5 = 0.0f, SumgpuTime6 = 0.0f, SumgpuTime7 = 0.0f;
    double duration = 0.0f, duration_iteration = 0.0f;
    int kol_hash_fail = 0;
    const int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;
    const int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    const int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;

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
    if (!load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value)) {
        std::cerr << "Failed to load matrix from file: " << NAME_FILE_GRAPH << std::endl;
        return -1;
    }

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
        //std::cout << "go_all_agent_omp";
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

        #pragma omp parallel 
        {
            // Поиск максимума и минимума
            double maxOf = -std::numeric_limits<double>::max();
            double minOf = std::numeric_limits<double>::max();

            #pragma omp for
            for (int i = 0; i < ANT_SIZE; ++i) {
                if (antOF[i] != ZERO_HASH_RESULT) {
                    if (antOF[i] > maxOf) maxOf = antOF[i];
                    if (antOF[i] < minOf) minOf = antOF[i];
                }
            }
            #pragma omp critical
            {
                if (maxOf > global_maxOf) global_maxOf = maxOf;
                if (minOf < global_minOf) global_minOf = minOf;
            }
        }
   
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

#pragma omp parallel 
        {
            // Поиск максимума и минимума
            double maxOf = -std::numeric_limits<double>::max();
            double minOf = std::numeric_limits<double>::max();

#pragma omp for
            for (int i = 0; i < ANT_SIZE; ++i) {
                if (antOF[i] != ZERO_HASH_RESULT) {
                    if (antOF[i] > maxOf) maxOf = antOF[i];
                    if (antOF[i] < minOf) minOf = antOF[i];
                }
            }
#pragma omp critical
            {
                if (maxOf > global_maxOf) global_maxOf = maxOf;
                if (minOf < global_minOf) global_minOf = minOf;
            }
        }

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

int start_NON_CUDA_transp_OMP_time() {
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
        go_mass_probability_transp_OMP_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

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
        go_all_agent_transp_OMP_non_cuda_time(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail, SumTimeHashTotal, SumTimeOF, SumTimeHashSearch, SumTimeHashSave, SumTimeSearchAgent);

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
        add_pheromon_iteration_transp_OMP_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

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
    std::cout << "Time non CUDA_transp_OMP_time;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA_transp_OMP_time;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

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

int start_NON_CUDA_transp_non_hash_OMP_optimized() {
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
        go_mass_probability_transp_OMP_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

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
        go_all_agent_transp_non_cuda_non_hash_OMP_optimized(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, kol_hash_fail, SumgpuTime5);

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
        add_pheromon_iteration_transp_OMP_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

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
    std::cout << "Time non CUDA_transp non hash optimized;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << "0" << "; " << SumgpuTime5 << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA_transp non hash optimized:;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << "0" << "; " << SumgpuTime5 << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
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

// Структура для классической хэш-таблицы
struct HashEntry_classic {
    std::vector<int> key;
    double value;

    // Конструктор по умолчанию
    HashEntry_classic() : value(0.0) {}

    // Проверка на пустоту
    bool isEmpty() const {
        return key.empty();
    }

    // Проверка совпадения ключей
    bool keyEquals(const std::vector<int>& other) const {
        if (key.size() != other.size()) return false;
        for (size_t i = 0; i < key.size(); ++i) {
            if (key[i] != other[i]) return false;
        }
        return true;
    }
};

// ----------------- Функция инициализации хэш-таблицы -----------------
void initializeHashTable_classic(HashEntry_classic* hashTable, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        hashTable[i] = HashEntry_classic(); // Инициализируем каждый элемент
    }
    //std::cout << "Classic hash table initialized with size: " << size << std::endl;
}

// ----------------- Оптимизированная генерация ключа -----------------
unsigned long long generateKey_classic(const std::vector<int>& path) {
    unsigned long long key = 0;
    unsigned long long factor = 1;

    for (int val : path) {
        key += static_cast<unsigned long long>(val) * factor;

        // Проверка переполнения
        if (factor > ULLONG_MAX / MAX_VALUE_SIZE) {
            // Используем хэширование при переполнении
            key = key * 131 + val; // Простая хэш-функция
            factor = 1;
        }
        else {
            factor *= MAX_VALUE_SIZE;
        }
    }
    return key;
}

// ----------------- Быстрая хэш-функция -----------------
unsigned long long hashFunction_classic_fast(const std::vector<int>& path) {
    // Полиномиальное хэширование для лучшего распределения
    const unsigned long long prime = 1099511628211ULL;
    unsigned long long hash = 14695981039346656037ULL;

    for (int val : path) {
        hash ^= static_cast<unsigned long long>(val);
        hash *= prime;
    }

    return hash % HASH_TABLE_SIZE;
}

// ----------------- Альтернативная хэш-функция -----------------
unsigned long long hashFunction_classic_simple(const std::vector<int>& path) {
    unsigned long long hash = 0;

    for (int val : path) {
        hash = hash * 31 + val; // Простая, но эффективная хэш-функция
    }

    return hash % HASH_TABLE_SIZE;
}

// ----------------- Оптимизированный поиск в хэш-таблице -----------------
double getCachedResultOptimized_classic_ant(HashEntry_classic* hashTable, const std::vector<int>& path) {
    unsigned long long key_hash = hashFunction_classic_fast(path);

    for (int i = 0; i < MAX_PROBES; i++) {
        unsigned long long new_idx = (key_hash + static_cast<unsigned long long>(i * i)) % HASH_TABLE_SIZE;

        if (hashTable[new_idx].keyEquals(path)) {
            return hashTable[new_idx].value; // Найдено
        }
        if (hashTable[new_idx].isEmpty()) {
            return -1.0; // Не найдено и слот пуст
        }
    }
    return -1.0; // Не найдено после максимального количества проб
}

// ----------------- Оптимизированное сохранение в хэш-таблицу -----------------
bool saveToCacheOptimized_classic_ant(HashEntry_classic* hashTable, const std::vector<int>& path, double value) {
    unsigned long long key_hash = hashFunction_classic_fast(path);

    for (int i = 0; i < MAX_PROBES; i++) {
        unsigned long long new_idx = (key_hash + static_cast<unsigned long long>(i * i)) % HASH_TABLE_SIZE;

        if (hashTable[new_idx].isEmpty()) {
            // Успешно вставлено
            hashTable[new_idx].key = path;
            hashTable[new_idx].value = value;
            return true;
        }
        else if (hashTable[new_idx].keyEquals(path)) {
            // Ключ уже существует - обновление значения
            hashTable[new_idx].value = value;
            return true;
        }
    }

    // Таблица переполнена
    std::cerr << "Warning: Hash table full, could not insert path" << std::endl;
    return false;
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
static void start_ant_classic() {
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

static void start_ant_classic_non_hash() {
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


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#include <immintrin.h> // Для AVX

// Подготовка массива для вероятностного поиска
void go_mass_probability_AVX_non_cuda(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    const int remainder = MAX_VALUE_SIZE % CONST_AVX;

    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        const int base_idx = MAX_VALUE_SIZE * tx;

        // 1. Вычисляем sumVector для феромонов
        double sumVector = 0.0;

        // Векторизованное суммирование
        __m256d sum_pheromon_AVX = _mm256_setzero_pd();
        int i = 0;
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d pheromonValues = _mm256_loadu_pd(&pheromon[base_idx + i]);
            sum_pheromon_AVX = _mm256_add_pd(sum_pheromon_AVX, pheromonValues);
        }

        // Горизонтальное суммирование
        double temp[CONST_AVX];
        _mm256_storeu_pd(temp, sum_pheromon_AVX);
        for (int j = 0; j < CONST_AVX; j++) {
            sumVector += temp[j];
        }

        // Остаточные элементы
        for (; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[base_idx + i];
        }

        // Обработка нулевой суммы
        if (sumVector == 0.0) {
            // Установка равномерного распределения
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = uniform_prob;
            norm_matrix_probability[base_idx] = cumulative;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += uniform_prob;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
            continue;
        }

        // 2. Нормируем значения феромонов
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
        __m256d sumVector_AVX = _mm256_set1_pd(sumVector);

        i = 0;
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d pheromonValues = _mm256_loadu_pd(&pheromon[base_idx + i]);
            __m256d normValues = _mm256_div_pd(pheromonValues, sumVector_AVX);
            _mm256_storeu_pd(&pheromon_norm[i], normValues);
        }

        // Остаточные элементы для нормирования
        for (; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[base_idx + i] / sumVector;
        }

        // 3. Вычисляем svertka и их сумму
        double svertka[MAX_VALUE_SIZE] = { 0 };
        __m256d sumSvertka_AVX = _mm256_setzero_pd();
        __m256d zero_AVX = _mm256_setzero_pd();
        __m256d one_AVX = _mm256_set1_pd(1.0);

        i = 0;
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d kolEnterValues = _mm256_loadu_pd(&kol_enter[base_idx + i]);
            __m256d pheromonNormValues = _mm256_loadu_pd(&pheromon_norm[i]);

            // Проверка условий: kol_enter != 0 AND pheromon_norm != 0
            __m256d kolNonZeroMask = _mm256_cmp_pd(kolEnterValues, zero_AVX, _CMP_NEQ_OQ);
            __m256d pheromonNonZeroMask = _mm256_cmp_pd(pheromonNormValues, zero_AVX, _CMP_NEQ_OQ);
            __m256d conditionMask = _mm256_and_pd(kolNonZeroMask, pheromonNonZeroMask);

            // Вычисление: 1.0 / kol_enter + pheromon_norm
            __m256d invKolEnter = _mm256_div_pd(one_AVX, kolEnterValues);
            __m256d svertkaValues = _mm256_add_pd(invKolEnter, pheromonNormValues);

            // Применяем маску: если условие false -> 0.0
            svertkaValues = _mm256_blendv_pd(zero_AVX, svertkaValues, conditionMask);

            _mm256_storeu_pd(&svertka[i], svertkaValues);
            sumSvertka_AVX = _mm256_add_pd(sumSvertka_AVX, svertkaValues);
        }

        // Остаточные элементы для svertka
        for (; i < MAX_VALUE_SIZE; i++) {
            if (kol_enter[base_idx + i] != 0.0 && pheromon_norm[i] != 0.0) {
                svertka[i] = 1.0 / kol_enter[base_idx + i] + pheromon_norm[i];
            }
            else {
                svertka[i] = 0.0;
            }
        }

        // Суммируем svertka
        double sumSvertka = 0.0;
        _mm256_storeu_pd(temp, sumSvertka_AVX);
        for (int j = 0; j < CONST_AVX; j++) {
            sumSvertka += temp[j];
        }

        // Добавляем остаточные элементы к сумме
        for (i = MAX_VALUE_SIZE - remainder; i < MAX_VALUE_SIZE; i++) {
            sumSvertka += svertka[i];
        }

        // 4. Вычисляем кумулятивные вероятности
        if (sumSvertka == 0.0) {
            // Равномерное распределение если все svertka нулевые
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = uniform_prob;
            norm_matrix_probability[base_idx] = cumulative;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += uniform_prob;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
        else {
            double invSumSvertka = 1.0 / sumSvertka;
            double cumulative = svertka[0] * invSumSvertka;
            norm_matrix_probability[base_idx] = cumulative;

            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += svertka[i] * invSumSvertka;
                norm_matrix_probability[base_idx + i] = cumulative;
            }

            // Гарантируем, что последнее значение равно 1.0
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
    }
}
// Базовая версия OpenMP 2.0/3.0/3.1 с AVX
void go_mass_probability_AVX_OMP_non_cuda_2_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    const int remainder = MAX_VALUE_SIZE % CONST_AVX;

#pragma omp parallel for schedule(static)
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        const int base_idx = MAX_VALUE_SIZE * tx;

        // 1. Вычисляем sumVector для феромонов
        double sumVector = 0.0;

        // Векторизованное суммирование
        __m256d sum_pheromon_AVX = _mm256_setzero_pd();
        int i = 0;
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d pheromonValues = _mm256_loadu_pd(&pheromon[base_idx + i]);
            sum_pheromon_AVX = _mm256_add_pd(sum_pheromon_AVX, pheromonValues);
        }

        // Горизонтальное суммирование
        alignas(32) double temp[CONST_AVX];
        _mm256_store_pd(temp, sum_pheromon_AVX);
        for (int j = 0; j < CONST_AVX; j++) {
            sumVector += temp[j];
        }

        // Остаточные элементы
        for (; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[base_idx + i];
        }

        // Обработка нулевой суммы
        if (sumVector == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = uniform_prob;
            norm_matrix_probability[base_idx] = cumulative;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += uniform_prob;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
            continue;
        }

        // 2. Нормируем значения феромонов
        alignas(32) double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
        __m256d sumVector_AVX = _mm256_set1_pd(sumVector);

        i = 0;
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d pheromonValues = _mm256_loadu_pd(&pheromon[base_idx + i]);
            __m256d normValues = _mm256_div_pd(pheromonValues, sumVector_AVX);
            _mm256_store_pd(&pheromon_norm[i], normValues);
        }

        // Остаточные элементы для нормирования
        for (; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[base_idx + i] / sumVector;
        }

        // 3. Вычисляем svertka и их сумму
        alignas(32) double svertka[MAX_VALUE_SIZE] = { 0 };
        __m256d sumSvertka_AVX = _mm256_setzero_pd();
        __m256d zero_AVX = _mm256_setzero_pd();
        __m256d one_AVX = _mm256_set1_pd(1.0);

        i = 0;
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d kolEnterValues = _mm256_loadu_pd(&kol_enter[base_idx + i]);
            __m256d pheromonNormValues = _mm256_load_pd(&pheromon_norm[i]);

            // Проверка условий: kol_enter != 0 AND pheromon_norm != 0
            __m256d kolNonZeroMask = _mm256_cmp_pd(kolEnterValues, zero_AVX, _CMP_NEQ_OQ);
            __m256d pheromonNonZeroMask = _mm256_cmp_pd(pheromonNormValues, zero_AVX, _CMP_NEQ_OQ);
            __m256d conditionMask = _mm256_and_pd(kolNonZeroMask, pheromonNonZeroMask);

            // Вычисление: 1.0 / kol_enter + pheromon_norm
            __m256d invKolEnter = _mm256_div_pd(one_AVX, kolEnterValues);
            __m256d svertkaValues = _mm256_add_pd(invKolEnter, pheromonNormValues);

            // Применяем маску: если условие false -> 0.0
            svertkaValues = _mm256_blendv_pd(zero_AVX, svertkaValues, conditionMask);

            _mm256_store_pd(&svertka[i], svertkaValues);
            sumSvertka_AVX = _mm256_add_pd(sumSvertka_AVX, svertkaValues);
        }

        // Остаточные элементы для svertka
        for (; i < MAX_VALUE_SIZE; i++) {
            if (kol_enter[base_idx + i] != 0.0 && pheromon_norm[i] != 0.0) {
                svertka[i] = 1.0 / kol_enter[base_idx + i] + pheromon_norm[i];
            }
            else {
                svertka[i] = 0.0;
            }
        }

        // Суммируем svertka
        double sumSvertka = 0.0;
        _mm256_store_pd(temp, sumSvertka_AVX);
        for (int j = 0; j < CONST_AVX; j++) {
            sumSvertka += temp[j];
        }

        // Добавляем остаточные элементы к сумме
        for (i = MAX_VALUE_SIZE - remainder; i < MAX_VALUE_SIZE; i++) {
            sumSvertka += svertka[i];
        }

        // 4. Вычисляем кумулятивные вероятности
        if (sumSvertka == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = uniform_prob;
            norm_matrix_probability[base_idx] = cumulative;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += uniform_prob;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
        else {
            double invSumSvertka = 1.0 / sumSvertka;
            double cumulative = svertka[0] * invSumSvertka;
            norm_matrix_probability[base_idx] = cumulative;

            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += svertka[i] * invSumSvertka;
                norm_matrix_probability[base_idx + i] = cumulative;
            }

            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
    }
}
#if _OPENMP >= 201307  // OpenMP 4.0+
void go_mass_probability_AVX_OMP_non_cuda_4_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    const int remainder = MAX_VALUE_SIZE % CONST_AVX;

    // OpenMP 4.0: separate simd directive для внешнего цикла
#pragma omp parallel for schedule(static)
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        const int base_idx = MAX_VALUE_SIZE * tx;

        // 1. Вычисляем sumVector для феромонов
        double sumVector = 0.0;

        // Векторизованное суммирование с выровненной памятью
        __m256d sum_pheromon_AVX = _mm256_setzero_pd();
        int i = 0;
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d pheromonValues = _mm256_loadu_pd(&pheromon[base_idx + i]);
            sum_pheromon_AVX = _mm256_add_pd(sum_pheromon_AVX, pheromonValues);
        }

        // Горизонтальное суммирование
        alignas(32) double temp[CONST_AVX];
        _mm256_store_pd(temp, sum_pheromon_AVX);
        for (int j = 0; j < CONST_AVX; j++) {
            sumVector += temp[j];
        }

        // Остаточные элементы
        for (; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[base_idx + i];
        }

        if (sumVector == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = uniform_prob;
            norm_matrix_probability[base_idx] = cumulative;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += uniform_prob;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
            continue;
        }

        // 2. Нормируем значения феромонов
        alignas(32) double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
        __m256d sumVector_AVX = _mm256_set1_pd(sumVector);

        i = 0;
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d pheromonValues = _mm256_loadu_pd(&pheromon[base_idx + i]);
            __m256d normValues = _mm256_div_pd(pheromonValues, sumVector_AVX);
            _mm256_store_pd(&pheromon_norm[i], normValues);
        }

        for (; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[base_idx + i] / sumVector;
        }

        // 3. Вычисляем svertka и их сумму
        alignas(32) double svertka[MAX_VALUE_SIZE] = { 0 };
        __m256d sumSvertka_AVX = _mm256_setzero_pd();
        __m256d zero_AVX = _mm256_setzero_pd();
        __m256d one_AVX = _mm256_set1_pd(1.0);

        i = 0;
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d kolEnterValues = _mm256_loadu_pd(&kol_enter[base_idx + i]);
            __m256d pheromonNormValues = _mm256_load_pd(&pheromon_norm[i]);

            __m256d kolNonZeroMask = _mm256_cmp_pd(kolEnterValues, zero_AVX, _CMP_NEQ_OQ);
            __m256d pheromonNonZeroMask = _mm256_cmp_pd(pheromonNormValues, zero_AVX, _CMP_NEQ_OQ);
            __m256d conditionMask = _mm256_and_pd(kolNonZeroMask, pheromonNonZeroMask);

            __m256d invKolEnter = _mm256_div_pd(one_AVX, kolEnterValues);
            __m256d svertkaValues = _mm256_add_pd(invKolEnter, pheromonNormValues);
            svertkaValues = _mm256_blendv_pd(zero_AVX, svertkaValues, conditionMask);

            _mm256_store_pd(&svertka[i], svertkaValues);
            sumSvertka_AVX = _mm256_add_pd(sumSvertka_AVX, svertkaValues);
        }

        for (; i < MAX_VALUE_SIZE; i++) {
            if (kol_enter[base_idx + i] != 0.0 && pheromon_norm[i] != 0.0) {
                svertka[i] = 1.0 / kol_enter[base_idx + i] + pheromon_norm[i];
            }
            else {
                svertka[i] = 0.0;
            }
        }

        double sumSvertka = 0.0;
        _mm256_store_pd(temp, sumSvertka_AVX);
        for (int j = 0; j < CONST_AVX; j++) {
            sumSvertka += temp[j];
        }

        for (i = MAX_VALUE_SIZE - remainder; i < MAX_VALUE_SIZE; i++) {
            sumSvertka += svertka[i];
        }

        // 4. Вычисляем кумулятивные вероятности
        if (sumSvertka == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = uniform_prob;
            norm_matrix_probability[base_idx] = cumulative;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += uniform_prob;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
        else {
            double invSumSvertka = 1.0 / sumSvertka;
            double cumulative = svertka[0] * invSumSvertka;
            norm_matrix_probability[base_idx] = cumulative;

            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += svertka[i] * invSumSvertka;
                norm_matrix_probability[base_idx + i] = cumulative;
            }

            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
    }
}
#endif
#if _OPENMP >= 201511  // OpenMP 4.5+
void go_mass_probability_AVX_OMP_non_cuda_4_5(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    const int remainder = MAX_VALUE_SIZE % CONST_AVX;

    // OpenMP 4.5: if clause для условного выполнения
#if defined(__clang__)
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(static) // if(PARAMETR_SIZE > 100)
#endif
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        const int base_idx = MAX_VALUE_SIZE * tx;

        double sumVector = 0.0;
        __m256d sum_pheromon_AVX = _mm256_setzero_pd();
        int i = 0;

        // Автоматическая векторизация внутренних циклов
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d pheromonValues = _mm256_loadu_pd(&pheromon[base_idx + i]);
            sum_pheromon_AVX = _mm256_add_pd(sum_pheromon_AVX, pheromonValues);
        }

        alignas(32) double temp[CONST_AVX];
        _mm256_store_pd(temp, sum_pheromon_AVX);
        for (int j = 0; j < CONST_AVX; j++) {
            sumVector += temp[j];
        }

        for (; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[base_idx + i];
        }

        if (sumVector == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = uniform_prob;
            norm_matrix_probability[base_idx] = cumulative;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += uniform_prob;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
            continue;
        }

        alignas(32) double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
        __m256d sumVector_AVX = _mm256_set1_pd(sumVector);

        i = 0;
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d pheromonValues = _mm256_loadu_pd(&pheromon[base_idx + i]);
            __m256d normValues = _mm256_div_pd(pheromonValues, sumVector_AVX);
            _mm256_store_pd(&pheromon_norm[i], normValues);
        }

        for (; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[base_idx + i] / sumVector;
        }

        alignas(32) double svertka[MAX_VALUE_SIZE] = { 0 };
        __m256d sumSvertka_AVX = _mm256_setzero_pd();
        __m256d zero_AVX = _mm256_setzero_pd();
        __m256d one_AVX = _mm256_set1_pd(1.0);

        i = 0;
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d kolEnterValues = _mm256_loadu_pd(&kol_enter[base_idx + i]);
            __m256d pheromonNormValues = _mm256_load_pd(&pheromon_norm[i]);

            __m256d kolNonZeroMask = _mm256_cmp_pd(kolEnterValues, zero_AVX, _CMP_NEQ_OQ);
            __m256d pheromonNonZeroMask = _mm256_cmp_pd(pheromonNormValues, zero_AVX, _CMP_NEQ_OQ);
            __m256d conditionMask = _mm256_and_pd(kolNonZeroMask, pheromonNonZeroMask);

            __m256d invKolEnter = _mm256_div_pd(one_AVX, kolEnterValues);
            __m256d svertkaValues = _mm256_add_pd(invKolEnter, pheromonNormValues);
            svertkaValues = _mm256_blendv_pd(zero_AVX, svertkaValues, conditionMask);

            _mm256_store_pd(&svertka[i], svertkaValues);
            sumSvertka_AVX = _mm256_add_pd(sumSvertka_AVX, svertkaValues);
        }

        for (; i < MAX_VALUE_SIZE; i++) {
            if (kol_enter[base_idx + i] != 0.0 && pheromon_norm[i] != 0.0) {
                svertka[i] = 1.0 / kol_enter[base_idx + i] + pheromon_norm[i];
            }
            else {
                svertka[i] = 0.0;
            }
        }

        double sumSvertka = 0.0;
        _mm256_store_pd(temp, sumSvertka_AVX);
        for (int j = 0; j < CONST_AVX; j++) {
            sumSvertka += temp[j];
        }

        for (i = MAX_VALUE_SIZE - remainder; i < MAX_VALUE_SIZE; i++) {
            sumSvertka += svertka[i];
        }

        if (sumSvertka == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = uniform_prob;
            norm_matrix_probability[base_idx] = cumulative;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += uniform_prob;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
        else {
            double invSumSvertka = 1.0 / sumSvertka;
            double cumulative = svertka[0] * invSumSvertka;
            norm_matrix_probability[base_idx] = cumulative;

            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += svertka[i] * invSumSvertka;
                norm_matrix_probability[base_idx + i] = cumulative;
            }

            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
    }
}
#endif
#if _OPENMP >= 201811  // OpenMP 5.0+
void go_mass_probability_AVX_OMP_non_cuda_5_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    const int remainder = MAX_VALUE_SIZE % CONST_AVX;

    // OpenMP 5.0: loop transformation hints
#pragma omp parallel for schedule(static)
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        const int base_idx = MAX_VALUE_SIZE * tx;

        double sumVector = 0.0;
        __m256d sum_pheromon_AVX = _mm256_setzero_pd();
        int i = 0;

        // OpenMP 5.0: оптимизированные циклы
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d pheromonValues = _mm256_loadu_pd(&pheromon[base_idx + i]);
            sum_pheromon_AVX = _mm256_add_pd(sum_pheromon_AVX, pheromonValues);
        }

        alignas(32) double temp[CONST_AVX];
        _mm256_store_pd(temp, sum_pheromon_AVX);
        for (int j = 0; j < CONST_AVX; j++) {
            sumVector += temp[j];
        }

        for (; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[base_idx + i];
        }

        if (sumVector == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = uniform_prob;
            norm_matrix_probability[base_idx] = cumulative;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += uniform_prob;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
            continue;
        }

        alignas(32) double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
        __m256d sumVector_AVX = _mm256_set1_pd(sumVector);

        i = 0;
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d pheromonValues = _mm256_loadu_pd(&pheromon[base_idx + i]);
            __m256d normValues = _mm256_div_pd(pheromonValues, sumVector_AVX);
            _mm256_store_pd(&pheromon_norm[i], normValues);
        }

        for (; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[base_idx + i] / sumVector;
        }

        alignas(32) double svertka[MAX_VALUE_SIZE] = { 0 };
        __m256d sumSvertka_AVX = _mm256_setzero_pd();
        __m256d zero_AVX = _mm256_setzero_pd();
        __m256d one_AVX = _mm256_set1_pd(1.0);

        i = 0;
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d kolEnterValues = _mm256_loadu_pd(&kol_enter[base_idx + i]);
            __m256d pheromonNormValues = _mm256_load_pd(&pheromon_norm[i]);

            __m256d kolNonZeroMask = _mm256_cmp_pd(kolEnterValues, zero_AVX, _CMP_NEQ_OQ);
            __m256d pheromonNonZeroMask = _mm256_cmp_pd(pheromonNormValues, zero_AVX, _CMP_NEQ_OQ);
            __m256d conditionMask = _mm256_and_pd(kolNonZeroMask, pheromonNonZeroMask);

            __m256d invKolEnter = _mm256_div_pd(one_AVX, kolEnterValues);
            __m256d svertkaValues = _mm256_add_pd(invKolEnter, pheromonNormValues);
            svertkaValues = _mm256_blendv_pd(zero_AVX, svertkaValues, conditionMask);

            _mm256_store_pd(&svertka[i], svertkaValues);
            sumSvertka_AVX = _mm256_add_pd(sumSvertka_AVX, svertkaValues);
        }

        for (; i < MAX_VALUE_SIZE; i++) {
            if (kol_enter[base_idx + i] != 0.0 && pheromon_norm[i] != 0.0) {
                svertka[i] = 1.0 / kol_enter[base_idx + i] + pheromon_norm[i];
            }
            else {
                svertka[i] = 0.0;
            }
        }

        double sumSvertka = 0.0;
        _mm256_store_pd(temp, sumSvertka_AVX);
        for (int j = 0; j < CONST_AVX; j++) {
            sumSvertka += temp[j];
        }

        for (i = MAX_VALUE_SIZE - remainder; i < MAX_VALUE_SIZE; i++) {
            sumSvertka += svertka[i];
        }

        if (sumSvertka == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = uniform_prob;
            norm_matrix_probability[base_idx] = cumulative;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += uniform_prob;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
        else {
            double invSumSvertka = 1.0 / sumSvertka;
            double cumulative = svertka[0] * invSumSvertka;
            norm_matrix_probability[base_idx] = cumulative;

            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += svertka[i] * invSumSvertka;
                norm_matrix_probability[base_idx + i] = cumulative;
            }

            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
    }
}
#endif
#if _OPENMP >= 202011  // OpenMP 5.1+
void go_mass_probability_AVX_OMP_non_cuda_5_1(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    const int remainder = MAX_VALUE_SIZE % CONST_AVX;

    // OpenMP 5.1: error recovery и улучшенное управление памятью
#pragma omp parallel for schedule(nonmonotonic:static)
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        const int base_idx = MAX_VALUE_SIZE * tx;

        double sumVector = 0.0;
        __m256d sum_pheromon_AVX = _mm256_setzero_pd();
        int i = 0;

        // OpenMP 5.1: оптимизированные циклы с неблокирующими операциями
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d pheromonValues = _mm256_loadu_pd(&pheromon[base_idx + i]);
            sum_pheromon_AVX = _mm256_add_pd(sum_pheromon_AVX, pheromonValues);
        }

        alignas(32) double temp[CONST_AVX];
        _mm256_store_pd(temp, sum_pheromon_AVX);

        // OpenMP 5.1: улучшенное скалярное суммирование
#pragma omp simd reduction(+:sumVector)
        for (int j = 0; j < CONST_AVX; j++) {
            sumVector += temp[j];
        }

        for (; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[base_idx + i];
        }

        if (sumVector == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = uniform_prob;
            norm_matrix_probability[base_idx] = cumulative;

            // OpenMP 5.1: оптимизированный цикл для равномерного распределения
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += uniform_prob;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
            continue;
        }

        alignas(32) double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
        __m256d sumVector_AVX = _mm256_set1_pd(sumVector);

        i = 0;
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d pheromonValues = _mm256_loadu_pd(&pheromon[base_idx + i]);
            __m256d normValues = _mm256_div_pd(pheromonValues, sumVector_AVX);
            _mm256_store_pd(&pheromon_norm[i], normValues);
        }

        for (; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[base_idx + i] / sumVector;
        }

        alignas(32) double svertka[MAX_VALUE_SIZE] = { 0 };
        __m256d sumSvertka_AVX = _mm256_setzero_pd();
        __m256d zero_AVX = _mm256_setzero_pd();
        __m256d one_AVX = _mm256_set1_pd(1.0);

        i = 0;
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d kolEnterValues = _mm256_loadu_pd(&kol_enter[base_idx + i]);
            __m256d pheromonNormValues = _mm256_load_pd(&pheromon_norm[i]);

            // OpenMP 5.1: оптимизированные маскированные операции
            __m256d kolNonZeroMask = _mm256_cmp_pd(kolEnterValues, zero_AVX, _CMP_NEQ_OQ);
            __m256d pheromonNonZeroMask = _mm256_cmp_pd(pheromonNormValues, zero_AVX, _CMP_NEQ_OQ);
            __m256d conditionMask = _mm256_and_pd(kolNonZeroMask, pheromonNonZeroMask);

            __m256d invKolEnter = _mm256_div_pd(one_AVX, kolEnterValues);
            __m256d svertkaValues = _mm256_add_pd(invKolEnter, pheromonNormValues);
            svertkaValues = _mm256_blendv_pd(zero_AVX, svertkaValues, conditionMask);

            _mm256_store_pd(&svertka[i], svertkaValues);
            sumSvertka_AVX = _mm256_add_pd(sumSvertka_AVX, svertkaValues);
        }

        for (; i < MAX_VALUE_SIZE; i++) {
            if (kol_enter[base_idx + i] != 0.0 && pheromon_norm[i] != 0.0) {
                svertka[i] = 1.0 / kol_enter[base_idx + i] + pheromon_norm[i];
            }
            else {
                svertka[i] = 0.0;
            }
        }

        double sumSvertka = 0.0;
        _mm256_store_pd(temp, sumSvertka_AVX);

#pragma omp simd reduction(+:sumSvertka)
        for (int j = 0; j < CONST_AVX; j++) {
            sumSvertka += temp[j];
        }

        for (i = MAX_VALUE_SIZE - remainder; i < MAX_VALUE_SIZE; i++) {
            sumSvertka += svertka[i];
        }

        if (sumSvertka == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = uniform_prob;
            norm_matrix_probability[base_idx] = cumulative;

            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += uniform_prob;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
        else {
            double invSumSvertka = 1.0 / sumSvertka;
            double cumulative = svertka[0] * invSumSvertka;
            norm_matrix_probability[base_idx] = cumulative;

            // OpenMP 5.1: оптимизированный цикл кумулятивных сумм
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += svertka[i] * invSumSvertka;
                norm_matrix_probability[base_idx + i] = cumulative;
            }

            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
    }
}
#endif
#if _OPENMP >= 202111  // OpenMP 5.2+
void go_mass_probability_AVX_OMP_non_cuda_5_2(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    const int remainder = MAX_VALUE_SIZE % CONST_AVX;

    // OpenMP 5.2: assume clauses и улучшенный контроль памяти
#pragma omp parallel for schedule(static)
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        // OpenMP 5.2: assume clauses для оптимизатора
#if !defined(__clang__)
#pragma omp assume MAX_VALUE_SIZE % 4 == 0  // Предполагаем кратность 4 для AVX
#pragma omp assume PARAMETR_SIZE <= 1024    // Предполагаем разумный размер
#endif

        const int base_idx = MAX_VALUE_SIZE * tx;

        double sumVector = 0.0;
        __m256d sum_pheromon_AVX = _mm256_setzero_pd();
        int i = 0;

        // OpenMP 5.2: оптимизированные AVX циклы с assume
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d pheromonValues = _mm256_loadu_pd(&pheromon[base_idx + i]);
            sum_pheromon_AVX = _mm256_add_pd(sum_pheromon_AVX, pheromonValues);
        }

        alignas(32) double temp[CONST_AVX];
        _mm256_store_pd(temp, sum_pheromon_AVX);

        // OpenMP 5.2: улучшенное скалярное суммирование
#if !defined(__clang__)
#pragma omp assume CONST_AVX == 4  // Предполагаем размер AVX регистра
#endif
        for (int j = 0; j < CONST_AVX; j++) {
            sumVector += temp[j];
        }

        // Обработка остаточных элементов с assume
#if !defined(__clang__)
#pragma omp assume remainder < 4
#endif
        for (; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[base_idx + i];
        }

        if (sumVector == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = uniform_prob;
            norm_matrix_probability[base_idx] = cumulative;

            // OpenMP 5.2: оптимизированный цикл с assume
#if !defined(__clang__)
#pragma omp assume MAX_VALUE_SIZE > 1
#endif
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += uniform_prob;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
            continue;
        }

        alignas(32) double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
        __m256d sumVector_AVX = _mm256_set1_pd(sumVector);

        i = 0;
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d pheromonValues = _mm256_loadu_pd(&pheromon[base_idx + i]);
            __m256d normValues = _mm256_div_pd(pheromonValues, sumVector_AVX);
            _mm256_store_pd(&pheromon_norm[i], normValues);
        }

        for (; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[base_idx + i] / sumVector;
        }

        alignas(32) double svertka[MAX_VALUE_SIZE] = { 0 };
        __m256d sumSvertka_AVX = _mm256_setzero_pd();
        __m256d zero_AVX = _mm256_setzero_pd();
        __m256d one_AVX = _mm256_set1_pd(1.0);

        i = 0;
        for (; i <= MAX_VALUE_SIZE - CONST_AVX; i += CONST_AVX) {
            __m256d kolEnterValues = _mm256_loadu_pd(&kol_enter[base_idx + i]);
            __m256d pheromonNormValues = _mm256_load_pd(&pheromon_norm[i]);

            // OpenMP 5.2: оптимизированные сравнения с assume
            __m256d kolNonZeroMask = _mm256_cmp_pd(kolEnterValues, zero_AVX, _CMP_NEQ_OQ);
            __m256d pheromonNonZeroMask = _mm256_cmp_pd(pheromonNormValues, zero_AVX, _CMP_NEQ_OQ);
            __m256d conditionMask = _mm256_and_pd(kolNonZeroMask, pheromonNonZeroMask);

            __m256d invKolEnter = _mm256_div_pd(one_AVX, kolEnterValues);
            __m256d svertkaValues = _mm256_add_pd(invKolEnter, pheromonNormValues);
            svertkaValues = _mm256_blendv_pd(zero_AVX, svertkaValues, conditionMask);

            _mm256_store_pd(&svertka[i], svertkaValues);
            sumSvertka_AVX = _mm256_add_pd(sumSvertka_AVX, svertkaValues);
        }

        // Обработка остаточных элементов для svertka
#if !defined(__clang__)
#pragma omp assume remainder < 4
#endif
        for (; i < MAX_VALUE_SIZE; i++) {
            if (kol_enter[base_idx + i] != 0.0 && pheromon_norm[i] != 0.0) {
                svertka[i] = 1.0 / kol_enter[base_idx + i] + pheromon_norm[i];
            }
            else {
                svertka[i] = 0.0;
            }
        }

        double sumSvertka = 0.0;
        _mm256_store_pd(temp, sumSvertka_AVX);

#if !defined(__clang__)
#pragma omp assume CONST_AVX == 4
#endif
        for (int j = 0; j < CONST_AVX; j++) {
            sumSvertka += temp[j];
        }

        // Суммирование остаточных элементов
#if !defined(__clang__)
#pragma omp assume remainder < 4
#endif
        for (i = MAX_VALUE_SIZE - remainder; i < MAX_VALUE_SIZE; i++) {
            sumSvertka += svertka[i];
        }

        if (sumSvertka == 0.0) {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = uniform_prob;
            norm_matrix_probability[base_idx] = cumulative;
#if !defined(__clang__)
#pragma omp assume MAX_VALUE_SIZE > 1
#endif
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += uniform_prob;
                norm_matrix_probability[base_idx + i] = cumulative;
            }
            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
        else {
            double invSumSvertka = 1.0 / sumSvertka;
            double cumulative = svertka[0] * invSumSvertka;
            norm_matrix_probability[base_idx] = cumulative;

            // OpenMP 5.2: оптимизированный цикл кумулятивных вероятностей
#if !defined(__clang__)
#pragma omp assume MAX_VALUE_SIZE > 1
#endif
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                cumulative += svertka[i] * invSumSvertka;
                norm_matrix_probability[base_idx + i] = cumulative;
            }

            norm_matrix_probability[base_idx + MAX_VALUE_SIZE - 1] = 1.0;
        }
    }
}
#endif
void go_mass_probability_AVX_OMP_non_cuda(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
#if _OPENMP >= 202111  // OpenMP 5.2+
    go_mass_probability_AVX_OMP_non_cuda_5_2(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 202011  // OpenMP 5.1+
    go_mass_probability_AVX_OMP_non_cuda_5_1(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201811  // OpenMP 5.0+
    go_mass_probability_AVX_OMP_non_cuda_5_0(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201511  // OpenMP 4.5+
    go_mass_probability_AVX_OMP_non_cuda_4_5(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201307  // OpenMP 4.0+
    go_mass_probability_AVX_OMP_non_cuda_4_0(pheromon, kol_enter, norm_matrix_probability);
#else  // OpenMP 2.0/3.0/3.1
    go_mass_probability_AVX_OMP_non_cuda_2_0(pheromon, kol_enter, norm_matrix_probability);
#endif
}
void go_mass_probability_AVX_non_cuda_4(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //MAX_VALUE_SIZE=4
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        //Загрузка данных в вектора
        __m256d pheromonValues_AVX = _mm256_loadu_pd(&pheromon[MAX_VALUE_SIZE * tx]);
        __m256d kolEnterValues_AVX = _mm256_loadu_pd(&kol_enter[MAX_VALUE_SIZE * tx]);
        // Вычисляем sumVector
        double sumVector = pheromon[MAX_VALUE_SIZE * tx] + pheromon[MAX_VALUE_SIZE * tx + 1] + pheromon[MAX_VALUE_SIZE * tx + 2] + pheromon[MAX_VALUE_SIZE * tx + 3];
        __m256d pheromonNormValues_AVX = _mm256_div_pd(pheromonValues_AVX, _mm256_set1_pd(sumVector)); // Нормируем значения феромона

        __m256d mask_AVX = _mm256_cmp_pd(kolEnterValues_AVX, _mm256_setzero_pd(), _CMP_NEQ_OQ); // Создаем маску для проверки условий
        mask_AVX = _mm256_and_pd(mask_AVX, _mm256_cmp_pd(pheromonNormValues_AVX, _mm256_setzero_pd(), _CMP_NEQ_OQ));
        // Вычисляем svertka с учетом условий
        __m256d oneOverKolEnter_AVX = _mm256_div_pd(_mm256_set1_pd(1.0), kolEnterValues_AVX);
        __m256d svertkaValues_AVX = _mm256_add_pd(oneOverKolEnter_AVX, pheromonNormValues_AVX);
        svertkaValues_AVX = _mm256_blendv_pd(_mm256_setzero_pd(), svertkaValues_AVX, mask_AVX);

        double svertka[MAX_VALUE_SIZE] = { 0 };
        // Суммируем значения из вектора svertka
        _mm256_storeu_pd(svertka, svertkaValues_AVX);
        sumVector = svertka[0] + svertka[1] + svertka[2] + svertka[3];
        // Заполняем norm_matrix_probability
        if (sumVector != 0) { // Проверка на деление на ноль
            norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
            norm_matrix_probability[MAX_VALUE_SIZE * tx + 1] = (svertka[1] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx];
            norm_matrix_probability[MAX_VALUE_SIZE * tx + 2] = (svertka[2] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + 1];
            norm_matrix_probability[MAX_VALUE_SIZE * tx + 3] = (svertka[3] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + 2];
        }
    }
}
void go_mass_probability_AVX_OMP_non_cuda_4( double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {

    alignas(32) const double uniform_probs[4] = { 0.25, 0.5, 0.75, 1.0 };
    const __m256d uniform_avx = _mm256_load_pd(uniform_probs);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d epsilon = _mm256_set1_pd(1e-12);

    // Условный параллелизм только для OpenMP 4.5+
#if _OPENMP >= 201511 && !defined(__clang__)
#pragma omp parallel for schedule(static) // if(PARAMETR_SIZE > 100)
#else
#pragma omp parallel for schedule(static)
#endif
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        const int base_idx = tx * 4; // MAX_VALUE_SIZE = 4

        // 1. Загрузка данных
        __m256d pheromon_vals = _mm256_loadu_pd(&pheromon[base_idx]);
        __m256d kol_enter_vals = _mm256_loadu_pd(&kol_enter[base_idx]);

        // 2. Суммирование феромонов (оптимальный способ для 4 элементов)
        alignas(32) double temp[4];
        _mm256_store_pd(temp, pheromon_vals);
        double sum_pheromon = temp[0] + temp[1] + temp[2] + temp[3];

        if (sum_pheromon < 1e-12) {
            _mm256_storeu_pd(&norm_matrix_probability[base_idx], uniform_avx);
            continue;
        }

        // 3. Нормализация феромонов
        __m256d inv_sum_pheromon = _mm256_set1_pd(1.0 / sum_pheromon);
        __m256d pheromon_norm = _mm256_mul_pd(pheromon_vals, inv_sum_pheromon);

        // 4. Вычисление svertka с условиями
        __m256d kol_non_zero = _mm256_cmp_pd(kol_enter_vals, epsilon, _CMP_GT_OQ);
        __m256d pheromon_non_zero = _mm256_cmp_pd(pheromon_norm, epsilon, _CMP_GT_OQ);
        __m256d mask = _mm256_and_pd(kol_non_zero, pheromon_non_zero);

        __m256d safe_kol = _mm256_max_pd(kol_enter_vals, epsilon);
        __m256d inv_kol = _mm256_div_pd(one, safe_kol);
        __m256d svertka = _mm256_add_pd(inv_kol, pheromon_norm);
        svertka = _mm256_blendv_pd(zero, svertka, mask);

        // 5. Суммирование svertka
        _mm256_store_pd(temp, svertka);
        double sum_svertka = temp[0] + temp[1] + temp[2] + temp[3];

        if (sum_svertka > 1e-12) {
            // 6. Нормализация и кумулятивное суммирование
            __m256d inv_sum_s = _mm256_set1_pd(1.0 / sum_svertka);
            __m256d normalized = _mm256_mul_pd(svertka, inv_sum_s);

            _mm256_store_pd(temp, normalized);

            // Простое и эффективное кумулятивное суммирование
            temp[1] += temp[0];
            temp[2] += temp[1];
            temp[3] = 1.0; // Гарантия точности

            _mm256_storeu_pd(&norm_matrix_probability[base_idx], _mm256_load_pd(temp));
        }
        else {
            _mm256_storeu_pd(&norm_matrix_probability[base_idx], uniform_avx);
        }
    }
}
void go_mass_probability_not_f_AVX_non_cuda_4(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    //MAX_VALUE_SIZE=4
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        //Загрузка данных в вектора
        __m256d pheromonValues_AVX = _mm256_loadu_pd(&pheromon[MAX_VALUE_SIZE * tx]);
        __m256d kolEnterValues_AVX = _mm256_loadu_pd(&kol_enter[MAX_VALUE_SIZE * tx]);
        // Вычисляем sumVector
        double sumVector = pheromon[MAX_VALUE_SIZE * tx] + pheromon[MAX_VALUE_SIZE * tx + 1] + pheromon[MAX_VALUE_SIZE * tx + 2] + pheromon[MAX_VALUE_SIZE * tx + 3];
        __m256d pheromonNormValues_AVX = _mm256_div_pd(pheromonValues_AVX, _mm256_set1_pd(sumVector)); // Нормируем значения феромона

        __m256d mask_AVX = _mm256_cmp_pd(kolEnterValues_AVX, _mm256_setzero_pd(), _CMP_NEQ_OQ); // Создаем маску для проверки условий
        mask_AVX = _mm256_and_pd(mask_AVX, _mm256_cmp_pd(pheromonNormValues_AVX, _mm256_setzero_pd(), _CMP_NEQ_OQ));
        // Вычисляем svertka с учетом условий
        __m256d oneOverKolEnter_AVX = _mm256_div_pd(_mm256_set1_pd(1.0), kolEnterValues_AVX);
        __m256d svertkaValues_AVX = _mm256_add_pd(oneOverKolEnter_AVX, pheromonNormValues_AVX);
        svertkaValues_AVX = _mm256_blendv_pd(_mm256_setzero_pd(), svertkaValues_AVX, mask_AVX);

        double svertka[MAX_VALUE_SIZE] = { 0 };
        // Суммируем значения из вектора svertka
        _mm256_storeu_pd(svertka, svertkaValues_AVX);
        sumVector = svertka[0] + svertka[1] + svertka[2] + svertka[3];
        // Заполняем norm_matrix_probability
        __m256d normalizedResult_AVX = _mm256_div_pd(svertkaValues_AVX, _mm256_set1_pd(sumVector)); // Нормируем значения svertkaValues_AVX
        _mm256_storeu_pd(&norm_matrix_probability[MAX_VALUE_SIZE * tx], normalizedResult_AVX);
    }
}
void go_mass_probability_not_f_AVX_OMP_non_cuda_4_fixed( double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {

    alignas(32) static const __m256d zero = _mm256_setzero_pd();
    alignas(32) static const __m256d one = _mm256_set1_pd(1.0);
    alignas(32) static const __m256d uniform = _mm256_set_pd(0.75, 0.5, 0.25, 0.0); // Для cumulative
    constexpr double epsilon = 1e-12;

#if _OPENMP >= 201511 && !defined(__clang__)
#pragma omp parallel // if(PARAMETR_SIZE > 100)
#else
#pragma omp parallel
#endif
    {
        __m256d pheromon_vals, kol_enter_vals, pheromon_norm, mask, inv_kol, svertka;
        __m256d sum_avx, normalized;
        double sum_pheromon, sum_svertka;
        alignas(32) double temp[4];

#pragma omp for schedule(static) nowait
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            const int base_idx = 4 * tx; // MAX_VALUE_SIZE = 4

            // Загрузка данных
            pheromon_vals = _mm256_load_pd(&pheromon[base_idx]);
            kol_enter_vals = _mm256_load_pd(&kol_enter[base_idx]);

            // Суммирование феромонов
            sum_avx = _mm256_hadd_pd(pheromon_vals, pheromon_vals);
            sum_pheromon = ((double*)&sum_avx)[0] + ((double*)&sum_avx)[2];

            if (sum_pheromon < epsilon) {
                // Равномерное распределение с cumulative sum
                _mm256_store_pd(&norm_matrix_probability[base_idx],
                    _mm256_add_pd(uniform, _mm256_set1_pd(0.25)));
                continue;
            }

            // Нормализация феромонов
            pheromon_norm = _mm256_div_pd(pheromon_vals, _mm256_set1_pd(sum_pheromon));

            // Условия с безопасными порогами
            mask = _mm256_and_pd(
                _mm256_cmp_pd(kol_enter_vals, _mm256_set1_pd(epsilon), _CMP_GT_OQ),
                _mm256_cmp_pd(pheromon_norm, _mm256_set1_pd(epsilon), _CMP_GT_OQ)
            );

            // Безопасное вычисление svertka
            __m256d safe_kol = _mm256_max_pd(kol_enter_vals, _mm256_set1_pd(epsilon));
            inv_kol = _mm256_div_pd(one, safe_kol);
            svertka = _mm256_add_pd(inv_kol, pheromon_norm);
            svertka = _mm256_blendv_pd(zero, svertka, mask);

            // Суммирование svertka
            sum_avx = _mm256_hadd_pd(svertka, svertka);
            sum_svertka = ((double*)&sum_avx)[0] + ((double*)&sum_avx)[2];

            if (sum_svertka > epsilon) {
                // Нормализация и cumulative sum
                normalized = _mm256_div_pd(svertka, _mm256_set1_pd(sum_svertka));
                _mm256_store_pd(temp, normalized);

                // Кумулятивное суммирование
                temp[1] += temp[0];
                temp[2] += temp[1];
                temp[3] = 1.0; // Гарантия

                _mm256_store_pd(&norm_matrix_probability[base_idx], _mm256_load_pd(temp));
            }
            else {
                // Равномерное распределение
                _mm256_store_pd(&norm_matrix_probability[base_idx],
                    _mm256_add_pd(uniform, _mm256_set1_pd(0.25)));
            }
        }
    }
}
// Обновление слоев графа
void add_pheromon_iteration_AVX_non_cuda(double* __restrict pheromon, double* __restrict kol_enter, int* __restrict agent_node, double* __restrict OF) {
    // Испарение весов-феромона
    __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);
    for (int i = 0; i < PARAMETR_SIZE * MAX_VALUE_SIZE; i += CONST_AVX) {
        if (i + CONST_AVX < PARAMETR_SIZE * MAX_VALUE_SIZE) { // Проверка на выход за пределы массива
            __m256d pheromonValues_AVX = _mm256_loadu_pd(&pheromon[i]); // Загружаем 4 значения из pheromon
            pheromonValues_AVX = _mm256_mul_pd(pheromonValues_AVX, parametRovector_AVX);  // Умножаем на PARAMETR_RO
            _mm256_storeu_pd(&pheromon[i], pheromonValues_AVX); // Сохраняем обратно в pheromon
        }
    }

    for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
        // Добавление весов-феромона
        for (int i = 0; i < ANT_SIZE; ++i) {
            int k = agent_node[i * PARAMETR_SIZE + tx];
            if (k >= 0 && k < MAX_VALUE_SIZE) { // Проверка на выход за пределы массива kol_enter
                kol_enter[MAX_VALUE_SIZE * tx + k]++;
                // Проверяем условие и обновляем pheromon
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
// Базовая версия OpenMP 2.0/3.0/3.1
void add_pheromon_iteration_AVX_OMP_non_cuda_2_0(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);
    const size_t total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int remainder = total_size % CONST_AVX;

    // 1. Испарение феромонов - векторизованное
#pragma omp parallel for schedule(static)
    for (int i = 0; i < total_size - remainder; i += CONST_AVX) {
        __m256d pheromonValues = _mm256_loadu_pd(&pheromon[i]);
        pheromonValues = _mm256_mul_pd(pheromonValues, parametRovector_AVX);
        _mm256_storeu_pd(&pheromon[i], pheromonValues);
    }

    // Обработка остаточных элементов
    if (remainder > 0) {
#pragma omp parallel for schedule(static)
        for (int i = total_size - remainder; i < total_size; i++) {
            pheromon[i] *= PARAMETR_RO;
        }
    }

    // 2. Добавление феромонов
#pragma omp parallel
    {
        double* local_pheromon_add = (double*)calloc(total_size, sizeof(double));
        int* local_kol_enter_add = (int*)calloc(total_size, sizeof(int));

#pragma omp for nowait
        for (int i = 0; i < ANT_SIZE; i++) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            const double min1_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT > agent_of) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            const double min2_value = PARAMETR_Q / ((agent_of == 0) ? 1e-7 : agent_of);
#elif OPTIMIZE_MAX
            const double max_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                int k = agent_path[tx];
                if (k >= 0 && k < MAX_VALUE_SIZE) {
                    int idx = MAX_VALUE_SIZE * tx + k;
                    local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_1
                    if (min1_value > 0) local_pheromon_add[idx] += min1_value;
#elif OPTIMIZE_MIN_2
                    local_pheromon_add[idx] += min2_value;
#elif OPTIMIZE_MAX
                    local_pheromon_add[idx] += max_value;
#endif
                }
            }
        }

#pragma omp critical
        {
            for (int idx = 0; idx < total_size; idx++) {
                kol_enter[idx] += local_kol_enter_add[idx];
                pheromon[idx] += local_pheromon_add[idx];
            }
        }

        free(local_pheromon_add);
        free(local_kol_enter_add);
    }
}
#if _OPENMP >= 201307  // OpenMP 4.0+
void add_pheromon_iteration_AVX_OMP_non_cuda_4_0(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);
    const size_t total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int remainder = total_size % CONST_AVX;

    // 1. Испарение феромонов с SIMD
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < total_size - remainder; i += CONST_AVX) {
        __m256d pheromonValues = _mm256_loadu_pd(&pheromon[i]);
        pheromonValues = _mm256_mul_pd(pheromonValues, parametRovector_AVX);
        _mm256_storeu_pd(&pheromon[i], pheromonValues);
    }

    // Обработка остаточных элементов
    if (remainder > 0) {
#pragma omp parallel for schedule(static)
        for (int i = total_size - remainder; i < total_size; i++) {
            pheromon[i] *= PARAMETR_RO;
        }
    }

    // 2. Добавление феромонов с оптимизированными thread-local буферами
#pragma omp parallel
    {
        double* local_pheromon_add = (double*)calloc(total_size, sizeof(double));
        int* local_kol_enter_add = (int*)calloc(total_size, sizeof(int));

#pragma omp for nowait
        for (int i = 0; i < ANT_SIZE; i++) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            const double min1_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT > agent_of) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            const double min2_value = PARAMETR_Q / ((agent_of == 0) ? 1e-7 : agent_of);
#elif OPTIMIZE_MAX
            const double max_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                int k = agent_path[tx];
                if (k >= 0 && k < MAX_VALUE_SIZE) {
                    int idx = MAX_VALUE_SIZE * tx + k;
                    local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_1
                    if (min1_value > 0) local_pheromon_add[idx] += min1_value;
#elif OPTIMIZE_MIN_2
                    local_pheromon_add[idx] += min2_value;
#elif OPTIMIZE_MAX
                    local_pheromon_add[idx] += max_value;
#endif
                }
            }
        }

        // ВСЕ потоки работают параллельно
#pragma omp parallel for
        for (int idx = 0; idx < total_size; idx++) {
#pragma omp atomic
            kol_enter[idx] += local_kol_enter_add[idx];
#pragma omp atomic  
            pheromon[idx] += local_pheromon_add[idx];
        }

        free(local_pheromon_add);
        free(local_kol_enter_add);
    }
}
#endif
#if _OPENMP >= 201511  // OpenMP 4.5+
void add_pheromon_iteration_AVX_OMP_non_cuda_4_5(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);
    const size_t total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int remainder = total_size % CONST_AVX;

    // 1. Испарение феромонов с if clause
#if _OPENMP >= 201511 && !defined(__clang__)
#pragma omp parallel for simd schedule(static) if(total_size > 1000)
#else
#pragma omp parallel for simd schedule(static)
#endif
    for (int i = 0; i < total_size - remainder; i += CONST_AVX) {
        __m256d pheromonValues = _mm256_loadu_pd(&pheromon[i]);
        pheromonValues = _mm256_mul_pd(pheromonValues, parametRovector_AVX);
        _mm256_storeu_pd(&pheromon[i], pheromonValues);
    }

    // Обработка остаточных элементов
    if (remainder > 0) {
#if _OPENMP >= 201511 && !defined(__clang__)
#pragma omp parallel for schedule(static) if(remainder > 10)
#else
#pragma omp parallel for schedule(static)
#endif

        for (int i = total_size - remainder; i < total_size; i++) {
            pheromon[i] *= PARAMETR_RO;
        }
    }

    // 2. Добавление феромонов с улучшенным планированием
#pragma omp parallel
    {
        double* local_pheromon_add = (double*)calloc(total_size, sizeof(double));
        int* local_kol_enter_add = (int*)calloc(total_size, sizeof(int));
#if defined(__clang__)
#pragma omp for schedule(static) nowait
#else
#pragma omp for schedule(static) nowait // if(ANT_SIZE > 100)
#endif
        for (int i = 0; i < ANT_SIZE; i++) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            const double min1_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT > agent_of) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            const double min2_value = PARAMETR_Q / ((agent_of == 0) ? 1e-7 : agent_of);
#elif OPTIMIZE_MAX
            const double max_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                int k = agent_path[tx];
                if (k >= 0 && k < MAX_VALUE_SIZE) {
                    int idx = MAX_VALUE_SIZE * tx + k;
                    local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_1
                    if (min1_value > 0) local_pheromon_add[idx] += min1_value;
#elif OPTIMIZE_MIN_2
                    local_pheromon_add[idx] += min2_value;
#elif OPTIMIZE_MAX
                    local_pheromon_add[idx] += max_value;
#endif
                }
            }
        }

        // ВСЕ потоки работают параллельно
#pragma omp parallel for
        for (int idx = 0; idx < total_size; idx++) {
#pragma omp atomic
            kol_enter[idx] += local_kol_enter_add[idx];
#pragma omp atomic  
            pheromon[idx] += local_pheromon_add[idx];
        }

        free(local_pheromon_add);
        free(local_kol_enter_add);
    }
}
#endif
#if _OPENMP >= 201811  // OpenMP 5.0+
void add_pheromon_iteration_AVX_OMP_non_cuda_5_0(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);
    const size_t total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int remainder = total_size % CONST_AVX;

    // 1. Испарение феромонов с loop трансформацией
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < total_size - remainder; i += CONST_AVX) {
        __m256d pheromonValues = _mm256_loadu_pd(&pheromon[i]);
        pheromonValues = _mm256_mul_pd(pheromonValues, parametRovector_AVX);
        _mm256_storeu_pd(&pheromon[i], pheromonValues);
    }

    // Обработка остаточных элементов
    if (remainder > 0) {
#pragma omp parallel for schedule(static)
        for (int i = total_size - remainder; i < total_size; i++) {
            pheromon[i] *= PARAMETR_RO;
        }
    }

    // 2. Добавление феромонов с nonmonotonic scheduling
#pragma omp parallel
    {
        double* local_pheromon_add = (double*)calloc(total_size, sizeof(double));
        int* local_kol_enter_add = (int*)calloc(total_size, sizeof(int));

        // Инициализация буферов
        for (int idx = 0; idx < total_size; idx++) {
            local_pheromon_add[idx] = 0.0;
            local_kol_enter_add[idx] = 0;
        }

        // OpenMP 5.0: nonmonotonic scheduling
#pragma omp for //schedule(nonmonotonic:static) nowait
        for (int i = 0; i < ANT_SIZE; i++) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            const double min1_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT > agent_of) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            const double min2_value = PARAMETR_Q / ((agent_of == 0) ? 1e-7 : agent_of);
#elif OPTIMIZE_MAX
            const double max_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                int k = agent_path[tx];
                if (k >= 0 && k < MAX_VALUE_SIZE) {
                    int idx = MAX_VALUE_SIZE * tx + k;
                    local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_1
                    if (min1_value > 0) local_pheromon_add[idx] += min1_value;
#elif OPTIMIZE_MIN_2
                    local_pheromon_add[idx] += min2_value;
#elif OPTIMIZE_MAX
                    local_pheromon_add[idx] += max_value;
#endif
                }
            }
        }

        // ВСЕ потоки работают параллельно
#pragma omp parallel for
        for (int idx = 0; idx < total_size; idx++) {
#pragma omp atomic
            kol_enter[idx] += local_kol_enter_add[idx];
#pragma omp atomic  
            pheromon[idx] += local_pheromon_add[idx];
        }

        free(local_pheromon_add);
        free(local_kol_enter_add);
    }
}
#endif
#if _OPENMP >= 202011  // OpenMP 5.1+
void add_pheromon_iteration_AVX_OMP_non_cuda_5_1(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);
    const size_t total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int remainder = total_size % CONST_AVX;

    // 1. Испарение феромонов с неблокирующими операциями
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < total_size - remainder; i += CONST_AVX) {
        __m256d pheromonValues = _mm256_loadu_pd(&pheromon[i]);
        pheromonValues = _mm256_mul_pd(pheromonValues, parametRovector_AVX);
        _mm256_storeu_pd(&pheromon[i], pheromonValues);
    }

    // Обработка остаточных элементов
    if (remainder > 0) {
#pragma omp parallel for schedule(static)
        for (int i = total_size - remainder; i < total_size; i++) {
            pheromon[i] *= PARAMETR_RO;
        }
    }

    // 2. Добавление феромонов с error recovery features
#pragma omp parallel
    {
        double* local_pheromon_add = (double*)calloc(total_size, sizeof(double));
        int* local_kol_enter_add = (int*)calloc(total_size, sizeof(int));

        // Инициализация буферов
        for (int idx = 0; idx < total_size; idx++) {
            local_pheromon_add[idx] = 0.0;
            local_kol_enter_add[idx] = 0;
        }

        // OpenMP 5.1: улучшенное планирование
#pragma omp for //schedule(nonmonotonic:static) nowait
        for (int i = 0; i < ANT_SIZE; i++) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            const double min1_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT > agent_of) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            const double min2_value = PARAMETR_Q / ((agent_of == 0) ? 1e-7 : agent_of);
#elif OPTIMIZE_MAX
            const double max_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                int k = agent_path[tx];
                if (k >= 0 && k < MAX_VALUE_SIZE) {
                    int idx = MAX_VALUE_SIZE * tx + k;
                    local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_1
                    if (min1_value > 0) local_pheromon_add[idx] += min1_value;
#elif OPTIMIZE_MIN_2
                    local_pheromon_add[idx] += min2_value;
#elif OPTIMIZE_MAX
                    local_pheromon_add[idx] += max_value;
#endif
                }
            }
        }

        // ВСЕ потоки работают параллельно
#pragma omp parallel for
        for (int idx = 0; idx < total_size; idx++) {
#pragma omp atomic
            kol_enter[idx] += local_kol_enter_add[idx];
#pragma omp atomic  
            pheromon[idx] += local_pheromon_add[idx];
        }

        free(local_pheromon_add);
        free(local_kol_enter_add);
    }
}
#endif
#if _OPENMP >= 202111  // OpenMP 5.2+
void add_pheromon_iteration_AVX_OMP_non_cuda_5_2(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);
    const size_t total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int remainder = total_size % CONST_AVX;

    // 1. Испарение феромонов с assume clauses
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < total_size - remainder; i += CONST_AVX) {
#if !defined(__clang__)
#pragma omp assume aligned(pheromon:32)
#endif
        __m256d pheromonValues = _mm256_load_pd(&pheromon[i]);
        pheromonValues = _mm256_mul_pd(pheromonValues, parametRovector_AVX);
        _mm256_store_pd(&pheromon[i], pheromonValues);
    }

    // Обработка остаточных элементов
    if (remainder > 0) {
#if !defined(__clang__)
#pragma omp assume (remainder < CONST_AVX)
#endif
#pragma omp parallel for schedule(static)
        for (int i = total_size - remainder; i < total_size; i++) {
            pheromon[i] *= PARAMETR_RO;
        }
    }

    // 2. Добавление феромонов с latest OpenMP 5.2 features
#pragma omp parallel
    {
        // OpenMP 5.2: aligned allocation
        double* local_pheromon_add = (double*)ALIGNED_ALLOC(32, total_size * sizeof(double));
        int* local_kol_enter_add = (int*)ALIGNED_ALLOC(32, total_size * sizeof(int));

        // Инициализация буферов
#if !defined(__clang__)
#pragma omp assume (total_size > 0)
#endif
        for (int idx = 0; idx < total_size; idx++) {
            local_pheromon_add[idx] = 0.0;
            local_kol_enter_add[idx] = 0;
        }

        // OpenMP 5.2: assume clauses для оптимизатора
#if !defined(__clang__)
#pragma omp assume (MAX_VALUE_SIZE > 0)
#pragma omp assume (PARAMETR_SIZE > 0)
#endif
#pragma omp for schedule(static) nowait
        for (int i = 0; i < ANT_SIZE; i++) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            const double min1_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT > agent_of) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            const double min2_value = PARAMETR_Q / ((agent_of == 0) ? 1e-7 : agent_of);
#elif OPTIMIZE_MAX
            const double max_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];

            // OpenMP 5.2: assume для лучшей оптимизации
#if !defined(__clang__)
#pragma omp assume (PARAMETR_SIZE <= 1000)
#endif
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                int k = agent_path[tx];
                if (k >= 0 && k < MAX_VALUE_SIZE) {
                    int idx = MAX_VALUE_SIZE * tx + k;
                    local_kol_enter_add[idx]++;

#if OPTIMIZE_MIN_1
                    if (min1_value > 0) local_pheromon_add[idx] += min1_value;
#elif OPTIMIZE_MIN_2
                    local_pheromon_add[idx] += min2_value;
#elif OPTIMIZE_MAX
                    local_pheromon_add[idx] += max_value;
#endif
                }
            }
        }

        // ВСЕ потоки работают параллельно
#pragma omp parallel for
        for (int idx = 0; idx < total_size; idx++) {
#pragma omp atomic
            kol_enter[idx] += local_kol_enter_add[idx];
#pragma omp atomic  
            pheromon[idx] += local_pheromon_add[idx];
        }

        ALIGNED_FREE(local_pheromon_add);
        ALIGNED_FREE(local_kol_enter_add);
    }
}
#endif
void add_pheromon_iteration_AVX_OMP_non_cuda(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
#if _OPENMP >= 202111  // OpenMP 5.2+
    add_pheromon_iteration_AVX_OMP_non_cuda_5_2(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 202011  // OpenMP 5.1+
    add_pheromon_iteration_AVX_OMP_non_cuda_5_1(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 201811  // OpenMP 5.0+
    add_pheromon_iteration_AVX_OMP_non_cuda_5_0(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 201511  // OpenMP 4.5+
    add_pheromon_iteration_AVX_OMP_non_cuda_4_5(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 201307  // OpenMP 4.0+
    add_pheromon_iteration_AVX_OMP_non_cuda_4_0(pheromon, kol_enter, agent_node, OF);
#else  // OpenMP 2.0/3.0/3.1
    add_pheromon_iteration_AVX_OMP_non_cuda_2_0(pheromon, kol_enter, agent_node, OF);
#endif
}
// Базовая версия OpenMP 2.0/3.0/3.1
void add_pheromon_iteration_AVX_OMP_non_cuda_4_2_0(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const size_t total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // 1. Испарение феромонов
#pragma omp parallel for schedule(static)
    for (int i = 0; i < total_size; i += CONST_AVX) {
        __m256d pheromon_vec = _mm256_loadu_pd(&pheromon[i]);
        _mm256_storeu_pd(&pheromon[i], _mm256_mul_pd(pheromon_vec, _mm256_set1_pd(PARAMETR_RO)));
    }

    // 2. Fallback для старых версий OpenMP
#pragma omp parallel
    {
        double* local_pheromon = (double*)calloc(total_size, sizeof(double));
        int* local_kol_enter = (int*)calloc(total_size, sizeof(int));

#pragma omp for nowait
        for (int i = 0; i < ANT_SIZE; i++) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            double add_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT > agent_of) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            double add_value = PARAMETR_Q / ((agent_of == 0) ? 1e-7 : agent_of);
#elif OPTIMIZE_MAX
            double add_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];

            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                int k = agent_path[tx];
                if (k >= 0 && k < MAX_VALUE_SIZE) {
                    int idx = MAX_VALUE_SIZE * tx + k;
                    local_kol_enter[idx]++;
#if OPTIMIZE_MIN_1
                    if (add_value > 0) local_pheromon[idx] += add_value;
#else
                    local_pheromon[idx] += add_value;
#endif
                }
            }
        }

#pragma omp critical
        {
            for (int idx = 0; idx < total_size; idx++) {
                kol_enter[idx] += local_kol_enter[idx];
                pheromon[idx] += local_pheromon[idx];
            }
        }

        free(local_pheromon);
        free(local_kol_enter);
    }
}
#if _OPENMP >= 201307  // OpenMP 4.0+
void add_pheromon_iteration_AVX_OMP_non_cuda_4_4_0(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const size_t total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // 1. Испарение феромонов с SIMD
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < total_size; i += CONST_AVX) {
        __m256d pheromon_vec = _mm256_loadu_pd(&pheromon[i]);
        _mm256_storeu_pd(&pheromon[i], _mm256_mul_pd(pheromon_vec, _mm256_set1_pd(PARAMETR_RO)));
    }

    // 2. Thread-local буферы с оптимизациями OpenMP 4.0
#pragma omp parallel
    {
        double* local_pheromon = (double*)calloc(total_size, sizeof(double));
        int* local_kol_enter = (int*)calloc(total_size, sizeof(int));

#pragma omp for nowait
        for (int i = 0; i < ANT_SIZE; i++) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            double add_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT > agent_of) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            double add_value = PARAMETR_Q / ((agent_of == 0) ? 1e-7 : agent_of);
#elif OPTIMIZE_MAX
            double add_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];

            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                int k = agent_path[tx];
                if (k >= 0 && k < MAX_VALUE_SIZE) {
                    int idx = MAX_VALUE_SIZE * tx + k;
                    local_kol_enter[idx]++;
#if OPTIMIZE_MIN_1
                    if (add_value > 0) local_pheromon[idx] += add_value;
#else
                    local_pheromon[idx] += add_value;
#endif
                }
            }
        }

        for (int idx = 0; idx < total_size; idx++) {
#pragma omp atomic
            kol_enter[idx] += local_kol_enter[idx];
#pragma omp atomic  
            pheromon[idx] += local_pheromon[idx];
        }

        free(local_pheromon);
        free(local_kol_enter);
    }
}
#endif
#if _OPENMP >= 201511  // OpenMP 4.5+
void add_pheromon_iteration_AVX_OMP_non_cuda_4_4_5(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const size_t total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // 1. Испарение феромонов с if clause
#if _OPENMP >= 201511 && !defined(__clang__)
#pragma omp parallel for simd schedule(static) if(total_size > 1000)
#else
#pragma omp parallel for simd schedule(static)
#endif
    for (int i = 0; i < total_size; i += CONST_AVX) {
        __m256d pheromon_vec = _mm256_loadu_pd(&pheromon[i]);
        _mm256_storeu_pd(&pheromon[i], _mm256_mul_pd(pheromon_vec, _mm256_set1_pd(PARAMETR_RO)));
    }

    // 2. Thread-local буферы с улучшенным планированием
#pragma omp parallel
    {
        double* local_pheromon = (double*)calloc(total_size, sizeof(double));
        int* local_kol_enter = (int*)calloc(total_size, sizeof(int));

#if defined(__clang__)
#pragma omp for schedule(static) nowait
#else
#pragma omp for schedule(static) nowait // if(ANT_SIZE > 100)
#endif
        for (int i = 0; i < ANT_SIZE; i++) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            double add_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT > agent_of) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            double add_value = PARAMETR_Q / ((agent_of == 0) ? 1e-7 : agent_of);
#elif OPTIMIZE_MAX
            double add_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];

            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                int k = agent_path[tx];
                if (k >= 0 && k < MAX_VALUE_SIZE) {
                    int idx = MAX_VALUE_SIZE * tx + k;
                    local_kol_enter[idx]++;
#if OPTIMIZE_MIN_1
                    if (add_value > 0) local_pheromon[idx] += add_value;
#else
                    local_pheromon[idx] += add_value;
#endif
                }
            }
        }

        for (int idx = 0; idx < total_size; idx++) {
#pragma omp atomic
            kol_enter[idx] += local_kol_enter[idx];
#pragma omp atomic  
            pheromon[idx] += local_pheromon[idx];
        }

        free(local_pheromon);
        free(local_kol_enter);
    }
}
#endif
#if _OPENMP >= 201811  // OpenMP 5.0+
void add_pheromon_iteration_AVX_OMP_non_cuda_4_5_0(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const size_t total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // 1. Испарение феромонов
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < total_size; i += CONST_AVX) {
        __m256d pheromon_vec = _mm256_loadu_pd(&pheromon[i]);
        _mm256_storeu_pd(&pheromon[i], _mm256_mul_pd(pheromon_vec, _mm256_set1_pd(PARAMETR_RO)));
    }

    // 2. Array reduction (OpenMP 5.0+)
    // Исправлено: правильное использование array reduction
#pragma omp parallel
    {
        // Локальные массивы для каждого потока
        double* local_pheromon = (double*)calloc(total_size, sizeof(double));
        int* local_kol_enter = (int*)calloc(total_size, sizeof(int));

#pragma omp for nowait
        for (int i = 0; i < ANT_SIZE; i++) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            double add_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT > agent_of) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            double add_value = PARAMETR_Q / ((agent_of == 0) ? 1e-7 : agent_of);
#elif OPTIMIZE_MAX
            double add_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];

            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                int k = agent_path[tx];
                if (k >= 0 && k < MAX_VALUE_SIZE) {
                    int idx = MAX_VALUE_SIZE * tx + k;
                    local_kol_enter[idx]++;
#if OPTIMIZE_MIN_1
                    if (add_value > 0) local_pheromon[idx] += add_value;
#else
                    local_pheromon[idx] += add_value;
#endif
                }
            }
        }

        // Исправлено: использование array reduction для объединения результатов
#pragma omp for simd
        for (int idx = 0; idx < total_size; idx++) {
#pragma omp atomic
            kol_enter[idx] += local_kol_enter[idx];
#pragma omp atomic
            pheromon[idx] += local_pheromon[idx];
        }

        free(local_pheromon);
        free(local_kol_enter);
    }
}
#endif
#if _OPENMP >= 202011  // OpenMP 5.1+
void add_pheromon_iteration_AVX_OMP_non_cuda_4_5_1(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const size_t total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // 1. Испарение феромонов
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < total_size; i += CONST_AVX) {
        __m256d pheromon_vec = _mm256_loadu_pd(&pheromon[i]);
        _mm256_storeu_pd(&pheromon[i], _mm256_mul_pd(pheromon_vec, _mm256_set1_pd(PARAMETR_RO)));
    }

    // 2. Улучшенная версия с nonmonotonic scheduling (OpenMP 5.1+)
#pragma omp parallel
    {
        double* local_pheromon = (double*)calloc(total_size, sizeof(double));
        int* local_kol_enter = (int*)calloc(total_size, sizeof(int));

#if defined(__clang__)
#pragma omp for schedule(static) nowait
#elif
#pragma omp for schedule(nonmonotonic: static) nowait
#endif
        for (int i = 0; i < ANT_SIZE; i++) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            double add_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT > agent_of) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            double add_value = PARAMETR_Q / ((agent_of == 0) ? 1e-7 : agent_of);
#elif OPTIMIZE_MAX
            double add_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];

            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                int k = agent_path[tx];
                if (k >= 0 && k < MAX_VALUE_SIZE) {
                    int idx = MAX_VALUE_SIZE * tx + k;
                    local_kol_enter[idx]++;
#if OPTIMIZE_MIN_1
                    if (add_value > 0) local_pheromon[idx] += add_value;
#else
                    local_pheromon[idx] += add_value;
#endif
                }
            }
        }

        // Исправлено: эффективное объединение результатов
#pragma omp for simd
        for (int idx = 0; idx < total_size; idx++) {
#pragma omp atomic
            kol_enter[idx] += local_kol_enter[idx];
#pragma omp atomic
            pheromon[idx] += local_pheromon[idx];
        }

        free(local_pheromon);
        free(local_kol_enter);
    }
}
#endif
#if _OPENMP >= 202111  // OpenMP 5.2+
void add_pheromon_iteration_AVX_OMP_non_cuda_4_5_2(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const size_t total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;

    // 1. Испарение феромонов с assume clauses
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < total_size; i += CONST_AVX) {
#if !defined(__clang__) && defined(__INTEL_COMPILER)
        // assume aligned только для компиляторов, которые это поддерживают
#pragma omp assume aligned(pheromon:32)
        __m256d pheromon_vec = _mm256_load_pd(&pheromon[i]);
#else
        __m256d pheromon_vec = _mm256_loadu_pd(&pheromon[i]);
#endif
        _mm256_storeu_pd(&pheromon[i], _mm256_mul_pd(pheromon_vec, _mm256_set1_pd(PARAMETR_RO)));
    }

    // 2. Оптимизированная версия с assume для OpenMP 5.2
#pragma omp parallel
    {
        double* local_pheromon = (double*)calloc(total_size, sizeof(double));
        int* local_kol_enter = (int*)calloc(total_size, sizeof(int));

#if !defined(__clang__) && defined(__INTEL_COMPILER)
#pragma omp assume (MAX_VALUE_SIZE == 4)
#pragma omp assume (total_size % 4 == 0)
#endif

#pragma omp for schedule(static) nowait
        for (int i = 0; i < ANT_SIZE; i++) {
            const double agent_of = OF[i];

#if OPTIMIZE_MIN_1
            double add_value = (MAX_PARAMETR_VALUE_TO_MIN_OPT > agent_of) ? PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of) : 0.0;
#elif OPTIMIZE_MIN_2
            double add_value = PARAMETR_Q / ((agent_of == 0) ? 1e-7 : agent_of);
#elif OPTIMIZE_MAX
            double add_value = PARAMETR_Q * agent_of;
#endif

            const int* agent_path = &agent_node[i * PARAMETR_SIZE];

            // OpenMP 5.2: assume для лучшей оптимизации
#if !defined(__clang__) && defined(__INTEL_COMPILER)
#pragma omp assume PARAMETR_SIZE <= 100
#endif
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                int k = agent_path[tx];
                if (k >= 0 && k < MAX_VALUE_SIZE) {
                    int idx = MAX_VALUE_SIZE * tx + k;
                    local_kol_enter[idx]++;
#if OPTIMIZE_MIN_1
                    if (add_value > 0) local_pheromon[idx] += add_value;
#else
                    local_pheromon[idx] += add_value;
#endif
                }
            }
        }

        // Исправлено: эффективное объединение с SIMD
#pragma omp for simd
        for (int idx = 0; idx < total_size; idx++) {
#pragma omp atomic
            kol_enter[idx] += local_kol_enter[idx];
#pragma omp atomic
            pheromon[idx] += local_pheromon[idx];
        }

        free(local_pheromon);
        free(local_kol_enter);
    }
}
#endif
// Универсальная функция с автоматическим выбором версии
void add_pheromon_iteration_AVX_OMP_non_cuda_4(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
#if _OPENMP >= 202111  // OpenMP 5.2+
    add_pheromon_iteration_AVX_OMP_non_cuda_4_5_2(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 202011  // OpenMP 5.1+
    add_pheromon_iteration_AVX_OMP_non_cuda_4_5_1(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 201811  // OpenMP 5.0+
    add_pheromon_iteration_AVX_OMP_non_cuda_4_5_0(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 201511  // OpenMP 4.5+
    add_pheromon_iteration_AVX_OMP_non_cuda_4_4_5(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 201307  // OpenMP 4.0+
    add_pheromon_iteration_AVX_OMP_non_cuda_4_4_0(pheromon, kol_enter, agent_node, OF);
#else  // OpenMP 2.0/3.0/3.1
    add_pheromon_iteration_AVX_OMP_non_cuda_4_2_0(pheromon, kol_enter, agent_node, OF);
#endif
}int start_NON_CUDA_AVX_time() {
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
        go_mass_probability_AVX_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

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
        add_pheromon_iteration_AVX_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

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
    std::cout << "Time non CUDA_AVX_time;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA_AVX_time;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}
int start_NON_CUDA_AVX() {
    auto start = std::chrono::high_resolution_clock::now();
    double SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumTimeHashTotal = 0.0f, SumTimeOF = 0.0f, SumTimeHashSearch = 0.0f, SumTimeHashSave = 0.0f;
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
        go_mass_probability_AVX_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

        // Вычисление пути агентов
        auto end_temp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> current_time = end_temp - start;
        go_all_agent_non_cuda(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail);

        // Обновление весов-феромонов
        add_pheromon_iteration_AVX_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

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
    std::cout << "Time non CUDA AVX;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA AVX;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}
int start_NON_CUDA_AVX_non_hash() {
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
        go_mass_probability_AVX_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

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
        add_pheromon_iteration_AVX_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

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
    std::cout << "Time non CUDA AVX non hash;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << "0" << "; " << SumgpuTime5 << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA AVX non hash:;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << "0" << "; " << SumgpuTime5 << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    return 0;
}

int start_NON_CUDA_AVX_OMP_time() {
    auto start = std::chrono::high_resolution_clock::now();
    double SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumgpuTime4 = 0.0f, SumgpuTime5 = 0.0f, SumTimeHashTotal = 0.0f, SumTimeOF = 0.0f, SumTimeHashSearch = 0.0f, SumTimeHashSave = 0.0f, SumTimeSearchAgent = 0.0f;
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
        go_mass_probability_AVX_OMP_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

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
        add_pheromon_iteration_AVX_OMP_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

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
    std::cout << "Time non CUDA_AVX_OMP_time;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA_AVX_OMP_time;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}
int start_NON_CUDA_AVX_OMP() {
    auto start = std::chrono::high_resolution_clock::now();
    double SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumgpuTime4 = 0.0f, SumgpuTime5 = 0.0f, SumTimeHashTotal = 0.0f, SumTimeOF = 0.0f, SumTimeHashSearch = 0.0f, SumTimeHashSave = 0.0f;
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
        go_mass_probability_AVX_OMP_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

        // Вычисление пути агентов
        auto end_temp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> current_time = end_temp - start;
        go_all_agent_omp(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail, SumgpuTime4, SumgpuTime5);

        // Обновление весов-феромонов
        add_pheromon_iteration_AVX_OMP_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

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
    std::cout << "Time non CUDA AVX OMP;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA AVX OMP;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}
int start_NON_CUDA_AVX_OMP_non_hash() {
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
        go_mass_probability_AVX_OMP_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

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
        go_all_agent_omp_non_hash(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, kol_hash_fail, SumgpuTime4, SumgpuTime5);

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
        add_pheromon_iteration_AVX_OMP_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

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
    std::cout << "Time non CUDA AVX OMP non hash;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << "0" << "; " << SumgpuTime5 << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA AVX OMP non hash:;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << "0" << "; " << SumgpuTime5 << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    return 0;
}

void go_all_agent_non_cuda_time_4(int gpuTime, double* __restrict parametr, double* __restrict norm_matrix_probability, double* __restrict agent, int* __restrict agent_node, double* __restrict OF, HashEntry* __restrict hashTable, int& kol_hash_fail, double& totalHashTime, double& totalOFTime, double& HashTimeSave, double& HashTimeSearch, double& SumTimeSearch) {
    // Генератор случайных чисел
    std::default_random_engine generator(123 + gpuTime); // Используем gpuTime как начальное значение
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int bx = 0; bx < ANT_SIZE; bx++) { // Проходим по всем агентам
        auto start_ant = std::chrono::high_resolution_clock::now();
        bool go_4 = true;
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            int k = 0;
            if (go_4) {
                double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
                // Определение номера значения
                while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                    k++;
                }
            }
            // Запись подматрицы блока в глобальную память
            agent_node[bx * PARAMETR_SIZE + tx] = k;
            agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
            go_4 = (k != MAX_VALUE_SIZE - 1);
        }
        auto end_ant = std::chrono::high_resolution_clock::now();
        SumTimeSearch += std::chrono::duration<double, std::milli>(end_ant - start_ant).count();
        auto start = std::chrono::high_resolution_clock::now();
        // Проверка наличия решения в Хэш-таблице
        double cachedResult = getCachedResultOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx);
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
            saveToCacheOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx, OF[bx]);
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
                    cachedResult = getCachedResultOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx);
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
                saveToCacheOptimized_non_cuda(hashTable, &agent_node[bx * PARAMETR_SIZE], bx, OF[bx]);
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
void go_all_agent_OMP_non_cuda_time_4(int gpuTime, double* __restrict parametr, double* __restrict norm_matrix_probability, double* __restrict agent, int* __restrict agent_node, double* __restrict OF, HashEntry* __restrict hashTable, int& kol_hash_fail, double& totalHashTime, double& totalOFTime, double& HashTimeSave, double& HashTimeSearch, double& SumTimeSearch) {
    int local_kol_hash_fail = 0;
    double local_totalHashTime = 0.0;
    double local_totalOFTime = 0.0;
    double local_HashTimeSave = 0.0;
    double local_HashTimeSearch = 0.0;
    double local_SumTimeSearch = 0.0;

    // Выбор директивы parallel в зависимости от версии OpenMP
#if _OPENMP >= 201511  // OpenMP 4.5+
#if defined(__clang__)
#pragma omp parallel reduction(+:local_kol_hash_fail, local_totalHashTime, local_totalOFTime, local_HashTimeSave, local_HashTimeSearch, local_SumTimeSearch)
#else
#pragma omp parallel reduction(+:local_kol_hash_fail, local_totalHashTime, local_totalOFTime, local_HashTimeSave, local_HashTimeSearch, local_SumTimeSearch) // if(ANT_SIZE > 100)
#endif
#else
#pragma omp parallel reduction(+:local_kol_hash_fail, local_totalHashTime, local_totalOFTime, local_HashTimeSave, local_HashTimeSearch, local_SumTimeSearch)
#endif
    {
        uint64_t seed = 123 + gpuTime + omp_get_thread_num();

        // Выбор директивы for в зависимости от версии OpenMP
#if _OPENMP >= 201811 && !defined(__clang__) // OpenMP 5.0+
#pragma omp for schedule(nonmonotonic:static)
#elif _OPENMP >= 201511  // OpenMP 4.5+
#if defined(__clang__)
#pragma omp for schedule(static)
#else
#pragma omp for schedule(static) // if(ANT_SIZE > 100)
#endif
#else
#pragma omp for schedule(static)
#endif
        for (int bx = 0; bx < ANT_SIZE; bx++) {
            // OpenMP 5.2: assume clauses для оптимизатора
#if _OPENMP >= 202111 && !defined(__clang__)  // OpenMP 5.2+
#pragma omp assume (MAX_VALUE_SIZE == 4)
#pragma omp assume (PARAMETR_SIZE <= 100)
#endif

            const int agent_base = bx * PARAMETR_SIZE;
            int* current_agent_node = &agent_node[agent_base];
            double* current_agent = &agent[agent_base];

            auto start_ant = std::chrono::high_resolution_clock::now();

            // Упрощенная генерация пути (MAX_VALUE_SIZE = 4)
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                double randomValue = unified_fast_random(seed);
                const double* probs = &norm_matrix_probability[4 * tx];

                // Развернутый цикл для MAX_VALUE_SIZE = 4
                int k = 0;
                if (randomValue > probs[0]) k = 1;
                if (randomValue > probs[1]) k = 2;
                if (randomValue > probs[2]) k = 3;
                if (randomValue > probs[3]) k = 4;

                // Корректируем k если он вышел за границы
                if (k >= 4) k = 3;

                current_agent_node[tx] = k;
                current_agent[tx] = parametr[tx * 4 + k];
            }

            auto end_ant = std::chrono::high_resolution_clock::now();
            local_SumTimeSearch += std::chrono::duration<double, std::milli>(end_ant - start_ant).count();

            auto start_hash = std::chrono::high_resolution_clock::now();

            double cachedResult = -1.0;

            // Критическая секция для поиска в хеш-таблице
#pragma omp critical(hash_lookup)
            {
                cachedResult = getCachedResultOptimized_non_cuda(hashTable, current_agent_node, bx);
            }

            auto end_hash = std::chrono::high_resolution_clock::now();
            local_HashTimeSearch += std::chrono::duration<double, std::milli>(end_hash - start_hash).count();

            if (cachedResult == -1.0) {
                auto start_of = std::chrono::high_resolution_clock::now();
                OF[bx] = BenchShafferaFunction_non_cuda(current_agent);
                auto end_of = std::chrono::high_resolution_clock::now();
                local_totalOFTime += std::chrono::duration<double, std::milli>(end_of - start_of).count();

                auto start_save = std::chrono::high_resolution_clock::now();
#pragma omp critical(hash_save)
                {
                    saveToCacheOptimized_non_cuda(hashTable, current_agent_node, bx, OF[bx]);
                }
                auto end_save = std::chrono::high_resolution_clock::now();
                local_HashTimeSave += std::chrono::duration<double, std::milli>(end_save - start_save).count();
            }
            else {
                local_kol_hash_fail++;

                switch (TYPE_ACO) {
                case 0: // ACOCN
                    OF[bx] = cachedResult;
                    break;

                case 1: // ACOCNI
                    OF[bx] = ZERO_HASH_RESULT;
                    break;

                case 2: // ACOCCyN
                {
                    double currentCachedResult = cachedResult;
                    int nom_iteration = 0;

                    while ((currentCachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION)) {
                        auto start_cycle = std::chrono::high_resolution_clock::now();

                        // Упрощенная регенерация пути
#if _OPENMP >= 202111 && !defined(__clang__)  // OpenMP 5.2+
#pragma omp assume (PARAMETR_SIZE <= 100)
#endif
                        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                            double randomValue = unified_fast_random(seed);
                            const double* probs = &norm_matrix_probability[4 * tx];

                            int k = 0;
                            if (randomValue > probs[0]) k = 1;
                            if (randomValue > probs[1]) k = 2;
                            if (randomValue > probs[2]) k = 3;
                            if (randomValue > probs[3]) k = 4;
                            if (k >= 4) k = 3;

                            current_agent_node[tx] = k;
                            current_agent[tx] = parametr[tx * 4 + k];
                        }

                        auto end_cycle = std::chrono::high_resolution_clock::now();
                        local_SumTimeSearch += std::chrono::duration<double, std::milli>(end_cycle - start_cycle).count();

                        auto start_hash_cycle = std::chrono::high_resolution_clock::now();
#pragma omp critical(hash_lookup)
                        {
                            currentCachedResult = getCachedResultOptimized_non_cuda(hashTable, current_agent_node, bx);
                        }
                        auto end_hash_cycle = std::chrono::high_resolution_clock::now();
                        local_HashTimeSearch += std::chrono::duration<double, std::milli>(end_hash_cycle - start_hash_cycle).count();

                        nom_iteration++;
                        local_kol_hash_fail++;
                    }

                    auto start_of_cycle = std::chrono::high_resolution_clock::now();
                    OF[bx] = BenchShafferaFunction_non_cuda(current_agent);
                    auto end_of_cycle = std::chrono::high_resolution_clock::now();
                    local_totalOFTime += std::chrono::duration<double, std::milli>(end_of_cycle - start_of_cycle).count();

                    auto start_save_cycle = std::chrono::high_resolution_clock::now();
#pragma omp critical(hash_save)
                    {
                        saveToCacheOptimized_non_cuda(hashTable, current_agent_node, bx, OF[bx]);
                    }
                    auto end_save_cycle = std::chrono::high_resolution_clock::now();
                    local_HashTimeSave += std::chrono::duration<double, std::milli>(end_save_cycle - start_save_cycle).count();
                }
                break;

                default:
                    OF[bx] = cachedResult;
                    break;
                }
            }

            auto end_total = std::chrono::high_resolution_clock::now();
            local_totalHashTime += std::chrono::duration<double, std::milli>(end_total - start_hash).count();
        }
    }

    kol_hash_fail += local_kol_hash_fail;
    totalHashTime += local_totalHashTime;
    totalOFTime += local_totalOFTime;
    HashTimeSave += local_HashTimeSave;
    HashTimeSearch += local_HashTimeSearch;
    SumTimeSearch += local_SumTimeSearch;
}
int start_NON_CUDA_AVX4_time() {
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
        go_mass_probability_AVX_non_cuda_4(pheromon_value, kol_enter_value, norm_matrix_probability);

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
        go_all_agent_non_cuda_time_4(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail, SumTimeHashTotal, SumTimeOF, SumTimeHashSearch, SumTimeHashSave, SumTimeSearchAgent);

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
        add_pheromon_iteration_AVX_OMP_non_cuda_4(pheromon_value, kol_enter_value, ant_parametr, antOF);

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
    std::cout << "Time non CUDA_AVX_time_4;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA_AVX_time_4;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}
int start_NON_CUDA_AVX4_OMP_time() {
    auto start = std::chrono::high_resolution_clock::now();
    double SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumTimeHashTotal = 0.0f, SumTimeOF = 0.0f, SumTimeHashSearch = 0.0f, SumTimeHashSave = 0.0f, SumTimeSearchAgent = 0.0f;
    double duration = 0.0f, duration_iteration = 0.0f;
    int kol_hash_fail = 0;
    const int kol_shag_stat = KOL_ITERATION / KOL_STAT_LEVEL;

    const int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    const int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
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
        go_mass_probability_AVX_OMP_non_cuda_4(pheromon_value, kol_enter_value, norm_matrix_probability);

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
        go_all_agent_OMP_non_cuda_time_4(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail, SumTimeHashTotal, SumTimeOF, SumTimeHashSearch, SumTimeHashSave, SumTimeSearchAgent);

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
        add_pheromon_iteration_AVX_OMP_non_cuda_4(pheromon_value, kol_enter_value, ant_parametr, antOF);

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
    std::cout << "Time non CUDA_AVX_OMP_time_4;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA_AVX_OMP_time_4;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}

// Подготовка массива для вероятностного поиска
void go_mass_probability_transp_AVX_non_cuda(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
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
        for (int tx = 0; tx < PARAMETR_SIZE; tx += CONST_AVX) {
            // Загружаем значения из pheromon_value и sumVectorT
            __m256d pheromon_values_AVX = _mm256_loadu_pd(&pheromon[tx + i * PARAMETR_SIZE]);
            __m256d sum_vector_AVX = _mm256_loadu_pd(&sumVectorT[tx]);

            // Выполняем деление
            __m256d pheromon_norm_values_AVX = _mm256_div_pd(pheromon_values_AVX, sum_vector_AVX);
            _mm256_storeu_pd(&pheromon_norm[tx + i * PARAMETR_SIZE], pheromon_norm_values_AVX);
        }
    }
    // Вычисление Z и P
    double* svertka = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx += CONST_AVX) {
            // Загружаем значения из kol_enter_value и pheromon_norm
            __m256d kol_enter_values_AVX = _mm256_loadu_pd(&kol_enter[tx + i * PARAMETR_SIZE]);
            __m256d pheromon_norm_values_AVX = _mm256_loadu_pd(&pheromon_norm[tx + i * PARAMETR_SIZE]);
            // Создаем маску для проверки условий (kol_enter_value[tx + i * PARAMETR_SIZE] != 0) && (pheromon_norm[tx + i * PARAMETR_SIZE] != 0)
            __m256d zero_vector_AVX = _mm256_setzero_pd();
            __m256d condition_mask_AVX = _mm256_and_pd(
                _mm256_cmp_pd(kol_enter_values_AVX, zero_vector_AVX, _CMP_NEQ_OQ),
                _mm256_cmp_pd(pheromon_norm_values_AVX, zero_vector_AVX, _CMP_NEQ_OQ)
            );
            // Вычисляем svertka
            __m256d one_vector_AVX = _mm256_set1_pd(1.0);
            __m256d svertka_values_AVX = _mm256_blendv_pd(
                zero_vector_AVX, _mm256_add_pd(_mm256_div_pd(one_vector_AVX, kol_enter_values_AVX), pheromon_norm_values_AVX),

                condition_mask_AVX
            );
            //__m256d svertka_values_AVX = _mm256_add_pd(_mm256_div_pd(one_vector_AVX, kol_enter_values_AVX), pheromon_norm_values_AVX);
            // Сохраняем значения в svertka
            _mm256_storeu_pd(&svertka[tx + i * PARAMETR_SIZE], svertka_values_AVX);
        }
    }
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            sumVectorZ[tx] += svertka[tx + i * PARAMETR_SIZE];
        }
    }
    // Вычисление F
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx += CONST_AVX) {
            __m256d svertka_values_AVX = _mm256_loadu_pd(&svertka[tx + i * PARAMETR_SIZE]);
            __m256d sum_vector_z_AVX = _mm256_loadu_pd(&sumVectorZ[tx]);
            __m256d norm_matrix_probability_AVX = _mm256_div_pd(svertka_values_AVX, sum_vector_z_AVX);
            if (i == 0) {
                // Нормализация для первой строки
                _mm256_storeu_pd(&norm_matrix_probability[tx + i * PARAMETR_SIZE], norm_matrix_probability_AVX);
            }
            else {
                // Нормализация для остальных строк
                __m256d previous_norm_values_AVX = _mm256_loadu_pd(&norm_matrix_probability[tx + (i - 1) * PARAMETR_SIZE]);
                __m256d norm_matrix_probability_values_AVX = _mm256_add_pd(norm_matrix_probability_AVX, previous_norm_values_AVX);
                _mm256_storeu_pd(&norm_matrix_probability[tx + i * PARAMETR_SIZE], norm_matrix_probability_values_AVX);
            }
        }
    }
    delete[] pheromon_norm;
    delete[] svertka;
}
// Базовая версия OpenMP 2.0/3.0/3.1
void go_mass_probability_transp_AVX_OMP_non_cuda_2_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    const int total_cells = MAX_VALUE_SIZE * PARAMETR_SIZE;

    // Выделяем память с выравниванием
    double* pheromon_norm = (double*)_mm_malloc(total_cells * sizeof(double), 32);
    double* svertka = (double*)_mm_malloc(total_cells * sizeof(double), 32);

    // Инициализация массивов
#pragma omp parallel for schedule(static)
    for (int i = 0; i < total_cells; i++) {
        pheromon_norm[i] = 0.0;
        svertka[i] = 0.0;
    }

    alignas(32) double sumVectorT[PARAMETR_SIZE] = { 0.0 };
    alignas(32) double sumVectorZ[PARAMETR_SIZE] = { 0.0 };

    // 1. Вычисление сумм T_i
#pragma omp parallel
    {
        alignas(32) double local_sumT[PARAMETR_SIZE] = { 0.0 };

#pragma omp for nowait
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                local_sumT[tx] += pheromon[tx + i * PARAMETR_SIZE];
            }
        }

#pragma omp critical
        {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                sumVectorT[tx] += local_sumT[tx];
            }
        }
    }

    // 2. Вычисление Tnorm
#pragma omp parallel for schedule(static)
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            int idx = tx + i * PARAMETR_SIZE;
            pheromon_norm[idx] = (sumVectorT[tx] != 0.0) ?
                pheromon[idx] / sumVectorT[tx] : 0.0;
        }
    }

    // 3. Вычисление svertka и сумм Z
#pragma omp parallel
    {
        alignas(32) double local_sumZ[PARAMETR_SIZE] = { 0.0 };

#pragma omp for nowait
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                int idx = tx + i * PARAMETR_SIZE;
                double val = 0.0;
                if (kol_enter[idx] != 0.0 && pheromon_norm[idx] != 0.0) {
                    val = 1.0 / kol_enter[idx] + pheromon_norm[idx];
                }
                svertka[idx] = val;
                local_sumZ[tx] += val;
            }
        }

#pragma omp critical
        {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                sumVectorZ[tx] += local_sumZ[tx];
            }
        }
    }

    // 4. Вычисление кумулятивных вероятностей
#pragma omp parallel for schedule(static)
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        if (sumVectorZ[tx] != 0.0) {
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                int idx = tx + i * PARAMETR_SIZE;
                double prob = svertka[idx] / sumVectorZ[tx];
                cumulative += prob;
                norm_matrix_probability[idx] = cumulative;
            }
            // Гарантируем, что последнее значение равно 1.0
            if (MAX_VALUE_SIZE > 0) {
                norm_matrix_probability[tx + (MAX_VALUE_SIZE - 1) * PARAMETR_SIZE] = 1.0;
            }
        }
        else {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                int idx = tx + i * PARAMETR_SIZE;
                cumulative += uniform_prob;
                norm_matrix_probability[idx] = cumulative;
            }
            if (MAX_VALUE_SIZE > 0) {
                norm_matrix_probability[tx + (MAX_VALUE_SIZE - 1) * PARAMETR_SIZE] = 1.0;
            }
        }
    }

    _mm_free(pheromon_norm);
    _mm_free(svertka);
}
#if _OPENMP >= 201307  // OpenMP 4.0+
void go_mass_probability_transp_AVX_OMP_non_cuda_4_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    const int total_cells = MAX_VALUE_SIZE * PARAMETR_SIZE;

    double* pheromon_norm = (double*)_mm_malloc(total_cells * sizeof(double), 32);
    double* svertka = (double*)_mm_malloc(total_cells * sizeof(double), 32);

    // Инициализация с SIMD
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < total_cells; i++) {
        pheromon_norm[i] = 0.0;
        svertka[i] = 0.0;
    }

    alignas(32) double sumVectorT[PARAMETR_SIZE] = { 0.0 };
    alignas(32) double sumVectorZ[PARAMETR_SIZE] = { 0.0 };

    // 1. Вычисление сумм T_i с оптимизациями OpenMP 4.0
#pragma omp parallel
    {
        alignas(32) double local_sumT[PARAMETR_SIZE] = { 0.0 };

#pragma omp for nowait
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                local_sumT[tx] += pheromon[tx + i * PARAMETR_SIZE];
            }
        }

#pragma omp critical
        {
#pragma omp simd
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                sumVectorT[tx] += local_sumT[tx];
            }
        }
    }

    // 2. Вычисление Tnorm с SIMD
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            int idx = tx + i * PARAMETR_SIZE;
            pheromon_norm[idx] = (sumVectorT[tx] != 0.0) ?
                pheromon[idx] / sumVectorT[tx] : 0.0;
        }
    }

    // 3. Вычисление svertka и сумм Z
#pragma omp parallel
    {
        alignas(32) double local_sumZ[PARAMETR_SIZE] = { 0.0 };

#pragma omp for nowait
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                int idx = tx + i * PARAMETR_SIZE;
                double val = 0.0;
                if (kol_enter[idx] != 0.0 && pheromon_norm[idx] != 0.0) {
                    val = 1.0 / kol_enter[idx] + pheromon_norm[idx];
                }
                svertka[idx] = val;
                local_sumZ[tx] += val;
            }
        }

#pragma omp critical
        {
#pragma omp simd
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                sumVectorZ[tx] += local_sumZ[tx];
            }
        }
    }

    // 4. Вычисление кумулятивных вероятностей
#pragma omp parallel for schedule(static)
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        if (sumVectorZ[tx] != 0.0) {
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                int idx = tx + i * PARAMETR_SIZE;
                double prob = svertka[idx] / sumVectorZ[tx];
                cumulative += prob;
                norm_matrix_probability[idx] = cumulative;
            }
            if (MAX_VALUE_SIZE > 0) {
                norm_matrix_probability[tx + (MAX_VALUE_SIZE - 1) * PARAMETR_SIZE] = 1.0;
            }
        }
        else {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                int idx = tx + i * PARAMETR_SIZE;
                cumulative += uniform_prob;
                norm_matrix_probability[idx] = cumulative;
            }
            if (MAX_VALUE_SIZE > 0) {
                norm_matrix_probability[tx + (MAX_VALUE_SIZE - 1) * PARAMETR_SIZE] = 1.0;
            }
        }
    }

    _mm_free(pheromon_norm);
    _mm_free(svertka);
}
#endif
#if _OPENMP >= 201511  // OpenMP 4.5+
void go_mass_probability_transp_AVX_OMP_non_cuda_4_5(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    const int total_cells = MAX_VALUE_SIZE * PARAMETR_SIZE;

    double* pheromon_norm = (double*)_mm_malloc(total_cells * sizeof(double), 32);
    double* svertka = (double*)_mm_malloc(total_cells * sizeof(double), 32);

    // Инициализация с if clause
#ifdef __clang__
#pragma omp parallel for simd schedule(static)
#else
#pragma omp parallel for simd schedule(static) // if(TOTAL_CELLS > 1000)
#endif
    for (int i = 0; i < total_cells; i++) {
        pheromon_norm[i] = 0.0;
        svertka[i] = 0.0;
    }

    alignas(32) double sumVectorT[PARAMETR_SIZE] = { 0.0 };
    alignas(32) double sumVectorZ[PARAMETR_SIZE] = { 0.0 };

    // 1. Вычисление сумм T_i с улучшенным планированием
#pragma omp parallel
    {
        alignas(32) double local_sumT[PARAMETR_SIZE] = { 0.0 };

        // Clang-compatible: убираем if clause или используем условную компиляцию
#if defined(__clang__)
#pragma omp for schedule(static) nowait
#else
#pragma omp for schedule(static) nowait // if(MAX_VALUE_SIZE > 100)
#endif
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                local_sumT[tx] += pheromon[tx + i * PARAMETR_SIZE];
            }
        }

#pragma omp critical
        {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                sumVectorT[tx] += local_sumT[tx];
            }
        }
    }

    // 2. Вычисление Tnorm
#if defined(__clang__)
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(static) // if(MAX_VALUE_SIZE > 100)
#endif
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            int idx = tx + i * PARAMETR_SIZE;
            pheromon_norm[idx] = (sumVectorT[tx] != 0.0) ? pheromon[idx] / sumVectorT[tx] : 0.0;
        }
    }

    // 3. Вычисление svertka и сумм Z
#pragma omp parallel
    {
        alignas(32) double local_sumZ[PARAMETR_SIZE] = { 0.0 };

#if defined(__clang__)
#pragma omp for schedule(static) nowait
#else
#pragma omp for schedule(static) nowait // if(MAX_VALUE_SIZE > 100)
#endif
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                int idx = tx + i * PARAMETR_SIZE;
                double val = 0.0;
                if (kol_enter[idx] != 0.0 && pheromon_norm[idx] != 0.0) {
                    val = 1.0 / kol_enter[idx] + pheromon_norm[idx];
                }
                svertka[idx] = val;
                local_sumZ[tx] += val;
            }
        }

#pragma omp critical
        {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                sumVectorZ[tx] += local_sumZ[tx];
            }
        }
    }

    // 4. Вычисление кумулятивных вероятностей
        // Clang-compatible: убираем if clause или используем условную компиляцию
#if defined(__clang__)
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(static) // if(PARAMETR_SIZE > 100)
#endif
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        if (sumVectorZ[tx] != 0.0) {
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                int idx = tx + i * PARAMETR_SIZE;
                double prob = svertka[idx] / sumVectorZ[tx];
                cumulative += prob;
                norm_matrix_probability[idx] = cumulative;
            }
            if (MAX_VALUE_SIZE > 0) {
                norm_matrix_probability[tx + (MAX_VALUE_SIZE - 1) * PARAMETR_SIZE] = 1.0;
            }
        }
        else {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                int idx = tx + i * PARAMETR_SIZE;
                cumulative += uniform_prob;
                norm_matrix_probability[idx] = cumulative;
            }
            if (MAX_VALUE_SIZE > 0) {
                norm_matrix_probability[tx + (MAX_VALUE_SIZE - 1) * PARAMETR_SIZE] = 1.0;
            }
        }
    }

    _mm_free(pheromon_norm);
    _mm_free(svertka);
}
#endif
#if _OPENMP >= 201811  // OpenMP 5.0+
void go_mass_probability_transp_AVX_OMP_non_cuda_5_0(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    const int total_cells = MAX_VALUE_SIZE * PARAMETR_SIZE;

    double* pheromon_norm = (double*)_mm_malloc(total_cells * sizeof(double), 32);
    double* svertka = (double*)_mm_malloc(total_cells * sizeof(double), 32);

    // OpenMP 5.0: loop transformation hints
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < total_cells; i++) {
        pheromon_norm[i] = 0.0;
        svertka[i] = 0.0;
    }

    alignas(32) double sumVectorT[PARAMETR_SIZE] = { 0.0 };
    alignas(32) double sumVectorZ[PARAMETR_SIZE] = { 0.0 };

    // 1. Вычисление сумм T_i с nonmonotonic scheduling
#pragma omp parallel
    {
        alignas(32) double local_sumT[PARAMETR_SIZE] = { 0.0 };

        // OpenMP 5.0: nonmonotonic scheduling
#pragma omp for //schedule(nonmonotonic:static) nowait
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                local_sumT[tx] += pheromon[tx + i * PARAMETR_SIZE];
            }
        }

#pragma omp critical
        {
#ifdef __clang__
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                sumVectorT[tx] += local_sumT[tx];
            }
#else
#pragma omp simd
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                sumVectorT[tx] += local_sumT[tx];
            }
#endif
        }
    }

    // 2. Вычисление Tnorm с улучшенной векторизацией
#pragma omp parallel for schedule(static)
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            int idx = tx + i * PARAMETR_SIZE;
            pheromon_norm[idx] = (sumVectorT[tx] != 0.0) ?
                pheromon[idx] / sumVectorT[tx] : 0.0;
        }
    }

    // 3. Вычисление svertka и сумм Z с nonmonotonic scheduling
#pragma omp parallel
    {
        alignas(32) double local_sumZ[PARAMETR_SIZE] = { 0.0 };

        // OpenMP 5.0: nonmonotonic scheduling
#pragma omp for //schedule(nonmonotonic:static) nowait
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                int idx = tx + i * PARAMETR_SIZE;
                double val = 0.0;
                if (kol_enter[idx] != 0.0 && pheromon_norm[idx] != 0.0) {
                    val = 1.0 / kol_enter[idx] + pheromon_norm[idx];
                }
                svertka[idx] = val;
                local_sumZ[tx] += val;
            }
        }

#pragma omp critical
        {
#ifdef __clang__
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                sumVectorZ[tx] += local_sumZ[tx];
            }
#else
#pragma omp simd
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                sumVectorZ[tx] += local_sumZ[tx];
            }
#endif
        }
    }

    // 4. Вычисление кумулятивных вероятностей с loop трансформацией
#pragma omp parallel for schedule(static)
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        if (sumVectorZ[tx] != 0.0) {
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                int idx = tx + i * PARAMETR_SIZE;
                double prob = svertka[idx] / sumVectorZ[tx];
                cumulative += prob;
                norm_matrix_probability[idx] = cumulative;
            }
            if (MAX_VALUE_SIZE > 0) {
                norm_matrix_probability[tx + (MAX_VALUE_SIZE - 1) * PARAMETR_SIZE] = 1.0;
            }
        }
        else {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                int idx = tx + i * PARAMETR_SIZE;
                cumulative += uniform_prob;
                norm_matrix_probability[idx] = cumulative;
            }
            if (MAX_VALUE_SIZE > 0) {
                norm_matrix_probability[tx + (MAX_VALUE_SIZE - 1) * PARAMETR_SIZE] = 1.0;
            }
        }
    }

    _mm_free(pheromon_norm);
    _mm_free(svertka);
}
#endif
#if _OPENMP >= 202011  // OpenMP 5.1+
void go_mass_probability_transp_AVX_OMP_non_cuda_5_1(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    const int total_cells = MAX_VALUE_SIZE * PARAMETR_SIZE;

    // OpenMP 5.1: aligned allocation с error recovery
    double* pheromon_norm = (double*)ALIGNED_ALLOC(32, total_cells * sizeof(double));
    double* svertka = (double*)ALIGNED_ALLOC(32, total_cells * sizeof(double));

    if (!pheromon_norm || !svertka) {
        // Error recovery: fallback to standard allocation
        if (pheromon_norm) free(pheromon_norm);
        if (svertka) free(svertka);
        pheromon_norm = (double*)malloc(total_cells * sizeof(double));
        svertka = (double*)malloc(total_cells * sizeof(double));
    }

    // OpenMP 5.1: неблокирующие операции
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < total_cells; i++) {
        pheromon_norm[i] = 0.0;
        svertka[i] = 0.0;
    }

    alignas(32) double sumVectorT[PARAMETR_SIZE] = { 0.0 };
    alignas(32) double sumVectorZ[PARAMETR_SIZE] = { 0.0 };

    // 1. Вычисление сумм T_i с error recovery features
#pragma omp parallel
    {
        alignas(32) double local_sumT[PARAMETR_SIZE] = { 0.0 };

        // OpenMP 5.1: улучшенное планирование
#pragma omp for //schedule(nonmonotonic:static) nowait
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                local_sumT[tx] += pheromon[tx + i * PARAMETR_SIZE];
            }
        }

#pragma omp critical
        {
            // OpenMP 5.1: выравнивание памяти для векторизации
#pragma omp simd aligned(sumVectorT, local_sumT:32)
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                sumVectorT[tx] += local_sumT[tx];
            }
        }
    }

    // 2. Вычисление Tnorm с улучшенным управлением памятью
#pragma omp parallel for schedule(static)
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            int idx = tx + i * PARAMETR_SIZE;
            pheromon_norm[idx] = (sumVectorT[tx] != 0.0) ?
                pheromon[idx] / sumVectorT[tx] : 0.0;
        }
    }

    // 3. Вычисление svertka и сумм Z с error recovery
#pragma omp parallel
    {
        alignas(32) double local_sumZ[PARAMETR_SIZE] = { 0.0 };

#pragma omp for //schedule(nonmonotonic:static) nowait
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                int idx = tx + i * PARAMETR_SIZE;
                double val = 0.0;
                if (kol_enter[idx] != 0.0 && pheromon_norm[idx] != 0.0) {
                    val = 1.0 / kol_enter[idx] + pheromon_norm[idx];
                }
                svertka[idx] = val;
                local_sumZ[tx] += val;
            }
        }

#pragma omp critical
        {
#pragma omp simd aligned(sumVectorZ, local_sumZ:32)
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                sumVectorZ[tx] += local_sumZ[tx];
            }
        }
    }

    // 4. Вычисление кумулятивных вероятностей с улучшенным управлением ошибками
#pragma omp parallel for schedule(static)
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        if (sumVectorZ[tx] != 0.0) {
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                int idx = tx + i * PARAMETR_SIZE;
                double prob = svertka[idx] / sumVectorZ[tx];
                cumulative += prob;
                norm_matrix_probability[idx] = cumulative;
            }
            // OpenMP 5.1: гарантия численной стабильности
            if (MAX_VALUE_SIZE > 0) {
                norm_matrix_probability[tx + (MAX_VALUE_SIZE - 1) * PARAMETR_SIZE] = 1.0;
            }
        }
        else {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = 0.0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                int idx = tx + i * PARAMETR_SIZE;
                cumulative += uniform_prob;
                norm_matrix_probability[idx] = cumulative;
            }
            if (MAX_VALUE_SIZE > 0) {
                norm_matrix_probability[tx + (MAX_VALUE_SIZE - 1) * PARAMETR_SIZE] = 1.0;
            }
        }
    }

    // OpenMP 5.1: безопасное освобождение памяти
    if (pheromon_norm) ALIGNED_FREE(pheromon_norm);
    if (svertka) ALIGNED_FREE(svertka);
}
#endif
#if _OPENMP >= 202111  // OpenMP 5.2+
void go_mass_probability_transp_AVX_OMP_non_cuda_5_2(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
    const int total_cells = MAX_VALUE_SIZE * PARAMETR_SIZE;

    // OpenMP 5.2: assume clauses для оптимизатора
#if !defined(__clang__)
#pragma omp assume (MAX_VALUE_SIZE > 0)
#pragma omp assume (PARAMETR_SIZE > 0)
#pragma omp assume (total_cells > 0)
#endif

// OpenMP 5.2: aligned allocation с assume
    double* pheromon_norm = (double*)ALIGNED_ALLOC(32, total_cells * sizeof(double));
    double* svertka = (double*)ALIGNED_ALLOC(32, total_cells * sizeof(double));
#if !defined(__clang__)
#pragma omp assume aligned(pheromon_norm, svertka:32)
#endif
    // Инициализация с assume clauses
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < total_cells; i++) {
#if !defined(__clang__)
#pragma omp assume (i < total_cells)
#endif
        pheromon_norm[i] = 0.0;
        svertka[i] = 0.0;
    }

    alignas(32) double sumVectorT[PARAMETR_SIZE] = { 0.0 };
    alignas(32) double sumVectorZ[PARAMETR_SIZE] = { 0.0 };

    // 1. Вычисление сумм T_i с assume clauses
#pragma omp parallel
    {
        alignas(32) double local_sumT[PARAMETR_SIZE] = { 0.0 };

#if !defined(__clang__)
#pragma omp assume (MAX_VALUE_SIZE <= 10000)
#endif
#pragma omp for schedule(static) nowait
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
#if !defined(__clang__)
#pragma omp assume (i < MAX_VALUE_SIZE)
#endif
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
#if !defined(__clang__)
#pragma omp assume (tx < PARAMETR_SIZE)
#endif
                local_sumT[tx] += pheromon[tx + i * PARAMETR_SIZE];
            }
        }

#pragma omp critical
        {
#if !defined(__clang__)
#pragma omp assume aligned(sumVectorT, local_sumT:32)
#endif
#pragma omp simd
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                sumVectorT[tx] += local_sumT[tx];
            }
        }
    }

    // 2. Вычисление Tnorm с assume для численной стабильности
#pragma omp parallel for schedule(static)
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
#if !defined(__clang__)
#pragma omp assume (i < MAX_VALUE_SIZE)
#endif
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
#if !defined(__clang__)
#pragma omp assume (tx < PARAMETR_SIZE)
#endif
            int idx = tx + i * PARAMETR_SIZE;
#if !defined(__clang__)
#pragma omp assume (idx < total_cells)

            // OpenMP 5.2: assume для ветвления
#pragma omp assume (sumVectorT[tx] >= 0.0)
#endif
            pheromon_norm[idx] = (sumVectorT[tx] != 0.0) ?
                pheromon[idx] / sumVectorT[tx] : 0.0;
        }
    }

    // 3. Вычисление svertka и сумм Z с assume
#pragma omp parallel
    {
        alignas(32) double local_sumZ[PARAMETR_SIZE] = { 0.0 };
#if !defined(__clang__)
#pragma omp assume (MAX_VALUE_SIZE <= 10000)
#endif
#pragma omp for schedule(static) nowait
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
#if !defined(__clang__)
#pragma omp assume (i < MAX_VALUE_SIZE)
#endif
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
#if !defined(__clang__)
#pragma omp assume (tx < PARAMETR_SIZE)
#endif
                int idx = tx + i * PARAMETR_SIZE;
#if !defined(__clang__)
#pragma omp assume (idx < total_cells)
#endif

                double val = 0.0;
                // OpenMP 5.2: assume для условий
#if !defined(__clang__)
#pragma omp assume noalias(kol_enter, pheromon_norm)
#endif
                if (kol_enter[idx] != 0.0 && pheromon_norm[idx] != 0.0) {
                    val = 1.0 / kol_enter[idx] + pheromon_norm[idx];
                }
                svertka[idx] = val;
                local_sumZ[tx] += val;
            }
        }

#pragma omp critical
        {
#if !defined(__clang__)
#pragma omp assume aligned(sumVectorZ, local_sumZ:32)
#endif
#pragma omp simd
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                sumVectorZ[tx] += local_sumZ[tx];
            }
        }
    }

    // 4. Вычисление кумулятивных вероятностей с assume clauses
#pragma omp parallel for schedule(static)
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
#if !defined(__clang__)
#pragma omp assume (tx < PARAMETR_SIZE)
#endif

        // OpenMP 5.2: assume для численной стабильности
#if !defined(__clang__)
#pragma omp assume (sumVectorZ[tx] >= 0.0)
#endif
        if (sumVectorZ[tx] != 0.0) {
            double cumulative = 0.0;
#if !defined(__clang__)
#pragma omp assume (MAX_VALUE_SIZE > 0)
#endif
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
#if !defined(__clang__)
#pragma omp assume (i < MAX_VALUE_SIZE)
#endif
                int idx = tx + i * PARAMETR_SIZE;
#if !defined(__clang__)
#pragma omp assume (idx < total_cells)
#endif
                double prob = svertka[idx] / sumVectorZ[tx];
                cumulative += prob;
                norm_matrix_probability[idx] = cumulative;
            }
            if (MAX_VALUE_SIZE > 0) {
                norm_matrix_probability[tx + (MAX_VALUE_SIZE - 1) * PARAMETR_SIZE] = 1.0;
            }
        }
        else {
            double uniform_prob = 1.0 / MAX_VALUE_SIZE;
            double cumulative = 0.0;
#if !defined(__clang__)
#pragma omp assume (MAX_VALUE_SIZE > 0)
#endif
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
#if !defined(__clang__)
#pragma omp assume (i < MAX_VALUE_SIZE)
#endif
                int idx = tx + i * PARAMETR_SIZE;
#if !defined(__clang__)
#pragma omp assume (idx < total_cells)
#endif

                cumulative += uniform_prob;
                norm_matrix_probability[idx] = cumulative;
            }
            if (MAX_VALUE_SIZE > 0) {
                norm_matrix_probability[tx + (MAX_VALUE_SIZE - 1) * PARAMETR_SIZE] = 1.0;
            }
        }
    }

    ALIGNED_FREE(pheromon_norm);
    ALIGNED_FREE(svertka);
}
#endif
void go_mass_probability_transp_AVX_OMP_non_cuda(double* __restrict pheromon, double* __restrict kol_enter, double* __restrict norm_matrix_probability) {
#if _OPENMP >= 202111  // OpenMP 5.2+
    go_mass_probability_transp_AVX_OMP_non_cuda_5_2(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 202011  // OpenMP 5.1+
    go_mass_probability_transp_AVX_OMP_non_cuda_5_1(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201811  // OpenMP 5.0+
    go_mass_probability_transp_AVX_OMP_non_cuda_5_0(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201511  // OpenMP 4.5+
    go_mass_probability_transp_AVX_OMP_non_cuda_4_5(pheromon, kol_enter, norm_matrix_probability);
#elif _OPENMP >= 201307  // OpenMP 4.0+
    go_mass_probability_transp_AVX_OMP_non_cuda_4_0(pheromon, kol_enter, norm_matrix_probability);
#else  // OpenMP 2.0/3.0/3.1
    go_mass_probability_transp_AVX_OMP_non_cuda_2_0(pheromon, kol_enter, norm_matrix_probability);
#endif
}
// Обновление слоев графа
void add_pheromon_iteration_transp_AVX_non_cuda(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    // Испарение весов-феромона
    __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);
    for (int i = 0; i < PARAMETR_SIZE * MAX_VALUE_SIZE; i += CONST_AVX) {
        if (i + CONST_AVX < PARAMETR_SIZE * MAX_VALUE_SIZE) { // Проверка на выход за пределы массива
            __m256d pheromonValues_AVX = _mm256_loadu_pd(&pheromon[i]); // Загружаем 4 значения из pheromon
            pheromonValues_AVX = _mm256_mul_pd(pheromonValues_AVX, parametRovector_AVX);  // Умножаем на PARAMETR_RO
            _mm256_storeu_pd(&pheromon[i], pheromonValues_AVX); // Сохраняем обратно в pheromon
        }
    }

    for (int i = 0; i < ANT_SIZE; ++i) {
        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            int k = agent_node[tx + i * PARAMETR_SIZE];
            if ((k >= 0) && (k < MAX_VALUE_SIZE)) {
                kol_enter[tx + k * PARAMETR_SIZE]++;
                if (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i] > 0) {
                    pheromon[tx + k * PARAMETR_SIZE] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i]); // MIN
                }
            }
        }
    }
}
// Базовая версия OpenMP 2.0/3.0/3.1
void add_pheromon_iteration_transp_AVX_OMP_non_cuda_2_0(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int remainder = total_size % CONST_AVX;
    const __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);

    // 1. Испарение феромонов - векторизованное
#pragma omp parallel for schedule(static)
    for (int i = 0; i < total_size - remainder; i += CONST_AVX) {
        __m256d pheromonValues_AVX = _mm256_loadu_pd(&pheromon[i]);
        pheromonValues_AVX = _mm256_mul_pd(pheromonValues_AVX, parametRovector_AVX);
        _mm256_storeu_pd(&pheromon[i], pheromonValues_AVX);
    }

    // Обработка остаточных элементов
    if (remainder > 0) {
#pragma omp parallel for schedule(static)
        for (int i = total_size - remainder; i < total_size; i++) {
            pheromon[i] *= PARAMETR_RO;
        }
    }

    // 2. Добавление феромонов с atomic операциями
#pragma omp parallel for schedule(static)
    for (int i = 0; i < ANT_SIZE; ++i) {
        const double agent_of = OF[i];
        const double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of;
        const double add_value = (delta > 0) ? PARAMETR_Q * delta : 0.0;

        const int* agent_path = &agent_node[i * PARAMETR_SIZE];
        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            int k = agent_path[tx];
            if ((k >= 0) && (k < MAX_VALUE_SIZE)) {
                int idx = tx + k * PARAMETR_SIZE;
#pragma omp atomic
                kol_enter[idx]++;

                if (add_value > 0) {
#pragma omp atomic
                    pheromon[idx] += add_value;
                }
            }
        }
    }
}
#if _OPENMP >= 201307  // OpenMP 4.0+
void add_pheromon_iteration_transp_AVX_OMP_non_cuda_4_0(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int remainder = total_size % CONST_AVX;
    const __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);

    // 1. Испарение феромонов с SIMD
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < total_size - remainder; i += CONST_AVX) {
        __m256d pheromonValues_AVX = _mm256_loadu_pd(&pheromon[i]);
        pheromonValues_AVX = _mm256_mul_pd(pheromonValues_AVX, parametRovector_AVX);
        _mm256_storeu_pd(&pheromon[i], pheromonValues_AVX);
    }

    // Обработка остаточных элементов
    if (remainder > 0) {
#pragma omp parallel for schedule(static)
        for (int i = total_size - remainder; i < total_size; i++) {
            pheromon[i] *= PARAMETR_RO;
        }
    }

    // 2. Добавление феромонов с оптимизированными atomic операциями
#pragma omp parallel for schedule(static)
    for (int i = 0; i < ANT_SIZE; ++i) {
        const double agent_of = OF[i];
        const double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of;
        const double add_value = (delta > 0) ? PARAMETR_Q * delta : 0.0;

        const int* agent_path = &agent_node[i * PARAMETR_SIZE];
        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            int k = agent_path[tx];
            if ((k >= 0) && (k < MAX_VALUE_SIZE)) {
                int idx = tx + k * PARAMETR_SIZE;
#pragma omp atomic update
                kol_enter[idx]++;

                if (add_value > 0) {
#pragma omp atomic update
                    pheromon[idx] += add_value;
                }
            }
        }
    }
}
#endif
#if _OPENMP >= 201511  // OpenMP 4.5+
void add_pheromon_iteration_transp_AVX_OMP_non_cuda_4_5(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int remainder = total_size % CONST_AVX;
    const __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);

    // 1. Испарение феромонов с if clause
#if _OPENMP >= 201511 && !defined(__clang__)
#pragma omp parallel for simd schedule(static) if(total_size > 1000)
#else
#pragma omp parallel for simd schedule(static)
#endif
    for (int i = 0; i < total_size - remainder; i += CONST_AVX) {
        __m256d pheromonValues_AVX = _mm256_loadu_pd(&pheromon[i]);
        pheromonValues_AVX = _mm256_mul_pd(pheromonValues_AVX, parametRovector_AVX);
        _mm256_storeu_pd(&pheromon[i], pheromonValues_AVX);
    }

    // Обработка остаточных элементов
    if (remainder > 0) {
#if _OPENMP >= 201511 && !defined(__clang__)
#pragma omp parallel for schedule(static) if(remainder > 10)
#else
#pragma omp parallel for schedule(static)
#endif
        for (int i = total_size - remainder; i < total_size; i++) {
            pheromon[i] *= PARAMETR_RO;
        }
    }

    // 2. Добавление феромонов с улучшенным планированием
            // Clang-compatible: убираем if clause или используем условную компиляцию
#if defined(__clang__)
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(static) // if(ANT_SIZE > 100)
#endif
    for (int i = 0; i < ANT_SIZE; ++i) {
        const double agent_of = OF[i];
        const double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of;
        const double add_value = (delta > 0) ? PARAMETR_Q * delta : 0.0;

        const int* agent_path = &agent_node[i * PARAMETR_SIZE];
        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            int k = agent_path[tx];
            if ((k >= 0) && (k < MAX_VALUE_SIZE)) {
                int idx = tx + k * PARAMETR_SIZE;
#pragma omp atomic update
                kol_enter[idx]++;

                if (add_value > 0) {
#pragma omp atomic update
                    pheromon[idx] += add_value;
                }
            }
        }
    }
}
#endif
#if _OPENMP >= 201811  // OpenMP 5.0+
void add_pheromon_iteration_transp_AVX_OMP_non_cuda_5_0(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int remainder = total_size % CONST_AVX;
    const __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);

    // 1. Испарение феромонов с loop трансформацией
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < total_size - remainder; i += CONST_AVX) {
        __m256d pheromonValues_AVX = _mm256_loadu_pd(&pheromon[i]);
        pheromonValues_AVX = _mm256_mul_pd(pheromonValues_AVX, parametRovector_AVX);
        _mm256_storeu_pd(&pheromon[i], pheromonValues_AVX);
    }

    // Обработка остаточных элементов
    if (remainder > 0) {
#pragma omp parallel for schedule(static)
        for (int i = total_size - remainder; i < total_size; i++) {
            pheromon[i] *= PARAMETR_RO;
        }
    }

    // 2. Добавление феромонов с nonmonotonic scheduling
#pragma omp parallel for schedule(nonmonotonic:static)
    for (int i = 0; i < ANT_SIZE; ++i) {
        const double agent_of = OF[i];
        const double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of;
        const double add_value = (delta > 0) ? PARAMETR_Q * delta : 0.0;

        const int* agent_path = &agent_node[i * PARAMETR_SIZE];
        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            int k = agent_path[tx];
            if ((k >= 0) && (k < MAX_VALUE_SIZE)) {
                int idx = tx + k * PARAMETR_SIZE;
#pragma omp atomic update
                kol_enter[idx]++;

                if (add_value > 0) {
#pragma omp atomic update
                    pheromon[idx] += add_value;
                }
            }
        }
    }
}
#endif
#if _OPENMP >= 202011  // OpenMP 5.1+
void add_pheromon_iteration_transp_AVX_OMP_non_cuda_5_1(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int remainder = total_size % CONST_AVX;
    const __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);

    // 1. Испарение феромонов с неблокирующими операциями
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < total_size - remainder; i += CONST_AVX) {
        __m256d pheromonValues_AVX = _mm256_loadu_pd(&pheromon[i]);
        pheromonValues_AVX = _mm256_mul_pd(pheromonValues_AVX, parametRovector_AVX);
        _mm256_storeu_pd(&pheromon[i], pheromonValues_AVX);
    }

    // Обработка остаточных элементов
    if (remainder > 0) {
#pragma omp parallel for schedule(static)
        for (int i = total_size - remainder; i < total_size; i++) {
            pheromon[i] *= PARAMETR_RO;
        }
    }

    // 2. Добавление феромонов с error recovery features
#pragma omp parallel for schedule(nonmonotonic:static)
    for (int i = 0; i < ANT_SIZE; ++i) {
        const double agent_of = OF[i];
        const double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of;
        const double add_value = (delta > 0) ? PARAMETR_Q * delta : 0.0;

        const int* agent_path = &agent_node[i * PARAMETR_SIZE];
        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            int k = agent_path[tx];
            if ((k >= 0) && (k < MAX_VALUE_SIZE)) {
                int idx = tx + k * PARAMETR_SIZE;
                // OpenMP 5.1: улучшенные atomic операции
#pragma omp atomic update
                kol_enter[idx]++;

                if (add_value > 0) {
#pragma omp atomic update
                    pheromon[idx] += add_value;
                }
            }
        }
    }
}
#endif
#if _OPENMP >= 202111  // OpenMP 5.2+
void add_pheromon_iteration_transp_AVX_OMP_non_cuda_5_2(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
    const int total_size = PARAMETR_SIZE * MAX_VALUE_SIZE;
    const int remainder = total_size % CONST_AVX;
    const __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);

    // OpenMP 5.2: assume clauses для оптимизатора
#if !defined(__clang__)
#pragma omp assume (MAX_VALUE_SIZE > 0)
#pragma omp assume (PARAMETR_SIZE > 0)
#pragma omp assume (total_size > 0)
#endif

// 1. Испарение феромонов с assume clauses
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < total_size - remainder; i += CONST_AVX) {
#if !defined(__clang__)
#pragma omp assume aligned(pheromon:32)
#endif
        __m256d pheromonValues_AVX = _mm256_load_pd(&pheromon[i]);
        pheromonValues_AVX = _mm256_mul_pd(pheromonValues_AVX, parametRovector_AVX);
        _mm256_store_pd(&pheromon[i], pheromonValues_AVX);
    }

    // Обработка остаточных элементов
    if (remainder > 0) {
#if !defined(__clang__)
#pragma omp assume (remainder < CONST_AVX)
#endif
#pragma omp parallel for schedule(static)
        for (int i = total_size - remainder; i < total_size; i++) {
            pheromon[i] *= PARAMETR_RO;
        }
    }

    // 2. Добавление феромонов с latest OpenMP 5.2 features
#pragma omp parallel for schedule(static)
    for (int i = 0; i < ANT_SIZE; ++i) {
        const double agent_of = OF[i];

        // OpenMP 5.2: assume для численной стабильности
#if !defined(__clang__)
#pragma omp assume (MAX_PARAMETR_VALUE_TO_MIN_OPT >= 0.0)
#pragma omp assume (agent_of >= 0.0)
#endif

        const double delta = MAX_PARAMETR_VALUE_TO_MIN_OPT - agent_of;
        const double add_value = (delta > 0) ? PARAMETR_Q * delta : 0.0;

        const int* agent_path = &agent_node[i * PARAMETR_SIZE];

        // OpenMP 5.2: assume для лучшей оптимизации
#if !defined(__clang__)
#pragma omp assume (PARAMETR_SIZE <= 1000)
#endif
        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            int k = agent_path[tx];
            if ((k >= 0) && (k < MAX_VALUE_SIZE)) {
                int idx = tx + k * PARAMETR_SIZE;

                // OpenMP 5.2: улучшенные atomic операции
#pragma omp atomic update
                kol_enter[idx]++;

                if (add_value > 0) {
#pragma omp atomic update
                    pheromon[idx] += add_value;
                }
            }
        }
    }
}
#endif
void add_pheromon_iteration_transp_AVX_OMP_non_cuda(double* __restrict pheromon, double* __restrict kol_enter, const int* __restrict agent_node, const double* __restrict OF) {
#if _OPENMP >= 202111  // OpenMP 5.2+
    add_pheromon_iteration_transp_AVX_OMP_non_cuda_5_2(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 202011  // OpenMP 5.1+
    add_pheromon_iteration_transp_AVX_OMP_non_cuda_5_1(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 201811  // OpenMP 5.0+
    add_pheromon_iteration_transp_AVX_OMP_non_cuda_5_0(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 201511  // OpenMP 4.5+
    add_pheromon_iteration_transp_AVX_OMP_non_cuda_4_5(pheromon, kol_enter, agent_node, OF);
#elif _OPENMP >= 201307  // OpenMP 4.0+
    add_pheromon_iteration_transp_AVX_OMP_non_cuda_4_0(pheromon, kol_enter, agent_node, OF);
#else  // OpenMP 2.0/3.0/3.1
    add_pheromon_iteration_transp_AVX_OMP_non_cuda_2_0(pheromon, kol_enter, agent_node, OF);
#endif
}
int start_NON_CUDA_transp_AVX_time() {
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
        go_mass_probability_transp_AVX_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

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
        add_pheromon_iteration_transp_AVX_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

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
    std::cout << "Time non CUDA_transp_AVX_time;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA_transp_AVX_time;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}
int start_NON_CUDA_transp_AVX() {
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
        go_mass_probability_transp_AVX_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

        // Вычисление пути агентов
        auto end_temp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> current_time = end_temp - start;
        go_all_agent_transp_non_cuda(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail);

        // Обновление весов-феромонов
        add_pheromon_iteration_transp_AVX_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

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
    std::cout << "Time non CUDA_transp_AVX;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA_transp_AVX;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}
int start_NON_CUDA_transp_AVX_non_hash() {
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
        go_mass_probability_transp_AVX_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

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
        add_pheromon_iteration_transp_AVX_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

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
    std::cout << "Time non CUDA_transp_AVX non hash;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << "0" << "; " << SumgpuTime5 << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA_transp_AVX non hash:;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << "0" << "; " << SumgpuTime5 << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    return 0;
}

int start_NON_CUDA_transp_AVX_OMP_time() {
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
        go_mass_probability_transp_AVX_OMP_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

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
        add_pheromon_iteration_transp_AVX_OMP_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

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
    std::cout << "Time non CUDA_transp_AVX_OMP_time;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA_transp_AVX_OMP_time;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}
int start_NON_CUDA_transp_AVX_OMP() {
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
        go_mass_probability_transp_AVX_OMP_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

        // Вычисление пути агентов
        auto end_temp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> current_time = end_temp - start;
        go_all_agent_transp_non_cuda(int(current_time.count() * CONST_RANDOM), parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail);

        // Обновление весов-феромонов
        add_pheromon_iteration_transp_AVX_OMP_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

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
    std::cout << "Time non CUDA_transp_AVX_OMP;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA_transp_AVX_OMP;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}
int start_NON_CUDA_transp_AVX_OMP_non_hash() {
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
        go_mass_probability_transp_AVX_OMP_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);

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
        add_pheromon_iteration_transp_AVX_OMP_non_cuda(pheromon_value, kol_enter_value, ant_parametr, antOF);

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
    std::cout << "Time non CUDA_transp_AVX_OMP non hash;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << "0" << "; " << SumgpuTime5 << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA_transp_AVX_OMP non hash:;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << "0" << "; " << SumgpuTime5 << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    return 0;
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
    if (GO_NON_CUDA_TRANSP_OMP_TIME) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_NON_CUDA_transp_OMP_time();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_NON_CUDA_transp_OMP_time();
            i = i + 1;
        }
        // Остановка таймера
        auto end2 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time non CUDA_transp_OMP_time:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("non CUDA transp OMP Time");
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
    if (GO_NON_CUDA_TRANSP_NON_HASH_OMP_OPT) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_NON_CUDA_transp_non_hash_OMP_optimized();
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_NON_CUDA_transp_non_hash_OMP_optimized();
            i = i + 1;
        }
        // Остановка таймера
        auto end2 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time non CUDA transp non hash OMP Optimized:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("non CUDA transp non hash OMP Optimized");
    }
    if (MAX_VALUE_SIZE % CONST_AVX == 0) {
        if (GO_NON_CUDA_AVX_TIME) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_NON_CUDA_AVX_time();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start2 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_NON_CUDA_AVX_time();
                i = i + 1;
            }
            // Остановка таймера
            auto end2 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end2 - start2;
            std::string message = "Time non CUDA_AVX_time:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("non CUDA AVX Time");
        }
        if (GO_NON_CUDA_AVX) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_NON_CUDA_AVX();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start2 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_NON_CUDA_AVX();
                i = i + 1;
            }
            // Остановка таймера
            auto end2 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end2 - start2;
            std::string message = "Time non CUDA AVX:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("non CUDA AVX");
        }
        if (GO_NON_CUDA_AVX_NON_HASH) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_NON_CUDA_AVX_non_hash();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start2 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_NON_CUDA_AVX_non_hash();
                i = i + 1;
            }
            // Остановка таймера
            auto end2 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end2 - start2;
            std::string message = "Time non CUDA AVX non hash:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("non CUDA AVX non hash");
        }
        if (GO_NON_CUDA_AVX_OMP_TIME) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_NON_CUDA_AVX_OMP_time();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start2 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_NON_CUDA_AVX_OMP_time();
                i = i + 1;
            }
            // Остановка таймера
            auto end2 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end2 - start2;
            std::string message = "Time non CUDA_AVX_OMP_time:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("non CUDA AVX_OMP Time");
        }
        if (GO_NON_CUDA_AVX_OMP) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_NON_CUDA_AVX_OMP();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start2 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_NON_CUDA_AVX_OMP();
                i = i + 1;
            }
            // Остановка таймера
            auto end2 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end2 - start2;
            std::string message = "Time non CUDA AVX_OMP:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("non CUDA AVX_OMP");
        }
        if (GO_NON_CUDA_AVX_OMP_NON_HASH) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_NON_CUDA_AVX_OMP_non_hash();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start2 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_NON_CUDA_AVX_OMP_non_hash();
                i = i + 1;
            }
            // Остановка таймера
            auto end2 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end2 - start2;
            std::string message = "Time non CUDA AVX_OMP non hash:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("non CUDA AVX_OMP non hash");
        }
    }
    if (PARAMETR_SIZE % CONST_AVX == 0) {
        if (GO_NON_CUDA_TRANSP_AVX_TIME) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_NON_CUDA_transp_AVX_time();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start2 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_NON_CUDA_transp_AVX_time();
                i = i + 1;
            }
            // Остановка таймера
            auto end2 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end2 - start2;
            std::string message = "Time non CUDA_transp_AVX_time:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("non CUDA transp_AVX Time");
        }
        if (GO_NON_CUDA_TRANSP_AVX) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_NON_CUDA_transp_AVX();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start2 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_NON_CUDA_transp_AVX();
                i = i + 1;
            }
            // Остановка таймера
            auto end2 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end2 - start2;
            std::string message = "Time non CUDA transp_AVX:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("non CUDA transp_AVX");
        }
        if (GO_NON_CUDA_TRANSP_AVX_NON_HASH) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_NON_CUDA_transp_AVX_non_hash();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start2 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_NON_CUDA_transp_AVX_non_hash();
                i = i + 1;
            }
            // Остановка таймера
            auto end2 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end2 - start2;
            std::string message = "Time non CUDA transp_AVX non hash:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("non CUDA transp_AVX non hash");
        }
        if (GO_NON_CUDA_TRANSP_AVX_OMP_TIME) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_NON_CUDA_transp_AVX_OMP_time();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start2 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_NON_CUDA_transp_AVX_OMP_time();
                i = i + 1;
            }
            // Остановка таймера
            auto end2 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end2 - start2;
            std::string message = "Time non CUDA_transp_AVX_OMP_time:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("non CUDA transp_AVX_OMP Time");
        }
        if (GO_NON_CUDA_TRANSP_AVX_OMP) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_NON_CUDA_transp_AVX_OMP();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start2 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_NON_CUDA_transp_AVX_OMP();
                i = i + 1;
            }
            // Остановка таймера
            auto end2 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end2 - start2;
            std::string message = "Time non CUDA transp_AVX_OMP:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("non CUDA transp_AVX_OMP");
        }
        if (GO_NON_CUDA_TRANSP_AVX_OMP_NON_HASH) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_NON_CUDA_transp_AVX_OMP_non_hash();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start2 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_NON_CUDA_transp_AVX_OMP_non_hash();
                i = i + 1;
            }
            // Остановка таймера
            auto end2 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end2 - start2;
            std::string message = "Time non CUDA transp_AVX_OMP non hash:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("non CUDA transp_AVX_OMP non hash");
        }
    }
    if (MAX_VALUE_SIZE == CONST_AVX)
    {
        if (GO_NON_CUDA_AVX_TIME_4) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_NON_CUDA_AVX4_time();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start1 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_NON_CUDA_AVX4_time();
                i = i + 1;
            }
            // Остановка таймера
            auto end1 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end1 - start1;
            // Вывод информации на экран и в лог-файл
            std::string message = "Time non CUDA AVX time 4:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("Time non CUDA AVX time 4");
        }
        if (GO_NON_CUDA_AVX_OMP_TIME_4) {
            int j = 0;
            while (j < KOL_PROGREV)
            {
                std::cout << "PROGREV " << j << " ";
                start_NON_CUDA_AVX4_OMP_time();
                j = j + 1;
            }
            // Запуск таймера
            clear_all_stat();
            auto start1 = std::chrono::high_resolution_clock::now();
            int i = 0;
            while (i < KOL_PROGON_STATISTICS)
            {
                std::cout << i << " ";
                start_NON_CUDA_AVX4_OMP_time();
                i = i + 1;
            }
            // Остановка таймера
            auto end1 = std::chrono::high_resolution_clock::now();
            // Вычисление времени выполнения
            std::chrono::duration<double, std::milli> duration = end1 - start1;
            // Вывод информации на экран и в лог-файл
            std::string message = "Time non CUDA AVX OMP time 4:;" + std::to_string(duration.count()) + ";sec";
            std::cout << message << std::endl;
            logFile << message << std::endl; // Запись в лог-файл
            save_all_stat_text_file("Time non CUDA AVX OMP time 4");
        }
    }
    // Закрытие лог-файла
    logFile.close();
    outfile.close();
}