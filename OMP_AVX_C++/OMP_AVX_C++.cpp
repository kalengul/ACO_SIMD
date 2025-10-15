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

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#include <immintrin.h> // Для AVX
// Подготовка массива для вероятностного поиска
void go_mass_probability_AVX_non_cuda(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        // Вычисляем sumVector
        double sumVector = 0.0;
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормируем значения
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 }; // Инициализация массива
        for (int i = 0; i < MAX_VALUE_SIZE; i += CONST_AVX) {
            if (i + CONST_AVX <= MAX_VALUE_SIZE) { // Проверка границ
                __m256d pheromonValues_AVX = _mm256_loadu_pd(&pheromon[MAX_VALUE_SIZE * tx + i]);
                __m256d normValues_AVX = _mm256_div_pd(pheromonValues_AVX, _mm256_set1_pd(sumVector));
                _mm256_storeu_pd(&pheromon_norm[i], normValues_AVX);
            }
        }

        // Вычисляем svertka и sumVector
        sumVector = 0.0;
        double svertka[MAX_VALUE_SIZE] = { 0 }; // Инициализация массива
        __m256d sumVector_AVX = _mm256_setzero_pd();
        for (int i = 0; i < MAX_VALUE_SIZE; i += CONST_AVX) {
            if (i + CONST_AVX <= MAX_VALUE_SIZE) { // Проверка границ
                __m256d kolEnterValues_AVX = _mm256_loadu_pd(&kol_enter[MAX_VALUE_SIZE * tx + i]);
                __m256d pheromonNormValues_AVX = _mm256_loadu_pd(&pheromon_norm[i]);
                __m256d svertkaValues_AVX;

                // Создаем маску для проверки условий
                __m256d mask_AVX = _mm256_cmp_pd(kolEnterValues_AVX, _mm256_setzero_pd(), _CMP_NEQ_OQ);
                mask_AVX = _mm256_and_pd(mask_AVX, _mm256_cmp_pd(pheromonNormValues_AVX, _mm256_setzero_pd(), _CMP_NEQ_OQ));

                // Вычисляем svertka с учетом условий
                __m256d oneOverKolEnter_AVX = _mm256_div_pd(_mm256_set1_pd(1.0), kolEnterValues_AVX);
                svertkaValues_AVX = _mm256_add_pd(oneOverKolEnter_AVX, pheromonNormValues_AVX);
                svertkaValues_AVX = _mm256_blendv_pd(_mm256_setzero_pd(), svertkaValues_AVX, mask_AVX);

                // Сохраняем результат
                _mm256_storeu_pd(&svertka[i], svertkaValues_AVX);
                // Суммируем svertka
                sumVector_AVX = _mm256_add_pd(sumVector_AVX, svertkaValues_AVX);
            }
        }
        // Суммируем значения из вектора svertka
        double temp[CONST_AVX] = { 0 };
        _mm256_storeu_pd(temp, sumVector_AVX);
        for (int j = 0; j < CONST_AVX; j++) {
            sumVector += temp[j];
        }

        // Заполняем norm_matrix_probability
        if (sumVector != 0) { // Проверка на деление на ноль
            norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
            }
        }
    }
}
void go_mass_probability_AVX_OMP_non_cuda(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        // Вычисляем sumVector
        double sumVector = 0.0;
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормируем значения
        double pheromon_norm[MAX_VALUE_SIZE] = { 0 }; // Инициализация массива
#pragma omp parallel for
        for (int i = 0; i < MAX_VALUE_SIZE; i += CONST_AVX) {
            if (i + CONST_AVX <= MAX_VALUE_SIZE) { // Проверка границ
                __m256d pheromonValues_AVX = _mm256_loadu_pd(&pheromon[MAX_VALUE_SIZE * tx + i]);
                __m256d normValues_AVX = _mm256_div_pd(pheromonValues_AVX, _mm256_set1_pd(sumVector));
                _mm256_storeu_pd(&pheromon_norm[i], normValues_AVX);
            }
        }

        // Вычисляем svertka и sumVector
        sumVector = 0.0;
        double svertka[MAX_VALUE_SIZE] = { 0 }; // Инициализация массива
        __m256d sumVector_AVX = _mm256_setzero_pd();
#pragma omp parallel for reduction(+:sumVector)
        for (int i = 0; i < MAX_VALUE_SIZE; i += CONST_AVX) {
            if (i + CONST_AVX <= MAX_VALUE_SIZE) { // Проверка границ
                __m256d kolEnterValues_AVX = _mm256_loadu_pd(&kol_enter[MAX_VALUE_SIZE * tx + i]);
                __m256d pheromonNormValues_AVX = _mm256_loadu_pd(&pheromon_norm[i]);
                __m256d svertkaValues_AVX;

                // Создаем маску для проверки условий
                __m256d mask_AVX = _mm256_cmp_pd(kolEnterValues_AVX, _mm256_setzero_pd(), _CMP_NEQ_OQ);
                mask_AVX = _mm256_and_pd(mask_AVX, _mm256_cmp_pd(pheromonNormValues_AVX, _mm256_setzero_pd(), _CMP_NEQ_OQ));

                // Вычисляем svertka с учетом условий
                __m256d oneOverKolEnter_AVX = _mm256_div_pd(_mm256_set1_pd(1.0), kolEnterValues_AVX);
                svertkaValues_AVX = _mm256_add_pd(oneOverKolEnter_AVX, pheromonNormValues_AVX);
                svertkaValues_AVX = _mm256_blendv_pd(_mm256_setzero_pd(), svertkaValues_AVX, mask_AVX);

                // Сохраняем результат
                _mm256_storeu_pd(&svertka[i], svertkaValues_AVX);
                // Суммируем svertka
                sumVector_AVX = _mm256_add_pd(sumVector_AVX, svertkaValues_AVX);
            }
        }
        // Суммируем значения из вектора svertka
        double temp[CONST_AVX] = { 0 };
        _mm256_storeu_pd(temp, sumVector_AVX);
        for (int j = 0; j < CONST_AVX; j++) {
            sumVector += temp[j];
        }

        // Заполняем norm_matrix_probability
        if (sumVector != 0) { // Проверка на деление на ноль
            norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
            }
        }
    }
}
void go_mass_probability_AVX_non_cuda_4(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
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
void go_mass_probability_not_f_AVX_non_cuda_4(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
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
// Обновление слоев графа
void add_pheromon_iteration_AVX_non_cuda(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
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
void add_pheromon_iteration_AVX_OMP_non_cuda(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    // Испарение весов-феромона
    __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);

#pragma omp parallel for
    for (int i = 0; i < PARAMETR_SIZE * MAX_VALUE_SIZE; i += CONST_AVX) {
        if (i + CONST_AVX < PARAMETR_SIZE * MAX_VALUE_SIZE) { // Проверка на выход за пределы массива
            __m256d pheromonValues_AVX = _mm256_loadu_pd(&pheromon[i]); // Загружаем 4 значения из pheromon
            pheromonValues_AVX = _mm256_mul_pd(pheromonValues_AVX, parametRovector_AVX);  // Умножаем на PARAMETR_RO
            _mm256_storeu_pd(&pheromon[i], pheromonValues_AVX); // Сохраняем обратно в pheromon
        }
    }

#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
        // Добавление весов-феромона
        for (int i = 0; i < ANT_SIZE; ++i) {
            int k = agent_node[i * PARAMETR_SIZE + tx];
            if (k >= 0 && k < MAX_VALUE_SIZE) { // Проверка на выход за пределы массива kol_enter
                // Используем атомарное обновление для kol_enter
#pragma omp atomic
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
void add_pheromon_iteration_AVX_OMP_non_cuda_4(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    // Испарение весов-феромона
    __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);

#pragma omp parallel for
    for (int i = 0; i < PARAMETR_SIZE * MAX_VALUE_SIZE; i += CONST_AVX) {
        __m256d pheromonValues_AVX = _mm256_loadu_pd(&pheromon[i]); // Загружаем 4 значения из pheromon
        pheromonValues_AVX = _mm256_mul_pd(pheromonValues_AVX, parametRovector_AVX);  // Умножаем на PARAMETR_RO
        _mm256_storeu_pd(&pheromon[i], pheromonValues_AVX); // Сохраняем обратно в pheromon
    }

#pragma omp parallel for
    for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
        // Добавление весов-феромона
        for (int i = 0; i < ANT_SIZE; ++i) {
            int k = agent_node[i * PARAMETR_SIZE + tx];
            if (k >= 0 && k < MAX_VALUE_SIZE) { // Проверка на выход за пределы массива kol_enter
                // Используем атомарное обновление для kol_enter
#pragma omp atomic
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


int start_NON_CUDA_AVX_time() {
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

void go_all_agent_non_cuda_time_4(int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail, double& totalHashTime, double& totalOFTime, double& HashTimeSave, double& HashTimeSearch, double& SumTimeSearch) {
    // Генератор случайных чисел
    std::default_random_engine generator(123 + gpuTime); // Используем gpuTime как начальное значение
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int bx = 0; bx < ANT_SIZE; bx++) { // Проходим по всем агентам
        auto start_ant = std::chrono::high_resolution_clock::now();
        bool go_4 = true;
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            // Определение номера значения
            int k = 0;
            while (go_4 && k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                k++;
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
void go_mass_probability_transp_AVX_OMP_non_cuda(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    // Сумма T_i для Tnorm
    double sumVectorT[PARAMETR_SIZE] = { 0.0 };
    double sumVectorZ[PARAMETR_SIZE] = { 0.0 };

#pragma omp parallel for
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
#pragma omp atomic
            sumVectorT[tx] += pheromon[tx + i * PARAMETR_SIZE];
        }
    }

    // Вычисление Tnorm
    double* pheromon_norm = new double[MAX_VALUE_SIZE * PARAMETR_SIZE];
#pragma omp parallel for
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
#pragma omp parallel for
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx += CONST_AVX) {
            // Загружаем значения из kol_enter_value и pheromon_norm
            __m256d kol_enter_values_AVX = _mm256_loadu_pd(&kol_enter[tx + i * PARAMETR_SIZE]);
            __m256d pheromon_norm_values_AVX = _mm256_loadu_pd(&pheromon_norm[tx + i * PARAMETR_SIZE]);

            // Создаем маску для проверки условий
            __m256d zero_vector_AVX = _mm256_setzero_pd();
            __m256d condition_mask_AVX = _mm256_and_pd(
                _mm256_cmp_pd(kol_enter_values_AVX, zero_vector_AVX, _CMP_NEQ_OQ),
                _mm256_cmp_pd(pheromon_norm_values_AVX, zero_vector_AVX, _CMP_NEQ_OQ)
            );

            // Вычисляем svertka
            __m256d one_vector_AVX = _mm256_set1_pd(1.0);
            __m256d svertka_values_AVX = _mm256_blendv_pd(
                zero_vector_AVX,
                _mm256_add_pd(_mm256_div_pd(one_vector_AVX, kol_enter_values_AVX), pheromon_norm_values_AVX),
                condition_mask_AVX
            );

            // Сохраняем значения в svertka
            _mm256_storeu_pd(&svertka[tx + i * PARAMETR_SIZE], svertka_values_AVX);
        }
    }

#pragma omp parallel for
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
#pragma omp atomic
            sumVectorZ[tx] += svertka[tx + i * PARAMETR_SIZE];
        }
    }

    // Вычисление F
#pragma omp parallel for
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
void add_pheromon_iteration_transp_AVX_OMP_non_cuda(double* pheromon, double* kol_enter, int* agent_node, double* OF) {
    // Испарение весов-феромона
    __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);
#pragma omp parallel for
    for (int i = 0; i < PARAMETR_SIZE * MAX_VALUE_SIZE; i += CONST_AVX) {
        if (i + CONST_AVX < PARAMETR_SIZE * MAX_VALUE_SIZE) { // Проверка на выход за пределы массива
            __m256d pheromonValues_AVX = _mm256_loadu_pd(&pheromon[i]); // Загружаем 4 значения из pheromon
            pheromonValues_AVX = _mm256_mul_pd(pheromonValues_AVX, parametRovector_AVX);  // Умножаем на PARAMETR_RO
            _mm256_storeu_pd(&pheromon[i], pheromonValues_AVX); // Сохраняем обратно в pheromon
        }
    }
#pragma omp parallel for
    for (int i = 0; i < ANT_SIZE; ++i) {
        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            int k = agent_node[tx + i * PARAMETR_SIZE];
            if ((k >= 0) && (k < MAX_VALUE_SIZE)) {
                // Обновление kol_enter, используя атомарное обновление
#pragma omp atomic
                kol_enter[tx + k * PARAMETR_SIZE]++;
                // Обновление pheromon с учетом условий
                if (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i] > 0) {
#pragma omp atomic
                    pheromon[tx + k * PARAMETR_SIZE] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i]); // MIN
                }
            }
        }
    }
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

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//Eigen/Dense
/*
#include <Eigen/Dense>

void load_matrix_non_cuda(const std::string& filename, Eigen::VectorXd& parametr_value,
    Eigen::VectorXd& pheromon_value, Eigen::VectorXd& kol_enter_value) {
    // Реализуйте вашу функцию загрузки данных здесь
}

double BenchShafferaFunction_non_cuda(const Eigen::VectorXd& agent) {
    // Реализуйте вашу функцию оценки здесь
    return 0.0; // Замените на реальную реализацию
}

void matrix_ACO_non_hash() {
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;

    Eigen::VectorXd parametr_value(kolBytes_matrix_graph);
    Eigen::VectorXd pheromon_value(kolBytes_matrix_graph);
    Eigen::VectorXd kol_enter_value(kolBytes_matrix_graph);
    Eigen::VectorXd norm_matrix_probability(kolBytes_matrix_graph);
    Eigen::VectorXd ant(kolBytes_matrix_ant);
    Eigen::VectorXi ant_parametr(kolBytes_matrix_ant);
    Eigen::VectorXd antOF(ANT_SIZE);
    Eigen::VectorXd agent(PARAMETR_SIZE);

    std::default_random_engine generator(123);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);

    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            double sumVector = pheromon_value.segment(MAX_VALUE_SIZE * tx, MAX_VALUE_SIZE).sum();
            Eigen::VectorXd pheromon_norm = pheromon_value.segment(MAX_VALUE_SIZE * tx, MAX_VALUE_SIZE) / sumVector;

            double svertka[MAX_VALUE_SIZE] = { 0 };
            sumVector = 0;

            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                if (kol_enter_value(MAX_VALUE_SIZE * tx + i) != 0 && pheromon_norm(i) != 0) {
                    svertka[i] = 1.0 / kol_enter_value(MAX_VALUE_SIZE * tx + i) + pheromon_norm(i);
                }
                else {
                    svertka[i] = 0.0;
                }
                sumVector += svertka[i];
            }

            norm_matrix_probability(MAX_VALUE_SIZE * tx) = svertka[0] / sumVector;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability(MAX_VALUE_SIZE * tx + i) = (svertka[i] / sumVector) + norm_matrix_probability(MAX_VALUE_SIZE * tx + i - 1);
            }
        }

        for (int bx = 0; bx < ANT_SIZE; bx++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                double randomValue = distribution(generator);
                int k = 0;
                while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability(MAX_VALUE_SIZE * tx + k)) {
                    k++;
                }
                ant_parametr(bx * PARAMETR_SIZE + tx) = k;
                agent(tx) = parametr_value(tx * MAX_VALUE_SIZE + k);
            }
            antOF(bx) = BenchShafferaFunction_non_cuda(agent);
        }

        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            pheromon_value.segment(MAX_VALUE_SIZE * tx, MAX_VALUE_SIZE) *= PARAMETR_RO;
        }

        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            for (int i = 0; i < ANT_SIZE; ++i) {
                int k = ant_parametr(i * PARAMETR_SIZE + tx);
                kol_enter_value(MAX_VALUE_SIZE * tx + k)++;
                if (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF(i) > 0) {
                    pheromon_value(MAX_VALUE_SIZE * tx + k) += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF(i)); // MIN
                }
            }
        }
    }
}
*/
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//armadillo
/*
#include <armadillo>
#include <iostream>
#include <random>

void load_matrix_non_cuda(const std::string& filename, arma::vec& parametr_value,
    arma::vec& pheromon_value, arma::vec& kol_enter_value) {
    // Реализуйте вашу функцию загрузки данных здесь
}

double BenchShafferaFunction_non_cuda(const arma::vec& agent) {
    // Реализуйте вашу функцию оценки здесь
    return 0.0; // Замените на реальную реализацию
}

void matrix_ACO_non_hash() {
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;

    arma::vec parametr_value(kolBytes_matrix_graph);
    arma::vec pheromon_value(kolBytes_matrix_graph);
    arma::vec kol_enter_value(kolBytes_matrix_graph);
    arma::vec norm_matrix_probability(kolBytes_matrix_graph);
    arma::vec ant(kolBytes_matrix_ant);
    arma::uvec ant_parametr(kolBytes_matrix_ant); // Используем uvec для индексов
    arma::vec antOF(ANT_SIZE);
    arma::vec agent(PARAMETR_SIZE);

    std::default_random_engine generator(123);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);

    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            double sumVector = arma::sum(pheromon_value.subvec(MAX_VALUE_SIZE * tx, MAX_VALUE_SIZE * (tx + 1) - 1));
            arma::vec pheromon_norm = pheromon_value.subvec(MAX_VALUE_SIZE * tx, MAX_VALUE_SIZE * (tx + 1) - 1) / sumVector;

            arma::vec svertka(MAX_VALUE_SIZE);
            sumVector = 0;

            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                if (kol_enter_value(MAX_VALUE_SIZE * tx + i) != 0 && pheromon_norm(i) != 0) {
                    svertka(i) = 1.0 / kol_enter_value(MAX_VALUE_SIZE * tx + i) + pheromon_norm(i);
                }
                else {
                    svertka(i) = 0.0;
                }
                sumVector += svertka(i);
            }

            norm_matrix_probability(MAX_VALUE_SIZE * tx) = svertka(0) / sumVector;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability(MAX_VALUE_SIZE * tx + i) = (svertka(i) / sumVector) + norm_matrix_probability(MAX_VALUE_SIZE * tx + i - 1);
            }
        }

        for (int bx = 0; bx < ANT_SIZE; bx++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                double randomValue = distribution(generator);
                int k = 0;
                while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability(MAX_VALUE_SIZE * tx + k)) {
                    k++;
                }
                ant_parametr(bx * PARAMETR_SIZE + tx) = k;
                agent(tx) = parametr_value(tx * MAX_VALUE_SIZE + k);
            }
            antOF(bx) = BenchShafferaFunction_non_cuda(agent);
        }

        pheromon_value *= PARAMETR_RO; // Умножаем всю матрицу феромонов на коэффициент

        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            for (int i = 0; i < ANT_SIZE; ++i) {
                int k = ant_parametr(i * PARAMETR_SIZE + tx);
                kol_enter_value(MAX_VALUE_SIZE * tx + k)++;
                if (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF(i) > 0) {
                    pheromon_value(MAX_VALUE_SIZE * tx + k) += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF(i)); // MIN
                }
            }
        }
    }
}
*/
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// Intel MKL
/*
#include <iostream>
#include <vector>
#include <random>
#include <mkl.h> // Подключаем Intel MKL
#include <cmath>

void matrix_ACO_non_hash() {
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;

    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* agent = new double[PARAMETR_SIZE];

    std::default_random_engine generator(123);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);

    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            double sumVector = cblas_dasum(MAX_VALUE_SIZE, pheromon_value + MAX_VALUE_SIZE * tx, 1);
            cblas_dscal(MAX_VALUE_SIZE, 1.0 / sumVector, pheromon_value + MAX_VALUE_SIZE * tx, 1);

            double svertka[MAX_VALUE_SIZE] = { 0 };
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                if (kol_enter_value[MAX_VALUE_SIZE * tx + i] != 0 && pheromon_value[MAX_VALUE_SIZE * tx + i] != 0) {
                    svertka[i] = 1.0 / kol_enter_value[MAX_VALUE_SIZE * tx + i] + pheromon_value[MAX_VALUE_SIZE * tx + i];
                }
                else {
                    svertka[i] = 0.0;
                }
            }

            sumVector = cblas_dasum(MAX_VALUE_SIZE, svertka, 1);
            norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
            }
        }

        for (int bx = 0; bx < ANT_SIZE; bx++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                double randomValue = distribution(generator);
                int k = 0;
                while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                    k++;
                }
                ant_parametr[bx * PARAMETR_SIZE + tx] = k;
                agent[tx] = parametr_value[tx * MAX_VALUE_SIZE + k];
            }
            antOF[bx] = BenchShafferaFunction_non_cuda(agent);
        }

        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            cblas_dscal(MAX_VALUE_SIZE, PARAMETR_RO, pheromon_value + MAX_VALUE_SIZE * tx, 1);
        }

        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            for (int i = 0; i < ANT_SIZE; ++i) {
                int k = ant_parametr[i * PARAMETR_SIZE + tx];
                kol_enter_value[MAX_VALUE_SIZE * tx + k]++;
                if (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF[i] > 0) {
                    pheromon_value[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF[i]);
                }
            }
        }
    }

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] agent;
}
*/
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//Blaze
/*
#include <blaze/Blaze.h>

// Прототипы ваших функций
void load_matrix_non_cuda(const std::string& filename, blaze::DynamicMatrix<double>& parametr_value,
    blaze::DynamicMatrix<double>& pheromon_value, blaze::DynamicMatrix<double>& kol_enter_value);
double BenchShafferaFunction_non_cuda(const blaze::DynamicVector<double>& agent);

void matrix_ACO_non_hash() {
    // Создаем матрицы и векторы с помощью Blaze
    blaze::DynamicMatrix<double> parametr_value(MAX_VALUE_SIZE, PARAMETR_SIZE);
    blaze::DynamicMatrix<double> pheromon_value(MAX_VALUE_SIZE, PARAMETR_SIZE);
    blaze::DynamicMatrix<double> kol_enter_value(MAX_VALUE_SIZE, PARAMETR_SIZE);
    blaze::DynamicMatrix<double> norm_matrix_probability(MAX_VALUE_SIZE, PARAMETR_SIZE);
    blaze::DynamicMatrix<double> ant(ANT_SIZE, PARAMETR_SIZE);
    blaze::DynamicVector<int> ant_parametr(ANT_SIZE * PARAMETR_SIZE);
    blaze::DynamicVector<double> antOF(ANT_SIZE);
    blaze::DynamicVector<double> agent(PARAMETR_SIZE);

    std::default_random_engine generator(123);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);

    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            double sumVector = 0;
            blaze::DynamicVector<double> pheromon_norm(MAX_VALUE_SIZE);

            // Суммируем феромоны
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                sumVector += pheromon_value(i, tx);
            }

            // Нормируем феромоны
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                pheromon_norm[i] = pheromon_value(i, tx) / sumVector;
            }

            sumVector = 0;
            blaze::DynamicVector<double> svertka(MAX_VALUE_SIZE);

            // Считаем свертку
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                if ((kol_enter_value(i, tx) != 0) && (pheromon_norm[i] != 0)) {
                    svertka[i] = 1.0 / kol_enter_value(i, tx) + pheromon_norm[i];
                }
                else {
                    svertka[i] = 0.0;
                }
                sumVector += svertka[i];
            }

            norm_matrix_probability(0, tx) = svertka[0] / sumVector;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability(i, tx) = (svertka[i] / sumVector) + norm_matrix_probability(i - 1, tx);
            }
        }

        for (int bx = 0; bx < ANT_SIZE; bx++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                double randomValue = distribution(generator);
                int k = 0;
                while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability(k, tx)) {
                    k++;
                }
                ant_parametr[bx * PARAMETR_SIZE + tx] = k;
                agent[tx] = parametr_value(k, tx);
            }
            antOF[bx] = BenchShafferaFunction_non_cuda(agent);
        }

        // Обновление феромонов
        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
                pheromon_value(i, tx) *= PARAMETR_RO;
            }
        }

        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            for (int i = 0; i < ANT_SIZE; ++i) {
                int k = ant_parametr[i * PARAMETR_SIZE + tx];
                kol_enter_value(k, tx)++;
                if (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF[i] > 0) {
                    pheromon_value(k, tx) += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF[i]); // MIN
                }
            }
        }
    }
}

*/
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// Ceres Solver
/*
#include <ceres/ceres.h>

struct PheromoneCostFunction {
    PheromoneCostFunction(double target_value) : target_value(target_value) {}

    template <typename T>
    bool operator()(const T* const pheromone, T* residual) const {
        // Целевая функция: минимизируем разницу между феромоном и целевым значением
        residual[0] = pheromone[0] - T(target_value);
        return true;
    }

    double target_value;
};

void matrix_ACO_non_hash() {
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* agent = new double[PARAMETR_SIZE];

    std::default_random_engine generator(123);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);

    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            double sumVector = 0;
            double pheromon_norm[MAX_VALUE_SIZE] = { 0 };

            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                sumVector += pheromon_value[MAX_VALUE_SIZE * tx + i];
            }

            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                pheromon_norm[i] = pheromon_value[MAX_VALUE_SIZE * tx + i] / sumVector;
            }

            sumVector = 0;
            double svertka[MAX_VALUE_SIZE] = { 0 };

            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                if ((kol_enter_value[MAX_VALUE_SIZE * tx + i] != 0) && (pheromon_norm[i] != 0)) {
                    svertka[i] = 1.0 / kol_enter_value[MAX_VALUE_SIZE * tx + i] + pheromon_norm[i];
                }
                else {
                    svertka[i] = 0.0;
                }
                sumVector += svertka[i];
            }

            norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
            }
        }

        for (int bx = 0; bx < ANT_SIZE; bx++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                double randomValue = distribution(generator);
                int k = 0;
                while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                    k++;
                }
                ant_parametr[bx * PARAMETR_SIZE + tx] = k;
                agent[tx] = parametr_value[tx * MAX_VALUE_SIZE + k];
            }
            antOF[bx] = BenchShafferaFunction_non_cuda(agent);
        }

        // Обновление феромонов с использованием Ceres Solver
        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            ceres::Problem problem;

            for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
                // Создаем новую переменную для феромона
                double* pheromone_param = &pheromon_value[MAX_VALUE_SIZE * tx + i];

                // Создаем стоимость для оптимизации
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<PheromoneCostFunction, 1, 1>(
                        new PheromoneCostFunction(MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF[i])
                    ),
                    nullptr,
                    pheromone_param
                );
            }

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = true;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
        }

        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
                kol_enter_value[MAX_VALUE_SIZE * tx + i]++;
            }
        }
    }

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] agent;
}
*/
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// Windows Proc
/*
// Структура для передачи параметров в поток
#include <windows.h>
struct TaskParams {
    double* parametr;
    double* norm_matrix_probability;
    double* agent;
    int* agent_node;
    double* OF;
    HashEntry* hashTable;
    int bx;
    int& kol_hash_fail; // Ссылка на переменную для подсчета хэш-неудач
};

DWORD WINAPI WorkerThread(LPVOID lpParam) {
    TaskParams* params = reinterpret_cast<TaskParams*>(lpParam);

    std::default_random_engine generator(rand()); // Генератор случайных чисел для каждого потока
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    double* agent = params->agent;
    int* agent_node = params->agent_node;
    double* OF = params->OF;
    HashEntry* hashTable = params->hashTable;
    int bx = params->bx;
    int& kol_hash_fail = params->kol_hash_fail;

    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double randomValue = distribution(generator);
        int k = 0;

        while (k < MAX_VALUE_SIZE && randomValue > params->norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
            k++;
        }

        agent_node[bx * PARAMETR_SIZE + tx] = k;
        agent[bx * PARAMETR_SIZE + tx] = params->parametr[tx * MAX_VALUE_SIZE + k];
    }

    double cachedResult = getCachedResultOptimized_non_cuda(hashTable, agent_node, bx);
    int nom_iteration = 0;

    if (cachedResult == -1.0) {
        OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
        saveToCacheOptimized_non_cuda(hashTable, agent_node, bx, OF[bx]);
    }
    else {
        switch (TYPE_ACO) {
        case 0: // ACOCN
            OF[bx] = cachedResult;
            kol_hash_fail++;
            break;
        case 1: // ACOCNI
            OF[bx] = ZERO_HASH_RESULT;
            kol_hash_fail++;
            break;
        case 2: // ACOCCyN
            while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION)) {
                for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                    double randomValue = distribution(generator);
                    int k = 0;

                    while (k < MAX_VALUE_SIZE && randomValue > params->norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                        k++;
                    }

                    agent_node[bx * PARAMETR_SIZE + tx] = k;
                    agent[bx * PARAMETR_SIZE + tx] = params->parametr[tx * MAX_VALUE_SIZE + k];
                }
                cachedResult = getCachedResultOptimized_non_cuda(hashTable, agent_node, bx);
                nom_iteration++;
                kol_hash_fail++;
            }

            OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
            saveToCacheOptimized_non_cuda(hashTable, agent_node, bx, OF[bx]);
            break;
        default:
            OF[bx] = cachedResult;
            kol_hash_fail++;
            break;
        }
    }

    return 0;
}

void go_all_agent_non_cuda(double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail) {
    HANDLE threads[ANT_SIZE];
    TaskParams taskParams[ANT_SIZE];

    for (int bx = 0; bx < ANT_SIZE; bx++) {
        taskParams[bx] = { parametr, norm_matrix_probability, agent, agent_node, OF, hashTable, bx, kol_hash_fail };

        threads[bx] = CreateThread(nullptr, 0, WorkerThread, &taskParams[bx], 0, nullptr);
        if (threads[bx] == nullptr) {
            std::cerr << "Error creating thread: " << GetLastError() << std::endl;
            return;
        }
    }

    WaitForMultipleObjects(ANT_SIZE, threads, TRUE, INFINITE);

    for (int bx = 0; bx < ANT_SIZE; bx++) {
        CloseHandle(threads[bx]);
    }

    std::cout << "All tasks are completed." << std::endl;
}

void go_all_agent_MPI(int rank_MSI, int size_MSI, int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, int* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail, double& totalHashTime, double& totalOFTime) {
    // Генератор случайных чисел
    std::default_random_engine generator(123 + gpuTime); // Используем gpuTime как начальное значение
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int bx = rank_MSI; bx < ANT_SIZE; bx+=size_MSI) { // Проходим по всем агентам
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
        double cachedResult = getCachedResultOptimized_non_cuda(hashTable, agent_node, bx);

        int nom_iteration = 0;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            auto start_OF = std::chrono::high_resolution_clock::now();
            OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
            auto end_OF = std::chrono::high_resolution_clock::now();
            totalOFTime += std::chrono::duration<double, std::milli>(end_OF - start_OF).count();
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

                //auto start_OF_2 = std::chrono::high_resolution_clock::now();
                OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
                //auto end_OF_2 = std::chrono::high_resolution_clock::now();
                //totalOFTime += std::chrono::duration<double, std::milli>(end_OF_2 - start_OF_2).count();
                saveToCacheOptimized_non_cuda(hashTable, agent_node, bx, OF[bx]);
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
*/
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
void matrix_ACO2_non_hash() {
    double SumTime1 = 0.0, SumTime2 = 0.0;

    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* agent = new double[PARAMETR_SIZE];
    std::default_random_engine generator(123);
    std::default_random_engine generator1(123);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        load_matrix_transp_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);
        auto start1 = std::chrono::high_resolution_clock::now();
        //Сумма Тi для Tnorm
        double sumVectorT[PARAMETR_SIZE] = { 0 };
        double sumVectorZ[PARAMETR_SIZE] = { 0 };
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            sumVectorT[tx] = 0.0;
            sumVectorZ[tx] = 0.0;
        }
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                sumVectorT[tx] += pheromon_value[tx + i * PARAMETR_SIZE];
            }
        }
        //Вычисление Tnorm
        double* pheromon_norm = new double[kolBytes_matrix_graph];
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                pheromon_norm[tx + i * PARAMETR_SIZE] = pheromon_value[tx + i * PARAMETR_SIZE] / sumVectorT[tx];
            }
        }
        //Вычисление Z и P
        double* svertka = new double[kolBytes_matrix_graph];
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                if ((kol_enter_value[tx + i * PARAMETR_SIZE] != 0) && (pheromon_norm[tx + i * PARAMETR_SIZE] != 0)) {
                    svertka[tx + i * PARAMETR_SIZE] = 1.0 / kol_enter_value[tx + i * PARAMETR_SIZE] + pheromon_norm[tx + i * PARAMETR_SIZE];
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
        auto end1 = std::chrono::high_resolution_clock::now();
        SumTime1 = std::chrono::duration<float, std::milli>(end1 - start1).count();
        std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << SumTime1 << std::endl;
        for (int i = 0; i < PARAMETR_SIZE; ++i) {
            for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                std::cout << parametr_value[i + j * PARAMETR_SIZE] << "(" << pheromon_value[i + j * PARAMETR_SIZE] << ", " << kol_enter_value[i + j * PARAMETR_SIZE] << "-> " << pheromon_norm[i + j * PARAMETR_SIZE] << "+" << svertka[i + j * PARAMETR_SIZE] << ";" << norm_matrix_probability[i + j * PARAMETR_SIZE] << ") "; // Индексируем элементы
            }
            std::cout << std::endl; // Переход на новую строку
        }


        start1 = std::chrono::high_resolution_clock::now();
        //Сумма Тi для Tnorm
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            sumVectorT[tx] = 0.0;
            sumVectorZ[tx] = 0.0;
        }
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                sumVectorT[tx] += pheromon_value[tx + i * PARAMETR_SIZE];
            }
        }
        //Вычисление Tnorm
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx += CONST_AVX) {
                // Загружаем значения из pheromon_value и sumVectorT
                __m256d pheromon_values_AVX = _mm256_loadu_pd(&pheromon_value[tx + i * PARAMETR_SIZE]);
                __m256d sum_vector_AVX = _mm256_loadu_pd(&sumVectorT[tx]);

                // Выполняем деление
                __m256d pheromon_norm_values_AVX = _mm256_div_pd(pheromon_values_AVX, sum_vector_AVX);
                _mm256_storeu_pd(&pheromon_norm[tx + i * PARAMETR_SIZE], pheromon_norm_values_AVX);
            }
        }
        // Вычисление Z и P
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx += CONST_AVX) {
                // Загружаем значения из kol_enter_value и pheromon_norm
                __m256d kol_enter_values_AVX = _mm256_loadu_pd(&kol_enter_value[tx + i * PARAMETR_SIZE]);
                __m256d pheromon_norm_values_AVX = _mm256_loadu_pd(&pheromon_norm[tx + i * PARAMETR_SIZE]);

                // Создаем маску для проверки условий
                //(kol_enter_value[tx + i * PARAMETR_SIZE] != 0) && (pheromon_norm[tx + i * PARAMETR_SIZE] != 0)
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
        end1 = std::chrono::high_resolution_clock::now();
        SumTime1 = std::chrono::duration<float, std::milli>(end1 - start1).count();
        std::cout << "?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????" << std::endl;
        std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << SumTime1 << std::endl;
        for (int i = 0; i < PARAMETR_SIZE; ++i) {
            for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                std::cout << parametr_value[i + j * PARAMETR_SIZE] << "(" << pheromon_value[i + j * PARAMETR_SIZE] << ", " << kol_enter_value[i + j * PARAMETR_SIZE] << "-> " << pheromon_norm[i + j * PARAMETR_SIZE] << "+" << svertka[i + j * PARAMETR_SIZE] << ";" << norm_matrix_probability[i + j * PARAMETR_SIZE] << ") "; // Индексируем элементы
            }
            std::cout << std::endl; // Переход на новую строку
        }

        for (int bx = 0; bx < ANT_SIZE; bx++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                double randomValue = distribution(generator);
                int k = 0;
                while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[tx + k * PARAMETR_SIZE]) {
                    k++;
                }
                std::cout << randomValue << "; " << k << ")";
                ant_parametr[bx * PARAMETR_SIZE + tx] = k;
                agent[tx] = parametr_value[tx + k * PARAMETR_SIZE];
            }
            antOF[bx] = BenchShafferaFunction_non_cuda(agent);
        }
        std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
        for (int i = 0; i < ANT_SIZE; ++i) {
            for (int j = 0; j < PARAMETR_SIZE; ++j) {
                std::cout << ant_parametr[i * PARAMETR_SIZE + j] << " ";

            }
            std::cout << "-> " << antOF[i] << std::endl;

        }
        /*
        for (int bx = 0; bx < ANT_SIZE; bx++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx += CONST_AVX) {
                __m256d randomValues_AVX = _mm256_set_pd(distribution(generator), distribution(generator),
                    distribution(generator), distribution(generator));
                __m256d k_AVX = _mm256_setzero_pd(); // Для хранения значений k
                __m256d indices_AVX = _mm256_set_pd(3.0, 2.0, 1.0, 0.0); // Индексы для доступа к norm_matrix_probability
                for (int kIndex = 0; kIndex < MAX_VALUE_SIZE; ++kIndex) {
                    // Загружаем значения из norm_matrix_probability
                    __m256d norm_matrix_probability_AVX = _mm256_loadu_pd(&norm_matrix_probability[tx + kIndex * PARAMETR_SIZE]);
                    // Сравниваем randomValues с norm_matrix_probability
                    __m256d cmpResult = _mm256_cmp_pd(randomValues_AVX, norm_matrix_probability_AVX, _CMP_GT_OS);
                    // Увеличиваем k для тех, кто меньше
                    k_AVX = _mm256_add_pd(k_AVX, _mm256_and_pd(cmpResult, indices_AVX));
                }


                // Сохраняем k в ant_parametr
                _mm256_storeu_pd(&ant_parametr[bx * PARAMETR_SIZE + tx], k_AVX);

                // Получаем значения agent на основе k
                for (int i = 0; i < 4; ++i) {
                    int index = static_cast<int>(_mm256_extract_epi64(k_AVX, i)); // Извлекаем значение k
                    agent[tx + i] = parametr_value[tx + index * PARAMETR_SIZE];
                }
            }
            antOF[bx] = BenchShafferaFunction_non_cuda(agent);
        }
        std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
        for (int i = 0; i < ANT_SIZE; ++i) {
            for (int j = 0; j < PARAMETR_SIZE; ++j) {
                std::cout << ant_parametr[i * PARAMETR_SIZE + j] << " ";

            }
            std::cout << "-> " << antOF[i] << std::endl;

        }*/
        for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
            for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                pheromon_value[tx + i * PARAMETR_SIZE] *= PARAMETR_RO;
            }
        }

        for (int i = 0; i < ANT_SIZE; ++i) {
            for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                int k = int(ant_parametr[tx + i * PARAMETR_SIZE]);
                kol_enter_value[tx + k * PARAMETR_SIZE]++;
                //            pheromon[tx + k * PARAMETR_SIZE] += PARAMETR_Q * OF[i]; // MAX
                //            pheromon[tx + k * PARAMETR_SIZE] += PARAMETR_Q / OF[i]; // MIN
                if (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF[i] > 0) {
                    pheromon_value[tx + k * PARAMETR_SIZE] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF[i]); // MIN
                }
            }
        }
        std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        std::cout << "Matrix1 (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << SumTime1 << std::endl;
        for (int i = 0; i < PARAMETR_SIZE; ++i) {
            for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                std::cout << parametr_value[i + j * PARAMETR_SIZE] << "(" << pheromon_value[i + j * PARAMETR_SIZE] << ", " << kol_enter_value[i + j * PARAMETR_SIZE] << ") "; // Индексируем элементы
            }
            std::cout << std::endl; // Переход на новую строку
        }

        __m256d parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);
        for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx += CONST_AVX) {
                //pheromon_value[tx + i * PARAMETR_SIZE] *= PARAMETR_RO;
                // Загружаем 4 значения из pheromon_value
                __m256d pheromonValues_AVX = _mm256_loadu_pd(&pheromon_value[tx + i * PARAMETR_SIZE]);
                // Умножаем на PARAMETR_RO
                pheromonValues_AVX = _mm256_mul_pd(pheromonValues_AVX, parametRovector_AVX);
                // Сохраняем обратно в pheromon_value
                _mm256_storeu_pd(&pheromon_value[tx + i * PARAMETR_SIZE], pheromonValues_AVX);
            }
        }

        for (int i = 0; i < ANT_SIZE; ++i) {
            for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                int k = int(ant_parametr[tx + i * PARAMETR_SIZE]);
                kol_enter_value[tx + k * PARAMETR_SIZE]++;
                //            pheromon[tx + k * PARAMETR_SIZE] += PARAMETR_Q * OF[i]; // MAX
                //            pheromon[tx + k * PARAMETR_SIZE] += PARAMETR_Q / OF[i]; // MIN
                if (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF[i] > 0) {
                    pheromon_value[tx + k * PARAMETR_SIZE] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF[i]); // MIN
                }
            }
        }
        std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        std::cout << "Matrix2 (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << SumTime1 << std::endl;
        for (int i = 0; i < PARAMETR_SIZE; ++i) {
            for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                std::cout << parametr_value[i + j * PARAMETR_SIZE] << "(" << pheromon_value[i + j * PARAMETR_SIZE] << ", " << kol_enter_value[i + j * PARAMETR_SIZE] << ") "; // Индексируем элементы
            }
            std::cout << std::endl; // Переход на новую строку
        }

        load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);
        start1 = std::chrono::high_resolution_clock::now();
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            double sumVector = 0;
            double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
            double svertka[MAX_VALUE_SIZE] = { 0 };
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                sumVector += pheromon_value[MAX_VALUE_SIZE * tx + i];
            }
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                pheromon_norm[i] = pheromon_value[MAX_VALUE_SIZE * tx + i] / sumVector;
            }
            sumVector = 0;
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                if ((kol_enter_value[MAX_VALUE_SIZE * tx + i] != 0) && (pheromon_norm[i] != 0)) {
                    svertka[i] = 1.0 / kol_enter_value[MAX_VALUE_SIZE * tx + i] + pheromon_norm[i];
                }
                else
                {
                    svertka[i] = 0.0;
                }
                sumVector += svertka[i];
            }

            norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
            }
        }
        end1 = std::chrono::high_resolution_clock::now();
        SumTime2 = std::chrono::duration<float, std::milli>(end1 - start1).count();
        std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << SumTime2 << std::endl;
        for (int i = 0; i < PARAMETR_SIZE; ++i) {
            for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "+ " << svertka[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
            }
            std::cout << std::endl; // Переход на новую строку
        }
        start1 = std::chrono::high_resolution_clock::now();
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            // Вычисляем sumVector
            double sumVector = 0.0;
            __m256d sumVectorAVX = _mm256_setzero_pd();
            for (int i = 0; i < MAX_VALUE_SIZE; i += 4) {
                __m256d pheromonValues = _mm256_loadu_pd(&pheromon_value[MAX_VALUE_SIZE * tx + i]);
                sumVectorAVX = _mm256_add_pd(sumVectorAVX, pheromonValues);
            }
            // Суммируем значения из вектора
            double temp[4];
            _mm256_storeu_pd(temp, sumVectorAVX);
            for (int j = 0; j < 4; j++) {
                sumVector += temp[j];
            }
            // Нормируем значения
            double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
            for (int i = 0; i < MAX_VALUE_SIZE; i += 4) {
                __m256d pheromonValues = _mm256_loadu_pd(&pheromon_value[MAX_VALUE_SIZE * tx + i]);
                __m256d normValues = _mm256_div_pd(pheromonValues, _mm256_set1_pd(sumVector));
                _mm256_storeu_pd(&pheromon_norm[i], normValues);
            }
            // Вычисляем svertka и sumVector
            sumVector = 0.0;
            double svertka[MAX_VALUE_SIZE] = { 0 };
            sumVectorAVX = _mm256_setzero_pd();
            for (int i = 0; i < MAX_VALUE_SIZE; i += 4) {
                __m256d kolEnterValues = _mm256_loadu_pd(&kol_enter_value[MAX_VALUE_SIZE * tx + i]);
                __m256d pheromonNormValues = _mm256_loadu_pd(&pheromon_norm[i]);
                __m256d svertkaValues;
                // Создаем маску для проверки условий
                __m256d mask = _mm256_cmp_pd(kolEnterValues, _mm256_setzero_pd(), _CMP_NEQ_OQ);
                mask = _mm256_and_pd(mask, _mm256_cmp_pd(pheromonNormValues, _mm256_setzero_pd(), _CMP_NEQ_OQ));
                // Вычисляем svertka с учетом условий
                __m256d oneOverKolEnter = _mm256_div_pd(_mm256_set1_pd(1.0), kolEnterValues);
                svertkaValues = _mm256_add_pd(oneOverKolEnter, pheromonNormValues);
                svertkaValues = _mm256_blendv_pd(_mm256_setzero_pd(), svertkaValues, mask);
                // Сохраняем результат
                _mm256_storeu_pd(&svertka[i], svertkaValues);
                // Суммируем svertka
                sumVectorAVX = _mm256_add_pd(sumVectorAVX, svertkaValues);
            }
            // Суммируем значения из вектора svertka
            _mm256_storeu_pd(temp, sumVectorAVX);
            for (int j = 0; j < 4; j++) {
                sumVector += temp[j];
            }
            // Заполняем norm_matrix_probability
            norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
            }
        }
        end1 = std::chrono::high_resolution_clock::now();
        SumTime2 = std::chrono::duration<float, std::milli>(end1 - start1).count();
        std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << SumTime2 << std::endl;
        for (int i = 0; i < PARAMETR_SIZE; ++i) {
            for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "+ " << svertka[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
            }
            std::cout << std::endl; // Переход на новую строку
        }

        for (int bx = 0; bx < ANT_SIZE; bx++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                double randomValue = distribution(generator1);
                int k = 0;
                while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                    k++;
                }
                ant_parametr[bx * PARAMETR_SIZE + tx] = k;
                agent[tx] = parametr_value[tx * MAX_VALUE_SIZE + k];
            }
            antOF[bx] = BenchShafferaFunction_non_cuda(agent);
        }
        std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
        for (int i = 0; i < ANT_SIZE; ++i) {
            for (int j = 0; j < PARAMETR_SIZE; ++j) {
                std::cout << ant_parametr[i * PARAMETR_SIZE + j] << " ";

            }
            std::cout << "-> " << antOF[i] << std::endl;

        }
        /*
        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
                pheromon_value[MAX_VALUE_SIZE * tx + i] *= PARAMETR_RO;
            }
        }
        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            for (int i = 0; i < ANT_SIZE; ++i) {
                int k = int(ant_parametr[i * PARAMETR_SIZE + tx]);
                kol_enter_value[MAX_VALUE_SIZE * tx + k]++;
                //            pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * OF[i]; // MAX
                //            pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q / OF[i]; // MIN
                if (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF[i] > 0) {
                    pheromon_value[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF[i]); // MIN
                }
            }
        }
        std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << SumTime2 << std::endl;
        for (int i = 0; i < PARAMETR_SIZE; ++i) {
            for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
            }
            std::cout << std::endl; // Переход на новую строку
        }

        */
        // Умножение pheromon_value на PARAMETR_RO
        parametRovector_AVX = _mm256_set1_pd(PARAMETR_RO);
        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            for (int i = 0; i < MAX_VALUE_SIZE; i += CONST_AVX) {
                // Загружаем 4 значения из pheromon_value
                __m256d pheromonValues_AVX = _mm256_loadu_pd(&pheromon_value[MAX_VALUE_SIZE * tx + i]);
                // Умножаем на PARAMETR_RO
                pheromonValues_AVX = _mm256_mul_pd(pheromonValues_AVX, parametRovector_AVX);
                // Сохраняем обратно в pheromon_value
                _mm256_storeu_pd(&pheromon_value[MAX_VALUE_SIZE * tx + i], pheromonValues_AVX);
            }
        }

        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            for (int i = 0; i < ANT_SIZE; ++i) {
                int k = int(ant_parametr[i * PARAMETR_SIZE + tx]);
                kol_enter_value[MAX_VALUE_SIZE * tx + k]++;
                //            pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * OF[i]; // MAX
                //            pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q / OF[i]; // MIN
                if (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF[i] > 0) {
                    pheromon_value[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF[i]); // MIN
                }
            }
        }
        std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << SumTime2 << std::endl;
        for (int i = 0; i < PARAMETR_SIZE; ++i) {
            for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
            }
            std::cout << std::endl; // Переход на новую строку
        }

    }
    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] agent;
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
    }
    // Закрытие лог-файла
    logFile.close();
    outfile.close();
}


//код без лишнего для матричного ACO
void matrix_ACO() {
    int kol_hash_fail = 0;
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    HashEntry* hashTable = new HashEntry[HASH_TABLE_SIZE];
    initializeHashTable_non_cuda(hashTable, HASH_TABLE_SIZE);
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* agent = new double[PARAMETR_SIZE];
    std::default_random_engine generator(123);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            double sumVector = 0;
            double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                sumVector += pheromon_value[MAX_VALUE_SIZE * tx + i];
            }
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                pheromon_norm[i] = pheromon_value[MAX_VALUE_SIZE * tx + i] / sumVector;
            }
            sumVector = 0;
            double svertka[MAX_VALUE_SIZE] = { 0 };
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                if ((kol_enter_value[MAX_VALUE_SIZE * tx + i] != 0) && (pheromon_norm[i] != 0)) {
                    svertka[i] = 1.0 / kol_enter_value[MAX_VALUE_SIZE * tx + i] + pheromon_norm[i];
                }
                else
                {
                    svertka[i] = 0.0;
                }
                sumVector += svertka[i];
            }
            norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
            }
        }

        for (int bx = 0; bx < ANT_SIZE; bx++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                double randomValue = distribution(generator);
                int k = 0;
                while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                    k++;
                }
                ant_parametr[bx * PARAMETR_SIZE + tx] = k;
                agent[tx] = parametr_value[tx * MAX_VALUE_SIZE + k];
            }
            double cachedResult = getCachedResultOptimized_non_cuda(hashTable, ant_parametr, bx);
            int nom_iteration = 0;
            if (cachedResult == -1.0) {
                antOF[bx] = BenchShafferaFunction_non_cuda(agent);
                saveToCacheOptimized_non_cuda(hashTable, ant_parametr, bx, antOF[bx]);
            }
            else {
                switch (TYPE_ACO) {
                case 0: // ACOCN
                    antOF[bx] = cachedResult;
                    kol_hash_fail = kol_hash_fail + 1;
                    break;
                case 1: // ACOCNI
                    antOF[bx] = ZERO_HASH_RESULT;
                    kol_hash_fail = kol_hash_fail + 1;
                    break;
                case 2: // ACOCCyN
                    while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                    {
                        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
                            double randomValue = distribution(generator);
                            int k = 0;
                            while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                                k++;
                            }
                            ant_parametr[bx * PARAMETR_SIZE + tx] = k;
                            agent[tx] = parametr_value[tx * MAX_VALUE_SIZE + k];
                        }
                        cachedResult = getCachedResultOptimized_non_cuda(hashTable, ant_parametr, bx);
                        nom_iteration = nom_iteration + 1;
                        kol_hash_fail = kol_hash_fail + 1;
                    }

                    antOF[bx] = BenchShafferaFunction_non_cuda(agent);
                    saveToCacheOptimized_non_cuda(hashTable, ant_parametr, bx, antOF[bx]);
                    break;
                default:
                    antOF[bx] = cachedResult;
                    kol_hash_fail = kol_hash_fail + 1;
                    break;
                }
            }
        }

        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
                pheromon_value[MAX_VALUE_SIZE * tx + i] *= PARAMETR_RO;
            }
        }
        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            for (int i = 0; i < ANT_SIZE; ++i) {
                int k = int(ant_parametr[i * PARAMETR_SIZE + tx]);
                kol_enter_value[MAX_VALUE_SIZE * tx + k]++;
                //            pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * OF[i]; // MAX
                //            pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q / OF[i]; // MIN
                if (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF[i] > 0) {
                    pheromon_value[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF[i]); // MIN
                }
            }
        }

    }
    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] agent;
}

void matrix_ACO_non_hash() {
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    int* ant_parametr = new int[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* agent = new double[PARAMETR_SIZE];
    std::default_random_engine generator(123);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
            double sumVector = 0;
            double pheromon_norm[MAX_VALUE_SIZE] = { 0 };
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                sumVector += pheromon_value[MAX_VALUE_SIZE * tx + i];
            }
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                pheromon_norm[i] = pheromon_value[MAX_VALUE_SIZE * tx + i] / sumVector;
            }
            sumVector = 0;
            double svertka[MAX_VALUE_SIZE] = { 0 };
            for (int i = 0; i < MAX_VALUE_SIZE; i++) {
                if ((kol_enter_value[MAX_VALUE_SIZE * tx + i] != 0) && (pheromon_norm[i] != 0)) {
                    svertka[i] = 1.0 / kol_enter_value[MAX_VALUE_SIZE * tx + i] + pheromon_norm[i];
                }
                else
                {
                    svertka[i] = 0.0;
                }
                sumVector += svertka[i];
            }
            norm_matrix_probability[MAX_VALUE_SIZE * tx] = svertka[0] / sumVector;
            for (int i = 1; i < MAX_VALUE_SIZE; i++) {
                norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i] / sumVector) + norm_matrix_probability[MAX_VALUE_SIZE * tx + i - 1];
            }
        }

        for (int bx = 0; bx < ANT_SIZE; bx++) {
            for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
                double randomValue = distribution(generator);
                int k = 0;
                while (k < MAX_VALUE_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                    k++;
                }
                ant_parametr[bx * PARAMETR_SIZE + tx] = k;
                agent[tx] = parametr_value[tx * MAX_VALUE_SIZE + k];
            }
            antOF[bx] = BenchShafferaFunction_non_cuda(agent);
        }

        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
                pheromon_value[MAX_VALUE_SIZE * tx + i] *= PARAMETR_RO;
            }
        }
        for (int tx = 0; tx < PARAMETR_SIZE; ++tx) {
            for (int i = 0; i < ANT_SIZE; ++i) {
                int k = int(ant_parametr[i * PARAMETR_SIZE + tx]);
                kol_enter_value[MAX_VALUE_SIZE * tx + k]++;
                //            pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * OF[i]; // MAX
                //            pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q / OF[i]; // MIN
                if (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF[i] > 0) {
                    pheromon_value[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - antOF[i]); // MIN
                }
            }
        }

    }
    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    delete[] agent;
}

/*
Давайте рассмотрим детально каждую предложенную технику и реализуем соответствующие фрагменты кода.

1. Метод псевдонима (Alias Method)

Метод псевдонима позволяет создать специальную таблицу, которая преобразует равномерное распределение случайных чисел в произвольное заданное распределение вероятностей. Выбор одного значения из списка вероятностей осуществляется за константное время ($O(1)$), что делает этот подход крайне эффективным.

Структура Alias Table Entry:

struct AliasTableEntry {
    uint32_t prob_idx;       // Индекс оригинальной вероятности
    uint32_t alias_idx;      // Индекс альтернативного варианта ("псевдонима")
    float prob_weight;        // Относительный вес вероятности
};

Построение таблицы псевдонимов:

void build_alias_table(AliasTableEntry* table, const float* probabilities, int num_entries) {
    // Массивы для подсчета малых и больших вероятностей
    std::vector<uint32_t> small(num_entries), large(num_entries);
    std::queue<uint32_t> smallQueue, largeQueue;

    // Начальная инициализация очереди
    for(uint32_t i = 0; i < num_entries; ++i) {
        table[i].prob_idx = i;
        table[i].prob_weight = probabilities[i] * num_entries;
        if(table[i].prob_weight < 1.0f) {
            smallQueue.push(i);
        } else {
            largeQueue.push(i);
        }
    }

    // Основной цикл формирования таблицы
    while(!smallQueue.empty() && !largeQueue.empty()) {
        uint32_t less = smallQueue.front(); smallQueue.pop();
        uint32_t more = largeQueue.front(); largeQueue.pop();

        table[less].alias_idx = more;
        table[more].prob_weight -= (1.0f - table[less].prob_weight);

        if(table[more].prob_weight < 1.0f) {
            smallQueue.push(more);
        } else {
            largeQueue.push(more);
        }
    }

    // Завершаем оставшиеся крупные элементы
    while(!largeQueue.empty()) {
        uint32_t idx = largeQueue.front(); largeQueue.pop();
        table[idx].alias_idx = idx;
    }

    // Осталось обработать маленькие элементы
    while(!smallQueue.empty()) {
        uint32_t idx = smallQueue.front(); smallQueue.pop();
        table[idx].alias_idx = idx;
    }
}

Финальный выбор случайного индекса:

uint32_t select_random_entry(float rnd_num, const AliasTableEntry* table, int num_entries) {
    int idx = floor(rnd_num * num_entries);
    float random_prob = rnd_num * num_entries - idx;

    if(random_prob < table[idx].prob_weight) {
        return table[idx].prob_idx;
    } else {
        return table[idx].alias_idx;
    }
}

Примеры использования метода псевдонима:

Предположим, у вас есть массив вероятностей и вы хотите сделать быстрый выбор.

float probabilities[] = {0.1, 0.2, 0.3, 0.4}; // Сумма равна единице
AliasTableEntry table[sizeof(probabilities)/sizeof(probabilities[0])];
build_alias_table(table, probabilities, sizeof(probabilities)/sizeof(probabilities[0]));

std::default_random_engine gen(std::chrono::system_clock::now().time_since_epoch().count());
std::uniform_real_distribution<float> dist(0.0, 1.0);

// Сделаем несколько быстрых выборов
for(int i = 0; i < 10; ++i) {
    float rnd = dist(gen);
    uint32_t selected = select_random_entry(rnd, table, sizeof(probabilities)/sizeof(probabilities[0]));
    printf("Selected %d\n", selected);
}

2. SIMD-ускорение (AVX2)

Intel Advanced Vector Extensions (AVX2) поддерживают одновременную обработку 8 пар вещественных чисел двойной точности (double) за одну инструкцию. Используя SIMD-векторизацию, можно значительно ускорить вычислительные этапы, такие как расчеты вероятностей и выборку индексов.

Пример реализации выборки с использованием AVX2:

Сначала определим макросы для удобного обращения к AVX2-инструкциям:

#include <immintrin.h>

#define LOAD_M256D(addr) _mm256_loadu_pd((const double*) addr)
#define STORE_M256D(addr, val) _mm256_storeu_pd((double*) addr, val)
#define SET_ONE _mm256_set1_pd(1.0)
#define CMP_GT(a,b) _mm256_cmp_pd(a, b, _CMP_GT_OQ)
#define MOVEMASK(pd) _mm256_movemask_pd(pd)

Теперь реализуйте процедуру выборки с использованием SIMD:

int select_random_simd(double randomValue, const double* probabilities, int length) {
    __m256d comp_value = SET_ONE * randomValue;
    __m256d current_probs;
    int first_valid = -1;

    for(int i = 0; i < length; i += 8) {
        current_probs = LOAD_M256D(&probabilities[i]);
        __m256d comparison = CMP_GT(comp_value, current_probs);
        int bitmask = MOVEMASK(comparison);

        if(bitmask != 0) {
            first_valid = i + __builtin_ctzll(bitmask); // Получаем первый ненулевой бит
            break;
        }
    }

    return first_valid != -1 ? first_valid : length - 1;
}

Как работает выборка?

Этот пример перебирает массив вероятностей, сравнивая каждый блок из 8 элементов с пороговым значением (случайным числом). Результат сравнения сохраняется в виде битовой маски, где единица означает, что соответствующее значение меньше порога. Затем определяется первая позиция, где произошло превышение порога.

Полностью интегрированный пример:

#include <iostream>
#include <random>
#include <cstdio>
#include <immintrin.h>

// Определение макросов
#define LOAD_M256D(addr) _mm256_loadu_pd((const double*) addr)
#define STORE_M256D(addr, val) _mm256_storeu_pd((double*) addr, val)
#define SET_ONE _mm256_set1_pd(1.0)
#define CMP_GT(a,b) _mm256_cmp_pd(a, b, _CMP_GT_OQ)
#define MOVEMASK(pd) _mm256_movemask_pd(pd)

// Функция быстрой выборки с использованием AVX2
int select_random_simd(double randomValue, const double* probabilities, int length) {
    __m256d comp_value = SET_ONE * randomValue;
    __m256d current_probs;
    int first_valid = -1;

    for(int i = 0; i < length; i += 8) {
        current_probs = LOAD_M256D(&probabilities[i]);
        __m256d comparison = CMP_GT(comp_value, current_probs);
        int bitmask = MOVEMASK(comparison);

        if(bitmask != 0) {
            first_valid = i + __builtin_ctzll(bitmask); // Найти первую позицию
            break;
        }
    }

    return first_valid != -1 ? first_valid : length - 1;
}

int main() {
    // Случайные вероятности
    double probabilities[] = {0.1, 0.2, 0.3, 0.4};
    std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Несколько примеров выборки
    for(int i = 0; i < 10; ++i) {
        double rnd = dist(engine);
        int sel = select_random_simd(rnd, probabilities, sizeof(probabilities)/sizeof(probabilities[0]));
        std::cout << "Selected: " << sel << "\n";
    }

    return 0;
}

3. Parallel Processing (OpenMP)

Параллельное исполнение обеспечивает возможность разделить большую задачу на части, каждая из которых исполняется отдельным потоком. Библиотека OpenMP предоставляет удобный способ запуска параллельных процессов с минимальными изменениями исходного кода.

Рассмотрим простейший пример парализации основного цикла:

#include <omp.h>

void process_ants_parallel(int NUM_THREADS, int PARAMETR_SIZE, int ANT_SIZE) {
    #pragma omp parallel for num_threads(NUM_THREADS)
    for(int bx = 0; bx < ANT_SIZE; bx++) {
        for(int tx = 0; tx < PARAMETR_SIZE; tx++) {
            // Ваш внутренний код тут...
        }
    }
}

Важные моменты:

Атрибут #pragma omp parallel for автоматически распределяет итерации цикла между указанными потоками.
Используйте разумное количество потоков (обычно равное количеству физических ядер процессора).

Подытоживая:

Мы рассмотрели три различных способа повышения эффективности вашей программы:

Alias Method — чрезвычайно эффективный способ преобразования равномерного распределения в произвольное с постоянной скоростью выборки.
SIMD (AVX2) — ускоряет массовые арифметические операции, повышая производительность при обработке блоков данных.
Parallel Processing (OpenMP) — простой и надежный способ достижения высокой производительности за счет многоядерных процессоров.

Каждый из этих подходов направлен на разные аспекты проблемы и позволяет добиться значительных улучшений в зависимости от конкретных требований и ограничений среды разработки.
*/