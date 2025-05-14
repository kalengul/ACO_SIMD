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
#include <mpi.h>
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
    double mean() {
        return count > 0 ? sum / count : 0.0;
    }
    double variance() {
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
unsigned long long generateKey_non_cuda(const double* agent_node, int bx) {
    unsigned long long key = 0;
    unsigned long long factor = 1;
    for (int i = 0; i < PARAMETR_SIZE; i++) {
        int val = static_cast<int>(agent_node[bx * PARAMETR_SIZE + i]);
        //        std::cout << val << " ";
        key += val * factor;
        factor *= MAX_VALUE_SIZE;
    }
    //    std::cout <<" key=" << key;
    //    std::cout << std::endl;
    return key;
}

// ----------------- Hash Table Search with Quadratic Probing -----------------
double getCachedResultOptimized_non_cuda(HashEntry* hashTable, const double* agent_node, int bx) {
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
        idx = (idx + i * i) % HASH_TABLE_SIZE; // Quadratic probing
        i++;
    }
    return -1.0; // Not found after maximum probes
}

// ----------------- Hash Table Insertion with Quadratic Probing -----------------
void saveToCacheOptimized_non_cuda(HashEntry* hashTable, const double* agent_node, int bx, double value) {
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

        idx = (idx + i * i) % HASH_TABLE_SIZE; // Quadratic probing
        i++;
    }
    // If the table is full, handle the error or ignore
}

// Функция для цвычисления параметра х при  параметрическом графе
double go_x_non_cuda(double* parametr, int start_index, int kol_parametr) {
    double sum = 0.0;
    for (int i = 1; i < kol_parametr; ++i) {
        sum += parametr[start_index + i];
    }
    return parametr[start_index] * sum; // Умножаем на первый параметр в диапазоне
}

// Функция для цвычисления параметра х1 при 40 параметрическом графе
double go_x1_21_non_cuda(double* parametr) {
    return parametr[0] * (parametr[1] + parametr[2] + parametr[3] + parametr[4] + parametr[5] + parametr[6] + parametr[7] + parametr[8] + parametr[9] + parametr[10] + parametr[11] + parametr[12] + parametr[13] + parametr[14] + parametr[15] + parametr[16] + parametr[17] + parametr[18] + parametr[19] + parametr[20]);
}

// Функция для цвычисления параметра х2 при 40 параметрическом графе
double go_x2_21_non_cuda(double* parametr) {
    return parametr[21] * (parametr[22] + parametr[23] + parametr[24] + parametr[25] + parametr[26] + parametr[27] + parametr[28] + parametr[29] + parametr[30] + parametr[31] + parametr[32] + parametr[33] + parametr[34] + parametr[35] + parametr[36] + parametr[37] + parametr[38] + parametr[39] + parametr[40] + parametr[41]);
}
// Функция для вычисления параметра x1 при 12 параметрическом графе
double go_x1_6_non_cuda(double* parametr) {
    return parametr[0] * (parametr[1] + parametr[2] + parametr[3] + parametr[4] + parametr[5]);
}

// Функция для вычисления параметра x2 при 12 параметрическом графе
double go_x2_6_non_cuda(double* parametr) {
    return parametr[6] * (parametr[7] + parametr[8] + parametr[9] + parametr[10] + parametr[11]);
}

// Функция для целевой функции Шаффера
double BenchShafferaFunction_non_cuda_2x(double* parametr) {
    double x1 = go_x1_21_non_cuda(parametr);
    double x2 = go_x2_21_non_cuda(parametr);
    double r = sqrt(x1 * x1 + x2 * x2);
    double sin_r = sin(r);
    return 1.0 / 2.0 - (sin_r * sin_r - 0.5) / (1.0 + 0.001 * (x1 * x1 + x2 * x2));
}

// Функция для целевой функции Шаффера с 100 переменными
double BenchShafferaFunction_non_cuda(double* parametr) {
    double sum = 0.0;
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

// Функция для вычисления вероятностной формулы
double probability_formula_non_cuda(double pheromon, double kol_enter) {
    double res = 0;
    if ((kol_enter != 0) && (pheromon != 0)) {
        res = 1.0 / kol_enter + pheromon;
    }
    return res;
}

// Подготовка массива для вероятностного поиска
void go_mass_probability_non_cuda(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    // Нормализация слоя с феромоном
    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double sumVector = 0;
        double pheromon_norm[MAX_VALUE_SIZE];

        // Суммируем значения феромонов
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
        }

        // Нормализуем значения феромонов
        for (int i = 0; i < MAX_VALUE_SIZE; i++) {
            pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
        }

        sumVector = 0;
        double svertka[MAX_VALUE_SIZE];

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

// Функция для вычисления пути агентов на CPU
void go_all_agent_non_cuda_time(int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail, float& totalHashTime, float& totalOFTime, float& HashTimeSave, float& HashTimeSearch, float& SumTimeSearch) {
    // Генератор случайных чисел
    std::default_random_engine generator(123 + gpuTime); // Используем gpuTime как начальное значение
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int bx = 0; bx < ANT_SIZE; bx++) { // Проходим по всем агентам
        auto start_ant = std::chrono::high_resolution_clock::now();
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            // Определение номера значения
            int k = 0;
            while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
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
                        while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
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
void go_all_agent_non_cuda(int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail) {
    // Генератор случайных чисел
    std::default_random_engine generator(123 + gpuTime); // Используем gpuTime как начальное значение
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int bx = 0; bx < ANT_SIZE; bx++) { // Проходим по всем агентам
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            // Определение номера значения
            int k = 0;
            while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
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
                        while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
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

void process_agent(int bx, double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail) {
    std::default_random_engine generator(rand()); // Генератор случайных чисел
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
        double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]

        // Определение номера значения
        int k = 0;
        while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
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
                    while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
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

void go_all_agent_non_cuda_thread(double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail, int num_threads) {
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
/*
// Структура для передачи параметров в поток
#include <windows.h>
struct TaskParams {
    double* parametr;
    double* norm_matrix_probability;
    double* agent;
    double* agent_node;
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
    double* agent_node = params->agent_node;
    double* OF = params->OF;
    HashEntry* hashTable = params->hashTable;
    int bx = params->bx;
    int& kol_hash_fail = params->kol_hash_fail;

    for (int tx = 0; tx < PARAMETR_SIZE; tx++) {
        double randomValue = distribution(generator);
        int k = 0;

        while (k < PARAMETR_SIZE && randomValue > params->norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
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

                    while (k < PARAMETR_SIZE && randomValue > params->norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
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

void go_all_agent_non_cuda(double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail) {
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
*/
void go_all_agent_MPI(int rank_MSI, int size_MSI, int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail, float& totalHashTime, float& totalOFTime) {
    // Генератор случайных чисел
    std::default_random_engine generator(123 + gpuTime); // Используем gpuTime как начальное значение
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int bx = rank_MSI; bx < ANT_SIZE; bx+=size_MSI) { // Проходим по всем агентам
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            // Определение номера значения
            int k = 0;
            while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
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
                        while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
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

void go_all_agent_non_cuda_non_hash(int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF, int& kol_hash_fail, float& totalOFTime) {
    // Генератор случайных чисел
    std::default_random_engine generator(123 + gpuTime); // Используем gpuTime как начальное значение
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int bx = 0; bx < ANT_SIZE; bx++) { // Проходим по всем агентам
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            double randomValue = distribution(generator); // Генерация случайного числа в диапазоне [0, 1]
            // Определение номера значения
            int k = 0;
            while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
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

// Обновление слоев графа
void add_pheromon_iteration_non_cuda(double* pheromon, double* kol_enter, double* agent_node, double* OF) {
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
            kol_enter[MAX_VALUE_SIZE * tx + k]++;
            //            pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * OF[i]; // MAX
            //            pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q / OF[i]; // MIN
            if (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i] > 0) {
                pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i]); // MIN
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

int start_NON_CUDA_time() {
    auto start = std::chrono::high_resolution_clock::now();
    float SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumTimeHashTotal = 0.0f, SumTimeOF = 0.0f, SumTimeHashSearch = 0.0f, SumTimeHashSave = 0.0f, SumTimeSearchAgent = 0.0f;
    float duration = 0.0f, duration_iteration = 0.0f;
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
    double* ant_parametr = new double[kolBytes_matrix_ant];
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
        go_all_agent_non_cuda_time(current_time.count() * 1000, parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail, SumTimeHashTotal, SumTimeOF, SumTimeHashSearch, SumTimeHashSave, SumTimeSearchAgent);



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
        SumgpuTime1 += std::chrono::duration<float, std::milli>(end_iter - start1).count();
        SumgpuTime2 += std::chrono::duration<float, std::milli>(end_iter - start2).count();
        SumgpuTime3 += std::chrono::duration<float, std::milli>(end_iter - start3).count();
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

    // Освобождение выделенной памяти
    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    auto end_all = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double, std::milli>(end_all - start).count();
    std::cout << "Time non CUDA_time;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA_time;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << SumTimeSearchAgent << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}

int start_NON_CUDA_thread() {
    auto start = std::chrono::high_resolution_clock::now();
    float SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumTimeHashTotal = 0.0f, SumTimeOF = 0.0f, SumTimeHashSearch = 0.0f, SumTimeHashSave = 0.0f, SumTimeSearchAgent = 0.0f;
    float duration = 0.0f, duration_iteration = 0.0f;
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
    double* ant_parametr = new double[kolBytes_matrix_ant];
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

    // Освобождение выделенной памяти
    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    auto end_all = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double, std::milli>(end_all - start).count();
    std::cout << "Time non CUDA thread;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA thread;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}

int start_NON_CUDA() {
    auto start = std::chrono::high_resolution_clock::now();
    float SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumTimeHashTotal = 0.0f, SumTimeOF = 0.0f, SumTimeHashSearch = 0.0f, SumTimeHashSave = 0.0f, SumTimeSearchAgent = 0.0f;
    float duration = 0.0f, duration_iteration = 0.0f;
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
    double* ant_parametr = new double[kolBytes_matrix_ant];
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
        auto end_temp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> current_time = end_temp - start;
        go_all_agent_non_cuda(current_time.count() * 1000, parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail);

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

    // Освобождение выделенной памяти
    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    auto end_all = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double, std::milli>(end_all - start).count();
    std::cout << "Time non CUDA;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumTimeHashTotal << "; " << SumTimeOF << "; " << SumTimeHashSearch << "; " << SumTimeHashSave << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}

int start_MPI(int rank, int size) {
    auto start = std::chrono::high_resolution_clock::now();

    float SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumgpuTime4 = 0.0f, SumgpuTime5 = 0.0f, SumgpuTime6 = 0.0f, SumgpuTime7 = 0.0f;
    float duration = 0.0f, duration_iteration = 0.0f;
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
    double* ant_parametr = new double[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    if (rank == 0) {
        // Загрузка матрицы из файла
        load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);
    }
    // Распространение параметров среди всех процессов
    MPI_Bcast(parametr_value, kolBytes_matrix_graph, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pheromon_value, kolBytes_matrix_graph, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(kol_enter_value, kolBytes_matrix_graph, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double start_iteration = MPI_Wtime();
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        double start1 = MPI_Wtime();
        if (rank == 0) {
            // Расчет нормализованной вероятности
            go_mass_probability_non_cuda(pheromon_value, kol_enter_value, norm_matrix_probability);
            MPI_Bcast(norm_matrix_probability, kolBytes_matrix_graph, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (PRINT_INFORMATION) {
                std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
                for (int i = 0; i < PARAMETR_SIZE; ++i) {
                    for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                        std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                    }
                    std::cout << std::endl; // Переход на новую строку
                }
            }
        }
        // Вычисление пути агентов
        double start2 = MPI_Wtime();
        double end_temp = MPI_Wtime();
        double current_time = end_temp - start_iteration;
        go_all_agent_MPI(rank, size,current_time * 1000, parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail, SumgpuTime4, SumgpuTime5);

        double maxOf = -std::numeric_limits<double>::max();
        double minOf = std::numeric_limits<double>::max();
        double start3 = MPI_Wtime();
        if (rank == 0) {
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
            MPI_Bcast(pheromon_value, kolBytes_matrix_graph, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(kol_enter_value, kolBytes_matrix_graph, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            // Поиск максимума и минимума

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
            double end_iter = MPI_Wtime();
            SumgpuTime1 += (end_iter - start1) * 1000;
            SumgpuTime2 += (end_iter - start2) * 1000;
            SumgpuTime3 += (end_iter - start3) * 1000;
            if (PRINT_INFORMATION) {
                std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << std::endl;
            }
            if ((nom_iter + 1) % kol_shag_stat == 0) {
                int NomStatistics = nom_iter / kol_shag_stat;
                if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
                update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, SumgpuTime6, SumgpuTime7, 0, global_minOf, global_maxOf, kol_hash_fail);
            }
        }
    }
    double end = MPI_Wtime();
    duration_iteration += (end - start_iteration) * 1000;

    // Освобождение выделенной памяти
    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    MPI_Finalize();
    auto end_all = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double, std::milli>(end_all - start).count();
    std::cout << "Time MPI;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumgpuTime4 << "; " << SumgpuTime5 << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time MPI;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumgpuTime4 << "; " << SumgpuTime5 << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}

int start_NON_CUDA_non_hash() {
    auto start = std::chrono::high_resolution_clock::now();
    float SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumgpuTime5 = 0.0f, SumgpuTime6 = 0.0f, SumgpuTime7 = 0.0f;
    float duration = 0.0f, duration_iteration = 0.0f;
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
    double* ant_parametr = new double[kolBytes_matrix_ant];
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
        go_all_agent_non_cuda_non_hash(current_time.count() * 100000, parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, kol_hash_fail, SumgpuTime5);

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
        SumgpuTime1 += std::chrono::duration<float, std::milli>(end_iter - start1).count();
        SumgpuTime2 += std::chrono::duration<float, std::milli>(end_iter - start2).count();
        SumgpuTime3 += std::chrono::duration<float, std::milli>(end_iter - start3).count();
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

    // Освобождение выделенной памяти
    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    auto end_all = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double, std::milli>(end_all - start).count();
    std::cout << "Time non CUDA non hash;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << "0" << "; " << SumgpuTime5 << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA non hash:;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << "0" << "; " << SumgpuTime5 << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}
/*
int start_MPI() {
    int rank, size;
    MPI_Init(0,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto start = MPI_Wtime();
    float SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f, SumgpuTime4 = 0.0f, SumgpuTime5 = 0.0f, SumgpuTime6 = 0.0f, SumgpuTime6 = 0.0f, SumgpuTime7 = 0.0f;
    float duration = 0.0f, duration_iteration = 0.0f;
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
    double* ant_parametr = new double[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];

    // Загрузка матрицы из файла
    load_matrix_non_cuda(NAME_FILE_GRAPH, parametr_value, pheromon_value, kol_enter_value);

    auto start_iteration = MPI_Wtime();
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        auto start1 = MPI_Wtime();
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

        auto start2 = MPI_Wtime();
        auto end_temp = MPI_Wtime();
        double current_time = end_temp - start;
        go_all_agent_non_cuda(current_time * 1000, parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail, SumgpuTime4, SumgpuTime5, SumgpuTime6);



        if (PRINT_INFORMATION) {
            std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";

                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        auto start3 = MPI_Wtime();
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
        auto end_iter = MPI_Wtime();
        SumgpuTime1 += (end_iter - start1) * 1000;
        SumgpuTime2 += (end_iter - start2) * 1000;
        SumgpuTime3 += (end_iter - start3) * 1000;
        if (PRINT_INFORMATION) {
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << std::endl;
        }
        if ((nom_iter + 1) % kol_shag_stat == 0) {
            int NomStatistics = nom_iter / kol_shag_stat;
            if (PRINT_INFORMATION) { std::cout << "nom_iter=" << nom_iter << " " << kol_shag_stat << " NomStatistics=" << NomStatistics << " "; }
            update_all_Stat(NomStatistics, 0, 0, SumgpuTime1, SumgpuTime2, SumgpuTime3, SumgpuTime4, SumgpuTime5, global_minOf, global_maxOf, kol_hash_fail);
        }
    }
    auto end = MPI_Wtime();
    duration_iteration += (end - start_iteration) * 1000;

    // Освобождение выделенной памяти
    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    auto end_all = MPI_Wtime();
    duration += (end_all - start) * 1000;
    std::cout << "Time non CUDA;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << SumgpuTime4 << "; " << SumgpuTime5 << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA;" << duration << "; " << duration_iteration << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << SumgpuTime4 << "; " << SumgpuTime5 << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}
*/

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
                Node node;
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
                    currentNode.KolSolutionNorm = static_cast<double>(currentNode.KolSolution) / MaxK;
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
        idx = (idx + i * i) % HASH_TABLE_SIZE; // Квадратичное пробирование
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

        idx = (idx + i * i) % HASH_TABLE_SIZE; // Квадратичное пробирование
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
    float SumTime5 = 0.0f;
    float SumTime6 = 0.0f;
    float SumTime7 = 0.0f;
    float duration_iteration = 0.0f;
    float duration = 0.0f;
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
    float SumTime5 = 0.0f;
    float SumTime6 = 0.0f;
    float SumTime7 = 0.0f;
    float duration_iteration = 0.0f;
    float duration = 0.0f;
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

int main(int argc, char** argv) {

    int rank, size;
    if (GO_MPI) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
    if (!GO_MPI || (GO_MPI && rank == 0)) {
        // Открытие лог-файла
        std::cout << __cplusplus << std::endl;
        logFile.open("log.txt");
        if (!logFile.is_open()) {
            std::cerr << "Ошибка открытия лог-файла!" << std::endl;
            return 1; // Возврат с ошибкой
        }
        //Создание векторов для статистики 


        std::cout << "PARAMETR_SIZE: " << PARAMETR_SIZE << "; "
            << "MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << "; "
            << "ANT_SIZE: " << ANT_SIZE << "; "
            << "NAME_FILE_GRAPH: " << NAME_FILE_GRAPH << "; "
            << "KOL_ITERATION: " << KOL_ITERATION << "; "
            << "KOL_PROGON_STATISTICS: " << KOL_PROGON_STATISTICS << "; "
            << "PARAMETR_Q: " << PARAMETR_Q << "; "
            << "PARAMETR_RO: " << PARAMETR_RO << "; "
            << "HASH_TABLE_SIZE: " << HASH_TABLE_SIZE << "; "
            << "MAX_PROBES: " << MAX_PROBES << "; "
            << "TYPE_ACO: " << TYPE_ACO << "; "
            << "ACOCCyN_KOL_ITERATION: " << ACOCCyN_KOL_ITERATION << "; "
            << "PRINT_INFORMATION: " << (PRINT_INFORMATION ? "true" : "false")
            << std::endl;
        std::cout << "START MPI ";
        logFile << "PARAMETR_SIZE: " << PARAMETR_SIZE << "; "
            << "MAX_VALUE_SIZE: " << MAX_VALUE_SIZE << "; "
            << "NAME_FILE_GRAPH: " << NAME_FILE_GRAPH << "; "
            << "ANT_SIZE: " << ANT_SIZE << "; "
            << "KOL_ITERATION: " << KOL_ITERATION << "; "
            << "KOL_PROGON_STATISTICS: " << KOL_PROGON_STATISTICS << "; "
            << "PARAMETR_Q: " << PARAMETR_Q << "; "
            << "PARAMETR_RO: " << PARAMETR_RO << "; "
            << "HASH_TABLE_SIZE: " << HASH_TABLE_SIZE << "; "
            << "MAX_PROBES: " << MAX_PROBES << "; "
            << "TYPE_ACO: " << TYPE_ACO << "; "
            << "ACOCCyN_KOL_ITERATION: " << ACOCCyN_KOL_ITERATION << "; "
            << "PRINT_INFORMATION: " << (PRINT_INFORMATION ? "true" : "false")
            << std::endl;
        std::cout << "START MPI ";
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
            int numAnts = ANT_SIZE;
            int numIterations = KOL_ITERATION;
            double evaporationRate = 0.999; // Параметр испарения
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
            int numAnts = ANT_SIZE;
            int numIterations = KOL_ITERATION;
            double evaporationRate = 0.999; // Параметр испарения
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
        std::cout << "START MPI 0";
    }
    std::cout << "START MPI ";
    if (GO_MPI) {
        int j = 0;
        while (j < KOL_PROGREV)
        {
            std::cout << "PROGREV " << j << " ";
            start_MPI(rank, size);
            j = j + 1;
        }
        // Запуск таймера
        clear_all_stat();
        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            std::cout << i << " ";
            start_MPI(rank, size);
            i = i + 1;
        }
        // Остановка таймера
        auto end2 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double, std::milli> duration = end2 - start2;
        std::string message = "Time MPI:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
        save_all_stat_text_file("MPI");
    }
    

    // Закрытие лог-файла
    logFile.close();
    outfile.close();
}
