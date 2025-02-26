#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <limits.h>
#include <vector>
#include <random>
#include <ctime>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "matrix_loader.h"
//#include "non_cuda.h"
#include "parametrs.h" 

// Глобальная переменная для лог-файла
std::ofstream logFile;
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
        hashTable[i].key = 0;
        hashTable[i].value = 0.0;
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
__device__ unsigned long long generateKey(const double* agent_node, int bx) {
    unsigned long long key = 0;
    unsigned long long factor = 1;
    for (int i = 0; i < PARAMETR_SIZE; i++) {
        int val = static_cast<int>(agent_node[bx * PARAMETR_SIZE + i]);
        key += val * factor;
        factor *= MAX_VALUE_SIZE;
    }
    return key;
}

// ----------------- Hash Table Search with Quadratic Probing -----------------
__device__ double getCachedResultOptimized(HashEntry* hashTable, const double* agent_node, int bx) {
    unsigned long long key = generateKey(agent_node, bx);
    unsigned long long idx = betterHashFunction(key);
    int i = 1;
    while (i <= MAX_PROBES) {
        if (hashTable[idx].key == key) {
            return hashTable[idx].value; // Found
        }
        if (hashTable[idx].key == 0) {
            return -1.0; // Not found and slot is empty
        }
        idx = (idx + i * i) % HASH_TABLE_SIZE; // Quadratic probing
        i++;
    }
    return -1.0; // Not found after maximum probes
}

// ----------------- Hash Table Insertion with Quadratic Probing -----------------
__device__ void saveToCacheOptimized(HashEntry* hashTable, const double* agent_node, int bx, double value) {
    unsigned long long key = generateKey(agent_node, bx);
    unsigned long long idx = betterHashFunction(key);
    int i = 1;

    while (i <= MAX_PROBES) {
        unsigned long long expected = 0;
        unsigned long long desired = key;
        unsigned long long old = atomicCAS(&(hashTable[idx].key), expected, desired);
        if (old == expected || old == key) {
            // Successfully inserted or key already exists
            hashTable[idx].value = value;
            return;
        }
        idx = (idx + i * i) % HASH_TABLE_SIZE; // Quadratic probing
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

// Функция для цвычисления параметра х при  параметрическом графе
__device__ double go_x(double* parametr, int start_index, int kol_parametr) {
    double sum = 0.0;
    for (int i = 1; i < kol_parametr; ++i) {
        sum += parametr[start_index+i];
    }
    return parametr[start_index] * sum; // Умножаем на первый параметр в диапазоне
}

// Функция для целевой функции Шаффера с 100 переменными
__device__ double BenchShafferaFunction(double* parametr) {
    double sum = 0.0;
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

__device__ double Bench4Function(double* parametr) {
    double p0 = go_x(parametr,0, PARAMETR_SIZE_ONE_X);
    double p1 = go_x(parametr, PARAMETR_SIZE_ONE_X, PARAMETR_SIZE_ONE_X);
    double a1 = p0 * p0;
    double a2 = p1 * p1;
    double a = 1.0 - sqrt(a1 + a2) / 3.1415926;
    double OF = cos(p0) * cos(p1) * exp(fabs(a)); // Используем fabs для абсолютного значения
    return OF * OF; // Возвращаем OF в квадрате
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
    double pheromon_norm[MAX_VALUE_SIZE];
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
    }
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
    }
    sumVector = 0;
    double svertka[MAX_VALUE_SIZE];

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

__global__ void go_mass_probability_only(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (параметра) 
    if (tx < PARAMETR_SIZE) {
        go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);
    }
}


__device__ void go_ant_path(int tx, int bx, curandState state, double* parametr, double* norm_matrix_probability, double* agent, double* agent_node) {
    double randomValue = curand_uniform(&state) * 1; // Генерация случайного числа в диапазоне [0, 1]
    //Определение номера значения
    int k = 0;
    while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
        k++;
    }
    // Запись подматрицы блока в глобальную память
    // каждый поток записывает один элемент
    agent_node[bx * PARAMETR_SIZE + tx] = k;
    agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
}
//Вычисление пути агентов
// parametr - матрица с значениями параметров для вычисления х1 и х2
// norm_matrix_probability - нормализованная матрица вероятностей выбора вершины
// HashEntry - хэш-таблица
// agent - значения параметров для агента (СКОРЕЕ ВСЕГО НЕ НУЖНО ВООБЩЕ)
// agent_node - номер значения для каждого параметра для пути агента
// OF - значения целевой функции для агента
__global__ void go_all_agent_only(int* gpuTime, double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (агента) 
    if (bx < ANT_SIZE) {
        //int tx = threadIdx.x;  
        int seed = 123 + bx * ANT_SIZE + gpuTime[0];

        // Генерация случайного числа с использованием curand
        curandState state;
        curand_init(seed, 0, 0, &state); // Инициализация состояния генератора случайных чисел

        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            double randomValue = curand_uniform(&state) * 1; // Генерация случайного числа в диапазоне [0, 1]
            //Определение номера значения
            int k = 0;
            while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                k++;
            }
            // Запись подматрицы блока в глобальную память
            // каждый поток записывает один элемент
            agent_node[bx * PARAMETR_SIZE + tx] = k;
            agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
        }
        // Проверка наличия решения в Хэш-таблице
        double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
        int nom_iteration = 0;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
            saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            //Если значение в Хэш-найдено, то агент "нулевой"
            atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                break;
            case 1: // ACOCNI
                OF[bx] = ZERO_HASH_RESULT;
                break;
            case 2: // ACOCCyN
                while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                {

                    for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                        double randomValue = curand_uniform(&state) * 1; // Генерация случайного числа в диапазоне [0, 1]
                        //Определение номера значения
                        int k = 0;
                        while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                            k++;
                        }
                        // Запись подматрицы блока в глобальную память
                        // каждый поток записывает один элемент
                        agent_node[bx * PARAMETR_SIZE + tx] = k;
                        agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
                    }
                    // Проверка наличия решения в Хэш-таблице
                    double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                    nom_iteration = nom_iteration + 1;
                }
                OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                break;
            }
        }
        // Обновление максимального и минимального значений с использованием атомарных операций
    atomicMax(maxOf_dev, OF[bx]);
    atomicMin(minOf_dev, OF[bx]);
    }
}

__global__ void go_all_agent(int* gpuTime, double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev , int* kol_hash_fail) {
    int bx = blockIdx.x;  // индекс  (столбца)
    int tx = threadIdx.x; // индекс  (агента) 
    int seed = 123 +bx * ANT_SIZE + tx * PARAMETR_SIZE + gpuTime[0];

    // Генерация случайного числа с использованием curand
    curandState state;
    curand_init(seed, 0, 0, &state); // Инициализация состояния генератора случайных чисел
    double randomValue = curand_uniform(&state) * 1; // Генерация случайного числа в диапазоне [0, 1]

    //Определение номера значения
    int k = 0;
    while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
       k++;
    }

    // Запись подматрицы блока в глобальную память
    // каждый поток записывает один элемент
    agent_node[bx * PARAMETR_SIZE + tx] = k;
    agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
    
    __syncthreads();
    /*
    // Проверка наличия решения в Хэш-таблице
    unsigned long long key = 0;
    unsigned long long factor = 1;
    for (int i = 0; i < PARAMETR_SIZE; i++) {
        int val = static_cast<int>(agent_node[bx * PARAMETR_SIZE + i]);
        key += val * factor;
        factor *= MAX_VALUE_SIZE;
    }
    //unsigned long long key = generateKey(agent_node, bx);
    unsigned long long m = 0xc6a4a7935bd1e995;
    int r = 47;
    unsigned long long h = seed ^ (8 * m);

    unsigned long long ks = key;
    ks *= m;
    ks ^= k >> r;
    ks *= m;

    h ^= ks;
    h *= m;

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    unsigned long long idx = h % HASH_TABLE_SIZE;
    //unsigned long long idx = murmurHash64A(key) % HASH_TABLE_SIZE;
    //unsigned long long idx = betterHashFunction(key);
    int i = 1;
    double cachedResult = -1.0; // Not found after maximum probes
    while (i <= MAX_PROBES) {
        if (hashTable[idx].key == key) {
            cachedResult = hashTable[idx].value; // Found
        }
        if (hashTable[idx].key == 0) {
            cachedResult = -1.0; // Not found and slot is empty
        }
        idx = (idx + i * i) % HASH_TABLE_SIZE; // Quadratic probing
        i++;
    }
    */
    double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
    int nom_iteration = 0;
    if (cachedResult == -1.0) {
        // Если значение не найденов ХЭШ, то заносим новое значение
        OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
        saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
    }
    else {
        //Если значение в Хэш-найдено, то агент "нулевой"
        atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
        //Поиск алгоритма для нулевого агента
        switch (TYPE_ACO) {
        case 0: // ACOCN
            OF[bx] = cachedResult;
            break;
        case 1: // ACOCNI
            OF[bx] = ZERO_HASH_RESULT;
            break;
        case 2: // ACOCCyN
            while ((cachedResult != -1.0) && (nom_iteration< ACOCCyN_KOL_ITERATION))
            {
                double randomValue = curand_uniform(&state) * 1; // Генерация случайного числа в диапазоне [0, 1]

                //Определение номера значения
                int k = 0;
                while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                    k++;
                }
                // Запись подматрицы блока в глобальную память
                // каждый поток записывает один элемент
                agent_node[bx * PARAMETR_SIZE + tx] = k;
                agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];

                // Проверка наличия решения в Хэш-таблице
                double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                nom_iteration= nom_iteration+1;
            }
            OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
            break;
        default:
            OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
            break;
        }
        

    }
    // Обновление максимального и минимального значений с использованием атомарных операций
    atomicMax(maxOf_dev, OF[bx]);
    atomicMin(minOf_dev, OF[bx]);
}

__global__ void go_all_agent_non_hash(int* gpuTime, double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = blockIdx.x;  // индекс  (столбца)
    int tx = threadIdx.x; // индекс  (агента) 
    int seed = 123 + bx * ANT_SIZE + tx * PARAMETR_SIZE + gpuTime[0];

    // Генерация случайного числа с использованием curand
    curandState state;
    curand_init(seed, 0, 0, &state); // Инициализация состояния генератора случайных чисел
    double randomValue = curand_uniform(&state) * 1; // Генерация случайного числа в диапазоне [0, 1]

    //Определение номера значения
    int k = 0;
    while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
        k++;
    }

    // Запись подматрицы блока в глобальную память
    // каждый поток записывает один элемент
    agent_node[bx * PARAMETR_SIZE + tx] = k;
    agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];

    __syncthreads();
    OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
    // Обновление максимального и минимального значений с использованием атомарных операций
    atomicMax(maxOf_dev, OF[bx]);
    atomicMin(minOf_dev, OF[bx]);
}

//Обновление слоев графа
// pheromon - слой с весами (феромоном)
// kol_enter - слой с количеством посещений вершины
// agent_node - пути агентов
// OF - значение целевой функции для каждого агента
__device__ void add_pheromon_iteration(int tx, double* pheromon, double* kol_enter, double* agent_node, double* OF) {
    //Испарение весов-феромона
    for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
        pheromon[MAX_VALUE_SIZE * tx + i] = pheromon[MAX_VALUE_SIZE * tx + i] * PARAMETR_RO;
    }
    //Добавление весов-феромона
    for (int i = 0; i < ANT_SIZE; ++i) {
        int k = int(agent_node[i * PARAMETR_SIZE + tx]);
        kol_enter[MAX_VALUE_SIZE * tx + k]++;
        //        pheromon[MAX_VALUE_SIZE * tx + k] = pheromon[MAX_VALUE_SIZE * tx + k] + PARAMETR_Q * OF[i]; //MAX
                //if (OF[i] == 0) { OF[i] = 0.0000001; }
                //pheromon[MAX_VALUE_SIZE * tx + k] = pheromon[MAX_VALUE_SIZE * tx + k] + PARAMETR_Q / OF[i]; //MIN
        pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q * (MAX_PARAMETR_VALUE_TO_MIN_OPT - OF[i]); // MIN
    }
    //        for (int i = 0; i < PARAMETR_SIZE; ++i) {
    //           kol_enter[MAX_VALUE_SIZE * i + int(agent_node[tx * PARAMETR_SIZE + i])]++;
}

__global__ void add_pheromon_iteration_thread(double* pheromon, double* kol_enter, double* agent_node, double* OF){
//    int bx = blockIdx.x; // индекс блока (не требуется)
    int tx = threadIdx.x; // индекс потока (параметра)
    add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
}

__global__ void add_pheromon_iteration_only(double* pheromon, double* kol_enter, double* agent_node, double* OF) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (параметра) 
    if (tx < PARAMETR_SIZE) {
        add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
    }
}

__global__ void go_mass_probability_and_add_pheromon_iteration(double* pheromon, double* kol_enter, double* norm_matrix_probability, double* agent_node, double* OF) {
    //    int bx = blockIdx.x; // индекс блока (не требуется)
    int tx = threadIdx.x; // индекс потока (столбца)
    //Испарение весов-феромона
    add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
    //Нормализация слоя с феромоном
    go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);
}

__global__ void go_mass_probability_and_add_pheromon_iteration_only(double* pheromon, double* kol_enter, double* norm_matrix_probability, double* agent_node, double* OF) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (параметра) 
    if (tx < PARAMETR_SIZE) {
        //Испарение весов-феромона
        add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
        //Нормализация слоя с феромоном
        go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);
    }
}

__global__ void go_all_agent_opt(double* pheromon, double* kol_enter, int* gpuTime, double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = blockIdx.x; // индекс блока (агента)
    int tx = threadIdx.x; // индекс потока (столбца)
    int seed = 1230 +bx * ANT_SIZE + tx * PARAMETR_SIZE;

    // Генерация случайного числа с использованием curand
    curandState state;
    curand_init(seed, 0, 0, &state); // Инициализация состояния генератора случайных чисел
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        //Нормализация слоя с феромоном
        go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);
    
        double randomValue = curand_uniform(&state) * 1; // Генерация случайного числа в диапазоне [0, 1]
        //Определение номера значения
        int k = 0;
        while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
            k++;
        }
        // Запись подматрицы блока в глобальную память
        // каждый поток записывает один элемент
        agent_node[bx * PARAMETR_SIZE + tx] = k;
        agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
        __syncthreads();
        // Проверка наличия решения в Хэш-таблице
        double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
        int nom_iteration = 0;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
            saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
            //Если значение в Хэш-найдено, то агент "нулевой"
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                break;
            case 1: // ACOCNI
                OF[bx] = ZERO_HASH_RESULT;
                break;
            case 2: // ACOCCyN
                while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                {
                    double randomValue = curand_uniform(&state) * 1; // Генерация случайного числа в диапазоне [0, 1]

                    //Определение номера значения
                    int k = 0;
                    while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                        k++;
                    }
                    // Запись подматрицы блока в глобальную память
                    // каждый поток записывает один элемент
                    agent_node[bx * PARAMETR_SIZE + tx] = k;
                    agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];

                    // Проверка наличия решения в Хэш-таблице
                    double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                    nom_iteration = nom_iteration + 1;
                }
                OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                break;
            }


        }
        // Обновление максимального и минимального значений с использованием атомарных операций
        atomicMax(maxOf_dev, OF[bx]);
        atomicMin(minOf_dev, OF[bx]);
        //Испарение весов-феромона
        add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
    }
}

__global__ void go_all_agent_opt_non_hash(double* pheromon, double* kol_enter, int* gpuTime, double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = blockIdx.x; // индекс блока (агента)
    int tx = threadIdx.x; // индекс потока (столбца)
    int seed = 1230 + bx * ANT_SIZE + tx * PARAMETR_SIZE;

    // Генерация случайного числа с использованием curand
    curandState state;
    curand_init(seed, 0, 0, &state); // Инициализация состояния генератора случайных чисел
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        //Нормализация слоя с феромоном
        go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);

        double randomValue = curand_uniform(&state) * 1; // Генерация случайного числа в диапазоне [0, 1]
        //Определение номера значения
        int k = 0;
        while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
            k++;
        }
        // Запись подматрицы блока в глобальную память
        // каждый поток записывает один элемент
        agent_node[bx * PARAMETR_SIZE + tx] = k;
        agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
        OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
        // Обновление максимального и минимального значений с использованием атомарных операций
        atomicMax(maxOf_dev, OF[bx]);
        atomicMin(minOf_dev, OF[bx]);
        //Испарение весов-феромона
        add_pheromon_iteration(tx, pheromon, kol_enter, agent_node, OF);
    }
}

__global__ void go_all_agent_opt_only(double* pheromon, double* kol_enter, int* gpuTime, double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF, HashEntry* hashTable, double* maxOf_dev, double* minOf_dev, int* kol_hash_fail) {
    int bx = threadIdx.x + blockIdx.x * blockDim.x;  // индекс  (агента) 
    if (bx<ANT_SIZE){
    //int tx = threadIdx.x; // индекс потока (столбца)
    int seed = 1230 + bx * ANT_SIZE + threadIdx.x * PARAMETR_SIZE;

    // Генерация случайного числа с использованием curand
    curandState state;
    curand_init(seed, 0, 0, &state); // Инициализация состояния генератора случайных чисел
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            //Нормализация слоя с феромоном
            go_mass_probability(tx, pheromon, kol_enter, norm_matrix_probability);
        }
        for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
            double randomValue = curand_uniform(&state) * 1; // Генерация случайного числа в диапазоне [0, 1]
            //Определение номера значения
            int k = 0;
            while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                k++;
            }
            // Запись подматрицы блока в глобальную память
            // каждый поток записывает один элемент
            agent_node[bx * PARAMETR_SIZE + tx] = k;
            agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
        }
        // Проверка наличия решения в Хэш-таблице
        double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
        int nom_iteration = 0;
        if (cachedResult == -1.0) {
            // Если значение не найденов ХЭШ, то заносим новое значение
            OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
            saveToCacheOptimized(hashTable, agent_node, bx, OF[bx]);
        }
        else {
            atomicAdd(&kol_hash_fail[0], 1); // Атомарное инкрементирование
            //Если значение в Хэш-найдено, то агент "нулевой"
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                break;
            case 1: // ACOCNI
                OF[bx] = ZERO_HASH_RESULT;
                break;
            case 2: // ACOCCyN
                while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                {
                    for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                        double randomValue = curand_uniform(&state) * 1; // Генерация случайного числа в диапазоне [0, 1]
                        //Определение номера значения
                        int k = 0;
                        while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                            k++;
                        }
                        // Запись подматрицы блока в глобальную память
                        // каждый поток записывает один элемент
                        agent_node[bx * PARAMETR_SIZE + tx] = k;
                        agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
                    }
                    // Проверка наличия решения в Хэш-таблице
                    double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                    nom_iteration = nom_iteration + 1;
                }
                OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
                break;
            case 3: // ACOCCyI

                while (cachedResult != -1.0)
                {
                    nom_iteration = nom_iteration + 1;
                    for (int tx = 0; tx < PARAMETR_SIZE; tx++) { // Проходим по всем параметрам
                        double randomValue = curand_uniform(&state) * 1; // Генерация случайного числа в диапазоне [0, 1]
                        //Определение номера значения
                        int k = 0;
                        while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
                            k++;
                        }
                        // Запись подматрицы блока в глобальную память
                        // каждый поток записывает один элемент
                        agent_node[bx * PARAMETR_SIZE + tx] = k;
                        agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
                    }
                    // Проверка наличия решения в Хэш-таблице
                    double cachedResult = getCachedResultOptimized(hashTable, agent_node, bx);
                }
                OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                break;
            }


        }
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

// Функция для загрузки матрицы из файла
bool load_matrix(const std::string & filename, double* parametr_value, double* pheromon_value, double* kol_enter_value) 
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

static int start_CUDA() {
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    float gpuTime = 0.0f;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float gpuTime3 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int i_gpuTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&startAll);
    cudaEventCreate(&startAll1);
    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop);
    cudaEventRecord(startAll, 0);
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
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
    double* ant_parametr = new double[kolBytes_matrix_ant];
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
    double* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int* gpuTime_dev = nullptr;


    cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph);

    cudaMalloc((void**)&antdev, numBytes_matrix_ant);
    cudaMalloc((void**)&antOFdev, numBytes_ant);
    cudaMalloc((void**)&maxOf_dev, sizeof(double));
    cudaMalloc((void**)&minOf_dev, sizeof(double));
    cudaMalloc((void**)&kol_hash_fail, sizeof(int));
    cudaMalloc((void**)&ant_parametr_dev, numBytes_matrix_ant);
    //cudaMalloc((void**)&cache_dev, TABLE_SIZE * sizeof(HashEntry));
    cudaMalloc((void**)&gpuTime_dev, sizeof(int));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry));
    int threads_init_hash = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threads_init_hash - 1) / threads_init_hash;
    initializeHashTable << <blocks_init_hash, threads_init_hash >> > (hashTable_dev, HASH_TABLE_SIZE);

    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    dim3 kol_ant(ANT_SIZE);




    cudaEventRecord(start, 0);
    cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaEventRecord(startAll1, 0);
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        cudaMemcpy(gpuTime_dev, &i_gpuTime, sizeof(int), cudaMemcpyHostToDevice);
        cudaEventRecord(start1, 0);
        go_mass_probability_thread << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);

        if (PRINT_INFORMATION) {
            cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        cudaEventRecord(start2, 0);
        go_all_agent << <kol_ant, kol_parametr >> > (gpuTime_dev, parametr_value_dev, norm_matrix_probability_dev, antdev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);

        if (PRINT_INFORMATION) {
            cudaMemcpy(ant, antdev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
            cudaMemcpy(ant_parametr, ant_parametr_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
            cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost);
            std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        cudaEventRecord(start3, 0);
        add_pheromon_iteration_thread << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        cudaEventElapsedTime(&gpuTime1, start1, stop);
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        cudaEventElapsedTime(&gpuTime2, start2, stop);
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        cudaEventElapsedTime(&gpuTime3, start3, stop);
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
            cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime1, startAll1, stop);
    cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost);

    // Освобождение ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(startAll1);
    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(start3);

    cudaFree(parametr_value_dev);
    cudaFree(pheromon_value_dev);
    cudaFree(kol_enter_value_dev);
    cudaFree(norm_matrix_probability_dev);
    cudaFree(antdev);
    cudaFree(ant_parametr_dev);
    cudaFree(antOFdev);
    cudaFree(hashTable_dev);
    cudaFree(gpuTime_dev);
    cudaFree(maxOf_dev);
    cudaFree(minOf_dev);
    cudaFree(kol_hash_fail);

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime, startAll, stop);

    cudaEventDestroy(startAll);
    cudaEventDestroy(stop);
    std::cout << "Time CUDA:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    return 0;
}

static int start_CUDA_non_hash() {
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    float gpuTime = 0.0f;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float gpuTime3 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int i_gpuTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&startAll);
    cudaEventCreate(&startAll1);
    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop);
    cudaEventRecord(startAll, 0);
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
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
    double* ant_parametr = new double[kolBytes_matrix_ant];
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
    double* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int* gpuTime_dev = nullptr;


    cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph);

    cudaMalloc((void**)&antdev, numBytes_matrix_ant);
    cudaMalloc((void**)&antOFdev, numBytes_ant);
    cudaMalloc((void**)&maxOf_dev, sizeof(double));
    cudaMalloc((void**)&minOf_dev, sizeof(double));
    cudaMalloc((void**)&kol_hash_fail, sizeof(int));
    cudaMalloc((void**)&ant_parametr_dev, numBytes_matrix_ant);
    //cudaMalloc((void**)&cache_dev, TABLE_SIZE * sizeof(HashEntry));
    cudaMalloc((void**)&gpuTime_dev, sizeof(int));

    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    dim3 kol_ant(ANT_SIZE);




    cudaEventRecord(start, 0);
    cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaEventRecord(startAll1, 0);
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        cudaMemcpy(gpuTime_dev, &i_gpuTime, sizeof(int), cudaMemcpyHostToDevice);
        cudaEventRecord(start1, 0);
        go_mass_probability_thread << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);

        if (PRINT_INFORMATION) {
            cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        cudaEventRecord(start2, 0);
        go_all_agent_non_hash << <kol_ant, kol_parametr >> > (gpuTime_dev, parametr_value_dev, norm_matrix_probability_dev, antdev, ant_parametr_dev, antOFdev,maxOf_dev, minOf_dev, kol_hash_fail);

        if (PRINT_INFORMATION) {
            cudaMemcpy(ant, antdev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
            cudaMemcpy(ant_parametr, ant_parametr_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
            cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost);
            std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        cudaEventRecord(start3, 0);
        add_pheromon_iteration_thread << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        cudaEventElapsedTime(&gpuTime1, start1, stop);
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        cudaEventElapsedTime(&gpuTime2, start2, stop);
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        cudaEventElapsedTime(&gpuTime3, start3, stop);
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
            cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime1, startAll1, stop);
    cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost);

    // Освобождение ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(startAll1);
    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(start3);

    cudaFree(parametr_value_dev);
    cudaFree(pheromon_value_dev);
    cudaFree(kol_enter_value_dev);
    cudaFree(norm_matrix_probability_dev);
    cudaFree(antdev);
    cudaFree(ant_parametr_dev);
    cudaFree(antOFdev);
    cudaFree(gpuTime_dev);
    cudaFree(maxOf_dev);
    cudaFree(minOf_dev);
    cudaFree(kol_hash_fail);

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime, startAll, stop);

    cudaEventDestroy(startAll);
    cudaEventDestroy(stop);
    std::cout << "Time CUDA non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    return 0;
}


static int start_CUDA_ant() {
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    float gpuTime = 0.0f;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float gpuTime3 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int i_gpuTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&startAll);
    cudaEventCreate(&startAll1);
    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop);
    cudaEventRecord(startAll, 0);
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
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
    double* ant_parametr = new double[kolBytes_matrix_ant];
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
    double* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int* gpuTime_dev = nullptr;
    int numBlocks = 0; 
    int numThreads = 0;


    cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph);

    cudaMalloc((void**)&antdev, numBytes_matrix_ant);
    cudaMalloc((void**)&antOFdev, numBytes_ant);
    cudaMalloc((void**)&maxOf_dev, sizeof(double));
    cudaMalloc((void**)&minOf_dev, sizeof(double));
    cudaMalloc((void**)&kol_hash_fail, sizeof(int));
    cudaMalloc((void**)&ant_parametr_dev, numBytes_matrix_ant);
    cudaMalloc((void**)&gpuTime_dev, sizeof(int));
    
    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock> ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 0;
        numThreads = ANT_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);

    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    //dim3 kol_ant(ANT_SIZE);

    

    
    cudaEventRecord(start, 0);
    cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaEventRecord(startAll1, 0);
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        cudaMemcpy(gpuTime_dev, &i_gpuTime, sizeof(int), cudaMemcpyHostToDevice);
        cudaEventRecord(start1, 0);
        go_mass_probability_thread << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);

        if (PRINT_INFORMATION) {
            cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> "<< norm_matrix_probability[i * MAX_VALUE_SIZE + j] <<") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        cudaEventRecord(start2, 0);
        go_all_agent_only << <numBlocks, numThreads >> > (gpuTime_dev, parametr_value_dev, norm_matrix_probability_dev, antdev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
        
        if (PRINT_INFORMATION) {
            cudaMemcpy(ant, antdev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
            cudaMemcpy(ant_parametr, ant_parametr_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
            cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost);
            std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i]  << std::endl;

            }
        }

        cudaEventRecord(start3, 0);
        add_pheromon_iteration_thread << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        cudaEventElapsedTime(&gpuTime1, start1, stop);
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        cudaEventElapsedTime(&gpuTime2, start2, stop);
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        cudaEventElapsedTime(&gpuTime3, start3, stop);
        SumgpuTime3 = SumgpuTime3 + gpuTime3;
        i_gpuTime = int(gpuTime*1000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(gpuTime*1000) << "x" << ANT_SIZE << "):" << std::endl;
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
            cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime1, startAll1, stop);
    cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost);

    // Освобождение ресурсов
    // Освобождение ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(startAll1);
    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(start3);

    cudaFree(parametr_value_dev);
    cudaFree(pheromon_value_dev);
    cudaFree(kol_enter_value_dev);
    cudaFree(norm_matrix_probability_dev);
    cudaFree(antdev);
    cudaFree(ant_parametr_dev);
    cudaFree(antOFdev);
    cudaFree(hashTable_dev);
    cudaFree(gpuTime_dev);
    cudaFree(maxOf_dev);
    cudaFree(minOf_dev);
    cudaFree(kol_hash_fail);

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime, startAll, stop);

    cudaEventDestroy(startAll);
    cudaEventDestroy(stop);
    std::cout << "Time CUDA ant:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA ant:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;

    return 0;
}

static int start_CUDA_ant_par() {
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    float gpuTime = 0.0f;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float gpuTime3 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int i_gpuTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&startAll);
    cudaEventCreate(&startAll1);
    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop);
    cudaEventRecord(startAll, 0);
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
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
    double* ant_parametr = new double[kolBytes_matrix_ant];
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
    double* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int* gpuTime_dev = nullptr;
    int numBlocks = 0;
    int numThreads = 0;
    int numBlocksParametr = 0;
    int numThreadsParametr = 0;


    cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph);

    cudaMalloc((void**)&antdev, numBytes_matrix_ant);
    cudaMalloc((void**)&antOFdev, numBytes_ant);
    cudaMalloc((void**)&maxOf_dev, sizeof(double));
    cudaMalloc((void**)&minOf_dev, sizeof(double));
    cudaMalloc((void**)&kol_hash_fail, sizeof(int));
    cudaMalloc((void**)&ant_parametr_dev, numBytes_matrix_ant);
    cudaMalloc((void**)&gpuTime_dev, sizeof(int));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock > ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 0;
        numThreads = ANT_SIZE;
    }
    if (threadsPerBlock > PARAMETR_SIZE) {
        numBlocksParametr = (PARAMETR_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreadsParametr = MAX_THREAD_CUDA;
    }
    else {
        numBlocksParametr = 0;
        numThreadsParametr = PARAMETR_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);

    // Установка конфигурации запуска ядра
    //dim3 kol_parametr(PARAMETR_SIZE);
    //dim3 kol_ant(ANT_SIZE);




    cudaEventRecord(start, 0);
    cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaEventRecord(startAll1, 0);
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        cudaMemcpy(gpuTime_dev, &i_gpuTime, sizeof(int), cudaMemcpyHostToDevice);
        cudaEventRecord(start1, 0);
        go_mass_probability_only << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);

        if (PRINT_INFORMATION) {
            cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        cudaEventRecord(start2, 0);
        go_all_agent_only << <numBlocks, numThreads >> > (gpuTime_dev, parametr_value_dev, norm_matrix_probability_dev, antdev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);

        if (PRINT_INFORMATION) {
            cudaMemcpy(ant, antdev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
            cudaMemcpy(ant_parametr, ant_parametr_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
            cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost);
            std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        cudaEventRecord(start3, 0);
        add_pheromon_iteration_only << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        cudaEventElapsedTime(&gpuTime1, start1, stop);
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        cudaEventElapsedTime(&gpuTime2, start2, stop);
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        cudaEventElapsedTime(&gpuTime3, start3, stop);
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
            cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime1, startAll1, stop);
    cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost);

    // Освобождение ресурсов
    // Освобождение ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(startAll1);
    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(start3);

    cudaFree(parametr_value_dev);
    cudaFree(pheromon_value_dev);
    cudaFree(kol_enter_value_dev);
    cudaFree(norm_matrix_probability_dev);
    cudaFree(antdev);
    cudaFree(ant_parametr_dev);
    cudaFree(antOFdev);
    cudaFree(hashTable_dev);
    cudaFree(gpuTime_dev);
    cudaFree(maxOf_dev);
    cudaFree(minOf_dev);
    cudaFree(kol_hash_fail);

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime, startAll, stop);

    cudaEventDestroy(startAll);
    cudaEventDestroy(stop);
    std::cout << "Time CUDA ant par:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA ant par:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;

    return 0;
}

static int start_CUDA_opt() {
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    float gpuTime = 0.0f;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float gpuTime3 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int i_gpuTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&startAll);
    cudaEventCreate(&startAll1);
    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop);
    cudaEventRecord(startAll, 0);
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
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
    double* ant_parametr = new double[kolBytes_matrix_ant];
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
    double* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int* gpuTime_dev = nullptr;

    cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph);

    cudaMalloc((void**)&antdev, numBytes_matrix_ant);
    cudaMalloc((void**)&antOFdev, numBytes_ant);
    cudaMalloc((void**)&maxOf_dev, sizeof(double));
    cudaMalloc((void**)&minOf_dev, sizeof(double));
    cudaMalloc((void**)&kol_hash_fail, sizeof(int));
    cudaMalloc((void**)&ant_parametr_dev, numBytes_matrix_ant);
    cudaMalloc((void**)&gpuTime_dev, sizeof(int));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry));
    int threads_init_hash = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threads_init_hash - 1) / threads_init_hash;
    initializeHashTable << <blocks_init_hash, threads_init_hash >> > (hashTable_dev, HASH_TABLE_SIZE);

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

    cudaEventRecord(start, 0);
    cudaMemcpy(ant_parametr_dev, ant_parametr, numBytes_matrix_ant, cudaMemcpyHostToDevice);
    cudaMemcpy(antOFdev, antOF, numBytes_ant, cudaMemcpyHostToDevice);
    cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaEventRecord(startAll1, 0);
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        cudaMemcpy(gpuTime_dev, &i_gpuTime, sizeof(int), cudaMemcpyHostToDevice);
        cudaEventRecord(start1, 0);
        go_mass_probability_and_add_pheromon_iteration << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev, ant_parametr_dev, antOFdev);

        if (PRINT_INFORMATION) {
            cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        cudaEventRecord(start2, 0);
        go_all_agent << <kol_ant, kol_parametr >> > (gpuTime_dev, parametr_value_dev, norm_matrix_probability_dev, antdev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);

        if (PRINT_INFORMATION) {
            cudaMemcpy(ant, antdev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
            cudaMemcpy(ant_parametr, ant_parametr_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
            cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost);
            std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        cudaEventElapsedTime(&gpuTime1, start1, stop);
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        cudaEventElapsedTime(&gpuTime2, start2, stop);
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        cudaEventElapsedTime(&gpuTime3, start3, stop);
        SumgpuTime3 = SumgpuTime3 + gpuTime3;

        i_gpuTime = int(gpuTime*1000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(gpuTime*1000) << "x" << ANT_SIZE << "):" << std::endl;
            cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime1, startAll1, stop);

    cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost);
    // Освобождение ресурсов
    // Освобождение ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(startAll1);
    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(start3);

    cudaFree(parametr_value_dev);
    cudaFree(pheromon_value_dev);
    cudaFree(kol_enter_value_dev);
    cudaFree(norm_matrix_probability_dev);
    cudaFree(antdev);
    cudaFree(ant_parametr_dev);
    cudaFree(antOFdev);
    cudaFree(hashTable_dev);
    cudaFree(gpuTime_dev);
    cudaFree(maxOf_dev);
    cudaFree(minOf_dev);
    cudaFree(kol_hash_fail);

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime, startAll, stop);

    cudaEventDestroy(startAll);
    cudaEventDestroy(stop);
    std::cout << "Time CUDA opt:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA opt:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;

    return 0;
}

static int start_CUDA_opt_non_hash() {
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    float gpuTime = 0.0f;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float gpuTime3 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int i_gpuTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&startAll);
    cudaEventCreate(&startAll1);
    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop);
    cudaEventRecord(startAll, 0);
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
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
    double* ant_parametr = new double[kolBytes_matrix_ant];
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
    double* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int* gpuTime_dev = nullptr;

    cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph);

    cudaMalloc((void**)&antdev, numBytes_matrix_ant);
    cudaMalloc((void**)&antOFdev, numBytes_ant);
    cudaMalloc((void**)&maxOf_dev, sizeof(double));
    cudaMalloc((void**)&minOf_dev, sizeof(double));
    cudaMalloc((void**)&kol_hash_fail, sizeof(int));
    cudaMalloc((void**)&ant_parametr_dev, numBytes_matrix_ant);
    cudaMalloc((void**)&gpuTime_dev, sizeof(int));

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

    cudaEventRecord(start, 0);
    cudaMemcpy(ant_parametr_dev, ant_parametr, numBytes_matrix_ant, cudaMemcpyHostToDevice);
    cudaMemcpy(antOFdev, antOF, numBytes_ant, cudaMemcpyHostToDevice);
    cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaEventRecord(startAll1, 0);
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        cudaMemcpy(gpuTime_dev, &i_gpuTime, sizeof(int), cudaMemcpyHostToDevice);
        cudaEventRecord(start1, 0);
        go_mass_probability_and_add_pheromon_iteration << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev, ant_parametr_dev, antOFdev);

        if (PRINT_INFORMATION) {
            cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        cudaEventRecord(start2, 0);
        go_all_agent_non_hash << <kol_ant, kol_parametr >> > (gpuTime_dev, parametr_value_dev, norm_matrix_probability_dev, antdev, ant_parametr_dev, antOFdev, maxOf_dev, minOf_dev, kol_hash_fail);

        if (PRINT_INFORMATION) {
            cudaMemcpy(ant, antdev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
            cudaMemcpy(ant_parametr, ant_parametr_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
            cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost);
            std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        cudaEventElapsedTime(&gpuTime1, start1, stop);
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        cudaEventElapsedTime(&gpuTime2, start2, stop);
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        cudaEventElapsedTime(&gpuTime3, start3, stop);
        SumgpuTime3 = SumgpuTime3 + gpuTime3;

        i_gpuTime = int(gpuTime * 1000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(gpuTime * 1000) << "x" << ANT_SIZE << "):" << std::endl;
            cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime1, startAll1, stop);

    cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost);
    // Освобождение ресурсов
    // Освобождение ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(startAll1);
    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(start3);

    cudaFree(parametr_value_dev);
    cudaFree(pheromon_value_dev);
    cudaFree(kol_enter_value_dev);
    cudaFree(norm_matrix_probability_dev);
    cudaFree(antdev);
    cudaFree(ant_parametr_dev);
    cudaFree(antOFdev);
    cudaFree(gpuTime_dev);
    cudaFree(maxOf_dev);
    cudaFree(minOf_dev);
    cudaFree(kol_hash_fail);

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime, startAll, stop);

    cudaEventDestroy(startAll);
    cudaEventDestroy(stop);
    std::cout << "Time CUDA opt non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA opt non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;

    return 0;
}

static int start_CUDA_opt_ant() {
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    float gpuTime = 0.0f;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float gpuTime3 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int i_gpuTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&startAll);
    cudaEventCreate(&startAll1);
    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop);
    cudaEventRecord(startAll, 0);
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
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
    double* ant_parametr = new double[kolBytes_matrix_ant];
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
    double* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int* gpuTime_dev = nullptr;
    int numBlocks = 0;
    int numThreads = 0;


    cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph);

    cudaMalloc((void**)&antdev, numBytes_matrix_ant);
    cudaMalloc((void**)&antOFdev, numBytes_ant);
    cudaMalloc((void**)&maxOf_dev, sizeof(double));
    cudaMalloc((void**)&minOf_dev, sizeof(double));
    cudaMalloc((void**)&kol_hash_fail, sizeof(int));
    cudaMalloc((void**)&ant_parametr_dev, numBytes_matrix_ant);
    cudaMalloc((void**)&gpuTime_dev, sizeof(int));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock > ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 0;
        numThreads = ANT_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);

    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    //dim3 kol_ant(ANT_SIZE);

    //Заполнение начальными значениями ant_parametr_dev, antOFdev
    for (int i = 0; i < ANT_SIZE; ++i) {
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            ant_parametr[i * PARAMETR_SIZE + j] = 0;
        }
        antOF[i] = 1;

    }

    cudaEventRecord(start, 0);
    cudaMemcpy(ant_parametr_dev, ant_parametr, numBytes_matrix_ant, cudaMemcpyHostToDevice);
    cudaMemcpy(antOFdev, antOF, numBytes_ant, cudaMemcpyHostToDevice);
    cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaEventRecord(startAll1, 0);
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        cudaMemcpy(gpuTime_dev, &i_gpuTime, sizeof(int), cudaMemcpyHostToDevice);
        cudaEventRecord(start1, 0);
        go_mass_probability_and_add_pheromon_iteration << <1, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev, ant_parametr_dev, antOFdev);

        if (PRINT_INFORMATION) {
            cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        cudaEventRecord(start2, 0);
        go_all_agent_only << <numBlocks, numThreads >> > (gpuTime_dev, parametr_value_dev, norm_matrix_probability_dev, antdev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);

        if (PRINT_INFORMATION) {
            cudaMemcpy(ant, antdev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
            cudaMemcpy(ant_parametr, ant_parametr_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
            cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost);
            std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        cudaEventElapsedTime(&gpuTime1, start1, stop);
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        cudaEventElapsedTime(&gpuTime2, start2, stop);
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        cudaEventElapsedTime(&gpuTime3, start3, stop);
        SumgpuTime3 = SumgpuTime3 + gpuTime3;

        i_gpuTime = int(gpuTime * 1000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(gpuTime * 1000) << "x" << ANT_SIZE << "):" << std::endl;
            cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime1, startAll1, stop);

    cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost);
    // Освобождение ресурсов
    // Освобождение ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(startAll1);
    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(start3);

    cudaFree(parametr_value_dev);
    cudaFree(pheromon_value_dev);
    cudaFree(kol_enter_value_dev);
    cudaFree(norm_matrix_probability_dev);
    cudaFree(antdev);
    cudaFree(ant_parametr_dev);
    cudaFree(antOFdev);
    cudaFree(hashTable_dev);
    cudaFree(gpuTime_dev);
    cudaFree(maxOf_dev);
    cudaFree(minOf_dev);
    cudaFree(kol_hash_fail);

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime, startAll, stop);

    cudaEventDestroy(startAll);
    cudaEventDestroy(stop);
    std::cout << "Time CUDA opt ant:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA opt ant:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;

    return 0;
}

static int start_CUDA_opt_ant_par() {
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    float gpuTime = 0.0f;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float gpuTime3 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int i_gpuTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&startAll);
    cudaEventCreate(&startAll1);
    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop);
    cudaEventRecord(startAll, 0);
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
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
    double* ant_parametr = new double[kolBytes_matrix_ant];
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
    double* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int* gpuTime_dev = nullptr;
    int numBlocks = 0;
    int numThreads = 0;
    int numBlocksParametr = 0;
    int numThreadsParametr = 0;


    cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph);

    cudaMalloc((void**)&antdev, numBytes_matrix_ant);
    cudaMalloc((void**)&antOFdev, numBytes_ant);
    cudaMalloc((void**)&maxOf_dev, sizeof(double));
    cudaMalloc((void**)&minOf_dev, sizeof(double));
    cudaMalloc((void**)&kol_hash_fail, sizeof(int));
    cudaMalloc((void**)&ant_parametr_dev, numBytes_matrix_ant);
    cudaMalloc((void**)&gpuTime_dev, sizeof(int));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock > ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 0;
        numThreads = ANT_SIZE;
    }
    if (threadsPerBlock > PARAMETR_SIZE) {
        numBlocksParametr = (PARAMETR_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreadsParametr = MAX_THREAD_CUDA;
    }
    else {
        numBlocksParametr = 0;
        numThreadsParametr = PARAMETR_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);

    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    //dim3 kol_ant(ANT_SIZE);

    //Заполнение начальными значениями ant_parametr_dev, antOFdev
    for (int i = 0; i < ANT_SIZE; ++i) {
        for (int j = 0; j < PARAMETR_SIZE; ++j) {
            ant_parametr[i * PARAMETR_SIZE + j] = 0;
        }
        antOF[i] = 1;

    }

    cudaEventRecord(start, 0);
    cudaMemcpy(ant_parametr_dev, ant_parametr, numBytes_matrix_ant, cudaMemcpyHostToDevice);
    cudaMemcpy(antOFdev, antOF, numBytes_ant, cudaMemcpyHostToDevice);
    cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaEventRecord(startAll1, 0);
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        cudaMemcpy(gpuTime_dev, &i_gpuTime, sizeof(int), cudaMemcpyHostToDevice);
        cudaEventRecord(start1, 0);
        go_mass_probability_and_add_pheromon_iteration << <numBlocksParametr, numThreadsParametr >> > (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev, ant_parametr_dev, antOFdev);

        if (PRINT_INFORMATION) {
            cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
            std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
            for (int i = 0; i < PARAMETR_SIZE; ++i) {
                for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                    std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << "-> " << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
                }
                std::cout << std::endl; // Переход на новую строку
            }
        }
        cudaEventRecord(start2, 0);
        go_all_agent_only << <numBlocks, numThreads >> > (gpuTime_dev, parametr_value_dev, norm_matrix_probability_dev, antdev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);

        if (PRINT_INFORMATION) {
            cudaMemcpy(ant, antdev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
            cudaMemcpy(ant_parametr, ant_parametr_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
            cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost);
            std::cout << "ANT (" << ANT_SIZE << "):" << std::endl;
            for (int i = 0; i < ANT_SIZE; ++i) {
                for (int j = 0; j < PARAMETR_SIZE; ++j) {
                    std::cout << ant[i * PARAMETR_SIZE + j] << " ";
                    //std::cout << ant_parametr[i * PARAMETR_SIZE + j] << "(" << ant[i * PARAMETR_SIZE + j] << ") "; 
                }
                std::cout << "-> " << antOF[i] << std::endl;

            }
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        cudaEventElapsedTime(&gpuTime1, start1, stop);
        SumgpuTime1 = SumgpuTime1 + gpuTime1;
        cudaEventElapsedTime(&gpuTime2, start2, stop);
        SumgpuTime2 = SumgpuTime2 + gpuTime2;
        cudaEventElapsedTime(&gpuTime3, start3, stop);
        SumgpuTime3 = SumgpuTime3 + gpuTime3;

        i_gpuTime = int(gpuTime * 1000);
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;

        if (PRINT_INFORMATION) {
            std::cout << "h_seeds (" << int(gpuTime * 1000) << "x" << ANT_SIZE << "):" << std::endl;
            cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
            std::cout << nom_iter << "   MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " GMIN OF DEV -> " << *global_minOf_in_device << "  GMAX OF DEV-> " << *global_maxOf_in_device << " Time: " << gpuTime << " ms " << std::endl;
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime1, startAll1, stop);

    cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost);
    // Освобождение ресурсов
    // Освобождение ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(startAll1);
    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(start3);

    cudaFree(parametr_value_dev);
    cudaFree(pheromon_value_dev);
    cudaFree(kol_enter_value_dev);
    cudaFree(norm_matrix_probability_dev);
    cudaFree(antdev);
    cudaFree(ant_parametr_dev);
    cudaFree(antOFdev);
    cudaFree(hashTable_dev);
    cudaFree(gpuTime_dev);
    cudaFree(maxOf_dev);
    cudaFree(minOf_dev);
    cudaFree(kol_hash_fail);

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime, startAll, stop);

    cudaEventDestroy(startAll);
    cudaEventDestroy(stop);
    std::cout << "Time CUDA opt ant par:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA opt ant par:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;

    return 0;
}

static int start_CUDA_opt_one_GPU() {
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    float gpuTime = 0.0f;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float gpuTime3 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int i_gpuTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&startAll);
    cudaEventCreate(&startAll1);
    cudaEventCreate(&stop);
    cudaEventRecord(startAll, 0);
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
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
    double* ant_parametr = new double[kolBytes_matrix_ant];
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
    double* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int* gpuTime_dev = nullptr;

    cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph);

    cudaMalloc((void**)&antdev, numBytes_matrix_ant);
    cudaMalloc((void**)&antOFdev, numBytes_ant);
    cudaMalloc((void**)&maxOf_dev, sizeof(double));
    cudaMalloc((void**)&minOf_dev, sizeof(double));
    cudaMalloc((void**)&kol_hash_fail, sizeof(int));
    cudaMalloc((void**)&ant_parametr_dev, numBytes_matrix_ant);
    cudaMalloc((void**)&gpuTime_dev, sizeof(int));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry));
    int threads_init_hash = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threads_init_hash - 1) / threads_init_hash;
    initializeHashTable << <blocks_init_hash, threads_init_hash >> > (hashTable_dev, HASH_TABLE_SIZE);

    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    dim3 kol_ant(ANT_SIZE);

    cudaEventRecord(start, 0);
    cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaEventRecord(startAll1, 0);
    go_all_agent_opt << <kol_ant, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, gpuTime_dev, parametr_value_dev, norm_matrix_probability_dev, antdev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime1, startAll1, stop);
    cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost);
    // Освобождение ресурсов
    // Освобождение ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(startAll1);

    cudaFree(parametr_value_dev);
    cudaFree(pheromon_value_dev);
    cudaFree(kol_enter_value_dev);
    cudaFree(norm_matrix_probability_dev);
    cudaFree(antdev);
    cudaFree(ant_parametr_dev);
    cudaFree(antOFdev);
    cudaFree(hashTable_dev);
    cudaFree(gpuTime_dev);
    cudaFree(maxOf_dev);
    cudaFree(minOf_dev);
    cudaFree(kol_hash_fail);

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime, startAll, stop);

    cudaEventDestroy(startAll);
    cudaEventDestroy(stop);
    std::cout << "Time CUDA opt one:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device  << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA opt one:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;

    return 0;
}

static int start_CUDA_opt_one_GPU_non_hash() {
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    float gpuTime = 0.0f;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float gpuTime3 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int i_gpuTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&startAll);
    cudaEventCreate(&startAll1);
    cudaEventCreate(&stop);
    cudaEventRecord(startAll, 0);
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
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
    double* ant_parametr = new double[kolBytes_matrix_ant];
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
    double* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int* gpuTime_dev = nullptr;

    cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph);

    cudaMalloc((void**)&antdev, numBytes_matrix_ant);
    cudaMalloc((void**)&antOFdev, numBytes_ant);
    cudaMalloc((void**)&maxOf_dev, sizeof(double));
    cudaMalloc((void**)&minOf_dev, sizeof(double));
    cudaMalloc((void**)&kol_hash_fail, sizeof(int));
    cudaMalloc((void**)&ant_parametr_dev, numBytes_matrix_ant);
    cudaMalloc((void**)&gpuTime_dev, sizeof(int));

    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    dim3 kol_ant(ANT_SIZE);

    cudaEventRecord(start, 0);
    cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaEventRecord(startAll1, 0);
    go_all_agent_opt_non_hash << <kol_ant, kol_parametr >> > (pheromon_value_dev, kol_enter_value_dev, gpuTime_dev, parametr_value_dev, norm_matrix_probability_dev, antdev, ant_parametr_dev, antOFdev, maxOf_dev, minOf_dev, kol_hash_fail);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime1, startAll1, stop);
    cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost);
    // Освобождение ресурсов
    // Освобождение ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(startAll1);

    cudaFree(parametr_value_dev);
    cudaFree(pheromon_value_dev);
    cudaFree(kol_enter_value_dev);
    cudaFree(norm_matrix_probability_dev);
    cudaFree(antdev);
    cudaFree(ant_parametr_dev);
    cudaFree(antOFdev);
    cudaFree(gpuTime_dev);
    cudaFree(maxOf_dev);
    cudaFree(minOf_dev);
    cudaFree(kol_hash_fail);

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime, startAll, stop);

    cudaEventDestroy(startAll);
    cudaEventDestroy(stop);
    std::cout << "Time CUDA opt one non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA opt one non hash:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;

    return 0;
}

static int start_CUDA_opt_one_GPU_ant() {
    // Создание обработчиков событий CUDA
    cudaEvent_t start, startAll, startAll1, start1, start2, start3, stop;
    float gpuTime = 0.0f;
    float AllgpuTime = 0.0f;
    float AllgpuTime1 = 0.0f;
    float gpuTime1 = 0.0f;
    float gpuTime2 = 0.0f;
    float gpuTime3 = 0.0f;
    float SumgpuTime1 = 0.0f;
    float SumgpuTime2 = 0.0f;
    float SumgpuTime3 = 0.0f;
    int i_gpuTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&startAll);
    cudaEventCreate(&startAll1);
    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop);
    cudaEventRecord(startAll, 0);
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
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
    double* ant_parametr = new double[kolBytes_matrix_ant];
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
    double* ant_parametr_dev = nullptr;
    double* maxOf_dev = nullptr;
    double* minOf_dev = nullptr;
    int* kol_hash_fail = nullptr;
    int* gpuTime_dev = nullptr;

    cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph);

    cudaMalloc((void**)&antdev, numBytes_matrix_ant);
    cudaMalloc((void**)&antOFdev, numBytes_ant);
    cudaMalloc((void**)&maxOf_dev, sizeof(double));
    cudaMalloc((void**)&minOf_dev, sizeof(double));
    cudaMalloc((void**)&kol_hash_fail, sizeof(int));
    cudaMalloc((void**)&ant_parametr_dev, numBytes_matrix_ant);
    cudaMalloc((void**)&gpuTime_dev, sizeof(int));

    // Allocate memory for the hash table on the device
    HashEntry* hashTable_dev = nullptr;
    cudaMalloc((void**)&hashTable_dev, HASH_TABLE_SIZE * sizeof(HashEntry));
    const int threadsPerBlock = MAX_THREAD_CUDA;
    int blocks_init_hash = (HASH_TABLE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    if (threadsPerBlock > ANT_SIZE) {
        numBlocks = (ANT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        numThreads = MAX_THREAD_CUDA;
    }
    else {
        numBlocks = 0;
        numThreads = ANT_SIZE;
    }
    initializeHashTable << <blocks_init_hash, threadsPerBlock >> > (hashTable_dev, HASH_TABLE_SIZE);

    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    //dim3 kol_ant(ANT_SIZE);

    cudaEventRecord(start, 0);
    cudaMemcpy(maxOf_dev, &global_maxOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(minOf_dev, &global_minOf, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaEventRecord(startAll1, 0);
    go_all_agent_opt_only << <numBlocks, numThreads >> > (pheromon_value_dev, kol_enter_value_dev, gpuTime_dev, parametr_value_dev, norm_matrix_probability_dev, antdev, ant_parametr_dev, antOFdev, hashTable_dev, maxOf_dev, minOf_dev, kol_hash_fail);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime1, startAll1, stop);
    cudaMemcpy(global_maxOf_in_device, maxOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(global_minOf_in_device, minOf_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(kol_hash_fail_in_device, kol_hash_fail, sizeof(int), cudaMemcpyDeviceToHost);
    // Освобождение ресурсов
    // Освобождение ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(startAll1);
    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(start3);

    cudaFree(parametr_value_dev);
    cudaFree(pheromon_value_dev);
    cudaFree(kol_enter_value_dev);
    cudaFree(norm_matrix_probability_dev);
    cudaFree(antdev);
    cudaFree(ant_parametr_dev);
    cudaFree(antOFdev);
    cudaFree(hashTable_dev);
    cudaFree(gpuTime_dev);
    cudaFree(maxOf_dev);
    cudaFree(minOf_dev);
    cudaFree(kol_hash_fail);

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&AllgpuTime, startAll, stop);

    cudaEventDestroy(startAll);
    cudaEventDestroy(stop);
    std::cout << "Time CUDA opt one ant:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;
    logFile << "Time CUDA opt one ant:;" << AllgpuTime << "; " << AllgpuTime1 << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << *global_minOf_in_device << "; " << *global_maxOf_in_device << ";" << *kol_hash_fail_in_device / PARAMETR_SIZE << ";" << std::endl;

    return 0;
}


// ----------------- Kernel: Initializing Hash Table -----------------
void initializeHashTable_non_cuda(HashEntry* hashTable, int size) {
    for (int i = 0; i < size; i++) {
        hashTable[i].key = 0;
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
        if (hashTable[idx].key == 0) {
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
        unsigned long long expected = 0;
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
    if ((kol_enter != 0)&&(pheromon != 0)) {
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

// Функция для генерации случайного числа в диапазоне [0, 1]
//double generate_random_value_non_cuda(int seed) {
//    srand(seed);
//    return static_cast<double>(rand()) / RAND_MAX; // Генерация случайного числа в диапазоне [0, 1]
//}

// Функция для вычисления пути агентов на CPU
void go_all_agent_non_cuda(int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF, HashEntry* hashTable, int& kol_hash_fail) {
    // Генератор случайных чисел
    std::default_random_engine generator(123 +gpuTime); // Используем gpuTime как начальное значение
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
            kol_hash_fail = kol_hash_fail + 1;
            //Поиск алгоритма для нулевого агента
            switch (TYPE_ACO) {
            case 0: // ACOCN
                OF[bx] = cachedResult;
                break;
            case 1: // ACOCNI
                OF[bx] = 0;
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
                    double cachedResult = getCachedResultOptimized_non_cuda(hashTable, agent_node, bx);
                    nom_iteration = nom_iteration + 1;
                }
                OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
                break;
            default:
                OF[bx] = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                break;
            }


        }
    }
}

void go_all_agent_non_cuda_non_hash(int gpuTime, double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF, int& kol_hash_fail) {
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

        OF[bx] = BenchShafferaFunction_non_cuda(&agent[bx * PARAMETR_SIZE]);
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
            pheromon[MAX_VALUE_SIZE * tx + k] += PARAMETR_Q *(MAX_PARAMETR_VALUE_TO_MIN_OPT-OF[i]); // MIN
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

int start_NON_CUDA() {
    auto start = std::chrono::high_resolution_clock::now();
    float SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f;
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
        std::chrono::duration<double> current_time = end_temp - start;
        go_all_agent_non_cuda(current_time.count()*100000, parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, hashTable, kol_hash_fail);
        
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
            if (antOF[i] > maxOf) {
                maxOf = antOF[i];
            }
            if (antOF[i] < minOf) {
                minOf = antOF[i];
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
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_iteration = end - start_iteration;

    // Освобождение выделенной памяти
    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    auto end_all = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_all - start;
    std::cout << "Time non CUDA;" << duration.count() << "; " << duration_iteration.count() << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA:;" << duration.count() << "; " << duration_iteration.count() << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

    return 0;
}

int start_NON_CUDA_non_hash() {
    auto start = std::chrono::high_resolution_clock::now();
    float SumgpuTime1 = 0.0f, SumgpuTime2 = 0.0f, SumgpuTime3 = 0.0f;
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
        std::chrono::duration<double> current_time = end_temp - start;
        go_all_agent_non_cuda_non_hash(current_time.count() * 100000, parametr_value, norm_matrix_probability, ant, ant_parametr, antOF, kol_hash_fail);

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
            if (antOF[i] > maxOf) {
                maxOf = antOF[i];
            }
            if (antOF[i] < minOf) {
                minOf = antOF[i];
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
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_iteration = end - start_iteration;

    // Освобождение выделенной памяти
    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;
    auto end_all = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_all - start;
    std::cout << "Time non CUDA non hash;" << duration.count() << "; " << duration_iteration.count() << "; " << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time non CUDA non hash:;" << duration.count() << "; " << duration_iteration.count() << SumgpuTime1 << "; " << SumgpuTime2 << "; " << SumgpuTime3 << ";" << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

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

void start_ant_classic() {
    auto start = std::chrono::high_resolution_clock::now();
    double global_maxOf = -std::numeric_limits<double>::max();
    double global_minOf = std::numeric_limits<double>::max();
    float SumTime1 = 0.0f;
    float duration_iteration = 0.0f;
    int kol_hash_fail = 0;
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

            Ant ants[ANT_SIZE]; // Массив муравьев
            for (int nom_ant = 0; nom_ant < ANT_SIZE; nom_ant++) { // Проходим по всем агентам
                //std::cout << " nom_ant:" << nom_ant;
                double objValue = 0;
                ants[nom_ant] = Ant(pg); // Инициализация каждого муравья с графом
                std::vector<int> path = ants[nom_ant].findPath(); // Поиск пути
                //Проверка Хэш функции
                // Проверка наличия решения в Хэш-таблице
                double cachedResult = getCachedResultOptimized_classic_ant(hashTable, path);
                //std::cout << " cachedResult:" << cachedResult;
                int nom_iteration = 0;
                if (cachedResult == -1.0) {
                    // Если значение не найденов ХЭШ, то заносим новое значение
                    objValue = ants[nom_ant].evaluateObjectiveFunction(); // Оценка целевой функции
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
                        objValue = 0;
                        break;
                    case 2: // ACOCCyN
                        while ((cachedResult != -1.0) && (nom_iteration < ACOCCyN_KOL_ITERATION))
                        {
                            nom_iteration = nom_iteration + 1;
                            std::vector<int> path = ants[nom_ant].findPath(); // Поиск пути
                            double cachedResult = getCachedResultOptimized_classic_ant(hashTable, path);
                        }
                        objValue = ants[nom_ant].evaluateObjectiveFunction(); // Оценка целевой функции
                        saveToCacheOptimized_classic_ant(hashTable, path, objValue);
                        break;
                    default:
                        objValue = cachedResult; // Обработка случая, если TYPE_ACO не соответствует ни одному из вариантов
                        break;
                    }
                }
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
        }
        auto end_iteration = std::chrono::high_resolution_clock::now();
        duration_iteration = std::chrono::duration<float, std::milli> (end_iteration - start_iteration).count();
        
    }
    else {
        std::cerr << "Graph Load Error" << std::endl;
    }
    delete[] hashTable; // Освобождаем память
    auto end_all = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_all - start;
    std::cout << "Time classical ACO;" << duration.count() << "; " << duration_iteration << "; " << SumTime1 << "; " << "0" << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time classical ACO;" << duration.count() << "; " << duration_iteration << "; " << SumTime1 << "; " << "0" << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

}

void start_ant_classic_non_hash() {
    auto start = std::chrono::high_resolution_clock::now();
    double global_maxOf = -std::numeric_limits<double>::max();
    double global_minOf = std::numeric_limits<double>::max();
    float SumTime1 = 0.0f;
    float duration_iteration = 0.0f;
    int kol_hash_fail = 0;
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

            Ant ants[ANT_SIZE]; // Массив муравьев
            for (int nom_ant = 0; nom_ant < ANT_SIZE; nom_ant++) { // Проходим по всем агентам
                //std::cout << " nom_ant:" << nom_ant;
                double objValue = 0;
                ants[nom_ant] = Ant(pg); // Инициализация каждого муравья с графом
                std::vector<int> path = ants[nom_ant].findPath(); // Поиск пути
                //Проверка Хэш функции
                objValue = ants[nom_ant].evaluateObjectiveFunction(); // Оценка целевой функции
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
        }
        auto end_iteration = std::chrono::high_resolution_clock::now();
        duration_iteration = std::chrono::duration<float, std::milli>(end_iteration - start_iteration).count();

    }
    else {
        std::cerr << "Graph Load Error" << std::endl;
    }
    auto end_all = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_all - start;
    std::cout << "Time classical ACO non hash;" << duration.count() << "; " << duration_iteration << "; " << SumTime1 << "; " << "0" << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;
    logFile << "Time classical ACO non hash;" << duration.count() << "; " << duration_iteration << "; " << SumTime1 << "; " << "0" << "; " << "0" << "; " << global_minOf << "; " << global_maxOf << "; " << kol_hash_fail << "; " << std::endl;

}


int main(int argc, char* argv[]) {
    // Открытие лог-файла
    logFile.open("log.txt");
    if (!logFile.is_open()) {
        std::cerr << "Ошибка открытия лог-файла!" << std::endl;
        return 1; // Возврат с ошибкой
    }
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
 
    if (GO_CUDA) {
        // Запуск таймера
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            start_CUDA();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double> duration = end1 - start1;
        // Вывод информации на экран и в лог-файл
        std::string message = "Time CUDA:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
    }

    if (GO_CUDA_NON_HASH) {
        // Запуск таймера
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            start_CUDA_non_hash();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double> duration = end1 - start1;
        // Вывод информации на экран и в лог-файл
        std::string message = "Time CUDA non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
    }

    if (GO_CUDA_ANT) {
        // Запуск таймера
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            start_CUDA_ant();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double> duration = end1 - start1;
        std::string message = "Time CUDA ant:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
    }

    if (GO_CUDA_ANT_PAR) {
        // Запуск таймера
        auto start1 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            start_CUDA_ant_par();
            i = i + 1;
        }
        // Остановка таймера
        auto end1 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double> duration = end1 - start1;
        std::string message = "Time CUDA ant par:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
    }
 
    if (GO_CUDA_OPT) {
        // Запуск таймера
        auto start = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            start_CUDA_opt();
            i = i + 1;
        }
        // Остановка таймера
        auto end = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double> duration = end - start;
        std::string message = "Time CUDA opt:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
    }

    if (GO_CUDA_OPT_NON_HASH) {
        // Запуск таймера
        auto start = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            start_CUDA_opt_non_hash();
            i = i + 1;
        }
        // Остановка таймера
        auto end = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double> duration = end - start;
        std::string message = "Time CUDA opt non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
    }

    if (GO_CUDA_OPT_ANT) {
        // Запуск таймера
        auto start = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            start_CUDA_opt_ant();
            i = i + 1;
        }
        // Остановка таймера
        auto end = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double> duration = end - start;
        std::string message = "Time CUDA opt ant:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
    }

    if (GO_CUDA_OPT_ANT_PAR) {
        // Запуск таймера
        auto start = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            start_CUDA_opt_ant_par();
            i = i + 1;
        }
        // Остановка таймера
        auto end = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double> duration = end - start;
        std::string message = "Time CUDA opt ant par:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
    }
 
    if (GO_CUDA_ONE_OPT) {
        // Запуск таймера
        auto start0 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            start_CUDA_opt_one_GPU();
            i = i + 1;
        }
        // Остановка таймера
        auto end0 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double> duration = end0 - start0;
        std::string message = "Time CUDA opt_one_GPU:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
    }

    if (GO_CUDA_ONE_OPT_NON_HASH) {
        // Запуск таймера
        auto start = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            start_CUDA_opt_one_GPU_non_hash();
            i = i + 1;
        }
        // Остановка таймера
        auto end = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double> duration = end - start;
        std::string message = "Time CUDA opt_one_GPU non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
    }

    if (GO_CUDA_ONE_OPT_ANT) {
        // Запуск таймера
        auto start0 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            start_CUDA_opt_one_GPU_ant();
            i = i + 1;
        }
        // Остановка таймера
        auto end0 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double> duration = end0 - start0;
        std::string message = "Time CUDA opt_one_GPU ant:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
    }
     
    if (GO_NON_CUDA) {
        // Запуск таймера
        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            start_NON_CUDA();
            i = i + 1;
        }
        // Остановка таймера
        auto end2 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double> duration = end2 - start2;
        std::string message = "Time non CUDA:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
    }

    if (GO_NON_CUDA_NON_HASH) {
        // Запуск таймера
        auto start2 = std::chrono::high_resolution_clock::now();
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
            start_NON_CUDA_non_hash();
            i = i + 1;
        }
        // Остановка таймера
        auto end2 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double> duration = end2 - start2;
        std::string message = "Time non CUDA non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
    }

    if (GO_CLASSIC_ACO) {
        // Запуск таймера
        auto start3 = std::chrono::high_resolution_clock::now();
        int numAnts = ANT_SIZE;
        int numIterations = KOL_ITERATION;
        double evaporationRate = 0.999; // Параметр испарения
        int i = 0;
        while (i < KOL_PROGON_STATISTICS)
        {
           
            start_ant_classic();

            i = i + 1;
            
        }
        // Остановка таймера
        auto end3 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double> duration = end3 - start3;
        std::string message = "Time Classic ACO:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
    }

    if (GO_CLASSIC_ACO_NON_HASH) {
        // Запуск таймера
        auto start3 = std::chrono::high_resolution_clock::now();
        int numAnts = ANT_SIZE;
        int numIterations = KOL_ITERATION;
        double evaporationRate = 0.999; // Параметр испарения
        int i = 0;
        while (i < KOL_PROGON_STATISTICS){
            start_ant_classic_non_hash();
            i = i + 1;
        }
        // Остановка таймера
        auto end3 = std::chrono::high_resolution_clock::now();
        // Вычисление времени выполнения
        std::chrono::duration<double> duration = end3 - start3;
        std::string message = "Time Classic ACO non hash:;" + std::to_string(duration.count()) + ";sec";
        std::cout << message << std::endl;
        logFile << message << std::endl; // Запись в лог-файл
    }

    // Закрытие лог-файла
    logFile.close();
}
