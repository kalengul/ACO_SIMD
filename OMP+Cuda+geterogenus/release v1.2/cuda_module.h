#ifndef CUDA_MODULE_H
#define CUDA_MODULE_H

#ifdef __cplusplus
extern "C" {
#endif

    // ==================== КОНФИГУРАЦИЯ DLL ====================
    // Определение макроса для экспорта/импорта функций в DLL
    // При сборке DLL используется __declspec(dllexport)
    // При использовании DLL используется __declspec(dllimport)

#ifdef BUILD_CUDA_DLL
    // Режим сборки библиотеки - экспортируем функции
#define CUDA_API __declspec(dllexport)
#else
    // Режим использования библиотеки - импортируем функции
#define CUDA_API __declspec(dllimport)
#endif

// ==================== КОНСТАНТЫ ДЛЯ КОМПИЛЯЦИИ ====================
// Эти константы должны быть синхронизированы с cuda_module.cu

#ifndef MAX_VALUE_SIZE
#define MAX_VALUE_SIZE 4
#endif

#ifndef PARAMETR_SIZE
#define PARAMETR_SIZE 1344
#endif

#ifndef ANT_SIZE
#define ANT_SIZE 500
#endif

#ifndef HASH_TABLE_SIZE
#define HASH_TABLE_SIZE 10000000
#endif

// ==================== ОПРЕДЕЛЕНИЯ ФУНКЦИЙ ====================

/**
 * @brief Инициализация CUDA с хэш-таблицей
 *
 * @param parametr_value Указатель на массив параметров размером MAX_VALUE_SIZE * PARAMETR_SIZE
 * @param pheromon_value Указатель на массив феромонов размером MAX_VALUE_SIZE * PARAMETR_SIZE
 * @param kol_enter_value Указатель на массив количества вхождений размером MAX_VALUE_SIZE * PARAMETR_SIZE
 * @return true - инициализация успешна, false - ошибка
 *
 * Функция выделяет память на GPU, копирует данные и инициализирует хэш-таблицу.
 * Требует вызова cuda_cleanup() после завершения работы.
 */
    CUDA_API bool cuda_initialize(const double* parametr_value,
        const double* pheromon_value,
        const double* kol_enter_value);

    /**
     * @brief Инициализация CUDA без хэш-таблицы
     *
     * @param parametr_value Указатель на массив параметров
     * @param pheromon_value Указатель на массив феромонов
     * @param kol_enter_value Указатель на массив количества вхождений
     * @return true - инициализация успешна, false - ошибка
     *
     * Более легкая версия инициализации без выделения памяти для хэш-таблицы.
     * Подходит для случаев, когда кэширование результатов не требуется.
     */
    CUDA_API bool cuda_initialize_non_hash(const double* parametr_value,
        const double* pheromon_value,
        const double* kol_enter_value);

    /**
     * @brief Асинхронный запуск вычислений (упрощенная версия)
     *
     * @param norm_matrix_probability Нормализованная матрица вероятностей
     * @param ant_parametr Массив параметров муравьев (вход)
     * @param antOF Массив значений целевой функции (выход)
     * @param iteration Номер итерации (используется для seed генератора случайных чисел)
     * @param completion_callback Функция обратного вызова после завершения вычислений
     *
     * Упрощенная версия для тестирования базовой функциональности CUDA.
     * Вычисления выполняются асинхронно, callback вызывается по завершении.
     */
    CUDA_API void cuda_run_async(const double* norm_matrix_probability,
        const int* ant_parametr,
        double* antOF,
        int iteration,
        void (*completion_callback)(double*, int, int));

    /**
     * @brief Запуск итерации с хэш-таблицей
     *
     * @param norm_matrix_probability Нормализованная матрица вероятностей
     * @param ant_parametr Массив выбранных параметров муравьев (выход)
     * @param antOF Массив значений целевой функции (выход)
     * @param ant_size Количество муравьев
     * @param global_minOf Минимальное значение целевой функции (выход)
     * @param global_maxOf Максимальное значение целевой функции (выход)
     * @param kol_hash_fail Количество попаданий в хэш-таблицу (статистика)
     * @param time_all Общее время выполнения (аккумулируется)
     * @param time_function Время вычисления целевой функции (аккумулируется)
     * @param iteration Номер итерации
     * @param completion_callback Функция обратного вызова
     *
     * Основная функция для выполнения одной итерации алгоритма с кэшированием
     * результатов в хэш-таблице. Измеряет время выполнения.
     */
    CUDA_API void cuda_run_iteration(const double* norm_matrix_probability,
        int* ant_parametr,
        double* antOF,
        int ant_size,
        double* global_minOf,
        double* global_maxOf,
        int* kol_hash_fail,
        double* time_all,
        double* time_function,
        int iteration,
        void (*completion_callback)(double*, int, int));

    /**
     * @brief Запуск итерации без хэш-таблицы
     *
     * @param norm_matrix_probability Нормализованная матрица вероятностей
     * @param ant_parametr Массив выбранных параметров муравьев (выход)
     * @param antOF Массив значений целевой функции (выход)
     * @param ant_size Количество муравьев
     * @param global_minOf Минимальное значение целевой функции (выход)
     * @param global_maxOf Максимальное значение целевой функции (выход)
     * @param time_all Общее время выполнения (аккумулируется)
     * @param time_function Время вычисления целевой функции (аккумулируется)
     * @param iteration Номер итерации
     * @param completion_callback Функция обратного вызова
     *
     * Версия без кэширования результатов. Быстрее для небольших задач
     * или когда вычисления быстро выполняются.
     */
    CUDA_API void cuda_run_iteration_non_hash(const double* norm_matrix_probability,
        int* ant_parametr,
        double* antOF,
        int ant_size,
        double* global_minOf,
        double* global_maxOf,
        double* time_all,
        double* time_function,
        int iteration,
        void (*completion_callback)(double*, int, int));

    /**
     * @brief Освобождение ресурсов CUDA
     *
     * Функция освобождает всю выделенную память на GPU и уничтожает CUDA stream.
     * Должна вызываться перед завершением программы или при переинициализации.
     */
    CUDA_API void cuda_cleanup();

    /**
     * @brief Синхронизация с выполнением операций на GPU
     *
     * Блокирует выполнение до завершения всех операций в CUDA stream.
     * Полезно для измерения точного времени выполнения или перед выходом.
     */
    CUDA_API void cuda_synchronize();

    /**
     * @brief Получение информации о версии и оборудовании
     *
     * @return Строка с информацией о CUDA устройстве и версии модуля
     *
     * Возвращает статическую строку с описанием:
     * - Версия модуля
     * - Информация о доступных GPU устройствах
     * - Характеристики устройств
     */
    CUDA_API const char* cuda_get_version();

    

#ifdef __cplusplus
}
#endif

// ==================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================
/*
Пример последовательности вызовов:

1. Инициализация:
   double params[MAX_VALUE_SIZE * PARAMETR_SIZE];
   double pheromones[MAX_VALUE_SIZE * PARAMETR_SIZE];
   double counts[MAX_VALUE_SIZE * PARAMETR_SIZE];
   // ... заполнение данных ...
   bool success = cuda_initialize(params, pheromones, counts);

2. Выполнение итераций:
   double probabilities[MAX_VALUE_SIZE * PARAMETR_SIZE];
   int ant_params[PARAMETR_SIZE * ant_size];
   double antOF[ant_size];
   double minVal, maxVal;
   int hashHits = 0;
   double total_time = 0, func_time = 0;

   for (int iter = 0; iter < num_iterations; iter++) {
       // ... обновление probabilities ...
       cuda_run_iteration(probabilities, ant_params, antOF, ant_size,
                         &minVal, &maxVal, &hashHits,
                         &total_time, &func_time, iter, my_callback);
   }

3. Завершение:
   cuda_cleanup();
*/

#endif // CUDA_MODULE_H