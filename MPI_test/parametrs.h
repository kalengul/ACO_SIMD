#pragma once
// parameters.h
#ifndef PARAMETERS_H
#define PARAMETERS_H

// Определение параметров
// 42, 84, 100, 168, 336, 672, 1344, 2732
#define PARAMETR_SIZE 42   // Количество параметров 
#define PARAMETR_SIZE_ONE_X 21    // Количество параметров на оди x 21
#define MAX_VALUE_SIZE 5    // Максимальное количество значений у параметров
#define ANT_SIZE 500      // Максимальное количество агентов
#define KOL_ITERATION 500   // Количество итераций ММК
#define KOL_STAT_LEVEL 20    // Количество этапов сбора статистики
#define KOL_PROGON_STATISTICS 50 //Для сбора статистики
#define KOL_PROGREV 5 //Количество итераций для начальнойго запуска системы
#define PARAMETR_Q 1        // Параметр ММК для усиления феромона Q
#define MAX_PARAMETR_VALUE_TO_MIN_OPT 1        // Максимальное значение параметра чтобы выполнять разницу max-x
#define PARAMETR_RO 0.999     // Параметр ММК для испарения феромона RO
#define HASH_TABLE_SIZE 10000000 // Hash table size (10 million entries)
#define MAX_PROBES 10000 // Maximum number of probes for collision resolution
#define MAX_THREAD_CUDA 1024
#define ZERO_HASH_RESULT -1000
#define NAME_FILE_GRAPH "Parametr_Graph/test42.txt"
#define TYPE_ACO 2
#define ACOCCyN_KOL_ITERATION 50
#define PRINT_INFORMATION 0
#define CPU_RANDOM 0
#define KOL_THREAD_CPU_ANT 12

#define GO_NON_CUDA 0
#define GO_NON_CUDA_TIME 0
#define GO_NON_CUDA_NON_HASH 0
#define GO_NON_CUDA_THREAD 0
#define GO_CLASSIC_ACO 0
#define GO_CLASSIC_ACO_NON_HASH 0
#define GO_OMP 0
#define GO_OMP_NON_HASH 0
#define GO_MPI 1 //mpiexec -n 4 MPI.exe
/*
#define koef1 1    // Параметр koef1
#define koef2 1    // Параметр koef2
#define koef3 1    // Параметр koef3
#define alf1 1    // Параметр alf1
#define alf2 1    // Параметр alf2
#define alf3 1    // Параметр alf3
*/

#endif // PARAMETERS_H
