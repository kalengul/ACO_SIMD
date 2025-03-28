#pragma once
// parameters.h
#ifndef PARAMETERS_H
#define PARAMETERS_H

// Определение параметров
#define PARAMETR_SIZE 168   // Количество параметров
#define PARAMETR_SIZE_ONE_X 21    // Количество параметров на оди x
#define MAX_VALUE_SIZE 5    // Максимальное количество значений у параметров
#define ANT_SIZE 500      // Максимальное количество агентов
#define KOL_ITERATION 500   // Количество итераций ММК
#define KOL_PROGON_STATISTICS 30 //Для сбора статистики
#define PARAMETR_Q 1        // Параметр ММК для усиления феромона Q
#define MAX_PARAMETR_VALUE_TO_MIN_OPT 1        // Максимальное значение параметра чтобы выполнять разницу max-x
#define PARAMETR_RO 0.999     // Параметр ММК для испарения феромона RO
#define HASH_TABLE_SIZE 10000000 // Hash table size (10 million entries)
#define MAX_PROBES 10000 // Maximum number of probes for collision resolution
#define MAX_THREAD_CUDA 512
#define ZERO_HASH_RESULT -1000
#define NAME_FILE_GRAPH "Parametr_Graph/test168.txt"
#define TYPE_ACO 2
#define ACOCCyN_KOL_ITERATION 100
#define PRINT_INFORMATION 0

#define GO_CUDA 0
#define GO_CUDA_ANT 0
#define GO_CUDA_ANT_PAR 0
#define GO_CUDA_NON_HASH 0
#define GO_CUDA_OPT 0
#define GO_CUDA_OPT_ANT 0
#define GO_CUDA_OPT_ANT_PAR 0
#define GO_CUDA_OPT_NON_HASH 0
#define GO_CUDA_ONE_OPT 0
#define GO_CUDA_ONE_OPT_ANT 1
#define GO_CUDA_ONE_OPT_NON_HASH 0

#define GO_NON_CUDA 0
#define GO_NON_CUDA_NON_HASH 0
#define GO_CLASSIC_ACO 0
#define GO_CLASSIC_ACO_NON_HASH 0
/*
#define koef1 1    // Параметр koef1
#define koef2 1    // Параметр koef2
#define koef3 1    // Параметр koef3
#define alf1 1    // Параметр alf1
#define alf2 1    // Параметр alf2
#define alf3 1    // Параметр alf3
*/
#endif // PARAMETERS_H
