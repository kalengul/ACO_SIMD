// Auto-generated parameters for size 5376
// Generated at 25.11.2025 15:57:36,86

#ifdef _WIN32
#define WINDOWS_OPTIMIZATION 1
#pragma once
#endif
// parameters.h
#ifndef PARAMETERS_H
#define PARAMETERS_H

// 42, 84, 168, 336, 672, 1344, 2688, 5376, 10752, 21504, 43008, 86016, 172032, 344064, 688128, 1376256
#define KOL_GPU 4
#define PARAMETR_SIZE 2688
#define SET_PARAMETR_SIZE_ONE_X 21    // ���������� ���������� �� ��� x 21 (6)
#define MAX_VALUE_SIZE 5    // ������������ ���������� �������� � ���������� 5 (100)
#define NAME_FILE_GRAPH "Parametr_Graph/test2688.txt"
#define ANT_SIZE 500
#define KOL_ITERATION 500   // ���������� �������� ��� 500
#define KOL_STAT_LEVEL 20    // ���������� ������ ����� ���������� 20
#define KOL_PROGON_STATISTICS 50 //��� ����� ���������� 50
#define KOL_PROGREV 5 //���������� �������� ��� ����������� ������� ������� 5
#define PARAMETR_Q 1        // �������� ��� ��� �������� �������� Q 
#define MAX_PARAMETR_VALUE_TO_MIN_OPT 1        // ������������ �������� ��������� ����� ��������� ������� max-x
#define PARAMETR_RO 0.999     // �������� ��� ��� ��������� �������� RO
//#define HASH_TABLE_SIZE 1048576  // 2^20
#define GO_HASH 1
#define HASH_TABLE_SIZE 67108864  // 2^26 (������ ���� �������� ������)
#define MAX_PROBES 10000 // Maximum number of probes for collision resolution
#define MAX_THREAD_CUDA 512 //1024
#define ZERO_HASH_RESULT -1
#define ZERO_HASH 0xFFFFFFFFFFFFFFFFULL
#define TYPE_ACO 1
#define ACOCCyN_KOL_ITERATION 50
#define PRINT_INFORMATION 0
#define CPU_RANDOM 0
#define KOL_THREAD_CPU_ANT 12
#define CONST_AVX 4 //double = 4, floaf,int = 8
#define CONST_RANDOM 100000
#define MAX_CONST 8192 //8192 Constant Memory: 65536 bytes
#define MAX_SHARED 6144 //6144 Shared memory per block: 49152 bytes (48 KB)
#define BIN_SEARCH 0
#define NON_WHILE_ANT 1
#define BLOCK_SIZE 64
#define WARP_SIZE 32
#define AGENTS_PER_BLOCK 8
#define TILE_SIZE 256  
#define MAX_SHARED_MEMORY_KB 48  // �������������� �������� ��� ����������� GPU
#define MAX_PARAMETR_SIZE_LIMIT ((MAX_SHARED_MEMORY_KB * 1024) / sizeof(double))
#define GO_ALG_MINMAX 1
#define PAR_MAX_ALG_MINMAX 1000
#define PAR_MIN_ALG_MINMAX 1
#define SHAFFERA 1
#define CARROM_TABLE 0
#define RASTRIGIN 0
#define ACKLEY 0
#define SPHERE 0
#define GRIEWANK 0
#define ZAKHAROV 0
#define SCHWEFEL 0
#define LEVY 0 //�� ��������
#define MICHAELWICZYNSKI 0
#define DELT4 0
#define OPTIMIZE_MIN_1 1
#define OPTIMIZE_MIN_2 0
#define OPTIMIZE_MAX 0

#define M_PI 3.1415926535897932384626433832795028841971
#define M_E 2.7182818284590452353602874713526624977572
#endif // PARAMETERS_H
