#pragma once
// parameters.h
#ifndef PARAMETERS_H
#define PARAMETERS_H

// ����������� ����������
// 10000 - 6 12 24 48 96 198
/*
#define PARAMETR_SIZE 6   // ���������� ���������� 21*x (6*�)
#define PARAMETR_SIZE_ONE_X 3    // ���������� ���������� �� ��� x 21 (6)
#define MAX_VALUE_SIZE 100000    // ������������ ���������� �������� � ���������� 5 (100)
#define NAME_FILE_GRAPH "Parametr_Graph/test42_100000.txt"
 */
// 10000 - 8 16 32 64 128 256
/*
#define PARAMETR_SIZE 16   // ���������� ���������� 21*x (6*�)
#define PARAMETR_SIZE_ONE_X 4    // ���������� ���������� �� ��� x 21 (6)
#define MAX_VALUE_SIZE 10000    // ������������ ���������� �������� � ���������� 5 (100)
#define NAME_FILE_GRAPH "Parametr_Graph/test84_10000.txt"
*/
// 1000 - 10 20 40 80 160 320
/*
#define PARAMETR_SIZE 80   // ���������� ���������� 21*x (6*�)
#define PARAMETR_SIZE_ONE_X 5    // ���������� ���������� �� ��� x 21 (6)
#define MAX_VALUE_SIZE 1000    // ������������ ���������� �������� � ���������� 5 (100)
#define NAME_FILE_GRAPH "Parametr_Graph/test336_1000.txt"
*/
// 100 - 12 24 48 96 192 384
/*
#define PARAMETR_SIZE 768   // ���������� ���������� 21*x (6*�)
#define PARAMETR_SIZE_ONE_X 6    // ���������� ���������� �� ��� x 21 (6)
#define MAX_VALUE_SIZE 100    // ������������ ���������� �������� � ���������� 5 (100)
#define NAME_FILE_GRAPH "Parametr_Graph/test2688_100.txt"
*/
// 42, 84, 168, 336, 672, 1344, 2688

#define PARAMETR_SIZE 336   // ���������� ���������� 21*x (6*�)
#define PARAMETR_SIZE_ONE_X 21    // ���������� ���������� �� ��� x 21 (6)
#define MAX_VALUE_SIZE 5    // ������������ ���������� �������� � ���������� 5 (100)
#define NAME_FILE_GRAPH "Parametr_Graph/test336.txt"

#define ANT_SIZE 500      // ������������ ���������� ������� 500
#define KOL_ITERATION 500   // ���������� �������� ��� 500
#define KOL_STAT_LEVEL 20    // ���������� ������ ����� ���������� 20
#define KOL_PROGON_STATISTICS 50 //��� ����� ���������� 50
#define KOL_PROGREV 5 //���������� �������� ��� ����������� ������� ������� 5
#define PARAMETR_Q 1        // �������� ��� ��� �������� �������� Q 
#define MAX_PARAMETR_VALUE_TO_MIN_OPT 1        // ������������ �������� ��������� ����� ��������� ������� max-x
#define PARAMETR_RO 0.999     // �������� ��� ��� ��������� �������� RO
#define HASH_TABLE_SIZE 10000000 // Hash table size (10 million entries)
#define MAX_PROBES 10000 // Maximum number of probes for collision resolution
#define MAX_THREAD_CUDA 1024 //512
#define ZERO_HASH_RESULT 100000000
#define TYPE_ACO 2
#define ACOCCyN_KOL_ITERATION 50
#define PRINT_INFORMATION 0
#define CPU_RANDOM 0
#define KOL_THREAD_CPU_ANT 12
#define CONST_AVX 4 //double = 4, floaf,int = 8
#define CONST_RANDOM 100000
#define MAX_CONST 8000

/*
// ��� ������ �������
#define GO_CUDA_TIME 1 //�� 672
#define GO_CUDA 1  //�� 672
#define GO_CUDA_NON_HASH 1 //�� 672
#define GO_CUDA_BLOCK_TIME 0 //PROGREV 0 Time CUDA Time only block:;1.56838e+07; 1.56838e+07; 1.56837e+07; 1.56836e+07; 485.034;0;0;0.0832818; 0.901077;23530;
#define GO_CUDA_ANT 1
#define GO_CUDA_ANT_PAR 1
#define GO_CUDA_ANT_NON_HASH 1
#define GO_CUDA_ANT_ADD_CPU 0
#define GO_CUDA_ANT_ADD_CPU_TIME 1
#define GO_CUDA_ANT_ADD_CPU_NON_HASH 0
#define GO_CUDA_ANT_ADD_CPU_OPTMEM 0
#define GO_CUDA_ANT_ADD_CPU2_TIME 1 //�� ����������, ����� ��� ������������ ��������
#define GO_CUDA_OPT 1
#define GO_CUDA_OPT_TIME 1
#define GO_CUDA_OPT_NON_HASH 1
#define GO_CUDA_OPT_ANT 1
#define GO_CUDA_OPT_ANT_NON_HASH 1
#define GO_CUDA_OPT_ANT_PAR 1
#define GO_CUDA_ONE_OPT 1  //�� ����������
#define GO_CUDA_ONE_OPT_LOCAL 0  //�� ����������
#define GO_CUDA_ONE_OPT_NON_HASH 0 //�� ����������
#define GO_CUDA_ONE_OPT_ANT 1 // �� �������� �� 336
#define GO_CUDA_ONE_OPT_ANT_LOCAL 1 // �� �������� �� 336
#define GO_CUDA_ONE_OPT_ANT_NON_HASH 0 // �� �������� �� 336

#define GO_NON_CUDA_TIME 1
#define GO_NON_CUDA 1
#define GO_NON_CUDA_NON_HASH 1
#define GO_NON_CUDA_THREAD 0 //������������, �� 672 ����� ����� PROGREV 0 Time non CUDA thread;137785; 0; 0; 0; 0; 0; 0; 0; 0; 0.235521; 0.777734; 12433261;
#define GO_CLASSIC_ACO 1
#define GO_CLASSIC_ACO_NON_HASH 1
#define GO_OMP 1
#define GO_OMP_NON_HASH 1
#define GO_MPI 0
#define GO_NON_CUDA_TRANSP_TIME 1
#define GO_NON_CUDA_TRANSP 1
#define GO_NON_CUDA_TRANSP_NON_HASH 1
#define GO_NON_CUDA_AVX_TIME 1
#define GO_NON_CUDA_AVX 1
#define GO_NON_CUDA_AVX_NON_HASH 1
#define GO_NON_CUDA_AVX_OMP_TIME 1
#define GO_NON_CUDA_AVX_OMP 1
#define GO_NON_CUDA_AVX_OMP_NON_HASH 1
#define GO_CUDA_ANT_ADD_CPU_AVX 0
#define GO_CUDA_ANT_ADD_CPU_AVX_TIME 1
#define GO_CUDA_ANT_ADD_CPU_AVX_NON_HASH 0
#define GO_NON_CUDA_TRANSP_AVX_TIME 1
#define GO_NON_CUDA_TRANSP_AVX 1
#define GO_NON_CUDA_TRANSP_AVX_NON_HASH 1
#define GO_NON_CUDA_TRANSP_AVX_OMP_TIME 1
#define GO_NON_CUDA_TRANSP_AVX_OMP 1
#define GO_NON_CUDA_TRANSP_AVX_OMP_NON_HASH 1
#define GO_CUDA_ANT_ADD_CPU_AVX_TRANSP 0
#define GO_CUDA_ANT_ADD_CPU_AVX_TRANSP_TIME 1
#define GO_CUDA_ANT_ADD_CPU_AVX_TRANSP_NON_HASH 0
#define GO_CUDA_CONST 1  //�� 672
#define GO_CUDA_ANT_CONST 1
#define GO_CUDA_ANT_PAR_CONST 1
#define GO_CUDA_ANT_ADD_CPU_CONST 1
#define GO_CUDA_OPT_CONST 1
#define GO_CUDA_OPT_ANT_CONST 1
#define GO_CUDA_OPT_ANT_PAR_CONST 1
*/

#define GO_CUDA 0  //�� 672
#define GO_CUDA_TIME 0 //�� 672
#define GO_CUDA_NON_HASH 0 //�� 672
#define GO_CUDA_BLOCK_TIME 0 //PROGREV 0 Time CUDA Time only block:;1.56838e+07; 1.56838e+07; 1.56837e+07; 1.56836e+07; 485.034;0;0;0.0832818; 0.901077;23530;
#define GO_CUDA_ANT 0
#define GO_CUDA_ANT_NON_HASH 0
#define GO_CUDA_ANT_PAR 0
#define GO_CUDA_ANT_ADD_CPU 0 //0
#define GO_CUDA_ANT_ADD_CPU_TIME 0 //0
#define GO_CUDA_ANT_ADD_CPU_NON_HASH 0 //0
#define GO_CUDA_ANT_ADD_CPU_OPTMEM 0
#define GO_CUDA_ANT_ADD_CPU2_TIME 0 //�� ����������, ����� ��� ������������ ��������
#define GO_CUDA_OPT 0
#define GO_CUDA_OPT_TIME 0
#define GO_CUDA_OPT_ANT 0
#define GO_CUDA_OPT_ANT_NON_HASH 0
#define GO_CUDA_OPT_ANT_PAR 0
#define GO_CUDA_OPT_NON_HASH 0
#define GO_CUDA_ONE_OPT 1  //�� ����������
#define GO_CUDA_ONE_OPT_LOCAL 0  //�� ����������
#define GO_CUDA_ONE_OPT_NON_HASH 0 //�� ����������
#define GO_CUDA_ONE_OPT_ANT 1 // �� �������� �� 336
#define GO_CUDA_ONE_OPT_ANT_LOCAL 1 // �� �������� �� 336
#define GO_CUDA_ONE_OPT_ANT_NON_HASH 0 // �� �������� �� 336

#define GO_NON_CUDA 0
#define GO_NON_CUDA_TIME 0
#define GO_NON_CUDA_NON_HASH 0
#define GO_NON_CUDA_THREAD 0 //������������, �� 672 ����� ����� PROGREV 0 Time non CUDA thread;137785; 0; 0; 0; 0; 0; 0; 0; 0; 0.235521; 0.777734; 12433261;
#define GO_CLASSIC_ACO 0
#define GO_CLASSIC_ACO_NON_HASH 0
#define GO_OMP 0
#define GO_OMP_NON_HASH 0
#define GO_MPI 0
#define GO_NON_CUDA_TRANSP_TIME 0
#define GO_NON_CUDA_TRANSP 0
#define GO_NON_CUDA_TRANSP_NON_HASH 0
#define GO_NON_CUDA_AVX_TIME 0
#define GO_NON_CUDA_AVX 0
#define GO_NON_CUDA_AVX_NON_HASH 0
#define GO_NON_CUDA_AVX_OMP_TIME 0
#define GO_NON_CUDA_AVX_OMP 0
#define GO_NON_CUDA_AVX_OMP_NON_HASH 0
#define GO_CUDA_ANT_ADD_CPU_AVX 0 //0
#define GO_CUDA_ANT_ADD_CPU_AVX_TIME 0 //0
#define GO_CUDA_ANT_ADD_CPU_AVX_NON_HASH 0 //0
#define GO_NON_CUDA_TRANSP_AVX_TIME 0
#define GO_NON_CUDA_TRANSP_AVX 0
#define GO_NON_CUDA_TRANSP_AVX_NON_HASH 0
#define GO_NON_CUDA_TRANSP_AVX_OMP_TIME 0
#define GO_NON_CUDA_TRANSP_AVX_OMP 0
#define GO_NON_CUDA_TRANSP_AVX_OMP_NON_HASH 0
#define GO_CUDA_ANT_ADD_CPU_AVX_TRANSP 0  //0
#define GO_CUDA_ANT_ADD_CPU_AVX_TRANSP_TIME 0 //0
#define GO_CUDA_ANT_ADD_CPU_AVX_TRANSP_NON_HASH 0 //0
#define GO_CUDA_CONST 1  //�� 672
#define GO_CUDA_ANT_CONST 1
#define GO_CUDA_ANT_PAR_CONST 1
#define GO_CUDA_ANT_ADD_CPU_CONST 1
#define GO_CUDA_OPT_CONST 1
#define GO_CUDA_OPT_ANT_CONST 1
#define GO_CUDA_OPT_ANT_PAR_CONST 1

/*
#define koef1 1    // �������� koef1
#define koef2 1    // �������� koef2
#define koef3 1    // �������� koef3
#define alf1 1    // �������� alf1
#define alf2 1    // �������� alf2
#define alf3 1    // �������� alf3
*/

#endif // PARAMETERS_H
