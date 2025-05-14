#pragma once
// parameters.h
#ifndef PARAMETERS_H
#define PARAMETERS_H

// ����������� ����������
// 42, 84, 100, 168, 336, 672, 1344, 2732
#define PARAMETR_SIZE 42   // ���������� ���������� 
#define PARAMETR_SIZE_ONE_X 21    // ���������� ���������� �� ��� x 21
#define MAX_VALUE_SIZE 5    // ������������ ���������� �������� � ����������
#define ANT_SIZE 500      // ������������ ���������� �������
#define KOL_ITERATION 500   // ���������� �������� ���
#define KOL_STAT_LEVEL 20    // ���������� ������ ����� ����������
#define KOL_PROGON_STATISTICS 50 //��� ����� ����������
#define KOL_PROGREV 5 //���������� �������� ��� ����������� ������� �������
#define PARAMETR_Q 1        // �������� ��� ��� �������� �������� Q
#define MAX_PARAMETR_VALUE_TO_MIN_OPT 1        // ������������ �������� ��������� ����� ��������� ������� max-x
#define PARAMETR_RO 0.999     // �������� ��� ��� ��������� �������� RO
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
#define koef1 1    // �������� koef1
#define koef2 1    // �������� koef2
#define koef3 1    // �������� koef3
#define alf1 1    // �������� alf1
#define alf2 1    // �������� alf2
#define alf3 1    // �������� alf3
*/

#endif // PARAMETERS_H
