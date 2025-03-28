#pragma once
// parameters.h
#ifndef PARAMETERS_H
#define PARAMETERS_H

// ����������� ����������
#define PARAMETR_SIZE 168   // ���������� ����������
#define PARAMETR_SIZE_ONE_X 21    // ���������� ���������� �� ��� x
#define MAX_VALUE_SIZE 5    // ������������ ���������� �������� � ����������
#define ANT_SIZE 500      // ������������ ���������� �������
#define KOL_ITERATION 500   // ���������� �������� ���
#define KOL_PROGON_STATISTICS 30 //��� ����� ����������
#define PARAMETR_Q 1        // �������� ��� ��� �������� �������� Q
#define MAX_PARAMETR_VALUE_TO_MIN_OPT 1        // ������������ �������� ��������� ����� ��������� ������� max-x
#define PARAMETR_RO 0.999     // �������� ��� ��� ��������� �������� RO
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
#define koef1 1    // �������� koef1
#define koef2 1    // �������� koef2
#define koef3 1    // �������� koef3
#define alf1 1    // �������� alf1
#define alf2 1    // �������� alf2
#define alf3 1    // �������� alf3
*/
#endif // PARAMETERS_H
