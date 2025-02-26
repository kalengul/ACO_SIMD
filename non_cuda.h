#pragma once
// non_cuda.h
#ifndef NON_CUDA_H
#define NON_CUDA_H

int start_NON_CUDA();

double go_x1_6_non_cuda(double* parametr);
double go_x2_6_non_cuda(double* parametr);
double BenchShafferaFunction_non_cuda(double* parametr);
double probability_formula_non_cuda(double pheromon, double kol_enter);
void go_mass_probability_non_cuda(double* pheromon, double* kol_enter, double* norm_matrix_probability);
void go_all_agent_non_cuda(int* gpuTime, double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF);
void add_pheromon_iteration_non_cuda(double* pheromon, double* kol_enter, double* agent_node, double* OF);

#endif // NON_CUDA_H
