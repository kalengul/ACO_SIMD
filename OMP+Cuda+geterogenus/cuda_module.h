#ifndef CUDA_MODULE_H
#define CUDA_MODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef BUILD_CUDA_DLL
#define CUDA_API __declspec(dllexport)
#else
#define CUDA_API __declspec(dllimport)
#endif

CUDA_API bool cuda_initialize(const double* parametr_value, const double* pheromon_value, const double* kol_enter_value);
CUDA_API bool cuda_initialize_non_hash(const double* parametr_value, const double* pheromon_value, const double* kol_enter_value);
CUDA_API void cuda_run_async(const double* norm_matrix_probability,
                            const int* ant_parametr,
                            double* antOF,
                            int iteration,
                            void (*completion_callback)(double*, int, int));
CUDA_API void cuda_run_iteration(const double* norm_matrix_probability, int* ant_parametr, double* antOF, int ant_size, double* global_minOf, double* global_maxOf, int* kol_hash_fail, double* time_all, double* time_function, int iteration, void (*completion_callback)(double*, int, int));
CUDA_API void cuda_run_iteration_non_hash(const double* norm_matrix_probability, int* ant_parametr, double* antOF, int ant_size, double* global_minOf, double* global_maxOf, double* time_all, double* time_function, int iteration, void (*completion_callback)(double*, int, int));

CUDA_API void cuda_cleanup();
CUDA_API void cuda_synchronize();
CUDA_API const char* cuda_get_version();

#ifdef __cplusplus
}
#endif

#endif // CUDA_MODULE_H