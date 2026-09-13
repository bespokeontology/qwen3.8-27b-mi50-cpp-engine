// host syntax stub for rocblas.h (tools/hipstub): types and enums used by src/q27_rb.cpp only
#pragma once
#include <hip/hip_runtime.h>
typedef struct _rocblas_handle* rocblas_handle;
typedef int rocblas_status; enum { rocblas_status_success = 0 };
typedef int rocblas_operation; enum { rocblas_operation_none = 111, rocblas_operation_transpose = 112 };
typedef int rocblas_datatype; enum { rocblas_datatype_i8_r = 160, rocblas_datatype_i32_r = 162 };
typedef int rocblas_gemm_algo; enum { rocblas_gemm_algo_standard = 0 };
enum { rocblas_gemm_flags_none = 0, rocblas_gemm_flags_pack_int8x4 = 1 };
typedef int rocblas_pointer_mode; enum { rocblas_pointer_mode_host = 0 };
inline rocblas_status rocblas_create_handle(rocblas_handle*) { return 0; }
inline rocblas_status rocblas_set_stream(rocblas_handle, hipStream_t) { return 0; }
inline rocblas_status rocblas_set_pointer_mode(rocblas_handle, rocblas_pointer_mode) { return 0; }
inline rocblas_status rocblas_gemm_ex(rocblas_handle, rocblas_operation, rocblas_operation, int, int, int, const void*, const void*, rocblas_datatype, int, const void*, rocblas_datatype, int, const void*, const void*, rocblas_datatype, int, void*, rocblas_datatype, int, rocblas_datatype, rocblas_gemm_algo, int, unsigned) { return 0; }
enum { rocblas_gemm_algo_solution_index = 1 };
typedef int rocblas_int;
inline rocblas_status rocblas_gemm_ex_get_solutions(rocblas_handle, rocblas_operation, rocblas_operation, int, int, int, const void*, const void*, rocblas_datatype, int, const void*, rocblas_datatype, int, const void*, const void*, rocblas_datatype, int, void*, rocblas_datatype, int, rocblas_datatype, rocblas_gemm_algo, unsigned, rocblas_int*, rocblas_int* n) { if (n) *n = 0; return 0; }
