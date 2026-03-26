#ifndef UTIL_H
#define UTIL_H
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_ERROR(ans)                      \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

/**
 * @brief Helper function for CHECK_ERROR macro to handle CUDA errors
 * @param code CUDA error code to check
 * @param file Source file where the error occurred
 * @param line Line number where the error occurred
 * @param abort Whether to abort execution on error (default: true)
 */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#endif
