#pragma once

#include <iostream>
#include <cstdint>

#ifndef __CUDACC__
    #include <chrono>
#endif

#ifndef __CUDACC__
    #define TIMERSTART(label)                                                  \
        std::chrono::time_point<std::chrono::system_clock> a ## label, b ## label; \
        a ## label = std::chrono::system_clock::now();
#else
    #define TIMERSTART(label)                                                  \
        cudaEvent_t start ## label, stop ## label;                                 \
        float time ## label;                                                     \
        cudaEventCreate(&start ## label);                                        \
        cudaEventCreate(&stop ## label);                                         \
        cudaEventRecord(start ## label, 0);
#endif

#ifndef __CUDACC__
    #define TIMERSTOP(label)                                                   \
        b ## label = std::chrono::system_clock::now();                           \
        std::chrono::duration<double> delta ## label = b ## label-a ## label;        \
        std::cout << "# elapsed time ("<< #label <<"): "                       \
                  << delta ## label.count()  << "s" << std::endl;
#else
    #define TIMERSTOP(label)                                                   \
        cudaEventRecord(stop ## label, 0);                                   \
        cudaEventSynchronize(stop ## label);                                 \
        cudaEventElapsedTime(&time ## label, start ## label, stop ## label);     \
        std::cout << "TIMING: " << time ## label << " ms (" << #label << ")" \
                  << std::endl;
#endif


#ifdef __CUDACC__
    #define CUERR {                                                            \
                cudaError_t err;                                                       \
                if ((err = cudaGetLastError()) != cudaSuccess) {                       \
                        std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "    \
                                  << __FILE__ << ", line " << __LINE__ << std::endl;       \
                        exit(1);                                                           \
                }                                                                      \
}
    #define CUSPAERR(status, errorMsg){ \
                if(status!= 0) { \
                        switch (status) \
                        { \
                        case CUSPARSE_STATUS_NOT_INITIALIZED: \
                                std::cout << "CuSparse error: " <<"CUSPARSE_STATUS_NOT_INITIALIZED " << status <<" "<< errorMsg << std::endl; \
                                throw std::exception();  \
                        case CUSPARSE_STATUS_ALLOC_FAILED: \
                                std::cout << "CuSparse error: " <<"CUSPARSE_STATUS_ALLOC_FAILED " << status <<" "<< errorMsg << std::endl; \
                                throw std::exception();  \
                        case CUSPARSE_STATUS_INVALID_VALUE: \
                                std::cout << "CuSparse error:" <<"CUSPARSE_STATUS_INVALID_VALUE " << status <<" "<< errorMsg << std::endl; \
                                throw std::exception();  \
                        case CUSPARSE_STATUS_ARCH_MISMATCH: \
                                std::cout << "CuSparse error:" <<"CUSPARSE_STATUS_ARCH_MISMATCH " << status <<" "<< errorMsg << std::endl; \
                                throw std::exception();  \
                        case CUSPARSE_STATUS_MAPPING_ERROR: \
                                std::cout << "CuSparse error:" <<"CUSPARSE_STATUS_MAPPING_ERROR " << status <<" "<< errorMsg << std::endl; \
                                throw std::exception();  \
                        case CUSPARSE_STATUS_EXECUTION_FAILED: \
                                std::cout << "CuSparse error:" <<"CUSPARSE_STATUS_EXECUTION_FAILED " << status <<" "<< errorMsg << std::endl; \
                                throw std::exception();  \
                        case CUSPARSE_STATUS_INTERNAL_ERROR: \
                                std::cout << "CuSparse error:" <<"CUSPARSE_STATUS_INTERNAL_ERROR " << status <<" "<< errorMsg << std::endl; \
                                throw std::exception();  \
                        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: \
                                std::cout << "CuSparse error:" <<"CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED " << status <<" "<< errorMsg << std::endl; \
                                throw std::exception();  \
                        case CUSPARSE_STATUS_ZERO_PIVOT: \
                                std::cout << "CuSparse error:" <<"CUSPARSE_STATUS_ZERO_PIVOT " << status <<" "<< errorMsg << std::endl; \
                                throw std::exception();  \
                        } \
                } \
}


// transfer constants
    #define H2D (cudaMemcpyHostToDevice)
    #define D2H (cudaMemcpyDeviceToHost)
    #define H2H (cudaMemcpyHostToHost)
    #define D2D (cudaMemcpyDeviceToDevice)
#endif


// safe division
#define SDIV(x,y)(((x)+(y)-1)/(y))

// cross platform classifiers
#ifdef __CUDACC__
    #define HOSTDEVICEQUALIFIER  __host__ __device__
#else
    #define HOSTDEVICEQUALIFIER
#endif

#ifdef __CUDACC__
    #define INLINEQUALIFIER  __forceinline__
#else
    #define INLINEQUALIFIER inline
#endif

#ifdef __CUDACC__
    #define GLOBALQUALIFIER  __global__
#else
    #define GLOBALQUALIFIER
#endif

#ifdef __CUDACC__
    #define DEVICEQUALIFIER  __device__
#else
    #define DEVICEQUALIFIER
#endif

#ifdef __CUDACC__
    #define HOSTQUALIFIER  __host__
#else
    #define HOSTQUALIFIER
#endif
