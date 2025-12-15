#ifndef NATIVE_CHECK_H
#define NATIVE_CHECK_H

#include <immintrin.h>

#if (defined(__x86_64__) && defined(__AVX__))
#define SIMD_MAX_CAPACITY 256  // __AVX__ 的 SIMD 最大并行计算容量为 256 位

#elif (defined(__aarch64__) && defined(__ARM_NEON))

#define SIMD_MAX_CAPACITY 128  // __ARM_NEON 的 SIMD 最大并行计算容量为 128 位

#else

// 若不支持 SIMD 指令集，则不定义 SIMD_MAX_CAPACITY，使用普通 C 语言实现

#endif


#ifdef SIMD_MAX_CAPACITY
size_t GetOneBitCodeSimdBlockNum(size_t dim) {
    return (dim + SIMD_MAX_CAPACITY - 1) / SIMD_MAX_CAPACITY;   // 向上取整
}
size_t GetBytesPerSimdBlock() {
    return SIMD_MAX_CAPACITY / 8;    // 32 或 16
}
#endif

#endif  // NATIVE_CHECK_H