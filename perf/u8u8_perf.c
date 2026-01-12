#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define N (1000 * 1000) /* adjust if you want longer runs */
#define DIM 512
#define REPEATS 10

/* Scalar fallback */
static int32_t dot_u8_scalar(const uint8_t* a, const uint8_t* b) {
    int32_t sum = 0;
    for (int i = 0; i < DIM; i++) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }
    return sum;
}

#if defined(__AVX2__)
#include <immintrin.h>
/* 16-byte chunks, AVX2 widening mul */
static int32_t dot_u8_avx2(const uint8_t* a, const uint8_t* b) {
    __m256i acc32 = _mm256_setzero_si256();
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int i = 0; i < DIM; i += 16) {
        __m128i va8 = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i vb8 = _mm_loadu_si128((const __m128i*)(b + i));
        __m256i va16 = _mm256_cvtepu8_epi16(va8);
        __m256i vb16 = _mm256_cvtepu8_epi16(vb8);
        __m256i prod16 = _mm256_mullo_epi16(va16, vb16);
        __m256i sum32 = _mm256_madd_epi16(prod16, ones16);
        acc32 = _mm256_add_epi32(acc32, sum32);
    }
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(acc32),
                                   _mm256_extracti128_si256(acc32, 1));
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return _mm_cvtsi128_si32(sum128);
}
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
static int32_t dot_u8_neon(const uint8_t* a, const uint8_t* b) {
    uint64x2_t acc64 = vdupq_n_u64(0);
    for (int i = 0; i < DIM; i += 16) {
        uint8x16_t va = vld1q_u8(a + i);
        uint8x16_t vb = vld1q_u8(b + i);
        uint16x8_t prod0 = vmull_u8(vget_low_u8(va), vget_low_u8(vb));
        uint16x8_t prod1 = vmull_u8(vget_high_u8(va), vget_high_u8(vb));
        uint32x4_t sum0 = vpaddlq_u16(prod0);
        uint32x4_t sum1 = vpaddlq_u16(prod1);
        acc64 = vaddq_u64(acc64, vpaddlq_u32(sum0));
        acc64 = vaddq_u64(acc64, vpaddlq_u32(sum1));
    }
    return (int32_t)(vgetq_lane_u64(acc64, 0) + vgetq_lane_u64(acc64, 1));
}
#endif

typedef int32_t (*dot_fn)(const uint8_t*, const uint8_t*);

static double bench(const char* name, dot_fn fn) {
    static volatile uint8_t a[DIM];
    static volatile uint8_t b[DIM];
    for (int i = 0; i < DIM; i++) {
        a[i] = (uint8_t)((i % 13) + 1);
        b[i] = (uint8_t)((DIM - i) % 17 + 1);
    }
    int32_t total = 0;
    double sum_ops = 0.0;

    for (int r = 0; r < REPEATS; r++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (long i = 0; i < N; i++) {
            total += fn((const uint8_t*)a, (const uint8_t*)b);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        sum_ops += N / dt;
    }

    double avg_ops = sum_ops / REPEATS;
    printf("%-12s: %.2f million ops/s (avg over %d, total=%d)\n", name, avg_ops / 1e6, REPEATS, total);
    return avg_ops;
}

int main(void) {
    bench("scalar", dot_u8_scalar);

#if defined(__AVX2__)
    bench("avx2", dot_u8_avx2);
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    bench("neon", dot_u8_neon);
#endif

    return 0;
}