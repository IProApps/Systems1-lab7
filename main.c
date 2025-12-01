#include "edit_distance.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void generate_random_string(char *str, size_t len) {
    for (size_t i = 0; i < len; i++) {
        str[i] = 'A' + (rand() % 26);
    }
    str[len] = '\0';
}

void benchmark_edit_distance(size_t len) {
    printf("\n========================================\n");
    printf("Benchmarking with string length: %zu\n", len);
    printf("========================================\n");
    
    char *str1 = malloc(len + 1);
    char *str2 = malloc(len + 1);
    
    if (!str1 || !str2) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }
    
    srand(time(NULL));
    generate_random_string(str1, len);
    generate_random_string(str2, len);
    
    printf("Generated random strings of length %zu\n", len);
    printf("Sample (first 50 chars of each):\n");
    printf("str1: %.50s...\n", str1);
    printf("str2: %.50s...\n", str2);
    printf("\n");
    
    printf("Computing edit distance...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    int distance = cse2421_edit_distance(str1, str2, len);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + 
                     (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    printf("\nResults:\n");
    printf("  Edit distance: %d\n", distance);
    printf("  Execution time: %.6f seconds\n", elapsed);
    printf("  Throughput: %.2f million cell updates/sec\n", 
           (double)(len * len) / (elapsed * 1000000.0));
    
    free(str1);
    free(str2);
}

int main(int argc, char *argv[]) {
    printf("Edit Distance Benchmark\n");
    printf("=======================\n\n");
    
    printf("Configuration:\n");
    printf("  Number of threads: 8\n");
    printf("  Tile size: 256\n");
    printf("  Parallelization: Wavefront tiling with pthreads\n");
    printf("  Vectorization: AVX2 intrinsics\n\n");
    
    // Test with different sizes
    size_t test_sizes[] = {100, 1000, 10000};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int i = 0; i < num_tests; i++) {
        benchmark_edit_distance(test_sizes[i]);
    }
    
    // Main benchmark with length 1,000,000
    printf("\n\n");
    printf("========================================\n");
    printf("MAIN BENCHMARK: len = 1,000,000\n");
    printf("========================================\n");
    benchmark_edit_distance(1000000);
    
    printf("\n\nBenchmark complete!\n");
    return 0;
}