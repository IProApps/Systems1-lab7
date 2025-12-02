#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h>
#include <stdatomic.h>
#include <sched.h>
#include "edit_distance.h"

// --- Tuning Parameters ---
#define TILE_SIZE 256 
#define ALIGNMENT 32

// --- Data Structures ---

typedef struct {
    const char *strA;
    const char *strB;
    int lenA;
    int lenB;
    
    int num_tiles_row;
    int num_tiles_col;

    // CHANGED: Use int32_t pointers, but we will index them with size_t
    int32_t *h_boundaries; 
    int32_t *v_boundaries; 
    int32_t *corners;      

    pthread_barrier_t barrier;
    atomic_int tile_job_index; 
    int current_wave_k; 
    int num_threads;
} GlobalContext;

typedef struct {
    int thread_id;
    GlobalContext *ctx;
} ThreadArgs;


/*
custom aligned allocation, which is required for SIMD AVX2 shenanigans. 
Done in this way to ensure it will run on both linux and windows, since there are different calls for different systems for this method. 
*/
void* aligned_malloc(size_t size) {
    void *ptr;
    if (posix_memalign(&ptr, ALIGNMENT, size) != 0) return NULL;
    memset(ptr, 0, size);
    return ptr;
}

// --- Tile Kernel (Optimized Scalar) ---
void compute_tile_kernel(
    const char *subA, 
    const char *subB,
    const int32_t *top_vals,
    const int32_t *left_vals,
    int32_t top_left_val,
    int32_t *out_bottom,
    int32_t *out_right
) {
    int32_t p1[TILE_SIZE + 1];   
    int32_t curr[TILE_SIZE + 1]; 
    
    p1[0] = top_left_val;
    memcpy(&p1[1], top_vals, TILE_SIZE * sizeof(int32_t));
    
    for(int r = 0; r < TILE_SIZE; r++) {
        char char_a = subA[r];
        curr[0] = left_vals[r];
        
        for(int c = 0; c < TILE_SIZE; c++) {
            char char_b = subB[c];
            int cost = (char_a == char_b) ? 0 : 1;
            
            int ins = p1[c+1] + 1; 
            int del = curr[c] + 1; 
            int sub = p1[c] + cost;
            
            int min = (ins < del) ? ins : del;
            curr[c+1] = (sub < min) ? sub : min;
        }
        
        out_right[r] = curr[TILE_SIZE];
        memcpy(p1, curr, (TILE_SIZE + 1) * sizeof(int32_t));
    }
    memcpy(out_bottom, &p1[1], TILE_SIZE * sizeof(int32_t));
}

// --- Thread Worker ---
void* worker_thread(void *arg) {
    ThreadArgs *args = (ThreadArgs*)arg;
    GlobalContext *ctx = args->ctx;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(args->thread_id % 64, &cpuset); 
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    while (1) {
        int rc = pthread_barrier_wait(&ctx->barrier);
        if (ctx->current_wave_k == -1) break;

        while (1) {
            int task_idx = atomic_fetch_add(&ctx->tile_job_index, 1);
            
            int k = ctx->current_wave_k;
            int min_r = (k < ctx->num_tiles_col) ? 0 : (k - ctx->num_tiles_col + 1);
            int max_r = (k < ctx->num_tiles_row) ? k : (ctx->num_tiles_row - 1);
            int count = max_r - min_r + 1;
            
            if (task_idx >= count) break; 

            int r = min_r + task_idx;
            int c = k - r;
            
            // --- INDEXING FIX: Use size_t and force 64-bit math ---
            
            // 1. Inputs
            // String offsets (size_t to be safe)
            const char *sA = ctx->strA + ((size_t)r * TILE_SIZE);
            const char *sB = ctx->strB + ((size_t)c * TILE_SIZE);
            
            int32_t stack_buf_top[TILE_SIZE]; 
            int32_t *input_top;
            int32_t top_left_corner;
            
            if (r == 0) {
                for(int i=0; i<TILE_SIZE; i++) stack_buf_top[i] = (c * TILE_SIZE) + i + 1;
                input_top = stack_buf_top;
                top_left_corner = (c == 0) ? 0 : (c * TILE_SIZE);
            } else {
                // CHANGED: (size_t) casts to prevent int overflow during multiply
                size_t idx = ((size_t)(r - 1) * ctx->num_tiles_col + c) * TILE_SIZE;
                input_top = &ctx->h_boundaries[idx];
                
                if (c == 0) top_left_corner = r * TILE_SIZE;
                else {
                    size_t c_idx = (size_t)(r-1) * ctx->num_tiles_col + (c-1);
                    top_left_corner = ctx->corners[c_idx];
                }
            }

            int32_t stack_buf_left[TILE_SIZE];
            int32_t *input_left;
            
            if (c == 0) {
                for(int i=0; i<TILE_SIZE; i++) stack_buf_left[i] = (r * TILE_SIZE) + i + 1;
                input_left = stack_buf_left;
            } else {
                // CHANGED: (size_t) casts
                size_t idx = ((size_t)r * ctx->num_tiles_col + (c-1)) * TILE_SIZE;
                input_left = &ctx->v_boundaries[idx];
            }
            
            // 2. Outputs
            // CHANGED: (size_t) casts
            size_t out_h_idx = ((size_t)r * ctx->num_tiles_col + c) * TILE_SIZE;
            int32_t *output_bottom = &ctx->h_boundaries[out_h_idx];
            
            size_t out_v_idx = ((size_t)r * ctx->num_tiles_col + c) * TILE_SIZE;
            int32_t *output_right = &ctx->v_boundaries[out_v_idx];
            
            // 3. Compute
            compute_tile_kernel(sA, sB, input_top, input_left, top_left_corner, output_bottom, output_right);
            
            // 4. Store Corner
            size_t corner_idx = (size_t)r * ctx->num_tiles_col + c;
            ctx->corners[corner_idx] = output_right[TILE_SIZE - 1]; 
        }

        pthread_barrier_wait(&ctx->barrier);
    }
    return NULL;
}

int levenshtein_tiled(const char *s1, const char *s2, int len, int num_threads) {
    GlobalContext ctx;
    ctx.strA = s1;
    ctx.strB = s2;
    ctx.lenA = len;
    ctx.lenB = len;
    ctx.num_threads = num_threads;
    
    ctx.num_tiles_row = (len + TILE_SIZE - 1) / TILE_SIZE;
    ctx.num_tiles_col = (len + TILE_SIZE - 1) / TILE_SIZE;
    
    // CHANGED: Use size_t for block counting
    size_t num_blocks = (size_t)ctx.num_tiles_row * ctx.num_tiles_col;
    
    // Check for OOM / Safe Allocation
    // Note: calloc takes size_t, so we are good if we cast
    printf("Allocating %.2f GB for boundaries...\n", (double)(num_blocks * TILE_SIZE * 4 * 2) / 1024 / 1024 / 1024);
    
    ctx.h_boundaries = calloc(num_blocks * TILE_SIZE, sizeof(int32_t));
    ctx.v_boundaries = calloc(num_blocks * TILE_SIZE, sizeof(int32_t));
    ctx.corners      = calloc(num_blocks, sizeof(int32_t));
    
    if (!ctx.h_boundaries || !ctx.v_boundaries || !ctx.corners) {
        fprintf(stderr, "Memory Allocation Failed! Reduce string size or increase RAM.\n");
        exit(1);
    }
    
    pthread_barrier_init(&ctx.barrier, NULL, num_threads + 1);
    atomic_init(&ctx.tile_job_index, 0);
    
    pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);
    ThreadArgs *t_args = malloc(sizeof(ThreadArgs) * num_threads);

    for (int i = 0; i < num_threads; i++) {
        t_args[i].thread_id = i;
        t_args[i].ctx = &ctx;
        pthread_create(&threads[i], NULL, worker_thread, &t_args[i]);
    }

    int max_wave = ctx.num_tiles_row + ctx.num_tiles_col - 2;
    
    for (int k = 0; k <= max_wave; k++) {
        ctx.current_wave_k = k;
        atomic_store(&ctx.tile_job_index, 0);
        pthread_barrier_wait(&ctx.barrier);
        pthread_barrier_wait(&ctx.barrier);
    }
    
    ctx.current_wave_k = -1;
    pthread_barrier_wait(&ctx.barrier); 
    
    for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);

    int result = ctx.corners[num_blocks - 1];
    
    free(ctx.h_boundaries);
    free(ctx.v_boundaries);
    free(ctx.corners);
    free(threads);
    free(t_args);
    pthread_barrier_destroy(&ctx.barrier);
    
    return result;
}

int cse2421_edit_distance(const char* str1, const char *str2, size_t len) {

    //initialize a default amount of threads for the linux box
    int threads = 64;

    //pads out the strings to a clean multiple of the tile size to make handing boundaries simpler with minimal performance consequences
    int padded_len = ((len + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

    printf("Length: %d (Padded: %d), Threads: %d\n", len, padded_len, threads);

    //standardized lengths to be susceptible to complete tiling over the strings' array
    char *str1_padded = aligned_malloc((size_t)padded_len + 1);
    char *str2_padded = aligned_malloc((size_t)padded_len + 1);

    if(!str1_padded || !str2_padded) {
        fprintf(stderr, "Failed to allocate strings.\n");
        return 1;
    }

    strcopy(str1_padded, str1);
    strcopy(str2_padded, str2);
    for(size_t i=len; i<padded_len; i++) { 
        str1_padded[i] = 'A'; 
        str2_padded[i] = 'A'; 
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int dist = levenshtein_tiled(str1_padded, str2_padded, padded_len, threads);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double t = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Dist: %d\n", dist);
    printf("Time: %.4f s\n", t);
    printf("MCPS: %.2f\n", (double)len*len/1e6/t);

    return dist;
}

// int main(int argc, char **argv) {
//     int len = 30000;
//     int threads = 64; 

//     if (argc > 1) len = atoi(argv[1]);
//     if (argc > 2) threads = atoi(argv[2]);

//     int padded_len = ((len + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    
//     printf("Length: %d (Padded: %d), Threads: %d\n", len, padded_len, threads);
    
//     // Allocating strings (might be large)
//     char *s1 = aligned_malloc((size_t)padded_len + 1);
//     char *s2 = aligned_malloc((size_t)padded_len + 1);
    
//     if(!s1 || !s2) {
//         fprintf(stderr, "Failed to allocate strings.\n");
//         return 1;
//     }

//     srand(time(NULL));
//     const char c[] = "QWERTYUIOPASDFGHJKLZXCVBNM";
//     // Fill slightly faster
//     for(size_t i=0; i<len; i++) { s1[i] = c[rand()%26]; s2[i] = c[rand()%26]; }
//     for(size_t i=len; i<padded_len; i++) { s1[i] = 'A'; s2[i] = 'A'; }

//     struct timespec start, end;
//     clock_gettime(CLOCK_MONOTONIC, &start);

//     int dist = levenshtein_tiled(s1, s2, padded_len, threads);

//     clock_gettime(CLOCK_MONOTONIC, &end);
//     double t = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

//     printf("Dist: %d\n", dist);
//     printf("Time: %.4f s\n", t);
//     printf("MCPS: %.2f\n", (double)len*len/1e6/t);

//     return 0;
// }
