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

//tile size point of control
#define TILE_SIZE 512
#define THREADS 64

//honestly this is just here to remember
#define ALIGNMENT 32

//context struct for the entire program to simplify logic
typedef struct
{
    const char *strA;
    const char *strB;
    int lenA;
    int lenB;

    int num_tiles_row;
    int num_tiles_col;

    //boundaries are int32_t because I got a segfault and changed a bunch of stuff
    int32_t *h_boundaries;
    int32_t *v_boundaries;
    int32_t *corners;

    pthread_barrier_t barrier;
    atomic_int tile_job_index;
    int current_wave_k;
    int num_threads;
} GlobalContext;

//simple struct for threads
typedef struct
{
    int thread_id;
    GlobalContext *ctx;
} ThreadArgs;


//custom aligned allocation, which is required for SIMD AVX2 shenanigans.
void *aligned_malloc(size_t size)
{
    void *ptr;
    if (posix_memalign(&ptr, ALIGNMENT, size) != 0)
        return NULL;
    memset(ptr, 0, size);
    return ptr;
}

//scalar implementation of computing a tile, mostly used for benchmarking
void compute_tile_kernel(
    const char *subA,
    const char *subB,
    const int32_t *top_vals,
    const int32_t *left_vals,
    int32_t top_left_val,
    int32_t *out_bottom,
    int32_t *out_right)
{
    int32_t p1[TILE_SIZE + 1];
    int32_t curr[TILE_SIZE + 1];

    p1[0] = top_left_val;
    memcpy(&p1[1], top_vals, TILE_SIZE * sizeof(int32_t));

    for (int r = 0; r < TILE_SIZE; r++)
    {
        char char_a = subA[r];
        curr[0] = left_vals[r];

        for (int c = 0; c < TILE_SIZE; c++)
        {
            char char_b = subB[c];
            int cost = (char_a == char_b) ? 0 : 1;

            int ins = p1[c + 1] + 1;
            int del = curr[c] + 1;
            int sub = p1[c] + cost;

            int min = (ins < del) ? ins : del;
            curr[c + 1] = (sub < min) ? sub : min;
        }

        out_right[r] = curr[TILE_SIZE];
        memcpy(p1, curr, (TILE_SIZE + 1) * sizeof(int32_t));
    }
    memcpy(out_bottom, &p1[1], TILE_SIZE * sizeof(int32_t));
}

//proper tile calculations using AVX2 calculations (which mostly just prerenders stuff). Check docs for explanation
void compute_tile_AVX2(
    const char *subA,
    const char *subB,
    const int32_t *top_vals,
    const int32_t *left_vals,
    int32_t top_left_val,
    int32_t *out_bottom,
    int32_t *out_right)
{
    int32_t __attribute__((aligned(32))) prev_row[TILE_SIZE + 1];
    int32_t __attribute__((aligned(32))) curr_row[TILE_SIZE + 1];
    
    __m256i ones = _mm256_set1_epi32(1);
    
    prev_row[0] = top_left_val;
    memcpy(&prev_row[1], top_vals, TILE_SIZE * sizeof(int32_t));

    for (int r = 0; r < TILE_SIZE; r++) {
        char a = subA[r];
        curr_row[0] = left_vals[r];
        
        for (int c = 0; c < TILE_SIZE; c += 8) {
            // Vectorized character comparison and cost calculation
            __m128i subB_chars_128 = _mm_loadl_epi64((const __m128i*)&subB[c]);
            __m256i subB_chars = _mm256_cvtepi8_epi32(subB_chars_128);
            __m256i char_a_32 = _mm256_set1_epi32((int32_t)a);
            __m256i eq_mask = _mm256_cmpeq_epi32(char_a_32, subB_chars);
            __m256i cost = _mm256_andnot_si256(eq_mask, ones);
            
            // Load independent values
            __m256i prev_diag = _mm256_loadu_si256((const __m256i*)&prev_row[c]);
            __m256i prev_top = _mm256_loadu_si256((const __m256i*)&prev_row[c + 1]);
            
            // Compute substitution and insertion costs (fully parallel)
            __m256i sub = _mm256_add_epi32(prev_diag, cost);
            __m256i ins = _mm256_add_epi32(prev_top, ones);
            
            // Store temporarily for scalar processing
            int32_t sub_arr[8], ins_arr[8];
            _mm256_storeu_si256((__m256i*)sub_arr, sub);
            _mm256_storeu_si256((__m256i*)ins_arr, ins);
            
            // Sequential computation with dependency chain
            for (int i = 0; i < 8; i++) {
                int32_t del = curr_row[c + i] + 1;
                int32_t min_val = (ins_arr[i] < del) ? ins_arr[i] : del;
                curr_row[c + i + 1] = (sub_arr[i] < min_val) ? sub_arr[i] : min_val;
            }
        }
        
        out_right[r] = curr_row[TILE_SIZE];
        memcpy(prev_row, curr_row, (TILE_SIZE + 1) * sizeof(int32_t));
    }
    
    memcpy(out_bottom, &prev_row[1], TILE_SIZE * sizeof(int32_t));
}

//main thread function
void *worker_thread(void *arg)
{
    ThreadArgs *args = (ThreadArgs *)arg;
    GlobalContext *ctx = args->ctx;

    //establishes and sets the specific thread to a cpu core on the machine
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(args->thread_id % THREADS, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    while (1)
    {
        //waits to begin, and checks to see if all waves are finished before working 
        pthread_barrier_wait(&ctx->barrier);
        if (ctx->current_wave_k == -1)
            break;

        while (1)
        {
            int task_idx = atomic_fetch_add(&ctx->tile_job_index, 1);
            //this uses a queue-less method of fetching instructions by taking an atomic number-assigned space in the wave.

            int k = ctx->current_wave_k;
            int min_r = (k < ctx->num_tiles_col) ? 0 : (k - ctx->num_tiles_col + 1);
            int max_r = (k < ctx->num_tiles_row) ? k : (ctx->num_tiles_row - 1);
            int count = max_r - min_r + 1;

            if (task_idx >= count)
                break;

            int r = min_r + task_idx;
            int c = k - r;

            //uses size_t to avoid out-of-bound segfaults for integer overflows
            const char *sA = ctx->strA + ((size_t)r * TILE_SIZE);
            const char *sB = ctx->strB + ((size_t)c * TILE_SIZE);

            int32_t stack_buf_top[TILE_SIZE];
            int32_t *input_top;
            int32_t top_left_corner;

            if (r == 0)
            {
                for (int i = 0; i < TILE_SIZE; i++) {
                    stack_buf_top[i] = (c * TILE_SIZE) + i + 1;
                }
                input_top = stack_buf_top;
                top_left_corner = (c == 0) ? 0 : (c * TILE_SIZE);
            }
            else
            {
                //(size_t) casts to prevent int overflow during multiply
                size_t idx = ((size_t)(r - 1) * ctx->num_tiles_col + c) * TILE_SIZE;
                input_top = &ctx->h_boundaries[idx];

                if (c == 0)
                    top_left_corner = r * TILE_SIZE;
                else
                {
                    size_t c_idx = (size_t)(r - 1) * ctx->num_tiles_col + (c - 1);
                    top_left_corner = ctx->corners[c_idx];
                }
            }

            int32_t stack_buf_left[TILE_SIZE];
            int32_t *input_left;

            if (c == 0)
            {
                for (int i = 0; i < TILE_SIZE; i++)
                    stack_buf_left[i] = (r * TILE_SIZE) + i + 1;
                input_left = stack_buf_left;
            }
            else
            {
                //(size_t) casts
                size_t idx = ((size_t)r * ctx->num_tiles_col + (c - 1)) * TILE_SIZE;
                input_left = &ctx->v_boundaries[idx];
            }

            //outputs
            size_t out_h_idx = ((size_t)r * ctx->num_tiles_col + c) * TILE_SIZE;
            int32_t *output_bottom = &ctx->h_boundaries[out_h_idx];

            size_t out_v_idx = ((size_t)r * ctx->num_tiles_col + c) * TILE_SIZE;
            int32_t *output_right = &ctx->v_boundaries[out_v_idx];

            //compute
            compute_tile_AVX2(sA, sB, input_top, input_left, top_left_corner, output_bottom, output_right);

            //store Corner
            size_t corner_idx = (size_t)r * ctx->num_tiles_col + c;
            ctx->corners[corner_idx] = output_right[TILE_SIZE - 1];
        }

        //ending barrier to ensure synchronization
        pthread_barrier_wait(&ctx->barrier);
    }
    return NULL;
}

int levenshtein_tiled(const char *s1, const char *s2, int len, int num_threads)
{
    GlobalContext ctx;
    ctx.strA = s1;
    ctx.strB = s2;
    ctx.lenA = len;
    ctx.lenB = len;
    ctx.num_threads = num_threads;

    //note: these are using the padding algorithm to ensure a clean tiling
    ctx.num_tiles_row = (len + TILE_SIZE - 1) / TILE_SIZE;
    ctx.num_tiles_col = (len + TILE_SIZE - 1) / TILE_SIZE;

    size_t num_blocks = (size_t)ctx.num_tiles_row * ctx.num_tiles_col;

    //check for Safe Allocation
    //note: calloc takes size_t, so we are good if we cast
    printf("Allocating %.2f GB for boundaries...\n", (double)(num_blocks * TILE_SIZE * 4 * 2) / 1024 / 1024 / 1024);

    ctx.h_boundaries = calloc(num_blocks * TILE_SIZE, sizeof(int32_t));
    ctx.v_boundaries = calloc(num_blocks * TILE_SIZE, sizeof(int32_t));
    ctx.corners = calloc(num_blocks, sizeof(int32_t));

    if (!ctx.h_boundaries || !ctx.v_boundaries || !ctx.corners)
    {
        fprintf(stderr, "Memory Allocation Failed! Reduce string size or increase RAM.\n");
        exit(1);
    }

    pthread_barrier_init(&ctx.barrier, NULL, num_threads + 1);
    atomic_init(&ctx.tile_job_index, 0);

    pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);
    ThreadArgs *t_args = malloc(sizeof(ThreadArgs) * num_threads);

    for (int i = 0; i < num_threads; i++)
    {
        t_args[i].thread_id = i;
        t_args[i].ctx = &ctx;
        pthread_create(&threads[i], NULL, worker_thread, &t_args[i]);
    }

    int max_wave = ctx.num_tiles_row + ctx.num_tiles_col - 2;

    for (int k = 0; k <= max_wave; k++)
    {
        ctx.current_wave_k = k;
        atomic_store(&ctx.tile_job_index, 0);
        pthread_barrier_wait(&ctx.barrier);
        pthread_barrier_wait(&ctx.barrier);
    }

    ctx.current_wave_k = -1;
    pthread_barrier_wait(&ctx.barrier);

    for (int i = 0; i < num_threads; i++)
        pthread_join(threads[i], NULL);

    int result = ctx.corners[num_blocks - 1];

    free(ctx.h_boundaries);
    free(ctx.v_boundaries);
    free(ctx.corners);
    free(threads);
    free(t_args);
    pthread_barrier_destroy(&ctx.barrier);

    return result;
}

int cse2421_edit_distance(const char *str1, const char *str2, size_t len)
{


    // pads out the strings to a clean multiple of the tile size to make handing boundaries simpler with minimal performance consequences
    size_t padded_len = ((len + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

    printf("Length: %zu (Padded: %zu), Threads: %d\n", len, padded_len, THREADS);

    // standardized lengths to be susceptible to complete tiling over the strings' array
    char *str1_padded = aligned_malloc((size_t)padded_len + 1);
    char *str2_padded = aligned_malloc((size_t)padded_len + 1);

    if (!str1_padded || !str2_padded)
    {
        fprintf(stderr, "Failed to allocate strings.\n");
        return 1;
    }

    strcpy(str1_padded, str1);
    strcpy(str2_padded, str2);
    for (size_t i = len; i < padded_len; i++)
    {
        str1_padded[i] = 'A';
        str2_padded[i] = 'A';
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int dist = levenshtein_tiled(str1_padded, str2_padded, padded_len, THREADS);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double t = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Dist: %d\n", dist);
    printf("Time: %.4f s\n", t);
    printf("MCPS: %.2f\n", (double)len * len / 1e6 / t);

    return dist;
}
