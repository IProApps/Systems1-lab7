#include "edit_distance.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MIN3(a,b,c) MIN(MIN(a,b),c)
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define TILE_SIZE 256
#define NUM_THREADS 16

typedef struct {
    const char *str1;
    const char *str2;
    size_t len;
    int **diagonals;
    int max_diag_stored;
    int num_tiles;
    volatile int *tile_ready;
    volatile int *next_wavefront_tile;
    pthread_mutex_t *wavefront_lock;
} shared_data_t;

// Simplified diagonal access
static inline int get_diag(int **diag, int i, int j, int max_stored) {
    int d = (i + j) % max_stored;
    return diag[d][MIN(i, j)];
}

static inline void set_diag(int **diag, int i, int j, int val, int max_stored) {
    int d = (i + j) % max_stored;
    diag[d][MIN(i, j)] = val;
}

// True AVX2 SIMD minimum
static inline __m256i min3_avx2(__m256i a, __m256i b, __m256i c) {
    return _mm256_min_epi32(_mm256_min_epi32(a, b), c);
}

void compute_tile_avx2(shared_data_t *shared, int tile_i, int tile_j) {
    int si = tile_i * TILE_SIZE;
    int sj = tile_j * TILE_SIZE;
    int ei = MIN(si + TILE_SIZE, shared->len + 1);
    int ej = MIN(sj + TILE_SIZE, shared->len + 1);
    
    // Process anti-diagonals within tile
    for (int d = si + sj; d < ei + ej - 1; d++) {
        int start_i = MAX(si, d - ej + 1);
        int end_i = MIN(ei - 1, d - sj);
        
        int i = start_i;
        
        // Handle boundaries
        while (i <= end_i && (i == 0 || (d - i) == 0)) {
            int j = d - i;
            if (j >= sj && j < ej) {
                set_diag(shared->diagonals, i, j, (i == 0) ? j : i, shared->max_diag_stored);
            }
            i++;
        }
        
        // AVX2 vectorized interior - 8 cells at once
        for (; i + 7 <= end_i; i += 8) {
            int costs[8] __attribute__((aligned(32)));
            int diag_vals[8] __attribute__((aligned(32)));
            int top_vals[8] __attribute__((aligned(32)));
            int left_vals[8] __attribute__((aligned(32)));
            int results[8] __attribute__((aligned(32)));
            int valid[8];
            
            // Gather phase
            for (int k = 0; k < 8; k++) {
                int ii = i + k;
                int jj = d - ii;
                
                if (jj < sj || jj >= ej || ii >= ei) {
                    valid[k] = 0;
                    costs[k] = diag_vals[k] = top_vals[k] = left_vals[k] = 0;
                    continue;
                }
                
                valid[k] = 1;
                costs[k] = (shared->str1[ii - 1] != shared->str2[jj - 1]) ? 1 : 0;
                diag_vals[k] = get_diag(shared->diagonals, ii - 1, jj - 1, shared->max_diag_stored);
                top_vals[k] = get_diag(shared->diagonals, ii - 1, jj, shared->max_diag_stored);
                left_vals[k] = get_diag(shared->diagonals, ii, jj - 1, shared->max_diag_stored);
            }
            
            // SIMD computation
            __m256i cost_vec = _mm256_load_si256((__m256i*)costs);
            __m256i diag_vec = _mm256_load_si256((__m256i*)diag_vals);
            __m256i top_vec = _mm256_load_si256((__m256i*)top_vals);
            __m256i left_vec = _mm256_load_si256((__m256i*)left_vals);
            __m256i ones = _mm256_set1_epi32(1);
            
            __m256i diag_cost = _mm256_add_epi32(diag_vec, cost_vec);
            __m256i top_plus1 = _mm256_add_epi32(top_vec, ones);
            __m256i left_plus1 = _mm256_add_epi32(left_vec, ones);
            
            __m256i result_vec = min3_avx2(diag_cost, top_plus1, left_plus1);
            _mm256_store_si256((__m256i*)results, result_vec);
            
            // Scatter phase
            for (int k = 0; k < 8; k++) {
                if (valid[k]) {
                    int ii = i + k;
                    int jj = d - ii;
                    set_diag(shared->diagonals, ii, jj, results[k], shared->max_diag_stored);
                }
            }
        }
        
        // Remaining cells
        for (; i <= end_i; i++) {
            int j = d - i;
            if (j < sj || j >= ej) continue;
            
            int cost = (shared->str1[i - 1] != shared->str2[j - 1]) ? 1 : 0;
            int diag = get_diag(shared->diagonals, i - 1, j - 1, shared->max_diag_stored);
            int top = get_diag(shared->diagonals, i - 1, j, shared->max_diag_stored);
            int left = get_diag(shared->diagonals, i, j - 1, shared->max_diag_stored);
            
            set_diag(shared->diagonals, i, j, MIN3(diag + cost, top + 1, left + 1), shared->max_diag_stored);
        }
    }
}

void *worker_thread(void *arg) {
    shared_data_t *shared = (shared_data_t *)arg;
    int num_wavefronts = 2 * shared->num_tiles - 1;
    
    // Process wavefronts
    for (int wave = 0; wave < num_wavefronts; wave++) {
        while (1) {
            // Try to claim a tile from this wavefront
            pthread_mutex_lock(shared->wavefront_lock);
            int tile_idx = __atomic_load_n(&shared->next_wavefront_tile[wave], __ATOMIC_ACQUIRE);
            
            // Calculate how many tiles are in this wavefront
            int wave_start_i = MAX(0, wave - (shared->num_tiles - 1));
            int wave_end_i = MIN(wave, shared->num_tiles - 1);
            int tiles_in_wave = wave_end_i - wave_start_i + 1;
            
            if (tile_idx >= tiles_in_wave) {
                pthread_mutex_unlock(shared->wavefront_lock);
                break; // No more tiles in this wavefront
            }
            
            __atomic_store_n(&shared->next_wavefront_tile[wave], tile_idx + 1, __ATOMIC_RELEASE);
            pthread_mutex_unlock(shared->wavefront_lock);
            
            // Calculate actual tile coordinates
            int tile_i = wave_start_i + tile_idx;
            int tile_j = wave - tile_i;
            
            // Wait for dependencies
            if (tile_i > 0) {
                int dep_idx = (tile_i - 1) * shared->num_tiles + tile_j;
                while (!__atomic_load_n(&shared->tile_ready[dep_idx], __ATOMIC_ACQUIRE)) {
                    __builtin_ia32_pause();
                }
            }
            
            if (tile_j > 0) {
                int dep_idx = tile_i * shared->num_tiles + (tile_j - 1);
                while (!__atomic_load_n(&shared->tile_ready[dep_idx], __ATOMIC_ACQUIRE)) {
                    __builtin_ia32_pause();
                }
            }
            
            // Compute tile
            compute_tile_avx2(shared, tile_i, tile_j);
            
            // Mark done
            int my_idx = tile_i * shared->num_tiles + tile_j;
            __atomic_store_n(&shared->tile_ready[my_idx], 1, __ATOMIC_RELEASE);
        }
    }
    
    return NULL;
}

int cse2421_edit_distance(const char *str1, const char *str2, size_t len) {
    if (len == 0) return 0;
    
    // Small input optimization
    if (len < 1000) {
        int *prev = malloc((len + 1) * sizeof(int));
        int *curr = malloc((len + 1) * sizeof(int));
        
        for (size_t i = 0; i <= len; i++) prev[i] = i;
        
        for (size_t i = 1; i <= len; i++) {
            curr[0] = i;
            for (size_t j = 1; j <= len; j++) {
                int cost = (str1[i - 1] != str2[j - 1]);
                curr[j] = MIN3(prev[j - 1] + cost, prev[j] + 1, curr[j - 1] + 1);
            }
            int *tmp = prev; prev = curr; curr = tmp;
        }
        
        int res = prev[len];
        free(prev);
        free(curr);
        return res;
    }
    
    int num_tiles = (len + TILE_SIZE - 1) / TILE_SIZE;
    int max_stored = 2 * TILE_SIZE + 2;
    
    // Allocate diagonal storage
    int **diagonals = malloc(max_stored * sizeof(int *));
    for (int d = 0; d < max_stored; d++) {
        diagonals[d] = aligned_alloc(32, (len + 1) * sizeof(int));
        memset(diagonals[d], 0, (len + 1) * sizeof(int));
    }
    
    // Initialize boundaries
    for (int d = 0; d <= len && d < max_stored; d++) {
        for (int i = 0; i <= d && i <= len; i++) {
            int j = d - i;
            if (j <= len) {
                set_diag(diagonals, i, j, (i == 0) ? j : (j == 0) ? i : 0, max_stored);
            }
        }
    }
    
    // Setup shared data
    shared_data_t shared;
    shared.str1 = str1;
    shared.str2 = str2;
    shared.len = len;
    shared.diagonals = diagonals;
    shared.max_diag_stored = max_stored;
    shared.num_tiles = num_tiles;
    shared.tile_ready = calloc(num_tiles * num_tiles, sizeof(int));
    
    // Wavefront coordination
    int num_wavefronts = 2 * num_tiles - 1;
    shared.next_wavefront_tile = calloc(num_wavefronts, sizeof(int));
    shared.wavefront_lock = malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(shared.wavefront_lock, NULL);
    
    // Create worker threads
    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, worker_thread, &shared);
    }
    
    // Wait for completion
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Extract result
    int result = get_diag(diagonals, len, len, max_stored);
    
    // Cleanup
    pthread_mutex_destroy(shared.wavefront_lock);
    free(shared.wavefront_lock);
    free((void *)shared.next_wavefront_tile);
    free((void *)shared.tile_ready);
    
    for (int d = 0; d < max_stored; d++) {
        free(diagonals[d]);
    }
    free(diagonals);
    
    return result;
}