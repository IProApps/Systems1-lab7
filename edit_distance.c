#include "edit_distance.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MIN3(a,b,c) MIN(MIN(a,b),c)
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define TILE_SIZE 128
#define NUM_THREADS 8

typedef struct {
    const char *str1;
    const char *str2;
    size_t len;
    int **dp;
    int num_tiles;
    volatile int *tile_ready;
    volatile int *next_tile_in_wave;
    pthread_mutex_t wave_lock;
} shared_data_t;

// Compute a single tile
void compute_tile(shared_data_t *shared, int tile_i, int tile_j) {
    size_t start_i = tile_i * TILE_SIZE;
    size_t start_j = tile_j * TILE_SIZE;
    size_t end_i = MIN(start_i + TILE_SIZE, shared->len + 1);
    size_t end_j = MIN(start_j + TILE_SIZE, shared->len + 1);
    
    for (size_t i = start_i; i < end_i; i++) {
        for (size_t j = start_j; j < end_j; j++) {
            if (i == 0) {
                shared->dp[i][j] = j;
            } else if (j == 0) {
                shared->dp[i][j] = i;
            } else {
                int cost = (shared->str1[i - 1] != shared->str2[j - 1]) ? 1 : 0;
                shared->dp[i][j] = MIN3(
                    shared->dp[i - 1][j - 1] + cost,
                    shared->dp[i - 1][j] + 1,
                    shared->dp[i][j - 1] + 1
                );
            }
        }
    }
}

void *worker_thread(void *arg) {
    shared_data_t *shared = (shared_data_t *)arg;
    int num_wavefronts = 2 * shared->num_tiles - 1;
    
    // Process each wavefront
    for (int wave = 0; wave < num_wavefronts; wave++) {
        // Calculate wavefront parameters
        int start_tile_i = MAX(0, wave - (shared->num_tiles - 1));
        int end_tile_i = MIN(wave, shared->num_tiles - 1);
        int tiles_in_wave = end_tile_i - start_tile_i + 1;
        
        while (1) {
            // Atomically claim a tile from this wavefront
            pthread_mutex_lock(&shared->wave_lock);
            int local_idx = shared->next_tile_in_wave[wave];
            if (local_idx >= tiles_in_wave) {
                pthread_mutex_unlock(&shared->wave_lock);
                break;
            }
            shared->next_tile_in_wave[wave] = local_idx + 1;
            pthread_mutex_unlock(&shared->wave_lock);
            
            // Calculate tile coordinates
            int tile_i = start_tile_i + local_idx;
            int tile_j = wave - tile_i;
            
            // Wait for dependencies (tile above and tile to the left)
            if (tile_i > 0) {
                int dep_idx = (tile_i - 1) * shared->num_tiles + tile_j;
                while (!__atomic_load_n(&shared->tile_ready[dep_idx], __ATOMIC_ACQUIRE)) {
                    // Spin wait
                }
            }
            
            if (tile_j > 0) {
                int dep_idx = tile_i * shared->num_tiles + (tile_j - 1);
                while (!__atomic_load_n(&shared->tile_ready[dep_idx], __ATOMIC_ACQUIRE)) {
                    // Spin wait
                }
            }
            
            // Compute the tile
            compute_tile(shared, tile_i, tile_j);
            
            // Mark this tile as complete
            int my_idx = tile_i * shared->num_tiles + tile_j;
            __atomic_store_n(&shared->tile_ready[my_idx], 1, __ATOMIC_RELEASE);
        }
    }
    
    return NULL;
}

int cse2421_edit_distance(const char *str1, const char *str2, size_t len) {
    if (len == 0) return 0;
    
    // For small inputs, use simple two-row algorithm
    if (len < 512) {
        int *prev = malloc((len + 1) * sizeof(int));
        int *curr = malloc((len + 1) * sizeof(int));
        
        for (size_t i = 0; i <= len; i++) {
            prev[i] = i;
        }
        
        for (size_t i = 1; i <= len; i++) {
            curr[0] = i;
            for (size_t j = 1; j <= len; j++) {
                int cost = (str1[i - 1] != str2[j - 1]) ? 1 : 0;
                curr[j] = MIN3(prev[j - 1] + cost, prev[j] + 1, curr[j - 1] + 1);
            }
            int *tmp = prev;
            prev = curr;
            curr = tmp;
        }
        
        int result = prev[len];
        free(prev);
        free(curr);
        return result;
    }
    
    // Allocate full DP table for tiled parallel computation
    int **dp = malloc((len + 1) * sizeof(int *));
    for (size_t i = 0; i <= len; i++) {
        dp[i] = malloc((len + 1) * sizeof(int));
    }
    
    // Calculate number of tiles
    int num_tiles = (len + TILE_SIZE) / TILE_SIZE;
    
    // Setup shared data structure
    shared_data_t shared;
    shared.str1 = str1;
    shared.str2 = str2;
    shared.len = len;
    shared.dp = dp;
    shared.num_tiles = num_tiles;
    
    // Allocate tile synchronization arrays
    int total_tiles = num_tiles * num_tiles;
    shared.tile_ready = calloc(total_tiles, sizeof(int));
    
    int num_wavefronts = 2 * num_tiles - 1;
    shared.next_tile_in_wave = calloc(num_wavefronts, sizeof(int));
    
    pthread_mutex_init(&shared.wave_lock, NULL);
    
    // Create worker threads
    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, worker_thread, &shared);
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Extract result
    int result = dp[len][len];
    
    // Cleanup
    pthread_mutex_destroy(&shared.wave_lock);
    free((void *)shared.tile_ready);
    free((void *)shared.next_tile_in_wave);
    
    for (size_t i = 0; i <= len; i++) {
        free(dp[i]);
    }
    free(dp);
    
    return result;
}