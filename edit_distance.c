#include "edit_distance.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>
#include <unistd.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MIN3(a,b,c) MIN(MIN(a,b),c)
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define TILE_SIZE 256
#define NUM_THREADS 4

typedef struct {
    const char *str1;
    const char *str2;
    size_t len;
    int **diagonals;  // Store anti-diagonals
    int max_diag_stored;
    int tile_i;
    int tile_j;
    int tile_size;
    int num_tiles;
    volatile int **tile_done;
    pthread_mutex_t *tile_locks;
    pthread_cond_t *tile_conds;
    pthread_mutex_t diag_lock;
} tile_work_t;

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t notify;
    pthread_t *threads;
    tile_work_t **task_queue;
    int queue_head;
    int queue_tail;
    int queue_size;
    int queue_capacity;
    int thread_count;
    int shutdown;
} threadpool_t;

// Get value from anti-diagonal storage
// Cell (i,j) is on anti-diagonal d = i+j
// Position in that diagonal: min(i,j)
static inline int get_diag_cell(int **diagonals, int i, int j, int len, int max_stored) {
    if (i < 0 || j < 0 || i > len || j > len) return 0;
    
    int diag_num = i + j;
    int diag_stored = diag_num % max_stored;
    
    // Position in anti-diagonal
    int pos = (i <= j) ? i : j;
    
    return diagonals[diag_stored][pos];
}

static inline void set_diag_cell(int **diagonals, int i, int j, int val, int len, int max_stored) {
    int diag_num = i + j;
    int diag_stored = diag_num % max_stored;
    
    int pos = (i <= j) ? i : j;
    
    diagonals[diag_stored][pos] = val;
}

void compute_tile_antidiag(tile_work_t *work) {
    int tile_start_i = work->tile_i * work->tile_size;
    int tile_start_j = work->tile_j * work->tile_size;
    int tile_end_i = MIN(tile_start_i + work->tile_size, work->len + 1);
    int tile_end_j = MIN(tile_start_j + work->tile_size, work->len + 1);
    
    // Process tile by anti-diagonals within the tile
    int min_diag = tile_start_i + tile_start_j;
    int max_diag = (tile_end_i - 1) + (tile_end_j - 1);
    
    pthread_mutex_lock(&work->diag_lock);
    
    for (int d = min_diag; d <= max_diag; d++) {
        // For diagonal d, find cells (i,j) where i+j=d and both are in tile bounds
        int start_i = MAX(tile_start_i, d - tile_end_j + 1);
        int end_i = MIN(tile_end_i - 1, d - tile_start_j);
        
        for (int i = start_i; i <= end_i; i++) {
            int j = d - i;
            
            if (j < tile_start_j || j >= tile_end_j) continue;
            
            int val;
            if (i == 0) {
                val = j;
            } else if (j == 0) {
                val = i;
            } else {
                int cost = (work->str1[i - 1] == work->str2[j - 1]) ? 0 : 1;
                int from_diag = get_diag_cell(work->diagonals, i - 1, j - 1, work->len, work->max_diag_stored);
                int from_top = get_diag_cell(work->diagonals, i - 1, j, work->len, work->max_diag_stored);
                int from_left = get_diag_cell(work->diagonals, i, j - 1, work->len, work->max_diag_stored);
                val = MIN3(from_diag + cost, from_top + 1, from_left + 1);
            }
            
            set_diag_cell(work->diagonals, i, j, val, work->len, work->max_diag_stored);
        }
    }
    
    pthread_mutex_unlock(&work->diag_lock);
}

void *thread_do_work(void *pool_ptr) {
    threadpool_t *pool = (threadpool_t *)pool_ptr;
    
    while(1) {
        pthread_mutex_lock(&pool->lock);
        
        while (pool->queue_size == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->notify, &pool->lock);
        }
        
        if (pool->shutdown && pool->queue_size == 0) {
            pthread_mutex_unlock(&pool->lock);
            pthread_exit(NULL);
        }
        
        tile_work_t *work = pool->task_queue[pool->queue_head];
        pool->queue_head = (pool->queue_head + 1) % pool->queue_capacity;
        pool->queue_size--;
        
        pthread_mutex_unlock(&pool->lock);
        
        if (work) {
            // Wait for dependencies
            if (work->tile_i > 0) {
                int dep_idx = (work->tile_i - 1) * work->num_tiles + work->tile_j;
                pthread_mutex_lock(&work->tile_locks[dep_idx]);
                while (!work->tile_done[work->tile_i - 1][work->tile_j]) {
                    pthread_cond_wait(&work->tile_conds[dep_idx], &work->tile_locks[dep_idx]);
                }
                pthread_mutex_unlock(&work->tile_locks[dep_idx]);
            }
            
            if (work->tile_j > 0) {
                int dep_idx = work->tile_i * work->num_tiles + (work->tile_j - 1);
                pthread_mutex_lock(&work->tile_locks[dep_idx]);
                while (!work->tile_done[work->tile_i][work->tile_j - 1]) {
                    pthread_cond_wait(&work->tile_conds[dep_idx], &work->tile_locks[dep_idx]);
                }
                pthread_mutex_unlock(&work->tile_locks[dep_idx]);
            }
            
            // Compute the tile
            compute_tile_antidiag(work);
            
            // Mark as done
            int tile_idx = work->tile_i * work->num_tiles + work->tile_j;
            pthread_mutex_lock(&work->tile_locks[tile_idx]);
            work->tile_done[work->tile_i][work->tile_j] = 1;
            pthread_cond_broadcast(&work->tile_conds[tile_idx]);
            pthread_mutex_unlock(&work->tile_locks[tile_idx]);
            
            free(work);
        }
    }
}

threadpool_t *threadpool_create(int thread_count) {
    threadpool_t *pool = malloc(sizeof(threadpool_t));
    pool->thread_count = thread_count;
    pool->shutdown = 0;
    pool->queue_head = 0;
    pool->queue_tail = 0;
    pool->queue_size = 0;
    pool->queue_capacity = 10000;
    pool->task_queue = malloc(sizeof(tile_work_t *) * pool->queue_capacity);
    
    pthread_mutex_init(&pool->lock, NULL);
    pthread_cond_init(&pool->notify, NULL);
    
    pool->threads = malloc(sizeof(pthread_t) * thread_count);
    for (int i = 0; i < thread_count; i++) {
        pthread_create(&pool->threads[i], NULL, thread_do_work, pool);
    }
    
    return pool;
}

void threadpool_add(threadpool_t *pool, tile_work_t *work) {
    pthread_mutex_lock(&pool->lock);
    
    pool->task_queue[pool->queue_tail] = work;
    pool->queue_tail = (pool->queue_tail + 1) % pool->queue_capacity;
    pool->queue_size++;
    
    pthread_cond_signal(&pool->notify);
    pthread_mutex_unlock(&pool->lock);
}

void threadpool_destroy(threadpool_t *pool) {
    pthread_mutex_lock(&pool->lock);
    pool->shutdown = 1;
    pthread_mutex_unlock(&pool->lock);
    
    pthread_cond_broadcast(&pool->notify);
    
    for (int i = 0; i < pool->thread_count; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    free(pool->threads);
    free(pool->task_queue);
    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->notify);
    free(pool);
}

int cse2421_edit_distance(const char *str1, const char *str2, size_t len) {
    if (len == 0) return 0;
    
    // For small inputs, use simple sequential algorithm
    if (len < 1000) {
        int *prev_row = malloc((len + 1) * sizeof(int));
        int *curr_row = malloc((len + 1) * sizeof(int));
        
        for (size_t i = 0; i <= len; i++) {
            prev_row[i] = i;
        }
        
        for (size_t i = 1; i <= len; i++) {
            curr_row[0] = i;
            for (size_t j = 1; j <= len; j++) {
                int cost = (str1[i - 1] == str2[j - 1]) ? 0 : 1;
                curr_row[j] = MIN3(prev_row[j - 1] + cost, prev_row[j] + 1, curr_row[j - 1] + 1);
            }
            int *temp = prev_row;
            prev_row = curr_row;
            curr_row = temp;
        }
        
        int result = prev_row[len];
        free(prev_row);
        free(curr_row);
        return result;
    }
    
    int tile_size = TILE_SIZE;
    int num_tiles = (len + tile_size) / tile_size;
    
    // Allocate anti-diagonal storage
    // We need to store enough diagonals to handle tile dependencies
    // Maximum anti-diagonal number is 2*len, but we only keep recent ones
    int max_diag_stored = 2 * tile_size + 2;
    
    int **diagonals = malloc(max_diag_stored * sizeof(int *));
    for (int d = 0; d < max_diag_stored; d++) {
        // Each anti-diagonal d has min(d+1, 2*len+1-d) elements
        // For simplicity, allocate max size (len+1)
        diagonals[d] = aligned_alloc(32, (len + 1) * sizeof(int));
        memset(diagonals[d], 0, (len + 1) * sizeof(int));
    }
    
    // Initialize first row and column (anti-diagonals 0 through len)
    for (int d = 0; d <= len && d < max_diag_stored; d++) {
        int diag_stored = d % max_diag_stored;
        for (int i = 0; i <= d && i <= len; i++) {
            int j = d - i;
            if (j > len) continue;
            
            int val;
            if (i == 0) val = j;
            else if (j == 0) val = i;
            else val = 0;  // Will be computed
            
            int pos = (i <= j) ? i : j;
            diagonals[diag_stored][pos] = val;
        }
    }
    
    threadpool_t *pool = threadpool_create(NUM_THREADS);
    
    int total_tiles = num_tiles * num_tiles;
    pthread_mutex_t *tile_locks = malloc(sizeof(pthread_mutex_t) * total_tiles);
    pthread_cond_t *tile_conds = malloc(sizeof(pthread_cond_t) * total_tiles);
    volatile int **tile_done = malloc(sizeof(volatile int *) * num_tiles);
    pthread_mutex_t diag_lock;
    pthread_mutex_init(&diag_lock, NULL);
    
    for (int i = 0; i < num_tiles; i++) {
        tile_done[i] = calloc(num_tiles, sizeof(volatile int));
    }
    
    for (int i = 0; i < total_tiles; i++) {
        pthread_mutex_init(&tile_locks[i], NULL);
        pthread_cond_init(&tile_conds[i], NULL);
    }
    
    // Submit all tiles - they'll execute in wavefront order due to dependencies
    for (int tile_i = 0; tile_i < num_tiles; tile_i++) {
        for (int tile_j = 0; tile_j < num_tiles; tile_j++) {
            tile_work_t *work = malloc(sizeof(tile_work_t));
            work->str1 = str1;
            work->str2 = str2;
            work->len = len;
            work->diagonals = diagonals;
            work->max_diag_stored = max_diag_stored;
            work->tile_i = tile_i;
            work->tile_j = tile_j;
            work->tile_size = tile_size;
            work->num_tiles = num_tiles;
            work->tile_done = tile_done;
            work->tile_locks = tile_locks;
            work->tile_conds = tile_conds;
            work->diag_lock = diag_lock;
            
            threadpool_add(pool, work);
        }
    }
    
    // Wait for last tile
    int last_tile_idx = (num_tiles - 1) * num_tiles + (num_tiles - 1);
    pthread_mutex_lock(&tile_locks[last_tile_idx]);
    while (!tile_done[num_tiles - 1][num_tiles - 1]) {
        pthread_cond_wait(&tile_conds[last_tile_idx], &tile_locks[last_tile_idx]);
    }
    pthread_mutex_unlock(&tile_locks[last_tile_idx]);
    
    int result = get_diag_cell(diagonals, len, len, len, max_diag_stored);
    
    threadpool_destroy(pool);
    
    pthread_mutex_destroy(&diag_lock);
    
    for (int i = 0; i < total_tiles; i++) {
        pthread_mutex_destroy(&tile_locks[i]);
        pthread_cond_destroy(&tile_conds[i]);
    }
    
    for (int i = 0; i < num_tiles; i++) {
        free((void *)tile_done[i]);
    }
    
    free(tile_done);
    free(tile_locks);
    free(tile_conds);
    
    for (int d = 0; d < max_diag_stored; d++) {
        free(diagonals[d]);
    }
    free(diagonals);
    
    return result;
}