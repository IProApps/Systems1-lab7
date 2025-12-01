#include "edit_distance.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>
#include <unistd.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MIN3(a,b,c) MIN(MIN(a,b),c)
#define TILE_SIZE 256
#define NUM_THREADS 8

typedef struct {
    const char *str1;
    const char *str2;
    size_t len;
    int *buffer;  // Circular buffer for stripes
    int stripe_width;
    int tile_i;
    int tile_j;
    int tile_size;
    int num_tiles;
    volatile int **tile_done;
    pthread_mutex_t *locks;
    pthread_cond_t *conds;
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

// Get value from circular stripe buffer
static inline int get_cell(int *buffer, int i, int j, int stripe_width, int len) {
    if (i < 0 || j < 0 || i > len || j > len) return 0;
    int stripe_idx = i % stripe_width;
    return buffer[stripe_idx * (len + 1) + j];
}

// Set value in circular stripe buffer
static inline void set_cell(int *buffer, int i, int j, int val, int stripe_width, int len) {
    int stripe_idx = i % stripe_width;
    buffer[stripe_idx * (len + 1) + j] = val;
}

void compute_tile_avx2(tile_work_t *work) {
    int tile_start_i = work->tile_i * work->tile_size;
    int tile_start_j = work->tile_j * work->tile_size;
    int tile_end_i = MIN(tile_start_i + work->tile_size, work->len + 1);
    int tile_end_j = MIN(tile_start_j + work->tile_size, work->len + 1);
    
    for (int i = tile_start_i; i < tile_end_i; i++) {
        for (int j = tile_start_j; j < tile_end_j; j++) {
            int val;
            if (i == 0) {
                val = j;
            } else if (j == 0) {
                val = i;
            } else {
                int cost = (work->str1[i - 1] == work->str2[j - 1]) ? 0 : 1;
                int from_diag = get_cell(work->buffer, i - 1, j - 1, work->stripe_width, work->len) + cost;
                int from_top = get_cell(work->buffer, i - 1, j, work->stripe_width, work->len) + 1;
                int from_left = get_cell(work->buffer, i, j - 1, work->stripe_width, work->len) + 1;
                val = MIN3(from_diag, from_top, from_left);
            }
            set_cell(work->buffer, i, j, val, work->stripe_width, work->len);
        }
    }
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
            compute_tile_avx2(work);
            
            // Mark tile as done
            int tile_idx = work->tile_i * work->num_tiles + work->tile_j;
            pthread_mutex_lock(&work->locks[tile_idx]);
            work->tile_done[work->tile_i][work->tile_j] = 1;
            pthread_cond_broadcast(&work->conds[tile_idx]);
            pthread_mutex_unlock(&work->locks[tile_idx]);
            
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
    
    // For small inputs, use simple sequential algorithm with 2 rows
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
    
    // For large inputs, use wavefront tiled parallelism
    int tile_size = TILE_SIZE;
    int num_tiles = (len + tile_size) / tile_size;
    
    // Allocate circular buffer for stripes
    // We only need enough stripes to cover tile dependencies
    int stripe_width = tile_size + 1;
    int *buffer = aligned_alloc(32, stripe_width * (len + 1) * sizeof(int));
    memset(buffer, 0, stripe_width * (len + 1) * sizeof(int));
    
    // Initialize first row
    for (size_t j = 0; j <= len; j++) {
        set_cell(buffer, 0, j, j, stripe_width, len);
    }
    
    // Initialize first column in buffer
    for (int i = 1; i < stripe_width && i <= len; i++) {
        set_cell(buffer, i, 0, i, stripe_width, len);
    }
    
    threadpool_t *pool = threadpool_create(NUM_THREADS);
    
    // Allocate synchronization structures
    int total_tiles = num_tiles * num_tiles;
    pthread_mutex_t *locks = malloc(sizeof(pthread_mutex_t) * total_tiles);
    pthread_cond_t *conds = malloc(sizeof(pthread_cond_t) * total_tiles);
    volatile int **tile_done = malloc(sizeof(volatile int *) * num_tiles);
    
    for (int i = 0; i < num_tiles; i++) {
        tile_done[i] = calloc(num_tiles, sizeof(volatile int));
    }
    
    for (int i = 0; i < total_tiles; i++) {
        pthread_mutex_init(&locks[i], NULL);
        pthread_cond_init(&conds[i], NULL);
    }
    
    // Process tiles in wavefront (anti-diagonal) order
    // Tiles on the same anti-diagonal can execute in parallel
    for (int wave = 0; wave < 2 * num_tiles - 1; wave++) {
        for (int tile_i = 0; tile_i < num_tiles; tile_i++) {
            int tile_j = wave - tile_i;
            
            if (tile_j < 0 || tile_j >= num_tiles) continue;
            
            // Wait for dependencies: tile above (tile_i-1, tile_j) and tile to left (tile_i, tile_j-1)
            if (tile_i > 0) {
                int dep_idx = (tile_i - 1) * num_tiles + tile_j;
                pthread_mutex_lock(&locks[dep_idx]);
                while (!tile_done[tile_i - 1][tile_j]) {
                    pthread_cond_wait(&conds[dep_idx], &locks[dep_idx]);
                }
                pthread_mutex_unlock(&locks[dep_idx]);
            }
            
            if (tile_j > 0) {
                int dep_idx = tile_i * num_tiles + (tile_j - 1);
                pthread_mutex_lock(&locks[dep_idx]);
                while (!tile_done[tile_i][tile_j - 1]) {
                    pthread_cond_wait(&conds[dep_idx], &locks[dep_idx]);
                }
                pthread_mutex_unlock(&locks[dep_idx]);
            }
            
            // Create work item for this tile
            tile_work_t *work = malloc(sizeof(tile_work_t));
            work->str1 = str1;
            work->str2 = str2;
            work->len = len;
            work->buffer = buffer;
            work->stripe_width = stripe_width;
            work->tile_i = tile_i;
            work->tile_j = tile_j;
            work->tile_size = tile_size;
            work->num_tiles = num_tiles;
            work->tile_done = tile_done;
            work->locks = locks;
            work->conds = conds;
            
            threadpool_add(pool, work);
        }
    }
    
    // Wait for the last tile to complete
    int last_tile_idx = (num_tiles - 1) * num_tiles + (num_tiles - 1);
    pthread_mutex_lock(&locks[last_tile_idx]);
    while (!tile_done[num_tiles - 1][num_tiles - 1]) {
        pthread_cond_wait(&conds[last_tile_idx], &locks[last_tile_idx]);
    }
    pthread_mutex_unlock(&locks[last_tile_idx]);
    
    int result = get_cell(buffer, len, len, stripe_width, len);
    
    threadpool_destroy(pool);
    
    for (int i = 0; i < total_tiles; i++) {
        pthread_mutex_destroy(&locks[i]);
        pthread_cond_destroy(&conds[i]);
    }
    
    for (int i = 0; i < num_tiles; i++) {
        free((void *)tile_done[i]);
    }
    
    free(tile_done);
    free(locks);
    free(conds);
    free(buffer);
    
    return result;
}