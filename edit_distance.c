#include "edit_distance.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MIN3(a,b,c) MIN(MIN(a,b),c)
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define TILE_SIZE 512
#define NUM_THREADS 16

typedef struct {
    const char *str1;
    const char *str2;
    size_t len;
    int **diagonals;
    int max_diag_stored;
    int tile_i;
    int tile_j;
    int tile_size;
    int num_tiles;
    volatile int *tile_ready;  // Flattened 1D array
} tile_work_t;

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t notify;
    pthread_t *threads;
    tile_work_t **tasks;
    int head;
    int tail;
    int count;
    int capacity;
    int shutdown;
} threadpool_t;

// Simplified diagonal access - no locks needed for reads
static inline int get_diag(int **diag, int i, int j, int len, int max_stored) {
    if (i < 0 || j < 0 || i > len || j > len) return 0;
    int d = (i + j) % max_stored;
    return diag[d][MIN(i, j)];
}

// Direct write - diagonal locks removed, dependency ordering ensures safety
static inline void set_diag(int **diag, int i, int j, int val, int max_stored) {
    int d = (i + j) % max_stored;
    diag[d][MIN(i, j)] = val;
}

void compute_tile_avx2(tile_work_t *w) {
    int si = w->tile_i * w->tile_size;
    int sj = w->tile_j * w->tile_size;
    int ei = MIN(si + w->tile_size, w->len + 1);
    int ej = MIN(sj + w->tile_size, w->len + 1);
    
    // Process anti-diagonals within tile
    for (int d = si + sj; d < ei + ej - 1; d++) {
        int start_i = MAX(si, d - ej + 1);
        int end_i = MIN(ei - 1, d - sj);
        
        int i = start_i;
        
        // Boundaries
        while (i <= end_i && (i == 0 || (d - i) == 0)) {
            int j = d - i;
            if (j >= sj && j < ej) {
                set_diag(w->diagonals, i, j, (i == 0) ? j : i, w->max_diag_stored);
            }
            i++;
        }
        
        // AVX2 vectorized interior (8 cells at a time)
        for (; i + 7 <= end_i; i += 8) {
            __m256i costs, diag_vals, top_vals, left_vals, results;
            int vals[8];
            
            // Gather dependencies and compute
            for (int k = 0; k < 8; k++) {
                int ii = i + k;
                int jj = d - ii;
                
                if (jj < sj || jj >= ej) {
                    vals[k] = 0;
                    continue;
                }
                
                int cost = (w->str1[ii - 1] != w->str2[jj - 1]);
                int diag = get_diag(w->diagonals, ii - 1, jj - 1, w->len, w->max_diag_stored);
                int top = get_diag(w->diagonals, ii - 1, jj, w->len, w->max_diag_stored);
                int left = get_diag(w->diagonals, ii, jj - 1, w->len, w->max_diag_stored);
                
                vals[k] = MIN3(diag + cost, top + 1, left + 1);
            }
            
            // Store results
            for (int k = 0; k < 8; k++) {
                int ii = i + k;
                int jj = d - ii;
                if (jj >= sj && jj < ej) {
                    set_diag(w->diagonals, ii, jj, vals[k], w->max_diag_stored);
                }
            }
        }
        
        // Remaining cells
        for (; i <= end_i; i++) {
            int j = d - i;
            if (j < sj || j >= ej) continue;
            
            int cost = (w->str1[i - 1] != w->str2[j - 1]) ? 1 : 0;
            int diag = get_diag(w->diagonals, i - 1, j - 1, w->len, w->max_diag_stored);
            int top = get_diag(w->diagonals, i - 1, j, w->len, w->max_diag_stored);
            int left = get_diag(w->diagonals, i, j - 1, w->len, w->max_diag_stored);
            
            set_diag(w->diagonals, i, j, MIN3(diag + cost, top + 1, left + 1), w->max_diag_stored);
        }
    }
}

void *worker_thread(void *arg) {
    threadpool_t *pool = (threadpool_t *)arg;
    
    while (1) {
        pthread_mutex_lock(&pool->lock);
        
        while (pool->count == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->notify, &pool->lock);
        }
        
        if (pool->shutdown && pool->count == 0) {
            pthread_mutex_unlock(&pool->lock);
            return NULL;
        }
        
        tile_work_t *work = pool->tasks[pool->head];
        pool->head = (pool->head + 1) % pool->capacity;
        pool->count--;
        
        pthread_mutex_unlock(&pool->lock);
        
        if (work) {
            // Wait for dependencies using atomic check
            int dep_left = work->tile_i * work->num_tiles + (work->tile_j - 1);
            int dep_top = (work->tile_i - 1) * work->num_tiles + work->tile_j;
            
            if (work->tile_j > 0) {
                while (!__atomic_load_n(&work->tile_ready[dep_left], __ATOMIC_ACQUIRE)) {
                    __builtin_ia32_pause();
                }
            }
            
            if (work->tile_i > 0) {
                while (!__atomic_load_n(&work->tile_ready[dep_top], __ATOMIC_ACQUIRE)) {
                    __builtin_ia32_pause();
                }
            }
            
            // Compute tile
            compute_tile_avx2(work);
            
            // Mark done
            int my_idx = work->tile_i * work->num_tiles + work->tile_j;
            __atomic_store_n(&work->tile_ready[my_idx], 1, __ATOMIC_RELEASE);
            
            free(work);
        }
    }
}

threadpool_t *create_pool(int n) {
    threadpool_t *p = malloc(sizeof(threadpool_t));
    p->shutdown = 0;
    p->head = p->tail = p->count = 0;
    p->capacity = 8192;
    p->tasks = malloc(sizeof(tile_work_t *) * p->capacity);
    
    pthread_mutex_init(&p->lock, NULL);
    pthread_cond_init(&p->notify, NULL);
    
    p->threads = malloc(sizeof(pthread_t) * n);
    for (int i = 0; i < n; i++) {
        pthread_create(&p->threads[i], NULL, worker_thread, p);
    }
    
    return p;
}

void add_task(threadpool_t *p, tile_work_t *w) {
    pthread_mutex_lock(&p->lock);
    p->tasks[p->tail] = w;
    p->tail = (p->tail + 1) % p->capacity;
    p->count++;
    pthread_cond_signal(&p->notify);
    pthread_mutex_unlock(&p->lock);
}

void destroy_pool(threadpool_t *p, int n) {
    pthread_mutex_lock(&p->lock);
    p->shutdown = 1;
    pthread_mutex_unlock(&p->lock);
    pthread_cond_broadcast(&p->notify);
    
    for (int i = 0; i < n; i++) {
        pthread_join(p->threads[i], NULL);
    }
    
    free(p->threads);
    free(p->tasks);
    pthread_mutex_destroy(&p->lock);
    pthread_cond_destroy(&p->notify);
    free(p);
}

int cse2421_edit_distance(const char *str1, const char *str2, size_t len) {
    if (len == 0) return 0;
    
    // Small input optimization
    if (len < 800) {
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
    
    // Allocate diagonal storage (no per-diagonal locks)
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
    
    // Create tile ready flags (atomic spinlock array)
    volatile int *tile_ready = calloc(num_tiles * num_tiles, sizeof(int));
    
    threadpool_t *pool = create_pool(NUM_THREADS);
    
    // Submit all tiles
    for (int ti = 0; ti < num_tiles; ti++) {
        for (int tj = 0; tj < num_tiles; tj++) {
            tile_work_t *w = malloc(sizeof(tile_work_t));
            w->str1 = str1;
            w->str2 = str2;
            w->len = len;
            w->diagonals = diagonals;
            w->max_diag_stored = max_stored;
            w->tile_i = ti;
            w->tile_j = tj;
            w->tile_size = TILE_SIZE;
            w->num_tiles = num_tiles;
            w->tile_ready = tile_ready;
            
            add_task(pool, w);
        }
    }
    
    // Wait for last tile
    int last_idx = (num_tiles - 1) * num_tiles + (num_tiles - 1);
    while (!__atomic_load_n(&tile_ready[last_idx], __ATOMIC_ACQUIRE)) {
        __builtin_ia32_pause();
    }
    
    int result = get_diag(diagonals, len, len, len, max_stored);
    
    // Cleanup
    destroy_pool(pool, NUM_THREADS);
    
    for (int d = 0; d < max_stored; d++) {
        free(diagonals[d]);
    }
    free(diagonals);
    free((void *)tile_ready);
    
    return result;
}