#include "edit_distance.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

// Reference implementation - standard dynamic programming
int reference_edit_distance(const char *s1, const char *s2, size_t len) {
    int *prev = (int *)malloc((len + 1) * sizeof(int));
    int *curr = (int *)malloc((len + 1) * sizeof(int));
    
    if (!prev || !curr) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    
    // Initialize first row
    for (size_t i = 0; i <= len; i++) {
        prev[i] = i;
    }
    
    // Fill the matrix
    for (size_t i = 1; i <= len; i++) {
        curr[0] = i;
        for (size_t j = 1; j <= len; j++) {
            int cost = (s1[i-1] == s2[j-1]) ? 0 : 1;
            int ins = prev[j] + 1;
            int del = curr[j-1] + 1;
            int sub = prev[j-1] + cost;
            
            int min = (ins < del) ? ins : del;
            curr[j] = (sub < min) ? sub : min;
        }
        int *temp = prev;
        prev = curr;
        curr = temp;
    }
    
    int result = prev[len];
    free(prev);
    free(curr);
    return result;
}

// Generate random string with given alphabet size
void generate_random_string(char *str, size_t len, int alphabet_size) {
    for (size_t i = 0; i < len; i++) {
        str[i] = 'A' + (rand() % alphabet_size);
    }
    str[len] = '\0';
}

// Test with specific patterns
void test_pattern_based() {
    printf("=== Pattern-Based Tests ===\n");
    
    size_t test_sizes[] = {10, 100, 500, 1000, 5000, 10000};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int i = 0; i < num_tests; i++) {
        size_t len = test_sizes[i];
        char *str1 = (char *)malloc(len + 1);
        char *str2 = (char *)malloc(len + 1);
        
        // Test 1: Identical strings
        generate_random_string(str1, len, 26);
        memcpy(str2, str1, len + 1);
        
        int result = cse2421_edit_distance(str1, str2, len);
        int expected = 0;
        printf("  Identical[%6zu]: %d (expected %d) %s\n", 
               len, result, expected, (result == expected) ? "✓" : "✗ FAILED");
        assert(result == expected);
        
        // Test 2: One difference
        generate_random_string(str1, len, 26);
        memcpy(str2, str1, len + 1);
        if (len > 0) str2[len/2] = (str2[len/2] == 'A') ? 'Z' : 'A';
        
        result = cse2421_edit_distance(str1, str2, len);
        expected = (len > 0) ? 1 : 0;
        printf("  OneDiff[%6zu]:   %d (expected %d) %s\n", 
               len, result, expected, (result == expected) ? "✓" : "✗ FAILED");
        assert(result == expected);
        
        // Test 3: Completely different
        for (size_t j = 0; j < len; j++) {
            str1[j] = 'A';
            str2[j] = 'Z';
        }
        str1[len] = '\0';
        str2[len] = '\0';
        
        result = cse2421_edit_distance(str1, str2, len);
        expected = len;
        printf("  AllDiff[%6zu]:   %d (expected %zu) %s\n", 
               len, result, len, (result == expected) ? "✓" : "✗ FAILED");
        assert(result == expected);
        
        free(str1);
        free(str2);
    }
    printf("\n");
}

// Test with random inputs of various sizes and alphabet sizes
void test_random_inputs() {
    printf("=== Random Input Tests ===\n");
    
    size_t test_sizes[] = {10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000};
    int alphabet_sizes[] = {2, 4, 10, 26};
    
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int num_alphabets = sizeof(alphabet_sizes) / sizeof(alphabet_sizes[0]);
    
    int test_count = 0;
    int pass_count = 0;
    
    for (int s = 0; s < num_sizes; s++) {
        size_t len = test_sizes[s];
        
        for (int a = 0; a < num_alphabets; a++) {
            int alphabet = alphabet_sizes[a];
            
            // Run multiple random tests for each configuration
            int num_trials = (len <= 1000) ? 10 : ((len <= 10000) ? 5 : 2);
            
            for (int trial = 0; trial < num_trials; trial++) {
                char *str1 = (char *)malloc(len + 1);
                char *str2 = (char *)malloc(len + 1);
                
                if (!str1 || !str2) {
                    fprintf(stderr, "Memory allocation failed\n");
                    exit(1);
                }
                
                generate_random_string(str1, len, alphabet);
                generate_random_string(str2, len, alphabet);
                
                clock_t start = clock();
                int result = cse2421_edit_distance(str1, str2, len);
                clock_t end = clock();
                double time_impl = ((double)(end - start)) / CLOCKS_PER_SEC;
                
                start = clock();
                int reference = reference_edit_distance(str1, str2, len);
                end = clock();
                double time_ref = ((double)(end - start)) / CLOCKS_PER_SEC;
                
                test_count++;
                int passed = (result == reference);
                if (passed) pass_count++;
                
                printf("  Len=%6zu Alph=%2d Trial=%2d: result=%5d ref=%5d (%.4fs vs %.4fs) %s\n",
                       len, alphabet, trial + 1, result, reference, 
                       time_impl, time_ref,
                       passed ? "✓" : "✗ FAILED");
                
                if (!passed) {
                    fprintf(stderr, "\nFAILURE DETAILS:\n");
                    fprintf(stderr, "Length: %zu, Alphabet: %d\n", len, alphabet);
                    fprintf(stderr, "First 50 chars of str1: %.50s\n", str1);
                    fprintf(stderr, "First 50 chars of str2: %.50s\n", str2);
                    fprintf(stderr, "Result: %d, Expected: %d\n", result, reference);
                    free(str1);
                    free(str2);
                    exit(1);
                }
                
                free(str1);
                free(str2);
            }
        }
    }
    
    printf("\n=== Summary ===\n");
    printf("Total tests: %d\n", test_count);
    printf("Passed: %d\n", pass_count);
    printf("Failed: %d\n", test_count - pass_count);
    printf("\n");
}

// Stress test with edge cases
void test_edge_cases() {
    printf("=== Edge Case Tests ===\n");
    
    // Empty strings
    char empty1[] = "";
    char empty2[] = "";
    int result = cse2421_edit_distance(empty1, empty2, 0);
    printf("  Empty strings: %d (expected 0) %s\n", 
           result, (result == 0) ? "✓" : "✗ FAILED");
    assert(result == 0);
    
    // Single character - same
    char single1[] = "A";
    char single2[] = "A";
    result = cse2421_edit_distance(single1, single2, 1);
    printf("  Single char (same): %d (expected 0) %s\n", 
           result, (result == 0) ? "✓" : "✗ FAILED");
    assert(result == 0);
    
    // Single character - different
    char single3[] = "A";
    char single4[] = "B";
    result = cse2421_edit_distance(single3, single4, 1);
    printf("  Single char (diff): %d (expected 1) %s\n", 
           result, (result == 1) ? "✓" : "✗ FAILED");
    assert(result == 1);
    
    // All same character
    size_t len = 1000;
    char *same1 = (char *)malloc(len + 1);
    char *same2 = (char *)malloc(len + 1);
    memset(same1, 'A', len);
    memset(same2, 'A', len);
    same1[len] = '\0';
    same2[len] = '\0';
    result = cse2421_edit_distance(same1, same2, len);
    printf("  All same (%zu chars): %d (expected 0) %s\n", 
           len, result, (result == 0) ? "✓" : "✗ FAILED");
    assert(result == 0);
    free(same1);
    free(same2);
    
    printf("\n");
}

int main() {
    // Seed with fixed value for reproducibility
    srand(42);
    
    printf("========================================\n");
    printf("Comprehensive Random Test Suite\n");
    printf("Testing cse2421_edit_distance\n");
    printf("========================================\n\n");
    
    test_edge_cases();
    test_pattern_based();
    test_random_inputs();
    
    printf("========================================\n");
    printf("ALL TESTS PASSED! ✓\n");
    printf("========================================\n");
    
    return 0;
}