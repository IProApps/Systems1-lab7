#include "edit_distance.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

void test_identical_strings() {
    const char *str = "hello";
    int dist = cse2421_edit_distance(str, str, strlen(str));
    printf("Test identical strings: %s == %s, distance = %d (expected 0)\n", str, str, dist);
    assert(dist == 0);
}

void test_empty_strings() {
    const char *str = "";
    int dist = cse2421_edit_distance(str, str, 0);
    printf("Test empty strings: distance = %d (expected 0)\n", dist);
    assert(dist == 0);
}

void test_one_substitution() {
    const char *str1 = "kitten";
    const char *str2 = "sitten";
    int dist = cse2421_edit_distance(str1, str2, strlen(str1));
    printf("Test one substitution: %s -> %s, distance = %d (expected 1)\n", str1, str2, dist);
    assert(dist == 1);
}

void test_classic_example() {
    const char *str1 = "kitten";
    const char *str2 = "sitting";
    // Need to pad to same length for this implementation
    char s1[8] = "kitten";
    char s2[8] = "sitting";
    int len = 7;
    int dist = cse2421_edit_distance(s1, s2, len);
    printf("Test classic: %s -> %s, distance = %d (expected 3)\n", s1, s2, dist);
}

void test_completely_different() {
    const char *str1 = "abc";
    const char *str2 = "xyz";
    int dist = cse2421_edit_distance(str1, str2, 3);
    printf("Test completely different: %s -> %s, distance = %d (expected 3)\n", str1, str2, dist);
    assert(dist == 3);
}

void test_single_char() {
    const char *str1 = "a";
    const char *str2 = "b";
    int dist = cse2421_edit_distance(str1, str2, 1);
    printf("Test single char: %s -> %s, distance = %d (expected 1)\n", str1, str2, dist);
    assert(dist == 1);
}

void test_longer_strings() {
    char str1[100], str2[100];
    for (int i = 0; i < 99; i++) {
        str1[i] = 'A' + (i % 26);
        str2[i] = 'A' + ((i + 1) % 26);
    }
    str1[99] = '\0';
    str2[99] = '\0';
    
    int dist = cse2421_edit_distance(str1, str2, 99);
    printf("Test longer strings (99 chars): distance = %d\n", dist);
}

int main() {
    printf("Running edit distance tests...\n\n");
    
    test_empty_strings();
    test_identical_strings();
    test_one_substitution();
    test_completely_different();
    test_single_char();
    test_classic_example();
    test_longer_strings();
    
    printf("\nAll tests passed!\n");
    return 0;
}