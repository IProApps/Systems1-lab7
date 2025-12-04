CC = gcc
CFLAGS = -Wall -O3 -march=native -pthread
LDFLAGS = -pthread -lm

TARGET = lab7
TEST = test_edit_distance
RANDOM_TEST = test_random

OBJS = edit_distance.o main.o
TEST_OBJS = edit_distance.o test_edit_distance.o
RANDOM_TEST_OBJS = edit_distance.o test_random.o

all: $(TARGET) $(TEST) $(RANDOM_TEST)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(TEST): $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(RANDOM_TEST): $(RANDOM_TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

edit_distance.o: edit_distance.c edit_distance.h
	$(CC) $(CFLAGS) -c edit_distance.c

main.o: main.c edit_distance.h
	$(CC) $(CFLAGS) -c main.c

test_edit_distance.o: test_edit_distance.c edit_distance.h
	$(CC) $(CFLAGS) -c test_edit_distance.c

test_random.o: test_random.c edit_distance.h
	$(CC) $(CFLAGS) -c test_random.c

# Run all tests
test: $(TEST) $(RANDOM_TEST)
	@echo "========================================="
	@echo "Running basic test suite..."
	@echo "========================================="
	./$(TEST)
	@echo ""
	@echo "========================================="
	@echo "Running random test suite..."
	@echo "========================================="
	./$(RANDOM_TEST)

# Run only the random tests
test-random: $(RANDOM_TEST)
	./$(RANDOM_TEST)

# Run only the basic tests
test-basic: $(TEST)
	./$(TEST)

clean:
	rm -f $(TARGET) $(TEST) $(RANDOM_TEST) *.o

.PHONY: all clean test test-random test-basic