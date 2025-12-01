CC = gcc
CFLAGS = -Wall -O3 -march=native -pthread -march=native
LDFLAGS = -pthread -lm

TARGET = lab7
TEST = test_edit_distance

OBJS = edit_distance.o main.o
TEST_OBJS = edit_distance.o test_edit_distance.o

all: $(TARGET) $(TEST)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(TEST): $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

edit_distance.o: edit_distance.c edit_distance.h
	$(CC) $(CFLAGS) -c edit_distance.c

main.o: main.c edit_distance.h
	$(CC) $(CFLAGS) -c main.c

test_edit_distance.o: test_edit_distance.c edit_distance.h
	$(CC) $(CFLAGS) -c test_edit_distance.c

clean:
	rm -f $(TARGET) $(TEST) *.o

.PHONY: all clean