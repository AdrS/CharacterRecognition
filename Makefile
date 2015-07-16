CC = gcc
CCFLAGS = -c -Wall
CLFLAGS = -o

all: main
main: logisticRegression.o mnist.o main.o
	$(CC) logisticRegression.o mnist.o main.o $(CLFLAGS) main

main.o: main.c sample.h
	$(CC) $(CCFLAGS) main.c

logisticRegression.o: logisticRegression.c logisticRegression.h sample.h 
	$(CC) $(CCFLAGS) logisticRegression.c

mnist.o: mnist.c mnist.h sample.h
	$(CC) $(CCFLAGS) mnist.c

clean:
	rm *.o
