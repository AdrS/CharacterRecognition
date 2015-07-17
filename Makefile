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

#START TESTING CODE
neuralNetworkTests: neuralNetworkTests.o neuralNetwork.o vector.o
	$(CC) neuralNetworkTests.o neuralNetwork.o vector.o $(CLFLAGS) main

neuralNetworkTests.o: neuralNetwork.c
	$(CC) $(CCFLAGS) neuralNetworkTests.c

vectorTests: vectorTests.o vector.o
	$(CC) vectorTests.o vector.o $(CLFLAGS) main

vectorTests.o: vectorTests.c
	$(CC) $(CCFLAGS) vectorTests.c
#END TESTING CODE

neuralNetwork.o: neuralNetwork.h neuralNetwork.c
	$(CC) $(CCFLAGS) neuralNetwork.c

vector.o: vector.c vector.h
	$(CC) $(CCFLAGS) vector.c

mnist.o: mnist.c mnist.h sample.h
	$(CC) $(CCFLAGS) mnist.c

clean:
	rm *.o
