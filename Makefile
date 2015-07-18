CC = gcc
CCFLAGS = -c -Wall
CLFLAGS = -o

all: main
main: logisticRegression.o mnist.o main.o functions.o sample.o vector.o
	$(CC) logisticRegression.o mnist.o main.o functions.o sample.o vector.o $(CLFLAGS) main

#TODO: does sample.h need to be here? (analogous question for other recipies)
main.o: main.c sample.h
	$(CC) $(CCFLAGS) main.c

logisticRegression.o: logisticRegression.c logisticRegression.h
	$(CC) $(CCFLAGS) logisticRegression.c

#START TESTING CODE
neuralNetworkTests: neuralNetworkTests.o neuralNetwork.o vector.o functions.o sample.o
	$(CC) neuralNetworkTests.o neuralNetwork.o vector.o functions.o sample.o $(CLFLAGS) main

neuralNetworkTests.o: neuralNetwork.c
	$(CC) $(CCFLAGS) neuralNetworkTests.c

vectorTests: vectorTests.o vector.o
	$(CC) vectorTests.o vector.o $(CLFLAGS) main

vectorTests.o: vectorTests.c
	$(CC) $(CCFLAGS) vectorTests.c
#END TESTING CODE

sample.o: sample.c sample.h
	$(CC) $(CCFLAGS) sample.c
	
neuralNetwork.o: neuralNetwork.h neuralNetwork.c
	$(CC) $(CCFLAGS) neuralNetwork.c

vector.o: vector.c vector.h
	$(CC) $(CCFLAGS) vector.c

functions.o: functions.c functions.h
	$(CC) $(CCFLAGS) functions.c

mnist.o: mnist.c mnist.h sample.h
	$(CC) $(CCFLAGS) mnist.c

clean:
	rm *.o
