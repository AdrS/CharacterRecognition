#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "logisticRegression.h"
#include "mnist.h"
#include "sample.h"

///////////////////////////////////////////////////////////////////////////////////
///////////////////TODO: add momentum
///////////////////////////////////////////////////////////////////////////////////
void usage() {
	printf("usage: main [OPTIONS]\n--seed - specify unsigned int to use as seed\n");
	printf("--epochs - number of epochs to run for\n");
	printf("--rate - learing rate\n");
	printf("--size - batch size\n");
	exit(-2);
}
int main(const int argc, const char** argv) {
	//TRY USING SMALLER BATCH SIZES
	LogisticRegressionClassifier classifier;
	Sample* trainingSamples;
	Sample* testingSamples;
	double learningRate = 0.1;
	double output[10];
	unsigned int maxEpochs = 25, i, correct = 0, seed = 6082015, batchSize = 600;
	i = 1;
	while(i < argc) {
		if(strcmp(argv[i],"--seed") == 0) {
			i++;
			if(i < argc) {
				seed = atol(argv[i]);
				if(seed == 0) {
					printf("error: could not parse parameter \"%s\" as number\n", argv[i]);
					usage();
				}
			} else {
				printf("error: value for seed not specified\n");
				usage();
			}
			i++;
		} else if(strcmp(argv[i], "--epochs") == 0) {
			i++;
			if(i < argc) {
				maxEpochs = atol(argv[i]);
				if(maxEpochs == 0) {
					printf("error: could not parse parameter \"%s\" as number\n", argv[i]);
					usage();
				}
			} else {
				printf("error: value for max epochs not specified\n");
				usage();
			}
			i++;
		} else if(strcmp(argv[i], "--rate") == 0) {
			i++;
			if(i < argc) {
				learningRate = atof(argv[i]);
				if(learningRate <= 0.0) {
					printf("error: invalid learing rate \"%s\"\n", argv[i]);
					usage();
				}
			} else {
				printf("error: value for learning rate not specified\n");
				usage();
			}
			i++;
		} else if(strcmp(argv[i], "--size") == 0) {
			i++;
			if(i < argc) {
				batchSize = atol(argv[i]);
				if(batchSize == 0) {
					printf("error: could not parse parameter \"%s\" as number\n", argv[i]);
					usage();
				}
			} else {
				printf("error: value for batch size not specified\n");
				usage();
			}
			i++;
		} else {
			printf("error: unrecognized option \"%s\"\n", argv[i]);
			usage();
		}
	}
	printf("Learning rate: %f\n", learningRate);
	printf("Max epochs: %d\n", maxEpochs);
	printf("Seed: %d\n", seed);
	printf("Batch size: %d\n", batchSize);
	int guess;
	if(createClassifier(&classifier, 28*28, 10, seed)) {
		printf("error: could not create classifier");
		return -1;
	}
	trainingSamples = load("../datasets/train-images-idx3-ubyte","../datasets/train-labels-idx1-ubyte");
	testingSamples = load("../datasets/t10k-images-idx3-ubyte","../datasets/t10k-labels-idx1-ubyte");
	if(!trainingSamples) {
		printf("error: could not load training samples\n");
		return -1;
	}
	if(!testingSamples) {
		printf("error: could not load testing samples\n");
		return -1;
	}
	//printWeights(NULL, &classifier);
	printf("Training ... \n");
	if(batchTrain(&classifier, trainingSamples, 60000, maxEpochs, learningRate, batchSize)) {
		printf("error: could not complete training\n");
		return -1;
	}
	//the moment of truth! time to test
	for(i = 0; i < 10000; i++) {
		guess = classify(&classifier,testingSamples[i].inputs, output);
		if(guess < 0) {
			printf("error: something went wrong during testing\n");
			return -1;
		}
		if(testingSamples[i].outputs[guess]) {
			correct++;
		}
	}
	printf("Learning rate: %f\n", learningRate);
	printf("Max epochs: %d\n", maxEpochs);
	printf("Seed: %d\n", seed);
	printf("Batch size: %d\n", batchSize);
	printf("%d of %d (%f%%) correctly classified\n", correct, 10000, correct/100.0);
	return 0;
}
