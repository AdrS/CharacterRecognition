#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "neuralNetwork.h"
#include "sample.h"

int main() {
	unsigned int xorLayerSizes[] = {2, 4, 1};
	double s1i[] = {0.0, 0.0};
	double s1o[] = {0.0};
	double s2i[] = {0.0, 1.0};
	double s2o[] = {1.0};
	double s3i[] = {1.0, 0.0};
	double s3o[] = {1.0};
	double s4i[] = {1.0, 1.0};
	double s4o[] = {0.0};
	Sample xorSamples[] = {{s1i, s1o},
				{s2i, s2o},
				{s3i, s3o},
				{s4i, s4o}};
	int ret;
	NeuralNetwork xorNet;
	//srand(time(NULL));
	puts("creating net ...");
	ret = createNet(&xorNet, xorLayerSizes, 3, LOGISTIC);
	if(ret) {
		printf("could not create net\n");
		return ret;
	}
#if 0
	puts("ORIGINAL SAMPLES");
	printSamples(NULL, xorSamples, 4, 2, 1);
	shuffleSamples(xorSamples, 4);
	puts("\nSHUFFLED SAMPLES");
	printSamples(NULL, xorSamples, 4, 2, 1);
	putchar('\n');
	
	puts("BEFORE TRAINGING");
#endif
	printNet(stdout, &xorNet, 1);
	feedForward(&xorNet, xorSamples[2].inputs);
	//trainNet(&xorNet, xorSamples, 4, 20, 4, 0.1);
	puts("AFTER TRAINGING");
	printNet(stdout, &xorNet, 1);
	deleteNet(&xorNet, 0);
	/*
	NeuralNetwork nn;
	unsigned int layerSizes[] = {2, 4, 7, 1};
	ret = createNet(&nn, layerSizes, 4, LOGISTIC);
	if(ret) {
		printf("could not create net\n");
		return ret;
	}
	printNet(stdout, &nn);
	deleteNet(&nn, 0);
	*/
	return 0;
}
