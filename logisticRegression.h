#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "sample.h"

#ifndef __LOGISTIC_REGRESSION_HEADER__
#define __LOGISTIC_REGRESSION_HEADER__

typedef struct {
	unsigned int features;
	unsigned int classes;
	double** weights; //a features + 1 by classes dimenstion matrix
	unsigned int seed;
} LogisticRegressionClassifier;

//returns 0 on success, -1 on failure, -2 on invalid parameters
int createClassifier(LogisticRegressionClassifier* classifier,
	unsigned int features, unsigned int classes, unsigned int seed);

//probabilites of class membership are stored in output and the index of the maximum 
//is returned (or -2 for invalid params)
int classify(LogisticRegressionClassifier* classifier, double* input, double* output);

//f(x) = 1/(1 + exp(-x))
double logisticFunction(double x);

//f'(x) = exp(-x)/(1 + exp(-x))^2
double logisticFunctionDerivative(double x);

//if output is null, stdout is used
void printWeights(FILE* output, LogisticRegressionClassifier* classifier);

int batchTrain(LogisticRegressionClassifier* classifier, Sample* samples,
	unsigned int numberOfSamples, unsigned int maxEpochs,
	double learningRate, unsigned int batchSize);

#endif
