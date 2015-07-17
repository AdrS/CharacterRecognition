#include "logisticRegression.h"

int isValid(LogisticRegressionClassifier* classifier, char deep) {
	int i;
	char valid = classifier && classifier->weights && classifier->features && classifier->classes;
	if(valid && deep) {
		for(i = 0; i < classifier->features + 1; i++) {
			if(!classifier->weights[i])
				return -1;
		}
	}
	return valid;
}
int createClassifier(LogisticRegressionClassifier* classifier,
	unsigned int features, unsigned int classes, unsigned int seed) {
	int i, j;
	srand(seed);
	if(!classifier || features == 0 || classes == 0) {
		return -2;
	}
	classifier->features = features;
	classifier->classes = classes;
	classifier->seed = seed;
	classifier->weights = (double**)malloc(sizeof(double*)* (features + 1));
	if(!classifier->weights) {
		fprintf(stderr,"error: could not allocate memory for classifier");
		return -1;
	}
	for(i = 0; i < features + 1; i++) {
		classifier->weights[i] = (double*)malloc(sizeof(double) * classes);
		if(!classifier->weights[i]) {
			//free memory allready allocated
			for(j = 0; j < i; j++) {
				free(classifier->weights[j]);
			}
			fprintf(stderr,"error: could not allocate memory for classifier");
			return -1;
		}
		for(j = 0; j < classifier->classes; j++) {
			classifier->weights[i][j] = randomInt(-0.5, 0.5);
		}
	}
	return 0;
}
int classify(LogisticRegressionClassifier* classifier, double* input, double* output) {
	int i, j;
	double weightedSum;
	double totalSum = 0.0;
	double max;
	if(!isValid(classifier,0) || !input || !output) {
		return -2;
	}
	//TODO: refactor into a allocateMultidimentional array class
	for(i = 0; i < classifier->classes; i++) {
		weightedSum = 0.0;
		for(j = 0; j < classifier->features; j++) {
			weightedSum += classifier->weights[j][i]*input[j];
		}
		//add in bias
		weightedSum += classifier->weights[j][i];
		output[i] = logisticFunction(weightedSum);
		totalSum += output[i];
	}
	//scale outputs so that they add to 1 and find largests
	max = -1.0;
	for(i = 0; i < classifier->classes; i++) {
		if(output[i] > max) {
			max = output[i];
			j = i;
		}
		output[i] /= totalSum;
	}
	//return index of max
	return j;
}
int batchTrain(LogisticRegressionClassifier* classifier, Sample* samples, 
	unsigned int numberOfSamples, unsigned int maxEpochs,
	double learningRate, unsigned int batchSize) {
	double** gradients;
	double* input;	//sample inputs do not have space for bias input
	double* output;
	double totalError;
	double error;
	double tmp;
	unsigned int i, j, k, epoch, sampleIndex;
	if(!(isValid(classifier, 0) && samples && numberOfSamples && maxEpochs &&
		learningRate > 0 && batchSize)) {
		return -2;
	}
	output = (double*)malloc(sizeof(double*)*classifier->classes);
	input = (double*)malloc(sizeof(double*)*(classifier->features + 1));
	//TODO: clean up the memory managment code (make in like the MNIST loader code)
	//allocate memory for gradient matrix
	gradients = (double**)malloc(sizeof(double*)*(classifier->features + 1));
	if(!gradients || !output || !input) {
		free(output);
		free(input);
		free(gradients);
		fprintf(stderr,"error: could not allocate memory for gradients");
		return -1;
	}
	for(i = 0; i < classifier->features + 1; i++) {
		gradients[i] = (double*)malloc(sizeof(double) * classifier->classes);
		if(!gradients[i]) {
			//free memory allready allocated
			for(j = 0; j < i; j++) {
				free(gradients[j]);
			}
			free(gradients);
			free(output);
			free(input);
			fprintf(stderr,"error: could not allocate memory for gradients");
			return -1;
		}
	}
	//end boiler plate bs
	input[classifier->features] = 1.0;
	printf("Let the training begin\n");
	epoch = 0;
	while(epoch < maxEpochs) {
		//initialize gradients to 0
		for(i = 0; i < classifier->features + 1; i++) {
			for(j = 0; j < classifier->classes; j++) {
				gradients[i][j] = 0.0;
			}
		}
		totalError = 0.0;
		for(i = 0; i < batchSize; i++, sampleIndex = (sampleIndex + 1) % numberOfSamples) {
			//copy over sample input
			for(j = 0; j < classifier->features; j++) {
				input[j] = samples[sampleIndex].inputs[j];
			}
			classify(classifier, input, output);
			//for each possible weight, calculate the gradient/error
			for(j = 0; j < classifier->classes; j++) {
				error = samples[sampleIndex].outputs[j] - output[j];
				totalError += fabs(error);
				for(k = 0; k < classifier->features + 1; k++) {
					//if input is negative, then shift in other direction
					//gradients[k][j] += error * (samples[sampleIndex].inputs[k] > 0 ? 1.0 : -1.0);
					//gradients[k][j] += error * samples[sampleIndex].inputs[k];
					//Activation function is nonlinear, so derivative stuff gets complicated
					tmp = samples[sampleIndex].inputs[k];
					gradients[k][j] += error * logisticFunctionDerivative(tmp)*tmp;
				}
			}
		}
		for(i = 0; i < classifier->features + 1; i++) {
			for(j = 0; j < classifier->classes; j++) {
				classifier->weights[i][j] += learningRate * gradients[i][j];
				//TODO: should I add weight decay
				//weight decay is to penalize large weights which are associated with overfitting
				//classifier->weights[i][j] *= 0.9;//(1 - decay);
			}
		}
		if(epoch % (maxEpochs/ 15) == 0) {
			//TODO: make the output specifiable
			fprintf(stdout,"epoch: %d error: %f\n", epoch, totalError);
		}
		epoch++;
	}
	//free memory when done TODO: refact this into a freeMultidimentionalArray function
	for(i = 0; i < classifier->features + 1; i++) {
		free(gradients[i]);
	}
	free(gradients);
	free(output);
	free(input);
	return 0;
}
void printWeights(FILE* output, LogisticRegressionClassifier* classifier) {
	int i, j;
	if(!output) {
		output = stdout;
	}
	if(!isValid(classifier,0)) {
		return;
	}
	fprintf(output, "Features: %d\nClasses: %d\n", classifier->features, classifier->classes);
	fprintf(output, "Seed: %d\n", classifier->seed);
	fprintf(output, "Weights (rows are input feature, columns are output class, cells are weights\n");
	for(i = 0; i < classifier->features; i++) {
		for(j = 0; j < classifier->classes - 1; j++) {
			fprintf(output, "%6f\t", classifier->weights[i][j]);
		}
		fprintf(output, "%6f\n", classifier->weights[i][j]);
	}
	fprintf(output, "Bias weights\n");
	for(i = 0; i < classifier->classes - 1; i++) {
		fprintf(output, "%6f\t", classifier->weights[classifier->features][i]);
	}
	fprintf(output, "%6f\n", classifier->weights[classifier->features][i]);
}
