#include <stdio.h>
#include "vector.h"
#include "functions.h"
#include "sample.h"

#ifndef __NEURAL_NETWORK__
#define __NEURAL_NETWORK__
typedef enum {
	LOGISTIC,
	IDENTITY,
	HYPERBOLIC_TANGENT
} ActivationFunctionType; 
typedef struct {
	//there must be at least an input and an output layer
	unsigned int layers;
	unsigned int* layerSizes;
	double** biases;
	//TODO: accumulatedBiasDeltas is a better name
	double** biasDeltas;
	double** biasDelta;
	//the rows of each weight matrix are for the destination neuron
	//cols are for inputting neuron
	double*** weights;
	double*** weightDeltas;
	double*** weightDelta;
	double** activations;
	double** preActivations;
	double (*activationFunction)(double);
	double (*activationFunctionDerivative)(double);
	//this is for intermal memory bookeeping
	void* _allocated;
	double* scratchPaper;
} NeuralNetwork;

//returns 0 on success, -1 on failure, -2 on invalid params
int createNet(NeuralNetwork* net, unsigned int* sizes, unsigned int layers, ActivationFunctionType type);
//returns 0 on success, -1 on failure
//the neural net stores a pointer to the sizes array passed to createNet
//freeLayerSizes specifies whether that should be freed
//this function assumes that the net was created with createNet
int deleteNet(NeuralNetwork* net, char freeLayerSizes);
void printNet(FILE* out, NeuralNetwork* net, char printEverything);

//input should be a pointer to layerSizes[0] doubles
//on success a pointer to the activation output layer is returned
//on failure or invalid params, null is returned
double* feedForward(NeuralNetwork* net, double* input);

//returns 0 on success, -1 on failure, -2 on invalid params
int trainNet(NeuralNetwork* net, Sample* samples, unsigned int numberOfSamples,
	unsigned int epochs, unsigned int batchSize, double learningRate);
int isValidNet(NeuralNetwork* net);
//gives initializes all deltas to 0
void initializeDeltas(NeuralNetwork* net);
//this is where the backpropagation takes place, target is desired output
void updateDeltas(NeuralNetwork* net, double* target);
//scalar should be learningRate/batch size
void updateWeights(NeuralNetwork* net, double scalar);

#endif
