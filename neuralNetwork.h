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
	double** biasDeltas;
	double*** weights;
	double*** weightDeltas;
	double** activations;	//TODO: set up this
	double (*activationFunction)(double);
	double (*activationFunctionDerivative)(double);
	//this is for intermal memory bookeeping
	void* _allocated;
} NeuralNetwork;

//returns 0 on success, -1 on failure, -2 on invalid params
int createNet(NeuralNetwork* net, unsigned int* sizes, unsigned int layers, ActivationFunctionType type);
//returns 0 on success, -1 on failure
//the neural net stores a pointer to the sizes array passed to createNet
//freeLayerSizes specifies whether that should be freed
//this function assumes that the net was created with createNet
int deleteNet(NeuralNetwork* net, char freeLayerSizes);
void printNet(FILE* out, NeuralNetwork* net, char printDeltas);

//feedForward
//returns 0 on success, -1 on failure, -2 on invalid params
int trainNet(NeuralNetwork* net, Sample* samples, unsigned int numberOfSamples,
	unsigned int epochs, unsigned int batchSize, double learningRate);
int isValidNet(NeuralNetwork* net);
#endif
