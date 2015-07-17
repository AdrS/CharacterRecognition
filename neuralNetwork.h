#include <stdio.h>
#include "vector.h"

#ifndef __NEURAL_NETWORK__
#define __NEURAL_NETWORK__

typedef struct {
	//there must be at least an input and an output layer
	unsigned int layers;
	unsigned int* layerSizes;
	double** biases;
	double*** weights;
	double (*activationFunction)(double);
	//this is for intermal memory bookeeping
	void* _allocated;
} NeuralNetwork;

//returns 0 on success, -1 on failure, -2 on invalid params
int createNet(NeuralNetwork* net, unsigned int* sizes, unsigned int layers);
//returns 0 on success, -1 on failure
//the neural net stores a pointer to the sizes array passed to createNet
//freeLayerSizes specifies whether that should be freed
//this function assumes that the net was created with createNet
int deleteNet(NeuralNetwork* net, char freeLayerSizes);
void printNet(FILE* out, NeuralNetwork* net);

#endif
