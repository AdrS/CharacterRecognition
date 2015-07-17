#include <stdio.h>
#include "neuralNetwork.h"

int main() {
	unsigned int layerSizes[] = {2, 4, 7, 1};
	NeuralNetwork nn;
	createNet(&nn, layerSizes, 4);
	printNet(stdout, &nn);
	deleteNet(&nn, 0);
	return 0;
}
