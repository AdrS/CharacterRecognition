#include <stdio.h>
#include "neuralNetwork.h"

int main() {
	unsigned int layerSizes[] = {2, 4, 7, 1};
	int ret;
	NeuralNetwork nn;
	ret = createNet(&nn, layerSizes, 4, LOGISTIC);
	if(ret) {
		printf("could not create net\n");
		return ret;
	}
	printNet(stdout, &nn);
	deleteNet(&nn, 0);
	return 0;
}
