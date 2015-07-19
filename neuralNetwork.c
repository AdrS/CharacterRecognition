#include <stdlib.h>
#include "neuralNetwork.h"

int createNet(NeuralNetwork* net, unsigned int* sizes, unsigned int layers, ActivationFunctionType type) {
	unsigned int totalMemForWeights = 0, totalMemForPointers = 0, i, j, k;
	unsigned int totalMemForActivations = 0;
	void* allocated;
	//there must at least be an input and an output layer
	if(!net || !sizes || layers < 2) {
		return -2;
	}
	switch (type) {
		case LOGISTIC:
			net->activationFunction = logisticFunction;
			net->activationFunctionDerivative = logisticFunctionDerivative;
		break;
		case IDENTITY:
			net->activationFunction = identity;
			net->activationFunctionDerivative = identity;
		break;
		case HYPERBOLIC_TANGENT:
			net->activationFunction = tanh;
			net->activationFunctionDerivative = hyperbolicTangentDerivative;
		break;
		default:
			return -2;
	}
	if(sizes[0] < 1) {
		return -2;
	}
	//TODO: check the memory counting code
	//calculate memory requirements
	//memory for input layer
	totalMemForActivations += sizes[0];
	for(i = 1; i < layers; i++) {
		//layers must have at lest one neuron
		if(sizes[i] < 1) {
			return -2;
		}
		//one bias weight for each neuron except those in input layer
		//there are sizes[i] desitination neurons and sizes[i - 1] source neurons
		//==> sizes[i - 1] * sizes[i] weights between them
		//==> total space for ith layer = sizes[i] * (sizes[i - 1] + 1)
		totalMemForWeights += sizes[i] * (sizes[i - 1] + 1);
		//for each source there will be an array of output connection weights
		totalMemForPointers += sizes[i - 1];
		totalMemForActivations += sizes[i];
	}
	//all layers have biases and weight matricies except 1st
	totalMemForPointers += 2*(layers - 1); 
	totalMemForWeights *= 2; //these are doubled because for each weight there is a weight delta
	totalMemForPointers *= 2;
	totalMemForPointers += layers;	//activation vector for each layer

	//all pointers are 1 word, so sizeof(void*) should equal sizeof(double*) and sizeof(double**)
	allocated = malloc((totalMemForWeights + totalMemForActivations)*sizeof(double) +
				totalMemForPointers * sizeof(void*));
	if(!allocated) {
		fprintf(stderr,"error: allocation failed in 'createNet'\n");
		return -1;
	}
	//do not start actual initilization until we know there is memory
	net->layers = layers;
	net->layerSizes = sizes;
	net->_allocated = allocated;

	//divy up memory
	net->biases = allocated;
	allocated += sizeof(double*) * (layers - 1);
	net->biasDeltas = allocated;
	allocated += sizeof(double*) * (layers - 1);
	net->weights = allocated;
	allocated += sizeof(double**) * (layers - 1);
	net->weightDeltas = allocated;
	allocated += sizeof(double**) * (layers - 1);
	net->activations = allocated;
	allocated += sizeof(double*) * layers;
	net->activations[0] = allocated;
	allocated += sizeof(double) * sizes[0];
	for(i = 1; i < layers; i++) {
		//give space for the biases coming into layer i
		net->biases[i - 1] = allocated;
		allocated += sizeof(double) * sizes[i];
		net->biasDeltas[i - 1] = allocated;
		allocated += sizeof(double) * sizes[i];

		//initialize biases
		for(j = 0; j < sizes[i]; j++) {
			net->biases[i - 1][j] = sampleGuassianDistribution(0.0, 1.0);
		}
		//do not bother initializing the deltas (will be done at beggining of each training epoch)
		net->weights[i - 1] = allocated;
		allocated += sizeof(double**) * sizes[i - 1];
		net->weightDeltas[i - 1] = allocated;
		allocated += sizeof(double**) * sizes[i - 1];
		for(j = 0; j < sizes[i - 1]; j++) {
			net->weights[i - 1][j] = allocated;
			allocated += sizeof(double*) * sizes[i];
			net->weightDeltas[i - 1][j] = allocated;
			allocated += sizeof(double*) * sizes[i];
			for(k = 0; k < sizes[i]; k++) {
				net->weights[i - 1][j][k] = sampleGuassianDistribution(0.0, 1.0);
			}
		}
		//do not bother initializing the activations either
		net->activations[i] = allocated;
		allocated += sizeof(double) * sizes[i];
	}
	return 0;
}
int deleteNet(NeuralNetwork* net, char freeLayerSizes) {
	if(!net) {
		//done already!
		return 0;
	}
	if(freeLayerSizes) {
		free(net->layerSizes);
		net->layerSizes = NULL;
	}
	free(net->_allocated);
	net->_allocated = NULL;
	net->biases = NULL;
	net->weights = NULL;
	net->biasDeltas = NULL;
	net->weightDeltas = NULL;
	net->activations = NULL;
	net->activationFunction = NULL;
	net->activationFunctionDerivative = NULL;
	return 0;
}
int isValidNet(NeuralNetwork* net) {
	if(!net || !net->layerSizes || !net->biases || !net->weights || !net->activationFunction
		|| !net->activationFunctionDerivative || !net->biasDeltas || !net->weightDeltas
		|| !net->activations || net->layers < 2) {
		return 0;
	}
	return 1;
}
void initializeDeltas(NeuralNetwork* net) {
	//TODO: this logic would be easier if all deltas were in one coninuous piece of memory
	unsigned int i, j, k;
	if(!isValidNet(net)) {
		return;
	}
	//for each layer except the input
	for(i = 0; i < net->layers - 1; i++) {
		//for each destination neuron
		for(j = 0; j < net->layerSizes[i + 1]; j++) {
			//initialize bias
			net->biasDeltas[i][j] = 0.0;
			//initialize weights for each incomming
			for(k = 0; k < net->layerSizes[i]; k++) {
				net->weightDeltas[i][k][j] = 0.0;
			}
		}
	}
}
void updateWeights(NeuralNetwork* net, double scalar) {
	unsigned int i, j, k;
	//TODO: this is a helper methods, so the checks are unessicary mostly
	//add preprocessor statements to dissable this?
	//same goes for initializeDeltas, and probably other functions
	if(!isValidNet(net)) {
		return;
	}
	//for each layer except the input
	for(i = 0; i < net->layers - 1; i++) {
		//for each destination neuron
		for(j = 0; j < net->layerSizes[i + 1]; j++) {
			//update bias
			net->biases[i][j] -= scalar * net->biasDeltas[i][j];
			//update weights for each incomming
			for(k = 0; k < net->layerSizes[i]; k++) {
				net->weights[i][k][j] -= scalar * net->weightDeltas[i][k][j];
			}
		}
	}
}
int trainNet(NeuralNetwork* net, Sample* samples, unsigned int numberOfSamples,
	unsigned int epochs, unsigned int batchSize, double learningRate) {
	unsigned int epoch = 0;
	if(!samples || epochs < 1 || batchSize < 1 || batchSize > numberOfSamples
		|| learningRate <= 0.0 || !isValidNet(net)) {
		return -2;
	}
	while(epoch < epochs) {
		shuffleSamples(samples, numberOfSamples);
		//for each mini batch
			//set accumulated deltas to 0
			initializeDeltas(net);
			//for each sample
				//update deltas
			//update weights + biases
			//if number of samples is not a multiple of batchSize
			//then the last batch will have a different size
			//when this is fully implemented change the 2nd param to account for this
			updateWeights(net, learningRate/(double)batchSize); //TODO: could use some more testing
		epoch++;
	}
	return 0;
}
double* feedForward(NeuralNetwork* net, double* inputs) {
	unsigned int i, cSize, nSize;
	double* currentLayer;
	double* nextLayer;
	if(!inputs || !isValidNet(net)) {
		return NULL;
	}
	//copy inputs (for later reference)
	for(i = 0; i < net->layerSizes[0]; i++) {
		net->activations[0][i] = inputs[i];
	}
	//TODO: finish this
	//feed forward to each successive layer
	for(i = 1; i < net->layers; i++) {
		currentLayer = net->activations[i - 1];
		nextLayer = net->activations[i];
		cSize = net->layerSizes[i - 1];
		nSize = net->layerSizes[i];
		//TODO: I SCREWED UP THE WEIGHT MATRIX FORMAT SHOULD BE dest src instead of vice verse
		//matrixVectorProduct(net->weights[i - 1], currentLayer, nextLayer, cSize, nSize);
		//add biases wa => wa + b
		add(nextLayer, net->biases[i - 1], nextLayer, nSize);
		//apply activation function wa + b => f(wa + b)
		applyOnEach(nextLayer, nextLayer, net->activationFunction, nSize);
	}
	return net->activations[i - 1];
}
void printNet(FILE* out, NeuralNetwork* net, char printEverything) {
	//TODO: this could be refactored to be shorted and more concise
	//Some kind of a function like printEntry(vector, heading fmt str, args ..)?
	unsigned int i;
	if(!out || !isValidNet(net)) {
		return;
	}
	fprintf(out, "Layers: 1 input, %d hidden, and 1 output\n", net->layers - 2);
	fprintf(out, "Entry i, j of the connect weights table is the weight from\n");
	fprintf(out, "the ith entry of the previous layer the the jth entry of the current layer\n");
	fprintf(out, "Input layer (%d nodes):\n", net->layerSizes[0]);
	if(printEverything) {
		printVector(out, net->activations[0], net->layerSizes[0]);
	}
	fputc('\n', out);
	fputc('\n', out);
	for(i = 1; i < net->layers - 1; i++) {
		fprintf(out, "Hidden layer %d (%d nodes):\n", i, net->layerSizes[i]);
		if(printEverything) {
			printVector(out, net->activations[i], net->layerSizes[i]);
			fputc('\n', out);
		}
		fprintf(out, "Bias weights:\n");
		printVector(out, net->biases[i - 1], net->layerSizes[i]);
		if(printEverything) {
			fprintf(out, "\nBias weight deltas:\n");
			printVector(out, net->biasDeltas[i - 1], net->layerSizes[i]);
		}
		fprintf(out, "\n\nWeights (rows incoming node, cols desination node):\n");
		printMatrix(out, net->weights[i - 1], net->layerSizes[i - 1], net->layerSizes[i]);
		if(printEverything) {
			fprintf(out, "Weight deltas:\n");
			printMatrix(out, net->weightDeltas[i - 1], net->layerSizes[i - 1], net->layerSizes[i]);
		}
	}
	fprintf(out, "\nOutput layer (%d nodes):\n", net->layerSizes[i]);
	if(printEverything) {
		printVector(out, net->activations[i], net->layerSizes[i]);
		fputc('\n', out);
	}
	fprintf(out, "Bias weights:\n");
	printVector(out, net->biases[i - 1], net->layerSizes[i]);
	if(printEverything) {
		fprintf(out, "\nBias weight deltas:\n");
		printVector(out, net->biasDeltas[i - 1], net->layerSizes[i]);
	}
	fprintf(out, "\n\nWeights (rows incoming node, cols desination node):\n");
	printMatrix(out, net->weights[i - 1], net->layerSizes[i - 1], net->layerSizes[i]);
	if(printEverything) {
		fprintf(out, "Weight deltas:\n");
		printMatrix(out, net->weightDeltas[i - 1], net->layerSizes[i - 1], net->layerSizes[i]);
	}
}
