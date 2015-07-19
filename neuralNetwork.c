#include <stdlib.h>
#include "neuralNetwork.h"

int createNet(NeuralNetwork* net, unsigned int* sizes, unsigned int layers, ActivationFunctionType type) {
	unsigned int totalMemForWeights = 0, totalMemForPointers = 0, i, j, k;
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
	}
	//all layers have biases and weight matricies except 1st
	totalMemForPointers += 2*(layers - 1); 
	totalMemForWeights *= 2; //these are doubled because for each weight there is a weight delta
	totalMemForPointers *= 2;

	//TODO: #######################################################################	
	//	allocate memory for activations
	//TODO: #######################################################################	
	//all pointers are 1 word, so sizeof(void*) should equal sizeof(double*) and sizeof(double**)
	allocated = malloc(totalMemForWeights*sizeof(double) + totalMemForPointers * sizeof(void*));
	if(!allocated) {
		fprintf(stderr,"error: allocation failed in 'createNet'\n");
		return -1;
	}
	//do not start actual initilization until we know there is memory
	net->layers = layers;
	net->layerSizes = sizes;
	net->_allocated = allocated;

	//divy up memory for all layers with inputs
	net->biases = allocated;
	allocated += sizeof(double*) * (layers - 1);
	net->biasDeltas = allocated;
	allocated += sizeof(double*) * (layers - 1);
	net->weights = allocated;
	allocated += sizeof(double**) * (layers - 1);
	net->weightDeltas = allocated;
	allocated += sizeof(double**) * (layers - 1);
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
	net->activationFunction = NULL;
	net->activationFunctionDerivative = NULL;
	return 0;
}
int isValidNet(NeuralNetwork* net) {
	if(!net || !net->layerSizes || !net->biases || !net->weights || !net->activationFunction
		|| !net->activationFunctionDerivative || !net->biasDeltas || !net->weightDeltas
		|| !net->activations) {
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
		epoch++;
	}
	return 0;
}
void printNet(FILE* out, NeuralNetwork* net, char printDeltas) {
	unsigned int i;
	if(!out || !isValidNet(net)) {
		return;
	}
	fprintf(out, "Layers: 1 input, %d hidden, and 1 output\n", net->layers - 2);
	fprintf(out, "Entry i, j of the connect weights table is the weight from\n");
	fprintf(out, "the ith entry of the previous layer the the jth entry of the current layer\n");
	fprintf(out, "Input layer (%d nodes):\n\n", net->layerSizes[0]);
	for(i = 1; i < net->layers - 1; i++) {
		fprintf(out, "Hidden layer %d (%d nodes):\n", i, net->layerSizes[i]);
		fprintf(out, "Bias weights:\n");
		printVector(out, net->biases[i - 1], net->layerSizes[i]);
		if(printDeltas) {
			fprintf(out, "\nBias weight deltas:\n");
			printVector(out, net->biasDeltas[i - 1], net->layerSizes[i]);
		}
		fprintf(out, "\n\nWeights (rows incoming node, cols desination node):\n");
		printMatrix(out, net->weights[i - 1], net->layerSizes[i - 1], net->layerSizes[i]);
		if(printDeltas) {
			fprintf(out, "Weight deltas:\n");
			printMatrix(out, net->weightDeltas[i - 1], net->layerSizes[i - 1], net->layerSizes[i]);
		}
	}
	fprintf(out, "\nOutput layer (%d nodes):\n", net->layerSizes[i]);
	fprintf(out, "Bias weights:\n");
	printVector(out, net->biases[i - 1], net->layerSizes[i]);
	if(printDeltas) {
		fprintf(out, "\nBias weight deltas:\n");
		printVector(out, net->biasDeltas[i - 1], net->layerSizes[i]);
	}
	fprintf(out, "\n\nWeights (rows incoming node, cols desination node):\n");
	printMatrix(out, net->weights[i - 1], net->layerSizes[i - 1], net->layerSizes[i]);
	if(printDeltas) {
		fprintf(out, "Weight deltas:\n");
		printMatrix(out, net->weightDeltas[i - 1], net->layerSizes[i - 1], net->layerSizes[i]);
	}
}
