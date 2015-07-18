#include "sample.h"

void shuffleSamples(Sample* samples, unsigned int number) {
	if(!samples) {
		return;
	}
}
int printSamples(FILE* out, Sample* samples, unsigned int number, unsigned int inputs, unsigned int outputs) {
	unsigned int i;
	if(!samples || number < 1) {
		return -2;
	}
	if(!out) {
		out = stdout;
	}
	for(i = 0; i < number; i++) {
		//TODO: check return values of all printing functions
		fprintf(out, "Sample %d:\nInputs: ", i + 1);
		printVector(out, samples[i].inputs, inputs);
		fprintf(out, "\nOutputs: ");
		printVector(out, samples[i].outputs, outputs);
		fputc('\n', out);
	}
	return 0;
}
