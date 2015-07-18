#include <stdio.h>
#include "functions.h"
#include "vector.h"

#ifndef __SAMPLE_HEADER__
#define __SAMPLE_HEADER__

typedef struct {
	double* inputs;
	double* outputs;
} Sample;

void shuffleSamples(Sample* samples, unsigned int number);
//if out is null then stdout is used, returns -2 on invalid params, 0 on success
int printSamples(FILE* out, Sample* samples, unsigned int number, unsigned int inputs, unsigned int outputs);

#endif
