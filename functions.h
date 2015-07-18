#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>

#ifndef __FUNCTIONS_HEADER__
#define __FUNCTIONS_HEADER__

//returns a pseudo random number in [0, 1]
double random();
//returns a pseudo random from the specified range
double randomRange(double min, double max);
//returns a pseudo random from [min, max]
int randomInt(int min, int max);
double sampleGuassianDistribution(double mean, double stdev);
//f(x) = 1/(1 + exp(-x))
double logisticFunction(double x);
//f'(x) = exp(-x)/(1 + exp(-x))^2
double logisticFunctionDerivative(double x);

//(tanh)' = sech^2
//hyperbolic tangent is already implemented in math.h as tanh
double hyperbolicTangentDerivative(double x);

double identity(double x);

#endif
