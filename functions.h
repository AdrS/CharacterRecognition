#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>

#ifndef __FUNCTIONS_HEADER__
#define __FUNCTIONS_HEADER__

//returns a pseudo random number in from [0, 1]
double random();
double randomInt(double min, double max);
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
