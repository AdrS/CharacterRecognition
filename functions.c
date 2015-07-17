#include "functions.h"

double random() {
	return (double)rand()/RAND_MAX;
}
double sampleGuassianDistribution(double mean, double stdev) {
	//see https://en.wikipedia.org/wiki/Box-Muller_transform
	static double z1;
	static char leftOvers = 0;
	double u1, u2, z0;
	if(leftOvers) {
		return z1 * stdev + mean;
	}
	do {
		u1 = random();
		u2 = random();
	} while(u1 <= DBL_EPSILON);
	z0 = sqrt(-2.0 * log(u1)) * cos(2 * 3.14159265358979323846 * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(2 * 3.14159265358979323846 * u2);
	return z0 * stdev + mean;
}
