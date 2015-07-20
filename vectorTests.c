#include <stdio.h>
#include <string.h>
#include <math.h>
#include "vector.h"

int main() {
	double a[] = {1.0, 2.0, 3.0};
	double b[] = {-3.0, 0.0, 1.0};
	double c[3];
	double v[] = {1.0, 2.0, 3.0, 4.0, 5.0};
	double r1[] = {1.0, -1.0, 1.0/3.0}, r2[] = {0.0, 1.0, 2.0}, r3[] = {1.0, 0.0, 2.0};
	double r4[] = {1.0, 0.0, 2.0}, r5[] = {0.0, -1.0, 0.0};
	double* m[] = {r1, r2, r3, r4, r5};
	double mp[5];
				
				
	printf("a = ");
	printVector(stdout, a, 3);
	printf("\nb = ");
	printVector(stdout, b, 3);
	printf("\ninner product: %f\n", innerProduct(a, b, 3));
	printf("|a| = %f\n", magnitude(a,3));
	scale(a, c, -2.0, 3);
	printf("-2.0a = ");
	printVector(stdout, c, 3);
	printf("\na + b = ");
	add(a, b, c, 3);
	printVector(stdout, c, 3);
	printf("\nb - a = ");
	subtract(a, b, c, 3);
	printVector(stdout, c, 3);
	printf("\nabs(b - a) = ");
	applyOnEach(c, c, fabs, 3);
	printf("\na * b = ");
	hadamardProduct(a, b, c, 3);
	printVector(stdout, c, 3);
	printf("\nm = \n");
	printMatrix(stdout, m, 5, 3);
	printf("\nma = ");
	matrixVectorProduct(m, a, mp, 5, 3);
	printVector(stdout, mp, 5);
	
	printf("\nv = ");
	printVector(stdout, v, 5);
	printf("\nmTv = \n");
	matrixTransposeVectorProduct(m, v, c, 5, 3);
	printVector(stdout, c, 3);
	
	return 0;
}
