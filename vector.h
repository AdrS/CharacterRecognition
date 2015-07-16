#include <math.h>
#include <stdio.h>
#include <assert.h>

#ifndef __VECTOR_MATH__
#define __VECTOR_MATH__
//GENERAL NOTES:
//all vectors are represented by arrays of doubles and are "column" vectors
//all matricies are arrays of arrays of doubles
//all parameters are assumed to not be null and have the correct dimensions
double innerProduct(double* a, double* b, unsigned int components);
double magnitude(double* a, unsigned int components);
void scale(double* src, double* dest, double scalar, unsigned int components);
void add(double* a, double* b, double* sum, unsigned int components);
//subtracts a from b
void subtract(double* a, double* b, double* difference, unsigned int components);
//v must have cols entries and product must point to rows worth of allocated doubles
void matrixVectorProduct(double** m, double* v, double* product, unsigned int rows, unsigned int cols);
//applies func to each element in src and stores the result in dest
void applyOnEach(double* src, double* dest, double (*func)(double), unsigned int components);
void printVector(FILE* out, double* v, unsigned int components);
void printMatrix(FILE* out, double** m, unsigned int rows, unsigned int cols);

#endif