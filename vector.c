#include "vector.h"

double innerProduct(double* a, double* b, unsigned int components) {
	double sum = 0.0;
	unsigned int i;
	assert(a && b);
	for(i = 0; i < components; i++) {
		sum += a[i] * b[i];
	}
	return sum;
}
double magnitude(double* a, unsigned int components) {
	return sqrt(innerProduct(a, a, components));
}
void scale(double* src, double* dest, double scalar, unsigned int components) {
	unsigned int i;
	assert(src && dest);
	for(i = 0; i < components; i++) {
		dest[i] = src[i] * scalar;
	}
}
void add(double* a, double* b, double* sum, unsigned int components) {
	unsigned int i;
	assert(a && b && sum);
	for(i = 0; i < components; i++) {
		sum[i] = a[i] + b[i];
	}
}
void subtract(double* a, double* b, double* difference, unsigned int components) {
	unsigned int i;
	assert(a && b && difference);
	for(i = 0; i < components; i++) {
		difference[i] = b[i] - a[i];
	}
}
void hadamardProduct(double* a, double* b, double* product, unsigned int components) {
	unsigned int i;
	assert(a && b && product);
	for(i = 0; i < components; i++) {
		product[i] = a[i] * b[i];
	}
}
void matrixVectorProduct(double** m, double* v, double* product, unsigned int rows, unsigned int cols) {
	unsigned int i, j;
	double sum;
	assert(m && v && product);
	for(i = 0; i < rows; i++) {
		assert(m[i]);
		sum = 0.0;
		for(j = 0; j < cols; j++) {
			sum += m[i][j] * v[j];
		}
		product[i] = sum;
	}
}
void matrixTransposeVectorProduct(double** m, double* v, double* product, unsigned int rows, unsigned int cols) {
	//rows and cols are the dimensions of the original matrix, so mT is a cols x rows matrix
	//v must have rows components then and product must have cols components
	unsigned int i, j;
	double sum;
	assert(m && v && product);
	for(i = 0; i < cols; i++) {
		//assert(m[i]);
		sum = 0.0;
		for(j = 0; j < rows; j++) {
			sum += m[j][i] * v[j];
		}
		product[i] = sum;
	}
}
void applyOnEach(double* src, double* dest, double (*func)(double), unsigned int components) {
	unsigned int i;
	assert(src && dest && func);
	for(i = 0; i < components; i++) {
		dest[i] = func(src[i]);
	}
}
void printVector(FILE* out, double* v, unsigned int components) {
	//TODO: should I return number of characters written???
	unsigned int i;
	assert(out && v);
	fputc('[', out);
	for(i = 0; i < components - 1; i++) {
		fprintf(out, "%f ", v[i]);
	}
	if(components > 0) {
		fprintf(out, "%f", v[i]);
	}
	fputc(']', out);
}
void printMatrix(FILE* out, double** m, unsigned int rows, unsigned int cols) {
	unsigned int i, j;
	assert(out && m);
	fputc('[', out);
	//print first row
	if(rows > 0) {
		assert(m[0]);
		fputc('[', out);
		for(i = 0; i < cols - 1; i++) {
			fprintf(out, "%f ", m[0][i]);
		}
		if(cols > 0) {
			fprintf(out, "%f", m[0][i]);
		}
		fputc(']', out);
	}
	//print all other rows indented by a space
	for(i = 1; i < rows; i++) {
		assert(m[i]);
		fputs("\n [", out);
		for(j = 0; j < cols - 1; j++) {
			fprintf(out, "%f ", m[i][j]);
		}
		if(cols > 0) {
			fprintf(out, "%f", m[i][j]);
		}
		fputc(']', out);
	}
	fputc(']', out);
	fputc('\n', out);
}
