#include "mnist.h"

uint32_t readAsLittleEndian(uint8_t* buffer) {
	if(!buffer) {
		return 0;
	}
	uint32_t temp = buffer[0];
	temp = (temp << 8) | buffer[1];
	temp = (temp << 8) | buffer[2];
	temp = (temp << 8) | buffer[3];
	return temp;
}
Sample* load(const char* imageFile, const char* labelFile) {
	//MNIST files and file formate at http://yann.lecun.com/exdb/mnist/
	FILE* ifile, *lfile;
	uint8_t buffer[IMAGE_SIZE];
	uint32_t numberOfImages, i, j;
	void* memoryOffset;
	Sample* samples;
	ifile = fopen(imageFile, "rb");
	if(!ifile) {
		fprintf(stderr,"error: could not open file \"%s\"\n", imageFile);
		return NULL;
	}
	lfile = fopen(labelFile, "rb");
	if(!lfile) {
		fprintf(stderr,"error: could not open file \"%s\"\n", labelFile);
		fclose(ifile);
		return NULL;
	}
	//4 bytes magic number
	//4 bytes number of images
	//4 bytes number of rows
	//4 bytes number of columns
	//unsigned bytes all the pixels (28*28 byte each)
	if(fread((void*)buffer, 1, 16, ifile) != 16) {
		fprintf(stderr,"error: could not read headers for image file\n");
		goto error;
	}
	if(readAsLittleEndian(buffer) != 0x00000803) {
		fprintf(stderr,"error: invalid magic number in image file\n");
		goto error;
	}
	numberOfImages = readAsLittleEndian(buffer + 4);
	printf("There are %d images\n", numberOfImages);
	if(readAsLittleEndian(buffer + 8) != 28 || readAsLittleEndian(buffer + 12) != 28) {
		fprintf(stderr,"error: unexpected dimensions for images\n");
		goto error;
	}
	//allocate all space at once, then divy it up
	samples = (Sample*)malloc((sizeof(Sample) + sizeof(double)*(IMAGE_SIZE + 10))*numberOfImages);
	if(!samples) {
		fprintf(stderr,"error: could not allocate memory\n");
		goto error;
	}
	//skip over space for samples
	memoryOffset = (void*)((void*)samples + sizeof(Sample) * numberOfImages);
	for(i = 0; i < numberOfImages; i++) {
		//give each sample some space for input and output vectors
		samples[i].inputs = (double*)memoryOffset;
		memoryOffset += sizeof(double)*IMAGE_SIZE;
		samples[i].outputs = (double*)memoryOffset;
		memoryOffset += sizeof(double)*10;
		//read in sample image
		if(fread((void*)buffer, 1, IMAGE_SIZE, ifile) != IMAGE_SIZE) {
			fprintf(stderr,"error: could not read all images\n");
			goto error;
		}
		//scale the inputs (0 - white 255 - black/ink)
		for(j = 0; j < IMAGE_SIZE; j++) {
			samples[i].inputs[j] = (double)buffer[j]/255.0;
		}
	}
	fclose(ifile);
	//read in the labels
	if(fread((void*)buffer, 1, 8, lfile) != 8) {
		fprintf(stderr,"error: could not read headers for label file\n");
		goto error;
	}
	if(readAsLittleEndian(buffer) != 0x00000801) {
		fprintf(stderr,"error: invalid magic number in label file\n");
		goto error;
	}
	if(readAsLittleEndian(buffer + 4) != numberOfImages) {
		fprintf(stderr, "error: number of labels does not match number of images\n");
		goto error;
	}
	for(i = 0; i < numberOfImages; i++) {
		buffer[0] = fgetc(lfile);
		if(buffer[0] == EOF) {
			fprintf(stderr, "error: unexpected end of file\n");
			goto error;
		}
		if(buffer[0] < 0 || buffer[0] > 9) {
			fprintf(stderr, "error: invalid label\n");
			fprintf(stderr, "file: %s line: %d\n", labelFile, i);
			goto error;
		}
		for(j = 0; j < 10; j++) {
			samples[i].outputs[j] = j == buffer[0] ? 1.0 : 0.0;
		}
	}
	fclose(ifile);
	fclose(lfile);
	return samples;
error:
	fclose(ifile);
	fclose(lfile);
	free(samples);
	return NULL;
}
