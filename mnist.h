#include <stdio.h>
#include "sample.h"
#include "stdint.h"
#include "stdlib.h"
#include "string.h"

#ifndef __MNIST_LOADER_HEADER__
#define __MNIST_LOADER_HEADER__
#define IMAGE_SIZE 28*28

//on failure or invalid parmeters returns NULL
Sample* load(const char* imageFile, const char* labelFile);

#endif
