/*
 * Rules:
 *  - plain C
 *  - no dynamic memory allocation allowed
 *  - only inference pass
 *  - as simple and straightforward as possible
 *  - the code must be written while consuming at least 660ml of beer.
 *  - AI guided coding is allowed but has to be limited to its minium
 */
#include <stdio.h>

#define SHAPE_0 128

typedef struct {

} shape_2d;

/*
 * This comuptes the matrix multiplicaton as follows
 *    Y = A @ B
 * where the shapes are exepect as
 *    A[X, Y]
 *    B[Z, Y]
 *    Y[X, Z]
 */
void mm(float A[SHAPE_0], float B[SHAPE_0], float C[SHAPE_0]) {
  // do some wild computations
}

#ifdef BERT_IMPLEMENTATION

int main(int argc, char **argv) {

  // Feed forward network
  // Y = X @ W
  //
  // X: [B, A, B];
  // W: [B, B, C];
  // Y: [B, A, C];

  printf("Hello world!\n");
}

#endif
