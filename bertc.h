/*
 * Rules:
 *  - plain C
 *  - no dynamic memory allocation allowed
 *  - only inference pass
 *  - as simple and straightforward as possible
 *  - the code must be written while consuming at least 660ml of beer.
 *  - AI guided coding is allowed but has to be limited to its minium
 */
#include <assert.h>
#include <malloc.h>
#include <stdbool.h>
#include <stdio.h>

#define SHAPE_0 128
#define tensor_t float

typedef struct {
  size_t *dims;
  size_t ndim;
} Shape;

typedef struct {
  tensor_t *data;
  Shape shape;

} Tensor;

typedef struct {
  size_t dims[1];
  tensor_t *data;
} Tensor1D;

typedef struct {
  size_t dims[2];
  tensor_t *data;
} Tensor2D;

typedef struct {
  size_t dims[3];
  tensor_t *data;
} Tensor3D;

/*
 * This comuptes the matrix multiplicaton as follows
 *    C = A @ B
 * where the shapes are exepect as
 *    A[X, Y]
 *    B[Z, Y]
 *    C[X, Z]
 *
 * NOTE:
 *    C has to be zero initialized data structure
 *
 */
void mm(Tensor2D A, Tensor2D B, Tensor2D *C) {
  assert(C != NULL);
  assert(A.dims[1] == B.dims[1]);
  assert(C->dims[0] == A.dims[0]);
  assert(C->dims[1] == B.dims[0]);

  /*
   * i -> X
   * j -> Z
   * k -> Y
   */
  for (size_t i = 0; i < A.dims[0]; ++i) {
    for (size_t j = 0; j < B.dims[0]; ++j) {
      for (size_t k = 0; k < A.dims[1]; ++k) {
        // C->data[i, j] += A.data[i, k] * B.data[j, k];
        size_t a_offset = A.dims[0] * i + k;
        size_t b_offset = B.dims[0] * j + k;
        size_t c_offset = C->dims[0] * i + j;
        *(C->data + c_offset) = *(A.data + a_offset) * *(B.data + b_offset);
      }
    }
  }
}
