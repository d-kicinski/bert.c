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
#include <stdlib.h>

#define SHAPE_0 128
#define tensor_t float

#define len(a) (sizeof(a) / sizeof(float))

float *fill_random(float *a, size_t sz) {
  for (int i = 0; i < sz; ++i) {
    a[i] = rand() / RAND_MAX;
  }
  return a;
}

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
 *  So this is just a simple matrix multiplication but the B matrix is
 *  assumed to be transposed
 *
 * NOTE:
 *    C has to be zero initialized data structure
 *
 */
void mm_bt(Tensor2D A, Tensor2D B_T, Tensor2D *C) {
  assert(C != NULL);
  assert(A.dims[1] == B_T.dims[1]);
  assert(C->dims[0] == A.dims[0]);
  assert(C->dims[1] == B_T.dims[0]);

  /*
   * i -> X
   * j -> Z
   * k -> Y
   */

  size_t X = A.dims[0];
  size_t Y = A.dims[1];
  size_t Z = B_T.dims[0];

  for (size_t i = 0; i < X; ++i) {
    for (size_t j = 0; j < Z; ++j) {
      tensor_t cumsum = 0;
      for (size_t k = 0; k < Y; ++k) {
        // C->data[i, j] += A.data[i, k] * B.data[j, k];
        size_t a_offset = Y * i + k;
        size_t b_offset = Y * j + k;
        cumsum += *(A.data + a_offset) * *(B_T.data + b_offset);
      }
      size_t c_offset = X * i + j;
      *(C->data + c_offset) += cumsum;
    }
  }
}

/*
 * This comuptes the matrix multiplicaton as follows
 *    C = A @ B
 * where the shapes are exepect as
 *    A[X, Y]
 *    B[Y, Z]
 *    C[X, Z]
 *
 * NOTE:
 *    C has to be zero initialized data structure
 *
 */
void mm(Tensor2D A, Tensor2D B, Tensor2D *C) {
  assert(C != NULL);
  assert(A.dims[1] == B.dims[0]);
  assert(C->dims[0] == A.dims[0]);
  assert(C->dims[1] == B.dims[1]);

  /* variable -> axis along we iterate
   * i -> X
   * j -> y
   * k -> z
   */
  size_t X = A.dims[0];
  size_t Y = A.dims[1];
  size_t Z = B.dims[1];
  for (size_t i = 0; i < X; ++i) {
    for (size_t j = 0; j < Y; ++j) {
      for (size_t k = 0; k < Z; ++k) {
        // C->data[i, k] += A.data[i, j] * B.data[j, k];
        size_t a_offset = Y * i + j;
        size_t b_offset = Z * j + k;
        size_t c_offset = Z * i + k;
        *(C->data + c_offset) += *(A.data + a_offset) * *(B.data + b_offset);
      }
    }
  }
}
