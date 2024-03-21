#include <bertc.h>

bool test_mm() {
  Tensor2D A = {.dims = {2, 4}, .data = calloc(4, sizeof(tensor_t))};

  Tensor2D B = {.dims = {4, 4}, .data = calloc(16, sizeof(tensor_t))};
  Tensor2D C = {.dims = {2, 4}, .data = calloc(8, sizeof(tensor_t))};

  mm(A, B, &C);

  return true;
}

int main(int argc, char **argv) {

  test_mm();
}
