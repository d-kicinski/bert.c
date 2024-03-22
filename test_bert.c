#include <bertc.h>
#include <stdalign.h>

bool test_mm() {
  Tensor2D A = {.dims = {2, 4}, .data = calloc(4, sizeof(tensor_t))};

  Tensor2D B = {.dims = {4, 4}, .data = calloc(16, sizeof(tensor_t))};
  Tensor2D C = {.dims = {2, 4}, .data = calloc(8, sizeof(tensor_t))};

  mm(A, B, &C);

  return true;
}

void bertc_mm() {
  float a_data[384 * 384];
  Tensor2D A = {.dims = {384, 384}, .data = fill_random(a_data, len(a_data))};

  float b_data[384 * 256];
  Tensor2D B = {.dims = {384, 256}, .data = fill_random(b_data, len(b_data))};

  float c_data[384 * 256];
  Tensor2D C = {.dims = {384, 256}, .data = fill_random(c_data, len(c_data))};

  mm(A, B, &C);
}

void bertc_mm_bt() {
  float a_data[384 * 384];
  Tensor2D A = {.dims = {384, 384}, .data = fill_random(a_data, len(a_data))};

  float b_data[256 * 384];
  Tensor2D B = {.dims = {256, 384}, .data = fill_random(b_data, len(b_data))};

  float c_data[384 * 256];
  Tensor2D C = {.dims = {384, 256}, .data = fill_random(c_data, len(c_data))};

  mm_bt(A, B, &C);
}

int main(int argc, char **argv) {
  test_mm();
  bertc_mm();
  bertc_mm_bt();
}
