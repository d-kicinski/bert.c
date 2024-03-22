#include <benchmark/benchmark.h>
#include <bertc.h>



static void bertc_mm(benchmark::State &state) {
  float a_data[384 * 384];
  Tensor2D A = {.dims = {384, 384}, .data = fill_random(a_data, len(a_data))};

  float b_data[384 * 256];
  Tensor2D B = {.dims = {384, 256}, .data = fill_random(b_data, len(b_data))};

  float c_data[384 * 256];
  Tensor2D C = {.dims = {384, 256}, .data = fill_random(c_data, len(c_data))};

  for (auto _ : state) {
    mm(A, B, &C);
  }
}

// Similarly, we can define another benchmark
static void bertc_mm_bt(benchmark::State &state) {
  float a_data[384 * 384];
  Tensor2D A = {.dims = {384, 384}, .data = fill_random(a_data, len(a_data))};

  float b_data[256 * 384];
  Tensor2D B = {.dims = {256, 384}, .data = fill_random(b_data, len(b_data))};

  float c_data[384 * 256];
  Tensor2D C = {.dims = {384, 256}, .data = fill_random(c_data, len(c_data))};

  for (auto _ : state) {
    mm_bt(A, B, &C);
  }
}


static void bertc_mm_aligned(benchmark::State &state) {
  alignas(64) float a_data[384 * 384];
  Tensor2D A = {.dims = {384, 384}, .data = fill_random(a_data, len(a_data))};

  alignas(64) float b_data[384 * 256];
  Tensor2D B = {.dims = {384, 256}, .data = fill_random(b_data, len(b_data))};

  alignas(64) float c_data[384 * 256];
  Tensor2D C = {.dims = {384, 256}, .data = fill_random(c_data, len(c_data))};

  for (auto _ : state) {
    mm(A, B, &C);
  }
}

static void bertc_mm_bt_aligned(benchmark::State &state) {
  alignas(64) float a_data[384 * 384];
  Tensor2D A = {.dims = {384, 384}, .data = fill_random(a_data, len(a_data))};

  alignas(64) float b_data[256 * 384];
  Tensor2D B = {.dims = {256, 384}, .data = fill_random(b_data, len(b_data))};

  alignas(64) float c_data[384 * 256];
  Tensor2D C = {.dims = {384, 256}, .data = fill_random(c_data, len(c_data))};

  for (auto _ : state) {
    mm_bt(A, B, &C);
  }
}


BENCHMARK(bertc_mm);
BENCHMARK(bertc_mm_bt);
BENCHMARK(bertc_mm_aligned);
BENCHMARK(bertc_mm_bt_aligned);

BENCHMARK_MAIN();