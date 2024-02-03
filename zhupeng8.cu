#include <iostream>
#include <unistd.h>
#include <vector>

inline int dimension_convert(int x, int y, int cols) { return x * cols + y; }

void print_matrix(float *m, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%lf,", m[i * cols + j]);
    }
    printf("\n");
  }
}

void gen_matrix(int rows, int cols, float *x, float *y) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      x[i * cols + j] = (float)((i + j) % 100) / 2.0;
      y[i * cols + j] = (float)3.25 * ((i + j) % 100);
    }
  }
}

void cuda_malloc_matrix(float **m, int rows, int cols) {
  cudaMalloc((void ***)&m, rows * sizeof(float *));
  for (int i = 0; i < rows; ++i) {
    cudaMalloc((void **)&(m[i]), cols * sizeof(float));
  }
}

void cuda_h_to_d_memcpy(float **h_m, float **d_m, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    cudaMemcpy(d_m, h_m, cols * sizeof(float), cudaMemcpyHostToDevice);
  }
}

void cuda_d_to_h_memcpy(float **d_m, float **h_m, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    cudaMemcpy(h_m, d_m, cols * sizeof(float), cudaMemcpyDeviceToHost);
  }
}

void cpu_calculate(float *x, float *y, int rows, int cols, float *z) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      z[dimension_convert(i, j, cols)] =
          (i == 0 ? 0 : x[dimension_convert(i, j, cols)]) +
          x[dimension_convert(i, j, cols)] +
          (i == rows - 1 ? 0 : x[dimension_convert(i + 1, j, cols)]) -
          (j < 2 ? 0 : y[dimension_convert(i, j - 2, cols)]) -
          (j < 1 ? 0 : y[dimension_convert(i, j - 1, cols)]) -
          y[dimension_convert(i, j, cols)];
    }
  }
}

void cpy_and_calculate(float *x, float *y, int rows, int cols) {
  // malloc and copy to device
  // float *d_x, *d_y;
  // cuda_malloc_matrix(d_x, rows, cols);
  // cuda_malloc_matrix(d_y, rows, cols);
  // cuda_h_to_d_memcpy(x, d_x, rows, cols);

  float *z = static_cast<float *>(malloc(rows * cols * sizeof(float)));
  cpu_calculate(x, y, rows, cols, z);
  print_matrix(z, rows, cols);
}

int main(int argc, char *argv[]) {

  std::string rows_str, cols_str;
  int rows, cols;

  if ((argc <= 2)) {
    std::cerr << "Error: not enough input for rows and colsumn number.\n";
    return -1;
  } else if (argc > 3) {
    std::cerr << "Error: too many input arguments.\n";
    return -1;
  } else {
    rows_str = argv[1];
    cols_str = argv[2];

    try {
      rows = std::stoi(rows_str);
      cols = std::stoi(cols_str);
    } catch (std::exception &e) {
      std::cerr
          << "Error, failed to convert rows/cols number to integer, error "
             "message:"
          << e.what() << '\n';
      return -1;
    }
  }

  printf("specified rows:%d, cols:%d\n", rows, cols);

  float *x, *y;
  x = static_cast<float *>(malloc(rows * cols * sizeof(float)));
  y = static_cast<float *>(malloc(rows * cols * sizeof(float)));

  gen_matrix(rows, cols, x, y);

  print_matrix(x, rows, cols);
  print_matrix(y, rows, cols);

  return 0;
}