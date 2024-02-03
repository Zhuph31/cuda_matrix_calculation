#include <iostream>
#include <unistd.h>
#include <vector>

void print_matrix(float **m, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%lf,", m[i][j]);
    }
    printf("\n");
  }
}

void gen_matrix(int rows, int cols, float **x, float **y) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      x[i][j] = (float)((i + j) % 100) / 2.0;
      y[i][j] = (float)3.25 * ((i + j) % 100);
    }
  }
}

void cpu_malloc(float ***m, int rows, int cols) {
  *m = (float **)malloc(rows * sizeof(float *));
  for (size_t i = 0; i < rows; i++) {
    (*m)[i] = (float *)malloc(cols * sizeof(float));
  }
}

void cpu_free(float **m, int rows) {
  for (int i = 0; i < rows; ++i) {
    free(m[i]);
  }
  free(m);
}

void cpu_calculate(float **x, float **y, int rows, int cols, float **z) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      z[i][j] = (i == 0 ? 0 : x[i - 1][j]) + x[i][j] +
                (i == rows - 1 ? 0 : x[i + 1][j]) - (j < 2 ? 0 : y[i][j - 2]) -
                (j < 1 ? 0 : y[i][j - 1]) - y[i][j];
    }
  }
}

void cuda_malloc_matrix(float **m, int rows, int cols) {
  cudaMalloc((void ***)&m, rows * sizeof(float *));
  for (int i = 0; i < rows; ++i) {
    cudaMalloc((void **)&(m[i]), cols * sizeof(float));
  }
}

void cuda_free_matrix(float **m, int rows) {
  for (int i = 0; i < rows; ++i) {
    cudaFree(m[i]);
  }
  cudaFree(m);
}

void cuda_h_to_d_memcpy(float **d_m, float **h_m, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    cudaMemcpy(d_m, h_m, cols * sizeof(float), cudaMemcpyHostToDevice);
  }
}

void cuda_d_to_h_memcpy(float **h_m, float **d_m, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    cudaMemcpy(h_m, d_m, cols * sizeof(float), cudaMemcpyDeviceToHost);
  }
}

void cpy_and_calculate(float **x, float **y, int rows, int cols) {
  printf("start cpu calculation\n");

  // CPU calculation
  float **cpu_z;
  cpu_malloc(&cpu_z, rows, cols);
  cpu_calculate(x, y, rows, cols, cpu_z);
  print_matrix(cpu_z, rows, cols);
  cpu_free(cpu_z, rows);

  // GPU calculation
  float **d_x, **d_y, **d_z;
  cuda_malloc_matrix(d_x, rows, cols);
  cuda_malloc_matrix(d_y, rows, cols);
  cuda_malloc_matrix(d_z, rows, cols);

  cuda_h_to_d_memcpy(d_x, x, rows, cols);
  cuda_h_to_d_memcpy(d_y, y, rows, cols);

  float **test;
  cpu_malloc(&test, rows, cols);

  cuda_d_to_h_memcpy(test, d_x, rows, cols);
  printf("print test\n");
  print_matrix(test, rows, cols);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
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

  float **x, **y;
  cpu_malloc(&x, rows, cols);
  cpu_malloc(&y, rows, cols);

  gen_matrix(rows, cols, x, y);

  print_matrix(x, rows, cols);
  print_matrix(y, rows, cols);

  cpy_and_calculate(x, y, rows, cols);

  cpu_free(x, rows);
  cpu_free(y, rows);

  return 0;
}