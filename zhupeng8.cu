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

void cpy_and_calculate(float **x, float **y, int rows, int cols) {
  printf("start cpu calculation\n");
  // CPU calculation
  float **cpu_z;
  cpu_malloc(&cpu_z, rows, cols);
  cpu_calculate(x, y, rows, cols, cpu_z);
  printf("debug cpu result\n");
  print_matrix(cpu_z, rows, cols);
  cpu_free(cpu_z, rows);

  printf("start gpu calculation\n");
  // GPU calculation
  float *d_x, *d_y, *d_z;
  printf("cpy\n");

  int elements = rows * cols;
  cudaMalloc((void **)&d_x, elements * sizeof(float));
  cudaMalloc((void **)&d_y, elements * sizeof(float));
  cudaMalloc((void **)&d_z, elements * sizeof(float));

  cudaMemcpy(d_x, x, elements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, elements * sizeof(float), cudaMemcpyHostToDevice);

  float **test;
  cpu_malloc(&test, rows, cols);
  cudaMemcpy(test, d_x, elements * sizeof(float), cudaMemcpyDeviceToHost);

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