#include <iostream>
#include <unistd.h>
#include <vector>

// #define NUM_BANKS 32
// #define LOG_NUM_BANKS 5
// #define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
int block_size = 512;

#define gpu_err_check(ans) gpu_err_check_impl((ans), __FILE__, __LINE__)
inline void gpu_err_check_impl(cudaError_t code, const char *file, int line,
                               bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s:%d\n", cudaGetErrorString(code), file,
            line);
    if (abort) {
      fflush(stderr);
      exit(code);
    }
  }
}

void flatten_matrix(float **matrix, float **vec, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      (*vec)[i * cols + j] = matrix[i][j];
    }
  }
}

void print_flat_matrix(float *v, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%lf,", v[i * cols + j]);
    }
    printf("\n");
  }
}

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

__global__ void basic_impl(const float *x, const float *y, float *z, int rows,
                           int cols) {
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  int block_offset = block_id * blockDim.x;
  int idx = block_offset + thread_id;

  if (idx < rows * cols) {
    int row = idx / rows, col = idx % rows;

    float elem1 = row == 0 ? 0 : x[(row - 1) * cols + col];
    float elem2 = x[row * cols + col];
    float elem3 = row == rows - 1 ? 0 : x[(row + 1) * cols + col];
    float elem4 = col < 2 ? 0 : y[row * cols + col - 2];
    float elem5 = col < 1 ? 0 : y[row * cols + col - 1];
    float elem6 = y[row * cols + col];
    printf("idx:%d, row:%d, col:%d, x_elem:%lf, y_elem:%lf, "
           "elements:%lf,%lf,%lf,%lf,%lf,%lf\n",
           idx, row, col, x[idx], y[row * cols + col], elem1, elem2, elem3,
           elem4, elem5, elem6);

    z[idx] = elem1 + elem2 + elem3 - elem4 - elem5 - elem6;
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
  printf("cpu result finished\n\n");

  print_matrix(x, rows, cols);
  print_matrix(y, rows, cols);
  printf("\n");

  // flatten matrix for gpu memcpy
  int elements = rows * cols;
  float *x_flat = (float *)malloc(elements * sizeof(float));
  float *y_flat = (float *)malloc(elements * sizeof(float));
  flatten_matrix(x, &x_flat, rows, cols);
  flatten_matrix(y, &y_flat, rows, cols);

  // GPU calculation
  printf("start gpu calculation\n");
  float *d_x, *d_y, *d_z;

  cudaMalloc((void **)&d_x, elements * sizeof(float));
  cudaMalloc((void **)&d_y, elements * sizeof(float));
  cudaMalloc((void **)&d_z, elements * sizeof(float));
  float *h_z = (float *)malloc(elements * sizeof(float));

  gpu_err_check(cudaMemcpy(d_x, x_flat, elements * sizeof(float),
                           cudaMemcpyHostToDevice));
  gpu_err_check(cudaMemcpy(d_y, y_flat, elements * sizeof(float),
                           cudaMemcpyHostToDevice));

  int grid_dim = (elements + block_size - 1) / block_size;
  printf("grid_dim:%d\n", grid_dim);
  basic_impl<<<grid_dim, block_size>>>(d_x, d_y, d_z, rows, cols);

  cudaMemcpy(h_z, d_z, elements * sizeof(float), cudaMemcpyDeviceToHost);
  print_flat_matrix(h_z, rows, cols);
  printf("gpu result finished\n");

  // check results
  float *cpu_res_flat = (float *)malloc(elements * sizeof(float));
  flatten_matrix(cpu_z, &cpu_res_flat, rows, cols);
  for (int idx = 0; idx < elements; ++idx) {
    if (cpu_res_flat[idx] != h_z[idx]) {
      std::cerr << "Error: CPU and GPU result does not match\n";
      exit(-1);
    }
  }
  free(cpu_res_flat);
  free(h_z);
  free(x_flat);
  free(y_flat);

  cpu_free(cpu_z, rows);

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

  printf("\nprint x and y\n");
  print_matrix(x, rows, cols);
  printf("\n");
  print_matrix(y, rows, cols);
  printf("print x and y finished\n\n");

  cpy_and_calculate(x, y, rows, cols);

  cpu_free(x, rows);
  cpu_free(y, rows);

  return 0;
}