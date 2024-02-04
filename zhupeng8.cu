#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

// #define NUM_BANKS 32
// #define LOG_NUM_BANKS 5
// #define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
int block_size = 1024;

class TimeCost {
  double get_timestamp() const {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
  }

  double start_ts;

public:
  TimeCost() { start_ts = get_timestamp(); }
  double get_elapsed() const { return get_timestamp() - start_ts; }
};

struct ExecRecord {
  double total_gpu_time;
  double cpu_gpu_transfer_time;
  double kernel_time;
  double gpu_cpu_transfer_time;
  float z_value;
  void print() const {
    printf("%.6f %.6f %.6f %.6f %.6f\n", total_gpu_time, cpu_gpu_transfer_time,
           kernel_time, gpu_cpu_transfer_time, z_value);
  }
};

struct ExecRecords {
  double cpu_record;
  struct GPURecords {
    ExecRecord basic;
    ExecRecord shared_memory;
  } gpu_records;
};

inline void check_kernel_err() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: kernel invoke failed, %s\n",
            cudaGetErrorString(err));
    exit(-1);
  }
}

#define gpu_err_check(ans) gpu_err_check_impl((ans), __FILE__, __LINE__)
inline void gpu_err_check_impl(cudaError_t code, const char *file, int line,
                               bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "Error: cuda func failed, %s %s:%d\n",
            cudaGetErrorString(code), file, line);
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

void compare_flat_matrix(float *l, float *r, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (l[i * cols + j] != r[i * cols + j]) {
        printf("\033[1;31m%.6f/%.6f,\033[0m", l[i * cols + j], r[i * cols + j]);
      } else {
        printf("\033[1;32m%.6f,\033[0m", l[i * cols + j]);
      }
    }
    printf("\n");
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
    int row = idx / cols, col = idx % cols;

    float elem1 = row == 0 ? 0 : x[(row - 1) * cols + col];
    float elem2 = x[row * cols + col];
    float elem3 = row == rows - 1 ? 0 : x[(row + 1) * cols + col];
    float elem4 = col < 2 ? 0 : y[row * cols + col - 2];
    float elem5 = col < 1 ? 0 : y[row * cols + col - 1];
    float elem6 = y[row * cols + col];

#ifdef DEBUG
    printf("idx:%d, row:%d, col:%d, x_elem:%lf, y_elem:%lf, "
           "elements:%lf,%lf,%lf,%lf,%lf,%lf\n",
           idx, row, col, x[idx], y[row * cols + col], elem1, elem2, elem3,
           elem4, elem5, elem6);
#endif

    z[idx] = elem1 + elem2 + elem3 - elem4 - elem5 - elem6;
  }
}

__global__ void shared_memory_impl(const float *x, const float *y, float *z,
                                   int rows, int cols, int block_elems) {
  extern __shared__ int temp[];

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  // int block_offset = block_id * blockDim.x;
  int block_elem_offset = block_elems * block_id;
  int idx = block_elem_offset + thread_id;

  int x_offset = 0, y_offset = block_elems, z_offset = 2 * block_elems;

  // only block elems number of threads, and idx cannot exceed total elements
  if (thread_id < block_elems && idx < rows * cols) {
    // copy data to shared memory
    temp[x_offset + thread_id] = x[idx];
    temp[y_offset + thread_id] = y[idx];

    // calculate using shared memory
    int row = idx / cols, col = idx % cols;

    float elem1 =
        row == 0 ? 0
                 : temp[x_offset + (row - 1) * cols + col - block_elem_offset];
    float elem2 = temp[x_offset + row * cols + col - block_elem_offset];
    float elem3 =
        row == rows - 1
            ? 0
            : temp[x_offset + (row + 1) * cols + col - block_elem_offset];
    float elem4 =
        col < 2 ? 0 : temp[y_offset + row * cols + col - 2 - block_elem_offset];
    float elem5 =
        col < 1 ? 0 : temp[y_offset + row * cols + col - 1 - block_elem_offset];
    float elem6 = temp[y_offset + row * cols + col - block_elem_offset];

    temp[z_offset + idx - block_elem_offset] =
        elem1 + elem2 + elem3 - elem4 - elem5 - elem6;

    // copy back to globak memory
    z[idx] = temp[z_offset + idx - block_elem_offset];
  }
}

double cpu_cal_and_record(float **x, float **y, int rows, int cols,
                          float ***cpu_z) {
  TimeCost cpu_tc;
  cpu_malloc(cpu_z, rows, cols);
  cpu_calculate(x, y, rows, cols, *cpu_z);
  return cpu_tc.get_elapsed();
}

void check_results(float **cpu_z, float *h_z, int rows, int cols, int elements,
                   const std::string &mode) {
  float *cpu_res_flat = (float *)malloc(elements * sizeof(float));
  flatten_matrix(cpu_z, &cpu_res_flat, rows, cols);
  for (int idx = 0; idx < elements; ++idx) {
    if (cpu_res_flat[idx] != h_z[idx]) {
      printf("\033[1;31mError: mode %s CPU and GPU result does not "
             "match\n\033[0m\n",
             mode.c_str());

#ifdef DEBUG
      printf("debug mode\n");
      compare_flat_matrix(cpu_res_flat, h_z, rows, cols);
#endif
      exit(-1);
    }
  }
  printf("\033[1;32mVerifies, results match.\033[0m\n");
  free(cpu_res_flat);
}

ExecRecords calculate_and_compare(float **x, float **y, int rows, int cols) {
  ExecRecords records;

  float **cpu_z;
  records.cpu_record = cpu_cal_and_record(x, y, rows, cols, &cpu_z);

  // flatten matrix for gpu memcpy
  int elements = rows * cols;
  float *x_flat = (float *)malloc(elements * sizeof(float));
  float *y_flat = (float *)malloc(elements * sizeof(float));
  flatten_matrix(x, &x_flat, rows, cols);
  flatten_matrix(y, &y_flat, rows, cols);

  // GPU malloc
  float *d_x, *d_y, *d_z;
  cudaMalloc((void **)&d_x, elements * sizeof(float));
  cudaMalloc((void **)&d_y, elements * sizeof(float));
  cudaMalloc((void **)&d_z, elements * sizeof(float));
  float *h_z = (float *)malloc(elements * sizeof(float));

  // basic execution
  {
    ExecRecord record;
    TimeCost total_gpu_time, cpu_gpu_transfer_time;
    gpu_err_check(cudaMemcpy(d_x, x_flat, elements * sizeof(float),
                             cudaMemcpyHostToDevice));
    gpu_err_check(cudaMemcpy(d_y, y_flat, elements * sizeof(float),
                             cudaMemcpyHostToDevice));
    record.cpu_gpu_transfer_time = cpu_gpu_transfer_time.get_elapsed();

    int grid_dim = (elements + block_size - 1) / block_size;
    printf("grid_dim:%d\n", grid_dim);
    TimeCost kernel_time;
    basic_impl<<<grid_dim, block_size>>>(d_x, d_y, d_z, rows, cols);
    check_kernel_err();
    cudaDeviceSynchronize();
    record.kernel_time = kernel_time.get_elapsed();

    TimeCost gpu_cpu_transfer_time;
    cudaMemcpy(h_z, d_z, elements * sizeof(float), cudaMemcpyDeviceToHost);
    record.gpu_cpu_transfer_time = gpu_cpu_transfer_time.get_elapsed();

    record.total_gpu_time = total_gpu_time.get_elapsed();
    record.z_value = h_z[5 * cols + 5];

    records.gpu_records.basic = record;

    check_results(cpu_z, h_z, rows, cols, elements, "basic");
  }

  // shared memory
  {
    ExecRecord record;
    TimeCost total_gpu_time, cpu_gpu_transfer_time;
    gpu_err_check(cudaMemcpy(d_x, x_flat, elements * sizeof(float),
                             cudaMemcpyHostToDevice));
    gpu_err_check(cudaMemcpy(d_y, y_flat, elements * sizeof(float),
                             cudaMemcpyHostToDevice));
    record.cpu_gpu_transfer_time = cpu_gpu_transfer_time.get_elapsed();

    int grid_dim = (elements + block_size - 1) / block_size;
    TimeCost kernel_time;
    shared_memory_impl<<<grid_dim, block_size>>>(d_x, d_y, d_z, rows, cols,
                                                 block_size / 3);
    cudaDeviceSynchronize();
    record.kernel_time = kernel_time.get_elapsed();

    TimeCost gpu_cpu_transfer_time;
    cudaMemcpy(h_z, d_z, elements * sizeof(float), cudaMemcpyDeviceToHost);
    record.gpu_cpu_transfer_time = gpu_cpu_transfer_time.get_elapsed();

    record.total_gpu_time = total_gpu_time.get_elapsed();
    record.z_value = h_z[5 * cols + 5];

    records.gpu_records.shared_memory = record;

    check_results(cpu_z, h_z, rows, cols, elements, "basic");
  }

  free(h_z);
  free(x_flat);
  free(y_flat);

  cpu_free(cpu_z, rows);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);

  return records;
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

#ifdef DEBUG
  printf("\nprint x and y\n");
  print_matrix(x, rows, cols);
  printf("\n");
  print_matrix(y, rows, cols);
  printf("print x and y finished\n\n");
#endif

  ExecRecords records = calculate_and_compare(x, y, rows, cols);

  printf("%.6f\n", records.cpu_record);
  records.gpu_records.basic.print();
  records.gpu_records.shared_memory.print();

  cpu_free(x, rows);
  cpu_free(y, rows);

  return 0;
}