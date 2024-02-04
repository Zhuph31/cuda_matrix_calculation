#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

// #define NUM_BANKS 32
// #define LOG_NUM_BANKS 5
// #define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
int block_size = 1024;
int n_stream = 10;

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
    ExecRecord shared_tiling;
    ExecRecord basic_streaming;
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
                           int cols, int stream_id, int stream_elem_offset,
                           int stream_elements) {
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  int block_offset = block_id * blockDim.x;

  if (thread_id >= stream_elements) {
    return;
  }

  int idx = block_offset + thread_id + stream_elem_offset;

  int elements = rows * cols;

  if (stream_elements <= 0 || stream_elem_offset >= elements) {
    return;
  }

#ifdef DEBUG
  if (idx == 0) {
    printf("rows:%d, cols:%d\n", rows, cols);
  }
#endif

  int row = idx / cols, col = idx % cols;

  float elem1 = row == 0 ? 0 : x[(row - 1) * cols + col];
  float elem2 = x[row * cols + col];
  float elem3 = row == rows - 1 ? 0 : x[(row + 1) * cols + col];
  float elem4 = col < 2 ? 0 : y[row * cols + col - 2];
  float elem5 = col < 1 ? 0 : y[row * cols + col - 1];
  float elem6 = y[row * cols + col];

  z[idx] = elem1 + elem2 + elem3 - elem4 - elem5 - elem6;

#ifdef DEBUG
  printf("stream:%d, idx:%d, row:%d, col:%d, x_elem:%lf, y_elem:%lf, "
         "elements:%lf,%lf,%lf,%lf,%lf,%lf, z_elem:%lf\n",
         stream_id, idx, row, col, x[idx], y[idx], elem1, elem2, elem3, elem4,
         elem5, elem6, z[idx]);
#endif
}

__global__ void shared_memory_impl(const float *x, const float *y, float *z,
                                   int rows, int cols, int elements) {
  extern __shared__ float temp[];

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  int block_offset = block_id * blockDim.x;
  int idx = block_offset + thread_id;

  // tiling, 6 times z elements size

  int x1_offset = 0, x2_offset = elements, x3_offset = 2 * elements,
      y1_offset = 3 * elements, y2_offset = 4 * elements,
      y3_offset = 5 * elements;

  if (idx < rows * cols) {
    int row = idx / cols, col = idx % cols;

    // copy data to shared memory
    temp[x1_offset + thread_id] = row == 0 ? 0 : x[(row - 1) * cols + col];
    temp[x2_offset + thread_id] = x[row * cols + col];
    temp[x3_offset + thread_id] =
        row >= rows - 1 ? 0 : x[(row + 1) * cols + col];
    temp[y1_offset + thread_id] = col < 2 ? 0 : y[row * cols + col - 2];
    temp[y2_offset + thread_id] = col < 1 ? 0 : y[row * cols + col - 1];
    temp[y3_offset + thread_id] = y[row * cols + col];

#ifdef DEBUG
    printf("idx:%d, row:%d, col:%d, copy:%lf, %lf, %lf, %lf, %lf, %lf\n", idx,
           row, col, temp[x1_offset + thread_id], temp[x2_offset + thread_id],
           temp[x3_offset + thread_id], temp[y1_offset + thread_id],
           temp[y2_offset + thread_id], temp[y3_offset + thread_id]);
#endif

    // copy back to global memory
    z[idx] = temp[x1_offset + thread_id] + temp[x2_offset + thread_id] +
             temp[x3_offset + thread_id] - temp[y1_offset + thread_id] -
             temp[y2_offset + thread_id] - temp[y3_offset + thread_id];
  }
}

__global__ void shared_tiling_impl(const float *x, const float *y, float *z,
                                   int rows, int cols, int inc_rows,
                                   int inc_cols) {
  extern __shared__ float temp[];

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;

  int tile_id = block_id;
  int total_tile_rows = (rows + inc_rows - 1) / inc_rows,
      total_tile_cols = (cols + inc_cols - 1) / inc_cols;
  int tile_row = tile_id / total_tile_cols,
      tile_col = tile_id % total_tile_cols;

  int tile_inner_row = thread_id / cols,
      tile_inner_col =
          thread_id % cols; // relative row & col inside current tile
  int tile_row_offset = tile_row * inc_rows,
      tile_col_offset = tile_col * inc_cols; // offset of tile begin row and col
  int global_row = tile_row_offset + tile_inner_row,
      global_col =
          tile_col_offset + tile_inner_col; // row and col in global matrix

#ifdef DEBUG
  if (thread_id == 0) {
    printf("block_id:%d, total_tile_rows:%d, total_tile_cols:%d, tile_row:%d, "
           "tile_col:%d, tile_row_offset:%d, tile_col_offset:%d, inc_rows:%d, "
           "inc_cols:%d\n",
           block_id, total_tile_rows, total_tile_cols, tile_row, tile_col,
           tile_row_offset, tile_col_offset, inc_rows, inc_cols);
  }
  __syncthreads();
  if (global_row >= 0 && global_row < rows && global_col >= 0 &&
      global_col < cols) {
    printf("thread_id:%d, tile_inner_row:%d, tile_inner_col:%d, global_row:%d, "
           "global_col:%d\n",
           thread_id, tile_inner_row, tile_inner_col, global_row, global_col);
  }
#endif

  int z_elems = inc_rows * inc_cols;
  int x_offset = 0, x1_offset = z_elems, x2_offset = x1_offset + inc_cols,
      y_offset = x2_offset + inc_cols, y1_offset = y_offset + z_elems,
      y2_offset = y1_offset + inc_rows;

  // copy to shared memory
  if (global_row >= 0 && global_row < rows && global_col >= 0 &&
      global_col < cols) {
    int global_idx = global_row * cols + global_col;
    int tile_inner_idx = tile_inner_row * total_tile_cols + tile_inner_col;

    // copy main matrix
    temp[x_offset + tile_inner_idx] = x[global_idx];
    temp[y_offset + tile_inner_idx] = y[global_idx];

    // copy extra 2 rows
    if (tile_inner_row == 0) {
      temp[x1_offset + tile_inner_idx] =
          global_row == 0 ? 0 : x[(global_row - 1) * cols + global_col];
    } else if (tile_inner_row == inc_rows - 1) {
      temp[x2_offset + tile_inner_idx] =
          global_row == rows - 1 ? 0 : x[(global_row + 1) * cols + global_col];
    }

    // copy extra 2 cols
    if (tile_inner_col == 0) {
      temp[y1_offset + tile_inner_idx] =
          global_col < 2 ? 0 : x[global_row * cols + global_col - 2];
      temp[y2_offset + tile_inner_idx] =
          global_col < 1 ? 0 : x[global_row * cols + global_col - 1];
    }

    // get 6 elements
    float elem1 = tile_inner_row == 0
                      ? temp[x1_offset + tile_inner_col]
                      : temp[x_offset + (tile_inner_row - 1) * total_tile_cols +
                             tile_inner_col];
    float elem2 = temp[x_offset + tile_inner_idx];
    float elem3 = tile_inner_row == inc_rows - 1
                      ? temp[x2_offset + tile_inner_col]
                      : temp[x_offset + (tile_inner_row + 1) * total_tile_cols +
                             tile_inner_col];
    float elem4 = tile_inner_col < 2 ? temp[y1_offset + tile_inner_row]
                                     : temp[y_offset + tile_inner_idx - 2];
    float elem5 = tile_inner_col < 1 ? temp[y2_offset + tile_inner_row]
                                     : temp[y_offset + tile_inner_idx - 1];
    float elem6 = temp[y_offset + tile_inner_idx];

#ifdef DEBUG
    printf(
        "global_idx:%d, global_row:%d, global_col:%d, tile_inner_row:%d, "
        "tile_inner_col:%d, inc_rows:%d, inc_cols:%d, copy:%lf, %lf, %lf, %lf, "
        "%lf, %lf\n",
        global_idx, global_row, global_col, tile_inner_row, tile_inner_col,
        inc_rows, inc_cols, elem1, elem2, elem3, elem4, elem5, elem6);
#endif

    // write directly into global memory
    z[global_idx] = elem1 + elem2 + elem3 - elem4 - elem5 - elem6;
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
      // break;
    }
  }
  printf("\033[1;32mVerifies mode %s, results match.\033[0m\n", mode.c_str());
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

  // basic execution
  {
    // GPU malloc
    float *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_x, elements * sizeof(float));
    cudaMalloc((void **)&d_y, elements * sizeof(float));
    cudaMalloc((void **)&d_z, elements * sizeof(float));
    float *h_z = (float *)malloc(elements * sizeof(float));

    ExecRecord record;
    TimeCost total_gpu_time, cpu_gpu_transfer_time;
    gpu_err_check(cudaMemcpy(d_x, x_flat, elements * sizeof(float),
                             cudaMemcpyHostToDevice));
    gpu_err_check(cudaMemcpy(d_y, y_flat, elements * sizeof(float),
                             cudaMemcpyHostToDevice));
    record.cpu_gpu_transfer_time = cpu_gpu_transfer_time.get_elapsed();

    int grid_dim = (elements + block_size - 1) / block_size;
    TimeCost kernel_time;
    basic_impl<<<grid_dim, block_size>>>(d_x, d_y, d_z, rows, cols, 0, 0,
                                         elements);
    record.kernel_time = kernel_time.get_elapsed();

    TimeCost gpu_cpu_transfer_time;
    cudaMemcpy(h_z, d_z, elements * sizeof(float), cudaMemcpyDeviceToHost);
    record.gpu_cpu_transfer_time = gpu_cpu_transfer_time.get_elapsed();

    record.total_gpu_time = total_gpu_time.get_elapsed();
    record.z_value = h_z[5 * cols + 5];

    records.gpu_records.basic = record;

    check_results(cpu_z, h_z, rows, cols, elements, "basic");

    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
  }

  // shared memory
  // {
  //   // GPU malloc
  //   float *d_x, *d_y, *d_z;
  //   cudaMalloc((void **)&d_x, elements * sizeof(float));
  //   cudaMalloc((void **)&d_y, elements * sizeof(float));
  //   cudaMalloc((void **)&d_z, elements * sizeof(float));
  //   float *h_z = (float *)malloc(elements * sizeof(float));

  //   ExecRecord record;
  //   TimeCost total_gpu_time, cpu_gpu_transfer_time;
  //   gpu_err_check(cudaMemcpy(d_x, x_flat, elements * sizeof(float),
  //                            cudaMemcpyHostToDevice));
  //   gpu_err_check(cudaMemcpy(d_y, y_flat, elements * sizeof(float),
  //                            cudaMemcpyHostToDevice));
  //   record.cpu_gpu_transfer_time = cpu_gpu_transfer_time.get_elapsed();

  //   int grid_dim = (elements + block_size - 1) / block_size;
  //   TimeCost kernel_time;
  //   shared_memory_impl<<<grid_dim, block_size,
  //                        block_size * 6 * sizeof(float)>>>(d_x, d_y, d_z,
  //                        rows,
  //                                                          cols, block_size);
  //   cudaDeviceSynchronize();
  //   check_kernel_err();
  //   record.kernel_time = kernel_time.get_elapsed();

  //   TimeCost gpu_cpu_transfer_time;
  //   cudaMemcpy(h_z, d_z, elements * sizeof(float), cudaMemcpyDeviceToHost);
  //   record.gpu_cpu_transfer_time = gpu_cpu_transfer_time.get_elapsed();

  //   record.total_gpu_time = total_gpu_time.get_elapsed();
  //   record.z_value = h_z[5 * cols + 5];

  //   records.gpu_records.shared_memory = record;

  //   check_results(cpu_z, h_z, rows, cols, elements, "shared");

  //   free(h_z);
  //   cudaFree(d_x);
  //   cudaFree(d_y);
  //   cudaFree(d_z);
  // }

  // basic + streaming memcpy
  {
    // GPU malloc
    float *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_x, elements * sizeof(float));
    cudaMalloc((void **)&d_y, elements * sizeof(float));
    cudaMalloc((void **)&d_z, elements * sizeof(float));
    float *h_z = (float *)malloc(elements * sizeof(float));

    // pin host side memroy
    cudaHostRegister(x_flat, elements * sizeof(float), 0);
    cudaHostRegister(y_flat, elements * sizeof(float), 0);
    cudaHostRegister(h_z, elements * sizeof(float), 0);

    ExecRecord record;
    TimeCost total_gpu_time, cpu_gpu_transfer_time;

    cudaStream_t stream[n_stream + 1];
    std::vector<int> streams_begin_elem_offset(n_stream + 1, 0),
        streams_elements(n_stream + 1, 0),
        streams_begin_byte_offset(n_stream + 1, 0),
        streams_bytes(n_stream + 1, 0);

    printf("starting multiple streams\n");

    int elements_per_stream = (elements + n_stream - 1) / n_stream;
    printf("elements per stream:%d\n", elements_per_stream);
    // elements_per_stream = elements_per_stream < 100 ? 100 :
    // elements_per_stream;

    // start streams & copy
    for (int i = 1; i <= n_stream; ++i) {
      cudaStreamCreate(&stream[i]);
      // printf("stream %d created\n", i);

      int begin_elem_offset = (i - 1) * elements_per_stream;

      if (begin_elem_offset >= elements) {
        // printf("abort creating stream %d cause not needed\n", i);
        continue;
      }

      int cur_stream_elements = elements_per_stream;
      if (begin_elem_offset + cur_stream_elements >= elements) {
        cur_stream_elements = elements - begin_elem_offset;
      }

      int begin_byte_offset = begin_elem_offset * sizeof(float),
          cur_stream_bytes = cur_stream_elements * sizeof(float);
      streams_begin_elem_offset[i] = begin_elem_offset;
      streams_begin_byte_offset[i] = begin_byte_offset;
      streams_elements[i] = cur_stream_elements;
      streams_bytes[i] = cur_stream_bytes;

      cudaMemcpyAsync(&(d_x[begin_elem_offset]), &(x_flat[begin_elem_offset]),
                      cur_stream_bytes, cudaMemcpyHostToDevice, stream[i]);
      cudaMemcpyAsync(&(d_y[begin_elem_offset]), &(y_flat[begin_elem_offset]),
                      cur_stream_bytes, cudaMemcpyHostToDevice, stream[i]);

#ifdef DEBUG
      printf("stream:%d, begin_elem_offset:%d, cur_stream_elements:%d, "
             "begin_byte_offset:%d, cur_stream_bytes:%d\n",
             i, begin_elem_offset, cur_stream_elements, begin_byte_offset,
             cur_stream_bytes);
#endif
    }

    for (int i = 0; i <= n_stream; ++i) {
      // printf("Synchronizing stream %d\n", i);
      cudaStreamSynchronize(stream[i]);
    }

    for (int i = 1; i <= n_stream; i++) {
      int grid_dim = (elements_per_stream + block_size - 1) / block_size;
      basic_impl<<<grid_dim, block_size, 0, stream[i]>>>(
          d_x, d_y, d_z, rows, cols, i, streams_begin_elem_offset[i],
          streams_elements[i]);
      cudaMemcpyAsync(&h_z[streams_begin_elem_offset[i]],
                      &d_z[streams_begin_elem_offset[i]], streams_bytes[i],
                      cudaMemcpyDeviceToHost, stream[i]);
    }

    for (int i = 0; i <= 10; i++) {
      // printf("Synchronizing stream %d\n", i);
      cudaStreamSynchronize(stream[i]);
    }

    record.total_gpu_time = total_gpu_time.get_elapsed();
    record.z_value = h_z[5 * cols + 5];

    records.gpu_records.basic_streaming = record;

    check_results(cpu_z, h_z, rows, cols, elements, "basic_streaming");

    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
  }

  // shared memory + tiling
  // {
  //   // GPU malloc
  //   float *d_x, *d_y, *d_z;
  //   cudaMalloc((void **)&d_x, elements * sizeof(float));
  //   cudaMalloc((void **)&d_y, elements * sizeof(float));
  //   cudaMalloc((void **)&d_z, elements * sizeof(float));
  //   float *h_z = (float *)malloc(elements * sizeof(float));

  //   ExecRecord record;
  //   TimeCost total_gpu_time, cpu_gpu_transfer_time;
  //   gpu_err_check(cudaMemcpy(d_x, x_flat, elements * sizeof(float),
  //                            cudaMemcpyHostToDevice));
  //   gpu_err_check(cudaMemcpy(d_y, y_flat, elements * sizeof(float),
  //                            cudaMemcpyHostToDevice));
  //   record.cpu_gpu_transfer_time = cpu_gpu_transfer_time.get_elapsed();

  //   int grid_dim = (elements + block_size - 1) / block_size;
  //   TimeCost kernel_time;

  //   // ? problem with irrational blockDim
  //   int inc_rows = sqrt(block_size), inc_cols = sqrt(block_size);
  //   inc_rows = inc_rows > rows ? rows : inc_rows;
  //   inc_cols = inc_cols > cols ? cols : inc_cols;
  //   shared_tiling_impl<<<grid_dim, block_size,
  //                        block_size * 6 * sizeof(float)>>>(
  //       d_x, d_y, d_z, rows, cols, inc_rows, inc_cols);
  //   // cudaDeviceSynchronize();
  //   check_kernel_err();
  //   record.kernel_time = kernel_time.get_elapsed();

  //   TimeCost gpu_cpu_transfer_time;
  //   cudaMemcpy(h_z, d_z, elements * sizeof(float), cudaMemcpyDeviceToHost);
  //   record.gpu_cpu_transfer_time = gpu_cpu_transfer_time.get_elapsed();

  //   record.total_gpu_time = total_gpu_time.get_elapsed();
  //   record.z_value = h_z[5 * cols + 5];

  //   records.gpu_records.shared_tiling = record;

  //   check_results(cpu_z, h_z, rows, cols, elements, "shared_tiling");

  //   free(h_z);
  //   cudaFree(d_x);
  //   cudaFree(d_y);
  //   cudaFree(d_z);
  // }

  free(x_flat);
  free(y_flat);

  cpu_free(cpu_z, rows);

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
  // records.gpu_records.shared_memory.print();
  // records.gpu_records.shared_tiling.print();
  records.gpu_records.basic_streaming.print();

  cpu_free(x, rows);
  cpu_free(y, rows);

  return 0;
}