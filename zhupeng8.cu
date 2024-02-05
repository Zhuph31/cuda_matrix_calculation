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
  ExecRecord() {
    total_gpu_time = -1;
    cpu_gpu_transfer_time = -1;
    kernel_time = -1;
    gpu_cpu_transfer_time = -1;
    z_value = -1;
  }
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

  int elements = rows * cols;
  if (thread_id >= stream_elements || stream_elements <= 0 ||
      stream_elem_offset >= elements) {
    if (thread_id == 0) {
      printf("stream id:%d, skip kernel execution\n", stream_id);
    }
    return;
  }

  int idx = block_offset + thread_id +
            stream_elem_offset; // idx in flattened global memory

#ifdef DEBUG
  if (idx == 0) {
    printf("rows:%d, cols:%d\n", rows, cols);
  }
  if (thread_id == 0) {
    printf("stream %d executing kernel\n", stream_id);
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
  printf(
      "basic debug stream:%d, idx:%d, row:%d, col:%d, x_elem:%lf, y_elem:%lf, "
      "elements:%lf,%lf,%lf,%lf,%lf,%lf, z_elem:%lf\n",
      stream_id, idx, row, col, x[idx], y[idx], elem1, elem2, elem3, elem4,
      elem5, elem6, z[idx]);
#endif
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
  float *x_flat, *y_flat;
  cudaMallocHost((void **)&x_flat, elements * sizeof(float),
                 cudaHostAllocWriteCombined);
  cudaMallocHost((void **)&y_flat, elements * sizeof(float),
                 cudaHostAllocWriteCombined);
  flatten_matrix(x, &x_flat, rows, cols);
  flatten_matrix(y, &y_flat, rows, cols);

  // basic execution
  // {
  //   // GPU malloc
  //   float *d_x, *d_y, *d_z;
  //   cudaMalloc((void **)&d_x, elements * sizeof(float));
  //   cudaMalloc((void **)&d_y, elements * sizeof(float));
  //   cudaMalloc((void **)&d_z, elements * sizeof(float));

  //   float *h_z;
  //   cudaMallocHost((void **)&h_z, elements * sizeof(float),
  //                  cudaHostAllocWriteCombined);

  //   ExecRecord record;
  //   TimeCost total_gpu_time, cpu_gpu_transfer_time;
  //   gpu_err_check(cudaMemcpy(d_x, x_flat, elements * sizeof(float),
  //                            cudaMemcpyHostToDevice));
  //   gpu_err_check(cudaMemcpy(d_y, y_flat, elements * sizeof(float),
  //                            cudaMemcpyHostToDevice));
  //   record.cpu_gpu_transfer_time = cpu_gpu_transfer_time.get_elapsed();

  //   int grid_dim = (elements + block_size - 1) / block_size;
  //   TimeCost kernel_time;
  //   basic_impl<<<grid_dim, block_size>>>(d_x, d_y, d_z, rows, cols, 0, 0,
  //                                        elements);
  //   record.kernel_time = kernel_time.get_elapsed();

  //   TimeCost gpu_cpu_transfer_time;
  //   cudaMemcpy(h_z, d_z, elements * sizeof(float), cudaMemcpyDeviceToHost);
  //   record.gpu_cpu_transfer_time = gpu_cpu_transfer_time.get_elapsed();

  //   record.total_gpu_time = total_gpu_time.get_elapsed();
  //   record.z_value = h_z[5 * cols + 5];

  //   records.gpu_records.basic = record;

  //   check_results(cpu_z, h_z, rows, cols, elements, "basic");

  //   cudaFree(d_z);
  //   cudaFree(d_x);
  //   cudaFree(d_y);
  //   cudaFree(h_z);
  // }

  // basic + streaming memcpy
  {
    // GPU malloc
    float *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_x, elements * sizeof(float));
    cudaMalloc((void **)&d_y, elements * sizeof(float));
    cudaMalloc((void **)&d_z, elements * sizeof(float));

    float *h_z;
    cudaMallocHost((void **)&h_z, elements * sizeof(float),
                   cudaHostAllocWriteCombined);

    ExecRecord record;

    std::vector<int> streams_begin_elem_offset(n_stream + 1, 0),
        streams_elements(n_stream + 1, 0),
        streams_begin_byte_offset(n_stream + 1, 0),
        streams_bytes(n_stream + 1, 0);

    int elements_per_stream = (elements + n_stream - 1) / n_stream;
    // printf("elements per stream:%d\n", elements_per_stream);

    // ? make each stream at least process 1 row, could be less effective for
    // ? small rows big cols
    elements_per_stream =
        elements_per_stream < cols ? cols : elements_per_stream;

    int rows_in_streams, cols_in_streams;

    if (elements_per_stream <= cols) {
      cols_in_streams = (cols + elements_per_stream - 1) / elements_per_stream;
      rows_in_streams = rows;
    } else {
      // if elements per stream is more than elements per row, we ceil elements
      // per stream to the nearest multiple of elements per row
      elements_per_stream += cols - (elements_per_stream % cols);
      cols_in_streams = 1;
      int rows_per_stream = elements_per_stream / cols;
      rows_in_streams = (rows + rows_per_stream - 1) / rows_per_stream;
    }

    // printf("elements_per_stream:%d, rows_in_streams:%d,
    // cols_in_streams:%d\n",
    //        elements_per_stream, rows_in_streams, cols_in_streams);

    // craete streams
    cudaStream_t stream[n_stream + 1];

    cudaEvent_t hToDCpyStartEvents[n_stream + 1],
        hToDCpyEndEvents[n_stream + 1], dToHCpyStartEvents[n_stream + 1],
        dToHCpyEndEvents[n_stream + 1], kernelStartEvents[n_stream + 1],
        kernelEndEvents[n_stream + 1];

    for (int i = 1; i <= n_stream; ++i) {
      cudaEventCreate(&hToDCpyStartEvents[i]);
      cudaEventCreate(&hToDCpyEndEvents[i]);
      cudaEventCreate(&dToHCpyStartEvents[i]);
      cudaEventCreate(&dToHCpyEndEvents[i]);
      cudaEventCreate(&kernelStartEvents[i]);
      cudaEventCreate(&kernelEndEvents[i]);
    }

    // start streams & copy
    TimeCost total_gpu_time;
    for (int i = 1; i <= n_stream; ++i) {
      // printf("ranging for i:%d\n", i);
      cudaStreamCreate(&stream[i]);
      int begin_elem_offset = (i - 1) * elements_per_stream;

      if (begin_elem_offset < elements) {
        // printf("abort creating stream %d cause not needed\n", i);

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

        cudaEventRecord(hToDCpyStartEvents[i], stream[i]);
        cudaMemcpyAsync(&(d_x[begin_elem_offset]), &(x_flat[begin_elem_offset]),
                        cur_stream_bytes, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&(d_y[begin_elem_offset]), &(y_flat[begin_elem_offset]),
                        cur_stream_bytes, cudaMemcpyHostToDevice, stream[i]);
        cudaEventRecord(hToDCpyEndEvents[i], stream[i]);
        // check_kernel_err();
        cudaStreamSynchronize(stream[i]);

#ifdef DEBUG
        printf("stream:%d, begin_elem_offset:%d, cur_stream_elements:%d, "
               "begin_byte_offset:%d, cur_stream_bytes:%d\n",
               i, begin_elem_offset, cur_stream_elements, begin_byte_offset,
               cur_stream_bytes);
#endif
      }

      // launch the kernel for the previous stream
      // printf("checking kernel launch for i:%d\n", i);
      if (i > 1 && streams_elements[i - 1] > 0) {
        int grid_dim = (streams_elements[i - 1] + block_size - 1) / block_size;
        // printf("starting kernel for stream:%d\n", i - 1);
        cudaEventRecord(kernelStartEvents[i - 1], stream[i - 1]);
        basic_impl<<<grid_dim, block_size, 0, stream[i - 1]>>>(
            d_x, d_y, d_z, rows, cols, i - 1, streams_begin_elem_offset[i - 1],
            streams_elements[i - 1]);
        cudaEventRecord(kernelEndEvents[i - 1], stream[i - 1]);
        cudaEventRecord(dToHCpyStartEvents[i - 1], stream[i - 1]);
        cudaMemcpyAsync(&h_z[streams_begin_elem_offset[i - 1]],
                        &d_z[streams_begin_elem_offset[i - 1]],
                        streams_bytes[i - 1], cudaMemcpyDeviceToHost,
                        stream[i - 1]);
        cudaEventRecord(dToHCpyEndEvents[i], stream[i - 1]);
      }

      // extra check for last stream
      if (i == n_stream && streams_elements[i] > 0) {
        int grid_dim = (streams_elements[i] + block_size - 1) / block_size;
        cudaEventRecord(kernelStartEvents[i], stream[i]);
        basic_impl<<<grid_dim, block_size, 0, stream[i]>>>(
            d_x, d_y, d_z, rows, cols, i, streams_begin_elem_offset[i],
            streams_elements[i]);
        cudaEventRecord(kernelEndEvents[i], stream[i]);
        cudaEventRecord(dToHCpyStartEvents[i], stream[i]);
        cudaMemcpyAsync(&h_z[streams_begin_elem_offset[i]],
                        &d_z[streams_begin_elem_offset[i]], streams_bytes[i],
                        cudaMemcpyDeviceToHost, stream[i]);
        cudaEventRecord(dToHCpyEndEvents[i], stream[i]);
      }
    }

    // ! when the current stream launch kernel, the last stream may have not
    // ! finished memcpy

    for (int i = 1; i <= n_stream; ++i) {
      // printf("Synchronizing stream %d\n", i);
      cudaStreamSynchronize(stream[i]);
    }

    double cpu_gpu_transfer_time = 0;
    double kernel_time = 0;
    double gpu_cpu_transfer_time = 0;
    for (int i = 1; i < n_stream; ++i) {
      float ms;
      cudaEventElapsedTime(&ms, hToDCpyStartEvents[i], hToDCpyEndEvents[i]);
      cpu_gpu_transfer_time += ms / 1000;
      cudaEventElapsedTime(&ms, kernelStartEvents[i], kernelEndEvents[i]);
      kernel_time += ms / 1000;
      cudaEventElapsedTime(&ms, dToHCpyStartEvents[i], dToHCpyStartEvents[i]);
      gpu_cpu_transfer_time += ms / 1000;
    }

    record.cpu_gpu_transfer_time = cpu_gpu_transfer_time;
    record.kernel_time = kernel_time;
    record.gpu_cpu_transfer_time = gpu_cpu_transfer_time;
    record.total_gpu_time = total_gpu_time.get_elapsed();
    record.z_value = h_z[5 * cols + 5];

    records.gpu_records.basic_streaming = record;

    check_results(cpu_z, h_z, rows, cols, elements, "basic_streaming");

    cudaFree(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
  }

  printf("final free\n");

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
  // records.gpu_records.basic.print();
  // records.gpu_records.shared_memory.print();
  // records.gpu_records.shared_tiling.print();
  records.gpu_records.basic_streaming.print();

  cpu_free(x, rows);
  cpu_free(y, rows);

  return 0;
}