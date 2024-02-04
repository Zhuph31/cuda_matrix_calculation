    // test copy
    float *test = (float *)malloc(elements * sizeof(float));
    cudaMemcpy(test, d_x, elements * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < elements; ++i) {
      printf("%lf,", test[i]);
    }
    printf("\n");
    exit(0);

