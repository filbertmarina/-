#include <stdio.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <string>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iomanip>
#include "mpi.h"



__global__ void calculationMatrix(double* new_arry, const double* old_array, size_t size, size_t groupSize)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    //printf("%d", size);
    if (i > 0 && i < groupSize - 1 && j > 0 && j < size - 1)
    {
        new_arry[i * size + j] = 0.25 * (old_array[i * size + j - 1] + old_array[(i - 1) * size + j] +
            old_array[(i + 1) * size + j] + old_array[i * size + j + 1]);
    }
}


__global__ void getDifferenceMatrix(const double* new_arry, const double* old_array, double* dif)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    dif[idx] = std::abs(old_array[idx] - new_arry[idx]);
}




int main(int argc, char** argv) {

    int SIZE;
    double err_max;
    int iter_max;

    SIZE = atoi(argv[2]);
    err_max = atof(argv[1]);
    iter_max = atoi(argv[3]);

    clock_t start;
    start = clock();

    int rank, sizeOfTheGroup;
    MPI_Init(&argc, &argv);
    //Функция возвращает номер процесса, вызвавшего эту функцию
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // колво процессов в комуникаторе, запись в sizeOfTheGroup
    MPI_Comm_size(MPI_COMM_WORLD, &sizeOfTheGroup);


    cudaSetDevice(rank);

    if (rank != 0)
        cudaDeviceEnablePeerAccess(rank - 1, 0);
    if (rank != sizeOfTheGroup - 1)
        cudaDeviceEnablePeerAccess(rank + 1, 0);

    size_t sizeOfAreaForOneProcess = SIZE / sizeOfTheGroup;
    size_t startYIdx = sizeOfAreaForOneProcess * rank;

 
    double* mas, * mas_old;
    cudaMallocHost(&mas, sizeof(double) * SIZE * SIZE);
    cudaMallocHost(&mas_old, sizeof(double) * SIZE * SIZE);

    std::memset(mas_old, 0, SIZE * SIZE * sizeof(double));


    mas[0] = 10;
    mas[SIZE - 1] = 20;
    mas[(SIZE) * (SIZE - 1)] = 20;
    mas[(SIZE) * (SIZE)-1] = 30;



    for (int i = 1; i < SIZE - 1; i++)
        mas[i] = mas[i - 1] + (mas[SIZE - 1] - mas[0]) / SIZE;

    for (int i = 1; i < SIZE - 1; i++) {
        mas[SIZE * (SIZE - 1) + i] = mas[SIZE * (SIZE - 1) + i - 1] + (mas[(SIZE) * (SIZE)-1] - mas[(SIZE) * (SIZE - 1)]) / SIZE;
        mas[(SIZE) * (i)] = mas[(i - 1) * (SIZE)] + (mas[(SIZE) * (SIZE - 1)] - mas[0]) / SIZE;
        mas[(SIZE) * (i)+(SIZE - 1)] = mas[(SIZE) * (i - 1) + (SIZE - 1)] + (mas[(SIZE) * (SIZE)-1] - mas[SIZE - 1]) / SIZE;
    }


    for (int i = 0; i < SIZE * SIZE; i++)
        mas_old[i] = mas[i];    



    int iter_count = 0; 
    double* error;
    cudaMallocHost(&error, sizeof(double));
    *error = 1.0;



    if (rank != 0 && rank != sizeOfTheGroup - 1)
    {
        sizeOfAreaForOneProcess += 2;
    }
    else
    {
        sizeOfAreaForOneProcess += 1;
    }

    size_t sizeOfAllocatedMemory = SIZE * sizeOfAreaForOneProcess;

    //выделяем память на gpu 
    double* d_mas, * d_mas_old, * d_dif;
    cudaMalloc((void**)&d_mas_old, sizeof(double) * sizeOfAllocatedMemory);
    cudaMalloc((void**)&d_mas, sizeof(double) * sizeOfAllocatedMemory);
    cudaMalloc((void**)&d_dif, sizeof(double) * sizeOfAllocatedMemory);



    /// ///////////////////////
    unsigned int threads_x = (SIZE < 1024) ? SIZE : 1024;// кол-во потоков 
    unsigned int blocks_y = sizeOfAreaForOneProcess;// колво блоков 
    unsigned int blocks_x = SIZE / threads_x;

    dim3 blockDim1(threads_x, 1);//- размеры одного блока в потоках
    dim3 gridDim1(blocks_x, blocks_y);//размер сетки в блоках
    /// ////////////////////////////////


    size_t offset = (rank != 0) ? SIZE : 0;

    cudaMemset(d_mas_old, 0, sizeof(double) * sizeOfAllocatedMemory);
    cudaMemset(d_mas, 0, sizeof(double) * sizeOfAllocatedMemory);

    // CPU to GPU
    cudaMemcpy(d_mas_old, mas_old + (startYIdx * SIZE) - offset, sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice); // (CPU) mas_old -> (GPU) d_mas_old
    cudaMemcpy(d_mas, mas + (startYIdx * SIZE) - offset, sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice); // (CPU) mas -> (GPU) d_mas

  
    double* max_error = 0;
    cudaMalloc((void**)&max_error, sizeof(double));




    size_t temp_storage_bytes = 0;
    double* temp_storage = NULL;
    //получаем размер временного буфера для редукции
    cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, d_dif, max_error, sizeOfAllocatedMemory);
    //выделяем память для буфера
    cudaMalloc((void**)&temp_storage, temp_storage_bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);


 

   
    while (iter_count < iter_max && (*error) > err_max) {
        iter_count += 1;
        // расчет матрицы
        calculationMatrix <<<gridDim1, blockDim1, 0, stream >>> (d_mas, d_mas_old, SIZE, sizeOfAreaForOneProcess);

     
        if (iter_count % 100 == 0) {

            getDifferenceMatrix <<<gridDim1, blockDim1, 0, stream >>> (d_mas, d_mas_old, d_dif);
            cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, d_dif, max_error, sizeOfAllocatedMemory); // нахождение максимума в разнице матрицы

            cudaStreamSynchronize(stream);

            //Объединяет значения из всех процессов и распределяет результат обратно во все процессы.
            MPI_Allreduce((void*)max_error, (void*)max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            cudaMemcpyAsync(error, max_error, sizeof(double), cudaMemcpyDeviceToHost, stream); // запись ошибки в переменную на host
            // Находим максимальную ошибку среди всех и передаём её всем процессам

        }
        cudaStreamSynchronize(stream);


        // Обмен "граничными" условиями каждой области
    // Обмен верхней границей
        if (rank != 0)
        {
            MPI_Sendrecv(d_mas + SIZE + 1, SIZE - 2, MPI_DOUBLE, rank - 1, 0, d_mas + 1, SIZE - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Обмен нижней границей
        if (rank != sizeOfTheGroup - 1)
        {
            MPI_Sendrecv(d_mas + (sizeOfAreaForOneProcess - 2) * SIZE + 1,
                SIZE - 2, MPI_DOUBLE, rank + 1, 0,
                d_mas + (sizeOfAreaForOneProcess - 1) * SIZE + 1,
                SIZE - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        double* c = d_mas_old;
        d_mas_old = d_mas;
        d_mas = c;
    }

    clock_t end = clock();
    if (rank == 0)
    {
        printf("%d %lf", (end - start) / CLOCKS_PER_SEC, &err);
    }

    //очитска памяти
    cudaFree(d_mas_old);
    cudaFree(d_mas);
    cudaFree(temp_storage);
    cudaFree(mas_old);
    cudaFree(mas);
    cudaFree(max_error);

    //Функция закрывает все MPI-процессы и ликвидирует все области связи
    MPI_Finalize();
        
    return 0;
}

