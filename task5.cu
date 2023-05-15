#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cuda.h>

#include <mpi.h>


__global__
void countnewmatrix(double* mas_old, double* mas, size_t size, size_t sizePerGpu)
{
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    	size_t i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (!(blockIdx.x == 0 || threadIdx.x == 0 || i == sizePerGpu - 1))
		mas[i * size + j] = 0.25 * (mas_old[i * size + j - 1] + mas_old[(i - 1) * size + j] + mas_old[(i + 1) * size + j] + mas_old[i * size + j + 1]);

}
__global__
void finderr(double* mas_old, double* mas, double* outMatrix, size_t size)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (!(blockIdx.x == 0 || threadIdx.x == 0))
		outMatrix[idx] = fabs(mas[idx] - mas_old[idx]);

}

int find_threads(int size) {
	if (size % 32 == 0)
		return size / 1024;

	return int(size / 1024) + 1;

}


int main(int argc, char** argv)
{
	int SIZE;
	double err_max;
	int iter_max;

	SIZE = atoi(argv[2]);
	err_max = atof(argv[1]);
	iter_max = atoi(argv[3]);


	clock_t start;
	start = clock();
	///////////////////////&

	
//функция инициализации MPI, создаем группу в которой наход все процессы и создается область связи описываемая коммуникатором MPI_COMM_WORLD
	MPI_Init(&argc, &argv);
	int rank, sizeOfTheGroup;

//delete 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//Функция определения числа процессов в области связи MPI_Comm_size
	MPI_Comm_size(MPI_COMM_WORLD, &sizeOfTheGroup);


	cudaSetDevice(rank);
//////
	if (rank != 0)
		cudaDeviceEnablePeerAccess(rank - 1, 0);
	if (rank != sizeOfTheGroup - 1)
		cudaDeviceEnablePeerAccess(rank + 1, 0);
///////

	size_t sizeOfAreaForOneProcess = SIZE / sizeOfTheGroup;
	size_t startYIdx = sizeOfAreaForOneProcess * rank;


	// Расчитываем, сколько памяти требуется процессу
	if (rank != 0 && rank != sizeOfTheGroup - 1)
	{
		sizeOfAreaForOneProcess += 2;
	}
	else
	{
		sizeOfAreaForOneProcess += 1;
	}

	size_t sizeOfAllocatedMemory = SIZE * sizeOfAreaForOneProcess;

/////////


	////////////////////
	double* mas;
	double* mas_old;
	cudaMallocHost(&mas, SIZE * SIZE * sizeof(double));
	cudaMallocHost(&mas_old, SIZE * SIZE * sizeof(double));


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






	double* mas_old_dev, * mas_dev, * deviceError, * errorMatrix, * tempStorage = NULL; //Device-accessible allocation of temporary storage


	////////////////////////////&
	unsigned int threads_x = (SIZE < 1024) ? SIZE : 1024;       // кол-во потоков в блоке 
	unsigned int blocks_y = sizeOfAreaForOneProcess;            // кол-во блоков по y в сетке 
	unsigned int blocks_x = SIZE / threads_x;              //кол-во блоков по x

	dim3 blockDim(threads_x, 1);//кол-во потоков в блоке 
	dim3 gridDim(blocks_x, blocks_y);//размер сетки 
	/////////////////////////////////////////

		// Выделяем память на девайсе
	cudaError_t cudaStatus_1 = cudaMalloc((void**)(&mas_old_dev), sizeof(double) * SIZE * SIZE);
	cudaError_t cudaStatus_2 = cudaMalloc((void**)(&mas_dev), sizeof(double) * SIZE * SIZE);
	cudaMalloc((void**)&deviceError, sizeof(double));
	cudaError_t cudaStatus_3 = cudaMalloc((void**)&errorMatrix, sizeof(double) * SIZE * SIZE);
	if (cudaStatus_1 != 0 || cudaStatus_2 != 0 || cudaStatus_3 != 0)
	{
		std::cout << "error" << std::endl;
		return -1;
	}

	////////////////////////&
		// Копируем часть заполненной матрицы в выделенную память, начиная с 1 строки
	size_t offset = (rank != 0) ? SIZE : 0;
	cudaMemset(mas_old_dev, 0, sizeof(double) * sizeOfAllocatedMemory);
	cudaMemset(mas_dev, 0, sizeof(double) * sizeOfAllocatedMemory);
	cudaMemcpy(mas_old_dev, mas_old + (startYIdx * SIZE) - offset, sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice);
	cudaMemcpy(mas_dev, mas + (startYIdx * SIZE) - offset, sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice);

	//////////////////

	size_t tempsize=0;
	// Determine temporary device storage requirements
	cub::DeviceReduce::Max(tempStorage, tempsize, errorMatrix, deviceError, SIZE * SIZE);
	// Allocate temporary storage
	cudaMalloc(&tempStorage, 0);

	int iter = 0;
	double* err;
	cudaMallocHost(&err, sizeof(double));
	*err = 1.0;

	cudaStream_t stream, memoryStream;
	cudaStreamCreate(&stream);
	cudaStreamCreate(&memoryStream);


	while ((iter < iter_max) && (*err) > err_max)
	{
		iter++;


		countnewmatrix <<<gridDim, blockDim, 0, stream >>> (mas_old_dev, mas_dev, SIZE, sizeOfAreaForOneProcess);


		if (iter % 100 == 0)
		{
			finderr <<<blocks_x * blocks_y, threads_x, 0, stream >>> (mas_old_dev, mas_dev, errorMatrix, SIZE);
			cub::DeviceReduce::Max(tempStorage, tempsize, errorMatrix, deviceError, sizeOfAllocatedMemory);

			cudaStreamSynchronize(stream);

				((void*)deviceError, (void*)deviceError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);//

			cudaMemcpyAsync(err, deviceError, sizeof(double), cudaMemcpyDeviceToHost, stream);
		}

		//Waits for stream tasks to complete.
		cudaStreamSynchronize(stream);

		// Обмен "граничными" условиями каждой области
		// Обмен верхней границей
		if (rank != 0)
		{
			MPI_Sendrecv(mas_dev + SIZE + 1, SIZE - 2, MPI_DOUBLE, rank - 1, 0, mas_dev + 1, SIZE - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		// Обмен нижней границей
		if (rank != sizeOfTheGroup - 1)
		{
			MPI_Sendrecv(mas_dev + (sizeOfAreaForOneProcess - 2) * SIZE + 1,
				SIZE - 2, MPI_DOUBLE, rank + 1, 0,
				mas_dev + (sizeOfAreaForOneProcess - 1) * SIZE + 1,
				SIZE - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		double* m = mas;
		mas = mas_old;
		mas_old = m;
	}

	clock_t end = clock();
	if (rank == 0)
	{
		printf("%d %lf", (end-start)/ CLOCKS_PER_SEC, &err);
	}


	// Высвобождение памяти
	cudaFree(mas_old_dev);
	cudaFree(mas_dev);
	cudaFree(errorMatrix);
	cudaFree(tempStorage);
	cudaFree(mas_old);
	cudaFree(mas);

//Функция закрывает все MPI-процессы и ликвидирует все области связи
	MPI_Finalize();

	return 0;
}
