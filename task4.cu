#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cuda.h>


__global__
void countnewmatrix(double* mas_old, double* mas, size_t size)
{
	size_t i = blockIdx.x;
	size_t j = threadIdx.x;

	assert(i * size + j > size * size);
	if (!(blockIdx.x == 0 || threadIdx.x == 0))
		mas[i * size + j] = 0.25 * (mas_old[i * size + j - 1] + mas_old[(i - 1) * size + j] + mas_old[(i + 1) * size + j] + mas_old[i * size + j + 1]);
	
}
__global__
void finderr(double* mas_old, double* mas, double* outMatrix, size_t size)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	assert(idx > size * size);
	if (!(blockIdx.x == 0 || threadIdx.x == 0))
		outMatrix[idx] = fabs(mas[idx] - mas_old[idx]);
	
}

int find_threads(int size){
	if (size%32==0)
		return size/1024;

	return int(size/1024)+1;

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


	double* mas ;
	double* mas_old;
	cudaMallocHost(&mas, SIZE*SIZE * sizeof(double));
	cudaMallocHost(&mas_old, SIZE*SIZE * sizeof(double));
	

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


	int iter = 0;
	double *err;
	*err = 1.0;


	for (int i = 0; i < SIZE * SIZE; i++) 
		mas_old[i] = mas[i];


	

	printf("start");
	cudaSetDevice(3);

	double* mas_old_dev, * mas_dev, * deviceError, * errorMatrix, * tempStorage = NULL; //Device-accessible allocation of temporary storage
	size_t tempStorageSize = 0;

	cudaError_t cudaStatus_1 = cudaMalloc((void**)(&mas_old_dev), sizeof(double) * SIZE*SIZE);
	cudaError_t cudaStatus_2 = cudaMalloc((void**)(&mas_dev), sizeof(double) * SIZE*SIZE);
	cudaMalloc((void**)&deviceError, sizeof(double));
	cudaError_t cudaStatus_3 = cudaMalloc((void**)&errorMatrix, sizeof(double) * SIZE*SIZE);
	if (cudaStatus_1 != 0 || cudaStatus_2 != 0 || cudaStatus_3 != 0)
	{
		std::cout << "error" << std::endl;
		return -1;
	}
	


	cudaStatus_1  = cudaMemcpy(mas_old_dev, mas_old, sizeof(double) * SIZE*SIZE, cudaMemcpyHostToDevice);
	cudaStatus_2  = cudaMemcpy(mas_dev, mas, sizeof(double) * SIZE*SIZE, cudaMemcpyHostToDevice);
	if (cudaStatus_1 != 0 || cudaStatus_2 != 0 )
	{
		std::cout << "error" << std::endl;
		return -1;
	}

	// Determine temporary device storage requirements
	cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, SIZE * SIZE, stream);
	// Allocate temporary storage
	cudaMalloc(&tempStorage, tempStorageSize);
	
	bool graphCreated = false;
	cudaStream_t stream, memoryStream;
	cudaStreamCreate(&stream);
	cudaStreamCreate(&memoryStream);
	cudaGraph_t graph;
	cudaGraphExec_t instance;
	int t= 1024;
	int b = find_threads(SIZE);

		while ((*err > err_max) && iter < iter_max) {

			 if(!graphCreated){
    				cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
				for(size_t i = 0; i <50; i++){
					countnewmatrix <<<b, t, 0, stream>>> (mas_old_dev, mas_dev, SIZE);
					countnewmatrix <<<b, t, 0, stream>>> (mas_dev, mas_old_dev, SIZE);		
		}
				finderr <<<b, t, 0 , stream>>> (mas_old_dev, mas_dev, errorMatrix, SIZE);

				cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, SIZE*SIZE, stream);
	
				
				cudaStreamEndCapture(stream, &graph);
				cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
				graphCreated=true;

			}
			else{
				cudaGraphLaunch(instance, stream);
				cudaMemcpyAsync(err, deviceError, sizeof(double), cudaMemcpyDeviceToHost, stream);
				cudaStreamSynchronize(stream);
				iter += 100;
				double ti = (double)(clock() - start) / CLOCKS_PER_SEC;
					printf(" time: %lf\n", ti);
					printf("%d  %lf", iter, &err);
					printf("\n");

				
			}
		}

	cudaFree(mas_old_dev);
	cudaFree(mas_dev);
	cudaFree(errorMatrix);
	cudaFree(tempStorage);
	
	free(mas_old);

	free(mas);

	double ti = (double)(clock() - start) / CLOCKS_PER_SEC;
	printf(" time: %lf\n", ti);
	return 0;

}
