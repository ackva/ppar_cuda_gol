
#include "utils.h"
#include <stdlib.h>

#include "life_kernel.cu"

void init_data(int * domain, int domain_x, int domain_y)
{
	for(int i = 0; i != domain_y; ++i) {
		for(int j = 0; j != domain_x; ++j) {
			domain[i * domain_x + j] = rand() % 3;
		}
	}
}

// Color display code contributed by Louis Beziaud, Simon Bihel and Rémi Hutin, PPAR 2016/2017
void print_domain(int* domain, int domain_x, int domain_y, int* red, int* blue) {
	if (red != NULL) *red = 0;
	if (blue != NULL) *blue = 0;
	for(int y = 0; y < domain_y; y++) {
		for(int x = 0; x < domain_x; x++) {
			int cell = domain[y * domain_x + x];
			switch(cell) {
				case 0:
					printf("\033[40m  \033[0m");
					break;
				case 1:
					printf("\033[41m  \033[0m");
					break;
				case 2:
					printf("\033[44m  \033[0m");
					break;
				default:
					break;
			}
			if(red != NULL && cell == 1) {
				(*red)++;
			} else if(blue != NULL && cell == 2) {
				(*blue)++;
			}
		}
		printf("\n");
	}
}

int main(int argc, char ** argv)
{
    // Definition of parameters
    int domain_x = 128;	// Multiple of threads_per_block * cells_per_word
    int domain_y = 128;
    
    int cells_per_word = 1;
    
    int steps = 2;
    
    int threads_per_block = 128;
    int blocks_x = domain_x / (threads_per_block * cells_per_word);
    int blocks_y = domain_y;
    
    dim3  grid(blocks_x, blocks_y);	// CUDA grid dimensions
    dim3  threads(threads_per_block);	// CUDA block dimensions

    // Allocation of arrays
    int * domain_gpu[2] = {NULL, NULL};

	// Arrays of dimensions domain.x * domain.y
	size_t domain_size = domain_x * domain_y / cells_per_word * sizeof(int);
	CUDA_SAFE_CALL(cudaMalloc((void**)&domain_gpu[0], domain_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&domain_gpu[1], domain_size));

    int * domain_cpu = (int*)malloc(domain_size);

	// Arrays of dimensions pitch * domain.y

	init_data(domain_cpu, domain_x, domain_y);
    CUDA_SAFE_CALL(cudaMemcpy(domain_gpu[0], domain_cpu, domain_size, cudaMemcpyHostToDevice));

    // Timer initialization
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));

    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    // Kernel execution
    int shared_mem_size = 0;
    for(int i = 0; i < steps; i++) {
	    life_kernel<<< grid, threads, shared_mem_size >>>(domain_gpu[i%2],
	    	domain_gpu[(i+1)%2], domain_x, domain_y);
	}

    // Stop timer
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    
    float elapsedTime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));	// In ms
    printf("GPU time: %f ms\n", elapsedTime);

    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));

    // Get results back
    CUDA_SAFE_CALL(cudaMemcpy(domain_cpu, domain_gpu[steps%2], domain_size, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(domain_gpu[0]));
    CUDA_SAFE_CALL(cudaFree(domain_gpu[1]));
    

    // Count colors
    int red = 0;
    int blue = 0;
    print_domain(domain_cpu, domain_x, domain_y, &red, &blue);
    printf("Red/Blue cells: %d/%d\n", red, blue);
    
    free(domain_cpu);
    
    return 0;
}

