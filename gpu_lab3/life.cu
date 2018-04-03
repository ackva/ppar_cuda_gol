
#include "utils.h"
#include <stdlib.h>
#include <getopt.h>
#include <ctype.h>
#include <unistd.h>

#include "life_kernel.cu"

#define WARP_SIZE 32
#define DEFAULT_DIM_X 64
#define DEFAULT_DIM_Y 64
#define DEFAULT_STEPS 2
#define DEFAULT_CELLS_PER_WORD 1

#define MIN(x,y) ((x < y) ? x : y)

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
    int domain_x = DEFAULT_DIM_X;
    int domain_y = DEFAULT_DIM_Y;
    int cells_per_word = DEFAULT_CELLS_PER_WORD;
    int steps = DEFAULT_STEPS;

    int c, cval;
    while ((c = getopt (argc, argv, "x:y:s:")) != -1)
    {
        switch (c)
        {
            case 'x':
                if ((cval = optarg) % WARP_SIZE == 0 && cval > 0)
                    domain_x = cval;
                else
                    fprintf (stderr, 
                        "Invalid domain size '%d' : \
                        dimension_x must be a positive multiple of warp size (%d).\n", cval, WARP_SIZE);
                break;
            case 'y':
                if ((cval = optarg) > 0)
                    domain_y = optarg;
                else
                    fprintf (stderr,
                        "Invalid domain size '%d' :\
                        dimension_y must be a positive integer.\n", cval);
                break;
            case 's':
                if ((cval = optarg) > 0)
                    steps = cval;
                else
                    fprintf (stderr,
                        "Invalid number of steps '%d' :\
                        must be a positive integer.\n", cval);
                break;
        }
    }

    int blocks_x = 1;    
    int threads_per_block = 1024;
    c = MIN (1, threads_per_block / domain_x);
    int blocks_y = domain_y / c;

    dim3  grid(blocks_x, blocks_y);	            // CUDA grid dimensions
    dim3  threads(threads_per_block / c, c);	// CUDA block dimensions

    // Allocation of arrays
    int * domain_gpu[2] = {NULL, NULL};

	// Arrays of dimensions domain.x * domain.y
	size_t domain_size = domain_x * domain_y / cells_per_word * sizeof(int);
	CUDA_SAFE_CALL(cudaMalloc((void**)&domain_gpu[0], domain_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&domain_gpu[1], domain_size));

    int * domain_cpu = (int*) malloc(domain_size);

	// Arrays of dimensions pitch * domain.y

	init_data(domain_cpu, domain_x, domain_y);
    CUDA_SAFE_CALL(cudaMemcpy(domain_gpu[0], domain_cpu, domain_size, cudaMemcpyHostToDevice));

    // Timer initialization
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));

    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    //  Compute the necessary shared memory size
    int sm_x = MIN(blocks_x + 2, domain_x);
    int sm_y = MIN(blocks_y + 2, domain_y);
    int shared_mem_size = sm_x * sm_y * sizeof(int);

    // Kernel execution
    for(int i = 0; i < steps; i++) {
	    life_kernel<<< grid, threads, shared_mem_size >>>(domain_gpu[i%2],
	    	domain_gpu[(i+1)%2], domain_x, domain_y, sm_x, sm_y);
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

