
// Reads a cell at (x+dx, y+dy)
__device__ int read_cell(int * source_domain, int x, int y, int dx, int dy,
    unsigned int domain_x, unsigned int domain_y)
{
    x = (unsigned int)(x + dx) % domain_x;	// Wrap around
    y = (unsigned int)(y + dy) % domain_y;
    return source_domain[y * domain_x + x];
}

// Compute kernel
__global__ void life_kernel(int * source_domain, int * dest_domain,
    int domain_x, int domain_y)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y;
    
    // Read cell
    int myself = read_cell(source_domain, tx, ty, 0, 0,
	                       domain_x, domain_y);
    
    // TODO: Read the 8 neighbors and count number of blue and red

	// TODO: Compute new value
	
	// TODO: Write it in dest_domain
}

