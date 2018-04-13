
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
    int myself = read_cell(source_domain, tx, ty, 0, 0,domain_x, domain_y);
    
    // Read the 8 neighbors and count number of blue and red
    int blue = 0, red = 0, alive = 0;

    //  if the cell is not empty, break out on alive neighboor count exceeding 3
    for (int x_offset = -1 ; x_offset < 2  && (!myself || (alive < 4)) ; x_offset++)
    {
        for (int y_offset = -1 ; y_offset < 2 && (!myself || (alive < 4)) ; y_offset++)
        {
            //  ignore self
            if (x_offset == 0 && y_offset == 0)
                continue;

            switch (read_cell (source_domain, tx, ty, x_offset, y_offset, domain_x, domain_y))
            {
                case 1: red++;  alive++;    break;
                case 2: blue++; alive++;    break;
                default:    break;
            }
        }
    }

	// Compute new value
    
    //  empty cell case
    if (!myself)
    {
        if (alive == 3)
            if (blue < red)
                myself = 1;
            else
                myself = 2;
    }
    //  live cell cases
    else
    {
        //  die cases
        if (alive != 2 && alive != 3)
            myself = 0;
        //  else survive
    }

	// Write it in dest_domain
    dest_domain[ty * domain_x + tx] = myself;
}

