
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
    int domain_x, int domain_y, int sm_x, int sm_y)
{
    //  thread IDs : absolute
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    //  thread IDs relative to the block
    int tx_r = threadIdx.x;
    int ty_r = threadIdx.y;

    extern __shared__ int subdomain[];
    
    /*  
     *  Load values in shared memory
     */

    //  first step : every thread reads its cell's upper-left neighbor
    subdomain[tx_r + ty_r * sm_x] = read_cell (source_domain, tx, ty, -1, -1, domain_x, domain_y);

    /*  second step : 
     *  - the last two rows of threads read their bottom-left neighbor
     */
    if (sm_y < domain_y)
        //  last two rows
        if (tx_r > blockDim.x - 3)
            subdomain[tx_r + (ty_r + 1) * sm_x] = read_cell (source_domain, tx, ty, -1, 1, domain_x, domain_y);

	/*
     *  Compute new value
     */

    //  read self
    int pos = ((tx_r + 1) % domain_x) + ((ty_r + 1) % domain_y) * sm_x;
    int myself, cell;

    int blue = 0, alive = 0;

    //  if the cell is not empty, break out on alive neighboor count exceeding 3
    for (int y_offset = -1 ; y_offset < 2  &&
                                (!myself || (alive < 4)) ; y_offset++)
    {
        for (int x_offset = -1 ; x_offset < 2 &&
                                (!myself || (alive < 4)) ; x_offset++)
        {
            //  read self
            if (x_offset == 0 && y_offset == 0)
                myself = subdomain[pos];
            else
            {
                //  if 7 values have been read and no live neighbor was found, don't read the last value
                if (y_offset == 1 && x_offset == 1 && alive == 0)
                    break;

                cell = subdomain[pos + x_offset + y_offset * sm_x];
                alive += cell;

                if (cell == 2)
                {
                    alive--;
                    blue++;
                }
            }
        }
    }

    //  empty cell case
    if (!myself)
    {
        if (alive == 3)
            if (blue == 0 || blue == 1)
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

        //  else survive : no changes needed
    }

	// Write it in dest_domain
    dest_domain[ty * domain_x + tx] = myself;

    __syncthreads();
}

