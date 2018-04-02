
// Reads a cell at (x+dx, y+dy)
__device__ int read_cell(int * source_domain, int x, int y, int dx, int dy,
    unsigned int domain_x, unsigned int domain_y)
{
    x = (unsigned int)(x + dx) % domain_x;	// Wrap around
    y = (unsigned int)(y + dy) % domain_y;
    return source_domain[y * domain_x + x];
}

/*
 *  Load the necessary values in shared memory
 *
 *  @param source_domain:   original values domain to read from
 *  @param tx:              thread ID relative to the grid's x dimension
 *  @param ty:              thread ID relative to the grid's y dimension
 *  @param tx_r:            thread ID relative to the block's x dimension
 *  @param ty_r:            thread ID relative to the block's y dimension
 *  @param dest_subdomain:  shared memory space to store the read values
 *  @param domain_x:        original domain's x dimension size
 *  @param domain_y:        original domain's y dimension size
 */
__device__ void read_to_sm (int * source_domain, int tx, int ty, int tx_r, int ty_r, int dest_subdomain[][],
    unsigned int domain_x, unsigned int domain_y)
{
    //  first step : every thread reads its cell's upper-left neighbor
    dest_subdomain[tx_r][ty_r] = read_cell (source_domain, tx, ty, -1, -1, domain_x, domain_y);

    /*  second step : 
     *  - the last two rows of threads read their immediate down neighbor
     */
    if (blockDim.y < domain_y)
        if (tx_r > blockDim.x - 3)
            dest_subdomain[tx_r + 1][ty_r + 2] = read_cell (source_domain, tx, ty, 0, 1, domain_x, domain_y);
}

/*
 *  Compute the cell's new value :
 *  - read the cell's starting value
 *  - read neighbor's values from shared memory
 *  - apply the life and death rules
 *
 *  @param subdomain:   the shared memory space containing the values
 *  @param tx_r:        thread ID relative to the block's x dimension
 *  @param ty_r:        thread ID relative to the block's y dimension
 *
 *  @return:            the cell's new value if a change is necessary
 *                      -1 if no change is needed
 */
__device__ int new_value (int subdomain[][], int tx_r, int ty_r)
{
    //  read self
    int myself = subdomain[tx_r + 1][ty_r + 1];

    int blue = 0, alive = 0;

    //  if the cell is not empty, break out on alive neighboor count exceeding 3
    for (int x_offset = -1 ; x_offset < 2  &&
                                (!myself || (alive < 4)) ; x_offset++)
    {
        for (int y_offset = -1 ; y_offset < 2 &&
                                (!myself || (alive < 4)) ; y_offset++)
        {
            //  ignore self
            if (x_offset == 0 && y_offset == 0)
                continue;

            //  if 7 values have been read and no live neighbor was found, don't read the last value
            if (y_offset == 1 && x_offset == 1 && alive == 0)
                break;

            switch (subdomain[tx_r + 1 + x_offset][ty_r + 1 + y_offset])
            {
                case 1: 
                    alive++;
                    break;
                case 2: 
                    blue++;
                    alive++;
                    break;
                default:    
                    break;
            }
        }
    }

    //  empty cell case
    if (!myself)
    {
        if (alive == 3)
            if (blue == 0 || blue == 1)
                return 1;
            else
                return 2;
    }
    //  live cell cases
    else
    {
        //  die cases
        if (alive != 2 && alive != 3)
            return 0;

        //  else survive : no changes needed
    }

    return myself;
}

// Compute kernel
__global__ void life_kernel(int * source_domain, int * dest_domain,
    int domain_x, int domain_y)
{
    //  thread IDs : absolute
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    //  thread IDs relative to the block
    int tx_r = threadIdx.x;
    int ty_r = threadIdx.y;

    extern _shared_ int subdomain[blockDim.x + 2][blockDim.y + 2];
    
    //  load values in shared memory
    read_to_sm (source_domain, tx, ty, tx_r, ty_r, subdomain, domain_x, domain_y);

	// Compute new value
    int change = new_value (subdomain, tx_r, ty_r);

	// Write it in dest_domain
    dest_domain[ty * domain_x + tx] = change;

    _syncthreads();
}

