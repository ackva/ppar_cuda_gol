#define BLOCK_DIM_X 128
#define BLOCK_DIM_Y 8

// Reads a cell at (x+dx, y+dy)
__device__ int read_cell(int * source_domain, int x, int y, int dx, int dy,
    unsigned int domain_x, unsigned int domain_y)
{
    x = (unsigned int)(x + dx) % domain_x;	// Wrap around
    y = (unsigned int)(y + dy) % domain_y;
    return source_domain[y * domain_x + x];
}

/*
 *  Read neighbor cells' values
 *
 *  @param subdomain:   shared memory space
 *  @param txr:         thread's Id relative to the block's x dimension
 *  @param tyr:         thread's Id relative to the block's y dimension
 *  
 *  @param self:        destination for the cell's value
 *  @param alive:       destination for the cell's live neighbor count
 *  @param blue:        destination for the cell's blue neighbor count
 *
 */
__device__ void read_values (int subdomain[][], int txr, int tyr,
    int *self, int *alive, int *blue)
{
    int self_ = 0, alive_ = 0, blue_ = 0, current;

    for (int y_offset = 0 ; y_offset < 3
                                && (!self_ || alive_ < 4) ; y_offset++)
    {
        for (int x_offset = 0 ; x_offset < 3
                                && (!self_ || alive_ < 4) ; x_offset++)
        {
            if (y_offset == 1 && x_offset == 1)
                self_ = subdomain[txr + 1][ty_r + 1];
            else
            {
                current = subdomain[tyr + y_offset][txr + x_offset];
                alive_ += current;

                if (current == 2)
                {
                    alive_--;
                    blue_++;
                }
            }
        }
    }

    *self = self_;
    *alive = alive_;
    *blue = blue_;
}

/*
 *  Compute the cell's next value
 *
 *  @param current: current value
 *  @param alive:   number of alive neighbor cells
 *  @param blue:    number of blue neighbor cells
 *
 *  @return:        the cell's next value
 */
__device__ int new_value (int current, int alive, int blue)
{
    if (alive > 3 || alive < 2)
        return 0;

    if (!current && alive == 3)
        if (blue < 2)
            return 1;
        else
            return 2;

    return current;
}

// Compute kernel
__global__ void life_kernel(int * source_domain, int * dest_domain,
    int domain_x, int domain_y)
{
    //  thread IDs relative to the block
    int tx_r = threadIdx.x;
    int ty_r = threadIdx.y;

    //  thread IDs : absolute
    int tx = blockIdx.x * blockDim.x + tx_r;
    int ty = blockIdx.y * blockDim.y + ty_r;

    //  shared memory space
    __shared__ int subdomain[BLOCK_DIM_Y + 2][BLOCK_DIM_X];
    
    /*  
     *  Load values in shared memory
     */

    //  first step : every thread reads its cell's upper-left neighbor
    subdomain[ty_r][tx_r] = read_cell (source_domain, tx, ty, -1, -1, domain_x, domain_y);

    /*  second step : 
     *  - the last two rows of threads read their bottom-left neighbor
     */
    if (ty_r >= BLOCK_DIM_Y - 2)
        subdomain[ty_r + 1][tx_r] = read_cell (source_domain, tx, ty, -1, 1, domain_x, domain_y);

	/*
     *  Read neighbor cell's values
     */
    int myself, alive, blue;
    read_values (subdomain, tx_r, ty_r, &myself, &alive, &blue);

	// Write it in dest_domain
    dest_domain[ty * domain_x + tx] = new_value (myself, alive, blue);

    __syncthreads();
}

