
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
    int myself = read_cell(source_domain, tx, ty, 0, 0, domain_x, domain_y);
     
    // TODO: Read the 8 neighbors and count number of blue and red
    int neighbors[8];
    neighbors[0] = read_cell(source_domain, tx, ty,  0, -1, domain_x, domain_y);
    neighbors[1] = read_cell(source_domain, tx, ty,  0,  1, domain_x, domain_y);
    neighbors[2] = read_cell(source_domain, tx, ty, -1, -1, domain_x, domain_y);
    neighbors[3] = read_cell(source_domain, tx, ty, -1,  0, domain_x, domain_y);
    neighbors[4] = read_cell(source_domain, tx, ty, -1,  1, domain_x, domain_y);
    neighbors[5] = read_cell(source_domain, tx, ty,  1, -1, domain_x, domain_y);
    neighbors[6] = read_cell(source_domain, tx, ty,  1,  0, domain_x, domain_y);
    neighbors[7] = read_cell(source_domain, tx, ty,  1,  1, domain_x, domain_y);
	
    // TODO: Compute new value

    int blue = 0, red = 0, empty = 0;

	int myselfNew = 0;
    for (int i = 0; i < 8; i++) {
        switch(neighbors[i]) {
          case 0 :
             empty++;
             break;
          case 1 :
             red++;
             break;
          case 2 :
             blue++;
             break;
          default :
             break;
    }
    }

    int alive = red + blue;

    if (alive < 2 || alive > 3)
        myselfNew = 0;
    else if (alive == 2)
        myselfNew = myself;
    else {
        if (red > blue)
            myselfNew = 1;
        else
            myselfNew = 2;
    }

	// TODO: Write it in dest_domain
    dest_domain[ (ty * domain_x) + tx] = myselfNew;

}

