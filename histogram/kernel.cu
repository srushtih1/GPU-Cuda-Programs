#include <stdio.h>
#define BLOCK_SIZE 512
#define MAX_BLOCK_NUM 16

__global__ void histo_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins)
{

    /*************************************************************************/
    // INSERT KERNEL CODE HERE

     __shared__ unsigned int pvt_histogram[4096];

    for (int i = threadIdx.x; i < num_bins; i += BLOCK_SIZE)
       pvt_histogram[i] = 0;
    __syncthreads();

    int start = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (start < num_elements)
    {
        atomicAdd(&(pvt_histogram[input[start]]), 1);
        start += stride;
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < num_bins; i += BLOCK_SIZE)
        atomicAdd(&(bins[i]), pvt_histogram[i]);

   /*************************************************************************/
}

void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

          /*************************************************************************/
    //INSERT CODE HERE

    dim3 Dim_Grid, Dim_Block;
    Dim_Block.x = BLOCK_SIZE; Dim_Block.y = Dim_Block.z = 1;
    int num_blocks = (num_elements-1)/BLOCK_SIZE+1;
    Dim_Grid.x = (num_blocks > MAX_BLOCK_NUM ? MAX_BLOCK_NUM : num_blocks);
    Dim_Grid.y = Dim_Grid.z = 1;

    histo_kernel<<<Dim_Grid, Dim_Block>>>(input, bins, num_elements, num_bins);

          /*************************************************************************/

}

