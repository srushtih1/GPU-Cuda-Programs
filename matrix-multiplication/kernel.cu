#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    __shared__ float A_matrix[TILE_SIZE][TILE_SIZE];
    __shared__ float B_matrix[TILE_SIZE][TILE_SIZE];

    int blkx = blockIdx.x;
    int blky = blockIdx.y;
    int thrdx = threadIdx.x;
    int thrdy = threadIdx.y;

    int row = blky * blockDim.y + thrdy;
    int col = blkx * blockDim.x + thrdx;

    float Cvalue = 0;

    // Loop over tiles in A and B
    for (int p = 0; p < (k-1)/TILE_SIZE+1; p++)
    {
        // Load tile into shared memory
        if (row < m && p*TILE_SIZE+thrdx < k)
            A_matrix[thrdy][thrdx] = A[row*k + p*TILE_SIZE+thrdx];
        else
            A_matrix[thrdy][thrdx] = 0.0;
        if (p*TILE_SIZE+thrdy < k && col < n)
            B_matrix[thrdy][thrdx] = B[(p*TILE_SIZE+thrdy)*n + col];
        else
            B_matrix[thrdy][thrdx] = 0.0;
        __syncthreads();
        
        if (row < m && col < n)
        {
            for (int i = 0; i < TILE_SIZE; i++)
                Cvalue += A_matrix[thrdy][i] * B_matrix[i][thrdx];
        }
        __syncthreads();
    }
    if (row < m && col < n)
        C[row*n + col] = Cvalue;
        
    /*************************************************************************/
}

void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
    /*************************************************************************/
    //INSERT CODE HERE
    dim3 dim_Grid, dim_Block;

    dim_Block.x = dim_Block.y = BLOCK_SIZE; 
    dim_Block.z = 1;
    dim_Grid.x = (n - 1) / BLOCK_SIZE + 1;
    dim_Grid.y = (m - 1) / BLOCK_SIZE + 1;
    dim_Grid.z = 1;

    /*************************************************************************/

    // Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE
    mysgemm<<<dim_Grid, dim_Block>>>(m, n, k, A, B, C);
	
    /*************************************************************************/
}
