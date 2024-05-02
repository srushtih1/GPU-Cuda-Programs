#include <stdio.h>

#define TILE_SIZE 16

__global__ void matAdd(int dim, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A + B
     *   where A is a (dim x dim) matrix
     *   where B is a (dim x dim) matrix
     *   where C is a (dim x dim) matrix
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row<dim){
	for (int col =0; col<dim;col++){
		int index_number = row * dim + col;
        	C[index_number] = A[index_number] + B[index_number];
    	}
    }

    /*************************************************************************/

}

void basicMatAdd(int dim, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    /*************************************************************************/
    //INSERT CODE HERE
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
    dim3 dimGrid((dim-1)/BLOCK_SIZE + 1, (dim-1)/BLOCK_SIZE + 1, 1);

    /*************************************************************************/

        // Invoke CUDA kernel -----------------------------------------------------
        matAdd<<<dimGrid,dimBlock>>>(dim,A,B,C);

    /*************************************************************************/
    //INSERT CODE HERE
    /*************************************************************************/

}
