/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512


__global__ void naiveReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // NAIVE REDUCTION IMPLEMENTATION

    __shared__ float partialSum[2*BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start_th = 2*blockIdx.x*blockDim.x;
    
    partialSum[t] = in[start_th + t];
    partialSum[blockDim.x+t] = in[start_th + blockDim.x+t];
    
    for (unsigned int step = 1;step <= blockDim.x;  step *= 2) 
    {
        __syncthreads();
            if (t % step == 0)
              partialSum[2*t]+= partialSum[2*t+step];
   }
    if (t==0){
        out[blockIdx.x]=partialSum[0];

}
}

__global__ void optimizedReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // OPTIMIZED REDUCTION IMPLEMENTATION

    __shared__ float redSum[2*BLOCK_SIZE];
    unsigned int start = 2*blockIdx.x * blockDim.x;
    redSum[threadIdx.x] = in[start + threadIdx.x];
    redSum[blockDim.x+threadIdx.x] = in[start + blockDim.x+threadIdx.x];

    for (unsigned int step = blockDim.x; step > 0; step /= 2)
    {
        if (threadIdx.x < step && start+step < size) 
                                
        {
            redSum[threadIdx.x] += redSum[threadIdx.x + step];
        }
        __syncthreads();
    }
     if (threadIdx.x==0)
    {
        out[blockIdx.x] = redSum[0];
    }

}
