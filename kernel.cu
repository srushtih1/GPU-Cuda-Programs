#include <stdio.h>
#include "data_declare.h"

#define BLOCK_SIZE 256

#define NUM_LAYERS    3
#define N             30
#define M             1
#define NUM_YEARS     280

__device__ void SetInput(NET* Net, REAL* Input)
{
  // printf("Entered SetInput method ....");
  
 // for (INT i=1; i<=Net->InputLayer->Units; i++) {
  for (INT i=1; i<=31; i++) {
    // printf("Index: %d, Input: %f\n", i, Input[i]);
    Net->InputLayer->Output[i] = Input[i-1];
    // printf("Input[i-1]: %f\n", Net->InputLayer->Output[i]);
  }
}

__device__ void PropagateLayer(NET* Net, LAYER* Lower, LAYER* Upper)
{
  INT  i,j;
  REAL Sum;
  // printf("Entered PropagateLayer method ....");

  for (i=1; i<=Upper->Units; i++) {
    // printf("Entered for loop upper ....");
    Sum = 0;
    for (j=0; j<=Lower->Units; j++) {
      // printf("Entered for loop lower ....");
      Sum += Upper->Weight[i][j] * Lower->Output[j];
      // printf("SUM : %f\n", Sum);
    }
    Upper->Output[i] = 1 / (1 + exp(-Net->Gain * Sum));
    // printf("Upper->Output[i] : %f\n", Upper->Output[i]);
  }
}

__device__ void PropagateNet(NET* Net)
{
  // printf("Entered PropagateNet method ....");
  
  for (INT l=0; l<NUM_LAYERS-1; l++) {
    // printf("Entered for loop layers ....");
    PropagateLayer(Net, Net->Layer[l], Net->Layer[l+1]);
  }
}

__device__ void GetOutput(NET* Net, REAL* Output)
{
  INT i;
  // printf("Entered GetOutput method ....");

  for (i=1; i<=Net->OutputLayer->Units; i++) {
    Output[i-1] = Net->OutputLayer->Output[i];
    // printf("Output[i-1] : %f\n", Output[i-1]);
  }
}

__device__ void ComputeOutputError(NET* Net, REAL* Target)
{
  INT  i;
  REAL Out, Err;
  // printf("Entered ComputeOutputError method ....");

  Net->Error = 0;
  for (i=1; i<=Net->OutputLayer->Units; i++) {
    Out = Net->OutputLayer->Output[i];
    Err = Target[i-1]-Out;
    Net->OutputLayer->Error[i] = Net->Gain * Out * (1-Out) * Err;
    Net->Error += 0.5 * sqr(Err);
    // printf("Net->Error : %f\n", Net->Error);
  }
}

__device__ void BackpropagateLayer(NET* Net, LAYER* Upper, LAYER* Lower)
{
  INT  i,j;
  REAL Out, Err;
  // printf("Entered BackpropagateLayer method ....");

  for (i=1; i<=Lower->Units; i++) {
    Out = Lower->Output[i];
    Err = 0;
    for (j=1; j<=Upper->Units; j++) {
      Err += Upper->Weight[j][i] * Upper->Error[j];
    }
    Lower->Error[i] = Net->Gain * Out * (1-Out) * Err;
    // printf("Lower->Error[i] : %f\n", Lower->Error[i]);
  }
}

__device__ void BackpropagateNet(NET* Net)
{
  INT l;
   
  for (l=NUM_LAYERS-1; l>1; l--) {
    BackpropagateLayer(Net, Net->Layer[l], Net->Layer[l-1]);
  }
}

__device__ void AdjustWeights(NET* Net)
{
  INT  l,i,j;
  REAL Out, Err, dWeight;
  // printf("Entered AdjustWeights method ....");

  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Out = Net->Layer[l-1]->Output[j];
        Err = Net->Layer[l]->Error[i];
        dWeight = Net->Layer[l]->dWeight[i][j];
        Net->Layer[l]->Weight[i][j] += Net->Eta * Err * Out + Net->Alpha * dWeight;
        Net->Layer[l]->dWeight[i][j] = Net->Eta * Err * Out;
        // printf("Net->Layer[l]->dWeight[i][j] : %f\n", Net->Layer[l]->dWeight[i][j]);
      }
    }
  }
}

__device__ void SimulateNet (NET* Net, REAL* Input, REAL* Output, REAL* Target, BOOL Training)
{
  // printf("Entered SimulateNet method ....");
  SetInput(Net, Input);
  PropagateNet(Net);
  GetOutput(Net, Output);
  ComputeOutputError(Net, Target);
  if (Training) {
    BackpropagateNet(Net);
    AdjustWeights(Net);
  }
}

__device__ INT RandomEqualINT_krnl(INT Low, INT High) {
  // printf("Entered RandomEqualINT_krnl method ....");
  return (High - Low + 1) + Low;
}

__global__ void TrainNet_krnl(NET* Net, REAL* Input, INT lwb, INT upb, INT n, INT m, INT tr_yeras)
{ 
  INT idx = threadIdx.x + blockIdx.x * blockDim.x;
  INT Year;
  REAL Output[M];

  if (idx >= tr_yeras) {
    return;
  }

  // printf("idx: %d\n", idx);

  // Year = RandomEqualINT_krnl(30, 179);

  if (idx < tr_yeras) {
    Year = RandomEqualINT_krnl(lwb, upb);
    if (Year >= 30 && Year < NUM_YEARS) {
      // printf("Enetered if year ....");
      SimulateNet(Net, &Input[Year - 30], Output, &Input[Year], TRUE);
      __syncthreads();
    }  
  }
}

__global__ void TestNet_krnl(NET* Net, REAL* Input, INT no_yeras, INT Year, REAL* Train_error, REAL* Test_error)
{ 
  INT idx = threadIdx.x + blockIdx.x * blockDim.x;
  REAL Output[M];

  REAL TrainError = 0;
  REAL TestError = 0;
  // printf("Enetered TestNet method....");
  Net->Error = 0;
  if (idx >= no_yeras) {
    return;
  }

  if (idx < no_yeras) {
      // printf("Enetered if loop....");
        SimulateNet(Net, &Input[Year-30], Output, &Input[Year], FALSE);
        TrainError += Net->Error;
        *Train_error = TrainError;
        // printf("TrainError :   %f , Train_error_Host:  %f\n", TrainError, *Train_error);
        __syncthreads();
        if(Year + 150 <= 259 ){
          SimulateNet(Net, &Input[Year-30 + 150], Output, &Input[Year + 150], FALSE);
          TestError += Net->Error;
          *Test_error = TestError;
          // printf("TestError :   %f , Test_error_Host:  %f\n", TestError, *Test_error);
        }
        __syncthreads();
  }
}

__global__ void EvaluateNet_krnl(NET* Net, REAL* Input, INT no_yeras, INT Year, REAL* Train_error, REAL* Test_error)
{ 
  INT idx = threadIdx.x + blockIdx.x * blockDim.x;
  REAL Output[M];

  REAL TrainError = 0;
  REAL TestError = 0;
  // printf("Enetered TestNet method....");
  Net->Error = 0;
  if (idx >= no_yeras) {
    return;
  }

  if (idx < no_yeras) {
      // printf("Enetered if loop....");
        SimulateNet(Net, &Input[Year-30], Output, &Input[Year], FALSE);
        TrainError += Net->Error;
        *Train_error = TrainError;
        // printf("TrainError :   %f , Train_error_Host:  %f\n", TrainError, *Train_error);
        __syncthreads();
  }
}

void TrainNet(NET* Net, INT epochs, REAL* Input) {
  INT totalIterations = epochs * 150;

  dim3 dim_Grid, dim_Block;
  dim_Block.x = BLOCK_SIZE; 
  dim_Block.y, dim_Block.z = 1;
  
  dim_Grid.x = (totalIterations-1)/BLOCK_SIZE+1;;
  dim_Grid.y = dim_Grid.z = 1;

  printf("dim_Block.x: %zu , dim_Grid.x: %zu \n", dim_Block.x, dim_Grid.x);

  printf("Executing TrainNet kernel...");

  for (INT i = 0; i < epochs; i++){
    TrainNet_krnl<<<dim_Grid, dim_Block>>>(Net, Input, 30, 179, 30, 1, 150);
  }

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error launching kernel: %s\n", cudaGetErrorString(err));
    exit(1);
  }
}

void TestNet(NET* Net, REAL* Input, REAL* Train_error, REAL* Test_error)
{
  INT totalIterations = NUM_YEARS;
  INT Year;

  dim3 dim_Grid, dim_Block;
  dim_Block.x = BLOCK_SIZE; 
  dim_Block.y, dim_Block.z = 1;
  
  dim_Grid.x = (totalIterations-1)/BLOCK_SIZE+1;;
  dim_Grid.y = dim_Grid.z = 1;

  printf("dim_Block.x: %zu , dim_Grid.x: %zu \n", dim_Block.x, dim_Grid.x);

  printf("Executing TestNet kernel...");

  //trainnet and trainnet
  for (Year=30; Year<=179; Year++){
    TestNet_krnl<<<dim_Grid, dim_Block>>>(Net, Input, 280, Year, Train_error, Test_error);
  }

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error launching kernel: %s\n", cudaGetErrorString(err));
    exit(1);
  }
}

void EvaluateNet(NET* Net, REAL* Input, REAL* Train_error, REAL* Test_error)
{
  INT totalIterations = NUM_YEARS;
  INT Year;

  dim3 dim_Grid, dim_Block;
  dim_Block.x = BLOCK_SIZE; 
  dim_Block.y, dim_Block.z = 1;
  
  dim_Grid.x = (totalIterations-1)/BLOCK_SIZE+1;;
  dim_Grid.y = dim_Grid.z = 1;

  printf("dim_Block.x: %zu , dim_Grid.x: %zu \n", dim_Block.x, dim_Grid.x);

  printf("Executing Evaluate kernel...");

  //trainnet and trainnet
  for (Year=260; Year<=279; Year++){
    EvaluateNet_krnl<<<dim_Grid, dim_Block>>>(Net, Input, 280, Year, Train_error, Test_error);
  }

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error launching kernel: %s\n", cudaGetErrorString(err));
    exit(1);
  }
}

