#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernel.cu"
#include "support.h"
#include "data_declare.h"

void InitializeRandoms()
{
  srand(4711);
}

REAL RandomEqualREAL(REAL Low, REAL High)
{
  return ((REAL) rand() / RAND_MAX) * (High-Low) + Low;
}     

void GenerateNetwork(NET* Net)
{
  INT l,i;

  Net->Layer = (LAYER**) calloc(NUM_LAYERS, sizeof(LAYER*));
  cudaMallocManaged(&Net->Layer, NUM_LAYERS * sizeof(LAYER*));
  
  cudaError_t err = cudaMallocManaged(&(Net->Layer), NUM_LAYERS * sizeof(LAYER*));
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
  }

  for (l=0; l<NUM_LAYERS; l++) {
    Net->Layer[l] = (LAYER*) malloc(sizeof(LAYER));
    cudaMallocManaged(&Net->Layer[l], sizeof(LAYER));

    Net->Layer[l]->Units  = Units[l];
    // cudaMallocManaged(&(Net->Layer[l]->Units), sizeof(INT));

    Net->Layer[l]->Output     = (REAL*)  calloc(Units[l]+1, sizeof(REAL));
    cudaMallocManaged(&Net->Layer[l]->Output, Units[l]+1 * sizeof(REAL));

    Net->Layer[l]->Error      = (REAL*)  calloc(Units[l]+1, sizeof(REAL));
    cudaMallocManaged(&Net->Layer[l]->Error, Units[l]+1 * sizeof(REAL));

    Net->Layer[l]->Weight     = (REAL**) calloc(Units[l]+1, sizeof(REAL*));
    cudaMallocManaged(&Net->Layer[l]->Weight, Units[l]+1 * sizeof(REAL*));

    Net->Layer[l]->WeightSave = (REAL**) calloc(Units[l]+1, sizeof(REAL*));
    cudaMallocManaged(&Net->Layer[l]->WeightSave, Units[l]+1 * sizeof(REAL*));

    Net->Layer[l]->dWeight    = (REAL**) calloc(Units[l]+1, sizeof(REAL*));
    cudaMallocManaged(&Net->Layer[l]->dWeight, Units[l]+1 * sizeof(REAL*));
    
    Net->Layer[l]->Output[0]  = BIAS;
    // cudaMallocManaged(&(Net->Layer[l]->Output[0]), sizeof(REAL));
      
    if (l != 0) {
      for (i=1; i<=Units[l]; i++) {
        Net->Layer[l]->Weight[i]     = (REAL*) calloc(Units[l-1]+1, sizeof(REAL));
        cudaMallocManaged(&Net->Layer[l]->Weight[i], Units[l-1]+1 * sizeof(REAL));

        Net->Layer[l]->WeightSave[i] = (REAL*) calloc(Units[l-1]+1, sizeof(REAL));
        cudaMallocManaged(&Net->Layer[l]->WeightSave[i], Units[l-1]+1 * sizeof(REAL));

        Net->Layer[l]->dWeight[i]    = (REAL*) calloc(Units[l-1]+1, sizeof(REAL));
        cudaMallocManaged(&Net->Layer[l]->dWeight[i], Units[l-1]+1 * sizeof(REAL));
      }
    }
  }
  Net->InputLayer  = Net->Layer[0];
  Net->OutputLayer = Net->Layer[NUM_LAYERS - 1];
  Net->Alpha       = 0.9;
  Net->Eta         = 0.25;
  Net->Gain        = 1;
}

void RandomWeights(NET* Net)
{
  INT l,i,j;
   
  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Net->Layer[l]->Weight[i][j] = RandomEqualREAL(-0.5, 0.5);
      }
    }
  }
}

void NormalizeSunspots()
{
  INT  Year;
  REAL Min, Max;
	
  Min = MAX_REAL;
  Max = MIN_REAL;
  for (Year=0; Year<NUM_YEARS; Year++) {
    Min = MIN(Min, Sunspots[Year]);
    Max = MAX(Max, Sunspots[Year]);
  }
  Mean = 0;
  for (Year=0; Year<NUM_YEARS; Year++) {
    Sunspots_[Year] = 
    Sunspots [Year] = ((Sunspots[Year]-Min) / (Max-Min)) * (HI-LO) + LO;
    Mean += Sunspots[Year] / NUM_YEARS;
  }
}


void InitializeApplication(NET* Net)
{
  INT  Year, i;
  REAL Out, Err;

  Net->Alpha = 0.5;
  Net->Eta   = 0.05;
  Net->Gain  = 1;

  NormalizeSunspots();
  TrainErrorPredictingMean = 0;
  for (Year=TRAIN_LWB; Year<=TRAIN_UPB; Year++) {
    for (i=0; i<M; i++) {
      Out = Sunspots[Year+i];
      Err = Mean - Out;
      TrainErrorPredictingMean += 0.5 * sqr(Err);
    }
  }
  TestErrorPredictingMean = 0;
  for (Year=TEST_LWB; Year<=TEST_UPB; Year++) {
    for (i=0; i<M; i++) {
      Out = Sunspots[Year+i];
      Err = Mean - Out;
      TestErrorPredictingMean += 0.5 * sqr(Err);
    }
  }
  f = fopen("BPN.txt", "w");
}

void PrintNetwork(const NET* Net) {
    printf("Network Information:\n");
    printf("Number of Layers: %d\n", NUM_LAYERS);
    printf("Alpha: %f\n", Net->Alpha);
    printf("Eta: %f\n", Net->Eta);
    printf("Gain: %f\n", Net->Gain);

    for (int l = 0; l < NUM_LAYERS; l++) {
        printf("\nLayer %d:\n", l);
        printf("Units: %d\n", Net->Layer[l]->Units);

        for (int i = 0; i <= Net->Layer[l]->Units; i++) {
            printf("  Unit %d:\n", i);
            printf("    Output: %f\n", Net->Layer[l]->Output[i]);
            printf("    Error: %f\n", Net->Layer[l]->Error[i]);

            if (l != 0 && i!=0) {
                for (int j = 1; j <= Units[l-1]+1; j++) {
                    printf("    Weight[%d][%d]: %f\n", i, j, Net->Layer[l]->Weight[i][j]);
                    // Add more details if needed
                }
            }
        }
    }
}

void SaveWeights(NET* Net)
{
  INT l,i,j;

  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Net->Layer[l]->WeightSave[i][j] = Net->Layer[l]->Weight[i][j];
      }
    }
  }
}


void RestoreWeights(NET* Net)
{
  INT l,i,j;

  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Net->Layer[l]->Weight[i][j] = Net->Layer[l]->WeightSave[i][j];
      }
    }
  }
}

void EvaluateNet_krnl(NET* Net)
{
  INT  Year;
  REAL Output [M];
  REAL Output_[M];

  fprintf(f, "\n\n\n");
  fprintf(f, "Year    Sunspots    Open-Loop Prediction    Closed-Loop Prediction\n");
  fprintf(f, "\n");
  for (Year=EVAL_LWB; Year<=EVAL_UPB; Year++) {
    SimulateNet_krnl(Net, &(Sunspots [Year-N]), Output,  &(Sunspots [Year]), FALSE);
    SimulateNet_krnl(Net, &(Sunspots_[Year-N]), Output_, &(Sunspots_[Year]), FALSE);
    Sunspots_[Year] = Output_[0];
    fprintf(f,"%d       %0.3f                   %0.3f                     %0.3f\n",
               FIRST_YEAR + Year,
               Sunspots[Year],
               Output [0],
               Output_[0]);
  }
}

void FinalizeApplication(NET* Net)
{
  fclose(f);
}

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;
    BOOL Stop;
    REAL MinTestError;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    dim3 dim_grid, dim_block;

    NET Net;
    REAL *Output = (REAL*) malloc ((sizeof(REAL) * M));
    REAL Input[NUM_YEARS];

    REAL Train_error = 0;
    REAL Test_error =0;

    NET *d_Net;
    REAL *d_Input;
    REAL *d_Output;
    REAL *d_Train_error;
    REAL *d_Test_error;
    size_t net_sz, input_sz, output_sz;

    net_sz = sizeof(NET);
    input_sz = sizeof(REAL) * NUM_YEARS;
    output_sz = sizeof(REAL) * M;

    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    int ndev;
    cudaGetDeviceCount(&ndev);
    if(ndev==0){
        printf(" GPU DEVICES ARE UNAVAILABLE\n\n");
        exit(-1);
            
    }else{
        printf("\nAvailable Number of GPU: %d\n",ndev);
    }

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Net: %u x %u\n    Output: %u x %u\n    Input: %u x %u\n", net_sz, input_sz,
        output_sz);

    for (int i = 0; i < NUM_YEARS; i++) {
      Input[i] = Sunspots[i];
    }
    printf("Input array:\n");
    for (int i = 0; i < NUM_YEARS; i++) {
        printf("%f ", Input[i]);
    }
    printf("\n");
    // Back propogation code ----------------------------------------------
    printf("Starting the Back propogation code..."); fflush(stdout);
    
    InitializeRandoms();
    printf("Completed InitializeRandoms...");
    GenerateNetwork(&Net);
    printf("Completed GenerateNetwork...");
    //PrintNetwork(&Net);

    RandomWeights(&Net);
    printf("Completed RandomWeights...");
    //PrintNetwork(&Net);

    InitializeApplication(&Net);
    printf("Completed InitializeApplication...");
    // PrintNetwork(&Net);

    Stop = FALSE;
    MinTestError = MAX_REAL;

    printf("Allocating device variables...."); fflush(stdout);
    startTime(&timer);

    cudaMalloc((void**)&d_Net, sizeof(NET));
    cudaMalloc((void**)&d_Input, sizeof(REAL) * NUM_YEARS);
    cudaMalloc((void**)&d_Output, sizeof(REAL) * M);
    
    cudaMalloc((REAL**)&d_Train_error, sizeof(REAL));
    cudaMalloc((REAL**)&d_Test_error, sizeof(REAL));

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cudaMemcpy(d_Net, &Net, sizeof(NET), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input, &Input, sizeof(REAL) * NUM_YEARS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Train_error, &Train_error, sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Test_error, &Test_error, sizeof(REAL), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    size_t free_byte, total_byte;
    free_byte, total_byte = cudaMemGetInfo(&free_byte, &total_byte);
    printf("Free GPU memory: %zu bytes, Total GPU memory: %zu bytes\n", free_byte, total_byte);

    printf("Launching kernel...");
    startTime(&timer);

    INT epochs = 10;
    
    do {
      TrainNet(d_Net, epochs, d_Input);
      printf("Copying data from device to host for TrainNet..."); fflush(stdout);
      startTime(&timer);
      // Copy the output data back to the CPU
      cudaMemcpy(&Net, d_Net, sizeof(NET), cudaMemcpyDeviceToHost);

      cudaDeviceSynchronize();
      stopTime(&timer); printf("%f s\n", elapsedTime(timer));

      TestNet(d_Net, d_Input, d_Train_error, d_Test_error);

      printf("Copying data from device to host for TestNet..."); fflush(stdout);
      startTime(&timer);
      // Copy the output data back to the CPU
      cudaMemcpy(&Train_error, d_Train_error, sizeof(REAL), cudaMemcpyDeviceToHost);
      cudaMemcpy(&Test_error, d_Test_error, sizeof(REAL), cudaMemcpyDeviceToHost);

      printf("Kernel code calculated Train Error: %f , Test error:  %f \n", Train_error, Test_error);
      printf("\nNMSE is %0.3f on Training Set and %0.3f on Test Set",
             Train_error / TrainErrorPredictingMean,
             Test_error / TestErrorPredictingMean);

      cudaDeviceSynchronize();
      stopTime(&timer); printf("Completion of kernel code for Training and Test....%f s\n", elapsedTime(timer));

      if (Test_error < MinTestError) {
        printf(" - saving Weights ...");
        MinTestError = Test_error;
        SaveWeights(&Net);
      }
      else if (Test_error > 1.2 * MinTestError) {
        printf(" - stopping Training and restoring Weights ...");
        Stop = TRUE;
        RestoreWeights(&Net);
      }
      printf(" - stopping Training ...");
      Stop = TRUE;
    } while (NOT Stop);
    
    TestNet(d_Net, d_Input, d_Train_error, d_Test_error);
    printf("Copying data from device to host for Final TestNet..."); fflush(stdout);
    startTime(&timer);
      // Copy the output data back to the CPU
    cudaMemcpy(&Train_error, d_Train_error, sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaMemcpy(&Test_error, d_Test_error, sizeof(REAL), cudaMemcpyDeviceToHost);

    printf("Kernel code calculated Train Error: %f , Test error:  %f \n", Train_error, Test_error);
    printf("\nNMSE is %0.3f on Training Set and %0.3f on Test Set",
             Train_error / TrainErrorPredictingMean,
             Test_error / TestErrorPredictingMean);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("Completion of kernel code for final TestNet....%f s\n", elapsedTime(timer));

    // EvaluateNet(d_Net, d_Input, d_Train_error, d_Test_error);
    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);
      // Copy the output data back to the CPU
    cudaMemcpy(&Net, d_Net, sizeof(NET), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("Completion of whole kernel code....%f s\n", elapsedTime(timer));

    EvaluateNet_krnl(&Net);
   
    FinalizeApplication(&Net);

    cudaFreeHost(d_Net);
    cudaFreeHost(d_Input);
    cudaFreeHost(d_Output);
    cudaFreeHost(d_Train_error);
    cudaFreeHost(d_Test_error);

    cudaFree(d_Net);
    cudaFree(d_Input);
    cudaFree(d_Output);
    cudaFree(d_Train_error);
    cudaFree(d_Test_error);
    return 0;
}
