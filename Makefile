
NVCC        = nvcc
ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3 -arch=sm_20
else
NVCC_FLAGS  = -O3 
endif
LD_FLAGS    = -lcudart 
EXE	        = back-propagation
OBJ	        = main.o support.o

default: $(EXE)

main.o: main.cu kernel.cu support.h data_declare.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
