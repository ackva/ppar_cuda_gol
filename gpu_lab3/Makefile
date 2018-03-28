EXECUTABLE  := life
OBJECTS := life.o utils.o

CUDA_PATH ?=/usr/local/cuda
NVCC=$(CUDA_PATH)/bin/nvcc

CUDAFLAGS := -g -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52
LDFLAGS := -g -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -c $<

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(OBJECTS)  $(LDFLAGS) -o $(EXECUTABLE)

life.o: life_kernel.cu

clean:
	rm -f $(EXECUTABLE) $(OBJECTS)

