%%writefile Addition_onGPU.cu

#include <cuda_runtime.h>
#include <iostream>

__global__ void addOne(int *data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Block %d, Thread %d -> global id %d\n",
           blockIdx.x, threadIdx.x, idx);

    data[idx]++;
}

int main()
{
    const int N=8;

    int h[N]={0,1,2,3,4,5,6,7};

    int *d;

    cudaMalloc(&d,N*sizeof(int));

    cudaMemcpy(d,h,
               N*sizeof(int),
               cudaMemcpyHostToDevice);

    addOne<<<2,N/2>>>(d);	 // 2 blocks, 4 threads each = 8 threads total

    cudaMemcpy(h,d,
               N*sizeof(int),
               cudaMemcpyDeviceToHost);
    
    std::cout<<"\n Result = ";

    for(auto x:h)
        std::cout<<x<<" ";

    cudaFree(d);
}