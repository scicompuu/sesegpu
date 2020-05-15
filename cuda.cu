#include "data.h"
#include <vector>
#include <cub/cub.cuh>

#include <assert.h>
#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
__device__ void cdpAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
   if (code != cudaSuccess)
   {
      printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(0);
   }
}

namespace chrono = std::chrono;

__managed__ uint8_t* cudadataptr;
__managed__ int cudaelems;


constexpr int bs = 8;
void __global__ computedivergence(long long* maxdivergence)
{
	long long divergence = 0;
    	typedef cub::WarpReduce<int> WarpReduce;
    	__shared__ typename WarpReduce::TempStorage temp_storage;

	for (int istep = 0; istep < bs; istep++)
	{
		int iindex = blockIdx.y * bs + istep;
		//for (int jstep = 0; jstep < bs; jstep++)
		int jstep = threadIdx.y;
		{
			int jindex = blockIdx.x * bs + jstep;
			int sum = 0;
			for (int pixel = threadIdx.x * 4; pixel < side * side; pixel += blockDim.x * 4)
			{
				// For all threads in a warp to issue memory accesses coalesced, we need each warp
				// to access exactly 4 bytes.
				for (int sub = 0; sub < 4; sub++)
				{
					int diff = (int) cudadataptr[iindex * side * side + pixel + sub] - (int) cudadataptr[jindex * side * side + pixel + sub];
					diff *= diff;
					sum += diff;
				}
			}
			sum = WarpReduce(temp_storage).Sum(sum);
			if (threadIdx.x == 0 && sum > divergence >> 32)
			{
				divergence = sum;
				divergence <<= 32;
				divergence += iindex << 16;
				divergence += jindex;
			}
		}
	}
	if (threadIdx.x == 0) atomicMax((unsigned long long*) maxdivergence, divergence);
}

void __host__ dowork(int elems, int& maxi, int& maxj, int& divergence)
{
	int ratio = elems / bs;
	
	long long* maxdiv;
	cudaMallocManaged(&maxdiv, sizeof(long long));
	*maxdiv = 0;
	dim3 dimGrid(ratio, ratio);
	dim3 dimBlock(32, bs);
	computedivergence<<<dimGrid, dimBlock>>>(maxdiv);
	cudaDeviceSynchronize();

	divergence = *maxdiv >> 32;
	maxi = *maxdiv >> 16;
	maxi &= (1 << 16) - 1;
	maxj = *maxdiv & (1 << 16) - 1;
}


int main()
{
  printf("CUDA with cub.\n");
  int elems = data.extent(0);
  cudaelems = elems;
  int* output;
  cudaMallocManaged(&output, sizeof(int) * 3);
  
  int& maxi = output[0];
  int& maxj = output[1];
  int& divergence = output[2];


  cudaMallocManaged(&cudadataptr, sizeof(uint8_t) * elems * side * side);
  printf("Procesing %d * %d elements\n", elems, elems);

  auto computestart = chrono::steady_clock::now();

for (int x = 0; x < 1; x++)
{
  mnisttype localdata = mnisttype(cudadataptr, elems);
  cudaMemcpy(cudadataptr, data.data(), sizeof(uint8_t) * elems * side * side, cudaMemcpyHostToDevice);
  cudaMemAdvise(cudadataptr, sizeof(uint8_t) * elems * side * side, cudaMemAdviseSetReadMostly, 0);


	cudaFuncSetAttribute(dowork,
			     cudaFuncAttributePreferredSharedMemoryCarveout,
			     cudaSharedmemCarveoutMaxL1);
dowork(elems, maxi, maxj, divergence);
  cudaError e = cudaDeviceSynchronize();
  if (e)
  {
	printf("Cuda error %d reported: %s\n", e, cudaGetErrorString(e));
  }
}
  auto firstend = chrono::steady_clock::now();
  
  std::cout << "First pass, in microseconds : " << chrono::duration_cast<chrono::microseconds>(firstend-computestart).count() << std::endl;
  printf("Maximum divergence at %d against %d with value %d\n", maxi, maxj, divergence);
}
