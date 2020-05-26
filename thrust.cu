#include "data.h"
#include <vector>
#include <chrono>
#include <iostream>
#include <functional>
#include <thrust/for_each.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>
#include <assert.h>
#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
__device__ void cdpAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(0);
   }
}

namespace chrono = std::chrono;

struct squarediff
{
	uint8_t const* i;
	uint8_t const* j;
	int __device__ operator() (int k)
	{
		int diff = i[k] - j[k];
		return diff * diff;
	}
};

template<class T1, class T2> void __global__ dowork(T1 indexeddivergence, int elems, T2 data, int* c)
{
  thrust::counting_iterator<int32_t> zeroit(0);
  
  thrust::for_each_n(thrust::device, zeroit, elems, [indexeddivergence, elems, data] (int i)
		     {
		       auto zeroit = thrust::make_counting_iterator(0);
		       
		       thrust::transform(thrust::device, zeroit, zeroit + elems, &indexeddivergence(i, 0), [i, data] (int j)
					 {
					   squarediff differ;
					   differ.i = &data(i, 0, 0);
					   differ.j = &data(j, 0, 0);
					   thrust::counting_iterator<int32_t> zeroit(0);
					   
					   return thrust::transform_reduce(thrust::seq, zeroit, zeroit + side * side, differ, 0, thrust::plus<int>());
					   
					 });
		     });
}

int main()
{
  printf("Thrust.\n");
  int elems = data.extent(0);
  int* divergence;
  cudaMallocManaged(&divergence, sizeof(int) * elems * elems);
  uint8_t* dataptr;
  cudaMallocManaged(&dataptr, sizeof(uint8_t) * elems * side * side);
  printf("Procesing %d * %d elements\n", elems, elems);
  auto indexeddivergence = stdex::basic_mdspan<int, stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>>(divergence, elems, elems);

  auto computestart = chrono::steady_clock::now();

  mnisttype localdata = mnisttype(dataptr, elems);
  cudaError e;
  e = cudaMemcpy(dataptr, data.data(), sizeof(uint8_t) * elems * side * side, cudaMemcpyHostToDevice);
  if (e)
  {
	printf("Cuda error %d reported: %s\n", e, cudaGetErrorString(e));
  }
  e= cudaMemAdvise(dataptr, sizeof(uint8_t) * elems * side * side, cudaMemAdviseSetReadMostly, 0);
  if (e)
  {
	printf("Cuda error %d reported: %s\n", e, cudaGetErrorString(e));
  }

  dowork<<<1,1>>>(indexeddivergence, elems, localdata, divergence);
  e = cudaPeekAtLastError();
  if (e)
  {
	printf("Cuda error %d reported: %s\n", e, cudaGetErrorString(e));
  }

  e = cudaDeviceSynchronize();
  if (e)
  {
	printf("Cuda error %d reported: %s\n", e, cudaGetErrorString(e));
  }
  auto firstend = chrono::steady_clock::now();

  int maxi = 0;
  int maxj = 0;
  for (int i = 0; i < elems; i++)
    {
      for (int j = 0; j < elems; j++)
	{
	  if (indexeddivergence(i, j) >= indexeddivergence(maxi, maxj))
	    {
	      maxi = i;
	      maxj = j;
	    }
	}
    }
  auto secondend = chrono::steady_clock::now();
  
  std::cout << "First pass, in microseconds : " << chrono::duration_cast<chrono::microseconds>(firstend-computestart).count() << std::endl;
  std::cout << "Second pass, in microseconds : " << chrono::duration_cast<chrono::microseconds>(secondend-firstend).count() << std::endl;
  printf("Maximum divergence at %d against %d with value %d\n", maxi, maxj, indexeddivergence(maxi, maxj));
}
