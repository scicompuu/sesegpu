#include "data.h"
#include <vector>
#include <chrono>
#include <iostream>
#include <thrust/functional.h>
#include <thrust/for_each.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

namespace chrono = std::chrono;

__managed__ uint8_t* cudadataptr;
__managed__ int cudaelems;

struct computej
{
	int i;

	int __device__ operator() (int j)
	{
	  auto summer = [] (int a, int b)
	    {
	      int diff = a - b;
	      return diff * diff;
	    };
	  mnisttype data = mnisttype(cudadataptr, cudaelems);
	  return thrust::inner_product(thrust::device, &data(i, 0, 0), &data(i, 0, 0) + side * side, &data(j, 0, 0), 0, thrust::plus<int>(), summer);

	}
};

void __global__ dowork(int elems, int& maxi, int& maxj, int& divergence)
{
	thrust::counting_iterator<int32_t> zeroit(0);
	computej computer;

	auto domaxj = [] (int i)
	{
	        computej computer;
		computer.i = i / cudaelems;
		return computer(i % cudaelems);
	};

	auto transformer = make_transform_iterator(thrust::make_counting_iterator(0), domaxj);
	int maxval = thrust::max_element(thrust::seq, transformer, transformer + cudaelems * cudaelems) - transformer;
	maxj = maxval % cudaelems;
	maxi = maxval / cudaelems;

	computer.i = maxi;
	divergence = computer(maxj);
}


int main()
{
  printf("Thrust (max).\n");
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

  mnisttype localdata = mnisttype(cudadataptr, elems);
  cudaError e;

  cudaMemcpy(cudadataptr, data.data(), sizeof(uint8_t) * elems * side * side, cudaMemcpyHostToDevice);
  cudaMemAdvise(cudadataptr, sizeof(uint8_t) * elems * side * side, cudaMemAdviseSetReadMostly, 0);
  dowork<<<1,1>>>(elems, maxi, maxj, divergence);

  e = cudaDeviceSynchronize();
  if (e)
  {
	printf("Cuda error %d reported: %s\n", e, cudaGetErrorString(e));
  }
  auto firstend = chrono::steady_clock::now();
  
  std::cout << "First pass, in microseconds : " << chrono::duration_cast<chrono::microseconds>(firstend-computestart).count() << std::endl;
  printf("Maximum divergence at %d against %d with value %d\n", maxi, maxj, divergence);
}
