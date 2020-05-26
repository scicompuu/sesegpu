#include "data.h"
#include <vector>
#include <chrono>
#include <iostream>
#include <omp.h>

namespace chrono = std::chrono;

int main()
{
  printf("OpenMP target.\n");
  int elems = data.extent(0);
  printf("Procesing %d * %d elements\n", elems, elems);
  // Relying on unified virtual memory
  auto computestart = chrono::steady_clock::now();

  // #pragma omp target data map(to:indexeddivergence)
  uint8_t* origdata = data.data();
  long long maxdivergence = 0;
  const int bs = 16;

#pragma omp target teams map(to:origdata[:elems*side*side]) map(tofrom:maxdivergence) reduction(max:maxdivergence)
  {
    mnisttype data = mnisttype(origdata, elems);
#pragma omp distribute collapse(2)
  for (int ibase = 0; ibase < elems; ibase+=bs)
    {
      for (int jbase = 0; jbase < elems; jbase+=bs)
	{
#pragma omp parallel for reduction(max:maxdivergence) collapse(2)
	  for (int istep = 0; istep < bs; istep++)
	    {
	      for (int jstep = 0; jstep < bs; jstep++)
		{
		  const int i = ibase + istep;
		  const int j = jbase + jstep;
		  int sum = 0;
		  //for (int y = 0; y < side; y++)
		    {
		      for (int x = 0; x < side * side; x++)
			{
			  int diff = (int) data(i, 0, x) - (int) data(j, 0, x);
			  sum += diff * diff;
			}
		    }
		  long long newdiv = sum;
		  newdiv <<= 32;
		  newdiv += i << 16;
		  newdiv += j;

		  if (newdiv > maxdivergence)
		    {
		      maxdivergence = newdiv;
		    }
		}
	    }
	}
    }
  }
  auto firstend = chrono::steady_clock::now();
  int maxi = maxdivergence >> 16;
  maxi &= (1 << 16) - 1;
  int maxj = maxdivergence;
  maxj &= (1 << 16) - 1;
  int divergence = maxdivergence >> 32;
  
  std::cout << "First pass, in microseconds : " << chrono::duration_cast<chrono::microseconds>(firstend-computestart).count() << std::endl;
  printf("Maximum divergence at %d against %d with value %d\n", maxi, maxj, divergence);

}
