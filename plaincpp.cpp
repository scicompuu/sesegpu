#include "data.h"
#include <vector>
#include <chrono>
#include <iostream>

namespace chrono = std::chrono;

int main()
{
  printf("Plain C++.\n");
  int elems = data.extent(0);
  std::vector<int> divergence;
  printf("Procesing %d * %d elements\n", elems, elems);
  divergence.resize(elems * elems);
  auto indexeddivergence = stdex::basic_mdspan<int, stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>>(divergence.data(), elems, elems);
  auto computestart = chrono::steady_clock::now();

  {

  for (int i = 0; i < elems; i++)
    {
      for (int j = 0; j < elems; j++)
	{
	  int sum = 0;
	  for (int y = 0; y < side; y++)
	    {
	      for (int x = 0; x < side; x++)
		{
		  int diff = (int) data(i, y, x) - (int) data(j, y, x);
		  sum += diff * diff;
		}
	    }
	  indexeddivergence(i, j) = sum;
	}
    }
  }
  auto firstend = chrono::steady_clock::now();
  int maxi = 0;
  int maxj = 0;
  for (int i = 0; i < elems; i++)
    {
      for (int j = 0; j < elems; j++)
	{
	  if (indexeddivergence(i, j) > indexeddivergence(maxi, maxj))
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
