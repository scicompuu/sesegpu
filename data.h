#include <cnpy.h>
#include <stdio.h>
#include <wordexp.h>
#include <experimental/mdspan>

namespace stdex = std::experimental;
constexpr int side = 28;
using mnisttype = stdex::basic_mdspan<uint8_t, stdex::extents<stdex::dynamic_extent, side, side> >;
// Global scope to keep
cnpy::NpyArray x_test;

mnisttype getdata()
{
  wordexp_t* exp = new wordexp_t;
  if (wordexp("~/.keras/datasets/mnist.npz", exp, WRDE_NOCMD) != 0 || exp->we_wordc != 1)
    {
      fprintf(stderr, "Error expanding.\n");
      std::terminate();
    }
  fprintf(stderr, "Loading file %s\n", exp->we_wordv[0]);
  auto inputdata = cnpy::npz_load(exp->we_wordv[0]);
  wordfree(exp);
  
  x_test = inputdata["x_test"];
  return mnisttype(x_test.data<uint8_t>(), (int) x_test.shape[0]);  
}

const mnisttype data = getdata();
