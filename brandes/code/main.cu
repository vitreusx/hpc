#include "edge.cuh"
#include "graph.h"
#include "stock.h"
#include "stride.cuh"
#include "vertex.cuh"
#include "virtual.cuh"
#include <fstream>
#include <iostream>
#include <memory>

int main(int argc, char **argv) {
  std::cout.sync_with_stdio(false);

  if (argc < 2 || argc > 3) {
    std::cout << "Usage: " << argv[0] << " input-file [output-file]"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::unique_ptr<std::ofstream> output_file;
  std::ostream *out_stream;
  if (argc == 3) {
    output_file = std::make_unique<std::ofstream>(argv[2]);
    out_stream = output_file.get();
  } else {
    out_stream = &std::cout;
  }

  auto input_file = std::ifstream(argv[1]);
  auto g = graph::from_stream(input_file);

  //  auto res = stock_impl(g);
  //  auto res = vertex::impl(g);
  //  auto res = edge::impl(g);
  //  auto res = virt::impl(g);
  auto res = stride::impl(g);

  for (int v = 0; v < g.n; ++v)
    *out_stream << res.bc[v] << '\n';

  std::cerr << res.kernel_ms << '\n';
  std::cerr << res.total_ms << '\n';

  return EXIT_SUCCESS;
}