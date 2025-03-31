#include "lqr_SIL_wrapper.hpp"
#include "python_control.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using FLOAT = typename lqr_SIL_wrapper::type::Value_Type;

constexpr std::size_t INPUT_SIZE = lqr_SIL_wrapper::INPUT_SIZE;
constexpr std::size_t STATE_SIZE = lqr_SIL_wrapper::STATE_SIZE;

lqr_SIL_wrapper::type lqr;

void initialize(void) { lqr = lqr_SIL_wrapper::make(); }

py::array_t<FLOAT> solve(void) {
  auto K = lqr.solve();

  py::array_t<FLOAT> result;
  result.resize({static_cast<int>(INPUT_SIZE), static_cast<int>(STATE_SIZE)});

  for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
    for (std::size_t j = 0; j < STATE_SIZE; ++j) {
      result.mutable_at(i, j) = K.access(i, j);
    }
  }

  return result;
}

py::array_t<FLOAT> get_K(void) {
  auto K = lqr.get_K();

  py::array_t<FLOAT> result;
  result.resize({static_cast<int>(INPUT_SIZE), static_cast<int>(STATE_SIZE)});

  for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
    for (std::size_t j = 0; j < STATE_SIZE; ++j) {
      result.mutable_at(i, j) = K.access(i, j);
    }
  }

  return result;
}

PYBIND11_MODULE(LqrSIL, m) {
  m.def("initialize", &initialize, "initialize lqr");
  m.def("solve", &solve, "solve discrete state space LQR");
  m.def("get_K", &get_K, "get K matrix");
}
