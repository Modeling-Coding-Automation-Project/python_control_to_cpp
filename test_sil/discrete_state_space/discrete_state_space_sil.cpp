#include "discrete_state_space_sil_wrapper.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void initialize(py::array_t<double> A, py::array_t<double> B,
                py::array_t<double> C, py::array_t<double> D, double dt) {

  py::buffer_info A_info = A.request();
  py::buffer_info B_info = B.request();
  py::buffer_info C_info = C.request();
  py::buffer_info D_info = D.request();

  if (discrete_state_space_wrapper::STATE_SIZE != A_info.shape[0]) {
    throw std::runtime_error(
        "A must have " +
        std::to_string(discrete_state_space_wrapper::STATE_SIZE) + " rows");
  }
}

PYBIND11_MODULE(DiscreteStateSpaceSIL, m) {
  m.def("initialize", &initialize, "initialize discrete state space");
}
