#include "discrete_state_space_SIL_wrapper.hpp"
#include "python_control.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

constexpr std::size_t INPUT_SIZE = discrete_state_space_wrapper::INPUT_SIZE;
constexpr std::size_t STATE_SIZE = discrete_state_space_wrapper::STATE_SIZE;
constexpr std::size_t OUTPUT_SIZE = discrete_state_space_wrapper::OUTPUT_SIZE;

discrete_state_space_wrapper::type sys;

void initialize(void) { sys = discrete_state_space_wrapper::make(); }

void update(py::array_t<double> U_in) {

  py::buffer_info U_info = U_in.request();

  /* check compatibility */
  if (INPUT_SIZE != U_info.shape[0]) {
    throw std::runtime_error("U must have " + std::to_string(INPUT_SIZE) +
                             " inputs.");
  }

  /* substitute */
  double *U_data_ptr = static_cast<double *>(U_info.ptr);
  PythonControl::StateSpaceInput_Type<double, INPUT_SIZE> U;
  for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
    U.access(i, 0) = U_data_ptr[i];
  }

  sys.update(U);
}

py::array_t<double> get_X(void) {
  auto X = sys.get_X();
  py::array_t<double> result;
  result.resize({static_cast<int>(STATE_SIZE), 1});

  for (std::size_t i = 0; i < STATE_SIZE; ++i) {
    result.mutable_at(i, 0) = X.access(i, 0);
  }

  return result;
}

py::array_t<double> get_Y(void) {
  auto Y = sys.get_Y();
  py::array_t<double> result;
  result.resize({static_cast<int>(OUTPUT_SIZE), 1});

  for (std::size_t i = 0; i < OUTPUT_SIZE; ++i) {
    result.mutable_at(i, 0) = Y.access(i, 0);
  }

  return result;
}

PYBIND11_MODULE(DiscreteStateSpaceSIL, m) {
  m.def("initialize", &initialize, "initialize discrete state space");
  m.def("update", &update, "update discrete state space");
  m.def("get_X", &get_X, "get state vector X");
  m.def("get_Y", &get_Y, "get output vector Y");
}
