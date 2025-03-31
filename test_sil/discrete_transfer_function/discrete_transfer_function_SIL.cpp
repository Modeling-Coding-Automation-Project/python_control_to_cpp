#include "discrete_transfer_function_SIL_wrapper.hpp"
#include "python_control.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using FLOAT = typename discrete_transfer_function_SIL_wrapper::type::Value_Type;

constexpr std::size_t INPUT_SIZE = 1;
constexpr std::size_t STATE_SIZE =
    discrete_transfer_function_SIL_wrapper::type::STATE_SIZE;
constexpr std::size_t OUTPUT_SIZE = 1;

discrete_transfer_function_SIL_wrapper::type sys;

void initialize(void) { sys = discrete_transfer_function_SIL_wrapper::make(); }

void update(py::array_t<FLOAT> U_in) {

  py::buffer_info U_info = U_in.request();

  /* check compatibility */
  if (INPUT_SIZE != U_info.shape[0]) {
    throw std::runtime_error("U must have " + std::to_string(INPUT_SIZE) +
                             " inputs.");
  }

  FLOAT *U_data_ptr = static_cast<FLOAT *>(U_info.ptr);

  sys.update(U_data_ptr[0]);
}

py::array_t<FLOAT> get_X(void) {
  auto X = sys.get_X();
  py::array_t<FLOAT> result;
  result.resize({static_cast<int>(STATE_SIZE), 1});

  for (std::size_t i = 0; i < STATE_SIZE; ++i) {
    result.mutable_at(i, 0) = X.access(i, 0);
  }

  return result;
}

py::array_t<FLOAT> get_y(void) {
  FLOAT y = sys.get_y();
  py::array_t<FLOAT> result;
  result.resize({static_cast<int>(OUTPUT_SIZE), 1});

  result.mutable_at(0, 0) = y;

  return result;
}

FLOAT solve_steady_state_and_input(FLOAT y_steady_state) {
  return sys.solve_steady_state_and_input(y_steady_state);
}

PYBIND11_MODULE(DiscreteTransferFunctionSIL, m) {
  m.def("initialize", &initialize, "initialize discrete state space");
  m.def("update", &update, "update discrete state space");
  m.def("get_X", &get_X, "get state vector X");
  m.def("get_y", &get_y, "get output vector Y");
  m.def("solve_steady_state_and_input", &solve_steady_state_and_input,
        "solve steady state and input");
}
