#include "pid_controller_SIL_wrapper.hpp"
#include "python_control.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using FLOAT = typename pid_controller_SIL_wrapper::type::Value_Type;

pid_controller_SIL_wrapper::type pid;

void initialize(void) { pid = pid_controller_SIL_wrapper::make(); }

FLOAT update(FLOAT error) { return pid.update(error); }

PYBIND11_MODULE(PidControllerSIL, m) {
  m.def("initialize", &initialize, "initialize least squares");
  m.def("update", &update, "update");
}
