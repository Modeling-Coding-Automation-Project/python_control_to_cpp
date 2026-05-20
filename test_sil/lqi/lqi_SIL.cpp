#include "lqi_SIL_wrapper.hpp"
#include "python_control.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using FLOAT = typename lqi_SIL_wrapper::type::Value_Type;

constexpr std::size_t INPUT_SIZE = lqi_SIL_wrapper::INPUT_SIZE;
constexpr std::size_t Q_EX_SIZE = lqi_SIL_wrapper::Q_EX_SIZE;

lqi_SIL_wrapper::type lqi;

void initialize(void) { lqi = lqi_SIL_wrapper::make(); }

py::array_t<FLOAT> solve(void) {
  auto K = lqi.solve();

  py::array_t<FLOAT> result;
  result.resize({static_cast<int>(INPUT_SIZE), static_cast<int>(Q_EX_SIZE)});

  for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
    for (std::size_t j = 0; j < Q_EX_SIZE; ++j) {
      result.mutable_at(i, j) = K.access(i, j);
    }
  }

  return result;
}

py::array_t<FLOAT> get_K(void) {
  auto K = lqi.get_K();

  py::array_t<FLOAT> result;
  result.resize({static_cast<int>(INPUT_SIZE), static_cast<int>(Q_EX_SIZE)});

  for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
    for (std::size_t j = 0; j < Q_EX_SIZE; ++j) {
      result.mutable_at(i, j) = K.access(i, j);
    }
  }

  return result;
}

void set_Eigen_solver_iteration_max(const int max_iteration) {

  lqi.set_Eigen_solver_iteration_max(static_cast<std::size_t>(max_iteration));
}

void set_Eigen_solver_iteration_max_for_eigen_vector(const int max_iteration) {

  lqi.set_Eigen_solver_iteration_max_for_eigen_vector(
      static_cast<std::size_t>(max_iteration));
}

PYBIND11_MODULE(LqiSIL, m) {
  m.def("initialize", &initialize, "initialize lqi");
  m.def("solve", &solve, "solve discrete state space lqi");
  m.def("get_K", &get_K, "get K matrix");
  m.def("set_Eigen_solver_iteration_max", &set_Eigen_solver_iteration_max,
        "set Eigen solver iteration max");
  m.def("set_Eigen_solver_iteration_max_for_eigen_vector",
        &set_Eigen_solver_iteration_max_for_eigen_vector,
        "set Eigen solver iteration max for eigen vector");
}
