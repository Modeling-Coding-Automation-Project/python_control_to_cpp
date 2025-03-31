#include "python_control.hpp"
#include "recursive_least_squares_SIL_wrapper.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using FLOAT = typename recursive_least_squares_SIL_wrapper::type::Value_Type;

constexpr std::size_t X_SIZE = recursive_least_squares_SIL_wrapper::X_SIZE;
constexpr std::size_t Y_SIZE = recursive_least_squares_SIL_wrapper::Y_SIZE;
constexpr std::size_t WEIGHTS_SIZE = X_SIZE + 1;

recursive_least_squares_SIL_wrapper::type rls;

void initialize(void) { rls = recursive_least_squares_SIL_wrapper::make(); }

void update(py::array_t<FLOAT> X_in, py::array_t<FLOAT> Y_in) {

  py::buffer_info X_info = X_in.request();
  py::buffer_info Y_info = Y_in.request();

  /* check compatibility */
  if (X_SIZE != X_info.shape[0]) {
    throw std::runtime_error("X must be " + std::to_string(X_SIZE) + " size.");
  }

  if (Y_SIZE != Y_info.shape[0]) {
    throw std::runtime_error("Y must be " + std::to_string(Y_SIZE) + " size.");
  }

  /* substitute */
  FLOAT *X_data_ptr = static_cast<FLOAT *>(X_info.ptr);
  PythonControl::StateSpaceState_Type<FLOAT, X_SIZE> X;
  for (std::size_t i = 0; i < X_SIZE; i++) {
    X.access(i, 0) = X_data_ptr[i];
  }

  FLOAT *Y_data_ptr = static_cast<FLOAT *>(Y_info.ptr);
  FLOAT y = Y_data_ptr[0];

  rls.update(X, y);
}

py::array_t<FLOAT> get_weights(void) {
  auto weights = rls.get_weights();

  py::array_t<FLOAT> result;
  result.resize({static_cast<int>(WEIGHTS_SIZE), 1});

  for (std::size_t i = 0; i < WEIGHTS_SIZE; ++i) {
    result.mutable_at(i, 0) = weights.access(i, 0);
  }

  return result;
}

PYBIND11_MODULE(RecursiveLeastSquaresSIL, m) {
  m.def("initialize", &initialize, "initialize recursive least squares");
  m.def("update", &update, "update recursive least squares");
  m.def("get_weights", &get_weights, "get weights");
}
