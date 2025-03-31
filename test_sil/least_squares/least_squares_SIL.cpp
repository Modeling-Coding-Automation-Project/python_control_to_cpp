#include "least_squares_SIL_wrapper.hpp"
#include "python_control.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using FLOAT = typename least_squares_SIL_wrapper::type::Value_Type;

constexpr std::size_t LS_NUMBER_OF_DATA =
    least_squares_SIL_wrapper::LS_NUMBER_OF_DATA;
constexpr std::size_t X_SIZE = least_squares_SIL_wrapper::X_SIZE;
constexpr std::size_t Y_SIZE = least_squares_SIL_wrapper::Y_SIZE;
constexpr std::size_t WEIGHTS_SIZE = X_SIZE + 1;

least_squares_SIL_wrapper::type ls;

void initialize(void) { ls = least_squares_SIL_wrapper::make(); }

void fit(py::array_t<FLOAT> X_in, py::array_t<FLOAT> Y_in) {

  py::buffer_info X_info = X_in.request();
  py::buffer_info Y_info = Y_in.request();

  /* check compatibility */
  if (LS_NUMBER_OF_DATA != X_info.shape[0]) {
    throw std::runtime_error("X must have " +
                             std::to_string(LS_NUMBER_OF_DATA) + " data.");
  }
  if (X_SIZE != X_info.shape[1]) {
    throw std::runtime_error("X must have " + std::to_string(X_SIZE) +
                             " columns.");
  }

  if (LS_NUMBER_OF_DATA != Y_info.shape[0]) {
    throw std::runtime_error("Y must have " +
                             std::to_string(LS_NUMBER_OF_DATA) + " data.");
  }
  if (Y_SIZE != Y_info.shape[1]) {
    throw std::runtime_error("Y must have " + std::to_string(Y_SIZE) +
                             " columns.");
  }

  /* substitute */
  FLOAT *X_data_ptr = static_cast<FLOAT *>(X_info.ptr);
  PythonNumpy::DenseMatrix_Type<FLOAT, LS_NUMBER_OF_DATA, X_SIZE> X;
  for (std::size_t i = 0; i < LS_NUMBER_OF_DATA; i++) {
    for (std::size_t j = 0; j < X_SIZE; j++) {
      X.access(i, j) = X_data_ptr[i * X_SIZE + j];
    }
  }

  FLOAT *Y_data_ptr = static_cast<FLOAT *>(Y_info.ptr);
  PythonNumpy::DenseMatrix_Type<FLOAT, LS_NUMBER_OF_DATA, Y_SIZE> Y;
  for (std::size_t i = 0; i < LS_NUMBER_OF_DATA; i++) {
    for (std::size_t j = 0; j < Y_SIZE; j++) {
      Y.access(i, j) = Y_data_ptr[i * Y_SIZE + j];
    }
  }

  ls.fit(X, Y);
}

py::array_t<FLOAT> get_weights(void) {
  auto weights = ls.get_weights();

  py::array_t<FLOAT> result;
  result.resize({static_cast<int>(WEIGHTS_SIZE), 1});

  for (std::size_t i = 0; i < WEIGHTS_SIZE; ++i) {
    result.mutable_at(i, 0) = weights.access(i, 0);
  }

  return result;
}

PYBIND11_MODULE(LeastSquaresSIL, m) {
  m.def("initialize", &initialize, "initialize least squares");
  m.def("fit", &fit, "fit least squares");
  m.def("get_weights", &get_weights, "get weights");
}
