#include "kalman_filter_fix_SIL_wrapper.hpp"
#include "python_control.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using FLOAT = typename kalman_filter_fix_SIL_wrapper::type::Value_Type;

constexpr std::size_t INPUT_SIZE = kalman_filter_fix_SIL_wrapper::INPUT_SIZE;
constexpr std::size_t STATE_SIZE = kalman_filter_fix_SIL_wrapper::STATE_SIZE;
constexpr std::size_t OUTPUT_SIZE = kalman_filter_fix_SIL_wrapper::OUTPUT_SIZE;

kalman_filter_fix_SIL_wrapper::type kf;

void initialize(void) { kf = kalman_filter_fix_SIL_wrapper::make(); }

void predict_and_update_with_fixed_G(py::array_t<FLOAT> U_in,
                                     py::array_t<FLOAT> Y_in) {

  py::buffer_info U_info = U_in.request();
  py::buffer_info Y_info = Y_in.request();

  /* check compatibility */
  if (INPUT_SIZE != U_info.shape[0]) {
    throw std::runtime_error("U must have " + std::to_string(INPUT_SIZE) +
                             " inputs.");
  }

  if (OUTPUT_SIZE != Y_info.shape[0]) {
    throw std::runtime_error("Y must have " + std::to_string(OUTPUT_SIZE) +
                             " outputs.");
  }

  /* substitute */
  FLOAT *U_data_ptr = static_cast<FLOAT *>(U_info.ptr);
  PythonControl::StateSpaceInput_Type<FLOAT, INPUT_SIZE> U;
  for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
    U.access(i, 0) = U_data_ptr[i];
  }

  FLOAT *Y_data_ptr = static_cast<FLOAT *>(Y_info.ptr);
  PythonControl::StateSpaceOutput_Type<FLOAT, OUTPUT_SIZE> Y;
  for (std::size_t i = 0; i < OUTPUT_SIZE; ++i) {
    Y.access(i, 0) = Y_data_ptr[i];
  }

  /* predict and update */
  kf.predict_and_update_with_fixed_G(U, Y);
}

void set_x_hat(py::array_t<FLOAT> x_hat_in) {
  py::buffer_info x_hat_info = x_hat_in.request();

  /* check compatibility */
  if (STATE_SIZE != x_hat_info.shape[0]) {
    throw std::runtime_error("x_hat must have " + std::to_string(STATE_SIZE) +
                             " states.");
  }

  /* substitute */
  FLOAT *x_hat_data_ptr = static_cast<FLOAT *>(x_hat_info.ptr);
  PythonControl::StateSpaceState_Type<FLOAT, STATE_SIZE> x_hat;
  for (std::size_t i = 0; i < STATE_SIZE; ++i) {
    x_hat.access(i, 0) = x_hat_data_ptr[i];
  }

  /* set */
  kf.set_x_hat(x_hat);
}

py::array_t<FLOAT> get_x_hat(void) {
  auto x_hat = kf.get_x_hat();

  py::array_t<FLOAT> result;
  result.resize({static_cast<int>(STATE_SIZE), 1});

  for (std::size_t i = 0; i < STATE_SIZE; ++i) {
    result.mutable_at(i, 0) = x_hat.access(i, 0);
  }

  return result;
}

PYBIND11_MODULE(KalmanFilterFixSIL, m) {
  m.def("initialize", &initialize, "initialize kalman filter");
  m.def("predict_and_update_with_fixed_G", &predict_and_update_with_fixed_G,
        "predict and update kalman filter with input and output");
  m.def("set_x_hat", &set_x_hat, "set x_hat");
  m.def("get_x_hat", &get_x_hat, "get x_hat");
}
