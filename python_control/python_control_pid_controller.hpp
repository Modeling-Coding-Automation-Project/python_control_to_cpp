#ifndef __PYTHON_CONTROL_PID_CONTROLLER_HPP__
#define __PYTHON_CONTROL_PID_CONTROLLER_HPP__

#include "python_numpy.hpp"

#include <limits>
#include <type_traits>

namespace PythonControl {

constexpr double PID_CONTROLLER_DIVISION_MIN = 1.0e-10;
constexpr double PID_CONTROLLER_MINIMUM_OUTPUT_DEFAULT =
    -std::numeric_limits<double>::infinity();
constexpr double PID_CONTROLLER_MAXIMUM_OUTPUT_DEFAULT =
    std::numeric_limits<double>::infinity();

/* Saturation */
template <typename T>
inline T saturation(const T &value, const T &min, const T &max) {
  T output = value;

  if (output < min) {
    output = min;
  } else if (output > max) {
    output = max;
  }

  return output;
}

/* PID Controller */
template <typename T> class DiscretePID_Controller {
private:
  /* Type */
  using _T = T;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Value data type must be float or double.");

public:
  /* Type */
  using Value_Type = _T;

public:
  /* Constructor */
  DiscretePID_Controller()
      : delta_time(static_cast<_T>(0)), Kp(static_cast<_T>(0)),
        Ki(static_cast<_T>(0)), Kd(static_cast<_T>(0)), N(static_cast<_T>(100)),
        Kb(static_cast<_T>(0)),
        minimum_output(static_cast<_T>(PID_CONTROLLER_MINIMUM_OUTPUT_DEFAULT)),
        maximum_output(static_cast<_T>(PID_CONTROLLER_MAXIMUM_OUTPUT_DEFAULT)),
        _integration_store(static_cast<_T>(0)),
        _differentiation_store(static_cast<_T>(0)),
        _PID_output(static_cast<_T>(0)),
        _saturated_PID_output(static_cast<_T>(0)) {}

  DiscretePID_Controller(const _T &delta_time, const _T &Kp, const _T &Ki,
                         const _T &Kd, const _T &N, const _T &Kb,
                         const _T &minimum_output, const _T &maximum_output)
      : delta_time(delta_time), Kp(Kp), Ki(Ki), Kd(Kd), N(N), Kb(Kb),
        minimum_output(minimum_output), maximum_output(maximum_output),
        _integration_store(static_cast<_T>(0)),
        _differentiation_store(static_cast<_T>(0)),
        _PID_output(static_cast<_T>(0)),
        _saturated_PID_output(static_cast<_T>(0)) {}

  /* Copy Constructor */
  DiscretePID_Controller(const DiscretePID_Controller<T> &input)
      : delta_time(input.delta_time), Kp(input.Kp), Ki(input.Ki), Kd(input.Kd),
        N(input.N), Kb(input.Kb), minimum_output(input.minimum_output),
        maximum_output(input.maximum_output),
        _integration_store(input._integration_store),
        _differentiation_store(input._differentiation_store),
        _PID_output(input._PID_output),
        _saturated_PID_output(input._saturated_PID_output) {}

  DiscretePID_Controller<T> &operator=(const DiscretePID_Controller<T> &input) {
    if (this != &input) {
      this->delta_time = input.delta_time;
      this->Kp = input.Kp;
      this->Ki = input.Ki;
      this->Kd = input.Kd;
      this->N = input.N;
      this->Kb = input.Kb;
      this->minimum_output = input.minimum_output;
      this->maximum_output = input.maximum_output;
      this->_integration_store = input._integration_store;
      this->_differentiation_store = input._differentiation_store;
      this->_PID_output = input._PID_output;
      this->_saturated_PID_output = input._saturated_PID_output;
    }
    return *this;
  }

  /* Move Constructor */
  DiscretePID_Controller(DiscretePID_Controller<T> &&input) noexcept
      : delta_time(std::move(input.delta_time)), Kp(std::move(input.Kp)),
        Ki(std::move(input.Ki)), Kd(std::move(input.Kd)), N(std::move(input.N)),
        Kb(std::move(input.Kb)),
        minimum_output(std::move(input.minimum_output)),
        maximum_output(std::move(input.maximum_output)),
        _integration_store(std::move(input._integration_store)),
        _differentiation_store(std::move(input._differentiation_store)),
        _PID_output(std::move(input._PID_output)),
        _saturated_PID_output(std::move(input._saturated_PID_output)) {}

  DiscretePID_Controller<T> &
  operator=(DiscretePID_Controller<T> &&input) noexcept {
    if (this != &input) {
      this->delta_time = std::move(input.delta_time);
      this->Kp = std::move(input.Kp);
      this->Ki = std::move(input.Ki);
      this->Kd = std::move(input.Kd);
      this->N = std::move(input.N);
      this->Kb = std::move(input.Kb);
      this->minimum_output = std::move(input.minimum_output);
      this->maximum_output = std::move(input.maximum_output);
      this->_integration_store = std::move(input._integration_store);
      this->_differentiation_store = std::move(input._differentiation_store);
      this->_PID_output = std::move(input._PID_output);
      this->_saturated_PID_output = std::move(input._saturated_PID_output);
    }
    return *this;
  }

public:
  /* Function */
  _T update(const _T &error) {
    this->_PID_output = this->_calculate_P_term(error) +
                        this->_calculate_I_term(error) +
                        this->_calculate_D_term(error);

    this->_saturated_PID_output = PythonControl::saturation<_T>(
        this->_PID_output, this->minimum_output, this->maximum_output);

    return this->_saturated_PID_output;
  }

  void reset(void) {
    this->_integration_store = static_cast<_T>(0);
    this->_differentiation_store = static_cast<_T>(0);
  }

  const T get_integration_store(void) const { return this->_integration_store; }

  const T get_differentiation_store(void) const {
    return this->_differentiation_store;
  }

private:
  /* Function */
  inline _T _back_calculation_for_I_term(const _T &error) {
    return this->Ki * error +
           this->Kb * (this->_saturated_PID_output - this->_PID_output);
  }

  inline _T _calculate_P_term(const _T &error) { return this->Kp * error; }

  inline _T _calculate_I_term(const _T &error) {

    this->_integration_store +=
        this->_back_calculation_for_I_term(error) * this->delta_time;
    return this->_integration_store;
  }

  inline _T _calculate_D_term(const _T &error) {
    /* incomplete differentiator */
    _T output = this->N * ((this->Kd * error) - this->_differentiation_store);

    this->_differentiation_store += output * this->delta_time;

    return output;
  }

public:
  /* Variables */
  _T delta_time;
  _T Kp;
  _T Ki;
  _T Kd;
  _T N;  // Filter coefficient
  _T Kb; // Back calculation coefficient

  _T minimum_output;
  _T maximum_output;

private:
  /* Variables */
  _T _integration_store;
  _T _differentiation_store;
  _T _PID_output;
  _T _saturated_PID_output;
};

/* make Discrete PID Controller */
template <typename T>
inline auto make_DiscretePID_Controller(
    const T &delta_time, const T &Kp, const T &Ki, const T &Kd, const T &N,
    const T &Kb, const T &minimum_output,
    const T &maximum_output) -> DiscretePID_Controller<T> {
  return DiscretePID_Controller<T>(delta_time, Kp, Ki, Kd, N, Kb,
                                   minimum_output, maximum_output);
}

/* Discrete PID Controller Type */
template <typename T>
using DiscretePID_Controller_Type = DiscretePID_Controller<T>;

} // namespace PythonControl

#endif // __PYTHON_CONTROL_PID_CONTROLLER__
