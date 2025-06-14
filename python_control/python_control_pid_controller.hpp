/**
 * @file python_control_pid_controller.hpp
 * @brief Discrete PID Controller implementation for C++.
 *
 * This header defines the PythonControl namespace, which provides a generic,
 * type-safe implementation of a discrete PID (Proportional-Integral-Derivative)
 * controller. The controller supports configurable gains, output saturation,
 * anti-windup via back-calculation, and a filtered derivative term. The code is
 * designed to be flexible and efficient, supporting both float and double
 * types.
 */
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

/**
 * @brief Clamps a value to a specified range.
 *
 * This function ensures that the input value does not exceed the specified
 * minimum and maximum bounds. If the value is below the minimum, it is set to
 * the minimum; if it exceeds the maximum, it is set to the maximum.
 *
 * @tparam T The type of the value, which must support comparison and
 * assignment.
 * @param value The input value to be clamped.
 * @param min The minimum allowable value.
 * @param max The maximum allowable value.
 * @return T The clamped value within the specified range.
 */
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

/**
 * @brief Discrete PID Controller class.
 *
 * This class implements a discrete PID controller with configurable gains for
 * proportional, integral, and derivative terms. It supports output saturation,
 * anti-windup via back-calculation, and a filtered derivative term.
 *
 * @tparam T The type of the controller's value (e.g., float or double).
 */
template <typename T> class DiscretePID_Controller {
protected:
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

  /**
   * @brief Updates the PID controller with the given error value.
   *
   * This function calculates the PID output based on the provided error value
   * and updates the internal state of the controller. It applies saturation to
   * the output to ensure it remains within specified limits.
   *
   * @param error The error value to be processed by the PID controller.
   * @return The saturated PID output.
   */
  _T update(const _T &error) {
    this->_PID_output = this->_calculate_P_term(error) +
                        this->_calculate_I_term(error) +
                        this->_calculate_D_term(error);

    this->_saturated_PID_output = PythonControl::saturation<_T>(
        this->_PID_output, this->minimum_output, this->maximum_output);

    return this->_saturated_PID_output;
  }

  /**
   * @brief Resets the internal state of the PID controller.
   *
   * This function clears the integration and differentiation stores,
   * effectively resetting the controller to its initial state.
   */
  void reset(void) {
    this->_integration_store = static_cast<_T>(0);
    this->_differentiation_store = static_cast<_T>(0);
  }

  /**
   * @brief Returns the current PID output.
   *
   * This function retrieves the last computed PID output value.
   *
   * @return The last computed PID output.
   */
  const T get_integration_store(void) const { return this->_integration_store; }

  /**
   * @brief Returns the current differentiation store.
   *
   * This function retrieves the last computed differentiation store value.
   *
   * @return The last computed differentiation store.
   */
  const T get_differentiation_store(void) const {
    return this->_differentiation_store;
  }

protected:
  /* Function */

  /**
   * @brief Back calculation for the integral term.
   *
   * This function computes the back calculation for the integral term based on
   * the error and the current PID output. It is used to prevent integral windup
   * by adjusting the integral term based on the difference between the
   * saturated and actual PID outputs.
   *
   * @param error The error value to be processed.
   * @return The computed back calculation for the integral term.
   */
  inline _T _back_calculation_for_I_term(const _T &error) {
    return this->Ki * error +
           this->Kb * (this->_saturated_PID_output - this->_PID_output);
  }

  /**
   * @brief Calculates the proportional term of the PID controller.
   *
   * This function computes the proportional term based on the error and the
   * proportional gain (Kp).
   *
   * @param error The error value to be processed.
   * @return The computed proportional term.
   */
  inline _T _calculate_P_term(const _T &error) { return this->Kp * error; }

  /**
   * @brief Calculates the integral term of the PID controller.
   *
   * This function computes the integral term based on the error and the
   * integral gain (Ki). It accumulates the integral of the error over time,
   * applying back calculation to prevent windup.
   *
   * @param error The error value to be processed.
   * @return The computed integral term.
   */
  inline _T _calculate_I_term(const _T &error) {

    this->_integration_store +=
        this->_back_calculation_for_I_term(error) * this->delta_time;
    return this->_integration_store;
  }

  /**
   * @brief Calculates the derivative term of the PID controller.
   *
   * This function computes the derivative term based on the error and the
   * derivative gain (Kd). It uses a filtered differentiation approach to reduce
   * noise sensitivity.
   *
   * @param error The error value to be processed.
   * @return The computed derivative term.
   */
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

protected:
  /* Variables */
  _T _integration_store;
  _T _differentiation_store;
  _T _PID_output;
  _T _saturated_PID_output;
};

/* make Discrete PID Controller */

/**
 * @brief Creates a DiscretePID_Controller object with default settings.
 *
 * This function template initializes a DiscretePID_Controller object with the
 * default delta time, gains, and output limits. It can be used for control
 * applications requiring a PID controller.
 *
 * @tparam T The type of the controller's value (e.g., float or double).
 * @return DiscretePID_Controller<T> The resulting DiscretePID_Controller
 * object.
 */
template <typename T>
inline auto make_DiscretePID_Controller(const T &delta_time, const T &Kp,
                                        const T &Ki, const T &Kd, const T &N,
                                        const T &Kb, const T &minimum_output,
                                        const T &maximum_output)
    -> DiscretePID_Controller<T> {
  return DiscretePID_Controller<T>(delta_time, Kp, Ki, Kd, N, Kb,
                                   minimum_output, maximum_output);
}

/* Discrete PID Controller Type */
template <typename T>
using DiscretePID_Controller_Type = DiscretePID_Controller<T>;

} // namespace PythonControl

#endif // __PYTHON_CONTROL_PID_CONTROLLER__
