/**
 * @file unscented_kalman_filter.cpp
 * @brief Demonstrates the use of an Unscented Kalman Filter (UKF) for state
 * estimation of a nonlinear bicycle model.
 *
 * This program sets up a plant model using a nonlinear bicycle model,
 * initializes process and measurement noise covariances, and defines the state
 * and measurement functions. It then constructs an Unscented Kalman Filter
 * (UKF) object and simulates the system over a number of time steps, applying
 * control inputs and updating the filter with delayed measurements. The
 * estimated states and true states are printed for comparison.
 */
#include <iostream>

#include "python_control.hpp"
#include "python_math.hpp"
#include "python_numpy.hpp"

#include <array>

using namespace PythonNumpy;
using namespace PythonControl;

constexpr std::size_t EKF_SIM_STEP_MAX = 200;

constexpr std::size_t NUMBER_OF_DELAY = 5;

constexpr std::size_t STATE_SIZE = 3;
constexpr std::size_t INPUT_SIZE = 2;
constexpr std::size_t OUTPUT_SIZE = 4;

/* model functions */
template <typename T> class BicycleModelParameter {
public:
  BicycleModelParameter(){};

  BicycleModelParameter(const T &delta_time, const T &wheelbase,
                        const T &landmark_1_x, const T &landmark_1_y,
                        const T &landmark_2_x, const T &landmark_2_y)
      : delta_time(delta_time), wheelbase(wheelbase),
        landmark_1_x(landmark_1_x), landmark_1_y(landmark_1_y),
        landmark_2_x(landmark_2_x), landmark_2_y(landmark_2_y) {}

public:
  T delta_time;
  T wheelbase;
  T landmark_1_x;
  T landmark_1_y;
  T landmark_2_x;
  T landmark_2_y;
};

template <typename T>
auto bicycle_model_state_function(const StateSpaceState_Type<T, STATE_SIZE> &X,
                                  const StateSpaceInput_Type<T, INPUT_SIZE> &U,
                                  const BicycleModelParameter<T> &parameters)
    -> StateSpaceState_Type<T, STATE_SIZE>;

template <typename T>
auto bicycle_model_measurement_function(
    const StateSpaceState_Type<T, STATE_SIZE> &X,
    const BicycleModelParameter<T> &parameters)
    -> StateSpaceOutput_Type<T, OUTPUT_SIZE>;

int main(void) {
  /* Create plant model */
  using X_Type = StateSpaceState_Type<double, STATE_SIZE>;
  using U_Type = StateSpaceInput_Type<double, INPUT_SIZE>;
  using Y_Type = StateSpaceOutput_Type<double, OUTPUT_SIZE>;

  auto Q = make_KalmanFilter_Q<STATE_SIZE>(0.01, 0.01, 0.01);

  using Q_Type = decltype(Q);

  auto R = make_KalmanFilter_R<OUTPUT_SIZE>(1.0, 1.0, 1.0, 1.0);

  using R_Type = decltype(R);

  /* Parameters */
  using Parameter_Type = BicycleModelParameter<double>;

  Parameter_Type parameters(0.1, 0.5, -1.0, -1.0, 10.0, 10.0);

  /* state and measurement functions */
  StateFunction_Object<X_Type, U_Type, BicycleModelParameter<double>>
      state_function = bicycle_model_state_function<double>;

  MeasurementFunction_Object<Y_Type, X_Type, BicycleModelParameter<double>>
      measurement_function = bicycle_model_measurement_function<double>;

  /* define EKF */
  UnscentedKalmanFilter<U_Type, Q_Type, R_Type, Parameter_Type, NUMBER_OF_DELAY>
      ukf(Q, R, state_function, measurement_function, parameters);

  /* simulation */
  auto x_true_initial = make_StateSpaceState<STATE_SIZE>(2.0, 6.0, 0.3);
  decltype(x_true_initial) x_true;

  auto u = make_StateSpaceInput<INPUT_SIZE>(2.0, 0.1);

  ukf.X_hat.template set<0, 0>(0.0);
  ukf.X_hat.template set<1, 0>(0.0);
  ukf.X_hat.template set<2, 0>(0.0);

  std::array<StateSpaceOutput_Type<double, OUTPUT_SIZE>, (NUMBER_OF_DELAY + 1)>
      y_store;

  std::size_t delay_index = 0;

  x_true = x_true_initial;
  for (std::size_t i = 0; i < EKF_SIM_STEP_MAX; i++) {
    x_true = bicycle_model_state_function<double>(x_true, u, parameters);
    y_store[delay_index] =
        bicycle_model_measurement_function<double>(x_true, parameters);

    // system delay
    delay_index++;
    if (delay_index > NUMBER_OF_DELAY) {
      delay_index = 0;
    }

    ukf.predict(u);
    ukf.update(y_store[delay_index]);

    for (std::size_t j = 0; j < STATE_SIZE; j++) {
      std::cout << "X_hat[" << j << "]: " << ukf.get_x_hat_without_delay()(j, 0)
                << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  for (std::size_t j = 0; j < STATE_SIZE; j++) {
    std::cout << "x_true[" << j << "]: " << x_true(j, 0) << ", ";
  }
  std::cout << std::endl;

  return 0;
}

template <typename T>
auto bicycle_model_state_function(const StateSpaceState_Type<T, STATE_SIZE> &X,
                                  const StateSpaceInput_Type<T, INPUT_SIZE> &U,
                                  const BicycleModelParameter<T> &parameters)
    -> StateSpaceState_Type<T, STATE_SIZE> {

  using namespace PythonMath;

  T x = X.template get<0, 0>();
  T y = X.template get<1, 0>();
  T theta = X.template get<2, 0>();
  T v = U.template get<0, 0>();
  T steering_angle = U.template get<1, 0>();

  T wheelbase = parameters.wheelbase;
  T delta_time = parameters.delta_time;

  return StateSpaceState_Type<T, STATE_SIZE>(
      {{-wheelbase * sin(theta) / tan(steering_angle) +
        wheelbase *
            sin(delta_time * v * tan(steering_angle) / wheelbase + theta) /
            tan(steering_angle) +
        x},
       {wheelbase * cos(theta) / tan(steering_angle) -
        wheelbase *
            cos(delta_time * v * tan(steering_angle) / wheelbase + theta) /
            tan(steering_angle) +
        y},
       {delta_time * v * tan(steering_angle) / wheelbase + theta}});
}

template <typename T>
auto bicycle_model_measurement_function(
    const StateSpaceState_Type<T, STATE_SIZE> &X,
    const BicycleModelParameter<T> &parameters)
    -> StateSpaceOutput_Type<T, OUTPUT_SIZE> {

  using namespace PythonMath;

  T x = X.template get<0, 0>();
  T y = X.template get<1, 0>();
  T theta = X.template get<2, 0>();

  T landmark_1_x = parameters.landmark_1_x;
  T landmark_2_x = parameters.landmark_2_x;
  T landmark_2_y = parameters.landmark_2_y;
  T landmark_1_y = parameters.landmark_1_y;

  T dif_1_x = landmark_1_x - x;
  T dif_1_y = landmark_1_y - y;
  T dif_2_x = landmark_2_x - x;
  T dif_2_y = landmark_2_y - y;

  return StateSpaceOutput_Type<T, OUTPUT_SIZE>(
      {{sqrt(dif_1_x * dif_1_x + dif_1_y * dif_1_y)},
       {-theta + atan2(dif_1_y, dif_1_x)},
       {sqrt(dif_2_x * dif_2_x + dif_2_y * dif_2_y)},
       {-theta + atan2(dif_2_y, dif_2_x)}});
}
