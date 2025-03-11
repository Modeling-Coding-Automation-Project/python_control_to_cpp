#include <iostream>

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <array>

using namespace PythonNumpy;
using namespace PythonControl;

constexpr std::size_t EKF_SIM_STEP_MAX = 500;

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
auto bicycle_model_state_function(const StateSpaceStateType<T, STATE_SIZE> &X,
                                  const StateSpaceInputType<T, INPUT_SIZE> &U,
                                  const BicycleModelParameter<T> &parameters)
    -> StateSpaceStateType<T, STATE_SIZE>;

template <typename T, typename A_Type>
auto bicycle_model_state_function_jacobian(
    const StateSpaceStateType<T, STATE_SIZE> &X,
    const StateSpaceInputType<T, INPUT_SIZE> &U,
    const BicycleModelParameter<T> &parameters) -> A_Type;

template <typename T>
auto bicycle_model_measurement_function(
    const StateSpaceStateType<T, STATE_SIZE> &X,
    const BicycleModelParameter<T> &parameters)
    -> StateSpaceOutputType<T, OUTPUT_SIZE>;

template <typename T, typename C_Type>
auto bicycle_model_measurement_function_jacobian(
    const StateSpaceStateType<T, STATE_SIZE> &X,
    const BicycleModelParameter<T> &parameters) -> C_Type;

int main(void) {
  /* Create plant model */
  using X_Type = StateSpaceStateType<double, STATE_SIZE>;
  using U_Type = StateSpaceInputType<double, INPUT_SIZE>;
  using Y_Type = StateSpaceOutputType<double, OUTPUT_SIZE>;

  using SparseAvailable_A =
      SparseAvailable<ColumnAvailable<true, false, true>,
                      ColumnAvailable<false, true, true>,
                      ColumnAvailable<false, false, true>>;

  using A_Type = SparseMatrix_Type<double, SparseAvailable_A>;

  using SparseAvailable_C = SparseAvailable<
      ColumnAvailable<true, true, false>, ColumnAvailable<true, true, true>,
      ColumnAvailable<true, true, false>, ColumnAvailable<true, true, true>>;

  using C_Type = SparseMatrix_Type<double, SparseAvailable_C>;

  auto Q = make_DiagMatrix<STATE_SIZE>(1.0, 1.0, 1.0);

  using Q_Type = decltype(Q);

  auto R = make_DiagMatrix<OUTPUT_SIZE>(10.0, 10.0, 10.0, 10.0);

  using R_Type = decltype(R);

  /* Parameters */
  using Parameter_Type = BicycleModelParameter<double>;

  Parameter_Type parameters(0.1, 0.5, 0.0, 0.0, 10.0, 10.0);

  /* state and measurement functions */
  StateFunction_Object<X_Type, U_Type, BicycleModelParameter<double>>
      state_function = bicycle_model_state_function<double>;

  StateFunctionJacobian_Object<A_Type, X_Type, U_Type,
                               BicycleModelParameter<double>>
      state_function_jacobian =
          bicycle_model_state_function_jacobian<double, A_Type>;

  MeasurementFunction_Object<Y_Type, X_Type, BicycleModelParameter<double>>
      measurement_function = bicycle_model_measurement_function<double>;

  MeasurementFunctionJacobian_Object<C_Type, X_Type,
                                     BicycleModelParameter<double>>
      measurement_function_jacobian =
          bicycle_model_measurement_function_jacobian<double, C_Type>;

  /* define EKF */
  ExtendedKalmanFilter<A_Type, C_Type, U_Type, Q_Type, R_Type, Parameter_Type>
      ekf(Q, R, state_function, state_function_jacobian, measurement_function,
          measurement_function_jacobian, parameters);

  /* simulation */
  auto x_true_initial = make_StateSpaceState<STATE_SIZE>(2.0, 6.0, 0.3);
  decltype(x_true_initial) x_true;

  auto u = make_StateSpaceInput<INPUT_SIZE>(1.1, 0.1);

  ekf.X_hat.template set<0, 0>(0.0);
  ekf.X_hat.template set<1, 0>(0.0);
  ekf.X_hat.template set<2, 0>(0.0);

  x_true = x_true_initial;
  for (std::size_t i = 0; i < EKF_SIM_STEP_MAX; i++) {
    x_true = bicycle_model_state_function<double>(x_true, u, parameters);
    auto y = bicycle_model_measurement_function<double>(x_true, parameters);

    ekf.predict(u);
    ekf.update(y);

    for (std::size_t j = 0; j < STATE_SIZE; j++) {
      std::cout << "X_hat[" << j << "]: " << ekf.X_hat(j, 0) << ", ";
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
auto bicycle_model_state_function(const StateSpaceStateType<T, STATE_SIZE> &X,
                                  const StateSpaceInputType<T, INPUT_SIZE> &U,
                                  const BicycleModelParameter<T> &parameters)
    -> StateSpaceStateType<T, STATE_SIZE> {

  T x = X.template get<0, 0>();
  T y = X.template get<1, 0>();
  T theta = X.template get<2, 0>();
  T v = U.template get<0, 0>();
  T steering_angle = U.template get<1, 0>();

  T wheelbase = parameters.wheelbase;
  T delta_time = parameters.delta_time;

  return StateSpaceStateType<T, STATE_SIZE>(
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

template <typename T, typename A_Type>
auto bicycle_model_state_function_jacobian(
    const StateSpaceStateType<T, STATE_SIZE> &X,
    const StateSpaceInputType<T, INPUT_SIZE> &U,
    const BicycleModelParameter<T> &parameters) -> A_Type {

  T theta = X.template get<2, 0>();
  T v = U.template get<0, 0>();
  T steering_angle = U.template get<1, 0>();

  T wheelbase = parameters.wheelbase;
  T delta_time = parameters.delta_time;

  A_Type A;

  A.template set<0, 0>(1);
  A.template set<0, 1>(0);
  A.template set<0, 2>(
      -wheelbase * cos(theta) / tan(steering_angle) +
      wheelbase *
          cos(delta_time * v * tan(steering_angle) / wheelbase + theta) /
          tan(steering_angle));
  A.template set<1, 0>(0);
  A.template set<1, 1>(1);
  A.template set<1, 2>(
      -wheelbase * sin(theta) / tan(steering_angle) +
      wheelbase *
          sin(delta_time * v * tan(steering_angle) / wheelbase + theta) /
          tan(steering_angle));
  A.template set<2, 0>(0);
  A.template set<2, 1>(0);
  A.template set<2, 2>(1);

  return A;
}

template <typename T>
auto bicycle_model_measurement_function(
    const StateSpaceStateType<T, STATE_SIZE> &X,
    const BicycleModelParameter<T> &parameters)
    -> StateSpaceOutputType<T, OUTPUT_SIZE> {

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

  return StateSpaceOutputType<T, OUTPUT_SIZE>(
      {{sqrt(dif_1_x * dif_1_x + dif_1_y * dif_1_y)},
       {-theta + atan2(dif_1_y, dif_1_x)},
       {sqrt(dif_2_x * dif_2_x + dif_2_y * dif_2_y)},
       {-theta + atan2(dif_2_y, dif_2_x)}});
}

template <typename T, typename C_Type>
auto bicycle_model_measurement_function_jacobian(
    const StateSpaceStateType<T, STATE_SIZE> &X,
    const BicycleModelParameter<T> &parameters) -> C_Type {

  T x = X.template get<0, 0>();
  T y = X.template get<1, 0>();

  T landmark_1_x = parameters.landmark_1_x;
  T landmark_2_x = parameters.landmark_2_x;
  T landmark_2_y = parameters.landmark_2_y;
  T landmark_1_y = parameters.landmark_1_y;

  C_Type C;

  T dif_1_x = landmark_1_x - x;
  T dif_1_y = landmark_1_y - y;
  T dif_2_x = landmark_2_x - x;
  T dif_2_y = landmark_2_y - y;

  C.template set<0, 0>((-landmark_1_x + x) /
                       sqrt(dif_1_x * dif_1_x + dif_1_y * dif_1_y));
  C.template set<0, 1>((-landmark_1_y + y) /
                       sqrt(dif_1_x * dif_1_x + dif_1_y * dif_1_y));
  C.template set<0, 2>(0);
  C.template set<1, 0>(-(-landmark_1_y + y) /
                       (dif_1_x * dif_1_x + dif_1_y * dif_1_y));
  C.template set<1, 1>(-(landmark_1_x - x) /
                       (dif_1_x * dif_1_x + dif_1_y * dif_1_y));
  C.template set<1, 2>(-1);
  C.template set<2, 0>((-landmark_2_x + x) /
                       sqrt(dif_2_x * dif_2_x + dif_2_y * dif_2_y));
  C.template set<2, 1>((-landmark_2_y + y) /
                       sqrt(dif_2_x * dif_2_x + dif_2_y * dif_2_y));
  C.template set<2, 2>(0);
  C.template set<3, 0>(-(-landmark_2_y + y) /
                       (dif_2_x * dif_2_x + dif_2_y * dif_2_y));
  C.template set<3, 1>(-(landmark_2_x - x) /
                       (dif_2_x * dif_2_x + dif_2_y * dif_2_y));
  C.template set<3, 2>(-1);

  return C;
}
