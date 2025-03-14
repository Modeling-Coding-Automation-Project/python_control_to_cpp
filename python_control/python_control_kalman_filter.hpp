#ifndef __PYTHON_CONTROL_KALMAN_FILTER_HPP__
#define __PYTHON_CONTROL_KALMAN_FILTER_HPP__

#include "python_control_state_space.hpp"
#include "python_numpy.hpp"

#include <functional>
#include <type_traits>

namespace PythonControl {

constexpr double KALMAN_FILTER_DIVISION_MIN = 1.0e-10;

template <typename DiscreteStateSpace_Type, typename Q_Type, typename R_Type>
class LinearKalmanFilter {
private:
  /* Type */
  using _T = typename DiscreteStateSpace_Type::Original_X_Type::Value_Type;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Matrix value data type must be float or double.");

  static constexpr std::size_t _STATE_SIZE =
      DiscreteStateSpace_Type::Original_X_Type::COLS;
  static constexpr std::size_t _INPUT_SIZE =
      DiscreteStateSpace_Type::Original_U_Type::COLS;
  static constexpr std::size_t _OUTPUT_SIZE =
      DiscreteStateSpace_Type::Original_Y_Type::COLS;

  using _C_P_CT_R_Inv_Type =
      PythonNumpy::LinalgSolverInv_Type<PythonNumpy::Matrix<
          PythonNumpy::DefDense, _T, _OUTPUT_SIZE, _OUTPUT_SIZE>>;

public:
  /* Type  */
  using P_Type = PythonNumpy::DenseMatrix_Type<_T, _STATE_SIZE, _STATE_SIZE>;

  using G_Type = PythonNumpy::DenseMatrix_Type<_T, _STATE_SIZE, _OUTPUT_SIZE>;

  /* Check Compatibility */
  static_assert(PythonNumpy::Is_Diag_Matrix<Q_Type>::value,
                "Q matrix must be diagonal matrix.");

  static_assert(PythonNumpy::Is_Diag_Matrix<R_Type>::value,
                "R matrix must be diagonal matrix.");

  /* Check Data Type */
  static_assert(
      std::is_same<typename Q_Type::Value_Type, _T>::value,
      "Data type of Q matrix must be same type as DiscreteStateSpace.");

  static_assert(
      std::is_same<typename R_Type::Value_Type, _T>::value,
      "Data type of R matrix must be same type as DiscreteStateSpace.");

public:
  /* Constructor */
  LinearKalmanFilter(){};

  LinearKalmanFilter(const DiscreteStateSpace_Type &DiscreteStateSpace,
                     const Q_Type &Q, const R_Type &R)
      : state_space(DiscreteStateSpace), Q(Q), R(R),
        P(PythonNumpy::make_DiagMatrixIdentity<_T, _STATE_SIZE>()
              .create_dense()),
        G(), _C_P_CT_R_inv_solver() {}

  /* Copy Constructor */
  LinearKalmanFilter(
      const LinearKalmanFilter<DiscreteStateSpace_Type, Q_Type, R_Type> &input)
      : state_space(input.state_space), Q(input.Q), R(input.R), P(input.P),
        G(input.G), _C_P_CT_R_inv_solver(input._C_P_CT_R_inv_solver) {}

  LinearKalmanFilter<DiscreteStateSpace_Type, Q_Type, R_Type> &
  operator=(const LinearKalmanFilter<DiscreteStateSpace_Type, Q_Type, R_Type>
                &input) {
    if (this != &input) {
      this->state_space = input.state_space;
      this->Q = input.Q;
      this->R = input.R;
      this->P = input.P;
      this->G = input.G;
      this->_C_P_CT_R_inv_solver = input._C_P_CT_R_inv_solver;
    }
    return *this;
  }

  /* Move Constructor */
  LinearKalmanFilter(LinearKalmanFilter<DiscreteStateSpace_Type, Q_Type, R_Type>
                         &&input) noexcept
      : state_space(std::move(input.state_space)), Q(std::move(input.Q)),
        R(std::move(input.R)), P(std::move(input.P)), G(std::move(input.G)),
        _C_P_CT_R_inv_solver(std::move(input._C_P_CT_R_inv_solver)) {}

  LinearKalmanFilter<DiscreteStateSpace_Type, Q_Type, R_Type> &
  operator=(LinearKalmanFilter<DiscreteStateSpace_Type, Q_Type, R_Type>
                &&input) noexcept {
    if (this != &input) {
      this->state_space = std::move(input.state_space);
      this->Q = std::move(input.Q);
      this->R = std::move(input.R);
      this->P = std::move(input.P);
      this->G = std::move(input.G);
      this->_C_P_CT_R_inv_solver = std::move(input._C_P_CT_R_inv_solver);
    }
    return *this;
  }

public:
  /* Function */
  inline void
  predict(const typename DiscreteStateSpace_Type::Original_U_Type &U) {

    this->state_space.X =
        this->state_space.A * this->state_space.X + this->state_space.B * U;
    this->P = this->state_space.A *
                  PythonNumpy::A_mul_BTranspose(this->P, this->state_space.A) +
              this->Q;
  }

  inline auto
  calc_y_dif(const typename DiscreteStateSpace_Type::Original_Y_Type &Y) ->
      typename DiscreteStateSpace_Type::Original_Y_Type {

    this->state_space.Y.push(this->state_space.C * this->state_space.X);

    typename DiscreteStateSpace_Type::Original_Y_Type Y_dif =
        Y - this->state_space.Y.get();

    // When there is no delay, you can use below.
    // typename DiscreteStateSpace_Type::Original_Y_Type Y_dif =
    // Y - this->C * this->state_space.X;

    return Y_dif;
  }

  inline void
  update(const typename DiscreteStateSpace_Type::Original_Y_Type &Y) {

    auto P_CT = PythonNumpy::A_mul_BTranspose(this->P, this->state_space.C);

    auto C_P_CT_R = this->state_space.C * P_CT + this->R;
    this->_C_P_CT_R_inv_solver.inv(C_P_CT_R);

    this->G = P_CT * this->_C_P_CT_R_inv_solver.get_answer();

    this->state_space.X = this->state_space.X + this->G * this->calc_y_dif(Y);

    this->P = (PythonNumpy::make_DiagMatrixIdentity<_T, _STATE_SIZE>() -
               this->G * this->state_space.C) *
              this->P;
  }

  inline void predict_and_update(
      const typename DiscreteStateSpace_Type::Original_U_Type &U,
      const typename DiscreteStateSpace_Type::Original_Y_Type &Y) {

    this->predict(U);
    this->update(Y);
  }

  // If G is known, you can use below "_fixed_G" functions.
  inline void predict_with_fixed_G(
      const typename DiscreteStateSpace_Type::Original_U_Type &U) {

    this->state_space.X =
        this->state_space.A * this->state_space.X + this->state_space.B * U;
  }

  inline void update_with_fixed_G(
      const typename DiscreteStateSpace_Type::Original_Y_Type &Y) {

    this->state_space.X = this->state_space.X + this->G * this->calc_y_dif(Y);
  }

  inline void predict_and_update_with_fixed_G(
      const typename DiscreteStateSpace_Type::Original_U_Type &U,
      const typename DiscreteStateSpace_Type::Original_Y_Type &Y) {

    this->predict_with_fixed_G(U);
    this->update_with_fixed_G(Y);
  }

  /* Get */
  inline auto get_x_hat(void) const ->
      typename DiscreteStateSpace_Type::Original_X_Type {
    return this->state_space.get_X();
  }

  /* Set */
  inline void
  set_x_hat(const typename DiscreteStateSpace_Type::Original_X_Type &x_hat) {
    this->state_space.X = x_hat;
  }

  inline void set_decay_rate_for_C_P_CT_R_inv_solver(const _T &decay_rate_in) {
    this->_C_P_CT_R_inv_solver.set_decay_rate(decay_rate_in);
  }

  inline void
  set_division_min_for_C_P_CT_R_inv_solver(const _T &division_min_in) {
    this->_C_P_CT_R_inv_solver.set_division_min(division_min_in);
  }

public:
  /* Variable */
  DiscreteStateSpace_Type state_space;
  Q_Type Q;
  R_Type R;
  P_Type P;
  G_Type G;

private:
  /* Variable */
  _C_P_CT_R_Inv_Type _C_P_CT_R_inv_solver;
};

/* make Linear Kalman Filter */
template <typename DiscreteStateSpace_Type, typename Q_Type, typename R_Type>
inline auto make_LinearKalmanFilter(DiscreteStateSpace_Type DiscreteStateSpace,
                                    Q_Type Q, R_Type R)
    -> LinearKalmanFilter<DiscreteStateSpace_Type, Q_Type, R_Type> {

  return LinearKalmanFilter<DiscreteStateSpace_Type, Q_Type, R_Type>(
      DiscreteStateSpace, Q, R);
}

/* Linear Kalman Filter Type */
template <typename DiscreteStateSpace_Type, typename Q_Type, typename R_Type>
using LinearKalmanFilter_Type =
    LinearKalmanFilter<DiscreteStateSpace_Type, Q_Type, R_Type>;

/* state and measurement function alias */
template <typename State_Type, typename Input_Type, typename Parameter_Type>
using StateFunction_Object = std::function<State_Type(
    const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename A_Type, typename State_Type, typename Input_Type,
          typename Parameter_Type>
using StateFunctionJacobian_Object = std::function<A_Type(
    const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename Output_Type, typename State_Type, typename Parameter_Type>
using MeasurementFunction_Object =
    std::function<Output_Type(const State_Type &, const Parameter_Type &)>;

template <typename C_Type, typename State_Type, typename Parameter_Type>
using MeasurementFunctionJacobian_Object =
    std::function<C_Type(const State_Type &, const Parameter_Type &)>;

/* Extended Kalman Filter */
template <typename A_Type, typename C_Type, typename U_Type, typename Q_Type,
          typename R_Type, typename Parameter_Type,
          std::size_t Number_Of_Delay = 0>
class ExtendedKalmanFilter {
private:
  /* Type */
  using _T = typename A_Type::Value_Type;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Matrix value data type must be float or double.");

  static constexpr std::size_t _STATE_SIZE = A_Type::COLS;
  static constexpr std::size_t _INPUT_SIZE = U_Type::COLS;
  static constexpr std::size_t _OUTPUT_SIZE = C_Type::COLS;

  using _State_Type = PythonControl::StateSpaceStateType<_T, _STATE_SIZE>;
  using _Measurement_Type =
      PythonControl::StateSpaceOutputType<_T, _OUTPUT_SIZE>;
  using _MeasurementStored_Type =
      PythonControl::DelayedVectorObject<_Measurement_Type, Number_Of_Delay>;

  using _StateFunction_Object =
      PythonControl::StateFunction_Object<_State_Type, U_Type, Parameter_Type>;
  using _StateFunctionJacobian_Object =
      PythonControl::StateFunctionJacobian_Object<A_Type, _State_Type, U_Type,
                                                  Parameter_Type>;
  using _MeasurementFunction_Object =
      PythonControl::MeasurementFunction_Object<_Measurement_Type, _State_Type,
                                                Parameter_Type>;
  using _MeasurementFunctionJacobian_Object =
      PythonControl::MeasurementFunctionJacobian_Object<C_Type, _State_Type,
                                                        Parameter_Type>;

  using _C_P_CT_R_Inv_Type =
      PythonNumpy::LinalgSolverInv_Type<PythonNumpy::Matrix<
          PythonNumpy::DefDense, _T, _OUTPUT_SIZE, _OUTPUT_SIZE>>;

public:
  /* Type  */
  using P_Type = PythonNumpy::DenseMatrix_Type<_T, _STATE_SIZE, _STATE_SIZE>;

  using G_Type = PythonNumpy::DenseMatrix_Type<_T, _STATE_SIZE, _OUTPUT_SIZE>;

  /* Check Compatibility */
  static_assert(PythonNumpy::Is_Diag_Matrix<Q_Type>::value,
                "Q matrix must be diagonal matrix.");

  static_assert(PythonNumpy::Is_Diag_Matrix<R_Type>::value,
                "R matrix must be diagonal matrix.");

  /* Check Data Type */
  static_assert(std::is_same<typename Q_Type::Value_Type, _T>::value,
                "Data type of Q matrix must be same type as A.");

  static_assert(std::is_same<typename R_Type::Value_Type, _T>::value,
                "Data type of R matrix must be same type as A.");

public:
  /* Constructor */
  ExtendedKalmanFilter(){};

  ExtendedKalmanFilter(
      const Q_Type &Q, const R_Type &R, _StateFunction_Object &state_function,
      _StateFunctionJacobian_Object &state_function_jacobian,
      _MeasurementFunction_Object &measurement_function,
      _MeasurementFunctionJacobian_Object &measurement_function_jacobian,
      const Parameter_Type &parameters)
      : A(), C(), Q(Q), R(R),
        P(PythonNumpy::make_DiagMatrixIdentity<_T, _STATE_SIZE>()
              .create_dense()),
        G(), X_hat(), Y_store(), parameters(parameters), _C_P_CT_R_inv_solver(),
        _state_function(state_function),
        _state_function_jacobian(state_function_jacobian),
        _measurement_function(measurement_function),
        _measurement_function_jacobian(measurement_function_jacobian) {}

  /* Copy Constructor */
  ExtendedKalmanFilter(
      const ExtendedKalmanFilter<A_Type, C_Type, U_Type, Q_Type, R_Type,
                                 Parameter_Type, Number_Of_Delay> &input)
      : A(input.A), C(input.C), Q(input.Q), R(input.R), P(input.P), G(input.G),
        X_hat(input.X_hat), Y_store(input.Y_store),
        parameters(input.parameters),
        _C_P_CT_R_inv_solver(input._C_P_CT_R_inv_solver),
        _state_function(input._state_function),
        _state_function_jacobian(input._state_function_jacobian),
        _measurement_function(input._measurement_function),
        _measurement_function_jacobian(input._measurement_function_jacobian) {}

  ExtendedKalmanFilter<A_Type, C_Type, U_Type, Q_Type, R_Type, Parameter_Type,
                       Number_Of_Delay> &
  operator=(
      const ExtendedKalmanFilter<A_Type, C_Type, U_Type, Q_Type, R_Type,
                                 Parameter_Type, Number_Of_Delay> &input) {
    if (this != &input) {
      this->A = input.A;
      this->C = input.C;
      this->Q = input.Q;
      this->R = input.R;
      this->P = input.P;
      this->G = input.G;
      this->X_hat = input.X_hat;
      this->Y_store = input.Y_store;
      this->parameters = input.parameters;
      this->_C_P_CT_R_inv_solver = input._C_P_CT_R_inv_solver;
      this->_state_function = input._state_function;
      this->_state_function_jacobian = input._state_function_jacobian;
      this->_measurement_function = input._measurement_function;
      this->_measurement_function_jacobian =
          input._measurement_function_jacobian;
    }
    return *this;
  }

  /* Move Constructor */
  ExtendedKalmanFilter(
      ExtendedKalmanFilter<A_Type, C_Type, U_Type, Q_Type, R_Type,
                           Parameter_Type, Number_Of_Delay> &&input)
      : A(std::move(input.A)), C(std::move(input.C)), Q(std::move(input.Q)),
        R(std::move(input.R)), P(std::move(input.P)), G(std::move(input.G)),
        X_hat(std::move(input.X_hat)), Y_store(std::move(input.Y_store)),
        parameters(std::move(input.parameters)),
        _C_P_CT_R_inv_solver(std::move(input._C_P_CT_R_inv_solver)),
        _state_function(input._state_function),
        _state_function_jacobian(input._state_function_jacobian),
        _measurement_function(input._measurement_function),
        _measurement_function_jacobian(input._measurement_function_jacobian) {}

  ExtendedKalmanFilter<A_Type, C_Type, U_Type, Q_Type, R_Type, Parameter_Type,
                       Number_Of_Delay> &
  operator=(ExtendedKalmanFilter<A_Type, C_Type, U_Type, Q_Type, R_Type,
                                 Parameter_Type, Number_Of_Delay> &&input) {
    if (this != &input) {
      this->A = std::move(input.A);
      this->C = std::move(input.C);
      this->Q = std::move(input.Q);
      this->R = std::move(input.R);
      this->P = std::move(input.P);
      this->G = std::move(input.G);
      this->X_hat = std::move(input.X_hat);
      this->Y_store = std::move(input.Y_store);
      this->parameters = std::move(input.parameters);
      this->_C_P_CT_R_inv_solver = std::move(input._C_P_CT_R_inv_solver);
      this->_state_function = std::move(input._state_function);
      this->_state_function_jacobian =
          std::move(input._state_function_jacobian);
      this->_measurement_function = std::move(input._measurement_function);
      this->_measurement_function_jacobian =
          std::move(input._measurement_function_jacobian);
    }
    return *this;
  }

public:
  /* Function */
  inline void predict(const U_Type &U) {

    this->A = this->_state_function_jacobian(this->X_hat, U, parameters);

    this->X_hat = this->_state_function(this->X_hat, U, parameters);
    this->P =
        this->A * PythonNumpy::A_mul_BTranspose(this->P, this->A) + this->Q;
  }

  inline auto calc_y_dif(const _Measurement_Type &Y) -> _Measurement_Type {

    this->Y_store.push(this->_measurement_function(this->X_hat, parameters));

    _Measurement_Type Y_dif = Y - this->Y_store.get();

    // When there is no delay, you can use below.
    // Y_dif = Y - this->_measurement_function(this->X_hat, parameters);

    return Y_dif;
  }

  inline void update(const _Measurement_Type &Y) {

    this->C = this->_measurement_function_jacobian(this->X_hat, parameters);

    auto P_CT = PythonNumpy::A_mul_BTranspose(this->P, this->C);

    auto C_P_CT_R = this->C * P_CT + this->R;
    this->_C_P_CT_R_inv_solver.inv(C_P_CT_R);

    this->G = P_CT * this->_C_P_CT_R_inv_solver.get_answer();

    this->X_hat = this->X_hat + this->G * this->calc_y_dif(Y);

    this->P = (PythonNumpy::make_DiagMatrixIdentity<_T, _STATE_SIZE>() -
               this->G * this->C) *
              this->P;
  }

  inline void predict_and_update(const U_Type &U, const _Measurement_Type &Y) {

    this->predict(U);
    this->update(Y);
  }

  /* Get */
  inline auto get_x_hat(void) const -> _State_Type { return this->X_hat; }

  /* Set */
  inline void set_x_hat(const _State_Type &x_hat) { this->X_hat = x_hat; }

  inline void set_decay_rate_for_C_P_CT_R_inv_solver(const _T &decay_rate_in) {
    this->_C_P_CT_R_inv_solver.set_decay_rate(decay_rate_in);
  }

  inline void
  set_division_min_for_C_P_CT_R_inv_solver(const _T &division_min_in) {
    this->_C_P_CT_R_inv_solver.set_division_min(division_min_in);
  }

public:
  /* Constant */
  static constexpr std::size_t NUMBER_OF_DELAY = Number_Of_Delay;

public:
  /* Variable */
  A_Type A;
  C_Type C;
  Q_Type Q;
  R_Type R;
  P_Type P;
  G_Type G;

  _State_Type X_hat;
  _MeasurementStored_Type Y_store;

  Parameter_Type parameters;

private:
  /* Variable */
  _C_P_CT_R_Inv_Type _C_P_CT_R_inv_solver;
  _StateFunction_Object _state_function;
  _StateFunctionJacobian_Object _state_function_jacobian;
  _MeasurementFunction_Object _measurement_function;
  _MeasurementFunctionJacobian_Object _measurement_function_jacobian;
};

/* Extended Kalman Filter Type */
template <typename A_Type, typename C_Type, typename U_Type, typename Q_Type,
          typename R_Type, typename Parameter_Type,
          std::size_t Number_Of_Delay = 0>
using ExtendedKalmanFilter_Type =
    ExtendedKalmanFilter<A_Type, C_Type, U_Type, Q_Type, R_Type, Parameter_Type,
                         Number_Of_Delay>;

} // namespace PythonControl

#endif // __PYTHON_CONTROL_KALMAN_FILTER_HPP__
