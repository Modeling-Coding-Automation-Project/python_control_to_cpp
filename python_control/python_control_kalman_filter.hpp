#ifndef __PYTHON_CONTROL_KALMAN_FILTER_HPP__
#define __PYTHON_CONTROL_KALMAN_FILTER_HPP__

#include "python_control_state_space.hpp"
#include "python_numpy.hpp"

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

  using _C_P_CT_R_Inv_Type = PythonNumpy::LinalgSolverInv_Type<
      PythonNumpy::Matrix<PythonNumpy::DefDense, _T, _STATE_SIZE, _STATE_SIZE>>;

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
      : state_space(DiscreteStateSpace), Q(Q), R(R), P(), G(),
        _C_P_CT_R_inv_solver() {}

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
  LinearKalmanFilter(
      LinearKalmanFilter<DiscreteStateSpace_Type, Q_Type, R_Type> &&input)
      : state_space(std::move(input.state_space)), Q(std::move(input.Q)),
        R(std::move(input.R)), P(std::move(input.P)), G(std::move(input.G)),
        _C_P_CT_R_inv_solver(std::move(input._C_P_CT_R_inv_solver)) {}

  LinearKalmanFilter<DiscreteStateSpace_Type, Q_Type, R_Type> &operator=(
      LinearKalmanFilter<DiscreteStateSpace_Type, Q_Type, R_Type> &&input) {
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
    this->state_space.P =
        this->state_space.A * PythonNumpy::A_mul_BTranspose(
                                  this->state_space.P * this->state_space.A) +
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
    // Y - this->state_space.C * this->state_space.X;

    return Y_dif;
  }

  inline void
  update(const typename DiscreteStateSpace_Type::Original_Y_Type &Y) {

    auto P_CT = PythonNumpy::A_mul_BTranspose(this->P, this->state_space.C);

    auto C_P_CT_R = this->state_space.C * P_CT + this->R;
    this->_C_P_CT_R_inv_solver.inv(C_P_CT_R);

    this->G = P_CT * this->_C_P_CT_R_inv_solver.get_answer();

    this->state_space.X = this->state_space.X + this->G * this->calc_y_dif(Y);

    this->state_space.P =
        (PythonNumpy::make_DiagMatrixIdentity<_T, _STATE_SIZE>() -
         this->G * this->state_space.C) *
        this->state_space.P;
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

} // namespace PythonControl

#endif // __PYTHON_CONTROL_KALMAN_FILTER_HPP__
