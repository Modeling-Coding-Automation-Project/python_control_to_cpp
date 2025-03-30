#ifndef __PYTHON_CONTROL_LEAST_SQUARES_HPP__
#define __PYTHON_CONTROL_LEAST_SQUARES_HPP__

#include "base_utility.hpp"
#include "python_numpy.hpp"

#include <array>
#include <type_traits>

namespace PythonControl {

constexpr double LEAST_SQUARES_DIVISION_MIN = 1.0e-10;

constexpr double LEAST_SQUARES_LAMBDA_FACTOR_DEFAULT = 0.9;
constexpr double LEAST_SQUARES_DELTA_DEFAULT = 0.1;

/* Least Squares Input, Output Type */
template <typename T, std::size_t Vector_Size>
using LeastSquaresInputType =
    PythonNumpy::DenseMatrix_Type<T, (Vector_Size + 1), 1>;

namespace MakeLeastSquaresInput {

template <std::size_t IndexCount, typename LeastSquaresInputType, typename T>
inline void assign_values(LeastSquaresInputType &input, T value_1) {

  static_assert(
      IndexCount < LeastSquaresInputType::COLS,
      "Number of arguments must be less than the number of input size.");

  input.template set<IndexCount, 0>(value_1);
  input.template set<(IndexCount + 1), 0>(static_cast<T>(1));
}

template <std::size_t IndexCount, typename LeastSquaresInputType, typename T,
          typename U, typename... Args>
inline void assign_values(LeastSquaresInputType &input, T value_1, U value_2,
                          Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");
  static_assert(
      IndexCount < LeastSquaresInputType::COLS,
      "Number of arguments must be less than the number of input size.");

  input.template set<IndexCount, 0>(value_1);

  assign_values<IndexCount + 1>(input, value_2, args...);
}

} // namespace MakeLeastSquaresInput

/* make Least Squares Input, Output  */
template <std::size_t Vector_Size, typename T, typename... Args>
inline auto make_LeastSquaresInput(T value_1, Args... args)
    -> LeastSquaresInputType<T, Vector_Size> {

  LeastSquaresInputType<T, Vector_Size> input;

  MakeLeastSquaresInput::assign_values<0>(input, value_1, args...);

  return input;
}

/* Least Squares Method */
template <typename X_Type> class LeastSquares {
private:
  /* Type */
  using _T = typename X_Type::Value_Type;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Value data type must be float or double.");

  using _Wights_Type = StateSpaceState_Type<_T, (X_Type::ROWS + 1)>;

  using _LstsqSolver_Type = PythonNumpy::LinalgLstsqSolver_Type<
      PythonNumpy::DenseMatrix_Type<_T, X_Type::COLS, _Wights_Type::COLS>,
      PythonNumpy::DenseMatrix_Type<_T, X_Type::COLS, 1>>;

public:
  /* Constant */
  static constexpr std::size_t NUMBER_OF_DATA = X_Type::COLS;

  using Y_Type = StateSpaceOutput_Type<_T, NUMBER_OF_DATA>;

public:
  /* Constructor */
  LeastSquares() : _weights(), _lstsq_solver() {}

  /* Copy Constructor */
  LeastSquares(const LeastSquares<X_Type> &input)
      : _weights(input._weights), _lstsq_solver(input._lstsq_solver) {}

  LeastSquares<X_Type> &operator=(const LeastSquares<X_Type> &input) {
    if (this != &input) {
      this->_weights = input._weights;
      this->_lstsq_solver = input._lstsq_solver;
    }
    return *this;
  }

  /* Move Constructor */
  LeastSquares(LeastSquares<X_Type> &&input) noexcept
      : _weights(std::move(input._weights)),
        _lstsq_solver(std::move(input._lstsq_solver)) {}

  LeastSquares<X_Type> &operator=(LeastSquares<X_Type> &&input) noexcept {
    if (this != &input) {
      this->_weights = std::move(input._weights);
      this->_lstsq_solver = std::move(input._lstsq_solver);
    }
    return *this;
  }

public:
  /* Function */
  inline void fit(const X_Type &X, const Y_Type &Y) {

    auto bias_vector = PythonNumpy::make_DenseMatrixOnes<_T, X_Type::COLS, 1>();

    auto X_ex = PythonNumpy::concatenate_horizontally(X, bias_vector);

    this->_weights = this->_lstsq_solver.solve(X_ex, Y);
  }

  inline auto predict(const X_Type &X) const -> Y_Type {

    auto bias_vector = PythonNumpy::make_DenseMatrixOnes<_T, X_Type::COLS, 1>();

    auto X_ex = PythonNumpy::concatenate_horizontally(X, bias_vector);

    return X_ex * this->_weights;
  }

  inline auto get_weights(void) const -> _Wights_Type { return this->_weights; }

  inline void set_lstsq_solver_decay_rate(const _T &decay_rate_in) {
    this->_lstsq_solver.set_decay_rate(decay_rate_in);
  }

  inline void set_lstsq_solver_division_min(const _T &division_min_in) {
    this->_lstsq_solver.set_division_min(division_min_in);
  }

private:
  /* Variables */
  _Wights_Type _weights;
  _LstsqSolver_Type _lstsq_solver;
};

/* make Least Squares */
template <typename X_Type>
inline auto make_LeastSquares(void) -> LeastSquares<X_Type> {
  return LeastSquares<X_Type>();
}

/* Least Squares type */
template <typename X_Type> using LeastSquares_Type = LeastSquares<X_Type>;

/* Recursive Least Squares Method */
template <typename X_Type> class RecursiveLeastSquares {
private:
  /* Type */
  using _T = typename X_Type::Value_Type;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Value data type must be float or double.");

  // plus 1 for bias
  using _Wights_Type = StateSpaceState_Type<_T, (X_Type::COLS + 1)>;

  using _P_Type =
      PythonNumpy::DenseMatrix_Type<_T, (X_Type::COLS + 1), (X_Type::COLS + 1)>;

public:
  /* Constant */
  static constexpr std::size_t RLS_SIZE = X_Type::COLS + 1;

public:
  /* Constructor */
  RecursiveLeastSquares()
      : _lambda_factor(static_cast<_T>(LEAST_SQUARES_LAMBDA_FACTOR_DEFAULT)),
        _lambda_factor_inv(
            static_cast<_T>(1) /
            static_cast<_T>(LEAST_SQUARES_LAMBDA_FACTOR_DEFAULT)),
        _weights(), _P(PythonNumpy::make_DiagMatrixFull<RLS_SIZE>(
                           static_cast<_T>(LEAST_SQUARES_DELTA_DEFAULT))
                           .create_dense()) {}

  RecursiveLeastSquares(const _T &lambda_in)
      : _lambda_factor(static_cast<_T>(lambda_in)),
        _lambda_factor_inv(static_cast<_T>(1) / static_cast<_T>(lambda_in)),
        _weights(), _P(PythonNumpy::make_DiagMatrixFull<RLS_SIZE>(
                           static_cast<_T>(LEAST_SQUARES_DELTA_DEFAULT))
                           .create_dense()) {}

  RecursiveLeastSquares(const _T &lambda_in, const _T &delta_in)
      : _lambda_factor(lambda_in),
        _lambda_factor_inv(static_cast<_T>(1) / lambda_in), _weights(),
        _P(PythonNumpy::make_DiagMatrixFull<RLS_SIZE>(delta_in)
               .create_dense()) {}

  /* Copy Constructor */
  RecursiveLeastSquares(const RecursiveLeastSquares<X_Type> &input)
      : _lambda_factor(input._lambda_factor),
        _lambda_factor_inv(input._lambda_factor_inv), _weights(input._weights),
        _P(input._P) {}

  RecursiveLeastSquares<X_Type> &
  operator=(const RecursiveLeastSquares<X_Type> &input) {
    if (this != &input) {
      this->_lambda_factor = input._lambda_factor;
      this->_lambda_factor_inv = input._lambda_factor_inv;
      this->_weights = input._weights;
      this->_P = input._P;
    }
    return *this;
  }

  /* Move Constructor */
  RecursiveLeastSquares(RecursiveLeastSquares<X_Type> &&input) noexcept
      : _lambda_factor(std::move(input._lambda_factor)),
        _lambda_factor_inv(std::move(input._lambda_factor_inv)),
        _weights(std::move(input._weights)), _P(std::move(input._P)) {}

  RecursiveLeastSquares<X_Type> &
  operator=(RecursiveLeastSquares<X_Type> &&input) noexcept {
    if (this != &input) {
      this->_lambda_factor = std::move(input._lambda_factor);
      this->_lambda_factor_inv = std::move(input._lambda_factor_inv);
      this->_weights = std::move(input._weights);
      this->_P = std::move(input._P);
    }
    return *this;
  }

public:
  /* Function */
  inline void set_lambda(const _T &lambda_in) {
    this->_lambda_factor = lambda_in;
    this->_lambda_factor_inv = static_cast<_T>(1) / lambda_in;
  }

  inline void update(const X_Type &X, const _T &y_true) {

    auto bias_vector = PythonNumpy::make_DenseMatrixOnes<_T, 1, 1>();

    auto X_ex = PythonNumpy::concatenate_vertically(X, bias_vector);

    auto y = PythonNumpy::ATranspose_mul_B(X_ex, this->_weights);

    auto y_dif = y_true - y.template get<0, 0>();

    auto P_x = this->_P * X_ex;

    // lambda_X_P is scalar
    _T lambda_X_P =
        this->_lambda_factor +
        PythonNumpy::ATranspose_mul_B(X_ex, P_x).template get<0, 0>();

    auto lambda_X_P_inv =
        static_cast<_T>(1) /
        Base::Utility::avoid_zero_divide(
            lambda_X_P, static_cast<_T>(LEAST_SQUARES_DIVISION_MIN));

    auto K = P_x * lambda_X_P_inv;

    this->_weights = this->_weights + K * y_dif;

    this->_P = (this->_P - K * PythonNumpy::ATranspose_mul_B(X_ex, this->_P)) *
               this->_lambda_factor_inv;
  }

  inline auto predict(const X_Type &X) const -> _T {

    auto bias_vector = PythonNumpy::make_DenseMatrixOnes<_T, 1, 1>();

    auto X_ex = PythonNumpy::concatenate_vertically(X, bias_vector);

    return PythonNumpy::ATranspose_mul_B(X_ex, this->_weights)
        .template get<0, 0>();
  }

  inline auto get_weights(void) const -> _Wights_Type { return this->_weights; }

  inline void set_inv_solver_decay_rate(const _T &decay_rate_in) {
    this->_lambda_X_P_Solver.set_decay_rate(decay_rate_in);
  }

  inline void set_inv_solver_division_min(const _T &division_min_in) {
    this->_lambda_X_P_Solver.set_division_min(division_min_in);
  }

private:
  /* Variables */
  _T _lambda_factor;
  _T _lambda_factor_inv;
  _Wights_Type _weights;
  _P_Type _P;
};

/* make Recursive Least Squares */
template <typename X_Type>
inline auto make_RecursiveLeastSquares(void) -> RecursiveLeastSquares<X_Type> {
  return RecursiveLeastSquares<X_Type>();
}

template <typename X_Type, typename T>
inline auto make_RecursiveLeastSquares(const T &lambda_in)
    -> RecursiveLeastSquares<X_Type> {
  return RecursiveLeastSquares<X_Type>(lambda_in);
}

template <typename X_Type, typename T>
inline auto make_RecursiveLeastSquares(const T &lambda_in, const T &delta_in)
    -> RecursiveLeastSquares<X_Type> {
  return RecursiveLeastSquares<X_Type>(lambda_in, delta_in);
}

/* Least Recursive Squares type */
template <typename X_Type>
using RecursiveLeastSquares_Type = RecursiveLeastSquares<X_Type>;

} // namespace PythonControl

#endif // __PYTHON_CONTROL_LEAST_SQUARES_HPP__
