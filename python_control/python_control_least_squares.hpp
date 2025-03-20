#ifndef __PYTHON_CONTROL_LEAST_SQUARES_HPP__
#define __PYTHON_CONTROL_LEAST_SQUARES_HPP__

#include "python_numpy.hpp"

#include <array>
#include <type_traits>

namespace PythonControl {

constexpr double LEAST_SQUARES_DIVISION_MIN = 1.0e-10;

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

  using _Wights_Type = StateSpaceStateType<_T, (X_Type::ROWS + 1)>;

  using _LstsqSolver_Type = PythonNumpy::LinalgLstsqSolver_Type<
      PythonNumpy::DenseMatrix_Type<_T, X_Type::COLS, _Wights_Type::COLS>,
      PythonNumpy::DenseMatrix_Type<_T, X_Type::COLS, 1>>;

public:
  /* Constant */
  static constexpr std::size_t NUMBER_OF_DATA = X_Type::COLS;

  using Y_Type = StateSpaceOutputType<_T, NUMBER_OF_DATA>;

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

    this->_weights = this->_lstsq_solver(X_ex, Y);
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

} // namespace PythonControl

#endif // __PYTHON_CONTROL_LEAST_SQUARES_HPP__
