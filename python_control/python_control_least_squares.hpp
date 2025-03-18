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
template <typename T, typename X_Type> class LeastSquares {
private:
  /* Type */
  using _T = T;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Value data type must be float or double.");

  using _Wights_Type = StateSpaceStateType<_T, X_Type::ROWS>;

  using _LstsqSolver_Type = PythonNumpy::LinalgLstsqSolver_Type<
      PythonNumpy::DenseMatrix_Type<_T, X_Type::COLS, _Wights_Type::COLS>,
      PythonNumpy::DenseMatrix_Type<_T, X_Type::COLS, 1>>;

public:
  /* Constant */
  static constexpr std::size_t NUMBER_OF_DATA = X_Type::COLS;

public:
  /* Constructor */
  LeastSquares() : _weights(), _lstsq_solver() {}

  /* Copy Constructor */
  LeastSquares(const LeastSquares<T, X_Type> &input)
      : _weights(input._weights), _lstsq_solver(input._lstsq_solver) {}

  LeastSquares<T, X_Type> &operator=(const LeastSquares<T, X_Type> &input) {
    if (this != &input) {
      this->_weights = input._weights;
      this->_lstsq_solver = input._lstsq_solver;
    }
    return *this;
  }

  /* Move Constructor */
  LeastSquares(LeastSquares<T, X_Type> &&input) noexcept
      : _weights(std::move(input._weights)),
        _lstsq_solver(std::move(input._lstsq_solver)) {}

  LeastSquares<T, X_Type> &operator=(LeastSquares<T, X_Type> &&input) noexcept {
    if (this != &input) {
      this->_weights = std::move(input._weights);
      this->_lstsq_solver = std::move(input._lstsq_solver);
    }
    return *this;
  }

public:
  /* Function */

private:
  /* Function */

private:
  /* Variables */
  _Wights_Type _weights;
  _LstsqSolver_Type _lstsq_solver;
};

} // namespace PythonControl

#endif // __PYTHON_CONTROL_LEAST_SQUARES_HPP__
