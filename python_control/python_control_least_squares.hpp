#ifndef __PYTHON_CONTROL_LEAST_SQUARES_HPP__
#define __PYTHON_CONTROL_LEAST_SQUARES_HPP__

#include "python_numpy.hpp"

#include <array>
#include <type_traits>

namespace PythonControl {

constexpr double LEAST_SQUARES_DIVISION_MIN = 1.0e-10;

/* Least Squares Input, Output Type */
template <typename T, std::size_t Vector_Size>
using LeastSquaresInput_Type =
    PythonNumpy::DenseMatrix_Type<T, (Vector_Size + 1), 1>;

namespace MakeLeastSquaresInput {

template <std::size_t IndexCount, typename LeastSquaresInput_Type, typename T>
inline void assign_values(LeastSquaresInput_Type &input, T value_1) {

  static_assert(
      IndexCount < LeastSquaresInput_Type::COLS,
      "Number of arguments must be less than the number of input size.");

  input.template set<IndexCount, 0>(value_1);
  input.template set<(IndexCount + 1), 0>(static_cast<T>(1));
}

template <std::size_t IndexCount, typename LeastSquaresInput_Type, typename T,
          typename U, typename... Args>
inline void assign_values(LeastSquaresInput_Type &input, T value_1, U value_2,
                          Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");
  static_assert(
      IndexCount < LeastSquaresInput_Type::COLS,
      "Number of arguments must be less than the number of input size.");

  input.template set<IndexCount, 0>(value_1);

  assign_values<IndexCount + 1>(input, value_2, args...);
}

} // namespace MakeLeastSquaresInput

/* make Least Squares Input, Output  */
template <std::size_t Vector_Size, typename T, typename... Args>
inline auto make_LeastSquaresInput(T value_1, Args... args)
    -> LeastSquaresInput_Type<T, Vector_Size> {

  LeastSquaresInput_Type<T, Vector_Size> input;

  MakeLeastSquaresInput::assign_values<0>(input, value_1, args...);

  return input;
}

} // namespace PythonControl

#endif // __PYTHON_CONTROL_LEAST_SQUARES_HPP__
