/**
 * @file python_control_transfer_function.hpp
 * @brief Discrete Transfer Function implementation for PythonControl C++
 * library.
 *
 * This header provides a template-based implementation of discrete transfer
 * functions, including utilities for constructing numerator and denominator
 * matrices, and converting transfer functions to state-space representations.
 * It supports single-input single-output (SISO) systems and provides methods
 * for updating, resetting, and solving steady-state responses of discrete
 * transfer functions.
 */
#ifndef __PYTHON_CONTROL_TRANSFER_FUNCTION_HPP__
#define __PYTHON_CONTROL_TRANSFER_FUNCTION_HPP__

#include "base_utility.hpp"
#include "python_control_state_space.hpp"
#include "python_numpy.hpp"

#include <type_traits>
#include <utility>

namespace PythonControl {

constexpr double TRANSFER_FUNCTION_DIVISION_MIN = 1.0e-10;

/* Numerator and Denominator definition */

/**
 * @brief Alias template for representing the numerator of a transfer function
 * as a sparse matrix.
 *
 * This type alias defines a transfer function numerator using a sparse matrix
 * type from the PythonNumpy module. The matrix is parameterized by the element
 * type `T` and the number of numerator coefficients `Numerator_Size`. The
 * matrix is constructed with dimensions determined by `Numerator_Size` rows and
 * 1 column, using the `DenseAvailable` trait to select the appropriate storage
 * format.
 *
 * @tparam T The data type of the matrix elements (e.g., double, float).
 * @tparam Numerator_Size The number of coefficients in the numerator (number of
 * rows).
 */
template <typename T, std::size_t Numerator_Size>
using TransferFunctionNumerator_Type = PythonNumpy::SparseMatrix_Type<
    T, PythonNumpy::DenseAvailable<Numerator_Size, 1>>;

/**
 * @brief Alias template for representing the denominator of a transfer function
 * as a sparse matrix.
 *
 * This type alias defines a transfer function denominator using a sparse matrix
 * type from the PythonNumpy module. The matrix is parameterized by the element
 * type `T` and the number of denominator coefficients `Denominator_Size`. The
 * matrix is constructed with dimensions determined by `Denominator_Size` rows
 * and 1 column, using the `DenseAvailable` trait to select the appropriate
 * storage format.
 *
 * @tparam T The data type of the matrix elements (e.g., double, float).
 * @tparam Denominator_Size The number of coefficients in the denominator
 * (number of rows).
 */
template <typename T, std::size_t Denominator_Size>
using TransferFunctionDenominator_Type = PythonNumpy::SparseMatrix_Type<
    T, PythonNumpy::DenseAvailable<Denominator_Size, 1>>;

namespace MakeNumerator {

/**
 * @brief Assigns values to the numerator matrix of a transfer function.
 *
 * This function recursively assigns values to the numerator matrix, ensuring
 * that the number of arguments does not exceed the number of columns in the
 * numerator matrix. It uses static assertions to enforce type consistency and
 * size constraints.
 *
 * @tparam IndexCount The current index in the numerator matrix being assigned.
 * @tparam TransferFunctionNumerator_Type The type of the numerator matrix.
 * @tparam T The type of the first value to assign.
 * @param numerator The numerator matrix to which values are assigned.
 * @param value_1 The first value to assign to the numerator matrix.
 */
template <std::size_t IndexCount, typename TransferFunctionNumerator_Type,
          typename T>
inline void assign_values(TransferFunctionNumerator_Type &numerator,
                          T value_1) {

  static_assert(
      IndexCount < TransferFunctionNumerator_Type::COLS,
      "Number of arguments must be less than the number of Numerator factor.");

  numerator.template set<IndexCount, 0>(value_1);
}

/**
 * @brief Assigns multiple values to the numerator matrix of a transfer
 * function.
 *
 * This function recursively assigns multiple values to the numerator matrix,
 * ensuring that all arguments are of the same type and that the number of
 * arguments does not exceed the number of columns in the numerator matrix. It
 * uses static assertions to enforce type consistency and size constraints.
 *
 * @tparam IndexCount The current index in the numerator matrix being assigned.
 * @tparam TransferFunctionNumerator_Type The type of the numerator matrix.
 * @tparam T The type of the first value to assign.
 * @tparam U The type of the second value to assign.
 * @param numerator The numerator matrix to which values are assigned.
 * @param value_1 The first value to assign to the numerator matrix.
 * @param value_2 The second value to assign to the numerator matrix.
 * @param args Additional values to assign, if any.
 */
template <std::size_t IndexCount, typename TransferFunctionNumerator_Type,
          typename T, typename U, typename... Args>
inline void assign_values(TransferFunctionNumerator_Type &numerator, T value_1,
                          U value_2, Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");
  static_assert(
      IndexCount < TransferFunctionNumerator_Type::COLS,
      "Number of arguments must be less than the number of Numerator factor.");

  numerator.template set<IndexCount, 0>(value_1);

  assign_values<IndexCount + 1>(numerator, value_2, args...);
}

} // namespace MakeNumerator

namespace MakeDenominator {

/**
 * @brief Assigns values to the denominator matrix of a transfer function.
 *
 * This function recursively assigns values to the denominator matrix,
 * ensuring that the number of arguments does not exceed the number of columns
 * in the denominator matrix. It uses static assertions to enforce type
 * consistency and size constraints.
 *
 * @tparam IndexCount The current index in the denominator matrix being
 * assigned.
 * @tparam TransferFunctionDenominator_Type The type of the denominator
 * matrix.
 * @tparam T The type of the first value to assign.
 * @param denominator The denominator matrix to which values are assigned.
 * @param value_1 The first value to assign to the denominator matrix.
 */
template <std::size_t IndexCount, typename TransferFunctionDenominator_Type,
          typename T>
inline void assign_values(TransferFunctionDenominator_Type &denominator,
                          T value_1) {

  static_assert(
      IndexCount < TransferFunctionDenominator_Type::COLS,
      "Number of arguments must be less than the number of Numerator factor.");

  denominator.template set<IndexCount, 0>(value_1);
}

/**
 * @brief Assigns multiple values to the denominator matrix of a transfer
 * function.
 *
 * This function recursively assigns multiple values to the denominator matrix,
 * ensuring that all arguments are of the same type and that the number of
 * arguments does not exceed the number of columns in the denominator matrix. It
 * uses static assertions to enforce type consistency and size constraints.
 *
 * @tparam IndexCount The current index in the denominator matrix being
 * assigned.
 * @tparam TransferFunctionDenominator_Type The type of the denominator
 * matrix.
 * @tparam T The type of the first value to assign.
 * @tparam U The type of the second value to assign.
 * @param denominator The denominator matrix to which values are assigned.
 * @param value_1 The first value to assign to the denominator matrix.
 * @param value_2 The second value to assign to the denominator matrix.
 * @param args Additional values to assign, if any.
 */
template <std::size_t IndexCount, typename TransferFunctionDenominator_Type,
          typename T, typename U, typename... Args>
inline void assign_values(TransferFunctionDenominator_Type &denominator,
                          T value_1, U value_2, Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");
  static_assert(
      IndexCount < TransferFunctionDenominator_Type::COLS,
      "Number of arguments must be less than the number of Numerator factor.");

  denominator.template set<IndexCount, 0>(value_1);

  assign_values<IndexCount + 1>(denominator, value_2, args...);
}

} // namespace MakeDenominator

/* make Numerator and Denominator */

/**
 * @brief Creates a TransferFunctionNumerator_Type object with specified values.
 *
 * This function constructs a TransferFunctionNumerator_Type object by
 * assigning values to its elements using a variadic template. It allows for a
 * variable number of arguments to be passed, which are assigned to the
 * numerator matrix.
 *
 * @tparam M The number of coefficients in the numerator (number of rows).
 * @tparam T The type of the first value to assign.
 * @tparam Args Additional types for subsequent values.
 * @param value_1 The first value to assign to the numerator matrix.
 * @param args Additional values to assign, if any.
 * @return A TransferFunctionNumerator_Type object initialized with the
 * provided values.
 */
template <std::size_t M, typename T, typename... Args>
inline auto make_TransferFunctionNumerator(T value_1, Args... args)
    -> TransferFunctionNumerator_Type<T, M> {

  TransferFunctionNumerator_Type<T, M> numerator;

  MakeNumerator::assign_values<0>(numerator, value_1, args...);

  return numerator;
}

/**
 * @brief Creates a TransferFunctionDenominator_Type object with specified
 * values.
 *
 * This function constructs a TransferFunctionDenominator_Type object by
 * assigning values to its elements using a variadic template. It allows for a
 * variable number of arguments to be passed, which are assigned to the
 * denominator matrix.
 *
 * @tparam M The number of coefficients in the denominator (number of rows).
 * @tparam T The type of the first value to assign.
 * @tparam Args Additional types for subsequent values.
 * @param value_1 The first value to assign to the denominator matrix.
 * @param args Additional values to assign, if any.
 * @return A TransferFunctionDenominator_Type object initialized with the
 * provided values.
 */
template <std::size_t M, typename T, typename... Args>
inline auto make_TransferFunctionDenominator(T value_1, Args... args)
    -> TransferFunctionNumerator_Type<T, M> {

  TransferFunctionNumerator_Type<T, M> denominator;

  MakeNumerator::assign_values<0>(denominator, value_1, args...);

  return denominator;
}

namespace ForDiscreteTransferFunction {

/* Create A type definition */
template <typename T, std::size_t N> struct DiscreteStateSpace_A_Type {
  /**
   * @brief Type definition for the A matrix in a discrete state-space system.
   *
   * This struct defines the type of the A matrix based on the number of states
   * (N). It uses PythonNumpy utilities to create a sparse matrix type that
   * concatenates available sparse structures vertically and horizontally.
   *
   * @tparam T The data type of the matrix elements (e.g., double, float).
   * @tparam N The number of states in the discrete state-space system.
   */
  using SparseAvailable_DiscreteStateSpace_A =
      PythonNumpy::ConcatenateSparseAvailableVertically<
          PythonNumpy::DenseAvailable<1, N>,
          PythonNumpy::ConcatenateSparseAvailableHorizontally<
              PythonNumpy::DiagAvailable<N - 1>,
              PythonNumpy::SparseAvailableEmpty<N - 1, 1>>>;

  /**
   * @brief Type definition for the A matrix in a discrete state-space system.
   *
   * This type alias uses PythonNumpy to create a sparse matrix type that
   * represents the A matrix with the specified element type T and the defined
   * sparse structure.
   *
   * @tparam T The data type of the matrix elements (e.g., double, float).
   */
  using type =
      PythonNumpy::SparseMatrix_Type<T, SparseAvailable_DiscreteStateSpace_A>;
};

/* Create B type definition */

/**
 * @brief Type definition for the B matrix in a discrete state-space system.
 *
 * This struct defines the type of the B matrix based on the number of inputs
 * (N). It uses PythonNumpy utilities to create a sparse matrix type that
 * concatenates available sparse structures vertically and horizontally.
 *
 * @tparam T The data type of the matrix elements (e.g., double, float).
 * @tparam N The number of inputs in the discrete state-space system.
 */
template <typename T, std::size_t N> struct DiscreteStateSpace_B_Type {
  using type = PythonNumpy::SparseMatrix_Type<
      T, PythonNumpy::ConcatenateSparseAvailableVertically<
             PythonNumpy::DenseAvailable<1, 1>,
             PythonNumpy::SparseAvailableEmpty<N - 1, 1>>>;
};

/* Create C type definition */
template <typename T, std::size_t Denominator_Size, std::size_t Den_Num_Dif>
struct DiscreteStateSpace_C_Type {
  /**
   * @brief Type definition for the C matrix in a discrete state-space system.
   *
   * This struct defines the type of the C matrix based on the number of
   * denominator differences (Den_Num_Dif) and the total size of the denominator
   * (Denominator_Size). It uses PythonNumpy utilities to create a sparse matrix
   * type that concatenates available sparse structures horizontally.
   *
   * @tparam T The data type of the matrix elements (e.g., double, float).
   * @tparam Denominator_Size The total size of the denominator.
   * @tparam Den_Num_Dif The number of differences in the denominator.
   */
  using type = PythonNumpy::SparseMatrix_Type<
      T, PythonNumpy::ConcatenateSparseAvailableHorizontally<
             PythonNumpy::DenseAvailable<1, (Den_Num_Dif - 1)>,
             PythonNumpy::DenseAvailable<1, (Denominator_Size - Den_Num_Dif)>>>;
};

template <typename T, std::size_t Denominator_Size>
struct DiscreteStateSpace_C_Type<T, Denominator_Size, 1> {
  /**
   * @brief Type definition for the C matrix in a discrete state-space system
   * with a single denominator difference.
   *
   * This struct defines the type of the C matrix when there is only one
   * difference in the denominator. It uses PythonNumpy utilities to create a
   * sparse matrix type that concatenates available sparse structures
   * horizontally.
   *
   * @tparam T The data type of the matrix elements (e.g., double, float).
   * @tparam Denominator_Size The total size of the denominator.
   */
  using type = PythonNumpy::SparseMatrix_Type<
      T, PythonNumpy::DenseAvailable<1, (Denominator_Size - 1)>>;
};

template <typename T, std::size_t Denominator_Size>
struct DiscreteStateSpace_C_Type<T, Denominator_Size, 0> {
  /**
   * @brief Type definition for the C matrix in a discrete state-space system
   * with no denominator differences.
   *
   * This struct defines the type of the C matrix when there are no differences
   * in the denominator. It uses PythonNumpy utilities to create a sparse matrix
   * type that consists of a single dense available column.
   *
   * @tparam T The data type of the matrix elements (e.g., double, float).
   * @tparam Denominator_Size The total size of the denominator.
   */
  using type = PythonNumpy::SparseMatrix_Type<
      T, PythonNumpy::DenseAvailable<1, (Denominator_Size - 1)>>;
};

/* Create D type definition */
template <typename T, bool IsStrictlyProper>
struct DiscreteStateSpace_D_Type {};

template <typename T> struct DiscreteStateSpace_D_Type<T, true> {
  /**
   * @brief Type definition for the D matrix in a discrete state-space system
   * when it is strictly proper.
   *
   * This struct defines the type of the D matrix as a sparse matrix with no
   * columns, indicating that there is no direct feedthrough from input to
   * output in a strictly proper system.
   *
   * @tparam T The data type of the matrix elements (e.g., double, float).
   */
  using type = PythonNumpy::SparseMatrix_Type<
      T, PythonNumpy::SparseAvailable<PythonNumpy::ColumnAvailable<false>>>;
};

template <typename T> struct DiscreteStateSpace_D_Type<T, false> {
  /**
   * @brief Type definition for the D matrix in a discrete state-space system
   * when it is not strictly proper.
   *
   * This struct defines the type of the D matrix as a sparse matrix with one
   * column, indicating that there is a direct feedthrough from input to output
   * in a non-strictly proper system.
   *
   * @tparam T The data type of the matrix elements (e.g., double, float).
   */
  using type = PythonNumpy::SparseMatrix_Type<
      T, PythonNumpy::SparseAvailable<PythonNumpy::ColumnAvailable<true>>>;
};

/* Set A value */
template <typename T, typename DiscreteStateSpace_A_type,
          typename Denominator_Type, std::size_t I, std::size_t LoopCount>
struct Set_A_Value {
  /**
   * @brief Sets the value of the A matrix at index I based on the denominator.
   *
   * This function sets the value of the A matrix at index I to a calculated
   * value based on the denominator. It recursively calls itself to set values
   * for subsequent indices until LoopCount reaches zero.
   *
   * @param A The A matrix to be modified.
   * @param denominator The denominator used for calculating the A matrix value.
   */
  static void set(DiscreteStateSpace_A_type &A,
                  const Denominator_Type &denominator) {
    A(I) = -denominator(I + 1) /
           Base::Utility::avoid_zero_divide(
               denominator(0),
               static_cast<T>(PythonControl::TRANSFER_FUNCTION_DIVISION_MIN));
    Set_A_Value<T, DiscreteStateSpace_A_type, Denominator_Type, (I + 1),
                (LoopCount - 1)>::set(A, denominator);
  }
};

template <typename T, typename DiscreteStateSpace_A_type,
          typename Denominator_Type, std::size_t I>
struct Set_A_Value<T, DiscreteStateSpace_A_type, Denominator_Type, I, 0> {
  /**
   * @brief Sets the value of the A matrix at index I based on the denominator.
   *
   * This function sets the value of the A matrix at index I to a calculated
   * value based on the denominator. It is the base case for recursion when
   * LoopCount reaches zero.
   *
   * @param A The A matrix to be modified.
   * @param denominator The denominator used for calculating the A matrix value.
   */
  static void set(DiscreteStateSpace_A_type &A,
                  const Denominator_Type &denominator) {
    A(I) = -denominator(I + 1) /
           Base::Utility::avoid_zero_divide(
               denominator(0),
               static_cast<T>(PythonControl::TRANSFER_FUNCTION_DIVISION_MIN));
  }
};

template <typename T, typename DiscreteStateSpace_A_type,
          typename std::size_t I, std::size_t LoopCount>
struct Set_A_Ones {
  /**
   * @brief Sets the value of the A matrix at index I to one.
   *
   * This function sets the value of the A matrix at index I to one and
   * recursively calls itself to set values for subsequent indices until
   * LoopCount reaches zero.
   *
   * @param A The A matrix to be modified.
   */
  static void set(DiscreteStateSpace_A_type &A) {
    A(I) = static_cast<T>(1);
    Set_A_Ones<T, DiscreteStateSpace_A_type, (I + 1), (LoopCount - 1)>::set(A);
  }
};

template <typename T, typename DiscreteStateSpace_A_type, std::size_t I>
struct Set_A_Ones<T, DiscreteStateSpace_A_type, I, 0> {
  /**
   * @brief Sets the value of the A matrix at index I to one.
   *
   * This function sets the value of the A matrix at index I to one. It is the
   * base case for recursion when LoopCount reaches zero.
   *
   * @param A The A matrix to be modified.
   */
  static void set(DiscreteStateSpace_A_type &A) { A(I) = static_cast<T>(1); }
};

/* Set C value */
template <typename T, typename DiscreteStateSpace_C_type,
          typename Numerator_Type, typename Denominator_Type,
          std::size_t C_INDEX_OFFSET, std::size_t I, std::size_t LoopCount,
          bool Is_Strictly_Proper>
struct Set_C_ValueElements {};

template <typename T, typename DiscreteStateSpace_C_type,
          typename Numerator_Type, typename Denominator_Type,
          std::size_t C_INDEX_OFFSET, std::size_t I, std::size_t LoopCount>
struct Set_C_ValueElements<T, DiscreteStateSpace_C_type, Numerator_Type,
                           Denominator_Type, C_INDEX_OFFSET, I, LoopCount,
                           true> {
  /**
   * @brief Sets the value of the C matrix at index I based on the numerator and
   * denominator.
   *
   * This function sets the value of the C matrix at index I to a calculated
   * value based on the numerator and the inverse of the first element of the
   * denominator. It recursively calls itself to set values for subsequent
   * indices until LoopCount reaches zero.
   *
   * @param C The C matrix to be modified.
   * @param numerator The numerator used for calculating the C matrix value.
   * @param denominator_0_inv The inverse of the first element of the
   * denominator.
   */
  static void set(DiscreteStateSpace_C_type &C, const Numerator_Type &numerator,
                  const T &denominator_0_inv) {
    C(I + C_INDEX_OFFSET) = numerator(I) * denominator_0_inv;

    Set_C_ValueElements<T, DiscreteStateSpace_C_type, Numerator_Type,
                        Denominator_Type, C_INDEX_OFFSET, (I + 1),
                        (LoopCount - 1), true>::set(C, numerator,
                                                    denominator_0_inv);
  }
};

template <typename T, typename DiscreteStateSpace_C_type,
          typename Numerator_Type, typename Denominator_Type,
          std::size_t C_INDEX_OFFSET, std::size_t I>
struct Set_C_ValueElements<T, DiscreteStateSpace_C_type, Numerator_Type,
                           Denominator_Type, C_INDEX_OFFSET, I, 0, true> {
  /**
   * @brief Sets the value of the C matrix at index I based on the numerator and
   * denominator.
   *
   * This function sets the value of the C matrix at index I to a calculated
   * value based on the numerator and the inverse of the first element of the
   * denominator. It is the base case for recursion when LoopCount reaches zero.
   *
   * @param C The C matrix to be modified.
   * @param numerator The numerator used for calculating the C matrix value.
   * @param denominator_0_inv The inverse of the first element of the
   * denominator.
   */
  static void set(DiscreteStateSpace_C_type &C, const Numerator_Type &numerator,
                  const T &denominator_0_inv) {
    C(I + C_INDEX_OFFSET) = numerator(I) * denominator_0_inv;
  }
};

template <typename T, typename DiscreteStateSpace_C_type,
          typename Numerator_Type, typename Denominator_Type,
          std::size_t C_INDEX_OFFSET, std::size_t I, std::size_t LoopCount>
struct Set_C_ValueElements<T, DiscreteStateSpace_C_type, Numerator_Type,
                           Denominator_Type, C_INDEX_OFFSET, I, LoopCount,
                           false> {
  /**
   * @brief Sets the value of the C matrix at index I based on the numerator and
   * denominator.
   *
   * This function sets the value of the C matrix at index I to a calculated
   * value based on the numerator and denominator. It recursively calls itself
   * to set values for subsequent indices until LoopCount reaches zero.
   *
   * @param C The C matrix to be modified.
   * @param numerator The numerator used for calculating the C matrix value.
   * @param denominator The denominator used for calculating the C matrix value.
   * @param denominator_0_inv The inverse of the first element of the
   * denominator.
   */
  static void set(DiscreteStateSpace_C_type &C, const Numerator_Type &numerator,
                  const Denominator_Type &denominator,
                  const T &denominator_0_inv) {
    C(I - 1) =
        numerator(I) * denominator_0_inv -
        denominator(I) * numerator(0) * (denominator_0_inv * denominator_0_inv);

    Set_C_ValueElements<T, DiscreteStateSpace_C_type, Numerator_Type,
                        Denominator_Type, C_INDEX_OFFSET, (I + 1),
                        (LoopCount - 1), false>::set(C, numerator, denominator,
                                                     denominator_0_inv);
  }
};

template <typename T, typename DiscreteStateSpace_C_type,
          typename Numerator_Type, typename Denominator_Type,
          std::size_t C_INDEX_OFFSET, std::size_t I>
struct Set_C_ValueElements<T, DiscreteStateSpace_C_type, Numerator_Type,
                           Denominator_Type, C_INDEX_OFFSET, I, 0, false> {
  /**
   * @brief Sets the value of the C matrix at index I based on the numerator and
   * denominator.
   *
   * This function sets the value of the C matrix at index I to a calculated
   * value based on the numerator and denominator. It is the base case for
   * recursion when LoopCount reaches zero.
   *
   * @param C The C matrix to be modified.
   * @param numerator The numerator used for calculating the C matrix value.
   * @param denominator The denominator used for calculating the C matrix value.
   * @param denominator_0_inv The inverse of the first element of the
   * denominator.
   */
  static void set(DiscreteStateSpace_C_type &C, const Numerator_Type &numerator,
                  const Denominator_Type &denominator,
                  const T &denominator_0_inv) {
    C(I - 1) =
        numerator(I) * denominator_0_inv -
        denominator(I) * numerator(0) * (denominator_0_inv * denominator_0_inv);
  }
};

template <typename T, typename DiscreteStateSpace_C_type,
          typename Numerator_Type, typename Denominator_Type,
          bool Is_Strictly_Proper>
struct Set_C_Value {};

template <typename T, typename DiscreteStateSpace_C_type,
          typename Numerator_Type, typename Denominator_Type>
struct Set_C_Value<T, DiscreteStateSpace_C_type, Numerator_Type,
                   Denominator_Type, true> {

  /**
   * @brief Sets the C matrix values for a strictly proper transfer function.
   *
   * This function sets the C matrix values based on the numerator and
   * denominator, ensuring that the first element of the denominator is not
   * zero. It uses a specialized method to handle strictly proper systems.
   *
   * @param C The C matrix to be modified.
   * @param numerator The numerator used for calculating the C matrix value.
   * @param denominator The denominator used for calculating the C matrix value.
   */
  static void set(DiscreteStateSpace_C_type &C, const Numerator_Type &numerator,
                  const Denominator_Type &denominator) {

    T denominator_0_inv =
        static_cast<T>(1) /
        Base::Utility::avoid_zero_divide(
            denominator(0),
            static_cast<T>(PythonControl::TRANSFER_FUNCTION_DIVISION_MIN));
    constexpr std::size_t C_INDEX_OFFSET =
        Denominator_Type::COLS - Numerator_Type::COLS - 1;

    Set_C_ValueElements<T, DiscreteStateSpace_C_type, Numerator_Type,
                        Denominator_Type, C_INDEX_OFFSET, 0,
                        (Numerator_Type::COLS - 1),
                        true>::set(C, numerator, denominator_0_inv);
  }
};

template <typename T, typename DiscreteStateSpace_C_type,
          typename Numerator_Type, typename Denominator_Type>
struct Set_C_Value<T, DiscreteStateSpace_C_type, Numerator_Type,
                   Denominator_Type, false> {
  /**
   * @brief Sets the C matrix values for a non-strictly proper transfer
   * function.
   *
   * This function sets the C matrix values based on the numerator and
   * denominator, ensuring that the first element of the denominator is not
   * zero. It uses a specialized method to handle non-strictly proper systems.
   *
   * @param C The C matrix to be modified.
   * @param numerator The numerator used for calculating the C matrix value.
   * @param denominator The denominator used for calculating the C matrix value.
   */
  static void set(DiscreteStateSpace_C_type &C, const Numerator_Type &numerator,
                  const Denominator_Type &denominator) {

    T denominator_0_inv =
        static_cast<T>(1) /
        Base::Utility::avoid_zero_divide(
            denominator(0),
            static_cast<T>(PythonControl::TRANSFER_FUNCTION_DIVISION_MIN));

    Set_C_ValueElements<T, DiscreteStateSpace_C_type, Numerator_Type,
                        Denominator_Type, 0, 1, (Numerator_Type::COLS - 2),
                        false>::set(C, numerator, denominator,
                                    denominator_0_inv);
  }
};

/* Set D value */
template <typename T, typename DiscreteStateSpace_D_type,
          typename Numerator_Type, typename Denominator_Type,
          bool Is_Strictly_Proper>
struct Set_D_Value {};

template <typename T, typename DiscreteStateSpace_D_type,
          typename Numerator_Type, typename Denominator_Type>
struct Set_D_Value<T, DiscreteStateSpace_D_type, Numerator_Type,
                   Denominator_Type, true> {
  /**
   * @brief Sets the D matrix value for a strictly proper transfer function.
   *
   * This function does nothing for strictly proper systems, as they do not have
   * a direct feedthrough term in the D matrix.
   *
   * @param D The D matrix to be modified (no modification occurs).
   * @param numerator The numerator used for calculating the D matrix value
   * (not used).
   * @param denominator The denominator used for calculating the D matrix value
   * (not used).
   */
  static void set(DiscreteStateSpace_D_type &D, const Numerator_Type &numerator,
                  const Denominator_Type &denominator) {
    /* Do Nothing */
    static_cast<void>(D);
    static_cast<void>(numerator);
    static_cast<void>(denominator);
  }
};

template <typename T, typename DiscreteStateSpace_D_type,
          typename Numerator_Type, typename Denominator_Type>
struct Set_D_Value<T, DiscreteStateSpace_D_type, Numerator_Type,
                   Denominator_Type, false> {
  /**
   * @brief Sets the D matrix value for a non-strictly proper transfer
   * function.
   *
   * This function sets the D matrix value based on the first element of the
   * numerator and denominator, ensuring that the denominator is not zero.
   *
   * @param D The D matrix to be modified.
   * @param numerator The numerator used for calculating the D matrix value.
   * @param denominator The denominator used for calculating the D matrix value.
   */
  static void set(DiscreteStateSpace_D_type &D, const Numerator_Type &numerator,
                  const Denominator_Type &denominator) {
    D(0) = numerator(0) /
           Base::Utility::avoid_zero_divide(
               denominator(0),
               static_cast<T>(PythonControl::TRANSFER_FUNCTION_DIVISION_MIN));
  }
};

/* Solve steady state and input */
template <typename T, typename State_Space_Type, bool Is_Strictly_Proper>
struct SolveSteadyStateAndInput {};

template <typename T, typename State_Space_Type>
struct SolveSteadyStateAndInput<T, State_Space_Type, true> {
  /**
   * @brief Solves for the steady state and input of a discrete transfer
   * function.
   *
   * This function calculates the steady state input based on the provided
   * steady state output and the state space representation of the transfer
   * function. It uses matrix operations to compute the input that results in
   * the desired steady state output.
   *
   * @param state_space The state space representation of the transfer function.
   * @param y_steady_state The desired steady state output.
   * @return The calculated steady state input.
   */
  static T solve(State_Space_Type &state_space, const T &y_steady_state) {
    T u_steady_state;

    auto I_A = PythonNumpy::make_DiagMatrixIdentity<
                   T, State_Space_Type::Original_X_Type::COLS>() -
               state_space.A;

    auto solver = PythonNumpy::make_LinalgSolver<decltype(I_A),
                                                 decltype(state_space.B)>();
    solver.solve(I_A, state_space.B);

    auto I_A_B = solver.get_answer();

    auto C_I_A_B = state_space.C * I_A_B;

    // This is a transfer function, so it is single input single output
    // system. Thus, "y_steady_state" and "C_I_A_B" are scalar.
    u_steady_state =
        y_steady_state /
        Base::Utility::avoid_zero_divide(
            C_I_A_B.template get<0, 0>(),
            static_cast<T>(PythonControl::TRANSFER_FUNCTION_DIVISION_MIN));

    state_space.X = I_A_B * u_steady_state;

    return u_steady_state;
  }
};

template <typename T, typename State_Space_Type>
struct SolveSteadyStateAndInput<T, State_Space_Type, false> {
  /**
   * @brief Solves for the steady state and input of a discrete transfer
   * function that is not strictly proper.
   *
   * This function calculates the steady state input based on the provided
   * steady state output and the state space representation of the transfer
   * function. It uses matrix operations to compute the input that results in
   * the desired steady state output.
   *
   * @param state_space The state space representation of the transfer function.
   * @param y_steady_state The desired steady state output.
   * @return The calculated steady state input.
   */
  static T solve(State_Space_Type &state_space, const T &y_steady_state) {
    T u_steady_state;

    auto I_A = PythonNumpy::make_DiagMatrixIdentity<
                   T, State_Space_Type::Original_X_Type::COLS>() -
               state_space.A;

    // This is a transfer function, so D is scalar.
    auto I_A_D_B_C = I_A * state_space.D.template get<0, 0>() +
                     state_space.B * state_space.C;

    auto B_y = state_space.B * y_steady_state;

    auto solver =
        PythonNumpy::make_LinalgSolver<decltype(I_A_D_B_C), decltype(B_y)>();
    solver.solve(I_A_D_B_C, B_y);

    state_space.X = solver.get_answer();

    u_steady_state =
        (y_steady_state -
         (state_space.C * state_space.X).template get<0, 0>()) /
        Base::Utility::avoid_zero_divide(
            state_space.D.template get<0, 0>(),
            static_cast<T>(PythonControl::TRANSFER_FUNCTION_DIVISION_MIN));

    return u_steady_state;
  }
};

} // namespace ForDiscreteTransferFunction

/* Discrete Transfer Function */

/**
 * @brief Represents a discrete transfer function with numerator and denominator
 * matrices.
 *
 * This class encapsulates the properties and methods of a discrete transfer
 * function, including its numerator and denominator matrices, state space
 * representation, and methods for solving steady state inputs and outputs.
 *
 * @tparam Numerator_Type_In The type of the numerator matrix.
 * @tparam Denominator_Type_In The type of the denominator matrix.
 * @tparam Number_Of_Delay The number of delays in the system (default is 0).
 */
template <typename Numerator_Type_In, typename Denominator_Type_In,
          std::size_t Number_Of_Delay = 0>
class DiscreteTransferFunction {
public:
  /* Type */
  using Numerator_Type = Numerator_Type_In;
  using Denominator_Type = Denominator_Type_In;

protected:
  /* Type */
  using _T = typename Numerator_Type::Value_Type;
  static_assert(
      std::is_same<_T, double>::value || std::is_same<_T, float>::value,
      "Numerator and denominator value data type must be float or double.");

  static_assert(Denominator_Type::COLS > 1,
                "Denominator must have at least 2 elements.");
  static constexpr std::size_t _DENOMINATOR_DIMENSION =
      Denominator_Type::COLS - 1;
  static_assert(Numerator_Type::COLS > 1,
                "Numerator must have at least 2 elements.");
  static constexpr std::size_t _NUMERATOR_DIMENSION = Numerator_Type::COLS - 1;

  static constexpr bool _IS_STRICTLY_PROPER =
      _NUMERATOR_DIMENSION < _DENOMINATOR_DIMENSION;

  using _DiscreteStateSpace_A_type =
      typename ForDiscreteTransferFunction::DiscreteStateSpace_A_Type<
          _T, _DENOMINATOR_DIMENSION>::type;

  using _DiscreteStateSpace_B_type =
      typename ForDiscreteTransferFunction::DiscreteStateSpace_B_Type<
          _T, _DENOMINATOR_DIMENSION>::type;

  using _DiscreteStateSpace_C_type =
      typename ForDiscreteTransferFunction::DiscreteStateSpace_C_Type<
          _T, Denominator_Type::COLS,
          (Denominator_Type::COLS - Numerator_Type::COLS)>::type;

  using _DiscreteStateSpace_D_type =
      typename ForDiscreteTransferFunction::DiscreteStateSpace_D_Type<
          _T, _IS_STRICTLY_PROPER>::type;

  using _U_Type = PythonControl::StateSpaceInput_Type<_T, 1>;

  using _State_Space_Type = PythonControl::DiscreteStateSpace<
      _DiscreteStateSpace_A_type, _DiscreteStateSpace_B_type,
      _DiscreteStateSpace_C_type, _DiscreteStateSpace_D_type, Number_Of_Delay>;

  /* Check Compatibility */
  /* Check Data Type */
  static_assert(std::is_same<typename Denominator_Type::Value_Type, _T>::value,
                "Data type of denominator must be same type as numerator.");

  /* Check Numerator and Denominator length */
  static_assert(Numerator_Type::COLS <= Denominator_Type::COLS,
                "Transfer function must be proper.");

public:
  /* Type */
  using Value_Type = _T;

public:
  /* Constructor */
  DiscreteTransferFunction() {}

  DiscreteTransferFunction(const Numerator_Type &numerator,
                           const Denominator_Type &denominator, _T delta_time) {
    /* Set A */
    ForDiscreteTransferFunction::Set_A_Value<
        _T, _DiscreteStateSpace_A_type, Denominator_Type, 0,
        (_DENOMINATOR_DIMENSION - 1)>::set(this->_state_space.A, denominator);

    ForDiscreteTransferFunction::Set_A_Ones<
        _T, _DiscreteStateSpace_A_type, _DENOMINATOR_DIMENSION,
        (2 * (_DENOMINATOR_DIMENSION - 1))>::set(this->_state_space.A);

    /* Set B */
    this->_state_space.B(0) = static_cast<_T>(1);

    /* Set C */
    ForDiscreteTransferFunction::Set_C_Value<
        _T, _DiscreteStateSpace_C_type, Numerator_Type, Denominator_Type,
        _IS_STRICTLY_PROPER>::set(this->_state_space.C, numerator, denominator);

    /* Set D */
    ForDiscreteTransferFunction::Set_D_Value<
        _T, _DiscreteStateSpace_D_type, Numerator_Type, Denominator_Type,
        _IS_STRICTLY_PROPER>::set(this->_state_space.D, numerator, denominator);

    /* Set delta time */
    this->_state_space.delta_time = delta_time;
  }

  /* Copy Constructor */
  DiscreteTransferFunction(
      const DiscreteTransferFunction<Numerator_Type, Denominator_Type> &input)
      : _state_space(input._state_space) {}

  DiscreteTransferFunction<Numerator_Type, Denominator_Type> &operator=(
      const DiscreteTransferFunction<Numerator_Type, Denominator_Type> &input) {
    if (this != &input) {
      this->_state_space = input._state_space;
    }
    return *this;
  }

  /* Move Constructor */
  DiscreteTransferFunction(
      DiscreteTransferFunction<Numerator_Type, Denominator_Type>
          &&input) noexcept
      : _state_space(std::move(input._state_space)) {}

  DiscreteTransferFunction<Numerator_Type, Denominator_Type> &
  operator=(DiscreteTransferFunction<Numerator_Type, Denominator_Type>
                &&input) noexcept {
    if (this != &input) {
      this->_state_space = std::move(input._state_space);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Returns the number of delays in the transfer function.
   *
   * This function returns the constant value representing the number of delays
   * in the transfer function.
   *
   * @return The number of delays.
   */
  static constexpr std::size_t get_number_of_delay(void) {
    return NUMBER_OF_DELAY;
  }

  /**
   * @brief Returns the state space representation of the transfer function.
   *
   * This function returns the state space representation of the transfer
   * function, which includes matrices A, B, C, D, and the state vector X.
   *
   * @return The state space representation.
   */
  auto get_X(void) const -> typename _State_Space_Type::Original_X_Type {
    return this->_state_space.X;
  }

  /**
   * @brief Returns the output of the transfer function.
   *
   * This function returns the output of the transfer function, which is
   * calculated as the product of the C matrix and the state vector X.
   *
   * @return The output value.
   */
  _T get_y(void) const {
    return this->_state_space.get_Y().template get<0, 0>();
  }

  /**
   * @brief Returns the input of the transfer function.
   *
   * This function returns the input of the transfer function, which is
   * calculated as the product of the D matrix and the state vector X.
   *
   * @return The input value.
   */
  void update(const _T &u) {
    auto input = make_StateSpaceInput<_U_Type::COLS>(u);

    this->_state_space.update(input);
  }

  /**
   * @brief Resets the numerator and denominator of the transfer function.
   *
   * This function resets the numerator and denominator matrices of the transfer
   * function to the provided values, updating the state space representation
   * accordingly.
   *
   * @param numerator The new numerator matrix.
   * @param denominator The new denominator matrix.
   */
  void reset_numerator_and_denominator(const Numerator_Type &numerator,
                                       const Denominator_Type &denominator) {
    /* Set A */
    ForDiscreteTransferFunction::Set_A_Value<
        _T, _DiscreteStateSpace_A_type, Denominator_Type, 0,
        (_DENOMINATOR_DIMENSION - 1)>::set(this->_state_space.A, denominator);

    ForDiscreteTransferFunction::Set_A_Ones<
        _T, _DiscreteStateSpace_A_type, _DENOMINATOR_DIMENSION,
        (2 * (_DENOMINATOR_DIMENSION - 1))>::set(this->_state_space.A);

    /* Set C */
    ForDiscreteTransferFunction::Set_C_Value<
        _T, _DiscreteStateSpace_C_type, Numerator_Type, Denominator_Type,
        _IS_STRICTLY_PROPER>::set(this->_state_space.C, numerator, denominator);

    /* Set D */
    ForDiscreteTransferFunction::Set_D_Value<
        _T, _DiscreteStateSpace_D_type, Numerator_Type, Denominator_Type,
        _IS_STRICTLY_PROPER>::set(this->_state_space.D, numerator, denominator);
  }

  /**
   * @brief Resets the state of the transfer function.
   *
   * This function resets the state of the transfer function, clearing the
   * internal state space representation.
   */
  void reset_state(void) { this->_state_space.reset_state(); }

  /**
   * @brief Solves for the steady state input based on the provided steady state
   * output.
   *
   * This function calculates the steady state input that would result in the
   * specified steady state output, using the state space representation of the
   * transfer function.
   *
   * @param y_steady_state The desired steady state output.
   * @return The calculated steady state input.
   */
  _T solve_steady_state_and_input(const _T &y_steady_state) {

    _T u_steady_state = ForDiscreteTransferFunction::SolveSteadyStateAndInput<
        _T, _State_Space_Type, _IS_STRICTLY_PROPER>::solve(this->_state_space,
                                                           y_steady_state);

    return u_steady_state;
  }

public:
  /* Constant */
  static constexpr std::size_t NUMBER_OF_DELAY = Number_Of_Delay;
  static constexpr std::size_t STATE_SIZE =
      _State_Space_Type::Original_X_Type::COLS;

protected:
  /* Variable */
  _State_Space_Type _state_space;
};

/* make Discrete Transfer Function */

/**
 * @brief Creates a DiscreteTransferFunction object with the specified
 * numerator, denominator, and delta time.
 *
 * This function constructs a DiscreteTransferFunction object using the provided
 * numerator and denominator matrices, along with the specified delta time.
 *
 * @tparam Numerator_Type The type of the numerator matrix.
 * @tparam Denominator_Type The type of the denominator matrix.
 * @param numerator The numerator matrix.
 * @param denominator The denominator matrix.
 * @param delta_time The time step for the discrete system.
 * @return A DiscreteTransferFunction object initialized with the given
 * parameters.
 */
template <typename Numerator_Type, typename Denominator_Type>
inline auto
make_DiscreteTransferFunction(Numerator_Type numerator,
                              Denominator_Type denominator,
                              typename Numerator_Type::Value_Type delta_time)
    -> DiscreteTransferFunction<Numerator_Type, Denominator_Type> {

  return DiscreteTransferFunction<Numerator_Type, Denominator_Type>(
      numerator, denominator, delta_time);
}

/**
 * @brief Creates a DiscreteTransferFunction object with the specified
 * numerator, denominator, and delta time, allowing for a specified number of
 * delays.
 *
 * This function constructs a DiscreteTransferFunction object using the provided
 * numerator and denominator matrices, along with the specified delta time and
 * number of delays.
 *
 * @tparam Number_Of_Delay The number of delays in the system (default is 0).
 * @tparam Numerator_Type The type of the numerator matrix.
 * @tparam Denominator_Type The type of the denominator matrix.
 * @param numerator The numerator matrix.
 * @param denominator The denominator matrix.
 * @param delta_time The time step for the discrete system.
 * @return A DiscreteTransferFunction object initialized with the given
 * parameters.
 */
template <std::size_t Number_Of_Delay, typename Numerator_Type,
          typename Denominator_Type>
inline auto
make_DiscreteTransferFunction(Numerator_Type numerator,
                              Denominator_Type denominator,
                              typename Numerator_Type::Value_Type delta_time)
    -> DiscreteTransferFunction<Numerator_Type, Denominator_Type,
                                Number_Of_Delay> {

  return DiscreteTransferFunction<Numerator_Type, Denominator_Type,
                                  Number_Of_Delay>(numerator, denominator,
                                                   delta_time);
}

/* Discrete Transfer Function Type */
template <typename Numerator_Type, typename Denominator_Type,
          std::size_t Number_Of_Delay = 0>
using DiscreteTransferFunction_Type =
    DiscreteTransferFunction<Numerator_Type, Denominator_Type, Number_Of_Delay>;

} // namespace PythonControl

#endif // __PYTHON_CONTROL_TRANSFER_FUNCTION_HPP__
