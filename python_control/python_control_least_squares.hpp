/**
 * @file python_control_least_squares.hpp
 * @brief Provides Least Squares and Recursive Least Squares algorithms for
 * regression and system identification.
 *
 * This header defines template-based C++ classes and utility functions for
 * performing Least Squares (LS) and Recursive Least Squares (RLS) computations.
 * The code is designed for use in control systems and modeling, supporting both
 * batch and online learning scenarios. It leverages custom matrix types and
 * utilities for efficient numerical operations, including bias handling and
 * regularization.
 */
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

/**
 * @brief Assigns values to specific positions in a LeastSquaresInputType
 * object.
 *
 * This function template assigns the provided value to the element at position
 * (IndexCount, 0) in the input object, and assigns the value 1 (cast to type T)
 * to the element at position (IndexCount + 1, 0). It uses a static assertion to
 * ensure that IndexCount is less than the number of columns in the input type.
 *
 * @tparam IndexCount The index at which to assign the first value.
 * @tparam LeastSquaresInputType The type of the input object, which must
 * provide COLS and a set method.
 * @tparam T The type of the value to assign.
 * @param input Reference to the input object to modify.
 * @param value_1 The value to assign at position (IndexCount, 0).
 */
template <std::size_t IndexCount, typename LeastSquaresInputType, typename T>
inline void assign_values(LeastSquaresInputType &input, T value_1) {

  static_assert(
      IndexCount < LeastSquaresInputType::COLS,
      "Number of arguments must be less than the number of input size.");

  input.template set<IndexCount, 0>(value_1);
  input.template set<(IndexCount + 1), 0>(static_cast<T>(1));
}

/**
 * @brief Recursively assigns values to a LeastSquaresInputType object.
 *
 * This function template assigns the first value to the element at position
 * (IndexCount, 0) in the input object, and then recursively calls itself to
 * assign the next value. It uses static assertions to ensure that all values
 * are of the same type and that IndexCount is less than the number of columns
 * in the input type.
 *
 * @tparam IndexCount The current index for assignment.
 * @tparam LeastSquaresInputType The type of the input object, which must
 * provide COLS and a set method.
 * @tparam T The type of the first value to assign.
 * @tparam U The type of the second value to assign.
 * @param input Reference to the input object to modify.
 * @param value_1 The first value to assign at position (IndexCount, 0).
 * @param value_2 The second value to assign at position (IndexCount + 1, 0).
 * @param args Additional values to assign in subsequent positions.
 */
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

/**
 * @brief Creates a LeastSquaresInputType object with the specified values.
 *
 * This function template initializes a LeastSquaresInputType object with the
 * provided values, assigning the first value to the first position and the
 * second value to the second position, and so on. It uses variadic templates to
 * handle an arbitrary number of arguments.
 *
 * @tparam Vector_Size The size of the vector in the LeastSquaresInputType.
 * @tparam T The type of the first value to assign.
 * @param value_1 The first value to assign at position (0, 0).
 * @param args Additional values to assign in subsequent positions.
 * @return LeastSquaresInputType<T, Vector_Size> The resulting input object.
 */
template <std::size_t Vector_Size, typename T, typename... Args>
inline auto make_LeastSquaresInput(T value_1, Args... args)
    -> LeastSquaresInputType<T, Vector_Size> {

  LeastSquaresInputType<T, Vector_Size> input;

  MakeLeastSquaresInput::assign_values<0>(input, value_1, args...);

  return input;
}

/* Least Squares Method */

/**
 * @brief Least Squares Method for regression and system identification.
 *
 * This class implements the Least Squares method, allowing for fitting a model
 * to data and making predictions. It supports batch learning with a fixed set
 * of input-output pairs.
 *
 * @tparam X_Type_In The type of the input data, which must provide COLS and
 * ROWS.
 */
template <typename X_Type_In> class LeastSquares {
public:
  /* Type */
  using X_Type = X_Type_In;

protected:
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

public:
  /* Type */
  using Value_Type = _T;

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

  /**
   * @brief Fits the model to the provided input and output data.
   *
   * This function computes the weights of the model using the least squares
   * method based on the provided input (X) and output (Y) data. It adds a bias
   * term to the input data before solving for the weights.
   *
   * @param X The input data matrix of shape (ROWS, COLS).
   * @param Y The output data vector of shape (ROWS, 1).
   */
  inline void fit(const X_Type &X, const Y_Type &Y) {

    auto bias_vector = PythonNumpy::make_DenseMatrixOnes<_T, X_Type::COLS, 1>();

    auto X_ex = PythonNumpy::concatenate_horizontally(X, bias_vector);

    this->_weights = this->_lstsq_solver.solve(X_ex, Y);
  }

  /**
   * @brief Fits the model to the provided input data and a single output value.
   *
   * This function computes the weights of the model using the least squares
   * method based on the provided input (X) and a single output value (y_true).
   * It adds a bias term to the input data before solving for the weights.
   *
   * @param X The input data matrix of shape (ROWS, COLS).
   * @param y_true The true output value.
   */
  inline auto predict(const X_Type &X) const -> Y_Type {

    auto bias_vector = PythonNumpy::make_DenseMatrixOnes<_T, X_Type::COLS, 1>();

    auto X_ex = PythonNumpy::concatenate_horizontally(X, bias_vector);

    return X_ex * this->_weights;
  }

  /**
   * @brief Predicts the output for a single input vector.
   *
   * This function computes the predicted output for a single input vector
   * using the learned weights. It adds a bias term to the input vector before
   * making the prediction.
   *
   * @param X The input vector of shape (1, COLS).
   * @return The predicted output value.
   */
  inline auto get_weights(void) const -> _Wights_Type { return this->_weights; }

  /**
   * @brief Sets the decay rate for the least squares solver.
   *
   * This function updates the decay rate used in the least squares solver,
   * which can affect the regularization of the solution.
   *
   * @param decay_rate_in The new decay rate to be set.
   */
  inline void set_lstsq_solver_decay_rate(const _T &decay_rate_in) {
    this->_lstsq_solver.set_decay_rate(decay_rate_in);
  }

  /**
   * @brief Sets the minimum division value for the least squares solver.
   *
   * This function updates the minimum division value used in the least squares
   * solver to avoid division by zero errors.
   *
   * @param division_min_in The new minimum division value to be set.
   */
  inline void set_lstsq_solver_division_min(const _T &division_min_in) {
    this->_lstsq_solver.set_division_min(division_min_in);
  }

protected:
  /* Variables */
  _Wights_Type _weights;
  _LstsqSolver_Type _lstsq_solver;
};

/* make Least Squares */

/**
 * @brief Creates a LeastSquares object.
 *
 * This function template initializes a LeastSquares object with the default
 * settings. It can be used to perform least squares regression and system
 * identification.
 *
 * @tparam X_Type The type of the input data, which must provide COLS and ROWS.
 * @return LeastSquares<X_Type> The resulting LeastSquares object.
 */
template <typename X_Type>
inline auto make_LeastSquares(void) -> LeastSquares<X_Type> {
  return LeastSquares<X_Type>();
}

/* Least Squares type */
template <typename X_Type> using LeastSquares_Type = LeastSquares<X_Type>;

/* Recursive Least Squares Method */

/**
 * @brief Recursive Least Squares Method for online regression and system
 * identification.
 *
 * This class implements the Recursive Least Squares method, allowing for
 * incremental updates to the model as new data becomes available. It supports
 * online learning scenarios with a focus on adaptive filtering and control.
 *
 * @tparam X_Type_In The type of the input data, which must provide COLS.
 */
template <typename X_Type_In> class RecursiveLeastSquares {
public:
  /* Type */
  using X_Type = X_Type_In;

protected:
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
  /* Type */
  using Value_Type = _T;

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

  /**
   * @brief Sets the regularization parameter (lambda) for the RLS algorithm.
   *
   * This function updates the lambda factor used in the RLS algorithm, which
   * controls the trade-off between fitting the data and regularization.
   *
   * @param lambda_in The new lambda factor to be set.
   */
  inline void set_lambda(const _T &lambda_in) {
    this->_lambda_factor = lambda_in;
    this->_lambda_factor_inv = static_cast<_T>(1) / lambda_in;
  }

  /**
   * @brief Updates the model with new input and true output values.
   *
   * This function performs an update step in the Recursive Least Squares
   * algorithm using the provided input (X) and true output (y_true). It
   * computes the new weights and updates the covariance matrix P.
   *
   * @param X The input data matrix of shape (ROWS, COLS).
   * @param y_true The true output value.
   */
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

  /**
   * @brief Predicts the output for a given input vector.
   *
   * This function computes the predicted output for a single input vector
   * using the learned weights. It adds a bias term to the input vector before
   * making the prediction.
   *
   * @param X The input vector of shape (1, COLS).
   * @return The predicted output value.
   */
  inline auto predict(const X_Type &X) const -> _T {

    auto bias_vector = PythonNumpy::make_DenseMatrixOnes<_T, 1, 1>();

    auto X_ex = PythonNumpy::concatenate_vertically(X, bias_vector);

    return PythonNumpy::ATranspose_mul_B(X_ex, this->_weights)
        .template get<0, 0>();
  }

  /**
   * @brief Predicts the output for a given input vector and returns the
   * weights.
   *
   * This function computes the predicted output for a single input vector
   * using the learned weights. It adds a bias term to the input vector before
   * making the prediction.
   *
   * @param X The input vector of shape (1, COLS).
   * @return The predicted output value.
   */
  inline auto get_weights(void) const -> _Wights_Type { return this->_weights; }

  /**
   * @brief Retrieves the covariance matrix P used in the RLS algorithm.
   *
   * This function returns the current covariance matrix P, which is used to
   * compute the Kalman gain and update the weights in the RLS algorithm.
   *
   * @return The covariance matrix P of shape (COLS + 1, COLS + 1).
   */
  inline void set_inv_solver_decay_rate(const _T &decay_rate_in) {
    this->_lambda_X_P_Solver.set_decay_rate(decay_rate_in);
  }

  /**
   * @brief Sets the minimum division value for the inverse solver in the RLS
   * algorithm.
   *
   * This function updates the minimum division value used in the inverse
   * solver to avoid division by zero errors during the RLS updates.
   *
   * @param division_min_in The new minimum division value to be set.
   */
  inline void set_inv_solver_division_min(const _T &division_min_in) {
    this->_lambda_X_P_Solver.set_division_min(division_min_in);
  }

protected:
  /* Variables */
  _T _lambda_factor;
  _T _lambda_factor_inv;
  _Wights_Type _weights;
  _P_Type _P;
};

/* make Recursive Least Squares */

/**
 * @brief Creates a RecursiveLeastSquares object with default settings.
 *
 * This function template initializes a RecursiveLeastSquares object with the
 * default lambda factor and delta value. It can be used for online regression
 * and system identification.
 *
 * @tparam X_Type The type of the input data, which must provide COLS.
 * @return RecursiveLeastSquares<X_Type> The resulting RecursiveLeastSquares
 * object.
 */
template <typename X_Type>
inline auto make_RecursiveLeastSquares(void) -> RecursiveLeastSquares<X_Type> {
  return RecursiveLeastSquares<X_Type>();
}

/**
 * @brief Creates a RecursiveLeastSquares object with a specified lambda factor.
 *
 * This function template initializes a RecursiveLeastSquares object with the
 * provided lambda factor and the default delta value. It can be used for online
 * regression and system identification.
 *
 * @tparam X_Type The type of the input data, which must provide COLS.
 * @param lambda_in The lambda factor to be set in the RLS algorithm.
 * @return RecursiveLeastSquares<X_Type> The resulting RecursiveLeastSquares
 * object.
 */
template <typename X_Type, typename T>
inline auto make_RecursiveLeastSquares(const T &lambda_in)
    -> RecursiveLeastSquares<X_Type> {
  return RecursiveLeastSquares<X_Type>(lambda_in);
}

/**
 * @brief Creates a RecursiveLeastSquares object with specified lambda and delta
 * values.
 *
 * This function template initializes a RecursiveLeastSquares object with the
 * provided lambda factor and delta value. It can be used for online regression
 * and system identification.
 *
 * @tparam X_Type The type of the input data, which must provide COLS.
 * @param lambda_in The lambda factor to be set in the RLS algorithm.
 * @param delta_in The delta value to be set in the RLS algorithm.
 * @return RecursiveLeastSquares<X_Type> The resulting RecursiveLeastSquares
 * object.
 */
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
