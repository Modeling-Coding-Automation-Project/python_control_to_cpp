/**
 * @file python_control_kalman_filter.hpp
 * @brief Kalman Filter implementations for PythonControl C++ library.
 *
 * This header provides C++ template implementations of Linear, Extended, and
 * Unscented Kalman Filters, supporting both standard and delayed-input systems.
 * It includes utility functions and types for Kalman filter weights, sigma
 * point calculations, and prediction/update operations. The code is designed
 * for high flexibility and type safety, leveraging template metaprogramming and
 * static assertions.
 */
#ifndef __PYTHON_CONTROL_KALMAN_FILTER_HPP__
#define __PYTHON_CONTROL_KALMAN_FILTER_HPP__

#include "python_control_state_space.hpp"
#include "python_math.hpp"
#include "python_numpy.hpp"

#include <functional>
#include <type_traits>

namespace PythonControl {

constexpr double KALMAN_FILTER_DIVISION_MIN = 1.0e-10;
constexpr std::size_t LKF_G_CONVERGE_REPEAT_MAX = 1000;

/* Kalman Filter Weight Type */
template <typename T, std::size_t Vector_Size>
using KalmanFilter_Q_Type = PythonNumpy::DiagMatrix_Type<T, Vector_Size>;

/* make Kalman Filter Weights  */

/**
 * @brief Creates a diagonal matrix for the process noise covariance (Q) used in
 * a Kalman Filter.
 *
 * This templated function constructs a diagonal matrix of size Vector_Size,
 * where the diagonal elements are initialized with the provided values. The
 * function is intended to facilitate the creation of the Q matrix for Kalman
 * Filter implementations.
 *
 * @tparam Vector_Size The size of the square matrix (number of states).
 * @tparam T The type of the matrix elements (e.g., float, double).
 * @tparam Args Variadic template parameter for additional values to fill the
 * diagonal.
 * @param value_1 The value for the first diagonal element.
 * @param args Additional values for the remaining diagonal elements.
 * @return KalmanFilter_Q_Type<T, Vector_Size> The resulting diagonal matrix of
 * type KalmanFilter_Q_Type.
 */
template <std::size_t Vector_Size, typename T, typename... Args>
inline auto make_KalmanFilter_Q(T value_1, Args... args)
    -> KalmanFilter_Q_Type<T, Vector_Size> {

  KalmanFilter_Q_Type<T, Vector_Size> input =
      PythonNumpy::make_DiagMatrix<Vector_Size>(value_1, args...);

  return input;
}

template <typename T, std::size_t Vector_Size>
using KalmanFilter_R_Type = PythonNumpy::DiagMatrix_Type<T, Vector_Size>;

/* make Kalman Filter Weights  */

/**
 * @brief Creates a diagonal matrix for the measurement noise covariance (R)
 * used in a Kalman Filter.
 *
 * This templated function constructs a diagonal matrix of size Vector_Size,
 * where the diagonal elements are initialized with the provided values. The
 * function is intended to facilitate the creation of the R matrix for Kalman
 * Filter implementations.
 *
 * @tparam Vector_Size The size of the square matrix (number of measurements).
 * @tparam T The type of the matrix elements (e.g., float, double).
 * @tparam Args Variadic template parameter for additional values to fill the
 * diagonal.
 * @param value_1 The value for the first diagonal element.
 * @param args Additional values for the remaining diagonal elements.
 * @return KalmanFilter_R_Type<T, Vector_Size> The resulting diagonal matrix of
 * type KalmanFilter_R_Type.
 */
template <std::size_t Vector_Size, typename T, typename... Args>
inline auto make_KalmanFilter_R(T value_1, Args... args)
    -> KalmanFilter_R_Type<T, Vector_Size> {

  KalmanFilter_R_Type<T, Vector_Size> input =
      PythonNumpy::make_DiagMatrix<Vector_Size>(value_1, args...);

  return input;
}

/* Unscented Kalman Filter Operation */
namespace UKF_Operation {

template <typename T, typename W_Type, std::size_t Index, std::size_t Rest>
struct SetRestOfW_Loop {
  /**
   * @brief Sets the specified weight value at the given index in the weight
   * matrix W, and recursively sets the same value for the remaining indices.
   *
   * @tparam T         The type of the weight value to set.
   * @tparam W_Type    The type of the weight matrix.
   * @tparam Index     The current index in the recursion.
   * @tparam Rest      The number of remaining indices to set.
   * @param W          Reference to the weight matrix to modify.
   * @param weight_to_set The weight value to assign at the current index.
   *
   * This function uses template recursion to set the specified weight value
   * at each index in the weight matrix W, starting from Index and continuing
   * for Rest elements.
   */
  static inline void set(W_Type &W, const T &weight_to_set) {
    W.template set<Index, Index>(weight_to_set);
    SetRestOfW_Loop<T, W_Type, Index + 1, Rest - 1>::set(W, weight_to_set);
  }
};

template <typename T, typename W_Type, std::size_t Index>
struct SetRestOfW_Loop<T, W_Type, Index, 0> {
  /**
   * @brief Base case for the recursive setting of weights in the weight matrix.
   *
   * @tparam T         The type of the weight value to set.
   * @tparam W_Type    The type of the weight matrix.
   * @tparam Index     The current index in the recursion.
   * @param W          Reference to the weight matrix to modify.
   * @param weight_to_set The weight value to assign at the current index.
   *
   * This function does nothing, serving as a base case for recursion.
   */
  static inline void set(W_Type &W, const T &weight_to_set) {
    W.template set<Index, Index>(weight_to_set);
  }
};

/**
 * @brief Sets the remaining weights in the weight matrix W starting from a
 * specified index.
 *
 * This alias template simplifies the process of setting the remaining weights
 * in the weight matrix W, starting from a given index and continuing to the
 * end of the matrix.
 *
 * @tparam T         The type of the weight value to set.
 * @tparam W_Type    The type of the weight matrix.
 * @tparam Start_Index The index from which to start setting weights.
 */
template <typename T, typename W_Type, std::size_t Start_Index>
using SetRestOfW =
    SetRestOfW_Loop<T, W_Type, Start_Index, (W_Type::COLS - 1 - Start_Index)>;

/* update sigma point matrix */
template <typename T, typename Kai_Type, typename X_Type, typename SP_Type,
          std::size_t Index, std::size_t End_Index>
struct UpdateSigmaPointMatrix_Loop {
  /**
   * @brief Updates the sigma point matrix Kai by setting the rows based on the
   * state vector X and the sigma points SP, scaled by a specified weight.
   *
   * @tparam T         The type of the weight to apply to the sigma points.
   * @tparam Kai_Type  The type of the sigma point matrix to update.
   * @tparam X_Type    The type of the state vector.
   * @tparam SP_Type   The type of the sigma points matrix.
   * @tparam Index     The current index in the recursion.
   * @tparam End_Index The end index for recursion.
   * @param Kai        Reference to the sigma point matrix to modify.
   * @param X          The state vector used for updating the sigma points.
   * @param SP         The sigma points matrix used for updating.
   * @param sigma_point_weight The weight applied to the sigma points.
   *
   * This function uses template recursion to set two rows in the Kai matrix
   * for each sigma point, one for the positive and one for the negative
   * weighted sigma point, and continues until all indices are processed.
   */
  static inline void set(Kai_Type &Kai, const X_Type &X, const SP_Type &SP,
                         const T &sigma_point_weight) {

    PythonNumpy::set_row<(Index + 1)>(
        Kai, X + sigma_point_weight * PythonNumpy::get_row<Index>(SP));
    PythonNumpy::set_row<(Index + X_Type::COLS + 1)>(
        Kai, X - sigma_point_weight * PythonNumpy::get_row<Index>(SP));

    UpdateSigmaPointMatrix_Loop<T, Kai_Type, X_Type, SP_Type, (Index + 1),
                                (End_Index - 1)>::set(Kai, X, SP,
                                                      sigma_point_weight);
  }
};

template <typename T, typename Kai_Type, typename X_Type, typename SP_Type,
          std::size_t Index>
struct UpdateSigmaPointMatrix_Loop<T, Kai_Type, X_Type, SP_Type, Index, 0> {
  /**
   * @brief Base case for the recursive update of the sigma point matrix.
   *
   * @tparam T         The type of the weight to apply to the sigma points.
   * @tparam Kai_Type  The type of the sigma point matrix to update.
   * @tparam X_Type    The type of the state vector.
   * @tparam SP_Type   The type of the sigma points matrix.
   * @tparam Index     The current index in the recursion.
   * @param Kai        Reference to the sigma point matrix to modify.
   * @param X          The state vector used for updating the sigma points.
   * @param SP         The sigma points matrix used for updating.
   * @param sigma_point_weight The weight applied to the sigma points.
   *
   * This function does nothing, serving as a base case for recursion.
   */
  static inline void set(Kai_Type &Kai, const X_Type &X, const SP_Type &SP,
                         const T &sigma_point_weight) {
    // Do nothing.
    static_cast<void>(Kai);
    static_cast<void>(X);
    static_cast<void>(SP);
    static_cast<void>(sigma_point_weight);
  }
};

/**
 * @brief Updates the sigma point matrix Kai based on the state vector X and
 * the sigma points SP, scaled by a specified weight.
 *
 * This alias template simplifies the process of updating the sigma point
 * matrix Kai, setting rows for both positive and negative weighted sigma
 * points.
 *
 * @tparam T         The type of the weight to apply to the sigma points.
 * @tparam Kai_Type  The type of the sigma point matrix to update.
 * @tparam X_Type    The type of the state vector.
 * @tparam SP_Type   The type of the sigma points matrix.
 */
template <typename T, typename Kai_Type, typename X_Type, typename SP_Type>
using UpdateSigmaPointMatrix =
    UpdateSigmaPointMatrix_Loop<T, Kai_Type, X_Type, SP_Type, 0, X_Type::COLS>;

/* calc state function with each sigma points */

template <typename Kai_Type, typename StateFunction_Object, typename U_Type,
          typename Parameter_Type, std::size_t Index>
struct StateFunctionEachSigmaPoints_Loop {
  /**
   * @brief Computes the state function for each sigma point in the Kai matrix
   * and updates the corresponding row in the Kai matrix.
   *
   * @tparam Kai_Type The type of the sigma point matrix.
   * @tparam StateFunction_Object The type of the state function object.
   * @tparam U_Type The type of the input vector.
   * @tparam Parameter_Type The type of the parameters for the state function.
   * @tparam Index The current index in the recursion.
   * @param Kai Reference to the sigma point matrix to modify.
   * @param state_function The state function object used for computation.
   * @param U The input vector used in the state function.
   * @param parameters The parameters for the state function.
   *
   * This function uses template recursion to compute the state function for
   * each sigma point and updates the corresponding row in the Kai matrix,
   * continuing until all indices are processed.
   */
  static inline void compute(Kai_Type &Kai,
                             const StateFunction_Object &state_function,
                             const U_Type &U,
                             const Parameter_Type &parameters) {

    PythonNumpy::set_row<Index>(
        Kai, state_function(PythonNumpy::get_row<Index>(Kai), U, parameters));

    StateFunctionEachSigmaPoints_Loop<Kai_Type, StateFunction_Object, U_Type,
                                      Parameter_Type,
                                      (Index - 1)>::compute(Kai, state_function,
                                                            U, parameters);
  }
};

template <typename Kai_Type, typename StateFunction_Object, typename U_Type,
          typename Parameter_Type>
struct StateFunctionEachSigmaPoints_Loop<Kai_Type, StateFunction_Object, U_Type,
                                         Parameter_Type, 0> {
  /**
   * @brief Base case for the recursive computation of the state function for
   * each sigma point.
   *
   * @param Kai Reference to the sigma point matrix to modify.
   * @param state_function The state function object used for computation.
   * @param U The input vector used in the state function.
   * @param parameters The parameters for the state function.
   * *
   * This function computes the state function for the first sigma point and
   * updates the corresponding row in the Kai matrix.
   * */
  static inline void compute(Kai_Type &Kai,
                             const StateFunction_Object &state_function,
                             const U_Type &U,
                             const Parameter_Type &parameters) {

    PythonNumpy::set_row<0>(
        Kai, state_function(PythonNumpy::get_row<0>(Kai), U, parameters));
  }
};

/**
 * @brief Computes the state function for each sigma point in the Kai matrix
 * and updates the corresponding rows.
 *
 * This alias template simplifies the process of computing the state function
 * for each sigma point, iterating through all indices in the Kai matrix.
 *
 * @tparam Kai_Type The type of the sigma point matrix.
 * @tparam StateFunction_Object The type of the state function object.
 * @tparam U_Type The type of the input vector.
 * @tparam Parameter_Type The type of the parameters for the state function.
 */
template <typename Kai_Type, typename StateFunction_Object, typename U_Type,
          typename Parameter_Type>
using StateFunctionEachSigmaPoints =
    StateFunctionEachSigmaPoints_Loop<Kai_Type, StateFunction_Object, U_Type,
                                      Parameter_Type, (Kai_Type::ROWS - 1)>;

/* average sigma points */
template <typename X_Type, typename W_Type, typename Kai_Type,
          std::size_t Index>
struct AverageSigmaPoints_Loop {
  /**
   * @brief Computes the average of sigma points weighted by the corresponding
   * weights and updates the state vector X_hat.
   *
   * @tparam X_Type The type of the state vector to update.
   * @tparam W_Type The type of the weight matrix.
   * @tparam Kai_Type The type of the sigma point matrix.
   * @tparam Index The current index in the recursion.
   * @param X_hat Reference to the state vector to update.
   * @param W The weight matrix used for averaging.
   * @param Kai The sigma point matrix containing the sigma points.
   *
   * This function uses template recursion to compute the weighted average of
   * sigma points and updates the state vector X_hat, continuing until all
   * indices are processed.
   */
  static inline void compute(X_Type &X_hat, const W_Type &W,
                             const Kai_Type &Kai) {
    X_hat = X_hat +
            W.template get<Index, Index>() * PythonNumpy::get_row<Index>(Kai);

    AverageSigmaPoints_Loop<X_Type, W_Type, Kai_Type, (Index - 1)>::compute(
        X_hat, W, Kai);
  }
};

template <typename X_Type, typename W_Type, typename Kai_Type>
struct AverageSigmaPoints_Loop<X_Type, W_Type, Kai_Type, 1> {
  /**
   * @brief Base case for the recursive averaging of sigma points.
   *
   * @tparam X_Type The type of the state vector to update.
   * @tparam W_Type The type of the weight matrix.
   * @tparam Kai_Type The type of the sigma point matrix.
   * @param X_hat Reference to the state vector to update.
   * @param W The weight matrix used for averaging.
   * @param Kai The sigma point matrix containing the sigma points.
   *
   * This function computes the weighted average for the last sigma point and
   * updates the state vector X_hat.
   */
  static inline void compute(X_Type &X_hat, const W_Type &W,
                             const Kai_Type &Kai) {
    X_hat = X_hat + W.template get<1, 1>() * PythonNumpy::get_row<1>(Kai);
  }
};

template <typename X_Type, typename W_Type, typename Kai_Type>
struct AverageSigmaPoints_Loop<X_Type, W_Type, Kai_Type, 0> {
  /**
   * @brief Base case for the recursive averaging of sigma points when no
   * sigma points are left to process.
   *
   * @param X_Type The type of the state vector to update.
   * @param W_Type The type of the weight matrix.
   * @param Kai_Type The type of the sigma point matrix.
   * @param X_hat Reference to the state vector to update.
   * @param W The weight matrix used for averaging.
   * @param Kai The sigma point matrix containing the sigma points.
   * *
   * This function does nothing, serving as a base case for recursion.
   */
  static inline void compute(X_Type &X_hat, const W_Type &W,
                             const Kai_Type &Kai) {
    // Do nothing.
    static_cast<void>(X_hat);
    static_cast<void>(W);
    static_cast<void>(Kai);
  }
};

/**
 * @brief Computes the average of sigma points weighted by the corresponding
 * weights and updates the state vector X_hat.
 *
 * This alias template simplifies the process of averaging sigma points,
 * iterating through all indices in the Kai matrix.
 *
 * @tparam X_Type The type of the state vector to update.
 * @tparam W_Type The type of the weight matrix.
 * @tparam Kai_Type The type of the sigma point matrix.
 */
template <typename X_Type, typename W_Type, typename Kai_Type>
using AverageSigmaPoints =
    AverageSigmaPoints_Loop<X_Type, W_Type, Kai_Type, (Kai_Type::ROWS - 1)>;

/* calc covariance matrix */
template <typename Kai_Type, typename X_Type, std::size_t Index>
struct SigmaPointsCovariance_Loop {
  /**
   * @brief Computes the covariance matrix of the sigma points by subtracting
   * the estimated state vector X_hat from each sigma point and storing the
   * result in the output matrix X_d.
   *
   * @tparam Kai_Type The type of the sigma point matrix.
   * @tparam X_Type The type of the estimated state vector.
   * @tparam Index The current index in the recursion.
   * @param X_d Reference to the output matrix where the covariance is stored.
   * @param Kai The sigma point matrix containing the sigma points.
   * @param X_hat The estimated state vector to subtract from each sigma point.
   *
   * This function uses template recursion to compute the covariance for each
   * sigma point and continues until all indices are processed.
   */
  static inline void compute(Kai_Type &X_d, const Kai_Type &Kai,
                             const X_Type &X_hat) {

    PythonNumpy::set_row<Index>(X_d, PythonNumpy::get_row<Index>(Kai) - X_hat);

    SigmaPointsCovariance_Loop<Kai_Type, X_Type, (Index - 1)>::compute(X_d, Kai,
                                                                       X_hat);
  }
};

template <typename Kai_Type, typename X_Type>
struct SigmaPointsCovariance_Loop<Kai_Type, X_Type, 0> {
  /**
   * @brief Base case for the recursive computation of the covariance matrix
   * of the sigma points.
   *
   * @tparam Kai_Type The type of the sigma point matrix.
   * @tparam X_Type The type of the estimated state vector.
   * @param X_d Reference to the output matrix where the covariance is stored.
   * @param Kai The sigma point matrix containing the sigma points.
   * @param X_hat The estimated state vector to subtract from each sigma point.
   *
   * This function computes the covariance for the first sigma point and
   * updates the corresponding row in the output matrix X_d.
   */
  static inline void compute(Kai_Type &X_d, const Kai_Type &Kai,
                             const X_Type &X_hat) {
    PythonNumpy::set_row<0>(X_d, PythonNumpy::get_row<0>(Kai) - X_hat);
  }
};

/**
 * @brief Computes the covariance matrix of the sigma points by subtracting
 * the estimated state vector X_hat from each sigma point and storing the
 * result in the output matrix X_d.
 *
 * This alias template simplifies the process of computing the covariance
 * matrix, iterating through all indices in the Kai matrix.
 *
 * @tparam Kai_Type The type of the sigma point matrix.
 * @tparam X_Type The type of the estimated state vector.
 */
template <typename Kai_Type, typename X_Type>
using SigmaPointsCovariance =
    SigmaPointsCovariance_Loop<Kai_Type, X_Type, (Kai_Type::ROWS - 1)>;

/* calc measurement function with each sigma points */
template <typename Nu_Type, typename Kai_Type,
          typename MeasurementFunction_Object, typename Parameter_Type,
          std::size_t Index>
struct MeasurementFunctionEachSigmaPoints_Loop {
  /**
   * @brief Computes the measurement function for each sigma point in the Kai
   * matrix and updates the corresponding row in the Nu matrix.
   *
   * @tparam Nu_Type The type of the measurement output matrix.
   * @tparam Kai_Type The type of the sigma point matrix.
   * @tparam MeasurementFunction_Object The type of the measurement function
   * object.
   * @tparam Parameter_Type The type of the parameters for the measurement
   * function.
   * @tparam Index The current index in the recursion.
   * @param Nu Reference to the measurement output matrix to modify.
   * @param Kai The sigma point matrix containing the sigma points.
   * @param measurement_function The measurement function object used for
   * computation.
   * @param parameters The parameters for the measurement function.
   *
   * This function uses template recursion to compute the measurement function
   * for each sigma point and updates the corresponding row in the Nu matrix,
   * continuing until all indices are processed.
   */
  static inline void
  compute(Nu_Type &Nu, Kai_Type &Kai,
          const MeasurementFunction_Object &measurement_function,
          const Parameter_Type &parameters) {

    PythonNumpy::set_row<Index>(
        Nu, measurement_function(PythonNumpy::get_row<Index>(Kai), parameters));

    MeasurementFunctionEachSigmaPoints_Loop<
        Nu_Type, Kai_Type, MeasurementFunction_Object, Parameter_Type,
        (Index - 1)>::compute(Nu, Kai, measurement_function, parameters);
  }
};

template <typename Nu_Type, typename Kai_Type,
          typename MeasurementFunction_Object, typename Parameter_Type>
struct MeasurementFunctionEachSigmaPoints_Loop<
    Nu_Type, Kai_Type, MeasurementFunction_Object, Parameter_Type, 0> {
  /**
   * @brief Base case for the recursive computation of the measurement
   * function for each sigma point.
   *
   * @param Nu Reference to the measurement output matrix to modify.
   * @param Kai The sigma point matrix containing the sigma points.
   * @param measurement_function The measurement function object used for
   * computation.
   * @param parameters The parameters for the measurement function.
   * *
   * This function computes the measurement function for the first sigma
   * point and updates the corresponding row in the Nu matrix.
   */
  static inline void
  compute(Nu_Type &Nu, Kai_Type &Kai,
          const MeasurementFunction_Object &measurement_function,
          const Parameter_Type &parameters) {

    PythonNumpy::set_row<0>(
        Nu, measurement_function(PythonNumpy::get_row<0>(Kai), parameters));
  }
};

/**
 * @brief Computes the measurement function for each sigma point in the Kai
 * matrix and updates the corresponding rows in the Nu matrix.
 *
 * This alias template simplifies the process of computing the measurement
 * function for each sigma point, iterating through all indices in the Kai
 * matrix.
 *
 * @tparam Nu_Type The type of the measurement output matrix.
 * @tparam Kai_Type The type of the sigma point matrix.
 * @tparam MeasurementFunction_Object The type of the measurement function
 * object.
 * @tparam Parameter_Type The type of the parameters for the measurement
 * function.
 */
template <typename Nu_Type, typename Kai_Type,
          typename MeasurementFunction_Object, typename Parameter_Type>
using MeasurementFunctionEachSigmaPoints =
    MeasurementFunctionEachSigmaPoints_Loop<
        Nu_Type, Kai_Type, MeasurementFunction_Object, Parameter_Type,
        (Kai_Type::ROWS - 1)>;

} // namespace UKF_Operation

namespace PredictOperation {

/* predict for Linear Kalman Filter */
template <std::size_t NumberOfDelay> struct Linear {
  /**
   * @brief Predicts the next state and updates the covariance matrix for a
   * Linear Kalman Filter.
   * @param state_space The discrete state space object containing the system
   * dynamics.
   * @param P The covariance matrix to be updated.
   * @param Q The process noise covariance matrix.
   * @param U_store The input store containing the control inputs.
   * @param _input_count The current input count, which determines if the
   * prediction should be executed or delayed.
   */
  template <typename DiscreteStateSpace_Type, typename P_Type, typename Q_Type,
            typename U_Store_Type>
  static void execute(DiscreteStateSpace_Type &state_space, P_Type &P,
                      const Q_Type &Q, const U_Store_Type &U_store,
                      std::size_t &_input_count) {

    if (_input_count < NumberOfDelay) {
      _input_count++;
    } else {
      state_space.X =
          state_space.A * state_space.X + state_space.B * U_store.get();
      P = state_space.A * PythonNumpy::A_mul_BTranspose(P, state_space.A) + Q;
    }
  }
};

template <> struct Linear<0> {
  /**
   * @brief Predicts the next state and updates the covariance matrix for a
   * Linear Kalman Filter without delay.
   * @param state_space The discrete state space object containing the system
   * dynamics.
   * @param P The covariance matrix to be updated.
   * @param Q The process noise covariance matrix.
   * @param U_store The input store containing the control inputs.
   * @param _input_count The current input count, which is ignored in this case.
   */
  template <typename DiscreteStateSpace_Type, typename P_Type, typename Q_Type,
            typename U_Store_Type>
  static void execute(DiscreteStateSpace_Type &state_space, P_Type &P,
                      const Q_Type &Q, const U_Store_Type &U_store,
                      std::size_t &_input_count) {
    static_cast<void>(_input_count);

    state_space.X =
        state_space.A * state_space.X + state_space.B * U_store.get();
    P = state_space.A * PythonNumpy::A_mul_BTranspose(P, state_space.A) + Q;
  }
};

/* predict for Extended Kalman Filter */
template <std::size_t NumberOfDelay> struct Extended {
  /**
   * @brief Predicts the next state and updates the covariance matrix for an
   * Extended Kalman Filter.
   * @param state_function_jacobian The Jacobian of the state function.
   * @param state_function The state function object.
   * @param A The Jacobian matrix to be updated.
   * @param P The covariance matrix to be updated.
   * @param Q The process noise covariance matrix.
   * @param U_store The input store containing the control inputs.
   * @param X_hat The estimated state vector to be updated.
   * @param parameters Additional parameters for the state function.
   * @param input_count The current input count, which determines if the
   * prediction should be executed or delayed.
   */
  template <typename StateFunction_Object,
            typename StateFunction_Jacobian_Object, typename A_Type,
            typename P_Type, typename Q_Type, typename U_Store_Type,
            typename X_Type, typename Parameter_Type>
  static void
  execute(const StateFunction_Jacobian_Object &state_function_jacobian,
          const StateFunction_Object &state_function, A_Type &A, P_Type &P,
          const Q_Type &Q, const U_Store_Type &U_store, X_Type &X_hat,
          const Parameter_Type parameters, std::size_t &input_count) {

    if (input_count < NumberOfDelay) {
      input_count++;
    } else {
      A = state_function_jacobian(X_hat, U_store.get(), parameters);

      X_hat = state_function(X_hat, U_store.get(), parameters);
      P = A * PythonNumpy::A_mul_BTranspose(P, A) + Q;
    }
  }
};

template <> struct Extended<0> {
  /**
   * @brief Predicts the next state and updates the covariance matrix for an
   * Extended Kalman Filter without delay.
   * @param state_function_jacobian The Jacobian of the state function.
   * @param state_function The state function object.
   * @param A The Jacobian matrix to be updated.
   * @param P The covariance matrix to be updated.
   * @param Q The process noise covariance matrix.
   * @param U_store The input store containing the control inputs.
   * @param X_hat The estimated state vector to be updated.
   * @param parameters Additional parameters for the state function.
   * @param input_count The current input count, which is ignored in this case.
   */
  template <typename StateFunction_Object,
            typename StateFunction_Jacobian_Object, typename A_Type,
            typename P_Type, typename Q_Type, typename U_Store_Type,
            typename X_Type, typename Parameter_Type>
  static void
  execute(const StateFunction_Jacobian_Object &state_function_jacobian,
          const StateFunction_Object &state_function, A_Type &A, P_Type &P,
          const Q_Type &Q, const U_Store_Type &U_store, X_Type &X_hat,
          const Parameter_Type parameters, std::size_t &input_count) {
    static_cast<void>(input_count);

    A = state_function_jacobian(X_hat, U_store.get(), parameters);

    X_hat = state_function(X_hat, U_store.get(), parameters);
    P = A * PythonNumpy::A_mul_BTranspose(P, A) + Q;
  }
};

/* predict for Unscented Kalman Filter */
template <std::size_t NumberOfDelay> struct Unscented {
  /**
   * @brief Predicts the next state and updates the covariance matrix for an
   * Unscented Kalman Filter.
   * @param state_function The state function object.
   * @param sigma_points_calculator The sigma points calculator object.
   * @param X_hat The estimated state vector to be updated.
   * @param U_store The input store containing the control inputs.
   * @param P The covariance matrix to be updated.
   * @param w_m The mean weight for the sigma points.
   * @param W The weight matrix for the sigma points.
   * @param Q The process noise covariance matrix.
   * @param X_d The output matrix for the sigma points covariance.
   * @param parameters Additional parameters for the state function.
   * @param input_count The current input count, which determines if the
   * prediction should be executed or delayed.
   */
  template <typename StateFunction_Object, typename SigmaPointsCalculator_Type,
            typename X_Type, typename U_Store_Type, typename P_Type, typename T,
            typename W_Type, typename Q_Type, typename Kai_Type,
            typename Parameter_Type>
  static void execute(StateFunction_Object &state_function,
                      SigmaPointsCalculator_Type &sigma_points_calculator,
                      X_Type &X_hat, const U_Store_Type &U_store, P_Type &P,
                      const T &w_m, const W_Type &W, const Q_Type &Q,
                      Kai_Type &X_d, const Parameter_Type &parameters,
                      std::size_t &input_count) {
    if (input_count < NumberOfDelay) {
      input_count++;
    } else {
      auto Kai = sigma_points_calculator.calculate(X_hat, P);

      UKF_Operation::StateFunctionEachSigmaPoints<
          Kai_Type, StateFunction_Object,
          typename U_Store_Type::Original_Vector_Type,
          Parameter_Type>::compute(Kai, state_function, U_store.get(),
                                   parameters);

      X_hat = w_m * PythonNumpy::get_row<0>(Kai);
      UKF_Operation::AverageSigmaPoints<X_Type, W_Type, Kai_Type>::compute(
          X_hat, W, Kai);

      UKF_Operation::SigmaPointsCovariance<Kai_Type, X_Type>::compute(X_d, Kai,
                                                                      X_hat);

      P = X_d * PythonNumpy::A_mul_BTranspose(W, X_d) + Q;
    }
  }
};

template <> struct Unscented<0> {
  /**
   * @brief Predicts the next state and updates the covariance matrix for an
   * Unscented Kalman Filter without delay.
   * @param state_function The state function object.
   * @param sigma_points_calculator The sigma points calculator object.
   * @param X_hat The estimated state vector to be updated.
   * @param U_store The input store containing the control inputs.
   * @param P The covariance matrix to be updated.
   * @param w_m The mean weight for the sigma points.
   * @param W The weight matrix for the sigma points.
   * @param Q The process noise covariance matrix.
   * @param X_d The output matrix for the sigma points covariance.
   * @param parameters Additional parameters for the state function.
   * @param input_count The current input count, which is ignored in this case.
   */
  template <typename StateFunction_Object, typename SigmaPointsCalculator_Type,
            typename X_Type, typename U_Store_Type, typename P_Type, typename T,
            typename W_Type, typename Q_Type, typename Kai_Type,
            typename Parameter_Type>
  static void execute(StateFunction_Object &state_function,
                      SigmaPointsCalculator_Type &sigma_points_calculator,
                      X_Type &X_hat, const U_Store_Type &U_store, P_Type &P,
                      const T &w_m, const W_Type &W, const Q_Type &Q,
                      Kai_Type &X_d, const Parameter_Type &parameters,
                      std::size_t &input_count) {

    static_cast<void>(input_count);

    auto Kai = sigma_points_calculator.calculate(X_hat, P);

    UKF_Operation::StateFunctionEachSigmaPoints<
        Kai_Type, StateFunction_Object,
        typename U_Store_Type::Original_Vector_Type,
        Parameter_Type>::compute(Kai, state_function, U_store.get(),
                                 parameters);

    X_hat = w_m * PythonNumpy::get_row<0>(Kai);
    UKF_Operation::AverageSigmaPoints<X_Type, W_Type, Kai_Type>::compute(
        X_hat, W, Kai);

    UKF_Operation::SigmaPointsCovariance<Kai_Type, X_Type>::compute(X_d, Kai,
                                                                    X_hat);

    P = X_d * PythonNumpy::A_mul_BTranspose(W, X_d) + Q;
  }
};

} // namespace PredictOperation

namespace GetXHatWithoutDelayOperation {

/* Get x_hat without delay for Linear Kalman Filter */
template <std::size_t NumberOfDelay> struct Linear {
  /**
   * @brief Computes the estimated state vector x_hat without delay for a Linear
   * Kalman Filter.
   * @param state_space The discrete state space object containing the system
   * dynamics.
   * @param input_count The number of inputs to consider in the prediction.
   * @return The estimated state vector x_hat.
   */
  template <typename DiscreteStateSpace_Type>
  static auto compute(const DiscreteStateSpace_Type &state_space,
                      const std::size_t &input_count) ->
      typename DiscreteStateSpace_Type::Original_X_Type {
    auto x_hat = state_space.get_X();
    std::size_t delay_index = state_space.U.get_delay_ring_buffer_index() +
                              NumberOfDelay - input_count;

    for (std::size_t i = 0; i < input_count; i++) {
      delay_index++;
      if (delay_index > NumberOfDelay) {
        delay_index = delay_index - NumberOfDelay - 1;
      }

      x_hat = state_space.A * x_hat +
              state_space.B * state_space.U.get_by_index(delay_index);
    }

    return x_hat;
  }
};

template <> struct Linear<0> {
  /**
   * @brief Computes the estimated state vector x_hat without delay for a Linear
   * Kalman Filter without considering input count.
   * @param state_space The discrete state space object containing the system
   * dynamics.
   * @return The estimated state vector x_hat.
   */
  template <typename DiscreteStateSpace_Type>
  static auto compute(const DiscreteStateSpace_Type &state_space) ->
      typename DiscreteStateSpace_Type::Original_X_Type {
    return state_space.get_X();
  }
};

/* Get x_hat without delay for Extended Kalman Filter */
template <std::size_t NumberOfDelay> struct Extended {
  /**
   * @brief Computes the estimated state vector x_hat without delay for an
   * Extended Kalman Filter.
   * @param state_function The state function object.
   * @param X_hat_in The initial estimated state vector.
   * @param U_store The input store containing the control inputs.
   * @param parameters Additional parameters for the state function.
   * @param input_count The number of inputs to consider in the prediction.
   * @return The estimated state vector x_hat.
   */
  template <typename StateFunction_Object, typename X_Type,
            typename U_Store_Type, typename Parameter_Type>
  static auto compute(StateFunction_Object &state_function,
                      const X_Type &X_hat_in, const U_Store_Type &U_store,
                      const Parameter_Type &parameters,
                      const std::size_t &input_count) -> X_Type {
    auto X_hat = X_hat_in;
    std::size_t delay_index =
        U_store.get_delay_ring_buffer_index() + NumberOfDelay - input_count;

    for (std::size_t i = 0; i < input_count; i++) {
      delay_index++;
      if (delay_index > NumberOfDelay) {
        delay_index = delay_index - NumberOfDelay - 1;
      }

      X_hat =
          state_function(X_hat, U_store.get_by_index(delay_index), parameters);
    }

    return X_hat;
  }
};

template <> struct Extended<0> {
  /**
   * @brief Computes the estimated state vector x_hat without delay for an
   * Extended Kalman Filter without considering input count.
   * @param state_function The state function object.
   * @param X_hat The initial estimated state vector.
   * @param U_store The input store containing the control inputs.
   * @param parameters Additional parameters for the state function.
   * @param input_count The number of inputs to consider in the prediction.
   * @return The estimated state vector x_hat.
   */
  template <typename StateFunction_Object, typename X_Type,
            typename U_Store_Type, typename Parameter_Type>
  static auto compute(StateFunction_Object &state_function, const X_Type &X_hat,
                      const U_Store_Type &U_store,
                      const Parameter_Type &parameters,
                      const std::size_t &input_count) -> X_Type {
    static_cast<void>(state_function);
    static_cast<void>(U_store);
    static_cast<void>(parameters);
    static_cast<void>(input_count);

    return X_hat;
  }
};

} // namespace GetXHatWithoutDelayOperation

/**
 * @brief Linear Kalman Filter class template.
 *
 * This class implements a linear Kalman filter for discrete state spaces.
 * It supports prediction and update operations, and it is designed to work
 * with diagonal covariance matrices for process noise (Q) and measurement
 * noise (R).
 *
 * @tparam DiscreteStateSpace_Type_In The type of the discrete state space.
 * @tparam Q_Type_In The type of the process noise covariance matrix.
 * @tparam R_Type_In The type of the measurement noise covariance matrix.
 */
template <typename DiscreteStateSpace_Type_In, typename Q_Type_In,
          typename R_Type_In>
class LinearKalmanFilter {
public:
  /* Type */
  using DiscreteStateSpace_Type = DiscreteStateSpace_Type_In;
  using Q_Type = Q_Type_In;
  using R_Type = R_Type_In;

protected:
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
      PythonNumpy::DenseMatrix_Type<_T, _OUTPUT_SIZE, _OUTPUT_SIZE>>;

public:
  /* Type  */
  using Value_Type = _T;

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
  LinearKalmanFilter()
      : state_space(), Q(), R(), P(), G(), _C_P_CT_R_inv_solver(),
        _input_count(static_cast<std::size_t>(0)) {}

  LinearKalmanFilter(const DiscreteStateSpace_Type &DiscreteStateSpace,
                     const Q_Type &Q, const R_Type &R)
      : state_space(DiscreteStateSpace), Q(Q), R(R),
        P(PythonNumpy::make_DiagMatrixIdentity<_T, _STATE_SIZE>()
              .create_dense()),
        G(), _C_P_CT_R_inv_solver(), _input_count(static_cast<std::size_t>(0)) {
  }

  /* Copy Constructor */
  LinearKalmanFilter(
      const LinearKalmanFilter<DiscreteStateSpace_Type, Q_Type, R_Type> &input)
      : state_space(input.state_space), Q(input.Q), R(input.R), P(input.P),
        G(input.G), _C_P_CT_R_inv_solver(input._C_P_CT_R_inv_solver),
        _input_count(input._input_count) {}

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
      this->_input_count = input._input_count;
    }
    return *this;
  }

  /* Move Constructor */
  LinearKalmanFilter(LinearKalmanFilter<DiscreteStateSpace_Type, Q_Type, R_Type>
                         &&input) noexcept
      : state_space(std::move(input.state_space)), Q(std::move(input.Q)),
        R(std::move(input.R)), P(std::move(input.P)), G(std::move(input.G)),
        _C_P_CT_R_inv_solver(std::move(input._C_P_CT_R_inv_solver)),
        _input_count(input._input_count) {}

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
      this->_input_count = input._input_count;
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Predicts the next state and updates the covariance matrix.
   *
   * This function performs the prediction step of the Kalman filter, updating
   * the state vector and covariance matrix based on the current input.
   *
   * @param U The control input to be applied for prediction.
   */
  inline void
  predict(const typename DiscreteStateSpace_Type::Original_U_Type &U) {

    this->state_space.U.push(U);

    PredictOperation::Linear<NUMBER_OF_DELAY>::execute(
        this->state_space, this->P, this->Q, this->state_space.U,
        this->_input_count);
  }

  /**
   * @brief Updates the state estimate and covariance matrix based on the
   * measurement.
   *
   * This function performs the update step of the Kalman filter, adjusting
   * the state vector and covariance matrix based on the observed measurement.
   *
   * @param Y The observed measurement to be used for updating the state.
   */
  inline void
  update(const typename DiscreteStateSpace_Type::Original_Y_Type &Y) {

    auto P_CT = PythonNumpy::A_mul_BTranspose(this->P, this->state_space.C);

    auto C_P_CT_R = this->state_space.C * P_CT + this->R;
    this->_C_P_CT_R_inv_solver.inv(C_P_CT_R);

    this->G = P_CT * this->_C_P_CT_R_inv_solver.get_answer();

    this->state_space.X =
        this->state_space.X +
        this->G * (Y - this->state_space.C * this->state_space.X);

    this->P = (PythonNumpy::make_DiagMatrixIdentity<_T, _STATE_SIZE>() -
               this->G * this->state_space.C) *
              this->P;
  }

  /**
   * @brief Predicts the next state and updates the state estimate based on the
   * measurement.
   *
   * This function combines the prediction and update steps of the Kalman filter
   * into a single operation.
   *
   * @param U The control input to be applied for prediction.
   * @param Y The observed measurement to be used for updating the state.
   */
  inline void predict_and_update(
      const typename DiscreteStateSpace_Type::Original_U_Type &U,
      const typename DiscreteStateSpace_Type::Original_Y_Type &Y) {

    this->predict(U);
    this->update(Y);
  }

  // If G is known, you can use below "_fixed_G" functions.

  /**
   * @brief Predicts the next state with a fixed G matrix.
   *
   * This function performs the prediction step of the Kalman filter using a
   * pre-defined G matrix, which is assumed to be constant.
   *
   * @param U The control input to be applied for prediction.
   */
  inline void predict_with_fixed_G(
      const typename DiscreteStateSpace_Type::Original_U_Type &U) {

    this->state_space.U.push(U);

    if (this->_input_count < NUMBER_OF_DELAY) {
      this->_input_count++;
    } else {
      this->state_space.X = this->state_space.A * this->state_space.X +
                            this->state_space.B * this->state_space.U.get();
    }
  }

  /**
   * @brief Updates the state estimate with a fixed G matrix.
   *
   * This function performs the update step of the Kalman filter using a
   * pre-defined G matrix, which is assumed to be constant.
   *
   * @param Y The observed measurement to be used for updating the state.
   */
  inline void update_with_fixed_G(
      const typename DiscreteStateSpace_Type::Original_Y_Type &Y) {

    this->state_space.X =
        this->state_space.X +
        this->G * (Y - this->state_space.C * this->state_space.X);
  }

  /**
   * @brief Predicts the next state and updates the state estimate with a fixed
   * G matrix.
   *
   * This function combines the prediction and update steps of the Kalman filter
   * using a pre-defined G matrix, which is assumed to be constant.
   *
   * @param U The control input to be applied for prediction.
   * @param Y The observed measurement to be used for updating the state.
   */
  inline void predict_and_update_with_fixed_G(
      const typename DiscreteStateSpace_Type::Original_U_Type &U,
      const typename DiscreteStateSpace_Type::Original_Y_Type &Y) {

    this->predict_with_fixed_G(U);
    this->update_with_fixed_G(Y);
  }

  /**
   * @brief Converges the G matrix to a stable value.
   *
   * This function iteratively updates the covariance matrix P and the G
   * matrix until convergence is achieved, ensuring that the Kalman filter
   * stabilizes.
   */
  inline void update_P_one_step(void) {
    this->P = this->state_space.A *
                  PythonNumpy::A_mul_BTranspose(this->P, this->state_space.A) +
              this->Q;

    auto P_CT = PythonNumpy::A_mul_BTranspose(this->P, this->state_space.C);

    auto C_P_CT_R = this->state_space.C * P_CT + this->R;
    this->_C_P_CT_R_inv_solver.inv(C_P_CT_R);

    this->G = P_CT * this->_C_P_CT_R_inv_solver.get_answer();

    this->P = (PythonNumpy::make_DiagMatrixIdentity<_T, _STATE_SIZE>() -
               this->G * this->state_space.C) *
              this->P;
  }

  /**
   * @brief Converges the G matrix by iteratively updating the covariance matrix
   * P and checking for convergence.
   *
   * This function performs multiple iterations to ensure that the G matrix
   * stabilizes, allowing the Kalman filter to operate effectively.
   */
  inline void converge_G(void) {
    this->update_P_one_step();

    for (std::size_t k = 0; k < PythonControl::LKF_G_CONVERGE_REPEAT_MAX; k++) {

      auto previous_G = this->G;
      this->update_P_one_step();
      auto G_diff = this->G - previous_G;

      bool is_converged = true;
      for (std::size_t i = 0; i < _STATE_SIZE; i++) {
        for (std::size_t j = 0; j < _OUTPUT_SIZE; j++) {

          if (PythonMath::abs(this->G.access(i, j)) >
              PythonControl::KALMAN_FILTER_DIVISION_MIN) {
            if (PythonMath::abs(G_diff.access(i, j) / this->G.access(i, j)) >
                PythonControl::KALMAN_FILTER_DIVISION_MIN) {
              is_converged = false;
            }
          }
        }
      }

      if (is_converged) {
        break;
      }
    }
  }

  /* Get */

  /**
   * @brief Retrieves the estimated state vector x_hat.
   *
   * This function returns the current estimated state vector from the state
   * space.
   *
   * @return The estimated state vector x_hat.
   */
  inline auto get_x_hat(void) const ->
      typename DiscreteStateSpace_Type::Original_X_Type {
    return this->state_space.get_X();
  }

  /**
   * @brief Computes the estimated state vector x_hat without considering
   * delays.
   *
   * This function calculates the estimated state vector based on the current
   * state space and input count, ignoring any delays in the input.
   *
   * @return The estimated state vector x_hat without delay.
   */
  inline auto get_x_hat_without_delay(void) const ->
      typename DiscreteStateSpace_Type::Original_X_Type {

    return GetXHatWithoutDelayOperation::Linear<NUMBER_OF_DELAY>::compute(
        this->state_space, this->_input_count);
  }

  /* Set */

  /**
   * @brief Sets the estimated state vector x_hat.
   *
   * This function updates the state space with a new estimated state vector.
   *
   * @param x_hat The new estimated state vector to be set.
   */
  inline void
  set_x_hat(const typename DiscreteStateSpace_Type::Original_X_Type &x_hat) {
    this->state_space.X = x_hat;
  }

  /**
   * @brief Sets the covariance matrix P.
   *
   * This function updates the covariance matrix used in the Kalman filter.
   *
   * @param P_in The new covariance matrix to be set.
   */
  inline void set_decay_rate_for_C_P_CT_R_inv_solver(const _T &decay_rate_in) {
    this->_C_P_CT_R_inv_solver.set_decay_rate(decay_rate_in);
  }

  /**
   * @brief Sets the minimum division value for the C_P_CT_R_inv_solver.
   *
   * This function updates the minimum division value used in the Kalman filter
   * to avoid division by zero errors.
   *
   * @param division_min_in The new minimum division value to be set.
   */
  inline void
  set_division_min_for_C_P_CT_R_inv_solver(const _T &division_min_in) {
    this->_C_P_CT_R_inv_solver.set_division_min(division_min_in);
  }

public:
  /* Constant */
  static constexpr std::size_t STATE_SIZE = _STATE_SIZE;
  static constexpr std::size_t INPUT_SIZE = _INPUT_SIZE;
  static constexpr std::size_t OUTPUT_SIZE = _OUTPUT_SIZE;

  static constexpr std::size_t NUMBER_OF_DELAY =
      DiscreteStateSpace_Type::NUMBER_OF_DELAY;

public:
  /* Variable */
  DiscreteStateSpace_Type state_space;
  Q_Type Q;
  R_Type R;
  P_Type P;
  G_Type G;

protected:
  /* Variable */
  _C_P_CT_R_Inv_Type _C_P_CT_R_inv_solver;
  std::size_t _input_count;
};

/* make Linear Kalman Filter */

/**
 * @brief Factory function to create a LinearKalmanFilter instance.
 *
 * This function constructs a LinearKalmanFilter object with the specified
 * discrete state space, process noise covariance matrix (Q), and measurement
 * noise covariance matrix (R).
 *
 * @param DiscreteStateSpace The discrete state space object.
 * @param Q The process noise covariance matrix.
 * @param R The measurement noise covariance matrix.
 * @return A LinearKalmanFilter instance.
 */
template <typename DiscreteStateSpace_Type, typename Q_Type, typename R_Type>
inline auto make_LinearKalmanFilter(DiscreteStateSpace_Type DiscreteStateSpace,
                                    Q_Type Q, R_Type R)
    -> LinearKalmanFilter<DiscreteStateSpace_Type, Q_Type, R_Type> {

  return LinearKalmanFilter<DiscreteStateSpace_Type, Q_Type, R_Type>(
      DiscreteStateSpace, Q, R);
}

/* Linear Kalman Filter Type */

/**
 * @brief Type alias for LinearKalmanFilter with specified types.
 *
 * This alias simplifies the usage of LinearKalmanFilter by providing a
 * convenient type name that includes the discrete state space, process noise
 * covariance matrix, and measurement noise covariance matrix types.
 *
 * @tparam DiscreteStateSpace_Type The type of the discrete state space.
 * @tparam Q_Type The type of the process noise covariance matrix.
 * @tparam R_Type The type of the measurement noise covariance matrix.
 */
template <typename DiscreteStateSpace_Type, typename Q_Type, typename R_Type>
using LinearKalmanFilter_Type =
    LinearKalmanFilter<DiscreteStateSpace_Type, Q_Type, R_Type>;

/* state and measurement function alias */

/**
 * @brief Type aliases for state and measurement function objects.
 *
 * These aliases define the types of function objects used for state and
 * measurement functions in Kalman filters, allowing for flexible function
 * signatures.
 *
 * @tparam State_Type The type of the state vector.
 * @tparam Input_Type The type of the input vector.
 * @tparam Parameter_Type The type of additional parameters for the functions.
 */
template <typename State_Type, typename Input_Type, typename Parameter_Type>
using StateFunction_Object = std::function<State_Type(
    const State_Type &, const Input_Type &, const Parameter_Type &)>;

/**
 * @brief Type alias for the Jacobian of the state function.
 *
 * This alias defines the type of the Jacobian function object for the state
 * function, which computes the Jacobian matrix given the state, input, and
 * parameters.
 *
 * @tparam A_Type The type of the Jacobian matrix.
 * @tparam State_Type The type of the state vector.
 * @tparam Input_Type The type of the input vector.
 * @tparam Parameter_Type The type of additional parameters for the function.
 */
template <typename A_Type, typename State_Type, typename Input_Type,
          typename Parameter_Type>
using StateFunctionJacobian_Object = std::function<A_Type(
    const State_Type &, const Input_Type &, const Parameter_Type &)>;

/**
 * @brief Type alias for the measurement function object.
 *
 * This alias defines the type of the measurement function object, which maps
 * the state and parameters to an output vector.
 *
 * @tparam Output_Type The type of the output vector.
 * @tparam State_Type The type of the state vector.
 * @tparam Parameter_Type The type of additional parameters for the function.
 */
template <typename Output_Type, typename State_Type, typename Parameter_Type>
using MeasurementFunction_Object =
    std::function<Output_Type(const State_Type &, const Parameter_Type &)>;

/**
 * @brief Type alias for the Jacobian of the measurement function.
 *
 * This alias defines the type of the Jacobian function object for the
 * measurement function, which computes the Jacobian matrix given the state and
 * parameters.
 *
 * @tparam C_Type The type of the Jacobian matrix.
 * @tparam State_Type The type of the state vector.
 * @tparam Parameter_Type The type of additional parameters for the function.
 */
template <typename C_Type, typename State_Type, typename Parameter_Type>
using MeasurementFunctionJacobian_Object =
    std::function<C_Type(const State_Type &, const Parameter_Type &)>;

/* Extended Kalman Filter */

/**
 * @brief Extended Kalman Filter class template.
 *
 * This class implements an extended Kalman filter for discrete state spaces.
 * It supports prediction and update operations, and it is designed to work
 * with diagonal covariance matrices for process noise (Q) and measurement
 * noise (R).
 *
 * @tparam A_Type_In The type of the state transition matrix.
 * @tparam C_Type_In The type of the measurement matrix.
 * @tparam U_Type_In The type of the input vector.
 * @tparam Q_Type_In The type of the process noise covariance matrix.
 * @tparam R_Type_In The type of the measurement noise covariance matrix.
 * @tparam Parameter_Type_In The type of additional parameters for the filter.
 * @tparam Number_Of_Delay The number of delays in the input store (default is
 * 0).
 */
template <typename A_Type_In, typename C_Type_In, typename U_Type_In,
          typename Q_Type_In, typename R_Type_In, typename Parameter_Type_In,
          std::size_t Number_Of_Delay = 0>
class ExtendedKalmanFilter {
public:
  /* Type */
  using A_Type = A_Type_In;
  using C_Type = C_Type_In;
  using U_Type = U_Type_In;
  using Q_Type = Q_Type_In;
  using R_Type = R_Type_In;
  using Parameter_Type = Parameter_Type_In;

protected:
  /* Type */
  using _T = typename A_Type::Value_Type;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Matrix value data type must be float or double.");

  static constexpr std::size_t _STATE_SIZE = A_Type::COLS;
  static constexpr std::size_t _INPUT_SIZE = U_Type::COLS;
  static constexpr std::size_t _OUTPUT_SIZE = C_Type::COLS;

  using _Input_Type = PythonControl::StateSpaceInput_Type<_T, _INPUT_SIZE>;
  using _State_Type = PythonControl::StateSpaceState_Type<_T, _STATE_SIZE>;
  using _Measurement_Type =
      PythonControl::StateSpaceOutput_Type<_T, _OUTPUT_SIZE>;

  using _InputStored_Type =
      PythonControl::DelayedVectorObject<_Input_Type, Number_Of_Delay>;

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

  using _C_P_CT_R_Inv_Type = PythonNumpy::LinalgSolverInv_Type<
      PythonNumpy::DenseMatrix_Type<_T, _OUTPUT_SIZE, _OUTPUT_SIZE>>;

public:
  /* Type  */
  using Value_Type = _T;

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
  ExtendedKalmanFilter()
      : A(), C(), Q(), R(),
        P(PythonNumpy::make_DiagMatrixIdentity<_T, _STATE_SIZE>()
              .create_dense()),
        G(), X_hat(), U_store(), parameters(), _C_P_CT_R_inv_solver(),
        _input_count(static_cast<std::size_t>(0)) {}

  ExtendedKalmanFilter(
      const Q_Type &Q, const R_Type &R, _StateFunction_Object &state_function,
      _StateFunctionJacobian_Object &state_function_jacobian,
      _MeasurementFunction_Object &measurement_function,
      _MeasurementFunctionJacobian_Object &measurement_function_jacobian,
      const Parameter_Type &parameters)
      : A(), C(), Q(Q), R(R),
        P(PythonNumpy::make_DiagMatrixIdentity<_T, _STATE_SIZE>()
              .create_dense()),
        G(), X_hat(), U_store(), parameters(parameters), _C_P_CT_R_inv_solver(),
        _state_function(state_function),
        _state_function_jacobian(state_function_jacobian),
        _measurement_function(measurement_function),
        _measurement_function_jacobian(measurement_function_jacobian),
        _input_count(static_cast<std::size_t>(0)) {}

  /* Copy Constructor */
  ExtendedKalmanFilter(
      const ExtendedKalmanFilter<A_Type, C_Type, U_Type, Q_Type, R_Type,
                                 Parameter_Type, Number_Of_Delay> &input)
      : A(input.A), C(input.C), Q(input.Q), R(input.R), P(input.P), G(input.G),
        X_hat(input.X_hat), U_store(input.U_store),
        parameters(input.parameters),
        _C_P_CT_R_inv_solver(input._C_P_CT_R_inv_solver),
        _state_function(input._state_function),
        _state_function_jacobian(input._state_function_jacobian),
        _measurement_function(input._measurement_function),
        _measurement_function_jacobian(input._measurement_function_jacobian),
        _input_count(input._input_count) {}

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
      this->U_store = input.U_store;
      this->parameters = input.parameters;
      this->_C_P_CT_R_inv_solver = input._C_P_CT_R_inv_solver;
      this->_state_function = input._state_function;
      this->_state_function_jacobian = input._state_function_jacobian;
      this->_measurement_function = input._measurement_function;
      this->_measurement_function_jacobian =
          input._measurement_function_jacobian;
      this->_input_count = input._input_count;
    }
    return *this;
  }

  /* Move Constructor */
  ExtendedKalmanFilter(
      ExtendedKalmanFilter<A_Type, C_Type, U_Type, Q_Type, R_Type,
                           Parameter_Type, Number_Of_Delay> &&input) noexcept
      : A(std::move(input.A)), C(std::move(input.C)), Q(std::move(input.Q)),
        R(std::move(input.R)), P(std::move(input.P)), G(std::move(input.G)),
        X_hat(std::move(input.X_hat)), U_store(std::move(input.U_store)),
        parameters(std::move(input.parameters)),
        _C_P_CT_R_inv_solver(std::move(input._C_P_CT_R_inv_solver)),
        _state_function(input._state_function),
        _state_function_jacobian(input._state_function_jacobian),
        _measurement_function(input._measurement_function),
        _measurement_function_jacobian(input._measurement_function_jacobian),
        _input_count(std::move(input._input_count)) {}

  ExtendedKalmanFilter<A_Type, C_Type, U_Type, Q_Type, R_Type, Parameter_Type,
                       Number_Of_Delay> &
  operator=(
      ExtendedKalmanFilter<A_Type, C_Type, U_Type, Q_Type, R_Type,
                           Parameter_Type, Number_Of_Delay> &&input) noexcept {
    if (this != &input) {
      this->A = std::move(input.A);
      this->C = std::move(input.C);
      this->Q = std::move(input.Q);
      this->R = std::move(input.R);
      this->P = std::move(input.P);
      this->G = std::move(input.G);
      this->X_hat = std::move(input.X_hat);
      this->U_store = std::move(input.U_store);
      this->parameters = std::move(input.parameters);
      this->_C_P_CT_R_inv_solver = std::move(input._C_P_CT_R_inv_solver);
      this->_state_function = std::move(input._state_function);
      this->_state_function_jacobian =
          std::move(input._state_function_jacobian);
      this->_measurement_function = std::move(input._measurement_function);
      this->_measurement_function_jacobian =
          std::move(input._measurement_function_jacobian);
      this->_input_count = std::move(input._input_count);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Predicts the next state based on the current input.
   *
   * This function performs the prediction step of the extended Kalman filter,
   * updating the state vector and covariance matrix based on the current input.
   *
   * @param U The control input to be applied for prediction.
   */
  inline void predict(const U_Type &U) {

    U_store.push(U);

    PredictOperation::Extended<NUMBER_OF_DELAY>::execute(
        this->_state_function_jacobian, this->_state_function, this->A, this->P,
        this->Q, this->U_store, this->X_hat, this->parameters,
        this->_input_count);
  }

  /**
   * @brief Updates the state estimate based on the measurement.
   *
   * This function performs the update step of the extended Kalman filter,
   * adjusting the state vector and covariance matrix based on the observed
   * measurement.
   *
   * @param Y The observed measurement to be used for updating the state.
   */
  inline void update(const _Measurement_Type &Y) {

    this->C = this->_measurement_function_jacobian(this->X_hat, parameters);

    auto P_CT = PythonNumpy::A_mul_BTranspose(this->P, this->C);

    auto C_P_CT_R = this->C * P_CT + this->R;
    this->_C_P_CT_R_inv_solver.inv(C_P_CT_R);

    this->G = P_CT * this->_C_P_CT_R_inv_solver.get_answer();

    this->X_hat =
        this->X_hat +
        this->G * (Y - this->_measurement_function(this->X_hat, parameters));

    this->P = (PythonNumpy::make_DiagMatrixIdentity<_T, _STATE_SIZE>() -
               this->G * this->C) *
              this->P;
  }

  /**
   * @brief Predicts the next state and updates the state estimate based on the
   * measurement.
   *
   * This function combines the prediction and update steps of the extended
   * Kalman filter into a single operation.
   *
   * @param U The control input to be applied for prediction.
   * @param Y The observed measurement to be used for updating the state.
   */
  inline void predict_and_update(const U_Type &U, const _Measurement_Type &Y) {

    this->predict(U);
    this->update(Y);
  }

  /**
   * @brief Calculates the state function for a given state and input.
   *
   * This function computes the next state based on the provided state vector,
   * input vector, and the state function defined for the extended Kalman
   * filter.
   *
   * @param X The current state vector.
   * @param U The control input vector.
   * @return The calculated next state vector.
   */
  inline auto calculate_state_function(const _State_Type &X,
                                       const U_Type &U) const -> _State_Type {
    return this->_state_function(X, U, this->parameters);
  }

  /**
   * @brief Calculates the measurement function for a given state.
   *
   * This function computes the expected measurement based on the provided
   * state vector and the measurement function defined for the extended Kalman
   * filter.
   *
   * @param X The current state vector.
   * @return The calculated measurement vector.
   */
  inline auto calculate_measurement_function(const _State_Type &X) const
      -> _Measurement_Type {
    return this->_measurement_function(X, this->parameters);
  }

  /* Get */

  /**
   * @brief Retrieves the estimated state vector x_hat.
   *
   * This function returns the current estimated state vector.
   *
   * @return The estimated state vector x_hat.
   */
  inline auto get_x_hat(void) const -> _State_Type { return this->X_hat; }

  /**
   * @brief Computes the estimated state vector x_hat without considering
   * delays.
   *
   * This function calculates the estimated state vector based on the current
   * state function, input store, and parameters, ignoring any delays in the
   * input.
   *
   * @return The estimated state vector x_hat without delay.
   */
  inline auto get_x_hat_without_delay(void) const -> _State_Type {

    return GetXHatWithoutDelayOperation::Extended<NUMBER_OF_DELAY>::compute(
        this->_state_function, this->X_hat, this->U_store, this->parameters,
        this->_input_count);
  }

  /* Set */

  /**
   * @brief Sets the estimated state vector x_hat.
   *
   * This function updates the estimated state vector with a new value.
   *
   * @param x_hat The new estimated state vector to be set.
   */
  inline void set_x_hat(const _State_Type &x_hat) { this->X_hat = x_hat; }

  /**
   * @brief Sets the covariance matrix P.
   *
   * This function updates the covariance matrix used in the extended Kalman
   * filter.
   *
   * @param P_in The new covariance matrix to be set.
   */
  inline void set_decay_rate_for_C_P_CT_R_inv_solver(const _T &decay_rate_in) {
    this->_C_P_CT_R_inv_solver.set_decay_rate(decay_rate_in);
  }

  /**
   * @brief Sets the minimum division value for the C_P_CT_R_inv_solver.
   *
   * This function updates the minimum division value used in the extended
   * Kalman filter to avoid division by zero errors.
   *
   * @param division_min_in The new minimum division value to be set.
   */
  inline void
  set_division_min_for_C_P_CT_R_inv_solver(const _T &division_min_in) {
    this->_C_P_CT_R_inv_solver.set_division_min(division_min_in);
  }

public:
  /* Constant */
  static constexpr std::size_t STATE_SIZE = _STATE_SIZE;
  static constexpr std::size_t INPUT_SIZE = _INPUT_SIZE;
  static constexpr std::size_t OUTPUT_SIZE = _OUTPUT_SIZE;

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
  _InputStored_Type U_store;

  Parameter_Type parameters;

protected:
  /* Variable */
  _C_P_CT_R_Inv_Type _C_P_CT_R_inv_solver;
  _StateFunction_Object _state_function;
  _StateFunctionJacobian_Object _state_function_jacobian;
  _MeasurementFunction_Object _measurement_function;
  _MeasurementFunctionJacobian_Object _measurement_function_jacobian;
  std::size_t _input_count;
};

/* Extended Kalman Filter Type */
template <typename A_Type, typename C_Type, typename U_Type, typename Q_Type,
          typename R_Type, typename Parameter_Type,
          std::size_t Number_Of_Delay = 0>
using ExtendedKalmanFilter_Type =
    ExtendedKalmanFilter<A_Type, C_Type, U_Type, Q_Type, R_Type, Parameter_Type,
                         Number_Of_Delay>;

/* Sigma Points Calculator */

/**
 * @brief Sigma Points Calculator class template.
 *
 * This class calculates sigma points for the Unscented Kalman Filter (UKF)
 * based on the state and covariance matrix. It provides methods to set the
 * sigma point weight and calculate the sigma points.
 *
 * @tparam State_Type The type of the state vector.
 * @tparam P_Type The type of the covariance matrix.
 */
template <typename State_Type, typename P_Type> class SigmaPointsCalculator {
protected:
  /* Type */
  using _T = typename State_Type::Value_Type;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Matrix value data type must be float or double.");

  static constexpr std::size_t _STATE_SIZE = State_Type::COLS;

  using _P_Chol_Solver_Type = PythonNumpy::LinalgSolverCholesky_Type<
      PythonNumpy::DenseMatrix_Type<_T, _STATE_SIZE, _STATE_SIZE>>;

public:
  /* Type */
  using Kai_Type =
      PythonNumpy::DenseMatrix_Type<_T, _STATE_SIZE, (2 * _STATE_SIZE + 1)>;

public:
  /* Constructor */
  SigmaPointsCalculator(){};

  /* Copy Constructor */
  SigmaPointsCalculator(const SigmaPointsCalculator<State_Type, P_Type> &input)
      : sigma_point_weight(input.sigma_point_weight),
        _P_cholesky_solver(input._P_cholesky_solver) {}

  SigmaPointsCalculator<State_Type, P_Type> &
  operator=(const SigmaPointsCalculator<State_Type, P_Type> &input) {
    if (this != &input) {
      this->sigma_point_weight = input.sigma_point_weight;
      this->_P_cholesky_solver = input._P_cholesky_solver;
    }
    return *this;
  }

  /* Move Constructor */
  SigmaPointsCalculator(
      SigmaPointsCalculator<State_Type, P_Type> &&input) noexcept
      : sigma_point_weight(std::move(input.sigma_point_weight)),
        _P_cholesky_solver(std::move(input._P_cholesky_solver)) {}

  SigmaPointsCalculator<State_Type, P_Type> &
  operator=(SigmaPointsCalculator<State_Type, P_Type> &&input) noexcept {
    if (this != &input) {
      this->sigma_point_weight = std::move(input.sigma_point_weight);
      this->_P_cholesky_solver = std::move(input._P_cholesky_solver);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Sets the weight for the sigma points.
   *
   * This function updates the sigma point weight used in the calculation of
   * sigma points.
   *
   * @param sigma_point_weight_in The new sigma point weight to be set.
   */
  inline void set_weight(const _T &sigma_point_weight_in) {
    this->sigma_point_weight = sigma_point_weight_in;
  }

  /**
   * @brief Calculates the sigma points based on the state and covariance
   * matrix.
   *
   * This function computes the sigma points using the provided state vector
   * and covariance matrix, applying the Cholesky decomposition to the
   * covariance matrix.
   *
   * @param X_in The state vector.
   * @param P_in The covariance matrix.
   * @return The calculated sigma points as a Kai_Type matrix.
   */
  inline auto calculate(const State_Type &X_in, const P_Type &P_in)
      -> Kai_Type {

    _P_cholesky_solver = PythonNumpy::make_LinalgSolverCholesky<P_Type>();
    Kai_Type Kai;

    auto SP = _P_cholesky_solver.solve(P_in);

    PythonNumpy::set_row<0>(Kai, X_in);
    UKF_Operation::UpdateSigmaPointMatrix<
        _T, Kai_Type, State_Type, decltype(SP)>::set(Kai, X_in, SP,
                                                     this->sigma_point_weight);

    return Kai;
  }

public:
  /* Variable */
  _T sigma_point_weight;

protected:
  /* Variable */
  _P_Chol_Solver_Type _P_cholesky_solver;
};

/* Unscented Kalman Filter */

/**
 * @brief Unscented Kalman Filter class template.
 *
 * This class implements an Unscented Kalman Filter (UKF) for discrete state
 * spaces. It supports prediction and update operations, and it is designed to
 * work with diagonal covariance matrices for process noise (Q) and measurement
 * noise (R).
 *
 * @tparam U_Type_In The type of the input vector.
 * @tparam Q_Type_In The type of the process noise covariance matrix.
 * @tparam R_Type_In The type of the measurement noise covariance matrix.
 * @tparam Parameter_Type_In The type of additional parameters for the filter.
 * @tparam Number_Of_Delay The number of delays in the input store (default is
 * 0).
 */
template <typename U_Type_In, typename Q_Type_In, typename R_Type_In,
          typename Parameter_Type_In, std::size_t Number_Of_Delay = 0>
class UnscentedKalmanFilter {
public:
  /* Type */
  using U_Type = U_Type_In;
  using Q_Type = Q_Type_In;
  using R_Type = R_Type_In;
  using Parameter_Type = Parameter_Type_In;

protected:
  /* Type */
  using _T = typename U_Type::Value_Type;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Matrix value data type must be float or double.");

  static constexpr std::size_t _STATE_SIZE = Q_Type::COLS;
  static constexpr std::size_t _INPUT_SIZE = U_Type::COLS;
  static constexpr std::size_t _OUTPUT_SIZE = R_Type::COLS;

  using _Input_Type = PythonControl::StateSpaceInput_Type<_T, _INPUT_SIZE>;
  using _State_Type = PythonControl::StateSpaceState_Type<_T, _STATE_SIZE>;
  using _Measurement_Type =
      PythonControl::StateSpaceOutput_Type<_T, _OUTPUT_SIZE>;

  using _InputStored_Type =
      PythonControl::DelayedVectorObject<_Input_Type, Number_Of_Delay>;

  using _StateFunction_Object =
      PythonControl::StateFunction_Object<_State_Type, U_Type, Parameter_Type>;
  using _MeasurementFunction_Object =
      PythonControl::MeasurementFunction_Object<_Measurement_Type, _State_Type,
                                                Parameter_Type>;

  using _P_Chol_Solver_Type = PythonNumpy::LinalgSolverCholesky_Type<
      PythonNumpy::DenseMatrix_Type<_T, _STATE_SIZE, _STATE_SIZE>>;

  using _P_YY_R_Inv_Type = PythonNumpy::LinalgSolverInv_Type<
      PythonNumpy::DenseMatrix_Type<_T, _OUTPUT_SIZE, _OUTPUT_SIZE>>;

  using _W_Type = PythonNumpy::DiagMatrix_Type<_T, (2 * _STATE_SIZE + 1)>;

  using _Kai_Type =
      PythonNumpy::DenseMatrix_Type<_T, _STATE_SIZE, (2 * _STATE_SIZE + 1)>;

  using _P_Type = PythonNumpy::DenseMatrix_Type<_T, _STATE_SIZE, _STATE_SIZE>;

  using _SigmaPointsCalculator_Type =
      SigmaPointsCalculator<_State_Type, _P_Type>;

public:
  /* Type  */
  using Value_Type = _T;

  using P_Type = _P_Type;
  using P_xy_Type =
      PythonNumpy::DenseMatrix_Type<_T, _STATE_SIZE, _OUTPUT_SIZE>;
  using P_yy_Type =
      PythonNumpy::DenseMatrix_Type<_T, _OUTPUT_SIZE, _OUTPUT_SIZE>;

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
  UnscentedKalmanFilter()
      : Q(), R(), P(PythonNumpy::make_DiagMatrixIdentity<_T, _STATE_SIZE>()
                        .create_dense()),
        G(), kappa(static_cast<_T>(0)), alpha(static_cast<_T>(0.5)),
        beta(static_cast<_T>(2)), w_m(static_cast<_T>(0)), W(), X_hat(), X_d(),
        U_store(), parameters(), _P_YY_R_inv_solver(), _state_function(),
        _measurement_function(), _predict_sigma_points_calculator(),
        _update_sigma_points_calculator(),
        _input_count(static_cast<std::size_t>(0)) {}

  UnscentedKalmanFilter(const Q_Type &Q, const R_Type &R,
                        _StateFunction_Object &state_function,
                        _MeasurementFunction_Object &measurement_function,
                        const Parameter_Type &parameters)
      : Q(Q), R(R), P(PythonNumpy::make_DiagMatrixIdentity<_T, _STATE_SIZE>()
                          .create_dense()),
        G(), kappa(static_cast<_T>(0)), alpha(static_cast<_T>(0.5)),
        beta(static_cast<_T>(2)), w_m(static_cast<_T>(0)), W(), X_hat(), X_d(),
        U_store(), parameters(parameters), _P_YY_R_inv_solver(),
        _state_function(state_function),
        _measurement_function(measurement_function),
        _predict_sigma_points_calculator(), _update_sigma_points_calculator(),
        _input_count(static_cast<std::size_t>(0)) {

    this->calculate_weights();
  }

  UnscentedKalmanFilter(const Q_Type &Q, const R_Type &R,
                        _StateFunction_Object &state_function,
                        _MeasurementFunction_Object &measurement_function,
                        const Parameter_Type &parameters, _T kappa_in)
      : Q(Q), R(R), P(PythonNumpy::make_DiagMatrixIdentity<_T, _STATE_SIZE>()
                          .create_dense()),
        G(), kappa(kappa_in), alpha(static_cast<_T>(0.5)),
        beta(static_cast<_T>(2)), w_m(static_cast<_T>(0)), W(), X_hat(), X_d(),
        U_store(), parameters(parameters), _P_YY_R_inv_solver(),
        _state_function(state_function),
        _measurement_function(measurement_function),
        _predict_sigma_points_calculator(), _update_sigma_points_calculator(),
        _input_count(static_cast<std::size_t>(0)) {

    this->calculate_weights();
  }

  UnscentedKalmanFilter(const Q_Type &Q, const R_Type &R,
                        _StateFunction_Object &state_function,
                        _MeasurementFunction_Object &measurement_function,
                        const Parameter_Type &parameters, _T kappa_in,
                        _T alpha_in)
      : Q(Q), R(R), P(PythonNumpy::make_DiagMatrixIdentity<_T, _STATE_SIZE>()
                          .create_dense()),
        G(), kappa(kappa_in), alpha(alpha_in), beta(static_cast<_T>(2)),
        w_m(static_cast<_T>(0)), W(), X_hat(), X_d(), U_store(),
        parameters(parameters), _P_YY_R_inv_solver(),
        _state_function(state_function),
        _measurement_function(measurement_function),
        _predict_sigma_points_calculator(), _update_sigma_points_calculator(),
        _input_count(static_cast<std::size_t>(0)) {

    this->calculate_weights();
  }

  UnscentedKalmanFilter(const Q_Type &Q, const R_Type &R,
                        _StateFunction_Object &state_function,
                        _MeasurementFunction_Object &measurement_function,
                        const Parameter_Type &parameters, _T kappa_in,
                        _T alpha_in, _T beta_in)
      : Q(Q), R(R), P(PythonNumpy::make_DiagMatrixIdentity<_T, _STATE_SIZE>()
                          .create_dense()),
        G(), kappa(kappa_in), alpha(alpha_in), beta(beta_in),
        w_m(static_cast<_T>(0)), W(), X_hat(), X_d(), U_store(),
        parameters(parameters), _P_YY_R_inv_solver(),
        _state_function(state_function),
        _measurement_function(measurement_function),
        _predict_sigma_points_calculator(), _update_sigma_points_calculator(),
        _input_count(static_cast<std::size_t>(0)) {

    this->calculate_weights();
  }

  /* Copy Constructor */
  UnscentedKalmanFilter(
      const UnscentedKalmanFilter<U_Type, Q_Type, R_Type, Parameter_Type,
                                  Number_Of_Delay> &input)
      : Q(input.Q), R(input.R), P(input.P), G(input.G), kappa(input.kappa),
        alpha(input.alpha), beta(input.beta), w_m(input.w_m), W(input.W),
        X_hat(input.X_hat), X_d(input.X_d), U_store(input.U_store),
        parameters(input.parameters),
        _P_YY_R_inv_solver(input._P_YY_R_inv_solver),
        _state_function(input._state_function),
        _measurement_function(input._measurement_function),
        _predict_sigma_points_calculator(
            input._predict_sigma_points_calculator),
        _update_sigma_points_calculator(input._update_sigma_points_calculator),
        _input_count(input._input_count) {}

  UnscentedKalmanFilter<U_Type, Q_Type, R_Type, Parameter_Type,
                        Number_Of_Delay> &
  operator=(const UnscentedKalmanFilter<U_Type, Q_Type, R_Type, Parameter_Type,
                                        Number_Of_Delay> &input) {
    if (this != &input) {
      this->Q = input.Q;
      this->R = input.R;
      this->P = input.P;
      this->G = input.G;
      this->kappa = input.kappa;
      this->alpha = input.alpha;
      this->beta = input.beta;
      this->w_m = input.w_m;
      this->W = input.W;
      this->X_hat = input.X_hat;
      this->X_d = input.X_d;
      this->U_store = input.U_store;
      this->parameters = input.parameters;
      this->_P_YY_R_inv_solver = input._P_YY_R_inv_solver;
      this->_state_function = input._state_function;
      this->_measurement_function = input._measurement_function;
      this->_predict_sigma_points_calculator =
          input._predict_sigma_points_calculator;
      this->_update_sigma_points_calculator =
          input._update_sigma_points_calculator;
      this->_input_count = input._input_count;
    }
    return *this;
  }

  /* Move Constructor */
  UnscentedKalmanFilter(
      UnscentedKalmanFilter<U_Type, Q_Type, R_Type, Parameter_Type,
                            Number_Of_Delay> &&input) noexcept
      : Q(std::move(input.Q)), R(std::move(input.R)), P(std::move(input.P)),
        G(std::move(input.G)), kappa(std::move(input.kappa)),
        alpha(std::move(input.alpha)), beta(std::move(input.beta)),
        w_m(std::move(input.w_m)), W(std::move(input.W)),
        X_hat(std::move(input.X_hat)), X_d(std::move(input.X_d)),
        U_store(std::move(input.U_store)),
        parameters(std::move(input.parameters)),
        _P_YY_R_inv_solver(std::move(input._P_YY_R_inv_solver)),
        _state_function(std::move(input._state_function)),
        _measurement_function(std::move(input._measurement_function)),
        _predict_sigma_points_calculator(
            std::move(input._predict_sigma_points_calculator)),
        _update_sigma_points_calculator(
            std::move(input._update_sigma_points_calculator)),
        _input_count(std::move(input._input_count)) {}

  UnscentedKalmanFilter<U_Type, Q_Type, R_Type, Parameter_Type,
                        Number_Of_Delay> &
  operator=(UnscentedKalmanFilter<U_Type, Q_Type, R_Type, Parameter_Type,
                                  Number_Of_Delay> &&input) noexcept {
    if (this != &input) {
      this->Q = std::move(input.Q);
      this->R = std::move(input.R);
      this->P = std::move(input.P);
      this->G = std::move(input.G);
      this->kappa = std::move(input.kappa);
      this->alpha = std::move(input.alpha);
      this->beta = std::move(input.beta);
      this->w_m = std::move(input.w_m);
      this->W = std::move(input.W);
      this->X_hat = std::move(input.X_hat);
      this->X_d = std::move(input.X_d);
      this->U_store = std::move(input.U_store);
      this->parameters = std::move(input.parameters);
      this->_P_YY_R_inv_solver = std::move(input._P_YY_R_inv_solver);
      this->_state_function = std::move(input._state_function);
      this->_measurement_function = std::move(input._measurement_function);
      this->_predict_sigma_points_calculator =
          std::move(input._predict_sigma_points_calculator);
      this->_update_sigma_points_calculator =
          std::move(input._update_sigma_points_calculator);
      this->_input_count = std::move(input._input_count);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Calculates the weights for the sigma points.
   *
   * This function computes the weights used in the Unscented Kalman Filter
   *
   * to determine the contribution of each sigma point to the state
   * estimation.
   * @details The weights are calculated based on the filter parameters
   * (alpha, beta, kappa) and the size of the state vector.
   */
  inline void calculate_weights(void) {
    _T lambda_weight = this->alpha * this->alpha *
                           (static_cast<_T>(_STATE_SIZE) + this->kappa) -
                       static_cast<_T>(_STATE_SIZE);

    this->w_m = lambda_weight / (static_cast<_T>(_STATE_SIZE) + lambda_weight);

    this->W.template set<0, 0>(this->w_m + static_cast<_T>(1) -
                               this->alpha * this->alpha + this->beta);

    UKF_Operation::SetRestOfW<_T, _W_Type, 1>::set(
        this->W,
        static_cast<_T>(1) / (static_cast<_T>(2) *
                              (static_cast<_T>(_STATE_SIZE) + lambda_weight)));

    _T sigma_point_weight =
        PythonMath::sqrt(static_cast<_T>(_STATE_SIZE) + lambda_weight);

    this->_predict_sigma_points_calculator.set_weight(sigma_point_weight);
    this->_update_sigma_points_calculator.set_weight(sigma_point_weight);
  }

  /**
   * @brief Predicts the next state based on the current input.
   *
   * This function performs the prediction step of the Unscented Kalman Filter,
   * updating the state vector and covariance matrix based on the current input.
   *
   * @param U The control input to be applied for prediction.
   */
  inline void predict(const U_Type &U) {

    U_store.push(U);

    PredictOperation::Unscented<NUMBER_OF_DELAY>::execute(
        this->_state_function, this->_predict_sigma_points_calculator,
        this->X_hat, this->U_store, this->P, this->w_m, this->W, this->Q,
        this->X_d, this->parameters, this->_input_count);
  }

  /**
   * @brief Updates the state estimate based on the measurement.
   *
   * This function performs the update step of the Unscented Kalman Filter,
   * adjusting the state vector and covariance matrix based on the observed
   * measurement.
   *
   * @param Y The observed measurement to be used for updating the state.
   */
  inline void update(const _Measurement_Type &Y) {

    auto Kai =
        this->_update_sigma_points_calculator.calculate(this->X_hat, this->P);

    auto Y_d = PythonNumpy::make_DenseMatrixZeros<_T, _OUTPUT_SIZE,
                                                  (2 * _STATE_SIZE + 1)>();
    using Y_d_Type = decltype(Y_d);

    UKF_Operation::MeasurementFunctionEachSigmaPoints<
        Y_d_Type, _Kai_Type, _MeasurementFunction_Object,
        Parameter_Type>::compute(Y_d, Kai, this->_measurement_function,
                                 parameters);

    _Measurement_Type Y_hat_m =
        PythonNumpy::make_DenseMatrixZeros<_T, _OUTPUT_SIZE, 1>();
    Y_hat_m = this->w_m * PythonNumpy::get_row<0>(Y_d);
    UKF_Operation::AverageSigmaPoints<_Measurement_Type, _W_Type,
                                      Y_d_Type>::compute(Y_hat_m, this->W, Y_d);

    UKF_Operation::SigmaPointsCovariance<Y_d_Type, _Measurement_Type>::compute(
        Y_d, Y_d, Y_hat_m);

    auto P_yy_R = Y_d * PythonNumpy::A_mul_BTranspose(this->W, Y_d) + this->R;
    auto P_xy = this->X_d * PythonNumpy::A_mul_BTranspose(this->W, Y_d);

    _P_YY_R_inv_solver.inv(P_yy_R);

    this->G = P_xy * this->_P_YY_R_inv_solver.get_answer();

    this->X_hat = this->X_hat + this->G * (Y - Y_hat_m);
    this->P = this->P - PythonNumpy::A_mul_BTranspose(this->G, P_xy);
  }

  /**
   * @brief Predicts the next state and updates the state estimate based on the
   * measurement.
   *
   * This function combines the prediction and update steps of the Unscented
   * Kalman Filter into a single operation.
   *
   * @param U The control input to be applied for prediction.
   * @param Y The observed measurement to be used for updating the state.
   */
  inline void predict_and_update(const U_Type &U, const _Measurement_Type &Y) {

    this->predict(U);
    this->update(Y);
  }

  /**
   * @brief Calculates the state function for a given state and input.
   *
   * This function computes the next state based on the provided state vector,
   * input vector, and the state function defined for the Unscented Kalman
   * filter.
   *
   * @param X The current state vector.
   * @param U The control input vector.
   * @return The calculated next state vector.
   */
  inline auto calculate_state_function(const _State_Type &X,
                                       const U_Type &U) const -> _State_Type {
    return this->_state_function(X, U, this->parameters);
  }

  /**
   * @brief Calculates the measurement function for a given state.
   *
   * This function computes the expected measurement based on the provided
   * state vector and the measurement function defined for the Unscented Kalman
   * filter.
   *
   * @param X The current state vector.
   * @return The calculated measurement vector.
   */
  inline auto calculate_measurement_function(const _State_Type &X) const
      -> _Measurement_Type {
    return this->_measurement_function(X, this->parameters);
  }

  /* Get */

  /**
   * @brief Retrieves the estimated state vector x_hat.
   *
   * This function returns the current estimated state vector.
   *
   * @return The estimated state vector x_hat.
   */
  inline auto get_x_hat(void) const -> _State_Type { return this->X_hat; }

  /**
   * @brief Computes the estimated state vector x_hat without considering
   * delays.
   *
   * This function calculates the estimated state vector based on the current
   * state function, input store, and parameters, ignoring any delays in the
   * input.
   *
   * @return The estimated state vector x_hat without delay.
   */
  inline auto get_x_hat_without_delay(void) const -> _State_Type {

    return GetXHatWithoutDelayOperation::Extended<NUMBER_OF_DELAY>::compute(
        this->_state_function, this->X_hat, this->U_store, this->parameters,
        this->_input_count);
  }

  /* Set */

  /**
   * @brief Sets the estimated state vector x_hat.
   *
   * This function updates the estimated state vector with a new value.
   *
   * @param x_hat The new estimated state vector to be set.
   */
  inline void set_x_hat(const _State_Type &x_hat) { this->X_hat = x_hat; }

  /**
   * @brief Sets the covariance matrix P.
   *
   * This function updates the covariance matrix used in the Unscented Kalman
   * Filter
   *
   * @param P_in The new covariance matrix to be set.
   */
  inline void set_decay_rate_for_P_YY_R_inv_solver(const _T &decay_rate_in) {
    this->_P_YY_R_inv_solver.set_decay_rate(decay_rate_in);
  }

  /**
   * @brief Sets the minimum division value for the P_YY_R_inv_solver.
   *
   * This function updates the minimum division value used in the Unscented
   * Kalman Filter to avoid division by zero errors.
   *
   * @param division_min_in The new minimum division value to be set.
   */
  inline void
  set_division_min_for_P_YY_R_inv_solver(const _T &division_min_in) {
    this->_P_YY_R_inv_solver.set_division_min(division_min_in);
  }

public:
  /* Constant */
  static constexpr std::size_t STATE_SIZE = _STATE_SIZE;
  static constexpr std::size_t INPUT_SIZE = _INPUT_SIZE;
  static constexpr std::size_t OUTPUT_SIZE = _OUTPUT_SIZE;

  static constexpr std::size_t NUMBER_OF_DELAY = Number_Of_Delay;

public:
  /* Variable */
  Q_Type Q;
  R_Type R;
  P_Type P;
  G_Type G;

  _T kappa;
  _T alpha;
  _T beta;
  _T w_m;
  _W_Type W;

  _State_Type X_hat;
  _Kai_Type X_d;
  _InputStored_Type U_store;

  Parameter_Type parameters;

protected:
  /* Variable */
  _P_YY_R_Inv_Type _P_YY_R_inv_solver;
  _StateFunction_Object _state_function;
  _MeasurementFunction_Object _measurement_function;

  _SigmaPointsCalculator_Type _predict_sigma_points_calculator;
  _SigmaPointsCalculator_Type _update_sigma_points_calculator;
  std::size_t _input_count;
};

/* Unscented Kalman Filter Type */
template <typename U_Type, typename Q_Type, typename R_Type,
          typename Parameter_Type, std::size_t Number_Of_Delay = 0>
using UnscentedKalmanFilter_Type =
    UnscentedKalmanFilter<U_Type, Q_Type, R_Type, Parameter_Type,
                          Number_Of_Delay>;

} // namespace PythonControl

#endif // __PYTHON_CONTROL_KALMAN_FILTER_HPP__
