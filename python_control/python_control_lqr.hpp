/**
 * @file python_control_lqr.hpp
 * @brief Linear Quadratic Regulator (LQR) and Linear Quadratic Integral (LQI)
 * controller implementation in C++.
 *
 * This header provides a C++ implementation of the Linear Quadratic Regulator
 * (LQR) and Linear Quadratic Integral (LQI) controllers, inspired by the Python
 * Control library. The implementation leverages template metaprogramming for
 * compile-time matrix size and type checks, and is designed to work with a
 * custom matrix and linear algebra library (PythonNumpy namespace).
 */
#ifndef __PYTHON_CONTROL_LQR_HPP__
#define __PYTHON_CONTROL_LQR_HPP__

#include "python_numpy.hpp"

#include <type_traits>

namespace PythonControl {

namespace LQR_Operation {

constexpr std::size_t HAMILTONIAN_COLUMN_SIZE = 2;
constexpr std::size_t HAMILTONIAN_ROW_SIZE = 2;

/**
 * @brief Type alias for a matrix representing the Hamiltonian in LQR.
 *
 * This type is used to represent the Hamiltonian matrix in the LQR problem,
 * which is constructed from the system dynamics (A), input (B), cost matrices
 * (Q, R), and their respective transposes.
 *
 * @tparam T The value type of the matrix elements (e.g., float, double).
 * @tparam State_Size The size of the state vector (number of states).
 */
template <typename T, std::size_t State_Size>
using V1_V2_Type = PythonNumpy::DenseMatrix_Type<PythonNumpy::Complex<T>,
                                                 State_Size, State_Size>;

/** * @brief Type alias for the inverse solver used in LQR operations.
 *
 * This type is used to solve the inverse of the V1 and V2 matrices in the LQR
 * problem, which are derived from the eigenvectors of the Hamiltonian matrix.
 *
 * @tparam T The value type of the matrix elements (e.g., float, double).
 * @tparam State_Size The size of the state vector (number of states).
 */
template <typename T, std::size_t State_Size>
using V1_V2_InvSolver_Type =
    PythonNumpy::LinalgSolverInv_Type<V1_V2_Type<T, State_Size>>;

template <typename A_Type, typename B_Type, typename Q_Type, typename R_Type>
struct Hamiltonian {

  using B_R_INV_BT_Type = PythonNumpy::A_mul_BTranspose_Type<
      PythonNumpy::A_Multiply_B_Type<B_Type, R_Type>, B_Type>;

  using Type = PythonNumpy::ConcatenateBlock_Type<
      HAMILTONIAN_COLUMN_SIZE, HAMILTONIAN_ROW_SIZE, A_Type, B_R_INV_BT_Type,
      Q_Type, PythonNumpy::Transpose_Type<A_Type>>;
};

} // namespace LQR_Operation

/* LQR with Arimoto Potter method */

/**
 * @brief Solves the LQR problem using the Arimoto-Potter method.
 *
 * This function computes the optimal gain matrix K for the LQR problem using
 * the Arimoto-Potter method, which involves solving a Hamiltonian matrix
 * eigenvalue problem.
 *
 * @tparam A_Type The type of the state transition matrix A.
 * @tparam B_Type The type of the input matrix B.
 * @tparam Q_Type The type of the state cost matrix Q.
 * @tparam R_Type The type of the input cost matrix R.
 * @tparam K_Type The type of the output gain matrix K.
 * @tparam Hamiltonian_Type The type of the Hamiltonian matrix.
 *
 * @param A The state transition matrix A.
 * @param B The input matrix B.
 * @param Q The state cost matrix Q.
 * @param R The input cost matrix R.
 * @param K The output gain matrix K to be computed.
 * @param R_inv_solver Solver for the inverse of R.
 * @param V1_inv_solver Solver for the inverse of V1 and V2 matrices.
 * @param eig_solver Solver for eigenvalues and eigenvectors of the Hamiltonian.
 * @param eigen_solver_is_ill Flag indicating if the eigenvalue solver is
 * ill-posed.
 */
template <typename A_Type, typename B_Type, typename Q_Type, typename R_Type,
          typename K_Type, typename Hamiltonian_Type>
inline void lqr_solve_with_arimoto_potter(
    const A_Type &A, const B_Type &B, const Q_Type &Q, const R_Type &R,
    K_Type &K, PythonNumpy::LinalgSolverInv_Type<R_Type> &R_inv_solver,
    LQR_Operation::V1_V2_InvSolver_Type<typename A_Type::Value_Type,
                                        A_Type::COLS> &V1_inv_solver,
    PythonNumpy::LinalgSolverEig_Type<Hamiltonian_Type> &eig_solver,
    bool &eigen_solver_is_ill) {

  using _T = typename A_Type::Value_Type;

  static constexpr std::size_t _State_Size = A_Type::COLS;

  R_inv_solver.inv(R);
  auto R_inv = R_inv_solver.get_answer();

  auto Hamiltonian =
      PythonNumpy::concatenate_block<LQR_Operation::HAMILTONIAN_COLUMN_SIZE,
                                     LQR_Operation::HAMILTONIAN_ROW_SIZE>(
          A, PythonNumpy::A_mul_BTranspose(-B * R_inv, B), -Q, -A.transpose());

  eig_solver.solve_eigen_values(Hamiltonian);

  auto eigen_values = eig_solver.get_eigen_values();

  eig_solver.solve_eigen_vectors(Hamiltonian);

  auto eigen_vectors = eig_solver.get_eigen_vectors();

  LQR_Operation::V1_V2_Type<_T, _State_Size> V1;
  LQR_Operation::V1_V2_Type<_T, _State_Size> V2;

  std::size_t minus_count = 0;
  eigen_solver_is_ill = true;
  for (std::size_t i = 0; i < (static_cast<std::size_t>(2) * _State_Size);
       i++) {

    if (eigen_values(i, 0).real < static_cast<_T>(0)) {

      for (std::size_t j = 0; j < _State_Size; j++) {
        V1(j, minus_count) = eigen_vectors(j, i);
        V2(j, minus_count) = eigen_vectors(j + _State_Size, i);
      }

      minus_count++;
      if (_State_Size == minus_count) {
        eigen_solver_is_ill = false;
        break;
      }
    }
  }

  V1_inv_solver.inv(V1);

  auto P = (V2 * V1_inv_solver.get_answer()).real();

  K = R_inv * PythonNumpy::ATranspose_mul_B(B, P);
}

/* Linear Quadratic Regulator */

/**
 * @brief Linear Quadratic Regulator (LQR) class.
 *
 * This class implements the Linear Quadratic Regulator (LQR) control
 * algorithm, which computes the optimal state feedback gain matrix K for a
 * linear system defined by the state transition matrix A, input matrix B,
 * state cost matrix Q, and input cost matrix R. The LQR controller aims to
 * minimize a quadratic cost function that penalizes deviations from the
 * desired state and control effort.
 *
 * * @tparam A_Type_In Type of the state transition matrix A.
 * * @tparam B_Type_In Type of the input matrix B.
 * * @tparam Q_Type_In Type of the state cost matrix Q.
 * * @tparam R_Type_In Type of the input cost matrix R.
 * * The class provides methods to compute the optimal gain matrix K using the
 * Arimoto-Potter method, which involves solving a Hamiltonian matrix
 */
template <typename A_Type_In, typename B_Type_In, typename Q_Type_In,
          typename R_Type_In>
class LQR {
public:
  /* Type */
  using A_Type = A_Type_In;
  using B_Type = B_Type_In;
  using Q_Type = Q_Type_In;
  using R_Type = R_Type_In;

protected:
  /* Type */
  using _T = typename A_Type::Value_Type;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Matrix value data type must be float or double.");

  using _Hamiltonian_Type =
      typename LQR_Operation::Hamiltonian<A_Type, B_Type, Q_Type, R_Type>::Type;

protected:
  /* Constant */
  static constexpr std::size_t _Input_Size = B_Type::ROWS;
  static constexpr std::size_t _State_Size = A_Type::COLS;

public:
  /* Type */
  using Value_Type = _T;

  using K_Type = PythonNumpy::DenseMatrix_Type<_T, _Input_Size, _State_Size>;

  /* Check Compatibility */
  static_assert(PythonNumpy::Is_Diag_Matrix<Q_Type>::value,
                "Q matrix must be diagonal matrix.");

  static_assert(PythonNumpy::Is_Diag_Matrix<R_Type>::value,
                "R matrix must be diagonal matrix.");

  /* Check Data Type */
  static_assert(std::is_same<typename B_Type::Value_Type, _T>::value,
                "Data type of B matrix must be same type as A matrix.");
  static_assert(std::is_same<typename Q_Type::Value_Type, _T>::value,
                "Data type of Q matrix must be same type as A matrix.");
  static_assert(std::is_same<typename R_Type::Value_Type, _T>::value,
                "Data type of R matrix must be same type as A matrix.");

  /* Check Matrix Column and Row length */
  static_assert((A_Type::ROWS == A_Type::COLS) &&
                    (B_Type::COLS == A_Type::COLS) &&
                    (Q_Type::ROWS == Q_Type::COLS) &&
                    (Q_Type::ROWS == A_Type::COLS) &&
                    (R_Type::ROWS == R_Type::COLS) &&
                    (R_Type::ROWS == B_Type::ROWS),
                "A, B, Q, R matrix size is not compatible");

public:
  /* Constructor */
  LQR()
      : _A(), _B(), _Q(), _R(), _K(), _R_inv_solver(), _V1_inv_solver(),
        eig_solver(), _eigen_solver_is_ill(false) {}

  LQR(const A_Type &A, const B_Type &B, const Q_Type &Q, const R_Type &R)
      : _A(A), _B(B), _Q(Q), _R(R), _K(), _R_inv_solver(), _V1_inv_solver(),
        eig_solver(), _eigen_solver_is_ill(false) {}

  /* Copy Constructor */
  LQR(const LQR<A_Type, B_Type, Q_Type, R_Type> &input)
      : _A(input._A), _B(input._B), _Q(input._Q), _R(input._R), _K(input._K),
        _R_inv_solver(input._R_inv_solver),
        _V1_inv_solver(input._V1_inv_solver), eig_solver(input.eig_solver),
        _eigen_solver_is_ill(input._eigen_solver_is_ill) {}

  LQR<A_Type, B_Type, Q_Type, R_Type> &
  operator=(const LQR<A_Type, B_Type, Q_Type, R_Type> &input) {
    if (this != &input) {
      this->_A = input._A;
      this->_B = input._B;
      this->_Q = input._Q;
      this->_R = input._R;
      this->_K = input._K;
      this->_R_inv_solver = input._R_inv_solver;
      this->_V1_inv_solver = input._V1_inv_solver;
      this->eig_solver = input.eig_solver;
      this->_eigen_solver_is_ill = input._eigen_solver_is_ill;
    }
    return *this;
  }

  /* Move Constructor */
  LQR(LQR<A_Type, B_Type, Q_Type, R_Type> &&input)
  noexcept
      : _A(std::move(input._A)), _B(std::move(input._B)),
        _Q(std::move(input._Q)), _R(std::move(input._R)),
        _K(std::move(input._K)), _R_inv_solver(std::move(input._R_inv_solver)),
        _V1_inv_solver(std::move(input._V1_inv_solver)),
        eig_solver(std::move(input.eig_solver)),
        _eigen_solver_is_ill(std::move(input._eigen_solver_is_ill)) {}

  LQR<A_Type, B_Type, Q_Type, R_Type> &
  operator=(LQR<A_Type, B_Type, Q_Type, R_Type> &&input) noexcept {
    if (this != &input) {
      this->_A = std::move(input._A);
      this->_B = std::move(input._B);
      this->_Q = std::move(input._Q);
      this->_R = std::move(input._R);
      this->_K = std::move(input._K);
      this->_R_inv_solver = std::move(input._R_inv_solver);
      this->_V1_inv_solver = std::move(input._V1_inv_solver);
      this->eig_solver = std::move(input.eig_solver);
      this->_eigen_solver_is_ill = std::move(input._eigen_solver_is_ill);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Solves the LQR problem using the Arimoto-Potter method.
   *
   * This function computes the optimal gain matrix K for the LQR problem using
   * the Arimoto-Potter method, which involves solving a Hamiltonian matrix
   * eigenvalue problem.
   *
   * @return The computed gain matrix K.
   */
  inline K_Type solve(void) {

    PythonControl::lqr_solve_with_arimoto_potter<A_Type, B_Type, Q_Type, R_Type,
                                                 K_Type, _Hamiltonian_Type>(
        this->_A, this->_B, this->_Q, this->_R, this->_K, this->_R_inv_solver,
        this->_V1_inv_solver, this->eig_solver, this->_eigen_solver_is_ill);

    return this->_K;
  }

  /**
   * @brief Computes the gain matrix K using the Arimoto-Potter method.
   *
   * This function computes the gain matrix K for the LQR problem using the
   * Arimoto-Potter method, which involves solving a Hamiltonian matrix
   * eigenvalue problem.
   *
   * @return The computed gain matrix K.
   */
  inline K_Type get_K() const { return this->_K; }

  /**
   * @brief Checks if the eigenvalue solver is ill-posed.
   *
   * This function returns a boolean indicating whether the eigenvalue solver
   * encountered an ill-posed problem during the computation of the gain matrix
   * K.
   *
   * @return True if the eigenvalue solver is ill-posed, false otherwise.
   */
  inline bool get_eigen_solver_is_ill() const {
    return this->_eigen_solver_is_ill;
  }

  /**
   * @brief Returns the state transition matrix A.
   *
   * This function returns the state transition matrix A used in the LQR
   * computation.
   *
   * @return The state transition matrix A.
   */
  inline void set_A(const A_Type &A) { this->_A = A; }

  /**
   * @brief Returns the input matrix B.
   *
   * This function returns the input matrix B used in the LQR computation.
   *
   * @return The input matrix B.
   */
  inline void set_B(const B_Type &B) { this->_B = B; }

  /**
   * @brief Returns the state cost matrix Q.
   *
   * This function returns the state cost matrix Q used in the LQR computation.
   *
   * @return The state cost matrix Q.
   */
  inline void set_Q(const Q_Type &Q) { this->_Q = Q; }

  /**
   * @brief Returns the input cost matrix R.
   *
   * This function returns the input cost matrix R used in the LQR computation.
   *
   * @return The input cost matrix R.
   */
  inline void set_R(const R_Type &R) { this->_R = R; }

  /**
   * @brief Returns the gain matrix K.
   *
   * This function returns the gain matrix K computed by the LQR algorithm.
   *
   * @return The gain matrix K.
   */
  inline void set_R_inv_division_min(const _T &division_min_in) {
    this->_R_inv_solver.set_division_min(division_min_in);
  }

  /**
   * @brief Returns the inverse solver for V1 and V2 matrices.
   *
   * This function returns the inverse solver used for the V1 and V2 matrices
   * in the LQR computation.
   *
   * @return The inverse solver for V1 and V2 matrices.
   */
  inline void set_V1_inv_decay_rate(const _T &decay_rate_in) {
    this->_V1_inv_solver.set_decay_rate(decay_rate_in);
  }

  /**
   * @brief Sets the division minimum for the inverse solver of V1 and V2
   * matrices.
   *
   * This function updates the minimum division value used in the inverse
   * solver to avoid division by zero errors during the LQR computation.
   *
   * @param division_min_in The new minimum division value to be set.
   */
  inline void set_V1_inv_division_min(const _T &division_min_in) {
    this->_V1_inv_solver.set_division_min(division_min_in);
  }

  /**
   * @brief Sets the maximum number of iterations for the eigenvalue solver.
   *
   * This function updates the maximum number of iterations allowed for the
   * eigenvalue solver used in the LQR computation.
   *
   * @param iteration_max The new maximum number of iterations to be set.
   */
  inline void set_Eigen_solver_iteration_max(const std::size_t &iteration_max) {
    this->eig_solver.set_iteration_max(iteration_max);
  }

  /**
   * @brief Sets the maximum number of iterations for the eigenvector solver.
   *
   * This function updates the maximum number of iterations allowed for the
   * eigenvector solver used in the LQR computation.
   *
   * @param iteration_max_for_eigen_vector The new maximum number of iterations
   * to be set for the eigenvector solver.
   */
  inline void set_Eigen_solver_iteration_max_for_eigen_vector(
      const std::size_t &iteration_max_for_eigen_vector) {
    this->eig_solver.set_iteration_max_for_eigen_vector(
        iteration_max_for_eigen_vector);
  }

  /**
   * @brief Sets the minimum division value for the eigenvalue solver.
   *
   * This function updates the minimum division value used in the eigenvalue
   * solver to avoid division by zero errors during the LQR computation.
   *
   * @param division_min_in The new minimum division value to be set.
   */
  inline void set_Eigen_solver_division_min(const _T &division_min_in) {
    this->eig_solver.set_division_min(division_min_in);
  }

  /**
   * @brief Sets the small value for the eigenvalue solver.
   *
   * This function updates the small value used in the eigenvalue solver to
   * handle numerical stability issues during the LQR computation.
   *
   * @param small_value_in The new small value to be set.
   */
  inline void set_Eigen_solver_small_value(const _T &small_value_in) {
    this->eig_solver.set_small_value(small_value_in);
  }

protected:
  /* Variable */
  A_Type _A;
  B_Type _B;
  Q_Type _Q;
  R_Type _R;
  K_Type _K;

  PythonNumpy::LinalgSolverInv_Type<R_Type> _R_inv_solver;
  LQR_Operation::V1_V2_InvSolver_Type<_T, _State_Size> _V1_inv_solver;
  PythonNumpy::LinalgSolverEig_Type<_Hamiltonian_Type> eig_solver;

  bool _eigen_solver_is_ill;
};

/* Make LQR */

/**
 * @brief Factory function to create an instance of the LQR class.
 *
 * This function constructs an LQR controller using the provided state
 * transition matrix A, input matrix B, state cost matrix Q, and input cost
 * matrix R. It returns an instance of the LQR class with these parameters.
 *
 * @param A The state transition matrix A.
 * @param B The input matrix B.
 * @param Q The state cost matrix Q.
 * @param R The input cost matrix R.
 * @return An instance of the LQR class initialized with the provided matrices.
 */
template <typename A_Type, typename B_Type, typename Q_Type, typename R_Type>
inline auto make_LQR(const A_Type &A, const B_Type &B, const Q_Type &Q,
                     const R_Type &R) -> LQR<A_Type, B_Type, Q_Type, R_Type> {

  return LQR<A_Type, B_Type, Q_Type, R_Type>(A, B, Q, R);
}

/* LQR Type */
template <typename A_Type, typename B_Type, typename Q_Type, typename R_Type>
using LQR_Type = LQR<A_Type, B_Type, Q_Type, R_Type>;

/* Linear Quadratic optimum control with Integral action */

/**
 * @brief Linear Quadratic Integral (LQI) class.
 *
 * This class implements the Linear Quadratic Integral (LQI) control
 * algorithm, which extends the Linear Quadratic Regulator (LQR) by
 * including an integral action to eliminate steady-state errors in the
 * control system. The LQI controller computes the optimal state feedback gain
 * matrix K for a linear system defined by the state transition matrix A,
 * input matrix B, state cost matrix Q, input cost matrix R, and output
 * matrix C. The LQI controller aims to minimize a quadratic cost function
 * that penalizes deviations from the desired state, control effort, and
 * output error.
 *
 * @tparam A_Type_In Type of the state transition matrix A.
 * @tparam B_Type_In Type of the input matrix B.
 * @tparam C_Type_In Type of the output matrix C.
 * @tparam Q_Type_In Type of the state cost matrix Q.
 * @tparam R_Type_In Type of the input cost matrix R.
 * The class provides methods to compute the optimal gain matrix K using the
 * Arimoto-Potter method, which involves solving a Hamiltonian matrix
 * eigenvalue problem.
 */
template <typename A_Type_In, typename B_Type_In, typename C_Type_In,
          typename Q_Type_In, typename R_Type_In>
class LQI {
public:
  /* Type */
  using A_Type = A_Type_In;
  using B_Type = B_Type_In;
  using C_Type = C_Type_In;
  using Q_Type = Q_Type_In;
  using R_Type = R_Type_In;

protected:
  /* Type */
  using _T = typename A_Type::Value_Type;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Matrix value data type must be float or double.");

  using _A_EX_Type = PythonNumpy::ConcatenateHorizontally_Type<
      PythonNumpy::ConcatenateVertically_Type<A_Type, C_Type>,
      PythonNumpy::SparseMatrixEmpty_Type<_T, (A_Type::COLS + C_Type::COLS),
                                          C_Type::COLS>>;

  using _B_EX_Type = PythonNumpy::ConcatenateVertically_Type<
      B_Type,
      PythonNumpy::SparseMatrixEmpty_Type<_T, C_Type::COLS, B_Type::ROWS>>;

  using _Hamiltonian_Type =
      typename LQR_Operation::Hamiltonian<_A_EX_Type, _B_EX_Type, Q_Type,
                                          R_Type>::Type;

protected:
  /* Constant */
  static constexpr std::size_t _Input_Size = B_Type::ROWS;
  static constexpr std::size_t _State_Size = A_Type::COLS;
  static constexpr std::size_t _Output_Size = C_Type::COLS;

public:
  /* Type */
  using Value_Type = _T;

  using K_Type = PythonNumpy::DenseMatrix_Type<_T, _Input_Size,
                                               (_State_Size + _Output_Size)>;

  /* Check Compatibility */
  static_assert(PythonNumpy::Is_Diag_Matrix<Q_Type>::value,
                "Q matrix must be diagonal matrix.");

  static_assert(PythonNumpy::Is_Diag_Matrix<R_Type>::value,
                "R matrix must be diagonal matrix.");

  /* Check Data Type */
  static_assert(std::is_same<typename B_Type::Value_Type, _T>::value,
                "Data type of B matrix must be same type as A matrix.");
  static_assert(std::is_same<typename C_Type::Value_Type, _T>::value,
                "Data type of C matrix must be same type as A matrix.");
  static_assert(std::is_same<typename Q_Type::Value_Type, _T>::value,
                "Data type of Q matrix must be same type as A matrix.");
  static_assert(std::is_same<typename R_Type::Value_Type, _T>::value,
                "Data type of R matrix must be same type as A matrix.");

  /* Check Matrix Column and Row length */
  static_assert((A_Type::ROWS == A_Type::COLS) &&
                    (B_Type::COLS == A_Type::COLS) &&
                    (C_Type::ROWS == A_Type::COLS) &&
                    (Q_Type::ROWS == Q_Type::COLS) &&
                    (Q_Type::ROWS == (A_Type::COLS + C_Type::COLS)) &&
                    (R_Type::ROWS == R_Type::COLS) &&
                    (R_Type::ROWS == B_Type::ROWS),
                "A, B, C, Q, R matrix size is not compatible");

public:
  /* Constructor */
  LQI()
      : _A(), _B(), _C(), _Q(), _R(), _K(), _R_inv_solver(), _V1_inv_solver(),
        _eig_solver(), _eigen_solver_is_ill(false) {}

  LQI(const A_Type &A, const B_Type &B, const C_Type &C, const Q_Type &Q,
      const R_Type &R)
      : _A(A), _B(B), _C(C), _Q(Q), _R(R), _K(), _R_inv_solver(),
        _V1_inv_solver(), _eig_solver(), _eigen_solver_is_ill(false) {}

  /* Copy Constructor */
  LQI(const LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &input)
      : _A(input._A), _B(input._B), _C(input._C), _Q(input._Q), _R(input._R),
        _K(input._K), _R_inv_solver(input._R_inv_solver),
        _V1_inv_solver(input._V1_inv_solver), _eig_solver(input._eig_solver),
        _eigen_solver_is_ill(input._eigen_solver_is_ill) {}

  LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &
  operator=(const LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &input) {
    if (this != &input) {
      this->_A = input._A;
      this->_B = input._B;
      this->_C = input._C;
      this->_Q = input._Q;
      this->_R = input._R;
      this->_K = input._K;
      this->_R_inv_solver = input._R_inv_solver;
      this->_V1_inv_solver = input._V1_inv_solver;
      this->_eig_solver = input._eig_solver;
      this->_eigen_solver_is_ill = input._eigen_solver_is_ill;
    }
    return *this;
  }

  /* Move Constructor */
  LQI(LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &&input)
  noexcept
      : _A(std::move(input._A)), _B(std::move(input._B)),
        _C(std::move(input._C)), _Q(std::move(input._Q)),
        _R(std::move(input._R)), _K(std::move(input._K)),
        _R_inv_solver(std::move(input._R_inv_solver)),
        _V1_inv_solver(std::move(input._V1_inv_solver)),
        _eig_solver(std::move(input._eig_solver)),
        _eigen_solver_is_ill(std::move(input._eigen_solver_is_ill)) {}

  LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &
  operator=(LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &&input) noexcept {
    if (this != &input) {
      this->_A = std::move(input._A);
      this->_B = std::move(input._B);
      this->_C = std::move(input._C);
      this->_Q = std::move(input._Q);
      this->_R = std::move(input._R);
      this->_K = std::move(input._K);
      this->_R_inv_solver = std::move(input._R_inv_solver);
      this->_V1_inv_solver = std::move(input._V1_inv_solver);
      this->_eig_solver = std::move(input._eig_solver);
      this->_eigen_solver_is_ill = std::move(input._eigen_solver_is_ill);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Solves the LQI problem using the Arimoto-Potter method.
   *
   * This function computes the optimal gain matrix K for the LQI problem using
   * the Arimoto-Potter method, which involves solving a Hamiltonian matrix
   * eigenvalue problem.
   *
   * @return The computed gain matrix K.
   */
  inline K_Type solve(void) {

    auto A_ex = PythonNumpy::concatenate_horizontally(
        PythonNumpy::concatenate_vertically(this->_A, this->_C),
        PythonNumpy::make_SparseMatrixEmpty<_T, (_State_Size + _Output_Size),
                                            _Output_Size>());

    auto B_ex = PythonNumpy::concatenate_vertically(
        this->_B,
        PythonNumpy::make_SparseMatrixEmpty<_T, _Output_Size, _Input_Size>());

    PythonControl::lqr_solve_with_arimoto_potter<
        _A_EX_Type, _B_EX_Type, Q_Type, R_Type, K_Type, _Hamiltonian_Type>(
        A_ex, B_ex, this->_Q, this->_R, this->_K, this->_R_inv_solver,
        this->_V1_inv_solver, this->_eig_solver, this->_eigen_solver_is_ill);

    return this->_K;
  }

  /**
   * @brief Computes the gain matrix K using the Arimoto-Potter method.
   *
   * This function computes the gain matrix K for the LQI problem using the
   * Arimoto-Potter method, which involves solving a Hamiltonian matrix
   * eigenvalue problem.
   *
   * @return The computed gain matrix K.
   */
  inline K_Type get_K() const { return this->_K; }

  /**
   * @brief Checks if the eigenvalue solver is ill-posed.
   *
   * This function returns a boolean indicating whether the eigenvalue solver
   * encountered an ill-posed problem during the computation of the gain matrix
   * K.
   *
   * @return True if the eigenvalue solver is ill-posed, false otherwise.
   */
  inline bool get_eigen_solver_is_ill() const {
    return this->_eigen_solver_is_ill;
  }

  /**
   * @brief Sets the state transition matrix A.
   *
   * This function updates the state transition matrix A used in the LQI
   * computation.
   *
   * @param A The new state transition matrix to be set.
   */
  inline void set_A(const A_Type &A) { this->_A = A; }

  /**
   * @brief Sets the input matrix B.
   *
   * This function updates the input matrix B used in the LQI computation.
   *
   * @param B The new input matrix to be set.
   */
  inline void set_B(const B_Type &B) { this->_B = B; }

  /**
   * @brief Sets the output matrix C.
   *
   * This function updates the output matrix C used in the LQI computation.
   *
   * @param C The new output matrix to be set.
   */
  inline void set_C(const C_Type &C) { this->_C = C; }

  /**
   * @brief Sets the state cost matrix Q.
   *
   * This function updates the state cost matrix Q used in the LQI computation.
   *
   * @param Q The new state cost matrix to be set.
   */
  inline void set_Q(const Q_Type &Q) { this->_Q = Q; }

  /**
   * @brief Sets the input cost matrix R.
   *
   * This function updates the input cost matrix R used in the LQI computation.
   *
   * @param R The new input cost matrix to be set.
   */
  inline void set_R(const R_Type &R) { this->_R = R; }

  /**
   * @brief Sets the division minimum for the inverse solver of R.
   *
   * This function updates the minimum division value used in the inverse
   * solver to avoid division by zero errors during the LQI computation.
   *
   * @param division_min_in The new minimum division value to be set.
   */
  inline void set_R_inv_division_min(const _T &division_min_in) {
    this->_R_inv_solver.set_division_min(division_min_in);
  }

  /**
   * @brief Sets the decay rate for the inverse solver of V1 and V2 matrices.
   *
   * This function updates the decay rate used in the inverse solver to
   * control the convergence behavior during the LQI computation.
   *
   * @param decay_rate_in The new decay rate to be set.
   */
  inline void set_V1_inv_decay_rate(const _T &decay_rate_in) {
    this->_V1_inv_solver.set_decay_rate(decay_rate_in);
  }

  /**
   * @brief Sets the division minimum for the inverse solver of V1 and V2
   * matrices.
   *
   * This function updates the minimum division value used in the inverse
   * solver to avoid division by zero errors during the LQI computation.
   *
   * @param division_min_in The new minimum division value to be set.
   */
  inline void set_V1_inv_division_min(const _T &division_min_in) {
    this->_V1_inv_solver.set_division_min(division_min_in);
  }

  /**
   * @brief Sets the maximum number of iterations for the eigenvalue solver.
   *
   * This function updates the maximum number of iterations allowed for the
   * eigenvalue solver used in the LQI computation.
   *
   * @param iteration_max The new maximum number of iterations to be set.
   */
  inline void set_Eigen_solver_iteration_max(const std::size_t &iteration_max) {
    this->_eig_solver.set_iteration_max(iteration_max);
  }

  /**
   * @brief Sets the maximum number of iterations for the eigenvector solver.
   *
   * This function updates the maximum number of iterations allowed for the
   * eigenvector solver used in the LQI computation.
   *
   * @param iteration_max_for_eigen_vector The new maximum number of iterations
   * to be set for the eigenvector solver.
   */
  inline void set_Eigen_solver_iteration_max_for_eigen_vector(
      const std::size_t &iteration_max_for_eigen_vector) {
    this->_eig_solver.set_iteration_max_for_eigen_vector(
        iteration_max_for_eigen_vector);
  }

  /**
   * @brief Sets the minimum division value for the eigenvalue solver.
   *
   * This function updates the minimum division value used in the eigenvalue
   * solver to avoid division by zero errors during the LQI computation.
   *
   * @param division_min_in The new minimum division value to be set.
   */
  inline void set_Eigen_solver_division_min(const _T &division_min_in) {
    this->_eig_solver.set_division_min(division_min_in);
  }

  /**
   * @brief Sets the small value for the eigenvalue solver.
   *
   * This function updates the small value used in the eigenvalue solver to
   * handle numerical stability issues during the LQI computation.
   *
   * @param small_value_in The new small value to be set.
   */
  inline void set_Eigen_solver_small_value(const _T &small_value_in) {
    this->_eig_solver.set_small_value(small_value_in);
  }

protected:
  /* Variable */
  A_Type _A;
  B_Type _B;
  C_Type _C;
  Q_Type _Q;
  R_Type _R;
  K_Type _K;

  PythonNumpy::LinalgSolverInv_Type<R_Type> _R_inv_solver;
  LQR_Operation::V1_V2_InvSolver_Type<_T, (_State_Size + _Output_Size)>
      _V1_inv_solver;
  PythonNumpy::LinalgSolverEig_Type<_Hamiltonian_Type> _eig_solver;

  bool _eigen_solver_is_ill;
};

/* Make LQI */

/**
 * @brief Factory function to create an instance of the LQI class.
 *
 * This function constructs an LQI controller using the provided state
 * transition matrix A, input matrix B, output matrix C, state cost matrix Q,
 * and input cost matrix R. It returns an instance of the LQI class with
 * these parameters.
 *
 * @param A The state transition matrix A.
 * @param B The input matrix B.
 * @param C The output matrix C.
 * @param Q The state cost matrix Q.
 * @param R The input cost matrix R.
 * @return An instance of the LQI class initialized with the provided matrices.
 */
template <typename A_Type, typename B_Type, typename C_Type, typename Q_Type,
          typename R_Type>
inline auto make_LQI(const A_Type &A, const B_Type &B, const C_Type &C,
                     const Q_Type &Q, const R_Type &R)
    -> LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> {

  return LQI<A_Type, B_Type, C_Type, Q_Type, R_Type>(A, B, C, Q, R);
}

/* LQI Type */
template <typename A_Type, typename B_Type, typename C_Type, typename Q_Type,
          typename R_Type>
using LQI_Type = LQI<A_Type, B_Type, C_Type, Q_Type, R_Type>;

} // namespace PythonControl

#endif // __PYTHON_CONTROL_LQR_HPP__
