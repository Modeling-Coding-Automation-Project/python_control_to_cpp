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
#ifndef PYTHON_CONTROL_LQR_HPP_
#define PYTHON_CONTROL_LQR_HPP_

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

/* LQR Arimoto Potter Solver */

/**
 * @brief Solver class for the LQR problem using the Arimoto-Potter method.
 *
 * This class encapsulates the Arimoto-Potter method for solving the LQR
 * problem. It computes the optimal gain matrix K by constructing and solving
 * a Hamiltonian matrix eigenvalue problem.
 *
 * @tparam A_Type The type of the state transition matrix A.
 * @tparam B_Type The type of the input matrix B.
 * @tparam Q_Type The type of the state cost matrix Q.
 * @tparam R_Type The type of the input cost matrix R.
 */
template <typename A_Type, typename B_Type, typename Q_Type, typename R_Type>
class LQR_ArimotoPotterSolver {
protected:
  /* Type */
  using T_ = typename A_Type::Value_Type;

  using Hamiltonian_Type_ =
      typename LQR_Operation::Hamiltonian<A_Type, B_Type, Q_Type, R_Type>::Type;

protected:
  /* Constant */
  static constexpr std::size_t Input_Size_ = B_Type::COLS;
  static constexpr std::size_t State_Size_ = A_Type::ROWS;

public:
  /* Type */
  using Value_Type = T_;

  using K_Type = PythonNumpy::DenseMatrix_Type<T_, Input_Size_, State_Size_>;

public:
  /* Constructor */
  LQR_ArimotoPotterSolver()
      : K_(), R_inv_solver_(), V1_inv_solver_(), _eig_solver(),
        _eigen_solver_is_ill(false) {}

  /* Copy Constructor */
  LQR_ArimotoPotterSolver(
      const LQR_ArimotoPotterSolver<A_Type, B_Type, Q_Type, R_Type> &input)
      : K_(input.K_), R_inv_solver_(input.R_inv_solver_),
        V1_inv_solver_(input.V1_inv_solver_), _eig_solver(input._eig_solver),
        _eigen_solver_is_ill(input._eigen_solver_is_ill) {}

  LQR_ArimotoPotterSolver<A_Type, B_Type, Q_Type, R_Type> &operator=(
      const LQR_ArimotoPotterSolver<A_Type, B_Type, Q_Type, R_Type> &input) {
    if (this != &input) {
      this->K_ = input.K_;
      this->R_inv_solver_ = input.R_inv_solver_;
      this->V1_inv_solver_ = input.V1_inv_solver_;
      this->_eig_solver = input._eig_solver;
      this->_eigen_solver_is_ill = input._eigen_solver_is_ill;
    }
    return *this;
  }

  /* Move Constructor */
  LQR_ArimotoPotterSolver(
      LQR_ArimotoPotterSolver<A_Type, B_Type, Q_Type, R_Type> &&input) noexcept
      : K_(std::move(input.K_)), R_inv_solver_(std::move(input.R_inv_solver_)),
        V1_inv_solver_(std::move(input.V1_inv_solver_)),
        _eig_solver(std::move(input._eig_solver)),
        _eigen_solver_is_ill(std::move(input._eigen_solver_is_ill)) {}

  LQR_ArimotoPotterSolver<A_Type, B_Type, Q_Type, R_Type> &
  operator=(LQR_ArimotoPotterSolver<A_Type, B_Type, Q_Type, R_Type>
                &&input) noexcept {
    if (this != &input) {
      this->K_ = std::move(input.K_);
      this->R_inv_solver_ = std::move(input.R_inv_solver_);
      this->V1_inv_solver_ = std::move(input.V1_inv_solver_);
      this->_eig_solver = std::move(input._eig_solver);
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
   * @param A The state transition matrix A.
   * @param B The input matrix B.
   * @param Q The state cost matrix Q.
   * @param R The input cost matrix R.
   * @return The computed gain matrix K.
   */
  inline auto solve(const A_Type &A, const B_Type &B, const Q_Type &Q,
                    const R_Type &R) -> K_Type {

    this->R_inv_solver_.inv(R);
    auto R_inv = this->R_inv_solver_.get_answer();

    auto Hamiltonian =
        PythonNumpy::concatenate_block<LQR_Operation::HAMILTONIAN_COLUMN_SIZE,
                                       LQR_Operation::HAMILTONIAN_ROW_SIZE>(
            A, PythonNumpy::A_mul_BTranspose(-B * R_inv, B), -Q,
            -A.transpose());

    this->_eig_solver.solve_eigen_values(Hamiltonian);

    auto eigen_values = this->_eig_solver.get_eigen_values();

    this->_eig_solver.solve_eigen_vectors(Hamiltonian);

    auto eigen_vectors = this->_eig_solver.get_eigen_vectors();

    LQR_Operation::V1_V2_Type<T_, State_Size_> V1;
    LQR_Operation::V1_V2_Type<T_, State_Size_> V2;

    std::size_t minus_count = 0;
    this->_eigen_solver_is_ill = true;
    for (std::size_t i = 0; i < (static_cast<std::size_t>(2) * State_Size_);
         i++) {

      if (eigen_values(i, 0).real < static_cast<T_>(0)) {

        for (std::size_t j = 0; j < State_Size_; j++) {
          V1(j, minus_count) = eigen_vectors(j, i);
          V2(j, minus_count) = eigen_vectors(j + State_Size_, i);
        }

        minus_count++;
        if (State_Size_ == minus_count) {
          this->_eigen_solver_is_ill = false;
          break;
        }
      }
    }

    this->V1_inv_solver_.inv(V1);

    auto P = (V2 * this->V1_inv_solver_.get_answer()).real();

    this->K_ = R_inv * PythonNumpy::ATranspose_mul_B(B, P);

    return this->K_;
  }

  /**
   * @brief Returns the computed gain matrix K.
   *
   * @return The gain matrix K.
   */
  inline auto get_K() const -> K_Type { return this->K_; }

  /**
   * @brief Checks if the eigenvalue solver is ill-posed.
   *
   * @return True if the eigenvalue solver is ill-posed, false otherwise.
   */
  inline auto get_eigen_solver_is_ill() const -> bool {
    return this->_eigen_solver_is_ill;
  }

  /**
   * @brief Sets the division minimum for the inverse solver of R.
   *
   * @param division_min_in The new minimum division value to be set.
   */
  inline void set_R_inv_division_min(const T_ &division_min_in) {
    this->R_inv_solver_.set_division_min(division_min_in);
  }

  /**
   * @brief Sets the decay rate for the inverse solver of V1 and V2 matrices.
   *
   * @param decay_rate_in The new decay rate to be set.
   */
  inline void set_V1_inv_decay_rate(const T_ &decay_rate_in) {
    this->V1_inv_solver_.set_decay_rate(decay_rate_in);
  }

  /**
   * @brief Sets the division minimum for the inverse solver of V1 and V2
   * matrices.
   *
   * @param division_min_in The new minimum division value to be set.
   */
  inline void set_V1_inv_division_min(const T_ &division_min_in) {
    this->V1_inv_solver_.set_division_min(division_min_in);
  }

  /**
   * @brief Sets the maximum number of iterations for the eigenvalue solver.
   *
   * @param iteration_max The new maximum number of iterations to be set.
   */
  inline void set_Eigen_solver_iteration_max(const std::size_t &iteration_max) {
    this->_eig_solver.set_iteration_max(iteration_max);
  }

  /**
   * @brief Sets the maximum number of iterations for the eigenvector solver.
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
   * @param division_min_in The new minimum division value to be set.
   */
  inline void set_Eigen_solver_division_min(const T_ &division_min_in) {
    this->_eig_solver.set_division_min(division_min_in);
  }

  /**
   * @brief Sets the small value for the eigenvalue solver.
   *
   * @param small_value_in The new small value to be set.
   */
  inline void set_Eigen_solver_small_value(const T_ &small_value_in) {
    this->_eig_solver.set_small_value(small_value_in);
  }

protected:
  /* Variable */
  K_Type K_;

  PythonNumpy::LinalgSolverInv_Type<R_Type> R_inv_solver_;
  LQR_Operation::V1_V2_InvSolver_Type<T_, State_Size_> V1_inv_solver_;
  PythonNumpy::LinalgSolverEig_Type<Hamiltonian_Type_> _eig_solver;

  bool _eigen_solver_is_ill;
};

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
  using T_ = typename A_Type::Value_Type;
  static_assert(std::is_same<T_, double>::value ||
                    std::is_same<T_, float>::value,
                "Matrix value data type must be float or double.");

  using Solver_Type_ = LQR_ArimotoPotterSolver<A_Type, B_Type, Q_Type, R_Type>;

protected:
  /* Constant */
  static constexpr std::size_t Input_Size_ = B_Type::COLS;
  static constexpr std::size_t State_Size_ = A_Type::ROWS;

public:
  /* Type */
  using Value_Type = T_;

  using K_Type = PythonNumpy::DenseMatrix_Type<T_, Input_Size_, State_Size_>;

  /* Check Compatibility */
  static_assert(PythonNumpy::Is_Diag_Matrix<Q_Type>::value,
                "Q matrix must be diagonal matrix.");

  static_assert(PythonNumpy::Is_Diag_Matrix<R_Type>::value,
                "R matrix must be diagonal matrix.");

  /* Check Data Type */
  static_assert(std::is_same<typename B_Type::Value_Type, T_>::value,
                "Data type of B matrix must be same type as A matrix.");
  static_assert(std::is_same<typename Q_Type::Value_Type, T_>::value,
                "Data type of Q matrix must be same type as A matrix.");
  static_assert(std::is_same<typename R_Type::Value_Type, T_>::value,
                "Data type of R matrix must be same type as A matrix.");

  /* Check Matrix Column and Row length */
  static_assert((A_Type::COLS == A_Type::ROWS) &&
                    (B_Type::ROWS == A_Type::ROWS) &&
                    (Q_Type::COLS == Q_Type::ROWS) &&
                    (Q_Type::COLS == A_Type::ROWS) &&
                    (R_Type::COLS == R_Type::ROWS) &&
                    (R_Type::COLS == B_Type::COLS),
                "A, B, Q, R matrix size is not compatible");

public:
  /* Constructor */
  LQR() : A_(), B_(), Q_(), R_(), K_(), _arimoto_potter_solver() {}

  LQR(const A_Type &A, const B_Type &B, const Q_Type &Q, const R_Type &R)
      : A_(A), B_(B), Q_(Q), R_(R), K_(), _arimoto_potter_solver() {}

  /* Copy Constructor */
  LQR(const LQR<A_Type, B_Type, Q_Type, R_Type> &input)
      : A_(input.A_), B_(input.B_), Q_(input.Q_), R_(input.R_), K_(input.K_),
        _arimoto_potter_solver(input._arimoto_potter_solver) {}

  LQR<A_Type, B_Type, Q_Type, R_Type> &
  operator=(const LQR<A_Type, B_Type, Q_Type, R_Type> &input) {
    if (this != &input) {
      this->A_ = input.A_;
      this->B_ = input.B_;
      this->Q_ = input.Q_;
      this->R_ = input.R_;
      this->K_ = input.K_;
      this->_arimoto_potter_solver = input._arimoto_potter_solver;
    }
    return *this;
  }

  /* Move Constructor */
  LQR(LQR<A_Type, B_Type, Q_Type, R_Type> &&input) noexcept
      : A_(std::move(input.A_)), B_(std::move(input.B_)),
        Q_(std::move(input.Q_)), R_(std::move(input.R_)),
        K_(std::move(input.K_)),
        _arimoto_potter_solver(std::move(input._arimoto_potter_solver)) {}

  LQR<A_Type, B_Type, Q_Type, R_Type> &
  operator=(LQR<A_Type, B_Type, Q_Type, R_Type> &&input) noexcept {
    if (this != &input) {
      this->A_ = std::move(input.A_);
      this->B_ = std::move(input.B_);
      this->Q_ = std::move(input.Q_);
      this->R_ = std::move(input.R_);
      this->K_ = std::move(input.K_);
      this->_arimoto_potter_solver = std::move(input._arimoto_potter_solver);
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

    this->K_ = this->_arimoto_potter_solver.solve(this->A_, this->B_, this->Q_,
                                                  this->R_);

    return this->K_;
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
  inline K_Type get_K() const { return this->K_; }

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
    return this->_arimoto_potter_solver.get_eigen_solver_is_ill();
  }

  /**
   * @brief Returns the state transition matrix A.
   *
   * This function returns the state transition matrix A used in the LQR
   * computation.
   *
   * @return The state transition matrix A.
   */
  inline void set_A(const A_Type &A) { this->A_ = A; }

  /**
   * @brief Returns the input matrix B.
   *
   * This function returns the input matrix B used in the LQR computation.
   *
   * @return The input matrix B.
   */
  inline void set_B(const B_Type &B) { this->B_ = B; }

  /**
   * @brief Returns the state cost matrix Q.
   *
   * This function returns the state cost matrix Q used in the LQR computation.
   *
   * @return The state cost matrix Q.
   */
  inline void set_Q(const Q_Type &Q) { this->Q_ = Q; }

  /**
   * @brief Returns the input cost matrix R.
   *
   * This function returns the input cost matrix R used in the LQR computation.
   *
   * @return The input cost matrix R.
   */
  inline void set_R(const R_Type &R) { this->R_ = R; }

  /**
   * @brief Returns the gain matrix K.
   *
   * This function returns the gain matrix K computed by the LQR algorithm.
   *
   * @return The gain matrix K.
   */
  inline void set_R_inv_division_min(const T_ &division_min_in) {
    this->_arimoto_potter_solver.set_R_inv_division_min(division_min_in);
  }

  /**
   * @brief Returns the inverse solver for V1 and V2 matrices.
   *
   * This function returns the inverse solver used for the V1 and V2 matrices
   * in the LQR computation.
   *
   * @return The inverse solver for V1 and V2 matrices.
   */
  inline void set_V1_inv_decay_rate(const T_ &decay_rate_in) {
    this->_arimoto_potter_solver.set_V1_inv_decay_rate(decay_rate_in);
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
  inline void set_V1_inv_division_min(const T_ &division_min_in) {
    this->_arimoto_potter_solver.set_V1_inv_division_min(division_min_in);
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
    this->_arimoto_potter_solver.set_Eigen_solver_iteration_max(iteration_max);
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
    this->_arimoto_potter_solver
        .set_Eigen_solver_iteration_max_for_eigen_vector(
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
  inline void set_Eigen_solver_division_min(const T_ &division_min_in) {
    this->_arimoto_potter_solver.set_Eigen_solver_division_min(division_min_in);
  }

  /**
   * @brief Sets the small value for the eigenvalue solver.
   *
   * This function updates the small value used in the eigenvalue solver to
   * handle numerical stability issues during the LQR computation.
   *
   * @param small_value_in The new small value to be set.
   */
  inline void set_Eigen_solver_small_value(const T_ &small_value_in) {
    this->_arimoto_potter_solver.set_Eigen_solver_small_value(small_value_in);
  }

protected:
  /* Variable */
  A_Type A_;
  B_Type B_;
  Q_Type Q_;
  R_Type R_;
  K_Type K_;

  Solver_Type_ _arimoto_potter_solver;
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
  using T_ = typename A_Type::Value_Type;
  static_assert(std::is_same<T_, double>::value ||
                    std::is_same<T_, float>::value,
                "Matrix value data type must be float or double.");

  using A_EX_Type_ = PythonNumpy::ConcatenateHorizontally_Type<
      PythonNumpy::ConcatenateVertically_Type<A_Type, C_Type>,
      PythonNumpy::SparseMatrixEmpty_Type<T_, (A_Type::ROWS + C_Type::ROWS),
                                          C_Type::ROWS>>;

  using B_EX_Type_ = PythonNumpy::ConcatenateVertically_Type<
      B_Type,
      PythonNumpy::SparseMatrixEmpty_Type<T_, C_Type::ROWS, B_Type::COLS>>;

  using Solver_Type_ =
      LQR_ArimotoPotterSolver<A_EX_Type_, B_EX_Type_, Q_Type, R_Type>;

protected:
  /* Constant */
  static constexpr std::size_t Input_Size_ = B_Type::COLS;
  static constexpr std::size_t State_Size_ = A_Type::ROWS;
  static constexpr std::size_t Output_Size_ = C_Type::ROWS;

public:
  /* Type */
  using Value_Type = T_;

  using K_Type = PythonNumpy::DenseMatrix_Type<T_, Input_Size_,
                                               (State_Size_ + Output_Size_)>;

  /* Check Compatibility */
  static_assert(PythonNumpy::Is_Diag_Matrix<Q_Type>::value,
                "Q matrix must be diagonal matrix.");

  static_assert(PythonNumpy::Is_Diag_Matrix<R_Type>::value,
                "R matrix must be diagonal matrix.");

  /* Check Data Type */
  static_assert(std::is_same<typename B_Type::Value_Type, T_>::value,
                "Data type of B matrix must be same type as A matrix.");
  static_assert(std::is_same<typename C_Type::Value_Type, T_>::value,
                "Data type of C matrix must be same type as A matrix.");
  static_assert(std::is_same<typename Q_Type::Value_Type, T_>::value,
                "Data type of Q matrix must be same type as A matrix.");
  static_assert(std::is_same<typename R_Type::Value_Type, T_>::value,
                "Data type of R matrix must be same type as A matrix.");

  /* Check Matrix Column and Row length */
  static_assert((A_Type::COLS == A_Type::ROWS) &&
                    (B_Type::ROWS == A_Type::ROWS) &&
                    (C_Type::COLS == A_Type::ROWS) &&
                    (Q_Type::COLS == Q_Type::ROWS) &&
                    (Q_Type::COLS == (A_Type::ROWS + C_Type::ROWS)) &&
                    (R_Type::COLS == R_Type::ROWS) &&
                    (R_Type::COLS == B_Type::COLS),
                "A, B, C, Q, R matrix size is not compatible");

public:
  /* Constructor */
  LQI() : A_(), B_(), C_(), Q_(), R_(), K_(), _arimoto_potter_solver() {}

  LQI(const A_Type &A, const B_Type &B, const C_Type &C, const Q_Type &Q,
      const R_Type &R)
      : A_(A), B_(B), C_(C), Q_(Q), R_(R), K_(), _arimoto_potter_solver() {}

  /* Copy Constructor */
  LQI(const LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &input)
      : A_(input.A_), B_(input.B_), C_(input.C_), Q_(input.Q_), R_(input.R_),
        K_(input.K_), _arimoto_potter_solver(input._arimoto_potter_solver) {}

  LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &
  operator=(const LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &input) {
    if (this != &input) {
      this->A_ = input.A_;
      this->B_ = input.B_;
      this->C_ = input.C_;
      this->Q_ = input.Q_;
      this->R_ = input.R_;
      this->K_ = input.K_;
      this->_arimoto_potter_solver = input._arimoto_potter_solver;
    }
    return *this;
  }

  /* Move Constructor */
  LQI(LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &&input) noexcept
      : A_(std::move(input.A_)), B_(std::move(input.B_)),
        C_(std::move(input.C_)), Q_(std::move(input.Q_)),
        R_(std::move(input.R_)), K_(std::move(input.K_)),
        _arimoto_potter_solver(std::move(input._arimoto_potter_solver)) {}

  LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &
  operator=(LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &&input) noexcept {
    if (this != &input) {
      this->A_ = std::move(input.A_);
      this->B_ = std::move(input.B_);
      this->C_ = std::move(input.C_);
      this->Q_ = std::move(input.Q_);
      this->R_ = std::move(input.R_);
      this->K_ = std::move(input.K_);
      this->_arimoto_potter_solver = std::move(input._arimoto_potter_solver);
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
        PythonNumpy::concatenate_vertically(this->A_, this->C_),
        PythonNumpy::make_SparseMatrixEmpty<T_, (State_Size_ + Output_Size_),
                                            Output_Size_>());

    auto B_ex = PythonNumpy::concatenate_vertically(
        this->B_,
        PythonNumpy::make_SparseMatrixEmpty<T_, Output_Size_, Input_Size_>());

    this->K_ =
        this->_arimoto_potter_solver.solve(A_ex, B_ex, this->Q_, this->R_);

    return this->K_;
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
  inline K_Type get_K() const { return this->K_; }

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
    return this->_arimoto_potter_solver.get_eigen_solver_is_ill();
  }

  /**
   * @brief Sets the state transition matrix A.
   *
   * This function updates the state transition matrix A used in the LQI
   * computation.
   *
   * @param A The new state transition matrix to be set.
   */
  inline void set_A(const A_Type &A) { this->A_ = A; }

  /**
   * @brief Sets the input matrix B.
   *
   * This function updates the input matrix B used in the LQI computation.
   *
   * @param B The new input matrix to be set.
   */
  inline void set_B(const B_Type &B) { this->B_ = B; }

  /**
   * @brief Sets the output matrix C.
   *
   * This function updates the output matrix C used in the LQI computation.
   *
   * @param C The new output matrix to be set.
   */
  inline void set_C(const C_Type &C) { this->C_ = C; }

  /**
   * @brief Sets the state cost matrix Q.
   *
   * This function updates the state cost matrix Q used in the LQI computation.
   *
   * @param Q The new state cost matrix to be set.
   */
  inline void set_Q(const Q_Type &Q) { this->Q_ = Q; }

  /**
   * @brief Sets the input cost matrix R.
   *
   * This function updates the input cost matrix R used in the LQI computation.
   *
   * @param R The new input cost matrix to be set.
   */
  inline void set_R(const R_Type &R) { this->R_ = R; }

  /**
   * @brief Sets the division minimum for the inverse solver of R.
   *
   * This function updates the minimum division value used in the inverse
   * solver to avoid division by zero errors during the LQI computation.
   *
   * @param division_min_in The new minimum division value to be set.
   */
  inline void set_R_inv_division_min(const T_ &division_min_in) {
    this->_arimoto_potter_solver.set_R_inv_division_min(division_min_in);
  }

  /**
   * @brief Sets the decay rate for the inverse solver of V1 and V2 matrices.
   *
   * This function updates the decay rate used in the inverse solver to
   * control the convergence behavior during the LQI computation.
   *
   * @param decay_rate_in The new decay rate to be set.
   */
  inline void set_V1_inv_decay_rate(const T_ &decay_rate_in) {
    this->_arimoto_potter_solver.set_V1_inv_decay_rate(decay_rate_in);
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
  inline void set_V1_inv_division_min(const T_ &division_min_in) {
    this->_arimoto_potter_solver.set_V1_inv_division_min(division_min_in);
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
    this->_arimoto_potter_solver.set_Eigen_solver_iteration_max(iteration_max);
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
    this->_arimoto_potter_solver
        .set_Eigen_solver_iteration_max_for_eigen_vector(
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
  inline void set_Eigen_solver_division_min(const T_ &division_min_in) {
    this->_arimoto_potter_solver.set_Eigen_solver_division_min(division_min_in);
  }

  /**
   * @brief Sets the small value for the eigenvalue solver.
   *
   * This function updates the small value used in the eigenvalue solver to
   * handle numerical stability issues during the LQI computation.
   *
   * @param small_value_in The new small value to be set.
   */
  inline void set_Eigen_solver_small_value(const T_ &small_value_in) {
    this->_arimoto_potter_solver.set_Eigen_solver_small_value(small_value_in);
  }

protected:
  /* Variable */
  A_Type A_;
  B_Type B_;
  C_Type C_;
  Q_Type Q_;
  R_Type R_;
  K_Type K_;

  Solver_Type_ _arimoto_potter_solver;
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

#endif // PYTHON_CONTROL_LQR_HPP_
