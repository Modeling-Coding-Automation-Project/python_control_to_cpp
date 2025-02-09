#ifndef __PYTHON_CONTROL_LQR_HPP__
#define __PYTHON_CONTROL_LQR_HPP__

#include "python_numpy.hpp"

#include <type_traits>

namespace PythonControl {

namespace LQR_Operation {

template <typename T, std::size_t State_Size>
using V1_V2_Type = PythonNumpy::DenseMatrix_Type<PythonNumpy::Complex<T>,
                                                 State_Size, State_Size>;

template <typename T, std::size_t State_Size>
using V1_V2_InvSolver_Type =
    PythonNumpy::LinalgSolverInv_Type<V1_V2_Type<T, State_Size>>;

template <typename A_Type, typename B_Type, typename Q_Type, typename R_Type>
struct Hamiltonian {

  using B_R_INV_BT_Type = PythonNumpy::A_mul_BTranspose_Type<
      PythonNumpy::A_Multiply_B_Type<B_Type, R_Type>, B_Type>;

  using Type =
      PythonNumpy::ConcatenateBlock_Type<A_Type, B_R_INV_BT_Type, Q_Type,
                                         PythonNumpy::Transpose_Type<A_Type>>;
};

} // namespace LQR_Operation

/* LQR with Arimoto Potter method */
template <typename A_Type, typename B_Type, typename Q_Type, typename R_Type,
          typename K_Type, typename Hamiltonian_Type>
inline void lqr_solve_with_arimoto_potter(
    const A_Type &A, const B_Type &B, const Q_Type &Q, const R_Type &R,
    K_Type &K, PythonNumpy::LinalgSolverInv_Type<R_Type> &R_inv_solver,
    LQR_Operation::V1_V2_InvSolver_Type<typename A_Type::Value_Type,
                                        A_Type::COLS> &V1_inv_solver,
    PythonNumpy::LinalgSolverEig_Type<Hamiltonian_Type> &eig_solver) {

  using _T = typename A_Type::Value_Type;

  static constexpr std::size_t _State_Size = A_Type::COLS;

  R_inv_solver.inv(R);
  auto R_inv = R_inv_solver.get_answer();

  auto Hamiltonian = PythonNumpy::concatenate_block(
      A, PythonNumpy::A_mul_BTranspose(-B * R_inv, B), -Q, -A.transpose());

  eig_solver.solve_eigen_values(Hamiltonian);

  auto eigen_values = eig_solver.get_eigen_values();

  eig_solver.solve_eigen_vectors(Hamiltonian);

  auto eigen_vectors = eig_solver.get_eigen_vectors();

  LQR_Operation::V1_V2_Type<_T, _State_Size> V1;
  LQR_Operation::V1_V2_Type<_T, _State_Size> V2;

  std::size_t minus_count = 0;
  for (std::size_t i = 0; i < (static_cast<std::size_t>(2) * _State_Size);
       i++) {

    if (eigen_values(i, 0).real < static_cast<_T>(0)) {

      for (std::size_t j = 0; j < _State_Size; j++) {
        V1(j, minus_count) = eigen_vectors(j, i);
        V2(j, minus_count) = eigen_vectors(j + _State_Size, i);
      }

      minus_count++;
      if (_State_Size == minus_count) {
        break;
      }
    }
  }

  V1_inv_solver.inv(V1);

  auto P = (V2 * V1_inv_solver.get_answer()).real();

  K = R_inv * PythonNumpy::ATranspose_mul_B(B, P);
}

/* Linear Quadratic Regulator */
template <typename A_Type, typename B_Type, typename Q_Type, typename R_Type>
class LQR {
private:
  /* Type */
  using _T = typename A_Type::Value_Type;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Matrix value data type must be float or double.");

  static_assert(PythonNumpy::Is_Diag_Matrix<Q_Type>::value,
                "Q matrix must be diagonal matrix.");

  static_assert(PythonNumpy::Is_Diag_Matrix<R_Type>::value,
                "Q matrix must be diagonal matrix.");

  using _Hamiltonian_Type =
      typename LQR_Operation::Hamiltonian<A_Type, B_Type, Q_Type, R_Type>::Type;

private:
  /* Constant */
  static constexpr std::size_t _Input_Size = B_Type::ROWS;
  static constexpr std::size_t _State_Size = A_Type::COLS;

public:
  /* Type */
  using K_Type = PythonNumpy::DenseMatrix_Type<_T, _Input_Size, _State_Size>;

  /* Check Compatibility */
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
        eig_solver() {}

  LQR(const A_Type &A, const B_Type &B, const Q_Type &Q, const R_Type &R)
      : _A(A), _B(B), _Q(Q), _R(R), _K(), _R_inv_solver(), _V1_inv_solver(),
        eig_solver() {}

  /* Copy Constructor */
  LQR(const LQR<A_Type, B_Type, Q_Type, R_Type> &input)
      : _A(input._A), _B(input._B), _Q(input._Q), _R(input._R), _K(input._K),
        _R_inv_solver(input._R_inv_solver),
        _V1_inv_solver(input._V1_inv_solver), eig_solver(input.eig_solver) {}

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
    }
    return *this;
  }

  /* Move Constructor */
  LQR(LQR<A_Type, B_Type, Q_Type, R_Type> &&input) noexcept
      : _A(std::move(input._A)), _B(std::move(input._B)),
        _Q(std::move(input._Q)), _R(std::move(input._R)),
        _K(std::move(input._K)), _R_inv_solver(std::move(input._R_inv_solver)),
        _V1_inv_solver(std::move(input._V1_inv_solver)),
        eig_solver(std::move(input.eig_solver)) {}

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
    }
    return *this;
  }

public:
  /* Function */
  inline K_Type solve(void) {

    PythonControl::lqr_solve_with_arimoto_potter<A_Type, B_Type, Q_Type, R_Type,
                                                 K_Type, _Hamiltonian_Type>(
        this->_A, this->_B, this->_Q, this->_R, this->_K, this->_R_inv_solver,
        this->_V1_inv_solver, this->eig_solver);

    return this->_K;
  }

  inline K_Type get_K() const { return this->_K; }

  inline void set_A(const A_Type &A) { this->_A = A; }

  inline void set_B(const B_Type &B) { this->_B = B; }

  inline void set_Q(const Q_Type &Q) { this->_Q = Q; }

  inline void set_R(const R_Type &R) { this->_R = R; }

  // inline void set_R_inv_decay_rate(const T &decay_rate_in) {
  //   this->decay_rate = decay_rate_in;
  // }

  // inline void set_R_inv_division_min(const T &division_min_in) {
  //   this->division_min = division_min_in;
  // }

private:
  /* Variable */
  A_Type _A;
  B_Type _B;
  Q_Type _Q;
  R_Type _R;
  K_Type _K;

  PythonNumpy::LinalgSolverInv_Type<R_Type> _R_inv_solver;
  LQR_Operation::V1_V2_InvSolver_Type<_T, _State_Size> _V1_inv_solver;
  PythonNumpy::LinalgSolverEig_Type<_Hamiltonian_Type> eig_solver;
};

/* Make LQR */
template <typename A_Type, typename B_Type, typename Q_Type, typename R_Type>
inline auto make_LQR(const A_Type &A, const B_Type &B, const Q_Type &Q,
                     const R_Type &R) -> LQR<A_Type, B_Type, Q_Type, R_Type> {

  return LQR<A_Type, B_Type, Q_Type, R_Type>(A, B, Q, R);
}

/* LQR Type */
template <typename A_Type, typename B_Type, typename Q_Type, typename R_Type>
using LQR_Type = LQR<A_Type, B_Type, Q_Type, R_Type>;

/* Linear Quadratic optimum control with Integral action */
template <typename A_Type, typename B_Type, typename C_Type, typename Q_Type,
          typename R_Type>
class LQI {
private:
  /* Type */
  using _T = typename A_Type::Value_Type;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Matrix value data type must be float or double.");

  static_assert(PythonNumpy::Is_Diag_Matrix<Q_Type>::value,
                "Q matrix must be diagonal matrix.");

  static_assert(PythonNumpy::Is_Diag_Matrix<R_Type>::value,
                "Q matrix must be diagonal matrix.");

  using _A_EX_Type = PythonNumpy::ConcatenateHorizontally_Type<
      PythonNumpy::ConcatenateVertically_Type<A_Type, C_Type>,
      decltype(PythonNumpy::make_SparseMatrixEmpty<
               _T, (A_Type::COLS + C_Type::COLS), C_Type::COLS>())>;

  using _B_EX_Type = PythonNumpy::ConcatenateVertically_Type<
      B_Type, decltype(PythonNumpy::make_SparseMatrixEmpty<_T, C_Type::COLS,
                                                           B_Type::ROWS>())>;

  using _Hamiltonian_Type =
      typename LQR_Operation::Hamiltonian<_A_EX_Type, _B_EX_Type, Q_Type,
                                          R_Type>::Type;

private:
  /* Constant */
  static constexpr std::size_t _Input_Size = B_Type::ROWS;
  static constexpr std::size_t _State_Size = A_Type::COLS;
  static constexpr std::size_t _Output_Size = C_Type::COLS;

public:
  /* Type */
  using K_Type = PythonNumpy::DenseMatrix_Type<_T, _Input_Size,
                                               (_State_Size + _Output_Size)>;

  /* Check Compatibility */
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
        _eig_solver() {}

  LQI(const A_Type &A, const B_Type &B, const C_Type &C, const Q_Type &Q,
      const R_Type &R)
      : _A(A), _B(B), _C(C), _Q(Q), _R(R), _K(), _R_inv_solver(),
        _V1_inv_solver(), _eig_solver() {}

  /* Copy Constructor */
  LQI(const LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &input)
      : _A(input._A), _B(input._B), _C(input._C), _Q(input._Q), _R(input._R),
        _K(input._K), _R_inv_solver(input._R_inv_solver),
        _V1_inv_solver(input._V1_inv_solver), _eig_solver(input._eig_solver) {}

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
      this->_eig_solver = input.eig_solver;
    }
    return *this;
  }

  /* Move Constructor */
  LQI(LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &&input) noexcept
      : _A(std::move(input._A)), _B(std::move(input._B)),
        _C(std::move(input._C)), _Q(std::move(input._Q)),
        _R(std::move(input._R)), _K(std::move(input._K)),
        _R_inv_solver(std::move(input._R_inv_solver)),
        _V1_inv_solver(std::move(input._V1_inv_solver)),
        _eig_solver(std::move(input._eig_solver)) {}

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
    }
    return *this;
  }

public:
  /* Function */
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
        this->_V1_inv_solver, this->_eig_solver);

    return this->_K;
  }

  inline K_Type get_K() const { return this->_K; }

  inline void set_A(const A_Type &A) { this->_A = A; }

  inline void set_B(const B_Type &B) { this->_B = B; }

  inline void set_C(const C_Type &C) { this->_C = C; }

  inline void set_Q(const Q_Type &Q) { this->_Q = Q; }

  inline void set_R(const R_Type &R) { this->_R = R; }

private:
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
};

/* Make LQI */
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
