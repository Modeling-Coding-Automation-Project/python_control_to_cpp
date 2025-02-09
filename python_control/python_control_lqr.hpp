#ifndef __PYTHON_CONTROL_LQR_HPP__
#define __PYTHON_CONTROL_LQR_HPP__

#include "python_numpy.hpp"

#include <type_traits>

namespace PythonControl {

/* LQR with Arimoto Potter method */
template <typename A_Type, typename B_Type, typename Q_Type, typename R_Type,
          typename K_Type>
inline void lqr_solve_with_arimoto_potter(const A_Type &A, const B_Type &B,
                                          const Q_Type &Q, const R_Type &R,
                                          K_Type &K) {

  using _T = typename A_Type::Value_Type;
  static constexpr std::size_t _State_Size = A_Type::COLS;

  auto R_inv_solver = PythonNumpy::make_LinalgSolverInv<R_Type>();
  R_inv_solver.inv(R);
  auto R_inv = R_inv_solver.get_answer();

  auto Hamiltonian = PythonNumpy::concatenate_horizontally(
      PythonNumpy::concatenate_vertically(A, -Q),
      PythonNumpy::concatenate_vertically(
          PythonNumpy::A_mul_BTranspose(-B * R_inv, B), -A.transpose()));

  auto eig_solver = PythonNumpy::make_LinalgSolverEig<decltype(Hamiltonian)>();
  eig_solver.solve_eigen_values(Hamiltonian);

  auto eigen_values = eig_solver.get_eigen_values();

  eig_solver.solve_eigen_vectors(Hamiltonian);

  auto eigen_vectors = eig_solver.get_eigen_vectors();

  PythonNumpy::DenseMatrix_Type<PythonNumpy::Complex<_T>, _State_Size,
                                _State_Size>
      V1;
  PythonNumpy::DenseMatrix_Type<PythonNumpy::Complex<_T>, _State_Size,
                                _State_Size>
      V2;

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
  auto V1_inv_solver = PythonNumpy::make_LinalgSolverInv<decltype(V1)>();
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
  LQR(const A_Type &A, const B_Type &B, const Q_Type &Q, const R_Type &R)
      : _A(A), _B(B), _Q(Q), _R(R) {}

  /* Copy Constructor */
  LQR(const LQR<A_Type, B_Type, Q_Type, R_Type> &input)
      : _A(input._A), _B(input._B), _Q(input._Q), _R(input._R) {}

  LQR<A_Type, B_Type, Q_Type, R_Type> &
  operator=(const LQR<A_Type, B_Type, Q_Type, R_Type> &input) {
    if (this != &input) {
      this->_A = input._A;
      this->_B = input._B;
      this->_Q = input._Q;
      this->_R = input._R;
    }
    return *this;
  }

  /* Move Constructor */
  LQR(LQR<A_Type, B_Type, Q_Type, R_Type> &&input)
  noexcept
      : _A(std::move(input._A)), _B(std::move(input._B)),
        _Q(std::move(input._Q)), _R(std::move(input._R)) {}

  LQR<A_Type, B_Type, Q_Type, R_Type> &
  operator=(LQR<A_Type, B_Type, Q_Type, R_Type> &&input) noexcept {
    if (this != &input) {
      this->_A = std::move(input._A);
      this->_B = std::move(input._B);
      this->_Q = std::move(input._Q);
      this->_R = std::move(input._R);
    }
    return *this;
  }

public:
  /* Function */
  inline K_Type solve(void) {

    PythonControl::lqr_solve_with_arimoto_potter(this->_A, this->_B, this->_Q,
                                                 this->_R, this->_K);

    return this->_K;
  }

  inline K_Type get_K() const { return this->_K; }

  inline void set_A(const A_Type &A) { this->_A = A; }

  inline void set_B(const B_Type &B) { this->_B = B; }

  inline void set_Q(const Q_Type &Q) { this->_Q = Q; }

  inline void set_R(const R_Type &R) { this->_R = R; }

private:
  /* Variable */
  A_Type _A;
  B_Type _B;
  Q_Type _Q;
  R_Type _R;

  K_Type _K;
};

/* Make LQR */
template <typename A_Type, typename B_Type, typename Q_Type, typename R_Type>
inline auto make_LQR(const A_Type &A, const B_Type &B, const Q_Type &Q,
                     const R_Type &R) -> LQR<A_Type, B_Type, Q_Type, R_Type> {

  return LQR<A_Type, B_Type, Q_Type, R_Type>(A, B, Q, R);
}

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
  LQI(const A_Type &A, const B_Type &B, const C_Type &C, const Q_Type &Q,
      const R_Type &R)
      : _A(A), _B(B), _C(C), _Q(Q), _R(R) {}

  /* Copy Constructor */
  LQI(const LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &input)
      : _A(input._A), _B(input._B), _C(input._C), _Q(input._Q), _R(input._R) {}

  LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &
  operator=(const LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &input) {
    if (this != &input) {
      this->_A = input._A;
      this->_B = input._B;
      this->_C = input._C;
      this->_Q = input._Q;
      this->_R = input._R;
    }
    return *this;
  }

  /* Move Constructor */
  LQI(LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &&input)
  noexcept
      : _A(std::move(input._A)), _B(std::move(input._B)),
        _C(std::move(input._C)), _Q(std::move(input._Q)),
        _R(std::move(input._R)) {}

  LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &
  operator=(LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> &&input) noexcept {
    if (this != &input) {
      this->_A = std::move(input._A);
      this->_B = std::move(input._B);
      this->_C = std::move(input._C);
      this->_Q = std::move(input._Q);
      this->_R = std::move(input._R);
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

    PythonControl::lqr_solve_with_arimoto_potter(A_ex, B_ex, this->_Q, this->_R,
                                                 this->_K);

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
};

/* Make LQI */
template <typename A_Type, typename B_Type, typename C_Type, typename Q_Type,
          typename R_Type>
inline auto make_LQI(const A_Type &A, const B_Type &B, const C_Type &C,
                     const Q_Type &Q, const R_Type &R)
    -> LQI<A_Type, B_Type, C_Type, Q_Type, R_Type> {

  return LQI<A_Type, B_Type, C_Type, Q_Type, R_Type>(A, B, C, Q, R);
}

} // namespace PythonControl

#endif // __PYTHON_CONTROL_LQR_HPP__
