#ifndef __PYTHON_CONTROL_LQR_HPP__
#define __PYTHON_CONTROL_LQR_HPP__

#include "python_numpy.hpp"

#include <type_traits>

namespace PythonControl {

/* Linear Quadratic Regulator */
template <typename A_Type, typename B_Type, typename Q_Type, typename R_Type>
class LQR {
private:
  /* Type */
  using _T = typename A_Type::Value_Type;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Matrix value data type must be float or double.");

  static constexpr std::size_t _Input_Size = B_Type::ROWS;
  static constexpr std::size_t _State_Size = A_Type::COLS;

public:
  /* Type */
  using K_Type =
      PythonNumpy::Matrix<PythonNumpy::DefDense, _T, _Input_Size, _State_Size>;

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

public:
  /* Function */
  K_Type solve_continuous() {

    auto R_inv_solver = PythonNumpy::make_LinalgSolver(this->_R);
    auto R_inv = R_inv_solver.get_answer();

    auto Hamiltonian_Left =
        PythonNumpy::concatenate_vertically(this->_A, -this->_Q);
    auto Hamiltonian_Right = PythonNumpy::concatenate_vertically(
        PythonNumpy::A_mul_BTranspose(-this->_B * R_inv_solver.get_answer(),
                                      this->_B),
        -this->_A.transpose());

    auto Hamiltonian = PythonNumpy::concatenate_horizontally(Hamiltonian_Left,
                                                             Hamiltonian_Right);

    auto eig_solver = PythonNumpy::make_LinalgSolverEig(Hamiltonian);

    auto eigen_values = eig_solver.get_eigen_values();

    eig_solver.solve_eigen_vectors(Hamiltonian);

    auto eigen_vectors = eig_solver.get_eigen_vectors();

    PythonNumpy::Matrix<PythonNumpy::DefDense, PythonNumpy::Complex<_T>,
                        _State_Size, _State_Size>
        V1;
    PythonNumpy::Matrix<PythonNumpy::DefDense, PythonNumpy::Complex<_T>,
                        _State_Size, _State_Size>
        V2;

    for (std::size_t i = 0; i < (static_cast<std::size_t>(2) * _State_Size);
         i++) {

      if (eigen_values(i, 0).real < static_cast<_T>(0)) {

        for (std::size_t j = 0; j < _State_Size; j++) {
          V1(j, i) = eigen_vectors(j, i);
          V2(j, i) = eigen_vectors(j + _State_Size, i);
        }
      }
    }

    auto V1_inv_solver = PythonNumpy::make_LinalgSolver(V1);

    auto P = (V2 * V1_inv_solver.get_answer()).real();

    this->_K = R_inv * this->_B.transpose() * P;

    return this->_K;
  }

  K_Type get_K() const { return this->_K; }

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

} // namespace PythonControl

#endif // __PYTHON_CONTROL_LQR_HPP__
