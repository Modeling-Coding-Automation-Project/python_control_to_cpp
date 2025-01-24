#ifndef __PYTHON_CONTROL_STATE_SPACE_HPP__
#define __PYTHON_CONTROL_STATE_SPACE_HPP__

#include "python_numpy.hpp"

#include <type_traits>

namespace PythonControl {

constexpr double STATE_SPACE_DIVISION_MIN = 1.0e-10;

template <typename T, std::size_t Input_Size>
using StateSpaceInputType =
    PythonNumpy::Matrix<PythonNumpy::DefDense, T, Input_Size, 1>;

namespace MakeStateSpaceInput {

template <std::size_t IndexCount, typename StateSpaceInputType, typename T>
inline void assign_values(StateSpaceInputType &input, T value_1) {

  static_assert(
      IndexCount < StateSpaceInputType::COLS,
      "Number of arguments must be less than the number of input size.");

  input.template set<IndexCount, 0>(value_1);
}

template <std::size_t IndexCount, typename StateSpaceInputType, typename T,
          typename U, typename... Args>
inline void assign_values(StateSpaceInputType &input, T value_1, U value_2,
                          Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");
  static_assert(
      IndexCount < StateSpaceInputType::COLS,
      "Number of arguments must be less than the number of input size.");

  input.template set<IndexCount, 0>(value_1);

  assign_values<IndexCount + 1>(input, value_2, args...);
}

} // namespace MakeStateSpaceInput

/* make State Space Input  */
template <std::size_t Input_Size, typename T, typename... Args>
inline auto make_StateSpaceInput(T value_1, Args... args)
    -> StateSpaceInputType<T, Input_Size> {

  StateSpaceInputType<T, Input_Size> input;

  MakeStateSpaceInput::assign_values<0>(input, value_1, args...);

  return input;
}

namespace ForDiscreteStateSpace {

/* Substitute Y values to Ring Buffer */
template <typename T, typename Original_Y_Type, typename Y_with_Delay_Type,
          std::size_t Count>
struct SubstituteYRingBuffer {
  static void compute(Y_with_Delay_Type &Y_buffer, const Original_Y_Type &Y,
                      const std::size_t &buffer_index) {
    Y_buffer(Count, buffer_index) = Y.template get<Count, 0>();
    SubstituteYRingBuffer<T, Original_Y_Type, Y_with_Delay_Type,
                          (Count - 1)>::compute(Y_buffer, Y, buffer_index);
  }
};

template <typename T, typename Original_Y_Type, typename Y_with_Delay_Type>
struct SubstituteYRingBuffer<T, Original_Y_Type, Y_with_Delay_Type, 0> {
  static void compute(Y_with_Delay_Type &Y_buffer, const Original_Y_Type &Y,
                      const std::size_t &buffer_index) {
    Y_buffer(0, buffer_index) = Y.template get<0, 0>();
  }
};

/* Substitute row from DenseMatrix to DenseRow */
template <typename T, std::size_t M, std::size_t N, std::size_t Count>
struct SubstituteDenseMatrixRow {
  static void
  compute(const PythonNumpy::Matrix<PythonNumpy::DefDense, T, M, N> &matrix,
          PythonNumpy::Matrix<PythonNumpy::DefDense, T, M, 1> &row,
          const std::size_t &row_index) {
    row.template set<Count, 0>(matrix(Count, row_index));
    SubstituteDenseMatrixRow<T, M, N, (Count - 1)>::compute(matrix, row,
                                                            row_index);
  }
};

template <typename T, std::size_t M, std::size_t N>
struct SubstituteDenseMatrixRow<T, M, N, 0> {
  static void
  compute(const PythonNumpy::Matrix<PythonNumpy::DefDense, T, M, N> &matrix,
          PythonNumpy::Matrix<PythonNumpy::DefDense, T, M, 1> &row,
          const std::size_t &row_index) {
    row.template set<0, 0>(matrix(0, row_index));
  }
};

/* Update Ring Buffer Index */
template <std::size_t Number_Of_Buffer> struct UpdateRingBufferIndex {
  static void compute(std::size_t &index) {
    if (index < Number_Of_Buffer) {
      index++;
    } else {
      index = static_cast<std::size_t>(0);
    }
  }
};

template <> struct UpdateRingBufferIndex<0> {
  static void compute(std::size_t &index) {
    /* Do Nothing */
    static_cast<void>(index);
  }
};

} // namespace ForDiscreteStateSpace

/* Discrete State Space */
template <typename A_Type, typename B_Type, typename C_Type, typename D_Type,
          std::size_t Number_Of_Delay = 0>
class DiscreteStateSpace {
private:
  /* Type */
  using _T = typename A_Type::Value_Type;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Matrix value data type must be float or double.");

  static constexpr std::size_t _Input_Size = B_Type::ROWS;
  static constexpr std::size_t _State_Size = A_Type::COLS;
  static constexpr std::size_t _Output_Size = C_Type::COLS;

public:
  using Original_U_Type = PythonControl::StateSpaceInputType<_T, _Input_Size>;
  using Original_X_Type =
      PythonNumpy::Matrix<PythonNumpy::DefDense, _T, _State_Size, 1>;
  using Original_Y_Type =
      PythonNumpy::Matrix<PythonNumpy::DefDense, _T, _Output_Size, 1>;

  using Y_with_Delay_Type =
      PythonNumpy::Matrix<PythonNumpy::DefDense, _T, _Output_Size,
                          (1 + Number_Of_Delay)>;

  /* Check Compatibility */
  /* Check Data Type */
  static_assert(std::is_same<typename B_Type::Value_Type, _T>::value,
                "Data type of B matrix must be same type as A matrix.");

  static_assert(std::is_same<typename C_Type::Value_Type, _T>::value,
                "Data type of C matrix must be same type as A matrix.");

  static_assert(std::is_same<typename D_Type::Value_Type, _T>::value,
                "Data type of D matrix must be same type as A matrix.");

  /* Check Matrix Column and Row length */
  static_assert((A_Type::ROWS == A_Type::COLS) &&
                    (B_Type::COLS == A_Type::COLS) &&
                    (C_Type::ROWS == A_Type::COLS) &&
                    (D_Type::COLS == C_Type::COLS) &&
                    (D_Type::ROWS == B_Type::ROWS),
                "A, B, C, D matrix size is not compatible");

public:
  /* Constructor */
  DiscreteStateSpace()
      : A(), B(), C(), D(),
        delta_time(static_cast<_T>(PythonControl::STATE_SPACE_DIVISION_MIN)),
        _delay_ring_buffer_index(static_cast<std::size_t>(0)) {}

  DiscreteStateSpace(const A_Type &A_in)
      : A(A_in), B(), C(), D(),
        delta_time(static_cast<_T>(PythonControl::STATE_SPACE_DIVISION_MIN)),
        _delay_ring_buffer_index(static_cast<std::size_t>(0)) {}

  DiscreteStateSpace(const A_Type &A_in, const B_Type &B_in)
      : A(A_in), B(B_in), C(), D(),
        delta_time(static_cast<_T>(PythonControl::STATE_SPACE_DIVISION_MIN)),
        _delay_ring_buffer_index(static_cast<std::size_t>(0)) {}

  DiscreteStateSpace(const A_Type &A_in, const B_Type &B_in, const C_Type &C_in)
      : A(A_in), B(B_in), C(C_in), D(),
        delta_time(static_cast<_T>(PythonControl::STATE_SPACE_DIVISION_MIN)),
        _delay_ring_buffer_index(static_cast<std::size_t>(0)) {}

  DiscreteStateSpace(const A_Type &A_in, const B_Type &B_in, const C_Type &C_in,
                     const D_Type &D_in)
      : A(A_in), B(B_in), C(C_in), D(D_in),
        delta_time(static_cast<_T>(PythonControl::STATE_SPACE_DIVISION_MIN)),
        _delay_ring_buffer_index(static_cast<std::size_t>(0)) {}

  DiscreteStateSpace(const A_Type &A_in, const B_Type &B_in, const C_Type &C_in,
                     const D_Type &D_in, const _T &delta_time)
      : A(A_in), B(B_in), C(C_in), D(D_in), delta_time(delta_time),
        _delay_ring_buffer_index(static_cast<std::size_t>(0)) {}

  /* Copy Constructor */
  DiscreteStateSpace(
      const DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> &input)
      : A(input.A), B(input.B), C(input.C), D(input.D),
        delta_time(input.delta_time), X(input.X), X_initial(input.X_initial),
        Y(input.Y), _delay_ring_buffer_index(input._delay_ring_buffer_index) {}

  DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> &
  operator=(const DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> &input) {
    if (this != &input) {
      this->A = input.A;
      this->B = input.B;
      this->C = input.C;
      this->D = input.D;
      this->delta_time = input.delta_time;
      this->X = input.X;
      this->X_initial = input.X_initial;
      this->Y = input.Y;
      this->_delay_ring_buffer_index = input._delay_ring_buffer_index;
    }
    return *this;
  }

  /* Move Constructor */
  DiscreteStateSpace(
      DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> &&input) noexcept
      : A(std::move(input.A)), B(std::move(input.B)), C(std::move(input.C)),
        D(std::move(input.D)), delta_time(std::move(input.delta_time)),
        X(std::move(input.X)), X_initial(std::move(input.X_initial)),
        Y(std::move(input.Y)),
        _delay_ring_buffer_index(std::move(input._delay_ring_buffer_index)) {}

  DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> &operator=(
      DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> &&input) noexcept {
    if (this != &input) {
      this->A = std::move(input.A);
      this->B = std::move(input.B);
      this->C = std::move(input.C);
      this->D = std::move(input.D);
      this->delta_time = std::move(input.delta_time);
      this->X = std::move(input.X);
      this->X_initial = std::move(input.X_initial);
      this->Y = std::move(input.Y);
      this->_delay_ring_buffer_index =
          std::move(input._delay_ring_buffer_index);
    }
    return *this;
  }

public:
  /* Function */
  constexpr std::size_t get_number_of_delay(void) const {
    return NUMBER_OF_DELAY;
  }

  auto get_X(void) const -> Original_X_Type { return this->X; }

  auto get_Y(void) const -> Original_Y_Type {
    Original_Y_Type result;

    ForDiscreteStateSpace::SubstituteDenseMatrixRow<
        _T, _Output_Size, (1 + Number_Of_Delay),
        (_Output_Size - 1)>::compute(this->Y, result,
                                     this->_delay_ring_buffer_index);

    return result;
  }

  const std::size_t get_delay_ring_buffer_index(void) const {
    return this->_delay_ring_buffer_index;
  }

  void update(const Original_U_Type &U) {

    ForDiscreteStateSpace::SubstituteYRingBuffer<
        _T, Original_Y_Type, Y_with_Delay_Type,
        (_Output_Size - 1)>::compute(this->Y, (this->C * this->X + this->D * U),
                                     this->_delay_ring_buffer_index);

    this->X = this->A * this->X + this->B * U;

    ForDiscreteStateSpace::UpdateRingBufferIndex<Number_Of_Delay>::compute(
        this->_delay_ring_buffer_index);
  }

  void reset_state(void) { this->X = this->X_initial; }

public:
  /* Constant */
  static constexpr std::size_t NUMBER_OF_DELAY = Number_Of_Delay;

public:
  /* Variable */
  A_Type A;
  B_Type B;
  C_Type C;
  D_Type D;
  _T delta_time;

  Original_X_Type X;
  Original_X_Type X_initial;

  Y_with_Delay_Type Y;

private:
  /* Variable */
  std::size_t _delay_ring_buffer_index;
};

/* Make Discrete State Space */
template <typename A_Type, typename B_Type, typename C_Type, typename D_Type>
inline auto make_DiscreteStateSpace(A_Type A, B_Type B, C_Type C, D_Type D,
                                    typename A_Type::Value_Type delta_time)
    -> DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> {

  return DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type>(A, B, C, D,
                                                            delta_time);
}

template <std::size_t Number_Of_Delay, typename A_Type, typename B_Type,
          typename C_Type, typename D_Type>
inline auto make_DiscreteStateSpace(A_Type A, B_Type B, C_Type C, D_Type D,
                                    typename A_Type::Value_Type delta_time)
    -> DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type, Number_Of_Delay> {

  return DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type, Number_Of_Delay>(
      A, B, C, D, delta_time);
}

} // namespace PythonControl

#endif // __PYTHON_CONTROL_STATE_SPACE_HPP__
