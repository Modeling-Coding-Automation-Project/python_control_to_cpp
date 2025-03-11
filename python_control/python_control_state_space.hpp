#ifndef __PYTHON_CONTROL_STATE_SPACE_HPP__
#define __PYTHON_CONTROL_STATE_SPACE_HPP__

#include "python_numpy.hpp"

#include <array>
#include <type_traits>

namespace PythonControl {

constexpr double STATE_SPACE_DIVISION_MIN = 1.0e-10;

namespace ForDelayedVector {

/* Substitute Vector values to Ring Buffer */
template <typename T, typename Original_Vector_Type,
          typename Vector_with_Delay_Type, std::size_t Count>
struct SubstituteVectorRingBuffer {
  static void compute(Vector_with_Delay_Type &Vector_buffer,
                      const Original_Vector_Type &Vector,
                      const std::size_t &buffer_index) {

    Vector_buffer[buffer_index](Count, 0) = Vector.template get<Count, 0>();
    SubstituteVectorRingBuffer<T, Original_Vector_Type, Vector_with_Delay_Type,
                               (Count - 1)>::compute(Vector_buffer, Vector,
                                                     buffer_index);
  }
};

template <typename T, typename Original_Vector_Type,
          typename Vector_with_Delay_Type>
struct SubstituteVectorRingBuffer<T, Original_Vector_Type,
                                  Vector_with_Delay_Type, 0> {

  static void compute(Vector_with_Delay_Type &Vector_buffer,
                      const Original_Vector_Type &Vector,
                      const std::size_t &buffer_index) {
    Vector_buffer[buffer_index](0, 0) = Vector.template get<0, 0>();
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

} // namespace ForDelayedVector

/* Delayed Vector Object */
template <typename Vector_Type, std::size_t Number_Of_Delay>
class DelayedVectorObject {
private:
  /* Type */
  using _T = typename Vector_Type::Value_Type;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Matrix value data type must be float or double.");

  using _Vector_with_Delay_Type =
      std::array<PythonNumpy::DenseMatrix_Type<_T, Vector_Type::COLS, 1>,
                 (1 + Number_Of_Delay)>;

public:
  /* Constructor */
  DelayedVectorObject() : _store(), _delay_ring_buffer_index(0) {}

  /* Copy Constructor */
  DelayedVectorObject(
      const DelayedVectorObject<Vector_Type, Number_Of_Delay> &input)
      : _store(input._store),
        _delay_ring_buffer_index(input._delay_ring_buffer_index) {}

  DelayedVectorObject<Vector_Type, Number_Of_Delay> &
  operator=(const DelayedVectorObject<Vector_Type, Number_Of_Delay> &input) {
    if (this != &input) {
      this->_store = input._store;
      this->_delay_ring_buffer_index = input._delay_ring_buffer_index;
    }
    return *this;
  }

  /* Move Constructor */
  DelayedVectorObject(
      DelayedVectorObject<Vector_Type, Number_Of_Delay> &&input) noexcept
      : _store(std::move(input._store)),
        _delay_ring_buffer_index(std::move(input._delay_ring_buffer_index)) {}

  DelayedVectorObject<Vector_Type, Number_Of_Delay> &operator=(
      DelayedVectorObject<Vector_Type, Number_Of_Delay> &&input) noexcept {
    if (this != &input) {
      this->_store = std::move(input._store);
      this->_delay_ring_buffer_index =
          std::move(input._delay_ring_buffer_index);
    }
    return *this;
  }

public:
  /* Function */
  inline void push(const Vector_Type &vector) {

    ForDelayedVector::SubstituteVectorRingBuffer<
        _T, Vector_Type, _Vector_with_Delay_Type,
        (Vector_Type::COLS - 1)>::compute(this->_store, vector,
                                          this->_delay_ring_buffer_index);

    ForDelayedVector::UpdateRingBufferIndex<Number_Of_Delay>::compute(
        this->_delay_ring_buffer_index);
  }

  inline auto get(void) const -> Vector_Type {

    return this->_store[this->_delay_ring_buffer_index];
  }

  template <std::size_t Index> inline auto access(void) -> _T & {
    static_assert(Index < Vector_Type::COLS,
                  "Index must be less than vector size.");

    return this->_store[this->_delay_ring_buffer_index].access(Index, 0);
  }

  inline const std::size_t get_delay_ring_buffer_index(void) const {
    return this->_delay_ring_buffer_index;
  }

public:
  /* Constant */
  static constexpr std::size_t VECTOR_SIZE = Vector_Type::COLS;
  static constexpr std::size_t NUMBER_OF_DELAY = Number_Of_Delay;

private:
  /* Variable */
  _Vector_with_Delay_Type _store;
  std::size_t _delay_ring_buffer_index;
};

template <typename T, std::size_t Vector_Size>
using StateSpaceInputType = PythonNumpy::DenseMatrix_Type<T, Vector_Size, 1>;

template <typename T, std::size_t Vector_Size>
using StateSpaceStateType = PythonNumpy::DenseMatrix_Type<T, Vector_Size, 1>;

template <typename T, std::size_t Vector_Size>
using StateSpaceOutputType = PythonNumpy::DenseMatrix_Type<T, Vector_Size, 1>;

namespace MakeStateSpaceVector {

template <std::size_t IndexCount, typename StateSpaceVectorType, typename T>
inline void assign_values(StateSpaceVectorType &input, T value_1) {

  static_assert(
      IndexCount < StateSpaceVectorType::COLS,
      "Number of arguments must be less than the number of input size.");

  input.template set<IndexCount, 0>(value_1);
}

template <std::size_t IndexCount, typename StateSpaceVectorType, typename T,
          typename U, typename... Args>
inline void assign_values(StateSpaceVectorType &input, T value_1, U value_2,
                          Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");
  static_assert(
      IndexCount < StateSpaceVectorType::COLS,
      "Number of arguments must be less than the number of input size.");

  input.template set<IndexCount, 0>(value_1);

  assign_values<IndexCount + 1>(input, value_2, args...);
}

} // namespace MakeStateSpaceVector

/* make State Space Vector  */
template <std::size_t Vector_Size, typename T, typename... Args>
inline auto make_StateSpaceInput(T value_1, Args... args)
    -> StateSpaceInputType<T, Vector_Size> {

  StateSpaceInputType<T, Vector_Size> input;

  MakeStateSpaceVector::assign_values<0>(input, value_1, args...);

  return input;
}

template <std::size_t Vector_Size, typename T, typename... Args>
inline auto make_StateSpaceState(T value_1, Args... args)
    -> StateSpaceStateType<T, Vector_Size> {

  StateSpaceStateType<T, Vector_Size> input;

  MakeStateSpaceVector::assign_values<0>(input, value_1, args...);

  return input;
}

template <std::size_t Vector_Size, typename T, typename... Args>
inline auto make_StateSpaceOutput(T value_1, Args... args)
    -> StateSpaceOutputType<T, Vector_Size> {

  StateSpaceOutputType<T, Vector_Size> input;

  MakeStateSpaceVector::assign_values<0>(input, value_1, args...);

  return input;
}

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
  using Original_X_Type = PythonNumpy::DenseMatrix_Type<_T, _State_Size, 1>;
  using Original_Y_Type = PythonNumpy::DenseMatrix_Type<_T, _Output_Size, 1>;

  using Y_with_Delay_Type =
      PythonNumpy::DenseMatrix_Type<_T, _Output_Size, (1 + Number_Of_Delay)>;

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
        delta_time(static_cast<_T>(PythonControl::STATE_SPACE_DIVISION_MIN)) {}

  DiscreteStateSpace(const A_Type &A_in)
      : A(A_in), B(), C(), D(),
        delta_time(static_cast<_T>(PythonControl::STATE_SPACE_DIVISION_MIN)) {}

  DiscreteStateSpace(const A_Type &A_in, const B_Type &B_in)
      : A(A_in), B(B_in), C(), D(),
        delta_time(static_cast<_T>(PythonControl::STATE_SPACE_DIVISION_MIN)) {}

  DiscreteStateSpace(const A_Type &A_in, const B_Type &B_in, const C_Type &C_in)
      : A(A_in), B(B_in), C(C_in), D(),
        delta_time(static_cast<_T>(PythonControl::STATE_SPACE_DIVISION_MIN)) {}

  DiscreteStateSpace(const A_Type &A_in, const B_Type &B_in, const C_Type &C_in,
                     const D_Type &D_in)
      : A(A_in), B(B_in), C(C_in), D(D_in),
        delta_time(static_cast<_T>(PythonControl::STATE_SPACE_DIVISION_MIN)) {}

  DiscreteStateSpace(const A_Type &A_in, const B_Type &B_in, const C_Type &C_in,
                     const D_Type &D_in, const _T &delta_time)
      : A(A_in), B(B_in), C(C_in), D(D_in), delta_time(delta_time) {}

  /* Copy Constructor */
  DiscreteStateSpace(
      const DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> &input)
      : A(input.A), B(input.B), C(input.C), D(input.D),
        delta_time(input.delta_time), X(input.X), X_initial(input.X_initial),
        Y(input.Y) {}

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
    }
    return *this;
  }

  /* Move Constructor */
  DiscreteStateSpace(
      DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> &&input) noexcept
      : A(std::move(input.A)), B(std::move(input.B)), C(std::move(input.C)),
        D(std::move(input.D)), delta_time(std::move(input.delta_time)),
        X(std::move(input.X)), X_initial(std::move(input.X_initial)),
        Y(std::move(input.Y)) {}

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
    }
    return *this;
  }

public:
  /* Function */
  constexpr std::size_t get_number_of_delay(void) const {
    return NUMBER_OF_DELAY;
  }

  inline auto get_X(void) const -> Original_X_Type { return this->X; }

  inline auto get_Y(void) const -> Original_Y_Type { return this->Y.get(); }

  template <std::size_t Index> inline auto access_Y(void) -> _T & {
    static_assert(Index < _Output_Size, "Index must be less than output size.");

    return this->Y.template access<Index>();
  }

  inline const std::size_t get_delay_ring_buffer_index(void) const {
    return this->Y.get_delay_ring_buffer_index();
  }

  inline void update(const Original_U_Type &U) {

    this->_calc_output_function(U);

    this->_calc_state_function(U);
  }

  inline void reset_state(void) { this->X = this->X_initial; }

  inline auto output_function(const Original_X_Type &X_in,
                              const Original_U_Type &U) const
      -> Original_Y_Type {

    Original_Y_Type Y_out = this->C * X_in + this->D * U;

    return Y_out;
  }

  inline auto state_function(const Original_X_Type &X_in,
                             const Original_U_Type &U) const
      -> Original_X_Type {

    Original_X_Type X_out = this->A * X_in + this->B * U;

    return X_out;
  }

private:
  /* Function */
  inline void _calc_output_function(const Original_U_Type &U) {

    this->Y.push(this->output_function(this->X, U));
  }

  inline void _calc_state_function(const Original_U_Type &U) {

    this->X = this->state_function(this->X, U);
  }

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

  DelayedVectorObject<Original_Y_Type, Number_Of_Delay> Y;
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

/* Discrete State Space Type */
template <typename A_Type, typename B_Type, typename C_Type, typename D_Type,
          std::size_t Number_Of_Delay = 0>
using DiscreteStateSpace_Type =
    DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type, Number_Of_Delay>;

} // namespace PythonControl

#endif // __PYTHON_CONTROL_STATE_SPACE_HPP__
