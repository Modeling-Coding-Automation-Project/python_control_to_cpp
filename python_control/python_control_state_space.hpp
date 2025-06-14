/**
 * @file python_control_state_space.hpp
 * @brief State-space modeling utilities for discrete-time control systems in
 * C++.
 *
 * This header provides a set of templates and classes for representing and
 * manipulating discrete-time state-space models, including support for delayed
 * input vectors using ring buffers. The utilities are designed to facilitate
 * the implementation of control algorithms and simulation of dynamic systems,
 * with a focus on type safety and compile-time checks for matrix compatibility.
 */
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
  /**
   * @brief Computes and stores a value from the input vector into the buffer at
   * the specified index, then recursively processes the remaining elements.
   *
   * @tparam T The data type of the elements.
   * @tparam Original_Vector_Type The type of the original input vector.
   * @tparam Vector_with_Delay_Type The type of the buffer vector with delay.
   * @tparam Count The current index being processed (used for recursion).
   * @param Vector_buffer The buffer where values are stored, supporting delayed
   * access.
   * @param Vector The original input vector from which values are extracted.
   * @param buffer_index The index in the buffer where the value should be
   * stored.
   */
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
  /**
   * @brief Specialization for the base case of the recursive template, which
   * handles the last element of the vector.
   *
   * @tparam T The data type of the elements.
   * @tparam Original_Vector_Type The type of the original input vector.
   * @tparam Vector_with_Delay_Type The type of the buffer vector with delay.
   * @param Vector_buffer The buffer where values are stored, supporting delayed
   * access.
   * @param Vector The original input vector from which values are extracted.
   * @param buffer_index The index in the buffer where the value should be
   * stored.
   */
  static void compute(Vector_with_Delay_Type &Vector_buffer,
                      const Original_Vector_Type &Vector,
                      const std::size_t &buffer_index) {
    Vector_buffer[buffer_index](0, 0) = Vector.template get<0, 0>();
  }
};

/* Update Ring Buffer Index */
template <std::size_t Number_Of_Buffer> struct UpdateRingBufferIndex {
  /**
   * @brief Updates the index of the ring buffer to point to the next position.
   *
   * This function increments the index, wrapping around to zero if it exceeds
   * the number of buffers.
   *
   * @tparam Number_Of_Buffer The total number of buffers in the ring.
   * @param index The current index to be updated.
   */
  static void compute(std::size_t &index) {
    if (index < Number_Of_Buffer) {
      index++;
    } else {
      index = static_cast<std::size_t>(0);
    }
  }
};

template <> struct UpdateRingBufferIndex<0> {
  /**
   * @brief Specialization for the case where there are no buffers, which does
   * nothing.
   *
   * This function is a no-op since there are no buffers to update.
   *
   * @param index The current index (unused).
   */
  static void compute(std::size_t &index) {
    /* Do Nothing */
    static_cast<void>(index);
  }
};

} // namespace ForDelayedVector

/* Delayed Vector Object */

/**
 * @brief A class that implements a delayed vector object with a ring buffer
 * mechanism.
 *
 * This class allows for storing a vector with a specified number of delays,
 * enabling access to previous states of the vector. It is designed to work with
 * vectors of size 1 x N or N x 1, where N is the number of columns.
 *
 * @tparam Vector_Type The type of the vector, which must be a matrix type with
 * a single row or column.
 * @tparam Number_Of_Delay The number of delays to be stored in the ring buffer.
 */
template <typename Vector_Type, std::size_t Number_Of_Delay>
class DelayedVectorObject {
protected:
  /* Type */
  using _T = typename Vector_Type::Value_Type;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Matrix value data type must be float or double.");

  using _Vector_with_Delay_Type =
      std::array<PythonNumpy::DenseMatrix_Type<_T, Vector_Type::COLS, 1>,
                 (1 + Number_Of_Delay)>;

public:
  /* Type */
  using Value_Type = _T;
  using Original_Vector_Type = Vector_Type;

  /* Check Data Type */
  static_assert(std::is_same<typename Vector_Type::Value_Type, _T>::value,
                "Data type of vector must be same type as T.");
  /* Check Vector Size */
  static_assert(Vector_Type::ROWS == 1,
                "Vector size must be 1 x N or N x 1 matrix.");
  static_assert(Vector_Type::COLS > 0, "Vector size must be greater than 0.");

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

  /**
   * @brief Pushes a new vector into the delayed vector object, updating the
   * ring buffer.
   *
   * This function substitutes the current vector in the ring buffer with the
   * new vector, shifting previous values as necessary to maintain the delay
   * structure.
   *
   * @param vector The new vector to be pushed into the delayed vector object.
   */
  inline void push(const Vector_Type &vector) {

    ForDelayedVector::SubstituteVectorRingBuffer<
        _T, Vector_Type, _Vector_with_Delay_Type,
        (Vector_Type::COLS - 1)>::compute(this->_store, vector,
                                          this->_delay_ring_buffer_index);

    ForDelayedVector::UpdateRingBufferIndex<Number_Of_Delay>::compute(
        this->_delay_ring_buffer_index);
  }

  /**
   * @brief Pushes a new vector into the delayed vector object, updating the
   * ring buffer.
   *
   * This function substitutes the current vector in the ring buffer with the
   * new vector, shifting previous values as necessary to maintain the delay
   * structure.
   *
   * @param vector The new vector to be pushed into the delayed vector object.
   */
  inline auto get(void) const -> Vector_Type {

    return this->_store[this->_delay_ring_buffer_index];
  }

  /**
   * @brief Retrieves a vector from the delayed vector object by index.
   *
   * This function returns the vector at the specified index in the ring buffer,
   * allowing access to previous states of the vector.
   *
   * @param index The index of the vector to retrieve, where 0 is the latest
   * vector and higher indices correspond to older vectors.
   * @return The vector at the specified index.
   */
  inline auto get_by_index(std::size_t index) const -> Vector_Type {

    static_assert(Number_Of_Delay > 0,
                  "Number of delay must be greater than 0.");
    if (index > Number_Of_Delay) {
      index = Number_Of_Delay;
    }

    return this->_store[index];
  }

  /**
   * @brief Retrieves the latest vector from the delayed vector object.
   *
   * This function returns the most recent vector stored in the ring buffer,
   * which is the last vector pushed into the object.
   *
   * @return The latest vector stored in the delayed vector object.
   */
  inline auto get_latest(void) const -> Vector_Type {
    std::size_t index = this->_delay_ring_buffer_index;
    if (static_cast<std::size_t>(0) == index) {
      index = Number_Of_Delay;
    } else {
      index = index - 1;
    }

    return this->_store[index];
  }

  /**
   * @brief Accesses a specific element in the latest vector.
   *
   * This function allows direct access to an element in the latest vector
   * stored in the ring buffer, specified by its index.
   *
   * @tparam Index The index of the element to access, where 0 is the first
   * element.
   * @return A reference to the element at the specified index in the latest
   * vector.
   */
  template <std::size_t Index> inline auto access(void) -> _T & {
    static_assert(Index < Vector_Type::COLS,
                  "Index must be less than vector size.");

    return this->_store[this->_delay_ring_buffer_index].access(Index, 0);
  }

  /**
   * @brief Retrieves the current index of the ring buffer.
   *
   * This function returns the index of the ring buffer where the next vector
   * will be pushed, allowing for tracking of the current position in the
   * buffer.
   *
   * @return The current index of the ring buffer.
   */
  inline const std::size_t get_delay_ring_buffer_index(void) const {
    return this->_delay_ring_buffer_index;
  }

public:
  /* Constant */
  static constexpr std::size_t VECTOR_SIZE = Vector_Type::COLS;
  static constexpr std::size_t NUMBER_OF_DELAY = Number_Of_Delay;

protected:
  /* Variable */
  _Vector_with_Delay_Type _store;
  std::size_t _delay_ring_buffer_index;
};

template <typename T, std::size_t Vector_Size>
using StateSpaceInput_Type = PythonNumpy::DenseMatrix_Type<T, Vector_Size, 1>;

template <typename T, std::size_t Vector_Size>
using StateSpaceState_Type = PythonNumpy::DenseMatrix_Type<T, Vector_Size, 1>;

template <typename T, std::size_t Vector_Size>
using StateSpaceOutput_Type = PythonNumpy::DenseMatrix_Type<T, Vector_Size, 1>;

namespace MakeStateSpaceVector {

/**
 * @brief Assigns values to a StateSpaceVectorType object recursively.
 *
 * This function assigns values to the StateSpaceVectorType object at
 * specified indices, allowing for a variable number of arguments.
 *
 * @tparam IndexCount The current index count for recursion.
 * @tparam StateSpaceVectorType The type of the state space vector to be
 * modified.
 * @tparam T The type of the first value to assign.
 * @param input The state space vector object to modify.
 * @param value_1 The first value to assign.
 */
template <std::size_t IndexCount, typename StateSpaceVectorType, typename T>
inline void assign_values(StateSpaceVectorType &input, T value_1) {

  static_assert(
      IndexCount < StateSpaceVectorType::COLS,
      "Number of arguments must be less than the number of input size.");

  input.template set<IndexCount, 0>(value_1);
}

/**
 * @brief Assigns values to a StateSpaceVectorType object recursively with
 * multiple arguments.
 *
 * This function assigns values to the StateSpaceVectorType object at
 * specified indices, allowing for a variable number of arguments. It ensures
 * that all arguments are of the same type.
 *
 * @tparam IndexCount The current index count for recursion.
 * @tparam StateSpaceVectorType The type of the state space vector to be
 * modified.
 * @tparam T The type of the first value to assign.
 * @tparam U The type of the second value to assign.
 * @param input The state space vector object to modify.
 * @param value_1 The first value to assign.
 * @param value_2 The second value to assign.
 * @param args Additional values to assign, if any.
 */
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

/**
 * @brief Creates a StateSpaceInput_Type object with specified values.
 *
 * This function constructs a StateSpaceInput_Type object by assigning values
 * to its elements using a variadic template. It allows for a variable number of
 * arguments to be passed, which are assigned to the vector.
 *
 * @tparam Vector_Size The size of the state space vector.
 * @tparam T The type of the first value to assign.
 * @tparam Args Additional types for subsequent values.
 * @param value_1 The first value to assign to the vector.
 * @param args Additional values to assign, if any.
 * @return A StateSpaceInput_Type object initialized with the provided values.
 */
template <std::size_t Vector_Size, typename T, typename... Args>
inline auto make_StateSpaceInput(T value_1, Args... args)
    -> StateSpaceInput_Type<T, Vector_Size> {

  StateSpaceInput_Type<T, Vector_Size> input;

  MakeStateSpaceVector::assign_values<0>(input, value_1, args...);

  return input;
}

/**
 * @brief Creates a StateSpaceState_Type object with specified values.
 *
 * This function constructs a StateSpaceState_Type object by assigning values
 * to its elements using a variadic template. It allows for a variable number of
 * arguments to be passed, which are assigned to the vector.
 *
 * @tparam Vector_Size The size of the state space vector.
 * @tparam T The type of the first value to assign.
 * @tparam Args Additional types for subsequent values.
 * @param value_1 The first value to assign to the vector.
 * @param args Additional values to assign, if any.
 * @return A StateSpaceState_Type object initialized with the provided values.
 */
template <std::size_t Vector_Size, typename T, typename... Args>
inline auto make_StateSpaceState(T value_1, Args... args)
    -> StateSpaceState_Type<T, Vector_Size> {

  StateSpaceState_Type<T, Vector_Size> input;

  MakeStateSpaceVector::assign_values<0>(input, value_1, args...);

  return input;
}

/**
 * @brief Creates a StateSpaceOutput_Type object with specified values.
 *
 * This function constructs a StateSpaceOutput_Type object by assigning values
 * to its elements using a variadic template. It allows for a variable number of
 * arguments to be passed, which are assigned to the vector.
 *
 * @tparam Vector_Size The size of the state space vector.
 * @tparam T The type of the first value to assign.
 * @tparam Args Additional types for subsequent values.
 * @param value_1 The first value to assign to the vector.
 * @param args Additional values to assign, if any.
 * @return A StateSpaceOutput_Type object initialized with the provided values.
 */
template <std::size_t Vector_Size, typename T, typename... Args>
inline auto make_StateSpaceOutput(T value_1, Args... args)
    -> StateSpaceOutput_Type<T, Vector_Size> {

  StateSpaceOutput_Type<T, Vector_Size> input;

  MakeStateSpaceVector::assign_values<0>(input, value_1, args...);

  return input;
}

/* Discrete State Space */

/**
 * @brief A class representing a discrete-time state-space model.
 *
 * This class encapsulates the state-space representation of a discrete-time
 * system, including matrices A, B, C, and D, as well as the time step for
 * simulation. It provides methods for initializing and manipulating the state
 * space model.
 *
 * @tparam A_Type_In The type of the state transition matrix A.
 * @tparam B_Type_In The type of the input matrix B.
 * @tparam C_Type_In The type of the output matrix C.
 * @tparam D_Type_In The type of the feedforward matrix D.
 * @tparam Number_Of_Delay The number of delays in the input vector (default is
 * 0).
 */
template <typename A_Type_In, typename B_Type_In, typename C_Type_In,
          typename D_Type_In, std::size_t Number_Of_Delay = 0>
class DiscreteStateSpace {
public:
  /* Type */
  using A_Type = A_Type_In;
  using B_Type = B_Type_In;
  using C_Type = C_Type_In;
  using D_Type = D_Type_In;

protected:
  /* Type */
  using _T = typename A_Type::Value_Type;
  static_assert(std::is_same<_T, double>::value ||
                    std::is_same<_T, float>::value,
                "Matrix value data type must be float or double.");

  static constexpr std::size_t _Input_Size = B_Type::ROWS;
  static constexpr std::size_t _State_Size = A_Type::COLS;
  static constexpr std::size_t _Output_Size = C_Type::COLS;

public:
  /* Type */
  using Value_Type = _T;

  using Original_U_Type = PythonControl::StateSpaceInput_Type<_T, _Input_Size>;
  using Original_X_Type = PythonNumpy::DenseMatrix_Type<_T, _State_Size, 1>;
  using Original_Y_Type = PythonNumpy::DenseMatrix_Type<_T, _Output_Size, 1>;

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
        U(), X(), X_initial(), Y() {}

  DiscreteStateSpace(const A_Type &A_in)
      : A(A_in), B(), C(), D(),
        delta_time(static_cast<_T>(PythonControl::STATE_SPACE_DIVISION_MIN)),
        U(), X(), X_initial(), Y() {}

  DiscreteStateSpace(const A_Type &A_in, const B_Type &B_in)
      : A(A_in), B(B_in), C(), D(),
        delta_time(static_cast<_T>(PythonControl::STATE_SPACE_DIVISION_MIN)),
        U(), X(), X_initial(), Y() {}

  DiscreteStateSpace(const A_Type &A_in, const B_Type &B_in, const C_Type &C_in)
      : A(A_in), B(B_in), C(C_in), D(),
        delta_time(static_cast<_T>(PythonControl::STATE_SPACE_DIVISION_MIN)),
        U(), X(), X_initial(), Y() {}

  DiscreteStateSpace(const A_Type &A_in, const B_Type &B_in, const C_Type &C_in,
                     const D_Type &D_in)
      : A(A_in), B(B_in), C(C_in), D(D_in),
        delta_time(static_cast<_T>(PythonControl::STATE_SPACE_DIVISION_MIN)),
        U(), X(), X_initial(), Y() {}

  DiscreteStateSpace(const A_Type &A_in, const B_Type &B_in, const C_Type &C_in,
                     const D_Type &D_in, const _T &delta_time)
      : A(A_in), B(B_in), C(C_in), D(D_in), delta_time(delta_time), U(), X(),
        X_initial(), Y() {}

  /* Copy Constructor */
  DiscreteStateSpace(
      const DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> &input)
      : A(input.A), B(input.B), C(input.C), D(input.D),
        delta_time(input.delta_time), U(input.U), X(input.X),
        X_initial(input.X_initial), Y(input.Y) {}

  DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> &
  operator=(const DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> &input) {
    if (this != &input) {
      this->A = input.A;
      this->B = input.B;
      this->C = input.C;
      this->D = input.D;
      this->delta_time = input.delta_time;
      this->U = input.U;
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
        U(std::move(input.U)), X(std::move(input.X)),
        X_initial(std::move(input.X_initial)), Y(std::move(input.Y)) {}

  DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> &operator=(
      DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> &&input) noexcept {
    if (this != &input) {
      this->A = std::move(input.A);
      this->B = std::move(input.B);
      this->C = std::move(input.C);
      this->D = std::move(input.D);
      this->delta_time = std::move(input.delta_time);
      this->U = std::move(input.U);
      this->X = std::move(input.X);
      this->X_initial = std::move(input.X_initial);
      this->Y = std::move(input.Y);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Returns the number of delays in the input vector.
   *
   * This function provides the number of delays that the state-space model
   * supports, which is determined at compile time.
   *
   * @return The number of delays in the input vector.
   */
  constexpr std::size_t get_number_of_delay(void) const {
    return NUMBER_OF_DELAY;
  }

  /**
   * @brief Returns the delta time used in the state-space model.
   *
   * This function provides the time step used for simulation in the discrete
   * state-space model.
   *
   * @return The delta time as a value of type _T.
   */
  inline auto get_X(void) const -> Original_X_Type { return this->X; }

  /**
   * @brief Returns the initial state of the system.
   *
   * This function provides the initial state vector of the system, which is
   * used to reset the state when needed.
   *
   * @return The initial state vector as a value of type Original_X_Type.
   */
  inline auto get_Y(void) const -> Original_Y_Type { return this->Y; }

  template <std::size_t Index> inline auto access_U(void) -> _T & {
    static_assert(Index < _Input_Size, "Index must be less than input size.");

    return this->U.template access<Index>();
  }

  inline const std::size_t get_delay_ring_buffer_index(void) const {
    return this->U.get_delay_ring_buffer_index();
  }

  /**
   * @brief Updates the state-space model with a new input vector.
   *
   * This function pushes a new input vector into the delayed input vector
   * object, calculates the output and state functions, and updates the internal
   * state accordingly.
   *
   * @param U_in The input vector to be processed, of type Original_U_Type.
   */
  inline void update(const Original_U_Type &U_in) {

    this->U.push(U_in);

    this->_calc_output_function();

    this->_calc_state_function();
  }

  /**
   * @brief Initializes the state-space model with a given initial state.
   *
   * This function sets the initial state of the system, which is used to reset
   * the state when needed.
   *
   * @param X_initial_in The initial state vector to be set, of type
   * Original_X_Type.
   */
  inline void reset_state(void) { this->X = this->X_initial; }

  /**
   * @brief Sets the initial state of the system.
   *
   * This function allows the user to set a specific initial state for the
   * system, which will be used when resetting the state.
   *
   * @param X_initial_in The initial state vector to be set, of type
   * Original_X_Type.
   */
  inline auto output_function(const Original_X_Type &X_in,
                              const Original_U_Type &U_in) const
      -> Original_Y_Type {

    Original_Y_Type Y_out = this->C * X_in + this->D * U_in;

    return Y_out;
  }

  /**
   * @brief Computes the state update function for the discrete state-space
   * model.
   *
   * This function calculates the next state of the system based on the current
   * state and input, using the state transition matrix A and input matrix B.
   *
   * @param X_in The current state vector, of type Original_X_Type.
   * @param U_in The input vector, of type Original_U_Type.
   * @return The next state vector as a value of type Original_X_Type.
   */
  inline auto state_function(const Original_X_Type &X_in,
                             const Original_U_Type &U_in) const
      -> Original_X_Type {

    Original_X_Type X_out = this->A * X_in + this->B * U_in;

    return X_out;
  }

protected:
  /* Function */

  /**
   * @brief Calculates the output function based on the current state and input.
   *
   * This function computes the output of the system using the output function
   * defined by the matrices C and D, applied to the current state and input.
   */
  inline void _calc_output_function(void) {

    this->Y = this->output_function(this->X, this->U.get());
  }

  /**
   * @brief Calculates the state update function based on the current state and
   * input.
   *
   * This function computes the next state of the system using the state
   * transition function defined by the matrices A and B, applied to the current
   * state and input.
   */
  inline void _calc_state_function(void) {

    this->X = this->state_function(this->X, this->U.get());
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

  DelayedVectorObject<Original_U_Type, Number_Of_Delay> U;

  Original_X_Type X;
  Original_X_Type X_initial;

  Original_Y_Type Y;
};

/* Make Discrete State Space */

/**
 * @brief Factory function to create an instance of the DiscreteStateSpace
 * class.
 *
 * This function constructs a discrete state-space model using the provided
 * matrices A, B, C, and D, along with the time step for simulation. It returns
 * an instance of the DiscreteStateSpace class with these parameters.
 *
 * @tparam A_Type The type of the state transition matrix A.
 * @tparam B_Type The type of the input matrix B.
 * @tparam C_Type The type of the output matrix C.
 * @tparam D_Type The type of the feedforward matrix D.
 * @param A The state transition matrix A.
 * @param B The input matrix B.
 * @param C The output matrix C.
 * @param D The feedforward matrix D.
 * @param delta_time The time step for simulation.
 * @return An instance of the DiscreteStateSpace class initialized
 * with the provided matrices and time step.
 */
template <typename A_Type, typename B_Type, typename C_Type, typename D_Type>
inline auto make_DiscreteStateSpace(A_Type A, B_Type B, C_Type C, D_Type D,
                                    typename A_Type::Value_Type delta_time)
    -> DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> {

  return DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type>(A, B, C, D,
                                                            delta_time);
}

/**
 * @brief Factory function to create an instance of the DiscreteStateSpace
 * class with a specified number of delays.
 *
 * This function constructs a discrete state-space model using the provided
 * matrices A, B, C, and D, along with the time step for simulation and the
 * number of delays in the input vector. It returns an instance of the
 * DiscreteStateSpace class with these parameters.
 *
 * @tparam Number_Of_Delay The number of delays in the input vector.
 * @tparam A_Type The type of the state transition matrix A.
 * @tparam B_Type The type of the input matrix B.
 * @tparam C_Type The type of the output matrix C.
 * @tparam D_Type The type of the feedforward matrix D.
 * @param A The state transition matrix A.
 * @param B The input matrix B.
 * @param C The output matrix C.
 * @param D The feedforward matrix D.
 * @param delta_time The time step for simulation.
 * @return An instance of the DiscreteStateSpace class initialized
 * with the provided matrices, time step, and number of delays.
 */
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
