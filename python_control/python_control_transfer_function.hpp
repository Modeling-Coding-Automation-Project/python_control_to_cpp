#ifndef __PYTHON_CONTROL_TRANSFER_FUNCTION_HPP__
#define __PYTHON_CONTROL_TRANSFER_FUNCTION_HPP__

#include "base_utility.hpp"
#include "python_control_state_space.hpp"
#include "python_numpy.hpp"

#include <type_traits>
#include <utility>

namespace PythonControl {

constexpr double TRANSFER_FUNCTION_DIVISION_MIN = 1.0e-10;

/* Numerator and Denominator definition */
template <typename T, std::size_t Numerator_Size>
using TransferFunctionNumeratorType = PythonNumpy::SparseMatrix_Type<
    T, PythonNumpy::DenseAvailable<Numerator_Size, 1>>;

template <typename T, std::size_t Denominator_Size>
using TransferFunctionDenominatorType = PythonNumpy::SparseMatrix_Type<
    T, PythonNumpy::DenseAvailable<Denominator_Size, 1>>;

namespace MakeNumerator {

template <std::size_t IndexCount, typename TransferFunctionNumeratorType,
          typename T>
inline void assign_values(TransferFunctionNumeratorType &numerator, T value_1) {

  static_assert(
      IndexCount < TransferFunctionNumeratorType::COLS,
      "Number of arguments must be less than the number of Numerator factor.");

  numerator.template set<IndexCount, 0>(value_1);
}

template <std::size_t IndexCount, typename TransferFunctionNumeratorType,
          typename T, typename U, typename... Args>
inline void assign_values(TransferFunctionNumeratorType &numerator, T value_1,
                          U value_2, Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");
  static_assert(
      IndexCount < TransferFunctionNumeratorType::COLS,
      "Number of arguments must be less than the number of Numerator factor.");

  numerator.template set<IndexCount, 0>(value_1);

  assign_values<IndexCount + 1>(numerator, value_2, args...);
}

} // namespace MakeNumerator

namespace MakeDenominator {

template <std::size_t IndexCount, typename TransferFunctionDenominatorType,
          typename T>
inline void assign_values(TransferFunctionDenominatorType &denominator,
                          T value_1) {

  static_assert(
      IndexCount < TransferFunctionDenominatorType::COLS,
      "Number of arguments must be less than the number of Numerator factor.");

  denominator.template set<IndexCount, 0>(value_1);
}

template <std::size_t IndexCount, typename TransferFunctionDenominatorType,
          typename T, typename U, typename... Args>
inline void assign_values(TransferFunctionDenominatorType &denominator,
                          T value_1, U value_2, Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");
  static_assert(
      IndexCount < TransferFunctionDenominatorType::COLS,
      "Number of arguments must be less than the number of Numerator factor.");

  denominator.template set<IndexCount, 0>(value_1);

  assign_values<IndexCount + 1>(denominator, value_2, args...);
}

} // namespace MakeDenominator

/* make Numerator and Denominator */
template <std::size_t M, typename T, typename... Args>
inline auto make_TransferFunctionNumerator(T value_1, Args... args)
    -> TransferFunctionNumeratorType<T, M> {

  TransferFunctionNumeratorType<T, M> numerator;

  MakeNumerator::assign_values<0>(numerator, value_1, args...);

  return numerator;
}

template <std::size_t M, typename T, typename... Args>
inline auto make_TransferFunctionDenominator(T value_1, Args... args)
    -> TransferFunctionNumeratorType<T, M> {

  TransferFunctionNumeratorType<T, M> denominator;

  MakeNumerator::assign_values<0>(denominator, value_1, args...);

  return denominator;
}

namespace ForDiscreteTransferFunction {

/* Create A type definition */
template <typename T, std::size_t N> struct DiscreteStateSpace_A_Type {

  using SparseAvailable_DiscreteStateSpace_A =
      PythonNumpy::ConcatenateSparseAvailableVertically<
          PythonNumpy::DenseAvailable<1, N>,
          PythonNumpy::ConcatenateSparseAvailableHorizontally<
              PythonNumpy::DiagAvailable<N - 1>,
              PythonNumpy::SparseAvailableEmpty<N - 1, 1>>>;

  using type =
      PythonNumpy::SparseMatrix_Type<T, SparseAvailable_DiscreteStateSpace_A>;
};

/* Create B type definition */
template <typename T, std::size_t N> struct DiscreteStateSpace_B_Type {
  using type = PythonNumpy::SparseMatrix_Type<
      T, PythonNumpy::ConcatenateSparseAvailableVertically<
             PythonNumpy::DenseAvailable<1, 1>,
             PythonNumpy::SparseAvailableEmpty<N - 1, 1>>>;
};

/* Create C type definition */
template <typename T, std::size_t Denominator_Size, std::size_t Den_Num_Dif>
struct DiscreteStateSpace_C_Type {
  using type = PythonNumpy::SparseMatrix_Type<
      T, PythonNumpy::ConcatenateSparseAvailableHorizontally<
             PythonNumpy::DenseAvailable<1, (Den_Num_Dif - 1)>,
             PythonNumpy::DenseAvailable<1, (Denominator_Size - Den_Num_Dif)>>>;
};

template <typename T, std::size_t Denominator_Size>
struct DiscreteStateSpace_C_Type<T, Denominator_Size, 1> {
  using type = PythonNumpy::SparseMatrix_Type<
      T, PythonNumpy::DenseAvailable<1, (Denominator_Size - 1)>>;
};

template <typename T, std::size_t Denominator_Size>
struct DiscreteStateSpace_C_Type<T, Denominator_Size, 0> {
  using type = PythonNumpy::SparseMatrix_Type<
      T, PythonNumpy::DenseAvailable<1, (Denominator_Size - 1)>>;
};

/* Create D type definition */
template <typename T, bool IsStrictlyProper>
struct DiscreteStateSpace_D_Type {};

template <typename T> struct DiscreteStateSpace_D_Type<T, true> {
  using type = PythonNumpy::SparseMatrix_Type<
      T, PythonNumpy::SparseAvailable<PythonNumpy::ColumnAvailable<false>>>;
};

template <typename T> struct DiscreteStateSpace_D_Type<T, false> {
  using type = PythonNumpy::SparseMatrix_Type<
      T, PythonNumpy::SparseAvailable<PythonNumpy::ColumnAvailable<true>>>;
};

/* Set A value */
template <typename T, typename DiscreteStateSpace_A_type,
          typename Denominator_Type, std::size_t I, std::size_t LoopCount>
struct Set_A_Value {
  static void set(DiscreteStateSpace_A_type &A,
                  const Denominator_Type &denominator) {
    A(I) = -denominator(I + 1) /
           Base::Utility::avoid_zero_divide(
               denominator(0),
               static_cast<T>(PythonControl::TRANSFER_FUNCTION_DIVISION_MIN));
    Set_A_Value<T, DiscreteStateSpace_A_type, Denominator_Type, (I + 1),
                (LoopCount - 1)>::set(A, denominator);
  }
};

template <typename T, typename DiscreteStateSpace_A_type,
          typename Denominator_Type, std::size_t I>
struct Set_A_Value<T, DiscreteStateSpace_A_type, Denominator_Type, I, 0> {
  static void set(DiscreteStateSpace_A_type &A,
                  const Denominator_Type &denominator) {
    A(I) = -denominator(I + 1) /
           Base::Utility::avoid_zero_divide(
               denominator(0),
               static_cast<T>(PythonControl::TRANSFER_FUNCTION_DIVISION_MIN));
  }
};

template <typename T, typename DiscreteStateSpace_A_type,
          typename std::size_t I, std::size_t LoopCount>
struct Set_A_Ones {
  static void set(DiscreteStateSpace_A_type &A) {
    A(I) = static_cast<T>(1);
    Set_A_Ones<T, DiscreteStateSpace_A_type, (I + 1), (LoopCount - 1)>::set(A);
  }
};

template <typename T, typename DiscreteStateSpace_A_type, std::size_t I>
struct Set_A_Ones<T, DiscreteStateSpace_A_type, I, 0> {
  static void set(DiscreteStateSpace_A_type &A) { A(I) = static_cast<T>(1); }
};

/* Set C value */
template <typename T, typename DiscreteStateSpace_C_type,
          typename Numerator_Type, typename Denominator_Type,
          std::size_t C_INDEX_OFFSET, std::size_t I, std::size_t LoopCount,
          bool Is_Strictly_Proper>
struct Set_C_ValueElements {};

template <typename T, typename DiscreteStateSpace_C_type,
          typename Numerator_Type, typename Denominator_Type,
          std::size_t C_INDEX_OFFSET, std::size_t I, std::size_t LoopCount>
struct Set_C_ValueElements<T, DiscreteStateSpace_C_type, Numerator_Type,
                           Denominator_Type, C_INDEX_OFFSET, I, LoopCount,
                           true> {
  static void set(DiscreteStateSpace_C_type &C, const Numerator_Type &numerator,
                  const T &denominator_0_inv) {
    C(I + C_INDEX_OFFSET) = numerator(I) * denominator_0_inv;

    Set_C_ValueElements<T, DiscreteStateSpace_C_type, Numerator_Type,
                        Denominator_Type, C_INDEX_OFFSET, (I + 1),
                        (LoopCount - 1), true>::set(C, numerator,
                                                    denominator_0_inv);
  }
};

template <typename T, typename DiscreteStateSpace_C_type,
          typename Numerator_Type, typename Denominator_Type,
          std::size_t C_INDEX_OFFSET, std::size_t I>
struct Set_C_ValueElements<T, DiscreteStateSpace_C_type, Numerator_Type,
                           Denominator_Type, C_INDEX_OFFSET, I, 0, true> {
  static void set(DiscreteStateSpace_C_type &C, const Numerator_Type &numerator,
                  const T &denominator_0_inv) {
    C(I + C_INDEX_OFFSET) = numerator(I) * denominator_0_inv;
  }
};

template <typename T, typename DiscreteStateSpace_C_type,
          typename Numerator_Type, typename Denominator_Type,
          std::size_t C_INDEX_OFFSET, std::size_t I, std::size_t LoopCount>
struct Set_C_ValueElements<T, DiscreteStateSpace_C_type, Numerator_Type,
                           Denominator_Type, C_INDEX_OFFSET, I, LoopCount,
                           false> {
  static void set(DiscreteStateSpace_C_type &C, const Numerator_Type &numerator,
                  const Denominator_Type &denominator,
                  const T &denominator_0_inv) {
    C(I - 1) =
        numerator(I) * denominator_0_inv -
        denominator(I) * numerator(0) * (denominator_0_inv * denominator_0_inv);

    Set_C_ValueElements<T, DiscreteStateSpace_C_type, Numerator_Type,
                        Denominator_Type, C_INDEX_OFFSET, (I + 1),
                        (LoopCount - 1), false>::set(C, numerator, denominator,
                                                     denominator_0_inv);
  }
};

template <typename T, typename DiscreteStateSpace_C_type,
          typename Numerator_Type, typename Denominator_Type,
          std::size_t C_INDEX_OFFSET, std::size_t I>
struct Set_C_ValueElements<T, DiscreteStateSpace_C_type, Numerator_Type,
                           Denominator_Type, C_INDEX_OFFSET, I, 0, false> {
  static void set(DiscreteStateSpace_C_type &C, const Numerator_Type &numerator,
                  const Denominator_Type &denominator,
                  const T &denominator_0_inv) {
    C(I - 1) =
        numerator(I) * denominator_0_inv -
        denominator(I) * numerator(0) * (denominator_0_inv * denominator_0_inv);
  }
};

template <typename T, typename DiscreteStateSpace_C_type,
          typename Numerator_Type, typename Denominator_Type,
          bool Is_Strictly_Proper>
struct Set_C_Value {};

template <typename T, typename DiscreteStateSpace_C_type,
          typename Numerator_Type, typename Denominator_Type>
struct Set_C_Value<T, DiscreteStateSpace_C_type, Numerator_Type,
                   Denominator_Type, true> {

  static void set(DiscreteStateSpace_C_type &C, const Numerator_Type &numerator,
                  const Denominator_Type &denominator) {

    T denominator_0_inv =
        static_cast<T>(1) /
        Base::Utility::avoid_zero_divide(
            denominator(0),
            static_cast<T>(PythonControl::TRANSFER_FUNCTION_DIVISION_MIN));
    constexpr std::size_t C_INDEX_OFFSET =
        Denominator_Type::COLS - Numerator_Type::COLS - 1;

    Set_C_ValueElements<T, DiscreteStateSpace_C_type, Numerator_Type,
                        Denominator_Type, C_INDEX_OFFSET, 0,
                        (Numerator_Type::COLS - 1),
                        true>::set(C, numerator, denominator_0_inv);
  }
};

template <typename T, typename DiscreteStateSpace_C_type,
          typename Numerator_Type, typename Denominator_Type>
struct Set_C_Value<T, DiscreteStateSpace_C_type, Numerator_Type,
                   Denominator_Type, false> {

  static void set(DiscreteStateSpace_C_type &C, const Numerator_Type &numerator,
                  const Denominator_Type &denominator) {

    T denominator_0_inv =
        static_cast<T>(1) /
        Base::Utility::avoid_zero_divide(
            denominator(0),
            static_cast<T>(PythonControl::TRANSFER_FUNCTION_DIVISION_MIN));

    Set_C_ValueElements<T, DiscreteStateSpace_C_type, Numerator_Type,
                        Denominator_Type, 0, 1, (Numerator_Type::COLS - 2),
                        false>::set(C, numerator, denominator,
                                    denominator_0_inv);
  }
};

/* Set D value */
template <typename T, typename DiscreteStateSpace_D_type,
          typename Numerator_Type, typename Denominator_Type,
          bool Is_Strictly_Proper>
struct Set_D_Value {};

template <typename T, typename DiscreteStateSpace_D_type,
          typename Numerator_Type, typename Denominator_Type>
struct Set_D_Value<T, DiscreteStateSpace_D_type, Numerator_Type,
                   Denominator_Type, true> {
  static void set(DiscreteStateSpace_D_type &D, const Numerator_Type &numerator,
                  const Denominator_Type &denominator) {
    /* Do Nothing */
    static_cast<void>(D);
    static_cast<void>(numerator);
    static_cast<void>(denominator);
  }
};

template <typename T, typename DiscreteStateSpace_D_type,
          typename Numerator_Type, typename Denominator_Type>
struct Set_D_Value<T, DiscreteStateSpace_D_type, Numerator_Type,
                   Denominator_Type, false> {
  static void set(DiscreteStateSpace_D_type &D, const Numerator_Type &numerator,
                  const Denominator_Type &denominator) {
    D(0) = numerator(0) /
           Base::Utility::avoid_zero_divide(
               denominator(0),
               static_cast<T>(PythonControl::TRANSFER_FUNCTION_DIVISION_MIN));
  }
};

/* Solve steady state and input */
template <typename T, typename State_Space_Type, bool Is_Strictly_Proper>
struct SolveSteadyStateAndInput {};

template <typename T, typename State_Space_Type>
struct SolveSteadyStateAndInput<T, State_Space_Type, true> {
  static T solve(State_Space_Type &state_space, const T &y_steady_state) {
    T u_steady_state;

    auto I_A = PythonNumpy::make_DiagMatrixIdentity<
                   T, State_Space_Type::Original_X_Type::COLS>() -
               state_space.A;

    auto solver = PythonNumpy::make_LinalgSolver<decltype(I_A),
                                                 decltype(state_space.B)>();
    solver.solve(I_A, state_space.B);

    auto I_A_B = solver.get_answer();

    auto C_I_A_B = state_space.C * I_A_B;

    // This is a transfer function, so it is single input single output
    // system. Thus, "y_steady_state" and "C_I_A_B" are scalar.
    u_steady_state =
        y_steady_state /
        Base::Utility::avoid_zero_divide(
            C_I_A_B.template get<0, 0>(),
            static_cast<T>(PythonControl::TRANSFER_FUNCTION_DIVISION_MIN));

    state_space.X = I_A_B * u_steady_state;

    return u_steady_state;
  }
};

template <typename T, typename State_Space_Type>
struct SolveSteadyStateAndInput<T, State_Space_Type, false> {
  static T solve(State_Space_Type &state_space, const T &y_steady_state) {
    T u_steady_state;

    auto I_A = PythonNumpy::make_DiagMatrixIdentity<
                   T, State_Space_Type::Original_X_Type::COLS>() -
               state_space.A;

    // This is a transfer function, so D is scalar.
    auto I_A_D_B_C = I_A * state_space.D.template get<0, 0>() +
                     state_space.B * state_space.C;

    auto B_y = state_space.B * y_steady_state;

    auto solver =
        PythonNumpy::make_LinalgSolver<decltype(I_A_D_B_C), decltype(B_y)>();
    solver.solve(I_A_D_B_C, B_y);

    state_space.X = solver.get_answer();

    u_steady_state =
        (y_steady_state -
         (state_space.C * state_space.X).template get<0, 0>()) /
        Base::Utility::avoid_zero_divide(
            state_space.D.template get<0, 0>(),
            static_cast<T>(PythonControl::TRANSFER_FUNCTION_DIVISION_MIN));

    return u_steady_state;
  }
};

} // namespace ForDiscreteTransferFunction

/* Discrete Transfer Function */
template <typename Numerator_Type, typename Denominator_Type,
          std::size_t Number_Of_Delay = 0>
class DiscreteTransferFunction {
private:
  /* Type */
  using _T = typename Numerator_Type::Value_Type;
  static_assert(
      std::is_same<_T, double>::value || std::is_same<_T, float>::value,
      "Numerator and denominator value data type must be float or double.");

  static_assert(Denominator_Type::COLS > 1,
                "Denominator must have at least 2 elements.");
  static constexpr std::size_t _DENOMINATOR_DIMENSION =
      Denominator_Type::COLS - 1;
  static_assert(Numerator_Type::COLS > 1,
                "Numerator must have at least 2 elements.");
  static constexpr std::size_t _NUMERATOR_DIMENSION = Numerator_Type::COLS - 1;

  static constexpr bool _IS_STRICTLY_PROPER =
      _NUMERATOR_DIMENSION < _DENOMINATOR_DIMENSION;

  using _DiscreteStateSpace_A_type =
      typename ForDiscreteTransferFunction::DiscreteStateSpace_A_Type<
          _T, _DENOMINATOR_DIMENSION>::type;

  using _DiscreteStateSpace_B_type =
      typename ForDiscreteTransferFunction::DiscreteStateSpace_B_Type<
          _T, _DENOMINATOR_DIMENSION>::type;

  using _DiscreteStateSpace_C_type =
      typename ForDiscreteTransferFunction::DiscreteStateSpace_C_Type<
          _T, Denominator_Type::COLS,
          (Denominator_Type::COLS - Numerator_Type::COLS)>::type;

  using _DiscreteStateSpace_D_type =
      typename ForDiscreteTransferFunction::DiscreteStateSpace_D_Type<
          _T, _IS_STRICTLY_PROPER>::type;

  using _U_Type = PythonControl::StateSpaceInputType<_T, 1>;

  using _State_Space_Type = PythonControl::DiscreteStateSpace<
      _DiscreteStateSpace_A_type, _DiscreteStateSpace_B_type,
      _DiscreteStateSpace_C_type, _DiscreteStateSpace_D_type, Number_Of_Delay>;

  /* Check Compatibility */
  /* Check Data Type */
  static_assert(std::is_same<typename Denominator_Type::Value_Type, _T>::value,
                "Data type of denominator must be same type as numerator.");

  /* Check Numerator and Denominator length */
  static_assert(Numerator_Type::COLS <= Denominator_Type::COLS,
                "Transfer function must be proper.");

public:
  /* Constructor */
  DiscreteTransferFunction() {}

  DiscreteTransferFunction(const Numerator_Type &numerator,
                           const Denominator_Type &denominator, _T delta_time) {
    /* Set A */
    ForDiscreteTransferFunction::Set_A_Value<
        _T, _DiscreteStateSpace_A_type, Denominator_Type, 0,
        (_DENOMINATOR_DIMENSION - 1)>::set(this->_state_space.A, denominator);

    ForDiscreteTransferFunction::Set_A_Ones<
        _T, _DiscreteStateSpace_A_type, _DENOMINATOR_DIMENSION,
        (2 * (_DENOMINATOR_DIMENSION - 1))>::set(this->_state_space.A);

    /* Set B */
    this->_state_space.B(0) = static_cast<_T>(1);

    /* Set C */
    ForDiscreteTransferFunction::Set_C_Value<
        _T, _DiscreteStateSpace_C_type, Numerator_Type, Denominator_Type,
        _IS_STRICTLY_PROPER>::set(this->_state_space.C, numerator, denominator);

    /* Set D */
    ForDiscreteTransferFunction::Set_D_Value<
        _T, _DiscreteStateSpace_D_type, Numerator_Type, Denominator_Type,
        _IS_STRICTLY_PROPER>::set(this->_state_space.D, numerator, denominator);

    /* Set delta time */
    this->_state_space.delta_time = delta_time;
  }

  /* Copy Constructor */
  DiscreteTransferFunction(
      const DiscreteTransferFunction<Numerator_Type, Denominator_Type> &input)
      : _state_space(input._state_space) {}

  DiscreteTransferFunction<Numerator_Type, Denominator_Type> &operator=(
      const DiscreteTransferFunction<Numerator_Type, Denominator_Type> &input) {
    if (this != &input) {
      this->_state_space = input._state_space;
    }
    return *this;
  }

  /* Move Constructor */
  DiscreteTransferFunction(
      DiscreteTransferFunction<Numerator_Type, Denominator_Type>
          &&input) noexcept
      : _state_space(std::move(input._state_space)) {}

  DiscreteTransferFunction<Numerator_Type, Denominator_Type> &
  operator=(DiscreteTransferFunction<Numerator_Type, Denominator_Type>
                &&input) noexcept {
    if (this != &input) {
      this->_state_space = std::move(input._state_space);
    }
    return *this;
  }

public:
  /* Function */
  constexpr std::size_t get_number_of_delay(void) const {
    return NUMBER_OF_DELAY;
  }

  auto get_X(void) const -> typename _State_Space_Type::Original_X_Type {
    return this->_state_space.X;
  }

  _T get_y(void) const {
    return this->_state_space.get_Y().template get<0, 0>();
  }

  void update(const _T &u) {
    auto input = make_StateSpaceInput<_U_Type::COLS>(u);

    this->_state_space.update(input);
  }

  void reset_numerator_and_denominator(const Numerator_Type &numerator,
                                       const Denominator_Type &denominator) {
    /* Set A */
    ForDiscreteTransferFunction::Set_A_Value<
        _T, _DiscreteStateSpace_A_type, Denominator_Type, 0,
        (_DENOMINATOR_DIMENSION - 1)>::set(this->_state_space.A, denominator);

    ForDiscreteTransferFunction::Set_A_Ones<
        _T, _DiscreteStateSpace_A_type, _DENOMINATOR_DIMENSION,
        (2 * (_DENOMINATOR_DIMENSION - 1))>::set(this->_state_space.A);

    /* Set C */
    ForDiscreteTransferFunction::Set_C_Value<
        _T, _DiscreteStateSpace_C_type, Numerator_Type, Denominator_Type,
        _IS_STRICTLY_PROPER>::set(this->_state_space.C, numerator, denominator);

    /* Set D */
    ForDiscreteTransferFunction::Set_D_Value<
        _T, _DiscreteStateSpace_D_type, Numerator_Type, Denominator_Type,
        _IS_STRICTLY_PROPER>::set(this->_state_space.D, numerator, denominator);
  }

  void reset_state(void) { this->_state_space.reset_state(); }

  _T solve_steady_state_and_input(const _T &y_steady_state) {

    _T u_steady_state = ForDiscreteTransferFunction::SolveSteadyStateAndInput<
        _T, _State_Space_Type, _IS_STRICTLY_PROPER>::solve(this->_state_space,
                                                           y_steady_state);

    return u_steady_state;
  }

public:
  /* Constant */
  static constexpr std::size_t NUMBER_OF_DELAY = Number_Of_Delay;

private:
  /* Variable */
  _State_Space_Type _state_space;
};

/* make Discrete Transfer Function */
template <typename Numerator_Type, typename Denominator_Type>
inline auto
make_DiscreteTransferFunction(Numerator_Type numerator,
                              Denominator_Type denominator,
                              typename Numerator_Type::Value_Type delta_time)
    -> DiscreteTransferFunction<Numerator_Type, Denominator_Type> {

  return DiscreteTransferFunction<Numerator_Type, Denominator_Type>(
      numerator, denominator, delta_time);
}

template <std::size_t Number_Of_Delay, typename Numerator_Type,
          typename Denominator_Type>
inline auto
make_DiscreteTransferFunction(Numerator_Type numerator,
                              Denominator_Type denominator,
                              typename Numerator_Type::Value_Type delta_time)
    -> DiscreteTransferFunction<Numerator_Type, Denominator_Type,
                                Number_Of_Delay> {

  return DiscreteTransferFunction<Numerator_Type, Denominator_Type,
                                  Number_Of_Delay>(numerator, denominator,
                                                   delta_time);
}

/* Discrete Transfer Function Type */
template <typename Numerator_Type, typename Denominator_Type,
          std::size_t Number_Of_Delay = 0>
using DiscreteTransferFunction_Type =
    DiscreteTransferFunction<Numerator_Type, Denominator_Type, Number_Of_Delay>;

} // namespace PythonControl

#endif // __PYTHON_CONTROL_TRANSFER_FUNCTION_HPP__
