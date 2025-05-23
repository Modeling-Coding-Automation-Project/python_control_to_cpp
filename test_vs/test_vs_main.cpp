﻿#include <type_traits>
#include <iostream>
#include <cmath>

#include "python_control.hpp"

#include "MCAP_tester.hpp"
#include "test_vs_data.hpp"
#include "test_vs_EKF_data.hpp"
#include "test_vs_LS_data.hpp"


using namespace Tester;


template <typename T>
void check_python_control_state_space(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;


    /* State Space 定義 (example 1) */
    using A_Type = DenseMatrix_Type<T, 2, 2>;
    using B_Type = DenseMatrix_Type<T, 2, 1>;
    using C_Type = DenseMatrix_Type<T, 1, 2>;
    using D_Type = DenseMatrix_Type<T, 1, 1>;

    A_Type A({
        {static_cast<T>(0.7), static_cast<T>(0.2)},
        {static_cast<T>(-0.3), static_cast<T>(0.8)}});
    B_Type B({
        {static_cast<T>(0.1)},
        {static_cast<T>(0.2)}});
    C_Type C(
        { {static_cast<T>(2), static_cast<T>(0)} });
    D_Type D(
        { {static_cast<T>(0)} });
    T dt = static_cast<T>(0.01);


    DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> sys_0;

    A_Type sys_0_A_answer({
        {static_cast<T>(0), static_cast<T>(0)},
        {static_cast<T>(0), static_cast<T>(0)}});

    tester.expect_near(sys_0.A.matrix.data, sys_0_A_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace 0 arguments.");

    DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> sys_1(A);

    A_Type sys_1_A_answer({
        {static_cast<T>(0.7), static_cast<T>(0.2)},
        {static_cast<T>(-0.3), static_cast<T>(0.8)}});

    tester.expect_near(sys_1.A.matrix.data, sys_1_A_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace 1 arguments.");

    DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> sys_2(A, B);

    B_Type sys_2_B_answer({
        {static_cast<T>(0.1)},
        {static_cast<T>(0.2)} });

    tester.expect_near(sys_2.B.matrix.data, sys_2_B_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace 2 arguments.");

    DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> sys_3(A, B, C);

    C_Type sys_3_C_answer(
        { {static_cast<T>(2), static_cast<T>(0)} });

    tester.expect_near(sys_3.C.matrix.data, sys_3_C_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace 3 arguments.");

    DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> sys_4(A, B, C, D);

    D_Type sys_4_D_answer(
        { {static_cast<T>(0)} });

    tester.expect_near(sys_4.D.matrix.data, sys_4_D_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace 4 arguments.");


    /* State Space 機能 */
    DiscreteStateSpace_Type<decltype(A), decltype(B), decltype(C), decltype(D)>
        sys = make_DiscreteStateSpace(A, B, C, D, dt);

    DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> sys_copy = sys;
    DiscreteStateSpace<A_Type, B_Type, C_Type, D_Type> sys_move = std::move(sys_copy);
    sys = sys_move;

    T dt_answer = static_cast<T>(0.01);

    tester.expect_near(sys.delta_time, dt_answer, NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace delta_time.");

    auto u_0 = make_StateSpaceInput<1>(static_cast<T>(1.0));
    auto x_0 = sys.get_X();
    x_0(0, 0) = static_cast<T>(0.1);
    x_0(1, 0) = static_cast<T>(0.2);

    auto x_1 = sys.state_function(x_0, u_0);

    decltype(x_1) x_1_answer({
        {static_cast<T>(0.21)},
        {static_cast<T>(0.33)}
        });

    tester.expect_near(x_1.matrix.data, x_1_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace state function.");

    auto y_1 = sys.output_function(x_0, u_0);

    decltype(y_1) y_1_answer({
        {static_cast<T>(0.2)}
        });

    tester.expect_near(y_1.matrix.data, y_1_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace output function.");


    /* State Space シミュレーション */
    DenseMatrix_Type<T, 2, TestData::SIM_SS_STEP_MAX> X_results;
    DenseMatrix_Type<T, 1, TestData::SIM_SS_STEP_MAX> Y_results;

    DenseMatrix_Type<T, TestData::SIM_SS_STEP_MAX, 2> X_results_exmaple_1_answer_Trans;
    for (std::size_t i = 0; i < X_results_exmaple_1_answer_Trans.cols(); i++) {
        for (std::size_t j = 0; j < X_results_exmaple_1_answer_Trans.rows(); j++) {

            X_results_exmaple_1_answer_Trans(i, j) =
                static_cast<T>(TestData::X_results_exmaple_1_answer(i, j));
        }
    }

    DenseMatrix_Type<T, TestData::SIM_SS_STEP_MAX, 1> Y_results_exmaple_1_answer_Trans;
    for (std::size_t i = 0; i <  Y_results_exmaple_1_answer_Trans.cols(); i++) {
        for (std::size_t j = 0; j <  Y_results_exmaple_1_answer_Trans.rows(); j++) {

             Y_results_exmaple_1_answer_Trans(i, j) =
                 static_cast<T>(TestData:: Y_results_exmaple_1_answer(i, j));
        }
    }


    for (std::size_t sim_step = 0; sim_step < TestData::SIM_SS_STEP_MAX; ++sim_step) {
        auto u = make_StateSpaceInput<1>(static_cast<T>(1.0));

        sys.update(u);

        X_results(0, sim_step) = sys.X(0, 0);
        X_results(1, sim_step) = sys.get_X()(1, 0);
        Y_results(0, sim_step) = sys.get_Y()(0, 0);
    }

    //for (std::size_t sim_step = 0; sim_step < TestData::SIM_SS_STEP_MAX; ++sim_step) {
    //    std::cout << "X" << "{" << sim_step << "}: "
    //        << X_results(0, sim_step)
    //        << ", " << X_results(1, sim_step) << std::endl;
    //}
    //std::cout << std::endl;

    //for (std::size_t sim_step = 0; sim_step < TestData::SIM_SS_STEP_MAX; ++sim_step) {
    //    std::cout << "Y" << "{" << sim_step << "}: "
    //        << Y_results(0, sim_step) << std::endl;
    //}

    tester.expect_near(X_results.matrix.data,
        X_results_exmaple_1_answer_Trans.transpose().matrix.data, NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace simulation X results.");

    tester.expect_near(Y_results.matrix.data,
        Y_results_exmaple_1_answer_Trans.transpose().matrix.data, NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace simulation Y results.");


    /* むだ時間 */
    constexpr std::size_t DELAY_STEP = 3;

    DiscreteStateSpace_Type<decltype(A), decltype(B), decltype(C), decltype(D), DELAY_STEP>
        sys_delay = make_DiscreteStateSpace<DELAY_STEP>(A, B, C, D, dt);

    DiscreteStateSpace<decltype(A), decltype(B), decltype(C), decltype(D), DELAY_STEP>
        sys_delay_copy = sys_delay;
    DiscreteStateSpace<decltype(A), decltype(B), decltype(C), decltype(D), DELAY_STEP>
        sys_delay_move = std::move(sys_delay_copy);
    sys_delay = sys_delay_move;

    tester.expect_near(static_cast<T>(sys_delay.get_number_of_delay()),
        static_cast<T>(DELAY_STEP), NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace delay_step.");

    auto u_delay = make_StateSpaceInput<1>(static_cast<T>(1.0));

    for (std::size_t i = 0; i < DELAY_STEP; i++) {
        sys_delay.update(u_delay);
    }
    std::size_t delay_ring_buffer_index = sys_delay.get_delay_ring_buffer_index();

    sys_delay.update(u_delay);

    tester.expect_near(static_cast<T>(delay_ring_buffer_index), static_cast<T>(DELAY_STEP), NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace delay ring buffer index.");

    auto U_latest = sys_delay.U.get_latest();

    tester.expect_near(U_latest(0, 0), static_cast<T>(1.0), NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace delay input latest.");

    T Y_delayed = sys_delay.get_Y()(0, 0);
    tester.expect_near(Y_delayed,
        Y_results_exmaple_1_answer_Trans(0, 0), NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace delay output.");

    sys_delay.update(u_delay);

    Y_delayed = sys_delay.get_Y()(0, 0);
    tester.expect_near(Y_delayed,
        Y_results_exmaple_1_answer_Trans(1, 0), NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace delay output 2.");

    auto U_1 = sys_delay.template access_U<0>();

    tester.expect_near(U_1, static_cast<T>(1.0), NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace delay input.");


    /* DCモーターモデル (example 2) */
    T DC_dt = static_cast<T>(0.01);

    //DenseMatrix_Type<T, 4, 4> A_example_2({
    //    {static_cast<T>(1.0), static_cast<T>(0.01), static_cast<T>(0.0), static_cast<T>(0.0)},
    //    {static_cast<T>(-0.51207708), static_cast<T>(0.99), static_cast<T>(0.02560385), static_cast<T>(0.0)},
    //    {static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(1.0), static_cast<T>(0.01)},
    //    {static_cast<T>(1.28019901), static_cast<T>(0.0), static_cast<T>(-0.06400995), static_cast<T>(0.898)}
    //    });
    auto A_example_2 = make_SparseMatrix<SparseAvailable<
        ColumnAvailable<true, true, false, false>,
        ColumnAvailable<true, true, true, false>,
        ColumnAvailable<false, false, true, true>,
        ColumnAvailable<true, false, true, true>>>(
            static_cast<T>(1.0),
            static_cast<T>(0.01),
            static_cast<T>(-0.51207708),
            static_cast<T>(0.99),
            static_cast<T>(0.02560385),
            static_cast<T>(1.0),
            static_cast<T>(0.01),
            static_cast<T>(1.28019901),
            static_cast<T>(-0.06400995),
            static_cast<T>(0.898));


    //DenseMatrix_Type<T, 4, 1> B_example_2({
    //    {static_cast<T>(0.0)},
    //    {static_cast<T>(0.0)},
    //    {static_cast<T>(0.0)},
    //    {static_cast<T>(0.01)}
    //    });
    auto B_example_2 = make_SparseMatrix<SparseAvailable<
        ColumnAvailable<false>,
        ColumnAvailable<false>,
        ColumnAvailable<false>,
        ColumnAvailable<true>>>(static_cast<T>(0.01));

    //DenseMatrix_Type<T, 2, 4> C_example_2({
    //    {static_cast<T>(1.0), static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(0.0)},
    //    {static_cast<T>(1280.19901), static_cast<T>(0.0), static_cast<T>(-64.0099503), static_cast<T>(0.0)}
    //    });
    auto C_example_2 = make_SparseMatrix<SparseAvailable<
        ColumnAvailable<true, false, false, false>,
        ColumnAvailable<true, false, true, false>>>(
            static_cast<T>(1.0),
            static_cast<T>(1280.19901),
            static_cast<T>(-64.0099503));

    //DenseMatrix_Type<T, 2, 1> D_example_2({
    //    {static_cast<T>(0.0)},
    //    {static_cast<T>(0.0)}
    //    });
    auto D_example_2 = make_SparseMatrixEmpty<T, 2, 1>();

    auto sys_dc = make_DiscreteStateSpace(A_example_2, B_example_2, C_example_2, D_example_2, DC_dt);


    /* DCモーターモデル シミュレーション */
    DenseMatrix_Type<T, 2, TestData::DC_MOTOR_SIM_SS_STEP_MAX> DC_motor_Y_results;

    DenseMatrix_Type<T, TestData::DC_MOTOR_SIM_SS_STEP_MAX, 2> Y_results_exmaple_2_answer_Trans;
    for (std::size_t i = 0; i < Y_results_exmaple_2_answer_Trans.cols(); i++) {
        for (std::size_t j = 0; j < Y_results_exmaple_2_answer_Trans.rows(); j++) {

            Y_results_exmaple_2_answer_Trans(i, j) =
                static_cast<T>(TestData::Y_results_exmaple_2_answer(i, j));
        }
    }

    for (std::size_t sim_step = 0; sim_step < TestData::DC_MOTOR_SIM_SS_STEP_MAX; ++sim_step) {
        auto u = make_StateSpaceInput<1>(static_cast<T>(1.0));

        sys_dc.update(u);

        DC_motor_Y_results(0, sim_step) = sys_dc.Y(0, 0);
        DC_motor_Y_results(1, sim_step) = sys_dc.Y(1, 0);
    }


    tester.expect_near(DC_motor_Y_results.matrix.data,
        Y_results_exmaple_2_answer_Trans.transpose().matrix.data, NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace DC motor simulation Y results.");

    /* 状態リセット */
    sys_dc.reset_state();

    DenseMatrix_Type<T, 4, 1> X_reset_results;

    tester.expect_near(sys_dc.get_X().matrix.data,
        X_reset_results.matrix.data, NEAR_LIMIT_STRICT,
        "check DiscreteStateSpace DC motor reset state.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void check_python_control_transfer_function(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* 分子3次 分母4次 伝達関数 定義 */
    auto numerator_3_4 = make_TransferFunctionNumerator<4>(
        static_cast<T>(0.0012642614672828678),
        static_cast<T>(0.0037594540384011665),
        static_cast<T>(-0.002781625665309928),
        static_cast<T>(-0.0009364784774175128)
    );

    auto denominator_3_4 = make_TransferFunctionDenominator<5>(
        static_cast<T>(1.0),
        static_cast<T>(-3.565195017021459),
        static_cast<T>(4.815115383504625),
        static_cast<T>(-2.9189348011558485),
        static_cast<T>(0.6703200460356397)
    );

    T dt = static_cast<T>(0.2);

    DiscreteTransferFunction_Type<decltype(numerator_3_4), decltype(denominator_3_4), 0>
        system_3_4 = make_DiscreteTransferFunction(numerator_3_4, denominator_3_4, dt);

    DiscreteTransferFunction_Type<decltype(numerator_3_4), decltype(denominator_3_4), 0>
        system_3_4_copy = system_3_4;
    DiscreteTransferFunction_Type<decltype(numerator_3_4), decltype(denominator_3_4), 0>
        system_3_4_move = std::move(system_3_4_copy);
    system_3_4 = system_3_4_move;

    DenseMatrix_Type<T, 1, TestData::SYSTEM_3_4_STEP_MAX> system_3_4_y;
    DenseMatrix_Type<T, TestData::SYSTEM_3_4_STEP_MAX, 1> system_3_4_y_answer_Trans;
    for (std::size_t i = 0; i < system_3_4_y_answer_Trans.cols(); i++) {
        for (std::size_t j = 0; j < system_3_4_y_answer_Trans.rows(); j++) {

            system_3_4_y_answer_Trans(i, j) =
                static_cast<T>(TestData::system_3_4_y_answer(i, j));
        }
    }

    /* 分子3次 分母4次 伝達関数 シミュレーション */
    T u = static_cast<T>(1);

    for (std::size_t i = 0; i < TestData::SYSTEM_3_4_STEP_MAX; i++) {
        system_3_4.update(u);

        system_3_4_y(0, i) = system_3_4.get_y();
    }

    tester.expect_near(system_3_4_y.matrix.data,
        system_3_4_y_answer_Trans.transpose().matrix.data, NEAR_LIMIT_STRICT,
        "check DiscreteTransferFunction 3,4, simulation Y results.");

    /* 分子3次 分母4次 伝達関数 状態と入力を出力から逆算 */
    T y_steady_state = static_cast<T>(1.0);

    T u_steady_state = system_3_4.solve_steady_state_and_input(y_steady_state);

    tester.expect_near(u_steady_state, u, NEAR_LIMIT_STRICT,
        "check DiscreteTransferFunction steady state input.");

    auto x_steady_state = system_3_4.get_X();

    decltype(x_steady_state) x_steady_state_answer({
        {static_cast<T>(765.93137255)},
        {static_cast<T>(765.93137255)},
        {static_cast<T>(765.93137255)},
        {static_cast<T>(765.93137255)}
        });

    tester.expect_near(x_steady_state.matrix.data, x_steady_state_answer.matrix.data,
        NEAR_LIMIT_STRICT * std::abs(x_steady_state_answer.template get<0, 0>()),
        "check DiscreteTransferFunction steady state state.");


    /* 分子3次 分母4次 状態リセット */
    system_3_4.reset_state();

    system_3_4.update(u);
    system_3_4.update(u);

    T y_reset = system_3_4.get_y();
    T y_reset_answer = static_cast<T>(TestData::system_3_4_y_answer(1, 0));

    tester.expect_near(y_reset, y_reset_answer, NEAR_LIMIT_STRICT * std::abs(y_reset_answer),
        "check DiscreteTransferFunction reset state.");

    /* 分子3次 分母4次 分子分母リセット */
    auto numerator_3_4_0 = make_TransferFunctionNumerator<4>(
        static_cast<T>(0),
        static_cast<T>(0),
        static_cast<T>(0),
        static_cast<T>(0)
    );

    auto denominator_3_4_0 = make_TransferFunctionDenominator<5>(
        static_cast<T>(1),
        static_cast<T>(0),
        static_cast<T>(0),
        static_cast<T>(0),
        static_cast<T>(0)
    );

    system_3_4.reset_numerator_and_denominator(numerator_3_4_0, denominator_3_4_0);

    system_3_4.update(u);
    system_3_4.update(u);

    T y_reset_numerator_denominator = system_3_4.get_y();
    T y_reset_numerator_denominator_answer = static_cast<T>(0);

    tester.expect_near(y_reset_numerator_denominator, y_reset_numerator_denominator_answer,
        NEAR_LIMIT_STRICT,
        "check DiscreteTransferFunction reset numerator and denominator.");


    /* 分子4次 分母4次 伝達関数 定義 */
    auto numerator_4_4 = make_TransferFunctionNumerator<5>(
        static_cast<T>(1.0),
        static_cast<T>(0.5),
        static_cast<T>(0.3),
        static_cast<T>(0.2),
        static_cast<T>(0.1)
    );

    auto denominator_4_4 = make_TransferFunctionDenominator<5>(
        static_cast<T>(1.1),
        static_cast<T>(-0.5),
        static_cast<T>(0.4),
        static_cast<T>(-0.3),
        static_cast<T>(0.2)
    );

    auto system_4_4 = make_DiscreteTransferFunction(numerator_4_4, denominator_4_4, dt);


    DenseMatrix_Type<T, 1, TestData::SYSTEM_4_4_STEP_MAX> system_4_4_y;
    DenseMatrix_Type<T, TestData::SYSTEM_4_4_STEP_MAX, 1> system_4_4_y_answer_Trans;
    for (std::size_t i = 0; i < system_4_4_y_answer_Trans.cols(); i++) {
        for (std::size_t j = 0; j < system_4_4_y_answer_Trans.rows(); j++) {

            system_4_4_y_answer_Trans(i, j) =
                static_cast<T>(TestData::system_4_4_y_answer(i, j));
        }
    }

    /* 分子4次 分母4次 伝達関数 シミュレーション */
    for (std::size_t i = 0; i < TestData::SYSTEM_4_4_STEP_MAX; i++) {
        system_4_4.update(u);

        system_4_4_y(0, i) = system_4_4.get_y();
    }

    tester.expect_near(system_4_4_y.matrix.data,
        system_4_4_y_answer_Trans.transpose().matrix.data, NEAR_LIMIT_STRICT,
        "check DiscreteTransferFunction  4,4, simulation Y results.");

    /* 分子4次 分母4次 伝達関数 状態と入力を出力から逆算 */
    T y_steady_state_4_4 = static_cast<T>(2.333315164634206);

    T u_steady_state_4_4 = system_4_4.solve_steady_state_and_input(y_steady_state_4_4);

    tester.expect_near(u_steady_state_4_4, u, NEAR_LIMIT_STRICT,
        "check DiscreteTransferFunction 4,4 steady state input.");

    auto x_steady_state_4_4 = system_4_4.get_X();

    decltype(x_steady_state_4_4) x_steady_state_4_4_answer({
        {static_cast<T>(1.22221271)},
        {static_cast<T>(1.22221271)},
        {static_cast<T>(1.22221271)},
        {static_cast<T>(1.22221271)}
        });

    tester.expect_near(x_steady_state_4_4.matrix.data, x_steady_state_4_4_answer.matrix.data,
        NEAR_LIMIT_STRICT * std::abs(x_steady_state_4_4_answer.template get<0, 0>()),
        "check DiscreteTransferFunction 4,4 steady state state.");


    /* 分子2次 分母4次 伝達関数 定義 */
    auto numerator_2_4 = make_TransferFunctionNumerator<3>(
        static_cast<T>(0.5),
        static_cast<T>(0.3),
        static_cast<T>(0.1)
    );

    auto denominator_2_4 = make_TransferFunctionDenominator<5>(
        static_cast<T>(1.0),
        static_cast<T>(-1.8),
        static_cast<T>(1.5),
        static_cast<T>(-0.7),
        static_cast<T>(0.2)
    );

    auto system_2_4 = make_DiscreteTransferFunction(numerator_2_4, denominator_2_4, dt);


    DenseMatrix_Type<T, 1, TestData::SYSTEM_2_4_STEP_MAX> system_2_4_y;
    DenseMatrix_Type<T, TestData::SYSTEM_2_4_STEP_MAX, 1> system_2_4_y_answer_Trans;
    for (std::size_t i = 0; i < system_2_4_y_answer_Trans.cols(); i++) {
        for (std::size_t j = 0; j < system_2_4_y_answer_Trans.rows(); j++) {

            system_2_4_y_answer_Trans(i, j) =
                static_cast<T>(TestData::system_2_4_y_answer(i, j));
        }
    }

    /* 分子2次 分母4次 伝達関数 シミュレーション */
    for (std::size_t i = 0; i < TestData::SYSTEM_2_4_STEP_MAX; i++) {
        system_2_4.update(u);

        system_2_4_y(0, i) = system_2_4.get_y();
    }

    tester.expect_near(system_2_4_y.matrix.data,
        system_2_4_y_answer_Trans.transpose().matrix.data, NEAR_LIMIT_STRICT,
        "check DiscreteTransferFunction  2,4, simulation Y results.");

    /* むだ時間 */
    constexpr std::size_t DELAY_STEP = 2;
    auto system_3_4_delay = make_DiscreteTransferFunction<DELAY_STEP>(numerator_3_4, denominator_3_4, dt);

    for (std::size_t i = 0; i < DELAY_STEP; i++) {
        system_3_4_delay.update(u);
    }
    system_3_4_delay.update(u);
    system_3_4_delay.update(u);

    tester.expect_near(system_3_4_delay.get_y(), static_cast<T>(TestData::system_3_4_y_answer(1, 0)),
        NEAR_LIMIT_STRICT,
        "check DiscreteTransferFunction 3,4, delay output.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void check_python_control_pid_controller(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-5);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;


    T dt = static_cast<T>(0.2);

    /* プラントモデル */
    auto numerator_plant = make_TransferFunctionNumerator<2>(
        static_cast<T>(0.015479737715070607),
        static_cast<T>(0.01497228851342225)
    );

    auto denominator_plant = make_TransferFunctionDenominator<3>(
        static_cast<T>(1.0),
        static_cast<T>(-1.9048374180359595),
        static_cast<T>(0.9048374180359595)
    );

    auto plant = make_DiscreteTransferFunction(numerator_plant, denominator_plant, dt);

    /* PID制御器 */
    DiscretePID_Controller<T> pid_controller;
    pid_controller.delta_time = dt;
    pid_controller.Kp = static_cast<T>(1.0);
    pid_controller.Ki = static_cast<T>(0.1);
    pid_controller.Kd = static_cast<T>(0.5);
    pid_controller.N = static_cast<T>(1.0) / pid_controller.delta_time;
    pid_controller.Kb = pid_controller.Ki;
    pid_controller.minimum_output = static_cast<T>(-1e6);
    pid_controller.maximum_output = static_cast<T>(1e6);

    DiscretePID_Controller_Type<T> pid_controller_copy = pid_controller;
    DiscretePID_Controller_Type<T> pid_controller_move = std::move(pid_controller_copy);
    pid_controller = pid_controller_move;

    /* シミュレーション */
    DenseMatrix_Type<T, 1, TestData::SYSTEM_PID_STEP_MAX> system_PID_y;
    DenseMatrix_Type<T, TestData::SYSTEM_PID_STEP_MAX, 1> system_PID_y_answer_Trans;
    for (std::size_t i = 0; i < system_PID_y_answer_Trans.cols(); i++) {
        for (std::size_t j = 0; j < system_PID_y_answer_Trans.rows(); j++) {

            system_PID_y_answer_Trans(i, j) =
                static_cast<T>(TestData::system_PID_y_answer(i, j));
        }
    }

    T r = static_cast<T>(1.0);

    for (std::size_t i = 0; i < TestData::SYSTEM_PID_STEP_MAX; i++) {
        T e = r - system_PID_y(0, i - 1);
        T u = pid_controller.update(e);

        plant.update(u);

        system_PID_y(0, i) = plant.get_y();
    }
    
    tester.expect_near(system_PID_y.matrix.data,
        system_PID_y_answer_Trans.transpose().matrix.data, NEAR_LIMIT_STRICT,
        "check DiscretePID_Controller simulation y results.");

    /* リセット */
    pid_controller.reset();

    tester.expect_near(pid_controller.get_integration_store(),
        static_cast<T>(0.0), NEAR_LIMIT_STRICT,
        "check DiscretePID_Controller integral store reset.");

    tester.expect_near(pid_controller.get_differentiation_store(),
        static_cast<T>(0.0), NEAR_LIMIT_STRICT,
        "check DiscretePID_Controller differenciation store reset.");


    T Kp = static_cast<T>(2.0);
    T Ki = static_cast<T>(0.2);
    T Kd = static_cast<T>(3.0);
    T N = static_cast<T>(1.0) / dt;
    T output_min = static_cast<T>(-0.5);
    T output_max = static_cast<T>(0.5);

    T error = static_cast<T>(1.0);

    /* 比例項 */
    auto pid_controller_P = make_DiscretePID_Controller(
        dt,
        Kp, static_cast<T>(0.0), static_cast<T>(0.0),
        static_cast<T>(0.0), static_cast<T>(0.0),
        static_cast<T>(-1e6), static_cast<T>(1e6));

    T u_P = pid_controller_P.update(error);

    tester.expect_near(u_P, Kp * error, NEAR_LIMIT_STRICT * std::abs(Kp * error),
        "check DiscretePID_Controller proportional term.");

    /* 積分項 */
    auto pid_controller_I = make_DiscretePID_Controller(
        dt,
        static_cast<T>(0.0), Ki, static_cast<T>(0.0),
        static_cast<T>(0.0), static_cast<T>(0.0),
        static_cast<T>(-1e6), static_cast<T>(1e6)
    );

    T u_I = pid_controller_I.update(error);

    tester.expect_near(u_I, Ki * dt * error, NEAR_LIMIT_STRICT * std::abs(Ki * dt * error),
        "check DiscretePID_Controller integral term.");

    /* 微分項 */
    auto pid_controller_D = make_DiscretePID_Controller(
        dt,
        static_cast<T>(0.0), static_cast<T>(0.0), Kd,
        N, static_cast<T>(0.0),
        static_cast<T>(-1e6), static_cast<T>(1e6)
    );

    T u_D = pid_controller_D.update(error);

    tester.expect_near(u_D, Kd / dt * error, NEAR_LIMIT_STRICT * std::abs(Kd / dt * error),
        "check DiscretePID_Controller differential term.");

    /* 積分項の飽和 */
    constexpr std::size_t SATURATION_STEP = 1000;

    auto pid_controller_I_saturation = make_DiscretePID_Controller(
        dt,
        static_cast<T>(0.0), Ki, static_cast<T>(0.0),
        static_cast<T>(0.0), Ki,
        output_min, output_max
    );

    T u_I_saturation = static_cast<T>(0.0);
    for (std::size_t i = 0; i < SATURATION_STEP; i++) {
        u_I_saturation = pid_controller_I_saturation.update(error);
    }

    tester.expect_near(u_I_saturation, output_max, NEAR_LIMIT_STRICT * std::abs(output_max),
        "check DiscretePID_Controller integral term saturation max.");

    tester.expect_near(pid_controller_I_saturation.get_integration_store(),
        output_max + error, NEAR_LIMIT_STRICT* std::abs(output_max + error),
        "check DiscretePID_Controller integral store saturation max.");

    for (std::size_t i = 0; i < SATURATION_STEP; i++) {
        u_I_saturation = pid_controller_I_saturation.update(-error);
    }

    tester.expect_near(u_I_saturation, output_min, NEAR_LIMIT_STRICT * std::abs(output_min),
        "check DiscretePID_Controller integral term saturation min.");

    tester.expect_near(pid_controller_I_saturation.get_integration_store(),
        output_min - error, NEAR_LIMIT_STRICT * std::abs(output_min - error),
        "check DiscretePID_Controller integral store saturation min.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void check_python_control_lqr(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-4) : T(1.0e-3);
    constexpr T NEAR_LIMIT_SOFT = 3.0e-1F;


    using SparseAvailable_Ac = SparseAvailable<
        ColumnAvailable<false, true, false, false>,
        ColumnAvailable<false, true, true, false>,
        ColumnAvailable<false, false, false, true>,
        ColumnAvailable<false, true, true, false>>;

    auto Ac = make_SparseMatrix<SparseAvailable_Ac>(
            static_cast<T>(1.0), static_cast<T>(-0.1),
            static_cast<T>(3.0), static_cast<T>(1.0),
            static_cast<T>(-0.5), static_cast<T>(30.0));

    //auto Ad = make_SparseMatrix<SparseAvailable<
    //    ColumnAvailable<true, true, false, false>,
    //    ColumnAvailable<false, true, true, false>,
    //    ColumnAvailable<false, false, true, true>,
    //    ColumnAvailable<false, true, true, true>>>(
    //        static_cast<T>(1.0), static_cast<T>(0.1),
    //        static_cast<T>(0.99), static_cast<T>(0.3),
    //        static_cast<T>(1.0), static_cast<T>(0.1),
    //        static_cast<T>(-0.05), static_cast<T>(3.0), static_cast<T>(1.0));

    using SparseAvailable_Bc = SparseAvailable<
        ColumnAvailable<false>,
        ColumnAvailable<true>,
        ColumnAvailable<false>,
        ColumnAvailable<true>>;

    auto Bc = make_SparseMatrix<SparseAvailable_Bc>(
            static_cast<T>(2.0), static_cast<T>(5.0));

    //auto Bd = make_SparseMatrix<SparseAvailable<
    //    ColumnAvailable<false>,
    //    ColumnAvailable<true>,
    //    ColumnAvailable<false>,
    //    ColumnAvailable<true>>>(static_cast<T>(0.2), static_cast<T>(0.5));

    //using SparseAvailable_Cc = SparseAvailable<
    //    ColumnAvailable<true, false, false, false>,
    //    ColumnAvailable<false, false, true, false>>;
    //auto Cc = make_SparseMatrix<SparseAvailable_Cc>(static_cast<T>(1.0), static_cast<T>(1.0));

    auto Q = make_DiagMatrix<4>(
        static_cast<T>(1), static_cast<T>(0),
        static_cast<T>(1), static_cast<T>(0));

    auto R = make_DiagMatrix<1>(static_cast<T>(1));


    /* LQR定義 */
    LQR_Type<decltype(Ac), decltype(Bc), decltype(Q), decltype(R)>
        lqr = make_LQR(Ac, Bc, Q, R);

    /* set */
    lqr.set_A(make_SparseMatrix<SparseAvailable_Ac>(
        static_cast<T>(0.0), static_cast<T>(0.0),
        static_cast<T>(0.0), static_cast<T>(0.0),
        static_cast<T>(0.0), static_cast<T>(0.0)));

    lqr.set_B(make_SparseMatrix<SparseAvailable_Bc>(
        static_cast<T>(0.0), static_cast<T>(0.0)));

    lqr.set_Q(make_DiagMatrix<4>(
        static_cast<T>(0), static_cast<T>(0),
        static_cast<T>(0), static_cast<T>(0)));

    lqr.set_R(make_DiagMatrix<1>(static_cast<T>(0)));

    lqr.set_A(Ac);
    lqr.set_B(Bc);
    lqr.set_Q(Q);
    lqr.set_R(R);

    LQR_Type<decltype(Ac), decltype(Bc), decltype(Q), decltype(R)> lqr_copy = lqr;
    LQR_Type<decltype(Ac), decltype(Bc), decltype(Q), decltype(R)> lqr_move = std::move(lqr_copy);
    lqr = lqr_move;

    /* LQR計算 */
    lqr.set_R_inv_division_min(static_cast<T>(1.0e-10));
    lqr.set_V1_inv_decay_rate(static_cast<T>(0));
    lqr.set_V1_inv_division_min(static_cast<T>(1.0e-10));
    lqr.set_Eigen_solver_iteration_max(10);
    lqr.set_Eigen_solver_iteration_max_for_eigen_vector(30);
    lqr.set_Eigen_solver_division_min(static_cast<T>(1.0e-20));
    lqr.set_Eigen_solver_small_value(static_cast<T>(1.0e-6));


    auto K = lqr.solve();
    K = lqr.get_K();

    auto K_answer = make_DenseMatrix<1, 4>(
        static_cast<T>(-1.0),
        static_cast<T>(-1.75585926),
        static_cast<T>(16.91449007),
        static_cast<T>(3.22735877) );

    tester.expect_near(K.matrix.data, K_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LQR solve continuous.");

    bool condition = lqr.get_eigen_solver_is_ill();

    tester.expect_near(condition, false, 0,
        "check LQR eigen solver is not ill.");


    /* LQI定義 */
    using SparseAvailable_Ac_easy_model = SparseAvailable<
        ColumnAvailable<true, true>,
        ColumnAvailable<true, false>>;

    using SparseAvailable_Bc_easy_model = SparseAvailable<
        ColumnAvailable<true>,
        ColumnAvailable<false>>;

    using SparseAvailable_Cc_easy_model = SparseAvailable<
        ColumnAvailable<false, true>>;

    auto Ac_easy_model = make_SparseMatrix<SparseAvailable_Ac_easy_model>(
        static_cast<T>(-2), static_cast<T>(-1), static_cast<T>(1));

    auto Bc_easy_model = make_SparseMatrix<SparseAvailable_Bc_easy_model>(
        static_cast<T>(1));

    auto Cc_easy_model = make_SparseMatrix<SparseAvailable_Cc_easy_model>(
        static_cast<T>(1));

    constexpr std::size_t Q_EX_SIZE = 3;

    auto Q_easy_model_ex = make_DiagMatrix<Q_EX_SIZE>(
        static_cast<T>(0), static_cast<T>(2), static_cast<T>(2));

    constexpr std::size_t R_EX_SIZE = 1;

    auto R_easy_model_ex = make_DiagMatrix<R_EX_SIZE>(static_cast<T>(1));

    LQI_Type<decltype(Ac_easy_model), decltype(Bc_easy_model), decltype(Cc_easy_model),
        decltype(Q_easy_model_ex), decltype(R_easy_model_ex)>
        lqi = make_LQI(Ac_easy_model, Bc_easy_model, Cc_easy_model,
            Q_easy_model_ex, R_easy_model_ex);

    /* set */
    lqi.set_A(make_SparseMatrixZeros<T, SparseAvailable_Ac_easy_model>());

    lqi.set_B(make_SparseMatrixZeros<T, SparseAvailable_Bc_easy_model>());

    lqi.set_C(make_SparseMatrixZeros<T, SparseAvailable_Cc_easy_model>());

    lqi.set_Q(make_DiagMatrixZeros<T, Q_EX_SIZE>());

    lqi.set_R(make_DiagMatrixZeros<T, R_EX_SIZE>());

    lqi.set_A(Ac_easy_model);
    lqi.set_B(Bc_easy_model);
    lqi.set_C(Cc_easy_model);
    lqi.set_Q(Q_easy_model_ex);
    lqi.set_R(R_easy_model_ex);

    LQI_Type<decltype(Ac_easy_model), decltype(Bc_easy_model), decltype(Cc_easy_model),
        decltype(Q_easy_model_ex), decltype(R_easy_model_ex)>
        lqi_copy = lqi;
    LQI_Type<decltype(Ac_easy_model), decltype(Bc_easy_model), decltype(Cc_easy_model),
        decltype(Q_easy_model_ex), decltype(R_easy_model_ex)>
        lqi_move = std::move(lqi_copy);
    lqi = lqi_move;

    /* LQI計算 */
    lqi.set_Eigen_solver_iteration_max(3);
    lqi.set_Eigen_solver_iteration_max_for_eigen_vector(8);

    auto K_ex = lqi.solve();
    K_ex = lqi.get_K();

    decltype(K_ex) K_ex_answer({ {
        static_cast<T>(0.95663669),
        static_cast<T>(2.37085025),
        static_cast<T>(1.41421356)
        } });

    tester.expect_near(K_ex.matrix.data, K_ex_answer.matrix.data, NEAR_LIMIT_SOFT,
        "check LQI solve continuous.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void check_python_control_linear_kalman_filter(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-4);
    constexpr T NEAR_LIMIT_SOFT = 1.0e-3F;


    using SparseAvailable_A = SparseAvailable<
        ColumnAvailable<true, true, false, false>,
        ColumnAvailable<false, true, true, false>,
        ColumnAvailable<false, false, true, true>,
        ColumnAvailable<false, false, false, true>>;

    auto A = make_SparseMatrix<SparseAvailable_A>(
        static_cast<T>(1.0), static_cast<T>(0.1),
        static_cast<T>(1.0), static_cast<T>(0.1),
        static_cast<T>(1.0), static_cast<T>(0.1),
        static_cast<T>(1.0));

    using SparseAvailable_B = SparseAvailable<
        ColumnAvailable<false, false>,
        ColumnAvailable<true, false>,
        ColumnAvailable<false, true>,
        ColumnAvailable<false, false>>;

    auto B = make_SparseMatrix<SparseAvailable_B>(
        static_cast<T>(0.1),
        static_cast<T>(0.1));

    using SparseAvailable_C = SparseAvailable<
        ColumnAvailable<true, false, false, false>,
        ColumnAvailable<false, false, true, false>>;

    auto C = make_SparseMatrix<SparseAvailable_C>(
        static_cast<T>(1.0),
        static_cast<T>(1.0));

    auto D = make_SparseMatrixEmpty<T, 2, 2>();

    T dt = static_cast<T>(0.1);

    constexpr std::size_t STATE_SIZE = decltype(A)::COLS;
    constexpr std::size_t INPUT_SIZE = decltype(B)::ROWS;
    constexpr std::size_t OUTPUT_SIZE = decltype(C)::COLS;

    auto sys = make_DiscreteStateSpace(A, B, C, D, dt);

    auto Q = make_KalmanFilter_Q<STATE_SIZE>(
        static_cast<T>(1), static_cast<T>(1),
        static_cast<T>(1), static_cast<T>(2));

    auto R = make_KalmanFilter_R<OUTPUT_SIZE>(
        static_cast<T>(10), static_cast<T>(10));

    /* カルマンフィルタ定義 */
    LinearKalmanFilter_Type<decltype(sys), decltype(Q), decltype(R)>
        lkf = make_LinearKalmanFilter(sys, Q, R);

    LinearKalmanFilter_Type<decltype(sys), decltype(Q), decltype(R)> lkf_copy = lkf;
    LinearKalmanFilter_Type<decltype(sys), decltype(Q), decltype(R)> lkf_move = std::move(lkf_copy);
    lkf = lkf_move;

    lkf.set_decay_rate_for_C_P_CT_R_inv_solver(static_cast<T>(0.0));
    lkf.set_division_min_for_C_P_CT_R_inv_solver(static_cast<T>(1.0e-10));

    /* シミュレーション準備 */
    lkf.set_x_hat(make_StateSpaceState<STATE_SIZE>(
        static_cast<T>(0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.0)));

    // data set
    std::array<StateSpaceState_Type<T, STATE_SIZE>, TestData::LKF_SIM_STEP_MAX> x_true;
    x_true[0](0, 0) = static_cast<T>(0.0);
    x_true[0](1, 0) = static_cast<T>(0.0);
    x_true[0](2, 0) = static_cast<T>(0.0);
    x_true[0](3, 0) = static_cast<T>(0.1);

    std::array<StateSpaceState_Type<T, STATE_SIZE>, TestData::LKF_SIM_STEP_MAX> x_estimate;
    x_estimate[0] = lkf.get_x_hat();

    std::array<StateSpaceOutput_Type<T, OUTPUT_SIZE>, TestData::LKF_SIM_STEP_MAX> y_measured;

    /* シミュレーション */
    for (std::size_t i = 1; i < TestData::LKF_SIM_STEP_MAX; i++) {
        auto u = make_StateSpaceInput<INPUT_SIZE>(
            static_cast<T>(TestData::lkf_test_input(i - 1, 0)),
            static_cast<T>(TestData::lkf_test_input(i - 1, 1))
        );

        // system response
        x_true[i] = A * x_true[i - 1] + B * u;
        y_measured[i] = C * x_true[i] + D * u;

        // kalman filter
        lkf.predict_and_update(u, y_measured[i]);
        x_estimate[i] = lkf.get_x_hat();
    }

    for (std::size_t i = TestData::LKF_SIM_STEP_MAX - 10; i < TestData::LKF_SIM_STEP_MAX; i++) {
        tester.expect_near(x_true[i].matrix.data, x_estimate[i].matrix.data, NEAR_LIMIT_SOFT,
            "check LinearKalmanFilter simulation x estimate.");
    }

    /* 遅れ込みのシミュレーション準備 */
    constexpr std::size_t NUMBER_OF_DELAY = 2;

    auto sys_delay = make_DiscreteStateSpace<NUMBER_OF_DELAY>(A, B, C, D, dt);
    auto lkf_delay = make_LinearKalmanFilter(sys_delay, Q, R);

    lkf_delay.set_x_hat(make_StateSpaceState<STATE_SIZE>(
        static_cast<T>(0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.0)));

    x_estimate[0] = lkf_delay.get_x_hat_without_delay();

    std::array<StateSpaceOutput_Type<T, OUTPUT_SIZE>, (NUMBER_OF_DELAY + 1)> y_store;

    std::size_t delay_index = 0;

    /* 遅れ込みのシミュレーション */
    for (std::size_t i = 1; i < TestData::LKF_SIM_STEP_MAX; i++) {
        auto u = make_StateSpaceInput<INPUT_SIZE>(
            static_cast<T>(TestData::lkf_test_input(i - 1, 0)),
            static_cast<T>(TestData::lkf_test_input(i - 1, 1))
        );

        // system response
        x_true[i] = A * x_true[i - 1] + B * u;
        auto y_next = C * x_true[i] + D * u;

        y_store[delay_index] = y_next;

        // system delay
        delay_index++;
        if (delay_index > NUMBER_OF_DELAY) {
            delay_index = 0;
        }

        y_measured[i] = y_store[delay_index];

        // kalman filter
        lkf_delay.predict_and_update(u, y_measured[i]);
        x_estimate[i] = lkf_delay.get_x_hat_without_delay();
    }

    for (std::size_t i = TestData::LKF_SIM_STEP_MAX - 10; i < TestData::LKF_SIM_STEP_MAX; i++) {
        tester.expect_near(x_true[i].matrix.data, x_estimate[i].matrix.data, NEAR_LIMIT_SOFT,
            "check LinearKalmanFilter with delay simulation x estimate.");
    }

    /* カルマンゲイン固定 */
    auto lkf_fixed = make_LinearKalmanFilter(sys, Q, R);

    lkf_fixed.set_x_hat(make_StateSpaceState<STATE_SIZE>(
        static_cast<T>(3.7633576),
        static_cast<T>(2.15584246),
        static_cast<T>(0.73995903),
        static_cast<T>(0.09986581)));

    lkf_fixed.G = make_DenseMatrix<STATE_SIZE, OUTPUT_SIZE>(
        static_cast<T>(0.33396284), static_cast<T>(0.00470285),
        static_cast<T>(0.27013879), static_cast<T>(0.06743136),
        static_cast<T>(0.00470287), static_cast<T>(0.3521885),
        static_cast<T>(-0.00624555), static_cast<T>(0.35992883)
    );

    auto u = make_StateSpaceInput<INPUT_SIZE>(
        static_cast<T>(0),
        static_cast<T>(0)
    );

    auto y = make_StateSpaceOutput<OUTPUT_SIZE>(
        static_cast<T>(3.7634),
        static_cast<T>(0.74)
    );

    lkf_fixed.predict_and_update_with_fixed_G(u, y);

    auto x_hat_fixed = lkf_fixed.get_x_hat();

    auto x_hat_fixed_answer = make_StateSpaceState<STATE_SIZE>(
        static_cast<T>(3.90691211),
        static_cast<T>(2.1709415),
        static_cast<T>(0.74542922),
        static_cast<T>(0.09763228)
    );

    tester.expect_near(x_hat_fixed.matrix.data, x_hat_fixed_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinearKalmanFilter with fixed G x estimate.");

    /* カルマンゲイン事前導出 */
    constexpr std::size_t STATE_SIZE_SS_1 = 2;
    constexpr std::size_t INPUT_SIZE_SS_1 = 1;
    constexpr std::size_t OUTPUT_SIZE_SS_1 = 1;

    auto A_ss_1 = make_DenseMatrix<STATE_SIZE_SS_1, STATE_SIZE_SS_1>(
        static_cast<T>(0.7), 
        static_cast<T>(0.2), 
        static_cast<T>(-0.3),
        static_cast<T>(0.8));

    auto B_ss_1 = make_DenseMatrix<STATE_SIZE_SS_1, INPUT_SIZE_SS_1>(
        static_cast<T>(0.1),
        static_cast<T>(0.2));

    using SparseAvailable_C_ss_1 = SparseAvailable<
        ColumnAvailable<true, false>>;

    auto C_ss_1 = make_SparseMatrix<SparseAvailable_C_ss_1>(
        static_cast<T>(1.0));

    auto D_ss_1 = make_SparseMatrixEmpty<T, OUTPUT_SIZE_SS_1, INPUT_SIZE_SS_1>();

    auto sys_ss_1 = make_DiscreteStateSpace(A_ss_1, B_ss_1, C_ss_1, D_ss_1, dt);

    auto Q_ss_1 = make_KalmanFilter_Q<STATE_SIZE_SS_1>(
        static_cast<T>(1.0), static_cast<T>(1.0));
    auto R_ss_1 = make_KalmanFilter_R<OUTPUT_SIZE_SS_1>(
        static_cast<T>(1.0));

    auto lkf_ss_1 = make_LinearKalmanFilter(sys_ss_1, Q_ss_1, R_ss_1);
    lkf_ss_1.converge_G();
    auto G_ss_1 = lkf_ss_1.G;

    auto G_ss_1_answer = make_DenseMatrix<STATE_SIZE_SS_1, OUTPUT_SIZE_SS_1>(
        static_cast<T>(0.5890584427270706),
        static_cast<T>(0.1491720712877476));

    tester.expect_near(G_ss_1.matrix.data, G_ss_1_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LinearKalmanFilter with pre-calculated G.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void check_python_control_extended_kalman_filter(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;

    MCAPTester<T> tester;

    //constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-4);
    constexpr T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* EKF定義準備 */
    constexpr std::size_t NUMBER_OF_DELAY = 5;

    constexpr std::size_t STATE_SIZE = EKF_TestData::STATE_SIZE;
    constexpr std::size_t INPUT_SIZE = EKF_TestData::INPUT_SIZE;
    constexpr std::size_t OUTPUT_SIZE = EKF_TestData::OUTPUT_SIZE;

    using X_Type = StateSpaceState_Type<T, STATE_SIZE>;
    using U_Type = StateSpaceInput_Type<T, INPUT_SIZE>;
    using Y_Type = StateSpaceOutput_Type<T, OUTPUT_SIZE>;

    using SparseAvailable_A = SparseAvailable<
        ColumnAvailable<true, false, true>,
        ColumnAvailable<false, true, true>,
        ColumnAvailable<false, false, true>
    >;

    using A_Type = SparseMatrix_Type<T, SparseAvailable_A>;

    using SparseAvailable_C = SparseAvailable<
        ColumnAvailable<true, true, false>,
        ColumnAvailable<true, true, true>,
        ColumnAvailable<true, true, false>,
        ColumnAvailable<true, true, true>
    >;

    using C_Type = SparseMatrix_Type<T, SparseAvailable_C>;


    auto Q = make_KalmanFilter_Q<STATE_SIZE>(
        static_cast<T>(1), static_cast<T>(1),
        static_cast<T>(1));

    using Q_Type = decltype(Q);

    auto R = make_KalmanFilter_R<OUTPUT_SIZE>(
        static_cast<T>(10), static_cast<T>(10),
        static_cast<T>(10), static_cast<T>(10));

    using R_Type = decltype(R);

    /* パラメータ */
    using Parameter_Type = EKF_TestData::BicycleModelParameter<T>;

    Parameter_Type parameters(
        static_cast<T>(0.1),
        static_cast<T>(0.5),
        static_cast<T>(-1.0),
        static_cast<T>(-1.0),
        static_cast<T>(10.0),
        static_cast<T>(10.0)
    );

    /* 状態方程式、出力方程式 */
    StateFunction_Object<X_Type,
        U_Type,
        EKF_TestData::BicycleModelParameter<T>> state_function;
    state_function = EKF_TestData::bicycle_model_state_function<T>;

    StateFunctionJacobian_Object<A_Type, X_Type, U_Type,
        EKF_TestData::BicycleModelParameter<T>> state_function_jacobian;
    state_function_jacobian = EKF_TestData::bicycle_model_state_function_jacobian<T, A_Type>;

    MeasurementFunction_Object<Y_Type, X_Type,
        EKF_TestData::BicycleModelParameter<T>> measurement_function;
    measurement_function = EKF_TestData::bicycle_model_measurement_function<T>;

    MeasurementFunctionJacobian_Object<C_Type, X_Type,
        EKF_TestData::BicycleModelParameter<T>> measurement_function_jacobian;
    measurement_function_jacobian = EKF_TestData::bicycle_model_measurement_function_jacobian<T, C_Type>;


    /* EKF定義 */
    ExtendedKalmanFilter<A_Type, C_Type, U_Type, Q_Type, R_Type, Parameter_Type, NUMBER_OF_DELAY>
        ekf(Q, R, state_function, state_function_jacobian,
            measurement_function, measurement_function_jacobian, parameters);

    ExtendedKalmanFilter_Type<A_Type, C_Type, U_Type, Q_Type, R_Type, Parameter_Type, NUMBER_OF_DELAY>
        ekf_copy = ekf;
    ExtendedKalmanFilter_Type<A_Type, C_Type, U_Type, Q_Type, R_Type, Parameter_Type, NUMBER_OF_DELAY>
        ekf_move = std::move(ekf_copy);
    ekf = ekf_move;

    /* シミュレーション */
    std::size_t simulation_steps = 200;

    auto x_true_initial = make_StateSpaceState<STATE_SIZE>(
        static_cast<T>(2), static_cast<T>(6), static_cast<T>(0.3)
    );
    decltype(x_true_initial) x_true;

    auto u = make_StateSpaceInput<INPUT_SIZE>(
        static_cast<T>(2.0), static_cast<T>(0.1)
    );

    ekf.X_hat.template set<0, 0>(static_cast<T>(0.0));
    ekf.X_hat.template set<1, 0>(static_cast<T>(0.0));
    ekf.X_hat.template set<2, 0>(static_cast<T>(0.0));

    constexpr std::size_t STORE_SIZE = 10;
    std::array<X_Type, STORE_SIZE> x_true_store;
    std::array<X_Type, STORE_SIZE> x_estimated_store;

    std::size_t store_index = 0;

    std::array<StateSpaceOutput_Type<T, OUTPUT_SIZE>, (NUMBER_OF_DELAY + 1)> y_store;

    std::size_t delay_index = 0;

    x_true = x_true_initial;
    for (std::size_t i = 0; i < simulation_steps; i++) {
        x_true = EKF_TestData::bicycle_model_state_function<T>(x_true, u, parameters);
        y_store[delay_index] = EKF_TestData::bicycle_model_measurement_function<T>(x_true, parameters);

        // system delay
        delay_index++;
        if (delay_index > NUMBER_OF_DELAY) {
            delay_index = 0;
        }

        ekf.predict(u);
        ekf.update(y_store[delay_index]);

        x_true_store[store_index] = x_true;
        x_estimated_store[store_index] = ekf.get_x_hat_without_delay();
        store_index++;
        
        if (store_index >= STORE_SIZE) {
            store_index = 0;
        }
    }

    for (std::size_t i = 0; i < STORE_SIZE; i++) {
        tester.expect_near(x_true_store[i].matrix.data, x_estimated_store[i].matrix.data, NEAR_LIMIT_SOFT,
            "check ExtendedKalmanFilter simulation x estimate.");
    }


    tester.throw_error_if_test_failed();
}

template <typename T>
void check_python_control_unscented_kalman_filter(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;

    MCAPTester<T> tester;

    //constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-4);
    constexpr T NEAR_LIMIT_SOFT = 1.0e-1F;

    /* UKF定義準備 */
    constexpr std::size_t NUMBER_OF_DELAY = 5;

    constexpr std::size_t STATE_SIZE = EKF_TestData::STATE_SIZE;
    constexpr std::size_t INPUT_SIZE = EKF_TestData::INPUT_SIZE;
    constexpr std::size_t OUTPUT_SIZE = EKF_TestData::OUTPUT_SIZE;

    using X_Type = StateSpaceState_Type<T, STATE_SIZE>;
    using U_Type = StateSpaceInput_Type<T, INPUT_SIZE>;
    using Y_Type = StateSpaceOutput_Type<T, OUTPUT_SIZE>;


    auto Q = make_KalmanFilter_Q<STATE_SIZE>(
        static_cast<T>(0.01), static_cast<T>(0.01),
        static_cast<T>(0.01));

    using Q_Type = decltype(Q);

    auto R = make_KalmanFilter_R<OUTPUT_SIZE>(
        static_cast<T>(1), static_cast<T>(1),
        static_cast<T>(1), static_cast<T>(1));

    using R_Type = decltype(R);

    /* パラメータ */
    using Parameter_Type = EKF_TestData::BicycleModelParameter<T>;

    Parameter_Type parameters(
        static_cast<T>(0.1),
        static_cast<T>(0.5),
        static_cast<T>(-1.0),
        static_cast<T>(-1.0),
        static_cast<T>(10.0),
        static_cast<T>(10.0)
    );

    /* 状態方程式、出力方程式 */
    StateFunction_Object<X_Type,
        U_Type,
        EKF_TestData::BicycleModelParameter<T>> state_function;
    state_function = EKF_TestData::bicycle_model_state_function<T>;

    MeasurementFunction_Object<Y_Type, X_Type,
        EKF_TestData::BicycleModelParameter<T>> measurement_function;
    measurement_function = EKF_TestData::bicycle_model_measurement_function<T>;

    /* UKF定義 */
    UnscentedKalmanFilter<U_Type, Q_Type, R_Type, Parameter_Type, NUMBER_OF_DELAY>
        ukf(Q, R, state_function, measurement_function, parameters,
            static_cast<T>(0), static_cast<T>(0), static_cast<T>(0));

    UnscentedKalmanFilter<U_Type, Q_Type, R_Type, Parameter_Type, NUMBER_OF_DELAY>
        ukf_copy = ukf;
    UnscentedKalmanFilter<U_Type, Q_Type, R_Type, Parameter_Type, NUMBER_OF_DELAY>
        ukf_move = std::move(ukf_copy);
    ukf = ukf_move;

    ukf.kappa = static_cast<T>(0);
    ukf.alpha = static_cast<T>(0.5);
    ukf.beta = static_cast<T>(2);
    ukf.calculate_weights();

    /* シミュレーション */
    std::size_t simulation_steps = 200;

    auto x_true_initial = make_StateSpaceState<STATE_SIZE>(
        static_cast<T>(2), static_cast<T>(6), static_cast<T>(0.3)
    );
    decltype(x_true_initial) x_true;

    auto u = make_StateSpaceInput<INPUT_SIZE>(
        static_cast<T>(2.0), static_cast<T>(0.1)
    );

    ukf.X_hat.template set<0, 0>(static_cast<T>(0.0));
    ukf.X_hat.template set<1, 0>(static_cast<T>(0.0));
    ukf.X_hat.template set<2, 0>(static_cast<T>(0.0));

    constexpr std::size_t STORE_SIZE = 10;
    std::array<X_Type, STORE_SIZE> x_true_store;
    std::array<X_Type, STORE_SIZE> x_estimated_store;

    std::size_t store_index = 0;

    std::array<StateSpaceOutput_Type<T, OUTPUT_SIZE>, (NUMBER_OF_DELAY + 1)> y_store;

    std::size_t delay_index = 0;

    x_true = x_true_initial;
    for (std::size_t i = 0; i < simulation_steps; i++) {
        x_true = EKF_TestData::bicycle_model_state_function<T>(x_true, u, parameters);
        y_store[delay_index] = EKF_TestData::bicycle_model_measurement_function<T>(x_true, parameters);

        // system delay
        delay_index++;
        if (delay_index > NUMBER_OF_DELAY) {
            delay_index = 0;
        }

        ukf.predict(u);
        ukf.update(y_store[delay_index]);

        x_true_store[store_index] = x_true;
        x_estimated_store[store_index] = ukf.get_x_hat_without_delay();
        store_index++;

        if (store_index >= STORE_SIZE) {
            store_index = 0;
        }
    }

    for (std::size_t i = 0; i < STORE_SIZE; i++) {
        tester.expect_near(x_true_store[i].matrix.data, x_estimated_store[i].matrix.data, NEAR_LIMIT_SOFT,
            "check UnscentedKalmanFilter simulation x estimate.");
    }


    tester.throw_error_if_test_failed();
}


template <typename T>
void check_python_control_least_squares(void) {
    using namespace PythonNumpy;
    using namespace PythonControl;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-4);
    //constexpr T NEAR_LIMIT_SOFT = 5.0e-2F;

    /* 最小二乗法定義 */
    constexpr std::size_t LS_NUMBER_OF_DATA = LS_TestData::LS_NUMBER_OF_DATA;
    constexpr std::size_t X_SIZE = LS_TestData::X_SIZE;
    constexpr std::size_t Y_SIZE = LS_TestData::Y_SIZE;

    using X_Type = DenseMatrix_Type<T, LS_NUMBER_OF_DATA, X_SIZE>;

    LeastSquares_Type<X_Type> ls = make_LeastSquares<X_Type>();
    LeastSquares_Type<X_Type> ls_copy = ls;
    LeastSquares_Type<X_Type> ls_move = std::move(ls_copy);
    ls = ls_move;


    /* 推定実行 */
    Matrix<DefDense, T, LS_NUMBER_OF_DATA, X_SIZE> X;
    for (std::size_t i = 0; i < LS_NUMBER_OF_DATA; i++) {
        for (std::size_t j = 0; j < X_SIZE; j++) {
            X(i, j) = static_cast<T>(LS_TestData::test_X(i, j));
        }
    }

    Matrix<DefDense, T, LS_NUMBER_OF_DATA, Y_SIZE> Y;
    for (std::size_t i = 0; i < LS_NUMBER_OF_DATA; i++) {
        for (std::size_t j = 0; j < Y_SIZE; j++) {
            Y(i, j) = static_cast<T>(LS_TestData::test_Y(i, j));
        }
    }

    ls.fit(X, Y);
    auto weights = ls.get_weights();

    Matrix<DefDense, T, (X_SIZE + 1), 1> weights_answer({
        { static_cast<T>(1.50065106)},
        { static_cast<T>(-0.79889541)},
        { static_cast<T>(0.28852566)}
        });

    tester.expect_near(weights.matrix.data, weights_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check LeastSquares fit.");

    auto Y_predicted = ls.predict(X);

    Matrix<DefDense, T, 3, Y_SIZE> Y_predicted_answer_part({
        {static_cast<T>(1.02100123)},
        {static_cast<T>(13.44101992)},
        {static_cast<T>(8.93927031)}
    });

    for (std::size_t i = 0; i < 3; i++) {
        tester.expect_near(Y_predicted(i, 0), Y_predicted_answer_part(i, 0), NEAR_LIMIT_STRICT,
            "check LeastSquares predict.");
    }


    /* 逐次最小二乗法定義 */
    constexpr std::size_t RLS_NUMBER_OF_DATA = RLS_TestData::RLS_NUMBER_OF_DATA;
    constexpr std::size_t RLS_X_SIZE = RLS_TestData::X_SIZE;
    constexpr std::size_t RLS_Y_SIZE = RLS_TestData::Y_SIZE;

    using RLS_X_Type = StateSpaceState_Type<T, RLS_X_SIZE>;

    RecursiveLeastSquares_Type<RLS_X_Type> rls =
        make_RecursiveLeastSquares<RLS_X_Type>(static_cast<T>(0), static_cast<T>(0.1));
    RecursiveLeastSquares_Type<RLS_X_Type> rls_copy = rls;
    RecursiveLeastSquares_Type<RLS_X_Type> rls_move = std::move(rls_copy);
    rls = rls_move;

    rls.set_lambda(static_cast<T>(0.9));

    /* 推定実行 */
    Matrix<DefDense, T, RLS_NUMBER_OF_DATA, RLS_X_SIZE> RLS_X;
    for (std::size_t i = 0; i < RLS_NUMBER_OF_DATA; i++) {
        for (std::size_t j = 0; j < RLS_X_SIZE; j++) {
            RLS_X(i, j) = static_cast<T>(RLS_TestData::test_X(i, j));
        }
    }

    Matrix<DefDense, T, RLS_NUMBER_OF_DATA, RLS_Y_SIZE> RLS_Y;
    for (std::size_t i = 0; i < RLS_NUMBER_OF_DATA; i++) {
        for (std::size_t j = 0; j < RLS_Y_SIZE; j++) {
            RLS_Y(i, j) = static_cast<T>(RLS_TestData::test_Y(i, j));
        }
    }

    Matrix<DefDense, T, (RLS_X_SIZE + 1), RLS_NUMBER_OF_DATA> RLS_weights_store;

    for (std::size_t i = 0; i < RLS_NUMBER_OF_DATA; i++) {
        RLS_X_Type x;
        x(0, 0) = RLS_X(i, 0);
        x(1, 0) = RLS_X(i, 1);

        rls.update(x, RLS_Y(i, 0));

        auto RLS_weights = rls.get_weights();
        RLS_weights_store(0, i) = RLS_weights(0, 0);
        RLS_weights_store(1, i) = RLS_weights(1, 0);
        RLS_weights_store(2, i) = RLS_weights(2, 0);
    }

    Matrix<DefDense, T, (RLS_X_SIZE + 1), 1> RLS_weight_answer_last({
        {static_cast<T>(0.49970984)},
        {static_cast<T>(-0.19947724)},
        {static_cast<T>(0.29828256)}
    });

    for (std::size_t i = 0; i < (RLS_X_SIZE + 1); i++) {
        tester.expect_near(
            RLS_weights_store(i, RLS_NUMBER_OF_DATA - 1), RLS_weight_answer_last(i, 0),
            NEAR_LIMIT_STRICT,
            "check RecursiveLeastSquares update.");
    }

    /* 予測 */
    RLS_X_Type x_last;
    x_last(0, 0) = RLS_X(RLS_NUMBER_OF_DATA - 1, 0);
    x_last(1, 0) = RLS_X(RLS_NUMBER_OF_DATA - 1, 1);

    auto RLS_Y_predicted = rls.predict(x_last);

    T RLS_Y_predicted_answer_last = static_cast<T>(-0.71824759);

    tester.expect_near(RLS_Y_predicted,
        RLS_Y_predicted_answer_last, NEAR_LIMIT_STRICT,
        "check RecursiveLeastSquares predict.");


    tester.throw_error_if_test_failed();
}


int main(void) {

    check_python_control_state_space<double>();

    check_python_control_state_space<float>();

    check_python_control_transfer_function<double>();

    check_python_control_transfer_function<float>();

    check_python_control_pid_controller<double>();

    check_python_control_pid_controller<float>();

    check_python_control_lqr<double>();

    check_python_control_lqr<float>();

    check_python_control_linear_kalman_filter<double>();

    check_python_control_linear_kalman_filter<float>();

    check_python_control_extended_kalman_filter<double>();

    check_python_control_extended_kalman_filter<float>();

    check_python_control_unscented_kalman_filter<double>();

    check_python_control_unscented_kalman_filter<float>();

    check_python_control_least_squares<double>();

    check_python_control_least_squares<float>();


    return 0;
}
