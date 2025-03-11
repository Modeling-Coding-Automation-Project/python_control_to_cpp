#ifndef TEST_VS_EKF_DATA_HPP
#define TEST_VS_EKF_DATA_HPP

#include "python_control.hpp"

namespace EKF_TestData {

using namespace Base::Math;
using namespace PythonNumpy;
using namespace PythonControl;

/* bicycle model example */
constexpr std::size_t STATE_SIZE = 3;
constexpr std::size_t INPUT_SIZE = 2;
constexpr std::size_t OUTPUT_SIZE = 4;

template <typename T>
class BicycleModelParameter {
public:
    BicycleModelParameter() {};

    BicycleModelParameter(const T& delta_time, const T& wheelbase,
        const T& landmark_1_x, const T& landmark_1_y,
        const T& landmark_2_x, const T& landmark_2_y)
        : delta_time(delta_time), wheelbase(wheelbase),
        landmark_1_x(landmark_1_x), landmark_1_y(landmark_1_y),
        landmark_2_x(landmark_2_x), landmark_2_y(landmark_2_y) {}

public:
    T delta_time;
    T wheelbase;
    T landmark_1_x;
    T landmark_1_y;
    T landmark_2_x;
    T landmark_2_y;
};

template <typename T>
StateSpaceStateType<T, STATE_SIZE> bicycle_model_state_function(
    const StateSpaceStateType<T, STATE_SIZE> &X,
    const StateSpaceInputType<T, INPUT_SIZE> &U,
    const BicycleModelParameter<T> &parameter) {


    T x = X.template get<0, 0>();
    T y = X.template get<1, 0>();
    T theta = X.template get<2, 0>();
    T v = U.template get<0, 0>();
    T steering_angle = U.template get<1, 0>();

    T wheelbase = parameter.wheelbase;
    T delta_time = parameter.delta_time;

    return StateSpaceStateType<T, STATE_SIZE>({
        {-wheelbase * sin(theta) / tan(steering_angle) + wheelbase * sin(delta_time * v * tan(steering_angle) / wheelbase + theta) / tan(steering_angle) + x},
        {wheelbase * cos(theta) / tan(steering_angle) - wheelbase * cos(delta_time * v * tan(steering_angle) / wheelbase + theta) / tan(steering_angle) + y},
        {delta_time * v * tan(steering_angle) / wheelbase + theta}
        });
}

template <typename T, typename A_Type>
A_Type bicycle_model_state_function_jacobian(
    const StateSpaceStateType<T, STATE_SIZE> &X,
    const StateSpaceInputType<T, INPUT_SIZE> &U,
    const BicycleModelParameter<T> &parameter) {

    T theta = X.template get<2, 0>();
    T v = U.template get<0, 0>();
    T steering_angle = U.template get<1, 0>();

    T wheelbase = parameter.wheelbase;
    T delta_time = parameter.delta_time;

    A_Type A;

    A.template set<0, 0>(1);
    A.template set<0, 1>(0);
    A.template set<0, 2>(-wheelbase * cos(theta) / tan(steering_angle) + wheelbase * cos(delta_time * v * tan(steering_angle) / wheelbase + theta) / tan(steering_angle));
    A.template set<1, 0>(0);
    A.template set<1, 1>(1);
    A.template set<1, 2>(-wheelbase * sin(theta) / tan(steering_angle) + wheelbase * sin(delta_time * v * tan(steering_angle) / wheelbase + theta) / tan(steering_angle));
    A.template set<2, 0>(0);
    A.template set<2, 1>(0);
    A.template set<2, 2>(1);

    return A;
}


} // namespace EKF_TestData

#endif // TEST_VS_EKF_DATA_HPP
