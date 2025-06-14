/**
 * @file least_squares_method.cpp
 * @brief Demonstrates the usage of a Least Squares fitting method using dense
 * matrices.
 *
 * This program initializes input (X) and output (Y) matrices with test data,
 * fits a Least Squares regression model, and outputs the resulting weights.
 * The code utilizes a LeastSquares class template, which is constructed using
 * a dense matrix type for the input data. The fit method computes the optimal
 * weights to minimize the squared error between the predicted and actual
 * outputs.
 */
#include <iostream>

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <array>

using namespace PythonNumpy;
using namespace PythonControl;

constexpr std::size_t LS_NUMBER_OF_DATA = 20;
constexpr std::size_t X_SIZE = 2;
constexpr std::size_t Y_SIZE = 1;

double get_LS_test_X(std::size_t i, std::size_t j);

double get_LS_test_Y(std::size_t i, std::size_t j);

int main(void) {
  /* define Least Squares */
  using X_Type = DenseMatrix_Type<double, LS_NUMBER_OF_DATA, X_SIZE>;
  auto ls = make_LeastSquares<X_Type>();

  /* fit */
  DenseMatrix_Type<double, LS_NUMBER_OF_DATA, X_SIZE> X;
  for (std::size_t i = 0; i < LS_NUMBER_OF_DATA; i++) {
    for (std::size_t j = 0; j < X_SIZE; j++) {
      X(i, j) = get_LS_test_X(i, j);
    }
  }

  DenseMatrix_Type<double, LS_NUMBER_OF_DATA, Y_SIZE> Y;
  for (std::size_t i = 0; i < LS_NUMBER_OF_DATA; i++) {
    for (std::size_t j = 0; j < Y_SIZE; j++) {
      Y(i, j) = get_LS_test_Y(i, j);
    }
  }

  ls.fit(X, Y);
  auto weights = ls.get_weights();

  for (std::size_t i = 0; i < (X_SIZE + 1); i++) {
    std::cout << "weights[" << i << "] = " << weights(i, 0) << std::endl;
  }

  return 0;
}

DenseMatrix_Type<double, LS_NUMBER_OF_DATA, X_SIZE>
    LS_test_X({{3.74540119, 6.11852895}, {9.50714306, 1.39493861},
               {7.31993942, 2.92144649}, {5.98658484, 3.66361843},
               {1.5601864, 4.56069984},  {1.5599452, 7.85175961},
               {0.58083612, 1.99673782}, {8.66176146, 5.14234438},
               {6.01115012, 5.92414569}, {7.08072578, 0.46450413},
               {0.20584494, 6.07544852}, {9.69909852, 1.70524124},
               {8.32442641, 0.65051593}, {2.12339111, 9.48885537},
               {1.81824967, 9.65632033}, {1.8340451, 8.08397348},
               {3.04242243, 3.04613769}, {5.24756432, 0.97672114},
               {4.31945019, 6.84233027}, {2.9122914, 4.40152494}});

DenseMatrix_Type<double, LS_NUMBER_OF_DATA, Y_SIZE> LS_test_Y(
    {{1.02314365},  {13.4341866},  {8.95097739},  {6.33677408},  {-1.00619163},
     {-3.66108659}, {-0.43941794}, {9.18073529},  {4.58479329},  {10.55119905},
     {-4.25274788}, {13.48144376}, {12.25144165}, {-4.11319608}, {-4.70228814},
     {-3.40553991}, {2.43015967},  {7.37233916},  {1.30855191},  {1.14336633}});

double get_LS_test_X(std::size_t i, std::size_t j) { return LS_test_X(i, j); }

double get_LS_test_Y(std::size_t i, std::size_t j) { return LS_test_Y(i, j); }
