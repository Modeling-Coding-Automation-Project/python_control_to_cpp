import numpy as np


class MCAPTester:
    def __init__(self):
        self.test_failed_flag = False

    def expect_near(self, actual, expected, tolerance, message):
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            if abs(actual - expected) <= tolerance:
                pass  # Do Nothing
            else:
                print(f"FAILURE: {message}")
                print()
                self.test_failed_flag = True
        elif isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
            if actual.shape != expected.shape:
                print(f"FAILURE: {message} Shape mismatch.")
                print()
                self.test_failed_flag = True
                return

            if not np.allclose(actual, expected, atol=tolerance):
                print(f"FAILURE: {message} Element mismatch.")
                print()
                self.test_failed_flag = True
        else:
            raise TypeError("Unsupported types for expect_near.")

    def expect_near_2d(self, actual, expected, tolerance, message):
        if not isinstance(actual, np.ndarray) or not isinstance(expected, np.ndarray):
            raise TypeError("Both actual and expected must be numpy arrays.")

        if actual.shape != expected.shape:
            print(f"FAILURE: {message} Shape mismatch.")
            print()
            self.test_failed_flag = True
            return

        if not np.allclose(actual, expected, atol=tolerance):
            print(f"FAILURE: {message} Element mismatch.")
            print()
            self.test_failed_flag = True

    def throw_error_if_test_failed(self):
        if self.test_failed_flag:
            raise RuntimeError("Test failed.")

    def reset_test_failed_flag(self):
        self.test_failed_flag = False
