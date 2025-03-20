import matplotlib.pyplot as plt
import numpy as np
import inspect


class SubplotsInfo:
    def __init__(self, signal_name, shape, column, row):
        self.signal_name = signal_name
        self.shape = shape
        self.column = column
        self.row = row


class Configuration:
    simulation_steps = 0

    subplots_shape = np.zeros((2, 1), dtype=int)
    subplots_signals_list = []


class SimulationPlotter:
    def __init__(self):
        self.configuration = Configuration()
        self.object_list = []
        self.name_to_object_dictionary = {}

    def append(self, object):
        self.object_list.append(object)

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the object_name that matches the matrix_in value
        object_name = None
        for name, value in caller_locals.items():
            if value is object:
                object_name = name
                break

        self.name_to_object_dictionary[object_name] = self.object_list

        self.update_configuration()

    def update_configuration(self):
        self.configuration.simulation_steps = len(self.object_list)

    def assign(self, signal_name, shape, column=0, row=0):
        shape = np.array([[shape[0]], [shape[1]]], dtype=int)

        self.configuration.subplots_signals_list.append(
            SubplotsInfo(signal_name, shape, column, row))

    def plot(self, suptitle=""):

        subplots_signals_list = self.configuration.subplots_signals_list

        if len(subplots_signals_list) == 0:
            print("No subplots to show.")
            return

        shape = np.zeros((2, 1), dtype=int)
        for signal_info in subplots_signals_list:
            if shape[0, 0] < signal_info.shape[0, 0]:
                shape[0, 0] = signal_info.shape[0, 0]
            if shape[1, 0] < signal_info.shape[1, 0]:
                shape[1, 0] = signal_info.shape[1, 0]

        self.configuration.subplots_shape = shape

        fig, axs = plt.subplots(shape[0, 0], shape[1, 0])
        fig.suptitle(suptitle)

        for signal_info in subplots_signals_list:
            signal_object_list = self.name_to_object_dictionary[signal_info.signal_name]

            signal = np.zeros((self.configuration.simulation_steps, 1))
            if isinstance(signal_object_list[0], np.ndarray):
                for i in range(self.configuration.simulation_steps):
                    signal[i, 0] = signal_object_list[i][signal_info.column,
                                                         signal_info.row]
            else:
                for i in range(self.configuration.simulation_steps):
                    signal[i, 0] = signal_object_list[i]

            label_name = signal_info.signal_name + \
                f"[{signal_info.column}, {signal_info.row}]"

            if shape[0] == 1 and shape[1] == 1:
                axs.plot(
                    range(self.configuration.simulation_steps),
                    signal,
                    label=label_name)
                axs.legend()
                axs.grid(True)
            else:
                axs[signal_info.row, signal_info.column].plot(
                    range(self.configuration.simulation_steps),
                    signal,
                    label=label_name)
                axs[signal_info.row, signal_info.column].legend()
                axs[signal_info.row, signal_info.column].grid(True)

        plt.show()
