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
    subplots_shape = np.zeros((2, 1), dtype=int)
    subplots_signals_list = []


class SimulationPlotter:
    def __init__(self):
        self.configuration = Configuration()
        self.name_to_object_dictionary = {}

    def append(self, object):
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

            # %% append object
        if object_name in self.name_to_object_dictionary:
            self.name_to_object_dictionary[object_name].append(object)
        else:
            self.name_to_object_dictionary[object_name] = [object]

    def assign(self, signal_name, position, column=0, row=0):
        shape = np.array([[position[0]], [position[1]]], dtype=int)

        self.configuration.subplots_signals_list.append(
            SubplotsInfo(signal_name, shape, column, row))

    def plot(self, suptitle=""):

        subplots_signals_list = self.configuration.subplots_signals_list

        if len(subplots_signals_list) == 0:
            print("No subplots to show.")
            return

        shape = np.zeros((2, 1), dtype=int)
        for signal_info in subplots_signals_list:
            if shape[0, 0] < signal_info.shape[0, 0] + 1:
                shape[0, 0] = signal_info.shape[0, 0] + 1
            if shape[1, 0] < signal_info.shape[1, 0] + 1:
                shape[1, 0] = signal_info.shape[1, 0] + 1

        self.configuration.subplots_shape = shape

        fig, axs = plt.subplots(shape[0, 0], shape[1, 0])
        fig.suptitle(suptitle)

        for signal_info in subplots_signals_list:
            signal_object_list = self.name_to_object_dictionary[signal_info.signal_name]

            steps = len(signal_object_list)

            signal = np.zeros((steps, 1))
            if isinstance(signal_object_list[0], np.ndarray):
                for i in range(steps):
                    signal[i, 0] = signal_object_list[i][signal_info.column,
                                                         signal_info.row]
            else:
                for i in range(steps):
                    signal[i, 0] = signal_object_list[i]

            label_name = signal_info.signal_name + \
                f"[{signal_info.column}, {signal_info.row}]"

            if shape[0] == 1 and shape[1] == 1:
                axs.plot(
                    range(steps), signal, label=label_name)
                axs.legend()
                axs.grid(True)
            elif shape[0] == 1:
                axs[signal_info.shape[1, 0]].plot(
                    range(steps), signal, label=label_name)
                axs[signal_info.shape[1, 0]].legend()
                axs[signal_info.shape[1, 0]].grid(True)
            elif shape[1] == 1:
                axs[signal_info.shape[0, 0]].plot(
                    range(steps), signal, label=label_name)
                axs[signal_info.shape[0, 0]].legend()
                axs[signal_info.shape[0, 0]].grid(True)
            else:
                axs[signal_info.shape[0, 0], signal_info.shape[1, 0]].plot(
                    range(steps), signal, label=label_name)
                axs[signal_info.shape[0, 0], signal_info.shape[1, 0]].legend()
                axs[signal_info.shape[0, 0],
                    signal_info.shape[1, 0]].grid(True)

        plt.show()
