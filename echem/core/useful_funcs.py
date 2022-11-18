import numpy as np


class ClassMethods:
    def check_existence(self, variable):
        """
        This function checks whether desires variable is not None
        :param variable: desired variable
        :return: nothing
        """
        if getattr(self, variable) is None:
            raise ValueError(f'{variable} is not defined')


def nearest_array_index(array, value):
    return (np.abs(array - value)).argmin()
