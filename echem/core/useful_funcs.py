import numpy as np
import subprocess


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


def is_float(element: any) -> bool:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def is_int(element: any) -> bool:
    if element is None:
        return False
    try:
        int(element)
        return True
    except ValueError:
        return False

def shell(cmd) -> str:
    '''
    Run shell command and return output as a string
    '''
    return subprocess.check_output(cmd, shell=True)
