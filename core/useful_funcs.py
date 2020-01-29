class ClassMethods:
    def check_existence(self, variable):
        """
        This function checks whether desires variable is not None
        :param variable: desired variable
        :return: nothing
        """
        if getattr(self, variable) is None:
            raise ValueError(f'{variable} is not defined')


def nearest_array_indices(array, value):
    i = 0
    while value < array[i]:
        i += 1
    return i - 1, i
