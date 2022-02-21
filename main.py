from math import sqrt


class KNearestNeighbour:

    """ Return the calculated euclidian distance between two vectors """
    @staticmethod
    def euclidian_distance(row1, row2):
        if len(row1) != len(row2):
            raise ValueError("The length of both rows must be equal.")

        sum_squares = 0
        for index, item in enumerate(row1):
            sum_squares += pow((row1[index] - row2[index]), 2)

        return sqrt(sum_squares)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    KNearestNeighbour.euclidian_distance()
