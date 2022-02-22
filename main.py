from math import sqrt


class KNearestNeighbour:

    """ Return the calculated euclidian distance between two vectors

    Ignores the last index of provided row1 and row2 based on data/iris.csv
    Ex: [1, 2, "excluded_last_value"]
    """
    @staticmethod
    def euclidian_distance(row1, row2):
        if len(row1) != len(row2):
            raise ValueError("The length of both rows must be equal.")

        sum_squares = 0
        for index in range(len(row1) - 1):
            sum_squares += pow((row1[index] - float(row2[index])), 2)

        return sqrt(sum_squares)

    def nearest_neighbour(self, training_row, num_neighbours=1):
        neighbours = []
        with open('./data/iris.csv') as data_file:
            line = data_file.readline()
            while line:
                distance = self.euclidian_distance(training_row, line.split(','))
                neighbours.append((distance, line))
                line = data_file.readline()

        neighbours.sort(key=lambda tup: tup[0])
        return neighbours[0:num_neighbours] if num_neighbours <= len(neighbours) - 1 else neighbours


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    KNearestNeighbour.euclidian_distance()
