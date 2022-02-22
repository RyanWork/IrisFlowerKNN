import unittest

from main import KNearestNeighbour


class Testing(unittest.TestCase):
    euclidian_distance_params = [
        [[1, "filler"], [0, "filler"], 1],
        [[0, "filler"], [0, "filler"], 0],
        [[1, 2, 3, "filler"], [1, 2, 3, "filler"], 0],
        [[5, 2, 7, "filler"], [3, 2, 11, "filler"], 4.47213595499958],
        [[2, 1, "filler"], [5, 14, "filler"], 13.341664064126334]
    ]

    def test_euclidian_distance(self):
        for params in self.euclidian_distance_params:
            row1 = params[0]
            row2 = params[1]
            expected_result = params[2]

            result = KNearestNeighbour.euclidian_distance(row1, row2)

            with self.subTest():
                self.assertEqual(result, expected_result)

    def test_euclidian_distance_value_error(self):
        row1 = [1, 2, 3]
        row2 = [1, 2]

        self.assertRaises(ValueError, KNearestNeighbour.euclidian_distance, row1, row2)

    def test_nearest_neighbour(self):
        training_row = [1, 2, 3, 4, 5]
        sut = KNearestNeighbour()

        result = sut.nearest_neighbour(training_row)

        self.assertEqual(result[0][0], 8.149233092751736)
        self.assertEqual(result[0][1], '6.4,2.8,5.6,2.2,Iris-virginica\n')

    def test_nearest_neighbour_max_num_neighbours_returns_entire_dataset(self):
        training_row = [1, 2, 3, 4, 5]
        sut = KNearestNeighbour()

        result = sut.nearest_neighbour(training_row, 9999)

        self.assertEqual(result[0][0], 8.149233092751736)
        self.assertEqual(result[0][1], '6.4,2.8,5.6,2.2,Iris-virginica\n')
        self.assertEqual(len(result), 150)



if __name__ == '__main__':
    unittest.main()
