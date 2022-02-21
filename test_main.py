import unittest

from main import KNearestNeighbour

euclidian_distance_params = [
    [[1, "filler"], [0, "filler"], 1],
    [[0, "filler"], [0, "filler"], 0],
    [[1, 2, 3, "filler"], [1, 2, 3, "filler"], 0],
    [[5, 2, 7, "filler"], [3, 2, 11, "filler"], 4.47213595499958],
    [[2, 1, "filler"], [5, 14, "filler"], 13.341664064126334]
]


class Testing(unittest.TestCase):
    def test_euclidian_distance(self):
        for params in euclidian_distance_params:
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


if __name__ == '__main__':
    unittest.main()
