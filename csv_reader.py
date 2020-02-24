import csv
import numpy as np


def read_csv(file_name):
    points = np.array([])

    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            points[line_count] = [{row[0]}, {row[1]}, {row[2]}]

    return points
