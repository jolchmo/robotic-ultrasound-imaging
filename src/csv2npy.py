import numpy as np
import csv

csv_reader = csv.reader(open('waypoints_loop_path.csv', 'r', encoding='utf-8-sig'))
waypoints = []
for row in csv_reader:

    waypoints.append([round(float(row[0]), 5), round(float(row[1]), 5), 1.07])

np.save('liver_loop_path.npy', waypoints)
