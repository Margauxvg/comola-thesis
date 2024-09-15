import numpy as np
import os

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

inputMatrix = np.loadtxt(fname = os.path.join(current_directory, "map.asc"), skiprows=6)
rows, columns = inputMatrix.shape
economy_landuses = np.array([3, 7])
results = 0

for i in range(rows):
    for j in range(columns):
        cellValue = inputMatrix[i][j]
        if cellValue in economy_landuses:
          results += pow(25, 2)
print("results", results)

# Define the filename
file_path = os.path.join(current_directory, "economy_output.csv")
f = open(file_path, "w")
f.write(str(results))
f.close