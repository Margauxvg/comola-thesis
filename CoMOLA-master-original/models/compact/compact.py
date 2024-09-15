import numpy as np
import os

def getsumofneighbors(matrix, i, j, value):
    if value < 0:
        return 0
    region = matrix[max(0, i-1) : i+2,
                    max(0, j-1) : j+2]
    region = region[region == value]
    return region.size - 1

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

inputMatrix = np.loadtxt(fname = os.path.join(current_directory, "map.asc"), skiprows=6)
rows, columns = inputMatrix.shape
results = 0

for i in range(rows):
    for j in range(columns):
        cellValue = inputMatrix[i][j]
        result = getsumofneighbors(inputMatrix, i, j, cellValue)
        print(result)
        results += result

print("results", results)
# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))
# Define the filename
f = open(os.path.join(current_directory, "compact_output.csv"), "w")
f.write(str(results))
f.close


