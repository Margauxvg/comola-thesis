import numpy as np
import os 

def getCompatibilityNeighbors(matrix, i, j, value):
    region = matrix[max(0, i-1) : i+2,
                    max(0, j-1) : j+2]
    rows, columns = region.shape
    sum = 0
    for i in range(rows):
        for j in range(columns):
            neighborValue  = region[i][j]
            if neighborValue > 0 and cellValue > 0:
                sum += compatibilityMatrix[value-1][neighborValue-1]
    return sum - 1
# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

inputMatrix = np.loadtxt(fname = os.path.join(current_directory, "map.asc"), skiprows=6, dtype=int)
compatibilityMatrix = np.loadtxt(fname = os.path.join(current_directory, "compatibilityMatrix.asc"), skiprows=0)
rows, columns = inputMatrix.shape
results = 0

for i in range(rows):
    for j in range(columns):
        cellValue = inputMatrix[i][j]
        result = getCompatibilityNeighbors(inputMatrix, i, j, cellValue)
        print(round(result))
        results += round(result)

print("results", results)

# Define the filename
f = open(os.path.join(current_directory, "compatibility_output.csv"), "w")
f.write(str(results))
f.close