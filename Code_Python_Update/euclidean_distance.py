from scipy.spatial import distance

# Define two points as tuples or lists
point1 = (2, 3, 4)
point2 = (1, 2, 5)

# Calculate Euclidean distance between the points
euclidean_dist = distance.euclidean(point1, point2)
print(euclidean_dist)