import math
import numpy as np

def euclidean_distance(point1, point2):
    return math.sqrt(math.pow(point1 - point2, 2) )


def nearest_neighbor(p, points, k=3):
    """Return the nearest neighbour of a point"""
    distances = []
    for point in points:
        dist = euclidean_distance(p, point)
        distances.append(dist)

    distances = np.array(distances)
    ind = np.argsort(distances)

    #print(type(ind))
    return np.array(points)[ind[0:k]], [ind[0:k]]

if __name__ == "__main__":
    arr = [12, 16, 22, 30, 35, 39, 42,
           45, 48, 50, 53, 55, 56]

    n = len(arr)
    x = 36
    k = 1

    print(nearest_neighbor(x, arr, k=k))