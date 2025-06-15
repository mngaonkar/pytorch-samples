import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
U, s, VT = np.linalg.svd(A)
print("U:\n", U)
print("s:\n", s)
print("VT:\n", VT)

# check if VT components are orthogonal
v1 = VT[0]
print("v1:", v1)

print(f" v1 @ v1T{v1 @ v1.T}")  # should be close to identity matrix
print(f"dot product of v1 with v1 transpose: {np.dot(v1, v1.T)}") 