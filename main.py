import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time

@njit
def mandelbrot(c:np.number, max_iterations=1000):
    bound = 2
    z=0
    i=0
    while abs(z) < bound and i < max_iterations:
        z = z*z + c
        i += 1
    return i

@njit
def mandelbrot_matrix(X:np.array,Y:np.array):
    matrix=np.zeros((X.size,Y.size))
    for i in range(X.size):
        for j in range(Y.size):
            z=complex(X[i], Y[j])
            matrix[j,i] = mandelbrot(z)
    return matrix


if __name__ == "__main__":
    N = 1000
    X=np.linspace(-1.5, 0.5 , N)
    Y=np.linspace(-1.1, 1.1, N)
    
    print("Started calculations...")
    
    start = time.time()
    matrix=mandelbrot_matrix(X,Y)
    end = time.time()
    
    print(f"Execution time: {end - start:.4f} seconds")
    
    plt.figure(figsize=(8, 8))
    contour = plt.contourf(X, Y, matrix, levels=100)
    plt.colorbar(contour, label='Iterations')
    plt.title("Mandelbrot Set")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()