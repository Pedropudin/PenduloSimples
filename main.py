import numpy as np
import scipy.constants
from math import sin
from math import tan

# ======= Constant Values ============== #
pi = scipy.constants.pi
g = 9.8
L = 1
T = 2*pi
alpha = 0.7
beta = 0.7

# ======= Simulation Variables ========= #
m = 100
h = T/(m+1)

# initialize theta with the initial guess
theta = np.array([0.7]*(m+2))
theta[0] = alpha
theta[m+1] = beta

# ======== Functions =================== #
def G_i(i, theta):
    if(i > 0 or i < (m+1)):
        return h**(-2)*(theta[i-1] - 2*theta[i] + theta[i+1]) + g/L*sin(theta[i])
    else:
        return theta[i]

def phi(i, theta):
    if(i > 0 or i < (m+1)):
        return (theta[i-1] - 2*theta[i] + theta[i+1])/(theta[i-1] - 2 + theta[i+1]) + tan(theta[i])
    else:
        return theta[i]

def print_jacobian_matrix(m):
    for i in range(m):
        for j in range(m):
            d = abs(i-j)
            if(d > 1):
                print("0 ", end="")
            elif(d == 1):
                print("1/h**2 ", end="")
            else:
                print(f"-2/h**2 + g/L*cos(theta_{i})")
        print()

"""
Resolve um sistema linear com `m` equações na forma Ax = B onde A é uma matriz triangular
Args:
    - m: quantidade de equações
    - A: matriz triangular de tamanho m por m
    - B: vetor dos resultados de tamanho m
    - upper: boleano indicando se é superior
Retr:
    - Vetor x de tamanho m
"""
def solve_triangular_linear_system(m, A, B, upper = True):
    #! check if everything is in the correct data type and shape
    res = np.zeros(m)

    if(upper):
        for i in range(m):
            s = 0
            for j in range(i):
                s += A[i][j]*res[j] # Maybe j and i are inverted
            res[i] = (B[i] - s)/A[i][i]
    else:
        for i in range(m-1,-1,-1):
            s = 0
            for j in range(m-1,i,-1):
                s += A[i][j]*res[j] # Maybe j and i are inverted
            res[i] = (B[i] - s)/A[i][i]


    return res

"""
Resolve um sistema linear com `m` equações na forma: Ax = B
Args:
    - m: quantidade de equações
    - A: matriz dos coeficientes de tamanho m por m
    - B: vetor dos resultados de tamanho m
Retr:
    - Vetor x de tamanho m
"""
def solve_linear_system(m, A, B):
    #! check if everything is in the correct data type and shape
    
    L, U = LU_decomposition(m, A)

    y = solve_triangular_linear_system(m,L,B, False)
    x = solve_triangular_linear_system(m,U,y, True)

    return x

"""
Faz a decomposição LU de uma matriz quadrada de tamanho `m`
Args:
    - m: tamanho da matriz
    - A: matriz
Retr:
    - Matriz L de tamanho m por m
    - Matriz U de tamanho m por m
"""
def LU_decomposition(m, A):
    U = np.zeros((m,m))
    L = np.zeros((m,m))

    #! Revisar isso depois, porque peguei do GFG
    for i in range(m):
        # Matriz Superior
        for k in range(i, m):

            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])

            U[i][k] = A[i][k] - sum

        # Matriz Inferior
        for k in range(i, m):
            if (i == k):
                L[i][i] = 1
            else:
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])

                L[k][i] = int((A[k][i] - sum) / U[i][i])

    return L,U

# ========= Main ======================= #
def main():
    a = np.array([
        [10,2],
        [1,0]
        ])

    b = np.array([1,1])

    r = solve_linear_system(2, a, b)

    print(r)


if __name__ == '__main__':
    main()

