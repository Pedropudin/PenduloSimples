import numpy as np
import scipy.constants
from math import sin
from math import cos
from math import tan

from animation import animate_pendulum

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
theta = np.array([0.7]*(m))

# ======== Functions =================== #
def G_i(i, theta, m, alpha, beta):
    if(i == 0):
        return h**(-2)*(alpha - 2*theta[i] + theta[i+1]) + g/L*sin(theta[i])
    elif(i == m-1):
        return h**(-2)*(theta[i-1] - 2*theta[i] + beta) + g/L*sin(theta[i])
    else:
        return h**(-2)*(theta[i-1] - 2*theta[i] + theta[i+1]) + g/L*sin(theta[i])

def create_G(m,theta, alpha, beta):
    res = np.zeros(m)

    for i in range(m): # revisar índices depois
        res[i] = G_i(i, theta, m, alpha, beta)

    return res

def create_jacobian(m,h,g,L,theta):
    res = np.zeros((m,m))

    for i in range(m):
        for j in range(m):
            d = abs(i-j)
            if(d > 1):
                res[i][j] = 0
            elif(d == 1):
                res[i][j] = 1/(h**2)
            else:
                res[i][j] = g/L*cos(theta[i]) - 2/(h**2)

    return res

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

def solve_newton_rhapson(m, h, g, L, alpha, beta, theta, err_max=0):
    err = []
    run = True

    while(run):
        J = create_jacobian(m, h, g, L, theta)
        G = create_G(m, theta, alpha, beta)
        dTheta = solve_linear_system(m, J, -G)
        theta_new = dTheta + theta

        err.append(np.max(np.abs(theta_new - theta)) / np.max(np.abs(theta_new)))
        print("Erro: ", err[-1])
        if(err[-1] < err_max):
            run = False

        theta = theta_new

    return theta

def solve_triangular_linear_system(m, A, B, lower = True):
    """
    Resolve um sistema linear com `m` equações na forma Ax = B onde A é uma matriz triangular
    Args:
        - m: quantidade de equações
        - A: matriz triangular de tamanho m por m
        - B: vetor dos resultados de tamanho m
        - lower: boleano indicando se é inferior
    Retr:
        - Vetor x de tamanho m
    """
    #! check if everything is in the correct data type and shape
    res = np.zeros(m)

    if(lower):
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

def solve_linear_system(m, A, B):
    """
    Resolve um sistema linear com `m` equações na forma: Ax = B
    Args:
        - m: quantidade de equações
        - A: matriz dos coeficientes de tamanho m por m
        - B: vetor dos resultados de tamanho m
    Retr:
        - Vetor x de tamanho m
    """
    #! check if everything is in the correct data type and shape
    
    L, U = LU_decomposition(m, A)

    y = solve_triangular_linear_system(m,L,B, True)
    x = solve_triangular_linear_system(m,U,y, False)

    return x

def LU_decomposition(m, A):
    """
    Faz a decomposição LU de uma matriz quadrada de tamanho `m`
    Args:
        - m: tamanho da matriz
        - A: matriz
    Retr:
        - Matriz L de tamanho m por m
        - Matriz U de tamanho m por m
    """
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

                L[k][i] = (A[k][i] - sum) / U[i][i]

    return L,U

# ========= Main ======================= #
def main():
    a = solve_newton_rhapson(m, h, g, L, alpha, beta, theta, 1e-3)

    full_theta = np.concatenate(([alpha], a, [beta]))
    animate_pendulum(full_theta, L, T)


if __name__ == '__main__':
    main()

