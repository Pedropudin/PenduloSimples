import numpy as np
import scipy.constants
from math import sin

# ======= Constant Values ============== #
g = scipy.constants.g
L = 10
T = 100
alpha = -10
beta = 10

# ======= Simulation Variables ========= #
m = 100
h = T/(m+1)

# initialize theta with the initial guess
theta = np.zeros(m+2)
theta[0] = alpha
theta[m+1] = beta

# ======== Functions =================== #
def G_i(i, theta):
    if(i > 0 or i < (m+1)):
        h**(-2)*(theta[i-1] - 2*theta[i] + theta[i+1]) + g/L*sin(theta[i])
    else:
        pass

# ========= Main ======================= #
def main():
    pass


if __name__ == '__main__':
    main()

