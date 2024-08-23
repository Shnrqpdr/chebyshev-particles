import numpy as np
import scipy as sc
import scipy.special as sp
from complex_quadrature import complex_quadrature

def angleExponential(exponential):
    return np.angle(exponential)

R = 1.0
EPSILON = 1/10
BETA = 5
GAMMA = -10
L_MIN = -10
L_MAX = 10
I = 0 + 1j
M = 1.0
HBAR = 1.0
C = 1/2

def getChebyshev2d():
    theta = np.linspace(0, 2*np.pi, 360)
    r = R + EPSILON*np.cos(BETA*theta)
    L = np.sqrt((BETA**2)*(EPSILON**2)*np.sin(BETA*theta) + (R + EPSILON*np.cos(BETA*theta))**2)

    return r, theta, L

def cbshv2d_theta(theta): 
    return R + EPSILON*np.cos(BETA*theta)

def incident_wave(k, r, theta, theta_0):

    return np.exp(I*k*r*np.cos(theta - theta_0))


def C_ext(k, l_linha, theta_0):
    integration = complex_quadrature(lambda theta: sp.jv(l_linha, k*cbshv2d_theta(theta))*incident_wave(k, cbshv2d_theta(theta), theta, theta_0)*(np.exp(-I*l_linha*theta)), -np.pi, np.pi)

    return integration[0]


def W_ext(k, l, l_linha):
    integration = complex_quadrature(lambda theta: sp.jv(l_linha, k*cbshv2d_theta(theta))*sp.hankel1(l, k*cbshv2d_theta(theta))*(np.exp(I*(l - l_linha)*theta)), -np.pi, np.pi)
    
    return integration[0]

def get_system_linear(k, theta_0):
    # Inicializa a matriz
    A = np.zeros(((np.abs(L_MAX) + np.abs(L_MIN)), (np.abs(L_MAX) + np.abs(L_MIN))), dtype=complex)
    B = np.zeros((np.abs(L_MAX) + np.abs(L_MIN), 1), dtype=complex)
    # Percorre a matriz. Ela deve começar em 0, mas L_MIN = -5
    # Então, l_linha = i - L_MAX = -5 quando i = 0
    #                            = -4 quando i = 1
    #                            ....
    #                            = 5 quando i = 10
    # O mesmo vale para l, usando "j"
    for i in range(0, (np.abs(L_MAX) + np.abs(L_MIN))):
        l_linha = i - L_MAX
        for j in range(0, (np.abs(L_MAX) + np.abs(L_MIN))):
            l = j - L_MAX

            coef = 0.0

            if(l_linha == l):
                coef = (1 + (I*C*GAMMA)/4)*W_ext(k, l, l_linha)
            else:
                coef = ((I*C*GAMMA)/4)*W_ext(k, l, l_linha)
            
            A[i, j] = coef

    for j in range(0, (np.abs(L_MAX) + np.abs(L_MIN))):
        l_linha = j - L_MAX

        coef = C_ext(k, l_linha, theta_0)

        B[j] = coef

    return A, B

def scattering_amplitude(k, x, theta):
    amp = 0.0

    for i in range(0, (np.abs(L_MAX) + np.abs(L_MIN))):
        l = i - L_MAX

        fatorHankelAssintotica = np.sqrt(2.0/(np.pi*k))*np.exp(-I*(2*l + 1)*(np.pi/4.0))
        amp = amp + (-1)*(I*C*GAMMA/4)*x[i]*fatorHankelAssintotica*np.exp(I*l*theta)

    return amp

def differential_cross_section(k, x):
    cs = 0.0
    for i in range(0, (np.abs(L_MAX) + np.abs(L_MIN))):
        l = i - L_MAX

        fatorHankelAssintotica = 2.0/(np.pi*k)
        # print('fatorHankelAssintotica', fatorHankelAssintotica)
        omega_l = (np.abs(x[i])**2)
        # print('omega_l', omega_l)
        cs = cs + (-1)*(C*GAMMA/16.0)*(np.abs(x[i])**2)*fatorHankelAssintotica
        # print('cs', cs)

    # print(type(cs))
    return cs

def total_cross_section(k, x):
    cs = 0.0
    for i in range(0, (np.abs(L_MAX) + np.abs(L_MIN))):
        l = i - L_MAX

        fatorHankelAssintotica = 2.0/(np.pi*k)
        cs = cs + (-1)*(C*GAMMA/16.0)*(np.abs(x[i])**2)*fatorHankelAssintotica

    # print(type(cs*2*np.pi))
    return cs*2*np.pi

