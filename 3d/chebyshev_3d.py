import numpy as np
import scipy as sc
import scipy.special as sp
from complex_quadrature import complex_dblquadrature

R = 1.0
EPSILON = 1/10
BETA = 5
ALPHA = 10
GAMMA = -10
L_MAX = 10
M_MIN = -L_MAX
M_MAX = L_MAX
I = 0 + 1j
M = 1.0
HBAR = 1.0
C = 1/2

def spherical_hn1(n,z,derivative=False):
    """ Spherical Hankel Function of the First Kind """
    return sp.spherical_jn(n,z,derivative)+1j*sp.spherical_yn(n,z,derivative)

def conjugate_sph_harm(m, l, phi, theta):
    return ((-1)**(-m))*sp.sph_harm(-m, l, phi, theta)

def getChebyshev3d():
    phi = np.linspace(0, 2*np.pi, 1000)
    theta = np.linspace(0, np.pi, 1000)   
    
    phi, theta = np.meshgrid(phi, theta)
    r = R + EPSILON*np.cos(BETA*phi)*np.cos(ALPHA*theta)

    return r, phi, theta, np.nan_to_num(surface_element(r, phi, theta))

def cbshv3d_thetaphi(phi, theta): 
    return R + EPSILON*np.cos(BETA*phi)*np.cos(ALPHA*theta)

def surface_element(r, phi, theta):
    r_phi = - EPSILON*BETA*np.sin(BETA*phi)*np.cos(ALPHA*theta)
    r_theta = - EPSILON*ALPHA*np.cos(BETA*phi)*np.sin(ALPHA*theta)
    
    surface_element = np.sqrt(((r_phi)**2 + (r**2)*(np.sin(theta)**2))*((r_theta)**2 + r**2) - (r_phi)*(r*np.sin(theta)*(np.cos(theta) - np.cos(phi)) + (r_theta)*(np.cos(phi)*np.cos(theta) + (np.sin(theta)**2))))
    return surface_element

def gamma(func, phi, theta):
    return func(phi, theta)

def incident_wave(k, r, phi, phi_0):

    return np.exp(I*k*r*np.cos(phi - phi_0))


def C_ext(k, l_linha, m_linha, phi_0):
    integration = complex_dblquadrature(lambda theta, phi:  
        gamma(lambda u, v: -10).flatten()*np.nan_to_num(surface_element(cbshv3d_thetaphi(phi, theta), phi, theta)).flatten()*sp.spherical_jn(l_linha, k*cbshv3d_thetaphi(phi, theta))*incident_wave(k, cbshv3d_thetaphi(phi, theta), phi, phi_0)*(sp.sph_harm(m_linha, l_linha, phi, theta)),
        0, 2*np.pi, -np.pi, np.pi)

    return integration[0]


def W_ext(k, l, l_linha, m_linha):
    integration = complex_dblquadrature(lambda theta, phi: 
        gamma(lambda u, v: -10).flatten()*np.nan_to_num(surface_element(cbshv3d_thetaphi(phi, theta), phi, theta)).flatten()*sp.spherical_jn(l_linha, k*cbshv3d_thetaphi(phi, theta))*spherical_hn1(l, k*cbshv3d_thetaphi(phi, theta))*(np.exp(I*(l - l_linha)*phi)), 
        0, 2*np.pi, -np.pi, np.pi)
    
    return integration[0]

def get_system_linear(k, phi_0):
    # Inicializa a matriz
    A = np.zeros(((np.abs(L_MAX) + np.abs(M_MIN)), (np.abs(L_MAX) + np.abs(M_MIN))), dtype=complex)
    B = np.zeros((np.abs(L_MAX) + np.abs(M_MIN), 1), dtype=complex)
    # Percorre a matriz. Ela deve começar em 0, mas L_MIN = -5
    # Então, l_linha = i - L_MAX = -5 quando i = 0
    #                            = -4 quando i = 1
    #                            ....
    #                            = 5 quando i = 10
    # O mesmo vale para l, usando "j"
    for i in range(0, (np.abs(L_MAX) + np.abs(M_MIN))):
        l_linha = i - M_MAX
        for j in range(0, (np.abs(L_MAX) + np.abs(M_MIN))):
            l = j - M_MAX

            coef = 0.0

            if(l_linha == l):
                coef = (1 + (I*C*k)*W_ext(k, l, l_linha))
            else:
                coef = (I*C*k)*W_ext(k, l, l_linha)
            
            A[i, j] = coef

    for j in range(0, (np.abs(L_MAX) + np.abs(M_MIN))):
        l_linha = j - L_MAX

        coef = C_ext(k, l_linha, phi_0)

        B[j] = coef

    return A, B


### TO DO ABAIXO

def scattering_amplitude(k, x, phi):
    amp = 0.0

    for i in range(0, (np.abs(M_MAX) + np.abs(M_MIN))):
        l = i - M_MAX

        fatorHankelAssintotica = np.sqrt(2.0/(np.pi*k))*np.exp(-I*(2*l + 1)*(np.pi/4.0))
        amp = amp + (-1)*(I*C*GAMMA/4)*x[i]*fatorHankelAssintotica*np.exp(I*l*phi)

    return amp

def differential_cross_section(k, x):
    cs = 0.0
    for i in range(0, (np.abs(L_MAX) + np.abs(M_MIN))):
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
    for i in range(0, (np.abs(L_MAX) + np.abs(M_MIN))):
        l = i - L_MAX

        fatorHankelAssintotica = 2.0/(np.pi*k)
        cs = cs + (-1)*(C*GAMMA/16.0)*(np.abs(x[i])**2)*fatorHankelAssintotica

    # print(type(cs*2*np.pi))
    return cs*2*np.pi

