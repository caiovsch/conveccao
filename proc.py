import numpy as np
from numba import njit
import timeit

''''''''''''''''''''''''''''''''''''
''' PARAMETROS DE ENTRADA '''

''' Parametros fisicos '''
COMPRIMENTO_X = 1
COMPRIMENTO_Y = 1
PRANDTL = 0.71  # Prandtl do ar
RAYLEIGH = 1e6
TEMPERATURA_ESQUERDA = 1
TEMPERATURA_DIREITA = 0
TEMPO_FINAL = 0.25

''' Parametros numericos '''
DELTA_T = 0.00002
N_DIV_X = 80
N_DIV_Y = 80  # numero de divisoes (numero de pontos - 1)

''' Parametros pos-processamento '''
NUMERO_FRAMES_TEMPO = 100
CAMINHO = "./resultados/"

''''''''''''''''''''''''''''''''''''
''' PRE-PROCESSAMENTO '''

delta_x = COMPRIMENTO_X/N_DIV_X
delta_y = COMPRIMENTO_Y/N_DIV_Y

x = np.arange(0, COMPRIMENTO_X + delta_x, delta_x)
y = np.arange(0, COMPRIMENTO_Y + delta_y, delta_y)
t = np.arange(0, TEMPO_FINAL + DELTA_T, DELTA_T)

''' Checa se os parametros garantem convergencia '''
if DELTA_T >= delta_x:
    msg = 'Para estabilidade, dt<dx'
    raise RuntimeError(msg)

if delta_x >= PRANDTL**0.5 or delta_y >= PRANDTL**0.5:
    msg = 'Para estabilidade, dx<1/(Re**0.5), dy<1/(Re**0.5)'
    raise RuntimeError(msg)

if DELTA_T >= 1/4/PRANDTL*delta_x**2 or DELTA_T >= 1/4/PRANDTL*delta_y**2:
    msg = 'Para estabilidade, dt<1/4*reynolds*dx**2, dt<1/4*reynolds*dy**2'
    raise RuntimeError(msg)

print(f'2 dt Ra /dx2 = {2*DELTA_T*RAYLEIGH/delta_x**2}')

''' Pre-alocando as matrizes da malha defasada'''
u = np.zeros([N_DIV_X+1, N_DIV_Y+2])  # componente x da velocidade
v = np.zeros([N_DIV_X+2, N_DIV_Y+1])  # componente y da velocidade
p = np.zeros([N_DIV_X+2, N_DIV_Y+2])  # pressao
theta = np.zeros_like(p)              # temperatura
u_star = np.zeros_like(u)
v_star = np.zeros_like(v)

''' Condicao inicial '''
theta[-1, :] = 2*TEMPERATURA_ESQUERDA

''' Pre-alocando as matrizes de recuperacao da malha original '''
uplot = np.zeros((N_DIV_X + 1, N_DIV_Y + 1))
vplot = np.zeros_like(uplot)
pplot = np.zeros_like(uplot)
thetaplot = np.zeros_like(uplot)

''' Variaves para salvar alguns frames '''
numero_passos_tempo = len(t) - 1
intervalo_captura = numero_passos_tempo // NUMERO_FRAMES_TEMPO
indices_tempo_selecionados = np.arange(0, len(t), intervalo_captura)

''' Pre-alocando as matrizes com os frames a salvar'''
frames_p = np.zeros((NUMERO_FRAMES_TEMPO+1, N_DIV_X+1, N_DIV_Y+1))
frames_u = np.zeros_like(frames_p)
frames_v = np.zeros_like(frames_p)
frames_theta = np.zeros_like(frames_p)

''''''''''''''''''''''''''''''''''''
''' PROCESSAMENTO '''


@njit
def calcular_u_star(u, v, N_DIV_X, N_DIV_Y, delta_x, delta_y, DELTA_T, u_star):

    for i in range(1, N_DIV_X):
        for j in range(0, N_DIV_Y):
            C1 = 1/4 * (v[i, j+1] + v[i-1, j+1] + v[i, j] + v[i-1, j])
            R = -DELTA_T * (u[i, j] * (u[i+1, j] - u[i-1, j])/(2*delta_x) + C1 * (u[i, j+1] - u[i, j-1])/(2*delta_y)) +\
                DELTA_T*PRANDTL * ((u[i+1, j] - 2*u[i, j] + u[i-1, j])/delta_x**2 + (u[i, j+1] - 2*u[i, j] + u[i, j-1])/delta_y**2)
            u_star[i, j] = u[i, j] + R

    # fronteira direta
    for j in range(0, N_DIV_Y):
        u_star[0, j] = 0
        u_star[N_DIV_X, j] = 0

    # pontos fantasmas
    for i in range(0, N_DIV_X+1):
        u_star[i, -1] = -u_star[i, 0]
        u_star[i, N_DIV_Y] = - u_star[i, N_DIV_Y - 1]

    return u_star


@njit
def calcular_v_star(u, v, N_DIV_X, N_DIV_Y, delta_x, delta_y, DELTA_T, v_star, theta):

    for i in range(0, N_DIV_X):
        for j in range(1, N_DIV_Y):
            C2 = 1/4 * (u[i+1, j] + u[i, j] + u[i+1, j-1] + u[i, j-1])
            R = -DELTA_T * (C2 * (v[i+1, j] - v[i-1, j])/(2*delta_x) + v[i, j] * (v[i, j+1] - v[i, j-1])/(2*delta_y)) + \
                DELTA_T*PRANDTL * ((v[i+1, j] - 2*v[i, j] + v[i-1, j])/delta_x**2 + (v[i, j+1] - 2*v[i, j] + v[i, j-1])/delta_y**2) +\
                DELTA_T*RAYLEIGH*PRANDTL * (theta[i, j] + theta[i, j-1])/2
            v_star[i, j] = v[i, j] + R

    # pontos fantasmas
    for j in range(0, N_DIV_Y+1):
        v_star[-1, j] = -v_star[0, j]
        v_star[N_DIV_X, j] = -v_star[N_DIV_X-1, j]

    # fronteira direta
    for i in range(0, N_DIV_X):
        v_star[i, 0] = 0
        v_star[i, N_DIV_Y] = 0

    return v_star


@njit
def calcular_theta(u, v, N_DIV_X, N_DIV_Y, delta_x, delta_y, DELTA_T, theta):

    theta_new = np.zeros_like(theta)

    # pontos internos
    for i in range(0, N_DIV_X):
        for j in range(0, N_DIV_Y):

            u_interp = (u[i - 1, j] + u[i, j]) / 2
            v_interp = (v[i, j - 1] + v[i, j]) / 2

            if i == 0 and j == 0:
                theta_new[i, j] = theta[i, j] -\
                                   DELTA_T * u_interp * (theta[i+1, j] - 2 + theta[i, j])/2/delta_x - \
                                   DELTA_T * v_interp * (theta[i, j+1] - theta[i, j])/2/delta_y + \
                                   DELTA_T * (2 - theta[i, j] - 2*theta[i, j] + theta[i+1, j])/delta_x**2 + \
                                   DELTA_T * (theta[i, j] - 2*theta[i, j] + theta[i, j+1])/delta_y**2

            elif i == 0 and j == N_DIV_Y - 1:
                theta_new[i, j] = theta[i, j] - \
                                  DELTA_T * u_interp * (theta[i + 1, j] - 2 + theta[i, j]) / 2 / delta_x - \
                                  DELTA_T * v_interp * (theta[i, j] - theta[i, j - 1]) / 2 / delta_y + \
                                  DELTA_T * (2 - theta[i, j] - 2 * theta[i, j] + theta[i + 1, j]) / delta_x ** 2 + \
                                  DELTA_T * (theta[i, j - 1] - 2 * theta[i, j] + theta[i, j]) / delta_y ** 2

            elif i == N_DIV_X - 1 and j == 0:
                theta_new[i, j] = theta[i, j] - \
                                  DELTA_T * u_interp * (-theta[i, j] - theta[i - 1, j]) / 2 / delta_x - \
                                  DELTA_T * v_interp * (theta[i, j + 1] - theta[i, j]) / 2 / delta_y + \
                                  DELTA_T * (theta[i - 1, j] - 2 * theta[i, j] - theta[i, j]) / delta_x ** 2 + \
                                  DELTA_T * (theta[i, j] - 2 * theta[i, j] + theta[i, j + 1]) / delta_y ** 2

            elif i == N_DIV_X - 1 and j == N_DIV_Y - 1:
                theta_new[i, j] = theta[i, j] - \
                                  DELTA_T * u_interp * (-theta[i, j] - theta[i - 1, j]) / 2 / delta_x - \
                                  DELTA_T * v_interp * (theta[i, j] - theta[i, j - 1]) / 2 / delta_y + \
                                  DELTA_T * (theta[i - 1, j] - 2 * theta[i, j] - theta[i, j]) / delta_x ** 2 + \
                                  DELTA_T * (theta[i, j - 1] - 2 * theta[i, j] + theta[i, j]) / delta_y ** 2

            elif i == 0 and j != 0 and j != N_DIV_Y - 1:
                theta_new[i, j] = theta[i, j] - \
                                  DELTA_T * u_interp * (theta[i + 1, j] - 2 + theta[i, j]) / 2 / delta_x - \
                                  DELTA_T * v_interp * (theta[i, j + 1] - theta[i, j - 1]) / 2 / delta_y + \
                                  DELTA_T * (2 - theta[i, j] - 2 * theta[i, j] + theta[i + 1, j]) / delta_x ** 2 + \
                                  DELTA_T * (theta[i, j - 1] - 2 * theta[i, j] + theta[i, j + 1]) / delta_y ** 2

            elif i == N_DIV_X - 1 and j != 0 and j != N_DIV_Y - 1:
                theta_new[i, j] = theta[i, j] - \
                                  DELTA_T * u_interp * (-theta[i, j] - theta[i - 1, j]) / 2 / delta_x - \
                                  DELTA_T * v_interp * (theta[i, j + 1] - theta[i, j - 1]) / 2 / delta_y + \
                                  DELTA_T * (theta[i - 1, j] - 2 * theta[i, j] - theta[i, j]) / delta_x ** 2 + \
                                  DELTA_T * (theta[i, j - 1] - 2 * theta[i, j] + theta[i, j + 1]) / delta_y ** 2

            elif j == 0 and i != 0 and i != N_DIV_X - 1:
                theta_new[i, j] = theta[i, j] - \
                                  DELTA_T * u_interp * (theta[i + 1, j] - theta[i - 1, j]) / 2 / delta_x - \
                                  DELTA_T * v_interp * (theta[i, j + 1] - theta[i, j]) / 2 / delta_y + \
                                  DELTA_T * (theta[i - 1, j] - 2 * theta[i, j] + theta[i + 1, j]) / delta_x ** 2 + \
                                  DELTA_T * (theta[i, j] - 2 * theta[i, j] + theta[i, j + 1]) / delta_y ** 2

            elif j == N_DIV_Y - 1 and i != 0 and i != N_DIV_X - 1:
                theta_new[i, j] = theta[i, j] - \
                                  DELTA_T * u_interp * (theta[i + 1, j] - theta[i - 1, j]) / 2 / delta_x - \
                                  DELTA_T * v_interp * (theta[i, j] - theta[i, j - 1]) / 2 / delta_y + \
                                  DELTA_T * (theta[i - 1, j] - 2 * theta[i, j] + theta[i + 1, j]) / delta_x ** 2 + \
                                  DELTA_T * (theta[i, j - 1] - 2 * theta[i, j] + theta[i, j]) / delta_y ** 2

            else:
                theta_new[i, j] = theta[i, j] -\
                                   DELTA_T * u_interp * (theta[i+1, j] - theta[i-1, j])/2/delta_x - \
                                   DELTA_T * v_interp * (theta[i, j+1] - theta[i, j-1])/2/delta_y + \
                                   DELTA_T * (theta[i-1, j] - 2*theta[i, j] + theta[i+1, j])/delta_x**2 + \
                                   DELTA_T * (theta[i, j-1] - 2*theta[i, j] + theta[i, j+1])/delta_y**2

    # pontos fantasmas
    theta_new[-1, 0:N_DIV_Y] = 2*TEMPERATURA_ESQUERDA - theta[0, 0:N_DIV_Y]  # esquerda, dirichilet
    theta_new[N_DIV_X, 0:N_DIV_Y] = theta[N_DIV_X-1, 0:N_DIV_Y]              # direita, dirichilet
    theta_new[0:N_DIV_X, N_DIV_Y] = theta[0:N_DIV_X, N_DIV_Y-1]              # topo, neumann
    theta_new[0:N_DIV_X, -1] = theta[0:N_DIV_X, 0]                           # base, neumann

    theta_new[-1, -1] = 2*TEMPERATURA_ESQUERDA - theta[0, 0]
    theta_new[-1, N_DIV_Y] = 2*TEMPERATURA_ESQUERDA - theta[0, N_DIV_Y - 1]
    theta_new[N_DIV_X, -1] = theta[N_DIV_X - 1, 0]
    theta_new[N_DIV_X, N_DIV_Y] = theta[N_DIV_X - 1, N_DIV_Y - 1]

    return theta_new


@njit
def calcular_pressao(u_star, v_star, N_DIV_X, N_DIV_Y, delta_x, delta_y, DELTA_T, tol, p):

    R = 0.0
    erro = 1000
    iter=0
    while erro > tol:
        iter += 1
        # print(iter)
        R_max = 0
        for i in range(0, N_DIV_X):
            for j in range(0, N_DIV_Y):

                if i == 0 and j == 0:
                    lamda = -(1 / delta_x ** 2 + 1 / delta_y ** 2)
                    R = (u_star[i+1, j] - u_star[i, j]) / DELTA_T / delta_x + (v_star[i, j + 1] - v_star[i, j]) / DELTA_T / delta_y - \
                        ((p[i+1, j] - p[i, j]) / delta_x ** 2 + (p[i, j + 1] - p[i, j]) / delta_y ** 2)

                elif i == 0 and j == N_DIV_Y-1:
                    lamda = -(1 / delta_x ** 2 + 1 / delta_y ** 2)
                    R = (u_star[i+1, j] - u_star[i, j]) / DELTA_T / delta_x + (v_star[i, j + 1] - v_star[i, j]) / DELTA_T / delta_y - \
                        ((p[i+1, j] - p[i, j]) / delta_x ** 2 + (-p[i, j] + p[i, j - 1]) / delta_y ** 2)

                elif i == N_DIV_X-1 and j == 0:
                    lamda = -(1 / delta_x ** 2 + 1 / delta_y ** 2)
                    R = (u_star[i+1, j] - u_star[i, j]) / DELTA_T / delta_x + (v_star[i, j + 1] - v_star[i, j]) / DELTA_T / delta_y - \
                        ((-p[i, j] + p[i-1, j]) / delta_x ** 2 + (p[i, j + 1] - p[i, j]) / delta_y ** 2)

                elif i == N_DIV_X-1 and j == N_DIV_Y-1:
                    lamda = -(1 / delta_x ** 2 + 1 / delta_y ** 2)
                    R = (u_star[i+1, j] - u_star[i, j]) / DELTA_T / delta_x + (v_star[i, j + 1] - v_star[i, j]) / DELTA_T / delta_y - \
                        ((-p[i, j] + p[i-1, j]) / delta_x ** 2 + (-p[i, j] + p[i, j - 1]) / delta_y ** 2)

                elif i == 0 and j != 0 and j != N_DIV_Y-1:
                    lamda = -(1 / delta_x ** 2 + 2 / delta_y ** 2)
                    R = (u_star[i+1, j] - u_star[i, j]) / DELTA_T / delta_x + (v_star[i, j + 1] - v_star[i, j]) / DELTA_T / delta_y - \
                        ((p[i+1, j] - p[i, j]) / delta_x ** 2 + (p[i, j + 1] - 2 * p[i, j] + p[i, j - 1]) / delta_y ** 2)

                elif i == N_DIV_X-1 and j != 0 and j != N_DIV_Y-1:
                    lamda = -(1 / delta_x ** 2 + 2 / delta_y ** 2)
                    R = (u_star[i+1, j] - u_star[i, j]) / DELTA_T / delta_x + (v_star[i, j + 1] - v_star[i, j]) / DELTA_T / delta_y - \
                        ((-p[i, j] + p[i-1, j]) / delta_x ** 2 + (p[i, j + 1] - 2 * p[i, j] + p[i, j - 1]) / delta_y ** 2)

                elif j == 0 and i != 0 and i != N_DIV_X-1:
                    lamda = -(2 / delta_x ** 2 + 1 / delta_y ** 2)
                    R = (u_star[i+1, j] - u_star[i, j]) / DELTA_T / delta_x + (v_star[i, j + 1] - v_star[i, j]) / DELTA_T / delta_y - \
                        ((p[i+1, j] - 2*p[i, j] + p[i-1, j]) / delta_x ** 2 + (p[i, j + 1] - p[i, j]) / delta_y ** 2)

                elif j == N_DIV_Y-1 and i != 0 and i != N_DIV_X-1:
                    lamda = -(2 / delta_x ** 2 + 1 / delta_y ** 2)
                    R = (u_star[i+1, j] - u_star[i, j]) / DELTA_T / delta_x + (v_star[i, j + 1] - v_star[i, j]) / DELTA_T / delta_y - \
                        ((p[i+1, j] - 2*p[i, j] + p[i-1, j]) / delta_x ** 2 + (-p[i, j] + p[i, j - 1]) / delta_y ** 2)

                else:
                    lamda = -(2 / delta_x ** 2 + 2 / delta_y ** 2)
                    R = (u_star[i+1, j] - u_star[i, j]) / DELTA_T / delta_x + (v_star[i, j + 1] - v_star[i, j]) / DELTA_T / delta_y - \
                        ((p[i+1, j] - 2*p[i, j] + p[i-1, j]) / delta_x ** 2 + (p[i, j + 1] - 2 * p[i, j] + p[i, j - 1]) / delta_y ** 2)

                R = R/lamda
                p[i, j] = p[i, j] + R

                if np.abs(R) > R_max:
                    R_max = np.abs(R)

        erro = R_max

    ''' FRONTEIRAS '''
    for i in range(0, N_DIV_X):
        p[i, -1] = p[i, 0]
        p[i, N_DIV_Y] = p[i, N_DIV_Y - 1]
    for j in range(0, N_DIV_Y):
        p[-1, j] = p[0, j]
        p[N_DIV_X, j] = p[N_DIV_X - 1, j]
    p[-1, -1] = p[0, 0]
    p[-1, N_DIV_Y] = p[0, N_DIV_Y - 1]
    p[N_DIV_X, -1] = p[N_DIV_X - 1, 0]
    p[N_DIV_X, N_DIV_Y] = p[N_DIV_X - 1, N_DIV_Y - 1]

    return p


@njit
def calcular_u(u_star, p, N_DIV_X, N_DIV_Y, delta_x, DELTA_T, u):
    for i in range(1, N_DIV_X):
        for j in range(-1, N_DIV_Y + 1):
            u[i, j] = u_star[i, j] - DELTA_T * (p[i, j] - p[i - 1, j]) / delta_x
    return u


@njit
def calcular_v(v_star, p, N_DIV_X, N_DIV_Y, delta_y, DELTA_T, v):
    for i in range(-1, N_DIV_X + 1):
        for j in range(1, N_DIV_Y):
            v[i, j] = v_star[i, j] - DELTA_T * (p[i, j] - p[i, j - 1]) / delta_y
    return v


@njit
def interpolar_malha(u, v, p, theta, N_DIV_X, N_DIV_Y, uplot, vplot, pplot, thetaplot):
    for i in range(0, N_DIV_X + 1):
        for j in range(0, N_DIV_Y + 1):
            uplot[i, j] = 1 / 2 * (u[i, j] + u[i, j - 1])
            vplot[i, j] = 1 / 2 * (v[i, j] + v[i - 1, j])
            pplot[i, j] = 1 / 4 * (p[i, j] + p[i - 1, j] + p[i, j - 1] + p[i - 1, j - 1])
            thetaplot[i, j] = 1 / 4 * (theta[i, j] + theta[i - 1, j] + theta[i, j - 1] + theta[i - 1, j - 1])
    return uplot, vplot, pplot, thetaplot


''' Salvando os frames da condicao inicial '''
frame_atual = 0
uplot, vplot, pplot, thetaplot = interpolar_malha(u, v, p, theta, N_DIV_X, N_DIV_Y, uplot, vplot, pplot, thetaplot)
indices_tempo_selecionados[0] = 0
frames_p[0] = pplot
frames_u[0] = uplot
frames_v[0] = vplot
frames_theta[0] = thetaplot

''' Avanco no tempo '''
start = timeit.default_timer()  # cronometro para avaliar custo computacional

for k in range(1, len(t)):

    if k % 1 == 0:
        print(f'tempo = {k * DELTA_T}')

    ''' CALCULO PARA CADA INSTANTE DE TEMPO '''
    u_star = calcular_u_star(u, v, N_DIV_X, N_DIV_Y, delta_x, delta_y, DELTA_T, u_star)
    v_star = calcular_v_star(u, v, N_DIV_X, N_DIV_Y, delta_x, delta_y, DELTA_T, v_star, theta)
    theta = calcular_theta(u, v, N_DIV_X, N_DIV_Y, delta_x, delta_y, DELTA_T, theta)
    p = calcular_pressao(u_star, v_star, N_DIV_X, N_DIV_Y, delta_x, delta_y, DELTA_T, 1e-5, p)
    u = calcular_u(u_star, p, N_DIV_X, N_DIV_Y, delta_x, DELTA_T, u)
    v = calcular_v(v_star, p, N_DIV_X, N_DIV_Y, delta_y, DELTA_T, v)

    ''' SALVAR ALGUNS INSTANTES DE TEMPO '''
    if k % intervalo_captura == 0:

        frame_atual += 1

        uplot, vplot, pplot, thetaplot = interpolar_malha(u, v, p, theta, N_DIV_X, N_DIV_Y, uplot, vplot, pplot, thetaplot)
        frames_u[frame_atual] = uplot
        frames_v[frame_atual] = vplot
        frames_p[frame_atual] = pplot
        frames_theta[frame_atual] = thetaplot

        # print(f'tempo={indices_tempo_selecionados[frame_atual]*DELTA_T}')

end = timeit.default_timer()
duracao = end - start
print(f'\nDuração de {duracao:.5f} s \n')

''''''''''''''''''''''''''''''''''''
''' SALVAR DADOS PARA O POS-PROCESAMENTO '''

np.savez(f'{CAMINHO}numero_rayleigh.npz', rayleigh=RAYLEIGH)
np.savez(f'{CAMINHO}malha.npz', x=x, y=y, t=t)
np.savez(f'{CAMINHO}frames.npz',
         indices_tempo_selecionados=indices_tempo_selecionados,
         frames_u=frames_u, frames_v=frames_v, frames_p=frames_p,
         frames_theta=frames_theta)
