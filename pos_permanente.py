import numpy as np
from matplotlib import pyplot as plt
from numba import njit

''''''''''''''''''''''''''''''''''''''''''
''' ABRIR OS DADOS DO PROCESSAMENTO '''

CAMINHO = "./resultados/"

numero_rayleigh = np.load(f'{CAMINHO}numero_rayleigh.npz')
rayleigh = numero_rayleigh['rayleigh']

malha = np.load(f'{CAMINHO}malha.npz')
x = malha['x']
y = malha['y']
t = malha['t']

frames = np.load(f'{CAMINHO}frames.npz')
indices_tempo_selecionados = frames['indices_tempo_selecionados']
frames_u = frames['frames_u']
frames_v = frames['frames_v']
frames_p = frames['frames_p']
frames_theta = frames['frames_theta']

frames_modulo_velocidade = (frames_u**2 + frames_v**2)**0.5

id = f'{rayleigh:.0f}'

N_x = len(x) - 1
N_y = len(y) - 1
dx = x[1] - x[0]
dy = y[1] - y[0]
dt = t[1] - t[0]

''''''''''''''''''''''''''''''''''''''''''
''' VARIAVEIS DO REGIME PERMANENTE '''

tempo = t[-1]
u = np.transpose(frames_u[-1])
v = np.transpose(frames_v[-1])
p = np.transpose(frames_p[-1])
theta = np.transpose(frames_theta[-1])

modulo_velocidade = (u**2 + v**2)**0.5


@njit
def calcular_velocidade_unitaria(u, v, modulo_velocidade, u_unitario, v_unitario):
    for i in range(0, len(x) - 1):
        for j in range(0, len(y) - 1):
            if modulo_velocidade[i, j] > 0.0:
                u_unitario[i, j] = u[i, j] / modulo_velocidade[i, j]
                v_unitario[i, j] = v[i, j] / modulo_velocidade[i, j]
    return u_unitario, v_unitario


@njit
def calcular_funcao_corrente(u, v, N_x, N_y, dx, dy, dt, tol, psi):
    lamda = -(2/dx**2+2/dy**2)
    erro = 100
    while erro > tol:
        R_max = 0
        for i in range(1, N_x):
            for j in range(1, N_y):
                R = -(v[i, j] - v[i-1, j])/dx + (u[i, j]-u[i, j-1])/dy - \
                    ((psi[i+1, j] - 2*psi[i, j] + psi[i-1, j])/dx**2 + (psi[i, j+1] - 2*psi[i, j] + psi[i, j-1])/dy**2)
                R = R/lamda
                psi[i, j] = psi[i, j] + R
                if abs(R) > R_max:
                    R_max = abs(R)
        erro = R_max
    return psi


def calcular_nusselt_medio_x0(y, delta_y, delta_x, theta):
    nusselt_medio_x0 = 0
    for j in range(len(y)-1):
        nusselt_medio_x0 += -(theta[1, j] - theta[0, j] + theta[1, j + 1] - theta[0, j + 1]) / delta_x * delta_y / 2
    return nusselt_medio_x0


u_unitario = np.zeros((N_x + 1, N_y + 1))
v_unitario = np.zeros((N_x + 1, N_y + 1))
u_unitario, v_unitario = calcular_velocidade_unitaria(u, v, modulo_velocidade, u_unitario, v_unitario)

psi = np.zeros_like(frames_u[-1])
psi = calcular_funcao_corrente(frames_u[-1], frames_v[-1], N_x, N_y, dx, dy, dt, 1e-8, psi)

''''''''''''''''''''''''''''''''''''''''''
''' GRAFICOS DO REGIME PERMANENTE '''

plt.style.use(['science', 'notebook'])
xx, yy = np.meshgrid(x, y)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

''' NUMERO DE NUSSELT '''
frames_nusselt = []
for i in range(len(indices_tempo_selecionados)):
    frames_nusselt.append(calcular_nusselt_medio_x0(y, dy, dx, frames_theta[i]))

grafico_nusselt = plt.subplots(figsize=(3.3, 3.0))
plt.plot(indices_tempo_selecionados[1:] * dt, frames_nusselt[1:], 'b.-')

plt.title(f'Ra={rayleigh:.0f}', fontsize=8)
plt.xlabel('$t$', fontsize=11)
plt.ylabel('Nusselt médio', fontsize=11)
plt.xticks(fontsize=11), plt.yticks(fontsize=11)
plt.grid(linestyle='--')

plt.savefig(f'{CAMINHO}{id}nusselt.png', format='png', dpi=600, bbox_inches='tight')
plt.show()

''' ISOTERMAS '''
grafico_temperatura = plt.subplots(figsize=(3.2, 3.2))
plt.contour(xx, yy, theta, levels=20, colors='black', linestyles='solid', linewidths=1)
plt.contourf(xx, yy, theta, levels=20, cmap=plt.cm.hot)
# plt.pcolormesh(xx, yy, theta, cmap=plt.cm.hot)
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)

plt.axis('scaled')
plt.title(f'Temperatura, t={tempo:.2f}, Ra={rayleigh:.0f}', fontsize=7)
plt.tick_params(axis='y', labelleft=False)
plt.tick_params(axis='x', labelbottom=False)

plt.savefig(f'{CAMINHO}{id}temperatura.png', format='png', dpi=600, bbox_inches='tight')
plt.show()

''' VETORES VELOCIDADE 
Plota-se o vetor unitario (direcao) da velocidade
'''
grafico_velocidade = plt.subplots(figsize=(3.2, 3.2))

# plt.pcolormesh(xx, yy, modulo_velocidade, cmap=plt.cm.BuGn)
# plt.pcolormesh(xx, yy, modulo_velocidade, cmap=plt.cm.viridis)
plt.contourf(xx, yy, modulo_velocidade, levels=30, cmap=plt.cm.viridis)
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)

step = 3
plt.quiver(x[::step], y[::step], u_unitario[::step, ::step], v_unitario[::step, ::step], units='xy')
plt.axis('scaled')
plt.title(f'Velocidade, t={tempo:.2f}, Ra={rayleigh:.0f}', fontsize=8)
plt.tick_params(axis='y', labelleft=False)
plt.tick_params(axis='x', labelbottom=False)

plt.savefig(f'{CAMINHO}{id}vetores_velocidade.png', format='png', dpi=600, bbox_inches='tight')
plt.show()

''' LINHAS DE CORRENTE '''
grafico_corrente = plt.subplots(figsize=(3.5, 3.5))
contour = plt.contour(xx, yy, np.transpose(psi), levels=20, colors='black', linestyles='solid', linewidths=1)

plt.axis('scaled')
plt.title(f't={tempo:.2f}, Ra={rayleigh:.0f}', fontsize=9)
plt.tick_params(axis='y', labelleft=False)
plt.tick_params(axis='x', labelbottom=False)

plt.savefig(f'{CAMINHO}{id}corrente.png', format='png', dpi=600, bbox_inches='tight')
plt.show()

''' CONTORNOS DE PRESSAO '''
grafico_pressao = plt.subplots(figsize=(3.2, 3.2))
plt.contour(xx, yy, p, levels=20, colors='black', linestyles='solid', linewidths=1)
plt.contourf(xx, yy, p, levels=20, cmap=plt.cm.rainbow)
# plt.pcolormesh(xx, yy, p, cmap=plt.cm.rainbow)
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)
plt.axis('scaled')
plt.title(f'Pressão, t={tempo:.2f}, Ra={rayleigh:.0f}', fontsize=7)
plt.tick_params(axis='y', labelleft=False)
plt.tick_params(axis='x', labelbottom=False)

plt.savefig(f'{CAMINHO}{id}pressao.png', format='png', dpi=600, bbox_inches='tight')
plt.show()

''' PARAMETROS '''
print(f'Passo da malha = {dx}')
print(f'Nusselt médio final = {frames_nusselt[-1]}')
print(f'Valor máximo da componente u = {np.max(frames_u[-1])}')
print(f'Valor máximo da componente v = {np.max(frames_v[-1])}')
