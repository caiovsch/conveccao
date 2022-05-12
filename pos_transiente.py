import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

''''''''''''''''''''''''''''''''''''''''''
''' ABRIR OS DADOS DO PROCESSAMENTO '''

CAMINHO = "./resultados/"

numero_rayleigh = np.load(f'{CAMINHO}numero_rayleigh.npz')
rayleigh = numero_rayleigh['rayleigh']

malha = np.load(f'{CAMINHO}malha.npz')
x = malha['x']
y = malha['y']
t = malha['t']

id = f'{rayleigh:.0f}'

frames = np.load(f'{CAMINHO}frames.npz')
indices_tempo_selecionados = frames['indices_tempo_selecionados']
frames_u = frames['frames_u']
frames_v = frames['frames_v']
frames_p = frames['frames_p']
frames_theta = frames['frames_theta']

frames_modulo_velocidade = (frames_u**2 + frames_v**2)**0.5

dt = t[1] - t[0]
xx, yy = np.meshgrid(x, y)


def mapa_filme(k):

    plt.clf()

    # plt.contour(xx, yy, np.transpose(frames_para_animar[k]), levels=20, colors='black', linestyles='solid', linewidths=1)
    plt.pcolormesh(xx, yy, np.transpose(frames_para_animar[k]), cmap=plt.cm.jet)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    cb.ax.tick_params(labelsize=12)

    plt.title(f'{titulo}, Ra={rayleigh:.0f}, t={dt*indices_tempo_selecionados[k]:.3f}', fontsize=12)
    plt.xlabel('$x$', fontsize=12)
    plt.ylabel('$y$', fontsize=12)
    plt.axis('scaled')
    plt.xticks(fontsize=12), plt.yticks(fontsize=12)

    return plt


''' GIF DA TEMPERATURA '''
titulo = 'Temperatura'
frames_para_animar = frames_theta
anim = animation.FuncAnimation(plt.figure(), mapa_filme, interval=1, frames=len(frames_theta), repeat=False)
anim.save(f'{CAMINHO}{id}temperatura.gif', writer='pillow', fps=15)


''' GIF DA VELOCIDADE '''
titulo = 'Velocidade'
frames_para_animar = frames_modulo_velocidade
anim = animation.FuncAnimation(plt.figure(), mapa_filme, interval=1, frames=len(frames_modulo_velocidade), repeat=False)
anim.save(f'{CAMINHO}{id}velocidade.gif', writer='pillow', fps=15)

''' GIF DA PRESSAO '''
titulo = 'Pressão'
frames_para_animar = frames_p
anim = animation.FuncAnimation(plt.figure(), mapa_filme, interval=1, frames=len(frames_p), repeat=False)
anim.save(f'{CAMINHO}{id}pressao.gif', writer='pillow', fps=15)

# plt.quiver(x, y, np.transpose(frames_u[k]), np.transpose(frames_v[k]))
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
