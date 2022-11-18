import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def arrow(u,v): 
    arrow_line = [[a,b] for a, b in zip(u,v)]
    return arrow_line


def plot_vectors_3d(vectors: np.array, origin=[0,0,0], plot_name='Vectors', ):

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(projection='3d')
    ax.set_title(plot_name, fontsize=15)

    max_dim1 = np.max(vectors, 0)[0]
    max_dim2 = np.max(vectors, 0)[1]
    max_dim3 = np.max(vectors, 0)[2]

    #l = max_dim
    #l_axis = np.arange(-l,l+1)
    #ax.view_init(15, 25, 'z')

    # ADD GRID MARKERS
    #for i in l_axis:
    #    for j in l_axis:
    #        for k in l_axis:
    #            ax.scatter(i,j,k, color='gray', marker='.', s=25)
    ax.scatter(*origin, color='crimson', marker='+', s=100) # grid

    for v in vectors:
        plt.quiver(*origin, *v, label=f'{v}')


    ax.set_xlim([-max_dim1, max_dim1+1])
    ax.set_ylim([-max_dim2, max_dim2+1])
    ax.set_zlim([-max_dim3, max_dim3+1])

    ax.set_xticks(np.arange(-max_dim1,max_dim1+1))
    ax.set_yticks(np.arange(-max_dim2,max_dim2+1))
    ax.set_zticks(np.arange(-max_dim3,max_dim3+1))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    plt.show()



def plot_2d_arrow(vectors: np.array, origin=[0,0], plot_name='Vectors' ):

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.set_title(plot_name, fontsize=15)

    # TODO: ASK TO PLOT ORIGIN
    ax.scatter(*origin, color='crimson', marker='+', s=100) # grid

    for v in vectors:
        ax.scatter(*v, color='crimson', marker='.', s=10, alpha=0.1) # grid
        plt.quiver(*origin, *v, color='gray', angles='xy', scale_units='xy', scale=1, label=f'{v.round(2)}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.legend()
    plt.show()



def plot_2d_arrow_color(vectors: np.array, origin=None, plot_name='Vectors' ):

    zero = np.zeros(vectors.shape) # DIFFERENT SHAPA THAN ORIGIN
    dim = vectors.shape[0]

    colors = np.arctan2(vectors[:,0], vectors[:,1])
    norm = Normalize()
    norm.autoscale(colors)
    colormap = cm.tab20b

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.set_title(plot_name, fontsize=15)
    # TODO: ASK TO PLOT ORIGIN
    ax.scatter([0,0],[0,0], color='crimson', marker='+', s=100) # ORIGIN CROSS

    ax.scatter(vectors[:,0],vectors[:,1], color='crimson', marker='.', s=10, alpha=0.1)
    plt.quiver(zero[:,0], zero[:,1], vectors[:,0],vectors[:,1], color=colormap(norm(colors)),
            angles='xy', scale_units='xy', scale=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # TODO: DIFFERENT LABELS PER VECTOR
    #plt.legend() 
    plt.show()



def plot_2d_line(vectors: np.array, origin=[0,0], plot_name='Vectors'):

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.set_title(plot_name, fontsize=15)

    # TODO: ASK TO PLOT ORIGIN
    ax.scatter(*origin, color='crimson', marker='+', s=100) # grid

    for v in vectors:
        ax.scatter(*v, color='crimson', marker='.', s=100, label=f'{v.round(2)}') # grid
        ax.plot(*arrow(origin,v), '--', color='gray') # (u1,v1), (u2,v2), (u3,v3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.legend()
    plt.show()