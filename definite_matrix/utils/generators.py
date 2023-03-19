import numpy as np
import pandas as pd

from scipy.linalg import fractional_matrix_power
from scipy.linalg import logm, expm


def identity_matrix():
    I = np.array([[1,0],
                 [0,1]])
    
    return I

def genrate_matrix(r=None):
    if r == None:
        r = np.random.rand()
        s = np.random.binomial(1, 0.5)
        if s:
            r = -r

    I = np.array([[1,0],
                 [0,1]])
    B = np.array([[0,1],
                 [1,0]]) 
    M = I + (B*r)
    
    return M

def get_field(A):
    x, y ,z = A[0][0], A[1][0], A[1][1]
    
    return x,y,z

def get_str(A):
    x, y ,z = A[0][0], np.round(A[1][0],2), A[1][1]
    
    return ','.join([str(s) for s in [x,y,z]])


def surface_points(size=10, limit=1):
    x = np.linspace(0, limit, size)
    z = np.linspace(0, limit, size)
    
    xs = []
    zs = []
    yas = []
    ybs = []
    for i in x:
        for j in z:
            a = np.sqrt(i*j)
            b = -1*np.sqrt(i*j)
            xs.append(i)
            zs.append(j)
            yas.append(a)
            ybs.append(b)
            
    return xs,yas,ybs,zs

def mesh_surface(size=10, limit=1):
    x = np.linspace(0, limit, size)
    z = np.linspace(0, limit, size)
    x, z = np.meshgrid(x, z)

    y1 = np.sqrt(x*z)
    y2 = -np.sqrt(x*z)

    return x,y1,y2,z


def upper_vectorization(A:np.array):
    triangle = np.triu(A, 1)
    vector = triangle[np.triu_indices(triangle.shape[0],1)]

    return vector


def geodesic_k(A, B):

    q = np.linalg.solve(A,B)
    e, _ = np.linalg.eig(q)
    dist = np.sqrt(np.sum(np.log(e)**2))

    return dist

def geodesic_distance(a,b):
    
    a_ = fractional_matrix_power(a, -0.5)
    q = a_@ b @a_
    q = logm(q)
    trace = np.trace(q**2)
    dist = np.sqrt(trace)
    
    return dist



def distances_two(A,B):
    
    va, vb = upper_vectorization(A), upper_vectorization(B)

    ed_ab = np.round(np.sqrt( (va-vb)**2 ), 2)[0]

    gdk_ab = np.round(geodesic_k(A,B), 2)

    gd_ab = np.round(geodesic_distance(A,B), 4)
    
    df = pd.DataFrame({'matrix':['AB'],
                       'euclidean':[ed_ab],
                       'geodesick':[gdk_ab],
                       #'geodesicd':[gd_ab],
                      })
                    
    print(A)
    print(df)

    
def distances_3(A,B,X):
    
    va, vb, vx = upper_vectorization(A), upper_vectorization(B), upper_vectorization(X)
    
    ed_ax = np.round(np.sqrt( (va-vx)**2 ), 2)[0]
    ed_xb = np.round(np.sqrt( (vx-vb)**2 ), 2)[0]
    ed_ab = np.round(np.sqrt( (va-vb)**2 ), 2)[0]
    
    gd_ax = np.round(geodesic_k(A,X), 2)
    gd_xb = np.round(geodesic_k(X,B), 2)
    gd_ab = np.round(geodesic_k(A,B), 2)
    
    df = pd.DataFrame({'matrix':['AX','XB','AB'],
                       'euclidean':[ed_ax,ed_xb,ed_ab],
                       'geodesic':[gd_ax,gd_xb,gd_ab],
                      })
                    
    
    print(df)

