# plotting
import matplotlib.pyplot as plt
# general math and science operations
import numpy as np
import scipy

# loading the data 
import scipy.io 
from scipy.sparse import csr_matrix 
from scipy.sparse import linalg

 # timing
from time import time
 # pretty progress bars
from tqdm.notebook import trange

import os
import numpy as np
from scipy.fftpack import fft2, fftshift, ifft2
import matplotlib.pyplot as plt
import scipy.sparse as sps
#from scipy.misc import imread
import cv2 # if imread does not work for you.
# N.B. You can install cv2 by calling 'conda install opencv' or 'pip install opencv' in the terminal
# If you have incompatibilities with hdf5, you can uninstall it by command line 'conda remove hdf5' or 
# 'pip remove hdf5' and then install opencv
from scipy.sparse.linalg import LinearOperator, svds
from pywt import wavedec2, waverec2, coeffs_to_array, array_to_coeffs
from collections import namedtuple
from math import sqrt

from .opt_types import *


# Mappings between n dimensional complex space and 2n dimensional real space
real2comp = lambda x: x[0:x.shape[0]//2] + 1j*x[x.shape[0]//2:]
comp2real = lambda x: np.append(x.real, x.imag)

# Load Data
#x = imread('blurredplate.jpg',flatten=True,mode='F')
BLUR_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),'blurredplatewatermark.jpg')
x = cv2.imread(BLUR_PATH , cv2.IMREAD_GRAYSCALE) #If imread does not work for you
# x = cv2.imread('blurredplate.jpg' , cv2.IMREAD_GRAYSCALE) #If imread does not work for you


x = x[60:188,40:296]
x = x/np.linalg.norm(x,ord=np.inf)
#x = x/np.linalg.norm(x)
imsize1 = x.shape[0]
imsize2 = x.shape[1]
imsize = imsize1*imsize2

_ImgShow=x

# Reshaping operators matrix to vector and vector to matrix
mat = lambda x: np.reshape(x,[imsize1,imsize2])
vec = lambda x: x.flatten()

# Set the measurement vector b
b = comp2real(vec(fft2(fftshift(x))))

# Roughly estimate the support of the blur kernel
K1 = 17
K2 = 17
Indw = np.zeros([imsize1,imsize2])
ind1 = np.int(imsize1/2-(K1+1)/2+1)
ind2 = np.int(imsize1/2+(K1+1)/2)
ind3 = np.int(imsize2/2-(K2+1)/2+1)
ind4 = np.int(imsize2/2+(K2+1)/2)
Indw[ind1:ind2,ind3:ind4] = 1
#above, for implementational simplicity we assume K1 and K2 odd, even  
#if they are even 1 pixel probably won't cause much trouble
_IndImg=-Indw
def setup_show():
    fig,ax=plt.subplots(ncols=2,figsize=[20,10])
    ax[0].imshow(x, cmap='gray')
    ax[0].set_title("Blurred image")
    ax[1].imshow(_IndImg, cmap='gray') # Shows the estimated support of blur kernel!
    ax[1].set_title("Roughly estimated blur kernel support")
    for a in ax:
        a.grid(False)
        a.set_xticks([])
        a.set_yticks([])

Indw = vec(Indw);
kernelsize = np.count_nonzero(Indw)
Indi = np.nonzero(Indw > 0)[0]
Indv = Indw[Indi]

# Define operators Bop and Cop
Bmat = sps.csr_matrix((Indv,(Indi,range(0,kernelsize))),shape=(imsize,kernelsize))
Bop = lambda x: mat(Bmat.dot(x))
BTop = lambda x: Bmat.T.dot(vec(x))

# Compute and display wavelet coefficients of the original and blurred image
l = coeffs_to_array(wavedec2(x, 'db1', level=4))[1]
Cop = lambda x: waverec2(array_to_coeffs(mat(x),l,output_format='wavedec2'),'db1')
CTop = lambda x: coeffs_to_array(wavedec2(x, 'db1', level=4))[0]

# Define operators
Aoper = lambda m,n,h: comp2real(1.0/sqrt(imsize)*n*vec(fft2(Cop(m))*fft2(Bop(h))))
AToper = {"matvec": lambda y,w: CTop(np.real(fft2(mat(np.conj(real2comp(y))*vec(fft2(Bop(w)))))))/sqrt(y.shape[0]/2.0),
         "rmatvec": lambda y,w: BTop(np.real(ifft2(mat(real2comp(y)*vec(ifft2(Cop(w)))))))*(y.shape[0]/2.0)**1.5}


def A_T(X):
    ATop1 = lambda w: AToper["matvec"](X,w)
    ATop2 = lambda w: AToper["rmatvec"](X,w)
    return LinearOperator((imsize,kernelsize), matvec=ATop1, rmatvec=ATop2)


def plot_func(mEst,i, C, x):
    xEst = -C(mEst);
    xEst = xEst - min(xEst.flatten())
    xEst = xEst/max(xEst.flatten())
    plt.imshow(xEst, cmap='gray')#,hold=False)
    plt.grid(False)
    plt.axis('off')
    plt.title(f"Iteration {i}")
    plt.show()


#######################################################
# 1.3 Comparing Prox and LMO
#######################################################


def eval_completion(dataset,proj_func,num_times=5,kappa=5000):
    """
    This helper functions loads the data for you, arranges it into the suitable vector form and then
    runs the timing on the provided projection or lmo
    """
    datasets = {
        "100k_MovieLens": './lib/part1/dataset/ml-100k/ub_base',
        "1M_MovieLens": './lib/part1/dataset/ml-1m/ml1m_base',
    }

    if dataset not in datasets.keys():
        raise ValueError("`dataset` needs to be one of: {datasets.keys()}")

    data = scipy.io.loadmat(datasets[dataset])  # load 100k dataset
    Rating = data['Rating'].flatten()
    UserID = data['UserID'].flatten() - 1  # Python indexing starts from 0 whereas Matlab from 1
    MovID = data['MovID'].flatten() - 1    # Python indexing starts from 0 whereas Matlab from 1

    nM = np.amax(data['MovID'])
    nU = np.amax(data['UserID'])
    total = 0
    Z = csr_matrix((Rating, (MovID, UserID)),shape=(nM, nU),dtype=float).toarray()

    for _ in trange(num_times):
        tstart = time()
        Z_proj = proj_func(Z, kappa)
        elapsed = time() - tstart
        total+=elapsed/num_times

    print('proj for {} data takes {} sec'.format(datasets[dataset], total))


#######################################################
# 1.4 Running Frank-Wolfe
#######################################################


plotF = lambda m,it: plot_func(m,it,Cop,x)

def run_frank_wolfe(f, opt_algorithm, maxit=200):    
    # keep track of objective value
    fx = np.array([])

    state = opt_algorithm.init_state()
    
    # The main loop    
    bar = trange(0, maxit+1)
    for _ in bar:
        
        # Print the objective values ...
        fx = np.append(fx, f(state.AX))
        bar.set_description('{:03d} | {:.4e}'.format(state.k, fx[-1]))

        state = opt_algorithm.state_update(f, state)
        
        # Show the reconstruction (at every 10 iteration) 
        if state.k%10 == 0:
            U,S,V = np.linalg.svd(state.X, full_matrices=0, compute_uv=1)
            plotF(U[:,0], state.k)

    return state


# NOTE: This experiment is based on the theory and the codes publised in
#'Blind Deconvolution using Convex Programming' by A.Ahmed, B.Recht and J.Romberg.
