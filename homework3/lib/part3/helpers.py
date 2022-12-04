import time 
from math import sqrt
import copy
import itertools

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from .projL1 import projL1


def projNuc(Z, kappa):
    #PROJNUC This function implements the projection onto nuclear norm ball.
    
    u,s,v = np.linalg.svd(Z, full_matrices=0)
    s = projL1(s,kappa)
    
    return u@np.diag(s)@v
    
    
## FUNCTIONS FOR THE SDP ROUNDING
def remap_centers(assign,k):
    class_vec = np.zeros([10,10])
    max_class = np.zeros([10,2])
    remap_vec = -1*np.ones([10])
    
    # Computing the number of assignments per class (the classes go 0-99, 100-199,...) 
    for l in range(10):
        class_loc = assign[100*(l):100*(l+1)-1]
        for i in range(10):
            class_vec[l,i] = sum(class_loc==i).item()
        max_class[l] = [np.max(class_vec[l,:]), np.argmax(class_vec[l,:])]

    # Remapping the cluster with the largest number of elements to the actual one iteratively
    # Plotting the evolution of class_vec, pos_remap and max_class helps to understand it 
    # easily ;)
    for l in range(10):  
        pos_remap = np.argmax(max_class[:,0])
        remap_vec[int(max_class[pos_remap,1])] = pos_remap
        class_vec[:,int(max_class[pos_remap,1])] = 0
        class_vec[pos_remap,:] = 0
        for i in range(10):
            max_class[i] = [np.max(class_vec[i,:]), np.argmax(class_vec[i,:])]

    return remap_vec

def sdp_rounding(X, k, digits):
    X = digits.dot(X)
    
    N=X.shape[1]
    # computation of an affinity matrix identifying repeated denoised points
    affinity=np.zeros([N,N]);
    for i in range(N): 
        for j in range(N):
            if np.linalg.norm(X[:,i]-X[:,j])<1e-3:
                affinity[i,j]=1;
                affinity[j,i]=1;
    # centers are k most popular points
    centers=np.zeros([k,k])
    for t in range(k):
        s = np.sum(affinity,0)
        idx = np.argmax(s)

        aux = copy.deepcopy(affinity[:,idx])
        centers[t,:]=X[:, idx]
        for i in range(N): 
            if aux[i]==1:
                affinity[i,:]=0
                affinity[:,i]=0
        
    # assignment of points to closest center
    ind=np.zeros([N,1]);
    for i in range(N):
        aux=np.zeros([k,1]);
        for t in range(k):
            aux[t,0]=np.linalg.norm(X[:,i].T - centers[t,:],2);
        ind[i,0]= np.argmin(aux);
    assignment=ind
    
    # remapping to the correct clusters, i.e. first cluster should be 0, ...
    assignment_remap = np.zeros([N,1]);
    remap_vec = remap_centers(assignment,k)
    centers_remap =np.zeros([k,k]); 
    for i in range(N):
            assignment_remap[i] = remap_vec[int(assignment[i])]
    for loc, map_ in enumerate(remap_vec):
        centers_remap[loc,:] = centers[int(map_),:]
    
    return centers_remap, assignment_remap

def misclassification_rate(assignment, labels):
    labels = labels-1
    return np.sum(assignment!=labels)/len(assignment)

def vis_samples(assignment, images, labels):
    assignment=assignment.astype(int)
    classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    labels = labels-1
    rand_samp = np.random.randint(0,1000,25)
    plt.figure(figsize=(7,7))
    for i,samp in enumerate(rand_samp):
        plt.subplot(5,5,i+1)
        plt.imshow(1-np.reshape(images[samp],[28,28]), cmap=plt.cm.gray)
        plt.title('Pred. {0}\n Orig. {1}'.format(classes[assignment[samp].item()],classes[labels[samp].item()]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def value_kmeans(points, labels):
    # This function computes the kmeans value (sum of squared
    # mean deviations from cluster centroid) of a 
    # provided partition of a provided set of points.
    
    k = 10
    points = points.T
    idxx = np.argsort(labels.T)
    idxx = np.squeeze(idxx)
    points = points[idxx,:]
            
    count = np.zeros([k,], int)
    
    for i in range(k):
        count[i] = np.int(np.sum(labels==(i)))
        
    idx = 0
    value = 0
    
    for t in range(k):
        cluster = points[idx:(idx+count[t]),:]
        center = np.matmul(np.ones([1, cluster.shape[0]]),cluster)/count[t]
        for i in range(count[t]):
            value = value + np.linalg.norm(cluster[i,:] - center)**2
        idx = (idx + count[t])
        
    return value


# Plotting function
def plot_func(cur_iter, feasibility1,feasibility2, objective, X, X_true, args):
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.loglog(cur_iter, feasibility1)#, 'go--', linewidth=2, markersize=12))
    plt.xlabel('iteration',fontsize=15)
    plt.ylabel('$\|A(X) - b\|$',fontsize=15)
    plt.grid(True)

    plt.subplot(222)
    plt.loglog(cur_iter, feasibility2)
    plt.xlabel('iteration',fontsize=15)
    plt.ylabel('dist$(B(X), (-\infty, 0])$',fontsize=15)
    plt.grid(True)
    plt.show()

    #plt.subplot(223)
    obj_res = np.reshape(np.abs(objective - args.opt_val)/args.opt_val, (len(objective),))
    plt.figure(figsize=(12, 8))
    plt.loglog((cur_iter), (obj_res))
    plt.xlabel('iteration',fontsize=15)
    plt.ylabel('$(f(X) - f^*)/f^*$',fontsize=15)
    plt.title('Relative objective residual',fontsize=15)
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.imshow(X)
    plt.title('HCGM solution',fontsize=15)
    plt.colorbar()
    
    plt.subplot(222)
    plt.imshow(X_true)
    plt.title('True solution',fontsize=15)
    plt.colorbar()
    plt.show()


def grad_F(X, beta, args):
    nchoosek_inds = list(itertools.combinations(range(args.p), 3))

    # Add the equality constraint
    [grad_val, displacement] = get_equality_constr_grad(X, beta, args)
    feas_eq = abs(displacement)
   
    # Add the triangle constraints. This needs to be scaled by d to match
    # the order of the equality constr
    n = len(nchoosek_inds)
    feas_ineq = 0
    for l in range(n):
        index_triplet = nchoosek_inds[l]
        [update_vector, row_idxs, col_idxs, part_feas_normsq] = get_triangle_constr_grad(X, beta, index_triplet)
        
        grad_val[row_idxs, col_idxs] = grad_val[row_idxs, col_idxs] + update_vector
        grad_val[col_idxs, row_idxs] = grad_val[col_idxs, row_idxs] + update_vector
        
        feas_ineq = feas_ineq + part_feas_normsq
    
    # Total gradient
    grad_val = grad_val + args.C
    feas_ineq = sqrt(feas_ineq)
    
    return (grad_val, feas_eq, feas_ineq)



def get_equality_constr_grad(X, beta, args):
    # X is a p x p matrix
    displacement = args.p * np.trace(X) - np.sum(X) - args.p**2/2
    grad_val = (args.p/beta) * (args.p * np.eye(args.p) - np.ones((args.p, args.p))) * displacement
    
    return (grad_val, displacement)



def get_triangle_constr_grad(X, beta, index_triplet):
    # X needs to be in a matrix shape, not vectorized
    #print(index_triplet)
    i = index_triplet[0]
    j = index_triplet[1]
    k = index_triplet[2]
    t1 = max(X[i, j] + X[j, k] - X[i, k] - X[j, j], 0.)
    t2 = max(X[i, k] + X[i, j] - X[j, k] - X[i, i], 0.)
    t3 = max(X[i, k] + X[j, k] - X[i, j] - X[k, k], 0.)

    grad_update = (1. / beta) * np.array([t1 + t2 - t3, # for Xij 
                                -t1 + t2 + t3, # for Xik
                                t1 - t2 + t3, # for Xjk
                                -t2/2., # for Xii - ./2 because we update these twice when we symmetrize
                                -t1/2., # for Xjj - ./2 because we update these twice when we symmetrize
                                -t3/2.]) # for Xkk - ./2 because we update these twice when we symmetrize

    update_row_idxs = [i, i, j, i, j, k]
    update_col_idxs = [j, k, k, i, j, k]
    part_feas_normsq = 2. * (t1**2 + t2**2 + t3**2)
    
    return (grad_update, update_row_idxs, update_col_idxs, part_feas_normsq)
