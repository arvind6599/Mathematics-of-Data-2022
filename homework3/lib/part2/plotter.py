# Plotting function
import matplotlib.pyplot as plt
import numpy as np

def plot_func(cur_iter, feasibility1,feasibility2, objective, X, opt_val):
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.loglog(cur_iter, feasibility1)#, 'go--', linewidth=2, markersize=12))
    plt.xlabel('iteration',fontsize=15)
    plt.ylabel('$\|X1-1\|/1$',fontsize=15)
    plt.grid(True)

    plt.subplot(222)
    plt.loglog(cur_iter, feasibility2)
    plt.xlabel('iteration',fontsize=15)
    plt.ylabel('dist$(X, \mathbb{R}^{n}_+)$',fontsize=15)
    plt.grid(True)
    plt.show()

    #plt.subplot(223)
    obj_res = np.reshape(np.abs(objective - opt_val)/opt_val, (len(objective),))
    plt.figure(figsize=(12, 8))
    plt.loglog((cur_iter), (obj_res))
    plt.xlabel('iteration',fontsize=15)
    plt.ylabel('$(f(X) - f^*)/f^*$',fontsize=15)
    plt.title('Relative objective residual',fontsize=15)
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(X)
    plt.title('SDP solution',fontsize=15)
    plt.colorbar()
    plt.show()
    
def plot_comp(it, f1,f2, obj, opt_val):
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.loglog(it[0], f1[0], label='PDHG')#, 'go--', linewidth=2, markersize=12))
    plt.loglog(it[1], f1[1], label='HCGM')#, 'go--', linewidth=2, markersize=12))
    plt.xlabel('t (sec)',fontsize=15)
    plt.ylabel('$\|X1-1\|/1$',fontsize=15)
    plt.grid(True)
    

    plt.subplot(222)
    plt.loglog(it[0], f2[0], label='PDHG' )
    plt.loglog(it[1], f2[1], label='HCGM' )
    plt.xlabel('t (sec)',fontsize=15)
    plt.ylabel('dist$(X, \mathbb{R}^{n}_+)$',fontsize=15)
    plt.grid(True)
#     plt.legend()
#     plt.savefig('feas_cluster.pdf')
    plt.show()

    #plt.subplot(223)
    obj_res_PDHG = np.reshape(np.abs(obj[0] - opt_val)/opt_val, (len(obj[0]),))
    obj_res_HCGM = np.reshape(np.abs(obj[1] - opt_val)/opt_val, (len(obj[1]),))
    plt.figure(figsize=(12, 8))
    plt.loglog((it[0]), (obj_res_PDHG), label='PDHG')
    plt.loglog((it[1]), (obj_res_HCGM), label='HCGM')
    plt.xlabel('t (sec)',fontsize=15)
    plt.ylabel('$(f(X) - f^*)/f^*$',fontsize=15)
    plt.title('Relative objective residual',fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)
#     plt.savefig('obj_cluster.pdf')
    plt.show()
    
    