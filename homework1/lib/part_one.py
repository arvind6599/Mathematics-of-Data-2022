from lib.opt_types import *
from lib.utils import draw_court
import numpy as np
import matplotlib.pyplot as plt


mu = 0.001



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def create_loss(A, b):
    def logistic(x):
        return np.average(np.log(1 + np.exp(-b * np.dot(A, x)))) + 0.5 * mu * np.dot(x, x)


    def grad_logistic(x):
        return 1/len(A)*np.dot(A.T, (-b * sigmoid(-b * np.dot(A, x)))) + mu * x

    def ith_grad(i, x):
        return  -b[i]*sigmoid(-b[i]*np.dot(A[i], x))*A[i] + mu*x
    
    return logistic, grad_logistic, ith_grad


A = np.load("lib/data/A.npy")
b = np.load("lib/data/b.npy")
logistic, grad_logistic, ith_grad = create_loss(A, b)
x_zero = np.zeros(A.shape[1])

f = Function(
    f=logistic,
    grad=grad_logistic,
    i_grad=ith_grad,
    minimum=0.6805235000445283,
    strng_cvx=mu,
    lips_grad=1/len(A)* (np.linalg.norm(A, ord="fro") ** 2) + mu,
    n = len(A),
    L_max = max([np.linalg.norm(A[i])**2 for i in range(len(A))]) + mu
)

A_full = np.load("lib/data/A_full.npy")
b_full = np.load("lib/data/b_full.npy")
logistic_full, grad_logistic_full, ith_grad_full = create_loss(A_full, b_full)

f_full_data = Function(
    f=logistic_full,
    grad=grad_logistic_full,
    i_grad=ith_grad_full,
    minimum=0.6703870057953005,
    strng_cvx=mu,
    lips_grad=1/len(A_full)* (np.linalg.norm(A_full, ord="fro") ** 2) + mu,
    n = len(A_full),
    L_max = max([np.linalg.norm(A_full[i])**2 for i in range(len(A_full))]) + mu
)

def train_accuracy(x: Vector, all=False):
    data = A
    labels = b
    if all == True:
        data = A_full
        labels = b_full
    preds = 2*(sigmoid(np.dot(A, x)) > 0.5) - 1

    plt.style.use("bmh")
    
    _, (bx, ax) = plt.subplots(1, 2, figsize=(15,6))
    bx.text(0.5, 0.5, "Accuracy: {:.1f}%".format(100*(1 - np.sum(preds != b)/len(b))), **dict(ha='center', va='center', fontsize=24))
    bx.set_xticks([])
    bx.set_yticks([])
    ax = draw_court(ax, 'k')
    ax.set_title("Golden State Warriors - Basketball shots")
    ax.set_xlim(-250,250)
    # Descending values along th y axis from bottom to top
    # in order to place the hoop by the top of plot
    ax.set_ylim(-47.5, 422.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(data[labels==1, 0], data[labels==1, 1], alpha=0.5, marker='o', s=25, label="Made Shot")
    ax.scatter(data[labels==-1, 0], data[labels==-1, 1], alpha=0.5, marker='x', s=25, label="Missed Shot")
    ax.scatter(data[preds != labels, 0], data[preds != labels, 1], alpha=0.5, marker='o', edgecolor="green", color="none", s=85, label="Missclassified")
    ax.legend(fontsize=14)

