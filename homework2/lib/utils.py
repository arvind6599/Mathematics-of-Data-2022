import torch
from torch.utils.data import Dataset
import uuid
import tempfile
import os
import torch
import math

from tqdm import tqdm

temp = tempfile.gettempdir()
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio

from torch import nn

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

temp = tempfile.gettempdir()


class NGaussians(Dataset):

    def __init__(self,
                 N=2,
                 moments=None,
                 inter_distance=6.0,
                 max_cols=None,
                 max_examples=None,
                 scaling=0.5):
        """

        :param N:
        :param moments:
        :param inter_distance:
        :param max_cols:
        :param max_examples: by default dim**2 *100
        """
        if moments is not None:
            assert len(moments) == N
            assert all([len(x) == 2 for x in moments])
            loc = torch.stack([
                x[0] if torch.is_tensor(x[0]) else torch.tensor(x[0])
                for x in moments
            ])
            scale = torch.stack([
                x[1] if torch.is_tensor(x[1]) else torch.tensor(x[1])
                for x in moments
            ])
        else:
            if max_cols is None:
                max_cols = int(np.ceil(np.sqrt(N)))
            x = torch.tensor(
                [inter_distance * (i % max_cols) for i in range(N)]
            )
            y = torch.tensor(
                [inter_distance * (i // max_cols) for i in range(N)]
            )
            loc = torch.stack([x, y], -1)
            scale = 0.5*torch.ones_like(loc)

        if max_examples is None:
            max_examples = int(10**loc.shape[-1] * loc.shape[0])
        loc = loc * scaling
        scale = scale * scaling

        mix = torch.distributions.Categorical(logits=torch.ones(len(loc)))
        comp = torch.distributions.Independent(
            torch.distributions.Normal(loc, scale), 1
        )
        self.dist = torch.distributions.MixtureSameFamily(mix, comp)
        self.max_examples = max_examples
        self.examples = self.dist.sample([max_examples])

    def __len__(self):
        return self.max_examples

    def __getitem__(self, item):
        return self.examples[item]

    def sample(self, *args, **kwargs):
        return self.dist.sample(*args, **kwargs)


def compare_samples_2D(sample_1,
                       sample_2,
                       filename,
                       x_lim=[-2, 8],
                       y_lim=[-2, 8]):
    """Plot real vs generated data in 2D"""
    plt.scatter(x=sample_1[:, 0], y=sample_1[:, 1], color='blue', alpha=.1)
    plt.scatter(x=sample_2[:, 0], y=sample_2[:, 1], color='red', alpha=.2)
    axes = plt.gca()
    axes.set_xlim(x_lim)
    axes.set_ylim(y_lim)
    axes.set_aspect('equal')
    if type(filename) is list:
        for fl in filename:
            plt.savefig(fl, bbox_inches='tight')
    else:
        plt.savefig(filename, bbox_inches='tight')
    plt.clf()

def animate(input_files, output_file, duration):
    """Generate an animated gif from a list of files"""
    images = [None] * len(input_files)
    for i, file in enumerate(input_files):
        images[i] = imageio.imread(file)
    imageio.mimsave(output_file, images, 'GIF', duration=duration)

def simplex_project(v):
    pos_part = torch.nn.ReLU(inplace=True)
    radius=1.0
    dim = 3
    mu, _ = torch.sort(v)
    cumul_sum = torch.divide(
        torch.flip(torch.cumsum(torch.flip(mu, [0]), dim=0), [0]) - radius, torch.arange(dim, 0, -1,device=mu.device))
    rho = torch.argmin(torch.where(mu > cumul_sum, torch.arange(dim,device=mu.device), dim))
    theta = cumul_sum[rho]
    v.add_(-theta)
    pos_part(v)

def run_alg(alg, f, x_init, y_init, step_size, n_iterations=1000):
    x, y = x_init.clone().requires_grad_(True), y_init.clone().requires_grad_(True)
    x_sequence = [x.detach().numpy()[:2].copy()]
    y_sequence = [y.detach().numpy()[:2].copy()]
    for _ in range(n_iterations):
        alg(f, x, y, step_size)
        x_sequence.append(x.detach().numpy()[:2].copy())
        y_sequence.append(y.detach().numpy()[:2].copy())
    return np.array(x_sequence), np.array(y_sequence)

def visualize_seq(L_x, L_y):
    plt.style.use('seaborn-poster')
    plt.gca().add_patch(Polygon([(0, 0), (1.0, 0), (0, 1.0)],
                                     facecolor='y', alpha=0.1))
    plt.scatter(L_x[:, 0], L_x[:, 1], alpha=0.5, s=15)
    plt.scatter(L_y[:, 0], L_y[:, 1], alpha=0.5, s=15)
    path = L_x.T
    plt.quiver(path[0,:-1], path[1,:-1], path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], scale_units='xy', angles='xy', scale=1, width=0.0015, color='b')
    path = L_y.T
    plt.quiver(path[0,:-1], path[1,:-1], path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], scale_units='xy', angles='xy', scale=1, width=0.0015, color='r')

    plt.xlim(1/3-0.75, 1/3+0.75)
    plt.ylim(1/3-0.75, 1/3+0.75)
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    


class GanTrainer():

    def __init__(
            self, batch_size, data, noise, make_gif=False, mode="penalty"):
        self.data = data
        self.noise = noise
        self.id = uuid.uuid4()
        self.snapshots = []
        self.checkpoints = []
        self.batch_size = batch_size
        self.make_gif = make_gif
        self.fixed_real_sample = data.dist.sample((1000,)).cpu().numpy()
        self.step=0
        self.mode=mode

    def _snapshot(self, g, ckpt=False):
        """Save an image of the current generated samples"""
        with torch.no_grad():
            gen_sample = g(self.noise.sample((self.batch_size,))).cpu().numpy()
        file_png = os.path.join(
                temp, str(self.id) + '_' + str(len(self.snapshots)) + '.png')
        filename = [file_png]
        if ckpt:
            file_pdf = os.path.join(
                    str(self.id) + '_' + str(len(self.checkpoints)) + '.pdf')
            filename.append(file_pdf)
        compare_samples_2D(
                self.fixed_real_sample, gen_sample, filename)
        self.snapshots.append(filename[0])
        if ckpt:
            self.checkpoints.append(filename[1])

    def render_gif(self, output_file, duration):
        """
        Render animated gif based on current snapshots

        Args:
            output_file (str): output_file
            duration (float): output video duration in seconds
        """
        animate(self.snapshots, output_file, duration)


    def alternating(self, n_iter, f, g, f_optim, g_optim, alternating_update, n_checkpoints, f_ratio=1):
        
        ckpts = math.floor(n_iter / n_checkpoints)

        bar=tqdm(range(n_iter))
        for _ in bar:
            self.step += 1
            noise = self.noise.sample([self.batch_size])
            real = self.data.sample([self.batch_size])
            l= alternating_update(self.step, f, g, f_optim, g_optim, noise, real, d_ratio=f_ratio)
            bar.set_description(f"W1:{l:0.2} G{self.step%f_ratio==0}")

            if self.make_gif:
                if _ % ckpts == 0:
                    self._snapshot(g, ckpt=True)
                else:
                    self._snapshot(g, ckpt=False)


def train(f, g, f_optim, g_optim, alternating_update, batch_size=200,
         n_iter=2000,
         hidden_dim=10,
         lr=1e-3,
         clip=1,
         f_ratio=5,
         seed=1,
         make_gif=True,
         device="cpu"):
    torch.manual_seed(seed)
    device = torch.device(device)

    # Define true distribution and noise distrubution
    # feel free to play with these to see the effects
    data = NGaussians(N=9)
    noise_mean = torch.zeros(2)
    noise_covariance = torch.eye(2)
    noise = torch.distributions.MultivariateNormal(noise_mean, noise_covariance)

    # plot the real data
    plt.scatter(data.examples[:, 0], data.examples[:, 1])
    plt.show()

    

    # Initialize trainer
    trainer = GanTrainer(batch_size,
                         data=data,
                         noise=noise,
                         make_gif=make_gif)

    # train and save GIF
    trainer.alternating(n_iter=n_iter,
                        f=f,
                        g=g,
                        f_optim=f_optim,
                        g_optim=g_optim,
                        alternating_update=alternating_update,
                        f_ratio=f_ratio,
                        n_checkpoints=4)
    trainer.render_gif('movie' + '.gif', duration=0.1)

