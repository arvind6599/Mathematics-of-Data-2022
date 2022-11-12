from lib.opt_types import *
from itertools import cycle
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Circle, Rectangle, Arc

plt.style.use("seaborn-paper")



def run(method: OptAlgorithm, f: Function, x_zero: Vector, iterations: int, disp=False) -> RunTrace:
    x_lst = []
    val_lst = []
    state = method.init_state(f, x_zero)
    iterations = range(iterations)
    if disp:
        iterations = tqdm(iterations, desc=method.name + (7 - len(method.name))*" ")
    for _ in iterations:
        x_lst.append(state.x_k)
        val_lst.append(f(state.x_k))
        state = method.state_update(f, state)

    return RunTrace(sequence=x_lst, values=val_lst)


def plot(methods: List[OptAlgorithm], f: Function, x_zero: Vector, max_iteration: int, title="") -> None:
    colors = cycle(
        [
            (0, 0, 1),
            (0, 0.5, 0),
            (1, 0, 0),
            (0, 0.75, 0.75),
            (0.75, 0, 0.75),
            (0.75, 0.75, 0),
            (0, 0, 0),
            (0.5, 1, 0.5),
            (0.5, 0, 0.5),
            (0.75, 0.25, 0.25),
        ]
    )
    for method in methods:
        run_trace = run(method, f, x_zero, max_iteration, disp=True)
        plt.plot(
            range(max_iteration),
            np.array(run_trace.values) - f.minimum,
            color=next(colors),
            lw=2,
            label=method.name,
        )
    plt.legend(fontsize=14)
    plt.xlabel("#iterations", fontsize=14)
    plt.ylabel("")
    plt.ylabel(r"$f(\mathbf{x}^k) - f^\star$", fontsize=14)
    plt.xscale("log")
    plt.yscale("log")
    plt.title(title, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.show()

def test(method: OptAlgorithm, maxiter=100):
    M = np.array([[1.0, 0.0], [0.0, 10.0]])
    quadratic = Function(f=lambda u: 0.5*np.dot(u, np.dot(M, u)), grad=lambda u: np.dot(M, u), minimum=0.0, lips_grad=10.0, strng_cvx=1.0)

    run_trace = run(method, quadratic, np.array([100.0, 100.0]), maxiter)

    f, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(12, 4))
    f.suptitle("TEST OF  " + method.name + " ", fontsize=14)

    ax_1.plot(
            range(maxiter),
            np.array(run_trace.values) - quadratic.minimum,
            color=(0, 0, 1),
            lw=2,
            label=method.name,
        )
    ax_1.legend(fontsize=14)
    ax_1.set_xlabel("#iterations", fontsize=14)
    ax_1.set_ylabel("")
    ax_1.set_ylabel(r"$f(\mathbf{x}^k) - f^\star$", fontsize=14)
    ax_1.set_xscale("log")
    ax_1.set_yscale("log")
    ax_1.grid()
    
    
    x_axis = np.linspace(-120, 120, 100)
    y_axis = np.linspace(-120, 120, 100)
    X, Y = np.meshgrid(x_axis, y_axis)
    def function(x, y):
        return 0.5*(x**2 + 10*y**2)

    Z = function(X, Y)
    contours = ax_2.contour(X, Y, Z, colors='black')
    ax_2.clabel(contours, inline=True, fontsize=8)

    path = np.array(run_trace.sequence).T
    ax_2.quiver(path[0,:-1], path[1,:-1], path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], scale_units='xy', angles='xy', scale=1, width=0.008)
    plt.show()

def test_composite(method):
    M = np.array([[1.0, 0.0], [0.0, 10.0]])
    f = Function(f=lambda u: 0.5*np.dot(u, np.dot(M, u)), grad=lambda u: np.dot(M, u), minimum=0.0, lips_grad=10.0, strng_cvx=1.0)
    g = Function(f=lambda x: 30*np.sum(np.abs(x), axis=0), subgrad= lambda x: 30*np.sign(x), prox=lambda gamma, x: np.sign(x)*np.maximum(np.abs(x) - 30*gamma, 0.0))
    composite_function = CompositeFunction(f = f, g=g, minimum=0.0)
    maxiter = 1000
    x_zero = np.array([10.0, 10.0])
    run_trace = run(method, composite_function, x_zero, maxiter)
    
    f, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(12, 4))
    f.suptitle("TEST OF  " + method.name + " ", fontsize=14)

    ax_1.plot(
            range(maxiter),
            np.array(run_trace.values) - composite_function.minimum,
            color=(0, 0, 1),
            lw=2,
            label=method.name,
        )
    ax_1.legend(fontsize=14)
    ax_1.set_xlabel("#iterations", fontsize=14)
    ax_1.set_ylabel("")
    ax_1.set_ylabel(r"$f(\mathbf{x}^k) - f^\star$", fontsize=14)
    ax_1.set_xscale("log")
    ax_1.set_yscale("log")
    ax_1.grid()

    x_axis = np.linspace(-12, 12, 100)
    y_axis = np.linspace(-12, 12, 100)
    X, Y = np.meshgrid(x_axis, y_axis)
    def function(x, y):
        return 0.5*(x**2 + 10*y**2) + 30*np.abs(x) + 30*np.abs(y)

    Z = function(X, Y)
    contours = ax_2.contour(X, Y, Z, colors='black')
    ax_2.clabel(contours, inline=True, fontsize=8)

    path = np.array(run_trace.sequence).T
    ax_2.quiver(path[0,:-1], path[1,:-1], path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], scale_units='xy', angles='xy', scale=1, width=0.008)
    plt.show()


def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

