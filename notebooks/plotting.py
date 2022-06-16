from firedrake import triplot
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def plot_mesh(mesh, fig=None, axes=None, **kwargs):
    """
    Helper function for mesh plotting.

    :arg mesh: the mesh to be plotted
    :kwarg fig: the matplotlib figure
    :kwarg axes: the matplotlib axes

    Additional kwargs are passed to :func:`triplot`.
    """
    kwargs.setdefault("interior_kw", {"linewidth": 0.5})
    kwargs.setdefault("boundary_kw", {"linewidth": 2.0})
    if fig is None and axes is None:
        fig, axes = plt.subplots(figsize=(5, 5))
    tp = triplot(mesh, axes=axes, **kwargs)
    axes.axis(False)
    return fig, axes, tp


def plot_video(fig, animate, results, dt):
    """
    Helper function for video plotting.

    :arg fig: the matplotlib figure
    :arg animate: a function saying how to plot each frame
    :arg results: a list of items to be plotted in each frame
    :arg dt: the timestep
    """
    interval = 4e3 * float(dt)
    animation = FuncAnimation(fig, animate, frames=results, interval=interval)
    plt.close()
    return animation.to_jshtml()
