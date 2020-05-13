import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

import logging

import george
from george import kernels


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.40)
x = np.linspace(0.0, 1.0, 100)


k1, k2 = (1e-3, 3.35)
k3 = 0
#kernel = k1 * kernels.ExpKernel(np.exp(k2))
#kernel = k1 * kernels.LinearKernel(log_gamma2=k2, order=3)
#kernel = k1 * kernels.ExpSquaredKernel(metric=k2)
kernel = k1 * kernels.RationalQuadraticKernel(log_alpha=k2, metric=1)

gp = george.GP(
    kernel,
    fit_white_noise=False
)
#    white_noise=k3,
#    fit_white_noise=True
#    )

gp.compute(x)

l, = ax.plot(x, gp.sample(), c="tab:blue")
l2, = ax.plot(x, gp.sample(), c="tab:blue", alpha=0.6)
l3, = ax.plot(x, gp.sample(), c="tab:blue", alpha=0.3)


axcolor = 'lightgoldenrodyellow'
ax_k1 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_k2 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_k3 = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)

s_k1 = Slider(ax_k1, 'k1', 1e-3, 10, valinit=k1, valstep=0.01)
s_k2 = Slider(ax_k2, 'k2', -6, 3, valinit=k2, valstep=0.01)
s_k3 = Slider(ax_k3, 'k3', -1, 1, valinit=k3, valstep=0.01)


def update(val):
    k1, k2, k3 = (s_k1.val, s_k2.val, s_k3.val)
    try:
        gp.set_parameter_vector([k1, k2, k3][:len(gp.get_parameter_vector())])#np.exp(k2)])#, k3])
        gp.compute(x, yerr=1e-6)

        y = gp.sample()
    except:
        logging.exception("this")
        None

    else:
        lims = np.hstack([y, ax.get_ylim(), ax.get_ylim()])
        lims = (np.min(lims), np.max(lims))

        new_l2 = l.get_ydata().copy()
        new_l3 = l2.get_ydata().copy()

        l.set_ydata(y)
        l2.set_ydata(new_l2)
        l3.set_ydata(new_l3)


        ax.set_ylim(*lims)
        fig.canvas.draw_idle()

s_k1.on_changed(update)
s_k2.on_changed(update)
s_k3.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
reset_lims = plt.axes([0.6, 0.025, 0.1, 0.04])
button_lims = Button(reset_lims, 'reset lims', color=axcolor, hovercolor='0.975')

def reset_lims(event):
    ax.set_ylim(-1, 1)
button_lims.on_clicked(reset_lims)

def reset(event):
    s_k1.reset()
    s_k2.reset()
    s_k3.reset()
    ax.set_ylim(-1, 1)
button.on_clicked(reset)


plt.show()

import threading
from time import sleep

def do():
    while True:
        sleep(0.1)
        update(None)

t = threading.Thread(target=do)
t.start()
