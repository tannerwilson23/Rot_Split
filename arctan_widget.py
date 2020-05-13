import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.40)
r = np.arange(0.0, 1.0, 0.001)

use_new = True

if use_new:
    offset = 0.5
    a0 = 10
    b0 = 9
    c0 = 0
    d0 = 450

    def get_omega(a, b, c, d):
        r_offset = r - 0.5
        return a * np.arctan((r_offset - c) * b) + d

    omega = get_omega(a0, b0, c0, d0)
    #omega = a0 * np.arctan((r - offset) * b0 - c0) + d0
    l, = plt.plot(r, omega, lw=2)
    #ax.margins(x=0)

    axcolor = 'lightgoldenrodyellow'
    ax_a = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_b = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_c = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
    ax_d = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)

    s_a = Slider(ax_a, 'a', 0, 100, valinit=a0, valstep=1)
    s_b = Slider(ax_b, 'b', -100, 100, valinit=b0, valstep=0.01)
    s_c = Slider(ax_c, 'c', -0.5, 0.5, valinit=c0, valstep=0.01)
    s_d = Slider(ax_d, 'd', 0, 500, valinit=d0, valstep=1)


    def update(val):
        a, b, c, d = (s_a.val, s_b.val, s_c.val, s_d.val)
        v = (r - offset) * b - c
        if np.any(np.abs(v) > 20):
            l.set_color("tab:red")
        else:
            l.set_color("tab:blue")
            
        #y = a * np.arctan((r - offset) * b - c) + d
        y = get_omega(a, b, c, d)
        l.set_ydata(y)
        ax.set_ylim(y.min(), y.max())
        fig.canvas.draw_idle()

else:
    offset = 0
    a0 = -20
    b0 = -20
    c0 = -13
    d0 = 430

    def get_omega(a, b, c, d):
        return a * np.arctan(r * b - c) + d


    omega = get_omega(a0, b0, c0, d0)
    l, = plt.plot(r, omega, lw=2)

    axcolor = 'lightgoldenrodyellow'
    ax_a = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_b = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_c = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
    ax_d = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)

    s_a = Slider(ax_a, 'a', -60, -10, valinit=a0, valstep=1)
    s_b = Slider(ax_b, 'b', -30, -10, valinit=b0, valstep=1)
    s_c = Slider(ax_c, 'c', -50, 0, valinit=c0, valstep=1)
    s_d = Slider(ax_d, 'd', 200, 500, valinit=d0, valstep=1)


    def update(val):
        a, b, c, d = (s_a.val, s_b.val, s_c.val, s_d.val)
            
        #y = a * np.arctan((r - offset) * b - c) + d
        y = get_omega(a, b, c, d)
        l.set_ydata(y)
        ax.set_ylim(y.min(), y.max())
        fig.canvas.draw_idle()


s_a.on_changed(update)
s_b.on_changed(update)
s_c.on_changed(update)
s_d.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    s_a.reset()
    s_b.reset()
    s_c.reset()
    s_d.reset()    
button.on_clicked(reset)


plt.show()