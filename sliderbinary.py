
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

import lightkurve as lk
import scipy.ndimage

import pymc3 as pm
import theano.tensor as tt
from exoplanet.gp import terms, GP

import exoplanet as xo



#download light curve and define all of the variables for that
search = lk.search_lightcurvefile('HD212771',mission = 'tess')

#search = lk.search_lightcurvefile('KIC10963065', cadence='short', mission='Kepler')
files = search.download_all()
#files = search[1:10].download_all()

lc = files.PDCSAP_FLUX.stitch()
lc = lc.remove_outliers().remove_nans()
lc_err = lc.flux_err
lc_smooth = lc.bin(binsize = 100)

flux = lc.flux
mean = np.mean(flux)
std = np.std(flux)
time = lc.astropy_time.value - lc.astropy_time.value[0]
plt.figure()
plt.plot(time,flux)

pg = lc.to_periodogram(method='lombscargle', normalization='psd', minimum_frequency = 0.01, maximum_frequency=10000)
#feel free to play around with smoothing here using the filter width
pg_smooth = pg.smooth(method='boxkernel', filter_width=100).power

power = np.array(pg.power)
pg_smooth = np.array(pg_smooth)
nu = np.array(pg.frequency)

#omega = np.linspace(10,400,5000)
##ask about this
#omega = 1e-6*2*np.pi*omega

omega = nu

y = np.ascontiguousarray(flux, dtype=np.float64)
norm = np.median(y)
y = (y / norm - 1)
#y = np.ones_like(y)*1e-4
plt.figure()
plt.plot(time,y)
plt.show()
#yerr = np.ascontiguousarray(lc_err, dtype=np.float64)
yerr = np.ascontiguousarray(lc_err, dtype=np.float64)/ norm
t = np.ascontiguousarray(time, dtype=np.float64)

opt = 0
bin = 0


def run_gp_single(Sgv,wgv,S1v,w1v,Q1v,opt = opt):
    
    if (opt == 1):
        print('Running Gp Single Optimiziation', 'Sgv',Sgv, 'wgv',wgv, 'S1v',S1v, 'w1v',w1v, 'Q1v', Q1v)
    with pm.Model() as model:

        logs2 = pm.Normal("logs2", mu=2 * np.log(np.mean(yerr)), sigma=100.0, testval = 100)


        logSg = pm.Normal("logSg", mu=Sgv, sigma= 1000.0, testval=Sgv)
        logwg = pm.Normal("logwg", mu = wgv, sigma = 1000.0, testval=wgv)
        logS1 = pm.Normal("logS1", mu=S1v, sigma=1000.0, testval=S1v)
        logw1 = pm.Normal("logw1", mu =w1v, sigma = 1000.0, testval=w1v)
        logQ1 = pm.Normal("logQ1", mu=Q1v, sigma=1000.0, testval=Q1v)


        # Set up the kernel an GP
        bg_kernel = terms.SHOTerm(log_S0=logSg, log_w0=logwg, Q=1.0 / np.sqrt(2))
        star_kernel1 = terms.SHOTerm(log_S0=logS1, log_w0=logw1, log_Q=logQ1)
        kernel = star_kernel1 + bg_kernel

        gp = GP(kernel, t, yerr ** 2 + pm.math.exp(logs2))
        gp_star1 = GP(star_kernel1, t, yerr ** 2 + pm.math.exp(logs2))
        gp_bg = GP(bg_kernel, t, yerr ** 2 + pm.math.exp(logs2))

        # Condition the GP on the observations and add the marginal likelihood
        # to the model
        gp.marginal("gp", observed=y)
        
        
    with model:
        val = gp.kernel.psd(omega)

        psd_init = xo.eval_in_model(val)

        bg_val = gp_bg.kernel.psd(omega)
        star_val_1 = gp_star1.kernel.psd(omega)



        bg_psd_init = xo.eval_in_model(bg_val)
        star_1_psd_init = xo.eval_in_model(star_val_1)


    #     print('done_init_plot')


        map_soln = model.test_point

        if (opt == 1):
            print('running opt')
            map_soln = xo.optimize(start=map_soln, vars=[logSg])
        #ask about this, do i need to scale when I show this?
            map_soln = xo.optimize(start=map_soln, vars=[logwg])
            map_soln = xo.optimize(start=map_soln, vars=[logw1])
            map_soln = xo.optimize(start=map_soln, vars=[logS1])
            map_soln = xo.optimize(start=map_soln)
            
            print(map_soln.values())
            mu, var = xo.eval_in_model(
                gp.predict(t, return_var=True), map_soln
            )
            
            plt.figure()
            plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0, label="data")
            sd = np.sqrt(var)
            art = plt.fill_between(t, mu + sd, mu - sd, color="C1", alpha=0.3)
            art.set_edgecolor("none")
            plt.plot(t, mu, color="C1", label="prediction")

            plt.legend(fontsize=12)
            plt.xlabel("t")
            plt.ylabel("y")
            plt.xlim(0, 10)
            _ = plt.ylim(-2.5, 2.5)



        psd_final = xo.eval_in_model(gp.kernel.psd(omega),map_soln)


        bg_psd_fin = xo.eval_in_model(bg_val,map_soln)
        star_1_psd_fin = xo.eval_in_model(star_val_1,map_soln)
    return psd_init, star_1_psd_init, bg_psd_init , psd_final, star_1_psd_fin, bg_psd_fin, map_soln



def run_gp_binary(Sg,wg,S1,w1,Q1,S2,w2,Q2,opt=opt):
    with pm.Model() as model:

        logs2 = pm.Normal("logs2", mu=2 * np.log(np.mean(yerr)), sigma=100.0, testval = -100)


        mean = pm.Normal("mean", mu=np.mean(y), sigma=1.0)
        logSg = pm.Normal("logSg", mu=0.0, sigma=15.0, testval=Sg)
        logwg = pm.Normal("logwg", mu = 0.0, sigma = 15.0, testval=wg-np.log(1e6))
        logS1 = pm.Normal("logS1", mu=0.0, sigma=15.0, testval=S1)
        logw1 = pm.Normal("logw1", mu = 0.0, sigma = 15.0, testval=w1-np.log(1e6))
        logQ1 = pm.Normal("logQ1", mu=0.0, sigma=15.0, testval=Q1)

        logS2 = pm.Normal("logS2", mu=0.0, sigma=15.0, testval=S2)
        logw2 = pm.Normal("logw2", mu = 0.0, sigma = 15.0, testval=w2-np.log(1e6))
        logQ2 = pm.Normal("logQ2", mu=0.0, sigma=15.0, testval=Q2)

        # Set up the kernel an GP
        bg_kernel = terms.SHOTerm(log_S0=logSg, log_w0=logwg, Q=1.0 / np.sqrt(2))
        star_kernel1 = terms.SHOTerm(log_S0=logS1, log_w0=logw1, log_Q=logQ1)
        star_kernel2 = terms.SHOTerm(log_S0=logS2, log_w0=logw2, log_Q=logQ2)
        kernel = star_kernel1 + star_kernel2 + bg_kernel

        gp = GP(kernel, t, yerr ** 2 + pm.math.exp(logs2), mean = mean)
        gp_star1 = GP(star_kernel1, t, yerr ** 2 + pm.math.exp(logs2), mean = mean)
        gp_bg = GP(bg_kernel, t, yerr ** 2 + pm.math.exp(logs2), mean = mean)
        gp_star2 = GP(star_kernel2, t, yerr ** 2 + pm.math.exp(logs2), mean = mean)

        # Condition the GP on the observations and add the marginal likelihood
        # to the model
        gp.marginal("gp", observed=y)

    with model:
        val = gp.kernel.psd(omega)

        psd_init = xo.eval_in_model(val)

        bg_val = gp_bg.kernel.psd(omega)
        star_val_1 = gp_star1.kernel.psd(omega)
        star_val_2 = gp_star2.kernel.psd(omega)


        bg_psd_init = xo.eval_in_model(bg_val)
        star_1_psd_init = xo.eval_in_model(star_val_1)
        star_2_psd_init = xo.eval_in_model(star_val_2)

    #     print('done_init_plot')


        map_soln = model.test_point

        if (opt == 1):
            print('running opt')
            map_soln = xo.optimize(start=map_soln, vars=[logSg])
        #ask about this, do i need to scale when I show this?
            map_soln = xo.optimize(start=map_soln, vars=[logwg])
            #map_soln = xo.optimize(start=map_soln, vars=[logS1,logw1])
            #map_soln = xo.optimize(start=map_soln, vars=[logS2,logw2])



        psd_final = xo.eval_in_model(gp.kernel.psd(omega),map_soln)



        bg_psd_fin = xo.eval_in_model(bg_val,map_soln)
        star_1_psd_fin = xo.eval_in_model(star_val_1,map_soln)
        star_2_psd_fin = xo.eval_in_model(star_val_2,map_soln)
    return psd_init, star_1_psd_init,star_2_psd_init, bg_psd_init , psd_final, star_1_psd_fin, star_2_psd_fin, bg_psd_fin, map_soln







Sg0 = -23.6
wg0 = 5.6
S10 = -24.3
w10 = 5.5
Q10 = 1.2
S20 = 0.0
w20 = 0.0
Q20 = 0.0


delta_vals = 0.1


psd_init, star_1_psd_init, bg_psd_init , psd_final, star_psd_fin, bg_psd_fin, map_soln = run_gp_single(Sg0, wg0, S10,w10, Q10)



omega2pi = omega
#omega2pi = omega/(2*np.pi)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)

ax.set_title('HD212771')
psd_init_line, = ax.plot(omega2pi, psd_init, label = 'Total')
star_1_psd_init_line, = ax.plot(omega2pi, star_1_psd_init, label = 'Star')
bg_psd_init_line, = ax.plot(omega2pi, bg_psd_init, label = 'BG')
#plt.plot(omega,psd_final, label = 'PSD following optimization')
#ask about this change, otherwise not on the same scale



true_data_line, = ax.plot(nu, power, alpha = 0.5)
#true_data_line, = ax.plot(1e-6*nu, power*1e6, alpha = 0.5)



ax.legend(loc = 'best')
ax.set_xscale('log')

ax.set_yscale('log')
#plt.xlim(10,500)
#plt.ylim(1e-12,1e-9)




plt.margins(x=0)

axcolor = 'lightgoldenrodyellow'
axSg = plt.axes([0.2, 0.25, 0.65, 0.03], facecolor=axcolor)
axWg = plt.axes([0.2, 0.2, 0.65, 0.03], facecolor=axcolor)
axS1 = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
axw1 = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
axQ1 = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)
axS2 = plt.axes([1.5, 0.15, 0.4, 0.03], facecolor=axcolor)
axw2 = plt.axes([1.5, 0.1, 0.4, 0.03], facecolor=axcolor)
axQ2 = plt.axes([1.5, 0.05, 0.4, 0.03], facecolor=axcolor)


sSg = Slider(axSg, 'Sg', -30, 30, valinit=Sg0, valstep=delta_vals)
swg = Slider(axWg, 'Wg', -30, 30, valinit=wg0, valstep=delta_vals)
sS1 = Slider(axS1, 'S1', -30, 30, valinit=S10, valstep=delta_vals)
sw1 = Slider(axw1, 'w1', -30, 30, valinit=w10, valstep=delta_vals)
sQ1 = Slider(axQ1, 'Q1', -30, 30, valinit=Q10, valstep=delta_vals)
sS2 = Slider(axS2, 'S2', -15, 15, valinit=S20, valstep=delta_vals)
sw2 = Slider(axw2, 'w2', -15, 15, valinit=w20, valstep=delta_vals)
sQ2 = Slider(axQ2, 'Q2', -15, 15, valinit=Q20, valstep=delta_vals)



def update(val, bin = bin, opt = opt):
    print(bin)
    Sg = sSg.val
    wg = swg.val
    S1 = sS1.val
    w1 = sw1.val
    Q1 = sQ1.val
    if (bin == 0):
        psd_init, star_1_psd_init, bg_psd_init , psd_final, star_psd_fin, bg_psd_fin, map_soln = run_gp_single(Sg, wg, S1,w1, Q1,opt)
    else:
        S2 = sS2.val
        w2 = sw2.val
        Q2 = sQ2.val
        psd_init, star_1_psd_init,star_2_psd_init, bg_psd_init , psd_final, star_1_psd_fin, star_2_psd_fin, bg_psd_fin, map_soln = run_gp_binary(Sg,wg,S1,w1,Q1,S2,w2,Q2,opt=opt)


    psd_init_line.set_ydata(psd_init)
    star_1_psd_init_line.set_ydata(star_1_psd_init)
    bg_psd_init_line.set_ydata(bg_psd_init)
    if (bin == 1):
        star_2_psd_init_line.set_ydata(star_2_psd_init)

    if (opt == 1):
        psd_final_line.set_ydata(psd_final)
        star_psd_fin_line.set_ydata(star_psd_fin)
        bg_psd_fin_line.set_ydata(bg_psd_fin)
        if (bin == 1):
            star_1_psd_init_line.set_ydata(star_1_psd_init)



    fig.canvas.draw_idle()


sSg.on_changed(update)
swg.on_changed(update)
sS1.on_changed(update)
sw1.on_changed(update)
sQ1.on_changed(update)

sS2.on_changed(update)
sw2.on_changed(update)
sQ2.on_changed(update)




rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('Single','Binary'), active=0)


def binary(label):

    if (label == 'Binary'):

        bin = 1
        Sg0 = -15.9
        wg0 = 6.
        S10 = -13.9
        w10 = 6.5
        Q10 = 1.2
        S20 = -9.9
        w20 = 6.3
        Q20 = 1.4



        psd_init, star_1_psd_init,star_2_psd_init, bg_psd_init , psd_final, star_1_psd_fin, star_2_psd_fin, bg_psd_fin, map_soln = run_gp_binary(Sg0,wg0,S10,w10,Q10,S20,w20,Q20,opt=opt)


        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)

        ax.set_title('HD212771')
        psd_init_line, = ax.plot(omega2pi, psd_init, label = 'Total')
        star_1_psd_init_line, = ax.plot(omega2pi, star_1_psd_init, label = 'Star 1')
        star_2_psd_init_line, = ax.plot(omega2pi, star_2_psd_init, label = 'Star 2')

        bg_psd_init_line, = ax.plot(omega2pi, bg_psd_init, label = 'BG')
        #plt.plot(omega,psd_final, label = 'PSD following optimization')
        #ask about this change, otherwise not on the same scale
        true_data_line, = ax.plot(nu, power, alpha = 0.5)
        ax.legend(loc = 'best')
        ax.set_xscale('log')

        ax.set_yscale('log')
        #plt.xlim(10,500)
        #plt.ylim(1e-12,1e-9)


        #move these over and add the new sliders
        axcolor = 'lightgoldenrodyellow'
        axSg = plt.axes([0.05, 0.25, 0.35, 0.03], facecolor=axcolor)
        axWg = plt.axes([0.05, 0.2, 0.35, 0.03], facecolor=axcolor)
        axS1 = plt.axes([0.05, 0.15, 0.35, 0.03], facecolor=axcolor)
        axw1 = plt.axes([0.05, 0.1, 0.35, 0.03], facecolor=axcolor)
        axQ1 = plt.axes([0.05, 0.05, 0.35, 0.03], facecolor=axcolor)

        axS2 = plt.axes([0.55, 0.15, 0.35, 0.03], facecolor=axcolor)
        axw2 = plt.axes([0.55, 0.1, 0.35, 0.03], facecolor=axcolor)
        axQ2 = plt.axes([0.55, 0.05, 0.35, 0.03], facecolor=axcolor)

        sSg = Slider(axSg, 'Sg', -30, 30, valinit=Sg0, valstep=delta_vals)
        swg = Slider(axWg, 'Wg', -30, 30, valinit=wg0, valstep=delta_vals)
        sS1 = Slider(axS1, 'S1', -30, 30, valinit=S10, valstep=delta_vals)
        sw1 = Slider(axw1, 'w1', -30, 30, valinit=w10, valstep=delta_vals)
        sQ1 = Slider(axQ1, 'Q1', -30, 30, valinit=Q10, valstep=delta_vals)

        sS2 = Slider(axS2, 'S2', -15, 15, valinit=S20, valstep=delta_vals)
        sw2 = Slider(axw2, 'w2', -15, 15, valinit=w20, valstep=delta_vals)
        sQ2 = Slider(axQ2, 'Q2', -15, 15, valinit=Q20, valstep=delta_vals)


        sSg.on_changed(update)
        swg.on_changed(update)
        sS1.on_changed(update)
        sw1.on_changed(update)
        sQ1.on_changed(update)

        sS2.on_changed(update)
        sw2.on_changed(update)
        sQ2.on_changed(update)

        button.on_clicked(optimize)
        radio.on_clicked(binary)

        fig.canvas.draw_idle()



radio.on_clicked(binary)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Optimize', color=axcolor, hovercolor='0.975')

def optimize(event):
    opt = 1

    Sg = sSg.val
    wg = swg.val
    S1 = sS1.val
    w1 = sw1.val
    Q1 = sQ1.val
    print(Sg, wg, S1, w1, Q1)
    if (bin == 0):
        psd_init, star_1_psd_init, bg_psd_init , psd_final, star_1_psd_fin, bg_psd_fin, map_soln = run_gp_single(Sg, wg, S1,w1, Q1, opt)
    else:
        S2 = sS2.val
        w2 = sw2.val
        Q2 = sQ2.val
        psd_init, star_1_psd_init,star_2_psd_init, bg_psd_init , psd_final, star_1_psd_fin, star_2_psd_fin, bg_psd_fin, map_soln = run_gp_binary(Sg,wg,S1,w1,Q1,S2,w2,Q2,opt)

    fig, ax = plt.subplots(2,1)
    #plt.subplots_adjust(bottom=0.35)

    ax[0].set_title('HD212771')
    psd_init_line, = ax[0].plot(omega2pi, psd_init, label = 'Total')
    star_1_psd_init_line, = ax[0].plot(omega2pi, star_1_psd_init, label = 'Star 1')
    if (bin == 1):
        star_2_psd_init_line, = ax[0].plot(omega2pi, star_2_psd_init, label = 'Star 2')


    bg_psd_init_line, = ax[0].plot(omega2pi, bg_psd_init, label = 'BG')
    #plt.plot(omega,psd_final, label = 'PSD following optimization')
    #ask about this change, otherwise not on the same scale
    true_data_line, = ax[0].plot(nu, power, alpha = 0.5)
    ax[0].legend(loc = 'best')
    ax[0].set_xscale('log')
    ax[0].set_xticklabels([])
    ax[0].set_yscale('log')
    #plt.xlim(10,500)
    #plt.ylim(1e-12,1e-9)

    logs2 = map_soln['logs2']


    #ax[1].set_title('HD212771 post optimization')

    psd_final_line, = ax[1].plot(omega2pi, psd_final, label = 'Total')
    star_1_psd_fin_line, = ax[1].plot(omega2pi, star_1_psd_fin, label = 'Star 1')
    
    if (bin == 1):
        star_2_psd_fin_line, = ax[1].plot(omega2pi, star_2_psd_fin, label = 'Star 2')


    bg_psd_fin_line, = ax[1].plot(omega2pi, bg_psd_fin,  label = 'BG')
    #ax[1].plot(omega2pi, np.exp(logs2)*np.ones_like(omega))
    #plt.plot(omega,psd_final, label = 'PSD following optimization')
    true_data_line2, = ax[1].plot(nu,power, alpha = 0.5)
    ax[1].legend(loc = 'best')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    #plt.xlim(10,500)
    #plt.ylim(1e-12,1e-9)

button.on_clicked(optimize)









plt.show()
