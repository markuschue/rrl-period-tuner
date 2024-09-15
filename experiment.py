import glob
import os
import sys
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.timeseries import LombScargle
from matplotlib.ticker import AutoMinorLocator
from matplotlib.widgets import Button, Slider, TextBox

from benchmark.period.functions import prepareTable
from benchmark.period.prueba_lombscargle import compute_period
from utils.data_utils import get_star_photometry


def compute_phase(key, period):
    rrl[key].loc[:, "PHASE"] = np.abs(
        (rrl[key][datestr]-zphase.values)/period - np.floor((rrl[key][datestr]-zphase.values)/period))
    return rrl[key].PHASE


def update(val):
    for ii in range(len(lclist)):
        if ii % 2 == 0:
            lclist[ii][0][0].set_xdata(
                compute_phase(lclist[ii][1], freq_slider.val))
        else:
            lclist[ii][0][0].set_xdata(compute_phase(
                lclist[ii][1], freq_slider.val)+1)
    fig.canvas.draw_idle()


def reset(event):
    freq_slider.reset()
    freq_slider.valstep = 0.0000005
    freq_slider.val = init_period
    freq_slider.valmin = init_period*0.95
    freq_slider.valmax = init_period*1.05
    currentstep.set_text("Current: %.9f" % freq_slider.valstep)
    currentperiod.set_text("Current: %.9f" % freq_slider.valinit)

    ax2.get_lines()[-1].remove()
    ax2.axvline(init_period, 0, 1, color='k', ls=(0, (5, 5)), alpha=0.5)

    if 'saved' in locals():
        saved.remove()

    return None


def confirm(event):
    if 'saved' in locals():
        saved.remove()

    saved = fig.text(0.85, 0.91, "New period saved into file!", size=15)

    rr.loc[rr["star"].str.contains(
        obj.split("_")[0]), "period"] = freq_slider.val
    Table.from_pandas(rr).write(
        '.\\data\\photometry\\periods.txt', format='ascii', overwrite=True)

    return None


def arrow_key_image_control(event):
    """
    This function takes an event from an mpl_connection
    and listens for key release events specifically from
    the keyboard arrow keys (left/right) and uses this
    input to advance/reverse to the next/previous image.
    """

    if 'saved' in locals():
        saved.remove()

    if event.key == 'left':
        new_val = freq_slider.val-freq_slider.valstep
        update(new_val)
        freq_slider.set_val(new_val)
        ax2.get_lines()[-1].remove()
        ax2.axvline(new_val, 0, 1, color='k', ls=(0, (5, 5)), alpha=0.5)
    elif event.key == 'right':
        new_val = freq_slider.val+freq_slider.valstep
        update(new_val)
        freq_slider.set_val(new_val)
        ax2.get_lines()[-1].remove()
        ax2.axvline(new_val, 0, 1, color='k', ls=(0, (5, 5)), alpha=0.5)

    currentperiod.set_text("Current: %.9f" % freq_slider.val)

    return None


def click_image_control(event):

    if 'saved' in locals():
        saved.remove()

    if event.inaxes == ax2:
        new_val = event.xdata
        ax2.get_lines()[-1].remove()

        ax2.axvline(new_val, 0, 1, color='k', ls=(0, (5, 5)), alpha=0.5)
        update(new_val)
        freq_slider.set_val(new_val)

    currentperiod.set_text("Current: %.9f" % freq_slider.val)

    return None


def submit_step(step):
    if 'saved' in locals():
        saved.remove()

    freq_slider.valstep = float(step)
    currentstep.set_text("Current: %.9f" % freq_slider.valstep)

    return None


def submit_period(new_period):
    if 'saved' in locals():
        saved.remove()

    freq_slider.val = float(new_period)
    freq_slider.valmin = float(new_period)*0.95
    freq_slider.valmax = float(new_period)*1.05
    freq_slider.set_val(float(new_period))
    update(new_period)
    currentperiod.set_text("Current: %.9f" % freq_slider.val)

    ax2.get_lines()[-1].remove()
    ax2.axvline(float(new_period), 0, 1, color='k', ls=(0, (5, 5)), alpha=0.5)

    return None


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 20
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)

resolution = (1920/96, 1080/96)


bdir = sys.argv[1]

if len(glob.glob(bdir+"/*final.fits")) > 0:
    magstr = "MAG_FINAL"
    magerrstr = "MAGERR_FINAL"
    final = "final"
    strfilt = -7
else:
    magstr = "MAG_AUTO_NORM"
    magerrstr = "MAGERR_AUTO"
    final = ""
    strfilt = -1

datestr = "DATE-OBS"

# List of objects, with their ID and periods.
obj = os.path.basename(os.path.normpath(bdir))
varid = Table.read(bdir+"/finalIDs_"+obj+".fits", hdu=1)[0][0]

# Load the general catalogs, with all the stars in the field.
cats = glob.glob(bdir+"/cat_"+obj+"*"+final+".fits")
data = {}
for line in cats:
    data[os.path.splitext(line)[0][strfilt]] = prepareTable(line, 1)

# Generate a dictionary extracting only the variable star from the general catalog.
rrl = {}
pd.options.mode.chained_assignment = None
for key in data:
    rrl[key] = data[key].loc[data[key].ID == varid, :]
rrl2 = get_star_photometry(bdir, star_id='GAIA DR3 ' + bdir.split('_')[-1])

# Define two dictionaries: colours for each filter, and artificial shift in each filter.
colours = plt.cm.rainbow(np.linspace(0, 1, len(rrl)))
factor = np.linspace(0, len(rrl)-1, len(rrl)) - np.floor(len(rrl)/2)

amplitude = {key: np.abs(rrl[key][magstr].min() -
                         rrl[key][magstr].max()) for key in rrl}
sort_keys = [list(amplitude.keys())[list(amplitude.values()).index(ii)]
             for ii in sorted(amplitude.values(), reverse=True)]

colours = {key: ii for key, ii in zip(sort_keys, colours)}
factor = {key: ii for key, ii in zip(sort_keys, factor)}
zerokey = list(factor.keys())[list(factor.values()).index(0)]
zeromedian = rrl[zerokey][magstr].median()

zphase = rrl[zerokey].loc[rrl[zerokey][magstr]
                          == rrl[zerokey][magstr].min(), datestr]


start = datetime.now()
best_period, possible_periods = compute_period(
    rrl[zerokey], datestr, magstr, magerrstr)
print(str(datetime.now() - start)+" s elapsed while computing period")

rr = Table.read('.\\data\\photometry\\periods.txt',
                format='ascii', names=["star", "period"]).to_pandas()
period_tabulated = rr.loc[rr["star"].str.contains(
    obj.split("_")[0]), "period"].iloc[0]
print("")
print("Computed period %.9f, tabulated period %.9f" %
      (best_period, period_tabulated))
print("Possible periods")
print(possible_periods)

init_period = period_tabulated

fig = plt.figure(figsize=(1920/(96*1.25), 1080/(96*1.25)), dpi=100)
ax1 = fig.add_subplot(122)
#  ax = fig.subplots(5,1,sharex=True)

fig.suptitle(obj.split("_")[1][:-3]+" " +
             obj.split("_")[1][-3:], size=24, y=0.925)
#  ax1.tick_params(labelcolor='none',which='both',top=False,boperiodom=False,left=False,right=False)
ax1.set_xlabel("Phase", size=15, y=1.05)
ax1.set_ylabel("Magnitude", size=15, x=0.25)

ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.grid(which='minor', alpha=0.5, linestyle=(0, (5, 10)), linewidth=0.5)
ax1.grid(which='major', alpha=0.85, linestyle=(0, (5, 10)), linewidth=0.5)

lowlimit_y = 1.75*(-1*factor[sort_keys[-1]]) + zeromedian
uplimit_y = 1.75*(-1*factor[sort_keys[0]]) + zeromedian
ax1.set_ylim(lowlimit_y, uplimit_y)
ax1.invert_yaxis()

lclist = []
for key in rrl:
    base = -1*factor[key] - rrl[key][magstr].median() + zeromedian
    mag = rrl[key][magstr]+base

    lclist.append([ax1.plot(compute_phase(key, init_period), mag, '.',
                  c=colours[key], ms=5, alpha=0.95, label=key+" + (%.1f)" % base), key])
    lclist.append([ax1.plot(compute_phase(key, init_period)+1,
                  mag, '.', c=colours[key], ms=5, alpha=0.95), key])

ax1.set_xlim(0, 2)
ax1.legend(loc=(1.025, 0.025), fontsize=12)

axfreq = fig.add_axes([0.25, 0.01, 0.5, 0.025])
freq_slider = Slider(
    ax=axfreq,
    label="Period [d]",
    valmin=init_period*0.95,
    valmax=init_period*1.05,
    valinit=init_period,
    valstep=0.0000005,
    valfmt='%.9f'
)

freq_slider.val = init_period
freq_slider.valinit = init_period

freq_slider.on_changed(update)

resetax = fig.add_axes([0.91, 0.5, 0.075, 0.045])
button = Button(resetax, 'Reset', hovercolor='0.975')
button.on_clicked(reset)

id1 = fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)

axbox = fig.add_axes([0.65, 0.935, 0.15, 0.04])
text_box = TextBox(axbox, "Period step", textalignment="center")
text_box.label.set_fontsize(16)
text_box.on_submit(submit_step)
currentstep = fig.text(0.65, 0.91, s=("Current: %.9f" %
                       freq_slider.valstep), fontsize=15)

axbox2 = fig.add_axes([0.2, 0.935, 0.15, 0.04])
text_box2 = TextBox(axbox2, "Period", textalignment="center")
text_box2.label.set_fontsize(16)
text_box2.on_submit(submit_period)
currentperiod = fig.text(0.2, 0.91, s=("Current: %.9f" %
                         freq_slider.val), fontsize=15)

confirmax = fig.add_axes([0.85, 0.935, 0.125, 0.045])
button2 = Button(confirmax, 'Confirm best period',
                 color='#99ff99', hovercolor='#00cc66')
button2.label.set_fontsize(14)
button2.on_clicked(confirm)


ax2 = fig.add_subplot(221)

epoch = rrl[zerokey][datestr]
mag = rrl[zerokey][magstr]
mag_err = rrl[zerokey][magerrstr]
freq, power = LombScargle(epoch, mag, mag_err).autopower(
    minimum_frequency=1, maximum_frequency=5)

periodogram = ax2.plot(1/freq, power)
ax2.set_ylim([min(power), max(power)*1.2])
ax2.set_xlim([min(1/freq), max(1/freq)])
ax2.set_xlabel("Period [d]", size=15, y=1.05)
ax2.set_ylabel("Power", size=15, x=0.25)
ax2.plot(1/freq[np.argmax(power)], max(power), 'r.', ms=5)

ax2.axvline(init_period, 0, 1, color='k', ls=(0, (5, 5)), alpha=0.5)

id2 = fig.canvas.mpl_connect('button_release_event', click_image_control)


ax3 = fig.add_subplot(223)

ax3.set_xlabel("Epoch", size=15, y=1.05)
ax3.set_ylabel("Magnitude", size=15, x=0.25)

ax3.xaxis.set_minor_locator(AutoMinorLocator())
ax3.yaxis.set_minor_locator(AutoMinorLocator())
ax3.grid(which='minor', alpha=0.5, linestyle=(0, (5, 10)), linewidth=0.5)
ax3.grid(which='major', alpha=0.85, linestyle=(0, (5, 10)), linewidth=0.5)
ax3.invert_yaxis()

for key in rrl:
    base = -1*factor[key] - rrl[key][magstr].median() + zeromedian
    mag = rrl[key][magstr]+base
    epoch = rrl[key][datestr]

    ax3.plot(epoch, mag, '.', c=colours[key], ms=7, alpha=0.95)


plt.show()
