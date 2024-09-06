import glob
import os
import sys
from datetime import datetime
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.timeseries import LombScargle
from matplotlib.ticker import AutoMinorLocator
from matplotlib.widgets import Button, Slider, TextBox

from utils.data_utils import get_star_photometry
from utils.period_utils import compute_period, prepareTable


class PeriodPlotter:
    def __init__(self, base_dir: str) -> None:
        self.bdir = base_dir
        self.datestr = "DATE-OBS"
        self.magstr, self.magerrstr, self.final, self.strfilt = self._get_field_names()
        self.obj = self._get_object_name()
        self.varid = self._load_varid()
        self.data = get_star_photometry(self.bdir)
        # self.data = {key: self.data[key]
        #              for key in self.data if key.startswith('Gaia')}

        self.colours, self.factor, self.sort_keys, self.zphase, self.zerokey, self.zeromedian = self._initialize_plot_data()
        self._compute_initial_period()

        self.fig, self.phase_plot, self.lclist = self._initialize_fig()
        self.freq_slider = self._initialize_slider()
        self._initialize_reset_button()
        self.currentstep, self.currentperiod = self._initialize_text_boxes()
        self._initialize_confirm_period_button()
        self.periodogram = self._initialize_periodogram()

        self.fig.canvas.mpl_connect(
            'button_release_event', self.click_image_control)

        self.lightcurve = self._initialize_lightcurve()

        self.fig.canvas.mpl_connect(
            'key_release_event', self.arrow_key_image_control)

    def _get_field_names(self) -> tuple[str, str, str, int]:
        if len(glob.glob(self.bdir + "*final.fits")) > 0:
            return "MAG_FINAL", "MAGERR_FINAL", "final", -7
        else:
            return "MAG_AUTO_NORM", "MAGERR_AUTO", "", -1

    def _get_object_name(self) -> str:
        return os.path.basename(os.path.normpath(self.bdir))

    def _load_varid(self) -> Any:
        return Table.read(self.bdir + f"finalIDs_{self.obj}.fits", hdu=1)[0][0]

    def _load_catalogs(self, files: list[str]) -> Dict[str, pd.DataFrame]:
        data = {}
        for line in files:
            key = os.path.splitext(line)[0][self.strfilt]
            data[key] = prepareTable(line, 1)
        return data

    def _initialize_plot_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, float], pd.Series, str, float]:
        colours = plt.cm.rainbow(np.linspace(0, 1, len(self.data)))
        factor = np.linspace(0, len(self.data) - 1,
                             len(self.data)) - np.floor(len(self.data) / 2)

        amplitude = {key: np.abs(self.data[key][self.magstr].min(
        ) - self.data[key][self.magstr].max()) for key in self.data}
        sort_keys = [list(amplitude.keys())[list(amplitude.values()).index(ii)]
                     for ii in sorted(amplitude.values(), reverse=True)]

        colours = {key: ii for key, ii in zip(sort_keys, colours)}
        factor = {key: ii for key, ii in zip(sort_keys, factor)}
        zerokey = list(factor.keys())[list(factor.values()).index(0)]
        zeromedian = self.data[zerokey][self.magstr].median()

        zphase = self.data[zerokey].loc[self.data[zerokey][self.magstr]
                                        == self.data[zerokey][self.magstr].min(), self.datestr]

        return colours, factor, sort_keys, zphase, zerokey, zeromedian

    def _initialize_fig(self):
        fig = plt.figure(figsize=(1920 / (96 * 1.25),
                         1080 / (96 * 1.25)), dpi=100)
        phase_plot = fig.add_subplot(122)

        fig.suptitle(self.obj.split("_")[
                     1][:-3] + " " + self.obj.split("_")[1][-3:], size=24, y=0.925)
        phase_plot.set_xlabel("Phase", size=15, y=1.05)
        phase_plot.set_ylabel("Magnitude", size=15, x=0.25)
        phase_plot.xaxis.set_minor_locator(AutoMinorLocator())
        phase_plot.yaxis.set_minor_locator(AutoMinorLocator())
        phase_plot.grid(which='minor', alpha=0.5,
                        linestyle=(0, (5, 10)), linewidth=0.5)
        phase_plot.grid(which='major', alpha=0.85,
                        linestyle=(0, (5, 10)), linewidth=0.5)
        lowlimit_y = 1.75 * \
            (-1*self.factor[self.sort_keys[-1]]) + self.zeromedian
        uplimit_y = 1.75*(-1*self.factor[self.sort_keys[0]]) + self.zeromedian
        phase_plot.set_ylim(lowlimit_y, uplimit_y)
        phase_plot.invert_yaxis()

        lclist = []

        for key in self.data:
            base = -1 * \
                self.factor[key] - self.data[key][self.magstr].median() + \
                self.zeromedian
            mag = self.data[key][self.magstr] + base

            lclist.append([phase_plot.plot(self.compute_phase(key, self.init_period), mag, '.',
                                           c=self.colours[key], ms=5, alpha=0.95, label=key + " + (%.1f)" % base), key])
            lclist.append([phase_plot.plot(self.compute_phase(key, self.init_period) + 1,
                                           mag, '.', c=self.colours[key], ms=5, alpha=0.95), key])

        phase_plot.set_xlim(0, 2)
        phase_plot.legend(loc=(1.025, 0.025), fontsize=12)

        return fig, phase_plot, lclist

    def _initialize_slider(self) -> Slider:
        axfreq = self.fig.add_axes([0.25, 0.01, 0.5, 0.025])
        slider = Slider(
            ax=axfreq,
            label="Period [d]",
            valmin=0,
            valmax=1,
            valinit=self.init_period,
            valstep=0.0005,
            valfmt='%.9f'
        )
        slider.val = self.init_period
        slider.valinit = self.init_period
        slider.on_changed(self.update)
        return slider

    def _initialize_text_boxes(self) -> Tuple[plt.Text, plt.Text]:
        axbox = self.fig.add_axes([0.65, 0.935, 0.15, 0.04])
        text_box = TextBox(axbox, "Period step", textalignment="center")
        text_box.label.set_fontsize(16)
        text_box.on_submit(self.submit_step)
        currentstep = self.fig.text(0.65, 0.91, s=(
            "Current: %.9f" % self.freq_slider.valstep), fontsize=15)

        axbox2 = self.fig.add_axes([0.2, 0.935, 0.15, 0.04])
        text_box2 = TextBox(axbox2, "Period", textalignment="center")
        text_box2.label.set_fontsize(16)
        text_box2.on_submit(self.submit_period)
        currentperiod = self.fig.text(0.2, 0.91, s=(
            "Current: %.9f" % self.freq_slider.val), fontsize=15)

        return currentstep, currentperiod

    def _initialize_confirm_period_button(self) -> None:
        confirmax = self.fig.add_axes([0.85, 0.935, 0.125, 0.045])
        button = Button(confirmax, 'Confirm best period',
                        color='#99ff99', hovercolor='#00cc66')
        button.label.set_fontsize(14)
        button.on_clicked(self.confirm)

    def _compute_initial_period(self) -> None:
        start = datetime.now()
        self.best_period, self.possible_periods = compute_period(
            self.data[self.zerokey], self.datestr, self.magstr, self.magerrstr)
        print(str(datetime.now() - start) +
              " s elapsed while computing period")

        self.rr = Table.read('.\\data\\photometry\\periods.txt',
                             format='ascii', names=["star", "period"]).to_pandas()
        period_tabulated = self.rr.loc[self.rr["star"].str.contains(
            self.obj.split("_")[0]), "period"].iloc[0]

        print(f"\nComputed period {self.best_period:.9f}, tabulated period {
              period_tabulated:.9f}")
        print("Possible periods")
        print(self.possible_periods)

        self.init_period = period_tabulated

    def _initialize_periodogram(self) -> plt.Axes:
        periodogram_plot = self.fig.add_subplot(221)

        epoch = self.data[self.zerokey][self.datestr]
        mag = self.data[self.zerokey][self.magstr]
        mag_err = self.data[self.zerokey][self.magerrstr]
        freq, power = LombScargle(epoch, mag, mag_err).autopower(
            minimum_frequency=1, maximum_frequency=5)

        periodogram_plot.plot(1/freq, power)
        periodogram_plot.set_ylim([min(power), max(power)*1.2])
        periodogram_plot.set_xlim([min(1/freq), max(1/freq)])
        periodogram_plot.set_xlabel("Period [d]", size=15, y=1.05)
        periodogram_plot.set_ylabel("Power", size=15, x=0.25)
        periodogram_plot.plot(1/freq[np.argmax(power)], max(power), 'r.', ms=5)

        periodogram_plot.axvline(self.init_period, 0, 1, color='k',
                                 ls=(0, (5, 5)), alpha=0.5)

        return periodogram_plot

    def _initialize_lightcurve(self) -> plt.Axes:
        lightcurve_plot = self.fig.add_subplot(223)

        lightcurve_plot.set_xlabel("Epoch", size=15, y=1.05)
        lightcurve_plot.set_ylabel("Magnitude", size=15, x=0.25)

        lightcurve_plot.xaxis.set_minor_locator(AutoMinorLocator())
        lightcurve_plot.yaxis.set_minor_locator(AutoMinorLocator())
        lightcurve_plot.grid(which='minor', alpha=0.5,
                             linestyle=(0, (5, 10)), linewidth=0.5)
        lightcurve_plot.grid(which='major', alpha=0.85,
                             linestyle=(0, (5, 10)), linewidth=0.5)
        lightcurve_plot.invert_yaxis()

        for key in self.data:
            base = -1*self.factor[key] - \
                self.data[key][self.magstr].median() + self.zeromedian
            mag = self.data[key][self.magstr]+base
            epoch = self.data[key][self.datestr]

            lightcurve_plot.plot(
                epoch, mag, '.', c=self.colours[key], ms=7, alpha=0.95)

        return lightcurve_plot

    def _initialize_reset_button(self) -> None:
        axreset = self.fig.add_axes([0.8, 0.01, 0.1, 0.04])
        button = Button(axreset, "Reset", hovercolor="0.975")
        button.on_clicked(self.reset)

    def compute_phase(self, key: str, period: float) -> np.ndarray:
        return ((self.data[key][self.datestr] - self.zphase.iloc[0]) / period) % 1

    def update(self, val: float) -> None:
        for ii in range(len(self.lclist)):
            if ii % 2 == 0:
                self.lclist[ii][0][0].set_xdata(
                    self.compute_phase(self.lclist[ii][1], self.freq_slider.val))
            else:
                self.lclist[ii][0][0].set_xdata(self.compute_phase(
                    self.lclist[ii][1], self.freq_slider.val)+1)
        self.fig.canvas.draw_idle()

    def arrow_key_image_control(self, event: Any) -> None:
        if event.key == 'left':
            new_val = self.freq_slider.val-self.freq_slider.valstep
            self.update(new_val)
            self.freq_slider.set_val(new_val)
            self.periodogram.get_lines()[-1].remove()
            self.periodogram.axvline(new_val, 0, 1, color='k',
                                     ls=(0, (5, 5)), alpha=0.5)
        elif event.key == 'right':
            new_val = self.freq_slider.val+self.freq_slider.valstep
            self.update(new_val)
            self.freq_slider.set_val(new_val)
            self.periodogram.get_lines()[-1].remove()
            self.periodogram.axvline(new_val, 0, 1, color='k',
                                     ls=(0, (5, 5)), alpha=0.5)

    def click_image_control(self, event: Any) -> None:
        if event.inaxes == self.periodogram:
            new_val = event.xdata
            self.periodogram.get_lines()[-1].remove()

            self.periodogram.axvline(new_val, 0, 1, color='k',
                                     ls=(0, (5, 5)), alpha=0.5)
            self.update(new_val)
            self.freq_slider.set_val(new_val)

        self.currentperiod.set_text("Current: %.9f" % self.freq_slider.val)

        return None

    def reset(self, event: Any) -> None:
        print("Resetting period to initial value")
        self.freq_slider.reset()
        self.freq_slider.valstep = 0.0000005
        self.freq_slider.val = self.init_period
        self.freq_slider.valmin = self.init_period*0.95
        self.freq_slider.valmax = self.init_period*1.05
        self.currentstep.set_text("Current: %.9f" % self.freq_slider.valstep)
        self.currentperiod.set_text("Current: %.9f" % self.freq_slider.valinit)

        self.periodogram.get_lines()[-1].remove()
        self.periodogram.axvline(self.init_period, 0, 1, color='k',
                                 ls=(0, (5, 5)), alpha=0.5)

    def submit_step(self, text: str) -> None:
        try:
            step = float(text)
            self.freq_slider.valstep = step
            self.currentstep.set_text("Current: %.9f" % step)
        except ValueError:
            print(f"Invalid step value: {text}")

    def submit_period(self, new_period: str) -> None:
        print(f"New period: {new_period}")
        try:
            self.freq_slider.val = float(new_period)
            self.freq_slider.valmin = float(new_period)*0.95
            self.freq_slider.valmax = float(new_period)*1.05
            self.freq_slider.set_val(float(new_period))
            self.update(float(new_period))
            self.currentperiod.set_text("Current: %.9f" % self.freq_slider.val)

            self.periodogram.get_lines()[-1].remove()
            self.periodogram.axvline(float(new_period), 0, 1,
                                     color='k', ls=(0, (5, 5)), alpha=0.5)
        except ValueError:
            print(f"Invalid period value: {self.text}")

    def confirm(self, event: Any) -> None:
        self.fig.text(
            0.85, 0.91, "New period saved into file!", size=15)

        self.rr.loc[self.rr["star"].str.contains(
            self.obj.split("_")[0]), "period"] = self.freq_slider.val
        Table.from_pandas(self.rr).write(
            '.\\data\\photometry\\periods.txt', format='ascii', overwrite=True)

        return None


if __name__ == "__main__":
    base_dir = ".\\data\\photometry\\GAIA03_1360607637502749440\\"
    plotter = PeriodPlotter(base_dir)
    plt.show()