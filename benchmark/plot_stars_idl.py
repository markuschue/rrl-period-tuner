import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_lcv_gaia_individual(data):
    tab = pd.read_csv(data)

    print(tab)

    id_star = tab['FIELD01']
    band = tab['FIELD03']
    time = tab['FIELD04']
    mag = tab['FIELD05']
    flux = tab['FIELD06']
    flux_err = tab['FIELD07']

    mag_g = mag[(band == 'G') & (mag > 0)]
    mag_bp = mag[(band == 'BP') & (mag > 0)]
    mag_rp = mag[(band == 'RP') & (mag > 0)]

    time_g = time[(band == 'G') & (time > 0)]
    time_bp = time[(band == 'BP') & (time > 0)]
    time_rp = time[(band == 'RP') & (time > 0)]

    flux_g = flux[(band == 'G') & (flux > 0)]
    flux_bp = flux[(band == 'BP') & (flux > 0)]
    flux_rp = flux[(band == 'RP') & (flux > 0)]

    flux_err_g = flux_err[(band == 'G') & (flux_err > 0)]
    flux_err_bp = flux_err[(band == 'BP') & (flux_err > 0)]
    flux_err_rp = flux_err[(band == 'RP') & (flux_err > 0)]

    err_mag_g = np.abs(-2.5 / flux_g / np.log(10)) * flux_err_g
    err_mag_bp = np.abs(-2.5 / flux_bp / np.log(10)) * flux_err_bp
    err_mag_rp = np.abs(-2.5 / flux_rp / np.log(10)) * flux_err_rp

    id2 = tab['FIELD001']
    ra = tab['FIELD003']
    dec = tab['FIELD004']
    gmean = tab['FIELD019']

    w = np.nonzero(id2 == data)[0]

    types = ['RR_Lyrae', 'Cepheid', 'Transit', 'Short_period', 'Long_period', 'Eclipsing_binary',
             'Rotation', 'MS_oscillator', 'AGN', 'Microlensing', 'Compact_companion']
    vec2 = [tab['FIELD125'][w], tab['FIELD126'][w], tab['FIELD127'][w], tab['FIELD128'][w], tab['FIELD129'][w], tab['FIELD130'][w],
            tab['FIELD131'][w], tab['FIELD132'][w], tab['FIELD133'][w], tab['FIELD134'][w], tab['FIELD135'][w]]

    # Flag is the type of star whose index has "true" in vec2:
    flag = [types[i] for i in range(len(types)) if vec2[i] == 'true'][0]

    if len(flag) == 0:
        flag = 'uncl'
    if len(flag) > 1:
        print(f"type greater than 1 for star {data}")
        flag = 'double'

    best = tab['FIELD052']

    best[best == 'DSCT|GDOR|SXPHE'] = 'DSCT_GDOR_SXPHE'
    best[best == ''] = 'UNCL'
    best[best == 'RR'] = 'RR_Lyrae'
    best[best == 'CEP'] = 'Cepheid'

    flag = best[w]

    tab.loc[tab['FIELD089'] > 0, 'FIELD089'] = 1. / \
        tab['FIELD089'][tab['FIELD089'] > 0]
    tab.loc[tab['FIELD117'] > 0, 'FIELD117'] = 1. / \
        tab['FIELD117'][tab['FIELD117'] > 0]
    tab.loc[tab['FIELD119'] > 0, 'FIELD119'] = 1. / \
        tab['FIELD119'][tab['FIELD119'] > 0]

    vec = [tab['FIELD056'][w], tab['FIELD058'][w], tab['FIELD082'][w],
           tab['FIELD093'][w], tab['FIELD095'][w], tab['FIELD089'][w],
           tab['FIELD117'][w], tab['FIELD119'][w]]

    period = vec[np.nonzero([len(vec[i]) > 0 for i in range(len(vec))])][0]

    with open('resumen.txt', 'a') as z:
        if len(period) == 0:
            print(f"No Gaia period found for star {data} {flag}")
            with open('no_period.csv', 'a') as u:
                u.write(f"{data},{ra[w].values[0]},{
                        dec[w].values[0]},{gmean[w].values[0]},{flag}\n")
            z.write(f"No Gaia period found for star {data} {ra[w].values[0]} {
                    dec[w].values[0]} {gmean[w].values[0]} {flag}\n")
            return
        else:
            print(f"Gaia period found for star {data} {period[0]} {flag}")
            with open('si_period.csv', 'a') as v:
                v.write(f"{data},{ra[w].values[0]},{dec[w].values[0]},{
                        gmean[w].values[0]},{flag},{period[0]}\n")
            z.write(f"Gaia period found for star {data} {ra[w].values[0]} {
                    dec[w].values[0]} {gmean[w].values[0]} {period[0]} {flag}\n")

    thre = 0.45

    for time_band, mag_band, err_mag_band, num_w_band in zip([time_g, time_bp, time_rp], [mag_g, mag_bp, mag_rp], [err_mag_g, err_mag_bp, err_mag_rp], [len(mag_g), len(mag_bp), len(mag_rp)]):
        for j in range(min([num_w_band - 1, 4])):
            srt = np.argsort(mag_band)
            mag_band = mag_band[srt]
            time_band = time_band[srt]
            if num_w_band > 2:
                if abs(mag_band[0] - mag_band[1]) > thre or abs(mag_band[0] - np.mean(mag_band)) > thre:
                    mag_band = mag_band[1:]
                    time_band = time_band[1:]
                num_w_band -= 1

    min_g = np.min(mag_g)
    zphase_g = time_g[np.argmin(mag_g)]
    ph_g = abs(((time_g - zphase_g) /
               period[0]) - np.floor((time_g - zphase_g) / period[0])) + 0.5
    max_g = np.max(mag_g)

    min_bp = np.min(mag_bp)
    zphase_bp = time_bp[np.argmin(mag_bp)] - zphase_g
    dph_bp = zphase_g - zphase_bp
    ph_bp = abs(((time_bp - zphase_bp) / period[0]) - np.floor(
        (time_bp - zphase_bp) / period[0])) + 0.5 - (dph_bp / period[0] - np.floor(dph_bp / period[0]))
    max_bp = np.max(mag_bp)

    min_rp = np.min(mag_rp)
    zphase_rp = time_rp[np.argmin(mag_rp)] - zphase_g
    dph_rp = zphase_g - zphase_rp
    ph_rp = abs(((time_rp - zphase_rp) / period[0]) - np.floor(
        (time_rp - zphase_rp) / period[0])) + 0.5 - (dph_rp / period[0] - np.floor(dph_rp / period[0]))
    max_rp = np.max(mag_rp)

    c1, c2, c3 = "sea green", "blue", "red"
    bp_shift, rp_shift = 0.75, -0.75
    xra = [-0.3, 1.3]
    thk = 3
    nsig = 3.0
    minimo = [min_g, min_bp, min_rp]
    maximo = [max_g, max_bp, max_rp]
    delt = maximo - minimo
    delta = 0.0

    ws = 0
    if ws == 1:
        fig, ax = plt.subplots(figsize=(18, 18))
    else:
        fig, ax = plt.subplots(figsize=(8, 8))

    midpt = np.mean([np.mean(mag_g), np.mean(mag_bp), np.mean(mag_rp)])
    factor = 1.4
    extra1 = midpt - np.mean(mag_bp)
    extra2 = midpt - np.mean(mag_rp)

    ax.errorbar(ph_bp + bp_shift, mag_bp + extra1, err_mag_bp,
                fmt='o', markersize=5, color=c2, ecolor=c2, alpha=0.8)
    ax.errorbar(ph_rp + rp_shift, mag_rp + extra2, err_mag_rp,
                fmt='o', markersize=5, color=c3, ecolor=c3, alpha=0.8)
    ax.errorbar(ph_g, mag_g, err_mag_g, fmt='o',
                markersize=5, color=c1, ecolor=c1, alpha=0.8)

    ax.set_ylim(midpt + delt[0] * factor, midpt - delt[0] * factor)
    ax.set_xlim(-0.3, 1.3)
    ax.invert_yaxis()
    ax.grid(True)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Mag")
    ax.legend(['G', 'BP', 'RP'])
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        plot_lcv_gaia_individual(sys.argv[1])
    else:
        print("Usage: python plot_stars.py <data>")
        sys.exit(1)
