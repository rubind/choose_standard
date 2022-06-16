import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import subprocess
import sys
from FileRead import readcol
from scipy.interpolate import interp1d
from random import choice

def eval_comb(items):
    jacobian = np.zeros([tot_standards, 5], dtype=np.float64)
    
    
    for i in range(tot_standards):

        jacobian[i, 0] = items[i][3]
        jacobian[i, 1] = 1.
        jacobian[i, 2] = items[i][4]
        jacobian[i, 3] = items[i][5]
        jacobian[i, 4] = items[i][6]

        
    wmat = np.dot(jacobian.T, jacobian)
    wmat += np.diag(0.00001*np.ones(5, dtype=np.float64))
    cmat = np.linalg.inv(wmat)

    diag_vals = np.sqrt(np.diag(cmat))
    return np.prod(diag_vals)
    
    
min_mag = 5
max_mag = 14
stds_per_hour = 4
#start_time_hours = -6.0 # -2.5 = 9:30 pm
min_alt = 30.

date_midnight = sys.argv[1]
n_tries = int(sys.argv[2])
tot_standards = int(sys.argv[3])
start_time_hours = float(sys.argv[4])



MLO = EarthLocation(lat=19.5364*u.deg, lon=-155.5765*u.deg, height=3400*u.m)
print(MLO)

utcoffset = -10*u.hour
midnight = Time(date_midnight + ' 00:00:01') - utcoffset # E.g., 2020-09-25


f = open("/Users/" + subprocess.getoutput("whoami") + "/Dropbox/SCP_Stuff/calspec/stars_for_wfc3/all_coords.csv", 'r')
lines = f.read().split('\n')
f.close()


[x, y] = readcol("/Users/" + subprocess.getoutput("whoami") + "/Dropbox/SCP_Stuff/calspec/alpha_lyr_stis_010.ascii", 'ff')
vegafn = interp1d(x, y, kind = 'linear')

std_RA_Dec_airmass_colors = []


cur_time = start_time_hours*1.
all_mags = {}

for i in tqdm.trange(tot_standards):
    std_RA_Dec_airmass_colors.append([])

    for line in lines:

        parsed = line.split(",")
        print("parsed", parsed)
        this_mag = float(parsed[2])

        if this_mag >= min_mag and this_mag <= max_mag:
            tmpitem = SkyCoord(parsed[5].strip() + " " + parsed[6].strip(), unit=(u.hourangle, u.deg), frame="icrs")
            frame_July13night = AltAz(obstime=midnight + cur_time * u.hour,
                                      location=MLO)

            tmp_altaz = tmpitem.transform_to(frame_July13night)

            if tmp_altaz.alt.deg > min_alt:
                [x, y] = readcol("/Users/" + subprocess.getoutput("whoami") + "/Dropbox/SCP_Stuff/calspec/" + parsed[7].strip() + ".ascii", 'ff')
                ifn = interp1d(x, y, kind = 'linear', bounds_error = False)

                U = -2.5*np.log10(np.mean(ifn(np.arange(3400., 3600.)))/np.mean(vegafn(np.arange(3400., 3600.))))
                B = -2.5*np.log10(np.mean(ifn(np.arange(4400., 4600.)))/np.mean(vegafn(np.arange(4400., 4600.))))
                V = -2.5*np.log10(np.mean(ifn(np.arange(5400., 5600.)))/np.mean(vegafn(np.arange(5400., 5600.))))
                I = -2.5*np.log10(np.mean(ifn(np.arange(7400., 7600.)))/np.mean(vegafn(np.arange(7400., 7600.))))


                std_RA_Dec_airmass_colors[-1].append([parsed[0], parsed[5].strip(), parsed[6].strip(), float(tmp_altaz.secz), U - V, B - V, V - I])
                all_mags[parsed[0]] = V
    plt.plot(cur_time, len(std_RA_Dec_airmass_colors[-1]), '.', color = 'b')
    cur_time += 1./stds_per_hour
plt.savefig("possible_standards_by_slot.pdf")
plt.close()


print(std_RA_Dec_airmass_colors)





best_val = 1e20


for i in tqdm.trange(n_tries):
    
    items = [choice(item) for item in std_RA_Dec_airmass_colors]
    
    this_val = eval_comb(items)
    if this_val < best_val:
        best_items = items
        best_val = this_val
        print("best ", best_val)
        for line in best_items:
            print(line[0], line[1].replace(" ", ":"), line[2].replace(" ", ":"), all_mags[line[0]], line[3])

        plt.figure(figsize = (16, 16))
        for j in range(4):
            for k in range(4):
                if k > j:
                    plt.subplot(4, 4, k*4 + 1 + j)
                    for line in best_items:
                        plt.plot(line[3 + j], line[3 + k], '.', color = 'b')
                        plt.text(line[3 + j], line[3 + k], line[0], fontsize = 4)
                        plt.xlabel(["Airmass", "$U-V$", "$B-V$", "V-I"][j])
                        plt.ylabel(["Airmass", "$U-V$", "$B-V$", "V-I"][k])
                        
        plt.savefig("color_color_best.pdf", bbox_inches = 'tight')
        plt.close()

