import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import plotterMethods as cplt

mpl.rcParams['font.family'] = "Helvetica"
mpl.rcParams['font.size'] = 18

dTimesData = np.genfromtxt('pes/butaDTimes.dat')
en = np.genfromtxt('pes/butaPES.dat')
S1 = en[:, 3]
S0 = en[:, 2]
CI = np.argmin(np.abs(S1-S0))
xCI = en[CI, 0]
yCI = en[CI, 1]
print(xCI, yCI)
dTimes = dTimesData[:, 2].reshape(1001, 1001)*0.02418884254
X = dTimesData[:, 0].reshape(1001, 1001)
Y = dTimesData[:, 1].reshape(1001, 1001)
spawns_f = np.genfromtxt('spawns/spawngeoms_f.dat')
spawns_t = np.genfromtxt('spawns/spawngeoms_t.dat')
fig = plt.figure(figsize=(2550./300., 3300./300.), dpi=300)
ax = fig.add_subplot(1,1,1)
args = {'levels': 200}
l1=cplt.plotContour(ax,[X,Y,dTimes],args,colorMap="viridis")
ax.plot(spawns_t[:,0], spawns_t[:,1],  "s",
        markerfacecolor='xkcd:aqua', markeredgecolor='xkcd:turquoise', 
        label='$\\tau_\mathrm{D} < 2 \\tau_\mathrm{AIMS}$',
        markersize=11)
ax.plot(spawns_f[:,0], spawns_f[:,1], "o",
        markerfacecolor='xkcd:grey', markeredgecolor='xkcd:black',
        label='$\\tau_\mathrm{D} \geq 2 \\tau_\mathrm{AIMS}$',
        markersize=11)
l5,=ax.plot(xCI, yCI, "o", markeredgecolor='xkcd:vermillion',
        markerfacecolor=(0,0,0,0), markersize=12, markeredgewidth=3,
        label='CI')
l6,=ax.plot(5.10, 0.0, "<", markeredgecolor='xkcd:orange',  markersize=13,
        markerfacecolor='xkcd:orange', label='FC')
ax.set_xlabel("X (bohr)")
ax.set_ylabel("Y (bohr)")
cb = fig.colorbar(l1,ax=ax,orientation='horizontal',location = "top", shrink=0.75)
cb.set_ticks(np.arange(1.9,2.7,0.2))
#fig.legend(loc='upper right', bbox_to_anchor=(0.9,0.725),
#           frameon=False, labelcolor='xkcd:white',
#           handlelength=1.)
plt.savefig("all.png")
plt.show()
