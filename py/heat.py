import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math
from matplotlib import ticker
import sys

def read_file(fname):
  count = 0;
  lim = 9;
  arr = [];
  f = open(fname);
  for line in f:
    if(count>=lim):
      arr.append([float(x) for x in line.split()])
    count = count + 1
  arr = np.asarray(arr);
  f.close()
  return arr

prefix = sys.argv[1]

# Extract data

mods = ['dsmc/',
        'rndNRG/','rndRW/','rndLGZ/',
        'sepNRG/','sepRW/','sepLGZ/',
        'rndNRGmin/','rndRWmin/','rndLGZmin/',
        'sepNRGmin/','sepRWmin/','sepLGZmin/'];
cases = ['p0/','p1/','p2/','p3/','p4/','p5/','p6/','p7/','p8/']

nmods = np.size(mods)
ncases = np.size(cases)
ncells = 5000

T = np.zeros((nmods,ncases,ncells));
q = np.zeros((nmods,ncases,ncells));

qm = np.zeros((nmods,ncases))
qsH = np.zeros((nmods,ncases));
qsL = np.zeros((nmods,ncases));
vs = np.zeros((nmods,ncases));

Ly = 4.0e-5
for i in range(0,nmods):
  for j in range(0,ncases):
    fn = 40000000
    smod = mods[i]
    scase = cases[j]

    fname = prefix+smod+scase+'state.'+str(fn)
    print(fname)
    state = read_file(fname)
    state = state[state[:,0].argsort()]
    T[i,j,:] = state[:,4]

    fname = prefix+smod+scase+'vel.'+str(fn)
    vel = read_file(fname)
    vel = vel[vel[:,0].argsort()]
    v = vel[:,3];
    vs[i,j] = (np.abs(v[-1])+np.abs(v[0]))*0.5

    fname = prefix+smod+scase+'vmom.'+str(fn)
    vmom = read_file(fname)
    vmom = vmom[vmom[:,0].argsort()]
    q[i,j,:] = vmom[:,9];
    qm[i,j] = np.mean(vmom[10:-10,9])
    qsH[i,j] = q[i,j,-1]
    qsL[i,j] = q[i,j,0]

#################################################################
# Plot high pressure
fig,ax = plt.subplots(2,3);

x = np.linspace(0,1,ncells)

# Temperature
ax[0,0].plot(x,T[0,0,:],'-k',markersize=8,label=r'$DSMC$')
ax[0,1].plot(x,T[0,0,:],'-k',markersize=8,label=r'$DSMC$')
ax[0,2].plot(x,T[0,0,:],'-k',markersize=8,label=r'$DSMC$')

ax[0,0].plot(x,T[1,0,:],'-r',markersize=8,label=r'$SWPM$')
ax[0,1].plot(x,T[2,0,:],'-r',markersize=8,label=r'$SWPM$')
ax[0,2].plot(x,T[3,0,:],'-r',markersize=8,label=r'$SWPM$')

ax[0,0].plot(x,T[4,0,:],'--g',markersize=8,label=r'$SWPM^G$')
ax[0,1].plot(x,T[5,0,:],'--g',markersize=8,label=r'$SWPM^G$')
ax[0,2].plot(x,T[6,0,:],'--g',markersize=8,label=r'$SWPM^G$')

ax[0,0].plot(x,T[7,0,:],':b',markersize=8,label=r'$SWPM^W$')
ax[0,1].plot(x,T[8,0,:],':b',markersize=8,label=r'$SWPM^W$')
ax[0,2].plot(x,T[9,0,:],':b',markersize=8,label=r'$SWPM^W$')

ax[0,0].plot(x,T[10,0,:],'-.c',markersize=8,label=r'$SWPM^{GW}$')
ax[0,1].plot(x,T[11,0,:],'-.c',markersize=8,label=r'$SWPM^{GW}$')
ax[0,2].plot(x,T[12,0,:],'-.c',markersize=8,label=r'$SWPM^{GW}$')

for j in range(3):
  ax[0,j].set_xlim([-0.05,1.05])
  ax[0,j].set_ylim([190,460])
  xval, yval = ax[0,j].get_xlim(), ax[0,j].get_ylim()
  xrange = xval[1]-xval[0]
  yrange = yval[1]-yval[0]
  ax[0,j].set_aspect((xrange/yrange), adjustable='box')
  ax[0,j].tick_params(axis='both',direction='in',which='both',right=True,top=True)
  ax[0,j].minorticks_on()

# Heat flux
ax[1,0].plot(x,q[0,0,:]*1e-5,'-k',markersize=8,label=r'$DSMC$')
ax[1,1].plot(x,q[0,0,:]*1e-5,'-k',markersize=8,label=r'$DSMC$')
ax[1,2].plot(x,q[0,0,:]*1e-5,'-k',markersize=8,label=r'$DSMC$')

ax[1,0].plot(x,q[1,0,:]*1e-5,'-r',markersize=8,label=r'$SWPM$')
ax[1,1].plot(x,q[2,0,:]*1e-5,'-r',markersize=8,label=r'$SWPM$')
ax[1,2].plot(x,q[3,0,:]*1e-5,'-r',markersize=8,label=r'$SWPM$')

ax[1,0].plot(x,q[4,0,:]*1e-5,'--g',markersize=8,label=r'$SWPM^G$')
ax[1,1].plot(x,q[5,0,:]*1e-5,'--g',markersize=8,label=r'$SWPM^G$')
ax[1,2].plot(x,q[6,0,:]*1e-5,'--g',markersize=8,label=r'$SWPM^G$')

ax[1,0].plot(x,q[7,0,:]*1e-5,':b',markersize=8,label=r'$SWPM^W$')
ax[1,1].plot(x,q[8,0,:]*1e-5,':b',markersize=8,label=r'$SWPM^W$')
ax[1,2].plot(x,q[9,0,:]*1e-5,':b',markersize=8,label=r'$SWPM^W$')

ax[1,0].plot(x,q[10,0,:]*1e-5,'-.c',markersize=8,label=r'$SWPM^{GW}$')
ax[1,1].plot(x,q[11,0,:]*1e-5,'-.c',markersize=8,label=r'$SWPM^{GW}$')
ax[1,2].plot(x,q[12,0,:]*1e-5,'-.c',markersize=8,label=r'$SWPM^{GW}$')

for j in range(3):
  ax[1,j].set_xlim([-0.05,1.05])
  ax[1,j].set_ylim([-4.0,2.0])
  xval, yval = ax[1,j].get_xlim(), ax[1,j].get_ylim()
  xrange = xval[1]-xval[0]
  yrange = yval[1]-yval[0]
  ax[1,j].set_aspect((xrange/yrange), adjustable='box')
  ax[1,j].tick_params(axis='both',direction='in',which='both',right=True,top=True)
  ax[1,j].minorticks_on()

# Labels
ax[0,0].legend()

ax[0,0].set_ylabel(r'$T$')
ax[1,0].set_ylabel(r'$q\times 10^5$')

ax[1,0].set_xlabel(r'$y/L_y$')
ax[1,1].set_xlabel(r'$y/L_y$')
ax[1,2].set_xlabel(r'$y/L_y$')

ax[0,0].set_title(r'$\mathrm{energy}$')
ax[0,1].set_title(r'$\mathrm{RW}$')
ax[0,2].set_title(r'$\mathrm{LGZ}$')

fig.set_size_inches(14.5, 8.5)

fig.tight_layout(pad=0.5)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig('TqhighP.png')

#################################################################
# Plot low pressure
fig,ax = plt.subplots(2,3);

x = np.linspace(0,1,ncells)

# Temperature
ax[0,0].plot(x,T[0,8,:],'-k',markersize=8,label=r'$DSMC$')
ax[0,1].plot(x,T[0,8,:],'-k',markersize=8,label=r'$DSMC$')
ax[0,2].plot(x,T[0,8,:],'-k',markersize=8,label=r'$DSMC$')

ax[0,0].plot(x,T[1,8,:],'-r',markersize=8,label=r'$SWPM$')
ax[0,1].plot(x,T[2,8,:],'-r',markersize=8,label=r'$SWPM$')
ax[0,2].plot(x,T[3,8,:],'-r',markersize=8,label=r'$SWPM$')

ax[0,0].plot(x,T[4,8,:],'--g',markersize=8,label=r'$SWPM^G$')
ax[0,1].plot(x,T[5,8,:],'--g',markersize=8,label=r'$SWPM^G$')
ax[0,2].plot(x,T[6,8,:],'--g',markersize=8,label=r'$SWPM^G$')

ax[0,0].plot(x,T[7,8,:],':b',markersize=8,label=r'$SWPM^W$')
ax[0,1].plot(x,T[8,8,:],':b',markersize=8,label=r'$SWPM^W$')
ax[0,2].plot(x,T[9,8,:],':b',markersize=8,label=r'$SWPM^W$')

ax[0,0].plot(x,T[10,8,:],'-.c',markersize=8,label=r'$SWPM^{GW}$')
ax[0,1].plot(x,T[11,8,:],'-.c',markersize=8,label=r'$SWPM^{GW}$')
ax[0,2].plot(x,T[12,8,:],'-.c',markersize=8,label=r'$SWPM^{GW}$')

for j in range(3):
  ax[0,j].set_xlim([-0.05,1.05])
  ax[0,j].set_ylim([270,300])
  xval, yval = ax[0,j].get_xlim(), ax[0,j].get_ylim()
  xrange = xval[1]-xval[0]
  yrange = yval[1]-yval[0]
  ax[0,j].set_aspect((xrange/yrange), adjustable='box')
  ax[0,j].tick_params(axis='both',direction='in',which='both',right=True,top=True)
  ax[0,j].minorticks_on()

# Heat flux
ax[1,0].plot(x,q[0,8,:]*1e-3,'-k',markersize=8,label=r'$DSMC$')
ax[1,1].plot(x,q[0,8,:]*1e-3,'-k',markersize=8,label=r'$DSMC$')
ax[1,2].plot(x,q[0,8,:]*1e-3,'-k',markersize=8,label=r'$DSMC$')

ax[1,0].plot(x,q[1,8,:]*1e-3,'-r',markersize=8,label=r'$SWPM$')
ax[1,1].plot(x,q[2,8,:]*1e-3,'-r',markersize=8,label=r'$SWPM$')
ax[1,2].plot(x,q[3,8,:]*1e-3,'-r',markersize=8,label=r'$SWPM$')

ax[1,0].plot(x,q[4,8,:]*1e-3,'--g',markersize=8,label=r'$SWPM^G$')
ax[1,1].plot(x,q[5,8,:]*1e-3,'--g',markersize=8,label=r'$SWPM^G$')
ax[1,2].plot(x,q[6,8,:]*1e-3,'--g',markersize=8,label=r'$SWPM^G$')

ax[1,0].plot(x,q[7,8,:]*1e-3,':b',markersize=8,label=r'$SWPM^W$')
ax[1,1].plot(x,q[8,8,:]*1e-3,':b',markersize=8,label=r'$SWPM^W$')
ax[1,2].plot(x,q[9,8,:]*1e-3,':b',markersize=8,label=r'$SWPM^W$')

ax[1,0].plot(x,q[10,8,:]*1e-3,'-.c',markersize=8,label=r'$SWPM^{GW}$')
ax[1,1].plot(x,q[11,8,:]*1e-3,'-.c',markersize=8,label=r'$SWPM^{GW}$')
ax[1,2].plot(x,q[12,8,:]*1e-3,'-.c',markersize=8,label=r'$SWPM^{GW}$')

for j in range(3):
  ax[1,j].set_xlim([-0.05,1.05])
  ax[1,j].set_ylim([-1.27,-1.25])
  xval, yval = ax[1,j].get_xlim(), ax[1,j].get_ylim()
  xrange = xval[1]-xval[0]
  yrange = yval[1]-yval[0]
  ax[1,j].set_aspect((xrange/yrange), adjustable='box')
  ax[1,j].tick_params(axis='both',direction='in',which='both',right=True,top=True)
  ax[1,j].minorticks_on()

# Labels
ax[0,0].legend()

ax[0,0].set_ylabel(r'$T$')
ax[1,0].set_ylabel(r'$q\times 10^4$')

ax[0,0].set_title(r'$\mathrm{energy}$')
ax[0,1].set_title(r'$\mathrm{RW}$')
ax[0,2].set_title(r'$\mathrm{LGZ}$')

ax[1,0].set_xlabel(r'$y/L_y$')
ax[1,1].set_xlabel(r'$y/L_y$')
ax[1,2].set_xlabel(r'$y/L_y$')

fig.set_size_inches(14.5, 8.5)

fig.tight_layout(pad=0.5)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig('TqlowP.png')

#################################################################
# Plot surface properties
fig,ax = plt.subplots(3,3);

patm = 101325
p = np.array([patm,0.5*patm,
                 patm*1e-1,0.5*patm*1e-1,
                 patm*1e-2,0.5*patm*1e-2,
                 patm*1e-3,0.5*patm*1e-3,
                 patm*1e-4])

vth = np.sqrt(2*1.38e-23*300/3.14159/4.65e-26)

# Mean heat flux
ax[0,0].plot(np.log(p),np.log(np.abs(qm[0,:])),'-k',markersize=8,label=r'$DSMC$')
ax[0,1].plot(np.log(p),np.log(np.abs(qm[0,:])),'-k',markersize=8,label=r'$DSMC$')
ax[0,2].plot(np.log(p),np.log(np.abs(qm[0,:])),'-k',markersize=8,label=r'$DSMC$')

ax[0,0].plot(np.log(p),np.log(np.abs(qm[1,:])),'o-r',markersize=8,label=r'$SWPM$')
ax[0,1].plot(np.log(p),np.log(np.abs(qm[2,:])),'o-r',markersize=8,label=r'$SWPM$')
ax[0,2].plot(np.log(p),np.log(np.abs(qm[3,:])),'o-r',markersize=8,label=r'$SWPM$')

ax[0,0].plot(np.log(p),np.log(np.abs(qm[4,:])),'^--g',markersize=8,label=r'$SWPM^G$')
ax[0,1].plot(np.log(p),np.log(np.abs(qm[5,:])),'^--g',markersize=8,label=r'$SWPM^G$')
ax[0,2].plot(np.log(p),np.log(np.abs(qm[6,:])),'^--g',markersize=8,label=r'$SWPM^G$')

ax[0,0].plot(np.log(p),np.log(np.abs(qm[7,:])),'v:b',markersize=8,label=r'$SWPM^W$')
ax[0,1].plot(np.log(p),np.log(np.abs(qm[8,:])),'v:b',markersize=8,label=r'$SWPM^W$')
ax[0,2].plot(np.log(p),np.log(np.abs(qm[9,:])),'v:b',markersize=8,label=r'$SWPM^W$')

ax[0,0].plot(np.log(p),np.log(np.abs(qm[10,:])),'s-.c',markersize=8,label=r'$SWPM^{GW}$')
ax[0,1].plot(np.log(p),np.log(np.abs(qm[11,:])),'s-.c',markersize=8,label=r'$SWPM^{GW}$')
ax[0,2].plot(np.log(p),np.log(np.abs(qm[12,:])),'s-.c',markersize=8,label=r'$SWPM^{GW}$')

for j in range(3):
  ax[0,j].set_xlim([2,12])
  ax[0,j].set_ylim([7,12])
  xval, yval = ax[0,j].get_xlim(), ax[0,j].get_ylim()
  xrange = xval[1]-xval[0]
  yrange = yval[1]-yval[0]
  ax[0,j].set_aspect((xrange/yrange), adjustable='box')
  ax[0,j].tick_params(axis='both',direction='in',which='both',right=True,top=True)
  ax[0,j].minorticks_on()

# Wall Heat flux
ax[1,0].plot(np.log(p),qsH[0,:]/qm[0,:],'-k',markersize=8,label=r'$DSMC$')
ax[1,1].plot(np.log(p),qsH[0,:]/qm[0,:],'-k',markersize=8,label=r'$DSMC$')
ax[1,2].plot(np.log(p),qsH[0,:]/qm[0,:],'-k',markersize=8,label=r'$DSMC$')

ax[1,0].plot(np.log(p),qsH[1,:]/qm[1,:],'o-r',markersize=8,label=r'$SWPM$')
ax[1,1].plot(np.log(p),qsH[2,:]/qm[2,:],'o-r',markersize=8,label=r'$SWPM$')
ax[1,2].plot(np.log(p),qsH[3,:]/qm[3,:],'o-r',markersize=8,label=r'$SWPM$')

ax[1,0].plot(np.log(p),qsH[4,:]/qm[4,:],'^--g',markersize=8,label=r'$SWPM^G$')
ax[1,1].plot(np.log(p),qsH[5,:]/qm[5,:],'^--g',markersize=8,label=r'$SWPM^G$')
ax[1,2].plot(np.log(p),qsH[6,:]/qm[6,:],'^--g',markersize=8,label=r'$SWPM^G$')

ax[1,0].plot(np.log(p),qsH[7,:]/qm[7,:],'v:b',markersize=8,label=r'$SWPM^W$')
ax[1,1].plot(np.log(p),qsH[8,:]/qm[8,:],'v:b',markersize=8,label=r'$SWPM^W$')
ax[1,2].plot(np.log(p),qsH[9,:]/qm[9,:],'v:b',markersize=8,label=r'$SWPM^W$')

ax[1,0].plot(np.log(p),qsH[10,:]/qm[10,:],'s-.c',markersize=8,label=r'$SWPM^{GW}$')
ax[1,1].plot(np.log(p),qsH[11,:]/qm[11,:],'s-.c',markersize=8,label=r'$SWPM^{GW}$')
ax[1,2].plot(np.log(p),qsH[12,:]/qm[12,:],'s-.c',markersize=8,label=r'$SWPM^{GW}$')

for j in range(3):
  ax[1,j].set_xlim([2,12])
  ax[1,j].set_ylim([-2.4,1.2])
  xval, yval = ax[1,j].get_xlim(), ax[1,j].get_ylim()
  xrange = xval[1]-xval[0]
  yrange = yval[1]-yval[0]
  ax[1,j].set_aspect((xrange/yrange), adjustable='box')
  ax[1,j].tick_params(axis='both',direction='in',which='both',right=True,top=True)
  ax[1,j].minorticks_on()

# Wall Normal velocity
ax[2,0].plot(np.log(p),vs[0,:]/vth*100,'-k',markersize=8,label=r'$DSMC$')
ax[2,1].plot(np.log(p),vs[0,:]/vth*100,'-k',markersize=8,label=r'$DSMC$')
ax[2,2].plot(np.log(p),vs[0,:]/vth*100,'-k',markersize=8,label=r'$DSMC$')

ax[2,0].plot(np.log(p),vs[1,:]/vth*100,'o-r',markersize=8,label=r'$SWPM$')
ax[2,1].plot(np.log(p),vs[2,:]/vth*100,'o-r',markersize=8,label=r'$SWPM$')
ax[2,2].plot(np.log(p),vs[3,:]/vth*100,'o-r',markersize=8,label=r'$SWPM$')

ax[2,0].plot(np.log(p),vs[4,:]/vth*100,'^--g',markersize=8,label=r'$SWPM^G$')
ax[2,1].plot(np.log(p),vs[5,:]/vth*100,'^--g',markersize=8,label=r'$SWPM^G$')
ax[2,2].plot(np.log(p),vs[6,:]/vth*100,'^--g',markersize=8,label=r'$SWPM^G$')

ax[2,0].plot(np.log(p),vs[7,:]/vth*100,'v:b',markersize=8,label=r'$SWPM^W$')
ax[2,1].plot(np.log(p),vs[8,:]/vth*100,'v:b',markersize=8,label=r'$SWPM^W$')
ax[2,2].plot(np.log(p),vs[9,:]/vth*100,'v:b',markersize=8,label=r'$SWPM^W$')

ax[2,0].plot(np.log(p),vs[10,:]/vth*100,'s-.c',markersize=8,label=r'$SWPM^{GW}$')
ax[2,1].plot(np.log(p),vs[11,:]/vth*100,'s-.c',markersize=8,label=r'$SWPM^{GW}$')
ax[2,2].plot(np.log(p),vs[12,:]/vth*100,'s-.c',markersize=8,label=r'$SWPM^{GW}$')

for j in range(3):
  ax[2,j].set_xlim([2,12])
  ax[2,j].set_ylim([-0.05,1.2])
  xval, yval = ax[2,j].get_xlim(), ax[2,j].get_ylim()
  xrange = xval[1]-xval[0]
  yrange = yval[1]-yval[0]
  ax[2,j].set_aspect((xrange/yrange), adjustable='box')
  ax[2,j].tick_params(axis='both',direction='in',which='both',right=True,top=True)
  ax[2,j].minorticks_on()

# Labels
ax[0,0].legend()

ax[0,0].set_ylabel(r'$ln(\overline{q})$')
ax[1,0].set_ylabel(r'$q_b/\overline{q}$')
ax[2,0].set_ylabel(r'$v_n/v_{th}\times100$')

ax[0,0].set_title(r'$\mathrm{energy}$')
ax[0,1].set_title(r'$\mathrm{RW}$')
ax[0,2].set_title(r'$\mathrm{LGZ}$')

ax[2,0].set_xlabel(r'$ln(p)$')
ax[2,1].set_xlabel(r'$ln(p)$')
ax[2,2].set_xlabel(r'$ln(p)$')

fig.set_size_inches(12.5, 12.5)

fig.tight_layout(pad=0.5)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig('wall.png')














