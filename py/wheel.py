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

def get_prop(T_in, F_in):
  R = 1e-4

  d = np.shape(T_in)[0]
  rD = np.sqrt(T_in[:,0]**2 + T_in[:,1]**2)/R
  vD = np.sqrt(F_in[:,4]**2 + F_in[:,5]**2)
  T = T_in[:,2]
  rho = F_in[:,3]
  nS = F_in[:,2]

  uniq_rD = np.unique(rD);
  nuniq = np.size(uniq_rD)
  arr = np.zeros((nuniq,6))
  for i in range(nuniq):
    ind = (rD == uniq_rD[i])
    arr[i,0] = uniq_rD[i]
    arr[i,1] = np.mean(vD[ind])
    arr[i,2] = np.mean(nS[ind])
    arr[i,3] = np.mean(rho[ind])
    arr[i,4] = np.mean(T[ind])
    arr[i,5] = np.mean(rho[ind]*T[ind]*1.38e-23)

  arr = arr[arr[:,0].argsort()]
  return arr

prefix = sys.argv[1]
t0 = int(sys.argv[2])
t1 = int(sys.argv[3])
t2 = int(sys.argv[4])
ts = int(sys.argv[5])

#################################################################
# Read Files

# DSMC
fname = prefix + 'dsmc/tmp_temp.' + str(t0); tmpD0 = read_file(fname);
fname = prefix + 'dsmc/tmp_flow.' + str(t0); flwD0 = read_file(fname);

fname = prefix + 'dsmc/tmp_temp.' + str(t1); tmpD1 = read_file(fname);
fname = prefix + 'dsmc/tmp_flow.' + str(t1); flwD1 = read_file(fname);

fname = prefix + 'dsmc/tmp_temp.' + str(t2); tmpD2 = read_file(fname);
fname = prefix + 'dsmc/tmp_flow.' + str(t2); flwD2 = read_file(fname);

fname = prefix + 'dsmc/std_temp.' + str(ts); tmpD3 = read_file(fname);
fname = prefix + 'dsmc/std_flow.' + str(ts); flwD3 = read_file(fname);

# rndRW
fname = prefix + 'rndRW/tmp_temp.' + str(t0); tmpR0 = read_file(fname);
fname = prefix + 'rndRW/tmp_flow.' + str(t0); flwR0 = read_file(fname);

fname = prefix + 'rndRW/tmp_temp.' + str(t1); tmpR1 = read_file(fname);
fname = prefix + 'rndRW/tmp_flow.' + str(t1); flwR1 = read_file(fname);

fname = prefix + 'rndRW/tmp_temp.' + str(t2); tmpR2 = read_file(fname);
fname = prefix + 'rndRW/tmp_flow.' + str(t2); flwR2 = read_file(fname);

fname = prefix + 'rndRW/std_temp.' + str(ts); tmpR3 = read_file(fname);
fname = prefix + 'rndRW/std_flow.' + str(ts); flwR3 = read_file(fname);

# sepRW
fname = prefix + 'sepRWmin/tmp_temp.' + str(t0); tmpS0 = read_file(fname);
fname = prefix + 'sepRWmin/tmp_flow.' + str(t0); flwS0 = read_file(fname);

fname = prefix + 'sepRWmin/tmp_temp.' + str(t1); tmpS1 = read_file(fname);
fname = prefix + 'sepRWmin/tmp_flow.' + str(t1); flwS1 = read_file(fname);

fname = prefix + 'sepRWmin/tmp_temp.' + str(t2); tmpS2 = read_file(fname);
fname = prefix + 'sepRWmin/tmp_flow.' + str(t2); flwS2 = read_file(fname);

fname = prefix + 'sepRWmin/std_temp.' + str(ts); tmpS3 = read_file(fname);
fname = prefix + 'sepRWmin/std_flow.' + str(ts); flwS3 = read_file(fname);

#################################################################
# Post-process

dsmc0 = get_prop(tmpD0, flwD0)
dsmc1 = get_prop(tmpD1, flwD1)
dsmc2 = get_prop(tmpD2, flwD2)
dsmc3 = get_prop(tmpD3, flwD3)

rndR0 = get_prop(tmpR0, flwR0)
rndR1 = get_prop(tmpR1, flwR1)
rndR2 = get_prop(tmpR2, flwR2)
rndR3 = get_prop(tmpR3, flwR3)

sepR0 = get_prop(tmpS0, flwS0)
sepR1 = get_prop(tmpS1, flwS1)
sepR2 = get_prop(tmpS2, flwS2)
sepR3 = get_prop(tmpS3, flwS3)

#################################################################
# Plot
fig,ax = plt.subplots(3,4);

t = np.array([t0,t1,t2,ts])*1e-11
tc = 3.14159/(4.0*np.sqrt(2))*2.107e-5/100
title_names = np.zeros(4,)
title_names[0] = np.ceil(t0/tc*1e-11)
title_names[1] = np.ceil(t1/tc*1e-11)
title_names[2] = np.ceil(t2/tc*1e-11)
title_names[3] = np.ceil(ts/tc*1e-11)
print(title_names)

Tw = 300
vw = 652.378

# Temperature
ax[0,0].plot(dsmc0[:,0],dsmc0[:,4]/Tw,'-k',markersize=8,label=r'$DSMC$')
ax[0,0].plot(rndR0[:,0],rndR0[:,4]/Tw,'-r',markersize=8,label=r'$SWPM$')
ax[0,0].plot(sepR0[:,0],sepR0[:,4]/Tw,'-b',markersize=8,label=r'$SWPM^{GW}$')

ax[0,1].plot(dsmc1[:,0],dsmc1[:,4]/Tw,'-k',markersize=8,label=r'$DSMC$')
ax[0,1].plot(rndR1[:,0],rndR1[:,4]/Tw,'-r',markersize=8,label=r'$SWPM$')
ax[0,1].plot(sepR1[:,0],sepR1[:,4]/Tw,'-b',markersize=8,label=r'$SWPM^{GW}$')

ax[0,2].plot(dsmc2[:,0],dsmc2[:,4]/Tw,'-k',markersize=8,label=r'$DSMC$')
ax[0,2].plot(rndR2[:,0],rndR2[:,4]/Tw,'-r',markersize=8,label=r'$SWPM$')
ax[0,2].plot(sepR2[:,0],sepR2[:,4]/Tw,'-b',markersize=8,label=r'$SWPM^{GW}$')

ax[0,3].plot(dsmc3[:,0],dsmc3[:,4]/Tw,'-k',markersize=8,label=r'$DSMC$')
ax[0,3].plot(rndR3[:,0],rndR3[:,4]/Tw,'-r',markersize=8,label=r'$SWPM$')
ax[0,3].plot(sepR3[:,0],sepR3[:,4]/Tw,'-b',markersize=8,label=r'$SWPM^{GW}$')

for j in range(4):
  ax[0,j].set_xlim([0,1])
  ax[0,j].set_ylim([0.0,2.0])
  xval, yval = ax[0,j].get_xlim(), ax[0,j].get_ylim()
  xrange = xval[1]-xval[0]
  yrange = yval[1]-yval[0]
  ax[0,j].set_aspect((xrange/yrange), adjustable='box')
  ax[0,j].tick_params(axis='both',direction='in',which='both',right=True,top=True)
  ax[0,j].minorticks_on()

# Speed
ax[1,0].plot(dsmc0[:,0],dsmc0[:,1]/vw,'-k',markersize=8,label=r'$DSMC$')
ax[1,0].plot(rndR0[:,0],rndR0[:,1]/vw,'-r',markersize=8,label=r'$SWPM$')
ax[1,0].plot(sepR0[:,0],sepR0[:,1]/vw,'-b',markersize=8,label=r'$SWPM^{GW}$')

ax[1,1].plot(dsmc1[:,0],dsmc1[:,1]/vw,'-k',markersize=8,label=r'$DSMC$')
ax[1,1].plot(rndR1[:,0],rndR1[:,1]/vw,'-r',markersize=8,label=r'$SWPM$')
ax[1,1].plot(sepR1[:,0],sepR1[:,1]/vw,'-b',markersize=8,label=r'$SWPM^{GW}$')

ax[1,2].plot(dsmc2[:,0],dsmc2[:,1]/vw,'-k',markersize=8,label=r'$DSMC$')
ax[1,2].plot(rndR2[:,0],rndR2[:,1]/vw,'-r',markersize=8,label=r'$SWPM$')
ax[1,2].plot(sepR2[:,0],sepR2[:,1]/vw,'-b',markersize=8,label=r'$SWPM^{GW}$')

ax[1,3].plot(dsmc3[:,0],dsmc3[:,1]/vw,'-k',markersize=8,label=r'$DSMC$')
ax[1,3].plot(rndR3[:,0],rndR3[:,1]/vw,'-r',markersize=8,label=r'$SWPM$')
ax[1,3].plot(sepR3[:,0],sepR3[:,1]/vw,'-b',markersize=8,label=r'$SWPM^{GW}$')

for j in range(4):
  ax[1,j].set_xlim([0,1])
  #ax[1,j].set_ylim([])
  xval, yval = ax[1,j].get_xlim(), ax[1,j].get_ylim()
  xrange = xval[1]-xval[0]
  yrange = yval[1]-yval[0]
  ax[1,j].set_aspect((xrange/yrange), adjustable='box')
  ax[1,j].tick_params(axis='both',direction='in',which='both',right=True,top=True)
  ax[1,j].minorticks_on()

# Number
ax[2,0].plot(dsmc0[:,0],dsmc0[:,2],'-k',markersize=8,label=r'$DSMC$')
ax[2,0].plot(rndR0[:,0],rndR0[:,2],'-r',markersize=8,label=r'$SWPM$')
ax[2,0].plot(sepR0[:,0],sepR0[:,2],'-b',markersize=8,label=r'$SWPM^{GW}$')

ax[2,1].plot(dsmc1[:,0],dsmc1[:,2],'-k',markersize=8,label=r'$DSMC$')
ax[2,1].plot(rndR1[:,0],rndR1[:,2],'-r',markersize=8,label=r'$SWPM$')
ax[2,1].plot(sepR1[:,0],sepR1[:,2],'-b',markersize=8,label=r'$SWPM^{GW}$')

ax[2,2].plot(dsmc2[:,0],dsmc2[:,2],'-k',markersize=8,label=r'$DSMC$')
ax[2,2].plot(rndR2[:,0],rndR2[:,2],'-r',markersize=8,label=r'$SWPM$')
ax[2,2].plot(sepR2[:,0],sepR2[:,2],'-b',markersize=8,label=r'$SWPM^{GW}$')

ax[2,3].plot(dsmc3[:,0],dsmc3[:,2],'-k',markersize=8,label=r'$DSMC$')
ax[2,3].plot(rndR3[:,0],rndR3[:,2],'-r',markersize=8,label=r'$SWPM$')
ax[2,3].plot(sepR3[:,0],sepR3[:,2],'-b',markersize=8,label=r'$SWPM^{GW}$')

for j in range(4):
  ax[2,j].set_xlim([0,1])
  ax[2,j].set_ylim([1e0,1e6])
  ax[2,j].set_yscale('log')
  xval, yval = ax[2,j].get_xlim(), ax[2,j].get_ylim()
  yval = np.log(yval)
  xrange = xval[1]-xval[0]
  yrange = yval[1]-yval[0]
  ax[2,j].set_aspect(2.5*(xrange/yrange), adjustable='box')
  ax[2,j].tick_params(axis='both',direction='in',which='both',right=True,top=True)
  ax[2,j].minorticks_on()

# Labels
ax[0,0].legend()

ax[0,0].set_ylabel(r'$T/T_b$')
ax[1,0].set_ylabel(r'$v/v_b$')
ax[2,0].set_ylabel(r'$N_c$')

ax[2,0].set_xlabel(r'$r/R$')
ax[2,1].set_xlabel(r'$r/R$')
ax[2,2].set_xlabel(r'$r/R$')
ax[2,3].set_xlabel(r'$r/R$')

ax[0,0].set_title(r'$\zeta=18$')
ax[0,1].set_title(r'$\zeta=45$')
ax[0,2].set_title(r'$\zeta=106$')
ax[0,3].set_title(r'$\zeta=556$')

fig.set_size_inches(12.5, 10.5)

fig.tight_layout(pad=0.5)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig('wheel.png')













