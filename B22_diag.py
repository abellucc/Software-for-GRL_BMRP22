# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:55:04 2021

@author: LEN00398C
"""
# This script is a diagnostic tool used to perform the analyses presented in Bellucci et al.(2022; hereafter B22).
# Specifically, the scripts implements a change point detection algorithm to identify changes in the statistical
# properties of an AMOC-AMV moving correlation data sequence. The algorithm is based on the Pruned Exact Linear Time
# (PELT) scheme (Killick et al., 2012) and requires the ruptures Python library, a package designed for the analysis
# and segmentation of non-stationary signals (Truong et al., 2020). 
# Additionally, this script does also implement continuous wavelet transform analysis (Terrence and Compo (1998) and 
# other statistical analyses as presented in B22.
#
# References
#
# Bellucci, A., Mattei, D., Ruggieri, P. and Famooss Paolini, L. (2022), Intermittent behavior in the AMOC-AMV relationship,
# Geophyiscal Research Letters, under review.
#
# Killick, R., Fearnhead, P., and Eckley, I. A. (2012), Optimal Detection of Changepoints With a Linear Computational Cost, 
# Journal of the American Statistical Association, 107:500, 1590-1598.
#
# Torrence, C., and Compo, G. P. (1998), A Practical Guide to Wavelet Analysis. 
# Bulletin of the American Meteorological Society, 79(1):61–78.
#
# Truong, C., Oudre, L., and Vayatis, N. Selective review of offline change point detection methods. 
# Signal Processing, 167:107299, 2020.  


import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import ruptures as rpt
from scipy.stats import t
import statsmodels.api as sm
import pycwt as wavelet

# Load unfiltered time series of AMV and AMOC index.

sam0_amocmean=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/sam0_amocmean.npy')
cccma2_amocmean=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/cccma2_amocmean.npy')
cccma1_amocmean=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/cccma1_amocmean.npy')
ham_amocmean=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/ham_amocmean.npy')
mpi_amocmean=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/mpi_amocmean.npy')
cesm2_amocmean=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/cesm2_amocmean.npy')

ec_earth3_r2_amocmean=np.loadtxt('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/ec-earth3_r2_amocmean.dat')
inm_cm5_0_amocmean=np.loadtxt('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/inm-cm5-0_amocmean.dat')
ipsl_cm6a_lr_amocmean=np.loadtxt('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/ipsl-cm6a-lr_amocmean.dat')
mri_esm2_0_amocmean=np.loadtxt('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/mri-esm2-0_amocmean.dat')

sam0_amv=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/sam0_amv.npy')
cccma2_amv=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/cccma2_amv.npy')
cccma1_amv=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/cccma1_amv.npy')
ham_amv=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/ham_amv.npy')
mpi_amv=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/mpi_amv.npy')
cesm2_amv=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/cesm2_amv.npy')

ec_earth3_r2_amv=np.loadtxt('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/ec-earth3_r2_amv.dat')
inm_cm5_0_amv=np.loadtxt('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/inm-cm5-0_amv.dat')
ipsl_cm6a_lr_amv=np.loadtxt('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/ipsl-cm6a-lr_amv.dat')
mri_esm2_0_amv=np.loadtxt('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/mri-esm2-0_amv.dat')

# Initialize CR and NCR regime counters

tot_reg_corr=0

tot_reg_nocorr=0

# Set the the time window for the running correlation (in years)

win=80


# Set multi-plot figure for multimodel running correlation

fig9, ax9 = plt.subplots(5, 2, constrained_layout=True, figsize=(14,12))

# Set parameters for STD centroids scatterplot

fig2 = plt.figure(figsize=(17,13), constrained_layout=True)
gs = gridspec.GridSpec(4, 4,width_ratios=[1,6,1,0.2],height_ratios=[1,6,1,0.2])
ax2 = plt.subplot(gs[1:3, :2])
plt.subplots_adjust(wspace=0.2, hspace=0.3) # For giving space, especially along vertical direction, to the main plot
title='STD centroids scatterplot'
ax2.set_title(title, fontsize=17)  
ax2.set_xlabel('AMOC STD (Sv)', fontsize=16)
ax2.set_ylabel('AMV STD (K)', fontsize=16)

##########################################################
#
#                  Main loop starts here
#
##########################################################

for h in range(0,10):
    
    
    if h == 0:
        name='cccma1'
        globals()['%s_amocmean' % name]=(globals()['%s_amocmean' % name]-globals()['%s_amocmean' % name].mean())/1e09
        globals()['%s_amv' % name]=(globals()['%s_amv' % name]-globals()['%s_amv' % name].mean())
        m=0
        l=0
        name_model='CANESM5 P1'
        mar='1'
    
    
    if h==1:
        name='sam0'
        globals()['%s_amocmean' % name]=(globals()['%s_amocmean' % name]-globals()['%s_amocmean' % name].mean())/1e09
        globals()['%s_amv' % name]=(globals()['%s_amv' % name]-globals()['%s_amv' % name].mean())
        m=0
        l=1
        name_model='SAM0-UNICON'
        mar='o'  
        
    if h==2:
        name='cesm2'
        globals()['%s_amocmean' % name]=(globals()['%s_amocmean' % name]-globals()['%s_amocmean' % name].mean())/1e09
        globals()['%s_amv' % name]=(globals()['%s_amv' % name]-globals()['%s_amv' % name].mean())
        m=1
        l=0
        name_model='CESM2'
        mar='v'           
    
    if h==3:
        name='mpi'
        globals()['%s_amocmean' % name]=(globals()['%s_amocmean' % name]-globals()['%s_amocmean' % name].mean())/1e09
        globals()['%s_amv' % name]=(globals()['%s_amv' % name]-globals()['%s_amv' % name].mean())
        m=1
        l=1
        name_model='MPI-ESM2-LR'
        mar='3'  
 
    if h==4:
        name='ham'
        globals()['%s_amocmean' % name]=(globals()['%s_amocmean' % name]-globals()['%s_amocmean' % name].mean())/1e09
        globals()['%s_amv' % name]=(globals()['%s_amv' % name]-globals()['%s_amv' % name].mean())
        m=2
        l=0
        name_model='MPI-ESM2-HAM'
        mar='*'

    if h==5:
        name='cccma2'
        globals()['%s_amocmean' % name]=(globals()['%s_amocmean' % name]-globals()['%s_amocmean' % name].mean())/1e09
        globals()['%s_amv' % name]=(globals()['%s_amv' % name]-globals()['%s_amv' % name].mean())
        m=2
        l=1
        name_model='CANESM5 P2'
        mar='2'
    
    if h==6:
        name='inm_cm5_0'
        globals()['%s_amocmean' % name]=(globals()['%s_amocmean' % name]-globals()['%s_amocmean' % name].mean())
        globals()['%s_amocmean' % name]=signal.detrend(globals()['%s_amocmean' % name])
        globals()['%s_amv' % name]=signal.detrend(globals()['%s_amv' % name])
        globals()['%s_amv' % name]
        m=3
        l=0
        name_model='INM CM5 0'
        mar='H'
    
    if h==7:
        name='ipsl_cm6a_lr'
        globals()['%s_amocmean' % name]=(globals()['%s_amocmean' % name]-globals()['%s_amocmean' % name].mean())
        m=3
        l=1
        name_model='IPSL CM6A LR'
        mar='+'
    
    if h==8:
        name='mri_esm2_0'
        globals()['%s_amocmean' % name]=(globals()['%s_amocmean' % name]-globals()['%s_amocmean' % name].mean())
        m=4
        l=0
        name_model='MRI ESM2 0'
        mar='D'
    
    if h==9:
        name='ec_earth3_r2'
        globals()['%s_amocmean' % name]=(globals()['%s_amocmean' % name]-globals()['%s_amocmean' % name].mean())
        m=4
        l=1
        name_model='EC-EARTH R2'
        mar='P'
        
        
        
    n_years=len(globals()['%s_amocmean' % name])
        
        
        
    
    # Apply Butterworth low-pass filter to AMOC and AMV time series
   
    order=5
    b, a = signal.butter(order, n_years/10, 'low', fs=n_years)  
    globals()['%s_amocmean' % name] = signal.filtfilt(b, a, globals()['%s_amocmean' % name])
   
   
    order=5
    b, a = signal.butter(order, n_years/10, 'low', fs=n_years)  
    globals()['%s_amv' % name] = signal.filtfilt(b, a, globals()['%s_amv' % name])
   
   
   
    # Find lag corresponding to max correlation between AMOC and AMV.
   
    lags = np.arange(-n_years + 1, n_years)
    ccov = np.correlate(globals()['%s_amocmean' % name] - globals()['%s_amocmean' % name].mean(), globals()['%s_amv' % name] - globals()['%s_amv' % name].mean(), mode='full')
    ccor = ccov / (n_years * globals()['%s_amocmean' % name].std() * globals()['%s_amv' % name].std())
   
    maxlag = lags[np.argmax(ccor)]
   
    # Remove lag.
   
    # Resize amv
    box=np.zeros(maxlag+n_years)
    box[0:maxlag+n_years]=globals()['%s_amv' % name][-maxlag:(n_years)]
    globals()['%s_amv' % name]=box
   
   
    # Resize amoc
    box=np.zeros(maxlag+n_years)
    box[0:maxlag+n_years]=globals()['%s_amocmean' % name][0:(n_years+maxlag)]
    globals()['%s_amocmean' % name]=box
   
    # Update time series length after resize 
    
    n_years=len(globals()['%s_amocmean' % name])
   
    # Compute running cross-correlation with using the time-window set in win variable
   
    s1 = pd.Series(globals()['%s_amv' % name])
    s2 = pd.Series(globals()['%s_amocmean' % name])
    globals()['%s_corr' % name]=s1.rolling(win).corr(s2)
    globals()['%s_corr' % name]=pd.DataFrame(globals()['%s_corr' % name])
    globals()['%s_corr' % name]=globals()['%s_corr' % name][0].to_numpy()
   
   
    # Apply change-point detection using PELT method as in Killick et al. (2012)
   
    # Remove nan values 
    globals()['%s_corr' % name]=globals()['%s_corr' % name][(win-1):]
   
    algo = rpt.Pelt(model='l1').fit(globals()['%s_corr' % name])
    result = algo.predict(pen=4)
    n_changes=len(result)
    changes=np.zeros((n_changes+1))
    changes[0]=0
    changes[1:n_changes+1]=result[0:n_changes]
      
           
    # Store running correlation segments identified by the detected change points
   
    for i in range(0,n_changes):
        globals()['segment%s' % i]=np.zeros((2))
        globals()['segment%s' % i][0]=changes[i]+(win//2-1)
        globals()['segment%s' % i][1]=changes[i+1]+(win//2-1)
        globals()['segment%s' % i]=globals()['segment%s' % i].astype(int)
   
   
   
    # Define procedure to compute effective degrees of freedom in statiscal significance evaluation
    # accounting for self-correlation between time series (Bretherton et al., 1999)
   
    def Neffective(N, m1, o1) :  
        o1_ac=pd.Series(sm.tsa.acf(o1, nlags=N)) # autocorrelation of o1 
        m1_ac=pd.Series(sm.tsa.acf(m1, nlags=N)) # autocorrelation of m1 
        lags = np.arange(0,N,1.)
        Neff = N / (2*(np.sum((1 -np.abs(lags)/N)*o1_ac*m1_ac))-(np.sum(o1_ac*m1_ac)))
        return Neff
   
   
   
    globals()['%s_crit_values' % name]=np.zeros((len(globals()['%s_corr' % name])))
   
    for k in range(0,len(globals()['%s_corr' % name])):
   
        p=0.99
        df=win
        df_eff=Neffective(df,globals()['%s_amocmean' % name][k:k+win],globals()['%s_amv' % name][k:k+win])
   
        Tst=t.ppf(p,df_eff)
        globals()['%s_crit_values' % name][k]= Tst/np.sqrt(Tst**2+df_eff-2)
   
   
    # Center the arrays
    box=np.zeros(n_years)
    box[0:(win//2-1)]=float('nan')
    box[n_years-(win//2):n_years]=float('nan')
    box[(win//2-1):n_years-(win//2)]=globals()['%s_corr' % name]
    globals()['%s_corr' % name]=box
    
    
    # Identify corr and no-corr regimes using a 60% threshold.
   
    # name_nocorr=np.chararray((0))
    num_corr=0
    num_nocorr=0
    for i in range (0,n_changes):
        count=0
        for j in range (globals()['segment%s' % i][0],globals()['segment%s' % i][1]):
            if globals()['%s_corr' % name][j] <= globals()['%s_crit_values' % name][j-((win//2)-1)]:
                count=count+1
        if count / (globals()['segment%s' % i][1]-globals()['segment%s' % i][0]) < 0.6:
            globals()['segment_corr%s' % num_corr]=globals()['segment%s' % i]
            num_corr=num_corr+1
        if count / (globals()['segment%s' % i][1]-globals()['segment%s' % i][0]) >= 0.6:      
            globals()['segment_nocorr%s' % num_nocorr]=globals()['segment%s' % i]
            num_nocorr=num_nocorr+1
       
           
           
    tot_reg_corr=tot_reg_corr+num_corr
    tot_reg_nocorr=tot_reg_nocorr+num_nocorr   
    
    
    
    ######################################################################
   
    # Create stamp-like plot of running correlation
   
    ######################################################################


    if h/2 - h//2 != 0:
        
        x=np.linspace(1,n_years,n_years)
       
        title='%s     %s years' % (name_model, n_years)
       
        ax9[m,l].set_title(title, fontsize=16)  
        ax9[m,l].set_xlim(1,n_years)
       
        ytick=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        ax9[m,l].set_ylim(-0.5,1)
        ax9[m,l].set_yticks(ytick)
        ax9[m,l].axes.yaxis.set_ticklabels([])
        
        
        ax9[m,l].scatter(x,globals()['%s_corr' % name], marker='.', s=10, color='black')
       
        ax9[m,l].tick_params(axis='y', which='major', labelsize=14)
         
        ax9[m,l].tick_params(axis='x', which='major', labelsize=14)
       
       
       
       
        # Draw vertical dashed lines to mark change-points
       
        for i in range (0,n_changes):
            ax9[m,l].axvline(globals()['segment%s' % i][0], linestyle='dashed', lw=0.8, color='black')
       
        ax9[m,l].axvline(globals()['segment%s' % i][1], linestyle='dashed', lw=0.8, color='black')  # https://likegeeks.com/matplotlib-tutorial/        https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
       
       
       
        # Number the regimes
       
        for i in range (0,n_changes):
            num='%s' % (i+1)
            ax9[m,l].text(globals()['segment%s' % i][0]+9, 0.8, num, color='black', fontsize=13)
        
       
        # Mark regimes with different colors: CR = pink; NCR = light blue.
       
        if num_corr>0:
            for i in range(0,num_corr):
                ax9[m,l].axvspan(globals()['segment_corr%s' % i][0], globals()['segment_corr%s' % i][1], facecolor='pink', alpha=0.5)
       
       
       
        if num_nocorr>0:
            for i in range(0,num_nocorr):
                ax9[m,l].axvspan(globals()['segment_nocorr%s' % i][0], globals()['segment_nocorr%s' % i][1], facecolor='lightblue', alpha=0.5)
       
       
       
        # Mark non significant points (in red)
       
        for i in range(0,(n_years-win-1)):
            if globals()['%s_corr' % name][i+((win//2)-1)] <= globals()['%s_crit_values' % name][i]:
                ax9[m,l].scatter(i+((win//2)-1), globals()['%s_corr' % name][i+((win//2)-1)], marker='.', s=20, color='red')
     
        
     
       
    if h/2 - h//2 == 0:

   
        x=np.linspace(1,n_years,n_years)
       
        title='%s     %s years' % (name_model, n_years)
       
        ax9[m,l].set_title(title, fontsize=16)  
        ax9[m,l].set_ylabel('Correlation Index', fontsize=16)
        ax9[m,l].set_xlim(1,n_years)
       
        ytick=[-0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00]
        ax9[m,l].set_ylim(-0.5,1)
        ax9[m,l].set_yticks(ytick)
                    
        ax9[m,l].scatter(x,globals()['%s_corr' % name], marker='.', s=10, color='black')
       
        ax9[m,l].tick_params(axis='y', which='major', labelsize=12)
        
        ax9[m,l].tick_params(axis='x', which='major', labelsize=14)
       
       
       
        for i in range (0,n_changes):
            ax9[m,l].axvline(globals()['segment%s' % i][0], linestyle='dashed', lw=0.8, color='black')
       
        ax9[m,l].axvline(globals()['segment%s' % i][1], linestyle='dashed', lw=0.8, color='black')  # https://likegeeks.com/matplotlib-tutorial/        https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
       

       
        for i in range (0,n_changes):
            num='%s' % (i+1)
            ax9[m,l].text(globals()['segment%s' % i][0]+9, 0.8, num, color='black', fontsize=13)
       
       
        if num_corr>0:
            for i in range(0,num_corr):
                ax9[m,l].axvspan(globals()['segment_corr%s' % i][0], globals()['segment_corr%s' % i][1], facecolor='pink', alpha=0.5)
       
       
        if num_nocorr>0:
            for i in range(0,num_nocorr):
                ax9[m,l].axvspan(globals()['segment_nocorr%s' % i][0], globals()['segment_nocorr%s' % i][1], facecolor='lightblue', alpha=0.5)
       
       
       
        for i in range(0,(n_years-win-1)):
            if globals()['%s_corr' % name][i+((win//2)-1)] <= globals()['%s_crit_values' % name][i]:
                ax9[m,l].scatter(i+((win//2)-1), globals()['%s_corr' % name][i+((win//2)-1)], marker='.', s=20, color='red')
       
   
    ax9[4,0].set_xlabel('Year', fontsize=18)
    
    ax9[4,1].set_xlabel('Year', fontsize=18)



    ##############################################################################
   
    # Diagnose running std for AMOC and AMV
   
    ##############################################################################
   
   
    # Create AMOC STD array
   
    globals()['%s_amocmean_std' % name]=np.zeros((n_years-(win-1)))
   
    for i in range (0, n_years-(win-1)):
        globals()['%s_amocmean_std' % name][i]=globals()['%s_amocmean' % name][i:i+win].std()
   
        
    box=np.zeros(n_years)
    box[0:(win//2-1)]=float('nan')
    box[n_years-(win//2):n_years]=float('nan')
    box[(win//2-1):n_years-(win//2)]=globals()['%s_amocmean_std' % name]
    globals()['%s_amocmean_std' % name]=box
   
   
    # Create AMV STD array
   
    globals()['%s_amv_std' % name]=np.zeros((n_years-(win-1)))
   
    for i in range (0, n_years-(win-1)):
        globals()['%s_amv_std' % name][i]=globals()['%s_amv' % name][i:i+win].std()
   
   
    box=np.zeros(n_years)
    box[0:(win//2-1)]=float('nan')
    box[n_years-(win//2):n_years]=float('nan')
    box[(win//2-1):n_years-(win//2)]=globals()['%s_amv_std' % name]
    globals()['%s_amv_std' % name]=box
   
   
   
    # Plot STD
   
    fig5, ax5 = plt.subplots(2, 1, constrained_layout=True, figsize=(12,6))
   
   
    x=np.linspace(1,n_years,n_years)
   
    title='STD %s     %s years' % (name, n_years)
   
    ax5[0].set_title(title, fontsize=17)  
    ax5[0].set_xlabel('Year', fontsize=16)
    ax5[0].set_ylabel('AMV Degrees (K)', fontsize=16)
    ax5[0].set_xlim(1,n_years)
   
    ytick=[0, 0.05, 0.1, 0.15, 0.20]
    ax5[0].set_ylim(0, 0.2)
    ax5[0].set_yticks(ytick)
   
   
    ax5[0].scatter(x,globals()['%s_amv_std' % name], marker='.', lw=0.4, color='black')
   
   
   
    # Mark regimes with different colors: CR = pink; NCR = light blue.
   
    if num_corr>0:
        for i in range(0,num_corr):
            ax5[0].axvspan(globals()['segment_corr%s' % i][0], globals()['segment_corr%s' % i][1], facecolor='pink', alpha=0.5)
   
    if num_nocorr>0:
        for i in range(0,num_nocorr):
            ax5[0].axvspan(globals()['segment_nocorr%s' % i][0], globals()['segment_nocorr%s' % i][1], facecolor='lightblue', alpha=0.5)
   
   
    #  Draw vertical dashed lines to mark change-points   
   
    for i in range (0,n_changes):
        ax5[0].axvline(globals()['segment%s' % i][0], linestyle='dashed', lw=0.8, color='black')
   
    ax5[0].axvline(globals()['segment%s' % i][1], linestyle='dashed', lw=0.8, color='black')  
   
    
    # Number regimes
   
    for i in range (0,n_changes):
        num='%s' % (i+1)
        ax5[0].text(globals()['segment%s' % i][0]+10, 0.21, num, color='black', fontsize=17)
   
   
    title='STD %s     %s years' % (name, n_years)
   
    ax5[1].set_title(title, fontsize=17)  
    ax5[1].set_xlabel('Year', fontsize=16)
    ax5[1].set_ylabel('AMOC Volume transport (Sv)', fontsize=16)
    ax5[1].set_xlim(1,n_years)
   
    ytick=[0, 0.5, 1, 1.5]
    ax5[1].set_ylim(0, 1.5)
    ax5[1].set_yticks(ytick)
    
   
    ax5[1].scatter(x,globals()['%s_amocmean_std' % name], marker='.', lw=0.4, color='black')
   
   
    # Mark regimes with different colors: CR = pink; NCR = light blue.
   
    if num_corr>0:
        for i in range(0,num_corr):
            ax5[1].axvspan(globals()['segment_corr%s' % i][0], globals()['segment_corr%s' % i][1], facecolor='pink', alpha=0.5)
   
   
    if num_nocorr>0:
        for i in range(0,num_nocorr):
            ax5[1].axvspan(globals()['segment_nocorr%s' % i][0], globals()['segment_nocorr%s' % i][1], facecolor='lightblue', alpha=0.5)
   
   
    # Draw vertical dash lines to mark the change-points
   
    for i in range (0,n_changes):
        ax5[1].axvline(globals()['segment%s' % i][0], linestyle='dashed', lw=0.8, color='black')
   
    ax5[1].axvline(globals()['segment%s' % i][1], linestyle='dashed', lw=0.8, color='black')  # https://likegeeks.com/matplotlib-tutorial/        https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
   
    
    # Number regimes
   
    for i in range (0,n_changes):
        num='%s' % (i+1)
        ax5[1].text(globals()['segment%s' % i][0]+10, 1.3, num, color='black', fontsize=17)
   
   

    # ##############################################################################
       
    # Wavelet analysis (see Torrence and Compo, 1998)
       
    # #############################################################################
    
    sam0_amocmean=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/sam0_amocmean.npy')
    cccma2_amocmean=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/cccma2_amocmean.npy')
    cccma1_amocmean=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/cccma1_amocmean.npy')
    ham_amocmean=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/ham_amocmean.npy')
    mpi_amocmean=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/mpi_amocmean.npy')
    cesm2_amocmean=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/cesm2_amocmean.npy')
    
    ec_earth3_r2_amocmean=np.loadtxt('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/ec-earth3_r2_amocmean.dat')
    inm_cm5_0_amocmean=np.loadtxt('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/inm-cm5-0_amocmean.dat')
    ipsl_cm6a_lr_amocmean=np.loadtxt('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/ipsl-cm6a-lr_amocmean.dat')
    mri_esm2_0_amocmean=np.loadtxt('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/mri-esm2-0_amocmean.dat')
    
    sam0_amv=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/sam0_amv.npy')
    cccma2_amv=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/cccma2_amv.npy')
    cccma1_amv=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/cccma1_amv.npy')
    ham_amv=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/ham_amv.npy')
    mpi_amv=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/mpi_amv.npy')
    cesm2_amv=np.load('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/cesm2_amv.npy')
    
    ec_earth3_r2_amv=np.loadtxt('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/ec-earth3_r2_amv.dat')
    inm_cm5_0_amv=np.loadtxt('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/inm-cm5-0_amv.dat')
    ipsl_cm6a_lr_amv=np.loadtxt('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/ipsl-cm6a-lr_amv.dat')
    mri_esm2_0_amv=np.loadtxt('/home/bellucci/work/AMOC_AMV/Denis/Dati_Paper_DM/unfilt/mri-esm2-0_amv.dat')
             
       
    n_years=len(globals()['%s_amocmean' % name])
             
       
    # Compute AMOC anomalies
       
       
    globals()['%s_amocmean' % name]=(globals()['%s_amocmean' % name]-globals()['%s_amocmean' % name].mean())
       
          
   # Find maximum covariance lag
          
    lags = np.arange(-n_years + 1, n_years)
    ccov = np.correlate(globals()['%s_amocmean' % name] - globals()['%s_amocmean' % name].mean(), globals()['%s_amv' % name] - globals()['%s_amv' % name].mean(), mode='full')
    ccor = ccov / (n_years * globals()['%s_amocmean' % name].std() * globals()['%s_amv' % name].std())
       
    maxlag = lags[np.argmax(ccor)]
    
       
    # resize amv
    box=np.zeros(maxlag+n_years)
    box[0:maxlag+n_years]=globals()['%s_amv' % name][-maxlag:(n_years)]
    globals()['%s_amv' % name]=box
       
             
    # resize amoc
       
    box=np.zeros(maxlag+n_years)
    box[0:maxlag+n_years]=globals()['%s_amocmean' % name][0:(n_years+maxlag)]
    globals()['%s_amocmean' % name]=box
          
       
    n_years=len(globals()['%s_amocmean' % name])
       
       
    fig7, ax7 = plt.subplots(2, 1, constrained_layout=True, figsize=(12,6))
       
       
    # Definition of parameters of our wavelet analysis.
    # select the mother wavelet, in this case the Morlet wavelet with
    # omega_0=6.
    
    dt = 1  # In years
    mother = wavelet.Morlet(6)
    s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
    dj = 1 / 12  # Twelve sub-octaves per octaves
       
       
       
    Xticks = [np.log10(10),np.log10(30),np.log10(100),np.log10(300)]
    xlabels=['10','30','100', '300']
       
       
    # Compute AMV cwt
       
    dat = (globals()['%s_amv' % name]-globals()['%s_amv' % name].mean())
    t0 = 1
    N = n_years
    tt= np.arange(0, N) * dt + t0
       
    J = np.log2(N/2) / dj  
       
       
    alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise
       
       
    std = dat.std()  # Standard deviation
    var = std ** 2  # Variance
    dat_norm = dat / std  # Normalized dataset
       
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J,
                                                          mother)
       
       
    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2
       
    #The power is significant where the ratio ``power / sig95 > 1``.
       
    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                              significance_level=0.95,
                                              wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95
       
       
       
    period = 1 / freqs
       
       
    #plot AMV power spectra
       
    #Wavelet powerspectrum
     
    title='%s     %s years     AMV' % (name, n_years)
       
    ax7[0].set_title(title, fontsize=17)
    levels = [0.125, 0.25, 0.5, 1, 2, 4, 8, 32]

    ax7[0].contourf(tt, np.log10(period), np.log2(power), np.log2(levels),
                extend='both', cmap='Spectral_r')
       
    extent = [tt.min(), tt.max(), 0, max(period)]
       
    ax7[0].contour(tt, np.log10(period), sig95, [-99, 1], colors='k', linewidths=2,
                extent=extent)
       
    ax7[0].fill(np.concatenate([tt, tt[-1:] + dt, tt[-1:] + dt,
                                tt[:1] - dt, tt[:1] - dt]),
            np.concatenate([np.log10(coi), [1e-9], np.log10(period[-1:]),
                                np.log10(period[-1:]), [1e-9]]),
            'k', alpha=0.3, hatch='x')
       
       
    ax7[0].set_yticks(Xticks)
    ax7[0].set_yticklabels(xlabels)
       
    ax7[0].set_ylim(np.log10(4),np.log10(300))
    ax7[0].set_xlim(1,n_years)
       
    ax7[0].set_ylabel('Period (years)', fontsize=12)
    
    # Mark CR and NCR regimes and change-points using different colors and dash lines on top of wavelet plot.
       
    if num_corr>0:
        for i in range(0,num_corr):
            ax7[0].axvspan(globals()['segment_corr%s' % i][0], globals()['segment_corr%s' % i][1], facecolor='pink', alpha=0.5)
       
       
    if num_nocorr>0:
        for i in range(0,num_nocorr):
            ax7[0].axvspan(globals()['segment_nocorr%s' % i][0], globals()['segment_nocorr%s' % i][1], facecolor='lightblue', alpha=0.5)
       
       
       
    for i in range (0,n_changes):
        ax7[0].axvline(globals()['segment%s' % i][0], linestyle='dashed', lw=0.8, color='black')
       
    ax7[0].axvline(globals()['segment%s' % i][1], linestyle='dashed', lw=0.8, color='black')
       
       
    for i in range (0,n_changes):
        num='%s' % (i+1)
        ax7[0].text(globals()['segment%s' % i][0]+10, 2.3, num, color='black', fontsize=17)      
    
    
       
    # Compute AMOC cwt
       
    dat = (globals()['%s_amocmean' % name]-globals()['%s_amocmean' % name].mean())
    t0 = 1
    N = n_years
    tt= np.arange(0, N) * dt + t0
       
    J = np.log2(N/2) / dj  
       
       
       
    alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise
       
       
    std = dat.std()  # Standard deviation
    var = std ** 2  # Variance
    
    dat_norm = dat / std  # Normalized dataset
       
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J,
                                                          mother)
       
       
    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2
       
    #The power is significant where the ratio ``power / sig95 > 1``.
       
    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                              significance_level=0.95,
                                              wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95
       
       
       
    period = 1 / freqs
       
       
       
    #plot amocmean power spectra
       
    #Wavelet powerspectrum
     
    title='%s     %s years     AMOC' % (name, n_years)
       
    ax7[1].set_title(title, fontsize=17)
    levels = [0.125, 0.25, 0.5, 1, 2, 4, 8, 32]
    
    ax7[1].contourf(tt, np.log10(period), np.log2(power), np.log2(levels),
                extend='both', cmap='Spectral_r')
       
    extent = [tt.min(), tt.max(), 0, max(period)]
       
    ax7[1].contour(tt, np.log10(period), sig95, [-99, 1], colors='k', linewidths=2,
                extent=extent)
       
    ax7[1].fill(np.concatenate([tt, tt[-1:] + dt, tt[-1:] + dt,
                                tt[:1] - dt, tt[:1] - dt]),
            np.concatenate([np.log10(coi), [1e-9], np.log10(period[-1:]),
                                np.log10(period[-1:]), [1e-9]]),
            'k', alpha=0.3, hatch='x')
       
       
    ax7[1].set_yticks(Xticks)
    ax7[1].set_yticklabels(xlabels)
       
    ax7[1].set_ylim(np.log10(4),np.log10(300))
    ax7[1].set_xlim(1,n_years)
       
    ax7[1].set_ylabel('Period (years)', fontsize=12)
    ax7[1].set_xlabel('year', fontsize=12)
       
    # Mark CR and NCR regimes and change-points using different colors and dash lines on top of wavelet plot.
   
    if num_corr>0:
        for i in range(0,num_corr):
            ax7[1].axvspan(globals()['segment_corr%s' % i][0], globals()['segment_corr%s' % i][1], facecolor='pink', alpha=0.5)
   
    if num_nocorr>0:
        for i in range(0,num_nocorr):
            ax7[1].axvspan(globals()['segment_nocorr%s' % i][0], globals()['segment_nocorr%s' % i][1], facecolor='lightblue', alpha=0.5)
   
   
    for i in range (0,n_changes):
        ax7[1].axvline(globals()['segment%s' % i][0], linestyle='dashed', lw=0.8, color='black')
   
    ax7[1].axvline(globals()['segment%s' % i][1], linestyle='dashed', lw=0.8, color='black')  # https://likegeeks.com/matplotlib-tutorial/        https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
   
   
    for i in range (0,n_changes):
        num='%s' % (i+1)
        ax7[1].text(globals()['segment%s' % i][0]+10, 2.3, num, color='black', fontsize=17)
       
    
    # Create the scatterplots
   
    fig1, ax1 = plt.subplots(1, 1, constrained_layout=True, figsize=(12,6))
   
   
    title='%s     %s years' % (name, n_years)
   
    ax1.set_title(title, fontsize=17)  
    ax1.set_xlabel('AMOC STD (Sv)', fontsize=16)
    ax1.set_ylabel('AMV STD (K)', fontsize=16)
   
    ax1.set_ylim(0, 0.26)
    ax1.set_xlim(0, 1.6)
   
    
    for i in range(0,num_nocorr):
        ax1.scatter(globals()['%s_amocmean_std' % name][globals()['segment_nocorr%s' % i][0] : globals()['segment_nocorr%s' % i][1]], globals()['%s_amv_std' % name][globals()['segment_nocorr%s' % i][0]: globals()['segment_nocorr%s' % i][1]], marker='.', lw=0.1, color='blue')
   
       
    for i in range(0,num_corr):
        ax1.scatter(globals()['%s_amocmean_std' % name][globals()['segment_corr%s' % i][0] : globals()['segment_corr%s' % i][1]], globals()['%s_amv_std' % name][globals()['segment_corr%s' % i][0]: globals()['segment_corr%s' % i][1]], marker='.', lw=0.1, color='red')
   
   
   # Store centroids coordinates in a single array
   
   
    # NCR regimes
   
    x_box_std=np.zeros((num_nocorr))
    y_box_std=np.zeros((num_nocorr))
   
    for i in range(0,num_nocorr):
       
        x_box_std[i]=globals()['%s_amocmean_std' % name][globals()['segment_nocorr%s' % i][0] : globals()['segment_nocorr%s' % i][1]].mean()/globals()['%s_amocmean_std' % name][39:len(globals()['%s_amocmean_std' % name])-40].std()
        y_box_std[i]=globals()['%s_amv_std' % name][globals()['segment_nocorr%s' % i][0]: globals()['segment_nocorr%s' % i][1]].mean()/globals()['%s_amv_std' % name][39:len(globals()['%s_amv_std' % name])-40].std()
        
   
    if  h==0:
        x_centr_nocorr_std=x_box_std
        y_centr_nocorr_std=y_box_std
       
       
    if h>0:
        x_centr_nocorr_std=np.concatenate([x_centr_nocorr_std,x_box_std])
        y_centr_nocorr_std=np.concatenate([y_centr_nocorr_std,y_box_std])


    # CR regimes

    x_box_std=np.zeros((num_corr))
    y_box_std=np.zeros((num_corr))
   
    for i in range(0,num_corr):
       
        x_box_std[i]=globals()['%s_amocmean_std' % name][globals()['segment_corr%s' % i][0] : globals()['segment_corr%s' % i][1]].mean()/globals()['%s_amocmean_std' % name][39:len(globals()['%s_amocmean_std' % name])-40].std()
        y_box_std[i]=globals()['%s_amv_std' % name][globals()['segment_corr%s' % i][0]: globals()['segment_corr%s' % i][1]].mean()/globals()['%s_amv_std' % name][39:len(globals()['%s_amv_std' % name])-40].std()

   
    if  h==0:
        x_centr_corr_std=x_box_std
        y_centr_corr_std=y_box_std
       
       
    if h>0:
        x_centr_corr_std=np.concatenate([x_centr_corr_std,x_box_std])
        y_centr_corr_std=np.concatenate([y_centr_corr_std,y_box_std])
   
   
   
   
    # Store normalized coordinates of centroids 
   
   
    # NCR regimes
   
    x_box_std=np.zeros((num_nocorr))
    y_box_std=np.zeros((num_nocorr))
   
    for i in range(0,num_nocorr):
       
        x_box_std[i]=globals()['%s_amocmean_std' % name][globals()['segment_nocorr%s' % i][0] : globals()['segment_nocorr%s' % i][1]].mean()/globals()['%s_amocmean_std' % name][39:len(globals()['%s_amocmean_std' % name])-40].std()  #ho diviso per la standard deviation della standard deviation in modo da normalizzare
        y_box_std[i]=globals()['%s_amv_std' % name][globals()['segment_nocorr%s' % i][0]: globals()['segment_nocorr%s' % i][1]].mean()/globals()['%s_amv_std' % name][39:len(globals()['%s_amv_std' % name])-40].std() #ho diviso per la standard deviation della standard deviation in modo da normalizzare
       
        
        globals()['x_centr_nocorr_std_%s' % name]=x_box_std
        globals()['y_centr_nocorr_std_%s' % name]=y_box_std
       
       
    # CR regimes

    x_box_std=np.zeros((num_corr))
    y_box_std=np.zeros((num_corr))
   
    for i in range(0,num_corr):
       
        x_box_std[i]=globals()['%s_amocmean_std' % name][globals()['segment_corr%s' % i][0] : globals()['segment_corr%s' % i][1]].mean()/globals()['%s_amocmean_std' % name][39:len(globals()['%s_amocmean_std' % name])-40].std()
        y_box_std[i]=globals()['%s_amv_std' % name][globals()['segment_corr%s' % i][0]: globals()['segment_corr%s' % i][1]].mean()/globals()['%s_amv_std' % name][39:len(globals()['%s_amv_std' % name])-40].std()
        

        globals()['x_centr_corr_std_%s' % name]=x_box_std
        globals()['y_centr_corr_std_%s' % name]=y_box_std
        
        ###############

    
    for i in range(0, num_nocorr):
        ax2.scatter( globals()['x_centr_nocorr_std_%s' % name], globals()['y_centr_nocorr_std_%s' % name],marker=mar, lw=2, color='blue')
       
    for i in range(0, num_corr):
        ax2.scatter( globals()['x_centr_corr_std_%s' % name], globals()['y_centr_corr_std_%s' % name],marker=mar, lw=2, color='red')
    
    
    
ax2.scatter( x_centr_nocorr_std.mean(), y_centr_nocorr_std.mean(),marker='o', s=400, lw=2, color='blue')
ax2.scatter( x_centr_nocorr_std.mean(), y_centr_nocorr_std.mean(),marker='o', s=50, lw=2, color='white')
ax2.errorbar(x_centr_nocorr_std.mean(), y_centr_nocorr_std.mean(), xerr=x_centr_nocorr_std.std(), yerr=y_centr_nocorr_std.std(),  fmt='', color='blue', elinewidth=1, capsize=7, capthick=1.5, barsabove='true')

ax2.scatter( x_centr_corr_std.mean(), y_centr_corr_std.mean(),marker='o', s=400, lw=2, color='red')
ax2.scatter( x_centr_corr_std.mean(), y_centr_corr_std.mean(),marker='o', s=50, lw=2, color='white')
ax2.errorbar(x_centr_corr_std.mean(), y_centr_corr_std.mean(), xerr=x_centr_corr_std.std(), yerr=y_centr_corr_std.std(),  fmt='', color='red', elinewidth=1, capsize=7, capthick=1.5, barsabove='true')

ax2.scatter(0.06, 0.55+0.153,  marker='o', lw=1, s=400,  color='red')
ax2.scatter(0.06, 0.55+0.153,  marker='o', lw=1, s=50,  color='white')
ax2.text(0.2, 0.5+0.153, 'Correlation regime centroid', color='red', fontsize=17)

ax2.scatter(0.06, 0.02+0.153,  marker='o', lw=1, s=400,  color='blue')
ax2.scatter(0.06, 0.02+0.153,  marker='o', lw=1, s=50,  color='white')
ax2.text(0.2, 0.1, 'No-correlation regime centroid', color='blue', fontsize=17)   
          
