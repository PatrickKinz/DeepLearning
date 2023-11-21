import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import QSM_and_qBOLD_functions as QQfunc
from matplotlib import colors
from matplotlib.figure import Figure

#%%
def plot_loss(history, keyword):
    plt.figure()
    plt.plot(history.history[keyword + 'loss'],'o-')
    plt.plot(history.history['val_'+keyword+'loss'],'o-')
    plt.yscale('log')
    plt.title('model ' +keyword+ 'loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def check_Params(Params_test,p,Number):
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    P0 = ax[0].imshow(Params_test[0][Number,:,:], cmap='gray')
    P0.set_clim(.0,1)
    ax[0].title.set_text('a')
    plt.colorbar(P0,ax=ax[0])
    P1 = ax[1].imshow(Params_test[1][Number,:,:], cmap='gray')
    P1.set_clim(.0,1)
    ax[1].title.set_text('b')
    plt.colorbar(P1,ax=ax[1])
    P2 = ax[2].imshow(Params_test[2][Number,:,:], cmap='gray')
    P2.set_clim(.0,1)
    ax[2].title.set_text('c')
    plt.colorbar(P2,ax=ax[2])
    P3 = ax[3].imshow(Params_test[3][Number,:,:], cmap='gray')
    P3.set_clim(.0,1)
    ax[3].title.set_text('d')
    plt.colorbar(P3,ax=ax[3])
    P4 = ax[4].imshow(Params_test[4][Number,:,:], cmap='gray')
    P4.set_clim(.0,1)
    ax[4].title.set_text('e')
    plt.colorbar(P4,ax=ax[4])
    P5 = ax[5].imshow(np.squeeze(p[0][Number,:,:,:]), cmap='gray')
    P5.set_clim(.0,1)
    plt.colorbar(P5,ax=ax[5])
    P6 = ax[6].imshow(np.squeeze(p[1][Number,:,:,:]), cmap='gray')
    P6.set_clim(.0,1)
    plt.colorbar(P6,ax=ax[6])
    P7 = ax[7].imshow(np.squeeze(p[2][Number,:,:,:]), cmap='gray')
    P7.set_clim(.0,1)
    plt.colorbar(P7,ax=ax[7])
    P8 = ax[8].imshow(np.squeeze(p[3][Number,:,:,:]), cmap='gray')
    P8.set_clim(.0,1)
    plt.colorbar(P8,ax=ax[8])
    P9 = ax[9].imshow(np.squeeze(p[4][Number,:,:,:]), cmap='gray')
    P9.set_clim(.0,1)
    plt.colorbar(P9,ax=ax[9])
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    ax[0].plot(Params_test[0][Number,15,:],'.')
    ax[0].plot(np.squeeze(p[0][Number,15,:,:]),'.')
    ax[0].set_ylim(0,1)
    ax[0].title.set_text('a')
    ax[1].plot(Params_test[1][Number,15,:],'.')
    ax[1].plot(np.squeeze(p[1][Number,15,:,:]),'.')
    ax[1].set_ylim(0,1)
    ax[1].title.set_text('b')
    ax[2].plot(Params_test[2][Number,15,:],'.')
    ax[2].plot(np.squeeze(p[2][Number,15,:,:]),'.')
    ax[2].set_ylim(0,1)
    ax[2].title.set_text('c')
    ax[3].plot(Params_test[3][Number,15,:],'.')
    ax[3].plot(np.squeeze(p[3][Number,15,:,:]),'.')
    ax[3].set_ylim(0,1)
    ax[3].title.set_text('d')
    ax[4].plot(Params_test[4][Number,15,:],'.')
    ax[4].plot(np.squeeze(p[4][Number,15,:,:]),'.')
    ax[4].set_ylim(0,1)
    ax[4].title.set_text('e')
    plt.show()


    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    ax[0].hist(Params_test[0][Number,:,:].ravel(),range=((0,1)))
    ax[0].title.set_text('a')
    ax[1].hist(Params_test[1][Number,:,:].ravel(),range=((0,1)))
    ax[1].title.set_text('b')
    ax[2].hist(Params_test[2][Number,:,:].ravel(),range=((0,1)))
    ax[2].title.set_text('c')
    ax[3].hist(Params_test[3][Number,:,:].ravel(),range=((0,1)))
    ax[3].title.set_text('d')
    ax[4].hist(Params_test[4][Number,:,:].ravel(),range=((0,1)))
    ax[4].title.set_text('e')
    ax[5].hist(np.squeeze(p[0][Number,:,:,:]).ravel(),range=((0,1)))
    ax[6].hist(np.squeeze(p[1][Number,:,:,:]).ravel(),range=((0,1)))
    ax[7].hist(np.squeeze(p[2][Number,:,:,:]).ravel(),range=((0,1)))
    ax[8].hist(np.squeeze(p[3][Number,:,:,:]).ravel(),range=((0,1)))
    ax[9].hist(np.squeeze(p[4][Number,:,:,:]).ravel(),range=((0,1)))
    plt.show()

def translate_Params(Params):
    S0 = Params[0]   #S0     = 1000 + 200 * randn(N).T
    R2 = (30-1) * Params[1] + 1  #from 1 to 30
    SaO2 = 0.98
    Y  = (SaO2 - 0.01) * Params[2] + 0.01   #from 1% to 98%
    nu = (0.1 - 0.001) * Params[3] + 0.001  #from 0.1% to 10%
    chi_nb = ( 0.1-(-0.1) ) * Params[4] - 0.1 #fr
    return [S0,R2,Y,nu,chi_nb]

def remove_air_and_CSF(Params,Seg):
    #make Mask in boolean array [False, True, True, ...]
    mask_tissue = Seg == 0 #tissue =0, air = 1, CSF =2
    #use Mask to filer. make sure everything is a numpy array and not a list
    output =[]
    for i in range(len(Params)):
        output.append(Params[i][mask_tissue])
    return output

#a= np.array([1,2,3])
#b= a==2
#print(b) #array[false,true,false]
#a[b] #array[2]


def check_Params_transformed(Params_test,p,Number,filename):
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    P0 = ax[0].imshow(Params_test[0][Number,:,:], cmap='inferno')
    P0.set_clim(.0,1)
    ax[0].title.set_text('$S_0$ [a.u.]')
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_ylabel('truth')
    plt.colorbar(P0,ax=ax[0])
    P1 = ax[1].imshow(Params_test[1][Number,:,:], cmap='inferno')
    P1.set_clim(0,30)
    ax[1].title.set_text('$R_2$ [Hz]')
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    plt.colorbar(P1,ax=ax[1])
    P2 = ax[2].imshow(Params_test[2][Number,:,:]*100, cmap='inferno')
    P2.set_clim(.0,1*100)
    ax[2].title.set_text('Y [%]')
    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)
    plt.colorbar(P2,ax=ax[2])
    P3 = ax[3].imshow(Params_test[3][Number,:,:]*100, cmap='inferno')
    P3.set_clim(0,0.1*100)
    ax[3].title.set_text('$v$ [%]')
    ax[3].get_xaxis().set_visible(False)
    ax[3].get_yaxis().set_visible(False)
    plt.colorbar(P3,ax=ax[3])
    P4 = ax[4].imshow(Params_test[4][Number,:,:]*1000, cmap='inferno')
    P4.set_clim(-.1*1000,.1*1000)
    ax[4].title.set_text('$\chi_{nb}$ [ppb]')
    ax[4].get_xaxis().set_visible(False)
    ax[4].get_yaxis().set_visible(False)
    plt.colorbar(P4,ax=ax[4])
    P5 = ax[5].imshow(np.squeeze(p[0][Number,:,:,:]), cmap='inferno')
    P5.set_clim(.0,1)
    ax[5].get_xaxis().set_visible(False)
    ax[5].get_yaxis().set_visible(False)
    ax[5].set_ylabel('prediction')
    plt.colorbar(P5,ax=ax[5])
    P6 = ax[6].imshow(np.squeeze(p[1][Number,:,:,:]), cmap='inferno')
    P6.set_clim(0,30)
    ax[6].get_xaxis().set_visible(False)
    ax[6].get_yaxis().set_visible(False)
    plt.colorbar(P6,ax=ax[6])
    P7 = ax[7].imshow(np.squeeze(p[2][Number,:,:,:])*100, cmap='inferno')
    P7.set_clim(.0,1*100)
    ax[7].get_xaxis().set_visible(False)
    ax[7].get_yaxis().set_visible(False)
    plt.colorbar(P7,ax=ax[7])
    P8 = ax[8].imshow(np.squeeze(p[3][Number,:,:,:])*100, cmap='inferno')
    P8.set_clim(.0,0.1*100)
    ax[8].get_xaxis().set_visible(False)
    ax[8].get_yaxis().set_visible(False)
    plt.colorbar(P8,ax=ax[8])
    P9 = ax[9].imshow(np.squeeze(p[4][Number,:,:,:])*1000, cmap='inferno')
    P9.set_clim(-.1*1000,.1*1000)
    ax[9].get_xaxis().set_visible(False)
    ax[9].get_yaxis().set_visible(False)
    plt.colorbar(P9,ax=ax[9])
    plt.tight_layout()
    plt.show()
    fig.savefig('plots/'+filename+'_slices.png')

    fig, axes = plt.subplots(nrows=1, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    ax[0].plot(Params_test[0][Number,15,:],'.')
    ax[0].plot(np.squeeze(p[0][Number,15,:,:]),'.')
    ax[0].set_ylim(0,1)
    ax[0].title.set_text('S0')
    ax[1].plot(Params_test[1][Number,15,:],'.')
    ax[1].plot(np.squeeze(p[1][Number,15,:,:]),'.')
    ax[1].set_ylim(0,30)
    ax[1].title.set_text('R2')
    ax[2].plot(Params_test[2][Number,15,:],'.')
    ax[2].plot(np.squeeze(p[2][Number,15,:,:]),'.')
    ax[2].set_ylim(0,1)
    ax[2].title.set_text('Y')
    ax[3].plot(Params_test[3][Number,15,:],'.')
    ax[3].plot(np.squeeze(p[3][Number,15,:,:]),'.')
    ax[3].set_ylim(0,0.1)
    ax[3].title.set_text('nu')
    ax[4].plot(Params_test[4][Number,15,:],'.')
    ax[4].plot(np.squeeze(p[4][Number,15,:,:]),'.')
    ax[4].set_ylim(-.1,.1)
    ax[4].title.set_text('chi_nb')
    plt.show()
    fig.savefig('plots/'+filename+'_line.png')


    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    ax[0].hist(Params_test[0][Number,:,:].ravel(),range=((0,1)))
    ax[0].title.set_text('S0')
    ax[1].hist(Params_test[1][Number,:,:].ravel(),range=((0,30)))
    ax[1].title.set_text('R2')
    ax[2].hist(Params_test[2][Number,:,:].ravel(),range=((0,1)))
    ax[2].title.set_text('Y')
    ax[3].hist(Params_test[3][Number,:,:].ravel(),range=((0,0.1)))
    ax[3].title.set_text('nu')
    ax[4].hist(Params_test[4][Number,:,:].ravel(),range=((-.1,.1)))
    ax[4].title.set_text('chi_nb')
    ax[5].hist(np.squeeze(p[0][Number,:,:,:]).ravel(),range=((0,1)))
    ax[6].hist(np.squeeze(p[1][Number,:,:,:]).ravel(),range=((0,30)))
    ax[7].hist(np.squeeze(p[2][Number,:,:,:]).ravel(),range=((0,1)))
    ax[8].hist(np.squeeze(p[3][Number,:,:,:]).ravel(),range=((0,.1)))
    ax[9].hist(np.squeeze(p[4][Number,:,:,:]).ravel(),range=((-.1,.1)))
    plt.show()

def check_Params_transformed_3D(Params_test,p,Number):
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    P0 = ax[0].imshow(Params_test[0][Number,:,:], cmap='gray')
    P0.set_clim(.0,1)
    ax[0].title.set_text('S0')
    plt.colorbar(P0,ax=ax[0])
    P1 = ax[1].imshow(Params_test[1][Number,:,:], cmap='gray')
    P1.set_clim(0,30)
    ax[1].title.set_text('R2')
    plt.colorbar(P1,ax=ax[1])
    P2 = ax[2].imshow(Params_test[2][Number,:,:], cmap='gray')
    P2.set_clim(.0,1)
    ax[2].title.set_text('Y')
    plt.colorbar(P2,ax=ax[2])
    P3 = ax[3].imshow(Params_test[3][Number,:,:], cmap='gray')
    P3.set_clim(0,0.1)
    ax[3].title.set_text('nu')
    plt.colorbar(P3,ax=ax[3])
    P4 = ax[4].imshow(Params_test[4][Number,:,:], cmap='gray')
    P4.set_clim(-.1,.1)
    ax[4].title.set_text('chi_nb')
    plt.colorbar(P4,ax=ax[4])
    P5 = ax[5].imshow(np.squeeze(p[0][Number,:,:,:]), cmap='gray')
    P5.set_clim(.0,1)
    plt.colorbar(P5,ax=ax[5])
    P6 = ax[6].imshow(np.squeeze(p[1][Number,:,:,:]), cmap='gray')
    P6.set_clim(0,30)
    plt.colorbar(P6,ax=ax[6])
    P7 = ax[7].imshow(np.squeeze(p[2][Number,:,:,:]), cmap='gray')
    P7.set_clim(.0,1)
    plt.colorbar(P7,ax=ax[7])
    P8 = ax[8].imshow(np.squeeze(p[3][Number,:,:,:]), cmap='gray')
    P8.set_clim(.0,0.1)
    plt.colorbar(P8,ax=ax[8])
    P9 = ax[9].imshow(np.squeeze(p[4][Number,:,:,:]), cmap='gray')
    P9.set_clim(-.1,.1)
    plt.colorbar(P9,ax=ax[9])
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    ax[0].plot(Params_test[0][Number,15,:],'.')
    ax[0].plot(np.squeeze(p[0][Number,15,:,:]),'.')
    ax[0].set_ylim(0,1)
    ax[0].title.set_text('S0')
    ax[1].plot(Params_test[1][Number,15,:],'.')
    ax[1].plot(np.squeeze(p[1][Number,15,:,:]),'.')
    ax[1].set_ylim(0,30)
    ax[1].title.set_text('R2')
    ax[2].plot(Params_test[2][Number,15,:],'.')
    ax[2].plot(np.squeeze(p[2][Number,15,:,:]),'.')
    ax[2].set_ylim(0,1)
    ax[2].title.set_text('Y')
    ax[3].plot(Params_test[3][Number,15,:],'.')
    ax[3].plot(np.squeeze(p[3][Number,15,:,:]),'.')
    ax[3].set_ylim(0,0.1)
    ax[3].title.set_text('nu')
    ax[4].plot(Params_test[4][Number,15,:],'.')
    ax[4].plot(np.squeeze(p[4][Number,15,:,:]),'.')
    ax[4].set_ylim(-.1,.1)
    ax[4].title.set_text('chi_nb')
    plt.show()


    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    ax[0].hist(Params_test[0][Number,:,:].numpy().ravel(),range=((0,1)))
    ax[0].title.set_text('S0')
    ax[1].hist(Params_test[1][Number,:,:].numpy().ravel(),range=((0,30)))
    ax[1].title.set_text('R2')
    ax[2].hist(Params_test[2][Number,:,:].numpy().ravel(),range=((0,1)))
    ax[2].title.set_text('Y')
    ax[3].hist(Params_test[3][Number,:,:].numpy().ravel(),range=((0,0.1)))
    ax[3].title.set_text('nu')
    ax[4].hist(Params_test[4][Number,:,:].numpy().ravel(),range=((-.1,.1)))
    ax[4].title.set_text('chi_nb')
    ax[5].hist(np.squeeze(p[0][Number,:,:,:]).ravel(),range=((0,1)))
    ax[6].hist(np.squeeze(p[1][Number,:,:,:]).ravel(),range=((0,30)))
    ax[7].hist(np.squeeze(p[2][Number,:,:,:]).ravel(),range=((0,1)))
    ax[8].hist(np.squeeze(p[3][Number,:,:,:]).ravel(),range=((0,.1)))
    ax[9].hist(np.squeeze(p[4][Number,:,:,:]).ravel(),range=((-.1,.1)))
    plt.show()

def check_Params_transformed_hist(Params_test,p,filename):
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    ax[0].hist(Params_test[0][:,:,:].ravel(),range=((0,1)))
    ax[0].title.set_text('S0')
    ax[1].hist(Params_test[1][:,:,:].ravel(),range=((0,30)))
    ax[1].title.set_text('R2')
    ax[2].hist(Params_test[2][:,:,:].ravel(),range=((0,1)))
    ax[2].title.set_text('Y')
    ax[3].hist(Params_test[3][:,:,:].ravel(),range=((0,0.1)))
    ax[3].title.set_text('nu')
    ax[4].hist(Params_test[4][:,:,:].ravel(),range=((-.1,.1)))
    ax[4].title.set_text('chi_nb')
    ax[5].hist(np.squeeze(p[0][:,:,:,:]).ravel(),range=((0,1)))
    ax[6].hist(np.squeeze(p[1][:,:,:,:]).ravel(),range=((0,30)))
    ax[7].hist(np.squeeze(p[2][:,:,:,:]).ravel(),range=((0,1)))
    ax[8].hist(np.squeeze(p[3][:,:,:,:]).ravel(),range=((0,.1)))
    ax[9].hist(np.squeeze(p[4][:,:,:,:]).ravel(),range=((-.1,.1)))
    plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(12,7))
    ax = axes.ravel()
    counts, xedges, yedges, im = ax[0].hist2d(x=Params_test[0][:,:,:].ravel(),y=np.squeeze(p[0][:,:,:,:]).ravel(),bins=30,range=((0,1),(0,1)),cmap='inferno')
    ax[0].title.set_text('$S_0$ [a.u.]')
    ax[0].set_xlabel('truth')
    ax[0].set_ylabel('prediction')
    cbar=fig.colorbar(im,ax=ax[0])
    cbar.formatter.set_powerlimits((0, 0))
    counts, xedges, yedges, im = ax[1].hist2d(x=Params_test[1][:,:,:].ravel(),y=np.squeeze(p[1][:,:,:,:]).ravel(),bins=30,range=((0,30),(0,30)),cmap='inferno')
    ax[1].title.set_text('$R_2$ [Hz]')
    ax[1].set_xlabel('truth')
    ax[1].set_ylabel('prediction')
    cbar=fig.colorbar(im,ax=ax[1])
    cbar.formatter.set_powerlimits((0, 0))
    counts_Y, xedges_Y, yedges, im = ax[2].hist2d(x=Params_test[2][:,:,:].ravel()*100,y=np.squeeze(p[2][:,:,:,:]).ravel()*100,bins=30,range=((0,98),(0,98)),cmap='inferno')
    ax[2].title.set_text('Y [%]')
    ax[2].set_xlabel('truth')
    ax[2].set_ylabel('prediction')
    cbar=fig.colorbar(im,ax=ax[2])
    cbar.formatter.set_powerlimits((0, 0))
    counts, xedges, yedges, im = ax[3].hist2d(x=Params_test[3][:,:,:].ravel()*100,y=np.squeeze(p[3][:,:,:,:]).ravel()*100,bins=30,range=((0,.1*100),(0,.1*100)),cmap='inferno')
    ax[3].title.set_text('$v$ [%]')
    ax[3].set_xlabel('truth')
    ax[3].set_ylabel('prediction')
    cbar=fig.colorbar(im,ax=ax[3])
    cbar.formatter.set_powerlimits((0, 0))
    counts, xedges, yedges, im = ax[4].hist2d(x=Params_test[4][:,:,:].ravel()*1000,y=np.squeeze(p[4][:,:,:,:]).ravel()*1000,bins=30,range=((-100,100),(-100,100)),cmap='inferno')
    ax[4].title.set_text('$\chi_{nb}$ [ppb]')
    ax[4].set_xlabel('truth')
    ax[4].set_ylabel('prediction')
    cbar=fig.colorbar(im,ax=ax[4])
    cbar.formatter.set_powerlimits((0, 0))
    ax[5].remove()
    plt.tight_layout()
    plt.show()
    fig.savefig('plots/'+filename+'.png')

    # collapse histogram along y axis. plot mean and std for each bin along xaxis. doesnt work like this :(
    #fig=plt.figure()
    #plt.errorbar(x=xedges_Y[1:]-(xedges_Y[1]-xedges_Y[0])/2,y=np.mean(counts_Y,axis=0),yerr=np.std(counts_Y,axis=0))
    #plt.show()



def check_full_confusion_matrix(Params_test,p,filename):
    """
    function that plots all combination of true param vs pred param as 2d histograms
    same params should show diagonals
    different params should show uniform results
    in total 25 histograms
    """
    factor = [1   ,1 ,100,100,1000]
    high =   [1.00,30,98,10, 100]
    #low =    [0.05, 5, 5, 1, -80]
    low =    [0   , 0, 0,0 ,-100]
    label = ['S$_0$','R$_2$','Y','$v$','$\chi_{nb}$']
    truth = []
    pred = []
    for i in range(5):
        truth.append(Params_test[i]*factor[i])
        pred.append(p[i]*factor[i])

    fig, axes = plt.subplots(nrows=5, ncols=5,figsize=(20,18))
    for i in range(5):
        for j in range(5):
            counts, xedges, yedges, im = axes[i,j].hist2d(x=truth[j],y=pred[i],bins=50,range=((low[j],high[j]),(low[i],high[i])),cmap='inferno')
            #axes[i,j].title.set_text('$S_0$ [a.u.]')
            axes[i,j].set_xlabel(label[j] + ' truth')
            axes[i,j].set_ylabel(label[i] + ' pred')
            cbar=fig.colorbar(im,ax=axes[i,j])
            cbar.formatter.set_powerlimits((0, 0))

    plt.tight_layout()
    plt.show()
    fig.savefig('plots/'+filename+'.png')
def check_full_confusion_matrix_normed(Params_test,p,filename):
    """
    function that plots all combination of true param vs pred param as 2d histograms
    same params should show diagonals
    different params should show uniform results
    in total 25 histograms
    """
    factor = [1   ,1 ,100,100,1000]
    high =   [1.00,30,98,10, 100]
    #low =    [0.05, 5, 5, 1, -80]
    low =    [0   , 0, 0,0 ,-100]
    label = ['S$_0$ [a.u.]','R$_2$ [Hz]','Y [%]','$v$ [%]','$\chi_{nb} [ppb]$']
    truth = []
    pred = []
    for i in range(5):
        truth.append(Params_test[i]*factor[i])
        pred.append(p[i]*factor[i])
    truth_histograms = []
    for i in range(5):
        hist, edges = np.histogram(truth[i],bins=50,range=(low[i],high[i]))
        truth_histograms.append(hist)

    fig = Figure(figsize=(16,13),constrained_layout=True)
    axes = fig.subplots(nrows=5, ncols=5)
    for i in range(5):
        for j in range(5):
            counts, xedges, yedges = np.histogram2d(x=truth[j],y=pred[i],bins=50,range=((low[j],high[j]),(low[i],high[i])))
            for k in range(50):
                for l in range(50):
                    counts[k,l]= 100*counts[k,l]/(truth_histograms[j][k]+1)
            im = axes[i,j].pcolormesh(xedges,yedges,counts.T,cmap='inferno',norm=colors.LogNorm(vmin=1, vmax=100))
            #axes[i,j].set_xlabel(label[j] + ' truth')
            #axes[i,j].set_ylabel(label[i] + ' pred')
            #cbar=fig.colorbar(im,ax=axes[i,j])
            #cbar.formatter.set_powerlimits((0, 0))
    cbar=fig.colorbar(im,ax=axes[:,4],shrink=0.6)
    cbar.ax.tick_params(labelsize='xx-large')
    for i in range(5):
        axes[i,0].set_ylabel(label[i] + ' pred', fontsize='xx-large')
    for j in range(5):
        axes[4,j].set_xlabel(label[j] + ' truth', fontsize='xx-large')
    #plt.tight_layout()
    plt.show()
    fig.savefig('plots/'+filename+'.png')


def check_Yv_confusion_matrix_normed(Params_test,p,filename):
    """
    function that plots all combination of true param vs pred param as 2d histograms
    same params should show diagonals
    different params should show uniform results
    in total 25 histograms
    """
    factor = [100,100]
    high =   [98,10]
    #low =    [0.05, 5, 5, 1, -80]
    low =    [ 0,0 ]
    label = ['Y [%]','$v$ [%]']
    truth = []
    pred = []
    for i in range(2):
        truth.append(Params_test[i]*factor[i])
        pred.append(p[i]*factor[i])
    truth_histograms = []
    for i in range(2):
        hist, edges = np.histogram(truth[i],bins=50,range=(low[i],high[i]))
        truth_histograms.append(hist)

    fig = Figure(figsize=(8,7),constrained_layout=True)
    axes = fig.subplots(nrows=2, ncols=2)
    for i in range(2):
        for j in range(2):
            counts, xedges, yedges = np.histogram2d(x=truth[j],y=pred[i],bins=50,range=((low[j],high[j]),(low[i],high[i])))
            for k in range(50):
                for l in range(50):
                    counts[k,l]= 100*counts[k,l]/(truth_histograms[j][k]+1)
            im = axes[i,j].pcolormesh(xedges,yedges,counts.T,cmap='inferno',norm=colors.LogNorm(vmin=1, vmax=100))
            #axes[i,j].set_xlabel(label[j] + ' truth')
            #axes[i,j].set_ylabel(label[i] + ' pred')
            #cbar=fig.colorbar(im,ax=axes[i,j])
            #cbar.formatter.set_powerlimits((0, 0))
    cbar=fig.colorbar(im,ax=axes[:,1],shrink=0.6)
    cbar.ax.tick_params(labelsize='xx-large')
    for i in range(2):
        axes[i,0].set_ylabel(label[i] + ' pred', fontsize='xx-large')
    for j in range(2):
        axes[1,j].set_xlabel(label[j] + ' truth', fontsize='xx-large')
    #plt.tight_layout()
    plt.show()
    fig.savefig('plots/'+filename+'.png')


def check_full_confusion_matrix_autonormed(Params_test,p,filename):
    """
    function that plots all combination of true param vs pred param as 2d histograms
    same params should show diagonals
    different params should show uniform results
    in total 25 histograms
    """
    factor = [1   ,1 ,100,100,1000]
    high =   [1.00,30,98,10, 100]
    #low =    [0.05, 5, 5, 1, -80]
    low =    [0   , 0, 0,0 ,-100]
    label = ['S$_0$','R$_2$','Y','$v$','$\chi_{nb}$']
    truth = []
    pred = []
    for i in range(5):
        truth.append(Params_test[i]*factor[i])
        pred.append(p[i]*factor[i])

    fig, axes = plt.subplots(nrows=5, ncols=5,figsize=(20,18))
    for i in range(5):
        for j in range(5):
            counts, xedges, yedges, im = axes[i,j].hist2d(x=truth[j],y=pred[i],bins=50,range=((low[j],high[j]),(low[i],high[i])),norm=colors.LogNorm(),cmap='inferno')
            #axes[i,j].title.set_text('$S_0$ [a.u.]')
            axes[i,j].set_xlabel(label[j] + ' truth')
            axes[i,j].set_ylabel(label[i] + ' pred')
            cbar=fig.colorbar(im,ax=axes[i,j])
            #cbar.formatter.set_powerlimits((0, 0))

    plt.tight_layout()
    plt.show()
    fig.savefig('plots/'+filename+'.png')

def check_full_confusion_matrix_normed(Params_test,p,filename):
    """
    function that plots all combination of true param vs pred param as 2d histograms
    same params should show diagonals
    different params should show uniform results
    in total 25 histograms
    """
    factor = [1   ,1 ,100,100,1000]
    high =   [1.00,30,98,10, 100]
    #low =    [0.05, 5, 5, 1, -80]
    low =    [0   , 0, 0,0 ,-100]
    label = ['S$_0$ [a.u.]','R$_2$ [Hz]','Y [%]','$v$ [%]','$\chi_{nb} [ppb]$']
    truth = []
    pred = []
    for i in range(5):
        truth.append(Params_test[i]*factor[i])
        pred.append(p[i]*factor[i])
    truth_histograms = []
    for i in range(5):
        hist, edges = np.histogram(truth[i],bins=50,range=(low[i],high[i]))
        truth_histograms.append(hist)

    fig = Figure(figsize=(16,13),constrained_layout=True)
    axes = fig.subplots(nrows=5, ncols=5)
    for i in range(5):
        for j in range(5):
            counts, xedges, yedges = np.histogram2d(x=truth[j],y=pred[i],bins=50,range=((low[j],high[j]),(low[i],high[i])))
            for k in range(50):
                for l in range(50):
                    counts[k,l]= 100*counts[k,l]/(truth_histograms[j][k]+1)
            im = axes[i,j].pcolormesh(xedges,yedges,counts.T,cmap='inferno',norm=colors.LogNorm(vmin=1, vmax=100))
            #axes[i,j].set_xlabel(label[j] + ' truth')
            #axes[i,j].set_ylabel(label[i] + ' pred')
            #cbar=fig.colorbar(im,ax=axes[i,j])
            #cbar.formatter.set_powerlimits((0, 0))
    cbar=fig.colorbar(im,ax=axes[:,4],shrink=0.6)
    cbar.ax.tick_params(labelsize='xx-large')
    for i in range(5):
        axes[i,0].set_ylabel(label[i] + ' pred', fontsize='xx-large')
    for j in range(5):
        axes[4,j].set_xlabel(label[j] + ' truth', fontsize='xx-large')
    #plt.tight_layout()
    plt.show()
    fig.savefig('plots/'+filename+'.png')


def check_Yv_confusion_matrix_normed(Params_test,p,filename):
    """
    function that plots all combination of true param vs pred param as 2d histograms
    same params should show diagonals
    different params should show uniform results
    in total 25 histograms
    """
    factor = [100,100]
    high =   [98,10]
    #low =    [0.05, 5, 5, 1, -80]
    low =    [ 0,0 ]
    label = ['Y [%]','$v$ [%]']
    truth = []
    pred = []
    for i in range(2):
        truth.append(Params_test[i]*factor[i])
        pred.append(p[i]*factor[i])
    truth_histograms = []
    for i in range(2):
        hist, edges = np.histogram(truth[i],bins=50,range=(low[i],high[i]))
        truth_histograms.append(hist)

    fig = Figure(figsize=(8,7),constrained_layout=True)
    axes = fig.subplots(nrows=2, ncols=2)
    for i in range(2):
        for j in range(2):
            counts, xedges, yedges = np.histogram2d(x=truth[j],y=pred[i],bins=50,range=((low[j],high[j]),(low[i],high[i])))
            for k in range(50):
                for l in range(50):
                    counts[k,l]= 100*counts[k,l]/(truth_histograms[j][k]+1)
            im = axes[i,j].pcolormesh(xedges,yedges,counts.T,cmap='inferno',norm=colors.LogNorm(vmin=1, vmax=100))
            #axes[i,j].set_xlabel(label[j] + ' truth')
            #axes[i,j].set_ylabel(label[i] + ' pred')
            #cbar=fig.colorbar(im,ax=axes[i,j])
            #cbar.formatter.set_powerlimits((0, 0))
    cbar=fig.colorbar(im,ax=axes[:,1],shrink=0.6)
    cbar.ax.tick_params(labelsize='xx-large')
    for i in range(2):
        axes[i,0].set_ylabel(label[i] + ' pred', fontsize='xx-large')
    for j in range(2):
        axes[1,j].set_xlabel(label[j] + ' truth', fontsize='xx-large')
    #plt.tight_layout()
    plt.show()
    fig.savefig('plots/'+filename+'.png')


def check_full_confusion_matrix_autonormed(Params_test,p,filename):
    """
    function that plots all combination of true param vs pred param as 2d histograms
    same params should show diagonals
    different params should show uniform results
    in total 25 histograms
    """
    factor = [1   ,1 ,100,100,1000]
    high =   [1.00,30,98,10, 100]
    #low =    [0.05, 5, 5, 1, -80]
    low =    [0   , 0, 0,0 ,-100]
    label = ['S$_0$','R$_2$','Y','$v$','$\chi_{nb}$']
    truth = []
    pred = []
    for i in range(5):
        truth.append(Params_test[i]*factor[i])
        pred.append(p[i]*factor[i])

    fig, axes = plt.subplots(nrows=5, ncols=5,figsize=(20,18))
    for i in range(5):
        for j in range(5):
            counts, xedges, yedges, im = axes[i,j].hist2d(x=truth[j],y=pred[i],bins=50,range=((low[j],high[j]),(low[i],high[i])),norm=colors.LogNorm(),cmap='inferno')
            #axes[i,j].title.set_text('$S_0$ [a.u.]')
            axes[i,j].set_xlabel(label[j] + ' truth')
            axes[i,j].set_ylabel(label[i] + ' pred')
            cbar=fig.colorbar(im,ax=axes[i,j])
            #cbar.formatter.set_powerlimits((0, 0))

    plt.tight_layout()
    plt.show()
    fig.savefig('plots/'+filename+'.png')

def check_full_confusion_matrix_normed_2(Params_test,p,filename):
    """
    function that plots all combination of true param vs pred param as 2d histograms
    same params should show diagonals
    different params should show uniform results
    in total 25 histograms
    """
    factor = [1   ,1 ,100,100,1000]
    high =   [1.00,30,98,10, 100]
    #low =    [0.05, 5, 5, 1, -80]
    low =    [0   , 0, 0,0 ,-100]
    label = ['S$_0$ [a.u.]','R$_2$ [Hz]','Y [%]','$v$ [%]','$\chi_{nb} [ppb]$']
    truth = []
    pred = []
    for i in range(5):
        truth.append(Params_test[i]*factor[i])
        pred.append(p[i]*factor[i])
    truth_histograms = []
    for i in range(5):
        hist, edges = np.histogram(truth[i],bins=50,range=(low[i],high[i]))
        truth_histograms.append(hist)

    fig = Figure(figsize=(16,13),constrained_layout=True)
    axes = fig.subplots(nrows=5, ncols=5)
    for i in range(5):
        for j in range(5):
            counts, xedges, yedges = np.histogram2d(x=truth[j],y=pred[i],bins=50,range=((low[j],high[j]),(low[i],high[i])))
            for k in range(50):
                for l in range(50):
                    counts[k,l]= 100*counts[k,l]/(truth_histograms[j][k]+1)
            im = axes[i,j].pcolormesh(xedges,yedges,counts.T,cmap='inferno',norm=colors.LogNorm(vmin=1, vmax=100))
            #axes[i,j].set_xlabel(label[j] + ' truth')
            #axes[i,j].set_ylabel(label[i] + ' pred')
            #cbar=fig.colorbar(im,ax=axes[i,j])
            #cbar.formatter.set_powerlimits((0, 0))
    cbar=fig.colorbar(im,ax=axes[:,4],shrink=0.6)
    cbar.ax.tick_params(labelsize='xx-large')
    for i in range(5):
        axes[i,0].set_ylabel(label[i] + ' pred', fontsize='xx-large')
    for j in range(5):
        axes[4,j].set_xlabel(label[j] + ' truth', fontsize='xx-large')
    #plt.tight_layout()
    plt.show()
    fig.savefig('plots/'+filename+'.png')


def check_Yv_confusion_matrix_normed(Params_test,p,filename):
    """
    function that plots all combination of true param vs pred param as 2d histograms
    same params should show diagonals
    different params should show uniform results
    in total 25 histograms
    """
    factor = [100,100]
    high =   [98,10]
    #low =    [0.05, 5, 5, 1, -80]
    low =    [ 0,0 ]
    label = ['Y [%]','$v$ [%]']
    truth = []
    pred = []
    for i in range(2):
        truth.append(Params_test[i]*factor[i])
        pred.append(p[i]*factor[i])
    truth_histograms = []
    for i in range(2):
        hist, edges = np.histogram(truth[i],bins=50,range=(low[i],high[i]))
        truth_histograms.append(hist)

    fig = Figure(figsize=(8,7),constrained_layout=True)
    axes = fig.subplots(nrows=2, ncols=2)
    for i in range(2):
        for j in range(2):
            counts, xedges, yedges = np.histogram2d(x=truth[j],y=pred[i],bins=50,range=((low[j],high[j]),(low[i],high[i])))
            for k in range(50):
                for l in range(50):
                    counts[k,l]= 100*counts[k,l]/(truth_histograms[j][k]+1)
            im = axes[i,j].pcolormesh(xedges,yedges,counts.T,cmap='inferno',norm=colors.LogNorm(vmin=1, vmax=100))
            #axes[i,j].set_xlabel(label[j] + ' truth')
            #axes[i,j].set_ylabel(label[i] + ' pred')
            #cbar=fig.colorbar(im,ax=axes[i,j])
            #cbar.formatter.set_powerlimits((0, 0))
    cbar=fig.colorbar(im,ax=axes[:,1],shrink=0.6)
    cbar.ax.tick_params(labelsize='xx-large')
    for i in range(2):
        axes[i,0].set_ylabel(label[i] + ' pred', fontsize='xx-large')
    for j in range(2):
        axes[1,j].set_xlabel(label[j] + ' truth', fontsize='xx-large')
    #plt.tight_layout()
    plt.show()
    fig.savefig('plots/'+filename+'.png')


def correlation_coef(x,y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    SPxy = np.sum((x - x_mean)*(y -y_mean))
    SQx = np.sum((x-x_mean)*(x-x_mean))
    SQy = np.sum((y-y_mean)*(y-y_mean))
    return SPxy/np.sqrt(SQx*SQy)

def check_correlation_coef(label_transformed,prediction_transformed,filename):
#%%
#label_transformed_array =np.array(label_transformed)
#label_transformed_array.shape
#prediction_transformed_array = np.array(prediction_transformed)
#Cov_array = np.corrcoef(label_transformed,prediction_transformed)
    Cov_array=np.zeros((5,5))
#print(Cov_array)
    for i in range(5):
        for j in range(5):
            Cov_array[i,j] = correlation_coef(label_transformed[j],prediction_transformed[i])
    Cov_array_round = np.round(Cov_array,3)
#    print(Cov_array_round)

    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(7,6))
    P0 = ax.imshow(Cov_array, cmap='bwr')
    P0.set_clim(-1,1)
    ax.title.set_text('Pearson correlation coefficient')
    plt.colorbar(P0,ax=ax)
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels(['S$_0$ true','R$_2$ true','Y true','$v$ true','$\chi_{nb}$ true'])
    ax.set_yticks([0,1,2,3,4])
    ax.set_yticklabels(['S$_0$ pred','R$_2$ pred','Y pred','$v$ pred','$\chi_{nb}$ pred'])
    for i in range(5):
        for j in range(5):
            c = Cov_array_round[j,i]
            ax.text(i, j, str(c), va='center', ha='center')
    fig.savefig('plots/'+ filename +'.png')





def check_nu_calc(Params_test,p,QSM_test):
    nu_calc = QQfunc.f_nu(p[2],p[4],QSM_test)

    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(5,5))
    counts, xedges, yedges, im = axes.hist2d(x=Params_test[3][:,:,:].ravel()*100,y=np.squeeze(nu_calc[:,:,:,:]).ravel()*100,bins=30,range=((0,10),(-5,15)),cmap='inferno')
    axes.title.set_text('$v$ [%]')
    axes.set_xlabel('truth')
    axes.set_ylabel('calculation')
    cbar=fig.colorbar(im,ax=axes)
    cbar.formatter.set_powerlimits((0, 0))
    axes.plot(np.linspace(0,10,10),np.linspace(0,10,10))
    plt.show()





def check_Params_transformed_hist_3D(Params_test,p):
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
    ax = axes.ravel()
    ax[0].hist(Params_test[0][:,:,:].numpy().ravel(),range=((0,1)))
    ax[0].title.set_text('S0')
    ax[1].hist(Params_test[1][:,:,:].numpy().ravel(),range=((0,30)))
    ax[1].title.set_text('R2')
    ax[2].hist(Params_test[2][:,:,:].numpy().ravel(),range=((0,1)))
    ax[2].title.set_text('Y')
    ax[3].hist(Params_test[3][:,:,:].numpy().ravel(),range=((0,0.1)))
    ax[3].title.set_text('nu')
    ax[4].hist(Params_test[4][:,:,:].numpy().ravel(),range=((-.1,.1)))
    ax[4].title.set_text('chi_nb')
    ax[5].hist(np.squeeze(p[0][:,:,:,:]).ravel(),range=((0,1)))
    ax[6].hist(np.squeeze(p[1][:,:,:,:]).ravel(),range=((0,30)))
    ax[7].hist(np.squeeze(p[2][:,:,:,:]).ravel(),range=((0,1)))
    ax[8].hist(np.squeeze(p[3][:,:,:,:]).ravel(),range=((0,.1)))
    ax[9].hist(np.squeeze(p[4][:,:,:,:]).ravel(),range=((-.1,.1)))
    plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(12,8))
    ax = axes.ravel()
    ax[0].hist2d(x=Params_test[0][:,:,:].numpy().ravel(),y=np.squeeze(p[0][:,:,:,:]).ravel(),bins=30,range=((0,1),(0,1)))
    ax[0].title.set_text('S0')
    ax[1].hist2d(x=Params_test[1][:,:,:].numpy().ravel(),y=np.squeeze(p[1][:,:,:,:]).ravel(),bins=30,range=((0,30),(0,30)))
    ax[1].title.set_text('R2')
    ax[2].hist2d(x=Params_test[2][:,:,:].numpy().ravel(),y=np.squeeze(p[2][:,:,:,:]).ravel(),bins=30,range=((0,1),(0,1)))
    ax[2].title.set_text('Y')
    ax[3].hist2d(x=Params_test[3][:,:,:].numpy().ravel(),y=np.squeeze(p[3][:,:,:,:]).ravel(),bins=30,range=((0,.1),(0,.1)))
    ax[3].title.set_text('nu')
    ax[4].hist2d(x=Params_test[4][:,:,:].numpy().ravel(),y=np.squeeze(p[4][:,:,:,:]).ravel(),bins=30,range=((-.1,.1),(-.1,.1)))
    ax[4].title.set_text('chi_nb')
    ax[5].remove()
    plt.show()
    fig.savefig('plots/CNN_Uniform_GESFIDE_16Echoes_evaluation.png')

def check_QSM(t,p,Number): #target prediction
    fig, axes = plt.subplots(nrows=2, ncols=1)
    ax = axes.ravel()
    P0=ax[0].imshow(t[Number,:,:,0], cmap='gray')
    ax[0].title.set_text('QSM')
    plt.colorbar(P0,ax=ax[0])
    P1=ax[1].imshow(p[Number,:,:,0], cmap='gray')
    #ax[1].title.set_text('QSM_pred')
    plt.colorbar(P1,ax=ax[1])
    plt.show()
    fig.savefig('plots/CNN_Uniform_GESFIDE_16Echoes_direkt_Full_check_QSM.png')

def check_qBOLD(t,p,Number): #target prediction
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))
    ax = axes.ravel()

    P0=ax[0].imshow(t[Number,:,:,0], cmap='gray')
    ax[0].title.set_text('3ms')
    #plt.colorbar(P0,ax=ax[0])
    P0.set_clim(.0,1)

    P1=ax[1].imshow(t[Number,:,:,3], cmap='gray')
    ax[1].title.set_text('9ms')
    #plt.colorbar(P1,ax=ax[1])
    P1.set_clim(.0,1)

    P2=ax[2].imshow(t[Number,:,:,7], cmap='gray')
    ax[2].title.set_text('21ms')
    #plt.colorbar(P2,ax=ax[2])
    P2.set_clim(.0,1)

    P3=ax[3].imshow(t[Number,:,:,11], cmap='gray')
    ax[3].title.set_text('33ms')
    #plt.colorbar(P3,ax=ax[3])
    P3.set_clim(.0,1)

    P4=ax[4].imshow(t[Number,:,:,15], cmap='gray')
    ax[4].title.set_text('45ms')
    plt.colorbar(P4,ax=ax[4])
    P4.set_clim(.0,1)

    P5=ax[5].imshow(p[Number,:,:,0], cmap='gray')
    #plt.colorbar(P5,ax=ax[5])
    P5.set_clim(.0,1)

    P6=ax[6].imshow(p[Number,:,:,3], cmap='gray')
    #plt.colorbar(P6,ax=ax[6])
    P6.set_clim(.0,1)

    P7=ax[7].imshow(p[Number,:,:,7], cmap='gray')
    #plt.colorbar(P7,ax=ax[7])
    P7.set_clim(.0,1)

    P8=ax[8].imshow(p[Number,:,:,11], cmap='gray')
    #plt.colorbar(P8,ax=ax[8])
    P8.set_clim(.0,1)

    P9=ax[9].imshow(p[Number,:,:,15], cmap='gray')
    plt.colorbar(P9,ax=ax[9])
    P9.set_clim(.0,1)
    plt.show()
    fig.savefig('plots/CNN_Uniform_GESFIDE_16Echoes_direkt_Full_check_qBOLD.png')

def check_Pixel(target,prediction,QSM_t,QSM_p,Number):
    t=np.array([3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48])/1000
    fig=plt.figure()
    plt.imshow(target[Number,:,:,0], cmap='gray')
    plt.plot([5,10,15,20,25],[15,15,15,15,15],'o')
    plt.show()
    fig.savefig('plots/CNN_Uniform_GESFIDE_16Echoes_direkt_Full_check_Pixel_overview.png')
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    ax = axes.ravel()
    ax[0].plot(t,target[Number,5,15,:],"o-r")
    ax[0].plot(t,prediction[Number,5,15,:],"o-b")
    ax[0].set_ylim(0,1)
    ax[1].plot("QSM",QSM_t[Number,5,15,0],"or")
    ax[1].plot("QSM",QSM_p[Number,5,15,0],"ob")
    ax[1].set_ylim(-0.15,0.15)
    plt.show()
    fig.savefig('plots/CNN_Uniform_GESFIDE_16Echoes_direkt_Full_check_Pixel_1.png')
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    ax = axes.ravel()
    ax[0].plot(t,target[Number,10,15,:],"o-r")
    ax[0].plot(t,prediction[Number,10,15,:],"o-b")
    ax[0].set_ylim(0,1)
    ax[1].plot("QSM",QSM_t[Number,10,15,0],"or")
    ax[1].plot("QSM",QSM_p[Number,10,15,0],"ob")
    ax[1].set_ylim(-0.15,0.15)
    plt.show(    )
    fig.savefig('plots/CNN_Uniform_GESFIDE_16Echoes_direkt_Full_check_Pixel_2.png')
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    ax = axes.ravel()
    ax[0].plot(t,target[Number,15,15,:],"o-r")
    ax[0].plot(t,prediction[Number,15,15,:],"o-b")
    ax[0].set_ylim(0,1)
    ax[1].plot("QSM",QSM_t[Number,15,15,0],"or")
    ax[1].plot("QSM",QSM_p[Number,15,15,0],"ob")
    ax[1].set_ylim(-0.15,0.15)
    plt.show()
    fig.savefig('plots/CNN_Uniform_GESFIDE_16Echoes_direkt_Full_check_Pixel_3.png')
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    ax = axes.ravel()
    ax[0].plot(t,target[Number,20,15,:],"o-r")
    ax[0].plot(t,prediction[Number,20,15,:],"o-b")
    ax[0].set_ylim(0,1)
    ax[1].plot("QSM",QSM_t[Number,20,15,0],"or")
    ax[1].plot("QSM",QSM_p[Number,20,15,0],"ob")
    ax[1].set_ylim(-0.15,0.15)
    plt.show()
    fig.savefig('plots/CNN_Uniform_GESFIDE_16Echoes_direkt_Full_check_Pixel_4.png')
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    ax = axes.ravel()
    ax[0].plot(t,target[Number,25,15,:],"o-r")
    ax[0].plot(t,prediction[Number,25,15,:],"o-b")
    ax[0].set_ylim(0,1)
    ax[1].plot("QSM",QSM_t[Number,25,15,0],"or")
    ax[1].plot("QSM",QSM_p[Number,25,15,0],"ob")
    ax[1].set_ylim(-0.15,0.15)
    plt.show()
    fig.savefig('plots/CNN_Uniform_GESFIDE_16Echoes_direkt_Full_check_Pixel_5.png')




def check_Params_transformed_hist_mean(Params_test,p,filename):
    """calculate mean and std for each bin of true values"""
