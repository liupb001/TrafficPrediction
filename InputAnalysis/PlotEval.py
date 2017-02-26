import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
plt.style.use('ggplot')

###### SETUP
verbose= True

path='C:\\CParisel\\Output'
location= ['Urban', 'Chevire', 'Bellevue']
methods= ['Real','Naive', 'GlobalPLSR','LocalPLSR', 'GradientBoosting']
# parameters
dataset= 1  # urban chevire or bellevue
method= 4
h= 1

def loadData(path, method, location, name):
    datapath= path+'\\'+method+'\\'+location
    fullname= os.path.join(datapath, name)
    data= genfromtxt(fullname, delimiter=',')
    return data

def computeMeans(d):
    return

def plot(title, d1, d2, d3, legend, xlabel, ylabel, filename):
    lw= 0.5
    horizon= ['30min', '1h', '1h30', '2h', '2h30', '3h00']
    x= np.arange(0, d1[0].size)
    fig, axes = plt.subplots(ncols=2, nrows=3,  sharex='col', sharey='row')
    for i in range(0,6):
        axes[int(i/2), i%2].plot(x, d1[i], label= legend[0], linewidth= lw)
        axes[int(i / 2), i % 2].plot(x, d2[i], label= legend[1], linewidth= lw)
        axes[int(i / 2), i % 2].plot(x, d3[i], label=legend[2], linewidth= lw)
        axes[int(i / 2), i % 2].set_title('h=%s'%horizon[i])
    st= fig.suptitle(title, fontsize=14)
    #plt.legend(loc='lower left')
    plt.legend()

    fig.tight_layout()
    # shift subplots down:
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)

    #fig.show()
    fig.savefig(filename, dpi=600)



###### RMSE
# by section
d1= loadData(path, methods[1], location[dataset], 'RMSEBySection.csv')
d2= loadData(path, methods[2], location[dataset], 'RMSEBySection.csv')
d3= loadData(path, methods[method], location[dataset], 'RMSEBySection.csv')
d1means= np.mean(d1,1) # compute means by horizons
d2means= np.mean(d2,1)
d3means= np.mean(d3,1)
legend= ['Naive', 'GlobalPLSR','LocalPLSR']
plot('RMSE by section (%s)'%location[dataset], d1, d2, d3, legend, 'sections', 'RMSE(percent)', '%sRMSEBySection'%location[dataset])
# by timestep
d1= loadData(path, methods[1], location[dataset], 'RMSEByTimeStep.csv')
d2= loadData(path, methods[2], location[dataset], 'RMSEByTimeStep.csv')
d3= loadData(path, methods[3], location[dataset], 'RMSEByTimeStep.csv')
#d1means= np.mean(d1,1) # compute means by horizons
#d2means= np.mean(d2,1)
legend= ['Naive', 'GlobalPLSR','LocalPLSR']
plot('RMSE by timestep (%s)'%location[dataset], d1, d2, d3, legend, 'timestep', 'RMSE(percent)', '%sRMSEByTimeStep'%location[dataset])

###### MAPE
# by section
d1= loadData(path, methods[1], location[dataset], 'MAPEBySection.csv')
d2= loadData(path, methods[2], location[dataset], 'MAPEBySection.csv')
d3= loadData(path, methods[3], location[dataset], 'MAPEBySection.csv')
d1means= np.mean(d1,1) # compute means by horizons
d2means= np.mean(d2,1)
legend= ['Naive', 'GlobalPLSR', 'LocalPLSR']
plot('MAPE by section (%s)'%location[dataset], d1, d2, d3, legend, 'sections', 'MAPE(percent)', '%sMAPEBySection'%location[dataset])




