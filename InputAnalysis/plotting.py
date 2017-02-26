import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline

def plotHeatmap(array2d):
    sb.heatmap(array2d, annot=False)
    plt.show()

def plotErrorBySector(title, data1, data1legend, data2, data2legend, xlabel, ylabel):
    S= data1.size
    x= np.arange(0, S)

    plt.plot(x, data1, 'k', label=data1legend)
    plt.plot(x, data2, 'r', label=data2legend)
    plt.legend()
    plt.title(title)
    plt.show()