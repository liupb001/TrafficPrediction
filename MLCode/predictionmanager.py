import numpy as np

prediction= 0

def setup(sections):
    global prediction
    global S
    S= sections
    # setup list of arrays
    # prediction[i] contains all the predictions for section i
    prediction= []
    for i in range(S): prediction.append(np.empty(1))


def update(section, pred):
    for i in range(section.size):
        s= section[i]
        if prediction[s].size<2:
            prediction[s]= pred[:,i]
        else:
            prediction[s]= np.vstack([prediction[s], pred[:,i]])

def computeMeans():
    T= prediction[0][0].size
    ypred= np.zeros([T,S])
    for i in range(len(prediction)):
        yp= prediction[i]
        ypred[:,i]= np.mean(yp, axis=0)
    return ypred

def computeMedian():
    T= prediction[0][0].size
    ypred= np.zeros([T,S])
    for i in range(len(prediction)):
        yp= prediction[i]
        ypred[:,i]= np.median(yp, axis=0)
    return ypred