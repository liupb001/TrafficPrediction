import pandas as pd

def printInfos(text, dt):
    size = dt.shape
    print('[%s] Rows (t): %d, Cols (S): %d' % (text, size[0], size[1]))

def checkFile(filename):
    dt= pd.read_csv(filename, sep= ',', header=None) # returns a dataframe
    dt= dt.transpose()
    printInfos('all', dt)
    return dt

def splitData(dt, split):
    # split data in training and testing
    dttrain= dt.iloc[0:split, :]
    printInfos('train', dttrain)
    dttest= dt.iloc[split:, :]
    printInfos('test', dttest)
    return dttrain, dttest

def buildCorr(dt, memory):
    corr0= dt.corr()

    #if memory > 0:
        # TODO
        # see page A


    corr= corr0
    return corr


def buildTrainingSetXYNoMemory(dt, h):
    # for X, remove the first h samples
    #  for Y, remove the last h samples
    X= dt.iloc[h:, :]
    X= X.reset_index()
    Y= dt.iloc[:-h, :]
    Y= Y.reset_index()
    printInfos("X", X)
    printInfos("Y", Y)
    return X, Y
