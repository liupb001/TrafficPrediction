from sklearn.cross_decomposition import PLSRegression
import numpy
import pandas as pd

pls2= 0
def regression(X,Y):
    global pls2
    c= 2  # default 2
    pls2= PLSRegression(n_components=c) # TODO try with more, understand meaning
    pls2.fit(X, Y)
    PLSRegression(copy=True, max_iter=500, n_components=c, scale=True, tol=1e-06)
    return pls2

def predict(X):
    global pls2
    Y_pred = pd.DataFrame(pls2.predict(X))
    return Y_pred

def evaluate(Xoriginal, Ytest, Ypredicted):
    # sum of square difference
    sq= (Xoriginal- Ytest)**2
    errRMSD= numpy.sum(sq)/Xoriginal.shape(0) # TODO


