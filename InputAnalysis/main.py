import os
import explorer
import PLSRTesting
import analysing
import plotting

# where to look for the raw data
#path= 'C:\\Utils\\stage\\LICIT\\datastudy\\'
path= 'C:\\CParisel\\Code\\datastudy'
filename= ['Urban_Matrix.csv', 'Chevire_Matrix.csv', 'Bellevue_Matrix.csv']

# setup
dataset= 2  # urban # 1 chevire
split= 2928
h= 6 # 1 to 6, predict next (1) or 6th future

# load data
datafile= os.path.join(path, filename[dataset])
print("File is %s"%datafile)
dt= explorer.checkFile(datafile)
dttrain, dttest= explorer.splitData(dt, split)

# format data for training
X, Y= explorer.buildTrainingSetXYNoMemory(dttrain, h)
Xtest, YReal= explorer.buildTrainingSetXYNoMemory(dttest, h) # for error computation

## NAIVE
Ypred= Xtest
NAIVE_RMSEBySector, NAIVE_MAPEBySector= analysing.evaluate(YReal, Ypred, "naive")

## PARTIAL LEAST SQUARE
pls2= PLSRTesting.regression(X, Y)
Ypred= PLSRTesting.predict(Xtest)
explorer.printInfos("Ypred", Ypred)
PLSRT_RMSEBySector, PLSRT_MAPEBySector= analysing.evaluate(YReal, Ypred, "PLSRT")


#############################################
## PLOTS

plotting.plotErrorBySector('naive vs plsrt h=%d'%h, NAIVE_MAPEBySector, 'naive mape', PLSRT_MAPEBySector, 'plsrt mape', 'sector', 'error')
plotting.plotErrorBySector('naive vs plsrt h=%d'%h, NAIVE_RMSEBySector, 'naive rmse', PLSRT_RMSEBySector, 'plsrt rmse', 'sector', 'error')





#PLSRTesting.evaluate()

# build additional features
#dtcorr= explorer.buildCorr(dttrain, 0) # no memory
#explorer.printInfos("Corr matrix", dtcorr)

#plotting.plotHeatmap(dtcorr)

print("Test is over")