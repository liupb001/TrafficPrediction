import os
import pandas as pd

global dataset, dataname
path= 'C:\\CParisel\\Code\\datastudy'
filename= ['Urban_Matrix.csv', 'Chevire_Matrix.csv', 'Bellevue_Matrix.csv']
#filename= ['Urban_noneg.csv', 'Chevire_noneg.csv', 'Bellevue_noneg.csv']
dataname= ['Urban', 'Chevire', 'Bellevue']
# setup
dataset= 2  # 0:urban 1:chevire 2:Bellevue

# load data
datafile= os.path.join(path, filename[dataset])
df= pd.read_csv(datafile, sep= ',', header=None) # returns a dataframe

# replace all neg by 0.1
c=  (df<0.1).sum().sum()
(S,T)= df.shape
total= S*T
pth= c*1000.0/total
print("%s has %d negative values out of %d (%4.2f per thousand)"%(dataname[dataset], c, total, pth))
df[df<0.1]= 0.1

# save data
n= '%s_noneg.csv'%dataname[dataset]
target= os.path.join(path, n)
df.to_csv(path_or_buf=target,index=False, header=False, float_format="%4.3f")




print('Finito %s'%n)
# aggregate data by day of the week


