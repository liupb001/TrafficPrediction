import os
import pandas as pd
from dateutil import parser, rrule
from datetime import datetime, time, date
import numpy as np
import seaborn as sb
import matplotlib.style as style
import matplotlib.pyplot as plt
style.use('ggplot')
import matplotlib.ticker as ticker

### functions for all sections
def addDateFields(df, T):
    df[(df > 0).all(1)]


    start_date= "20130901T000000" # September 1st, 00h00
    start = parser.parse(start_date)
    dates = list(rrule.rrule(rrule.MINUTELY, interval=30, count= T, dtstart=start))
    # add temporary column
    df['fulldate']= dates
    # add one column
    df['month']= df['fulldate'].apply(lambda x: x.month)
    # generate day
    df['day_of_week'] = df['fulldate'].apply(lambda x: x.weekday())
    # generate hour
    df['time_of_day'] = df['fulldate'].apply(lambda x: x.time())
    # generate working day
    df['working_day']= (df['day_of_week'] <= 4)
    #(df['day_of_week'] >= 0) & (df['day_of_week'] <= 4)
    # drop fulldate column
    df = df.drop('fulldate', axis= 1)
    return df

def interpretWorkdays(df):
    # drop non working day
    dfwd= df[df['working_day']== True]
    # drop columns (? useful)
    dfwd= dfwd.drop(['month', 'day_of_week', 'working_day'], axis=1)
    # groupby hour
    dfwd_g= dfwd.groupby(['time_of_day'], as_index= True)
    # create stats
    dfwd_mean= dfwd_g.aggregate(np.mean)
    dfwd_std = dfwd_g.apply(np.std)
    # heatmap
    title1= 'Mean speed per section during a working day'
    title2= 'Standard dev. of speed per section during a working day'
    savename= 'HourlyStat_workingday'
    plotHeatMap(dfwd_mean, dfwd_std, title1, title2, 'Section', 'Time of day', True, savename)
    #plotHeatMap(dfwd_mean, 'Mean speed per section during a working day', 'Section', 'Time of day', False)
    #plotHeatMap(dfwd_std, 'Standard dev. of speed per section during a working day', 'Section', 'Time of day', False)

def interpretOneDay(df, dayId, dayLabel):
    # drop non working day
    dfwd= df[df['day_of_week']== dayId]
    # drop columns (? useful)
    dfwd= dfwd.drop(['month', 'day_of_week', 'working_day'], axis=1)
    # groupby hour
    dfwd_g= dfwd.groupby(['time_of_day'], as_index= True)
    # create stats
    dfwd_mean= dfwd_g.aggregate(np.mean)
    dfwd_std = dfwd_g.apply(np.std)
    # heatmap
    title1= 'Mean speed per section on a %s' % dayLabel
    title2= 'Standard dev. of speed per section on a %s'%dayLabel
    savename= 'HourlyStat_%s'%dayLabel
    plotHeatMap(dfwd_mean, dfwd_std, title1,title2, 'Section', 'Time of day', True, savename)
    #plotHeatMap(dfwd_mean, 'Mean speed per section on a %s'%dayLabel, 'Section', 'Time of day', False)
    #plotHeatMap(dfwd_std, 'Standard dev. of speed per section on a %s'%dayLabel, 'Section', 'Time of day', False)

def plotHeatMap(data1, data2, title1, title2, xlabel, ylabel, save, filename):
    X,Y= data1.shape
    xticksskip= np.int(Y/20)

    f, ax = plt.subplots(figsize=(11, 9))
    plt.subplot(2, 1, 1)
    sb.set(style="white")
    sb.heatmap(data1, xticklabels=xticksskip, yticklabels=4, annot=False, cmap="YlOrBr")
    title1= title1 + ' (%s)'%dataname[dataset]
    plt.title(title1)
    #plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(rotation=0)
    plt.subplot(2, 1, 2)
    sb.heatmap(data2, xticklabels=xticksskip, yticklabels=4, annot=False, cmap="YlOrBr")
    title2 = title2 + ' (%s)' % dataname[dataset]
    plt.title(title2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(rotation=0)
    if save== True:
        filename= dataname[dataset] + '_' + filename
        plt.savefig(filename)
    else:
        plt.show()

def checkIdenticalTimes(data, times, save, filename):
    plt.figure()
    title= 'Comparing segments a given time for %s'%dataname[dataset]
    plt.xlabel('Segments')
    plt.ylabel('Speed')
    plt.title(title)
    d= data.loc[times, :].values
    d= d.T
    (X, Y)= d.shape
    xr= np.arange(0,X)
    for i in range(len(times)):
        d[:,i]= d[:,i]+2*i
        plt.plot(xr, d[:,i], label= str(times[i]))
    plt.legend(loc='best')
    if save == True:
        filename = dataname[dataset] + '_' + filename
        plt.savefig(filename)
    else:
        plt.show()

#def checkSavingDay(df):

# not used
def groupData(df, aggregations):
    # group data by day of the week, then remove columns which are not sections
    days= df.groupby(['day_of_week'], as_index=False)

    # group data by hour of the day
    hours= df.groupby(['time_of_day'])
    # group data by working day then time of day
    workinghours= df.groupby(['working_day','time_of_day'], as_index=False)

    workinghours.describe()
    #plt.figure()
    sb.set(color_codes=True)
    sb.heatmap(workinghours, annot= False).fig.show()
    plt.show()


def checkData(df,S):
    c = (df < 0.00001).sum().sum()
    print('negative speed: %d' % c)


### function for a close up on one section
def closeUp(df, s, location):
    # all days
    dall= df.iloc[:, s].values  # converts a numpy array
    dall= np.reshape(dall, (-1,48)) # 2D arrays (day x hour)
    # work days
    dwork= df[df['working_day']==True].iloc[:,s].values
    dwork = np.reshape(dwork, (-1, 48))  # 2D arrays (day x hour)

    plt.figure()
    #plt.subplot(2, 1, 1)
    plt.title('Speeds during any day - from 0:00 to 23:30  (section %d in %s)'%(s, location))
    sb.set(style="darkgrid", palette="Set2")
    sb.tsplot(dall, value='Speeds (all days)', err_style= 'unit_traces', ci=[95], time = np.linspace(0.0, 23.5, dwork.shape[1]))#,err_style="ci_band, unit_traces)
    #plt.subplot(2, 2, 1)
    plt.show()
    plt.figure()
    plt.title('Speeds during working days - from 0:00 to 23:30 (section %d in %s)'%(s, location))
    sb.set(style="darkgrid", palette="Set1")
    sb.tsplot(dwork, value='Speeds (working days)', err_style='unit_traces',    time = np.linspace(0.0, 23.5, dwork.shape[1]))
    plt.show()




global dataset, dataname
path= 'C:\\CParisel\\Code\\datastudy'
filename= ['Urban_Matrix.csv', 'Chevire_Matrix.csv', 'Bellevue_Matrix.csv']
#filename= ['Urban_noneg.csv', 'Chevire_noneg.csv', 'Bellevue_noneg.csv']
dataname= ['Urban', 'Chevire', 'Bellevue']
# setup
dataset= 1  # 0:urban 1:chevire 2:Bellevue

s= -1  # section (-1 means all)
days=['0_monday', '1_tuesday', '2_wednesday', '3_thursday', '4_friday', '5_saturday', '6_sunday']

# load data
datafile= os.path.join(path, filename[dataset])
df= pd.read_csv(datafile, sep= ',', header=None) # returns a dataframe
df= df.transpose()
(T,S)= df.shape
checkData(df, S)
# dt.infos()
# add time fields
'''
if s==-1:
    df= addDateFields(df, T)
    interpretWorkdays(df)
    for i in range(0,7):
        interpretOneDay(df, i, days[i])
    #interpretOneDay(df, 5, 'Saturday')
    #interpretOneDay(df, 6, 'Sunday')
else :
    df = addDateFields(df, T)
    closeUp(df, s, dataname[dataset])
'''
identicalTimes= [915, 1251, 2259, 2595, 3267,3939]
checkIdenticalTimes(df,identicalTimes, False, 'IdenticalTimes')


print('Finito STATS')
# aggregate data by day of the week





# aggregate data by hour of the day

