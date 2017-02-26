import os
import pandas as pd
from pandas import read_csv
from dateutil import parser, rrule
from datetime import datetime, time, date
import numpy as np
import seaborn as sns
import matplotlib.style as style
import matplotlib.pyplot as plt
style.use('ggplot')
sns.set(style="whitegrid")
import matplotlib.ticker as ticker

global dataset, dataname, SAVE

data_main_path = 'C:\\CParisel\\Finaldata\\'
ref_path = data_main_path+'Real\\' # real/reference data
#methods_folder = np.array(['MSVR', 'LR', 'SVR15', 'GBM15', 'RFSINGLE15', 'RFLOCAL15', 'RFGLOBAL'])#
#methods_folder = np.array(['MSVR', 'LR', 'SVR', 'GBM', 'RFUnique', 'RFLocal', 'RFGlobal'])
methods_folder = np.array(['MSVR', 'GBM', 'RFGlobal'])
#methods_select = np.array([1,      1,     1,        1,          1,          1,           1])
methods_select = np.array([1,          1,              1])
location= 'Chevire'
horizon= 6
days=['0_monday', '1_tuesday', '2_wednesday', '3_thursday', '4_friday', '5_saturday', '6_sunday']

SAVE= False


def compute_error(dref, data):
    sectors = dref.shape[1]
    RMSEBySector = np.zeros(shape=sectors, dtype=float)
    MAPEBySector = np.zeros(shape=sectors, dtype=float)

    for j in range(0, sectors):
        a = dref[:, j]
        b = data[:, j]
        rm = np.sqrt((a - b) ** 2)
        ma = np.absolute((a - b) / a) * 100.0
        rm[a <= 0.1] = 0
        ma[a <= 0.1] = 0
        RMSEBySector[j] = np.mean(rm)
        MAPEBySector[j] = np.mean(ma)

    meanRMSE = np.mean(RMSEBySector)
    meanMAPE = np.mean(MAPEBySector)
    print("MAPE %4.3f\t\t RMSE %4.3f\n" % (meanMAPE, meanRMSE))
    return RMSEBySector, MAPEBySector


def add_time_attribute(dt, missingValues):
    """
    Add time information : (sin,cos) for hour of the day and (0,1,2) for weekdays, saturdays and sundays
    :param dt: dataframe
    :param verbose:
    :return:
    """
    (T, S) = dt.shape
    # add all necessary fields
    start_date = "20131101T030000"  # November 1st, 00h00 horizon=6 (3:00am)
    start = parser.parse(start_date)
    dates = list(rrule.rrule(rrule.MINUTELY, interval=30, count=T, dtstart=start))
    dt['fulldate']= dates
    dt['month'] = dt['fulldate'].apply(lambda x: x.month)
    # generate day
    dt['day_of_week'] = dt['fulldate'].apply(lambda x: x.weekday())
    # generate hour
    dt['time_of_day'] = dt['fulldate'].apply(lambda x: x.time())
    # generate working day
    #dt['working_day'] = (dt['day_of_week'] <= 4)
    #dt['missing_values']= missingValues
    dt = dt.drop(['fulldate', 'month', 'day_of_week'], axis=1)
    return dt


def compareMAPE(dt):
    plt.figure()
    # check data available
    keep = []
    for i in range(len(methods_folder)):
        if methods_folder[i]: keep.append(methods_folder[i])
    keep = np.array(keep)
    # compare MAPE repartition between all method and build an histogram
    g= sns.PairGrid(dt[keep])
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_upper(sns.regplot, ci=75, color=sns.xkcd_rgb["amber"])  # color
    #g.map_diag(sns.kdeplot, lw=3)
    g.map_diag(plt.hist)
    g.set(alpha=0.5)
    plt.savefig('compareMAPE2%s' % horizon, dpi=600)


def computeMAPEPerSpeed(ref_data, d):
    ref_flat = ref_data.flatten()
    d_flat = d.flatten()
    ma = np.absolute((ref_flat - d_flat) / ref_flat) * 100.0
    #ma = np.sqrt((ref_flat - d_flat) ** 2)  # RMSE
    ma[ref_flat <= 0.1] = 0
    return ma


def compareMAPEperSpeed(dt):
    plt.figure()
    # round speed
    #dt['speed']= np.round(dt['raw_speed'])
    sorted= dt.sort_values(by= 'raw_speed') # sort dt by speed
    #sorted= sorted[sorted.MSVR<60.0]
    g = sns.jointplot("raw_speed", methods_folder[0], data=sorted)
    plt.savefig('compareMAPEperSpeed2%s' % horizon, dpi=600)

def showDistribution(dt, detail, cut=0):
    plt.figure()
   # if cut>0: dt = dt[dt.MSVR < cut]
    k= len(dt.index)
    print("[cut %d] %d %4.2f %4.2f %4.2f" % (cut, k, len(dt[dt.MSVR>cut].index)*100.0/k, len(dt[dt.GBM>cut].index)*100.0/k, len(dt[dt.RFGlobal>cut].index)*100.0/k))

    # ce qui est uniquement MSVR
    a= dt[dt.MSVR>cut]
    b = a[a.GBM>cut]
    c= b[b.RFGlobal>cut]
    print("cut GBM %d %4.2f"%(len(c.index), (len(c.index)-len(b.index))*100.0/k))


    #sns.violinplot(data=dt.loc[:, methods_folder], palette="Set3", bw=.2, cut=1, linewidth=1)
    sns.boxplot(data=dt.loc[:, methods_folder], palette="Set3", orient='h')
    plt.savefig('showDistribution%s%s' % (detail, horizon), dpi=600)

# MAIN PART #############################################################################
# preparing data ########################################################################
filename = 'SpeedHorizon%d.csv' % horizon
ref_data = np.genfromtxt('%s%s' % (ref_path, filename), delimiter=',')
dt_speed= pd.DataFrame(ref_data.flatten(), columns=['raw_speed'])
for i in range(len(methods_folder)):
    if methods_select[i]:
        d = np.genfromtxt('%s%s\\%s' % (data_main_path, methods_folder[i], filename), delimiter=',')
        rmse, mape = compute_error(ref_data, d)
        #mape= rmse
        if i == 0:
            dt = pd.DataFrame(mape, columns=[methods_folder[i]])
        else:
            dt[methods_folder[i]] = mape
        dt_speed[methods_folder[i]] = computeMAPEPerSpeed(ref_data, d)

# by timestep
dttime = []
for i in range(len(methods_folder)):
    if methods_select[i]:
        d = np.genfromtxt('%s%s\\%s' % (data_main_path, methods_folder[i], filename), delimiter=',')
        rmse, mape = compute_error(ref_data.transpose(), d.transpose())
        if i == 0:
            dttime = pd.DataFrame(mape, columns=[methods_folder[i]])
        else:
            dttime[methods_folder[i]]= mape

# add date and missing values
filename = 'C:\\CParisel\\data\\ChevireWideMissingValues_BoolVector.csv'
missingValues = np.genfromtxt(filename, delimiter=',', dtype='i8')
missingValues = missingValues[-ref_data.shape[0]:]
add_time_attribute(dt, missingValues)

# plot data      ########################################################################
#compareMAPE(dt)
#compareMAPEperSpeed(dt_speed)
showDistribution(dt_speed, "Detail", 8)
#showDistribution(dt, "Mean")