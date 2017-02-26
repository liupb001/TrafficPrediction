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

ref_path = 'C:\\CParisel\\Output\\Real\\Chevire\\' # real/reference data
data_main_path = 'C:\\CParisel\\Finaldata\\'
method= 'LR'
methods = ['MSVR', 'LR', 'SVR15', 'GBM15', 'RFSINGLE15', 'RFLOCAL15', 'RFGLOBAL']

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

def compute_per_speed(ref, d):
    ref_flat = ref.flatten()
    d_flat = d.flatten()
    a = np.absolute((ref_flat - d_flat) / ref_flat) * 100.0
    deviation = (ref_flat - d_flat)
    return deviation, a

#def plotvsSpeed():
    # plot speed difference vs real speed
    # plot MAPE, RMSE vs real speed

def plot_per_speed(ref, d):
    # remove all zeros (unusable data)
    d= d[ref>=0.101]  # may contain fake data
    ref= ref[ref>=0.101]
    dev, a= compute_per_speed(ref, d)
    # round values
    d= np.around(d, 1)
    ref = np.around(ref, 1)
    dev = np.around(dev, 1)
    a = np.around(a, 1)

    # build dataframe
    dt = pd.DataFrame(ref, columns=['raw_speed'])
    dt['predicted']= d
    dt['deviation']= dev
    dt['devperc']= a
    # pred speed
    lim= (0, np.max([np.max(ref), np.max(d)]))
    g = sns.jointplot("raw_speed", 'predicted', kind='kde', stat_func= None, data=dt,xlim= lim, ylim=lim)
    g.set_axis_labels('real speed', 'predicted speed')
    plt.savefig('.\\%s\\refVSpredSpeed_%d' % (method,horizon), dpi=600)
    # dev
    limx= (0, np.max(ref))
    g = sns.jointplot("raw_speed", 'deviation', kind='hex', stat_func= None, data=dt,xlim= limx)
    g.set_axis_labels('real speed', 'deviation')
    plt.savefig('.\\%s\\refVSdeviation_%d' % (method,horizon), dpi=600)
    # devpec
    g = sns.jointplot("raw_speed", 'devperc', kind='kde', stat_func=None, data=dt, xlim=limx)
    g.set_axis_labels('real speed', 'deviation (percentage)')
    plt.savefig('.\\%s\\refVSdeviationperc_%d' % (method, horizon), dpi=600)
    # boxplot



# MAIN PART #############################################################################
# preparing data ########################################################################

for h in np.arange(6):
    horizon= h+1
    filename = 'SpeedHorizon%d.csv' % horizon
    ref_data = np.genfromtxt('%s%s' % (ref_path, filename), delimiter=',')
    d = np.genfromtxt('%s%s\\%s' % (data_main_path, method, filename), delimiter=',')

    plot_per_speed(ref_data.flatten(), d.flatten())




