import sys
from os import path
import json
from time import time
from operator import itemgetter
import pandas
from pandas import read_csv
from dateutil import parser, rrule
import numpy as np
from sklearn import ensemble
from sklearn import neural_network
from sklearn import linear_model
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import evaluation
import predictionmanager
from timeit import default_timer as timer


# global variables
output_path = ''
cfg = ''
train_cut = 2928
#  dt = ''  # data frame containing all the features (speed) for 3 months and all segments
#  dt_time = ''  # data fram containing the time (day, hours...) information, same size as dt
#  dt_rows = ''  # number of rows (time)
#  dt_cols = ''  # number of columns (segment and clock)
#  correlation_matrix = ''
#  missing_values = ''


def init(argv):
    global output_path, cfg

    if len(argv) == 0:
        print("No output path defined")
        print("Usage : python mlmain.py output_folder")
        return
    output_path = str(argv[0])
    config_name = "param_%s.json" % str(argv[1])  # TODO create a sub folder of this name to save data
    # check if path exist
    if not(path.isdir(output_path)):
        print("Missing output directory")
        return
    # check if parameter file exist
    parameter_path = path.join(output_path, config_name)
    if not(path.exists(parameter_path)):
        print("Missing parameter file (param.txt) in output dir")
    # read parameter file and setup simulation
    with open(parameter_path, 'r') as json_file:
        cfg = json.load(json_file)


def load_file(datafile):
    dt = read_csv(datafile, sep=',', header=None)  # returns a data frame
    dt = dt.transpose()  # each section is a column
    return dt


def add_time_attribute(dt, miss_values):
    """
    Add time information : (sin,cos) for hour of the day and (0,1,2) for weekdays, saturdays and sundays
    :param dt: data frame
    :param miss_values:
    :return:
    """
    (T, S) = dt.shape
    # add all necessary fields
    start_date = "20130901T000000"  # September 1st, 00h00
    start = parser.parse(start_date)
    dates = list(rrule.rrule(rrule.MINUTELY, interval=30, count=T, dtstart=start))
    dt_time = pandas.DataFrame(dates, columns=['fulldate'])
    dt_time['month'] = dt_time['fulldate'].apply(lambda x: x.month)
    # generate day
    dt_time['day_of_week'] = dt_time['fulldate'].apply(lambda x: x.weekday())
    # generate hour
    dt_time['time_of_day'] = dt_time['fulldate'].apply(lambda x: x.time())
    # generate working day
    dt_time['working_day'] = (dt_time['day_of_week'] <= 4)
    dt_time['missing_values'] = miss_values

    # build (sin,cos) column for hour of the day (48 samples per day)
    # 24 hours clock : sin(2pi*hour/24), cos(2pi*hour/24)
    no_of_days = dt.shape[0]/48
    h = np.arange(0, 24, 0.5)
    time_sin = np.sin(2*np.pi*h/24)
    time_cos = np.cos(2*np.pi*h/24)
    if cfg["timeattr"]:
        dt['time_sin'] = np.tile(time_sin, no_of_days)
        dt['time_cos'] = np.tile(time_cos, no_of_days)
        # build day class
        day_type = [0, 0, 0, 0, 0, 1, 2]
        dt['day_class'] = dt_time['day_of_week'].apply(lambda x: day_type[x])
    # drop useless columns
    # df = dt.drop(['fulldate', 'month', 'day_of_week', 'time_of_day', 'working_day'], axis=1)

    #dt['missing_values']= dt_time['missing_values']
    #dt.to_csv("C:\\CParisel\\data\\ChevireComplete.csv", sep=',')
    return dt, dt_time


def prepare_data():
    data_name = '%s.csv' % cfg['location']
    datafile = path.join(cfg['datapath'], data_name)
    #datafile = path.join(cfg['datapathlinux'], data_name)
    print("Working with %s" % datafile)
    dt = load_file(datafile)
    data_name = '%sWideMissingValues_BoolVector.csv'%cfg['location']
    miss_values = read_csv(path.join(cfg['datapath'], data_name), sep=',', header=None)
    if cfg['corrmatrix'][0]:
        corr_matrix = build_correlation_matrix(dt, cfg['corrmatrix'][1])
    else:
        corr_matrix = 0
    dt, dt_time = add_time_attribute(dt, miss_values)
    return dt, dt_time, corr_matrix


def build_xy_set_single(dt, horizon, segment, history):
    # x has a single segment, y has a single segment
    v = dt.iloc[:, segment].values
    y = np.copy(v[horizon + history:])
    tmp = v[:-horizon]
    t = y.shape[0]
    x = np.zeros([t, history+1])
    for i in np.arange(t):
        x[i, :] = tmp[i:i+history+1]
    return x, y


def build_xy_set_local(dt, horizon, corr, time_attr, history, single_output):
    # keep all columns in corr and the last 3 columns
    (r, c) = dt.shape  # TODO : time is duplicated.... useful ?
    if time_attr:
        col_to_keep = np.hstack([corr, np.arange(c - 3, c)])
    else:
        col_to_keep = 0
    v = dt.iloc[:, col_to_keep].values
    # for Y, remove the first h samples
    if single_output:
        y = np.copy(v[horizon + history:, 0])
    else:
        y = np.copy(v[horizon + history:, :-3])  # all but time # TODO check
    # for X, remove the last h+t samples
    tmp = v[:-horizon, :]  # at time 0
    x = np.copy(tmp)
    for i in range(history):
        tmp = tmp[:-1, :]  # remove last time
        x = x[1:, :]  # remove first
        x = np.concatenate((x, tmp), axis=1)
    return x, y


def run_ml(dt, dt_time, corr_matrix):
    method_parameters = cfg['methodparam'][0]
    if cfg['methodname'] == 'GradientBoosting':
        clf = ensemble.GradientBoostingRegressor(**method_parameters)
    if cfg['methodname'] == 'RandomForest':
        clf = ensemble.RandomForestRegressor(**method_parameters)
    if cfg['methodname'] == 'AdaBoost':
        clf = ensemble.AdaBoostRegressor(**method_parameters)
    if cfg['methodname'] == 'ExtraTrees':
        clf = ensemble.ExtraTreesRegressor(**method_parameters)
    if cfg['methodname'] == 'Bagging':
        clf = ensemble.BaggingRegressor(**method_parameters)
    if cfg['methodname'] == 'NeuralNetwork':
        clf = neural_network.BernoulliRBM(**method_parameters)
    if cfg['methodname'] == 'LinearRegression':
        clf = linear_model.LinearRegression(**method_parameters)
    if cfg['methodname'] == 'SVR':
        clf = svm.SVR(**method_parameters)  # TODO try NuSVR et LinearSVR

    (no_of_samples, dt_cols) = dt.shape
    if cfg['timeattr']:
        time_attr = 3
    else:
        time_attr = 0
    no_of_segments = dt_cols - time_attr

    # compute each horizon one at a time
    for hor in range(cfg['horizon'][0], cfg['horizon'][1] + 1):
        print('Horizon= %d' % hor)
        # speedHorizon : no of prediction = size of test set - horizon
        speed_horizon = np.zeros([no_of_samples - train_cut - hor, no_of_segments])
        RMSEBySector = []
        MAPEBySector = []
        RMSEByTimestep = []
        MAPEByTimestep = []
        y_test = dt.iloc[train_cut + hor:, :].values
        if time_attr > 0:
            y_test = y_test[:, :-time_attr]  # drop time attribute

        missing = dt_time['missing_values'].values
        missing = missing[cfg['history']:train_cut]  # missing values for the training set
        ## TODO TIMER
        start = timer()
        ##
        # case unique - unique (all linear models)
        if cfg["inputtype"] == "unique" and cfg["outputtype"] == "unique":
            for segment in range(no_of_segments):
                print('\tSegment= %d' % segment)
                x, y = build_xy_set_single(dt, hor, segment, cfg['history'])
                # missing = dt_time['missing_values'].values
                # missing = missing[cfg['history']:train_cut]  # missing values for the training set
                x_train = x[:train_cut - cfg['history'], :]
                y_train = y[:train_cut - cfg['history']]
                x_train = x_train[np.logical_not(missing), :]
                y_train = y_train[np.logical_not(missing)]
                x_test = x[train_cut - cfg['history']:, :]
                # train and predict
                clf.fit(x_train, y_train)
                y_predicted = clf.predict(x_test)
                speed_horizon[:, segment] = y_predicted

        # SVR, RandomForest_sub, GBM
        if cfg["inputtype"] == "local" and cfg["outputtype"] == "unique":
            for segment in range(no_of_segments):
                print('\tSegment= %d' % segment)
                if not(cfg["corrmatrix"][0]):
                    x, y = build_xy_set_local(dt, hor, segment, cfg['timeattr'], cfg['history'], single_output= True)
                else:
                    x, y = build_xy_set_local(dt, hor, corr_matrix[segment, :], cfg['timeattr'], cfg['history'], single_output=True)
                # missing = dt_time['missing_values'].values
                # missing = missing[cfg['history']:train_cut]  # missing values for the training set
                x_train = x[:train_cut - cfg['history'], :]
                y_train = y[:train_cut - cfg['history']]
                x_train = x_train[np.logical_not(missing), :]
                y_train = y_train[np.logical_not(missing)]
                x_test = x[train_cut - cfg['history']:, :]
                # train and predict
                clf.fit(x_train, y_train)
                y_predicted = clf.predict(x_test)
                speed_horizon[:, segment] = y_predicted

        # only RF global so far
        if cfg['inputtype'] =="global" and cfg['outputtype'] == "global":
            x, y = build_Xy_set_global_history(dt, hor, cfg['timeattr'], cfg['history'])
            # missing = dt_time['missing_values'].values
            # missing = missing[cfg['history']:train_cut]  # missing values for the training set
            x_train = x[:train_cut - cfg['history'], :]
            y_train = y[:train_cut - cfg['history']]
            x_train = x_train[np.logical_not(missing), :]
            y_train = y_train[np.logical_not(missing), :]

            x_test = x[train_cut - cfg['history']:, :]

            clf.fit(x_train, y_train)
            speed_horizon = clf.predict(x_test)

        # RF local
        if cfg["inputtype"] == "local" and cfg["outputtype"] == "local":
            predictionmanager.setup(no_of_segments)
            for segment in range(no_of_segments):
            #for segment in range(5):
                print('\tSegment= %d' % segment)
                x, y = build_xy_set_local(dt, hor, corr_matrix[segment, :], cfg['timeattr'], cfg['history'], single_output=False)
                # missing = dt_time['missing_values'].values
                # missing = missing[cfg['history']:train_cut]  # missing values for the training set
                x_train = x[:train_cut - cfg['history'], :]
                y_train = y[:train_cut - cfg['history']]
                x_train = x_train[np.logical_not(missing), :]
                y_train = y_train[np.logical_not(missing)]
                x_test = x[train_cut - cfg['history']:, :]
                # train and predict
                clf.fit(x_train, y_train)
                y_predicted = clf.predict(x_test)
                #speed_horizon[:, segment] = y_predicted
                # save predicted segments
                predictionmanager.update(corr_matrix[segment, :], y_predicted)
            speed_horizon = predictionmanager.computeMeans()
        end = timer()

        # compute stats
        r, m = evaluation.computeError(y_test, speed_horizon, no_of_segments)
        RMSEBySector.append(r)
        MAPEBySector.append(m)
        r, m = evaluation.computeError(np.transpose(y_test), np.transpose(speed_horizon), no_of_segments)
        evaluation.fixTime(r, m, hor)
        RMSEByTimestep.append(r)
        MAPEByTimestep.append(m)
        # save data
        print('Save data')
        save(speed_horizon, RMSEBySector, MAPEBySector, RMSEByTimestep, MAPEByTimestep, hor, end-start)


def run_svr_temp(dt, dt_time):
    method_parameters = cfg['methodparam'][0]
    if cfg['methodname'] == 'SVR':
        clf = svm.SVR(**method_parameters)  # TODO try NuSVR et LinearSVR

    (no_of_samples, dt_cols) = dt.shape
    if cfg['timeattr']:
        time_attr = 3
    else:
        time_attr = 0
    no_of_segments = dt_cols - time_attr

    # compute each horizon one at a time
    for hor in range(cfg['horizon'][0], cfg['horizon'][1] + 1):
        print('Horizon= %d' % hor)
        # speedHorizon : no of prediction = size of test set - horizon
        speed_horizon = np.zeros([no_of_samples - train_cut - hor, no_of_segments])
        RMSEBySector = []
        MAPEBySector = []
        RMSEByTimestep = []
        MAPEByTimestep = []
        y_test = dt.iloc[train_cut + hor:, :].values
        if time_attr > 0:
            y_test = y_test[:, :-time_attr]  # drop time attribute

        if cfg["inputtype"] == "local" and cfg["outputtype"] == "unique":
            for segment in range(no_of_segments):
                print('\tSegment= %d' % segment)
                x, y = build_xy_set_local(dt, hor, segment, cfg['timeattr'], cfg['history'], single_output=True)
                missing = dt_time['missing_values'].values
                missing = missing[cfg['history']:train_cut]  # missing values for the training set
                x_train = x[:train_cut - cfg['history'], :]
                y_train = y[:train_cut - cfg['history']]
                x_train = x_train[np.logical_not(missing), :]
                y_train = y_train[np.logical_not(missing)]
                x_test = x[train_cut - cfg['history']:, :]
                # train and predict
                clf.fit(x_train, y_train)
                y_predicted = clf.predict(x_test)
                speed_horizon[:, segment] = y_predicted

        # compute stats
        r, m = evaluation.computeError(y_test, speed_horizon, no_of_segments)
        RMSEBySector.append(r)
        MAPEBySector.append(m)
        r, m = evaluation.computeError(np.transpose(y_test), np.transpose(speed_horizon), no_of_segments)
        evaluation.fixTime(r, m, hor)
        RMSEByTimestep.append(r)
        MAPEByTimestep.append(m)
        # save data
        print('Save data')
        save(speed_horizon, RMSEBySector, MAPEBySector, RMSEByTimestep, MAPEByTimestep, hor)


def split_data(dt, missing_values):
    dt['missing'] = missing_values
    split = 2928  # 2 months for training, one month for testing
    dt_train = dt.iloc[0:split, :]
    dt_test = dt.iloc[split:, :]
    # remove all columns with missing values on split sets
    dt_train = dt_train[dt_train['missing'] == 0]
    # dt_test = dt_test[dt_test['missing'] == 0]
    # remove column
    dt_train = dt_train.drop('missing', axis=1)
    # dt_test= dt_test.drop('missing', axis=1)
    return dt_train, dt_test


def build_correlation_matrix(dt, k=10):
    """
    Build the list of the k most correlated segments for each section (including the section itself)
    :param dt:
    :param k: number of neighbours
    :return:
    """
    arr = dt.corr().values
    # arr= abs(arr)
    res = arr.argsort(1)[:, ::-1][:, :k+1]  # take k highest (+1 for itself) correlated segment in order
    return res


def build_Xy_set_global(dt, h):
    X = dt.iloc[:-h, :].values # numpy array, remove last h row
    y = dt.iloc[h:, :].values # start at h
    return X, y


def build_Xy_set_local(dt, h, corr, timeAttr):
    # keep all columns in corr and the last 3 columns (if time attribute is used)
    (r, c) = dt.shape
    if timeAttr : colToKeep = np.hstack([corr, np.arange(c-3,c)])
    V = dt.iloc[:, colToKeep].values
    X = V[:-h, :]
    y = V[h:, 0]
    return X, y


def build_Xy_set_local_history(dt, horizon, corr, time_attr, history=1, univariate=1):
    # keep all columns in corr and the last 3 columns
    (r, c) = dt.shape  # TODO : time is duplicated.... useful ?
    if time_attr:
        col_to_keep = np.hstack([corr, np.arange(c - 3, c)])
    V = dt.iloc[:, col_to_keep].values
    # for Y, remove the first h samples
    if univariate:
        y = np.copy(V[horizon + history:, 0])
    else:
        y = np.copy(V[horizon + history:, :-3])  # all but time # TODO check
    # for X, remove the last h+t samples
    tmp = V[:-horizon, :]  # at time 0
    X = np.copy(tmp)
    for i in range(history):
        tmp = tmp[:-1, :]  # remove last time
        X = X[1:, :]  # remove first
        X = np.concatenate((X, tmp), axis= 1)
    return X, y


def build_Xy_set_global_history(dt, horizon, timeAttr, history):
    (r, c) = dt.shape # TODO : time is duplicated.... useful ?
    V = dt.values
    # for Y, remove the first h samples
    y = np.copy(V[horizon + history:, :-3])
    # for X, remove the last h+t samples
    tmp = V[:-horizon, :]  # at time 0
    X = np.copy(tmp)
    for i in range(history):
        tmp = tmp[:-1, :]  # remove last time
        X = X[1:, :]  # remove first
        X = np.concatenate((X, tmp), axis= 1)
    return X, y


def save(speedHorizon, RMSEBySector, MAPEBySector, RMSEByTimestep, MAPEByTimestep, horizon, duration= 0):
    # speed
    # saveData(savepath + 'GradientBoosting', speedHorizon, location[dataset], h)
    # stat
    fullname = 'SpeedHorizon%d.csv' % horizon
    name = path.join(output_path, fullname)
    # global stats
    name= path.join(output_path, 'stats.txt')
    with open(name, 'a') as f:
        f.write('%d\t%4.3f\t%d\n'%(horizon, np.mean(MAPEBySector), duration))
    # detailled stats
    fullname = 'SpeedHorizon%d.csv' % horizon
    name = path.join(output_path, fullname)
    np.savetxt(name, speedHorizon, delimiter=',', fmt='%.3f', newline='\n')
    name = path.join(output_path, 'RMSEBySection.csv')
    np.savetxt(name, RMSEBySector, delimiter=',', fmt='%.3f', newline='\n')
    name = path.join(output_path, 'MAPEBySection.csv')
    np.savetxt(name, MAPEBySector, delimiter=',', fmt='%.3f', newline='\n')
    name = path.join(output_path, 'RMSEByTimeStep.csv')
    np.savetxt(name, RMSEByTimestep, delimiter=',', fmt='%.3f', newline='\n')
    name = path.join(output_path, 'MAPEByTimeStep.csv')
    np.savetxt(name, MAPEByTimestep, delimiter=',', fmt='%.3f', newline='\n')

    # append mean MAPE

def runMethodRF():
    # load data
    dataname= '%s.csv'%cfg['location']
    datafile = path.join(cfg['datapath'], dataname)
    print("Working with %s" % datafile)
    dt = load_file(datafile)
    dataname= '%sWideMissingValues_BoolVector.csv'%cfg['location']
    missingValues= read_csv(path.join(cfg['datapath'], dataname), sep= ',', header=None)
    # prepare data
    (T,S)= dt.shape
    if cfg['corrmatrix'][0]:
        corrMatrix = build_correlation_matrix(dt, cfg['corrmatrix'][1])
    if cfg['timeattr']:
        dt, dttime = add_time_attribute(dt, missingValues)

    #dttrain, dttest = splitData(dt, missingValues)
    methodParam= cfg['methodparam'][0]

    ## methods
    if cfg['methodname'] == 'GradientBoosting':
        clf = ensemble.GradientBoostingRegressor(**methodParam)
    if cfg['methodname'] == 'RandomForest':
        clf= ensemble.RandomForestRegressor(**methodParam)
    if cfg['timeattr']:
        timeattr = 3
    else:
        timeattr = 0
    for h in range(cfg['horizon'][0], cfg['horizon'][1]+1):
        print('Horizon= %d'%h)
        # prepare outputs arrays
        # speedHorizon : no of prediction = size of test set - horizon
        speedHorizon = np.zeros([T - train_cut - h, S])
        RMSEBySector = []
        MAPEBySector = []
        RMSEByTimestep = []
        MAPEByTimestep = []

        Ytest= dt.iloc[train_cut + h:, :].values
        Ytest= Ytest[:, :-3] # drop time attribute
        if cfg['global'] and not(cfg['univariate']):
            X, Y = build_Xy_set_global_history(dt, h, cfg['timeattr'], cfg['history'])
            missing = dttime['missing_values'].values
            missing = missing[cfg['history']:train_cut]  # missing values for the training set
            Xtrain = X[:train_cut - cfg['history'], :]
            Ytrain = Y[:train_cut - cfg['history']]
            Xtrain = Xtrain[np.logical_not(missing), :]
            Ytrain = Ytrain[np.logical_not(missing), :]

            Xtest = X[train_cut - cfg['history']:, :]

            clf.fit(Xtrain, Ytrain)
            Ypred= clf.predict(Xtest)

            speedHorizon= Ypred

        # compute stats
        r,m= evaluation.computeError(Ytest, speedHorizon, S)
        RMSEBySector.append(r)
        MAPEBySector.append(m)
        r, m= evaluation.computeError(np.transpose(Ytest), np.transpose(speedHorizon), S)
        evaluation.fixTime(r,m, h)
        RMSEByTimestep.append(r)
        MAPEByTimestep.append(m)
        # save data
        print('Save data')
        save(speedHorizon, RMSEBySector, MAPEBySector, RMSEByTimestep, MAPEByTimestep, h)


def runMethodRFLocal(): # predict a set of segments
    # load data
    dataname = '%s.csv' % cfg['location']
    datafile = path.join(cfg['datapath'], dataname)
    print("Working with %s" % datafile)
    dt = load_file(datafile)
    dataname = '%sWideMissingValues_BoolVector.csv' % cfg['location']
    missingValues = read_csv(path.join(cfg['datapath'], dataname), sep=',', header=None)
    # prepare data
    (T, S) = dt.shape
    if cfg['corrmatrix'][0]:
        corrMatrix = build_correlation_matrix(dt, cfg['corrmatrix'][1])
    if cfg['timeattr']:
        dt, dttime = add_time_attribute(dt, missingValues)

    # dttrain, dttest = splitData(dt, missingValues)
    methodParam = cfg['methodparam'][0]

    ## methods
    if cfg['methodname'] == 'GradientBoosting':
        clf = ensemble.GradientBoostingRegressor(**methodParam)
    if cfg['methodname'] == 'RandomForest':
        clf = ensemble.RandomForestRegressor(**methodParam)
    if cfg['timeattr']:
        timeattr = 3
    else:
        timeattr = 0
    for h in range(cfg['horizon'][0], cfg['horizon'][1] + 1):
        print('Horizon= %d' % h)
        # prepare outputs arrays
        # speedHorizon : no of prediction = size of test set - horizon
        speedHorizon = np.zeros([T - train_cut - h, S])
        RMSEBySector = []
        MAPEBySector = []
        RMSEByTimestep = []
        MAPEByTimestep = []

        Ytest = dt.iloc[train_cut + h:, :].values
        Ytest = Ytest[:, :-3]  # drop time attribute

        predictionmanager.setup(S)
        if not (cfg['global']) and not(cfg['univariate']):
            for s in range(S):
                print('\tSegment= %d' % s)
                X, Y = build_Xy_set_local_history(dt, h, corrMatrix[s, :], cfg['timeattr'], cfg['history'], cfg['univariate'])
                # train for the first 2 months (modulo h) except when missing data is 1
                missing = dttime['missing_values'].values
                missing = missing[cfg['history']:train_cut]  # missing values for the training set
                Xtrain = X[:train_cut - cfg['history'], :]
                Ytrain = Y[:train_cut - cfg['history']]
                Xtrain = Xtrain[np.logical_not(missing), :]
                Ytrain = Ytrain[np.logical_not(missing)]
                # predict for the rest (keep missing values)
                Xtest = X[train_cut - cfg['history']:, :]
                # Ytest= Y[trainCut-cfg['history']:]

                # train and predict
                clf.fit(Xtrain, Ytrain)
                Ypred = clf.predict(Xtest)
                # save predicted segments
                predictionmanager.update(corrMatrix[s,:], Ypred)
                #speedHorizon[:, s] = Ypred

        # compute stats
        speedHorizon = predictionmanager.computeMeans()
        r, m = evaluation.computeError(Ytest, speedHorizon, S)
        RMSEBySector.append(r)
        MAPEBySector.append(m)
        r, m = evaluation.computeError(np.transpose(Ytest), np.transpose(speedHorizon), S)
        evaluation.fixTime(r, m, h)
        RMSEByTimestep.append(r)
        MAPEByTimestep.append(m)
        # save data
        print('Save data')
        save(speedHorizon, RMSEBySector, MAPEBySector, RMSEByTimestep, MAPEByTimestep, h)


def runMethodGBM():
    # load data
    dataname= '%s.csv'%cfg['location']
    datafile = path.join(cfg['datapath'], dataname)
    print("Working with %s" % datafile)
    dt = load_file(datafile)
    dataname= '%sWideMissingValues_BoolVector.csv'%cfg['location']
    missingValues= read_csv(path.join(cfg['datapath'], dataname), sep= ',', header=None)
    # prepare data
    (T,S)= dt.shape
    if cfg['corrmatrix'][0]:
        corrMatrix = build_correlation_matrix(dt, cfg['corrmatrix'][1])
    if cfg['timeattr']:
        dt, dttime = add_time_attribute(dt, missingValues)

    #dttrain, dttest = splitData(dt, missingValues)
    methodParam= cfg['methodparam'][0]

    ## methods
    if cfg['methodname'] == 'GradientBoosting':
        clf = ensemble.GradientBoostingRegressor(**methodParam)
    if cfg['methodname'] == 'RandomForest':
        clf= ensemble.RandomForestRegressor(**methodParam)
    if cfg['timeattr']:
        timeattr = 3
    else:
        timeattr = 0
    for h in range(cfg['horizon'][0], cfg['horizon'][1]+1):
        print('Horizon= %d'%h)
        # prepare outputs arrays
        # speedHorizon : no of prediction = size of test set - horizon
        speedHorizon = np.zeros([T - train_cut - h, S])
        RMSEBySector = []
        MAPEBySector = []
        RMSEByTimestep = []
        MAPEByTimestep = []

        Ytest= dt.iloc[train_cut + h:, :].values
        Ytest= Ytest[:, :-3] # drop time attribute
        if not(cfg['global']) and cfg['univariate']:
            for s in range(S):
                print('\tSegment= %d' % s)
                X, Y = build_Xy_set_local_history(dt, h, corrMatrix[s, :], cfg['timeattr'], cfg['history'], cfg['univariate'])
                # train for the first 2 months (modulo h) except when missing data is 1
                missing = dttime['missing_values'].values
                missing= missing[cfg['history']:train_cut]  # missing values for the training set
                Xtrain= X[:train_cut - cfg['history'], :]
                Ytrain= Y[:train_cut - cfg['history']]
                Xtrain= Xtrain[np.logical_not(missing),:]
                Ytrain= Ytrain[np.logical_not(missing)]
                # predict for the rest (keep missing values)
                Xtest= X[train_cut - cfg['history']:, :]
                #Ytest= Y[trainCut-cfg['history']:]

                # train and predict
                clf.fit(Xtrain, Ytrain)
                Ypred= clf.predict(Xtest)

                speedHorizon[:, s]= Ypred

        # compute stats
        r,m= evaluation.computeError(Ytest, speedHorizon, S)
        RMSEBySector.append(r)
        MAPEBySector.append(m)
        r, m= evaluation.computeError(np.transpose(Ytest), np.transpose(speedHorizon), S)
        evaluation.fixTime(r,m, h)
        RMSEByTimestep.append(r)
        MAPEByTimestep.append(m)
        # save data
        print('Save data')
        save(speedHorizon, RMSEBySector, MAPEBySector, RMSEByTimestep, MAPEByTimestep, h)




def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


def reportCsv(grid_scores, s):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:2]
    info= '%d\t'%s
    for i, score in enumerate(top_scores):
        b= '%4.4f\t%4.4f\t'%(score.mean_validation_score, np.std(score.cv_validation_scores))
        p= score.parameters
        c= '%3.3f\t%d\t%d\t%d\t'%(p.get('learning_rate'), p.get('n_estimators'), p.get('max_depth'), p.get('min_samples_split'))
        d= '%s\t%s\t'%(p.get('loss'), p.get('max_features'))
        info= '%s%s%s%s'%(info, b, c, d)
    print('%s'%info)
    return info


def runMethodSearch():
    dataname = '%s.csv' % cfg['location']
    datafile = path.join(cfg['datapath'], dataname)
    print("Working with %s" % datafile)
    dt = load_file(datafile)
    dataname = '%sWideMissingValues_BoolVector.csv' % cfg['location']
    missingValues = read_csv(path.join(cfg['datapath'], dataname), sep=',', header=None)
    (T, S) = dt.shape
    if cfg['timeattr']:
        dt = add_time_attribute(dt)
    if cfg['corrmatrix'][0]:
        corrMatrix = build_correlation_matrix(dt, cfg['corrmatrix'][1])
    dttrain, dttest = split_data(dt, missingValues)
    #methodParam = cfg['methodparam'][0]
    methodParam = {"loss": ['ls'],
                   "learning_rate": [0.01],
                   "n_estimators": [250],
                   "max_depth": [4, 6],
                   "min_samples_split": [1, 2],
                   "max_features": ['auto']}
    methodParam= {"loss": ['ls', 'lad', 'huber'], #, 'quantile'
              "learning_rate": [0.01],
              "n_estimators": [500,750],
              "max_depth": [4,6],
              "min_samples_split": [1,2,3],
              "max_features": ['auto', 'sqrt']} #, 'log2'

    ## methods
    if cfg['methodname'] == 'GradientBoosting':
        clf = ensemble.GradientBoostingRegressor()
    grid_search = GridSearchCV(clf, param_grid=methodParam, n_jobs=-1)

    for h in range(cfg['horizon'][0], cfg['horizon'][1] + 1):
        print('Horizon= %d' % h)
        Xtrain, Ytrain = build_Xy_set_global(dttrain, h)
        Xtest, Ytest = build_Xy_set_global(dttest, h)

        if cfg['timeattr']:
            skip = 3
        else:
            skip = 0
        name = path.join(output_path, 'gridsearch_%d.csv' % h)
        if cfg['global'] and cfg['univariate']:
            # fit and predict
            for s in range(S):
                print('\tSegment= %d' % s)
                Ytr = Ytrain[:, s]
                start = time()
                grid_search.fit(Xtrain, Ytr)
                print("GridSearchCV took %.0f m for %d candidate parameter settings."
                      % ((time() - start)/60, len(grid_search.grid_scores_)))
                #report(grid_search.grid_scores_)
                res= reportCsv(grid_search.grid_scores_, s)
                with open(name, 'a') as f:
                    f.write('%s\n' %res)


def runMethodNN():
    # load data
    dataname = '%s.csv' % cfg['location']
    datafile = path.join(cfg['datapath'], dataname)
    print("Working with %s" % datafile)
    dt = load_file(datafile)
    dataname = '%sWideMissingValues_BoolVector.csv' % cfg['location']
    missingValues = read_csv(path.join(cfg['datapath'], dataname), sep=',', header=None)
    # prepare data
    (T, S) = dt.shape
    if cfg['corrmatrix'][0]:
        corrMatrix = build_correlation_matrix(dt, cfg['corrmatrix'][1])
    if cfg['timeattr']:
        dt, dttime = add_time_attribute(dt, missingValues)

    # dttrain, dttest = splitData(dt, missingValues)
    methodParam = cfg['methodparam'][0]

    parameters= []

    ## methods
    if cfg['methodname'] == 'NeuralNetwork':
        clf = neural_network.MLRegressor(**parameters)
    if cfg['timeattr']:
        timeattr = 3
    else:
        timeattr = 0
    for h in range(cfg['horizon'][0], cfg['horizon'][1] + 1):
        print('Horizon= %d' % h)
        # prepare outputs arrays
        # speedHorizon : no of prediction = size of test set - horizon
        speedHorizon = np.zeros([T - train_cut - h, S])
        RMSEBySector = []
        MAPEBySector = []
        RMSEByTimestep = []
        MAPEByTimestep = []

        Ytest = dt.iloc[train_cut + h:, :].values
        Ytest = Ytest[:, :-3]  # drop time attribute
        if cfg['global'] and not (cfg['univariate']):
            X, Y = build_Xy_set_global_history(dt, h, cfg['timeattr'], cfg['history'])
            missing = dttime['missing_values'].values
            missing = missing[cfg['history']:train_cut]  # missing values for the training set
            Xtrain = X[:train_cut - cfg['history'], :]
            Ytrain = Y[:train_cut - cfg['history']]
            Xtrain = Xtrain[np.logical_not(missing), :]
            Ytrain = Ytrain[np.logical_not(missing), :]

            Xtest = X[train_cut - cfg['history']:, :]

            clf.fit(Xtrain, Ytrain)
            Ypred = clf.predict(Xtest)

            speedHorizon = Ypred

        # compute stats
        r, m = evaluation.computeError(Ytest, speedHorizon, S)
        RMSEBySector.append(r)
        MAPEBySector.append(m)
        r, m = evaluation.computeError(np.transpose(Ytest), np.transpose(speedHorizon), S)
        evaluation.fixTime(r, m, h)
        RMSEByTimestep.append(r)
        MAPEByTimestep.append(m)
        # save data
        print('Save data')
        save(speedHorizon, RMSEBySector, MAPEBySector, RMSEByTimestep, MAPEByTimestep, h)


if __name__ == "__main__":
    init(sys.argv[1:])
    data_frame, data_time, correlation_matrix = prepare_data()
    run_ml(data_frame, data_time, correlation_matrix)
    #run_svr_temp(data_frame, data_time)

    # runMethodRFLocal()