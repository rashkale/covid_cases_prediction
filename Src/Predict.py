import Data
import numpy as np
import datetime

def forecast(model, test, lag, seq, scaler):
    test = test.dropna().values
    test = test.astype('float32').reshape(-1,1)
    test = scaler.fit_transform(test.reshape(-1,1))
    result = list()
    for i in range(seq):
        X, y = Data.format_data(test, lag)
        ypredict = Data.get_predictions(model, X, scaler)
        ylast = ypredict[-1]
        yresult = round(ylast[0])
        if yresult < 0:
            yresult = yresult * -1
        result.append(yresult)
        ylast = scaler.transform(ylast.reshape(-1,1))
        test = np.append(test,ylast).reshape(-1,1)
    return result

def get_future_dates(start, n):
    daterange = list()
    for i in range(n): 
        start += datetime.timedelta(days=1)
        daterange.append(start)
    return daterange

def get_cumulative_data(startValue, ts):
    ts2 = list()
    for i in range(len(ts)):
        ts2.append(ts[i]+startValue)
        startValue = ts2[i]
    return ts2