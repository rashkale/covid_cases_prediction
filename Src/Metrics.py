from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
import math
import statistics
import numpy as np
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm


def evaluate_metrics(tsOriginal, tsPredict, suffix,metrics):
    r2s= r2_score(tsOriginal, tsPredict)
    metrics["R2Score"] = r2s
    print(suffix+ " R2-score: ", r2s)
    RSS =tsPredict.values-tsOriginal.values
    meanOrig = statistics.mean(tsOriginal.values)
    RMSE = math.sqrt((sum(RSS**2)/len(RSS)))
    metrics["RMSE"] = RMSE
    print(suffix+ ' NRMSE: %.4f'% (RMSE/meanOrig))
    metrics["NRMSE"] = RMSE/meanOrig
    return metrics

def evaluate_results(actual,predicted, suffix, metrics):
    # calculate root mean squared error
    r2s= r2_score(actual[0], predicted[:,0])
    print(suffix + " R2-score: ", r2s)
    metrics["R2Score"] = r2s
    RMSE = np.sqrt(mean_squared_error(actual[0], predicted[:,0]))
    metrics["RMSE"] = RMSE
    print(suffix + ' Score: %.2f RMSE' % (RMSE))
    Score = RMSE/np.mean(actual[0])
    print(suffix + ' Score: %.2f NRMSE' % (Score))
    metrics["NRMSE"] = Score


#define funtion to perform Augmented Dickey-Fuller test to check stationarity of series
def adf_test(ts):
    adf_result = sm.tsa.stattools.adfuller(ts)
    print("Adf statistic: ", adf_result[0])
    print("p-value", adf_result[1])
    print("Critical values: ")
    for key, value in adf_result[4].items():
        print('\t{}: {}'.format(key, value))