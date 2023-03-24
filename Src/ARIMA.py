
from statsmodels.tsa.arima.model import ARIMA
import pandas
from Metrics import evaluate_metrics

def build_ARIMA(tsTrain, p, d, q, metrics):
    model = ARIMA(tsTrain, order=(p,d,q), freq=tsTrain.index.inferred_freq)  
    results_ARIMA = model.fit()  
    evaluate_metrics(tsTrain,results_ARIMA.fittedvalues,"Train", metrics)
    return results_ARIMA


def predict_results(results_ARIMA,tsTrain, tsTest,metrics):        
    result_append = results_ARIMA.append(tsTest,freq=tsTest.index.inferred_freq)
    fc = result_append.forecast(tsTest.shape[0], alpha=0.05)  # 95% conf
    predict = result_append.predict(start=len(tsTrain), end=len(tsTrain)+len(tsTest)-1)
    fc_series = pandas.Series(predict, index=tsTest.index)
    evaluate_metrics(tsTest,fc_series,"Test", metrics)
    return fc_series