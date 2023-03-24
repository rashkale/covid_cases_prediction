import Data
import Metrics
import LSTM
import ARIMA
import Plot
import pandas

params = {"units" : 64,"layers" : 2,"dropout" : 0.0,"loss" : "mean_squared_error","optimizer" : "adam","activation" : "linear","epocs" : 26,"batchsize" : 1}

def run_default():
    ts_recovered, ts_confirmed, ts_deaths = import_data()
    analyse_data(ts_recovered, ts_confirmed, ts_deaths)
    ts_recovered_diff1, ts_confirmed_diff1, ts_deaths_diff1 = preprocess_data(ts_recovered, ts_confirmed, ts_deaths)
    # Mortality rate = (Number of Deaths/ Number of Confirmed cases) * 100
    tsMR = (ts_deaths/ts_confirmed)*100
    Plot.plot_timeseries(tsMR, 'red', 3, 1,"Mortality rate %")

    build_ARIMA(ts_confirmed_diff1)
    
    model_C, metricsTrain_C, metricsTest_C = build_LSTM(ts_confirmed_diff1,params, 8, 0.63, "Confirmed")

    model_R, metricsTrain_R, metricsTest_R = build_LSTM(ts_recovered_diff1,params, 8, 0.8, "Recovered")

    model_D, metricsTrain_D, metricsTest_D = build_LSTM(ts_deaths_diff1,params, 8, 0.63, "Deaths")

def import_data():
    #get data for recovered, confirmed and death cases
    df_recovered = Data.get_dataset("recovered")
    df_confirmed = Data.get_dataset("confirmed")
    df_deaths = Data.get_dataset("deaths")

    #get indivudual time series for recovered, confirmed and deaths
    ts_recovered = Data.convert_timeseries(df_recovered)
    ts_confirmed = Data.convert_timeseries(df_confirmed)
    ts_deaths = Data.convert_timeseries(df_deaths)

    return ts_recovered, ts_confirmed, ts_deaths


def analyse_data(ts_recovered, ts_confirmed, ts_deaths):
    #plot all three time series
    Plot.plot_timeseries(ts_confirmed, 'blue', 3, 1,"confirmed")
    Plot.plot_timeseries(ts_recovered, 'green', 3, 2,"recovered")
    Plot.plot_timeseries(ts_deaths, 'black', 3, 3,"deaths")

    #plotting the decomposed view of series
    decompose_ts = Plot.plot_decompose([ts_confirmed, ts_recovered, ts_deaths], ['blue', 'green', 'black'],["Confirmed","Recovered", "Deaths"])


    print("ADF test results with no differencing")
    Metrics.adf_test(ts_confirmed)


    print("ADF test results with 1 differencing")
    ts_confirmed_diff1 = ts_confirmed.diff().dropna()
    Metrics.adf_test(ts_confirmed_diff1)


def preprocess_data(ts_recovered, ts_confirmed, ts_deaths):
    #cleaning data
    ts_recovered_diff1 = Data.prune_to_date(ts_recovered,None,"Aug-04-2021").diff().dropna() #here we are dropping data after Aug-04-2021 because they have stopped updating recovered cases after that
    ts_deaths_diff1 = Data.remove_spikes(ts_deaths.diff().dropna(), 550) #here we are removing the spikes in the deaths data
    ts_confirmed_diff1 = Data.remove_negative(ts_confirmed.diff().dropna())

    Plot.plot_timeseries(ts_confirmed_diff1, 'blue', 3, 1,"confirmed")
    Plot.plot_timeseries(ts_recovered_diff1, 'green', 3, 2,"recovered")
    Plot.plot_timeseries(ts_deaths_diff1, 'black', 3, 3,"deaths")
    return ts_recovered_diff1, ts_confirmed_diff1, ts_deaths_diff1


    

def build_ARIMA(ts):
    Plot.plot_acf_pacf(ts)

    metricsARIMATrain = {"Model": "ARIMA", "Data": "Train"}
    metricsARIMATest = {"Model": "ARIMA", "Data": "Test"}

    tsTrain, tsTest = Data.train_test(ts, 0.62)
    results_ARIMA = ARIMA.build_ARIMA(tsTrain, 2,0,24,metricsARIMATrain)

    tsPredict = ARIMA.predict_results(results_ARIMA,tsTrain,tsTest,metricsARIMATest)
    Plot.plot_results(tsTrain,tsTest,tsPredict)


    #plotting residuals
    Plot.plot_residuals(pandas.DataFrame(results_ARIMA.resid))

    #printing the summary of ARIMA model
    results_ARIMA.summary()


def build_LSTM(ts, params, lag, split, type):
    model, metricsTrain, metricsTest = LSTM.pipeline(lag,split,ts, params, type)
    return model, metricsTrain, metricsTest



