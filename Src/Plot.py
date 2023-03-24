import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa import seasonal
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

#define funtion to plot the time series
def plot_timeseries(ts, color, total, pos,label):
    plt.figure(figsize=(30,20))
    plt.subplot(total, 1, pos)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel(label, fontsize=20)
    plt.plot(ts, color=color)

#define funtion to plot the decomposed components of the series
def plot_decompose(ts, color,label):
    
    fig, axes = plt.subplots(4, len(ts), figsize=(30, 10))
    for i in range(0, len(ts)):
        decompose_ts = seasonal.seasonal_decompose(ts[i].to_list(), period=365, model='additive')
        axes[0, i].plot(decompose_ts.observed, color=color[i])
        axes[0, i].set_ylabel('Observed')
        axes[1, i].plot(decompose_ts.trend, color=color[i])
        axes[1, i].set_ylabel('Trend')
        axes[2, i].plot(decompose_ts.seasonal, color = color[i])
        axes[2, i].set_ylabel('Seasonal')
        axes[3, i].plot(decompose_ts.resid, color=color[i])
        axes[3, i].set_ylabel('Residual')
        axes[3, i].set_xlabel(label[i])
    fig.tight_layout()
    return decompose_ts


#define function to plot acf and pacf graphs
def plot_acf_pacf(ts):
    plt.figure(figsize=(30,10))
    plt.subplot(211)
    plot_acf(ts,lags=100, ax= plt.gca());
    plt.subplot(212)
    plot_pacf(ts,lags=100, method='ywm', ax=plt.gca());

def plot_residuals(residuals):
    fig, ax = plt.subplots(1,2, figsize = (10, 3))
    #plt.figure(figsize=(30,10))
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.show()

def plot_results(tsTrain, tsTest, tsPredict):
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(tsTrain, label='training')
    plt.plot(tsTest, label='actual')
    plt.plot(tsPredict, label='predicted')    
    plt.show()


def plot_result(tsActual, tsTrainPred, tsTestPred, label):
    plt.figure(figsize=(30,10))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel(label, fontsize=20)
    plt.plot(tsActual, color = "blue")
    plt.plot(tsTrainPred, color = "green")
    plt.plot(tsTestPred, color = "orange")
    plt.legend(["Actual", "Train", "Test"], fontsize= 20)


def plot_metrics(ax, x, dfTrain, dfTest, metric):
    xpos = np.arange(len(x))  # the label locations
    width = 0.35  # the width of the bars
    ax.bar(xpos - width/2, dfTrain[metric], width, label = "Train")
    ax.bar(xpos + width/2, dfTest[metric], width, label = "Test")
    ax.set_ylabel(metric)
    #ax.set_title('Scores by group and gender')
    ax.set_xticks(xpos)
    ax.set_xticklabels(x, fontsize = 12)
    ax.legend()