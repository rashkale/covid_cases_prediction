import pandas
import numpy as np

gitlink = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{}_global.csv'
#get data from the github dataset link
def get_dataset(type):
    dataset = pandas.read_csv(gitlink.format(type))
    #filter data for country India
    dfSeries = dataset[dataset['Country/Region'].str.contains('India')]
    #drop columns which are not required
    dfSeries = dfSeries.drop(dfSeries.columns[[0, 1, 2, 3]], axis = 1)
    #take a transpose to convert columns(dates) into rows
    deSeries_T = dfSeries.T
    #rename the column with number of cases
    return deSeries_T.rename(columns={ deSeries_T.columns[0]: 'cases'})

#convert from dataset to pandas series
def convert_timeseries(df):
    df['date'] = df.index
    #format the date and convert into datetime datatype
    df['day'] = pandas.to_datetime(df['date'], format='%d/%m/%y', errors='ignore', infer_datetime_format=True) 
    #set the dates as the index for the series
    ts = df.set_index('day')["cases"]
    return ts

#define funtion to remove sudden the spkies from a series
def remove_spikes(ts, threshold):
    cnt = 0
    prev = 0
    for i, v in ts.items():
        if(cnt == 0):
            prev = ts[i]
            cnt+= 1
            continue
        if((ts[i] - prev) > threshold):
            ts[i] = prev
        prev = ts[i]
        cnt+= 1
    return ts

#define function to remove negative values
def remove_negative(ts):
    for i, v in ts.items():
        if(ts[i] < 0):
            ts[i] = 0
    return ts

def prune_to_date(ts, from_date, to_date):
    return ts[from_date:to_date]

def train_test(ts,per):
    count = round(len(ts) * per)
    return ts[:count], ts[count:]

def format_data(ds, lag):
    arrX, arrY = [], []
    for i in range(len(ds)-lag-1):
        arrX.append(ds[i:(i+lag), 0])
        arrY.append(ds[i + lag, 0])
    arrX = np.array(arrX)
    arrY = np.array(arrY)
    arrX = np.reshape(arrX, (arrX.shape[0], 1, lag))
    return arrX, arrY

def rearrange_data(predicted, base, start, stop):
    ds = base.iloc[start:stop].copy(deep =True)
    for i in range(0, len(ds)):
        ds.iloc[i:i+1] = predicted[i]
    return ds

def prepare_data(ts, lag, split):
    dataset = ts.dropna().values
    dataset = dataset.astype('float32').reshape(-1,1)
    
    # split into train and test sets
    train_size = int(len(dataset) * split)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    return train, test




