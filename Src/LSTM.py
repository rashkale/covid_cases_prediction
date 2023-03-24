import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import Data
import Metrics
import Plot

def get_model(trainX, trainY, units, epochs, batchsize, lag):
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units = units, input_shape=(1, lag),name= "LSTM_layer"))
    model.add(Dense(units = 1, name = "Dense_layer"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batchsize, verbose=0)
    return model

def get_predictions(model,dataX, scaler):
    # make predictions
    yPredict = model.predict(dataX)
    # invert normalisation
    yPredict = scaler.inverse_transform(yPredict)
    return yPredict

def get_improved_model(units, lag, layers, dropout, loss, optimizer, activation):
    # create and fit the improved LSTM network
    model = Sequential()
    model.add(LSTM(units = units, input_shape=(1, lag), return_sequences=True, name ="LSTM_layer_1"))
    for i in range(layers-1):
        if(i == layers-2):
            model.add(LSTM(units= units, name="LSTM_layer_"+str(i+2)))
        else:
            model.add(LSTM(units=units, return_sequences=True, name= "LSTM_layer_"+str(i+2)))
        model.add(Dropout(dropout, name ="Dropout_"+str(i+1)))     
    model.add(Dense(units = 1, activation = activation, name="Dense"))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def fit_improved_model(model, trainX, trainY,  epochs, batchsize):
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batchsize, verbose=0)
    return history

def pipeline(lag,split, ts, params, type):
    train, test = Data.prepare_data(ts, lag, split)

    scalerTrain = StandardScaler()
    train = scalerTrain.fit_transform(train.reshape(-1,1))

    scalerTest = StandardScaler()
    test = scalerTest.fit_transform(test.reshape(-1,1))

    trainX, trainY = Data.format_data(train, lag)
    testX, testY = Data.format_data(test, lag)
    metricsTrain = {"Model": type, "Data": "Train"}
    metricsTest = {"Model": type, "Data": "Test"}

    model = get_improved_model(params['units'],lag, params['layers'],params['dropout'], params['loss'], params['optimizer'], params['activation'])
    fit_improved_model(model,trainX,trainY,params['epocs'], params['batchsize'])
    #model = get_model(trainX, trainY, 20, 100,1, lag)
    trainPredict = get_predictions(model, trainX, scalerTrain)
    #print(trainPredict)
    testPredict = get_predictions(model, testX, scalerTest)
    trainY = scalerTrain.inverse_transform([trainY])
    testY = scalerTest.inverse_transform([testY])
    Metrics.evaluate_results(trainY, trainPredict, "Train", metricsTrain)
    Metrics.evaluate_results(testY, testPredict, "Test", metricsTest)
    tsTrainPred = Data.rearrange_data(trainPredict[:,0], ts, lag, trainPredict[:,0].shape[0]+lag)
    tsTestPred = Data.rearrange_data(testPredict[:,0], ts, trainPredict[:,0].shape[0]+1+lag+lag, trainPredict[:,0].shape[0]+testPredict[:,0].shape[0]+lag+lag-1)

    Plot.plot_result(ts,tsTrainPred,tsTestPred, "Result")
    return model, metricsTrain, metricsTest