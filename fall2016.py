from __future__ import division, print_function, absolute_import

import tflearn
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
import tensorflow as tf

time_window = 12
slide_time_window = 1
predict_steps = 3 #Number of steps in future to predict. Value of zero will
#imply the next step. Value of 1 implies to predict the next to next step in
#future and so on. Use this for model 1 and 2
#learn_rate = 0.33
learn_rate = 0.033
epochs = 2000
future_steps = 3 #Similar to predict_steps. Use this for model3. When using
#model 3, keep slide_time_window = 1
optimizer = 'adam'
lstm_units = 12
minibatch = 885

def generate_sin_data(step_radians=0.01, n=20):
    """Generates sine data points between 0 to n*pi in steps of step_radians."""
    num = np.arange(0, n*math.pi, step_radians)
    x_train = np.sin(num)
    test_num = np.arange(num[len(num) - time_window - predict_steps],
    (n+5)*math.pi, step_radians)
    x_test = np.sin(test_num)
    return x_train, x_test

def load_glucose_data():
    """Loads glucose data and scales the data values between 0 and 1"""
    df=pandas.read_csv("glucose_original.csv")
    df=df["Historic Glucose (mg/dL)"]
    f=df.tolist()
    f=np.array(f)
    max_f = np.amax(f)
    min_f = np.amin(f)
    f = (f - min_f)/(max_f - min_f)
    f_train = f[0:900]
    f_test = f[len(f_train)-time_window-predict_steps:]
    return f_train, f_test


#Prepare Data For Training/Testing
def model1_prepare_data(x):
    seq = []
    next_val = []

    for i in range(0, len(x) - time_window - predict_steps, slide_time_window):
        seq.append(x[i: i + time_window])
        next_val.append(x[i + time_window + predict_steps])

    next_val = np.reshape(next_val, [-1, 1])
    seq = np.reshape(seq, [-1, time_window, 1])

    inputs = np.array(seq)
    targets = np.array(next_val)

    return inputs, targets

def model2_prepare_data(x):
    seq = []
    next_val = []

    for i in range(0, len(x) - time_window - predict_steps, slide_time_window):
        seq.append(x[i: i + time_window])
        next_val.append(x[(i + time_window): (i+time_window+predict_steps+1)])

    seq = np.reshape(seq, [-1, time_window, 1])
    #next_val = np.reshape(next_val, [-1, predict_steps+1, 1])
    inputs = np.array(seq)
    targets = np.array(next_val)

    return inputs, targets

def model2_last_values(actual, predict):
    last_actual = []
    last_predict = []

    for i in range(0, len(actual)):
        last_actual.append(actual[i][predict_steps])
        last_predict.append(predict[i][predict_steps])

    return last_actual, last_predict

def model3_prepare_train_data(x):
    seq = []
    next_val = []

    for i in range(0, len(x) - time_window, slide_time_window):
        seq.append(x[i: i + time_window])
        next_val.append(x[i + time_window])

    next_val = np.reshape(next_val, [-1, 1])
    seq = np.reshape(seq, [-1, time_window, 1])

    inputs = np.array(seq)
    targets = np.array(next_val)

    return inputs, targets

def model3_prepare_test_data(x):
    seq = []
    next_val = []

    for i in range(0, len(x) - time_window - future_steps, slide_time_window):
        seq.append(x[i: i + time_window])
        next_val.append(x[i + time_window + future_steps])

    next_val = np.reshape(next_val, [-1, 1])
    seq = np.reshape(seq, [-1, time_window, 1])

    inputs = np.array(seq)
    targets = np.array(next_val)

    return inputs, targets

def model3_predict(seq, model):
    predict_Y = []

    for j in range(0, len(seq)):
        testwindow = seq[j]
        rstestwindow = np.reshape(testwindow, [1,time_window,1])
        for i in range(0, future_steps+1):
            predict_tw = model.predict(rstestwindow)
            nextwindow = np.append(rstestwindow[0][1:], predict_tw)
            rstestwindow = np.reshape(nextwindow, [1,time_window,1])
        predict_Y.append(predict_tw)

    rspredict_Y = np.reshape(predict_Y, [-1,1])
    return rspredict_Y

def add_noise(x, std_dev=0.08, mean=0):
    """Adds gaussian noise to the input array"""
    x = np.array(x)
    noise = std_dev*np.random.randn(x.shape[0]) + mean
    x = x + noise
    return x

def glu_previous_window(train, test):

    def prepare(x, now):
        x = [list(i) for i in x]
        windows=[]
        for i in range(0, len(x) - time_window - predict_steps,
        slide_time_window):
            windows.append(x[i: i + time_window])
        inputs = np.array(windows)

        targets=[]
        for i in range(0, len(now) - time_window - predict_steps,
        slide_time_window):
            #targets.append(now[(i + time_window):
            #(i+time_window+predict_steps+1)])
            #targets.append(now[i+time_window+predict_steps])
            targets.append(now[(i+time_window):(i+time_window+predict_steps+1)])
        targets = np.array(targets)
        #targets = np.reshape(targets, [-1,1])
        return inputs, targets

    train_now = train[96:]
    train_previous = train[:-96]
    tr = zip(train_now, train_previous)
    train_inputs, train_targets = prepare(tr, train_now)

    leftover = train[-96:]
    total_test = np.concatenate((leftover, test))
    test_now=test
    test_previous=total_test[:-96]
    ts = zip(test_now, test_previous)
    test_inputs, test_targets = prepare(ts, test_now)

    return train_inputs, train_targets, test_inputs, test_targets

def model4_classifier(train, test):

    def prepare(x):
        seq = []
        next_val = []

        for i in range(0, len(x) - time_window - predict_steps, slide_time_window):
            seq.append(x[i: i + time_window])
            if x[i + time_window + predict_steps]<80:
                next_val.append([1,0,0])
            elif x[i + time_window + predict_steps]<140:
                next_val.append([0,1,0])
            else:
                next_val.append([0,0,1])

        seq = np.reshape(seq, [-1, time_window, 1])

        inputs = np.array(seq)
        targets = np.array(next_val)

        return inputs, targets

    train_inputs, train_targets = prepare(train)
    test_inputs, test_targets = prepare(test)

    return train_inputs, train_targets, test_inputs, test_targets

#16/237, 76/327

# Build Network
def build_net():
    net = tflearn.input_data(shape=[None, time_window, 1])
    #net = tflearn.fully_connected(net, 10, activation='sigmoid')
    #net = tflearn.reshape(net, [-1, 10, 1])
    #net = tflearn.lstm(net, dropout=(1.0, 0.5), n_units=lstm_units, return_seq=True)
    #net = tflearn.lstm(net, n_units=lstm_units, return_seq=True)
    #net = tflearn.lstm(net, n_units=lstm_units, return_seq=True)
    #net = tflearn.lstm(net, n_units=lstm_units, return_seq=True)
    #net = tflearn.lstm(net, n_units=lstm_units, return_seq=True)
    #net = tflearn.lstm(net, n_units=lstm_units, return_seq=True)
    #net = tflearn.lstm(net, n_units=lstm_units, return_seq=True)
    #net = tflearn.lstm(net, n_units=lstm_units, return_seq=True)
    #net = tflearn.lstm(net, n_units=lstm_units, return_seq=True)
    #net = tflearn.lstm(net, dropout=(1.0, 0.5), n_units=lstm_units, return_seq=True)
    net = tflearn.lstm(net, n_units=lstm_units, return_seq=False)



    #net = tflearn.fully_connected(net, 8, activation='sigmoid')
    #net = tflearn.dropout(net, 0.5)
    #net = tflearn.fully_connected(net, lstm_units, activation='relu')
    #net = tflearn.dropout(net, 0.5)

    #For model 2
    net = tflearn.fully_connected(net, predict_steps+1, activation='sigmoid')

    #For model 1 and 3
    #net = tflearn.fully_connected(net, 1, activation='linear')

    #For model 4
    #net = tflearn.fully_connected(net, 3, activation='softmax')
    #net = tflearn.regression(net, metric='accuracy', optimizer=optimizer,
    #loss='categorical_crossentropy',
    #learning_rate=learn_rate, shuffle_batches=False)

    net = tflearn.regression(net, metric=None, optimizer=optimizer, loss='mean_square',
    learning_rate=learn_rate, shuffle_batches=False)

    model = tflearn.DNN(net, clip_gradients=0.0, tensorboard_verbose=0)
    return model

def build_net2():
    """Output units connected"""
    net = tflearn.input_data(shape=[None, time_window, 1])
    #net = tflearn.fully_connected(net, 10, activation='sigmoid')
    #net = tflearn.reshape(net, [-1, 10, 1])
    #net = tflearn.lstm(net, dropout=(1.0, 0.5), n_units=lstm_units, return_seq=True)
    net = tflearn.lstm(net, n_units=lstm_units, return_seq=False)
    #net = tflearn.fully_connected(net, 32, activation='relu')
    unit1 = tflearn.fully_connected(net, 1, activation='relu')
    merge1 = tflearn.merge([net, unit1], 'concat', axis=1)
    unit2 = tflearn.fully_connected(merge1, 1, activation='relu')
    merge2 = tflearn.merge([net, unit2], 'concat', axis=1)
    unit3 = tflearn.fully_connected(merge2, 1, activation='relu')
    merge3 = tflearn.merge([net, unit3], 'concat', axis=1)
    unit4 = tflearn.fully_connected(merge3, 1, activation='relu')
    outputs = tflearn.merge([unit1, unit2, unit3, unit4], 'concat', axis=1)

    net = tflearn.regression(outputs, metric=None, optimizer=optimizer, loss='mean_square',
    learning_rate=learn_rate, shuffle_batches=False)

    model = tflearn.DNN(net, clip_gradients=0.0, tensorboard_verbose=0)
    return model

def build_net3():
    """Bidirectional RNN"""
    net = tflearn.input_data(shape=[None, time_window, 1])
    lstm1 = tflearn.BasicLSTMCell(num_units=lstm_units)
    lstm2 = tflearn.BasicLSTMCell(num_units=lstm_units)
    #net = tflearn.fully_connected(net, 10, activation='sigmoid')
    #net = tflearn.reshape(net, [-1, 10, 1])
    #net = tflearn.lstm(net, dropout=(1.0, 0.5), n_units=lstm_units, return_seq=True)
    net = tflearn.bidirectional_rnn(net, lstm1, lstm2, return_seq=False)
    #net = tflearn.fully_connected(net, 5, activation='linear')
    #net = tflearn.dropout(net, 0.5)
    #net = tflearn.fully_connected(net, lstm_units, activation='relu')
    #net = tflearn.dropout(net, 0.5)

    #For model 2
    net = tflearn.fully_connected(net, predict_steps+1, activation='linear')

    #For model 1 and 3
    #net = tflearn.fully_connected(net, 1, activation='linear')

    #For model 4
    #net = tflearn.fully_connected(net, 3, activation='softmax')
    #net = tflearn.regression(net, metric='accuracy', optimizer=optimizer,
    #loss='categorical_crossentropy',
    #learning_rate=learn_rate, shuffle_batches=False)

    net = tflearn.regression(net, metric=None, optimizer=optimizer, loss='mean_square',
    learning_rate=learn_rate, shuffle_batches=False)

    model = tflearn.DNN(net, clip_gradients=0.0, tensorboard_verbose=0)
    return model
#Train Network
def fit_model(model, trainX, trainY, num_epochs=epochs, validation=0.0,
    mini_batch=minibatch, save='No'):
    model.fit(trainX, trainY, n_epoch=num_epochs, validation_set=validation,
    batch_size=mini_batch, shuffle=False, show_metric=True)
    if save=='yes' or 'y' or 'Y' or 'Yes' or 'YES':
        model.save('model1')

#mini_batch = 887 for glucose
# for sine mini_batch = 6270

def MSE(actual, predict):
    actual = np.array(actual)
    predict = np.array(predict)
    squared_loss = np.square(predict - actual)
    total_squared_loss = (1/(actual.shape[0]))*np.sum(squared_loss)
    return total_squared_loss

def plot(actual, predict, train_loss='NA', test_loss='NA',
save='model1.png'):
    # Plot test results
    plt.figure(figsize=(40,10))
    plt.suptitle('Prediction')
    plt.title('time_window='+str(time_window)+
    ', predict_steps='+str(predict_steps)+
    ', lstm_units='+str(lstm_units)+
    ', learn_rate='+str(learn_rate)+
    ', epochs='+str(epochs)+
    ', optimizer='+str(optimizer)+
    ', training_mse='+str(train_loss)+
    ', test_mse='+str(test_loss))
    plt.xlabel('Time Step')
    plt.ylabel('Glucose concentration scaled (mg/dL)')
    #plt.plot(actual, 'r.', label='Actual')
    #plt.plot(predict, 'b.', label='Predicted')
    plt.plot(actual, color='red', linestyle='dotted', marker='.', markerfacecolor='red',
             markersize='8', 		label='Actual')
    plt.plot(predict, color='blue', linestyle='dotted', marker='.', markerfacecolor='blue',
             markersize='8', 	label='Predicted')
    plt.grid(b=True, which='major', linestyle='dotted')
    plt.grid(b=True, which='minor', linestyle='dotted')
    plt.legend()
    #plt.show()
    plt.savefig(save)

def model1_run():
    train, test = load_glucose_data()
    trainX, trainY = model1_prepare_data(train)
    testX, testY = model1_prepare_data(test)
    #trainX, trainY, testX, testY = glu_previous_window(train, test)
    model=build_net3()
    fit_model(model, trainX, trainY)

    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)

    train_mse= MSE(trainY, train_predict)
    test_mse= MSE(testY, test_predict)

    print("Total training MSE is", train_mse)
    print("Total testing MSE is", test_mse)

    plot(trainY, train_predict,
    train_loss=train_mse, save='bi_glu_model1_train.png')
    plot(testY, test_predict, test_loss=test_mse,
    save='bi_glu_model1_test.png')


def model2_run():
    train, test = load_glucose_data()
    trainX, trainY = model2_prepare_data(train)
    testX, testY = model2_prepare_data(test)
    #print(np.shape(testX))
    #trainX, trainY, testX, testY = glu_previous_window(train, test)
    model = build_net()
    fit_model(model, trainX, trainY)


    test_predict = model.predict(testX)
    train_predict = model.predict(trainX)

    last_trainY, last_train_predict = model2_last_values(trainY, train_predict)
    last_testY, last_test_predict = model2_last_values(testY, test_predict)

    train_mse= MSE(last_trainY, last_train_predict)
    test_mse= MSE(last_testY, last_test_predict)

    def new1(allpredictions, lastvalues):
        """Ignore This"""
        lastvalues = np.array(lastvalues)
        lastvalues = np.reshape(lastvalues, [-1,1])
        inputs = allpredictions
        targets = lastvalues

        net = tflearn.input_data(shape=[None, 3])
        net = tflearn.fully_connected(net, 4, activation='sigmoid')
        net = tflearn.fully_connected(net, 1, activation='sigmoid')
        net = tflearn.regression(net, metric=None, optimizer='adam', loss='mean_square',
                                 learning_rate=0.01, shuffle_batches=False)
        model = tflearn.DNN(net, clip_gradients=0.0, tensorboard_verbose=0)
        model.fit(inputs, targets, n_epoch=20, validation_set=validation,
        batch_size=mini_batch, show_metric=True)


    print("Total training MSE is", train_mse)
    print("Total testing MSE is", test_mse)

    plot(last_trainY, last_train_predict,
    train_loss=train_mse,
    save='CONN_glu_model2_train.png')
    plot(last_testY, last_test_predict, test_loss=test_mse,
    save='CONN_glu_model2_test.png')

def model3_run():
    train, test = load_glucose_data()
    trainX, trainY = model3_prepare_train_data(train)
    testX, testY = model3_prepare_test_data(test)
    model = build_net()
    fit_model(model, trainX, trainY)
    test_predict = model3_predict(testX, model)

    test_mse= MSE(testY, test_predict)
    train_mse= 'NA'

    print("Total testing MSE is", test_mse)

    plot(testY, test_predict,test_loss=test_mse,
    save='glu_model3_test.png')

def model4_run():
    train, test = load_glucose_data()
    trainX, trainY, testX, testY = model4_classifier(train, test)
    #print(np.shape(trainX), np.shape(trainY))
    model=build_net()
    fit_model(model, trainX, trainY)

    print("Test accuracy is ", model.evaluate(testX, testY))


model2_run()
#train, test = load_glucose_data()
#plot(train, save='train_unscaled.png')
#plot(test, save='test_unscaled.png')
