
#Imports all the nueral nets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import activations
import matplotlib.pyplot as plt
import time
import os
import sys
import random
lib_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(lib_path)
sys.path.append('../package')
from package.curveplot import curveplot


#initialises signal dataframes
signal_300 = pd.DataFrame()
signal_400 = pd.DataFrame()
backround_all = pd.DataFrame()
#The signal frames I will use to test/train
signal_mass = [300, 400, 420, 440, 460, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2000]
epoch_length1 = 30


#Opens the csv files and unpacks them into respective dataframes
for each in signal_mass:
    df_temp = pd.read_csv(str(each) + ".csv", index_col=0)
    df_temp.drop(columns=["nTags", "MCChannelNumber"], inplace=True)
    if each == 300:
        test_signal_300 = pd.concat([df_temp], ignore_index=True)  
        test_signal_300["mass"] = 300 
    elif each == 400:
        test_signal_400 = pd.concat([df_temp], ignore_index=True)
        test_signal_400["mass"] = 400  
    elif each == 420:
        test_signal_420 = pd.concat([df_temp], ignore_index=True)
        test_signal_420["mass"] = 420  
    elif each == 440:
        test_signal_440 = pd.concat([df_temp], ignore_index=True)
        test_signal_440["mass"] = 440 
    elif each == 460:
        test_signal_460 = pd.concat([df_temp], ignore_index=True)
        test_signal_460["mass"] = 460 
    elif each == 500:
        test_signal_500 = pd.concat([df_temp], ignore_index=True)
        test_signal_500["mass"] = 500  
    elif each == 600:
        test_signal_600 = pd.concat([df_temp], ignore_index=True)
        test_signal_600["mass"] = 600  
    elif each == 700:
        test_signal_700 = pd.concat([df_temp], ignore_index=True)
        test_signal_700["mass"] = 700 
    elif each == 800:
        test_signal_800 = pd.concat([df_temp], ignore_index=True)
        test_signal_800["mass"] = 800 
    elif each == 900:
        test_signal_900 = pd.concat([df_temp], ignore_index=True)
        test_signal_900["mass"] = 900 
    elif each == 1000:
        test_signal_1000 = pd.concat([df_temp], ignore_index=True)
        test_signal_1000["mass"] = 1000 
    elif each == 1200:
        test_signal_1200 = pd.concat([df_temp], ignore_index=True)
        test_signal_1200["mass"] = 1200 
    elif each == 1400:
        test_signal_1400 = pd.concat([df_temp], ignore_index=True)
        test_signal_1400["mass"] = 1400  
    elif each == 1600:
        test_signal_1600 = pd.concat([df_temp], ignore_index=True) 
        test_signal_1600["mass"] = 1600 
    else:
        test_signal_2000 = pd.concat([df_temp], ignore_index=True)
        test_signal_2000["mass"] = 2000 

size_of_data_set = test_signal_300.shape[0] + test_signal_400.shape[0] + test_signal_420.shape[0] + test_signal_440.shape[0] + test_signal_460.shape[0] + test_signal_500.shape[0] + test_signal_600.shape[0] + test_signal_700.shape[0] + test_signal_800.shape[0]
+ test_signal_900.shape[0] + test_signal_1000.shape[0] + test_signal_1200.shape[0] + test_signal_1400.shape[0] + test_signal_1600.shape[0] + test_signal_2000.shape[0]
test_signal_300["weight"] = (test_signal_300.shape[0]/test_signal_300.shape[0])*test_signal_300["weight"]
test_signal_400["weight"] = (test_signal_300.shape[0]/test_signal_400.shape[0])*test_signal_400["weight"]
test_signal_420["weight"] = (test_signal_300.shape[0]/test_signal_420.shape[0])*test_signal_420["weight"]
test_signal_440["weight"] = (test_signal_300.shape[0]/test_signal_440.shape[0])*test_signal_440["weight"]
test_signal_460["weight"] = (test_signal_300.shape[0]/test_signal_460.shape[0])*test_signal_460["weight"]
test_signal_500["weight"] = (test_signal_300.shape[0]/test_signal_500.shape[0])*test_signal_500["weight"]
test_signal_600["weight"] = (test_signal_300.shape[0]/test_signal_600.shape[0])*test_signal_600["weight"]
test_signal_700["weight"] = (test_signal_300.shape[0]/test_signal_700.shape[0])*test_signal_700["weight"]
test_signal_800["weight"] = (test_signal_300.shape[0]/test_signal_800.shape[0])*test_signal_800["weight"]
test_signal_900["weight"] = (test_signal_300.shape[0]/test_signal_900.shape[0])*test_signal_900["weight"]
test_signal_1000["weight"] = (test_signal_300.shape[0]/test_signal_1000.shape[0])*test_signal_1000["weight"]
test_signal_1200["weight"] = (test_signal_300.shape[0]/test_signal_1200.shape[0])*test_signal_1200["weight"]
test_signal_1400["weight"] = (test_signal_300.shape[0]/test_signal_1400.shape[0])*test_signal_1400["weight"]
test_signal_1600["weight"] = (test_signal_300.shape[0]/test_signal_1600.shape[0])*test_signal_1600["weight"]
test_signal_2000["weight"] = (test_signal_300.shape[0]/test_signal_2000.shape[0])*test_signal_2000["weight"]


#unpacks the backround from the csv file into dataframe
background_all = pd.read_csv("background.csv", index_col=0)
background_all.drop(columns=["nTags", "MCChannelNumber"], inplace=True)
#background_all["weight"] = (background_all.shape[0]/test_signal_300.shape[0])*background_all["weight"]
masslist = []
for i in range(len(background_all)):
    masslist.append(signal_mass[random.randint(0,14)])
background_all.insert(17,"mass",masslist)
#combines all dataframes into 1 for the test datset
signal_all = pd.concat([test_signal_300,test_signal_400,test_signal_420,test_signal_440,test_signal_460,test_signal_500,test_signal_600,test_signal_700,test_signal_800,test_signal_900,test_signal_1000,test_signal_1200,test_signal_1400,test_signal_1600,test_signal_2000], ignore_index=True)

#Splits backround and signal into testing signal, testing backround,signal and backround
train_bkg, test_bkg = train_test_split(background_all, test_size=0.2)
train_signal, test_signal = train_test_split(signal_all, test_size=0.2)



#combines the test signals with the background so that we can use it as validation datasets in the the training.
test_x_300 = pd.concat([test_bkg, test_signal_300], ignore_index=True)
test_x_400 = pd.concat([test_bkg, test_signal_400], ignore_index=True)
test_x_420 = pd.concat([test_bkg, test_signal_420], ignore_index=True)
test_x_440 = pd.concat([test_bkg, test_signal_440], ignore_index=True)
test_x_460 = pd.concat([test_bkg, test_signal_460], ignore_index=True)
test_x_500 = pd.concat([test_bkg, test_signal_500], ignore_index=True)
test_x_600 = pd.concat([test_bkg, test_signal_600], ignore_index=True)
test_x_700 = pd.concat([test_bkg, test_signal_700], ignore_index=True)
test_x_800 = pd.concat([test_bkg, test_signal_800], ignore_index=True)
test_x_900 = pd.concat([test_bkg, test_signal_900], ignore_index=True)
test_x_1000 = pd.concat([test_bkg, test_signal_1000], ignore_index=True)
test_x_1200 = pd.concat([test_bkg, test_signal_1200], ignore_index=True)
test_x_1400 = pd.concat([test_bkg, test_signal_1400], ignore_index=True)
test_x_1600 = pd.concat([test_bkg, test_signal_1600], ignore_index=True)
test_x_2000 = pd.concat([test_bkg, test_signal_2000], ignore_index=True)
train_x = pd.concat([train_bkg, train_signal], ignore_index=True)
test_x = pd.concat([test_bkg, test_signal], ignore_index=True)

#DEfines which points are background (0) and which points are signal (1)
train_y = len(train_bkg) * [0] + len(train_signal) * [1]
test_y = len(test_bkg) * [0] + len(test_signal) * [1]
#Defines the weights of the data point and the dataset without the weight field
train_weight = train_x["weight"].to_numpy()
train_x.drop(columns=["weight"], inplace=True)
test_weight = test_x["weight"].to_numpy()
test_x.drop(columns=["weight"], inplace=True)

#Defines the above for every mass signal

#Rescales our data between -1 and 1
scaler = StandardScaler()
train_x_before = train_x
train_x = pd.DataFrame(scaler.fit_transform(train_x))
test_x = pd.DataFrame(scaler.transform(test_x))

#Defines the optimisation function we use that encodes the learning rate.
opt2 = keras.optimizers.Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999)
#defining Nueral net

#defines the shape of the nueral network
model = keras.models.Sequential()
model.add(keras.layers.Dense(100, input_dim=len(train_x_before.columns), activation="selu"))#the dimention of or data is how many columns train x has
model.add(keras.layers.Dropout(0.4))#Dropout is defined here to reduce overtraining
model.add(keras.layers.Dense(50, activation="selu"))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(50, activation="selu"))
model.add(keras.layers.Dense(1, activation=activations.sigmoid))
model.compile(loss="binary_crossentropy", optimizer=opt2, weighted_metrics=["accuracy"])#Setting loss to binary crossentropy makes the nueral etwork output a boolean 1 or 0
model.summary()
    
#trains the nueral net on the training data using the keras.fit function
history_300_selu = model.fit(train_x, np.array(train_y), sample_weight=train_weight, epochs=epoch_length1, validation_data=(test_x, np.array(test_y),test_weight), shuffle=True, batch_size=70)




#sets up the output of each fit function into its own dataframe
DF300_selu = pd.DataFrame(history_300_selu.history)


#plots each data frame as a line into the epochs vs loss graph
plt.plot(DF300_selu[["val_loss"]],label = "Validation set")
plt.plot(DF300_selu[["loss"]],label = "Training signal")
plt.legend()
plt.grid(True)
plt.xlabel("Epoch")
plt.ylabel("Validation dataset loss")
plt.gca()
plt.savefig('epochVSloss.pdf')


#plots the nueral net output against an arbritrary scale creating a new graph for each distinct signal mass
signal_mass = [300,400, 420, 440, 460, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2000]
for each_mass in signal_mass:
    test_bkg = test_bkg.assign(mass=each_mass)
    if each_mass == 300:
        thissignal = test_signal_300
    elif each_mass == 400:
        thissignal = test_signal_400
    elif each_mass == 420:
        thissignal = test_signal_420
    elif each_mass == 440:
        thissignal = test_signal_440
    elif each_mass == 460:
        thissignal = test_signal_460
    elif each_mass == 500:
        thissignal = test_signal_500
    elif each_mass == 600:
        thissignal = test_signal_600
    elif each_mass == 700:
        thissignal = test_signal_700
    elif each_mass == 800:
        thissignal = test_signal_800
    elif each_mass == 900:
        thissignal = test_signal_900
    elif each_mass == 1000:
        thissignal = test_signal_1000
    elif each_mass == 1200:
        thissignal = test_signal_1200
    elif each_mass == 1400:
        thissignal = test_signal_1400
    elif each_mass == 1600:
        thissignal = test_signal_1600
    else:
        thissignal = test_signal_2000
    

    bkgresultweight = test_bkg["weight"]
    sigresultweight = thissignal["weight"]
    bkgresult = model.predict(scaler.transform(test_bkg.drop(columns=["weight"]).to_numpy()))
    sigresult = model.predict(scaler.transform(thissignal.drop(columns=["weight"]).to_numpy()))
    
    bins = np.linspace(0, 1, 10)
    
   
    sighist, bin_edges = np.histogram(sigresult.flatten(), density=True, bins=bins, weights=sigresultweight.to_numpy())
    bkghist, bin_edges = np.histogram(bkgresult.flatten(), density=True, bins=bins, weights=bkgresultweight.to_numpy())
    
    curveplot([(bin_edges[0:-1] + bin_edges[1:])/2]*2, [sighist, bkghist], filename="output" + str(each_mass), ylimit=[0,20], labels=["sig", "bkg"], xlimit=[0, 1], 
              yshift=0.05, xshift=0.03, ylabel="arbitary unit", xlabel="NN output")
