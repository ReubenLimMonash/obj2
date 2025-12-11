'''
Date: 22/05/2023
Desc: To train a NN classifier to predict FANET reliability and failure modes
Modified: For v2 NN and for slurm
Modified: For downlink - load the dataset in two parts and train (the dataset is not split by modulation)
'''

import pandas as pd
import numpy as np 
import sklearn
import os
import math

# Import necessary modules
from sklearn.model_selection import train_test_split

# Keras specific
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization
from keras.utils import to_categorical 
import pickle

# Training params
# EPOCHS = 5
TEST_SPLIT = 0.20 # Test split percentage
checkpoint_filepath = '/home/rlim0005/nn_checkpoints/nn_v4_multimodulation_novideo_sinr_dl'
delay_threshold = 1

# NOTE: Make sure to run fanet_split_dataset_slurm.py first
dl_df_part_1 = pd.read_pickle("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/downlink_df_part_1.pkl")
dl_df_part_2 = pd.read_pickle("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/downlink_df_part_2.pkl")

# Filter out rows where mean / std dev of sinr is NaN
dl_df_part_1 = dl_df_part_1[dl_df_part_1['Mean_SINR'].notna()]
dl_df_part_1 = dl_df_part_1[dl_df_part_1['Std_Dev_SINR'].notna()]
dl_df_part_2 = dl_df_part_2[dl_df_part_2['Mean_SINR'].notna()]
dl_df_part_2 = dl_df_part_2[dl_df_part_2['Std_Dev_SINR'].notna()]

dl_df_part_1.sort_values(by = "U2G_H_Dist")
dl_df_part_2.sort_values(by = "U2G_H_Dist")

# Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
dl_df_part_1 = dl_df_part_1.loc[dl_df_part_1["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]
dl_df_part_2 = dl_df_part_2.loc[dl_df_part_2["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]

# Normalize data (Min Max Normalization between [-1,1])
max_mean_sinr = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
max_std_dev_sinr = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
min_mean_sinr = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
min_std_dev_sinr = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)

dl_df_part_1["Mean_SINR"] = dl_df_part_1["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
dl_df_part_1["Std_Dev_SINR"] = dl_df_part_1["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1)
dl_df_part_1["UAV_Sending_Interval"] = dl_df_part_1["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1})
dl_df_part_1['Packet_State'] = dl_df_part_1['Packet_State'].replace({"Reliable":0, "QUEUE_OVERFLOW":1, "RETRY_LIMIT_REACHED":2, "Delay_Exceeded":3})
dl_df_part_2["Mean_SINR"] = dl_df_part_2["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
dl_df_part_2["Std_Dev_SINR"] = dl_df_part_2["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1)
dl_df_part_2["UAV_Sending_Interval"] = dl_df_part_2["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1})
dl_df_part_2['Packet_State'] = dl_df_part_2['Packet_State'].replace({"Reliable":0, "QUEUE_OVERFLOW":1, "RETRY_LIMIT_REACHED":2, "Delay_Exceeded":3})

# Split to train and test
df_train_part_1, df_test_part_1 = train_test_split(dl_df_part_1, test_size=TEST_SPLIT, random_state=40, shuffle=False)
del dl_df_part_1 # To free up memory space
X_train_part_1 = df_train_part_1[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation"]].values
X_test_part_1 = df_test_part_1[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation"]].values
packet_state_train_part_1 = df_train_part_1['Packet_State'].values
packet_state_test_part_1 = df_test_part_1['Packet_State'].values

df_train_part_2, df_test_part_2 = train_test_split(dl_df_part_2, test_size=TEST_SPLIT, random_state=40, shuffle=False)
del dl_df_part_2 # To free up memory space
X_train_part_2 = df_train_part_2[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation"]].values
X_test_part_2 = df_test_part_2[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation"]].values
packet_state_train_part_2 = df_train_part_2['Packet_State'].values
packet_state_test_part_2 = df_test_part_2['Packet_State'].values

# Convert output data to categorical type
packet_state_train_part_1 = to_categorical(packet_state_train_part_1)
packet_state_test_part_1 = to_categorical(packet_state_test_part_1)
packet_state_train_part_2 = to_categorical(packet_state_train_part_2)
packet_state_test_part_2 = to_categorical(packet_state_test_part_2)

# For multiple output model
# Version 2: Add an additional hidden layer for each output layer (not shared among outputs)
# Version 4: Having only a single output layer for packet state
inputs = Input(shape=(4,))
base = Dense(100, activation='relu')(inputs)
base = BatchNormalization()(base)
base = Dense(50, activation='relu')(base)
base = BatchNormalization()(base)
base = Dense(25, activation='relu')(base)
base = BatchNormalization()(base)
base = Dense(10, activation='relu')(base)
base = BatchNormalization()(base)
packet_state_out = Dense(4, activation='softmax', name='packet_state')(base)
model = Model(inputs=inputs, outputs = packet_state_out)

# Compile the model
model.compile(optimizer='adam', 
              loss={'packet_state': 'categorical_crossentropy'},
              metrics={'packet_state': 'accuracy'})

# Record details of inputs and output for model
f = open(os.path.join(checkpoint_filepath,"model_details.txt"), "w")
f.write("Mean and Std Dev SINR are in dB\n")
f.write("Max Mean SINR (dB): {}\n".format(max_mean_sinr))
f.write("Min Mean SINR (dB): {}\n".format(min_mean_sinr))
f.write("Max Std Dev SINR (dB): {}\n".format(max_std_dev_sinr))
f.write("Min Std Dev SINR (dB): {}\n".format(min_std_dev_sinr))
f.write("[BPSK: 1, QPSK: 0.3333, QAM16: -0.3333, QAM64: -1]\n")
f.write("UAV Sending Interval: [10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1]\n")
f.write("Output: ['Reliable':0, 'QUEUE_OVERFLOW':1, 'RETRY_LIMIT_REACHED':2, 'Delay_Exceeded':3]\n")
f.close()

# TRAIN PART 1 --------------------------------------------------------------------------------
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_filepath,"model_part_1.{epoch:03d}-{val_loss:.4f}.h5"),
    save_weights_only=False,
    monitor='val_loss',
    mode='auto',
    save_freq='epoch')
history = model.fit(X_train_part_1, packet_state_train_part_1, epochs=2, callbacks=[model_checkpoint_callback], validation_data=(X_test_part_1, packet_state_test_part_1))
with open(os.path.join(checkpoint_filepath, 'trainHistoryDict_part1_22052023'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
# TRAIN PART 1 --------------------------------------------------------------------------------

# TRAIN PART 2 --------------------------------------------------------------------------------
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_filepath,"model_part_2.{epoch:03d}-{val_loss:.4f}.h5"),
    save_weights_only=False,
    monitor='val_loss',
    mode='auto',
    save_freq='epoch')
history = model.fit(X_train_part_2, packet_state_train_part_2, epochs=2, callbacks=[model_checkpoint_callback], validation_data=(X_test_part_2, packet_state_test_part_2))
with open(os.path.join(checkpoint_filepath, 'trainHistoryDict_part2_22052023'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
# TRAIN PART 2 --------------------------------------------------------------------------------

# Save final model
model.save(os.path.join(checkpoint_filepath,"final_model.h5"))
