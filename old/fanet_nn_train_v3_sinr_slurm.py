'''
Date: 03/03/2023
Desc: To train a NN classifier to predict FANET reliability and failure modes
Modified: For v2 NN and for slurm
'''

import pandas as pd
import numpy as np 
import sklearn
import os

# Import necessary modules
from sklearn.model_selection import train_test_split

# Keras specific
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input
from keras.utils import to_categorical 
import pickle

# Training params
EPOCHS = 30
TEST_SPLIT = 0.10 # Test split percentage
checkpoint_filepath = '/home/rlim0005/nn_checkpoints/nn_v3_hovering_sinr'
delay_threshold = 0.04
# checkpoint_filepath = "/home/research-student/omnet-fanet/nn_checkpoints"

# Compile all data
# dl_df_8uav = pd.read_hdf("/home/rlim0005/FANET_Dataset/Dataset_NP10000_BPSK_6-5Mbps/Dataset_NP10000_BPSK_6-5Mbps_8UAVs_processed_downlink.h5", '8_UAVs')
# dl_df_16uav = pd.read_hdf("/home/rlim0005/FANET_Dataset/Dataset_NP10000_BPSK_6-5Mbps/Dataset_NP10000_BPSK_6-5Mbps_16UAVs_processed_downlink.h5", '16_UAVs')
# dl_df_24uav = pd.read_hdf("/home/rlim0005/FANET_Dataset/Dataset_NP10000_BPSK_6-5Mbps/Dataset_NP10000_BPSK_6-5Mbps_24UAVs_processed_downlink.h5", '24_UAVs')
# dl_df_32uav = pd.read_hdf("/home/rlim0005/FANET_Dataset/Dataset_NP10000_BPSK_6-5Mbps/Dataset_NP10000_BPSK_6-5Mbps_32UAVs_processed_downlink.h5", '32_UAVs')
# dl_df_40uav = pd.read_hdf("/home/rlim0005/FANET_Dataset/Dataset_NP10000_BPSK_6-5Mbps/Dataset_NP10000_BPSK_6-5Mbps_40UAVs_processed_downlink.h5", '40_UAVs')
# dl_df = pd.concat([dl_df_8uav, dl_df_16uav, dl_df_24uav, dl_df_32uav, dl_df_40uav], ignore_index=True)

dl_df = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_64QAM_65Mbps_Hovering_8UAVs_processed_downlink.csv")

dl_df.sort_values(by = "U2G_H_Dist")

data_df = dl_df[["Mean_SINR", "Std_Dev_SINR", "Num_Members", "Bytes", "Sending_Interval", "Incorrectly_Received", "Queue_Overflow"]].copy()
data_df["Reliable"] = np.where(dl_df['Packet_State'] == "Reliable" , 1, 0)
data_df["Delay_Exceeded"] = np.where(dl_df['Delay'] > delay_threshold, 1, 0)

# Normalize data
max_mean_sinr = 6193 # The max mean SINR calculated at (0,24) is 6192.226659591827
max_std_dev_sinr = 2780 # The max std dev SINR calculated at (0,24) is 2779.468409580108
max_num_members = 39
max_bytes = 1145 # Should be 1144, but put 1145 just in case
# NOTE: Sending interval is already between 0 and 1 in the data
data_df["Mean_SINR"] = data_df["Mean_SINR"].div(max_mean_sinr)
data_df["Std_Dev_SINR"] = data_df["Std_Dev_SINR"].div(max_std_dev_sinr)
data_df["Num_Members"] = data_df["Num_Members"].div(max_num_members)
data_df["Bytes"] = data_df["Bytes"].div(max_bytes)

# Split to train and test
data_df_train, data_df_test = train_test_split(data_df, test_size=TEST_SPLIT, random_state=40, shuffle=False)
X_train = data_df_train[["Mean_SINR", "Std_Dev_SINR", "Num_Members", "Bytes", "Sending_Interval"]].values
X_test = data_df_test[["Mean_SINR", "Std_Dev_SINR", "Num_Members", "Bytes", "Sending_Interval"]].values
X_train_all = data_df[["Mean_SINR", "Std_Dev_SINR", "Num_Members", "Bytes", "Sending_Interval"]].values
reliability_train = data_df_train["Reliable"].values
reliability_test = data_df_test["Reliable"].values
reliability_train_all = data_df["Reliable"].values
incr_rcvd_train = data_df_train["Incorrectly_Received"].values
incr_rcvd_test = data_df_test["Incorrectly_Received"].values
incr_rcvd_train_all = data_df["Incorrectly_Received"].values
delay_excd_train = data_df_train["Delay_Exceeded"].values
delay_excd_test = data_df_test["Delay_Exceeded"].values
delay_excd_train_all = data_df["Delay_Exceeded"].values
queue_overflow_train = data_df_train["Queue_Overflow"].values
queue_overflow_test = data_df_test["Queue_Overflow"].values
queue_overflow_train_all = data_df["Queue_Overflow"].values

reliability_train = to_categorical(reliability_train) 
reliability_test = to_categorical(reliability_test)
reliability_train_all = to_categorical(reliability_train_all) 
incr_rcvd_train = to_categorical(incr_rcvd_train) 
incr_rcvd_test = to_categorical(incr_rcvd_test)
incr_rcvd_train_all = to_categorical(incr_rcvd_train_all) 
delay_excd_train = to_categorical(delay_excd_train) 
delay_excd_test = to_categorical(delay_excd_test)
delay_excd_train_all = to_categorical(delay_excd_train_all)
queue_overflow_train = to_categorical(queue_overflow_train) 
queue_overflow_test = to_categorical(queue_overflow_test)
queue_overflow_train_all = to_categorical(queue_overflow_train_all)


# For multiple output model
# Version 3: Large model additional hidden layer for each output layer (not shared among outputs)
inputs = Input(shape=(5,))
base = Dense(500, activation='relu')(inputs)
base = Dense(250, activation='relu')(base)
base = Dense(100, activation='relu')(base)
base = Dense(50, activation='relu')(base)
reliability_hl = Dense(25, activation='relu')(base)
reliability_hl = Dense(10, activation='relu')(reliability_hl)
incr_rcvd_out_hl = Dense(25, activation='relu')(base)
incr_rcvd_out_hl = Dense(10, activation='relu')(incr_rcvd_out_hl)
delay_excd_hl = Dense(25, activation='relu')(base)
delay_excd_hl = Dense(10, activation='relu')(delay_excd_hl)
queue_overflow_hl = Dense(25, activation='relu')(base)
queue_overflow_hl = Dense(10, activation='relu')(queue_overflow_hl)
reliability_out = Dense(2, activation='softmax', name='reliability')(reliability_hl)
incr_rcvd_out = Dense(8, activation='softmax', name='incorrectly_received')(incr_rcvd_out_hl)
delay_excd_out = Dense(2, activation='softmax', name='delay_exceeded')(delay_excd_hl)
queue_overflow_out = Dense(2, activation='softmax', name='queue_overflow')(queue_overflow_hl)
model = Model(inputs=inputs, outputs = [reliability_out, incr_rcvd_out, delay_excd_out, queue_overflow_out])

# Load pre-trained model for finetuning
# model = tf.keras.models.load_model(os.path.join(checkpoint_filepath, "model.010-2.0158.h5"), compile=False)

# Compile the model
model.compile(optimizer='adam', 
              loss={'reliability': 'binary_crossentropy',
                    'incorrectly_received': 'categorical_crossentropy',
                    'delay_exceeded': 'binary_crossentropy',
                    'queue_overflow': 'binary_crossentropy'},
              metrics={'reliability': 'accuracy',
                    'incorrectly_received': 'accuracy',
                    'delay_exceeded': 'accuracy',
                    'queue_overflow': 'accuracy'},)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_filepath,"model.{epoch:03d}-{val_loss:.4f}.h5"),
    save_weights_only=False,
    monitor='val_loss',
    mode='auto',
    save_freq='epoch')

# Y_train = [reliability_train, incr_rcvd_train, delay_excd_train, queue_overflow_train]
Y_train_all = [reliability_train_all, incr_rcvd_train_all, delay_excd_train_all, queue_overflow_train_all]
Y_test = [reliability_test, incr_rcvd_test, delay_excd_test, queue_overflow_test]
history = model.fit(X_train_all, Y_train_all, epochs=EPOCHS, callbacks=[model_checkpoint_callback], validation_data=(X_test, Y_test))
with open(os.path.join(checkpoint_filepath, 'trainHistoryDict_04042023'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Save final model
model.save(os.path.join(checkpoint_filepath,"final_model.h5"))