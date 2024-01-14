'''
Date Modified: 16/06/2023
Desc: To train logistic regression classifiers to predict FANET reliability and failure modes
Modified: For slurm
Modified: To use height and h_dist inputs instead of mean and std dev of SINR
Modified: To load the train dataset from "<modulation>_processed_train_video.csv" and the test dataset from "<modulation>_processed_holdout_video.csv"
'''

import pandas as pd
import numpy as np 
import os
import math
import pickle
import gc
from datetime import datetime

# Keras specific
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback

# Training params
EPOCHS = 5
checkpoint_filepath = '/home/wlau0003/Reuben_ws/nn_checkpoints/lr_multimodulation_video_nosinr_vid'
delay_threshold = 1

df_dtypes = {"TxTime": np.float32, "U2G_Distance": np.float32, "Height": np.int16,	"Num_Members": np.int16, "UAV_Sending_Interval": np.int16, "Bytes": np.int16, 
            "U2G_SINR": np.float32, "U2G_BER": np.float32, "Delay": np.float32, "Throughput": np.float32, "Queueing_Time": np.float32, "Packet_State": 'category', 
            "Retry_Count": np.int8, "Incorrectly_Received": np.int8, "Queue_Overflow": np.int8, "Packet_Name": 'string', "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
            "UAV_Sending_Interval": np.int16}

# Load training dataset ==========================================================================================================================
vid_df_bpsk = pd.read_csv("/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/BPSK_processed_train_video.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
vid_df_bpsk["Modulation"] = 1

vid_df_qpsk = pd.read_csv("/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QPSK_processed_train_video.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
vid_df_qpsk["Modulation"] = 0.3333

vid_df_qam16 = pd.read_csv("/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QAM16_processed_train_video.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
vid_df_qam16["Modulation"] = -0.3333

vid_df_qam64 = pd.read_csv("/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QAM64_processed_train_video.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
vid_df_qam64["Modulation"] = -1

vid_df_train = pd.concat([vid_df_bpsk, vid_df_qpsk, vid_df_qam16, vid_df_qam64], ignore_index=True)

vid_df_train.sort_values(by = "U2G_H_Dist")

    # Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
vid_df_train = vid_df_train.loc[vid_df_train["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]
# Load training dataset ==========================================================================================================================

# Load test dataset ==========================================================================================================================
vid_df_bpsk = pd.read_csv("/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/BPSK_processed_holdout_video.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
vid_df_bpsk["Modulation"] = 1

vid_df_qpsk = pd.read_csv("/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QPSK_processed_holdout_video.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
vid_df_qpsk["Modulation"] = 0.3333

vid_df_qam16 = pd.read_csv("/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QAM16_processed_holdout_video.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
vid_df_qam16["Modulation"] = -0.3333

vid_df_qam64 = pd.read_csv("/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QAM64_processed_holdout_video.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
vid_df_qam64["Modulation"] = -1

vid_df_holdout = pd.concat([vid_df_bpsk, vid_df_qpsk, vid_df_qam16, vid_df_qam64], ignore_index=True)

vid_df_holdout.sort_values(by = "U2G_H_Dist")

    # Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
vid_df_holdout = vid_df_holdout.loc[vid_df_holdout["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]
# Load test dataset ==========================================================================================================================

# Define ranges of input parameters
max_height = 300
min_height = 60
max_h_dist = 1200
min_h_dist = 0

# Normalize data (Min Max Normalization between [-1,1])
vid_df_train["Height"] = vid_df_train["Height"].apply(lambda x: 2*(x-min_height)/(max_height-min_height) - 1)
vid_df_train["U2G_H_Dist"] = vid_df_train["U2G_H_Dist"].apply(lambda x: 2*(x-min_h_dist)/(max_h_dist-min_h_dist) - 1)
vid_df_train["UAV_Sending_Interval"] = vid_df_train["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1})
vid_df_train['Packet_State'] = vid_df_train['Packet_State'].replace({"Reliable":0, "QUEUE_OVERFLOW":1, "RETRY_LIMIT_REACHED":2, "Delay_Exceeded":3})
vid_df_holdout["Height"] = vid_df_holdout["Height"].apply(lambda x: 2*(x-min_height)/(max_height-min_height) - 1)
vid_df_holdout["U2G_H_Dist"] = vid_df_holdout["U2G_H_Dist"].apply(lambda x: 2*(x-min_h_dist)/(max_h_dist-min_h_dist) - 1)
vid_df_holdout["UAV_Sending_Interval"] = vid_df_holdout["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1})
vid_df_holdout['Packet_State'] = vid_df_holdout['Packet_State'].replace({"Reliable":0, "QUEUE_OVERFLOW":1, "RETRY_LIMIT_REACHED":2, "Delay_Exceeded":3})

# Get inputs and outputs for train and test
X_train = vid_df_train[["U2G_H_Dist", "Height", "UAV_Sending_Interval", "Modulation"]].values
X_test = vid_df_holdout[["U2G_H_Dist", "Height", "UAV_Sending_Interval", "Modulation"]].values
# reliability_train = np.where(vid_df_train['Packet_State'] == 0, 1, 0)
# reliability_test = np.where(vid_df_holdout['Packet_State'] == 0, 1, 0)
# incr_rcvd_train = np.where(vid_df_train['Packet_State'] == 2, 1, 0)
# incr_rcvd_test = np.where(vid_df_holdout['Packet_State'] == 2, 1, 0)
# queue_overflow_train = np.where(vid_df_train['Packet_State'] == 1, 1, 0)
# queue_overflow_test = np.where(vid_df_holdout['Packet_State'] == 1, 1, 0)
delay_excd_train = np.where(vid_df_train['Packet_State'] == 3, 1, 0)
delay_excd_test = np.where(vid_df_holdout['Packet_State'] == 3, 1, 0)

# Clean up to save memory (so that oom don't make me cry)
del vid_df_train, vid_df_holdout, vid_df_bpsk, vid_df_qpsk, vid_df_qam16, vid_df_qam64
gc.collect()

# Build multiple models, one for each output
# reliability_model = Sequential()
# reliability_model.add(Dense(1, activation = 'sigmoid', input_dim = 4))
# reliability_model = tf.keras.models.load_model("/home/rlim0005/nn_checkpoints/lr_multimodulation_novideo_nosinr_dl/reliability_model.003-0.1469.h5", compile=False)
# incr_rcvd_model = Sequential()
# incr_rcvd_model.add(Dense(1, activation = 'sigmoid', input_dim = 4))
# queue_overflow_model = Sequential()
# queue_overflow_model.add(Dense(1, activation = 'sigmoid', input_dim = 4))
delay_excd_model = Sequential()
delay_excd_model.add(Dense(1, activation = 'sigmoid', input_dim = 4))

# Compile the models
# reliability_model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
# incr_rcvd_model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
# queue_overflow_model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
delay_excd_model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

# Record details of inputs and output for model
f = open(os.path.join(checkpoint_filepath,"model_details.txt"), "w")
f.write("Max Height (m): {}\n".format(max_height))
f.write("Min Height (m): {}\n".format(min_height))
f.write("Max H_Dist (m): {}\n".format(max_h_dist))
f.write("Min H_Dist (m): {}\n".format(min_h_dist))
f.write("[BPSK: 1, QPSK: 0.3333, QAM16: -0.3333, QAM64: -1]\n")
f.write("UAV Sending Interval: [10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1]\n")
f.write("Output: ['Reliable':0, 'QUEUE_OVERFLOW':1, 'RETRY_LIMIT_REACHED':2, 'Delay_Exceeded':3]\n")
f.close()

date = datetime.now()
date_str = date.strftime("%d%m%Y")

# Custom callback to clear memory to reduce RAM usage after each epoch
class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        keras.backend.clear_session()

# # Train reliability model ===============================================================
# model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
#     filepath=os.path.join(checkpoint_filepath,"reliability_model.{epoch:03d}-{val_loss:.4f}.h5"),
#     save_weights_only=False,
#     monitor='val_loss',
#     mode='auto',
#     save_freq='epoch')

# history = reliability_model.fit(X_train, reliability_train, epochs=EPOCHS, callbacks=[model_checkpoint_callback, ClearMemory()], validation_data=(X_test, reliability_test))
# with open(os.path.join(checkpoint_filepath, 'reliability_train_hist_{}'.format(date_str)), 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)
# # Save final model
# reliability_model.save(os.path.join(checkpoint_filepath,"reliability_final_model.h5"))
# del reliability_model, history # Free up memory
# gc.collect()
# # Train reliability model ===============================================================

# # Train incr rcvd model ===============================================================
# model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
#     filepath=os.path.join(checkpoint_filepath,"incr_rcvd_model.{epoch:03d}-{val_loss:.4f}.h5"),
#     save_weights_only=False,
#     monitor='val_loss',
#     mode='auto',
#     save_freq='epoch')

# history = incr_rcvd_model.fit(X_train, incr_rcvd_train, epochs=EPOCHS, callbacks=[model_checkpoint_callback, ClearMemory()], validation_data=(X_test, incr_rcvd_test))
# with open(os.path.join(checkpoint_filepath, 'incr_rcvd_train_hist_{}'.format(date_str)), 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)
# # Save final model
# incr_rcvd_model.save(os.path.join(checkpoint_filepath,"incr_rcvd_final_model.h5"))
# del incr_rcvd_model, history # Free up memory
# gc.collect()
# # Train incr rcvd model ===============================================================

# # Train queue_overflow model ===============================================================
# model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
#     filepath=os.path.join(checkpoint_filepath,"queue_overflow_model.{epoch:03d}-{val_loss:.4f}.h5"),
#     save_weights_only=False,
#     monitor='val_loss',
#     mode='auto',
#     save_freq='epoch')

# history = queue_overflow_model.fit(X_train, queue_overflow_train, epochs=EPOCHS, callbacks=[model_checkpoint_callback, ClearMemory()], validation_data=(X_test, queue_overflow_test))
# with open(os.path.join(checkpoint_filepath, 'queue_overflow_train_hist_{}'.format(date_str)), 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)
# # Save final model
# queue_overflow_model.save(os.path.join(checkpoint_filepath,"queue_overflow_final_model.h5"))
# del queue_overflow_model, history # Free up memory
# gc.collect()
# # Train queue_overflow model ===============================================================

# Train delay_excd model ===============================================================
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_filepath,"delay_excd_model.{epoch:03d}-{val_loss:.4f}.h5"),
    save_weights_only=False,
    monitor='val_loss',
    mode='auto',
    save_freq='epoch')

history = delay_excd_model.fit(X_train, delay_excd_train, epochs=EPOCHS, callbacks=[model_checkpoint_callback, ClearMemory()], validation_data=(X_test, delay_excd_test))
with open(os.path.join(checkpoint_filepath, 'delay_excd_train_hist_{}'.format(date_str)), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
# Save final model
delay_excd_model.save(os.path.join(checkpoint_filepath,"delay_excd_final_model.h5"))
# Train delay_excd model ===============================================================

