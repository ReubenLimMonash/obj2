'''
Date: 24/05/2023
Desc: To train logistic regression classifiers to predict FANET reliability and failure modes
'''

import pandas as pd
import numpy as np 
import sklearn
import os
import math
import gc
# Import necessary modules
from sklearn.model_selection import train_test_split

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
import pickle

# Training params
EPOCHS = 5
TEST_SPLIT = 0.30 # Test split percentage
checkpoint_filepath = '/home/rlim0005/nn_checkpoints/lr_multimodulation_novideo_sinr_ul'
delay_threshold = 1

df_dtypes = {"TxTime": np.float32, "U2G_Distance": np.float32, "Height": np.int16,	"Num_Members": np.int16, "UAV_Sending_Interval": np.int16, "Bytes": np.int16, 
            "U2G_SINR": np.float32, "U2G_BER": np.float32, "Delay": np.float32, "Throughput": np.float32, "Queueing_Time": np.float32, "Packet_State": 'category', 
            "Retry_Count": np.int8, "Incorrectly_Received": np.int8, "Queue_Overflow": np.int8, "Packet_Name": 'string', "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
            "UAV_Sending_Interval": np.int16}

ul_df_bpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/BPSK_processed_uplink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Incorrectly_Received", "Queue_Overflow",
                               "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
ul_df_bpsk["Modulation"] = 1
# Uncomment Below To Use One-Hot Encoding for Modulation
# ul_df_bpsk["Modulation"] = 0

ul_df_qpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QPSK_processed_uplink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Incorrectly_Received", "Queue_Overflow",
                               "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
ul_df_qpsk["Modulation"] = 0.3333
# Uncomment Below To Use One-Hot Encoding for Modulation
# ul_df_qpsk["Modulation"] = 1

ul_df_qam16 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM16_processed_uplink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Incorrectly_Received", "Queue_Overflow",
                               "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
ul_df_qam16["Modulation"] = -0.3333
# Uncomment Below To Use One-Hot Encoding for Modulation
# ul_df_qam16["Modulation"] = 2

ul_df_qam64 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM64_processed_uplink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Incorrectly_Received", "Queue_Overflow",
                               "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
ul_df_qam64["Modulation"] = -1
# Uncomment Below To Use One-Hot Encoding for Modulation
# ul_df_qam64["Modulation"] = 3

ul_df = pd.concat([ul_df_bpsk, ul_df_qpsk, ul_df_qam16, ul_df_qam64], ignore_index=True)
# ul_df = pd.concat([ul_df_bpsk, ul_df_qam64], ignore_index=True)

# Filter out rows where mean / std dev of sinr is NaN
ul_df = ul_df[ul_df['Mean_SINR'].notna()]
ul_df = ul_df[ul_df['Std_Dev_SINR'].notna()]

ul_df.sort_values(by = "U2G_H_Dist")

# Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
ul_df = ul_df.loc[ul_df["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]

# Normalize data (Min Max Normalization between [-1,1])
max_mean_sinr = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
max_std_dev_sinr = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
# max_uav_send_int = 1000 # The max UAV sending interval is 1000 ms
min_mean_sinr = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
min_std_dev_sinr = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)
# min_uav_send_int = 10 # The min UAV sending interval is 10 ms
ul_df["Mean_SINR"] = ul_df["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
ul_df["Std_Dev_SINR"] = ul_df["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1)
# ul_df["UAV_Sending_Interval"] = ul_df["UAV_Sending_Interval"].apply(lambda x: 2*(x-min_uav_send_int)/(max_uav_send_int-min_uav_send_int) - 1)
ul_df["UAV_Sending_Interval"] = ul_df["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1})
ul_df['Packet_State'] = ul_df['Packet_State'].replace({"Reliable":0, "QUEUE_OVERFLOW":1, "RETRY_LIMIT_REACHED":2, "Delay_Exceeded":3})

# Record details of inputs and output for model
f = open(os.path.join(checkpoint_filepath,"model_details.txt"), "w")
f.write("Mean and Std Dev SINR are in dB\n")
f.write("Max Mean SINR (dB): {}\n".format(max_mean_sinr))
f.write("Min Mean SINR (dB): {}\n".format(min_mean_sinr))
f.write("Max Std Dev SINR (dB): {}\n".format(max_std_dev_sinr))
f.write("Min Std Dev SINR (dB): {}\n".format(min_std_dev_sinr))
f.write("[BPSK: 1, QPSK: 0.3333, QAM16: -0.3333, QAM64: -1]\n")
f.write("UAV Sending Interval: [10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1]\n")
f.close()

# Split to train and test
data_df_train, data_df_test = train_test_split(ul_df, test_size=0.3, random_state=40, shuffle=False)
# X_train = data_df_train[["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval"]].values
X_test = data_df_test[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation"]].values
X_train_all = ul_df[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation"]].values
# Uncomment Below To Use One-Hot Encoding for Modulation ---------------------------------------
# modulation_train = to_categorical(data_df_train["Modulation"].values)
# modulation_test = to_categorical(data_df_test["Modulation"].values)
# modulation_train_all = to_categorical(ul_df["Modulation"].values)
# X_test = np.concatenate([data_df_test[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval"]].values, moduldata_df_testulation_test], axis=1)
# X_train_all = np.concatenate([ul_df[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval"]].values, modulation_train_all], axis=1)
# ----------------------------------------------------------------------------------------------
# reliability_train = np.where(data_df_train['Packet_State'] == 0, 1, 0)
reliability_test = np.where(data_df_test['Packet_State'] == 0, 1, 0)
reliability_train_all = np.where(ul_df['Packet_State'] == 0, 1, 0)
# incr_rcvd_train = np.where(data_df_train['Packet_State'] == 2, 1, 0)
incr_rcvd_test = np.where(data_df_test['Packet_State'] == 2, 1, 0)
incr_rcvd_train_all = np.where(ul_df['Packet_State'] == 2, 1, 0)
# queue_overflow_train = np.where(data_df_train['Packet_State'] == 1, 1, 0)
queue_overflow_test = np.where(data_df_test['Packet_State'] == 1, 1, 0)
queue_overflow_train_all = np.where(ul_df['Packet_State'] == 1, 1, 0)
# delay_excd_train = np.where(data_df_train['Packet_State'] == 3, 1, 0)
delay_excd_test = np.where(data_df_test['Packet_State'] == 3, 1, 0)
delay_excd_train_all = np.where(ul_df['Packet_State'] == 3, 1, 0)
del ul_df, data_df_train, data_df_test # To free up memory space
gc.collect()

# Build multiple models, one for each output
reliability_model = Sequential()
reliability_model.add(Dense(1, activation = 'sigmoid', input_dim = 4))
incr_rcvd_model = Sequential()
incr_rcvd_model.add(Dense(1, activation = 'sigmoid', input_dim = 4))
queue_overflow_model = Sequential()
queue_overflow_model.add(Dense(1, activation = 'sigmoid', input_dim = 4))
delay_excd_model = Sequential()
delay_excd_model.add(Dense(1, activation = 'sigmoid', input_dim = 4))

# Compile the models
reliability_model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
incr_rcvd_model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
queue_overflow_model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
delay_excd_model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

# # Train reliability model ===============================================================
# model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
#     filepath=os.path.join(checkpoint_filepath,"reliability_model.{epoch:03d}-{val_loss:.4f}.h5"),
#     save_weights_only=False,
#     monitor='val_loss',
#     mode='auto',
#     save_freq='epoch')

# history = reliability_model.fit(X_train_all, reliability_train_all, epochs=EPOCHS, callbacks=[model_checkpoint_callback], validation_data=(X_test, reliability_test))
# with open(os.path.join(checkpoint_filepath, 'reliability_train_hist_24052023'), 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)
# # Save final model
# reliability_model.save(os.path.join(checkpoint_filepath,"reliability_final_model.h5"))
# del reliability_model, history # Free up memory
# gc.collect()
# # Train reliability model ===============================================================

# Train incr rcvd model ===============================================================
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_filepath,"incr_rcvd_model.{epoch:03d}-{val_loss:.4f}.h5"),
    save_weights_only=False,
    monitor='val_loss',
    mode='auto',
    save_freq='epoch')

history = incr_rcvd_model.fit(X_train_all, incr_rcvd_train_all, epochs=EPOCHS, callbacks=[model_checkpoint_callback], validation_data=(X_test, incr_rcvd_test))
with open(os.path.join(checkpoint_filepath, 'incr_rcvd_train_hist_24052023'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
# Save final model
incr_rcvd_model.save(os.path.join(checkpoint_filepath,"incr_rcvd_final_model.h5"))
del incr_rcvd_model, history # Free up memory
gc.collect()
# Train incr rcvd model ===============================================================

# Train queue_overflow model ===============================================================
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_filepath,"queue_overflow_model.{epoch:03d}-{val_loss:.4f}.h5"),
    save_weights_only=False,
    monitor='val_loss',
    mode='auto',
    save_freq='epoch')

history = queue_overflow_model.fit(X_train_all, queue_overflow_train_all, epochs=EPOCHS, callbacks=[model_checkpoint_callback], validation_data=(X_test, queue_overflow_test))
with open(os.path.join(checkpoint_filepath, 'queue_overflow_train_hist_24052023'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
# Save final model
queue_overflow_model.save(os.path.join(checkpoint_filepath,"queue_overflow_final_model.h5"))
del queue_overflow_model, history # Free up memory
gc.collect()
# Train queue_overflow model ===============================================================

# Train delay_excd model ===============================================================
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_filepath,"delay_excd_model.{epoch:03d}-{val_loss:.4f}.h5"),
    save_weights_only=False,
    monitor='val_loss',
    mode='auto',
    save_freq='epoch')

history = delay_excd_model.fit(X_train_all, delay_excd_train_all, epochs=EPOCHS, callbacks=[model_checkpoint_callback], validation_data=(X_test, delay_excd_test))
with open(os.path.join(checkpoint_filepath, 'delay_excd_train_hist_24052023'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
# Save final model
delay_excd_model.save(os.path.join(checkpoint_filepath,"delay_excd_final_model.h5"))
# Train delay_excd model ===============================================================
