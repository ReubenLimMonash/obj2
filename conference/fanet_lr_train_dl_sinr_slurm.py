'''
Date: 25/05/2023
Desc: To train logistic regression classifiers to predict FANET reliability and failure modes
'''

import pandas as pd
import numpy as np 
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
EPOCHS = 1
TEST_SPLIT = 0.20 # Test split percentage
checkpoint_filepath = '/home/rlim0005/nn_checkpoints/lr_multimodulation_novideo_sinr_dl'
delay_threshold = 1

# NOTE: Make sure to run fanet_split_dataset_slurm.py first
dl_df = pd.read_pickle("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/downlink_df_all.pkl")

# Filter out rows where mean / std dev of sinr is NaN
dl_df = dl_df[dl_df['Mean_SINR'].notna()]
dl_df = dl_df[dl_df['Std_Dev_SINR'].notna()]

dl_df.sort_values(by = "U2G_H_Dist")

# Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
dl_df = dl_df.loc[dl_df["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]

# Normalize data (Min Max Normalization between [-1,1])
max_mean_sinr = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
max_std_dev_sinr = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
min_mean_sinr = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
min_std_dev_sinr = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)

dl_df["Mean_SINR"] = dl_df["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
dl_df["Std_Dev_SINR"] = dl_df["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1)
dl_df["UAV_Sending_Interval"] = dl_df["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1})
dl_df['Packet_State'] = dl_df['Packet_State'].replace({"Reliable":0, "QUEUE_OVERFLOW":1, "RETRY_LIMIT_REACHED":2, "Delay_Exceeded":3})

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

# Split to train and test
df_train, df_test = train_test_split(dl_df, test_size=TEST_SPLIT, random_state=40, shuffle=False)
del dl_df # To free up memory space
X_train = df_train[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation"]].values
X_test = df_test[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation"]].values
# ----------------------------------------------------------------------------------------------
reliability_train = np.where(df_train['Packet_State'] == 0, 1, 0)
reliability_test = np.where(df_test['Packet_State'] == 0, 1, 0)
incr_rcvd_train = np.where(df_train['Packet_State'] == 2, 1, 0)
incr_rcvd_test = np.where(df_test['Packet_State'] == 2, 1, 0)
queue_overflow_train = np.where(df_train['Packet_State'] == 1, 1, 0)
queue_overflow_test = np.where(df_test['Packet_State'] == 1, 1, 0)
delay_excd_train = np.where(df_train['Packet_State'] == 3, 1, 0)
delay_excd_test = np.where(df_test['Packet_State'] == 3, 1, 0)
del df_train, df_test # To free up memory space
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

# Train reliability model ===============================================================
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_filepath,"reliability_model.{epoch:03d}-{val_loss:.4f}.h5"),
    save_weights_only=False,
    monitor='val_loss',
    mode='auto',
    save_freq='epoch')

history = reliability_model.fit(X_train, reliability_train, epochs=EPOCHS, callbacks=[model_checkpoint_callback], validation_data=(X_test, reliability_test))
with open(os.path.join(checkpoint_filepath, 'reliability_train_hist_24052023'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
# Save final model
reliability_model.save(os.path.join(checkpoint_filepath,"reliability_final_model.h5"))
del reliability_model, history # Free up memory
gc.collect()
# Train reliability model ===============================================================

# Train incr rcvd model ===============================================================
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_filepath,"incr_rcvd_model.{epoch:03d}-{val_loss:.4f}.h5"),
    save_weights_only=False,
    monitor='val_loss',
    mode='auto',
    save_freq='epoch')

history = incr_rcvd_model.fit(X_train, incr_rcvd_train, epochs=EPOCHS, callbacks=[model_checkpoint_callback], validation_data=(X_test, incr_rcvd_test))
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

history = queue_overflow_model.fit(X_train, queue_overflow_train, epochs=EPOCHS, callbacks=[model_checkpoint_callback], validation_data=(X_test, queue_overflow_test))
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

history = delay_excd_model.fit(X_train, delay_excd_train, epochs=EPOCHS, callbacks=[model_checkpoint_callback], validation_data=(X_test, delay_excd_test))
with open(os.path.join(checkpoint_filepath, 'delay_excd_train_hist_24052023'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
# Save final model
delay_excd_model.save(os.path.join(checkpoint_filepath,"delay_excd_final_model.h5"))
# Train delay_excd model ===============================================================