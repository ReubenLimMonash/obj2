'''
Date: 26/06/2023
Desc: To train an autoencoder as a feature extractor for reliability and failure mode classification
Modified: To load the train dataset from "<modulation>_processed_train_uplink.csv" and the test dataset from "<modulation>_processed_holdout_uplink.csv"
'''

import pandas as pd
import numpy as np 
import sklearn
import os
import math
import pickle
import gc
from datetime import datetime

# Keras specific
import keras
from keras.models import Model
from keras.layers import Dense, Input

# Training params
EPOCHS = 1
checkpoint_filepath = '/home/rlim0005/nn_checkpoints/ae_multimodulation_novideo_sinr_ul'

df_dtypes = {"TxTime": np.float32, "U2G_Distance": np.float32, "Height": np.int16,	"Num_Members": np.int16, "UAV_Sending_Interval": np.int16, "Bytes": np.int16, 
            "U2G_SINR": np.float32, "U2G_BER": np.float32, "Delay": np.float32, "Throughput": np.float32, "Queueing_Time": np.float32, "Packet_State": 'category', 
            "Retry_Count": np.int8, "Incorrectly_Received": np.int8, "Queue_Overflow": np.int8, "Packet_Name": 'string', "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
            "UAV_Sending_Interval": np.int16}

# Load training dataset ==========================================================================================================================
ul_df_bpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/BPSK_processed_train_uplink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
ul_df_bpsk["Modulation"] = 1

ul_df_qpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QPSK_processed_train_uplink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
ul_df_qpsk["Modulation"] = 0.3333

ul_df_qam16 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM16_processed_train_uplink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
ul_df_qam16["Modulation"] = -0.3333

ul_df_qam64 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM64_processed_train_uplink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
ul_df_qam64["Modulation"] = -1

ul_df_train = pd.concat([ul_df_bpsk, ul_df_qpsk, ul_df_qam16, ul_df_qam64], ignore_index=True)

ul_df_train.sort_values(by = "U2G_H_Dist")

    # Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
ul_df_train = ul_df_train.loc[ul_df_train["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]
# Load training dataset ==========================================================================================================================

# Load test dataset ==========================================================================================================================
ul_df_bpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/BPSK_processed_holdout_uplink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
ul_df_bpsk["Modulation"] = 1

ul_df_qpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QPSK_processed_holdout_uplink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
ul_df_qpsk["Modulation"] = 0.3333

ul_df_qam16 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM16_processed_holdout_uplink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
ul_df_qam16["Modulation"] = -0.3333

ul_df_qam64 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM64_processed_holdout_uplink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
ul_df_qam64["Modulation"] = -1

ul_df_holdout = pd.concat([ul_df_bpsk, ul_df_qpsk, ul_df_qam16, ul_df_qam64], ignore_index=True)

ul_df_holdout.sort_values(by = "U2G_H_Dist")

    # Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
ul_df_holdout = ul_df_holdout.loc[ul_df_holdout["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]
# Load test dataset ==========================================================================================================================

# Define ranges of input parameters
max_mean_sinr = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
max_std_dev_sinr = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
min_mean_sinr = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
min_std_dev_sinr = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)

# Normalize data (Min Max Normalization between [-1,1])
ul_df_train["Mean_SINR"] = ul_df_train["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
ul_df_train["Std_Dev_SINR"] = ul_df_train["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1)
ul_df_train["UAV_Sending_Interval"] = ul_df_train["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1})
ul_df_holdout["Mean_SINR"] = ul_df_holdout["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
ul_df_holdout["Std_Dev_SINR"] = ul_df_holdout["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1)
ul_df_holdout["UAV_Sending_Interval"] = ul_df_holdout["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1})

# Get inputs and outputs for train and test
X_train = ul_df_train[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation"]].values
X_test = ul_df_holdout[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation"]].values

# Clean up to save memory (so that oom don't make me cry)
del ul_df_train, ul_df_holdout, ul_df_bpsk, ul_df_qpsk, ul_df_qam16, ul_df_qam64
gc.collect()

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

# Autoencoder for feature extraction (reliability prediction model)
inputs = Input(shape=(4,))
encoder = Dense(100, activation='relu', name='encoder')(inputs)
latent = Dense(50, activation='relu', name='latent')(encoder)
decoder = Dense(100, activation='relu', name='decoder')(latent)
recon_in = Dense(4, activation='linear', name='reconstructed_inputs')(decoder)
model = Model(inputs=inputs, outputs = recon_in)

# Compile the model
model.compile(optimizer='adam', 
              loss='mse',
              metrics='mse')

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_filepath,"model.{epoch:03d}-{val_loss:.4f}.h5"),
    save_weights_only=False,
    monitor='val_loss',
    mode='auto',
    save_freq='epoch')

date = datetime.now()
date_str = date.strftime("%d%m%Y")

history = model.fit(X_train, X_train, epochs=EPOCHS, callbacks=[model_checkpoint_callback], validation_data=(X_test, X_test))
with open(os.path.join(checkpoint_filepath, 'trainHistoryDict_{}'.format(date_str)), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Save final model
model.save(os.path.join(checkpoint_filepath,"final_model.h5"))
