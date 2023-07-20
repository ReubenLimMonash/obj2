'''
Date: 03/07/2023
Desc: To extract the eatures from autoencoder latent space and save it as inputs for training throughput monitoring.
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
from keras.layers import Dense, Input, BatchNormalization
from keras.utils import to_categorical 

# File paths
# autoencoder_model_path = '/home/rlim0005/nn_checkpoints/throughput_ae_multimodulation_novideo_sinr_ul/final_model.h5'
autoencoder_model_path = '/home/research-student/omnet-fanet/nn_checkpoints/throughput_ae_multimodulation_novideo_sinr_ul/final_model.h5'
# save_filepath = '/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo'
save_filepath = '/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_NoVideo'

df_dtypes = {"TxTime": np.float32, "U2G_Distance": np.float32, "Height": np.int16,	"Num_Members": np.int16, "UAV_Sending_Interval": np.int16, "Bytes": np.int16, 
            "U2G_SINR": np.float32, "U2G_BER": np.float32, "Delay": np.float32, "Throughput": np.float32, "Queueing_Time": np.float32, "Packet_State": 'category', 
            "Retry_Count": np.int8, "Incorrectly_Received": np.int8, "Queue_Overflow": np.int8, "Packet_Name": 'string', "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
            "UAV_Sending_Interval": np.int16}

# Load training dataset ==========================================================================================================================
# ul_df_bpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/BPSK_processed_throughput_uplink.csv",
ul_df_bpsk = pd.read_csv("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_NoVideo/BPSK_processed_throughput_uplink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
                    dtype=df_dtypes)
ul_df_bpsk["Modulation"] = 1

# ul_df_qpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QPSK_processed_throughput_uplink.csv",
ul_df_qpsk = pd.read_csv("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QPSK_processed_throughput_uplink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
                    dtype=df_dtypes)
ul_df_qpsk["Modulation"] = 0.3333

# ul_df_qam16 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM16_processed_throughput_uplink.csv",
ul_df_qam16 = pd.read_csv("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM16_processed_throughput_uplink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
                    dtype=df_dtypes)
ul_df_qam16["Modulation"] = -0.3333

# ul_df_qam64 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM64_processed_throughput_uplink.csv",
ul_df_qam64 = pd.read_csv("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM64_processed_throughput_uplink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
                    dtype=df_dtypes)
ul_df_qam64["Modulation"] = -1

ul_df_train = pd.concat([ul_df_bpsk, ul_df_qpsk, ul_df_qam16, ul_df_qam64], ignore_index=True)

# ul_df_train.sort_values(by = "Mean_SINR")
# Load training dataset ==========================================================================================================================

# Define ranges of input parameters
max_mean_sinr = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
max_std_dev_sinr = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
min_mean_sinr = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
min_std_dev_sinr = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)
max_throughput = 500000 # The max throughput is estimated as: 500,000 bytes/sec (uplink); 20,000 bytes/sec (downlink); 250,000 bytes/sec (video)
min_throughput = 0

# Normalize data (Min Max Normalization between [-1,1])
ul_df_train["Mean_SINR"] = ul_df_train["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
ul_df_train["Std_Dev_SINR"] = ul_df_train["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1)
ul_df_train["UAV_Sending_Interval"] = ul_df_train["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1})
ul_df_train["Throughput"] = ul_df_train["Throughput"].apply(lambda x: 2*(x-min_throughput)/(max_throughput-min_throughput) - 1)

# Get inputs and outputs for train and test
X_train = ul_df_train[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Throughput"]].values

# Clean up to save memory (so that oom don't make me cry)
del ul_df_train, ul_df_bpsk, ul_df_qpsk, ul_df_qam16, ul_df_qam64
gc.collect()

# Load the autoencoder feature extractor model
autoencoder = keras.models.load_model(autoencoder_model_path, compile=False)
autoencoder.compile(optimizer='adam', loss='mse', metrics='mse')
# Get the encoder part of the autoencoder
encoder_layer = autoencoder.get_layer('latent')
encoder = Model(inputs=autoencoder.input, outputs=encoder_layer.output)

# Get the autoencoder features from inputs
# ae_features_train = []
idx_train = math.floor(len(X_train)/1000)
for i in range(idx_train):
    # print(i)
    # ae_features_train.append(encoder.predict(X_train[i*1000:(i+1)*1000]))
    if i == 0:
        ae_features_train = encoder.predict(X_train[i*1000:(i+1)*1000])
    else:
        ae_features_train = np.concatenate((ae_features_train, encoder.predict(X_train[i*1000:(i+1)*1000])), axis=0)
if len(X_train) > idx_train*1000:
    # ae_features_train.append(encoder.predict(X_train[idx_train*1000:]))
    ae_features_train = np.concatenate((ae_features_train, encoder.predict(X_train[idx_train*1000:])), axis=0)

# Save the numpy array
np.save(os.path.join(save_filepath, "throughput_ae_features_novideo_sinr_ul.npy"), ae_features_train)
