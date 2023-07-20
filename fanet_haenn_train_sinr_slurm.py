'''
Date: 27/06/2023
Desc: To train an NN classifier for reliability prediction taking input from latent space of auto-encoder, for a hierarchical model
Modified: To load the train dataset from "<modulation>_processed_train_uplink.csv" and the test dataset from "<modulation>_processed_holdout_uplink.csv"
Modified: To combine the encoder part of the AE to the NN classifier, instead of extracting features and training the NN classifier in two separate steps
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
from keras.callbacks import Callback

# Training params
EPOCHS = 5
autoencoder_model_path = '/home/rlim0005/nn_checkpoints/ae_multimodulation_novideo_sinr_ul/final_model.h5'
checkpoint_filepath = '/home/rlim0005/nn_checkpoints/haenn_multimodulation_novideo_sinr_ul'
link_type = "uplink" # "uplink" / "downlink"
video_novideo = "NoVideo" # "NoVideo" / "Video" / "NoVideo_Part2"

df_dtypes = {"TxTime": np.float32, "U2G_Distance": np.float32, "Height": np.int16,	"Num_Members": np.int16, "UAV_Sending_Interval": np.int16, "Bytes": np.int16, 
            "U2G_SINR": np.float32, "U2G_BER": np.float32, "Delay": np.float32, "Throughput": np.float32, "Queueing_Time": np.float32, "Packet_State": 'category', 
            "Retry_Count": np.int8, "Incorrectly_Received": np.int8, "Queue_Overflow": np.int8, "Packet_Name": 'string', "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
            "UAV_Sending_Interval": np.int16}

# Load training dataset ==========================================================================================================================
df_bpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/BPSK_processed_train_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
df_bpsk["Modulation"] = 1

df_qpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QPSK_processed_train_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
df_qpsk["Modulation"] = 0.3333

df_qam16 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QAM16_processed_train_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
df_qam16["Modulation"] = -0.3333

df_qam64 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QAM64_processed_train_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
df_qam64["Modulation"] = -1

df_train = pd.concat([df_bpsk, df_qpsk, df_qam16, df_qam64], ignore_index=True)

df_train.sort_values(by = "U2G_H_Dist")

    # Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
df_train = df_train.loc[df_train["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]
# Load training dataset ==========================================================================================================================

# Load test dataset ==========================================================================================================================
df_bpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/BPSK_processed_holdout_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
df_bpsk["Modulation"] = 1

df_qpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QPSK_processed_holdout_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
df_qpsk["Modulation"] = 0.3333

df_qam16 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QAM16_processed_holdout_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
df_qam16["Modulation"] = -0.3333

df_qam64 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QAM64_processed_holdout_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
df_qam64["Modulation"] = -1

df_holdout = pd.concat([df_bpsk, df_qpsk, df_qam16, df_qam64], ignore_index=True)

df_holdout.sort_values(by = "U2G_H_Dist")

    # Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
df_holdout = df_holdout.loc[df_holdout["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]
# Load test dataset ==========================================================================================================================

# Define ranges of input parameters
max_mean_sinr = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
max_std_dev_sinr = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
min_mean_sinr = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
min_std_dev_sinr = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)

# Normalize data (Min Max Normalization between [-1,1])
df_train["Mean_SINR"] = df_train["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
df_train["Std_Dev_SINR"] = df_train["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1)
df_train["UAV_Sending_Interval"] = df_train["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1})
df_train['Packet_State'] = df_train['Packet_State'].replace({"Reliable":0, "QUEUE_OVERFLOW":1, "RETRY_LIMIT_REACHED":2, "Delay_Exceeded":3})
df_holdout["Mean_SINR"] = df_holdout["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
df_holdout["Std_Dev_SINR"] = df_holdout["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1)
df_holdout["UAV_Sending_Interval"] = df_holdout["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1})
df_holdout['Packet_State'] = df_holdout['Packet_State'].replace({"Reliable":0, "QUEUE_OVERFLOW":1, "RETRY_LIMIT_REACHED":2, "Delay_Exceeded":3})

# Get inputs and outputs for train and test
X_train = df_train[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation"]].values
X_test = df_holdout[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation"]].values
packet_state_train = df_train['Packet_State'].values
packet_state_test = df_holdout['Packet_State'].values

# Clean up to save memory (so that oom don't make me cry)
del df_train, df_holdout, df_bpsk, df_qpsk, df_qam16, df_qam64
gc.collect()

# Convert output data to categorical type
packet_state_train = to_categorical(packet_state_train)
packet_state_test = to_categorical(packet_state_test)

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

# Load the autoencoder feature extractor model
autoencoder = keras.models.load_model(autoencoder_model_path, compile=False)
autoencoder.compile(optimizer='adam', loss='mse', metrics='mse')
# Get the encoder part of the autoencoder
encoder_input = autoencoder.get_layer('encoder')
encoder_layer = autoencoder.get_layer('latent')
# Freeze the encoder part
encoder_input.trainable = False
encoder_layer.trainable = False
# Create hierarchical model
haenn = keras.Sequential(
    [
        Input(shape=(4,)),
        encoder_input,
        encoder_layer,
        Dense(25, activation='relu', name='classifier_1'),
        BatchNormalization(name='batch_norm_1'), 
        Dense(10, activation='relu', name='classifier_2'),
        BatchNormalization(name='batch_norm_2'), 
        Dense(4, activation='softmax', name='packet_state')
    ]
)

# Compile the model
haenn.compile(optimizer='adam', 
              loss={'packet_state': 'categorical_crossentropy'},
              metrics={'packet_state': 'accuracy'})

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_filepath,"model.{epoch:03d}-{val_loss:.4f}.h5"),
    save_weights_only=False,
    monitor='val_loss',
    mode='auto',
    save_freq='epoch')

# Custom callback to clear memory to reduce RAM usage after each epoch
class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        keras.backend.clear_session()

date = datetime.now()
date_str = date.strftime("%d%m%Y")

history = haenn.fit(X_train, packet_state_train, epochs=EPOCHS, callbacks=[model_checkpoint_callback, ClearMemory()], validation_data=(X_test, packet_state_test))
with open(os.path.join(checkpoint_filepath, 'trainHistoryDict_{}'.format(date_str)), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Save final model
haenn.save(os.path.join(checkpoint_filepath,"final_model.h5"))
