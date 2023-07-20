'''
Date: 26/06/2023
Desc: To train an NN classifier for throughput monitoring taking input from latent space of auto-encoder, for a hierarchical model
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
from keras.callbacks import Callback

# Training params
EPOCHS = 5
# checkpoint_filepath = '/home/rlim0005/nn_checkpoints/throughput_ae_multimodulation_novideo_sinr_ul'
# autoencoder_model_path = '/home/rlim0005/nn_checkpoints/ae_multimodulation_novideo_sinr_ul/final_model.h5'
autoencoder_model_path = '/home/research-student/omnet-fanet/nn_checkpoints/ae_multimodulation_novideo_sinr_ul/final_model.h5'
checkpoint_filepath = '/home/research-student/omnet-fanet/nn_checkpoints/throughput_ae_multimodulation_novideo_sinr_ul'
link_type = "uplink" # "uplink" / "downlink"
video_novideo = "NoVideo" # "NoVideo" / "Video" / "NoVideo_Part2"

df_dtypes = {"TxTime": np.float32, "U2G_Distance": np.float32, "Height": np.int16,	"Num_Members": np.int16, "UAV_Sending_Interval": np.int16, "Bytes": np.int16, 
            "U2G_SINR": np.float32, "U2G_BER": np.float32, "Delay": np.float32, "Throughput": np.float32, "Queueing_Time": np.float32, "Packet_State": 'category', 
            "Retry_Count": np.int8, "Incorrectly_Received": np.int8, "Queue_Overflow": np.int8, "Packet_Name": 'string', "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
            "UAV_Sending_Interval": np.int16}

# Load training dataset ==========================================================================================================================
# df_bpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/BPSK_processed_throughput_uplink.csv",
df_bpsk = pd.read_csv("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_{}/BPSK_processed_throughput_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
                    dtype=df_dtypes)
df_bpsk["Modulation"] = 1

# df_qpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QPSK_processed_throughput_uplink.csv",
df_qpsk = pd.read_csv("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_{}/QPSK_processed_throughput_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
                    dtype=df_dtypes)
df_qpsk["Modulation"] = 0.3333

# df_qam16 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM16_processed_throughput_uplink.csv",
df_qam16 = pd.read_csv("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_{}/QAM16_processed_throughput_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
                    dtype=df_dtypes)
df_qam16["Modulation"] = -0.3333

# df_qam64 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM64_processed_throughput_uplink.csv",
df_qam64 = pd.read_csv("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_{}/QAM64_processed_throughput_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
                    dtype=df_dtypes)
df_qam64["Modulation"] = -1

df_train = pd.concat([df_bpsk, df_qpsk, df_qam16, df_qam64], ignore_index=True)

# df_train.sort_values(by = "Mean_SINR")
# Load training dataset ==========================================================================================================================

# Define ranges of input parameters
max_mean_sinr = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
max_std_dev_sinr = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
min_mean_sinr = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
min_std_dev_sinr = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)
max_throughput = 500000 # The max throughput is estimated as: 500,000 bytes/sec (uplink); 20,000 bytes/sec (downlink); 250,000 bytes/sec (video)
min_throughput = 0

# Normalize data (Min Max Normalization between [-1,1])
df_train["Mean_SINR"] = df_train["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
df_train["Std_Dev_SINR"] = df_train["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1)
df_train["UAV_Sending_Interval"] = df_train["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1})
df_train["Throughput"] = df_train["Throughput"].apply(lambda x: 2*(x-min_throughput)/(max_throughput-min_throughput) - 1)

# Get inputs for training
X_train = df_train[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Throughput"]].values

# Get the outputs for training (1 for normal data)
normal_out = np.ones((len(X_train), 1))

# Clean up to save memory (so that oom don't make me cry)
del df_train, df_bpsk, df_qpsk, df_qam16, df_qam64
gc.collect()

# Record details of inputs and output for model
f = open(os.path.join(checkpoint_filepath,"model_details.txt"), "w")
f.write("Mean and Std Dev SINR are in dB\n")
f.write("Max Mean SINR (dB): {}\n".format(max_mean_sinr))
f.write("Min Mean SINR (dB): {}\n".format(min_mean_sinr))
f.write("Max Std Dev SINR (dB): {}\n".format(max_std_dev_sinr))
f.write("Min Std Dev SINR (dB): {}\n".format(min_std_dev_sinr))
f.write("Max Throughput (bytes/sec): {}\n".format(max_throughput))
f.write("Min Throughput (bytes/sec): {}\n".format(min_throughput))
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
        Input(shape=(5,)),
        encoder_input,
        encoder_layer,
        Dense(25, activation='relu', name='classifier_1'),
        BatchNormalization(name='batch_norm_1'), 
        Dense(10, activation='relu', name='classifier_2'),
        BatchNormalization(name='batch_norm_2'), 
        Dense(1, activation='linear', name='indicator')
    ]
)

# Compile the model
haenn.compile(optimizer='adam', 
              loss={'indicator': 'mae'},
              metrics={'indicator': 'accuracy'})

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

history = haenn.fit(X_train, normal_out, epochs=EPOCHS, callbacks=[model_checkpoint_callback, ClearMemory()], validation_data=(X_train, normal_out))
with open(os.path.join(checkpoint_filepath, 'trainHistoryDict_{}'.format(date_str)), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Save final model
haenn.save(os.path.join(checkpoint_filepath,"final_model.h5"))
