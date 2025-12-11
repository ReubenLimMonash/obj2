'''
Date Modified: 09/07/2023
Desc: To train a NN classifier to predict FANET reliability and failure modes
Modified: For v4 NN and for slurm
Modified: To use height and h_dist inputs instead of mean and std dev of SINR
Modified: To load the train dataset from "<modulation>_processed_train_uplink.csv" and the test dataset from "<modulation>_processed_holdout_uplink.csv"
'''

import pandas as pd
import numpy as np 
import sklearn
import os
import math
import pickle
from datetime import datetime
import gc

# Keras specific
import keras
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization
from keras.callbacks import Callback

# Training params
EPOCHS = 5
checkpoint_filepath = '/home/rlim0005/nn_checkpoints/throughput_predict_nn_v4_multimodulation_video_sinr_ul'
# checkpoint_filepath = '/home/clow0003/Reuben_ws/nn_checkpoints/throughput_predict_nn_v4_multimodulation_novideo_nosinr_dl'
link_type = "uplink" # "uplink" / "downlink" / "video"
video_novideo = "Video" # "NoVideo" / "Video" / "NoVideo_Part2"

df_dtypes = {"TxTime": np.float32, "U2G_Distance": np.float32, "Height": np.int16,	"Num_Members": np.int16, "UAV_Sending_Interval": np.int16, "Bytes": np.int16, 
            "U2G_SINR": np.float32, "U2G_BER": np.float32, "Delay": np.float32, "Throughput": np.float32, "Queueing_Time": np.float32, "Packet_State": 'category', 
            "Retry_Count": np.int8, "Incorrectly_Received": np.int8, "Queue_Overflow": np.int8, "Packet_Name": 'string', "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
            "UAV_Sending_Interval": np.int16}

# Load training dataset ==========================================================================================================================
# Part 1 ----------------------------------------------------------------------------------------------------------------------------------
df_bpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/BPSK_processed_train_{}.csv".format(video_novideo, link_type),
# df_bpsk = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/BPSK_processed_train_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
                    dtype=df_dtypes)
df_bpsk["Modulation"] = 1

df_qpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QPSK_processed_train_{}.csv".format(video_novideo, link_type),
# df_qpsk = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QPSK_processed_train_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
                    dtype=df_dtypes)
df_qpsk["Modulation"] = 0.3333

df_qam16 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QAM16_processed_train_{}.csv".format(video_novideo, link_type),
# df_qam16 = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QAM16_processed_train_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
                    dtype=df_dtypes)
df_qam16["Modulation"] = -0.3333

df_qam64 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QAM64_processed_train_{}.csv".format(video_novideo, link_type),
# df_qam64 = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QAM64_processed_train_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
                    dtype=df_dtypes)
df_qam64["Modulation"] = -1

df_train_1 = pd.concat([df_bpsk, df_qpsk, df_qam16, df_qam64], ignore_index=True)

# df_train_1.sort_values(by = "U2G_H_Dist")

    # Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
# df_train_1 = df_train_1.loc[df_train_1["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]

# If Part 2 COMMENTED OUT, UNCOMMENT BELOW
df_train = df_train_1

# # Part 2 ----------------------------------------------------------------------------------------------------------------------------------
# # df_bpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}_Part2/BPSK_processed_train_{}.csv".format(video_novideo, link_type),
# df_bpsk = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}_Part2/BPSK_processed_train_{}.csv".format(video_novideo, link_type),
#                     usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
#                     dtype=df_dtypes)
# df_bpsk["Modulation"] = 1

# # df_qpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}_Part2/QPSK_processed_train_{}.csv".format(video_novideo, link_type),
# df_qpsk = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}_Part2/QPSK_processed_train_{}.csv".format(video_novideo, link_type),
#                     usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
#                     dtype=df_dtypes)
# df_qpsk["Modulation"] = 0.3333

# # df_qam16 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}_Part2/QAM16_processed_train_{}.csv".format(video_novideo, link_type),
# df_qam16 = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}_Part2/QAM16_processed_train_{}.csv".format(video_novideo, link_type),
#                     usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
#                     dtype=df_dtypes)
# df_qam16["Modulation"] = -0.3333

# # df_qam64 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}_Part2/QAM64_processed_train_{}.csv".format(video_novideo, link_type),
# df_qam64 = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}_Part2/QAM64_processed_train_{}.csv".format(video_novideo, link_type),
#                     usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
#                     dtype=df_dtypes)
# df_qam64["Modulation"] = -1

# df_train_2 = pd.concat([df_bpsk, df_qpsk, df_qam16, df_qam64], ignore_index=True)

# df_train_2.sort_values(by = "Mean_SINR")

#     # Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
# # df_train_2 = df_train_2.loc[df_train_2["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]

# # Combine Part 1 & 2
# df_train = pd.concat([df_train_1, df_train_2])
# Load training dataset ==========================================================================================================================

# Load test dataset ==========================================================================================================================
# Part 1 ----------------------------------------------------------------------------------------------------------------------------------
df_bpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/BPSK_processed_holdout_{}.csv".format(video_novideo, link_type),
# df_bpsk = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/BPSK_processed_holdout_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
                    dtype=df_dtypes)
df_bpsk["Modulation"] = 1

df_qpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QPSK_processed_holdout_{}.csv".format(video_novideo, link_type),
# df_qpsk = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QPSK_processed_holdout_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
                    dtype=df_dtypes)
df_qpsk["Modulation"] = 0.3333

df_qam16 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QAM16_processed_holdout_{}.csv".format(video_novideo, link_type),
# df_qam16 = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QAM16_processed_holdout_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
                    dtype=df_dtypes)
df_qam16["Modulation"] = -0.3333

df_qam64 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QAM64_processed_holdout_{}.csv".format(video_novideo, link_type),
# df_qam64 = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QAM64_processed_holdout_{}.csv".format(video_novideo, link_type),
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
                    dtype=df_dtypes)
df_qam64["Modulation"] = -1

df_holdout_1 = pd.concat([df_bpsk, df_qpsk, df_qam16, df_qam64], ignore_index=True)

# df_holdout_1.sort_values(by = "Mean_SINR")

    # Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
# df_holdout_1 = df_holdout_1.loc[df_holdout_1["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]

# If Part 2 COMMENTED OUT, UNCOMMENT BELOW
df_holdout = df_holdout_1

# # Part 2 ----------------------------------------------------------------------------------------------------------------------------------
# df_bpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}_Part2/BPSK_processed_holdout_{}.csv".format(video_novideo, link_type),
#                     usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
#                     dtype=df_dtypes)
# df_bpsk["Modulation"] = 1

# df_qpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}_Part2/QPSK_processed_holdout_{}.csv".format(video_novideo, link_type),
#                     usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
#                     dtype=df_dtypes)
# df_qpsk["Modulation"] = 0.3333

# df_qam16 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}_Part2/QAM16_processed_holdout_{}.csv".format(video_novideo, link_type),
#                     usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
#                     dtype=df_dtypes)
# df_qam16["Modulation"] = -0.3333

# df_qam64 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}_Part2/QAM64_processed_holdout_{}.csv".format(video_novideo, link_type),
#                     usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Throughput"],
#                     dtype=df_dtypes)
# df_qam64["Modulation"] = -1

# df_holdout_2 = pd.concat([df_bpsk, df_qpsk, df_qam16, df_qam64], ignore_index=True)

# df_holdout_2.sort_values(by = "Mean_SINR")

#     # Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
# # df_holdout_2 = df_holdout_2.loc[df_holdout_2["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]
# df_holdout = pd.concat([df_holdout_1, df_holdout_2])
# Load test dataset ==========================================================================================================================

# Replace zero throughputs with 1
df_train.replace(to_replace=0, value=1, inplace=True)
df_holdout.replace(to_replace=0, value=1, inplace=True)

# Define ranges of input parameters
max_mean_sinr = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
max_std_dev_sinr = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
min_mean_sinr = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
min_std_dev_sinr = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)
# The max throughput is estimated as: 500,000 bytes/sec (uplink); 20,000 bytes/sec (downlink); 250,000 bytes/sec (video) [FOR 8 UAVs CASE ONLY]
if link_type == "uplink":
    # max_throughput = 10*math.log10(500000) 
    # min_throughput = 10*math.log10(1) # We make the min throughput to be 1 so that log(min_throughput) does not go to -inf
    max_throughput = 500000 
    min_throughput = 0
elif link_type == "downlink":
    # max_throughput = 10*math.log10(20000) 
    # min_throughput = 10*math.log10(1) # We make the min throughput to be 1 so that log(min_throughput) does not go to -inf
    max_throughput = 20000
    min_throughput = 0
elif link_type == "video":
    # max_throughput = 10*math.log10(250000) 
    # min_throughput = 10*math.log10(1) # We make the min throughput to be 1 so that log(min_throughput) does not go to -inf
    max_throughput = 250000
    min_throughput = 0

# Normalize data (Min Max Normalization between [-1,1])
df_train["Mean_SINR"] = df_train["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
df_train["Std_Dev_SINR"] = df_train["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1) # Convert to dB space
df_train["UAV_Sending_Interval"] = df_train["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1})
# df_train["Throughput"] = df_train["Throughput"].apply(lambda x: (10*math.log10(x)-min_throughput)/(max_throughput-min_throughput)) # Normalize throughput btw 0 and 1 so that ReLu can be used
df_train["Throughput"] = df_train["Throughput"].apply(lambda x: (x-min_throughput)/(max_throughput-min_throughput)) # Normalize throughput btw 0 and 1 so that ReLu can be used
df_holdout["Mean_SINR"] = df_holdout["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
df_holdout["Std_Dev_SINR"] = df_holdout["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1) # Convert to dB space
df_holdout["UAV_Sending_Interval"] = df_holdout["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1})
# df_holdout["Throughput"] = df_holdout["Throughput"].apply(lambda x: (10*math.log10(x)-min_throughput)/(max_throughput-min_throughput)) # Normalize throughput btw 0 and 1 so that ReLu can be used
df_holdout["Throughput"] = df_holdout["Throughput"].apply(lambda x: (x-min_throughput)/(max_throughput-min_throughput)) # Normalize throughput btw 0 and 1 so that ReLu can be used

# Get inputs and outputs for train and test
X_train = df_train[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation"]].values
X_test = df_holdout[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation"]].values
throughput_train = df_train['Throughput'].values
throughput_test = df_holdout['Throughput'].values

# Clean up to save memory (so that oom don't make me cry)
del df_train, df_holdout, df_bpsk, df_qpsk, df_qam16, df_qam64
gc.collect()

# For multiple output model
# Version 2: Add an additional hidden layer for each output layer (not shared among outputs)
# Version 4: Having only a single output layer for packet state
inputs = Input(shape=(4,))
base = Dense(100, activation='relu', name='dense_1')(inputs)
base = BatchNormalization(name='batch_norm_1')(base)
base = Dense(50, activation='relu', name='dense_2')(base)
base = BatchNormalization(name='batch_norm_2')(base)
base = Dense(25, activation='relu', name='dense_3')(base)
base = BatchNormalization(name='batch_norm_3')(base)
base = Dense(10, activation='relu', name='dense_4')(base)
base = BatchNormalization(name='batch_norm_4')(base)
throughput_out = Dense(1, activation='relu', name='throughput')(base) # Using ReLu as output can only be between 0 and 1
model = Model(inputs=inputs, outputs = throughput_out)

# Load pre-trained model for finetuning
# model = keras.models.load_model(os.path.join(checkpoint_filepath, "model.004-0.1376.h5"), compile=False)

# Compile the model
model.compile(optimizer='adam', 
              loss={'throughput': 'mse'},
              metrics={'throughput': 'accuracy'})

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

# Record details of inputs and output for model
f = open(os.path.join(checkpoint_filepath,"model_details.txt"), "w")
f.write("Mean and Std Dev SINR are in dB\n")
f.write("Max Mean SINR (dB): {}\n".format(max_mean_sinr))
f.write("Min Mean SINR (dB): {}\n".format(min_mean_sinr))
f.write("Max Std Dev SINR (dB): {}\n".format(max_std_dev_sinr))
f.write("Min Std Dev SINR (dB): {}\n".format(min_std_dev_sinr))
f.write("[BPSK: 1, QPSK: 0.3333, QAM16: -0.3333, QAM64: -1]\n")
f.write("UAV Sending Interval: [10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1]\n")
f.write("Max Throughput: {}\n".format(max_throughput))
f.write("Min Throughput: {}\n".format(min_throughput))
f.close()

history = model.fit(X_train, throughput_train, epochs=EPOCHS, callbacks=[model_checkpoint_callback, ClearMemory()], validation_data=(X_test, throughput_test))
with open(os.path.join(checkpoint_filepath, 'trainHistoryDict_{}'.format(date_str)), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Save final model
model.save(os.path.join(checkpoint_filepath,"final_model.h5"))
