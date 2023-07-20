'''
Date Modified: 07/06/2023
Desc: To train a NN classifier to predict FANET reliability and failure modes (FOR DOWNLINK)
Modified: For v4 NN and for slurm
Modified: To use height and h_dist inputs instead of mean and std dev of SINR
Modified: To load the train dataset from "<modulation>_processed_train_uplink.csv" and the test dataset from "<modulation>_processed_holdout_uplink.csv"
'''

import pandas as pd
import numpy as np 
import os
import pickle
import gc 

# Keras specific
import keras
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization
from keras.utils import to_categorical 

# Training params
EPOCHS = 5
# checkpoint_filepath = '/home/rlim0005/nn_checkpoints/nn_v4_multimodulation_video_nosinr_vid'
checkpoint_filepath = '/home/clow0003/Reuben_ws/nn_checkpoints/nn_v4_multimodulation_video_nosinr_vid'
delay_threshold = 1

df_dtypes = {"TxTime": np.float32, "U2G_Distance": np.float32, "Height": np.int16,	"Num_Members": np.int16, "UAV_Sending_Interval": np.int16, "Bytes": np.int16, 
            "U2G_SINR": np.float32, "U2G_BER": np.float32, "Delay": np.float32, "Throughput": np.float32, "Queueing_Time": np.float32, "Packet_State": 'category', 
            "Retry_Count": np.int8, "Incorrectly_Received": np.int8, "Queue_Overflow": np.int8, "Packet_Name": 'string', "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
            "UAV_Sending_Interval": np.int16}

# Load training dataset ==========================================================================================================================
# dl_df_bpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/BPSK_processed_train_video.csv",
dl_df_bpsk = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/BPSK_processed_train_video.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_bpsk["Modulation"] = 1

# dl_df_qpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QPSK_processed_train_video.csv",
dl_df_qpsk = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QPSK_processed_train_video.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qpsk["Modulation"] = 0.3333

# dl_df_qam16 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QAM16_processed_train_video.csv",
dl_df_qam16 = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QAM16_processed_train_video.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qam16["Modulation"] = -0.3333

# dl_df_qam64 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QAM64_processed_train_video.csv",
dl_df_qam64 = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QAM64_processed_train_video.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qam64["Modulation"] = -1

dl_df_train = pd.concat([dl_df_bpsk, dl_df_qpsk, dl_df_qam16, dl_df_qam64], ignore_index=True)

dl_df_train.sort_values(by = "U2G_H_Dist")

    # Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
dl_df_train = dl_df_train.loc[dl_df_train["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]
# Load training dataset ==========================================================================================================================

# Load test dataset ==========================================================================================================================
# dl_df_bpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/BPSK_processed_holdout_video.csv",
dl_df_bpsk = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/BPSK_processed_holdout_video.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_bpsk["Modulation"] = 1

# dl_df_qpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QPSK_processed_holdout_video.csv",
dl_df_qpsk = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QPSK_processed_holdout_video.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qpsk["Modulation"] = 0.3333

# dl_df_qam16 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QAM16_processed_holdout_video.csv",
dl_df_qam16 = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QAM16_processed_holdout_video.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qam16["Modulation"] = -0.3333

# dl_df_qam64 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QAM64_processed_holdout_video.csv",
dl_df_qam64 = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video/QAM64_processed_holdout_video.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qam64["Modulation"] = -1

dl_df_holdout = pd.concat([dl_df_bpsk, dl_df_qpsk, dl_df_qam16, dl_df_qam64], ignore_index=True)

dl_df_holdout.sort_values(by = "U2G_H_Dist")

    # Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
dl_df_holdout = dl_df_holdout.loc[dl_df_holdout["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]
# Load test dataset ==========================================================================================================================

# Define ranges of input parameters
max_height = 300
min_height = 60
max_h_dist = 1200
min_h_dist = 0

# Normalize data (Min Max Normalization between [-1,1])
dl_df_train["Height"] = dl_df_train["Height"].apply(lambda x: 2*(x-min_height)/(max_height-min_height) - 1)
dl_df_train["U2G_H_Dist"] = dl_df_train["U2G_H_Dist"].apply(lambda x: 2*(x-min_h_dist)/(max_h_dist-min_h_dist) - 1)
dl_df_train["UAV_Sending_Interval"] = dl_df_train["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1})
dl_df_train['Packet_State'] = dl_df_train['Packet_State'].replace({"Reliable":0, "QUEUE_OVERFLOW":1, "RETRY_LIMIT_REACHED":2, "Delay_Exceeded":3})
dl_df_holdout["Height"] = dl_df_holdout["Height"].apply(lambda x: 2*(x-min_height)/(max_height-min_height) - 1)
dl_df_holdout["U2G_H_Dist"] = dl_df_holdout["U2G_H_Dist"].apply(lambda x: 2*(x-min_h_dist)/(max_h_dist-min_h_dist) - 1)
dl_df_holdout["UAV_Sending_Interval"] = dl_df_holdout["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1})
dl_df_holdout['Packet_State'] = dl_df_holdout['Packet_State'].replace({"Reliable":0, "QUEUE_OVERFLOW":1, "RETRY_LIMIT_REACHED":2, "Delay_Exceeded":3})

# Split to train and test
X_train = dl_df_train[["U2G_H_Dist", "Height", "UAV_Sending_Interval", "Modulation"]].values
X_test = dl_df_holdout[["U2G_H_Dist", "Height", "UAV_Sending_Interval", "Modulation"]].values
packet_state_train = dl_df_train['Packet_State'].values
packet_state_test = dl_df_holdout['Packet_State'].values

# Clean up to save memory (so that oom don't make me cry)
del dl_df_train, dl_df_holdout, dl_df_bpsk, dl_df_qpsk, dl_df_qam16, dl_df_qam64
gc.collect()

# Convert output data to categorical type
packet_state_train = to_categorical(packet_state_train)
packet_state_test = to_categorical(packet_state_test)

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

# Load pre-trained model for finetuning
# model = tf.keras.models.load_model(os.path.join(checkpoint_filepath, "model.010-2.0158.h5"), compile=False)

# Compile the model
model.compile(optimizer='adam', 
              loss={'packet_state': 'categorical_crossentropy'},
              metrics={'packet_state': 'accuracy'})

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_filepath,"model.{epoch:03d}-{val_loss:.4f}.h5"),
    save_weights_only=False,
    monitor='val_loss',
    mode='auto',
    save_freq='epoch')

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

history = model.fit(X_train, packet_state_train, epochs=EPOCHS, callbacks=[model_checkpoint_callback], validation_data=(X_test, packet_state_test))
with open(os.path.join(checkpoint_filepath, 'trainHistoryDict_08052023'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Save final model
model.save(os.path.join(checkpoint_filepath,"final_model.h5"))
