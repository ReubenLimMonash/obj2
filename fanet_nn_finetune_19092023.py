'''
Date Modified: 19/07/2023
Desc: To train a NN classifier to predict FANET reliability and failure modes
Modified: To consolidate the different versions, training modes, models and dataset types into one script
'''

import pandas as pd
import numpy as np 
import math
import os
import pickle
import gc 
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
from sys import getsizeof

# Keras specific
import keras
from keras.models import Model, load_model
from keras.layers import Dense, Input, BatchNormalization, Activation, Dropout
from keras.utils import to_categorical 
from keras.callbacks import Callback
from keras import initializers, regularizers, optimizers, backend

def generate_reliability_train_test_dataset(dataset_details_df, test_split=0.2):
    # df_dtypes = {"Horizontal_Distance": np.float64, "Height": np.int16,	"U2G_Distance": np.int32, "UAV_Sending_Interval": np.float64, "Mean_SINR": np.float64, "Std_Dev_SINR": np.float64,
    #              "Modulation": 'string', "Num_Sent": np.int32, "Num_Reliable": np.int32, "Num_Delay_Excd": np.int32, "Num_Incr_Rcvd": np.int32, "Num_Q_Overflow": np.int32}
    # dataset_details = pd.read_csv(dataset_details_csv, 
    #                               usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Num_Sent", "Num_Reliable", "Num_Delay_Excd",
    #                                          "Num_Incr_Rcvd", "Num_Q_Overflow"],
    #                               dtype=df_dtypes)
    df_train_list = []
    df_test_list = []
    for row in tqdm(dataset_details_df.itertuples()):
        mean_sinr = row.Mean_SINR
        std_dev_sinr = row.Std_Dev_SINR
        uav_send_int = row.UAV_Sending_Interval
        modulation = row.Modulation
        mcs = row.MCS
        num_reliable = row.Num_Reliable
        num_delay_excd = row.Num_Delay_Excd
        num_incr_rcvd = row.Num_Incr_Rcvd
        num_q_overflow = row.Num_Q_Overflow

        if num_reliable > 1:
            reliable_packets = pd.DataFrame({"Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr, "UAV_Sending_Interval": uav_send_int, "MCS": mcs, "Packet_State": "Reliable"}, index=[0])
            reliable_packets = reliable_packets.loc[reliable_packets.index.repeat(num_reliable)]
            reliable_packets_train, reliable_packets_test = train_test_split(reliable_packets, test_size=test_split, random_state=40, shuffle=False)
        elif num_reliable == 1:
            reliable_packets_train = pd.DataFrame({"Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr, "UAV_Sending_Interval": uav_send_int, "MCS": mcs, "Packet_State": "Reliable"}, index=[0])
            reliable_packets_test = pd.DataFrame({})
        else:
            reliable_packets_train = pd.DataFrame({})
            reliable_packets_test = pd.DataFrame({})

        if num_delay_excd > 1:
            delay_excd_packets = pd.DataFrame({"Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr, "UAV_Sending_Interval": uav_send_int, "MCS": mcs, "Packet_State": "Delay_Exceeded"}, index=[0])
            delay_excd_packets = delay_excd_packets.loc[delay_excd_packets.index.repeat(num_delay_excd)]
            delay_excd_packets_train, delay_excd_packets_test = train_test_split(delay_excd_packets, test_size=test_split, random_state=40, shuffle=False)
        elif num_delay_excd == 1:
            delay_excd_packets_train = pd.DataFrame({"Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr, "UAV_Sending_Interval": uav_send_int, "MCS": mcs, "Packet_State": "Delay_Exceeded"}, index=[0])
            delay_excd_packets_test = pd.DataFrame({})
        else:
            delay_excd_packets_train = pd.DataFrame({})
            delay_excd_packets_test = pd.DataFrame({})

        if num_q_overflow > 1:
            q_overflow_packets = pd.DataFrame({"Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr, "UAV_Sending_Interval": uav_send_int, "MCS": mcs, "Packet_State": "QUEUE_OVERFLOW"}, index=[0])
            q_overflow_packets = q_overflow_packets.loc[q_overflow_packets.index.repeat(num_q_overflow)]
            q_overflow_packets_train, q_overflow_packets_test = train_test_split(q_overflow_packets, test_size=test_split, random_state=40, shuffle=False)
        elif num_q_overflow == 1:
            q_overflow_packets_train = pd.DataFrame({"Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr, "UAV_Sending_Interval": uav_send_int, "MCS": mcs, "Packet_State": "QUEUE_OVERFLOW"}, index=[0])
            q_overflow_packets_test = pd.DataFrame({})
        else:
            q_overflow_packets_train = pd.DataFrame({})
            q_overflow_packets_test = pd.DataFrame({})

        if num_incr_rcvd > 1:
            incr_rcvd_packets = pd.DataFrame({"Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr, "UAV_Sending_Interval": uav_send_int, "MCS": mcs, "Packet_State": "RETRY_LIMIT_REACHED"}, index=[0])
            incr_rcvd_packets = incr_rcvd_packets.loc[incr_rcvd_packets.index.repeat(num_incr_rcvd)]
            incr_rcvd_packets_train, incr_rcvd_packets_test = train_test_split(incr_rcvd_packets, test_size=test_split, random_state=40, shuffle=False)
        elif num_incr_rcvd == 1:
            incr_rcvd_packets_train = pd.DataFrame({"Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr, "UAV_Sending_Interval": uav_send_int, "MCS": mcs, "Packet_State": "RETRY_LIMIT_REACHED"}, index=[0])
            incr_rcvd_packets_test = pd.DataFrame({})
        else:
            incr_rcvd_packets_train = pd.DataFrame({})
            incr_rcvd_packets_test = pd.DataFrame({})
        df_train_list.append(pd.concat([reliable_packets_train, delay_excd_packets_train, q_overflow_packets_train, incr_rcvd_packets_train]))
        df_test_list.append(pd.concat([reliable_packets_test, delay_excd_packets_test, q_overflow_packets_test, incr_rcvd_packets_test]))

    df_train = pd.concat(df_train_list)
    df_test = pd.concat(df_test_list)
    return df_train, df_test

def normalize_data(df_in, columns=[], save_details_path=None):
    '''
    columns: The pandas data columns to normalize, given as a list of column names
    '''
    df = df_in.copy()
    # Define the ranges of parametrers
    max_mean_sinr = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
    max_std_dev_sinr = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
    min_mean_sinr = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
    min_std_dev_sinr = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)
    max_height = 300
    min_height = 60
    max_h_dist = 1200
    min_h_dist = 0
    max_mcs = 7
    min_mcs = 0

    # Normalize data (Min Max Normalization between [-1,1])
    if "Height" in columns:
        df["Height"] = df["Height"].apply(lambda x: 2*(x-min_height)/(max_height-min_height) - 1)
    if "U2G_H_Dist" in columns:
        df["U2G_H_Dist"] = df["U2G_H_Dist"].apply(lambda x: 2*(x-min_h_dist)/(max_h_dist-min_h_dist) - 1)
    if "Mean_SINR" in columns:
        df["Mean_SINR"] = df["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
    if "Std_Dev_SINR" in columns:
        df["Std_Dev_SINR"] = df["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1) # Convert to dB space
    if "UAV_Sending_Interval" in columns:
        df["UAV_Sending_Interval"] = df["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 66.7: 0.5, 100:1, 1000:2})
    if "Packet_State" in columns:
        df['Packet_State'] = df['Packet_State'].replace({"Reliable":0, "QUEUE_OVERFLOW":1, "RETRY_LIMIT_REACHED":2, "Delay_Exceeded":3})
    if "Modulation" in columns:
        df['Modulation'] = df['Modulation'].replace({"BPSK":1, "QPSK":0.3333, 16:-0.3333, "QAM-16":-0.3333, "QAM16":-0.3333, 64:-1, "QAM-64":-1, "QAM64":-1})
    if "MCS" in columns:
        df["MCS"] = df["MCS"].apply(lambda x: 2*(x-min_mcs)/(max_mcs-min_mcs) - 1)

    # Record details of inputs and output for model
    if save_details_path is not None:
        f = open(os.path.join(save_details_path,"model_details.txt"), "w")
        f.write("Max Height (m): {}\n".format(max_height))
        f.write("Min Height (m): {}\n".format(min_height))
        f.write("Max H_Dist (m): {}\n".format(max_h_dist))
        f.write("Min H_Dist (m): {}\n".format(min_h_dist))
        f.write("Max Mean SINR (dB): {}\n".format(max_mean_sinr))
        f.write("Min Mean SINR (dB): {}\n".format(min_mean_sinr))
        f.write("Max Std Dev SINR (dB): {}\n".format(max_std_dev_sinr))
        f.write("Min Std Dev SINR (dB): {}\n".format(min_std_dev_sinr))
        f.write("[BPSK: 1, QPSK: 0.3333, QAM16: -0.3333, QAM64: -1]\n")
        f.write("UAV Sending Interval: [10:-1, 20:-0.5, 40:0, 66.7: 0.5, 100:1, 1000:2]\n")
        f.write("Output: ['Reliable':0, 'QUEUE_OVERFLOW':1, 'RETRY_LIMIT_REACHED':2, 'Delay_Exceeded':3]\n")
        f.close()

    return df

def get_mcs_index(df_in):
    '''
    Gets the MCS index based on modulation and bitrate column of the df_in
    '''
    df = df_in.copy()
    df["MCS"] = ''
    df.loc[(df["Modulation"] == "BPSK") & (df["Bitrate"] == 6.5), "MCS"] = 0 # MCS Index 0
    df.loc[(df["Modulation"] == "QPSK") & (df["Bitrate"] == 13), "MCS"] = 1 # MCS Index 0
    df.loc[(df["Modulation"] == "QPSK") & (df["Bitrate"] == 19.5), "MCS"] = 2 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM16") & (df["Bitrate"] == 26), "MCS"] = 3 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM16") & (df["Bitrate"] == 39), "MCS"] = 4 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 52), "MCS"] = 5 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 58.5), "MCS"] = 6 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 65), "MCS"] = 7 # MCS Index 0

    return df

def build_nn_model_v4():
    # For multiple output model
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
    return model

def build_nn_model_v4_wobatchnorm():
    # For multiple output model
    # Version 4: Having only a single output layer for packet state
    inputs = Input(shape=(4,))
    base = Dense(100, activation='relu')(inputs)
    base = Dense(50, activation='relu')(base)
    base = Dense(25, activation='relu')(base)
    base = Dense(10, activation='relu')(base)
    packet_state_out = Dense(4, activation='softmax', name='packet_state')(base)
    model = Model(inputs=inputs, outputs = packet_state_out)
    return model

def build_nn_model_v5(l1_reg=0.001):
    # For multiple output model
    # Version 5: Added regularizations 
    init_weight = initializers.GlorotUniform(seed=0)
    regu_weight = regularizers.l1(l1_reg)

    inputs = Input(shape=(4,))
    base = Dense(100, kernel_initializer=init_weight, kernel_regularizer=regu_weight)(inputs)
    base = BatchNormalization()(base)
    base = Activation('relu')(base)
    base = Dropout(0.2)(base)
    base = Dense(50, kernel_initializer=init_weight, kernel_regularizer=regu_weight)(base)
    base = BatchNormalization()(base)
    base = Activation('relu')(base)
    base = Dropout(0.2)(base)
    base = Dense(25, kernel_initializer=init_weight, kernel_regularizer=regu_weight)(base)
    base = BatchNormalization()(base)
    base = Activation('relu')(base)
    base = Dropout(0.2)(base)
    base = Dense(10, kernel_initializer=init_weight, kernel_regularizer=regu_weight)(base)
    base = BatchNormalization()(base)
    base = Activation('relu')(base)
    base = Dropout(0.2)(base)
    packet_state_out = Dense(4, activation='softmax', name='packet_state')(base)
    model = Model(inputs=inputs, outputs = packet_state_out)
    return model

# Custom callback to clear memory to reduce RAM usage after each epoch
class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        keras.backend.clear_session()

if __name__ == "__main__":
    # Training params
    EPOCHS = 2 # 5
    BATCHSIZE = 64
    WORKER = 20 # No. of CPU for generator workers
    LR = 0.001 # Adam Learning Rate
    L1_REG = 0.001 # L1 Norm Weight Regularization
    # CHECKPOINT_FILEPATH = '/home/research-student/omnetpp_sim_results/PCAP_Test/DJISpark_ConstantSI/dl_nn_ckpt3'
    # DATASET_PATH = "/home/research-student/omnetpp_sim_results/PCAP_Test/DJISpark_ConstantSI/DJISpark_ConstantSI_Downlink_Reliability.csv"
    # CHECKPOINT_FILEPATH = '/home/rlim0005/nn_checkpoints/djispark_nnv5_ul'
    # DATASET_PATH = "/home/rlim0005/FANET_Dataset/Dataset_NP10000_DJISpark/DJI_Spark_Uplink_Reliability.csv"
    MODEL_FILEPATH = '/home/rlim0005/nn_checkpoints/djispark_nnv4_wobn_dl/model.010-0.2039.h5'
    CHECKPOINT_FILEPATH = '/home/rlim0005/nn_checkpoints/parrotar2_nnv4_wobn_finetune_dl'
    DATASET_PATH = "/home/rlim0005/FANET_Dataset/Dataset_NP10000_ParrotAR2/ParrotAR2_Downlink_Reliability.csv"

    # Create checkpoint filepath directory if not created
    if not os.path.isdir(CHECKPOINT_FILEPATH):
        os.mkdir(CHECKPOINT_FILEPATH)
        
    # Load dataset =================================================
    df_dtypes = {"Horizontal_Distance": np.float64, "Height": np.int16,	"U2G_Distance": np.int32, "UAV_Sending_Interval": np.float64, "Mean_SINR": np.float64, "Std_Dev_SINR": np.float64,
                 "Num_Sent": np.int32, "Num_Reliable": np.int32, "Num_Delay_Excd": np.int32, "Num_Incr_Rcvd": np.int32, "Num_Q_Overflow": np.int32, "Modulation": str, "Bitrate": np.float64}
    dataset_details_df = pd.read_csv(DATASET_PATH, 
                                usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Bitrate", "Num_Sent", "Num_Reliable", "Num_Delay_Excd",
                                            "Num_Incr_Rcvd", "Num_Q_Overflow"],
                                dtype=df_dtypes)
    dataset_details_df = get_mcs_index(dataset_details_df)
    dataset_details_df = normalize_data(dataset_details_df, columns=["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"], save_details_path=CHECKPOINT_FILEPATH) 
    df_train, df_test = generate_reliability_train_test_dataset(dataset_details_df, test_split=0.2)
    # Normalize output data
    df_train = normalize_data(df_train, columns=["Packet_State"], save_details_path=None)   
    df_test = normalize_data(df_test, columns=["Packet_State"], save_details_path=None)                         
    # Get only inputs and output(s) for model
    X_train = df_train[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"]].values
    X_test = df_test[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"]].values
    packet_state_train = df_train['Packet_State'].values
    packet_state_test = df_test['Packet_State'].values
    # Convert output data to categorical type
    packet_state_train = to_categorical(packet_state_train)
    packet_state_test = to_categorical(packet_state_test)
    # Clean up to save memory (so that oom don't make me cry)
    del df_train, df_test
    gc.collect()

    print("Train Data Size: ", getsizeof(X_train) + getsizeof(packet_state_train))
    print("Test Data Size: ", getsizeof(X_test) + getsizeof(packet_state_test))

    # Build model
    # model = build_nn_model_v5(l1_reg=L1_REG)
    # model = build_nn_model_v4()
    # model = build_nn_model_v4_wobatchnorm()

    # Load pre-trained model for finetuning
    model = load_model(MODEL_FILEPATH, compile=False)
    # model = keras.models.load_model(os.path.join(CHECKPOINT_FILEPATH, "model.004-0.2158.h5"), compile=False)

    # Compile the model
    optmz = optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optmz, 
                loss={'packet_state': 'categorical_crossentropy'},
                metrics={'packet_state': 'accuracy'})

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_FILEPATH,"model.{epoch:03d}-{val_loss:.4f}.h5"),
        save_weights_only=False,
        monitor='val_loss',
        mode='auto',
        save_freq='epoch')

    date = datetime.now()
    date_str = date.strftime("%d%m%Y")
    history = model.fit(X_train, packet_state_train, epochs=EPOCHS, batch_size=BATCHSIZE, callbacks=[model_checkpoint_callback, ClearMemory()], validation_data=(X_test, packet_state_test))
    
    with open(os.path.join(CHECKPOINT_FILEPATH, 'trainHistoryDict_{}'.format(date_str)), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # Save final model
    model.save(os.path.join(CHECKPOINT_FILEPATH,"final_model.h5"))