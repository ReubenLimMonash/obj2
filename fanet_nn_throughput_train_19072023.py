'''
Date Modified: 19/07/2023
Desc: To train a NN classifier to predict FANET throughput
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

# Keras specific
import keras
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization
from keras.utils import to_categorical 
from keras.callbacks import Callback

def load_train_holdout_dataset(dataset_path, link_type, video_novideo):
    df_dtypes = {"TxTime": np.float32, "U2G_Distance": np.float32, "Height": np.int16,	"Num_Members": np.int16, "UAV_Sending_Interval": np.int16, "Bytes": np.int16, 
            "U2G_SINR": np.float32, "U2G_BER": np.float32, "Throughput": np.float32, "Throughput": np.float32, "Queueing_Time": np.float32, "Packet_State": 'category', 
            "Retry_Count": np.int8, "Incorrectly_Received": np.int8, "Queue_Overflow": np.int8, "Packet_Name": 'string', "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,}
    # Load training dataset ==========================================================================================================================
    # Part 1 ----------------------------------------------------------------------------------------------------------------------------------
    df_bpsk = pd.read_csv(os.path.join(dataset_path, "Dataset_NP10000_MultiModulation_Hovering_{}/BPSK_processed_train_{}.csv".format(video_novideo, link_type)),
                        usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Throughput", "U2G_H_Dist", "Height"],
                        dtype=df_dtypes)
    df_bpsk["Modulation"] = 1

    df_qpsk = pd.read_csv(os.path.join(dataset_path, "Dataset_NP10000_MultiModulation_Hovering_{}/QPSK_processed_train_{}.csv".format(video_novideo, link_type)),
                        usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Throughput", "U2G_H_Dist", "Height"],
                        dtype=df_dtypes)
    df_qpsk["Modulation"] = 0.3333

    df_qam16 = pd.read_csv(os.path.join(dataset_path, "Dataset_NP10000_MultiModulation_Hovering_{}/QAM16_processed_train_{}.csv".format(video_novideo, link_type)),
                        usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Throughput", "U2G_H_Dist", "Height"],
                        dtype=df_dtypes)
    df_qam16["Modulation"] = -0.3333

    df_qam64 = pd.read_csv(os.path.join(dataset_path, "Dataset_NP10000_MultiModulation_Hovering_{}/QAM64_processed_train_{}.csv".format(video_novideo, link_type)),
    # df_qam64 = pd.read_csv("/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_{}/QAM64_processed_train_{}.csv".format(video_novideo, link_type),
                        usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Throughput", "U2G_H_Dist", "Height"],
                        dtype=df_dtypes)
    df_qam64["Modulation"] = -1

    df_train_1 = pd.concat([df_bpsk, df_qpsk, df_qam16, df_qam64], ignore_index=True)

    # df_train_1.sort_values(by = "U2G_H_Dist")

    # If Part 2 COMMENTED OUT, UNCOMMENT BELOW
    df_train = df_train_1

    # # Part 2 ----------------------------------------------------------------------------------------------------------------------------------
    # df_bpsk = pd.read_csv(os.path.join(dataset_path, "Dataset_NP10000_MultiModulation_Hovering_{}_Part2/BPSK_processed_train_{}.csv".format(video_novideo, link_type)),
    #                     usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Throughput", "U2G_H_Dist", "Height"],
    #                     dtype=df_dtypes)
    # df_bpsk["Modulation"] = 1

    # df_qpsk = pd.read_csv(os.path.join(dataset_path, "Dataset_NP10000_MultiModulation_Hovering_{}_Part2/QPSK_processed_train_{}.csv".format(video_novideo, link_type)),
    #                     usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Throughput", "U2G_H_Dist", "Height"],
    #                     dtype=df_dtypes)
    # df_qpsk["Modulation"] = 0.3333

    # df_qam16 = pd.read_csv(os.path.join(dataset_path, "Dataset_NP10000_MultiModulation_Hovering_{}_Part2/QAM16_processed_train_{}.csv".format(video_novideo, link_type)),
    #                     usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Throughput", "U2G_H_Dist", "Height"],
    #                     dtype=df_dtypes)
    # df_qam16["Modulation"] = -0.3333

    # # df_qam64 = pd.read_csv(os.path.join(dataset_path, "Dataset_NP10000_MultiModulation_Hovering_{}_Part2/QAM64_processed_train_{}.csv".format(video_novideo, link_type)),
    #                     usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Throughput", "U2G_H_Dist", "Height"],
    #                     dtype=df_dtypes)
    # df_qam64["Modulation"] = -1

    # df_train_2 = pd.concat([df_bpsk, df_qpsk, df_qam16, df_qam64], ignore_index=True)

    # df_train_2.sort_values(by = "Mean_SINR")

    # # Combine Part 1 & 2
    # df_train = pd.concat([df_train_1, df_train_2])
    # Load training dataset ==========================================================================================================================

    # Load test dataset ==========================================================================================================================
    # Part 1 ----------------------------------------------------------------------------------------------------------------------------------
    df_bpsk = pd.read_csv(os.path.join(dataset_path, "Dataset_NP10000_MultiModulation_Hovering_{}/BPSK_processed_holdout_{}.csv".format(video_novideo, link_type)),
                        usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Throughput", "U2G_H_Dist", "Height"],
                        dtype=df_dtypes)
    df_bpsk["Modulation"] = 1

    df_qpsk = pd.read_csv(os.path.join(dataset_path, "Dataset_NP10000_MultiModulation_Hovering_{}/QPSK_processed_holdout_{}.csv".format(video_novideo, link_type)),
                        usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Throughput", "U2G_H_Dist", "Height"],
                        dtype=df_dtypes)
    df_qpsk["Modulation"] = 0.3333

    df_qam16 = pd.read_csv(os.path.join(dataset_path, "Dataset_NP10000_MultiModulation_Hovering_{}/QAM16_processed_holdout_{}.csv".format(video_novideo, link_type)),
                        usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Throughput", "U2G_H_Dist", "Height"],
                        dtype=df_dtypes)
    df_qam16["Modulation"] = -0.3333

    df_qam64 = pd.read_csv(os.path.join(dataset_path, "Dataset_NP10000_MultiModulation_Hovering_{}/QAM64_processed_holdout_{}.csv".format(video_novideo, link_type)),
                        usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Throughput", "U2G_H_Dist", "Height"],
                        dtype=df_dtypes)
    df_qam64["Modulation"] = -1

    df_holdout_1 = pd.concat([df_bpsk, df_qpsk, df_qam16, df_qam64], ignore_index=True)

    # df_holdout_1.sort_values(by = "Mean_SINR")

    # If Part 2 COMMENTED OUT, UNCOMMENT BELOW
    df_holdout = df_holdout_1

    # # Part 2 ----------------------------------------------------------------------------------------------------------------------------------
    # df_bpsk = pd.read_csv(os.path.join(dataset_path, "Dataset_NP10000_MultiModulation_Hovering_{}_Part2/BPSK_processed_holdout_{}.csv".format(video_novideo, link_type)),
    #                     usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Throughput", "U2G_H_Dist", "Height"],
    #                     dtype=df_dtypes)
    # df_bpsk["Modulation"] = 1

    # df_qpsk = pd.read_csv(os.path.join(dataset_path, "Dataset_NP10000_MultiModulation_Hovering_{}_Part2/QPSK_processed_holdout_{}.csv".format(video_novideo, link_type)),
    #                     usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Throughput", "U2G_H_Dist", "Height"],
    #                     dtype=df_dtypes)
    # df_qpsk["Modulation"] = 0.3333

    # df_qam16 = pd.read_csv(os.path.join(dataset_path, "Dataset_NP10000_MultiModulation_Hovering_{}_Part2/QAM16_processed_holdout_{}.csv".format(video_novideo, link_type)),
    #                     usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Throughput", "U2G_H_Dist", "Height"],
    #                     dtype=df_dtypes)
    # df_qam16["Modulation"] = -0.3333

    # df_qam64 = pd.read_csv(os.path.join(dataset_path, "Dataset_NP10000_MultiModulation_Hovering_{}_Part2/QAM64_processed_holdout_{}.csv".format(video_novideo, link_type)),
    #                     usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Throughput", "U2G_H_Dist", "Height"],
    #                     dtype=df_dtypes)
    # df_qam64["Modulation"] = -1

    # df_holdout_2 = pd.concat([df_bpsk, df_qpsk, df_qam16, df_qam64], ignore_index=True)

    # df_holdout_2.sort_values(by = "Mean_SINR")

    # df_holdout = pd.concat([df_holdout_1, df_holdout_2])
    # Load test dataset ==========================================================================================================================

    return df_train, df_holdout

def generate_troughput_train_holdout_dataset(dataset_details_csv, holdout_split=0.2):
    df_dtypes = {"Horizontal_Distance": np.float32, "Height": np.int16,	"U2G_Distance": np.int32, "UAV_Sending_Interval": np.int16, "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
                 "Modulation": 'string', "Num_Sent": np.int32, "Num_Reliable": np.int32, "Num_Delay_Excd": np.int32, "Num_Incr_Rcvd": np.int32, "Num_Q_Overflow": np.int32}
    dataset_details = pd.read_csv(dataset_details_csv, 
                                  usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Throughput", "Num_Count"],
                                  dtype=df_dtypes)
    
    # For each scenario, get the throughput data and split it to train and holdout 
    scenarios = dataset_details[['Mean_SINR','Std_Dev_SINR','Modulation','UAV_Sending_Interval']].drop_duplicates()
    df_train_list = []
    df_holdout_list = []
    for scenario in tqdm(scenarios.itertuples()):
        scenario_df = dataset_details.loc[(dataset_details["Modulation"] == scenario.Modulation) & (dataset_details["UAV_Sending_Interval"] == scenario.UAV_Sending_Interval) & 
                          (dataset_details["Mean_SINR"] == scenario.Mean_SINR) & (dataset_details["Std_Dev_SINR"] == scenario.Std_Dev_SINR)]
        df_list = []
        for row in scenario_df.itertuples():
            throughput_data = {"Mean_SINR": row.Mean_SINR, "Std_Dev_SINR": row.Std_Dev_SINR, "UAV_Sending_Interval": row.UAV_Sending_Interval, "Modulation": row.Modulation, "Throughput": row.Throughput}
            # If Num_Count is one, just append the row to df_list
            if row.Num_Count == 1:
                df_list.append(throughput_data)
            else:
                df_list = df_list + [throughput_data.copy() for i in range(row.Num_Count)]
        df = pd.DataFrame(df_list)
        train, holdout = train_test_split(df, test_size=holdout_split, random_state=40, shuffle=True)
        df_train_list.append(train)
        df_holdout_list.append(holdout)
    df_train = pd.concat(df_train_list)
    df_holdout = pd.concat(df_holdout_list)
    return df_train, df_holdout


def filter_n_sort(df):
    # Filter out rows where mean / std dev of sinr is NaN
    df = df[df['Mean_SINR'].notna()]
    df = df[df['Std_Dev_SINR'].notna()]

    df.sort_values(by = "U2G_H_Dist")

    # Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
    df = df.loc[df["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]
    return df

def normalize_data(df, columns=[], link_type='uplink', save_details_path=None):
    '''
    columns: The pandas data columns to normalize, given as a list of column names
    '''
    # Define the ranges of parametrers
    max_mean_sinr = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
    max_std_dev_sinr = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
    min_mean_sinr = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
    min_std_dev_sinr = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)
    max_height = 300
    min_height = 60
    max_h_dist = 1200
    min_h_dist = 0
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
    if "Height" in columns:
        df["Height"] = df["Height"].apply(lambda x: 2*(x-min_height)/(max_height-min_height) - 1)
    if "U2G_H_Dist" in columns:
        df["U2G_H_Dist"] = df["U2G_H_Dist"].apply(lambda x: 2*(x-min_h_dist)/(max_h_dist-min_h_dist) - 1)
    if "Mean_SINR" in columns:
        df["Mean_SINR"] = df["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
    if "Std_Dev_SINR" in columns:
        df["Std_Dev_SINR"] = df["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1) # Convert to dB space
    if "UAV_Sending_Interval" in columns:
        df["UAV_Sending_Interval"] = df["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1})
    if "Packet_State" in columns:
        df['Packet_State'] = df['Packet_State'].replace({"Reliable":0, "QUEUE_OVERFLOW":1, "RETRY_LIMIT_REACHED":2, "Delay_Exceeded":3})
    if "Modulation" in columns:
        df['Modulation'] = df['Modulation'].replace({"BPSK":1, "QPSK":0.3333, 16:-0.3333, "QAM-16":-0.3333, "QAM16":-0.3333, 64:-1, "QAM-64":-1, "QAM64":-1})
    if "Throughput" in columns:
        df["Throughput"] = df["Throughput"].apply(lambda x: (x-min_throughput)/(max_throughput-min_throughput)) # Normalize throughput btw 0 and 1 so that ReLu can be used
    
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
        f.write("Max Throughput: {}\n".format(max_throughput))
        f.write("Min Throughput: {}\n".format(min_throughput))
        f.write("[BPSK: 1, QPSK: 0.3333, QAM16: -0.3333, QAM64: -1]\n")
        f.write("UAV Sending Interval: [10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1]\n")
        f.write("Output: ['Reliable':0, 'QUEUE_OVERFLOW':1, 'RETRY_LIMIT_REACHED':2, 'Delay_Exceeded':3]\n")
        f.close()

    return df

def build_nn_model_v4():
    # For multiple output model
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
    return model

# Custom callback to clear memory to reduce RAM usage after each epoch
class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        keras.backend.clear_session()

if __name__ == "__main__":
    # Training params
    EPOCHS = 20
    CHECKPOINT_FILEPATH = '/home/rlim0005/nn_checkpoints/throughput_predict_nn_v4_video_sinr_dl'
    DATASET_PATH = "/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_Video"
    # CHECKPOINT_FILEPATH = '/home/research-student/omnet-fanet/nn_checkpoints/throughput_video_sinr_dl_tmp'
    # DATASET_PATH = "/media/research-student/One Touch/FANET Datasets/Dataset_NP100000_MultiModulation_Hovering_Video/Test/Test_Dataset_1_Downlink_Throughput.csv"
    LINK_TYPE = "downlink" # "uplink" / "downlink" / "video"
    VIDEO_NOVIDEO = "Video" # "NoVideo" / "Video" / "NoVideo_Part2"

    # Load dataset =================================================
    # df_train, df_holdout = load_train_holdout_dataset(DATASET_PATH, LINK_TYPE, VIDEO_NOVIDEO)
    df_bpsk_train, df_bpsk_holdout = generate_troughput_train_holdout_dataset(os.path.join(DATASET_PATH, "BPSK/BPSK_Downlink_Throughput.csv"), holdout_split=0.2)
    df_qpsk_train, df_qpsk_holdout = generate_troughput_train_holdout_dataset(os.path.join(DATASET_PATH, "QPSK/QPSK_Downlink_Throughput.csv"), holdout_split=0.2)
    df_qam16_train, df_qam16_holdout = generate_troughput_train_holdout_dataset(os.path.join(DATASET_PATH, "QAM16/QAM16_Downlink_Throughput.csv"), holdout_split=0.2)
    df_qam64_train, df_qam64_holdout = generate_troughput_train_holdout_dataset(os.path.join(DATASET_PATH, "QAM64/QAM64_Downlink_Throughput.csv"), holdout_split=0.2)
    df_train = pd.concat([df_bpsk_train, df_qpsk_train, df_qam16_train, df_qam64_train])
    df_holdout = pd.concat([df_bpsk_holdout, df_qpsk_holdout, df_qam16_holdout, df_qam64_holdout])
    # Normalize data
    df_train = normalize_data(df_train, columns=["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Throughput"], link_type=LINK_TYPE, save_details_path=CHECKPOINT_FILEPATH)
    df_holdout = normalize_data(df_holdout, columns=["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Throughput"], link_type=LINK_TYPE, save_details_path=None)
    # Get only inputs and output(s) for model
    X_train = df_train[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation"]].values
    X_test = df_holdout[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation"]].values
    throughput_train = df_train['Throughput'].values
    throughput_test = df_holdout['Throughput'].values
    # Clean up to save memory (so that oom don't make me cry)
    del df_train, df_holdout
    gc.collect()
    # Load dataset =================================================

    # Build model
    model = build_nn_model_v4()

    # Load pre-trained model for finetuning
    # model = keras.models.load_model(os.path.join(CHECKPOINT_FILEPATH, "model.002-0.3125.h5"), compile=False)

    # Compile the model
    model.compile(optimizer='adam', 
              loss={'throughput': 'mse'},
              metrics={'throughput': 'accuracy'})

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_FILEPATH,"model.{epoch:03d}-{val_loss:.4f}.h5"),
        save_weights_only=False,
        monitor='val_loss',
        mode='auto',
        save_freq='epoch')

    date = datetime.now()
    date_str = date.strftime("%d%m%Y")
    history = model.fit(X_train, throughput_train, epochs=EPOCHS, callbacks=[model_checkpoint_callback, ClearMemory()], validation_data=(X_test, throughput_test))
    with open(os.path.join(CHECKPOINT_FILEPATH, 'trainHistoryDict_{}'.format(date_str)), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # Save final model
    model.save(os.path.join(CHECKPOINT_FILEPATH,"final_model.h5"))