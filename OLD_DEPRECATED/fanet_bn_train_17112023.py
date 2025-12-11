# Multinomail BN Model for UAV Comm. Reliability Prediction
# Date: 18/11/2023

import pandas as pd
import numpy as np 
import math
import os
from tqdm import tqdm
from sklearn.naive_bayes import MultinomialNB
from joblib import dump
from multiprocessing.pool import Pool
from itertools import repeat

def generate_reliability_dataset(dataset_details_df, test_split=0.2):
    # df_dtypes = {"Horizontal_Distance": np.float64, "Height": np.int16,	"U2G_Distance": np.int32, "UAV_Sending_Interval": np.float64, "Mean_SINR": np.float64, "Std_Dev_SINR": np.float64,
    #              "Modulation": 'string', "Num_Sent": np.int32, "Num_Reliable": np.int32, "Num_Delay_Excd": np.int32, "Num_Incr_Rcvd": np.int32, "Num_Q_Overflow": np.int32}
    # dataset_details = pd.read_csv(dataset_details_csv, 
    #                               usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Num_Sent", "Num_Reliable", "Num_Delay_Excd",
    #                                          "Num_Incr_Rcvd", "Num_Q_Overflow"],
    #                               dtype=df_dtypes)
    df_train_list = []
    for row in tqdm(dataset_details_df.itertuples()):
        mean_sinr = row.Mean_SINR_Class
        std_dev_sinr = row.Std_Dev_SINR_Class
        uav_send_int = row.UAV_Sending_Interval_Class
        mcs = row.MCS
        num_reliable = row.Num_Reliable
        num_delay_excd = row.Num_Delay_Excd
        num_incr_rcvd = row.Num_Incr_Rcvd
        num_q_overflow = row.Num_Q_Overflow

        if num_reliable > 0:
            reliable_packets = pd.DataFrame({"Mean_SINR_Class": mean_sinr, "Std_Dev_SINR_Class": std_dev_sinr, "UAV_Sending_Interval_Class": uav_send_int, "MCS": mcs, "Packet_State": 0}, index=[0])
            reliable_packets = reliable_packets.loc[reliable_packets.index.repeat(num_reliable)]
        else:
            reliable_packets = pd.DataFrame({})

        if num_delay_excd > 0:
            delay_excd_packets = pd.DataFrame({"Mean_SINR_Class": mean_sinr, "Std_Dev_SINR_Class": std_dev_sinr, "UAV_Sending_Interval_Class": uav_send_int, "MCS": mcs, "Packet_State": 1}, index=[0])
            delay_excd_packets = delay_excd_packets.loc[delay_excd_packets.index.repeat(num_delay_excd)]
        else:
            delay_excd_packets = pd.DataFrame({})

        if num_q_overflow > 0:
            q_overflow_packets = pd.DataFrame({"Mean_SINR_Class": mean_sinr, "Std_Dev_SINR_Class": std_dev_sinr, "UAV_Sending_Interval_Class": uav_send_int, "MCS": mcs, "Packet_State": 2}, index=[0])
            q_overflow_packets = q_overflow_packets.loc[q_overflow_packets.index.repeat(num_q_overflow)]
        else:
            q_overflow_packets = pd.DataFrame({})

        if num_incr_rcvd > 0:
            incr_rcvd_packets = pd.DataFrame({"Mean_SINR_Class": mean_sinr, "Std_Dev_SINR_Class": std_dev_sinr, "UAV_Sending_Interval_Class": uav_send_int, "MCS": mcs, "Packet_State": 3}, index=[0])
            incr_rcvd_packets = incr_rcvd_packets.loc[incr_rcvd_packets.index.repeat(num_incr_rcvd)]
        else:
            incr_rcvd_packets = pd.DataFrame({})
        df_train_list.append(pd.concat([reliable_packets, delay_excd_packets, q_overflow_packets, incr_rcvd_packets]))

    df_train = pd.concat(df_train_list)
    return df_train

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

def process_data_n_train_bn(dataset_path, num_bins=100):

    df_dtypes = {"Horizontal_Distance": np.float64, "Height": np.int16,	"U2G_Distance": np.int32, "UAV_Sending_Interval": np.float64, "Mean_SINR": np.float64, "Std_Dev_SINR": np.float64,
                "Num_Sent": np.int32, "Num_Reliable": np.int32, "Num_Delay_Excd": np.int32, "Num_Incr_Rcvd": np.int32, "Num_Q_Overflow": np.int32, "Modulation": str, "Bitrate": np.float64}
    dataset_details_df = pd.read_csv(dataset_path, 
                                usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Bitrate", "Num_Sent", "Num_Reliable", "Num_Delay_Excd",
                                            "Num_Incr_Rcvd", "Num_Q_Overflow"],
                                dtype=df_dtypes)
    dataset_details_df = get_mcs_index(dataset_details_df)

    # Change sending interval categorial to numeric
    dataset_details_df["UAV_Sending_Interval_Class"] = dataset_details_df["UAV_Sending_Interval"].replace({10:0, 20:1, 66.7:2, 100:3})

    # Quantize mean and std dev of sinr
    _, mean_sinr_bins = pd.qcut(dataset_details_df.Mean_SINR, q=num_bins, retbins=True)
    mean_sinr_bins = np.concatenate(([-np.inf], mean_sinr_bins[1:-1], [np.inf]))
    _, std_dev_sinr_bins = pd.qcut(dataset_details_df.Std_Dev_SINR, q=num_bins, retbins=True)
    std_dev_sinr_bins = np.concatenate(([-np.inf], std_dev_sinr_bins[1:-1], [np.inf]))

    dataset_details_df["Mean_SINR_Class"] = pd.cut(dataset_details_df.Mean_SINR, mean_sinr_bins, right=True, include_lowest=False, labels=False)
    dataset_details_df["Std_Dev_SINR_Class"] = pd.cut(dataset_details_df.Std_Dev_SINR, std_dev_sinr_bins, right=True, include_lowest=False, labels=False)


    # # Generate dataset samples
    df_train = generate_reliability_dataset(dataset_details_df)

    X = df_train[["Mean_SINR_Class", "Std_Dev_SINR_Class", "UAV_Sending_Interval_Class", "MCS"]].values
    packet_state_train = df_train['Packet_State'].values

    model = MultinomialNB(force_alpha=True)
    model.fit(X, packet_state_train)
    return model

if __name__ == "__main__":
    DATASET_PATH = "/home/rlim0005/FANET_Dataset/Dataset_NP10000_DJISpark/DJI_Spark_{}_Reliability.csv"
    SAVE_PATH = "/home/rlim0005/bn_ckpt/djispark_reliability_bn_{}.joblib"
    LINKS = ["Downlink", "Uplink", "Video"]

    dataset_paths = [DATASET_PATH.format(link) for link in LINKS]
    save_paths = [SAVE_PATH.format(link) for link in LINKS]
    num_bins = 100
    models = [] # The models saved will correspond to the order specified in LINKS
    with Pool(1) as pool: # If we use 3, OOM!
        for result in pool.starmap(process_data_n_train_bn, zip(dataset_paths, repeat(num_bins))):
            models.append(result)
    
    for i in range(len(LINKS)):
        dump(models[i], save_paths[i]) 
