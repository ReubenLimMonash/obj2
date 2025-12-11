"""
Date: 11/06/2024
Desc: Computes BN CPT for UAV communication reliability using dataset details CSV file
      Uses horizontal distance, height, MCS, and USI as inputs
"""

import pandas as pd
import numpy as np 
from joblib import dump

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

if __name__ == "__main__":
    DATASET_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/data_processed/{}_Reliability.csv"
    SAVE_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/bn_cpts/parrotar2_reliability_bn_{}_{}.{}"
    LINKS = ["Downlink", "Uplink", "Video"]
    # LINKS = ["Downlink"]
    HDIST_BINS = np.arange(0, 710, 10)
    HDIST_BINS[-1] = HDIST_BINS[-1] + 1 # To include 700 m in the last bin
    HEIGHT_BINS = np.arange(60, 330, 30)
    HEIGHT_BINS[-1] = HEIGHT_BINS[-1] + 1 # To include 300 m in the last bin (left closed ended, right open ended except for final bin)
    MCS_INDEXES = np.arange(0, 8, 1)
    USI = [10, 20, 66.7, 100] # Just listing to get number of possible USI (left closed ended, right open ended except for final bin)

    for link in LINKS:
        # Read dataset and discretize
        df_dtypes = {"Horizontal_Distance": np.float64, "Height": np.int16,	"U2G_Distance": np.int32, "UAV_Sending_Interval": np.float64, "Mean_SINR": np.float64, "Std_Dev_SINR": np.float64,
                "Num_Sent": np.int32, "Num_Reliable": np.int32, "Num_Delay_Excd": np.int32, "Num_Incr_Rcvd": np.int32, "Num_Q_Overflow": np.int32, "Modulation": str, "Bitrate": np.float64}
        dataset_details_df = pd.read_csv(DATASET_PATH.format(link), 
                                    usecols = ["Horizontal_Distance", "Height", "UAV_Sending_Interval", "Modulation", "Bitrate", "Num_Sent", "Num_Reliable", "Num_Delay_Excd",
                                                "Num_Incr_Rcvd", "Num_Q_Overflow"],
                                    dtype=df_dtypes)
        dataset_details_df = get_mcs_index(dataset_details_df)
        dataset_details_df["UAV_Sending_Interval_Class"] = dataset_details_df["UAV_Sending_Interval"].replace({10:0, 20:1, 66.7:2, 100:3})
        dataset_details_df["Height_Class"] = pd.cut(dataset_details_df["Height"], bins=HEIGHT_BINS, right=False, include_lowest=True, labels=np.arange(0, len(HEIGHT_BINS)-1).astype('str'))
        dataset_details_df["Horizontal_Distance_Class"] = pd.cut(dataset_details_df["Horizontal_Distance"], bins=HDIST_BINS, right=False, include_lowest=True, labels=np.arange(0, len(HDIST_BINS)-1).astype('str'))

        cpt_list = []
        for usi_class in np.arange(0, 4):
            for mcs_index in MCS_INDEXES:
                for hdist_bin in np.arange(0, len(HDIST_BINS)-1).astype('str'):
                    for height_bin in np.arange(0, len(HEIGHT_BINS)-1).astype('str'):
                            df = dataset_details_df.loc[(dataset_details_df["Horizontal_Distance_Class"] == hdist_bin) & (dataset_details_df["Height_Class"] == height_bin) & 
                                                        (dataset_details_df["UAV_Sending_Interval_Class"] == usi_class) & (dataset_details_df["MCS"] == mcs_index)]
                            num_sent = df["Num_Sent"].sum()
                            num_reliable = df["Num_Reliable"].sum()
                            num_delay_excd = df["Num_Delay_Excd"].sum()
                            num_incr_rcvd = df["Num_Incr_Rcvd"].sum()
                            num_q_overflow = df["Num_Q_Overflow"].sum()
                            cpt_list.append({"Horizontal_Distance_Class": hdist_bin, "Height_Class": height_bin, "UAV_Sending_Interval_Class": usi_class, "MCS": mcs_index,
                                            "Reliability": num_reliable/num_sent, "Prob_Delay_Excd": num_delay_excd/num_sent, "Prob_Queue_Overflow": num_q_overflow/num_sent, "Prob_Incr_Rcvd": num_incr_rcvd/num_sent})

        cpt_df = pd.DataFrame(cpt_list)
        cpt_df.to_csv(SAVE_PATH.format("CPT", link, "csv"), index=False)