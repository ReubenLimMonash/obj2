import pandas as pd
import numpy as np
import glob, math
from tqdm import tqdm
from multiprocessing.pool import Pool

# Modified Date: 18/04/2023
# Modified for new traffic model

def compile_dataset(processed_data_path):

    df_dtypes = {"TxTime": np.float32, "U2G_Distance": np.float32, "Height": np.int16,	"Num_Members": np.int16, "UAV_Sending_Interval": np.int16, "Bytes": np.int16, 
                "U2G_SINR": np.float32, "U2G_BER": np.float32, "Delay": np.float32, "Throughput": np.float32, "Queueing_Time": np.float32, "Packet_State": 'string', 
                "Retry_Count": np.int8, "Incorrectly_Received": np.int8, "Queue_Overflow": np.int8, "Packet_Name": 'string'}

    # Process and save uplink DF
    uplink_csvs = glob.glob(processed_data_path + "/*_uplink.csv")
    ul_df_list = []
    for csv_file in tqdm(uplink_csvs):
        df = pd.read_csv(csv_file, 
                        usecols = ['Packet_Name','U2G_H_Dist', 'Height', "Num_Members", "UAV_Sending_Interval", "Bytes", "U2G_SINR", "U2G_BER", 
                                "Delay", "Throughput", "Packet_State", "Retry_Count", "Incorrectly_Received", "Queue_Overflow", "Mean_SINR", "Std_Dev_SINR"],
                        dtype=df_dtypes)
        # Filter out rows where mean / std dev of sinr is NaN
        df = df[df['Mean_SINR'].notna()]
        df = df[df['Std_Dev_SINR'].notna()]
        # Let's cap the number of rows for each scenario at 100,000 packets for DL
        if len(df.index) > 100000:
            df = df.head(100000)
        ul_df_list.append(df)
    ul_df = pd.concat(ul_df_list, ignore_index=True)
    ul_df.to_csv(processed_data_path + "_uplink.csv", index=False)

    # Process and save downlink DF
    downlink_csvs = glob.glob(processed_data_path + "/*_downlink.csv")
    dl_df_list = []
    for csv_file in tqdm(downlink_csvs):
        df = pd.read_csv(csv_file, 
                        usecols = ['Packet_Name','U2G_H_Dist', 'Height', "Num_Members", "UAV_Sending_Interval", "Bytes", "U2G_SINR", "U2G_BER", 
                                "Delay", "Throughput", "Packet_State", "Retry_Count", "Incorrectly_Received", "Queue_Overflow", "Mean_SINR", "Std_Dev_SINR"],
                        dtype=df_dtypes)
        # Filter out rows where mean / std dev of sinr is NaN
        df = df[df['Mean_SINR'].notna()]
        df = df[df['Std_Dev_SINR'].notna()]
        # Let's cap the number of rows for each scenario at 100,000 packets for DL
        if len(df.index) > 100000:
            df = df.head(100000)
        dl_df_list.append(df)
    dl_df = pd.concat(dl_df_list, ignore_index=True)
    dl_df.to_csv(processed_data_path + "_downlink.csv", index=False)

if __name__ == "__main__":
    processed_data_paths = ["/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/BPSK_processed",
                            "/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QPSK_processed",
                            "/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM16_processed",
                            "/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM64_processed"]
    with Pool(4) as pool:
        pool.map(compile_dataset, processed_data_paths)