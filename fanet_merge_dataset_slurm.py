import pandas as pd
import numpy as np

# Load the dataset
df_dtypes = {"TxTime": np.float32, "U2G_Distance": np.float32, "Height": np.int16,	"Num_Members": np.int16, "UAV_Sending_Interval": np.int16, "Bytes": np.int16, 
            "U2G_SINR": np.float32, "U2G_BER": np.float32, "Delay": np.float32, "Throughput": np.float32, "Queueing_Time": np.float32, "Packet_State": 'category', 
            "Retry_Count": np.int8, "Incorrectly_Received": np.int8, "Queue_Overflow": np.int8, "Packet_Name": 'string', "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
            "UAV_Sending_Interval": np.int16}

dl_df_bpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/BPSK_processed_downlink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Incorrectly_Received", "Queue_Overflow",
                               "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_bpsk["Modulation"] = 1
# Uncomment Below To Use One-Hot Encoding for Modulation
# dl_df_bpsk["Modulation"] = 0

dl_df_qpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QPSK_processed_downlink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Incorrectly_Received", "Queue_Overflow",
                               "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qpsk["Modulation"] = 0.3333
# Uncomment Below To Use One-Hot Encoding for Modulation
# dl_df_qpsk["Modulation"] = 1

dl_df_qam16 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM16_processed_downlink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Incorrectly_Received", "Queue_Overflow",
                               "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qam16["Modulation"] = -0.3333
# Uncomment Below To Use One-Hot Encoding for Modulation
# dl_df_qam16["Modulation"] = 2

dl_df_qam64 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM64_processed_downlink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Incorrectly_Received", "Queue_Overflow",
                               "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qam64["Modulation"] = -1
# Uncomment Below To Use One-Hot Encoding for Modulation
# dl_df_qam64["Modulation"] = 3

dl_df = pd.concat([dl_df_bpsk, dl_df_qpsk, dl_df_qam16, dl_df_qam64], ignore_index=True)

dl_df.to_pickle("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/downlink_df_all.pkl")
print("Done")