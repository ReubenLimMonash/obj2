'''
Date: 22/05/2023
Desc: To train a BN classifier to predict FANET reliability and failure modes
'''

import pandas as pd
import numpy as np 
import os
import math
import pickle

# This function helps to calculate probability distribution, which goes into BBN (note, can handle up to 2 parents)
def cpt_probs(df, child, parents):
    try:
        # dependencies_arr = [pd.Categorical(df[parent],categories=df[parent].cat.categories.tolist()) for parent in parents]
        dependencies_arr = [df[parent] for parent in parents]
        # cpt = pd.crosstab(dependencies_arr, df[child], rownames=parents, colnames=[child], margins=False, normalize='index', dropna=False).sort_index().to_numpy().reshape(-1).tolist()
        cpt = pd.crosstab(dependencies_arr, df[child], rownames=parents, colnames=[child], margins=False, normalize='index', dropna=False).sort_index()
        return cpt
    except Exception as err:
        print(err)
        return None 
    
# Training params
save_cpt_filepath = '/home/rlim0005/cpt/bn_basic_multimodulation_novideo_sinr_dl'
delay_threshold = 1

df_dtypes = {"TxTime": np.float32, "U2G_Distance": np.float32, "Height": np.int16,	"Num_Members": np.int16, "UAV_Sending_Interval": np.int16, "Bytes": np.int16, 
            "U2G_SINR": np.float32, "U2G_BER": np.float32, "Delay": np.float32, "Throughput": np.float32, "Queueing_Time": np.float32, "Packet_State": 'category', 
            "Retry_Count": np.int8, "Incorrectly_Received": np.int8, "Queue_Overflow": np.int8, "Packet_Name": 'string', "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
            "UAV_Sending_Interval": np.int16}
print("Loading BPSK UL Data")
dl_df_bpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/BPSK_processed_downlink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Incorrectly_Received", "Queue_Overflow",
                               "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_bpsk["Modulation"] = "BPSK"
print("Loading QPSK UL Data")
dl_df_qpsk = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QPSK_processed_downlink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Incorrectly_Received", "Queue_Overflow",
                               "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qpsk["Modulation"] = "QPSK"
print("Loading QAM16 UL Data")
dl_df_qam16 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM16_processed_downlink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Incorrectly_Received", "Queue_Overflow",
                               "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qam16["Modulation"] = "QAM16"
print("Loading QAM64 UL Data")
dl_df_qam64 = pd.read_csv("/home/rlim0005/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/QAM64_processed_downlink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Incorrectly_Received", "Queue_Overflow",
                               "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qam64["Modulation"] = "QAM64"

dl_df = pd.concat([dl_df_bpsk, dl_df_qpsk, dl_df_qam16, dl_df_qam64], ignore_index=True)

print("Processing data")
# Filter out rows where mean / std dev of sinr is NaN
dl_df = dl_df[dl_df['Mean_SINR'].notna()]
dl_df = dl_df[dl_df['Std_Dev_SINR'].notna()]

dl_df.sort_values(by = "U2G_H_Dist")

# Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
dl_df = dl_df.loc[dl_df["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]

# Normalize the mean and std dev of SINR (Min Max Normalization between [-1,1])
max_mean_sinr = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
max_std_dev_sinr = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
min_mean_sinr = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
min_std_dev_sinr = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)
dl_df["Mean_SINR"] = dl_df["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
dl_df["Std_Dev_SINR"] = dl_df["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1)
dl_df["UAV_Sending_Interval_Class"] = dl_df["UAV_Sending_Interval"].replace({10:'vs', 20:'s', 40:'m', 100:'l', 1000:'vl'})
# dl_df['Packet_State'] = dl_df['Packet_State'].replace({"Reliable":0, "QUEUE_OVERFLOW":1, "RETRY_LIMIT_REACHED":2, "Delay_Exceeded":3})

# Discretize the mean and std dev of SINR
sinr_num_classes = 100
sinr_labels = [str(num) for num in np.arange(0,sinr_num_classes)+1]
sinr_class_bnd = np.linspace(-1, 1, sinr_num_classes, endpoint=False).tolist()
sinr_class_bnd.append(2) # Appending 2 to catch normalized inputs above 1
sinr_class_bnd[0] = -2 # Making the lowest boundary -2 to catch normalized inputs below -1
dl_df["Mean_SINR_Class"] = pd.cut(dl_df.Mean_SINR, sinr_class_bnd, right=False, include_lowest=True, labels=sinr_labels)
dl_df["Std_Dev_SINR_Class"] = pd.cut(dl_df.Std_Dev_SINR, sinr_class_bnd, right=False, include_lowest=True, labels=sinr_labels)

print("Computing CPT")
# Get the CPT for Packet_State
parents_pkt_state = ["Mean_SINR_Class", "Std_Dev_SINR_Class", "UAV_Sending_Interval_Class", "Modulation"]
pkt_state_cpt = cpt_probs(dl_df, child="Packet_State", parents=parents_pkt_state)

print("Saving...")
# Save the CPT as CSV
pkt_state_cpt.to_csv(os.path.join(save_cpt_filepath, "packet_state_dl_cpt.csv"))

# Record details of inputs and output for model
f = open(os.path.join(save_cpt_filepath,"model_details.txt"), "w")
f.write("Mean and Std Dev SINR are in dB\n")
f.write("Max Mean SINR (dB): {}\n".format(max_mean_sinr))
f.write("Min Mean SINR (dB): {}\n".format(min_mean_sinr))
f.write("Max Std Dev SINR (dB): {}\n".format(max_std_dev_sinr))
f.write("Min Std Dev SINR (dB): {}\n".format(min_std_dev_sinr))
f.write("Modulation Classes: [BPSK, QPSK, QAM16, QAM64]\n")
f.write("UAV Sending Interval Classes: [10:'vs', 20:'s', 40:'m', 100:'l', 1000:'vl']\n")
f.write("Output Classes: ['Reliable', 'QUEUE_OVERFLOW', 'RETRY_LIMIT_REACHED', 'Delay_Exceeded']\n")
f.close()

print("Done")