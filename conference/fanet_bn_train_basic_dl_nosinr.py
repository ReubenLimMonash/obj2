'''
Date: 14/06/2023
Desc: To train a BN classifier to predict FANET reliability and failure modes
Modified: To use h_dist and height as inputs instead of SINR
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
save_cpt_filepath = '/home/research-student/omnet-fanet/cpt/bn_basic_multimodulation_video_nosinr_dl'

df_dtypes = {"TxTime": np.float32, "U2G_Distance": np.float32, "Height": np.int16,	"Num_Members": np.int16, "UAV_Sending_Interval": np.int16, "Bytes": np.int16, 
            "U2G_SINR": np.float32, "U2G_BER": np.float32, "Delay": np.float32, "Throughput": np.float32, "Queueing_Time": np.float32, "Packet_State": 'category', 
            "Retry_Count": np.int8, "Incorrectly_Received": np.int8, "Queue_Overflow": np.int8, "Packet_Name": 'string', "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
            "UAV_Sending_Interval": np.int16}

print("Loading Datasets")
# Load training dataset ==========================================================================================================================
dl_df_bpsk = pd.read_csv("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_Video/BPSK_processed_train_downlink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_bpsk["Modulation"] = "BPSK"

dl_df_qpsk = pd.read_csv("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_Video/QPSK_processed_train_downlink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qpsk["Modulation"] = "QPSK"

dl_df_qam16 = pd.read_csv("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_Video/QAM16_processed_train_downlink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qam16["Modulation"] = "QAM16"

dl_df_qam64 = pd.read_csv("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_Video/QAM64_processed_train_downlink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qam64["Modulation"] = "QAM64"

dl_df_train = pd.concat([dl_df_bpsk, dl_df_qpsk, dl_df_qam16, dl_df_qam64], ignore_index=True)
# Load training dataset ==========================================================================================================================

# Load test dataset ==========================================================================================================================
dl_df_bpsk = pd.read_csv("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_Video/BPSK_processed_holdout_downlink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_bpsk["Modulation"] = "BPSK"

dl_df_qpsk = pd.read_csv("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_Video/QPSK_processed_holdout_downlink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qpsk["Modulation"] = "QPSK"

dl_df_qam16 = pd.read_csv("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_Video/QAM16_processed_holdout_downlink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qam16["Modulation"] = "QAM16"

dl_df_qam64 = pd.read_csv("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_Video/QAM64_processed_holdout_downlink.csv",
                    usecols = ["Mean_SINR", "Std_Dev_SINR", "Num_Members", "UAV_Sending_Interval", "Packet_State", "Delay", "U2G_H_Dist", "Height"],
                    dtype=df_dtypes)
dl_df_qam64["Modulation"] = "QAM64"

dl_df_holdout = pd.concat([dl_df_bpsk, dl_df_qpsk, dl_df_qam16, dl_df_qam64], ignore_index=True)
# Load test dataset ==========================================================================================================================

print("Processing Dataset")

dl_df = pd.concat([dl_df_train, dl_df_holdout], ignore_index=True)

# Filter out rows where U2G_H_Dist / Height of sinr is NaN
dl_df = dl_df[dl_df['U2G_H_Dist'].notna()]
dl_df = dl_df[dl_df['Height'].notna()]
dl_df.sort_values(by = "U2G_H_Dist")
# Drop rows where Packet State is FAILED or INTERFACE_DOWN (because we don't recognize the failure mode)
dl_df = dl_df.loc[dl_df["Packet_State"].isin(["Reliable", "Delay_Exceeded", "RETRY_LIMIT_REACHED", "QUEUE_OVERFLOW"])]

# Define ranges of input parameters
max_height = 300
min_height = 60
max_h_dist = 1200
min_h_dist = 0

# Normalize data (Min Max Normalization between [-1,1])
dl_df["Height"] = dl_df["Height"].apply(lambda x: 2*(x-min_height)/(max_height-min_height) - 1)
dl_df["U2G_H_Dist"] = dl_df["U2G_H_Dist"].apply(lambda x: 2*(x-min_h_dist)/(max_h_dist-min_h_dist) - 1)
dl_df["UAV_Sending_Interval_Class"] = dl_df["UAV_Sending_Interval"].replace({10:'vs', 20:'s', 40:'m', 100:'l', 1000:'vl'})
# dl_df['Packet_State'] = dl_df['Packet_State'].replace({"Reliable":0, "QUEUE_OVERFLOW":1, "RETRY_LIMIT_REACHED":2, "Delay_Exceeded":3})

# Discretize the h_dist and height of SINR
h_dist_num_classes = 61
h_dist_labels = [str(num) for num in np.arange(0,h_dist_num_classes)+1]
uav_swarm_radius = 5
h_dist_bnd_offset = 2 * uav_swarm_radius / max_h_dist
h_dist_class_bnd = np.linspace(-1, 1, h_dist_num_classes, endpoint=True)
h_dist_class_bnd[1:len(h_dist_class_bnd)] = h_dist_class_bnd[1:len(h_dist_class_bnd)] - h_dist_bnd_offset # Offset boundaries by radius
h_dist_class_bnd = h_dist_class_bnd.tolist()
h_dist_class_bnd.append(2) # Appending 2 to catch normalized inputs above 1
h_dist_class_bnd[0] = -2 # Making the lowest boundary -2 to catch normalized inputs below -1
height_num_classes = 9
height_labels = [str(num) for num in np.arange(0,height_num_classes)+1]
height_class_bnd = np.linspace(-1, 1, height_num_classes, endpoint=True).tolist()
height_class_bnd.append(2) # Appending 2 to catch normalized inputs above 1
height_class_bnd[0] = -2 # Making the lowest boundary -2 to catch normalized inputs below -1
dl_df["U2G_H_Dist_Class"] = pd.cut(dl_df.U2G_H_Dist, h_dist_class_bnd, right=False, include_lowest=True, labels=h_dist_labels)
dl_df["Height_Class"] = pd.cut(dl_df.Height, height_class_bnd, right=False, include_lowest=True, labels=height_labels)

print("Computing CPT")
# Get the CPT for Packet_State
parents_pkt_state = ["U2G_H_Dist_Class", "Height_Class", "UAV_Sending_Interval_Class", "Modulation"]
pkt_state_cpt = cpt_probs(dl_df, child="Packet_State", parents=parents_pkt_state)

# Validate CPT. The below should be zero
cpt_check = len(pkt_state_cpt.loc[(pkt_state_cpt["Delay_Exceeded"]==0) & (pkt_state_cpt["QUEUE_OVERFLOW"]==0) & (pkt_state_cpt["RETRY_LIMIT_REACHED"]==0) & (pkt_state_cpt["Reliable"]==0)])
if cpt_check:
    print("Missing values in CPT!")

# Save the CPT as CSV
pkt_state_cpt.to_csv(os.path.join(save_cpt_filepath, "packet_state_dl_cpt.csv"))

# Record details of inputs and output for model
f = open(os.path.join(save_cpt_filepath,"model_details.txt"), "w")
f.write("Max Height (m): {}\n".format(max_height))
f.write("Min Height (m): {}\n".format(min_height))
f.write("Max H_Dist (m): {}\n".format(max_h_dist))
f.write("Min H_Dist (m): {}\n".format(min_h_dist))
f.write("Modulation Classes: [BPSK, QPSK, QAM16, QAM64]\n")
f.write("UAV Sending Interval Classes: [10:'vs', 20:'s', 40:'m', 100:'l', 1000:'vl']\n")
f.write("Output Classes: ['Reliable', 'QUEUE_OVERFLOW', 'RETRY_LIMIT_REACHED', 'Delay_Exceeded']\n")
f.close()
