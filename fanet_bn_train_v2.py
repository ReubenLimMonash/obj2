# Date: 11/12/2022
import csv
import pandas as pd # for data manipulation 
import numpy as np
import networkx as nx # for drawing graphs
import matplotlib.pyplot as plt # for drawing graphs
import os, sys, glob, math, pickle

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

def euclidean_dist(row):
    # Function to calc euclidean distance on every df row 
    euc_dist = math.sqrt(row["U2G_Distance"]**2 + row["Height"]**2)
    return euc_dist

if __name__ == "__main__":
    csv_data_path = "/home/research-student/omnetpp_sim_results/No_ARP_CSV"
    downlink_csv = os.path.join(csv_data_path, "FANET_downlink_raw.csv")
    downlink_df = pd.read_csv(downlink_csv)

    # Add in some extra values I forgot to add in in the pre-processing step
    downlink_df["U2G_H_Dist"] = downlink_df.apply(lambda row: euclidean_dist(row), axis=1)
    e2e_delay = downlink_df["Delay"].to_numpy()
    jitter = e2e_delay[1:] - e2e_delay[0:-1]
    jitter = np.insert(jitter,0,0)
    downlink_df["Jitter"] = jitter

    # First, discretise the values to classes
    # Independent vars
    downlink_df["H_Dist_Class"] = pd.cut(downlink_df.U2G_H_Dist, bins=5, labels=['vs','s','m','l','vl'])
    downlink_df["Height_Class"] = pd.cut(downlink_df.Height, bins=3, labels=['s','m','l'])
    downlink_df["Num_Members_Class"] = pd.cut(downlink_df.Num_Members, bins=3, labels=['s','m','l'])
    downlink_df["Sending_Interval_Class"] = pd.cut(downlink_df.Sending_Interval, bins=3, labels=['s','m','l'])
    downlink_df["Packet_Size_Class"] = pd.cut(downlink_df.Bytes, bins=3, labels=['s','m','l'])
    # Second layer
    downlink_df["SINR_Class"] = pd.qcut(downlink_df.U2G_SINR, q=5, labels=['vs','s','m','l','vl'])
    downlink_df["Delay_Class"] = pd.qcut(downlink_df.Delay, q=5, labels=['vs','s','m','l','vl'])
    downlink_df["Throughput_Class"] = pd.qcut(downlink_df.Throughput, q=3, labels=['s','l'], duplicates='drop')
    downlink_df["Queueing_Time_Class"] = pd.qcut(downlink_df.Queueing_Time, q=3, labels=['s','l'], duplicates='drop')
    downlink_df["BER_Class"] = pd.qcut(downlink_df.U2G_BER, q=5, labels=['vs','s','m','l','vl'])
    downlink_df["Jitter_Class"] = pd.qcut(downlink_df.Jitter, q=3, labels=['s','m','l'])
    
    # Calculate the conditional probabilities table for each second layer class
    parents_1 = ["H_Dist_Class", "Height_Class", "Num_Members_Class", "Sending_Interval_Class", "Packet_Size_Class"]
    sinr_cpt = cpt_probs(downlink_df, child="SINR_Class", parents=parents_1)
    delay_cpt = cpt_probs(downlink_df, child="Delay_Class", parents=parents_1)
    throughput_cpt = cpt_probs(downlink_df, child="Throughput_Class", parents=parents_1)
    queueing_cpt = cpt_probs(downlink_df, child="Queueing_Time_Class", parents=parents_1)
    ber_cpt = cpt_probs(downlink_df, child="BER_Class", parents=parents_1)
    jitter_cpt = cpt_probs(downlink_df, child="Jitter_Class", parents=parents_1)
    parents_2 = ["SINR_Class", "Delay_Class", "Throughput_Class", "Queueing_Time_Class", "BER_Class", "Jitter_Class"]
    reliability_cpt = cpt_probs(downlink_df, child="Reliable", parents=parents_2)
    delay_exceeded_cpt = cpt_probs(downlink_df, child="Delay_Exceeded", parents=parents_2)
    num_dropped_cpt = cpt_probs(downlink_df, child="Number_Dropped", parents=parents_2)

    # Save the CPTs (to pickle)
    sinr_cpt.to_pickle("sinr_cpt.pkl")
    delay_cpt.to_pickle("delay_cpt.pkl")
    throughput_cpt.to_pickle("throughput_cpt.pkl")
    queueing_cpt.to_pickle("queueing_cpt.pkl")
    ber_cpt.to_pickle("ber_cpt.pkl")
    jitter_cpt.to_pickle("jitter_cpt.pkl")
    reliability_cpt.to_pickle("reliability_cpt.pkl")
    delay_exceeded_cpt.to_pickle("delay_exceeded_cpt.pkl")
    num_dropped_cpt.to_pickle("num_dropped_cpt.pkl")

    # Save the CPTs (to numpy)
    # np.save("sinr_cpt.npy", sinr_cpt.to_numpy(), allow_pickle=True)
    # np.save("delay_cpt.npy", delay_cpt.to_numpy(), allow_pickle=True)
    # np.save("throughput_cpt.npy", throughput_cpt.to_numpy(), allow_pickle=True)
    # np.save("queueing_cpt.npy", queueing_cpt.to_numpy(), allow_pickle=True)
    # np.save("ber_cpt.npy", ber_cpt.to_numpy(), allow_pickle=True)
    # np.save("jitter_cpt.npy", jitter_cpt.to_numpy(), allow_pickle=True)
    # np.save("reliability_cpt.npy", reliability_cpt.to_numpy(), allow_pickle=True)
    # np.save("delay_exceeded_cpt.npy", delay_exceeded_cpt.to_numpy(), allow_pickle=True)
    # np.save("num_dropped_cpt.npy", num_dropped_cpt.to_numpy(), allow_pickle=True)