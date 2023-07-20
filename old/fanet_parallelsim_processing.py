'''
Date: 18/06/2022
Desc: Processing of simulation data produced by method 2 using parallel runs
      Using Method 2 of data collection, multiple simulation runs were performed with the UAVs in linear mobility.
      Now we need to extract all files and compute reliability at points of interest
'''

import pandas as pd
import numpy as np
import os, sys, glob, math

def process_parallel_sim(sim_root_path, repeats, timestamps, num_members):
    '''
    This function calculates the reliability from n number of files at timestamps of interest, where n = repeats
    For distance points of interest, calculate the corresponding timestamps
    '''
    delay_th = 1
    num_success = np.zeros(len(timestamps)) # List to store number of successful packets receptions at each timestamps
    # num_gcs_sent = 0 # The number of GCS packets sent at each "point"
    for i in range(repeats):
        # For each repeats, get the number of reliably received packets 
        uavs_rx_df_list = [] # List to store all df for UAVs Rx app
        # gcs_tx_csv = os.path.join(sim_root_path, "/Run-{}_GCS-App[0]-Tx".format(i))
        # gcs_tx_df = pd.read_csv(gcs_tx_csv)
        gw_rx_csv = os.path.join(sim_root_path, "Run-{}_GW-App[0]-Rx.csv".format(i))
        gw_rx_df = pd.read_csv(gw_rx_csv)
        uavs_rx_df_list.append(gw_rx_df)
        for j in range(num_members):
            uav_csv = os.path.join(sim_root_path, "Run-{}_UAV-{}-Rx.csv".format(i,j))
            uav_rx_df = pd.read_csv(uav_csv)
            uavs_rx_df_list.append(uav_rx_df)

        uavs_rx_df = pd.concat(uavs_rx_df_list, ignore_index = True)
        uavs_reliable = uavs_rx_df[uavs_rx_df["Delay"] < delay_th] # Get the CNCData packets received reliably by all UAVs (delay < 1ms)
        # print(uavs_reliable)
        counter = 0
        for time in timestamps:
            # Check if the packet transmitted at time "time" is in uavs_reliable
            packet = uavs_reliable[uavs_reliable["TxTime"] == time]
            # print(packet)
            if not packet.empty:
                num_success[counter] += 1
            counter += 1
    reliability = num_success / repeats
    return reliability


if __name__ == "__main__":
    sim_root_path = "/home/reuben/omnetpp_sim_results/FANET_Verify/Method2"
    reliability = process_parallel_sim(sim_root_path, 5000, [0,1,2,3,4,5], 3)
    print(reliability)