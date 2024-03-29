'''
Date: 10/07/2023
Desc: To get the reliability of each simulation scenario from raw sim output files
'''

import pandas as pd # for data manipulation 
import numpy as np
# import matplotlib.pyplot as plt # for drawing graphs
import os, sys, glob, math
import time
from tqdm import tqdm


def compile_micro_sim_tx_rx(file_list):
    '''
    Function to compile data from the CSV files generated by each micro-simulation
    Update: To specifically return the tx_df, rx_df files only
    Input: file_list - List of simulation files belonging to a certain scenario (micro-sim)
    Output: rx_df list, tx_df list
    '''
    # Specify dtypes to save memory
    tx_df_dtypes = {"TxTime": np.float32, "Packet_Name": "str", "Packet_Seq": np.uint32, "Bytes": np.uint16, "Dest_Addr": 'str'}
    rx_df_dtypes = {"RxTime": np.float64, "TxTime": np.float32,	"Packet_Name": "str", "Bytes": np.uint16, "RSSI": 'str', "U2G_SINR": np.float32, "U2U_SINR": np.float32, 
                    "U2G_BER": np.float32, "U2U_BER": np.float32, "Src_Addr": 'str', "Dest_Addr": 'str', "Hop_Count": np.uint8, "Delay": np.float32, 
                    "Queueing_Time": np.float32, "Backoff_Time": np.float32, "U2G_Distance": np.float16, "Retry_Count": np.uint8}

    # Let's get the GCS dfs ===============================================================
    gcs_rx_file = [file for file in file_list if (('_GCS-' in file) and ('-Rx' in file))]
    gcs_tx_file = [file for file in file_list if (('_GCS-' in file) and ('-Tx' in file))]
    if len(gcs_rx_file) > 0:
        gcs_rx_df = pd.read_csv(gcs_rx_file[0], dtype=rx_df_dtypes)
    else:
        print("GCS RX File Missing")
        print(file_list[0])
    if len(gcs_tx_file) > 0:
        gcs_tx_df = pd.read_csv(gcs_tx_file[0], dtype=tx_df_dtypes)
    else:
        print("GCS TX File Missing")
        print(file_list[0])

    # Let's get the GW dfs ===============================================================
    gw_rx_file = [file for file in file_list if (('_GW-' in file) and ('-Rx' in file))]
    gw_tx_file = [file for file in file_list if (('_GW-' in file) and ('-Tx' in file))]
    if len(gw_rx_file) > 0:
        gw_rx_df = pd.read_csv(gw_rx_file[0], dtype=rx_df_dtypes)
    else:
        print("GW RX File Missing")
        print(file_list[0])
    if len(gw_tx_file) > 0:
        gw_tx_df = pd.read_csv(gw_tx_file[0], dtype=tx_df_dtypes)
    else:
        print("GW TX File Missing")
        print(file_list[0])

    # Let's get the UAVs dfs ===============================================================
    uavs_rx_df_list = []
    uavs_tx_df_list = []
    uav_rx_files = [file for file in file_list if (('_UAV-' in file) and ('-Rx' in file))]
    uav_tx_files = [file for file in file_list if (('_UAV-' in file) and ('-Tx' in file))]
    uav_rx_files.sort()
    uav_tx_files.sort()
    if len(uav_rx_files) > 0:
        for uav_rx_file in uav_rx_files:
            uavs_rx_df_list.append(pd.read_csv(uav_rx_file, dtype=rx_df_dtypes))
    else:
        print("UAV RX File(s) Missing")
        print(file_list[0])
    if len(uav_tx_files) > 0:
        for uav_tx_file in uav_tx_files:
            uavs_tx_df_list.append(pd.read_csv(uav_tx_file, dtype=tx_df_dtypes))
    else:
        print("UAV TX File(s) Missing")
        print(file_list[0])

    rx_df_list = [gcs_rx_df, gw_rx_df] + uavs_rx_df_list
    tx_df_list = [gcs_tx_df, gw_tx_df] + uavs_tx_df_list

    return rx_df_list, tx_df_list

def count_received_packets(rx_df, delay_threshold):
    """
    This function updates the state of received packets, whether "Reliable" or "Delay_Exceeded", and returns the number of reliable packets in rx_df
    rx_df: Rx DF
    delay_threshold: Delay threshold for denoting "Delay_Exceeded"
    """
    rx_df["Delay"] = rx_df['RxTime']-rx_df['TxTime']
    rx_df["Packet_State"] = pd.Categorical(np.where(rx_df['Delay'] > delay_threshold , "Delay_Exceeded", "Reliable"))
    num_reliable = len(rx_df.loc[rx_df["Packet_State"] == "Reliable"])
    return num_reliable

def process_scenario(scenario, sim_root_path, dl_delay_threshold, ul_delay_threshold, GX_GCS=0):
    '''
    GX_GCS is the x-coordinate of GCS. 
    '''
    # print(scenario)
    
    scenario_files = glob.glob(sim_root_path + "/{}_*.csv".format(scenario)) # Get list of csv files belonging to this scenario
    scenario_params = scenario.split('_')
    # num_member = int(scenario_params[0].split('-')[-1])
    # inter_uav_distance = int(scenario_params[1].split('-')[-1])
    height = int(scenario_params[2].split('-')[-1]) 
    h_dist = float(scenario_params[3].split('-')[-1]) - GX_GCS 
    # u2g_dist = math.sqrt(h_dist**2 + height**2)
    uav_sending_interval = int(scenario_params[5].split('-')[-1])
    modulation = scenario_params[4].split('-')[-1]
    rx_df_list, tx_df_list = compile_micro_sim_tx_rx(scenario_files)

    # Sort out which df is which
    gcs_tx_df = tx_df_list[0]
    gcs_rx_df = rx_df_list[0]
    uavs_tx_df = pd.concat(tx_df_list[1:len(tx_df_list)], ignore_index=True)
    uavs_rx_df = pd.concat(rx_df_list[1:len(rx_df_list)], ignore_index=True)

    # Take only the packets within a certain range
    gcs_tx_df = gcs_tx_df.sort_values("TxTime")
    uavs_tx_df = uavs_tx_df.sort_values("TxTime")
    gcs_tx_df = gcs_tx_df.iloc[:20000]
    uavs_tx_df = uavs_tx_df.iloc[:20000]
    gcs_tx_df["Packet_Full_Name"] = gcs_tx_df["Packet_Name"].astype("str") + "-" + gcs_tx_df["Packet_Seq"].astype("str")
    uavs_tx_df["Packet_Full_Name"] = uavs_tx_df["Packet_Name"].astype("str") + "-" + uavs_tx_df["Packet_Seq"].astype("str")
    dl_pkts = gcs_tx_df["Packet_Full_Name"].values
    ul_pkts = uavs_tx_df["Packet_Full_Name"].values
    uavs_rx_df = uavs_rx_df.loc[uavs_rx_df["Packet_Name"].isin(dl_pkts)]
    gcs_rx_df = gcs_rx_df.loc[gcs_rx_df["Packet_Name"].isin(ul_pkts)]

    num_reliable_downlink = count_received_packets(uavs_rx_df, dl_delay_threshold)
    num_reliable_uplink = count_received_packets(gcs_rx_df, ul_delay_threshold)
    num_sent_downlink = len(gcs_tx_df)
    num_sent_uplink = len(uavs_tx_df)
    uplink_reliability = num_reliable_uplink / num_sent_uplink
    downlink_reliability = num_reliable_downlink / num_sent_downlink
    
    return {"Horizontal_Distance": h_dist, "Height": height, "Modulation": modulation, "UAV_Sending_Interval": uav_sending_interval, 
            "Uplink_Num_Sent": num_sent_uplink, "Downlink_Num_Sent": num_sent_downlink, "Uplink_Reliability": uplink_reliability, "Downlink_Reliability": downlink_reliability}

if __name__ == "__main__":
    '''
    MAKE SURE TO SET "GX_GCS" and "sending_interval_range" in function process_sim_data_v2
    '''
    sim_root_path = "/media/research-student/One Touch/FANET Datasets/Dataset_NP500000_MultiModulation_Hovering_NoVideo/Test/Exp1"
    csv_save_path = "/media/research-student/One Touch/FANET Datasets/Dataset_NP500000_MultiModulation_Hovering_NoVideo/Test/Exp1_reliability_2.csv"
    # sim_root_path = "/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_NoVideo/Test/Tmp"
    # csv_save_path = "/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_NoVideo/Test/Exp1_reliability.csv"
    # sim_root_path = "/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/Test/Test_Dataset_1_100000"
    # csv_save_path = "/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/Test/Test_Dataset_1_100000_reliability.csv"
    dl_delay_threshold = 1
    ul_delay_threshold = 1
    GX_GCS = 0

    scenario_list = [csv.split('/')[-1][0:-11] for csv in glob.glob(sim_root_path + "/*GCS-Tx.csv")] # Get list of "unique" scenarios
    data = []
    for scenario in tqdm(scenario_list):
        data.append(process_scenario(scenario, sim_root_path, dl_delay_threshold, ul_delay_threshold, GX_GCS=GX_GCS))

    reliability_df = pd.DataFrame(data)
    reliability_df.sort_values(by = ["Modulation", "UAV_Sending_Interval", "Height", "Horizontal_Distance"], inplace = True)
    reliability_df.to_csv(csv_save_path, index=False)
