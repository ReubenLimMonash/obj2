'''
Date: 15/06/2022
Desc: Processing of simulation data, where everything is dumped to a single folder
'''

import pandas as pd
import numpy as np
import os, sys, glob, math
from sklearn.feature_selection import mutual_info_regression
import scipy.stats as stats

def rssi_to_np(rssi):
    # Function to convert rssi data from string (e.g. "435 pW") to exp (435e-12)
    rssi_num = np.zeros(rssi.shape)
    index = 0
    for r in rssi:
        num = r[0:-2]
        expn = r[-2:]
        # print(num)
        # print(expn)
        if expn == " W":
            # print(num)
            # print(index)
            rssi_num[index] = float(num)
        elif expn == "mW":
            rssi_num[index] = float(num) * 1e-3
        elif expn == "uW":
            rssi_num[index] = float(num) * 1e-6
        elif expn == "nW":
            rssi_num[index] = float(num) * 1e-9
        elif expn == "pW":
            rssi_num[index] = float(num) * 1e-12
        else:
            print(expn)
            raise ValueError("Unhandled unit prefix")
        index += 1
    return rssi_num

def process_micro_sim(file_list, delay_th):
    '''
    Function to process the CSV files generated by each micro-simulation
    Input: file_list - List of simulation files belonging to a certain scenario (micro-sim)
           delay_th - The delay threshold to calculate reliability
    '''
    uavs_rx_df_list = [] # List to store all df for UAVs Rx app
    uavs_tx_df_list = [] # List to store all df for UAVs Tx app
    for file in file_list:
        if ('_GCS-' in file) and ('-Tx' in file):
            # DOWNLINK
            # This is the GCS Tx file, recording the sent packets from GCS
            gcs_tx_df = pd.read_csv(file)
            num_gcs_sent = gcs_tx_df.shape[0] # The number of data entries in gcs_tx_df is the number of packets sent by the GCS
            # num_gcs_sent_members = gcs_tx_df[gcs_tx_df["Dest_Addr"] != "192.168.0.1"].shape[0] # The number of packets sent to member UAVs (192.168.0.1 is the addr of the GW)
            # num_gcs_sent_gw = gcs_tx_df[gcs_tx_df["Dest_Addr"] == "192.168.0.1"].shape[0] # The number of packets sent to the GW (192.168.0.1 is the addr of the GW)
        elif ('_GW-' in file) and ('-Rx' in file):
            # DOWNLINK
            # This is the gateway Rx file, let's get the information of packets received from GCS
            gw_rx_df = pd.read_csv(file)
            uavs_rx_df_list.append(gw_rx_df)
        elif ('_UAV-' in file) and ('-Rx' in file):
            # DOWNLINK
            # This is a UAV Rx file. To concatenate all such files into a single df
            uav_rx_df = pd.read_csv(file)
            # uav_cnc_data = uav_rx_df["CNCData" in uav_rx_df["Packet_Name"]] # Get the CNC Data received by this UAV
            # uav_cnc_reliable = uav_cnc_data[uav_cnc_data["Delay"] < delay_th] # Get the CNCData packets received reliably by this UAV (delay < 1ms)
            uavs_rx_df_list.append(uav_rx_df) # Append to list for concatenation later
        elif ('_GCS-' in file) and ('-Rx' in file):
            # UPLINK
            # This is a GCS Rx file, recording packets received from UAVs-
            gcs_rx_df = pd.read_csv(file)
            gcs_reliable = gcs_rx_df[gcs_rx_df["Delay"] < delay_th]
            num_gcs_rcvd = gcs_reliable.shape[0] # The number of data entries in gcs_reliable is the number of packets received reliably by the GCS
        elif ('_GW-' in file) and ('-Tx' in file):
            # UPLINK
            # This is the gateway Tx file, recording packet transmissions to GCS from gateway
            gw_tx_df = pd.read_csv(file)
            uavs_tx_df_list.append(gw_tx_df) # Append to list for concatenation later
        elif ('_UAV-' in file) and ('-Tx' in file):
            # DOWNLINK
            # This is a UAV Rx file. To concatenate all such files into a single df
            uav_tx_df = pd.read_csv(file)
            uavs_tx_df_list.append(uav_tx_df) # Append to list for concatenation later
        else:
            # This file type is not handled, pass 
            pass

    if uavs_rx_df_list:
        uavs_rx_df = pd.concat(uavs_rx_df_list, ignore_index = True)
        uavs_reliable = uavs_rx_df[uavs_rx_df["Delay"] < delay_th] # Get the CNCData packets received reliably by all UAVs (delay < 1ms)
        # Process DOWNLINK data (NOT INCLUDING GW)
        dl_reliability = uavs_reliable.shape[0] / num_gcs_sent # Calculate the communication reliability of GCS -> UAVs
        dl_avg_rssi = rssi_to_np(uavs_rx_df["RSSI"]).mean()
        dl_avg_sinr = uavs_rx_df["SINR"].mean()
        gw_avg_rssi = rssi_to_np(gw_rx_df["RSSI"]).mean()
        gw_avg_sinr = gw_rx_df["SINR"].mean()
        dl_avg_delay = uavs_rx_df["Delay"].mean()
        dl_avg_throughput = uavs_rx_df["Throughput"].mean()
        # dl_avg_distance = uavs_rx_df["Distance"].mean()
        dl_json = {"Reliability": dl_reliability, "Avg_RSSI": dl_avg_rssi, "Avg_SINR": dl_avg_sinr, "GW_Avg_RSSI": gw_avg_rssi, "GW_Avg_SINR": gw_avg_sinr, "Avg_Delay": dl_avg_delay, "Avg_Throughput": dl_avg_throughput}
    else:
        dl_json = None

    if uavs_tx_df_list:
        uavs_tx_df = pd.concat(uavs_tx_df_list, ignore_index = True)
        # Process UPLINK data
        num_uav_sent = uavs_tx_df.shape[0] # Total number of packets sent by the UAVs to the GCS
        ul_reliability = num_gcs_rcvd / num_uav_sent # Calculate the communication reliability of UAVs -> GCS   
        ul_avg_rssi = rssi_to_np(gcs_rx_df["RSSI"]).mean()
        ul_avg_sinr = gcs_rx_df["SINR"].mean()
        ul_avg_delay = gcs_rx_df["Delay"].mean()
        ul_avg_throughput = gcs_rx_df["Throughput"].mean()
        # ul_avg_distance = gcs_rx_df["Distance"].mean()
        ul_json = {"Reliability": ul_reliability, "Avg_RSSI": ul_avg_rssi, "Avg_SINR": ul_avg_sinr, "Avg_Delay": ul_avg_delay, "Avg_Throughput": ul_avg_throughput}
    else:
        ul_json = None

    return dl_json, ul_json

def process_reliability(sim_root_path):
    delay_th = 1 # Delay threshold of 1ms
    # Create pd df to store micro-sim processed data
    dl_df = pd.DataFrame(columns = ['Reliability', 'Avg_RSSI', 'Avg_SINR', 'Avg_Delay', 'Avg_Throughput', 'Swarm_Distance', 'Horizontal_Distance', 'Height', 'Inter_UAV_Distance', 'Num_Members', 'Packet_Size', 'Sending_Rate']) # Downlink dataframe
    ul_df = pd.DataFrame(columns = ['Reliability', 'Avg_RSSI', 'Avg_SINR', 'Avg_Delay', 'Avg_Throughput', 'Swarm_Distance', 'Horizontal_Distance', 'Height', 'Inter_UAV_Distance', 'Num_Members', 'Packet_Size', 'Sending_Rate']) # Uplink dataframe

    # Get list of "unique" scenarios
    scenario_list = [csv.split('/')[-1][0:-18] for csv in glob.glob(sim_root_path + "/*[[0]]-Tx.csv")]
    # print(scenario_list)

    # For each scenario, calculate the reliability and store the data in a df
    for scenario in scenario_list:
        scenario_files = glob.glob(sim_root_path + "/{}_*.csv".format(scenario)) # Get list of csv files belonging to this scenario
        # print(scenario_files)
        scenario_params = scenario.split('_')
        num_member = int(scenario_params[0].split('-')[-1])
        inter_uav_distance = int(scenario_params[1].split('-')[-1])
        height = int(scenario_params[2].split('-')[-1])
        swarm_hor_distance = int(scenario_params[3].split('-')[-1]) # Horizontal Swarm Distance
        swarm_distance = math.sqrt(int(height)**2 + swarm_hor_distance**2)
        packet_size = int(scenario_params[4].split('-')[-1])
        sending_rate = int(scenario_params[5].split('-')[-1])
        dl_data, ul_data = process_micro_sim(scenario_files, delay_th)
        if dl_data is not None:
            dl_data["Height"] = height
            dl_data["Swarm_Distance"] = swarm_distance
            dl_data["Horizontal_Distance"] = swarm_hor_distance
            dl_data["Inter_UAV_Distance"] = inter_uav_distance
            dl_data["Num_Members"] = num_member
            dl_data["Packet_Size"] = packet_size
            dl_data["Sending_Rate"] = sending_rate
            dl_df = pd.concat([dl_df, pd.DataFrame.from_records([dl_data])], ignore_index=True)
        if ul_data is not None:
            ul_data["Height"] = height
            ul_data["Swarm_Distance"] = swarm_distance
            ul_data["Horizontal_Distance"] = swarm_hor_distance
            ul_data["Inter_UAV_Distance"] = inter_uav_distance
            ul_data["Num_Members"] = num_member
            ul_data["Packet_Size"] = packet_size
            ul_data["Sending_Rate"] = sending_rate
            ul_df = pd.concat([ul_df, pd.DataFrame.from_records([ul_data])], ignore_index=True)
    
    return dl_df, ul_df

def correlation_study(dl_df, base_scenario_df):
    # Correlation with horizontal distance
    h_dist_df = dl_df.loc[(dl_df['Num_Members'] == 3) & (dl_df['Inter_UAV_Distance'] == 4) & (dl_df['Horizontal_Distance'] != 200) & (dl_df['Height'] == 100) & (dl_df['Packet_Size'] == 100) & (dl_df['Sending_Rate'] == 500)]
    h_dist_df = pd.concat([base_scenario_df, h_dist_df])
    h_dist = h_dist_df["Horizontal_Distance"].to_numpy()
    h_dist_reliability = h_dist_df["Reliability"].to_numpy()
    h_dist_MI = mutual_info_regression(h_dist.reshape((-1,1)), h_dist_reliability.reshape((-1,1)))
    h_dist_tau, h_dist_p_value = stats.kendalltau(h_dist, h_dist_reliability)
    h_dist_rho, h_dist_pval = stats.spearmanr(h_dist_reliability, h_dist)
    print("Correlation of h_dist - MI: {}, KendallTau: {}, SpearmanRho: {}".format(h_dist_MI, h_dist_tau, h_dist_rho))

    # Correlation with height
    height_df = dl_df.loc[(dl_df['Num_Members'] == 3) & (dl_df['Inter_UAV_Distance'] == 4) & (dl_df['Horizontal_Distance'] == 200) & (dl_df['Height'] != 100) & (dl_df['Packet_Size'] == 100) & (dl_df['Sending_Rate'] == 500)]
    height_df = pd.concat([base_scenario_df, height_df])
    height = height_df["Horizontal_Distance"].to_numpy()
    height_reliability = height_df["Reliability"].to_numpy()
    height_MI = mutual_info_regression(height.reshape((-1,1)), height_reliability.reshape((-1,1)))
    height_tau, height_p_value = stats.kendalltau(height, height_reliability)
    height_rho, height_pval = stats.spearmanr(height_reliability, height)
    print("Correlation of height - MI: {}, KendallTau: {}, SpearmanRho: {}".format(height_MI, height_tau, height_rho))

    # Correlation with inter-UAV distance
    iu_dist_df = dl_df.loc[(dl_df['Num_Members'] == 3) & (dl_df['Inter_UAV_Distance'] != 4) & (dl_df['Horizontal_Distance'] == 200) & (dl_df['Height'] == 100) & (dl_df['Packet_Size'] == 100) & (dl_df['Sending_Rate'] == 500)]
    iu_dist_df = pd.concat([base_scenario_df, iu_dist_df])
    iu_dist = iu_dist_df["Horizontal_Distance"].to_numpy()
    iu_dist_reliability = iu_dist_df["Reliability"].to_numpy()
    iu_dist_MI = mutual_info_regression(iu_dist.reshape((-1,1)), iu_dist_reliability.reshape((-1,1)))
    iu_dist_tau, iu_dist_p_value = stats.kendalltau(iu_dist, iu_dist_reliability)
    iu_dist_rho, iu_dist_pval = stats.spearmanr(iu_dist_reliability, iu_dist)
    print("Correlation of inter-UAV distance - MI: {}, KendallTau: {}, SpearmanRho: {}".format(iu_dist_MI, iu_dist_tau, iu_dist_rho))

    # Correlation with packet size
    ps_df = dl_df.loc[(dl_df['Num_Members'] == 3) & (dl_df['Inter_UAV_Distance'] == 4) & (dl_df['Horizontal_Distance'] == 200) & (dl_df['Height'] == 100) & (dl_df['Packet_Size'] != 100) & (dl_df['Sending_Rate'] == 500)]
    ps_df = pd.concat([base_scenario_df, ps_df])
    ps = ps_df["Horizontal_Distance"].to_numpy()
    ps_reliability = ps_df["Reliability"].to_numpy()
    ps_MI = mutual_info_regression(ps.reshape((-1,1)), ps_reliability.reshape((-1,1)))
    ps_tau, ps_p_value = stats.kendalltau(ps, ps_reliability)
    ps_rho, ps_pval = stats.spearmanr(ps_reliability, ps)
    print("Correlation of packet size - MI: {}, KendallTau: {}, SpearmanRho: {}".format(ps_MI, ps_tau, ps_rho))

    # Correlation with sending rate
    sr_df = dl_df.loc[(dl_df['Num_Members'] == 3) & (dl_df['Inter_UAV_Distance'] == 4) & (dl_df['Horizontal_Distance'] == 200) & (dl_df['Height'] == 100) & (dl_df['Packet_Size'] == 100) & (dl_df['Sending_Rate'] != 500)]
    sr_df = pd.concat([base_scenario_df, sr_df])
    sr = sr_df["Horizontal_Distance"].to_numpy()
    sr_reliability = sr_df["Reliability"].to_numpy()
    sr_MI = mutual_info_regression(sr.reshape((-1,1)), sr_reliability.reshape((-1,1)))
    sr_tau, sr_p_value = stats.kendalltau(sr, sr_reliability)
    sr_rho, sr_pval = stats.spearmanr(sr_reliability, sr)
    print("Correlation of sending rate - MI: {}, KendallTau: {}, SpearmanRho: {}".format(sr_MI, sr_tau, sr_rho))
    print(sr_df)

    # # Correlation with number of UAV members
    # num_uav_df = dl_df.loc[(dl_df['Num_Members'] != 3) & (dl_df['Inter_UAV_Distance'] == 4) & (dl_df['Horizontal_Distance'] == 200) & (dl_df['Height'] == 100) & (dl_df['Packet_Size'] == 100) & (dl_df['Sending_Rate'] == 500)]
    # num_uav_df = pd.concat([base_scenario_df, num_uav_df])
    # num_uav = num_uav_df["Horizontal_Distance"].to_numpy()
    # num_uav_reliability = num_uav_df["Reliability"].to_numpy()
    # num_uav_MI = mutual_info_regression(num_uav.reshape((-1,1)), num_uav_reliability.reshape((-1,1)))
    # num_uav_tau, num_uav_p_value = stats.kendalltau(num_uav, num_uav_reliability)
    # num_uav_rho, num_uav_pval = stats.spearmanr(num_uav_reliability, num_uav)
    # print("Correlation of packet size - MI: {}, KendallTau: {}, SpearmanRho: {}".format(num_uav_MI, num_uav_tau, num_uav_rho))

if __name__ == "__main__":
    # sim_root_path = "/home/reuben/omnetpp_sim_results/FANET_Corr"
    sim_root_path = "/home/reuben/omnetpp_sim_results/FANET_Corr2"
    # dl_df, ul_df = process_reliability(sim_root_path)
    # dl_df.to_csv(os.path.join(sim_root_path,"FANET_downlink.csv"), index=False)
    # ul_df.to_csv(os.path.join(sim_root_path,"FANET_uplink.csv"), index=False)

    # Analysing downlink correlation
    dl_df = pd.read_csv(os.path.join(sim_root_path,"FANET_downlink.csv"))
    base_scenario_df = dl_df.loc[(dl_df['Num_Members'] == 3) & (dl_df['Inter_UAV_Distance'] == 4) & (dl_df['Horizontal_Distance'] == 200) & (dl_df['Height'] == 100) & (dl_df['Packet_Size'] == 100) & (dl_df['Sending_Rate'] == 500)]
    correlation_study(dl_df, base_scenario_df)
    

