# Date: 27/1/2023
# Modified preprocessing script for OMNeT++ simulation data

import pandas as pd # for data manipulation 
import numpy as np
import matplotlib.pyplot as plt # for drawing graphs
import os, sys, glob, math
import time

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

def compile_micro_sim_data(file_list):
    '''
    Function to compile data from the CSV files generated by each micro-simulation
    Update: To specifically return the rx_df, tx_df, mon_df and pd_df in lists, so that specific dfs can be accessed (instead of aggregating UAV dfs)
    Input: file_list - List of simulation files belonging to a certain scenario (micro-sim)
    Output: concatenates the raw data to UL and DL dataframes
    '''

    # Let's get the GCS dfs ===============================================================
    gcs_rx_file = [file for file in file_list if (('_GCS-' in file) and ('-Rx' in file))]
    gcs_tx_file = [file for file in file_list if (('_GCS-' in file) and ('-Tx' in file))]
    gcs_mon_file = [file for file in file_list if (('_GCS-' in file) and ('Wlan' in file))]
    gcs_pd_file = [file for file in file_list if (('_GCS-' in file) and ('PacketDrop' in file))]
    if len(gcs_rx_file) > 0:
        gcs_rx_df = pd.read_csv(gcs_rx_file[0])
    else:
        print("GCS RX File Missing")
        print(file_list[0])
    if len(gcs_tx_file) > 0:
        gcs_tx_df = pd.read_csv(gcs_tx_file[0])
    else:
        print("GCS TX File Missing")
        print(file_list[0])
    if len(gcs_pd_file) > 0:
        gcs_pd_df = pd.read_csv(gcs_pd_file[0])
    else:
        print("GCS PD File Missing")
        print(file_list[0])
    if len(gcs_mon_file) > 0:
        gcs_mon_df = pd.read_csv(gcs_mon_file[0]) # Mon file is optional
        gcs_mon_df["Addr"] = "192.168.0.1"
    else: 
        gcs_mon_df = None

    # Let's get the GW dfs ===============================================================
    gw_rx_file = [file for file in file_list if (('_GW-' in file) and ('-Rx' in file))]
    gw_tx_file = [file for file in file_list if (('_GW-' in file) and ('-Tx' in file))]
    gw_mon_file = [file for file in file_list if (('_GW-' in file) and ('Wlan' in file))]
    gw_pd_file = [file for file in file_list if (('_GW-' in file) and ('PacketDrop' in file))]
    if len(gw_rx_file) > 0:
        gw_rx_df = pd.read_csv(gw_rx_file[0])
    else:
        print("GW RX File Missing")
        print(file_list[0])
    if len(gw_tx_file) > 0:
        gw_tx_df = pd.read_csv(gw_tx_file[0])
    else:
        print("GW TX File Missing")
        print(file_list[0])
    if len(gw_pd_file) > 0:
        gw_pd_df = pd.read_csv(gw_pd_file[0])
    else:
        print("GW PD File Missing")
        print(file_list[0])
    if len(gw_mon_file) > 0:
        gw_mon_df = pd.read_csv(gw_mon_file[0]) # Mon file is optional
        gw_mon_df["Addr"] = "192.168.0.2"
    else:
        gw_mon_df = None

    # Let's get the UAVs dfs ===============================================================
    uavs_rx_df_list = []
    uavs_tx_df_list = []
    uavs_mon_df_list = []
    uavs_pd_df_list = []
    uav_rx_files = [file for file in file_list if (('_UAV-' in file) and ('-Rx' in file))]
    uav_tx_files = [file for file in file_list if (('_UAV-' in file) and ('-Tx' in file))]
    uav_mon_files = [file for file in file_list if (('_UAV-' in file) and ('Wlan' in file))]
    uav_pd_files = [file for file in file_list if (('_UAV-' in file) and ('PacketDrop' in file))]
    uav_rx_files.sort()
    uav_tx_files.sort()
    uav_mon_files.sort()
    uav_pd_files.sort()
    if len(uav_rx_files) > 0:
        for uav_rx_file in uav_rx_files:
            uavs_rx_df_list.append(pd.read_csv(uav_rx_file))
    else:
        print("UAV RX File(s) Missing")
        print(file_list[0])
    if len(uav_tx_files) > 0:
        for uav_tx_file in uav_tx_files:
            uavs_tx_df_list.append(pd.read_csv(uav_tx_file))
    else:
        print("UAV TX File(s) Missing")
        print(file_list[0])
    if len(uav_pd_files) > 0:
        for uav_pd_file in uav_pd_files:
            uavs_pd_df_list.append(pd.read_csv(uav_pd_file))
    else:
        print("UAV PD File(s) Missing")
        print(file_list[0])
    if len(uav_mon_files) > 0: # UAV mon files are optional now
        uav_member_index = 3
        for uav_mon_file in uav_mon_files:
            uav_mon_df = pd.read_csv(uav_mon_file)
            uav_mon_df["Addr"] = "192.168.0." + str(uav_member_index)
            uavs_mon_df_list.append(uav_mon_df)
            uav_member_index += 1
    else:
        uavs_mon_df_list = []

    rx_df_list = [gcs_rx_df, gw_rx_df] + uavs_rx_df_list
    tx_df_list = [gcs_tx_df, gw_tx_df] + uavs_tx_df_list
    pd_df_list = [gcs_pd_df, gw_pd_df] + uavs_pd_df_list
    mon_df_list = [gcs_mon_df, gw_mon_df] + uavs_mon_df_list

    # UNCOMMENT BELOW IF RSSI DATA WILL BE USED
    for rx_df in rx_df_list:
        rx_df["RSSI"] = rssi_to_np(rx_df["RSSI"])
    for pd_df in pd_df_list:
        pd_df["RSSI"] = rssi_to_np(pd_df["RSSI"])
    for mon_df in mon_df_list:
        if mon_df is not None:
            mon_df["RSSI"] = rssi_to_np(mon_df["RSSI"])

    return rx_df_list, tx_df_list, pd_df_list, mon_df_list

def process_dropped_packets(tx_df, rx_df_list, pd_df_list, tx_index, delay_threshold, sending_interval=40, NP=10000):
    '''
    This function is to compile packet information from the tx, rx and pd dataframes, fopr downlink comm. (GCS to UAVs)
    tx_df: Tx DF 
    rx_df_list: List of Rx DFs, first one is for GCS, second for GW, subsequent DFs in the list for UAV 1, 2, ...
    pd_df_list: List of packet drop DFs, first one is for GCS, second for GW, subsequent DFs in the list for UAV 1, 2, ...
    tx_index: Index of Tx in pd_df_list (e.g. for GCS, tx_index = 0)
    delay_threshold: Delay threshold to consider if packet arrived too late
    sending_interval: Mean sending interval (used to determine UAV speed in the simulation)
    NP: Number of packets set for every 100m
    Output: pkt_df: DF containing info on packets from tx_df received and dropped 
    '''
    uav_speed = 100 * 1000 / NP / sending_interval # This is for estimating the U2G Distance when queue overflow happens (refer Omnet ini file)
    pkt_df = pd.DataFrame(columns = ['RxTime','TxTime','Packet_Name','Bytes','RSSI','U2G_SINR','U2U_SINR','U2G_BER','U2U_BER',
                                    'Hop_Count','Delay','Queueing_Time','Backoff_Time','U2G_Distance',
                                    'Incorrectly_Rcvd','Queue_Overflow','Interface_Down','Number_Dropped','Packet_State'])
    for index, row in tx_df.iterrows():
        packetName = row["Packet_Name"] + "-" + str(row["Packet_Seq"])
        dest_addr = row["Dest_Addr"]
        rx_index = int(dest_addr.split(".")[-1]) - 1
        rx_df = rx_df_list[rx_index]

        # For each packet in gcs_tx_df, get the packet drops from GW and corresponding UAV
        pkt_drops_tx = pd_df_list[tx_index].loc[(pd_df_list[tx_index]["Packet_Name"] == packetName)] # Packets dropped at the transmitter, to catch QUEUE_OVERFLOW and INTERFACE_DOWN
        pkt_drops_gw = pd_df_list[1].loc[(pd_df_list[1]["Packet_Name"] == packetName)] # Packets dropped at the gateway UAV
        if rx_index != 1: # If not the GW, include packet drops at receiver. Else no need, cos GW is Rx
            pkt_drops_rx = pd_df_list[rx_index].loc[(pd_df_list[rx_index]["Packet_Name"] == packetName)] # Packets dropped at the receiver (GCS / UAV)
            pkt_drops = pd.concat([pkt_drops_tx, pkt_drops_gw, pkt_drops_rx], ignore_index = True)
        else:
            pkt_drops = pd.concat([pkt_drops_tx, pkt_drops_gw], ignore_index = True)

        if not pkt_drops.empty: # Find the packet drops for this particular packet
            drop_reasons = pkt_drops["Packet_Drop_Reason"].values # List of pkt drop reasons at GW and Rx and Tx
            # Count the occurences of each failure modes for a particular packet
            incorrect_rcvd = np.count_nonzero(drop_reasons == "INCORRECTLY_RECEIVED")
            queue_overflow = np.count_nonzero(drop_reasons == "QUEUE_OVERFLOW")
            retry_limit_excd = np.count_nonzero(drop_reasons == "RETRY_LIMIT_REACHED")
            interface_down = np.count_nonzero(drop_reasons == "INTERFACE_DOWN")
            num_drops = len(drop_reasons) # This is for counting drops due to incorrectly received only

            # Update pkt_df 
            if (packetName not in rx_df["Packet_Name"].values):
                # If not received, add the data of failed packet
                rx_time = max(pkt_drops["RxTime"].values)
                tx_time = min(pkt_drops["TxTime"].values)
                bytes = row["Bytes"]
                rssi = pkt_drops["RSSI"].mean() # This should be taking the max RSSI, but since it is not used, leaving it as mean for now
                u2g_sinr = max(pkt_drops["U2G_SINR"].values)
                u2g_ber = max(pkt_drops["U2G_BER"].values)
                delay = max(pkt_drops["Delay"].values)
                queueing_time = max(pkt_drops["Queueing_Time"].values)
                backoff_time = max(pkt_drops["Backoff_Time"].values)
                u2g_distance = max(pkt_drops["U2G_Distance"].values)
                # Packet State Based on Failure Mode
                if retry_limit_excd > 0:
                    pkt_state = "RETRY_LIMIT_REACHED" # The packet failed to be received (RETRY_LIMIT_EXCEEDED)
                    # If packet was dropped due to retry limit reach at the GW, then there may not be any U2G distance recorded. But knowing the speed, we can compute it
                    if math.isnan(u2g_distance):
                        u2g_distance = uav_speed * rx_time
                elif queue_overflow > 0:
                    pkt_state = "QUEUE_OVERFLOW" # The packet failed due to queue buffer overflow
                    # If packet was dropped due to queue overflow, then there will not be any U2G distance recorded. But knowing the speed, we can compute it
                    if math.isnan(u2g_distance):
                        u2g_distance = uav_speed * rx_time
                elif interface_down > 0:
                    pkt_state = "INTERFACE_DOWN" # The packet failed due to interface down
                else:
                    pkt_state = "FAILED" # Unaccounted fail reason
                    print("Packet Failure Mode Unknown")
                # Check for U2U Data
                if (len(pkt_drops["U2U_SINR"].values) > 0): # There may not always be a U2U communication
                    u2u_sinr = max(pkt_drops["U2U_SINR"].values)
                    u2u_ber = max(pkt_drops["U2U_BER"].values)
                    hop_count = 2
                else:
                    u2u_sinr = None
                    u2u_ber = None
                    hop_count = 1

                failed_pkt = pd.DataFrame([{'RxTime': rx_time,'TxTime': tx_time,'Packet_Name': packetName,'Bytes': bytes,'RSSI': rssi,'U2G_SINR': u2g_sinr,'U2U_SINR': u2u_sinr,
                              'U2G_BER': u2g_ber,'U2U_BER': u2u_ber,'Hop_Count': hop_count,'Delay': delay,'Queueing_Time': queueing_time,'Backoff_Time': backoff_time,'U2G_Distance': u2g_distance,
                              'Incorrectly_Rcvd': incorrect_rcvd,'Queue_Overflow': queue_overflow,'Interface_Down': interface_down,'Number_Dropped': num_drops,'Packet_State': pkt_state}])
                pkt_df = pd.concat([pkt_df,failed_pkt], ignore_index = True)

            else:
                # If packet successfully received, update the number of tries and the reason for failed attempt(s) to the received packet info
                rcvd_pkt_df = rx_df.loc[(rx_df["Packet_Name"] == packetName)].copy()
                rcvd_pkt_df["Incorrectly_Rcvd"] = incorrect_rcvd
                rcvd_pkt_df["Queue_Overflow"] = queue_overflow
                rcvd_pkt_df["Interface_Down"] = interface_down
                rcvd_pkt_df["Number_Dropped"] = num_drops
                if rcvd_pkt_df["Delay"].values > delay_threshold:
                    rcvd_pkt_df["Packet_State"] = "DELAY_EXCEEDED"
                else:
                    rcvd_pkt_df["Packet_State"] = "RELIABLE"
                pkt_df = pd.concat([pkt_df,rcvd_pkt_df], ignore_index = True)

        elif (packetName in rx_df["Packet_Name"].values):
            # The packet was received without any retries
            rcvd_pkt_df = rx_df.loc[(rx_df["Packet_Name"] == packetName)].copy()
            rcvd_pkt_df["Incorrectly_Rcvd"] = 0
            rcvd_pkt_df["Queue_Overflow"] = 0
            rcvd_pkt_df["Interface_Down"] = 0
            rcvd_pkt_df["Number_Dropped"] = 0
            if rcvd_pkt_df["Delay"].values > delay_threshold:
                rcvd_pkt_df["Packet_State"] = "DELAY_EXCEEDED"
            else:
                rcvd_pkt_df["Packet_State"] = "RELIABLE"
            pkt_df = pd.concat([pkt_df,rcvd_pkt_df], ignore_index = True)
        
        # else:
        #     print("No packet drop recorded and packet not found in rx_df for packet: {}. This should not happen".format(packetName))

    pkt_df = pkt_df.sort_values("RxTime")
    pkt_df = pkt_df.reset_index()
    return pkt_df

def process_throughput(df, timeDiv):
    '''
    Function to calculate throughput data for a DataFrame
    timeDiv is the time division to use for calculating the throughput
    '''
    maxTime = math.ceil(float(df["RxTime"].max()))
    for i in range(math.ceil(maxTime / timeDiv)):
        df_in_range = df.loc[(df["RxTime"] >= (i*timeDiv)) & (df["RxTime"] < ((i+1)*timeDiv)) & (df["Packet_State"] == "RECEIVED")]
        totalBytes = df_in_range["Bytes"].sum()
        throughput = totalBytes / timeDiv
        df.loc[(df["RxTime"] >= (i*timeDiv)) & (df["RxTime"] < ((i+1)*timeDiv)), "Throughput"] = throughput
    return df

def process_sim_data(sim_root_path, delay_threshold, NP):
    # Concatenates all UL & DL results from sim_root_path into a single df
    scenario_list = [csv.split('/')[-1][0:-11] for csv in glob.glob(sim_root_path + "/*GCS-Tx.csv")] # Get list of "unique" scenarios

    # Dataframes to store UL & DL raw data
    dl_df = pd.DataFrame(columns = ['RxTime','TxTime','Packet_Name','Bytes','RSSI','U2G_SINR','U2U_SINR','U2G_BER','U2U_BER','Hop_Count','Throughput',
                                    'Delay','Queueing_Time','Backoff_Time','U2G_Distance','Height','Inter_UAV_Distance','Num_Members','Sending_Interval',
                                    'Incorrectly_Rcvd','Queue_Overflow','Interface_Down','Number_Dropped','Packet_State']) # Downlink dataframe
    ul_df = pd.DataFrame(columns = ['RxTime','TxTime','Packet_Name','Bytes','RSSI','U2G_SINR','U2U_SINR','U2G_BER','U2U_BER','Hop_Count','Throughput',
                                    'Delay','Queueing_Time','Backoff_Time','U2G_Distance','Height','Inter_UAV_Distance','Num_Members','Sending_Interval',
                                    'Incorrectly_Rcvd','Queue_Overflow','Interface_Down','Number_Dropped','Packet_State']) # Downlink dataframe

    # For each scenario, extract the UL and DL raw data
    # NP = 10000 # The number of packets set in the simulation for each 100m (refer to OMNeT++ ini sim file)
    # NP = input("Enter number of packets set in the simulation for each 100m (refer to OMNeT++ ini sim file)")
    for scenario in scenario_list:
        scenario_files = glob.glob(sim_root_path + "/{}_*.csv".format(scenario)) # Get list of csv files belonging to this scenario
        scenario_params = scenario.split('_')
        num_member = int(scenario_params[0].split('-')[-1])
        inter_uav_distance = int(scenario_params[1].split('-')[-1])
        height = int(scenario_params[2].split('-')[-1]) 
        sending_interval = int(scenario_params[5].split('-')[-1])
        rx_df_list, tx_df_list, pd_df_list, mon_df_list = compile_micro_sim_data(scenario_files)
        
        # Process the state of each packets sent in DL
        start_dl_time = time.time()
        dl_data = process_dropped_packets(tx_df_list[0], rx_df_list, pd_df_list, 0, delay_threshold, sending_interval, NP)
        if dl_data is not None:
            dl_data["Height"] = height
            dl_data["Inter_UAV_Distance"] = inter_uav_distance
            dl_data["Num_Members"] = num_member
            dl_data["Sending_Interval"] = sending_interval
            dl_data = process_throughput(dl_data, 1)
            dl_df = pd.concat([dl_df, dl_data], ignore_index=True)

        # Process the state of each packets sent in DL
        start_ul_time = time.time()
        for i in range(1,len(rx_df_list)):
            ul_data = process_dropped_packets(tx_df_list[i], rx_df_list, pd_df_list, i, delay_threshold, sending_interval, NP)
            if ul_data is not None:
                ul_data["Height"] = height
                ul_data["Inter_UAV_Distance"] = inter_uav_distance
                ul_data["Num_Members"] = num_member
                ul_data["Sending_Interval"] = sending_interval
                ul_data = process_throughput(ul_data, 1)
                ul_df = pd.concat([ul_df, ul_data], ignore_index=True)
        
        end_time = time.time()
        print("DL Time: {}".format(start_ul_time-start_dl_time))
        print("UL Time: {}".format(end_time-start_ul_time))
    
    return dl_df, ul_df

if __name__ == "__main__":
    # Let's get the data
    sim_root_path = "/home/research-student/omnetpp_sim_results/Dataset_NP10000_BPSK_6-5Mbps"
    delay_threshold = 0.04
    NP = 10000 # Number of packets set in the simulation for each 100m (refer to OMNeT++ ini sim file)
    dl_df, ul_df = process_sim_data(sim_root_path, delay_threshold=delay_threshold, NP=NP)
    # Save DF to CSV
    dl_df.to_csv(os.path.join(sim_root_path,"FANET_downlink_raw.csv"), index=False)
    ul_df.to_csv(os.path.join(sim_root_path,"FANET_uplink_raw.csv"), index=False)