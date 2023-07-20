'''
Date: 15/04/2023
Desc: Pandas data processing of FANET dataset with multiprocessing.
      V2 - Uses categorical data type for packet name
      Modified to process for 'hovering' dataset
      Modified for new traffic model
'''

import pandas as pd # for data manipulation 
import numpy as np
# import matplotlib.pyplot as plt # for drawing graphs
import os, sys, glob, math
import time
from multiprocessing.pool import Pool
from itertools import repeat
from tqdm import tqdm
from scipy import special

def h_dist_calc(row):
    # Function to calc euclidean distance on every df row 
    h_dist = math.sqrt(row["U2G_Distance"]**2 - row["Height"]**2)
    return h_dist

def q_func(x):
    q = 0.5 - 0.5*special.erf(x / np.sqrt(2))
    return q

def friis_calc(P,freq,dist,ple):
    '''
    Friis path loss equation
    P = Tx transmit power
    freq = Signal frequency
    dist = Transmission distance
    ple = Path loss exponent
    '''
    propagation_speed = 299792458
    l = propagation_speed / freq
    h_pl = P * l**2 / (16*math.pi**2)
    P_Rx = h_pl * dist**(-ple)
    return P_Rx

def plos_calc(h_dist, height_tx, height_rx, env='suburban'):
    '''
    % This function implements the LoS probability model from the paper
    % "Blockage Modeling for Inter-layer UAVs Communications in Urban
    % Environments" 
    % param h_dist    : horizontal distance between Tx and Rx (m)
    % param height_tx : height of Tx
    % param height_rx : height of Rx
    '''
    if env == 'suburban':
        a1 = 0.1
        a2 = 7.5e-4
        a3 = 8
    
    delta_h = height_tx - height_rx
    # pow_factor = 2 * h_dist * math.sqrt(a1*a2/math.pi) + a1 # NOTE: Use this pow_factor if assuming PPP building dist.
    pow_factor = h_dist * math.sqrt(a1*a2) # NOTE: Use this pow_factor if assuming ITU-R assumptions.
    if delta_h == 0:
        p = (1 - math.exp((-(height_tx)**2) / (2*a3**2))) ** pow_factor
    else:
        delta_h = abs(delta_h)
        p = (1 - (math.sqrt(2*math.pi)*a3 / delta_h) * abs(q_func(height_tx/a3) - q_func(height_rx/a3))) ** pow_factor
    return p

def sinr_lognormal_approx(h_dist, height, env='suburban'):
    '''
    To approximate the SNR from signal considering multipath fading and shadowing
    Assuming no interference due to CSMA, and fixed noise
    Inputs:
    h_dist = Horizontal Distance between Tx and Rx
    height = Height difference between Tx and Rx
    env = The operating environment (currently only suburban supported)
    '''
    # Signal properties
    P_Tx_dBm = 20 # Transmit power of 
    P_Tx = 10**(P_Tx_dBm/10) / 1000
    freq = 2.4e9 # Channel frequency (Hz)
    noise_dBm = -86
    noise = 10**(noise_dBm/10) / 1000
    if env == "suburban":
        # ENV Parameters Constants ----------------------------------
        # n_min = 2
        # n_max = 2.75
        # K_dB_min = 7.8
        # K_dB_max = 17.5
        # K_min = 10**(K_dB_min/10)
        # K_max = 10**(K_dB_max/10)
        # alpha = 11.25 # Env parameters for logarithm std dev of shadowing 
        # beta = 0.06 # Env parameters for logarithm std dev of shadowing 
        n_min = 2
        n_max = 2.75
        K_dB_min = 1.4922
        K_dB_max = 12.2272
        K_min = 10**(K_dB_min/10)
        K_max = 10**(K_dB_max/10)
        alpha = 11.1852 # Env parameters for logarithm std dev of shadowing 
        beta = 0.06 # Env parameters for logarithm std dev of shadowing 
        # -----------------------------------------------------------
    # Calculate fading parameters
    PLoS = plos_calc(h_dist, 0, height, env='suburban')
    theta_Rx = math.atan2(height, h_dist) * 180 / math.pi # Elevation angle in degrees
    ple = (n_min - n_max) * PLoS + n_max # Path loss exponent
    sigma_phi_dB = alpha*math.exp(-beta*theta_Rx)
    sigma_phi = 10**(sigma_phi_dB/10) # Logarithmic std dev of shadowing
    K = K_min * math.exp(math.log(K_max/K_min) * PLoS**2)
    omega = 1 # Omega of NCS (Rician)
    dist = math.sqrt(h_dist**2 + height**2)
    P_Rx = friis_calc(P_Tx, freq, dist, ple)
    # Approximate L-NCS RV (which is the SNR) as lognormal
    eta = math.log(10) / 10
    mu_phi = 10*math.log10(P_Rx)
    E_phi = math.exp(eta*mu_phi + eta**2*sigma_phi**2/2) # Mean of shadowing RV
    var_phi = math.exp(2*eta*mu_phi+eta**2*sigma_phi**2)*(math.exp(eta**2*sigma_phi**2)-1) # Variance of shadowing RV
    E_chi = (special.gamma(1+1)/(1+K))*special.hyp1f1(-1,1,-K)*omega
    var_chi = (special.gamma(1+2)/(1+K)**2)*special.hyp1f1(-2,1,-K)*omega**2 - E_chi**2
    E_SNR = E_phi * E_chi / noise # Theoretical mean of SINR
    var_SNR = ((var_phi+E_phi**2)*(var_chi+E_chi**2) - E_phi**2 * E_chi**2) / noise**2
    std_dev_SNR = math.sqrt(var_SNR)
    # sigma_ln = math.sqrt(math.log(var_SNR/E_SNR**2 + 1))
    # mu_ln = math.log(E_SNR) - sigma_ln**2/2
    return E_SNR, std_dev_SNR

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
    return rssi_num.astype(np.float32)

def compile_micro_sim_data_v2(file_list):
    '''
    Function to compile data from the CSV files generated by each micro-simulation
    Update: To specifically return the rx_df, tx_df, mon_df and pd_df in lists, so that specific dfs can be accessed (instead of aggregating UAV dfs)
    Input: file_list - List of simulation files belonging to a certain scenario (micro-sim)
    Output: concatenates the raw data to UL and DL dataframes
    '''
    # Specify dtypes to save memory
    tx_df_dtypes = {"TxTime": np.float32, "Packet_Name": "category", "Packet_Seq": np.uint32, "Bytes": np.uint16, "Dest_Addr": 'category'}
    rx_df_dtypes = {"RxTime": np.float64, "TxTime": np.float32,	"Packet_Name": "str", "Bytes": np.uint16, "RSSI": 'str', "U2G_SINR": np.float32, "U2U_SINR": np.float32, 
                    "U2G_BER": np.float32, "U2U_BER": np.float32, "Src_Addr": 'category', "Dest_Addr": 'category', "Hop_Count": np.uint8, "Delay": np.float32, 
                    "Queueing_Time": np.float32, "Backoff_Time": np.float32, "U2G_Distance": np.float16, "Retry_Count": np.uint8}
    pd_df_dtypes = {"RxTime": np.float64, "TxTime": np.float32,	"Packet_Name": "str", "Bytes": np.uint16, "RSSI": 'str', "U2G_SINR": np.float32, "U2U_SINR": np.float32, 
                    "U2G_BER": np.float32, "U2U_BER": np.float32, "Delay": np.float32, "Queueing_Time": np.float32, "Backoff_Time": np.float32, "U2G_Distance": np.float16,
                    "Has_Bit_Error": np.uint8, "Packet_Drop_Reason": 'category'}

    # Let's get the GCS dfs ===============================================================
    gcs_rx_file = [file for file in file_list if (('_GCS-' in file) and ('-Rx' in file))]
    gcs_tx_file = [file for file in file_list if (('_GCS-' in file) and ('-Tx' in file))]
    gcs_mon_file = [file for file in file_list if (('_GCS-' in file) and ('Wlan' in file))]
    gcs_pd_file = [file for file in file_list if (('_GCS-' in file) and ('PacketDrop' in file))]
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
    if len(gcs_pd_file) > 0:
        gcs_pd_df = pd.read_csv(gcs_pd_file[0], dtype=pd_df_dtypes)
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
        gw_rx_df = pd.read_csv(gw_rx_file[0], dtype=rx_df_dtypes)
    else:
        print("GW RX File Missing")
        print(file_list[0])
    if len(gw_tx_file) > 0:
        gw_tx_df = pd.read_csv(gw_tx_file[0], dtype=tx_df_dtypes)
    else:
        print("GW TX File Missing")
        print(file_list[0])
    if len(gw_pd_file) > 0:
        gw_pd_df = pd.read_csv(gw_pd_file[0], dtype=pd_df_dtypes)
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
    if len(uav_pd_files) > 0:
        for uav_pd_file in uav_pd_files:
            uavs_pd_df_list.append(pd.read_csv(uav_pd_file, dtype=pd_df_dtypes))
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

def process_received_packets_v2(rx_df, gw_pd_df, delay_threshold):
    """
    This function is to count the number of packets dropped due to queue overflow at the GW. 
    NOTE: ASSUMPTION - Queue overflow observed to only happen at GW
    The number of incorrectly received packets is the number of retries minus the number of packets dropped due to queue overflow
    This function also updates the state of received packets, whether "Reliable" or "Delay_Exceeded"
    rx_df: Rx DF
    gw_pd_df: Packet drop DF from Gateway
    """
    if "Unnamed: 16" in rx_df.columns:
        rx_df = rx_df.drop(["Unnamed: 16"], axis=1)

    rcvd_pkt_seq = rx_df["Packet_Seq"].values
    rcvd_pkt_type = rx_df["Packet_Type"].values
    retry_counts = rx_df["Retry_Count"].values
    incorrect_rcvd_list = np.zeros((1, len(rcvd_pkt_seq)))
    queue_overflow_list = np.zeros((1, len(rcvd_pkt_seq)))

    for i in tqdm(range(len(rcvd_pkt_seq))):
        retry_count = retry_counts[i]
        if retry_count <= 0: # If no packets dropped, no need to update anything
            pass
        else:
            pkt_drops_gw = gw_pd_df.loc[(gw_pd_df["Packet_Type"] == rcvd_pkt_type[i]) & (gw_pd_df["Packet_Seq"] == rcvd_pkt_seq[i])] # Packets dropped at the gateway UAV
            drop_reasons = pkt_drops_gw["Packet_Drop_Reason"].values # List of pkt drop reasons at GW and Rx and Tx
            queue_overflow = np.count_nonzero(drop_reasons == "QUEUE_OVERFLOW")
            incorrect_rcvd = retry_count - queue_overflow
            queue_overflow_list[0][i] = queue_overflow
            incorrect_rcvd_list[0][i] = incorrect_rcvd

    rx_df["Incorrectly_Received"] = pd.Categorical(incorrect_rcvd_list.tolist()[0])
    rx_df["Queue_Overflow"] = pd.Categorical(queue_overflow_list.tolist()[0])
    rx_df["Packet_State"] = pd.Categorical(np.where(rx_df['Delay'] > delay_threshold , "Delay_Exceeded", "Reliable"))
    return rx_df

def process_received_packets_v3(rx_df, pd_df_list, delay_threshold):
    """
    This function is to count the number of packets dropped due to queue overflow at the GW. 
    NOTE: ASSUMPTION - Queue overflow observed to only happen at GW
    Get the number of incorrectly received packets and queue overflows from the Tx, Rx and GW pd_df
    This function also updates the state of received packets, whether "Reliable" or "Delay_Exceeded"
    rx_df: Rx DF
    pd_df_list: List of packet drop DFs, first one is for GCS, second for GW, subsequent DFs in the list for UAV 1, 2, ...
    """
    
    if "Unnamed: 16" in rx_df.columns:
        rx_df = rx_df.drop(["Unnamed: 16"], axis=1)
    rx_df_dict = rx_df.to_dict('records')
    incorrect_rcvd_list = []
    queue_overflow_list = []
    
    for row in tqdm(rx_df_dict):
        packetType = row["Packet_Type"] # In TX CSV files, Packet_Type is called Packet_Name
        packetSeq = row["Packet_Seq"]
        if packetType == "CNCData": # If transmitter is the GCS, only include pkt dropped at GCS and Rx
            tx_index = 0 # Index 0 for GCS
            dest_addr = row["Dest_Addr"]
            rx_index = int(dest_addr.split(".")[-1]) - 1 # Get UAV Rx Index from Dest_Addr
            pkt_drops_tx = pd_df_list[tx_index].loc[(pd_df_list[tx_index]["Packet_Type"] == packetType) & (pd_df_list[tx_index]["Packet_Seq"] == packetSeq)] # Packets dropped at the transmitter, to catch QUEUE_OVERFLOW and INTERFACE_DOWN
            pkt_drops_rx = pd_df_list[rx_index].loc[(pd_df_list[rx_index]["Packet_Type"] == packetType) & (pd_df_list[rx_index]["Packet_Seq"] == packetSeq)] # Packets dropped at the receiver (GCS / UAV)
            pkt_drops = pd.concat([pkt_drops_tx, pkt_drops_rx], ignore_index = True)
        elif packetType == "GatewayData" or packetType == "GatewayVideo": # If transmitter is the GW, only include pkt dropped at GW and GCS
            tx_index = 1 # Index 1 for GW UAV
            rx_index = 0 # Index 0 for GCS
            pkt_drops_tx = pd_df_list[tx_index].loc[(pd_df_list[tx_index]["Packet_Type"] == packetType) & (pd_df_list[tx_index]["Packet_Seq"] == packetSeq)] # Packets dropped at the transmitter, to catch QUEUE_OVERFLOW and INTERFACE_DOWN
            pkt_drops_rx = pd_df_list[rx_index].loc[(pd_df_list[rx_index]["Packet_Type"] == packetType) & (pd_df_list[rx_index]["Packet_Seq"] == packetSeq)] # Packets dropped at the receiver (GCS / UAV)
            pkt_drops = pd.concat([pkt_drops_tx, pkt_drops_rx], ignore_index = True)
        else: # If the packet is not CNCData or GatewayData, it should be UAVData. Include pkt dropped at Rx, Tx and GW
            tx_index = int(packetType.split("_")[-1]) + 2 # Get UAV Tx Index from packet type
            rx_index = 0 # Index 0 for GCS
            pkt_drops_tx = pd_df_list[tx_index].loc[(pd_df_list[tx_index]["Packet_Type"] == packetType) & (pd_df_list[tx_index]["Packet_Seq"] == packetSeq)] # Packets dropped at the transmitter, to catch QUEUE_OVERFLOW and INTERFACE_DOWN
            pkt_drops_rx = pd_df_list[rx_index].loc[(pd_df_list[rx_index]["Packet_Type"] == packetType) & (pd_df_list[rx_index]["Packet_Seq"] == packetSeq)] # Packets dropped at the receiver (GCS / UAV)
            pkt_drops_gw = pd_df_list[1].loc[(pd_df_list[1]["Packet_Type"] == packetType) & (pd_df_list[1]["Packet_Seq"] == packetSeq)] # Packets dropped at the gateway UAV
            pkt_drops = pd.concat([pkt_drops_tx, pkt_drops_rx, pkt_drops_gw], ignore_index = True)
        # Get the packet drop reasons
        drop_reasons = pkt_drops["Packet_Drop_Reason"].values # List of pkt drop reasons at GW and Rx and Tx
        queue_overflow = np.count_nonzero(drop_reasons == "QUEUE_OVERFLOW")
        incorrect_rcvd = np.count_nonzero(drop_reasons == "INCORRECTLY_RECEIVED")
        queue_overflow_list.append(queue_overflow)
        incorrect_rcvd_list.append(incorrect_rcvd)

    rx_df["Delay"] = rx_df['RxTime']-rx_df['TxTime']
    rx_df["Incorrectly_Received"] = pd.Categorical(incorrect_rcvd_list)
    rx_df["Queue_Overflow"] = pd.Categorical(queue_overflow_list)
    rx_df["Packet_State"] = pd.Categorical(np.where(rx_df['Delay'] > delay_threshold , "Delay_Exceeded", "Reliable"))
    return rx_df

def process_received_packets_v4(rx_df, delay_threshold):
    """
    This function updates the state of received packets, whether "Reliable" or "Delay_Exceeded"
    rx_df: Rx DF
    delay_threshold: Delay threshold for denoting "Delay_Exceeded"
    """
    if "Unnamed: 16" in rx_df.columns:
        rx_df = rx_df.drop(["Unnamed: 16"], axis=1)
    rx_df["Delay"] = rx_df['RxTime']-rx_df['TxTime']
    rx_df["Packet_State"] = pd.Categorical(np.where(rx_df['Delay'] > delay_threshold , "Delay_Exceeded", "Reliable"))
    return rx_df

def process_dropped_packets_v8(tx_df, rx_df, pd_df_list, h_dist, height):
    uav_radius = 5
    '''
    Date Modified: 30/5/2023
    Update: Changed the algo to only process failed packets (found in Tx but missing from Rx), for packets sucessfully received, only update the retry count and the delay exceeded status
    Update: Remove recording of number of incr rcvd and queue overflow packets, since not used
    This function is to compile packet information from the tx, rx and pd dataframes, for downlink comm. (GCS to UAVs)
    tx_df: Tx DF 
    rx_df: Rx DF
    pd_df_list: List of packet drop DFs, first one is for GCS, second for GW, subsequent DFs in the list for UAV 1, 2, ...
    g2g_distance: Distance between gateway UAV and GCS
    Output: rx_df: Modified rx_df containing info on packets from tx_df received and dropped 
    '''
    # First, get the list of packets missing from Rx DF but transmitted in Tx DF
    packets_rcvd = rx_df["Packet_Name"].values
    # packets_rcvd = ["{}-{}".format(type_, seq_) for type_, seq_ in zip(rx_df["Packet_Type"].values, rx_df["Packet_Seq"].values)]
    # tx_df["Packet_Seq"] = tx_df["Packet_Seq"].apply(str)
    tx_df["Packet_Full_Name"] = tx_df["Packet_Name"].astype("str") + "-" + tx_df["Packet_Seq"].astype("str")
    tx_df_failed = tx_df.loc[~(tx_df["Packet_Full_Name"].isin(packets_rcvd))]
    tx_df_failed_dict = tx_df_failed.to_dict('records')
    failed_pkt_list = [] # Using list to store failed packet instead of doing pd.concat for appending every packets. This should be much faster
    # Only iterating through packets that failed to be received
    for row in tqdm(tx_df_failed_dict):
        # Use packet name to find the src of packet (NOTE: This makes the naming of packets important)
        packetName = row["Packet_Full_Name"]
        packetType = row["Packet_Name"] # In TX CSV files, Packet_Type is called Packet_Name
        packetSeq = row["Packet_Seq"]
        if packetType == "CNCData":
            tx_index = 0
        elif packetType == "GatewayData":
            tx_index = 1
        else:
            tx_index = int(packetType.split("_")[-1]) + 2 # If the packet is not CNCData or GatewayData, it should be UAVData
        dest_addr = row["Dest_Addr"]
        # src_addr = "192.168.0." + str(tx_index+1)
        rx_index = int(dest_addr.split(".")[-1]) - 1

        # For each packet in tx_df_failed, get the packet drops from GW and corresponding UAV
        pkt_drops_gw = pd_df_list[1].loc[(pd_df_list[1]["Packet_Type"] == packetType) & (pd_df_list[1]["Packet_Seq"] == packetSeq)] # Packets dropped at the gateway UAV
        if rx_index == 1: # If receiver is the GW, only include pkt dropped at Tx and GW
            pkt_drops_tx = pd_df_list[tx_index].loc[(pd_df_list[tx_index]["Packet_Type"] == packetType) & (pd_df_list[tx_index]["Packet_Seq"] == packetSeq)] # Packets dropped at the transmitter, to catch QUEUE_OVERFLOW and INTERFACE_DOWN
            pkt_drops = pd.concat([pkt_drops_tx, pkt_drops_gw], ignore_index = True)
        elif tx_index == 1: # If transmitter is the GW, only include pkt dropped at GW and Rx
            pkt_drops_rx = pd_df_list[rx_index].loc[(pd_df_list[rx_index]["Packet_Type"] == packetType) & (pd_df_list[rx_index]["Packet_Seq"] == packetSeq)] # Packets dropped at the receiver (GCS / UAV)
            pkt_drops = pd.concat([pkt_drops_rx, pkt_drops_gw], ignore_index = True)
        else:
            pkt_drops_tx = pd_df_list[tx_index].loc[(pd_df_list[tx_index]["Packet_Type"] == packetType) & (pd_df_list[tx_index]["Packet_Seq"] == packetSeq)] # Packets dropped at the transmitter, to catch QUEUE_OVERFLOW and INTERFACE_DOWN
            pkt_drops_rx = pd_df_list[rx_index].loc[(pd_df_list[rx_index]["Packet_Type"] == packetType) & (pd_df_list[rx_index]["Packet_Seq"] == packetSeq)] # Packets dropped at the receiver (GCS / UAV)
            pkt_drops = pd.concat([pkt_drops_tx, pkt_drops_rx, pkt_drops_gw], ignore_index = True)

        if not pkt_drops.empty: # Find the packet drops for this particular packet
            drop_reasons = pkt_drops["Packet_Drop_Reason"].values # List of pkt drop reasons at GW and Rx and Tx
            # Count the occurences of each failure modes for a particular packet
            incorrect_rcvd = np.count_nonzero(drop_reasons == "INCORRECTLY_RECEIVED")
            queue_overflow = np.count_nonzero(drop_reasons == "QUEUE_OVERFLOW")
            retry_limit_excd = np.count_nonzero(drop_reasons == "RETRY_LIMIT_REACHED")
            interface_down = np.count_nonzero(drop_reasons == "INTERFACE_DOWN")

            # If not received, add the data of failed packet
            rx_time = pkt_drops["RxTime"].max()
            tx_time = pkt_drops["TxTime"].max()
            bytes = row["Bytes"]
            # rssi = pkt_drops["RSSI"].max() # This should be taking the max RSSI, but since it is not used, leaving it as mean for now
            u2g_sinr = pkt_drops["U2G_SINR"].max()
            u2g_ber = pkt_drops["U2G_BER"].max()
            pkt_drops["Delay"] = pkt_drops['RxTime']-pkt_drops['TxTime']
            delay = pkt_drops["Delay"].max()
            u2g_distance = pkt_drops["U2G_Distance"].max()
            # Packet State Based on Failure Mode
            if interface_down > 0:
                pkt_state = "INTERFACE_DOWN" # The packet failed due to interface down
                if math.isnan(u2g_distance):
                    if rx_index <= 1: # The Rx is either GCS or GW
                        u2g_distance = math.sqrt(h_dist**2 + height**2)
                    else: # The Rx is a UAV
                        uav_index = rx_index - 2
                        num_uav = len(pd_df_list) - 2
                        uav_x = h_dist + uav_radius * math.cos(uav_index * 2 * math.pi / num_uav)
                        uav_y = uav_radius * math.sin(uav_index * 2 * math.pi / num_uav)
                        u2g_distance = math.sqrt(uav_x**2 + uav_y**2 + height**2)
            elif retry_limit_excd > 0:
                pkt_state = "RETRY_LIMIT_REACHED" # The packet failed to be received (RETRY_LIMIT_EXCEEDED)
                # If packet was dropped due to retry limit reach at the GW, then there may not be any U2G distance recorded. But knowing the speed, we can compute it
                if math.isnan(u2g_distance):
                    if rx_index <= 1: # The Rx is either GCS or GW
                        u2g_distance = math.sqrt(h_dist**2 + height**2)
                    else: # The Rx is a UAV
                        uav_index = rx_index - 2
                        num_uav = len(pd_df_list) - 2
                        uav_x = h_dist + uav_radius * math.cos(uav_index * 2 * math.pi / num_uav)
                        uav_y = uav_radius * math.sin(uav_index * 2 * math.pi / num_uav)
                        u2g_distance = math.sqrt(uav_x**2 + uav_y**2 + height**2)
            elif queue_overflow > 0:
                pkt_state = "QUEUE_OVERFLOW" # The packet failed due to queue buffer overflow
                # If packet was dropped due to queue overflow, then there will not be any U2G distance recorded. But knowing the speed, we can compute it
                if math.isnan(u2g_distance):
                    if rx_index <= 1: # The Rx is either GCS or GW
                        u2g_distance = math.sqrt(h_dist**2 + height**2)
                    else: # The Rx is a UAV
                        uav_index = rx_index - 2
                        num_uav = len(pd_df_list) - 2
                        uav_x = h_dist + uav_radius * math.cos(uav_index * 2 * math.pi / num_uav)
                        uav_y = uav_radius * math.sin(uav_index * 2 * math.pi / num_uav)
                        u2g_distance = math.sqrt(uav_x**2 + uav_y**2 + height**2)
            else:
                pkt_state = "FAILED" # Unaccounted fail reason
                print("Packet Failure Mode Unknown")

            failed_pkt = {'RxTime': rx_time,'TxTime': tx_time,'Packet_Name': packetName,'Bytes': bytes,'U2G_SINR': u2g_sinr,'U2G_BER': u2g_ber,
                          'Dest_Addr': dest_addr,'U2G_Distance': u2g_distance,'Retry_Count': len(drop_reasons),'Delay': delay, 'Packet_State': pkt_state}
            failed_pkt_list.append(failed_pkt)
        
    failed_pkt_df = pd.DataFrame(failed_pkt_list)
    rx_df = pd.concat([rx_df,failed_pkt_df], ignore_index = True)
    rx_df = rx_df.sort_values("RxTime")
    rx_df = rx_df.reset_index()
    return rx_df

def process_throughput(df, timeDiv):
    '''
    Function to calculate throughput data for a DataFrame
    timeDiv is the time division to use for calculating the throughput
    '''
    maxTime = math.ceil(float(df["RxTime"].max()))
    for i in range(math.ceil(maxTime / timeDiv)):
        df_in_range = df.loc[(df["RxTime"] >= (i*timeDiv)) & (df["RxTime"] < ((i+1)*timeDiv)) & (df["Packet_State"].isin(["Reliable","Delay_Exceeded"]))]
        totalBytes = df_in_range["Bytes"].sum()
        throughput = totalBytes / timeDiv
        df.loc[(df["RxTime"] >= (i*timeDiv)) & (df["RxTime"] < ((i+1)*timeDiv)), "Throughput"] = throughput
    return df

def get_pkt_type_cat(num_members):
    pkt_type_cat = ["UAVData_{}".format(i) for i in range(num_members)]
    pkt_type_cat.insert(0,"GatewayData")
    pkt_type_cat.insert(0,"CNCData")
    pkt_type_cat.append("GatewayVideo")
    return pkt_type_cat

def process_scenario_v2(scenario, sim_root_path, dl_delay_threshold, ul_delay_threshold, save_path, GX_GCS=0):
    '''
    GX_GCS is the x-coordinate of GCS. 
    '''
    # UNCOMMENT FOR DEBUGGING - If processed file exist, skip 
    if os.path.isfile(os.path.join(save_path,"{}_downlink.csv".format(scenario))):
        return

    print(scenario)
    
    scenario_files = glob.glob(sim_root_path + "/{}_*.csv".format(scenario)) # Get list of csv files belonging to this scenario
    scenario_params = scenario.split('_')
    num_member = int(scenario_params[0].split('-')[-1])
    inter_uav_distance = int(scenario_params[1].split('-')[-1])
    height = int(scenario_params[2].split('-')[-1]) 
    h_dist = float(scenario_params[3].split('-')[-1]) - GX_GCS 
    # u2g_dist = math.sqrt(h_dist**2 + height**2)
    uav_sending_interval = int(scenario_params[5].split('-')[-1])
    rx_df_list, tx_df_list, pd_df_list, mon_df_list = compile_micro_sim_data_v2(scenario_files)

    # Sort out which df is which
    gcs_tx_df = tx_df_list[0]
    gcs_rx_df = rx_df_list[0]
    uavs_tx_df = pd.concat(tx_df_list[1:len(tx_df_list)], ignore_index=True)
    uavs_rx_df = pd.concat(rx_df_list[1:len(rx_df_list)], ignore_index=True)

    # Convert Packet_Name in Rx_DF to categorical
    pkt_type_cat = get_pkt_type_cat(num_member) # Packet_Type categories
    if not gcs_rx_df.empty:
        gcs_rx_df[["Packet_Type", "Packet_Seq"]] = gcs_rx_df.Packet_Name.str.split("-",expand=True)
        gcs_rx_df["Packet_Type"] = pd.Categorical(gcs_rx_df["Packet_Type"].values, categories=pkt_type_cat)
        gcs_rx_df["Packet_Seq"] = gcs_rx_df.Packet_Seq.astype("uint32")
    else:
        gcs_rx_df["Packet_Type"] = []
        gcs_rx_df["Packet_Seq"] = []
    if not uavs_rx_df.empty:
        uavs_rx_df[["Packet_Type", "Packet_Seq"]] = uavs_rx_df.Packet_Name.str.split("-",expand=True)
        uavs_rx_df["Packet_Type"] = pd.Categorical(uavs_rx_df["Packet_Type"].values, categories=pkt_type_cat)
        uavs_rx_df["Packet_Seq"] = uavs_rx_df.Packet_Seq.astype("uint32")
    else:
        uavs_rx_df["Packet_Type"] = []
        uavs_rx_df["Packet_Seq"] = []
    for i in range(len(pd_df_list)):
        if not pd_df_list[i].empty:
            pd_df_list[i][["Packet_Type", "Packet_Seq"]] = pd_df_list[i].Packet_Name.str.split("-",expand=True)
            pd_df_list[i]["Packet_Type"] = pd.Categorical(pd_df_list[i]["Packet_Type"].values, categories=pkt_type_cat)
            pd_df_list[i]["Packet_Seq"] = pd_df_list[i].Packet_Seq.astype("uint32")
        else:
            pd_df_list[i]["Packet_Type"] = []
            pd_df_list[i]["Packet_Seq"] = []
    
    # Process the state of each packets sent in DL
    # uavs_rx_df = process_received_packets_v2(uavs_rx_df, pd_df_list[1], delay_threshold)
    # uavs_rx_df = process_received_packets_v3(uavs_rx_df, pd_df_list, dl_delay_threshold)
    uavs_rx_df = process_received_packets_v4(uavs_rx_df, dl_delay_threshold)
    
    dl_df = process_dropped_packets_v8(gcs_tx_df, uavs_rx_df, pd_df_list, h_dist=h_dist, height=height)
    if dl_df is not None:
        # dl_df = process_throughput(dl_df,1)
        dl_df["Height"] = height
        # dl_df["Inter_UAV_Distance"] = inter_uav_distance
        # dl_df["Num_Members"] = num_member
        dl_df["UAV_Sending_Interval"] = uav_sending_interval
        dl_df.drop(columns=["Packet_Type", "Packet_Seq"], inplace=True)
        # dl_df["U2G_H_Dist"] = dl_df.apply(h_dist_calc, axis=1)
        # dl_df[['Mean_SINR',"Std_Dev_SINR"]]= dl_df.apply(lambda row: sinr_lognormal_approx(row['U2G_H_Dist'],row['Height']),axis=1,result_type='expand')
        # Drop "index" column if its there
        if "index" in dl_df.columns:
            dl_df = dl_df.drop(["index"], axis=1)
        dl_df.to_csv(os.path.join(save_path,"{}_downlink.csv".format(scenario)), index=False)

    # Process the state of each packets sent in UL
    # gcs_rx_df = process_received_packets_v2(gcs_rx_df, pd_df_list[1], delay_threshold)
    # gcs_rx_df = process_received_packets_v3(gcs_rx_df, pd_df_list, ul_delay_threshold)
    gcs_rx_df = process_received_packets_v4(gcs_rx_df, ul_delay_threshold)

    ul_df = process_dropped_packets_v8(uavs_tx_df, gcs_rx_df, pd_df_list, h_dist=h_dist, height=height)
    if ul_df is not None:
        # ul_df = process_throughput(ul_df, 1)
        ul_df["Height"] = height
        # ul_df["Inter_UAV_Distance"] = inter_uav_distance
        # ul_df["Num_Members"] = num_member
        ul_df["UAV_Sending_Interval"] = uav_sending_interval
        ul_df.drop(columns=["Packet_Type", "Packet_Seq"], inplace=True)
        # ul_df["U2G_H_Dist"] = ul_df.apply(h_dist_calc, axis=1)
        # ul_df[['Mean_SINR',"Std_Dev_SINR"]]= ul_df.apply(lambda row: sinr_lognormal_approx(row['U2G_H_Dist'],row['Height']),axis=1,result_type='expand')
        # Drop "index" column if its there
        if "index" in ul_df.columns:
            ul_df = ul_df.drop(["index"], axis=1)
        ul_df.to_csv(os.path.join(save_path,"{}_uplink.csv".format(scenario)), index=False)

def process_sim_data_v2(sim_root_path, dl_delay_threshold, ul_delay_threshold, save_path):
    # Concatenates all UL & DL results from sim_root_path into a single df
    scenario_list = [csv.split('/')[-1][0:-11] for csv in glob.glob(sim_root_path + "/*GCS-Tx.csv")] # Get list of "unique" scenarios
    
    # For each scenario, extract the UL and DL raw data
    GX_GCS = 0 # The x-coord of GCS (refer to OMNeT++ ini sim file) (set to zero if already subtracted in OMNeT++)
    with Pool(64) as pool:
        pool.starmap(process_scenario_v2, zip(scenario_list, repeat(sim_root_path), repeat(dl_delay_threshold), repeat(ul_delay_threshold), repeat(save_path), repeat(GX_GCS)))
    # process_scenario_v2(scenario_list[0],sim_root_path,delay_threshold,NP,save_path,sending_interval_range)
    return 

if __name__ == "__main__":
    '''
    MAKE SURE TO SET "GX_GCS" and "sending_interval_range" in function process_sim_data_v2
    '''
    # sim_root_path = "/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_NoVideo/Test/Test_Dataset_1_10000"
    # save_path = "/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_NoVideo/Test/Test_Dataset_1_10000_processed"
    sim_root_path = "/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/Test/Test_Dataset_1_100000"
    save_path = "/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/Test/Test_Dataset_1_100000_processed"
    dl_delay_threshold = 1
    ul_delay_threshold = 1
    process_sim_data_v2(sim_root_path, dl_delay_threshold=dl_delay_threshold, ul_delay_threshold=ul_delay_threshold, save_path=save_path)

    
