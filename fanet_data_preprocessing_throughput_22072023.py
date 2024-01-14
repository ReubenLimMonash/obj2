'''
Date: 22/07/2023
Desc: Pandas data processing of FANET dataset with multiprocessing.
      FOR THROUGHPUT DATA
      Reads RX CSV file and calculates throughput for each time step
      Compressed data version: Returns dataframe of the number (of occurence) of each unique throughput reading measured in each scenario
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


def compile_micro_sim_data_rx_tx(file_list, video=False):
    '''
    Function to compile RX and TX data from the CSV files generated by each micro-simulation
    To specifically return the rx_df and tx_df in lists, so that specific dfs can be accessed (instead of aggregating UAV dfs)
    Input: file_list - List of simulation files belonging to a certain scenario (micro-sim)
            video - flag to indicate whether to compile video data
    Output: List of rx_df and tx_df dataframes
    '''
    # Specify dtypes to save memory
    tx_df_dtypes = {"TxTime": np.float32, "Packet_Name": "category", "Packet_Seq": np.uint32, "Bytes": np.uint16, "Dest_Addr": 'category'}
    rx_df_dtypes = {"RxTime": np.float64, "TxTime": np.float32,	"Packet_Name": "str", "Bytes": np.uint16, "RSSI": 'str', "U2G_SINR": np.float32, "U2U_SINR": np.float32, 
                    "U2G_BER": np.float32, "U2U_BER": np.float32, "Src_Addr": 'category', "Dest_Addr": 'category', "Hop_Count": np.uint8, "Delay": np.float32, 
                    "Queueing_Time": np.float32, "Backoff_Time": np.float32, "U2G_Distance": np.float16, "Retry_Count": np.uint8}

    # Let's get the RX dfs ===============================================================
    gcs_rx_file = [file for file in file_list if ('_GCS-Rx.csv' in file)]
    gw_rx_file = [file for file in file_list if (('_GW-Rx' in file))]
    uavs_rx_df_list = []
    uav_rx_files = [file for file in file_list if (('_UAV-' in file) and ('-Rx' in file))]
    if len(gcs_rx_file) > 0:
        gcs_rx_df = pd.read_csv(gcs_rx_file[0], dtype=rx_df_dtypes)
    else:
        print("GCS RX File Missing")
        print(file_list[0])

    if len(gw_rx_file) > 0:
        gw_rx_df = pd.read_csv(gw_rx_file[0], dtype=rx_df_dtypes)
    else:
        print("GW RX File Missing")
        print(file_list[0])
    
    if len(uav_rx_files) > 0:
        for uav_rx_file in uav_rx_files:
            uavs_rx_df_list.append(pd.read_csv(uav_rx_file, dtype=rx_df_dtypes))
    else:
        print("UAV RX File(s) Missing")
        print(file_list[0])

    # Let's get the TX dfs ===============================================================
    gcs_tx_file = [file for file in file_list if ('_GCS-Tx.csv' in file)]
    gw_tx_file = [file for file in file_list if (('_GW-Tx' in file))]
    uavs_tx_df_list = []
    uav_tx_files = [file for file in file_list if (('_UAV-' in file) and ('-Tx' in file))]

    if len(gcs_tx_file) > 0:
        gcs_tx_df = pd.read_csv(gcs_tx_file[0], dtype=tx_df_dtypes)
    else:
        print("GCS TX File Missing")
        print(file_list[0])
    
    if len(gw_tx_file) > 0:
        gw_tx_df = pd.read_csv(gw_tx_file[0], dtype=tx_df_dtypes)
    else:
        print("GW TX File Missing")
        print(file_list[0])

    if len(uav_tx_files) > 0:
        for uav_tx_file in uav_tx_files:
            uavs_tx_df_list.append(pd.read_csv(uav_tx_file, dtype=tx_df_dtypes))
    else:
        print("UAV TX File(s) Missing")
        print(file_list[0])

    rx_df_list = [gcs_rx_df, gw_rx_df] + uavs_rx_df_list
    tx_df_list = [gcs_tx_df, gw_tx_df] + uavs_tx_df_list

    # If video data available, compile it
    if video:
        gcs_video_rx_file = [file for file in file_list if ('_GCS-Video-Rx.csv' in file)]
        gw_video_tx_file = [file for file in file_list if ('_GW-Video-Tx.csv' in file)]
        if len(gcs_video_rx_file) > 0:
            gcs_video_rx_df = pd.read_csv(gcs_video_rx_file[0], dtype=rx_df_dtypes)
        else:
            print("GCS VIDEO RX File Missing")
            print(file_list[0])
        if len(gw_video_tx_file) > 0:
            gw_video_tx_df = pd.read_csv(gw_video_tx_file[0], dtype=tx_df_dtypes)
        else:
            print("GW VIDEO TX File Missing")
            print(file_list[0])
        video_df_list = [gw_video_tx_df, gcs_video_rx_df] # For video packets

        return rx_df_list, tx_df_list, video_df_list
    
    else:
        return rx_df_list, tx_df_list

def process_throughput_counts(rx_df, timeDiv, maxTime, delay_threshold):
    '''
    Function to calculate throughput data for a DataFrame based on time slots
    timeDiv is the time division (length of time slots) used for calculating the throughput
    maxTime is the max simulation time to consider for calculating the throughput
    delay_threshold is the threshold to consider for delay exceeded
    MODIFIED: Return only the number of each unique throughput occurence
    ''' 
    # maxTime = math.ceil(float(rx_df["RxTime"].max()))
    throughput_list = []
    rx_df["Delay"] = rx_df['RxTime']-rx_df['TxTime']
    # Let's get throughput data only from 
    for i in range(math.ceil(maxTime / timeDiv)):
        df_in_range = rx_df.loc[(rx_df["RxTime"] >= (i*timeDiv)) & (rx_df["RxTime"] < ((i+1)*timeDiv)) & (rx_df["Delay"] <= delay_threshold)]
        totalBytes = df_in_range["Bytes"].sum()
        throughput = totalBytes / timeDiv
        throughput_list.append(throughput)
    return np.unique(throughput_list, return_counts=True)

def process_throughput_time_slots(rx_df, timeDiv, maxTime, delay_threshold):
    '''
    Function to calculate throughput data for a DataFrame based on time slots
    timeDiv is the time division (length of time slots) used for calculating the throughput
    maxTime is the max simulation time to consider for calculating the throughput
    delay_threshold is the threshold to consider for delay exceeded
    Return: A DataFrame of throughput samples with its associated horizontal distance
    NOTE: Make sure the "Horizontal_Distance for each packet has been calculated
    ''' 
    rx_df.sort_values(["RxTime"], inplace=True)
    rx_df.reset_index(inplace=True, drop=True)
    throughput_list = []
    rx_df["Delay"] = rx_df['RxTime']-rx_df['TxTime']
    # Let's get throughput data only from 
    for i in range(math.ceil(maxTime / timeDiv)):
        df_in_range = rx_df.loc[(rx_df["RxTime"] >= (i*timeDiv)) & (rx_df["RxTime"] < ((i+1)*timeDiv)) & (rx_df["Delay"] <= delay_threshold)]
        if df_in_range.empty:
            throughput = 0
            HDist = np.NaN
        else: # Calculate total data size of packets in time slot, divided by time duration to rcvd these (start time from prev packet)
            firstIdx = df_in_range.index.values[0]
            if firstIdx == 0: # This should be the case of first time slot
                startTime = 0
            else:
                prevPkt = rx_df.iloc[firstIdx-1]
                startTime = prevPkt["RxTime"]
            endTime = df_in_range["RxTime"].max()
            totalBytes = df_in_range["Bytes"].sum()
            timePeriod = endTime - startTime
            throughput = totalBytes / timePeriod
            HDist = df_in_range["Horizontal_Distance"].max()
            
        throughput_list.append({"Horizontal_Distance": HDist, "Throughput": throughput})
    return pd.DataFrame(throughput_list)

def process_throughput_sliding_window_packets(rx_df, winSize, delay_threshold):
    '''
    Function to calculate throughput data for a DataFrame based on sliding window measured by no. of packets
    winSize is the sliding window size (in number of packets)
    delay_threshold is the threshold to consider for delay exceeded
    Return: A DataFrame of throughput samples with its associated horizontal distance
    NOTE: Make sure the "Horizontal_Distance for each packet has been calculated
    ''' 
    throughput_list = []
    rx_df["Delay"] = rx_df['RxTime']-rx_df['TxTime']
    df_reliable = rx_df.loc[(rx_df["Delay"] <= delay_threshold)].copy()
    df_reliable.sort_values(["RxTime"], inplace=True)

    # Let's get throughput data only from 
    for i in range(len(df_reliable)-winSize):
        df_in_window = df_reliable.iloc[i:i+winSize]
        totalBytes = df_in_window["Bytes"].sum()
        if i == 0:
            startTime = 0
        elif i > 0:
            startTime = df_reliable.iloc[i-1]["RxTime"]
        endTime = df_in_window["RxTime"].max()
        timePeriod = endTime - startTime
        throughput = totalBytes / timePeriod
        HDist = df_in_window["Horizontal_Distance"].max()
        throughput_list.append({"Horizontal_Distance": HDist, "Throughput": throughput})
    return pd.DataFrame(throughput_list)

def process_throughput_sliding_window_time_v1(rx_df, winSize, delay_threshold):
    '''
    Function to calculate throughput data for a DataFrame based on sliding window measured by time duration
    The sliding window will slide by next packet
    winSize is the sliding window size (in seconds)
    delay_threshold is the threshold to consider for delay exceeded
    Return: A DataFrame of throughput samples with its associated horizontal distance
    NOTE: Make sure the "Horizontal_Distance" for each packet has been calculated
    ''' 
    throughput_list = []
    rx_df["Delay"] = rx_df['RxTime']-rx_df['TxTime']
    df_reliable = rx_df.loc[(rx_df["Delay"] <= delay_threshold)].copy()
    df_reliable.sort_values(["RxTime"], inplace=True)
    df_reliable.reset_index(inplace=True, drop=True)
    
    # Loop through all packets 
    for row in df_reliable.itertuples():
        # If the Rx Time is less than winSize, skip (not enough packets to fill up window size)
        if row.RxTime < winSize:
            continue
        df_in_window = df_reliable.loc[(df_reliable["RxTime"]>=(row.RxTime-winSize)) & (df_reliable["RxTime"]<=row.RxTime)]
        if len(df_in_window) == 1:
            # If only one packet rcvd during this time period, calc throughput based on this one packet and previous rcvd packet
            if row.Index == 0:
                startTime = 0
            else:
                prevPkt = df_reliable.iloc[row.Index-1]
                startTime = prevPkt["RxTime"]
            totalBytes = row.Bytes
            endTime = row.RxTime
        else:
            if row.Index < len(df_in_window): # This should be the case of the first slide window
                startTime = 0
                totalBytes = df_in_window["Bytes"].sum()
                endTime = row.RxTime
            else:
                prevPkt = df_reliable.iloc[row.Index-len(df_in_window)]
                startTime = prevPkt["RxTime"]
                totalBytes = df_in_window["Bytes"].sum() 
                endTime = row.RxTime
        timePeriod = endTime - startTime
        throughput = totalBytes / timePeriod
        HDist = row.Horizontal_Distance
        throughput_list.append({"Horizontal_Distance": HDist, "Throughput": throughput})
    return pd.DataFrame(throughput_list)

def process_throughput_sliding_window_time_v2(rx_df, winSize, stride, max_time, delay_threshold):
    '''
    Function to calculate throughput data for a DataFrame based on sliding window measured by time duration
    The sliding window will slide by "stride" seconds
    winSize is the sliding window size (in seconds)
    maxTime is the max simulation time to consider for calculating the throughput (seconds)
    delay_threshold is the threshold to consider for delay exceeded
    Return: A DataFrame of throughput samples with its associated horizontal distance
    NOTE: Make sure the "Horizontal_Distance" for each packet has been calculated
    ''' 
    throughput_list = []
    rx_df["Delay"] = rx_df['RxTime']-rx_df['TxTime']
    # df_reliable = rx_df.loc[(rx_df["Delay"] <= delay_threshold)].copy()
    rx_df.sort_values(["RxTime"], inplace=True)
    rx_df.reset_index(inplace=True, drop=True)
    # Loop through all packets 
    for i in range(math.ceil((max_time - winSize) / stride)):
        rx_df_in_range = rx_df.loc[(rx_df["RxTime"] >= (i*stride)) & (rx_df["RxTime"] < (i*stride+winSize))]
        df_reliable = rx_df_in_range.loc[(rx_df_in_range["Delay"] <= delay_threshold)].copy()
        totalBytes = df_reliable["Bytes"].sum()
        throughput = totalBytes / winSize
        # TODO: Associate the throughput sample with max horizontal distance within reliable and dropped packets
        HDist = rx_df_in_range["Horizontal_Distance"].max()
        throughput_list.append({"Horizontal_Distance": HDist, "Throughput": throughput})
    return pd.DataFrame(throughput_list)

def process_scenario_v3(scenario, dl_slot_size, ul_slot_size, dl_delay_threshold, ul_delay_threshold, mode='timeSlot', vid_slot_size=1, vid_delay_threshold=0, GX_Offset=0):
    '''
    mode: 'timeSlot' / 'slideWindow' / 'count' (The throughput measurement mode)
    dl_slot_size: If mode=='timeSlot', this is the length of the slot size in seconds,
                  If mode=='slideWindow', this is the length of the sliding window in no. of packets
                  dl indicates Downlink. Likewise for ul (Uplink) and vid (Video)
    GX_Offset is the x-coordinate offset for h_dist in scenario file name
    NOTE: If vid_delay_threshold = 0 (default), then the script will consider no video data to process!!!
    '''

    print(scenario)
    
    scenario_files = glob.glob("{}_*.csv".format(scenario[0:-11])) # Get list of csv files belonging to this scenario
    scenario_params = scenario.split("/")[-1].split('_')
    height = int(scenario_params[2].split('-')[-1]) 
    modulation = scenario_params[4].split('-')[-1]
    uav_sending_interval = int(scenario_params[5].split('-')[-1])
    if modulation == '16':
        modulation = "QAM16"
    elif modulation == '64':
        modulation = "QAM64"

    if vid_delay_threshold == 0:
        rx_df_list, tx_df_list = compile_micro_sim_data_rx_tx(scenario_files, video=False)
    elif vid_delay_threshold > 0:
        rx_df_list, tx_df_list, video_df_list = compile_micro_sim_data_rx_tx(scenario_files, video=True)

    # Sort out which df is which
    gcs_rx_df = rx_df_list[0]
    gcs_tx_df = tx_df_list[0]
    uavs_rx_df = pd.concat(rx_df_list[1:len(rx_df_list)], ignore_index=True) # Includes GW Rx DF and all UAVs Rx DFs
    uavs_tx_df = pd.concat(tx_df_list[1:len(tx_df_list)], ignore_index=True) # Includes GW Tx DF and all UAVs Tx DFs
    
    if mode == 'count':
        h_dist = float(scenario_params[3].split('-')[-1]) - GX_Offset 
        # Calculate the mean and std dev of SINR for this scenario
        mean_sinr, std_dev_sinr = sinr_lognormal_approx(h_dist, height, env='suburban')
        # Get the count of different throughput measures in DL
        max_time = gcs_tx_df["TxTime"].max()
        throughput_data, counts = process_throughput_counts(uavs_rx_df, dl_slot_size, max_time, dl_delay_threshold)
        dl_throughput_list = []
        for i in range(len(throughput_data)):
            dl_throughput_list.append({"Horizontal_Distance": h_dist, "Height": height, "UAV_Sending_Interval": uav_sending_interval, "Modulation": modulation, "Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr,
                    "Throughput": throughput_data[i], "Num_Count": counts[i]})
        
        # Get the count of different throughput measures in UL
        max_time = uavs_tx_df["TxTime"].max()
        throughput_data, counts = process_throughput_counts(gcs_rx_df, ul_slot_size, max_time, ul_delay_threshold)
        ul_throughput_list = []
        for i in range(len(throughput_data)):
            ul_throughput_list.append({"Horizontal_Distance": h_dist, "Height": height, "UAV_Sending_Interval": uav_sending_interval, "Modulation": modulation, "Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr,
                    "Throughput": throughput_data[i], "Num_Count": counts[i]})
        
        # Get the count of different throughput measures in video link
        if vid_delay_threshold > 0:
            video_tx_df = video_df_list[0]
            video_rx_df = video_df_list[1]
            max_time = video_tx_df["TxTime"].max()
            throughput_data, counts = process_throughput_counts(video_rx_df, vid_slot_size, max_time, vid_delay_threshold)
            vid_throughput_list = []
            for i in range(len(throughput_data)):
                vid_throughput_list.append({"Horizontal_Distance": h_dist, "Height": height, "UAV_Sending_Interval": uav_sending_interval, "Modulation": modulation, "Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr,
                        "Throughput": throughput_data[i], "Num_Count": counts[i]})
            vid_throughput_df = pd.DataFrame(vid_throughput_list)

        dl_throughput_df = pd.DataFrame(dl_throughput_list)
        ul_throughput_df = pd.DataFrame(ul_throughput_list)
    
    elif mode == 'timeSlot':
        assert scenario_params[1].split('-')[0] == "UAVSpeed", "File name need to contain UAV Speed at second _ position"
        uav_speed = int(scenario_params[1].split('-')[-1]) 
        assert uav_speed > 0, "Mode timeSlot assumes UAVs are moving linearly with speed uav_speed, which cannot be 0"
        # Get the throughput measures in DL based on time slots
        max_time = gcs_tx_df["TxTime"].max()
        uavs_rx_df["Horizontal_Distance"] = uavs_rx_df["RxTime"] * uav_speed
        dl_throughput_df = process_throughput_time_slots(uavs_rx_df, dl_slot_size, max_time, dl_delay_threshold)
        # Only fill at first index ---------------------
        # dl_throughput_df.loc[dl_throughput_df.index[0], 'Height'] = height
        # dl_throughput_df.loc[dl_throughput_df.index[0], 'UAV_Sending_Interval'] = uav_sending_interval
        # dl_throughput_df.loc[dl_throughput_df.index[0], 'Modulation'] = modulation
        # dl_throughput_df.loc[dl_throughput_df.index[0], 'UAV_Speed'] = uav_speed
        # Fill values in all rows ---------------------
        dl_throughput_df['Height'] = height
        dl_throughput_df['UAV_Sending_Interval'] = uav_sending_interval
        dl_throughput_df['Modulation'] = modulation
        dl_throughput_df['UAV_Speed'] = uav_speed
        # dl_throughput_df.loc[dl_throughput_df.index[0], 'Mean_SINR'] = mean_sinr
        # dl_throughput_df.loc[dl_throughput_df.index[0], 'Std_Dev_SINR'] = std_dev_sinr
        
        # Get the throughput measures in UL based on time slots
        max_time = uavs_tx_df["TxTime"].max()
        gcs_rx_df["Horizontal_Distance"] = gcs_rx_df["RxTime"] * uav_speed
        ul_throughput_df = process_throughput_time_slots(gcs_rx_df, ul_slot_size, max_time, ul_delay_threshold)
        # Only fill at first index ---------------------
        # ul_throughput_df.loc[ul_throughput_df.index[0], 'Height'] = height
        # ul_throughput_df.loc[ul_throughput_df.index[0], 'UAV_Sending_Interval'] = uav_sending_interval
        # ul_throughput_df.loc[ul_throughput_df.index[0], 'Modulation'] = modulation
        # ul_throughput_df.loc[ul_throughput_df.index[0], 'UAV_Speed'] = uav_speed
        # Fill values in all rows ---------------------
        ul_throughput_df['Height'] = height
        ul_throughput_df['UAV_Sending_Interval'] = uav_sending_interval
        ul_throughput_df['Modulation'] = modulation
        ul_throughput_df['UAV_Speed'] = uav_speed
        # ul_throughput_df.loc[ul_throughput_df.index[0], 'Mean_SINR'] = mean_sinr
        # ul_throughput_df.loc[ul_throughput_df.index[0], 'Std_Dev_SINR'] = std_dev_sinr
        
        # Get the throughput measures in video link based on time slots
        if vid_delay_threshold > 0:
            video_tx_df = video_df_list[0]
            video_rx_df = video_df_list[1]
            max_time = video_tx_df["TxTime"].max()
            video_rx_df["Horizontal_Distance"] = video_rx_df["RxTime"] * uav_speed
            vid_throughput_df = process_throughput_time_slots(video_rx_df, vid_slot_size, max_time, vid_delay_threshold)
            # Only fill at first index -------------------
            # vid_throughput_df.loc[vid_throughput_df.index[0], 'Height'] = height
            # vid_throughput_df.loc[vid_throughput_df.index[0], 'UAV_Sending_Interval'] = uav_sending_interval
            # vid_throughput_df.loc[vid_throughput_df.index[0], 'Modulation'] = modulation
            # vid_throughput_df.loc[vid_throughput_df.index[0], 'UAV_Speed'] = uav_speed
            # Fill values in all rows ---------------------
            vid_throughput_df['Height'] = height
            vid_throughput_df['UAV_Sending_Interval'] = uav_sending_interval
            vid_throughput_df['Modulation'] = modulation
            vid_throughput_df['UAV_Speed'] = uav_speed
            # vid_throughput_df.loc[vid_throughput_df.index[0], 'Mean_SINR'] = mean_sinr
            # vid_throughput_df.loc[vid_throughput_df.index[0], 'Std_Dev_SINR'] = std_dev_sinr

    elif mode == 'slideWindow':
        assert scenario_params[1].split('-')[0] == "UAVSpeed", "File name need to contain UAV Speed at second _ position"
        uav_speed = int(scenario_params[1].split('-')[-1]) 
        assert uav_speed > 0, "Mode slideWindow assumes UAVs are moving linearly with speed uav_speed, which cannot be 0"
        # Get the throughput measures in DL based on time slots
        max_time = gcs_tx_df["TxTime"].max()
        uavs_rx_df["Horizontal_Distance"] = uavs_rx_df["RxTime"] * uav_speed
        dl_throughput_df = process_throughput_sliding_window_time_v1(uavs_rx_df, dl_slot_size, dl_delay_threshold)
        if dl_throughput_df.empty:
            dl_throughput_df = pd.DataFrame({"Horizontal_Distance": "NaN", "Throughput": "NaN"}, index=[0])
        # Only fill at first index ---------------------
        # dl_throughput_df.loc[dl_throughput_df.index[0], 'Height'] = height
        # dl_throughput_df.loc[dl_throughput_df.index[0], 'UAV_Sending_Interval'] = uav_sending_interval
        # dl_throughput_df.loc[dl_throughput_df.index[0], 'Modulation'] = modulation
        # dl_throughput_df.loc[dl_throughput_df.index[0], 'UAV_Speed'] = uav_speed
        # Fill values in all rows ---------------------
        dl_throughput_df['Height'] = height
        dl_throughput_df['UAV_Sending_Interval'] = uav_sending_interval
        dl_throughput_df['Modulation'] = modulation
        dl_throughput_df['UAV_Speed'] = uav_speed
        # dl_throughput_df.loc[dl_throughput_df.index[0], 'Mean_SINR'] = mean_sinr
        # dl_throughput_df.loc[dl_throughput_df.index[0], 'Std_Dev_SINR'] = std_dev_sinr
        
        # Get the throughput measures in UL based on time slots
        max_time = uavs_tx_df["TxTime"].max()
        gcs_rx_df["Horizontal_Distance"] = gcs_rx_df["RxTime"] * uav_speed
        ul_throughput_df = process_throughput_sliding_window_time_v1(gcs_rx_df, ul_slot_size, ul_delay_threshold)
        if ul_throughput_df.empty:
            ul_throughput_df = pd.DataFrame({"Horizontal_Distance": "NaN", "Throughput": "NaN"}, index=[0])
        # Only fill at first index ---------------------
        # ul_throughput_df.loc[ul_throughput_df.index[0], 'Height'] = height
        # ul_throughput_df.loc[ul_throughput_df.index[0], 'UAV_Sending_Interval'] = uav_sending_interval
        # ul_throughput_df.loc[ul_throughput_df.index[0], 'Modulation'] = modulation
        # ul_throughput_df.loc[ul_throughput_df.index[0], 'UAV_Speed'] = uav_speed
        # Fill values in all rows ---------------------
        ul_throughput_df['Height'] = height
        ul_throughput_df['UAV_Sending_Interval'] = uav_sending_interval
        ul_throughput_df['Modulation'] = modulation
        ul_throughput_df['UAV_Speed'] = uav_speed
        # ul_throughput_df.loc[ul_throughput_df.index[0], 'Mean_SINR'] = mean_sinr
        # ul_throughput_df.loc[ul_throughput_df.index[0], 'Std_Dev_SINR'] = std_dev_sinr
        
        # Get the throughput measures in video link based on time slots
        if vid_delay_threshold > 0:
            video_tx_df = video_df_list[0]
            video_rx_df = video_df_list[1]
            max_time = video_tx_df["TxTime"].max()
            video_rx_df["Horizontal_Distance"] = video_rx_df["RxTime"] * uav_speed
            vid_throughput_df = process_throughput_sliding_window_time_v1(video_rx_df, vid_slot_size, vid_delay_threshold)
            if vid_throughput_df.empty:
                vid_throughput_df = pd.DataFrame({"Horizontal_Distance": "NaN", "Throughput": "NaN"}, index=[0])
            # Only fill at first index -------------------
            # vid_throughput_df.loc[vid_throughput_df.index[0], 'Height'] = height
            # vid_throughput_df.loc[vid_throughput_df.index[0], 'UAV_Sending_Interval'] = uav_sending_interval
            # vid_throughput_df.loc[vid_throughput_df.index[0], 'Modulation'] = modulation
            # vid_throughput_df.loc[vid_throughput_df.index[0], 'UAV_Speed'] = uav_speed
            # Fill values in all rows ---------------------
            vid_throughput_df['Height'] = height
            vid_throughput_df['UAV_Sending_Interval'] = uav_sending_interval
            vid_throughput_df['Modulation'] = modulation
            vid_throughput_df['UAV_Speed'] = uav_speed
            # vid_throughput_df.loc[vid_throughput_df.index[0], 'Mean_SINR'] = mean_sinr
            # vid_throughput_df.loc[vid_throughput_df.index[0], 'Std_Dev_SINR'] = std_dev_sinr
    
    else:
        assert False, "Unknown mode. Should be one of: [count, timeSlot, slideWindow]"

    if vid_delay_threshold > 0:
        return (dl_throughput_df, ul_throughput_df, vid_throughput_df)
    else:
        return (dl_throughput_df, ul_throughput_df)
    
def process_mean_throughput(df):
    '''
    Calculate the mean throughput for each scenario
    '''
    scenarios = df[['Horizontal_Distance','Height','Modulation','UAV_Sending_Interval']].drop_duplicates()
    output_df_list = []
    for row in scenarios.itertuples():
        scenario = df.loc[(df["Modulation"] == row.Modulation) & (df["UAV_Sending_Interval"] == row.UAV_Sending_Interval) & 
                          (df["Horizontal_Distance"] == row.Horizontal_Distance) & (df["Height"] == row.Height)]
        total_throughput = scenario.Throughput * scenario.Num_Count
        total_throughput = np.sum(total_throughput.values)
        total_count = np.sum(scenario.Num_Count.values)
        mean_throughput = total_throughput / total_count
        mean_sinr, std_dev_sinr = sinr_lognormal_approx(row.Horizontal_Distance, row.Height, env='suburban')
        output_df_list.append({"Horizontal_Distance": row.Horizontal_Distance, "Height": row.Height, "UAV_Sending_Interval": row.UAV_Sending_Interval, 
                                "Modulation": row.Modulation, "Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr, "MeanThroughput": mean_throughput})
    return pd.DataFrame(output_df_list)

def process_sim_data_v3(sim_root_path, dl_slot_size, ul_slot_size, dl_delay_threshold, ul_delay_threshold, save_path, mode='timeSlot', vid_slot_size=1, vid_delay_threshold=0):
    '''
    mode: 'timeSlot' / 'slideWindow' / 'count' (The throughput measurement mode)
    dl_slot_size: If mode=='timeSlot', this is the length of the slot size in seconds,
                  If mode=='slideWindow', this is the length of the sliding window in no. of packets
                  dl indicates Downlink. Likewise for ul (Uplink) and vid (Video)
    '''
    
    # Concatenates all UL & DL results from sim_root_path into a single df
    scenario_list = [csv for csv in glob.glob(sim_root_path + "/*GCS-Tx.csv")] # Get list of "unique" scenarios
    # scenario_list = [csv for csv in glob.glob(sim_root_path + "/NumMember-7_UAVSpeed-*_Height-60_*_UAVSendingInterval-1000_GCS-Tx.csv")] # Get specific scenario
    # scenario_list = [csv for csv in glob.glob(sim_root_path + "/NumMember-7_InterUAVDistance-5_Height-60_Distance-*_Modulation-QAM-16_UAVSendingInterval-*_GCS-Tx.csv")] # Get list of "specific" scenarios

    # For each scenario, extract the UL and DL raw data
    GX_GCS = 0 # The x-coord of GCS (refer to OMNeT++ ini sim file) (set to zero if already subtracted in OMNeT++)
    dl_results = []
    ul_results = []
    vid_results = []
    with Pool(15) as pool:
        for result in pool.starmap(process_scenario_v3, zip(scenario_list, repeat(dl_slot_size), repeat(ul_slot_size), repeat(dl_delay_threshold), repeat(ul_delay_threshold), repeat(mode), repeat(vid_slot_size), repeat(vid_delay_threshold), repeat(GX_GCS))):
            dl_results.append(result[0])
            ul_results.append(result[1])
            if vid_delay_threshold > 0:
                vid_results.append(result[2])
    
    dl_df = pd.concat(dl_results)
    ul_df = pd.concat(ul_results)
    if vid_delay_threshold > 0:
        vid_df = pd.concat(vid_results)

    # # Gauss Markov Mobility Case ------------------------
    # uav_speed = 26
    # mobility_df = pd.read_csv(os.path.join(sim_root_path, "UAVSwarmMobility.csv"))
    # dl_df["RxTime"] = dl_df["Horizontal_Distance"] / uav_speed
    # ul_df["RxTime"] = ul_df["Horizontal_Distance"] / uav_speed
    # dl_df["Horizontal_Distance"] = np.interp(dl_df['RxTime'], mobility_df["Time"], mobility_df["X"]-500)
    # ul_df["Horizontal_Distance"] = np.interp(ul_df['RxTime'], mobility_df["Time"], mobility_df["X"]-500)
    # dl_df.drop("RxTime", axis=1, inplace=True)
    # ul_df.drop("RxTime", axis=1, inplace=True)
    # if vid_delay_threshold > 0:
    #     vid_df["RxTime"] = vid_df["Horizontal_Distance"] / uav_speed
    #     vid_df["Horizontal_Distance"] = np.interp(vid_df['RxTime'], mobility_df["Time"], mobility_df["X"]-500)
    #     vid_df.drop("RxTime", axis=1, inplace=True)
    # # --------------------------------------------------

    if mode == 'count':
        # Sort DF and get mean throughput
        dl_df.sort_values(["Modulation", "UAV_Sending_Interval", "Height", "Horizontal_Distance"], inplace=True)
        dl_df_mean_throughput = process_mean_throughput(dl_df)
        dl_df_mean_throughput.to_csv(save_path + "Downlink_MeanThroughput.csv", index=False)
        ul_df.sort_values(["Modulation", "UAV_Sending_Interval", "Height", "Horizontal_Distance"], inplace=True)
        ul_df_mean_throughput = process_mean_throughput(ul_df)
        ul_df_mean_throughput.to_csv(save_path + "Uplink_MeanThroughput.csv", index=False)
        if vid_delay_threshold > 0:
            vid_df.sort_values(["Modulation", "UAV_Sending_Interval", "Height", "Horizontal_Distance"], inplace=True)
            vid_df_mean_throughput = process_mean_throughput(vid_df)
            vid_df_mean_throughput.to_csv(save_path + "Video_MeanThroughput.csv", index=False)

    dl_df.to_csv(save_path + "Downlink_Throughput.csv", index=False)
    ul_df.to_csv(save_path + "Uplink_Throughput.csv", index=False)
    if vid_delay_threshold > 0:
        vid_df.to_csv(save_path + "Video_Throughput.csv", index=False)

    return 

if __name__ == "__main__":
    '''
    MAKE SURE TO SET "GX_GCS" and "sending_interval_range" in function process_sim_data_v2
    '''
    # sim_root_path = "/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_Video/Anomaly/TestCase8_Anomaly_Type2_2UAV_Far/data"
    # save_path = "/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_Video/Anomaly/TestCase8_Anomaly_Type2_2UAV_Far/TestCase8_Anomaly_Type2_2UAV_Far_"
    # sim_root_path = "/media/research-student/One Touch/FANET Datasets/Anomaly_Moving/Base_Case_QAM16/data"
    # save_path = "/media/research-student/One Touch/FANET Datasets/Anomaly_Moving/Base_Case_QAM16/Base_Case_QAM16_TimeSlot2.5s_"
    # sim_root_path = "/media/research-student/One Touch/FANET Datasets/Anomaly_Moving/Base_Case_QAM16_GaussMarkov/data-speed26"
    # save_path = "/media/research-student/One Touch/FANET Datasets/Anomaly_Moving/Base_Case_QAM16_GaussMarkov/Base_Case_QAM16_GaussMarkov_Speed26_timeSlot1s_"
    # save_path = "/media/research-student/One Touch/FANET Datasets/Dataset_NP100000_MultiModulation_Hovering_Video/QPSK/QPSK_"
    sim_root_path = "/media/research-student/One Touch/FANET Datasets/Anomaly_Moving/QAM16_4UAVI_Moderate/data/Run-*"
    save_path = "/media/research-student/One Touch/FANET Datasets/Anomaly_Moving/QAM16_4UAVI_Moderate/QAM16_4UAVI_Moderate_TimeSlot1s_"
    # sim_root_path = "/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP100000_MultiModulation_Hovering_Video/QAM16/data"
    # save_path = "/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP100000_MultiModulation_Hovering_Video/QAM16/Base_Case_QAM16_Static_"
    
    mode = "timeSlot" # 'timeSlot' / 'slideWindow' / 'count' (The throughput measurement mode)
    dl_slot_size = 1 # Either the length of time slots, or the sliding window size. For count, this is the timeDiv
    ul_slot_size = 1 # Either the length of time slots, or the sliding window size
    vid_slot_size = 1 # Either the length of time slots, or the sliding window size

    # mode = "count" # 'timeSlot' / 'slideWindow' / 'count' (The throughput measurement mode)
    # dl_slot_size = 1 # Either the length of time slots, or the sliding window size. For count, this is the timeDiv
    # ul_slot_size = 1 # Either the length of time slots, or the sliding window size
    # vid_slot_size = 1 # Either the length of time slots, or the sliding window size

    dl_delay_threshold = 0.04
    ul_delay_threshold = 0.04
    vid_delay_threshold = 1 # NOTE: Set to zero if no video data
    process_sim_data_v3(sim_root_path, dl_slot_size=dl_slot_size, ul_slot_size=ul_slot_size, 
                        dl_delay_threshold=dl_delay_threshold, ul_delay_threshold=ul_delay_threshold, save_path=save_path, mode=mode,
                        vid_slot_size=vid_slot_size, vid_delay_threshold=vid_delay_threshold)

    
