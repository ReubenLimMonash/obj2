'''
Date: 26/12/2023
Desc: Pandas data processing of FANET dataset with multiprocessing.
      FOR THROUGHPUT DATA
      Reads RX CSV file and calculates throughput for each time step
      Uses sliding window approach
      Modified: to processing DL GCS-2-UAV throughput individually per UAV, rather than combining everything
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

def compile_micro_sim_data_v2(file_list, video=False):
    '''
    Function to compile data from the CSV files generated by each micro-simulation
    Update: To specifically return the rx_df, tx_df, mon_df and pd_df in lists, so that specific dfs can be accessed (instead of aggregating UAV dfs)
    Input: file_list - List of simulation files belonging to a certain scenario (micro-sim)
            video - flag to indicate whether to compile video data
    Output: concatenates the raw data to UL and DL dataframes
    '''
    # Specify dtypes to save memory
    tx_df_dtypes = {"TxTime": np.float32, "Packet_Name": "category", "Packet_Seq": np.uint32, "Bytes": np.uint16, "Dest_Addr": 'category'}
    rx_df_dtypes = {"RxTime": np.float64, "TxTime": np.float32,	"Packet_Name": "str", "Bytes": np.uint16, "RSSI": 'str', "U2G_SINR": np.float32, "U2U_SINR": np.float32, 
                    "U2G_BER": np.float32, "U2U_BER": np.float32, "Src_Addr": 'category', "Dest_Addr": 'category', "Hop_Count": np.uint8, "Delay": np.float32, 
                    "Queueing_Time": np.float32, "Backoff_Time": np.float32, "U2G_Distance": np.float32, "Retry_Count": np.uint8}
    pd_df_dtypes = {"RxTime": np.float64, "TxTime": np.float32,	"Packet_Name": "str", "Bytes": np.uint16, "RSSI": 'str', "U2G_SINR": np.float32, "U2U_SINR": np.float32, 
                    "U2G_BER": np.float32, "U2U_BER": np.float32, "Delay": np.float32, "Queueing_Time": np.float32, "Backoff_Time": np.float32, "U2G_Distance": np.float16,
                    "Has_Bit_Error": np.uint8, "Packet_Drop_Reason": 'category'}

    # Let's get the GCS dfs ===============================================================
    gcs_rx_file = [file for file in file_list if ('_GCS-Rx.csv' in file)]
    gcs_tx_file = [file for file in file_list if ('_GCS-Tx.csv' in file)]
    gcs_mon_file = [file for file in file_list if ('_GCS-Wlan.csv' in file)]
    gcs_pd_file = [file for file in file_list if ('_GCS-PacketDrop.csv' in file)]
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
    gw_rx_file = [file for file in file_list if (('_GW-Rx' in file))]
    gw_tx_file = [file for file in file_list if (('_GW-Tx' in file))]
    gw_mon_file = [file for file in file_list if (('_GW-Wlan' in file))]
    gw_pd_file = [file for file in file_list if (('_GW-PacketDrop' in file))]
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

    # If video data available, compile it
    if video:
        gcs_video_rx_file = [file for file in file_list if ('_GCS-Video-Rx.csv' in file)]
        gw_video_tx_file = [file for file in file_list if ('_GW-Video-Tx.csv' in file)]
        gw_video_pd_file = [file for file in file_list if ('_GW-Video-PacketDrop.csv' in file)]
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
        if len(gw_video_pd_file) > 0:
            gw_video_pd_df = pd.read_csv(gw_video_pd_file[0], dtype=pd_df_dtypes)
        else:
            print("GW VIDEO PD File Missing")
            print(file_list[0])
        video_df_list = [gw_video_tx_df, gcs_video_rx_df, gw_video_pd_df] # For video packets

    if not video:
        return rx_df_list, tx_df_list, pd_df_list, mon_df_list
    else:
        return rx_df_list, tx_df_list, pd_df_list, mon_df_list, video_df_list

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

def process_throughput_sliding_window_time_v2(rx_df, pd_df, winSize, stride, max_time, delay_threshold):
    '''
    Function to calculate throughput data for a DataFrame based on sliding window measured by time duration
    The sliding window will slide by "stride" seconds
    winSize is the sliding window size (in seconds)
    maxTime is the max simulation time to consider for calculating the throughput (seconds)
    delay_threshold is the threshold to consider for delay exceeded
    rx_df - The rcvd DF of interest
    pd_df - The paccket drop df of transmitter, to get measured reliability
    Return: A DataFrame of throughput samples with its associated horizontal distance
    NOTE: Make sure the "Horizontal_Distance" for each packet has been calculated
    ''' 
    throughput_list = []
    rx_df["Delay"] = rx_df['RxTime']-rx_df['TxTime']
    # rx_df.sort_values(["RxTime"], inplace=True)
    # rx_df.reset_index(inplace=True, drop=True)
    # Slide the time window
    for i in range(math.ceil((max_time - winSize) / stride)):
        rx_df_in_range = rx_df.loc[(rx_df["RxTime"] >= (i*stride)) & (rx_df["RxTime"] < (i*stride+winSize))] # Get rcvd packets within time window
        pd_df_in_range = pd_df.loc[(pd_df["RxTime"] >= (i*stride)) & (pd_df["RxTime"] < (i*stride+winSize))] # Get dropped packets within time window (we base it on time of drop - 'RxTime')
        if not (rx_df_in_range.empty and pd_df_in_range.empty):
            df_reliable = rx_df_in_range.loc[(rx_df_in_range["Delay"] <= delay_threshold)].copy() # Get reliable packets within time window
            totalBytes = df_reliable["Bytes"].sum()
            throughput = totalBytes / winSize
            numReliable = len(df_reliable)
            numSent = len(rx_df_in_range) + len(pd_df_in_range)
            HDist = np.nanmax([rx_df_in_range["Horizontal_Distance"].max(), pd_df_in_range["Horizontal_Distance"].max()])
            throughput_list.append({"Time": i*stride+winSize, "Horizontal_Distance": HDist, "Throughput": throughput, "Measured_Reliability": numReliable/numSent})
    return pd.DataFrame(throughput_list)

def process_scenario_v4(scenario_path, save_path, dl_stride, ul_stride, dl_slot_size, ul_slot_size, dl_delay_threshold, ul_delay_threshold, mode='timeSlot', vid_stride=1, vid_slot_size=1, vid_delay_threshold=0, GX_Offset=0):
    '''
    Modified: To process all different runs of each sccenario (for measured throughput while UAV moving)
    mode: 'timeSlot' / 'slideWindow' / 'count' (The throughput measurement mode)
    dl_slot_size: If mode=='timeSlot', this is the length of the slot size in seconds,
                  If mode=='slideWindow', this is the length of the sliding window in no. of packets
                  dl indicates Downlink. Likewise for ul (Uplink) and vid (Video)
    stride: The sliding window will slide by "stride" seconds
    GX_Offset is the x-coordinate offset for h_dist in scenario file name
    NOTE: If vid_delay_threshold = 0 (default), then the script will consider no video data to process!!!
    '''
    scenario = scenario_path.split("/")[-1]

    # UNCOMMENT BELOW TO ONLY PROCESS SCENARIOS WHERE THE SAVE_PATH DIR IS NOT YET CREATED
    if os.path.isdir(os.path.join(save_path, scenario)):
        return

    print(scenario)
    scenario_params = scenario.split('_')
    height = float(scenario_params[2].split('-')[-1]) 
    modulation = scenario_params[4].split('-')[-1]
    bitrate = float(scenario_params[1].split('-')[-1]) 
    uav_sending_interval = float(scenario_params[5].split('-')[-1])
    uav_speed = float(scenario_params[0].split('-')[-1]) 
    assert uav_speed > 0, "Mode slideWindow assumes UAVs are moving linearly with speed uav_speed, which cannot be 0"
    if modulation == '16':
        modulation = "QAM16"
    elif modulation == '64':
        modulation = "QAM64"
    # Get MCS Index
    if modulation == "BPSK" and bitrate == 6.5:
        mcs_index = 0
    elif modulation == "QPSK" and bitrate == 13:
        mcs_index = 1
    elif modulation == "QPSK" and bitrate == 19.5:
        mcs_index = 2
    elif modulation == "QAM16" and bitrate == 26:
        mcs_index = 3
    elif modulation == "QAM16" and bitrate == 39:
        mcs_index = 4
    elif modulation == "QAM64" and bitrate == 52:
        mcs_index = 5
    elif modulation == "QAM64" and bitrate == 58.5:
        mcs_index = 6
    elif modulation == "QAM64" and bitrate == 65:
        mcs_index = 7
    else:
        mcs_index = np.nan

    dl_throughput_df_list = []
    ul_throughput_df_list = []
    vid_throughput_df_list = []
    runs = glob.glob("{}/Run-*_GCS-Tx.csv".format(scenario_path)) # Get the different runs for each scenario
    for run in runs:
        run_files = glob.glob("{}_*.csv".format(run[0:-11])) # Get list of csv files belonging to this scenario
        if vid_delay_threshold == 0:
            rx_df_list, tx_df_list, pd_df_list, mon_df_list = compile_micro_sim_data_v2(run_files, video=False)
        elif vid_delay_threshold > 0:
            rx_df_list, tx_df_list, pd_df_list, mon_df_list, video_df_list = compile_micro_sim_data_v2(run_files, video=True)

        # Sort out which df is which
        gcs_rx_df = rx_df_list[0]
        gcs_tx_df = tx_df_list[0]
        gcs_pd_df = pd_df_list[0]
        # uavs_rx_df = pd.concat(rx_df_list[1:len(rx_df_list)], ignore_index=True) # Includes GW Rx DF and all UAVs Rx DFs
        uavs_tx_df = pd.concat(tx_df_list[1:len(tx_df_list)], ignore_index=True) # Includes GW Tx DF and all UAVs Tx DFs
        uavs_pd_df = pd.concat(pd_df_list[1:len(pd_df_list)], ignore_index=True)
        
        # Get the throughput measures in DL FOR EACH INIVIDUAL UAV based on time slots
        max_time_uavs_rx = np.nanmax([df["RxTime"].max() for df in rx_df_list[1:len(rx_df_list)]])
        max_time = np.nanmax([gcs_tx_df["TxTime"].max(), max_time_uavs_rx, gcs_pd_df["RxTime"].max()])
        gcs_pd_df["Horizontal_Distance"] = gcs_pd_df["RxTime"] * uav_speed
        # uavs_rx_df["Horizontal_Distance"] = uavs_rx_df["RxTime"] * uav_speed
        # Remember, the RX DFs in rx_df_list is [GCS, Gateway, UAV-0, UAV-1, ...]
        uav_throughput_df_list = []
        for i in range(1,len(rx_df_list)):
            uav_rx_df = rx_df_list[i]
            uav_rx_df["Horizontal_Distance"] = uav_rx_df["RxTime"] * uav_speed
            uav_address = "192.168.0." + str(i+1) # 192.168.0.2 is for gateway, 192.168.0.3 is for UAV-0, so on ...
            pkts_tx_df = gcs_tx_df.loc[gcs_tx_df["Dest_Addr"] == uav_address] # DF of packets sent by GCS to this particular UAV
            pkts_tx_list = ["CNCData-{}".format(s) for s in pkts_tx_df["Packet_Seq"].values] # List of packets sent by GCS to this particular UAV
            pkts_pd_df = gcs_pd_df[gcs_pd_df["Packet_Name"].isin(pkts_tx_list)]
            uav_throughput_df = process_throughput_sliding_window_time_v2(uav_rx_df, pkts_pd_df, dl_slot_size, dl_stride, max_time, dl_delay_threshold)
            uav_throughput_df['Height'] = height
            uav_throughput_df['UAV_Sending_Interval'] = uav_sending_interval
            uav_throughput_df['MCS_Index'] = mcs_index
            uav_throughput_df['UAV_Speed'] = uav_speed
            uav_throughput_df_list.append(uav_throughput_df) # uav_throughput_df_list contains the throughput DF of [Gateway, UAV-0, UAV-1, ...]]
        dl_throughput_df_list.append(uav_throughput_df_list) # List of lists
        
        # Get the throughput measures in UL based on time slots
        max_time = np.nanmax([uavs_tx_df["TxTime"].max(), gcs_rx_df["RxTime"].max(), uavs_pd_df["RxTime"].max()])
        gcs_rx_df["Horizontal_Distance"] = gcs_rx_df["RxTime"] * uav_speed
        uavs_pd_df["Horizontal_Distance"] = uavs_pd_df["RxTime"] * uav_speed
        ul_throughput_df = process_throughput_sliding_window_time_v2(gcs_rx_df, uavs_pd_df, ul_slot_size, ul_stride, max_time, ul_delay_threshold)
        ul_throughput_df['Height'] = height
        ul_throughput_df['UAV_Sending_Interval'] = uav_sending_interval
        ul_throughput_df['MCS_Index'] = mcs_index
        ul_throughput_df['UAV_Speed'] = uav_speed
        ul_throughput_df_list.append(ul_throughput_df)

        # Get the throughput measures in video link based on time slots
        if vid_delay_threshold > 0:
            video_tx_df = video_df_list[0]
            video_rx_df = video_df_list[1]
            video_pd_df = video_df_list[2]
            max_time = np.nanmax([video_tx_df["TxTime"].max(), video_rx_df["RxTime"].max(), video_pd_df["RxTime"].max()])
            video_rx_df["Horizontal_Distance"] = video_rx_df["RxTime"] * uav_speed
            video_pd_df["Horizontal_Distance"] = video_pd_df["RxTime"] * uav_speed
            vid_throughput_df = process_throughput_sliding_window_time_v2(video_rx_df, video_pd_df, vid_slot_size, vid_stride, max_time, vid_delay_threshold)
            vid_throughput_df['Height'] = height
            vid_throughput_df['UAV_Sending_Interval'] = uav_sending_interval
            vid_throughput_df['MCS_Index'] = mcs_index
            vid_throughput_df['UAV_Speed'] = uav_speed
            vid_throughput_df_list.append(vid_throughput_df)

    # Create save file directory if not created
    if not os.path.isdir(os.path.join(save_path, scenario)):
        os.mkdir(os.path.join(save_path, scenario))

    num_uav = len(dl_throughput_df_list[0])
    for i in range(num_uav):
        dl_throughput_df = pd.concat([d[i] for d in dl_throughput_df_list])
        if i == 0: # This case is the Gateway UAV
            dl_throughput_df.to_csv(os.path.join(save_path, scenario, "Gateway_Downlink_Throughput.csv"), index=False)
        else:
            dl_throughput_df.to_csv(os.path.join(save_path, scenario, "UAV-{}_Downlink_Throughput.csv".format(i-1)), index=False)


    ul_throughput_df = pd.concat(ul_throughput_df_list)
    ul_throughput_df.to_csv(os.path.join(save_path, scenario, "Uplink_Throughput.csv"), index=False)
    
    if vid_delay_threshold > 0:
        vid_throughput_df = pd.concat(vid_throughput_df_list)
        vid_throughput_df.to_csv(os.path.join(save_path, scenario, "Video_Throughput.csv"), index=False)
    
    return
    
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

# def process_sim_data_v4(sim_root_path, stride, dl_slot_size, ul_slot_size, dl_delay_threshold, ul_delay_threshold, save_path, mode='timeSlot', vid_slot_size=1, vid_delay_threshold=0):
#     '''
#     Modified: Assuming the data for each scenario are in their own subdirectory under sim_root_path
#     mode: 'timeSlot' / 'slideWindow' / 'count' (The throughput measurement mode)
#     dl_slot_size: If mode=='timeSlot', this is the length of the slot size in seconds,
#                   If mode=='slideWindow', this is the length of the sliding window in no. of packets
#                   dl indicates Downlink. Likewise for ul (Uplink) and vid (Video)
    
#     '''
    
#     # Concatenates all UL & DL results from sim_root_path into a single df
#     scenario_list = [f.path for f in os.scandir(sim_root_path) if f.is_dir()] # Get list of "unique" scenarios

#     # For each scenario, extract the UL and DL raw data
#     GX_GCS = 0 # The x-coord of GCS (refer to OMNeT++ ini sim file) (set to zero if already subtracted in OMNeT++)
#     with Pool(15) as pool:
#         pool.starmap(process_scenario_v4, zip(scenario_list, repeat(save_path), repeat(stride), repeat(dl_slot_size), repeat(ul_slot_size), repeat(dl_delay_threshold), repeat(ul_delay_threshold), repeat(mode), repeat(vid_slot_size), repeat(vid_delay_threshold), repeat(GX_GCS)))
    
#     return 

if __name__ == "__main__":
    '''
    MAKE SURE TO SET "GX_GCS" and "sending_interval_range" in function process_sim_data_v2
    '''
    sim_root_path = "/media/research-student/One Touch/FANET Datasets/DJISpark_Measured_Throughput_10000Samples/data_test"
    save_path = "/media/research-student/One Touch/FANET Datasets/DJISpark_Measured_Throughput_10000Samples/data_test_processed_dl-5secWinSize1secStride_ul-1secWinSize100msecStride"
    # sim_root_path = "/home/wlau0003/Reuben_ws/FANET_Dataset/DJISpark_Throughput/data"
    # save_path = "/home/wlau0003/Reuben_ws/FANET_Dataset/DJISpark_Throughput/data_processed"
    
    mode = "timeSlot" # 'timeSlot' / 'slideWindow' / 'count' (The throughput measurement mode)
    dl_slot_size = 5 # Either the length of time slots, or the sliding window size. For count, this is the timeDiv
    ul_slot_size = 1 # Either the length of time slots, or the sliding window size
    vid_slot_size = 0 # Either the length of time slots, or the sliding window size

    dl_delay_threshold = 1
    ul_delay_threshold = 1
    vid_delay_threshold = 0 # NOTE: Set to zero if no video data

    dl_stride = 1 # 100ms stride
    ul_stride = 0.1 # 100ms stride
    vid_stride = 0 # 100ms stride
    
    scenario_list = [f.path for f in os.scandir(sim_root_path) if f.is_dir()] # Get list of "unique" scenarios

    # For each scenario, extract the UL and DL raw data
    GX_GCS = 0 # The x-coord of GCS (refer to OMNeT++ ini sim file) (set to zero if already subtracted in OMNeT++)
    with Pool(20) as pool:
        pool.starmap(process_scenario_v4, zip(scenario_list, repeat(save_path), repeat(dl_stride), repeat(ul_stride), repeat(dl_slot_size), repeat(ul_slot_size), repeat(dl_delay_threshold), repeat(ul_delay_threshold), repeat(mode), repeat(vid_stride), repeat(vid_slot_size), repeat(vid_delay_threshold), repeat(GX_GCS)))

    # process_sim_data_v4(sim_root_path, dl_slot_size=dl_slot_size, ul_slot_size=ul_slot_size, 
    #                     dl_delay_threshold=dl_delay_threshold, ul_delay_threshold=ul_delay_threshold, save_path=save_path, mode=mode,
    #                     vid_slot_size=vid_slot_size, vid_delay_threshold=vid_delay_threshold)

    
