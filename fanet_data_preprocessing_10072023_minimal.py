'''
Date: 10/07/2023
Desc: Pandas data processing of FANET dataset with multiprocessing.
      V2 - Uses categorical data type for packet name
      Modified to process for 'hovering' dataset
      Modified for new traffic model
      Modified for more efficient packet drop counting & throuhghput calculation
      Modified for storing only counts of reliable, delay excd, incr rcvd and q overflow packets
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

    return rx_df_list, tx_df_list, pd_df_list, mon_df_list

def process_micro_sim_reliability_data(rx_df, pd_df, delay_threshold):
    """
    Desc: Count the number of reliable, delay excd, incr rcvd and q overflow packets for each micro-sim
    ASSUMPTION: The total number of packets in uavs_rx_df and gcs_pd_df makes up the total number of packets transmitted in downlink (gcs_tx_df)
                RETRY_LIMIT_REACHED is due to incorrectly received in downlink
    """
    # Make sure that the packets recorded as retry limit reach and queue overflow in pd_df are not in fact received in rx_df
    rx_packets = rx_df["Packet_Name"].values
    pd_df = pd_df.loc[~pd_df["Packet_Name"].isin(rx_packets)]

    delay = rx_df['RxTime']-rx_df['TxTime']
    num_reliable = np.sum(np.where(delay > delay_threshold , 0, 1))
    num_delay_excd = len(rx_df) - num_reliable
    pkt_drop_counts = pd_df["Packet_Drop_Reason"].value_counts()
    if len(pkt_drop_counts) > 2:
        print("ALERT: More than 2 packet drop reason found!")
    
    if "RETRY_LIMIT_REACHED" in pkt_drop_counts:
        num_incr_rcvd = pkt_drop_counts["RETRY_LIMIT_REACHED"]
    else:
        num_incr_rcvd = 0
    if "QUEUE_OVERFLOW" in pkt_drop_counts:
        num_queue_overflow = pkt_drop_counts["QUEUE_OVERFLOW"]
    else:
        num_queue_overflow = 0

    return num_reliable, num_delay_excd, num_incr_rcvd, num_queue_overflow

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

def process_scenario_v3(scenario, sim_root_path, dl_delay_threshold, ul_delay_threshold, save_path, GX_Offset=0):
    '''
    GX_Offset is the x-coordinate offset for h_dist in scenario file name
    '''

    print(scenario)
    
    scenario_files = glob.glob(sim_root_path + "/{}_*.csv".format(scenario)) # Get list of csv files belonging to this scenario
    scenario_params = scenario.split('_')
    height = int(scenario_params[2].split('-')[-1]) 
    h_dist = float(scenario_params[3].split('-')[-1]) - GX_Offset 
    modulation = scenario_params[4].split('-')[-1]
    uav_sending_interval = int(scenario_params[5].split('-')[-1])
    if modulation == '16':
        modulation = "QAM16"
    elif modulation == '64':
        modulation = "QAM64"

    # Calculate the mean and std dev of SINR for this scenario
    mean_sinr, std_dev_sinr = sinr_lognormal_approx(h_dist, height, env='suburban')

    rx_df_list, tx_df_list, pd_df_list, mon_df_list = compile_micro_sim_data_v2(scenario_files)

    # Sort out which df is which
    gcs_tx_df = tx_df_list[0]
    gcs_rx_df = rx_df_list[0]
    gcs_pd_df = pd_df_list[0]
    uavs_tx_df = pd.concat(tx_df_list[1:len(tx_df_list)], ignore_index=True)
    uavs_rx_df = pd.concat(rx_df_list[1:len(rx_df_list)], ignore_index=True)
    uavs_pd_df = pd.concat(pd_df_list[1:len(rx_df_list)], ignore_index=True)
    
    # Get the count of each packet state in DL
    num_reliable, num_delay_excd, num_incr_rcvd, num_queue_overflow = process_micro_sim_reliability_data(uavs_rx_df, gcs_pd_df, dl_delay_threshold)
    num_pkts_sent = len(gcs_tx_df)
    if (num_pkts_sent != (num_reliable + num_delay_excd + num_incr_rcvd + num_queue_overflow)):
        print("ALERT: No. packets recorded in DL Tx DF does not match no. packets recorded in Rx DF and PD DF")
        print(scenario)
        num_pkts_sent = num_reliable + num_delay_excd + num_incr_rcvd + num_queue_overflow
    dl_result = {"Horizontal_Distance": h_dist, "Height": height, "UAV_Sending_Interval": uav_sending_interval, "Modulation": modulation, "Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr,
                 "Num_Sent": num_pkts_sent, "Num_Reliable": num_reliable, "Num_Delay_Excd": num_delay_excd, "Num_Incr_Rcvd": num_incr_rcvd, "Num_Q_Overflow": num_queue_overflow}
    
    # Get the count of each packet state in UL
    num_reliable, num_delay_excd, num_incr_rcvd, num_queue_overflow = process_micro_sim_reliability_data(gcs_rx_df, uavs_pd_df, ul_delay_threshold)
    num_pkts_sent = len(uavs_tx_df)
    if (num_pkts_sent != (num_reliable + num_delay_excd + num_incr_rcvd + num_queue_overflow)):
        print("ALERT: No. packets recorded in UL Tx DF does not match no. packets recorded in Rx DF and PD DF")
        print("Scenario:" + scenario)
        print(num_pkts_sent)
        num_pkts_sent = num_reliable + num_delay_excd + num_incr_rcvd + num_queue_overflow
        print(num_pkts_sent)
    ul_result = {"Horizontal_Distance": h_dist, "Height": height, "UAV_Sending_Interval": uav_sending_interval, "Modulation": modulation, "Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr,
                 "Num_Sent": num_pkts_sent, "Num_Reliable": num_reliable, "Num_Delay_Excd": num_delay_excd, "Num_Incr_Rcvd": num_incr_rcvd, "Num_Q_Overflow": num_queue_overflow}
    return (dl_result, ul_result)

def process_sim_data_v3(sim_root_path, dl_delay_threshold, ul_delay_threshold, save_path):
    # Concatenates all UL & DL results from sim_root_path into a single df
    scenario_list = [csv.split('/')[-1][0:-11] for csv in glob.glob(sim_root_path + "/*GCS-Tx.csv")] # Get list of "unique" scenarios
    # For each scenario, extract the UL and DL raw data
    GX_GCS = 0 # The x-coord of GCS (refer to OMNeT++ ini sim file) (set to zero if already subtracted in OMNeT++)
    dl_results = []
    ul_results = []
    with Pool(30) as pool:
        for result in pool.starmap(process_scenario_v3, zip(scenario_list, repeat(sim_root_path), repeat(dl_delay_threshold), repeat(ul_delay_threshold), repeat(save_path), repeat(GX_GCS))):
            dl_results.append(result[0])
            ul_results.append(result[1])
    # Save results to CSV
    dl_df = pd.DataFrame(dl_results)
    dl_df.sort_values(["Modulation", "UAV_Sending_Interval", "Height", "Horizontal_Distance"], inplace=True)
    dl_df.to_csv(save_path + "Downlink_Results.csv", index=False)
    ul_df = pd.DataFrame(ul_results)
    ul_df.sort_values(["Modulation", "UAV_Sending_Interval", "Height", "Horizontal_Distance"], inplace=True)
    ul_df.to_csv(save_path + "Uplink_Results.csv", index=False)
    return 

if __name__ == "__main__":
    '''
    MAKE SURE TO SET "GX_GCS" and "sending_interval_range" in function process_sim_data_v2
    '''
    sim_root_path = "/media/research-student/One Touch/FANET Datasets/Dataset_NP100000_MultiModulation_Hovering_NoVideo/Test/Test_Dataset_1_200000"
    save_path = "/media/research-student/One Touch/FANET Datasets/Dataset_NP100000_MultiModulation_Hovering_NoVideo/Test_Dataset_1_200000"
    # sim_root_path = "/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/Test/Test_Dataset_1_100000"
    # save_path = "/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP10000_MultiModulation_Hovering_NoVideo/Test/Test_Dataset_1_100000_processed"
    dl_delay_threshold = 1
    ul_delay_threshold = 1
    process_sim_data_v3(sim_root_path, dl_delay_threshold=dl_delay_threshold, ul_delay_threshold=ul_delay_threshold, save_path=save_path)

    
