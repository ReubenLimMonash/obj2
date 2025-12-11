'''
Date Modified: 25/05/2024
Desc: Pandas data processing of FANET dataset with multiprocessing.
      FOR RELIABILITY OF WAYPOINT MODE ONLY GCS BROADCAST (No Rebroadcasting or Retransmissions)
      Reads RX CSV file and calculates throughput for each time step
      Uses sliding window approach
      Modified: to processing DL GCS-2-UAV throughput individually per UAV, rather than combining everything
      Modified: For evaluating broadcast scenarios
      Modified: For saving results for each run separately
      Modified: Modified fanet_data_preprocessing_throughput_rebroadcast_07012024.py for high number of runs rather than high number of scenarios.
                Multiprocessing is done to compile the results from multiple runs in each scenario, rather than work on multiple scenarios in parallel
      Modified: To limit results up to a certain horizontal distance for waypoint mode.
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
    br_df_dtypes = {"TxTime": np.float32, "Packet_Name": "str", "Bytes": np.uint16, "Dest_Addr": 'category'}

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
    uavs_br_df_list = []
    uav_rx_files = [file for file in file_list if (('_UAV-' in file) and ('-Rx.csv' in file))]
    uav_tx_files = [file for file in file_list if (('_UAV-' in file) and ('-Tx.csv' in file))]
    uav_mon_files = [file for file in file_list if (('_UAV-' in file) and ('-Wlan.csv' in file))]
    uav_pd_files = [file for file in file_list if (('_UAV-' in file) and ('-PacketDrop.csv' in file))]
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

def count_packet_states(rx_df, pd_df, delay_threshold):
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

def process_scenario_broadcast(scenario_path, save_path, max_hdist_df, num_processes, 
                        dl_delay_threshold, ul_delay_threshold):
    '''
    Modified: To process all different runs of each sccenario (for measured throughput while UAV moving)
    Modified: For broadcast scenarios
    dl indicates Downlink. Likewise for ul (Uplink) and vid (Video)
    slot_size: the length of the slot size in seconds
    stride: The sliding window will slide by "stride" seconds
    delay_threshold: Delay Constraint for reliability calculation
    '''
    scenario = scenario_path.split("/")[-1]
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
    # if modulation == "BPSK" and bitrate == 6.5:
    #     mcs_index = 0
    # elif modulation == "QPSK" and bitrate == 13:
    #     mcs_index = 1
    # elif modulation == "QPSK" and bitrate == 19.5:
    #     mcs_index = 2
    # elif modulation == "QAM16" and bitrate == 26:
    #     mcs_index = 3
    # elif modulation == "QAM16" and bitrate == 39:
    #     mcs_index = 4
    # elif modulation == "QAM64" and bitrate == 52:
    #     mcs_index = 5
    # elif modulation == "QAM64" and bitrate == 58.5:
    #     mcs_index = 6
    # elif modulation == "QAM64" and bitrate == 65:
    #     mcs_index = 7
    # else:
    #     mcs_index = np.nan

    # Get the max hdist for this scenario
    if max_hdist_df == None:
        max_hdist = None
    else:
        max_hdist = max_hdist_df.loc[(max_hdist_df["Height"] == height) & (max_hdist_df["Bitrate"] == bitrate)]["Max_Horizontal_Distance"].values[0]

    # Create save file directory if not created
    if not os.path.isdir(os.path.join(save_path, scenario)):
        os.mkdir(os.path.join(save_path, scenario))

    metric_list = []
    runs = sorted(glob.glob("{}/Run-*_GCS-Tx.csv".format(scenario_path))) # Get the different runs for each scenario
    # run_number = np.arange(len(runs)) # This assumes that the runs are in ascending order with no breaks in between and starts at 0
    run_number = [run.split('/')[-1].split("_")[0].split("-")[-1] for run in runs]
    run_file_path = os.path.join(scenario_path, "Run-{}_*.csv")
    with Pool(num_processes) as pool:
        for result in pool.starmap(process_run, zip(run_number, repeat(run_file_path), repeat(uav_speed), repeat(max_hdist), 
                                                    repeat(dl_delay_threshold), repeat(ul_delay_threshold))):
            
            metric_list.append({"Run": result[0], "Num_Pkts_Reliable_DL": result[1], "Num_Pkts_Reliable_UL": result[2], 
                            "Useful_Packet_Reception_Ratio_DL": result[5], "Total_Reliability_DL": result[3], "Total_Reliability_UL": result[4], 
                            "DL_Num_Delay_Excd": result[6], "UL_Num_Delay_Excd": result[7], "UL_Num_Incr_Rcvd": result[8], "UL_Num_Queue_Overflow": result[9]})
    # Save metrics of each run to file
    metric_df = pd.DataFrame(metric_list)
    metric_df.to_csv(os.path.join(save_path, scenario, "Broadcast_Metrics.csv"), index=False)
    
    return
    
def process_run(run_number, file_path, uav_speed, max_hdist,
                dl_delay_threshold, ul_delay_threshold):
    '''
    Provide file_path such that the files for a particular run number can be identified using glob.glob(file_path.format(run_number))
    '''
    run_files = glob.glob(file_path.format(run_number)) # Get list of csv files belonging to this scenario

    rx_df_list, tx_df_list, pd_df_list, mon_df_list = compile_micro_sim_data_v2(run_files)

    # Sort out which df is which
    gcs_rx_df = rx_df_list[0]
    gcs_tx_df = tx_df_list[0]
    gcs_pd_df = pd_df_list[0]
    # uavs_rx_df = pd.concat(rx_df_list[1:len(rx_df_list)], ignore_index=True) # Includes GW Rx DF and all UAVs Rx DFs
    uavs_tx_df = pd.concat(tx_df_list[1:len(tx_df_list)], ignore_index=True) # Includes GW Tx DF and all UAVs Tx DFs
    uavs_pd_df = pd.concat(pd_df_list[1:len(pd_df_list)], ignore_index=True)

    if max_hdist != None:
        # Get the throughput measures in DL FOR EACH INIVIDUAL UAV based on time slots
        max_time = max_hdist / uav_speed # Use max_hdist to determine max_time
        # Filter DFs by max_time
        gcs_tx_df = gcs_tx_df.loc[gcs_tx_df["TxTime"] <= max_time]
        gcs_rx_df = gcs_rx_df.loc[gcs_rx_df["TxTime"] <= max_time]
        uavs_tx_df = uavs_tx_df.loc[uavs_tx_df["TxTime"] <= max_time]
        uavs_pd_df = uavs_pd_df.loc[uavs_pd_df["RxTime"] <= max_time] # Filtering by RxTime instead because TxTime is 0

    # Remember, the RX DFs in rx_df_list is [GCS, Gateway, UAV-0, UAV-1, ...]
    dl_num_rcvd = 0
    dl_num_reliable = 0
    dl_num_delay_excd = 0
    for i in range(1,len(rx_df_list)):
        uav_rx_df = rx_df_list[i].copy()
        if max_hdist != None:
            uav_rx_df = uav_rx_df.loc[uav_rx_df["TxTime"] <= max_time] # Filter by max time
        if not uav_rx_df.empty:
            dl_num_rcvd += len(uav_rx_df) 
            # Drop duplicate packets at Rx DF, keeping the first arrived
            uav_rx_df = uav_rx_df.sort_values(["Packet_Name", "RxTime"], ascending=[True, True])
            uav_rx_df = uav_rx_df.drop_duplicates(subset='Packet_Name', keep="first")
            uav_rx_df["Delay"] = uav_rx_df['RxTime']-uav_rx_df['TxTime'] # Calc delay of packets received
            uav_rx_df_reliable = uav_rx_df.loc[(uav_rx_df["Delay"] <= dl_delay_threshold)]
            dl_num_reliable += len(uav_rx_df_reliable) # Why not calc this from process_throughput_sliding_window_time_broadcast? Because of possibly overlapping time windows
            dl_num_delay_excd += len(uav_rx_df) - len(uav_rx_df_reliable)

    # Remove packets broadcasted by GCS in gcs_rx_df and uavs_pd_df (since it is not part of the UL flow)
    gcs_rx_df = gcs_rx_df.loc[['CNCData' not in name for name in gcs_rx_df.Packet_Name]].copy()
    uavs_pd_df = uavs_pd_df.loc[['CNCData' not in name for name in uavs_pd_df.Packet_Name]].copy()
    
    ''' Get metric of broadcast protocol '''
    # Downlink
    num_sent_dl = len(gcs_tx_df) * (len(rx_df_list)-1)
    total_reliability_dl = dl_num_reliable / num_sent_dl

    # Uplink
    ul_num_reliable, ul_num_delay_excd, ul_num_incr_rcvd, ul_num_queue_overflow = count_packet_states(gcs_rx_df, uavs_pd_df, ul_delay_threshold)
    num_sent_ul = len(uavs_tx_df)
    total_reliability_ul = ul_num_reliable / num_sent_ul
    if dl_num_rcvd > 0:
        uprr_dl = dl_num_reliable / dl_num_rcvd # Useful_Packet_Reception_Ratio_DL
    else:
        uprr_dl = np.nan

    return (run_number, dl_num_reliable, ul_num_reliable, total_reliability_dl, total_reliability_ul, uprr_dl,
            dl_num_delay_excd, ul_num_delay_excd, ul_num_incr_rcvd, ul_num_queue_overflow)

if __name__ == "__main__":
    '''
    MAKE SURE TO SET "GX_GCS" and "sending_interval_range" in function process_sim_data_v2
    '''
    # MAX_HDIST_FILE = "/home/research-student/omnet-fanet/data-processing-scripts/waypoint_max_hdist/Waypoint_Max_HDist_fm_Sim_50.csv"
    # MAX_HDIST_FILE = "/home/rlim0005/waypoint_max_hdist/Waypoint_Max_HDist_fm_Sim_50.csv"
    # max_hdist_df = pd.read_csv(MAX_HDIST_FILE)
    max_hdist_df = None

    sim_root_paths = ["/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/default",
                      "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case1",
                      "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case2",
                      "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case3",
                      "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case4",
                      "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case5",
                      "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case6",
                      "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case7",
                      "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case8",
                      "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case9",
                      "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case10"]
    save_paths = ["/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/default_processed",
                  "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case1_processed",
                    "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case2_processed",
                    "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case3_processed",
                    "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case4_processed",
                    "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case5_processed",
                    "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case6_processed",
                    "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case7_processed",
                    "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case8_processed",
                    "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case9_processed",
                    "/home/rlim0005/FANET_Dataset/DJISpark_Waypoint_Only_Broadcast_UAV_Interference/case10_processed"]
    
    dl_delay_threshold = 1
    ul_delay_threshold = 1

    for sim_root_path, save_path in zip(sim_root_paths, save_paths):
        # Create save file directory if not created
        if not os.path.isdir(os.path.join(save_path)):
            os.mkdir(os.path.join(save_path))

        scenario_list = [f.path for f in os.scandir(sim_root_path) if f.is_dir()] # Get list of "unique" scenarios
        num_processes = 64
        # For each scenario, extract the UL and DL raw data
        for scenario_path in scenario_list:
            process_scenario_broadcast(scenario_path, save_path, max_hdist_df, num_processes, dl_delay_threshold, ul_delay_threshold)


    
