# Date: 14/08/2024
# Desc: To test OCSVM accuracy in no interference / UAV interference / MANET interference
#       Tests each run individually instead of combining all test throughput samples
# NOTE: Each run is simulated up to d_max. No filtering by hdist needed

import pandas as pd
import numpy as np 
import math
import os
import glob
from pickle import load
from tqdm import tqdm
from sklearn.svm import OneClassSVM
from scipy import special
from pandarallel import pandarallel

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

# def get_measured_throughput(sim_root_path, link="Downlink", single_path = False):
#     '''
#     Function to load the processed measured throughput data from CSV files stored in different subdirs in sim_root_path
#     Modified: The throughput files for each UAV and the GCS are stored separately (rather than single Uplink/Downlink) and separated by runs.
#     '''
#     assert link in ["Downlink", "Uplink", "Video"], 'link must be one of "Downlink", "Uplink", "Video"'
#     df_list = []
#     if single_path:
#         scenario_list = [sim_root_path]
#     else:
#         scenario_list = [f.path for f in os.scandir(sim_root_path) if f.is_dir()] # Get list of "unique" scenarios
#     for scenario in tqdm(scenario_list):
#         # Get the measured throughput samples for UL/DL/Vid under this scenario
#         if link == "Downlink":
#             throughput_files = glob.glob(os.path.join(scenario, "Run-*_Downlink_Throughput.csv"))
#         elif link == "Uplink":
#             throughput_files = glob.glob(os.path.join(scenario, "Run-*_Uplink_Throughput.csv"))
#         elif link == "Video":
#             throughput_files = glob.glob(os.path.join(scenario, "Run-*_Video_Throughput.csv"))
        
#         for file in throughput_files:
#             measured_df = pd.read_csv(file)
#             df_list.append(measured_df)

#     return pd.concat(df_list)

def normalize_data(df_in, columns, save_details_path=None):
    '''
    columns: The pandas data columns to normalize, given as a list of column names
    '''
    df = df_in.copy()
    # Define the ranges of parametrers
    max_mean_sinr = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
    max_std_dev_sinr = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
    min_mean_sinr = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
    min_std_dev_sinr = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)
    max_height = 300
    min_height = 60
    max_h_dist = 1200
    min_h_dist = 0
    max_mcs = 7
    min_mcs = 0
    max_uav_speed = 26
    min_uav_speed = 0

    # Normalize data (Min Max Normalization between [-1,1])
    if "Height" in columns:
        df["Height"] = df["Height"].apply(lambda x: 2*(x-min_height)/(max_height-min_height) - 1)
    if "U2G_H_Dist" in columns:
        df["U2G_H_Dist"] = df["U2G_H_Dist"].apply(lambda x: 2*(x-min_h_dist)/(max_h_dist-min_h_dist) - 1)
    if "Horizontal_Distance" in columns:
        df["Horizontal_Distance"] = df["Horizontal_Distance"].apply(lambda x: 2*(x-min_h_dist)/(max_h_dist-min_h_dist) - 1)
    if "Mean_SINR" in columns:
        df["Mean_SINR"] = df["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
    if "Std_Dev_SINR" in columns:
        df["Std_Dev_SINR"] = df["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1) # Convert to dB space
    if "UAV_Sending_Interval" in columns:
        df["UAV_Sending_Interval"] = df["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 66.7: 0.5, 100:1, 1000:2})
    if "Packet_State" in columns:
        df['Packet_State'] = df['Packet_State'].replace({"Reliable":0, "QUEUE_OVERFLOW":1, "RETRY_LIMIT_REACHED":2, "Delay_Exceeded":3})
    if "MCS" in columns:
        df["MCS"] = df["MCS"].apply(lambda x: 2*(x-min_mcs)/(max_mcs-min_mcs) - 1)
    if "MCS_Index" in columns:
        df["MCS_Index"] = df["MCS_Index"].apply(lambda x: 2*(x-min_mcs)/(max_mcs-min_mcs) - 1)
    if "UAV_Speed" in columns:
        df["UAV_Speed"] = df["UAV_Speed"].apply(lambda x: 2*(x-min_uav_speed)/(max_uav_speed-min_uav_speed) - 1)

    # Record details of inputs and output for model
    if save_details_path is not None:
        f = open(os.path.join(save_details_path,"model_details.txt"), "w")
        f.write("Max Height (m): {}\n".format(max_height))
        f.write("Min Height (m): {}\n".format(min_height))
        f.write("Max H_Dist (m): {}\n".format(max_h_dist))
        f.write("Min H_Dist (m): {}\n".format(min_h_dist))
        f.write("Max Mean SINR (dB): {}\n".format(max_mean_sinr))
        f.write("Min Mean SINR (dB): {}\n".format(min_mean_sinr))
        f.write("Max Std Dev SINR (dB): {}\n".format(max_std_dev_sinr))
        f.write("Min Std Dev SINR (dB): {}\n".format(min_std_dev_sinr))
        f.write("Max UAV Speed (m/s): {}\n".format(max_uav_speed))
        f.write("Min UAV Speed (m/s): {}\n".format(min_uav_speed))
        f.write("[BPSK: 1, QPSK: 0.3333, QAM16: -0.3333, QAM64: -1]\n")
        f.write("UAV Sending Interval: [10:-1, 20:-0.5, 40:0, 66.7: 0.5, 100:1, 1000:2]\n")
        f.write("Output: ['Reliable':0, 'QUEUE_OVERFLOW':1, 'RETRY_LIMIT_REACHED':2, 'Delay_Exceeded':3]\n")
        f.close()

    return df

def get_mcs_index(df_in):
    '''
    Gets the MCS index based on modulation and bitrate column of the df_in
    '''
    df = df_in.copy()
    df["MCS"] = ''
    df.loc[(df["Modulation"] == "BPSK") & (df["Bitrate"] == 6.5), "MCS"] = 0 # MCS Index 0
    df.loc[(df["Modulation"] == "QPSK") & (df["Bitrate"] == 13), "MCS"] = 1 # MCS Index 1
    df.loc[(df["Modulation"] == "QPSK") & (df["Bitrate"] == 19.5), "MCS"] = 2 # MCS Index 2
    df.loc[(df["Modulation"] == "QAM16") & (df["Bitrate"] == 26), "MCS"] = 3 # MCS Index 3
    df.loc[(df["Modulation"] == "QAM16") & (df["Bitrate"] == 39), "MCS"] = 4 # MCS Index 4
    df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 52), "MCS"] = 5 # MCS Index 5
    df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 58.5), "MCS"] = 6 # MCS Index 6
    df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 65), "MCS"] = 7 # MCS Index 7

    return df

def norm_MCS(mcs_index):
    return 2*mcs_index/7 - 1

def get_MCS_bitrate(mcs_index):
    mcs_bitrate = {0: 6.5, 1: 13, 2: 19.5, 3: 26, 4: 39, 5: 52, 6: 58.5, 7: 65}
    return mcs_bitrate[mcs_index]

def get_MCS_index(mcs_bitrate):
    mcs_index = {6.5: 0, 13: 1, 19.5: 2, 26: 3, 39: 4, 52: 5, 58.5: 6, 65: 7}
    return mcs_index[mcs_bitrate]

if __name__ == "__main__":
    # pandarallel.initialize(progress_bar=False)
    ''' Define Paths Here'''
    OCSVM_MODEL_PATH = "/media/research-student/KingstonSSD/ocsvm_models"
    ROBUST_SCALER_PATH = "/media/research-student/KingstonSSD/ocsvm_models"
    SAVE_PATH = "/media/research-student/KingstonSSD/ocsvm_models"
    DATASET_NO_INT_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_ocsvm_test_no_int_processed" 
    DATASET_UAV_INT_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_ocsvm_test_uav_int_processed"
    DATASET_MANET_INT_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_ocsvm_test_manet_int_processed"
    RELIABILITY_TH = 0.99 # Threshold to evaluate interference scenarios
    pandarallel.initialize(progress_bar=False)

    # """ NO INT """
    # print("Testing No Int Scenarios")
    # scenarios_no_int = [x[0] for x in os.walk(DATASET_NO_INT_PATH)]
    # scenario_results = []
    # for scenario in tqdm(scenarios_no_int):
    #     if scenario == DATASET_NO_INT_PATH: # This is the root path
    #         continue
    #     # Get the parameters of the scenario
    #     scenario_name = scenario.split("/")[-1]
    #     params = scenario_name.split("_")
    #     bitrate = [x for x in params if "BitRate" in x][0].split('-')[-1]
    #     usi = [x for x in params if "UAVSendingInterval" in x][0].split('-')[-1]
    #     uavspeed = int([x for x in params if "UAVSpeed" in x][0].split('-')[-1])
    #     height = int([x for x in params if "Height" in x][0].split('-')[-1])
    #     mcs_index = get_MCS_index(float(bitrate))
    #     # Load OCSVM Model and Robust Scaler
    #     ocsvm_model_dl = load(open(os.path.join(OCSVM_MODEL_PATH, "ocsvm_Downlink_USI-{}_MCS-{}.pkl".format(usi, mcs_index)), 'rb'))
    #     ocsvm_model_ul = load(open(os.path.join(OCSVM_MODEL_PATH, "ocsvm_Uplink_USI-{}_MCS-{}.pkl".format(usi, mcs_index)), 'rb'))
    #     ocsvm_model_vid = load(open(os.path.join(OCSVM_MODEL_PATH, "ocsvm_Video_USI-{}_MCS-{}.pkl".format(usi, mcs_index)), 'rb'))
    #     robust_scaler_dl = load(open(os.path.join(ROBUST_SCALER_PATH, "robust_scaler_Downlink_USI-{}_MCS-{}.pkl".format(usi, mcs_index)), 'rb'))
    #     robust_scaler_ul = load(open(os.path.join(ROBUST_SCALER_PATH, "robust_scaler_Uplink_USI-{}_MCS-{}.pkl".format(usi, mcs_index)), 'rb'))
    #     robust_scaler_vid = load(open(os.path.join(ROBUST_SCALER_PATH, "robust_scaler_Video_USI-{}_MCS-{}.pkl".format(usi, mcs_index)), 'rb'))
    #     # Process each run
    #     ul_run_results = []
    #     dl_run_results = []
    #     vid_run_results = []
    #     # Each run should only have one uplink throughput, so use it to determine no. of runs
    #     uplink_throughput_files = glob.glob(os.path.join(scenario, "Run-*_Uplink_Throughput.csv"))
    #     downlink_throughput_files = glob.glob(os.path.join(scenario, "Run-*_Downlink_Throughput.csv"))
    #     overall_int_detected = 0
    #     for run_num in range(len(uplink_throughput_files)):
    #         ul_throughput_df = pd.read_csv(os.path.join(scenario, "Run-{}_Uplink_Throughput.csv".format(run_num)))
    #         vid_throughput_df = pd.read_csv(os.path.join(scenario, "Run-{}_Video_Throughput.csv".format(run_num)))
    #         dl_throughput_list = [l for l in downlink_throughput_files if "Run-{}_".format(run_num) in l]
    #         dl_df_list = []
    #         for file in dl_throughput_list:
    #             df = pd.read_csv(file)
    #             dl_df_list.append(df)
    #         dl_throughput_df = pd.concat(dl_df_list)
    #         # Normalize data
    #         # Downlink Throughput
    #         throughput = np.array(dl_throughput_df["Throughput"].values)
    #         dl_throughput_df["Throughput_Norm"] = robust_scaler_dl.transform(throughput.reshape(-1,1))           
    #         dl_throughput_df[['Mean_SINR',"Std_Dev_SINR"]] = dl_throughput_df.parallel_apply(lambda row: sinr_lognormal_approx(row['Horizontal_Distance'],row['Height']),axis=1,result_type='expand')
    #         dl_throughput_df = normalize_data(dl_throughput_df, columns=["Mean_SINR", "Std_Dev_SINR"])
    #         # Uplink Throughput
    #         throughput = np.array(ul_throughput_df["Throughput"].values)
    #         ul_throughput_df["Throughput_Norm"] = robust_scaler_ul.transform(throughput.reshape(-1,1))           
    #         ul_throughput_df[['Mean_SINR',"Std_Dev_SINR"]] = ul_throughput_df.parallel_apply(lambda row: sinr_lognormal_approx(row['Horizontal_Distance'],row['Height']),axis=1,result_type='expand')
    #         ul_throughput_df = normalize_data(ul_throughput_df, columns=["Mean_SINR", "Std_Dev_SINR"])
    #         # Video Throughput
    #         throughput = np.array(vid_throughput_df["Throughput"].values)
    #         vid_throughput_df["Throughput_Norm"] = robust_scaler_vid.transform(throughput.reshape(-1,1))           
    #         vid_throughput_df[['Mean_SINR',"Std_Dev_SINR"]] = vid_throughput_df.parallel_apply(lambda row: sinr_lognormal_approx(row['Horizontal_Distance'],row['Height']),axis=1,result_type='expand')
    #         vid_throughput_df = normalize_data(vid_throughput_df, columns=["Mean_SINR", "Std_Dev_SINR"])
    #         # Testing OCSVM Model
    #         # Downlink
    #         X_test_no_int = dl_throughput_df[["Mean_SINR", "Std_Dev_SINR", "Throughput_Norm"]].values
    #         y_pred_no_int_ocsvm = ocsvm_model_dl.predict(X_test_no_int)
    #         y_pred_no_int_ocsvm = (y_pred_no_int_ocsvm + 1) // 2 # To convert the range from -1:1 to 0:1 (0 for outlier, 1 for inlier)
    #         dl_accuracy_score_ocsvm = np.sum(y_pred_no_int_ocsvm) / len(X_test_no_int) # Ground truth is all 1 for normal test data
    #         dl_num_samples = len(X_test_no_int)
    #         if dl_accuracy_score_ocsvm < 1: 
    #             dl_int_detected = 1
    #         else:
    #             dl_int_detected = 0
    #         dl_run_results.append({"run_num": run_num, "dl_accuracy_score_ocsvm": dl_accuracy_score_ocsvm, "dl_num_samples": dl_num_samples, "dl_int_detected": dl_int_detected})
    #         # Uplink
    #         X_test_no_int = ul_throughput_df[["Mean_SINR", "Std_Dev_SINR", "Throughput_Norm"]].values
    #         y_pred_no_int_ocsvm = ocsvm_model_ul.predict(X_test_no_int)
    #         y_pred_no_int_ocsvm = (y_pred_no_int_ocsvm + 1) // 2 # To convert the range from -1:1 to 0:1 (0 for outlier, 1 for inlier)
    #         ul_accuracy_score_ocsvm = np.sum(y_pred_no_int_ocsvm) / len(X_test_no_int) # Ground truth is all 1 for normal test data
    #         ul_num_samples = len(X_test_no_int)
    #         if ul_accuracy_score_ocsvm < 1: 
    #             ul_int_detected = 1
    #         else:
    #             ul_int_detected = 0
    #         ul_run_results.append({"run_num": run_num, "ul_accuracy_score_ocsvm": ul_accuracy_score_ocsvm, "ul_num_samples": ul_num_samples, "ul_int_detected": ul_int_detected})
    #         # Video
    #         X_test_no_int = vid_throughput_df[["Mean_SINR", "Std_Dev_SINR", "Throughput_Norm"]].values
    #         y_pred_no_int_ocsvm = ocsvm_model_vid.predict(X_test_no_int)
    #         y_pred_no_int_ocsvm = (y_pred_no_int_ocsvm + 1) // 2 # To convert the range from -1:1 to 0:1 (0 for outlier, 1 for inlier)
    #         vid_accuracy_score_ocsvm = np.sum(y_pred_no_int_ocsvm) / len(X_test_no_int) # Ground truth is all 1 for normal test data
    #         vid_num_samples = len(X_test_no_int)
    #         if vid_accuracy_score_ocsvm < 1: 
    #             vid_int_detected = 1
    #         else:
    #             vid_int_detected = 0
    #         vid_run_results.append({"run_num": run_num, "vid_accuracy_score_ocsvm": vid_accuracy_score_ocsvm, "vid_num_samples": vid_num_samples, "vid_int_detected": vid_int_detected})

    #         # Check for overall interference detection
    #         if dl_int_detected == 1 or ul_int_detected == 1 or vid_int_detected == 1:
    #             overall_int_detected += 1
            
    #     # Record run results
    #     dl_run_results_df = pd.DataFrame(dl_run_results)
    #     dl_acc_avg = dl_run_results_df["dl_accuracy_score_ocsvm"].mean()
    #     dl_acc_max = dl_run_results_df["dl_accuracy_score_ocsvm"].max()
    #     dl_acc_min = dl_run_results_df["dl_accuracy_score_ocsvm"].min()
    #     dl_acc_std_dev = dl_run_results_df["dl_accuracy_score_ocsvm"].std()
    #     dl_num_samples_avg = dl_run_results_df["dl_num_samples"].mean()
    #     dl_percent_int_detected = dl_run_results_df["dl_int_detected"].sum() / len(dl_run_results_df)
    #     dl_run_results_df["dl_num_accuracy_samples"] = dl_run_results_df["dl_accuracy_score_ocsvm"] * dl_run_results_df["dl_num_samples"]
    #     dl_overall_acc = dl_run_results_df["dl_num_accuracy_samples"].sum() / dl_run_results_df["dl_num_samples"].sum() # Overall accuracy considers all samples from all runs

    #     ul_run_results_df = pd.DataFrame(ul_run_results)
    #     ul_acc_avg = ul_run_results_df["ul_accuracy_score_ocsvm"].mean()
    #     ul_acc_max = ul_run_results_df["ul_accuracy_score_ocsvm"].max()
    #     ul_acc_min = ul_run_results_df["ul_accuracy_score_ocsvm"].min()
    #     ul_acc_std_dev = ul_run_results_df["ul_accuracy_score_ocsvm"].std()
    #     ul_num_samples_avg = ul_run_results_df["ul_num_samples"].mean()
    #     ul_percent_int_detected = ul_run_results_df["ul_int_detected"].sum() / len(ul_run_results_df)
    #     ul_run_results_df["ul_num_accuracy_samples"] = ul_run_results_df["ul_accuracy_score_ocsvm"] * ul_run_results_df["ul_num_samples"]
    #     ul_overall_acc = ul_run_results_df["ul_num_accuracy_samples"].sum() / ul_run_results_df["ul_num_samples"].sum() # Overall accuracy considers all samples from all runs

    #     vid_run_results_df = pd.DataFrame(vid_run_results)
    #     vid_acc_avg = vid_run_results_df["vid_accuracy_score_ocsvm"].mean()
    #     vid_acc_max = vid_run_results_df["vid_accuracy_score_ocsvm"].max()
    #     vid_acc_min = vid_run_results_df["vid_accuracy_score_ocsvm"].min()
    #     vid_acc_std_dev = vid_run_results_df["vid_accuracy_score_ocsvm"].std()
    #     vid_num_samples_avg = vid_run_results_df["vid_num_samples"].mean()
    #     vid_percent_int_detected = vid_run_results_df["vid_int_detected"].sum() / len(vid_run_results_df)
    #     vid_run_results_df["vid_num_accuracy_samples"] = vid_run_results_df["vid_accuracy_score_ocsvm"] * vid_run_results_df["vid_num_samples"]
    #     vid_overall_acc = vid_run_results_df["vid_num_accuracy_samples"].sum() / vid_run_results_df["vid_num_samples"].sum() # Overall accuracy considers all samples from all runs

    #     overall_percent_int_detected = overall_int_detected / len(uplink_throughput_files)

    #     # Record scenario results
    #     scenario_results.append({"Scenario": scenario_name, "Num_Runs": len(uplink_throughput_files), "Overall_Percent_Int_Detected": overall_percent_int_detected,
    #                              "DL_Overall_Accuracy": dl_overall_acc, "DL_Accuracy_Avg": dl_acc_avg, "DL_Accuracy_Max": dl_acc_max, "DL_Accuracy_Min": dl_acc_min, "DL_Accuracy_Std_Dev": dl_acc_std_dev, "DL_Num_Samples_Avg": dl_num_samples_avg, "DL_Int_Detected_Percent": dl_percent_int_detected,
    #                              "UL_Overall_Accuracy": ul_overall_acc, "UL_Accuracy_Avg": ul_acc_avg, "UL_Accuracy_Max": ul_acc_max, "UL_Accuracy_Min": ul_acc_min, "UL_Accuracy_Std_Dev": ul_acc_std_dev, "UL_Num_Samples_Avg": ul_num_samples_avg, "UL_Int_Detected_Percent": ul_percent_int_detected,
    #                              "VID_Overall_Accuracy": vid_overall_acc, "VID_Accuracy_Avg": vid_acc_avg, "VID_Accuracy_Max": vid_acc_max, "VID_Accuracy_Min": vid_acc_min, "VID_Accuracy_Std_Dev": vid_acc_std_dev, "VID_Num_Samples_Avg": vid_num_samples_avg, "VID_Int_Detected_Percent": vid_percent_int_detected})

    # # Save results
    # scenario_results_df = pd.DataFrame(scenario_results)
    # scenario_results_df.to_csv(os.path.join(SAVE_PATH, "ocsvm_no_int_results.csv"))

    """ UAV INT """
    print("Testing UAV Int Scenarios")
    scenarios_no_int = [x[0] for x in os.walk(DATASET_UAV_INT_PATH)]
    scenario_results = []
    for scenario in tqdm(scenarios_no_int):
        if scenario == DATASET_UAV_INT_PATH: # This is the root path
            continue
        # Get the parameters of the scenario
        scenario_name = scenario.split("/")[-1]
        params = scenario_name.split("_")
        bitrate = [x for x in params if "BitRate" in x][0].split('-')[-1]
        usi = [x for x in params if "UAVSendingInterval" in x][0].split('-')[-1]
        uavspeed = int([x for x in params if "UAVSpeed" in x][0].split('-')[-1])
        height = int([x for x in params if "Height" in x][0].split('-')[-1])
        mcs_index = get_MCS_index(float(bitrate))
        # Load OCSVM Model and Robust Scaler
        ocsvm_model_dl = load(open(os.path.join(OCSVM_MODEL_PATH, "ocsvm_Downlink_USI-{}_MCS-{}.pkl".format(usi, mcs_index)), 'rb'))
        ocsvm_model_ul = load(open(os.path.join(OCSVM_MODEL_PATH, "ocsvm_Uplink_USI-{}_MCS-{}.pkl".format(usi, mcs_index)), 'rb'))
        ocsvm_model_vid = load(open(os.path.join(OCSVM_MODEL_PATH, "ocsvm_Video_USI-{}_MCS-{}.pkl".format(usi, mcs_index)), 'rb'))
        robust_scaler_dl = load(open(os.path.join(ROBUST_SCALER_PATH, "robust_scaler_Downlink_USI-{}_MCS-{}.pkl".format(usi, mcs_index)), 'rb'))
        robust_scaler_ul = load(open(os.path.join(ROBUST_SCALER_PATH, "robust_scaler_Uplink_USI-{}_MCS-{}.pkl".format(usi, mcs_index)), 'rb'))
        robust_scaler_vid = load(open(os.path.join(ROBUST_SCALER_PATH, "robust_scaler_Video_USI-{}_MCS-{}.pkl".format(usi, mcs_index)), 'rb'))
        # Process each run
        ul_run_results = []
        dl_run_results = []
        vid_run_results = []
        # Each run should only have one uplink throughput, so use it to determine no. of runs
        uplink_throughput_files = glob.glob(os.path.join(scenario, "Run-*_Uplink_Throughput.csv"))
        downlink_throughput_files = glob.glob(os.path.join(scenario, "Run-*_Downlink_Throughput.csv"))
        simulation_metric = pd.read_csv(os.path.join(scenario, "Simulation_Results.csv"))
        overall_int_detected = 0
        for run_num in range(len(uplink_throughput_files)): # Make sure run_num starts from 0
            # Get the link reliabilities for this run, if below 99% only test for that link
            dl_run_reliability = simulation_metric.loc[simulation_metric["Run"] == run_num]["Total_Reliability_DL"].values[0]
            ul_run_reliability = simulation_metric.loc[simulation_metric["Run"] == run_num]["Total_Reliability_UL"].values[0]
            vid_run_reliability = simulation_metric.loc[simulation_metric["Run"] == run_num]["Total_Reliability_VID"].values[0]
            # Load and normalize throughputs
            if ul_run_reliability < RELIABILITY_TH:
                ul_throughput_df = pd.read_csv(os.path.join(scenario, "Run-{}_Uplink_Throughput.csv".format(run_num)))
                throughput = np.array(ul_throughput_df["Throughput"].values)
                ul_throughput_df["Throughput_Norm"] = robust_scaler_ul.transform(throughput.reshape(-1,1))           
                ul_throughput_df[['Mean_SINR',"Std_Dev_SINR"]] = ul_throughput_df.parallel_apply(lambda row: sinr_lognormal_approx(row['Horizontal_Distance'],row['Height']),axis=1,result_type='expand')
                ul_throughput_df = normalize_data(ul_throughput_df, columns=["Mean_SINR", "Std_Dev_SINR"])
            else: 
                ul_throughput_df = None
            if vid_run_reliability < RELIABILITY_TH:
                vid_throughput_df = pd.read_csv(os.path.join(scenario, "Run-{}_Video_Throughput.csv".format(run_num)))
                throughput = np.array(vid_throughput_df["Throughput"].values)
                vid_throughput_df["Throughput_Norm"] = robust_scaler_vid.transform(throughput.reshape(-1,1))           
                vid_throughput_df[['Mean_SINR',"Std_Dev_SINR"]] = vid_throughput_df.parallel_apply(lambda row: sinr_lognormal_approx(row['Horizontal_Distance'],row['Height']),axis=1,result_type='expand')
                vid_throughput_df = normalize_data(vid_throughput_df, columns=["Mean_SINR", "Std_Dev_SINR"])
            else:
                vid_throughput_df = None
            if dl_run_reliability < RELIABILITY_TH:
                dl_throughput_list = [l for l in downlink_throughput_files if "Run-{}_".format(run_num) in l]
                dl_df_list = []
                for file in dl_throughput_list:
                    df = pd.read_csv(file)
                    dl_df_list.append(df)
                dl_throughput_df = pd.concat(dl_df_list)
                throughput = np.array(dl_throughput_df["Throughput"].values)
                dl_throughput_df["Throughput_Norm"] = robust_scaler_dl.transform(throughput.reshape(-1,1))           
                dl_throughput_df[['Mean_SINR',"Std_Dev_SINR"]] = dl_throughput_df.parallel_apply(lambda row: sinr_lognormal_approx(row['Horizontal_Distance'],row['Height']),axis=1,result_type='expand')
                dl_throughput_df = normalize_data(dl_throughput_df, columns=["Mean_SINR", "Std_Dev_SINR"])
            else:
                dl_throughput_df = None
            # Testing OCSVM Model
            dl_int_detected = 0
            ul_int_detected = 0
            vid_int_detected = 0
            # Downlink
            if dl_run_reliability < RELIABILITY_TH:
                X_test_no_int = dl_throughput_df[["Mean_SINR", "Std_Dev_SINR", "Throughput_Norm"]].values
                y_pred_no_int_ocsvm = ocsvm_model_dl.predict(X_test_no_int)
                y_pred_no_int_ocsvm = (y_pred_no_int_ocsvm + 1) // 2 # To convert the range from -1:1 to 0:1 (0 for outlier, 1 for inlier)
                dl_percent_outlier = 1 - np.sum(y_pred_no_int_ocsvm) / len(X_test_no_int) # Ground truth is all 1 for normal test data
                dl_num_samples = len(X_test_no_int)
                if dl_percent_outlier > 0: 
                    dl_int_detected = 1  
                dl_run_results.append({"run_num": run_num, "dl_percent_outlier": dl_percent_outlier, "dl_num_samples": dl_num_samples, "dl_int_detected": dl_int_detected})   
            # Uplink
            if ul_run_reliability < RELIABILITY_TH:
                X_test_no_int = ul_throughput_df[["Mean_SINR", "Std_Dev_SINR", "Throughput_Norm"]].values
                y_pred_no_int_ocsvm = ocsvm_model_ul.predict(X_test_no_int)
                y_pred_no_int_ocsvm = (y_pred_no_int_ocsvm + 1) // 2 # To convert the range from -1:1 to 0:1 (0 for outlier, 1 for inlier)
                ul_percent_outlier = 1 - np.sum(y_pred_no_int_ocsvm) / len(X_test_no_int) # Ground truth is all 1 for normal test data
                ul_num_samples = len(X_test_no_int)
                if ul_percent_outlier < 1: 
                    ul_int_detected = 1
                ul_run_results.append({"run_num": run_num, "ul_percent_outlier": ul_percent_outlier, "ul_num_samples": ul_num_samples, "ul_int_detected": ul_int_detected})
            # Video
            if vid_run_reliability < RELIABILITY_TH:
                X_test_no_int = vid_throughput_df[["Mean_SINR", "Std_Dev_SINR", "Throughput_Norm"]].values
                y_pred_no_int_ocsvm = ocsvm_model_vid.predict(X_test_no_int)
                y_pred_no_int_ocsvm = (y_pred_no_int_ocsvm + 1) // 2 # To convert the range from -1:1 to 0:1 (0 for outlier, 1 for inlier)
                vid_percent_outlier = np.sum(y_pred_no_int_ocsvm) / len(X_test_no_int) # Ground truth is all 1 for normal test data
                vid_num_samples = len(X_test_no_int)
                if vid_percent_outlier < 1: 
                    vid_int_detected = 1
                vid_run_results.append({"run_num": run_num, "vid_percent_outlier": vid_percent_outlier, "vid_num_samples": vid_num_samples, "vid_int_detected": vid_int_detected})

            # Check for overall interference detection
            if dl_int_detected == 1 or ul_int_detected == 1 or vid_int_detected == 1:
                overall_int_detected += 1
            
        # Record run results
        if len(dl_run_results) > 0:
            dl_run_results_df = pd.DataFrame(dl_run_results)
            dl_percent_outlier_avg = dl_run_results_df["dl_percent_outlier"].mean()
            dl_percent_outlier_max = dl_run_results_df["dl_percent_outlier"].max()
            dl_percent_outlier_min = dl_run_results_df["dl_percent_outlier"].min()
            dl_percent_outlier_std_dev = dl_run_results_df["dl_percent_outlier"].std()
            dl_num_samples_avg = dl_run_results_df["dl_num_samples"].mean()
            dl_percent_int_detected = dl_run_results_df["dl_int_detected"].sum() / len(dl_run_results_df)
            dl_run_results_df["dl_num_outlier"] = dl_run_results_df["dl_percent_outlier"] * dl_run_results_df["dl_num_samples"]
            dl_overall_percent_outlier = dl_run_results_df["dl_num_outlier"].sum() / dl_run_results_df["dl_num_samples"].sum() # Overall accuracy considers all samples from all runs
        else:
            dl_percent_outlier_avg = np.nan
            dl_percent_outlier_max = np.nan
            dl_percent_outlier_min = np.nan
            dl_percent_outlier_std_dev = np.nan
            dl_num_samples_avg = np.nan
            dl_percent_int_detected = np.nan
            dl_overall_percent_outlier = np.nan

        if len(ul_run_results) > 0:
            ul_run_results_df = pd.DataFrame(ul_run_results)
            ul_percent_outlier_avg = ul_run_results_df["ul_percent_outlier"].mean()
            ul_percent_outlier_max = ul_run_results_df["ul_percent_outlier"].max()
            ul_percent_outlier_min = ul_run_results_df["ul_percent_outlier"].min()
            ul_percent_outlier_std_dev = ul_run_results_df["ul_percent_outlier"].std()
            ul_num_samples_avg = ul_run_results_df["ul_num_samples"].mean()
            ul_percent_int_detected = ul_run_results_df["ul_int_detected"].sum() / len(ul_run_results_df)
            ul_run_results_df["ul_num_outlier"] = ul_run_results_df["ul_percent_outlier"] * ul_run_results_df["ul_num_samples"]
            ul_overall_percent_outlier = ul_run_results_df["ul_num_outlier"].sum() / ul_run_results_df["ul_num_samples"].sum() # Overall accuracy considers all samples from all runs
        else:
            ul_percent_outlier_avg = np.nan
            ul_percent_outlier_max = np.nan
            ul_percent_outlier_min = np.nan
            ul_percent_outlier_std_dev = np.nan
            ul_num_samples_avg = np.nan
            ul_percent_int_detected = np.nan
            ul_overall_percent_outlier = np.nan

        if len(vid_run_results) > 0:
            vid_run_results_df = pd.DataFrame(vid_run_results)
            vid_percent_outlier_avg = vid_run_results_df["vid_percent_outlier"].mean()
            vid_percent_outlier_max = vid_run_results_df["vid_percent_outlier"].max()
            vid_percent_outlier_min = vid_run_results_df["vid_percent_outlier"].min()
            vid_percent_outlier_std_dev = vid_run_results_df["vid_percent_outlier"].std()
            vid_num_samples_avg = vid_run_results_df["vid_num_samples"].mean()
            vid_percent_int_detected = vid_run_results_df["vid_int_detected"].sum() / len(vid_run_results_df)
            vid_run_results_df["vid_num_outlier"] = vid_run_results_df["vid_percent_outlier"] * vid_run_results_df["vid_num_samples"]
            vid_overall_percent_outlier = vid_run_results_df["vid_num_outlier"].sum() / vid_run_results_df["vid_num_samples"].sum() # Overall accuracy considers all samples from all runs
        else:
            vid_percent_outlier_avg = np.nan
            vid_percent_outlier_max = np.nan
            vid_percent_outlier_min = np.nan
            vid_percent_outlier_std_dev = np.nan
            vid_num_samples_avg = np.nan
            vid_percent_int_detected = np.nan
            vid_overall_percent_outlier = np.nan
        
        overall_percent_int_detected = overall_int_detected / len(uplink_throughput_files)

        # Record scenario results
        scenario_results.append({"Scenario": scenario_name, "Num_Runs": len(uplink_throughput_files), "Overall_Percent_Int_Detected": overall_percent_int_detected,
                                 "DL_Overall_Percent_Outlier": dl_overall_percent_outlier, "DL_Percent_Outlier_Avg": dl_percent_outlier_avg, "DL_Percent_Outlier_Max": dl_percent_outlier_max, "DL_Percent_Outlier_Min": dl_percent_outlier_min, "DL_Percent_Outlier_Std_Dev": dl_percent_outlier_std_dev, "DL_Num_Samples_Avg": dl_num_samples_avg, "DL_Int_Detected_Percent": dl_percent_int_detected,
                                 "UL_Overall_Percent_Outlier": ul_overall_percent_outlier, "UL_Percent_Outlier_Avg": ul_percent_outlier_avg, "UL_Percent_Outlier_Max": ul_percent_outlier_max, "UL_Percent_Outlier_Min": ul_percent_outlier_min, "UL_Percent_Outlier_Std_Dev": ul_percent_outlier_std_dev, "UL_Num_Samples_Avg": ul_num_samples_avg, "UL_Int_Detected_Percent": ul_percent_int_detected,
                                 "VID_Overall_Percent_Outlier": vid_overall_percent_outlier, "VID_Percent_Outlier_Avg": vid_percent_outlier_avg, "VID_Percent_Outlier_Max": vid_percent_outlier_max, "VID_Percent_Outlier_Min": vid_percent_outlier_min, "VID_Percent_Outlier_Std_Dev": vid_percent_outlier_std_dev, "VID_Num_Samples_Avg": vid_num_samples_avg, "VID_Int_Detected_Percent": vid_percent_int_detected})

    # Save results
    scenario_results_df = pd.DataFrame(scenario_results)
    scenario_results_df.to_csv(os.path.join(SAVE_PATH, "ocsvm_uav_int_results.csv"))