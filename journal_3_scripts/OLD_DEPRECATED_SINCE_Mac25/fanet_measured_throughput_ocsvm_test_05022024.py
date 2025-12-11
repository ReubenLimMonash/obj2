# Date: 05/02/2024
# Desc: To test OCSVM accuracy in non-anomalous data and LOF in anomalous data

import pandas as pd
import numpy as np 
import math
import os
import gc 
import glob
from pickle import load
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.svm import OneClassSVM
from sys import getsizeof
from scipy import special
import matplotlib.pyplot as plt
from pandarallel import pandarallel
import tensorflow as tf 
from itertools import product

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

def get_measured_throughput(sim_root_path, link="Downlink", single_path = False):
    '''
    Function to load the processed measured throughput data from CSV files stored in different subdirs in sim_root_path
    Modified: The throughput files for each UAV and the GCS are stored separately (rather than single Uplink/Downlink) and separated by runs.
    '''
    assert link in ["Downlink", "Uplink", "Video"], 'link must be one of "Downlink", "Uplink", "Video"'
    df_list = []
    if single_path:
        scenario_list = [sim_root_path]
    else:
        scenario_list = [f.path for f in os.scandir(sim_root_path) if f.is_dir()] # Get list of "unique" scenarios
    for scenario in tqdm(scenario_list):
        # Get the measured throughput samples for UL/DL/Vid under this scenario
        if link == "Downlink":
            throughput_files = glob.glob(os.path.join(scenario, "Run-*_Downlink_Throughput.csv"))
        elif link == "Uplink":
            throughput_files = glob.glob(os.path.join(scenario, "Run-*_Uplink_Throughput.csv"))
        elif link == "Video":
            throughput_files = glob.glob(os.path.join(scenario, "Run-*_Video_Throughput.csv"))
        
        for file in throughput_files:
            measured_df = pd.read_csv(file)
            df_list.append(measured_df)

    return pd.concat(df_list)

def normalize_data(df_in, columns, link, save_details_path=None):
    '''
    columns: The pandas data columns to normalize, given as a list of column names
    link is the link type, for choosing the range of measured throughput to scale the data
    '''
    assert link in ["Downlink", "Uplink", "Video"], 'link must be one of "Downlink", "Uplink", "Video"'
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
    if link == "Downlink":
        max_throughput = 16000 # Max measured throughput for DJI Spark Downlink is 15968 bytes/sec
    elif link == "Uplink":
        max_throughput = 565000 # Max measured throughput for DJI Spark Uplink is 564990 bytes/sec
    elif link == "Video":
        max_throughput = 300000 # Max measured throughput for DJI Spark Video Link is 298700 bytes/sec
    min_throughput = 0

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
    if "Modulation" in columns:
        df['Modulation'] = df['Modulation'].replace({"BPSK":1, "QPSK":0.3333, 16:-0.3333, "QAM-16":-0.3333, "QAM16":-0.3333, 64:-1, "QAM-64":-1, "QAM64":-1})
    if "MCS" in columns:
        df["MCS"] = df["MCS"].apply(lambda x: 2*(x-min_mcs)/(max_mcs-min_mcs) - 1)
    if "MCS_Index" in columns:
        df["MCS_Index"] = df["MCS_Index"].apply(lambda x: 2*(x-min_mcs)/(max_mcs-min_mcs) - 1)
    if "UAV_Speed" in columns:
        df["UAV_Speed"] = df["UAV_Speed"].apply(lambda x: 2*(x-min_uav_speed)/(max_uav_speed-min_uav_speed) - 1)
    if "Throughput" in columns:
        df["Throughput"] = df["Throughput"].apply(lambda x: 2*(x-min_throughput)/(max_throughput-min_throughput) - 1)


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
        f.write("Max Measured Throughput (bytes/sec): {}\n".format(max_throughput))
        f.write("Min Measured Throughput (bytes/sec): {}\n".format(min_throughput))
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

# Somehow this is needed to load the Downlink OCSVM LOF models
def custom_loss_score(y_true, y_pred, accuracy_goal):
    # LOWER BETTER
    y_true = (y_true + 1) / 2 # To convert the range from -1:1 to 0:1
    y_pred = (y_pred + 1) / 2 # To convert the range from -1:1 to 0:1
    sum_diff = np.abs(y_true - y_pred).sum() # y_true and y_pred should only be 0 or 1
    accuracy = 1 - sum_diff / len(y_true)
    return np.abs(accuracy - accuracy_goal) # How far is the accuracy from the goal?

if __name__ == "__main__":
    # pandarallel.initialize(progress_bar=False)
    ''' Define Paths Here'''
    DL_MODEL_PATH = "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJISpark/nn_checkpoints/djispark_nnv4_wobn_dl/model.010-0.2039.h5"
    UL_MODEL_PATH = "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJISpark/nn_checkpoints/djispark_nnv4_wobn_ul/model.010-0.1202.h5"
    VID_MODEL_PATH = "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJISpark/nn_checkpoints/djispark_nnv4_wobn_vid/model.010-0.2581.h5"
    OCSVM_MODEL_PATH = "/media/research-student/One Touch/FANET_Dataset/DJISpark_Measured_Throughput_10000Samples/ocsvm_models"
    ROBUST_SCALER_PATH = "/media/research-student/One Touch/FANET_Dataset/DJISpark_Measured_Throughput_10000Samples/ocsvm_models"
    LOF_MODEL_PATH = "/media/research-student/One Touch/FANET_Dataset/DJISpark_Measured_Throughput_10000Samples/lof_models"
    DATASET_PATH = "/media/research-student/One Touch/FANET_Dataset/DJISpark_Measured_Throughput_10000Samples/ocsvm_test_dataset" 
    SAVE_PATH = "/media/research-student/One Touch/FANET_Dataset/DJISpark_Measured_Throughput_10000Samples/ocsvm_models"
    CRIT_DIST_FILE = "/media/research-student/One Touch/FANET_Dataset/DJISpark_Measured_Throughput_10000Samples/ocsvm_models/Test_Dataset_Critical_Distances.csv" # Set to "" to calc critical distances for each scenario
    HEIGHTS = [75, 165, 285]
    LINKS = ["Downlink", "Uplink", "Video"]

    # Define Fixed Params
    MAX_MEAN_SINR = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
    MAX_STD_DEV_SINR = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
    MIN_MEAN_SINR = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
    MIN_STD_DEV_SINR = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)
    USI_NORM_DICT = {10:-1, 20:-0.5, 40:0, 66.7: 0.5, 100:1, 1000:2} # For normalization of USI

    '''Load the DNN reliability prediction models for defining regions of reliability under different MCS and USI'''
    dl_model = tf.keras.models.load_model(DL_MODEL_PATH, compile=False)
    dl_model.compile(optimizer='adam', 
                     loss={'packet_state': 'categorical_crossentropy'},
                     metrics={'packet_state': 'accuracy'})
    ul_model = tf.keras.models.load_model(UL_MODEL_PATH, compile=False)
    ul_model.compile(optimizer='adam', 
                     loss={'packet_state': 'categorical_crossentropy'},
                     metrics={'packet_state': 'accuracy'})
    vid_model = tf.keras.models.load_model(VID_MODEL_PATH, compile=False)
    vid_model.compile(optimizer='adam', 
                     loss={'packet_state': 'categorical_crossentropy'},
                     metrics={'packet_state': 'accuracy'})
    
    '''List out the different MCS and USI'''
    uav_send_int = [10, 20, 66.7, 100]
    mcs_index = np.arange(8).tolist()

    '''Get the critical distance for different heights for each combination of MCS and USI'''
    if CRIT_DIST_FILE == "":
        print("-------------- COMPUTING CRIT DISTS -------------")
        max_mean_sinr = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
        max_std_dev_sinr = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
        min_mean_sinr = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
        min_std_dev_sinr = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)
        uav_send_int_norm = {10:-1, 20:-0.5, 40:0, 66.7: 0.5, 100:1, 1000:2}
        horizontal_dist = np.linspace(0, 1200, 121, endpoint=True)
        reliability_th = 0.99 # Threshold for reliability value
        crit_dist_list = []
        for usi, mcs in tqdm(list(product(uav_send_int, mcs_index))):
            for height in HEIGHTS:
                mean_sinr = []
                std_dev_sinr = []
                for h_dist in horizontal_dist:
                    m, s = sinr_lognormal_approx(h_dist, height)
                    m = 2*(10*math.log10(m)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1
                    s = 2*(10*math.log10(s)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1
                    mean_sinr.append(m)
                    std_dev_sinr.append(s)
                inputs = np.hstack((np.array(mean_sinr).reshape(-1,1), np.array(std_dev_sinr).reshape(-1,1), 
                            np.ones((len(horizontal_dist),1))*uav_send_int_norm[usi], np.ones((len(horizontal_dist),1))*norm_MCS(mcs)))
                # Evaluate DL reliability at each horizontal_dist
                dl_predictions = dl_model.predict(inputs)
                dl_reliability = np.array([pred[0] >= reliability_th for pred in dl_predictions])
                if np.any(dl_reliability):
                    crit_distance = horizontal_dist[np.max(dl_reliability.nonzero())]
                    crit_dist_list.append({"MCS_Index": mcs, "USI": usi, "Link": "Downlink", "Height": height, "Critical_Distance": crit_distance})
                else:
                    crit_dist_list.append({"MCS_Index": mcs, "USI": usi, "Link": "Downlink", "Height": height, "Critical_Distance": 0})
                # Evaluate UL reliability at each horizontal_dist
                ul_predictions = ul_model.predict(inputs)
                ul_reliability = np.array([pred[0] >= reliability_th for pred in ul_predictions])
                if np.any(ul_reliability):
                    crit_distance = horizontal_dist[np.max(ul_reliability.nonzero())]
                    crit_dist_list.append({"MCS_Index": mcs, "USI": usi, "Link": "Uplink", "Height": height, "Critical_Distance": crit_distance})
                else:
                    crit_dist_list.append({"MCS_Index": mcs, "USI": usi, "Link": "Uplink", "Height": height, "Critical_Distance": 0})
                # Evaluate UL reliability at each horizontal_dist
                vid_predictions = vid_model.predict(inputs)
                vid_reliability = np.array([pred[0] >= reliability_th for pred in vid_predictions])
                if np.any(vid_reliability):
                    crit_distance = horizontal_dist[np.max(vid_reliability.nonzero())]
                    crit_dist_list.append({"MCS_Index": mcs, "USI": usi, "Link": "Video", "Height": height, "Critical_Distance": crit_distance})
                else:
                    crit_dist_list.append({"MCS_Index": mcs, "USI": usi, "Link": "Video", "Height": height, "Critical_Distance": 0}) 
        crit_dist_df = pd.DataFrame(crit_dist_list)
        crit_dist_df.to_csv(os.path.join(SAVE_PATH, "Test_Dataset_Critical_Distances.csv"), index=False)
    
    else:
        # Optionally, load crit_dist_df
        crit_dist_df = pd.read_csv(CRIT_DIST_FILE)
    '''Load and Process The Dataset'''
    print("-------------- LOADING AND PROCESSING TEST DATASET -------------")
    pandarallel.initialize(progress_bar=False)
    ocsvm_results_list = [] # To store the accuracy score for each OCSVM model
    for link in LINKS:
        print(link)
        ''' LOAD AND PROCESS TEST DATASET FOR PARTICULAR LINK'''
        print("### LOADING TEST DATASET ###")
        throughput_no_int_df = get_measured_throughput(os.path.join(DATASET_PATH, "data_no_interference_singlerun_processed"), link)
        throughput_uav_int_df = get_measured_throughput(os.path.join(DATASET_PATH, "data_uav_interference_singlerun_processed"), link)
        throughput_manet_int_df = get_measured_throughput(os.path.join(DATASET_PATH, "data_manet_interference_singlerun_processed"), link)
        # Get list of trained OCSVM models
        ocsvm_model_list = glob.glob(os.path.join(OCSVM_MODEL_PATH, "ocsvm_{}_*.pkl".format(link)))
        # For each OCSVM model listed in ocsvm_model_list, get the relevant test data df for it
        throughput_df_list = [] # To store the filtered DFs for different scenarios
        print("### TESTING OCSVM MODELS ###")
        for ocsvm_model_path in tqdm(ocsvm_model_list):
            # Load the OCSVM model
            ocsvm_model = load(open(ocsvm_model_path, 'rb'))
            # Get the USI and MCS
            model_name = ocsvm_model_path.split("/")[-1].split(".pkl")[0]
            usi = model_name.split("_")[2].split("-")[-1]
            mcs_index = model_name.split("_")[3].split("-")[-1]
            # Load the LOF model
            lof_model = load(open(os.path.join(LOF_MODEL_PATH, "lof_{}_USI-{}_MCS-{}.pkl".format(link, usi, mcs_index)), 'rb'))
            # Filter test datasets by USI and MCS
            context_throughput_no_int_df = throughput_no_int_df.loc[(throughput_no_int_df["UAV_Sending_Interval"]==float(usi)) & (throughput_no_int_df["MCS_Index"]==int(mcs_index))]
            context_throughput_uav_int_df = throughput_uav_int_df.loc[(throughput_uav_int_df["UAV_Sending_Interval"]==float(usi)) & (throughput_uav_int_df["MCS_Index"]==int(mcs_index))]
            context_throughput_manet_int_df = throughput_manet_int_df.loc[(throughput_manet_int_df["UAV_Sending_Interval"]==float(usi)) & (throughput_manet_int_df["MCS_Index"]==int(mcs_index))]
            # TODO: Revert when proper test dataset ready
            context_crit_dist_df = crit_dist_df.loc[(crit_dist_df["USI"]==float(usi)) & (crit_dist_df["MCS_Index"]==int(mcs_index)) & (crit_dist_df["Link"]==link)]
            # context_crit_dist_df = crit_dist_df.loc[(crit_dist_df["USI"]==float(usi)) & (crit_dist_df["MCS_Index"]==int(mcs_index))] # Take the min crit dist of all links
            # For each height, get the data up to the critical distance
            df = []
            df_uav = []
            df_manet = []
            # TODO: Revert when proper test dataset ready
            for h, cd in zip(context_crit_dist_df["Height"].values, context_crit_dist_df["Critical_Distance"].values):
            # for h in context_crit_dist_df["Height"].unique():
                # cd = context_crit_dist_df.loc[(context_crit_dist_df["Height"] == h)]["Critical_Distance"].min()
                # print(link, usi, mcs_index, h, cd)
                tmp_df = context_throughput_no_int_df.loc[(context_throughput_no_int_df["Height"]==h) & (context_throughput_no_int_df["Horizontal_Distance"]<=cd)]
                if not tmp_df.empty:
                    df.append(tmp_df)
                tmp_uav_df = context_throughput_uav_int_df.loc[(context_throughput_uav_int_df["Height"]==h) & (context_throughput_uav_int_df["Horizontal_Distance"]<=cd)]
                if not tmp_uav_df.empty:
                    df_uav.append(tmp_uav_df)
                tmp_manet_df = context_throughput_manet_int_df.loc[(context_throughput_manet_int_df["Height"]==h) & (context_throughput_manet_int_df["Horizontal_Distance"]<=cd)]
                if not tmp_manet_df.empty:
                    df_manet.append(tmp_manet_df)
            if len(df) > 0: # If data exist for this case, process it and append to throughput_df_list
                """ Process and Normalize The Data """
                robust_scaler = load(open(os.path.join(ROBUST_SCALER_PATH, "robust_scaler_{}_USI-{}_MCS-{}.pkl".format(link, usi, mcs_index)), 'rb'))
                # Test Dataset No Interference
                test_throughput_no_int_df = pd.concat(df)
                throughput = np.array(test_throughput_no_int_df["Throughput"].values)
                test_throughput_no_int_df["Throughput_Norm"] = robust_scaler.transform(throughput.reshape(-1,1))           
                test_throughput_no_int_df[['Mean_SINR',"Std_Dev_SINR"]] = test_throughput_no_int_df.parallel_apply(lambda row: sinr_lognormal_approx(row['Horizontal_Distance'],row['Height']),axis=1,result_type='expand')
                test_throughput_no_int_df = normalize_data(test_throughput_no_int_df, columns=["Mean_SINR", "Std_Dev_SINR",], link=link)
                # Test Dataset UAV Interference
                test_throughput_uav_int_df = pd.concat(df_uav)
                throughput = np.array(test_throughput_uav_int_df["Throughput"].values)
                test_throughput_uav_int_df["Throughput_Norm"] = robust_scaler.transform(throughput.reshape(-1,1))           
                test_throughput_uav_int_df[['Mean_SINR',"Std_Dev_SINR"]] = test_throughput_uav_int_df.parallel_apply(lambda row: sinr_lognormal_approx(row['Horizontal_Distance'],row['Height']),axis=1,result_type='expand')
                test_throughput_uav_int_df = normalize_data(test_throughput_uav_int_df, columns=["Mean_SINR", "Std_Dev_SINR",], link=link)
                # Test Dataset MANET Interference
                test_throughput_manet_int_df = pd.concat(df_manet)
                throughput = np.array(test_throughput_manet_int_df["Throughput"].values)
                test_throughput_manet_int_df["Throughput_Norm"] = robust_scaler.transform(throughput.reshape(-1,1))           
                test_throughput_manet_int_df[['Mean_SINR',"Std_Dev_SINR"]] = test_throughput_manet_int_df.parallel_apply(lambda row: sinr_lognormal_approx(row['Horizontal_Distance'],row['Height']),axis=1,result_type='expand')
                test_throughput_manet_int_df = normalize_data(test_throughput_manet_int_df, columns=["Mean_SINR", "Std_Dev_SINR",], link=link)
                """ Get Accuracy of OCSVM and LOF Models in Non-Anomalous Scenarios """
                X_test_no_int = test_throughput_no_int_df[["Mean_SINR", "Std_Dev_SINR", "Throughput_Norm"]].values
                y_pred_no_int_ocsvm = ocsvm_model.predict(X_test_no_int)
                y_pred_no_int_ocsvm = (y_pred_no_int_ocsvm + 1) // 2 # To convert the range from -1:1 to 0:1 (0 for outlier, 1 for inlier)
                non_anomalous_accuracy_score_ocsvm = np.sum(y_pred_no_int_ocsvm) / len(X_test_no_int) # Ground truth is all 1 for normal test data
                y_pred_no_int_lof = lof_model.predict(X_test_no_int)
                y_pred_no_int_lof = (y_pred_no_int_lof + 1) // 2 # To convert the range from -1:1 to 0:1 (0 for outlier, 1 for inlier)
                non_anomalous_accuracy_score_lof = np.sum(y_pred_no_int_lof) / len(X_test_no_int) # Ground truth is all 1 for normal test data
                """ Get Accuracy of OCSVM and LOF Models in UAV Interference Scenarios """
                X_test_uav_int = test_throughput_uav_int_df[["Mean_SINR", "Std_Dev_SINR", "Throughput_Norm"]].values
                y_pred_uav_int_ocsvm = ocsvm_model.predict(X_test_uav_int)
                y_pred_uav_int_ocsvm = (y_pred_uav_int_ocsvm + 1) // 2 # To convert the range from -1:1 to 0:1 (0 for outlier, 1 for inlier)
                percent_outlier_uav_int_ocsvm = 1 - np.sum(y_pred_uav_int_ocsvm) / len(X_test_uav_int) # Get the percent outlier predicted for OCSVM
                y_pred_uav_int_lof = lof_model.predict(X_test_uav_int)
                y_pred_uav_int_lof = (y_pred_uav_int_lof + 1) // 2 # To convert the range from -1:1 to 0:1 (0 for outlier, 1 for inlier)
                percent_outlier_uav_int_lof = 1 - np.sum(y_pred_uav_int_lof) / len(X_test_uav_int) # Get the percent outlier predicted for LOF
                similarity_uav_int_ocsvm_lof = accuracy_score(y_pred_uav_int_ocsvm, y_pred_uav_int_lof) # Similarity of predictions between OCSVM and LOF
                """ Get Accuracy of OCSVM and LOF Models in MANET Interference Scenarios """
                X_test_manet_int = test_throughput_manet_int_df[["Mean_SINR", "Std_Dev_SINR", "Throughput_Norm"]].values
                y_pred_manet_int_ocsvm = ocsvm_model.predict(X_test_manet_int)
                y_pred_manet_int_ocsvm = (y_pred_manet_int_ocsvm + 1) // 2 # To convert the range from -1:1 to 0:1 (0 for outlier, 1 for inlier)
                percent_outlier_manet_int_ocsvm = 1 - np.sum(y_pred_manet_int_ocsvm) / len(X_test_manet_int) # Get the percent outlier predicted for OCSVM
                y_pred_manet_int_lof = lof_model.predict(X_test_manet_int)
                y_pred_manet_int_lof = (y_pred_manet_int_lof + 1) // 2 # To convert the range from -1:1 to 0:1 (0 for outlier, 1 for inlier)
                percent_outlier_manet_int_lof = 1 - np.sum(y_pred_manet_int_lof) / len(X_test_manet_int) # Get the percent outlier predicted for LOF
                similarity_manet_int_ocsvm_lof = accuracy_score(y_pred_manet_int_ocsvm, y_pred_manet_int_lof) # Similarity of predictions between OCSVM and LOF
            else: # Else append empty DF
                non_anomalous_accuracy_score_ocsvm = np.nan
                non_anomalous_accuracy_score_lof = np.nan
                percent_outlier_uav_int_ocsvm = np.nan
                percent_outlier_uav_int_lof = np.nan
                percent_outlier_manet_int_ocsvm = np.nan
                percent_outlier_manet_int_lof = np.nan
                similarity_uav_int_ocsvm_lof = np.nan
                similarity_manet_int_ocsvm_lof = np.nan
            ocsvm_results_list.append({"Link": link, "USI": usi, "MCS_Index": mcs_index, 
                                       "Non-Anomalous_Accuracy_OCSVM": non_anomalous_accuracy_score_ocsvm, "Non-Anomalous_Accuracy_LOF": non_anomalous_accuracy_score_lof,
                                       "Percent_Outlier_UAV_Interference_OCSVM": percent_outlier_uav_int_ocsvm, "Percent_Outlier_UAV_Interference_LOF": percent_outlier_uav_int_lof,
                                       "Percent_Outlier_MANET_Interference_OCSVM": percent_outlier_manet_int_ocsvm, "Percent_Outlier_MANET_Interference_LOF": percent_outlier_manet_int_lof,
                                       "Similarity_UAV_Interference_OCSVM_LOF": similarity_uav_int_ocsvm_lof, "Similarity_MANET_Interference_OCSVM_LOF": similarity_manet_int_ocsvm_lof})

    ocsvm_results_df = pd.DataFrame(ocsvm_results_list)
    ocsvm_results_df.to_csv(os.path.join(SAVE_PATH, "OCSVM_Results.csv"), index=False)
