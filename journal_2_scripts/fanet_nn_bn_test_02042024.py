'''
Date: 03/04/2024
Desc: To compare the performance of NN, BN and calibrated NN model in predicting communication reliability
Metrics: Reliability class accuracy, Failure Mode accuracy, above reliability th accuracy
'''

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras import activations
from keras import backend as K
import pandas as pd # for data manipulation 
from scipy.optimize import minimize, differential_evolution
import numpy as np
import glob, math, os
from scipy import special
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

def euclidean_dist(row):
    # Function to calc euclidean distance on every df row 
    euc_dist = math.sqrt(row["U2G_Distance"]**2 - row["Height"]**2)
    return euc_dist

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
    elif env == 'urban':
        a1 = 0.3
        a2 = 5e-4
        a3 = 15
    
    delta_h = height_tx - height_rx
    # pow_factor = 2 * h_dist * math.sqrt(a1*a2/math.pi) + a1 # NOTE: Use this pow_factor if assuming PPP building dist.
    pow_factor = h_dist * math.sqrt(a1*a2) # NOTE: Use this pow_factor if assuming ITU-R assumptions.
    if delta_h == 0:
        p = (1 - math.exp((-(height_tx)**2) / (2*a3**2))) ** pow_factor
    else:
        if delta_h < 0:
            h1 = height_rx
            h2 = height_tx
        else:
            h1 = height_tx
            h2 = height_rx
        delta_h = abs(delta_h)
        p = (1 - (math.sqrt(2*math.pi)*a3 / delta_h) * abs(q_func(h1/a3) - q_func(h2/a3))) ** pow_factor
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
    elif env == "urban":
        n_min = 1.9
        n_max = 2.7
        K_dB_min = -5
        K_dB_max = 15
        K_min = 10**(K_dB_min/10)
        K_max = 10**(K_dB_max/10)
        alpha = 10.42 # Env parameters for logarithm std dev of shadowing 
        beta = 0.05 # Env parameters for logarithm std dev of shadowing 
    # Calculate fading parameters
    PLoS = plos_calc(h_dist, 0, height, env=env)
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

def normalize_data(df, columns=[], save_details_path=None):
    '''
    columns: The pandas data columns to normalize, given as a list of column names
    '''
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

    # Normalize data (Min Max Normalization between [-1,1])
    if "Height" in columns:
        df["Height"] = df["Height"].apply(lambda x: 2*(x-min_height)/(max_height-min_height) - 1)
    if "U2G_H_Dist" in columns:
        df["U2G_H_Dist"] = df["U2G_H_Dist"].apply(lambda x: 2*(x-min_h_dist)/(max_h_dist-min_h_dist) - 1)
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
        f.write("[BPSK: 1, QPSK: 0.3333, QAM16: -0.3333, QAM64: -1]\n")
        f.write("UAV Sending Interval: [10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1]\n")
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
    df.loc[(df["Modulation"] == "QPSK") & (df["Bitrate"] == 13.0), "MCS"] = 1 # MCS Index 0
    df.loc[(df["Modulation"] == "QPSK") & (df["Bitrate"] == 19.5), "MCS"] = 2 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM16") & (df["Bitrate"] == 26.0), "MCS"] = 3 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM16") & (df["Bitrate"] == 39.0), "MCS"] = 4 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 52.0), "MCS"] = 5 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 58.5), "MCS"] = 6 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 65.0), "MCS"] = 7 # MCS Index 0

    return df

def get_output_layer(model, layer_name):
    # From https://github.com/jacobgil/keras-cam/blob/master/model.py#L79
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

def find_nearest_value(value, array):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_index(value, array):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def build_nn_model_v4_wobatchnorm_noactivation():
    # For multiple output model
    # Version 4: Having only a single output layer for packet state
    inputs = Input(shape=(4,))
    base = Dense(100, activation='relu')(inputs)
    base = Dense(50, activation='relu')(base)
    base = Dense(25, activation='relu')(base)
    base = Dense(10, activation='relu')(base)
    packet_state_out = Dense(4, activation=None, name='packet_state_no_activation')(base)
    model = Model(inputs=inputs, outputs = packet_state_out)
    return model

if __name__ == "__main__":
    # EVERYTHING SHOULD CORRESPOND TO EACH OTHER
    # NN_MODELS = ["/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJISpark/nn_checkpoints/djispark_nnv4_wobn_dl_05042024/model.020-0.2034.h5",
    #               "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJISpark/nn_checkpoints/djispark_nnv4_wobn_ul_01042024/model.020-0.1199.h5",
    #               "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJISpark/nn_checkpoints/djispark_nnv4_wobn_vid_08042024/model.020-0.2590.h5",
    #               "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJIMavicAir/nn_checkpoints/djimavicair_nnv4_wobn_finetune_dl_08042024/model.020-0.1836.h5",
    #               "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJIMavicAir/nn_checkpoints/djimavicair_nnv4_wobn_finetune_ul_05042024/model.020-0.1229.h5",
    #               "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJIMavicAir/nn_checkpoints/djimavicair_nnv4_wobn_finetune_vid_15042024/model.020-0.2636.h5",
    #               "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_ParrotAR2/nn_checkpoints/parrotar2_nnv4_wobn_finetune_dl_10042024/model.020-0.2359.h5",
    #               "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_ParrotAR2/nn_checkpoints/parrotar2_nnv4_wobn_finetune_ul_10042024/model.020-0.1144.h5",
    #               "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_ParrotAR2/nn_checkpoints/parrotar2_nnv4_wobn_finetune_vid_15042024/model.020-0.2559.h5"]
    # NN_MODELS = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Downlink.round-0_split-6_0.2024.h5",
    #               "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Uplink.round-1_split-4_0.1346.h5",
    #               "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Video.round-0_split-8_0.2655.h5",
    #               "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/nn_ckpts/model_Downlink.round-0_split-7_0.2030.h5",
    #               "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/nn_ckpts/model_Uplink.round-0_split-9_0.1462.h5",
    #               "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/nn_ckpts/model_Video.round-0_split-8_0.2702.h5",
    #               "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/nn_ckpts/model_Downlink.round-0_split-9_0.2302.h5",
    #               "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/nn_ckpts/model_Uplink.round-1_split-9_0.1261.h5",
    #               "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/nn_ckpts/model_Video.round-0_split-1_0.2678.h5"]
    NN_MODELS = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Downlink.round-1_split-9_0.2027.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Uplink.round-0_split-9_0.1362.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Video.round-1_split-9_0.2720.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/nn_ckpts/model_Downlink.round-0_split-9_0.2039.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/nn_ckpts/model_Uplink.round-0_split-9_0.1462.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/nn_ckpts/model_Video.round-0_split-9_0.2725.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/nn_ckpts/model_Downlink.round-0_split-9_0.2302.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/nn_ckpts/model_Uplink.round-0_split-9_0.1268.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/nn_ckpts/model_Video.round-0_split-9_0.2686.h5"]
    # NN_MODELS = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Downlink.round-1_split-9_0.2027.h5",
    #               "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Uplink.round-1_split-9_0.1363.h5",
    #               "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Video.round-1_split-9_0.2720.h5",
    #               "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/nn_ckpts/model_Downlink.round-1_split-9_0.2108.h5",
    #               "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/nn_ckpts/model_Uplink.round-1_split-9_0.1468.h5",
    #               "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/nn_ckpts/model_Video.round-1_split-9_0.2737.h5",
    #               "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/nn_ckpts_retrain/model_Downlink.round-0_split-0_0.2321.h5",
    #               "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/nn_ckpts_retrain/model_Uplink.round-0_split-2_0.1278.h5",
    #               "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/nn_ckpts/model_Video.round-1_split-9_0.2698.h5"]
    # BN_MODELS correspond to NN_MODELS
    # BN_MODELS = ["/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJISpark/bn_ckpt/djispark_reliability_bn_CPT_Downlink.csv",
    #               "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJISpark/bn_ckpt/djispark_reliability_bn_CPT_Uplink.csv",
    #               "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJISpark/bn_ckpt/djispark_reliability_bn_CPT_Video.csv",
    #               "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJIMavicAir/bn_ckpt/djimavicair_reliability_bn_CPT_Downlink.csv",
    #               "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJIMavicAir/bn_ckpt/djimavicair_reliability_bn_CPT_Uplink.csv",
    #               "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJIMavicAir/bn_ckpt/djimavicair_reliability_bn_CPT_Video.csv",
    #               "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_ParrotAR2/bn_ckpt/parrotar2_reliability_bn_CPT_Downlink.csv",
    #               "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_ParrotAR2/bn_ckpt/parrotar2_reliability_bn_CPT_Uplink.csv",
    #               "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_ParrotAR2/bn_ckpt/parrotar2_reliability_bn_CPT_Video.csv"]
    BN_MODELS = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/bn_cpts/djispark_reliability_bn_CPT_Downlink.csv",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/bn_cpts/djispark_reliability_bn_CPT_Uplink.csv",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/bn_cpts/djispark_reliability_bn_CPT_Video.csv",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/bn_cpts/djimavicair_reliability_bn_CPT_Downlink.csv",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/bn_cpts/djimavicair_reliability_bn_CPT_Uplink.csv",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/bn_cpts/djimavicair_reliability_bn_CPT_Video.csv",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/bn_cpts/parrotar2_reliability_bn_CPT_Downlink.csv",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/bn_cpts/parrotar2_reliability_bn_CPT_Uplink.csv",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/bn_cpts/parrotar2_reliability_bn_CPT_Video.csv"]
    
    # NN_CALIBRATION_T_REL_CLASS = [0.9557206402188666, 1.2346639186950543, 0.5542500457173349, 0.6373688678731613, 0.9226032479078328, 1.3807097489210691, 0.9325759791937871, 0.5262883084699191, 0.8572615214148442]
    # NN_CALIBRATION_T_FAIL_CLASS = [0.9820691029067591, 1.2036481086991644, 0.7214191586160914, 0.8194965149005633, 1.1311071800932562, 0.9102878467033356, 0.9471861126724312, 0.8664543046440363, 0.625642688757027]
    # NN_CALIBRATION_T_F1_99 = [0.9444038738718499, 1.1494277249882265, 0.7100475414731251, 0.8183141026020989, 1.1213350284587758, 0.8964227658377057, 0.9471861126724312, 0.8663323103931915, 0.5979824643149598]
    # NN_CALIBRATION_T_F1_999 = [1.0403856550957589, 1.0947041510241227, 0.6707863324662134, 0.6178929785149132, 0.987124369961716, 0.6056186348626422, 0.865862468406123, 0.7757441037173518, 0.5406077162110817]
    NN_CALIBRATION_T_REL_CLASS = [0.922603247907832,0.836886680628517,1.181442319525770,0.695943774182421,0.977356604613259,1.038960337122830,0.909984079737721,0.698366414918771,0.971787764023628]
    NN_CALIBRATION_T_FAIL_CLASS = [0.303917065584829,0.920296935119101,0.350717794567156,0.316120754914620,0.300696504475379,0.920296935119101,0.920296935119101,0.920296935119101,0.920296935119101]
    NN_CALIBRATION_T_F1_99 = [0.818999033604945,1.087312760346270,0.858669947475337,0.796951993479910,0.960775197842332,0.844523227576418,0.793758852611129,1.065597390300470,0.875699333281340]
    NN_CALIBRATION_T_F1_999 = [0.779411217160536,1.157225181523750,0.599937827904500,0.559314656013430,0.889859201126994,0.663178585253110,0.863737664154734,0.934357536797462,0.778013117939757]
    NN_CALIBRATION_T_ACC_90 = [0.912663159480207,0.820144504692431,1.21358819712678,0.854196310294309,1.09470415102412,0.978148830572879,0.986659033027988,0.761941134913359,0.970326989540616]
    NN_CALIBRATION_T_ACC_99 = [0.815413750468373,1.08648885051258,0.859015293508804,0.797558307420914,0.987359474443279,0.843720268031539,0.793712931418296,1.06507326409254,0.874714345559215]
    NN_CALIBRATION_T_ACC_999 = [0.83938593546566,1.16017795210948,0.609308465589379,0.5671248296959,0.917065055936845,0.617897602874294,0.864148480049065,0.927027676627532,0.778013117939757 ]
    NN_CALIBRATION_T_SPECIFICITY_90 = [1.69337168339217,1.66494184128457,1.62717176597833,1.67067206490783,1.69388681134943,1.62717176597833,1.68294143647874,1.68294143647874,1.62757529834149]
    NN_CALIBRATION_T_SPECIFICITY_99 = [1.69561830451873,1.67551651082116,1.62717176597833,1.68294143647874,1.699369465806,1.69388681134943,1.68294143647874,1.24824445192228,1.69602271196495]
    NN_CALIBRATION_T_SPECIFICITY_999 = [1.69615238212838,1.68516226329027,1.62717176597833,1.65478099337274,1.66494184128457,1.69388681134943,1.69613224836134,1.65423484686518,1.68024126811074]
    NN_CALIBRATION_T_RECALL_90 = [0.305872952502177,0.305782888770369,0.389381167967074,0.350717794567156,0.308822700670607,0.443677835314617,0.350717794567156,0.300192076811355,0.443677835314617]
    NN_CALIBRATION_T_RECALL_99 = [0.443677835314617,0.400846486390357,0.406645786807994,0.406645786807994,0.443677835314617,0.30218054325172,0.318330425172217,0.311094380927188,0.443677835314617]
    NN_CALIBRATION_T_RECALL_999 = [0.443677835314617,0.443677835314617,0.401718190931968,0.443677835314617,0.443677835314617,0.318057706523196,0.318057706523196,0.443677835314617,0.30218054325172]
    TEST_DATASET = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/test_dataset_{}_processed/Downlink_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/test_dataset_{}_processed/Uplink_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/test_dataset_{}_processed/Video_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/test_dataset_{}_processed/Downlink_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/test_dataset_{}_processed/Uplink_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/test_dataset_{}_processed/Video_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/test_dataset_{}_processed/Downlink_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/test_dataset_{}_processed/Uplink_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/test_dataset_{}_processed/Video_Reliability.csv"]
    # SAVE_PATH = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/Results_NP100000_DJISpark_Downlink_Reliability.csv",
    #                   "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/Results_NP100000_DJISpark_Uplink_Reliability.csv",
    #                   "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/Results_NP100000_DJISpark_Video_Reliability.csv",
    #                   "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/Results_NP100000_DJIMavicAir_Downlink_Reliability.csv",
    #                   "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/Results_NP100000_DJIMavicAir_Uplink_Reliability.csv",
    #                   "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJIMavicAir/Results_NP100000_DJIMavicAir_Video_Reliability.csv",
    #                   "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/Results_NP100000_ParrotAR2_Downlink_Reliability.csv",
    #                   "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/Results_NP100000_ParrotAR2_Uplink_Reliability.csv",
    #                   "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_ParrotAR2/Results_NP100000_ParrotAR2_Video_Reliability.csv"]
    # METRIC_SAVE_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/Reliability_Prediction_Results_Obj2_final_model.csv"
    # CONFUSION_METRIC_SAVE_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/Reliability_Prediction_TP_TN_FP_FN_Obj2_temp.csv"
    SAVE_PATH = ["/media/research-student/KingstonSSD/FANET_Dataset/temp_obj2_results_alt/Results_NP100000_DJISpark_Downlink_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/temp_obj2_results_alt/Results_NP100000_DJISpark_Uplink_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/temp_obj2_results_alt/Results_NP100000_DJISpark_Video_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/temp_obj2_results_alt/Results_NP100000_DJIMavicAir_Downlink_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/temp_obj2_results_alt/Results_NP100000_DJIMavicAir_Uplink_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/temp_obj2_results_alt/Results_NP100000_DJIMavicAir_Video_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/temp_obj2_results_alt/Results_NP100000_ParrotAR2_Downlink_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/temp_obj2_results_alt/Results_NP100000_ParrotAR2_Uplink_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/temp_obj2_results_alt/Results_NP100000_ParrotAR2_Video_Reliability.csv"]
    METRIC_SAVE_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/temp_obj2_results_alt/Reliability_Prediction_Results_Obj2_final_model.csv"
    CONFUSION_METRIC_SAVE_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/temp_obj2_results_alt/Reliability_Prediction_TP_TN_FP_FN_Obj2_temp.csv"
    MIN_FAILURE_PROB = 0.5 # Sum of probabilities of all failure modes needs to be at least 50%
    MAX_HDIST = 700 # Since the test datasets goes up to 1200m
    F_BETA = 0.5 # Use 0.5 if precision is twice as important as recall; Use 2 if recall is twice as important as precision
    data_dtypes = {"Horizontal_Distance": np.float64, "Height":np.float64, "UAV_Sending_Interval": np.float64, "Modulation": 'str', "Bitrate": np.float64}
    # For BN Model
    HDIST_BIN = np.arange(0, 710, 10) # For associating each hdist to its nearest value in train dataset
    HEIGHT_BIN = np.arange(60, 330, 30) # For associating each height to its nearest value in train dataset
    metrics_list = [] # To store accuracy, F1, specificity, sensitivity
    confusion_metrics_list = [] # To store TP, TN, FP, FN
    for i in range(len(NN_MODELS)):
        ''' Load NN Model '''
        nn_model = tf.keras.models.load_model(NN_MODELS[i], compile=False)
        nn_model.compile(optimizer='adam', 
                    loss={'packet_state': 'categorical_crossentropy'},
                    metrics={'packet_state': 'accuracy'})
        nn_model_no_act = build_nn_model_v4_wobatchnorm_noactivation() # For testing calibrated NN
        nn_model_no_act.set_weights(nn_model.get_weights())
        ''' Load BN Model '''
        bn_cpt_df = pd.read_csv(BN_MODELS[i])
        # for j in [1, 2]: # FOR EACH TEST DATASET 1 and 2
        ''' Load Test Dataset '''
        test_dataset_path = TEST_DATASET[i]
        test_data_df_1 = pd.read_csv(TEST_DATASET[i].format(1), dtype=data_dtypes)
        test_data_df_2 = pd.read_csv(TEST_DATASET[i].format(2), dtype=data_dtypes)
        test_data_df = pd.concat([test_data_df_1, test_data_df_2], ignore_index=True)
        test_data_df = test_data_df.loc[test_data_df["Horizontal_Distance"] <= MAX_HDIST]
        test_data_df = get_mcs_index(test_data_df)
        test_data_df["Reliability"] = (test_data_df["Num_Reliable"] / test_data_df["Num_Sent"]).values
        test_data_df["Delay_Excd_Prob"] = (test_data_df["Num_Delay_Excd"] / test_data_df["Num_Sent"]).values
        test_data_df["Queue_Overflow_Prob"] = (test_data_df["Num_Q_Overflow"] / test_data_df["Num_Sent"]).values
        test_data_df["Incr_Rcvd_Prob"] = (test_data_df["Num_Incr_Rcvd"] / test_data_df["Num_Sent"]).values
        test_data_df["Failure_Mode"] = test_data_df[["Queue_Overflow_Prob", "Incr_Rcvd_Prob", "Delay_Excd_Prob"]].idxmax(axis=1)
        # test_data_df.loc[(test_data_df["Queue_Overflow_Prob"] < MIN_FAILURE_PROB) & (test_data_df["Incr_Rcvd_Prob"] < MIN_FAILURE_PROB) 
        #                 & (test_data_df["Delay_Excd_Prob"] < MIN_FAILURE_PROB),["Failure_Mode"]] = "None"
        test_data_df.loc[(test_data_df["Reliability"] >= (1-MIN_FAILURE_PROB)),["Failure_Mode"]] = "None" # To filter out failure modes where reliability 
        test_data_df["Reliable_State_90"] = test_data_df["Reliability"] >= 0.9
        test_data_df["Reliable_State_99"] = test_data_df["Reliability"] >= 0.99
        test_data_df["Reliable_State_999"] = test_data_df["Reliability"] >= 0.999
        ''' Test BN Model '''
        # Associating each horizontal distance and height with the closest values in training dataset:
        test_data_df["Horizontal_Distance_Class"] = pd.cut(test_data_df["Horizontal_Distance"], bins=HDIST_BIN, right=False, include_lowest=True, labels=np.arange(0, len(HDIST_BIN)-1))
        # test_data_df["Horizontal_Distance_Class"] = test_data_df["Horizontal_Distance"].apply(find_nearest_index, args=([HDIST_BIN]))
        test_data_df["Height_Class"] = pd.cut(test_data_df["Height"], bins=HEIGHT_BIN, right=False, include_lowest=True, labels=np.arange(0, len(HEIGHT_BIN)-1))
        # test_data_df["Height_Class"] = test_data_df["Height"].apply(find_nearest_index, args=([HEIGHT_BIN]))
        test_data_df["UAV_Sending_Interval_Class"] = test_data_df["UAV_Sending_Interval"].replace({10.0:0, 20.0:1, 66.7:2, 100.0:3}) # Change sending interval categorial to numeric
        test_data_df["Reliability_Class"] = pd.cut(test_data_df["Reliability"], bins=[-0.1,0.5,0.7,0.9,1], labels=["Low", "ModeratelyLow", "ModeratelyHigh", "High"])
        predicted_reliability = [] # To store reliability predictions
        predicted_incr_rcvd = [] # To store incorrectly received probability predictions
        predicted_delay_excd = [] # To store delay exceeded probability predictions
        predicted_q_ovflw = [] # To store queue overflow probability predictions
        for row in test_data_df.itertuples():
            # predictions = cpt_df.loc[(row.Horizontal_Distance_Class,row.Height_Class,row.UAV_Sending_Interval_Class,row.MCS)]
            bn_predictions = bn_cpt_df.loc[(bn_cpt_df["Horizontal_Distance_Class"]==row.Horizontal_Distance_Class) & (bn_cpt_df["Height_Class"]==row.Height_Class) &
                                           (bn_cpt_df["UAV_Sending_Interval_Class"]==row.UAV_Sending_Interval_Class) & (bn_cpt_df["MCS"]==row.MCS)]
            # print(bn_predictions)
            try:
                predicted_reliability.append(bn_predictions["Reliability"].values[0])
                predicted_delay_excd.append(bn_predictions["Prob_Delay_Excd"].values[0])
                predicted_q_ovflw.append(bn_predictions["Prob_Queue_Overflow"].values[0])
                predicted_incr_rcvd.append(bn_predictions["Prob_Incr_Rcvd"].values[0])
            except:
                print(row)
                print(bn_predictions)
                print(i)
                break
        test_data_df['BN_Predicted_Reliability'] = predicted_reliability
        test_data_df['BN_Predicted_Delay_Excd_Prob'] = predicted_delay_excd
        test_data_df['BN_Predicted_Queue_Overflow_Prob'] = predicted_q_ovflw
        test_data_df['BN_Predicted_Incr_Rcvd_Prob'] = predicted_incr_rcvd
        test_data_df["BN_Predicted_Reliability_Class"] = pd.cut(test_data_df["BN_Predicted_Reliability"], bins=[-0.1,0.5,0.7,0.9,1], labels=["Low", "ModeratelyLow", "ModeratelyHigh", "High"])
        test_data_df["BN_Predicted_Failure_Mode"] = test_data_df[["BN_Predicted_Queue_Overflow_Prob", "BN_Predicted_Incr_Rcvd_Prob", "BN_Predicted_Delay_Excd_Prob"]].idxmax(axis=1)
        # test_data_df.loc[(test_data_df["BN_Predicted_Queue_Overflow_Prob"] < MIN_FAILURE_PROB) & (test_data_df["BN_Predicted_Incr_Rcvd_Prob"] < MIN_FAILURE_PROB) 
        #                 & (test_data_df["BN_Predicted_Delay_Excd_Prob"] < MIN_FAILURE_PROB),["BN_Predicted_Failure_Mode"]] = "None"
        test_data_df["BN_Predicted_Reliable_State_90"] = test_data_df['BN_Predicted_Reliability'] >= 0.9
        test_data_df["BN_Predicted_Reliable_State_99"] = test_data_df['BN_Predicted_Reliability'] >= 0.99
        test_data_df["BN_Predicted_Reliable_State_999"] = test_data_df['BN_Predicted_Reliability'] >= 0.999
        ''' Test NN Model '''
        test_data_df = normalize_data(test_data_df, columns=["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"], save_details_path=None)
        nn_prediction = nn_model.predict(test_data_df[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"]].values)
        test_data_df['NN_Predicted_Reliability'] = [prob[0] for prob in nn_prediction]
        test_data_df['NN_Predicted_Queue_Overflow_Prob'] = [prob[1] for prob in nn_prediction]
        test_data_df['NN_Predicted_Incr_Rcvd_Prob'] = [prob[2] for prob in nn_prediction]
        test_data_df['NN_Predicted_Delay_Excd_Prob'] = [prob[3] for prob in nn_prediction]
        test_data_df["NN_Predicted_Reliability_Class"] = pd.cut(test_data_df["NN_Predicted_Reliability"], bins=[-0.1,0.5,0.7,0.9,1], labels=["Low", "ModeratelyLow", "ModeratelyHigh", "High"])
        test_data_df["NN_Predicted_Failure_Mode"] = test_data_df[["NN_Predicted_Queue_Overflow_Prob", "NN_Predicted_Incr_Rcvd_Prob", "NN_Predicted_Delay_Excd_Prob"]].idxmax(axis=1)
        # test_data_df.loc[(test_data_df["NN_Predicted_Queue_Overflow_Prob"] < MIN_FAILURE_PROB) & (test_data_df["NN_Predicted_Incr_Rcvd_Prob"] < MIN_FAILURE_PROB) 
        #                 & (test_data_df["NN_Predicted_Delay_Excd_Prob"] < MIN_FAILURE_PROB),["NN_Predicted_Failure_Mode"]] = "None"
        test_data_df["NN_Predicted_Reliable_State_90"] = test_data_df['NN_Predicted_Reliability'] >= 0.9
        test_data_df["NN_Predicted_Reliable_State_99"] = test_data_df['NN_Predicted_Reliability'] >= 0.99
        test_data_df["NN_Predicted_Reliable_State_999"] = test_data_df['NN_Predicted_Reliability'] >= 0.999
        ''' Test Calibrated NN Model '''
        T_REL_CLASS = NN_CALIBRATION_T_REL_CLASS[i]
        T_FAIL_CLASS = NN_CALIBRATION_T_FAIL_CLASS[i]
        T_ACC_90 = NN_CALIBRATION_T_ACC_90[i]
        T_ACC_99 = NN_CALIBRATION_T_ACC_99[i]
        T_ACC_999 = NN_CALIBRATION_T_ACC_999[i]
        T_SPECIFICITY_90 = NN_CALIBRATION_T_SPECIFICITY_90[i]
        T_SPECIFICITY_99 = NN_CALIBRATION_T_SPECIFICITY_99[i]
        T_SPECIFICITY_999 = NN_CALIBRATION_T_SPECIFICITY_999[i]
        T_RECALL_90 = NN_CALIBRATION_T_RECALL_90[i]
        T_RECALL_99 = NN_CALIBRATION_T_RECALL_99[i]
        T_RECALL_999 = NN_CALIBRATION_T_RECALL_999[i]
        nn_logits = nn_model_no_act.predict(test_data_df[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"]].values)
        calib_nn_prediction_reliability = [activations.softmax(K.constant([logits/T_REL_CLASS]), axis=-1)[0].numpy() for logits in nn_logits]
        calib_nn_prediction_failure_mode = [activations.softmax(K.constant([logits/T_FAIL_CLASS]), axis=-1)[0].numpy() for logits in nn_logits]
        test_data_df['CalibNN_Predicted_Reliability'] = [prob[0] for prob in calib_nn_prediction_reliability]
        test_data_df['CalibNN_Predicted_Queue_Overflow_Prob'] = [prob[1] for prob in calib_nn_prediction_failure_mode]
        test_data_df['CalibNN_Predicted_Incr_Rcvd_Prob'] = [prob[2] for prob in calib_nn_prediction_failure_mode]
        test_data_df['CalibNN_Predicted_Delay_Excd_Prob'] = [prob[3] for prob in calib_nn_prediction_failure_mode]
        test_data_df["CalibNN_Predicted_Reliability_Class"] = pd.cut(test_data_df["CalibNN_Predicted_Reliability"], bins=[-0.1,0.5,0.7,0.9,1], labels=["Low", "ModeratelyLow", "ModeratelyHigh", "High"])
        test_data_df["CalibNN_Predicted_Failure_Mode"] = test_data_df[["CalibNN_Predicted_Queue_Overflow_Prob", "CalibNN_Predicted_Incr_Rcvd_Prob", "CalibNN_Predicted_Delay_Excd_Prob"]].idxmax(axis=1)
        # test_data_df.loc[(test_data_df["CalibNN_Predicted_Queue_Overflow_Prob"] < MIN_FAILURE_PROB) & (test_data_df["CalibNN_Predicted_Incr_Rcvd_Prob"] < MIN_FAILURE_PROB) 
        #                 & (test_data_df["CalibNN_Predicted_Delay_Excd_Prob"] < MIN_FAILURE_PROB),["CalibNN_Predicted_Failure_Mode"]] = "None"
        # test_data_df["CalibNN_F1_Predicted_Reliability_99"] = np.array([activations.softmax(K.constant([logits/T_F1_99]), axis=-1)[0][0].numpy() for logits in nn_logits])
        # test_data_df["CalibNN_F1_Predicted_Reliability_999"] = np.array([activations.softmax(K.constant([logits/T_F1_999]), axis=-1)[0][0].numpy() for logits in nn_logits])
        # test_data_df["CalibNN_F1_Predicted_Reliable_State_99"] = test_data_df["CalibNN_F1_Predicted_Reliability_99"] >= 0.99
        # test_data_df["CalibNN_F1_Predicted_Reliable_State_999"] = test_data_df["CalibNN_F1_Predicted_Reliability_999"] >= 0.999
        test_data_df["CalibNN_Acc_Predicted_Reliability_90"] = np.array([activations.softmax(K.constant([logits/T_ACC_90]), axis=-1)[0][0].numpy() for logits in nn_logits])
        test_data_df["CalibNN_Acc_Predicted_Reliability_99"] = np.array([activations.softmax(K.constant([logits/T_ACC_99]), axis=-1)[0][0].numpy() for logits in nn_logits])
        test_data_df["CalibNN_Acc_Predicted_Reliability_999"] = np.array([activations.softmax(K.constant([logits/T_ACC_999]), axis=-1)[0][0].numpy() for logits in nn_logits])
        test_data_df["CalibNN_Acc_Predicted_Reliable_State_90"] = test_data_df["CalibNN_Acc_Predicted_Reliability_90"] >= 0.9
        test_data_df["CalibNN_Acc_Predicted_Reliable_State_99"] = test_data_df["CalibNN_Acc_Predicted_Reliability_99"] >= 0.99
        test_data_df["CalibNN_Acc_Predicted_Reliable_State_999"] = test_data_df["CalibNN_Acc_Predicted_Reliability_999"] >= 0.999
        test_data_df["CalibNN_Specificity_Predicted_Reliability_90"] = np.array([activations.softmax(K.constant([logits/T_SPECIFICITY_90]), axis=-1)[0][0].numpy() for logits in nn_logits])
        test_data_df["CalibNN_Specificity_Predicted_Reliability_99"] = np.array([activations.softmax(K.constant([logits/T_SPECIFICITY_99]), axis=-1)[0][0].numpy() for logits in nn_logits])
        test_data_df["CalibNN_Specificity_Predicted_Reliability_999"] = np.array([activations.softmax(K.constant([logits/T_SPECIFICITY_999]), axis=-1)[0][0].numpy() for logits in nn_logits])
        test_data_df["CalibNN_Specificity_Predicted_Reliable_State_90"] = test_data_df["CalibNN_Specificity_Predicted_Reliability_90"] >= 0.9
        test_data_df["CalibNN_Specificity_Predicted_Reliable_State_99"] = test_data_df["CalibNN_Specificity_Predicted_Reliability_99"] >= 0.99
        test_data_df["CalibNN_Specificity_Predicted_Reliable_State_999"] = test_data_df["CalibNN_Specificity_Predicted_Reliability_999"] >= 0.999
        test_data_df["CalibNN_Recall_Predicted_Reliability_90"] = np.array([activations.softmax(K.constant([logits/T_RECALL_90]), axis=-1)[0][0].numpy() for logits in nn_logits])
        test_data_df["CalibNN_Recall_Predicted_Reliability_99"] = np.array([activations.softmax(K.constant([logits/T_RECALL_99]), axis=-1)[0][0].numpy() for logits in nn_logits])
        test_data_df["CalibNN_Recall_Predicted_Reliability_999"] = np.array([activations.softmax(K.constant([logits/T_RECALL_999]), axis=-1)[0][0].numpy() for logits in nn_logits])
        test_data_df["CalibNN_Recall_Predicted_Reliable_State_90"] = test_data_df["CalibNN_Recall_Predicted_Reliability_90"] >= 0.9
        test_data_df["CalibNN_Recall_Predicted_Reliable_State_99"] = test_data_df["CalibNN_Recall_Predicted_Reliability_99"] >= 0.99
        test_data_df["CalibNN_Recall_Predicted_Reliable_State_999"] = test_data_df["CalibNN_Recall_Predicted_Reliability_999"] >= 0.999
        ''' Save Results '''
        test_data_df.to_csv(SAVE_PATH[i], index=False)
        ''' Calulate metrics '''
        # Reliability Class Accuracy
        bn_reliability_class_accuracy = accuracy_score(test_data_df["Reliability_Class"], test_data_df["BN_Predicted_Reliability_Class"])
        nn_reliability_class_accuracy = accuracy_score(test_data_df["Reliability_Class"], test_data_df["NN_Predicted_Reliability_Class"])
        calib_nn_reliability_class_accuracy = accuracy_score(test_data_df["Reliability_Class"], test_data_df["CalibNN_Predicted_Reliability_Class"])
        ''' Reliability Class Specificity and Sensitivity '''
        # BN Reliability Class Specificity and Sensitivity
        specificity = []
        sensitivity = []
        for l in ["Low", "ModeratelyLow", "ModeratelyHigh", "High"]:
            spec = recall_score(test_data_df["Reliability_Class"].to_numpy()==l, test_data_df["BN_Predicted_Reliability_Class"].to_numpy()==l, pos_label=False, average='binary', zero_division=np.nan)
            sens = recall_score(test_data_df["Reliability_Class"].to_numpy()==l, test_data_df["BN_Predicted_Reliability_Class"].to_numpy()==l, pos_label=True, average='binary', zero_division=np.nan)
            specificity.append(spec)
            sensitivity.append(sens)
        bn_reliability_class_specificity = np.nanmean(specificity)
        bn_reliability_class_sensitivity = np.nanmean(sensitivity)
        # NN Reliability Class Specificity and Sensitivity
        specificity = []
        sensitivity = []
        for l in ["Low", "ModeratelyLow", "ModeratelyHigh", "High"]:
            spec = recall_score(test_data_df["Reliability_Class"].to_numpy()==l, test_data_df["NN_Predicted_Reliability_Class"].to_numpy()==l, pos_label=False, average='binary', zero_division=np.nan)
            sens = recall_score(test_data_df["Reliability_Class"].to_numpy()==l, test_data_df["NN_Predicted_Reliability_Class"].to_numpy()==l, pos_label=True, average='binary', zero_division=np.nan)
            specificity.append(spec)
            sensitivity.append(sens)
        nn_reliability_class_specificity = np.nanmean(specificity)
        nn_reliability_class_sensitivity = np.nanmean(sensitivity)
        # Cal. NN Reliability Class Specificity and Sensitivity
        specificity = []
        sensitivity = []
        for l in ["Low", "ModeratelyLow", "ModeratelyHigh", "High"]:
            spec = recall_score(test_data_df["Reliability_Class"].to_numpy()==l, test_data_df["CalibNN_Predicted_Reliability_Class"].to_numpy()==l, pos_label=False, average='binary', zero_division=np.nan)
            sens = recall_score(test_data_df["Reliability_Class"].to_numpy()==l, test_data_df["CalibNN_Predicted_Reliability_Class"].to_numpy()==l, pos_label=True, average='binary', zero_division=np.nan)
            specificity.append(spec)
            sensitivity.append(sens)
        calib_nn_reliability_class_specificity = np.nanmean(specificity)
        calib_nn_reliability_class_sensitivity = np.nanmean(sensitivity)

        # Failure Mode Accuracy
        test_data_fail_df = test_data_df[test_data_df["Failure_Mode"] != "None"]
        failure_mode = test_data_fail_df["Failure_Mode"].replace({"Queue_Overflow_Prob":1, "Incr_Rcvd_Prob":2, "Delay_Excd_Prob":3, "None":4})
        bn_failure_mode_predicted = test_data_fail_df["BN_Predicted_Failure_Mode"].replace({"BN_Predicted_Queue_Overflow_Prob":1, "BN_Predicted_Incr_Rcvd_Prob":2, "BN_Predicted_Delay_Excd_Prob":3, "None":4})
        bn_failure_mode_accuracy = accuracy_score(failure_mode, bn_failure_mode_predicted)
        nn_failure_mode_predicted = test_data_fail_df["NN_Predicted_Failure_Mode"].replace({"NN_Predicted_Queue_Overflow_Prob":1, "NN_Predicted_Incr_Rcvd_Prob":2, "NN_Predicted_Delay_Excd_Prob":3, "None":4})
        nn_failure_mode_accuracy = accuracy_score(failure_mode, nn_failure_mode_predicted)
        calib_nn_failure_mode_predicted = test_data_fail_df["CalibNN_Predicted_Failure_Mode"].replace({"CalibNN_Predicted_Queue_Overflow_Prob":1, "CalibNN_Predicted_Incr_Rcvd_Prob":2, "CalibNN_Predicted_Delay_Excd_Prob":3, "None":4})
        calib_nn_failure_mode_accuracy = accuracy_score(failure_mode, calib_nn_failure_mode_predicted)
        ''' Failure Mode Specificity and Sensitivity ''' # NOTE: There should not be any None in the failure mode
        # BN Failure Mode Specificity and Sensitivity
        specificity = []
        sensitivity = []
        for l in [1,2,3]:
            spec = recall_score(failure_mode.to_numpy()==l, bn_failure_mode_predicted.to_numpy()==l, pos_label=False, average='binary', zero_division=np.nan)
            sens = recall_score(failure_mode.to_numpy()==l, bn_failure_mode_predicted.to_numpy()==l, pos_label=True, average='binary', zero_division=np.nan)
            specificity.append(spec)
            sensitivity.append(sens)
        bn_failure_mode_specificity = np.nanmean(specificity)
        bn_failure_mode_sensitivity = np.nanmean(sensitivity)
        # NN Failure Mode Specificity and Sensitivity
        specificity = []
        sensitivity = []
        for l in [1,2,3]:
            spec = recall_score(failure_mode.to_numpy()==l, nn_failure_mode_predicted.to_numpy()==l, pos_label=False, average='binary', zero_division=np.nan)
            sens = recall_score(failure_mode.to_numpy()==l, nn_failure_mode_predicted.to_numpy()==l, pos_label=True, average='binary', zero_division=np.nan)
            specificity.append(spec)
            sensitivity.append(sens)
        nn_failure_mode_specificity = np.nanmean(specificity)
        nn_failure_mode_sensitivity = np.nanmean(sensitivity)
        # Cal NN Failure Mode Specificity and Sensitivity
        specificity = []
        sensitivity = []
        for l in [1,2,3]:
            spec = recall_score(failure_mode.to_numpy()==l, calib_nn_failure_mode_predicted.to_numpy()==l, pos_label=False, average='binary', zero_division=np.nan)
            sens = recall_score(failure_mode.to_numpy()==l, calib_nn_failure_mode_predicted.to_numpy()==l, pos_label=True, average='binary', zero_division=np.nan)
            specificity.append(spec)
            sensitivity.append(sens)
        calib_nn_failure_mode_specificity = np.nanmean(specificity)
        calib_nn_failure_mode_sensitivity = np.nanmean(sensitivity)

        '''Reliabiliy State Accuracy'''
        bn_reliability_state_accuracy_90 = accuracy_score(test_data_df["Reliable_State_90"], test_data_df["BN_Predicted_Reliable_State_90"])
        bn_reliability_state_accuracy_99 = accuracy_score(test_data_df["Reliable_State_99"], test_data_df["BN_Predicted_Reliable_State_99"])
        bn_reliability_state_accuracy_999 = accuracy_score(test_data_df["Reliable_State_999"], test_data_df["BN_Predicted_Reliable_State_999"])
        nn_reliability_state_accuracy_90 = accuracy_score(test_data_df["Reliable_State_90"], test_data_df["NN_Predicted_Reliable_State_90"])
        nn_reliability_state_accuracy_99 = accuracy_score(test_data_df["Reliable_State_99"], test_data_df["NN_Predicted_Reliable_State_99"])
        nn_reliability_state_accuracy_999 = accuracy_score(test_data_df["Reliable_State_999"], test_data_df["NN_Predicted_Reliable_State_999"])
        calibnn_reliability_state_accuracy_90 = accuracy_score(test_data_df["Reliable_State_90"], test_data_df["CalibNN_Acc_Predicted_Reliable_State_90"])
        calibnn_reliability_state_accuracy_99 = accuracy_score(test_data_df["Reliable_State_99"], test_data_df["CalibNN_Acc_Predicted_Reliable_State_99"])
        calibnn_reliability_state_accuracy_999 = accuracy_score(test_data_df["Reliable_State_999"], test_data_df["CalibNN_Acc_Predicted_Reliable_State_999"])
        '''Reliabiliy State F1 Score'''
        # bn_reliability_state_f1_99 = f1_score(test_data_df["Reliable_State_99"], test_data_df["BN_Predicted_Reliable_State_99"])
        # bn_reliability_state_f1_999 = f1_score(test_data_df["Reliable_State_999"], test_data_df["BN_Predicted_Reliable_State_999"])
        # nn_reliability_state_f1_99 = f1_score(test_data_df["Reliable_State_99"], test_data_df["NN_Predicted_Reliable_State_99"])
        # nn_reliability_state_f1_999 = f1_score(test_data_df["Reliable_State_999"], test_data_df["NN_Predicted_Reliable_State_999"])
        # calibnn_f1_reliability_state_f1_99 = f1_score(test_data_df["Reliable_State_99"], test_data_df["CalibNN_F1_Predicted_Reliable_State_99"])
        # calibnn_f1_reliability_state_f1_999 = f1_score(test_data_df["Reliable_State_999"], test_data_df["CalibNN_F1_Predicted_Reliable_State_999"])
        # calibnn_specificity_reliability_state_f1_99 = f1_score(test_data_df["Reliable_State_99"], test_data_df["CalibNN_Specificity_Predicted_Reliable_State_99"])
        # calibnn_specificity_reliability_state_f1_999 = f1_score(test_data_df["Reliable_State_999"], test_data_df["CalibNN_Specificity_Predicted_Reliable_State_999"])
        # calibnn_recall_reliability_state_f1_99 = f1_score(test_data_df["Reliable_State_99"], test_data_df["CalibNN_Recall_Predicted_Reliable_State_99"])
        # calibnn_recall_reliability_state_f1_999 = f1_score(test_data_df["Reliable_State_999"], test_data_df["CalibNN_Recall_Predicted_Reliable_State_999"])
        '''Reliabiliy State F_beta Score'''
        # bn_reliability_state_fbeta_95 = fbeta_score(test_data_df["Reliable_State_95"], test_data_df["BN_Predicted_Reliable_State_95"], beta=F_BETA)
        # bn_reliability_state_fbeta_99 = fbeta_score(test_data_df["Reliable_State_99"], test_data_df["BN_Predicted_Reliable_State_99"], beta=F_BETA)
        # bn_reliability_state_fbeta_999 = fbeta_score(test_data_df["Reliable_State_999"], test_data_df["BN_Predicted_Reliable_State_999"], beta=F_BETA)
        # nn_reliability_state_fbeta_99 = fbeta_score(test_data_df["Reliable_State_99"], test_data_df["NN_Predicted_Reliable_State_99"], beta=F_BETA)
        # nn_reliability_state_fbeta_999 = fbeta_score(test_data_df["Reliable_State_999"], test_data_df["NN_Predicted_Reliable_State_999"], beta=F_BETA)
        # calibnn_reliability_state_fbeta_99 = fbeta_score(test_data_df["Reliable_State_99"], test_data_df["CalibNN_Predicted_Reliable_State_99"], beta=F_BETA)
        # calibnn_reliability_state_fbeta_999 = fbeta_score(test_data_df["Reliable_State_999"], test_data_df["CalibNN_Predicted_Reliable_State_999"], beta=F_BETA)
        '''Reliabiliy State Specificity Score'''
        bn_reliability_state_specificity_90 = recall_score(test_data_df["Reliable_State_90"], test_data_df["BN_Predicted_Reliable_State_90"], pos_label=False, average='binary')
        bn_reliability_state_specificity_99 = recall_score(test_data_df["Reliable_State_99"], test_data_df["BN_Predicted_Reliable_State_99"], pos_label=False, average='binary')
        bn_reliability_state_specificity_999 = recall_score(test_data_df["Reliable_State_999"], test_data_df["BN_Predicted_Reliable_State_999"], pos_label=False, average='binary')
        nn_reliability_state_specificity_90 = recall_score(test_data_df["Reliable_State_90"], test_data_df["NN_Predicted_Reliable_State_90"], pos_label=False, average='binary')
        nn_reliability_state_specificity_99 = recall_score(test_data_df["Reliable_State_99"], test_data_df["NN_Predicted_Reliable_State_99"], pos_label=False, average='binary')
        nn_reliability_state_specificity_999 = recall_score(test_data_df["Reliable_State_999"], test_data_df["NN_Predicted_Reliable_State_999"], pos_label=False, average='binary')
        # calibnn_f1_reliability_state_specificity_99 = recall_score(test_data_df["Reliable_State_99"], test_data_df["CalibNN_F1_Predicted_Reliable_State_99"], pos_label=False, average='binary')
        # calibnn_f1_reliability_state_specificity_999 = recall_score(test_data_df["Reliable_State_999"], test_data_df["CalibNN_F1_Predicted_Reliable_State_999"], pos_label=False, average='binary')
        calibnn_specificity_reliability_state_specificity_90 = recall_score(test_data_df["Reliable_State_90"], test_data_df["CalibNN_Specificity_Predicted_Reliable_State_90"], pos_label=False, average='binary')
        calibnn_specificity_reliability_state_specificity_99 = recall_score(test_data_df["Reliable_State_99"], test_data_df["CalibNN_Specificity_Predicted_Reliable_State_99"], pos_label=False, average='binary')
        calibnn_specificity_reliability_state_specificity_999 = recall_score(test_data_df["Reliable_State_999"], test_data_df["CalibNN_Specificity_Predicted_Reliable_State_999"], pos_label=False, average='binary')
        # calibnn_recall_reliability_state_specificity_99 = recall_score(test_data_df["Reliable_State_99"], test_data_df["CalibNN_Recall_Predicted_Reliable_State_99"], pos_label=False, average='binary')
        # calibnn_recall_reliability_state_specificity_999 = recall_score(test_data_df["Reliable_State_999"], test_data_df["CalibNN_Recall_Predicted_Reliable_State_999"], pos_label=False, average='binary')
        '''Reliabiliy State Recall (Sensitivity) Score'''
        bn_reliability_state_recall_90 = recall_score(test_data_df["Reliable_State_90"], test_data_df["BN_Predicted_Reliable_State_90"])
        bn_reliability_state_recall_99 = recall_score(test_data_df["Reliable_State_99"], test_data_df["BN_Predicted_Reliable_State_99"])
        bn_reliability_state_recall_999 = recall_score(test_data_df["Reliable_State_999"], test_data_df["BN_Predicted_Reliable_State_999"])
        nn_reliability_state_recall_90 = recall_score(test_data_df["Reliable_State_90"], test_data_df["NN_Predicted_Reliable_State_90"])
        nn_reliability_state_recall_99 = recall_score(test_data_df["Reliable_State_99"], test_data_df["NN_Predicted_Reliable_State_99"])
        nn_reliability_state_recall_999 = recall_score(test_data_df["Reliable_State_999"], test_data_df["NN_Predicted_Reliable_State_999"])
        # calibnn_f1_reliability_state_recall_99 = recall_score(test_data_df["Reliable_State_99"], test_data_df["CalibNN_F1_Predicted_Reliable_State_99"])
        # calibnn_f1_reliability_state_recall_999 = recall_score(test_data_df["Reliable_State_999"], test_data_df["CalibNN_F1_Predicted_Reliable_State_999"])
        # calibnn_specificity_reliability_state_recall_99 = recall_score(test_data_df["Reliable_State_99"], test_data_df["CalibNN_Specificity_Predicted_Reliable_State_99"])
        # calibnn_specificity_reliability_state_recall_999 = recall_score(test_data_df["Reliable_State_999"], test_data_df["CalibNN_Specificity_Predicted_Reliable_State_999"])
        calibnn_recall_reliability_state_recall_90 = recall_score(test_data_df["Reliable_State_90"], test_data_df["CalibNN_Recall_Predicted_Reliable_State_90"])
        calibnn_recall_reliability_state_recall_99 = recall_score(test_data_df["Reliable_State_99"], test_data_df["CalibNN_Recall_Predicted_Reliable_State_99"])
        calibnn_recall_reliability_state_recall_999 = recall_score(test_data_df["Reliable_State_999"], test_data_df["CalibNN_Recall_Predicted_Reliable_State_999"])
        # Get number of TP, FP, TN, FN
        # bn_reliability_state_99_CM = confusion_matrix(test_data_df["Reliable_State_99"], test_data_df["BN_Predicted_Reliable_State_99"])
        # bn_reliability_state_TN_99, bn_reliability_state_FP_99, bn_reliability_state_FN_99, bn_reliability_state_TP_99 = bn_reliability_state_99_CM.ravel()
        # bn_reliability_state_999_CM = confusion_matrix(test_data_df["Reliable_State_999"], test_data_df["BN_Predicted_Reliable_State_999"])
        # bn_reliability_state_TN_999, bn_reliability_state_FP_999, bn_reliability_state_FN_999, bn_reliability_state_TP_999 = bn_reliability_state_999_CM.ravel()
        # nn_reliability_state_99_CM = confusion_matrix(test_data_df["Reliable_State_99"], test_data_df["NN_Predicted_Reliable_State_99"])
        # nn_reliability_state_TN_99, nn_reliability_state_FP_99, nn_reliability_state_FN_99, nn_reliability_state_TP_99 = nn_reliability_state_99_CM.ravel()
        # nn_reliability_state_999_CM = confusion_matrix(test_data_df["Reliable_State_999"], test_data_df["NN_Predicted_Reliable_State_999"])
        # nn_reliability_state_TN_999, nn_reliability_state_FP_999, nn_reliability_state_FN_999, nn_reliability_state_TP_999 = nn_reliability_state_999_CM.ravel()
        # calibnn_f1_reliability_state_99_CM = confusion_matrix(test_data_df["Reliable_State_99"], test_data_df["CalibNN_F1_Predicted_Reliable_State_99"])
        # calibnn_f1_reliability_state_TN_99, calibnn_f1_reliability_state_FP_99, calibnn_f1_reliability_state_FN_99, calibnn_f1_reliability_state_TP_99 = calibnn_f1_reliability_state_99_CM.ravel()
        # calibnn_f1_reliability_state_999_CM = confusion_matrix(test_data_df["Reliable_State_999"], test_data_df["CalibNN_F1_Predicted_Reliable_State_999"])
        # calibnn_f1_reliability_state_TN_999, calibnn_f1_reliability_state_FP_999, calibnn_f1_reliability_state_FN_999, calibnn_f1_reliability_state_TP_999 = calibnn_f1_reliability_state_999_CM.ravel()
        # calibnn_specificity_reliability_state_99_CM = confusion_matrix(test_data_df["Reliable_State_99"], test_data_df["CalibNN_Specificity_Predicted_Reliable_State_99"])
        # calibnn_specificity_reliability_state_TN_99, calibnn_specificity_reliability_state_FP_99, calibnn_specificity_reliability_state_FN_99, calibnn_specificity_reliability_state_TP_99 = calibnn_specificity_reliability_state_99_CM.ravel()
        # calibnn_specificity_reliability_state_999_CM = confusion_matrix(test_data_df["Reliable_State_999"], test_data_df["CalibNN_Specificity_Predicted_Reliable_State_999"])
        # calibnn_specificity_reliability_state_TN_999, calibnn_specificity_reliability_state_FP_999, calibnn_specificity_reliability_state_FN_999, calibnn_specificity_reliability_state_TP_999 = calibnn_specificity_reliability_state_999_CM.ravel()
        # calibnn_recall_reliability_state_99_CM = confusion_matrix(test_data_df["Reliable_State_99"], test_data_df["CalibNN_Recall_Predicted_Reliable_State_99"])
        # calibnn_recall_reliability_state_TN_99, calibnn_recall_reliability_state_FP_99, calibnn_recall_reliability_state_FN_99, calibnn_recall_reliability_state_TP_99 = calibnn_recall_reliability_state_99_CM.ravel()
        # calibnn_recall_reliability_state_999_CM = confusion_matrix(test_data_df["Reliable_State_999"], test_data_df["CalibNN_Recall_Predicted_Reliable_State_999"])
        # calibnn_recall_reliability_state_TN_999, calibnn_recall_reliability_state_FP_999, calibnn_recall_reliability_state_FN_999, calibnn_recall_reliability_state_TP_999 = calibnn_recall_reliability_state_999_CM.ravel()
        # Reliabiliy State Max Err at False Positive
        # bn_predict_reliabiliy_95_df = test_data_df.loc[(test_data_df["BN_Predicted_Reliability"]>=0.95)]
        # bn_predict_reliabiliy_99_df = test_data_df.loc[(test_data_df["BN_Predicted_Reliability"]>=0.99)]
        # bn_predict_reliabiliy_999_df = test_data_df.loc[(test_data_df["BN_Predicted_Reliability"]>=0.999)]
        # nn_predict_reliabiliy_95_df = test_data_df.loc[(test_data_df["NN_Predicted_Reliability"]>=0.95)]
        # nn_predict_reliabiliy_99_df = test_data_df.loc[(test_data_df["NN_Predicted_Reliability"]>=0.99)]
        # nn_predict_reliabiliy_999_df = test_data_df.loc[(test_data_df["NN_Predicted_Reliability"]>=0.999)]
        # calibnn_predict_reliabiliy_95_df = test_data_df.loc[(test_data_df["CalibNN_Predicted_Reliability_95"]>=0.95)]
        # calibnn_predict_reliabiliy_99_df = test_data_df.loc[(test_data_df["CalibNN_Predicted_Reliability_99"]>=0.99)]
        # calibnn_predict_reliabiliy_999_df = test_data_df.loc[(test_data_df["CalibNN_Predicted_Reliability_999"]>=0.999)]
        # bn_maxae_reliable_95 = np.max(np.abs(bn_predict_reliabiliy_95_df['Reliability'].values - bn_predict_reliabiliy_95_df['BN_Predicted_Reliability'].values))
        # bn_maxae_reliable_99 = np.max(np.abs(bn_predict_reliabiliy_99_df['Reliability'].values - bn_predict_reliabiliy_99_df['BN_Predicted_Reliability'].values))
        # bn_maxae_reliable_999 = np.max(np.abs(bn_predict_reliabiliy_999_df['Reliability'].values - bn_predict_reliabiliy_999_df['BN_Predicted_Reliability'].values))
        # nn_maxae_reliable_95 = np.max(np.abs(nn_predict_reliabiliy_95_df['Reliability'].values - nn_predict_reliabiliy_95_df['NN_Predicted_Reliability'].values))
        # nn_maxae_reliable_99 = np.max(np.abs(nn_predict_reliabiliy_99_df['Reliability'].values - nn_predict_reliabiliy_99_df['NN_Predicted_Reliability'].values))
        # nn_maxae_reliable_999 = np.max(np.abs(nn_predict_reliabiliy_999_df['Reliability'].values - nn_predict_reliabiliy_999_df['NN_Predicted_Reliability'].values))
        # calibnn_maxae_reliable_95 = np.max(np.abs(calibnn_predict_reliabiliy_95_df['Reliability'].values - calibnn_predict_reliabiliy_95_df['CalibNN_Predicted_Reliability_95'].values))
        # calibnn_maxae_reliable_99 = np.max(np.abs(calibnn_predict_reliabiliy_99_df['Reliability'].values - calibnn_predict_reliabiliy_99_df['CalibNN_Predicted_Reliability_99'].values))
        # calibnn_maxae_reliable_999 = np.max(np.abs(calibnn_predict_reliabiliy_999_df['Reliability'].values - calibnn_predict_reliabiliy_999_df['CalibNN_Predicted_Reliability_999'].values))

        ''' Record metrics to csv '''
        metrics_list.append({"Test_Dataset": os.path.join(test_dataset_path.split("/")[-3], test_dataset_path.split("/")[-2], test_dataset_path.split("/")[-1]), 
                             "NN_Model": NN_MODELS[i], "Cal_T_Specificity_90": T_SPECIFICITY_90, "Cal_T_Specificity_99": T_SPECIFICITY_99, "Cal_T_Specificity_999": T_SPECIFICITY_999, "Cal_T_Sensitivity_90": T_RECALL_90, "Cal_T_Sensitivity_99": T_RECALL_99, "Cal_T_Sensitivity_999": T_RECALL_999,
                                "bn_reliability_class_accuracy": bn_reliability_class_accuracy, "nn_reliability_class_accuracy": nn_reliability_class_accuracy, "calibnn_reliability_class_accuracy": calib_nn_reliability_class_accuracy,
                                "bn_reliability_class_specificity": bn_reliability_class_specificity, "nn_reliability_class_specificity": nn_reliability_class_specificity, "calibnn_reliability_class_specificity": calib_nn_reliability_class_specificity,
                                "bn_reliability_class_sensitivity": bn_reliability_class_sensitivity, "nn_reliability_class_sensitivity": nn_reliability_class_sensitivity, "calibnn_reliability_class_sensitivity": calib_nn_reliability_class_sensitivity,
                                "bn_failure_mode_accuracy": bn_failure_mode_accuracy, "nn_failure_mode_accuracy": nn_failure_mode_accuracy, "calibnn_failure_mode_accuracy": calib_nn_failure_mode_accuracy,
                                "bn_failure_mode_specificity": bn_failure_mode_specificity, "nn_failure_mode_specificity": nn_failure_mode_specificity, "calibnn_failure_mode_specificity": calib_nn_failure_mode_specificity,
                                "bn_failure_mode_sensitivity": bn_failure_mode_sensitivity, "nn_failure_mode_sensitivity": nn_failure_mode_sensitivity, "calibnn_failure_mode_sensitivity": calib_nn_failure_mode_sensitivity,
                                "bn_reliability_state_accuracy_90": bn_reliability_state_accuracy_90, "bn_reliability_state_accuracy_99": bn_reliability_state_accuracy_99, "bn_reliability_state_accuracy_999": bn_reliability_state_accuracy_999,
                                "nn_reliability_state_accuracy_90": nn_reliability_state_accuracy_90, "nn_reliability_state_accuracy_99": nn_reliability_state_accuracy_99, "nn_reliability_state_accuracy_999": nn_reliability_state_accuracy_999,
                                "calibnn_reliability_state_accuracy_90": calibnn_reliability_state_accuracy_90, "calibnn_reliability_state_accuracy_99": calibnn_reliability_state_accuracy_99, "calibnn_reliability_state_accuracy_999": calibnn_reliability_state_accuracy_999,
                                # "bn_reliability_state_f1_99": bn_reliability_state_f1_99, "bn_reliability_state_f1_999": bn_reliability_state_f1_999,
                                # "nn_reliability_state_f1_99": nn_reliability_state_f1_99, "nn_reliability_state_f1_999": nn_reliability_state_f1_999,
                                # "calibnn_f1_reliability_state_f1_99": calibnn_f1_reliability_state_f1_99, "calibnn_f1_reliability_state_f1_999": calibnn_f1_reliability_state_f1_999,
                                # "calibnn_specificity_reliability_state_f1_99": calibnn_specificity_reliability_state_f1_99, "calibnn_specificity_reliability_state_f1_999": calibnn_specificity_reliability_state_f1_999,
                                # "calibnn_recall_reliability_state_f1_99": calibnn_recall_reliability_state_f1_99, "calibnn_recall_reliability_state_f1_999": calibnn_recall_reliability_state_f1_999,
                                "bn_reliability_state_specificity_90": bn_reliability_state_specificity_90, "bn_reliability_state_specificity_99": bn_reliability_state_specificity_99, "bn_reliability_state_specificity_999": bn_reliability_state_specificity_999,
                                "nn_reliability_state_specificity_90": nn_reliability_state_specificity_90, "nn_reliability_state_specificity_99": nn_reliability_state_specificity_99, "nn_reliability_state_specificity_999": nn_reliability_state_specificity_999,
                                # "calibnn_f1_reliability_state_specificity_99": calibnn_f1_reliability_state_specificity_99, "calibnn_f1_reliability_state_specificity_999": calibnn_f1_reliability_state_specificity_999,
                                "calibnn_specificity_reliability_state_specificity_90": calibnn_specificity_reliability_state_specificity_90, "calibnn_specificity_reliability_state_specificity_99": calibnn_specificity_reliability_state_specificity_99, "calibnn_specificity_reliability_state_specificity_999": calibnn_specificity_reliability_state_specificity_999,
                                # "calibnn_recall_reliability_state_specificity_99": calibnn_recall_reliability_state_specificity_99, "calibnn_recall_reliability_state_specificity_999": calibnn_recall_reliability_state_specificity_999,
                                "bn_reliability_state_recall_90": bn_reliability_state_recall_90, "bn_reliability_state_recall_99": bn_reliability_state_recall_99, "bn_reliability_state_recall_999": bn_reliability_state_recall_999,
                                "nn_reliability_state_recall_90": nn_reliability_state_recall_90, "nn_reliability_state_recall_99": nn_reliability_state_recall_99, "nn_reliability_state_recall_999": nn_reliability_state_recall_999,
                                # "calibnn_f1_reliability_state_recall_99": calibnn_f1_reliability_state_recall_99, "calibnn_f1_reliability_state_recall_999": calibnn_f1_reliability_state_recall_999,
                                # "calibnn_specificity_reliability_state_recall_99": calibnn_specificity_reliability_state_recall_99, "calibnn_specificity_reliability_state_recall_999": calibnn_specificity_reliability_state_recall_999,
                                "calibnn_recall_reliability_state_recall_90": calibnn_recall_reliability_state_recall_90, "calibnn_recall_reliability_state_recall_99": calibnn_recall_reliability_state_recall_99, "calibnn_recall_reliability_state_recall_999": calibnn_recall_reliability_state_recall_999})
        # confusion_metrics_list.append({"Test_Dataset": os.path.join(test_dataset_path.split("/")[-3], test_dataset_path.split("/")[-2], test_dataset_path.split("/")[-1]),
        #                                "bn_reliability_state_TN_99": bn_reliability_state_TN_99, "bn_reliability_state_TP_99": bn_reliability_state_TP_99, "bn_reliability_state_FN_99": bn_reliability_state_FN_99, "bn_reliability_state_FP_99": bn_reliability_state_FP_99,
        #                                "bn_reliability_state_TN_999": bn_reliability_state_TN_999, "bn_reliability_state_TP_999": bn_reliability_state_TP_999, "bn_reliability_state_FN_999": bn_reliability_state_FN_999, "bn_reliability_state_FP_999": bn_reliability_state_FP_999,
        #                                "nn_reliability_state_TN_99": nn_reliability_state_TN_99, "nn_reliability_state_TP_99": nn_reliability_state_TP_99, "nn_reliability_state_FN_99": nn_reliability_state_FN_99, "nn_reliability_state_FP_99": nn_reliability_state_FP_99,
        #                                "nn_reliability_state_TN_999": nn_reliability_state_TN_999, "nn_reliability_state_TP_999": nn_reliability_state_TP_999, "nn_reliability_state_FN_999": nn_reliability_state_FN_999, "nn_reliability_state_FP_999": nn_reliability_state_FP_999,
        #                                "calibnn_f1_reliability_state_TN_99": calibnn_f1_reliability_state_TN_99, "calibnn_f1_reliability_state_TP_99": calibnn_f1_reliability_state_TP_99, "calibnn_f1_reliability_state_FN_99": calibnn_f1_reliability_state_FN_99, "calibnn_f1_reliability_state_FP_99": calibnn_f1_reliability_state_FP_99,
        #                                "calibnn_f1_reliability_state_TN_999": calibnn_f1_reliability_state_TN_999, "calibnn_f1_reliability_state_TP_999": calibnn_f1_reliability_state_TP_999, "calibnn_f1_reliability_state_FN_999": calibnn_f1_reliability_state_FN_999, "calibnn_f1_reliability_state_FP_999": calibnn_f1_reliability_state_FP_999,
        #                                "calibnn_specificity_reliability_state_TN_99": calibnn_specificity_reliability_state_TN_99, "calibnn_specificity_reliability_state_TP_99": calibnn_specificity_reliability_state_TP_99, "calibnn_specificity_reliability_state_FN_99": calibnn_specificity_reliability_state_FN_99, "calibnn_specificity_reliability_state_FP_99": calibnn_specificity_reliability_state_FP_99,
        #                                "calibnn_specificity_reliability_state_TN_999": calibnn_specificity_reliability_state_TN_999, "calibnn_specificity_reliability_state_TP_999": calibnn_specificity_reliability_state_TP_999, "calibnn_specificity_reliability_state_FN_999": calibnn_specificity_reliability_state_FN_999, "calibnn_specificity_reliability_state_FP_999": calibnn_specificity_reliability_state_FP_999,
        #                                "calibnn_recall_reliability_state_TN_99": calibnn_recall_reliability_state_TN_99, "calibnn_recall_reliability_state_TP_99": calibnn_recall_reliability_state_TP_99, "calibnn_recall_reliability_state_FN_99": calibnn_recall_reliability_state_FN_99, "calibnn_recall_reliability_state_FP_99": calibnn_recall_reliability_state_FP_99,
        #                                "calibnn_recall_reliability_state_TN_999": calibnn_recall_reliability_state_TN_999, "calibnn_recall_reliability_state_TP_999": calibnn_recall_reliability_state_TP_999, "calibnn_recall_reliability_state_FN_999": calibnn_recall_reliability_state_FN_999, "calibnn_recall_reliability_state_FP_999": calibnn_recall_reliability_state_FP_999})
        # metrics_list.append({"Test_Dataset": os.path.join(test_dataset_path.split("/")[-3], test_dataset_path.split("/")[-2], test_dataset_path.split("/")[-1]), 
        #                      "bn_reliability_class_accuracy": bn_reliability_class_accuracy, "nn_reliability_class_accuracy": nn_reliability_class_accuracy, "calibnn_reliability_class_accuracy": calib_nn_reliability_class_accuracy,
        #                      "bn_failure_mode_accuracy": bn_failure_mode_accuracy, "nn_failure_mode_accuracy": nn_failure_mode_accuracy, "calibnn_failure_mode_accuracy": calib_nn_failure_mode_accuracy,
        #                      "bn_reliability_state_accuracy_95": bn_reliability_state_accuracy_95, "bn_reliability_state_accuracy_99": bn_reliability_state_accuracy_99, "bn_reliability_state_accuracy_999": bn_reliability_state_accuracy_999,
        #                      "nn_reliability_state_accuracy_95": nn_reliability_state_accuracy_95, "nn_reliability_state_accuracy_99": nn_reliability_state_accuracy_99, "nn_reliability_state_accuracy_999": nn_reliability_state_accuracy_999,
        #                      "calibnn_reliability_state_accuracy_95": calibnn_reliability_state_accuracy_95, "calibnn_reliability_state_accuracy_99": calibnn_reliability_state_accuracy_99, "calibnn_reliability_state_accuracy_999": calibnn_reliability_state_accuracy_999,
        #                      "bn_reliability_state_f1_95": bn_reliability_state_f1_95, "bn_reliability_state_f1_99": bn_reliability_state_f1_99, "bn_reliability_state_f1_999": bn_reliability_state_f1_999,
        #                      "nn_reliability_state_f1_95": nn_reliability_state_f1_95, "nn_reliability_state_f1_99": nn_reliability_state_f1_99, "nn_reliability_state_f1_999": nn_reliability_state_f1_999,
        #                      "calibnn_reliability_state_f1_95": calibnn_reliability_state_f1_95, "calibnn_f1_reliability_state_f1_99": calibnn_f1_reliability_state_f1_99, "calibnn_f1_reliability_state_f1_999": calibnn_f1_reliability_state_f1_999,
        #                      "bn_reliability_state_fbeta_95": bn_reliability_state_fbeta_95, "bn_reliability_state_fbeta_99": bn_reliability_state_fbeta_99, "bn_reliability_state_fbeta_999": bn_reliability_state_fbeta_999,
        #                      "nn_reliability_state_fbeta_95": nn_reliability_state_fbeta_95, "nn_reliability_state_fbeta_99": nn_reliability_state_fbeta_99, "nn_reliability_state_fbeta_999": nn_reliability_state_fbeta_999,
        #                      "calibnn_reliability_state_fbeta_95": calibnn_reliability_state_fbeta_95, "calibnn_reliability_state_fbeta_99": calibnn_reliability_state_fbeta_99, "calibnn_reliability_state_fbeta_999": calibnn_reliability_state_fbeta_999,
        #                      "bn_reliability_state_precision_95": bn_reliability_state_precision_95, "bn_reliability_state_specificity_99": bn_reliability_state_specificity_99, "bn_reliability_state_specificity_999": bn_reliability_state_specificity_999,
        #                      "nn_reliability_state_precision_95": nn_reliability_state_precision_95, "nn_reliability_state_specificity_99": nn_reliability_state_specificity_99, "nn_reliability_state_specificity_999": nn_reliability_state_specificity_999,
        #                      "calibnn_reliability_state_precision_95": calibnn_reliability_state_precision_95, "calibnn_f1_reliability_state_specificity_99": calibnn_f1_reliability_state_specificity_99, "calibnn_f1_reliability_state_specificity_999": calibnn_f1_reliability_state_specificity_999,
        #                      "bn_reliability_state_recall_95": bn_reliability_state_recall_95, "bn_reliability_state_recall_99": bn_reliability_state_recall_99, "bn_reliability_state_recall_999": bn_reliability_state_recall_999,
        #                      "nn_reliability_state_recall_95": nn_reliability_state_recall_95, "nn_reliability_state_recall_99": nn_reliability_state_recall_99, "nn_reliability_state_recall_999": nn_reliability_state_recall_999,
        #                      "calibnn_reliability_state_recall_95": calibnn_reliability_state_recall_95, "calibnn_f1_reliability_state_recall_99": calibnn_f1_reliability_state_recall_99, "calibnn_f1_reliability_state_recall_999": calibnn_f1_reliability_state_recall_999,
        #                      "bn_maxae_reliable_95": bn_maxae_reliable_95, "bn_maxae_reliable_99": bn_maxae_reliable_99, "bn_maxae_reliable_999": bn_maxae_reliable_999,
        #                      "nn_maxae_reliable_95": nn_maxae_reliable_95, "nn_maxae_reliable_99": nn_maxae_reliable_99, "nn_maxae_reliable_999": nn_maxae_reliable_999,
        #                      "calibnn_maxae_reliable_95": calibnn_maxae_reliable_95, "calibnn_maxae_reliable_99": calibnn_maxae_reliable_99, "calibnn_maxae_reliable_999": calibnn_maxae_reliable_999})
            
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(METRIC_SAVE_PATH)
    confusion_metrics_df = pd.DataFrame(confusion_metrics_list)
    confusion_metrics_df.to_csv(CONFUSION_METRIC_SAVE_PATH)