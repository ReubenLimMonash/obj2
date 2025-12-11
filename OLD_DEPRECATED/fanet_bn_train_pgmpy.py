'''
Date: 13/05/2024
Desc: To train BN model for communication reliability, using Mean and Std Dev as inputs.
      Using PGMPY library.
'''

import pandas as pd
import numpy as np 
import math
import os
from tqdm import tqdm
from joblib import dump
from multiprocessing.pool import Pool
from itertools import repeat
from scipy import special

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

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

def generate_reliability_dataset(dataset_details_df, test_split=0.2):
    # df_dtypes = {"Horizontal_Distance": np.float64, "Height": np.int16,	"U2G_Distance": np.int32, "UAV_Sending_Interval": np.float64, "Mean_SINR": np.float64, "Std_Dev_SINR": np.float64,
    #              "Modulation": 'string', "Num_Sent": np.int32, "Num_Reliable": np.int32, "Num_Delay_Excd": np.int32, "Num_Incr_Rcvd": np.int32, "Num_Q_Overflow": np.int32}
    # dataset_details = pd.read_csv(dataset_details_csv, 
    #                               usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Num_Sent", "Num_Reliable", "Num_Delay_Excd",
    #                                          "Num_Incr_Rcvd", "Num_Q_Overflow"],
    #                               dtype=df_dtypes)
    df_train_list = []
    for row in tqdm(dataset_details_df.itertuples()):
        mean_sinr = row.Mean_SINR_Class
        std_dev_sinr = row.Std_Dev_SINR_Class
        uav_send_int = row.UAV_Sending_Interval_Class
        mcs = row.MCS
        num_reliable = row.Num_Reliable
        num_delay_excd = row.Num_Delay_Excd
        num_incr_rcvd = row.Num_Incr_Rcvd
        num_q_overflow = row.Num_Q_Overflow

        if num_reliable > 0:
            reliable_packets = pd.DataFrame({"Mean_SINR_Class": mean_sinr, "Std_Dev_SINR_Class": std_dev_sinr, "UAV_Sending_Interval_Class": uav_send_int, "MCS": mcs, "Packet_State": 0}, index=[0])
            reliable_packets = reliable_packets.loc[reliable_packets.index.repeat(num_reliable)]
        else:
            reliable_packets = pd.DataFrame({})

        if num_delay_excd > 0:
            delay_excd_packets = pd.DataFrame({"Mean_SINR_Class": mean_sinr, "Std_Dev_SINR_Class": std_dev_sinr, "UAV_Sending_Interval_Class": uav_send_int, "MCS": mcs, "Packet_State": 1}, index=[0])
            delay_excd_packets = delay_excd_packets.loc[delay_excd_packets.index.repeat(num_delay_excd)]
        else:
            delay_excd_packets = pd.DataFrame({})

        if num_q_overflow > 0:
            q_overflow_packets = pd.DataFrame({"Mean_SINR_Class": mean_sinr, "Std_Dev_SINR_Class": std_dev_sinr, "UAV_Sending_Interval_Class": uav_send_int, "MCS": mcs, "Packet_State": 2}, index=[0])
            q_overflow_packets = q_overflow_packets.loc[q_overflow_packets.index.repeat(num_q_overflow)]
        else:
            q_overflow_packets = pd.DataFrame({})

        if num_incr_rcvd > 0:
            incr_rcvd_packets = pd.DataFrame({"Mean_SINR_Class": mean_sinr, "Std_Dev_SINR_Class": std_dev_sinr, "UAV_Sending_Interval_Class": uav_send_int, "MCS": mcs, "Packet_State": 3}, index=[0])
            incr_rcvd_packets = incr_rcvd_packets.loc[incr_rcvd_packets.index.repeat(num_incr_rcvd)]
        else:
            incr_rcvd_packets = pd.DataFrame({})
        df_train_list.append(pd.concat([reliable_packets, delay_excd_packets, q_overflow_packets, incr_rcvd_packets]))

    df_train = pd.concat(df_train_list)
    return df_train

def get_mcs_index(df_in):
    '''
    Gets the MCS index based on modulation and bitrate column of the df_in
    '''
    df = df_in.copy()
    df["MCS"] = ''
    df.loc[(df["Modulation"] == "BPSK") & (df["Bitrate"] == 6.5), "MCS"] = 0 # MCS Index 0
    df.loc[(df["Modulation"] == "QPSK") & (df["Bitrate"] == 13), "MCS"] = 1 # MCS Index 0
    df.loc[(df["Modulation"] == "QPSK") & (df["Bitrate"] == 19.5), "MCS"] = 2 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM16") & (df["Bitrate"] == 26), "MCS"] = 3 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM16") & (df["Bitrate"] == 39), "MCS"] = 4 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 52), "MCS"] = 5 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 58.5), "MCS"] = 6 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 65), "MCS"] = 7 # MCS Index 0

    return df

def process_n_generate_data(dataset_path, num_bins=100):

    df_dtypes = {"Horizontal_Distance": np.float64, "Height": np.int16,	"U2G_Distance": np.int32, "UAV_Sending_Interval": np.float64, "Mean_SINR": np.float64, "Std_Dev_SINR": np.float64,
                "Num_Sent": np.int32, "Num_Reliable": np.int32, "Num_Delay_Excd": np.int32, "Num_Incr_Rcvd": np.int32, "Num_Q_Overflow": np.int32, "Modulation": str, "Bitrate": np.float64}
    dataset_details_df = pd.read_csv(dataset_path, 
                                usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Bitrate", "Num_Sent", "Num_Reliable", "Num_Delay_Excd",
                                            "Num_Incr_Rcvd", "Num_Q_Overflow"],
                                dtype=df_dtypes)
    dataset_details_df = get_mcs_index(dataset_details_df)

    # Change sending interval categorial to numeric
    dataset_details_df["UAV_Sending_Interval_Class"] = dataset_details_df["UAV_Sending_Interval"].replace({10:0, 20:1, 66.7:2, 100:3})

    # Quantize mean and std dev of sinr
    _, mean_sinr_bins = pd.qcut(dataset_details_df.Mean_SINR, q=num_bins, retbins=True)
    mean_sinr_bins = np.concatenate(([-np.inf], mean_sinr_bins[1:-1], [np.inf]))
    _, std_dev_sinr_bins = pd.qcut(dataset_details_df.Std_Dev_SINR, q=num_bins, retbins=True)
    std_dev_sinr_bins = np.concatenate(([-np.inf], std_dev_sinr_bins[1:-1], [np.inf]))

    dataset_details_df["Mean_SINR_Class"] = pd.cut(dataset_details_df.Mean_SINR, mean_sinr_bins, right=True, include_lowest=False, labels=False)
    dataset_details_df["Std_Dev_SINR_Class"] = pd.cut(dataset_details_df.Std_Dev_SINR, std_dev_sinr_bins, right=True, include_lowest=False, labels=False)


    # # Generate dataset samples
    df_train = generate_reliability_dataset(dataset_details_df)

    # X = df_train[["Mean_SINR_Class", "Std_Dev_SINR_Class", "UAV_Sending_Interval_Class", "MCS"]].values
    # Y = df_train['Packet_State'].values

    return df_train, mean_sinr_bins, std_dev_sinr_bins # X, Y

if __name__ == '__main__':

    DATASET_PATHS = ["/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJISpark/data_processed/DJI_Spark_Downlink_Reliability.csv",
                    "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJISpark/data_processed/DJI_Spark_Uplink_Reliability.csv",
                    "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJISpark/data_processed/DJI_Spark_Video_Reliability.csv",
                    "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJIMavicAir/data_processed/DJI_MavicAir_Downlink_Reliability.csv",
                    "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJIMavicAir/data_processed/DJI_MavicAir_Uplink_Reliability.csv",
                    "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_DJIMavicAir/data_processed/DJI_MavicAir_Video_Reliability.csv",
                    "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_ParrotAR2/data_processed/ParrotAR2_Downlink_Reliability.csv",
                    "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_ParrotAR2/data_processed/ParrotAR2_Uplink_Reliability.csv",
                    "/media/research-student/One Touch/FANET_Dataset/Dataset_NP10000_ParrotAR2/data_processed/ParrotAR2_Video_Reliability.csv"]
    MODEL_NAMES = ["DJISpark_Downlink_Reliability_{}", "DJISpark_Uplink_Reliability_{}", "DJISpark_Video_Reliability_{}", 
                  "DJIMavicAir_Downlink_Reliability_{}", "DJIMavicAir_Uplink_Reliability_{}", "DJIMavicAir_Video_Reliability_{}",
                  "ParrotAR2_Downlink_Reliability_{}", "ParrotAR2_Uplink_Reliability_{}", "ParrotAR2_Video_Reliability_{}"]
    SAVE_PATH = "/home/research-student/omnet-fanet/bn_pgmpy"

    for dataset_path, model_name in zip(DATASET_PATHS, MODEL_NAMES):
        print(model_name)

        # Load Dataset and quantize mean and std dev of SINR
        print("========================== Loading Dataset ==========================")
        df_train, mean_sinr_bins, std_dev_sinr_bins = process_n_generate_data(dataset_path, num_bins=100)

        # Train BN
        print("========================== Training Model ==========================")
        model = BayesianNetwork([('Mean_SINR_Class', 'Packet_State'), ('Std_Dev_SINR_Class', 'Packet_State'), ('UAV_Sending_Interval_Class', 'Packet_State'), 
                                ('MCS', 'Packet_State'), ('Mean_SINR_Class', 'Std_Dev_SINR_Class')])
        model.fit(df_train)

        # Save model and bins
        print("========================== Saving Model ==========================")
        dump(model, os.path.join(SAVE_PATH, model_name.format("BN_Model.joblib")))
        dump(mean_sinr_bins, os.path.join(SAVE_PATH, model_name.format("Mean_Sinr_Bins.joblib")))
        dump(std_dev_sinr_bins, os.path.join(SAVE_PATH, model_name.format("Std_Dev_Sinr_Bins.joblib")))