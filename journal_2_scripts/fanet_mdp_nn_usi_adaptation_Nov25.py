"""
Date: 18/11/2025
Desc: Reliability-prediction-based USI adaptation using calibrated NN models.
      To write XML scenario scripts for test cases
Modified: Whenever adaptation is required at the next hdist, instead of predicting with each USI one by one, 
          we predict for all USI and choose the one that gives the highest mean reliability that meets the reliability requirement.
          The outcome of predictions for all USI are also recorded for analysis, with the chosen USI indicated.
          11/12/2025: Modified to consider all eligible USI (based on model predictions) at initialization for each test case, instead of just picking the highest USI that meets the reliability requirement.
"""

import tensorflow as tf
from keras import activations
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input
import pandas as pd # for data manipulation 
import numpy as np
import math, os
from scipy import special
from multiprocessing.pool import Pool
from itertools import repeat, product

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

def norm_MCS(mcs_index):
    return 2*mcs_index/7 - 1

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

def load_model(model_path):
    # Load the model without activation function, but with the trained weights
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', 
                loss={'packet_state': 'categorical_crossentropy'},
                metrics={'packet_state': 'accuracy'})
    return model

def mdp_nn(dl_model, ul_model, vid_model, height, mcs, reliability_th=0.99, init_hdist=0, dmax=500, hdist_step_size=5, save_file=None):
    '''
    Proposed communication reliability maintenance scheme
    Inputs:
    dl_model, ul_model, vid_model: The NN prediction models for DL, UL and VID links, respectively
    height: The height of the test case
    mcs: The MCS of the test case
    reliability_th: The required reliability level
    init_hdist: The starting horizontal distance
    dmax: The maximum horizontal distance (for recording purposes only)
    hdist_step_size: The step size for horizontal distance increment
    save_file: File to save results
    '''

    max_mean_sinr = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
    max_std_dev_sinr = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
    min_mean_sinr = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
    min_std_dev_sinr = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)

    usi_list = [10, 20, 66.7, 100] # Possible UAV sending intervals. List lowest first so that np.argmax will select the lowest first if there is a tie in mean reliability
    uav_send_int_norm = {10:-1, 20:-0.5, 40:0, 66.7: 0.5, 100:1, 1000:2}

    # Get initial USI that fulfills reliability requirement at starting point
    m, s = sinr_lognormal_approx(init_hdist, height)
    mean_sinr = 2*(10*math.log10(m)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1
    std_dev_sinr = 2*(10*math.log10(s)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1
    inputs = np.hstack((np.array([mean_sinr]*len(usi_list)).reshape(-1,1), np.array([std_dev_sinr]*len(usi_list)).reshape(-1,1), 
                        np.array([uav_send_int_norm[usi] for usi in usi_list]).reshape(-1,1), np.array([norm_MCS(mcs)]*len(usi_list)).reshape(-1,1)))
    dl_probs = dl_model.predict(inputs, verbose=0)
    init_usi_dl_rel = np.array([probs[0] for probs in dl_probs])
    ul_probs = ul_model.predict(inputs, verbose=0)
    init_usi_ul_rel = np.array([probs[0] for probs in ul_probs])
    vid_probs = vid_model.predict(inputs, verbose=0)
    init_usi_vid_rel = np.array([probs[0] for probs in vid_probs])
    init_usi_rel_check = np.array([1 if rel >= reliability_th else 0 for rel in init_usi_dl_rel]) & np.array([1 if rel >= reliability_th else 0 for rel in init_usi_ul_rel]) & np.array([1 if rel >= reliability_th else 0 for rel in init_usi_vid_rel]) 
    possible_indexes = np.where(init_usi_rel_check == 1)[0]
    if len(possible_indexes) == 0:
        print("No USI can fulfill reliability requirement at starting point. Abort MDP.")
        print(height, mcs)
        return
    possible_init_usi = [usi_list[i] for i in possible_indexes]

    for init_usi in possible_init_usi: # For each possible initial USI that fulfills reliability requirement, run the MDP
        state_record = []
        current_hdist = init_hdist
        curr_usi = init_usi
        state_record.append({"Horizontal_Distance": current_hdist, "Height": height, "UAV_Send_Interval": curr_usi, "MCS": mcs, 
                            "Status": "Chosen", # Status is to indicate whether the scheme is still running (Continue) or has ended (Dmax/Abort)
                            "DL_Reliability": init_usi_dl_rel[usi_list.index(curr_usi)], 
                            "UL_Reliability": init_usi_ul_rel[usi_list.index(curr_usi)], 
                            "Vid_Reliability": init_usi_vid_rel[usi_list.index(curr_usi)]})
        while True: 
            current_hdist += hdist_step_size # Move to next horizontal distance
            # Get the mean and std dev of SINR at this horizontal distance
            m, s = sinr_lognormal_approx(current_hdist, height)
            mean_sinr = 2*(10*math.log10(m)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1
            std_dev_sinr = 2*(10*math.log10(s)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1
            # Check reliability at current horizontal distance with current USI
            dl_probs = dl_model.predict([[mean_sinr, std_dev_sinr, uav_send_int_norm[curr_usi], norm_MCS(mcs)]], verbose=0)
            ul_probs = ul_model.predict([[mean_sinr, std_dev_sinr, uav_send_int_norm[curr_usi], norm_MCS(mcs)]], verbose=0)
            vid_probs = vid_model.predict([[mean_sinr, std_dev_sinr, uav_send_int_norm[curr_usi], norm_MCS(mcs)]], verbose=0)
            next_dl_rel = dl_probs[0][0]
            next_ul_rel = ul_probs[0][0]
            next_vid_rel = vid_probs[0][0]
            state_record.append({"Horizontal_Distance": current_hdist, "Height": height, "UAV_Send_Interval": curr_usi, "MCS": mcs, 
                            "Status": "Continue", # Status is to indicate whether the scheme is still running (Continue) or has ended (Dmax/Abort)
                            "DL_Reliability": next_dl_rel, "UL_Reliability": next_ul_rel, "Vid_Reliability": next_vid_rel})
            # Check if the next MCS is able to maintain the reliability of all links
            next_usi_rel_check = (next_dl_rel >= reliability_th) & (next_ul_rel >= reliability_th) & (next_vid_rel >= reliability_th)
            if not next_usi_rel_check:
                # Record that we need to adapt
                state_record[-1]["Status"] = "Adapt"
                # Predict reliabilities for all other USI (apart from current USI) at this horizontal distance
                usi_list_other = usi_list.copy()
                usi_list_other.remove(curr_usi)
                inputs = np.hstack((np.array([mean_sinr]*len(usi_list_other)).reshape(-1,1), np.array([std_dev_sinr]*len(usi_list_other)).reshape(-1,1), 
                                    np.array([uav_send_int_norm[usi] for usi in usi_list_other]).reshape(-1,1), np.array([norm_MCS(mcs)]*len(usi_list_other)).reshape(-1,1)))
                dl_probs = dl_model.predict(inputs, verbose=0)
                next_usi_dl_rel = np.array([probs[0] for probs in dl_probs])
                ul_probs = ul_model.predict(inputs, verbose=0)
                next_usi_ul_rel = np.array([probs[0] for probs in ul_probs])
                vid_probs = vid_model.predict(inputs, verbose=0)
                next_usi_vid_rel = np.array([probs[0] for probs in vid_probs])
                # Check if any USI is able to maintain the reliability of all links
                next_usi_rel_check = np.array([1 if rel >= reliability_th else 0 for rel in next_usi_dl_rel]) & np.array([1 if rel >= reliability_th else 0 for rel in next_usi_ul_rel]) & np.array([1 if rel >= reliability_th else 0 for rel in next_usi_vid_rel]) 
                if np.any(next_usi_rel_check):
                    # Get all indexes of USI that fulfill reliability requirement
                    reliable_usi_indexes = np.where(next_usi_rel_check == 1)[0]
                    # Choose the USI that gives the highest mean reliability
                    mean_reliabilities = [(next_usi_dl_rel[i] + next_usi_ul_rel[i] + next_usi_vid_rel[i])/3 for i in reliable_usi_indexes]
                    curr_usi_index = reliable_usi_indexes[np.argmax(mean_reliabilities)]
                    # Record all USI predictions
                    for i in range(len(next_usi_dl_rel)):
                        state_record.append({"Horizontal_Distance": current_hdist, "Height": height, "UAV_Send_Interval": usi_list_other[i], "MCS": mcs, 
                                    "Status": "Prediction", # Status is to indicate whether the scheme is still running (Continue) or has ended (Dmax/Abort)
                                    "DL_Reliability": next_usi_dl_rel[i], "UL_Reliability": next_usi_ul_rel[i], "Vid_Reliability": next_usi_vid_rel[i]})
                        if i == curr_usi_index:
                            state_record[-1]["Status"] = "Chosen"
                            curr_usi = usi_list_other[curr_usi_index]
                else:
                    # Record all prediction outcomes
                    for i in range(len(next_usi_dl_rel)):
                        state_record.append({"Horizontal_Distance": current_hdist, "Height": height, "UAV_Send_Interval": usi_list_other[i], "MCS": mcs, 
                                    "Status": "Prediction", # Status is to indicate whether the scheme is still running (Continue) or has ended (Dmax/Abort)
                                    "DL_Reliability": next_usi_dl_rel[i], "UL_Reliability": next_usi_ul_rel[i], "Vid_Reliability": next_usi_vid_rel[i]})
                    # Record the last state as Abort
                    state_record.append({"Horizontal_Distance": current_hdist, "Height": height, "UAV_Send_Interval": np.nan, "MCS": mcs, 
                                    "Status": "Abort", # Status is to indicate whether the scheme is still running (Continue) or has ended (Dmax/Abort)
                                    "DL_Reliability": np.nan, "UL_Reliability": np.nan, "Vid_Reliability": np.nan})
                    break
                
        mdp_actions = pd.DataFrame(state_record)
        mdp_actions["Dmax"] = None
        mdp_actions.loc[0, "Dmax"] = dmax

        if save_file is not None:
            mdp_actions.to_csv(save_file.format(height, mcs, init_usi), index=False)

    return 

def run_mdp(height, mcs, max_hdist_df, hdist_step_size, dl_model_path, ul_model_path, vid_model_path, reliability_th, save_path):
    # Load models
    dl_model = load_model(dl_model_path)
    ul_model = load_model(ul_model_path)
    vid_model = load_model(vid_model_path)
    # Run MDP
    print("Height: {}, MCS: {}".format(height, mcs))
    max_hdist = max_hdist_df.loc[(max_hdist_df["Height"]==height) & (max_hdist_df["MCS"]==mcs)]["D_max"].values[0]
    if max_hdist > 0:  # Exclude test cases with 0 max hdist 
        # hdist = list(np.append(np.linspace(0, max_hdist-max_hdist%hdist_step_size, int(max_hdist//hdist_step_size)+1, endpoint=True), np.array([max_hdist]))) # Evaluate for every hdist_step_size meters, includes last point (max_hdist)
        save_file = os.path.join(save_path, 'height-{}_mcs-{}_initUSI-{}_predictions.csv')
        mdp_nn(dl_model, ul_model, vid_model, height, mcs, reliability_th=reliability_th, init_hdist=0, dmax=max_hdist, hdist_step_size=hdist_step_size, save_file=save_file)
    print("Done - Height: {}, MCS: {}".format(height, mcs))
    return

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    ''' Set parameters '''
    DL_NN_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Downlink.round-1_split-9_0.2027.h5"
    UL_NN_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Uplink.round-1_split-9_0.1363.h5"
    VID_NN_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Video.round-1_split-9_0.2720.h5"
    RELIABILITY_TH = 0.99
    RELIABILITY_TH_STR = "99" # Remember to change this when changing RELIABILITY_TH
    MAX_HDIST_DF_PATH = "/home/research-student/omnet-fanet/data-processing-scripts/mdp_max_hdist_Oct25/MDP_USI_Max_HDist_fm_Sim_{}.csv".format(RELIABILITY_TH_STR)
    SAVE_PATH = "/home/research-student/omnet-fanet/data-processing-scripts/journal_2_scripts/predictions_usi_adaptation/predictions_5m_nn_{}".format(RELIABILITY_TH_STR)
    NUM_WORKERS = 32
    HDIST_STEP_SIZE = 5 # In m
    HEIGHTS = [75, 105, 135, 165, 195, 225, 255, 285]
    MCS_LIST = [0, 1, 2, 3, 4, 5, 6, 7] # MCS index list
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    """ Run MDP """
    max_hdist_df = pd.read_csv(MAX_HDIST_DF_PATH)
    var_params = [i for i in product(HEIGHTS, MCS_LIST)]
    with Pool(NUM_WORKERS) as pool:
        pool.starmap(run_mdp, zip(*zip(*var_params), repeat(max_hdist_df), repeat(HDIST_STEP_SIZE),
                    repeat(DL_NN_PATH), repeat(UL_NN_PATH), repeat(VID_NN_PATH), repeat(RELIABILITY_TH), repeat(SAVE_PATH)))