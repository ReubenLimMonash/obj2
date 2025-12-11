"""
Date: 30/10/2025
Desc: Reliability-prediction-based MCS+USI adaptation using calibrated NN models.
      Whenever adaptation is required at the next hdist, instead of predicting with each MCS then USI one by one, 
      we predict for all MCS and USI combinations and choose the one that gives the highest mean reliability that meets the reliability requirement.
      The outcome of predictions for all MCS and USI combos are also recorded for analysis, with the chosen combo indicated.
      11/12/2025: Modified to consider all eligible MCS+USI (based on model predictions) at initialization for each test case, instead of just picking the highest MCS and lowest USI that meets the reliability requirement.
"""

import tensorflow as tf
from keras import activations
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input
import pandas as pd
import numpy as np
import math, os
from scipy import special
from multiprocessing.pool import Pool
import itertools

def q_func(x):
    return 0.5 - 0.5*special.erf(x / np.sqrt(2))

def friis_calc(P,freq,dist,ple):
    propagation_speed = 299792458
    l = propagation_speed / freq
    h_pl = P * l**2 / (16*math.pi**2)
    P_Rx = h_pl * dist**(-ple)
    return P_Rx

def plos_calc(h_dist, height_tx, height_rx, env='suburban'):
    if env == 'suburban':
        a1,a2,a3 = 0.1, 7.5e-4, 8
    else:
        a1,a2,a3 = 0.3, 5e-4, 15
    delta_h = height_tx - height_rx
    pow_factor = h_dist * math.sqrt(a1*a2)
    if delta_h == 0:
        p = (1 - math.exp((-(height_tx)**2) / (2*a3**2))) ** pow_factor
    else:
        if delta_h < 0:
            h1 = height_rx; h2 = height_tx
        else:
            h1 = height_tx; h2 = height_rx
        p = (1 - (math.sqrt(2*math.pi)*a3 / abs(delta_h)) * abs(q_func(h1/a3) - q_func(h2/a3))) ** pow_factor
    return p

def sinr_lognormal_approx(h_dist, height, env='suburban'):
    P_Tx_dBm = 20
    P_Tx = 10**(P_Tx_dBm/10) / 1000
    freq = 2.4e9
    noise_dBm = -86
    noise = 10**(noise_dBm/10) / 1000
    if env == "suburban":
        n_min,n_max = 2,2.75
        K_dB_min,K_dB_max = 1.4922,12.2272
        K_min = 10**(K_dB_min/10); K_max = 10**(K_dB_max/10)
        alpha,beta = 11.1852, 0.06
    else:
        n_min,n_max = 1.9,2.7
        K_dB_min,K_dB_max = -5,15
        K_min = 10**(K_dB_min/10); K_max = 10**(K_dB_max/10)
        alpha,beta = 10.42, 0.05
    PLoS = plos_calc(h_dist, 0, height, env=env)
    theta_Rx = math.atan2(height, h_dist) * 180 / math.pi
    ple = (n_min - n_max) * PLoS + n_max
    sigma_phi_dB = alpha*math.exp(-beta*theta_Rx)
    sigma_phi = 10**(sigma_phi_dB/10)
    K = K_min * math.exp(math.log(K_max/K_min) * PLoS**2)
    omega = 1
    dist = math.sqrt(h_dist**2 + height**2)
    P_Rx = friis_calc(P_Tx, freq, dist, ple)
    eta = math.log(10) / 10
    mu_phi = 10*math.log10(P_Rx)
    E_phi = math.exp(eta*mu_phi + eta**2*sigma_phi**2/2)
    var_phi = math.exp(2*eta*mu_phi+eta**2*sigma_phi**2)*(math.exp(eta**2*sigma_phi**2)-1)
    E_chi = (special.gamma(2)/(1+K))*special.hyp1f1(-1,1,-K)*omega
    var_chi = (special.gamma(3)/(1+K)**2)*special.hyp1f1(-2,1,-K)*omega**2 - E_chi**2
    E_SNR = E_phi * E_chi / noise
    var_SNR = ((var_phi+E_phi**2)*(var_chi+E_chi**2) - E_phi**2 * E_chi**2) / noise**2
    std_dev_SNR = math.sqrt(var_SNR)
    return E_SNR, std_dev_SNR

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

def load_model_no_activation(model_path):
    # Load the model without activation function, but with the trained weights
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', 
                loss={'packet_state': 'categorical_crossentropy'},
                metrics={'packet_state': 'accuracy'})
    model_no_act = build_nn_model_v4_wobatchnorm_noactivation() # For testing calibrated NN
    model_no_act.set_weights(model.get_weights())
    return model_no_act

def find_last_occurrence_manual(my_list, element):
    for i in range(len(my_list) - 1, -1, -1):
        if my_list[i] == element:
            return i
    return np.nan

def mdp_nn_both(dl_model, ul_model, vid_model, dl_T, ul_T, vid_T, height, reliability_th=0.99, init_hdist=0, dmax=500, hdist_step_size=5, save_file=None):
    """
    Adapt both MCS and USI.
    dl_model, ul_model, vid_model: The NN prediction models for DL, UL and VID links, respectively
    dl_T, ul_T, vid_T: The calibration temperature for DL, UL and VID models, respectively
    height: The height of the test case
    reliability_th: The required reliability level
    init_hdist: The starting horizontal distance
    dmax: The maximum horizontal distance (for recording purposes only)
    hdist_step_size: The step size for horizontal distance increment
    save_file: File to save results
    """

    mcs_indexes = [0,1,2,3,4,5,6,7]
    usi_list = [10, 20, 66.7, 100]
    uav_send_int_norm = {10:-1, 20:-0.5, 40:0, 66.7: 0.5, 100:1, 1000:2}
    mcs_bitrate_map = {0: 6.5, 1: 13, 2: 19.5, 3: 26, 4: 39, 5: 52, 6: 58.5, 7: 65}
    bitrate_mcs_map = {v: k for k, v in mcs_bitrate_map.items()}

    # normalization bounds (same as other scripts)
    max_mean_sinr = 10*math.log10(1123)
    max_std_dev_sinr = 10*math.log10(466)
    min_mean_sinr = 10*math.log10(0.2)
    min_std_dev_sinr = 10*math.log10(0.7)

    # # Determine initial (MCS, USI) from ground truth (consistent with other scripts) ------------
    # dl_gt_df = pd.read_csv("/media/research-student/DataDrive/FANET_Dataset/complete_testing_dmax_dataset/data_processed_complete/Downlink_Reliability.csv")
    # ul_gt_df = pd.read_csv("/media/research-student/DataDrive/FANET_Dataset/complete_testing_dmax_dataset/data_processed_complete/Uplink_Reliability.csv")
    # vid_gt_df = pd.read_csv("/media/research-student/DataDrive/FANET_Dataset/complete_testing_dmax_dataset/data_processed_complete/Video_Reliability.csv")
    # dl_gt_df["Reliability"] = dl_gt_df["Num_Reliable"] / (dl_gt_df["Num_Reliable"] + dl_gt_df["Num_Delay_Excd"] + dl_gt_df["Num_Fail_Other"])
    # ul_gt_df["Reliability"] = ul_gt_df["Num_Reliable"] / (ul_gt_df["Num_Reliable"] + ul_gt_df["Num_Delay_Excd"] + ul_gt_df["Num_Fail_Other"])
    # vid_gt_df["Reliability"] = vid_gt_df["Num_Reliable"] / (vid_gt_df["Num_Reliable"] + vid_gt_df["Num_Delay_Excd"] + vid_gt_df["Num_Fail_Other"])
    # rel_df = dl_gt_df.merge(ul_gt_df, on=["Height","UAV_Sending_Interval","Horizontal_Distance","Bitrate"], suffixes=('_dl','_ul'))
    # rel_df = rel_df.merge(vid_gt_df, on=["Height","UAV_Sending_Interval","Horizontal_Distance","Bitrate"])
    # rel_df = rel_df.rename(columns={"Reliability": "Reliability_vid"})
    # # Filter rows at starting distance
    # rel_start = rel_df.loc[(rel_df["Height"]==height) & (rel_df["Horizontal_Distance"]==horizontal_dist[0])]
    # if rel_start.empty:
    #     print("No ground truth rows at start. Abort.")
    #     return
    # # search for initial pair: prefer highest MCS that has any USI meeting reliability
    # found_initial = False
    # for usi in usi_list:  
    #     rows_usi = rel_start.loc[rel_start["UAV_Sending_Interval"]==usi]
    #     if rows_usi.empty:
    #         continue
    #     # check per MCS if all links meet threshold
    #     rows_usi = rows_usi.copy()
    #     rows_usi["OK_dl"] = rows_usi["Reliability_dl"] >= reliability_th
    #     rows_usi["OK_ul"] = rows_usi["Reliability_ul"] >= reliability_th
    #     rows_usi["OK_vid"] = rows_usi["Reliability_vid"] >= reliability_th
    #     rows_usi["ALL_OK"] = rows_usi["OK_dl"] & rows_usi["OK_ul"] & rows_usi["OK_vid"]
    #     ok_rows = rows_usi.loc[rows_usi["ALL_OK"]]
    #     if not ok_rows.empty:
    #         # choose the largest MCS among ok rows (more frequent)
    #         chosen_row = ok_rows.sort_values(by="Bitrate").iloc[-1]
    #         curr_usi = usi
    #         curr_usi_index = usi_list.index(curr_usi)
    #         curr_MCS_index = bitrate_mcs_map[chosen_row["Bitrate"]]
    #         curr_MCS = mcs_indexes[curr_MCS_index]
    #         found_initial = True
    #         break
    # if not found_initial:
    #     print("No initial (MCS,USI) pair can fulfill reliability at start. Abort MDP.")
    #     print(height, horizontal_dist[-1])
    #     return
    # # -----------------------------------------------------------------------------------------

    # Get initial MCS that fulfills reliability requirement at starting point -----------------
    m, s = sinr_lognormal_approx(init_hdist, height)
    mean_sinr = 2*(10*math.log10(m)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1
    std_dev_sinr = 2*(10*math.log10(s)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1
    # Get all combinations of mcs_indexes and usi_kist
    all_combinations = list(itertools.product(usi_list, mcs_indexes)) # Putting usi_list first then mcs_indexes to match later indexing will make the algo favour lower USI over MCS when there are multiple best choices
    usi_combo_list = [usi for usi,_ in all_combinations]
    mcs_combo_list = [mcs for _,mcs in all_combinations]
    inputs = np.hstack((np.array([mean_sinr]*len(all_combinations)).reshape(-1,1), np.array([std_dev_sinr]*len(all_combinations)).reshape(-1,1), 
                        np.array([uav_send_int_norm[usi] for usi in usi_combo_list]).reshape(-1,1), np.array([norm_MCS(mcs) for mcs in mcs_combo_list]).reshape(-1,1)))
    dl_logits = dl_model.predict(inputs, verbose=0)
    init_mcs_usi_dl_rel = np.array([activations.softmax(K.constant([logits/dl_T]), axis=-1)[0][0].numpy() for logits in dl_logits])
    ul_logits = ul_model.predict(inputs, verbose=0)
    init_mcs_usi_ul_rel = np.array([activations.softmax(K.constant([logits/ul_T]), axis=-1)[0][0].numpy() for logits in ul_logits])
    vid_logits = vid_model.predict(inputs, verbose=0)
    init_mcs_usi_vid_rel = np.array([activations.softmax(K.constant([logits/vid_T]), axis=-1)[0][0].numpy() for logits in vid_logits])
    init_mcs_usi_rel_check = np.array([1 if rel >= reliability_th else 0 for rel in init_mcs_usi_dl_rel]) & np.array([1 if rel >= reliability_th else 0 for rel in init_mcs_usi_ul_rel]) & np.array([1 if rel >= reliability_th else 0 for rel in init_mcs_usi_vid_rel]) 
    possible_indexes = np.where(init_mcs_usi_rel_check == 1)[0]
    if len(possible_indexes) == 0:
        print("No MCS+USi combinations can fulfill reliability requirement at starting point. Abort MDP.")
        print(height)
        return
    possible_init_mcs_indexes = [mcs_combo_list[i] for i in possible_indexes]
    possible_init_usi_indexes = [usi_combo_list[i] for i in possible_indexes]

    for init_mcs, init_usi in zip(possible_init_mcs_indexes, possible_init_usi_indexes): # For each possible initial MCS and initial USI combo that fulfills reliability requirement, run the MDP
        state_record = []
        current_hdist = init_hdist
        curr_MCS = init_mcs
        curr_usi = init_usi
        state_record.append({"Horizontal_Distance": current_hdist, "Height": height, "UAV_Send_Interval": curr_usi, "MCS": curr_MCS, 
                            "Status": "Chosen", # Status is to indicate whether the scheme is still running (Continue) or has ended (Dmax/Abort)
                            "DL_Reliability": init_mcs_usi_dl_rel[all_combinations.index((curr_usi, curr_MCS))], 
                            "UL_Reliability": init_mcs_usi_ul_rel[all_combinations.index((curr_usi, curr_MCS))], 
                            "Vid_Reliability": init_mcs_usi_vid_rel[all_combinations.index((curr_usi, curr_MCS))]})
        while True: 
            current_hdist += hdist_step_size # Move to next horizontal distance  
            m, s = sinr_lognormal_approx(current_hdist, height)
            mean_sinr = 2*(10*math.log10(m)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1
            std_dev_sinr = 2*(10*math.log10(s)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1
            # Check reliability at current horizontal distance with current MCS and current USI
            dl_logits = dl_model.predict([[mean_sinr, std_dev_sinr, uav_send_int_norm[curr_usi], norm_MCS(curr_MCS)]], verbose=0)
            ul_logits = ul_model.predict([[mean_sinr, std_dev_sinr, uav_send_int_norm[curr_usi], norm_MCS(curr_MCS)]], verbose=0)
            vid_logits = vid_model.predict([[mean_sinr, std_dev_sinr, uav_send_int_norm[curr_usi], norm_MCS(curr_MCS)]], verbose=0)
            next_dl_rel = np.array([activations.softmax(K.constant([logits/dl_T]), axis=-1)[0][0].numpy() for logits in dl_logits])[0]
            next_ul_rel = np.array([activations.softmax(K.constant([logits/ul_T]), axis=-1)[0][0].numpy() for logits in ul_logits])[0]
            next_vid_rel = np.array([activations.softmax(K.constant([logits/vid_T]), axis=-1)[0][0].numpy() for logits in vid_logits])[0]
            state_record.append({"Horizontal_Distance": current_hdist, "Height": height, "UAV_Send_Interval": curr_usi, "MCS": curr_MCS, 
                            "Status": "Continue", # Status is to indicate whether the scheme is still running (Continue) or has ended (Dmax/Abort)
                            "DL_Reliability": next_dl_rel, "UL_Reliability": next_ul_rel, "Vid_Reliability": next_vid_rel})
            # Check if the next MCS and USI combo is able to maintain the reliability of all links
            next_mcs_usi_rel_check = (next_dl_rel >= reliability_th) & (next_ul_rel >= reliability_th) & (next_vid_rel >= reliability_th)
            if not next_mcs_usi_rel_check:
                # Record that we need to adapt
                state_record[-1]["Status"] = "Adapt"
                # Predict reliabilities for all other MCS and USI (apart from current combo) at this horizontal distance
                all_combinations = list(itertools.product(usi_list, mcs_indexes)) # Putting usi_list first then mcs_indexes to match later indexing will make the algo favour lower USI over MCS when there are multiple best choices
                usi_combo_list = [usi for usi,_ in all_combinations]
                mcs_combo_list = [mcs for _,mcs in all_combinations]
                for idx in range(len(mcs_combo_list)):
                    if (mcs_combo_list[idx] == curr_MCS) and (usi_combo_list[idx] == curr_usi):
                        curr_index = idx
                        break
                mcs_indexes_other = mcs_combo_list[:curr_index] + mcs_combo_list[curr_index+1:] # Remove the current combo
                usi_list_other = usi_combo_list[:curr_index] + usi_combo_list[curr_index+1:] # Remove the current combo
                # For each combo of mcs_indexes_other and usi_list_other, prepare input using numpy full
                mean_sinr_full = np.full((len(mcs_indexes_other),1), mean_sinr)
                std_dev_sinr_full = np.full((len(mcs_indexes_other),1), std_dev_sinr)
                uav_send_int_norm_full = np.array([uav_send_int_norm[usi] for usi in usi_list_other]).reshape(-1,1)
                mcs_norm_full = np.array([norm_MCS(mcs) for mcs in mcs_indexes_other]).reshape(-1,1)
                inputs = np.hstack((mean_sinr_full, std_dev_sinr_full, uav_send_int_norm_full, mcs_norm_full))
                dl_logits = dl_model.predict(inputs, verbose=0)
                next_combo_dl_rel = np.array([activations.softmax(K.constant([logits/dl_T]), axis=-1)[0][0].numpy() for logits in dl_logits])
                ul_logits = ul_model.predict(inputs, verbose=0)
                next_combo_ul_rel = np.array([activations.softmax(K.constant([logits/ul_T]), axis=-1)[0][0].numpy() for logits in ul_logits])
                vid_logits = vid_model.predict(inputs, verbose=0)
                next_combo_vid_rel = np.array([activations.softmax(K.constant([logits/vid_T]), axis=-1)[0][0].numpy() for logits in vid_logits])  
                # Check if any combo is able to maintain the reliability of all links
                next_combo_rel_check = np.array([1 if rel >= reliability_th else 0 for rel in next_combo_dl_rel]) \
                                    & np.array([1 if rel >= reliability_th else 0 for rel in next_combo_ul_rel]) \
                                    & np.array([1 if rel >= reliability_th else 0 for rel in next_combo_vid_rel]) 
                if np.any(next_combo_rel_check):
                    # Get all indexes of USI that fulfill reliability requirement
                    reliable_combo_indexes = np.where(next_combo_rel_check == 1)[0]
                    # Choose the USI that gives the highest mean reliability
                    mean_reliabilities = [(next_combo_dl_rel[i] + next_combo_ul_rel[i] + next_combo_vid_rel[i])/3 for i in reliable_combo_indexes]
                    curr_combo_index = reliable_combo_indexes[np.argmax(mean_reliabilities)]
                    # Record all combo's predictions
                    for i in range(len(next_combo_dl_rel)):
                        state_record.append({"Horizontal_Distance": current_hdist, "Height": height, "UAV_Send_Interval": usi_list_other[i], "MCS": mcs_indexes_other[i], 
                                    "Status": "Prediction", # Status is to indicate whether the scheme is still running (Continue) or has ended (Dmax/Abort)
                                    "DL_Reliability": next_combo_dl_rel[i], "UL_Reliability": next_combo_ul_rel[i], "Vid_Reliability": next_combo_vid_rel[i]})
                        if i == curr_combo_index:
                            state_record[-1]["Status"] = "Chosen"
                            curr_MCS = mcs_indexes_other[curr_combo_index]
                            curr_usi = usi_list_other[curr_combo_index]
                else:
                    # Record all prediction outcomes
                    for i in range(len(next_combo_dl_rel)):
                        state_record.append({"Horizontal_Distance": current_hdist, "Height": height, "UAV_Send_Interval": usi_list_other[i], "MCS": mcs_indexes_other[i], 
                                    "Status": "Prediction", # Status is to indicate whether the scheme is still running (Continue) or has ended (Dmax/Abort)
                                    "DL_Reliability": next_combo_dl_rel[i], "UL_Reliability": next_combo_ul_rel[i], "Vid_Reliability": next_combo_vid_rel[i]})
                    # Record the last state as Abort
                    state_record.append({"Horizontal_Distance": current_hdist, "Height": height, "UAV_Send_Interval": np.nan, "MCS": np.nan, 
                                    "Status": "Abort", # Status is to indicate whether the scheme is still running (Continue) or has ended (Dmax/Abort)
                                    "DL_Reliability": np.nan, "UL_Reliability": np.nan, "Vid_Reliability": np.nan})
                    break
            

        mdp_actions = pd.DataFrame(state_record)
        mdp_actions["Dmax"] = None
        mdp_actions.loc[0, "Dmax"] = dmax
        if save_file is not None:
            mdp_actions.to_csv(save_file.format(height, init_mcs, init_usi), index=False)

    return

def run_mdp(height, max_hdist_df, hdist_step_size, dl_model_path, ul_model_path, vid_model_path, 
            dl_T, ul_T, vid_T, reliability_th, save_path):
    dl_model = load_model_no_activation(dl_model_path)
    ul_model = load_model_no_activation(ul_model_path)
    vid_model = load_model_no_activation(vid_model_path)
    print("Height: {}".format(height))
    try:
        max_hdist = max_hdist_df.loc[(max_hdist_df["Height"]==height)]["D_max"].values[0]
    except Exception:
        print("Could not find D_max for Height: {}. Skipping.".format(height))
        return

    if max_hdist > 0:
        # hdist = list(np.append(np.linspace(0, max_hdist-max_hdist%hdist_step_size, int(max_hdist//hdist_step_size)+1, endpoint=True), np.array([max_hdist])))
        save_file = os.path.join(save_path, 'height-{}_initMCS-{}_initUSI-{}_predictions.csv')
        mdp_nn_both(dl_model, ul_model, vid_model, dl_T, ul_T, vid_T, height, reliability_th=reliability_th, init_hdist=0, dmax=max_hdist, hdist_step_size=hdist_step_size, save_file=save_file)
    print("Done - Height: {}".format(height))
    return

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    DL_NN_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Downlink.round-1_split-9_0.2027.h5"
    UL_NN_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Uplink.round-1_split-9_0.1363.h5"
    VID_NN_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Video.round-1_split-9_0.2720.h5"
    RELIABILITY_TH = 0.999
    RELIABILITY_TH_STR = "999"
    MAX_HDIST_DF_PATH = "/home/research-student/omnet-fanet/data-processing-scripts/mdp_max_hdist_Oct25/MDP_USInMCS_Max_HDist_fm_Sim_{}.csv".format(RELIABILITY_TH_STR)
    SAVE_PATH = "/home/research-student/omnet-fanet/data-processing-scripts/journal_2_scripts/predictions_mcs_n_usi_adaptation/predictions_5m_calnn_{}".format(RELIABILITY_TH_STR)
    NUM_WORKERS = 16
    HDIST_STEP_SIZE = 5
    HEIGHTS = [75, 105, 135, 165, 195, 225, 255, 285]
    # # Calibration T for sensitivity, reliability_th = 0.90
    # DL_T = 1.6933716833921753
    # UL_T = 1.6649418412845718
    # VID_T = 1.6271717659783327
    # # Calibration T for sensitivity, reliability_th = 0.99
    # DL_T = 1.69561830451873
    # UL_T = 1.67551651082116
    # VID_T = 1.62717176597833
    # Calibration T for sensitivity, reliability_th = 0.999
    DL_T = 1.69615238212838
    UL_T = 1.68516226329027
    VID_T = 1.62717176597833

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    max_hdist_df = pd.read_csv(MAX_HDIST_DF_PATH)
    # parallel over heights only; initial MCS/USI determined inside mdp_nn_both based on ground truth at start
    with Pool(NUM_WORKERS) as pool:
        pool.starmap(run_mdp, [(h, max_hdist_df, HDIST_STEP_SIZE, DL_NN_PATH, UL_NN_PATH, VID_NN_PATH, DL_T, UL_T, VID_T, RELIABILITY_TH, SAVE_PATH) for h in HEIGHTS])