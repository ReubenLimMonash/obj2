"""
Date: 30/10/2025
Desc: Reliability-prediction-based MCS+USI adaptation using calibrated NN models.
      MCS is adapted first; if no suitable MCS found, try next USI and restart MCS search.
      Records every prediction and stops when mission aborted (no (MCS,USI) can meet requirement).
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
from itertools import repeat, product

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

def mdp_nn_both(dl_model, ul_model, vid_model, dl_T, ul_T, vid_T, height, reliability_th=0.99, horizontal_dist=None, save_file=None):
    """
    Adapt both MCS and USI.
    Preference/order:
      - For each horizontal distance step: try to adapt MCS first (lower MCS indexes -> more robust).
      - If no MCS for current USI can satisfy reliability, try next USI and restart MCS search.
      - Continue until a (MCS,USI) pair is found or all USI exhausted -> Abort.
    """
    if horizontal_dist is None:
        horizontal_dist = np.linspace(0,600,61,endpoint=True)

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

    state_record = []
    end = 0

    # Determine initial (MCS, USI) from ground truth (consistent with other scripts) ------------
    dl_gt_df = pd.read_csv("/media/research-student/DataDrive/FANET_Dataset/complete_testing_dmax_dataset/data_processed_complete/Downlink_Reliability.csv")
    ul_gt_df = pd.read_csv("/media/research-student/DataDrive/FANET_Dataset/complete_testing_dmax_dataset/data_processed_complete/Uplink_Reliability.csv")
    vid_gt_df = pd.read_csv("/media/research-student/DataDrive/FANET_Dataset/complete_testing_dmax_dataset/data_processed_complete/Video_Reliability.csv")
    dl_gt_df["Reliability"] = dl_gt_df["Num_Reliable"] / (dl_gt_df["Num_Reliable"] + dl_gt_df["Num_Delay_Excd"] + dl_gt_df["Num_Fail_Other"])
    ul_gt_df["Reliability"] = ul_gt_df["Num_Reliable"] / (ul_gt_df["Num_Reliable"] + ul_gt_df["Num_Delay_Excd"] + ul_gt_df["Num_Fail_Other"])
    vid_gt_df["Reliability"] = vid_gt_df["Num_Reliable"] / (vid_gt_df["Num_Reliable"] + vid_gt_df["Num_Delay_Excd"] + vid_gt_df["Num_Fail_Other"])
    rel_df = dl_gt_df.merge(ul_gt_df, on=["Height","UAV_Sending_Interval","Horizontal_Distance","Bitrate"], suffixes=('_dl','_ul'))
    rel_df = rel_df.merge(vid_gt_df, on=["Height","UAV_Sending_Interval","Horizontal_Distance","Bitrate"])
    rel_df = rel_df.rename(columns={"Reliability": "Reliability_vid"})
    # Filter rows at starting distance
    rel_start = rel_df.loc[(rel_df["Height"]==height) & (rel_df["Horizontal_Distance"]==horizontal_dist[0])]
    if rel_start.empty:
        print("No ground truth rows at start. Abort.")
        return
    # search for initial pair: prefer highest MCS that has any USI meeting reliability
    found_initial = False
    for usi in usi_list:  
        rows_usi = rel_start.loc[rel_start["UAV_Sending_Interval"]==usi]
        if rows_usi.empty:
            continue
        # check per MCS if all links meet threshold
        rows_usi = rows_usi.copy()
        rows_usi["OK_dl"] = rows_usi["Reliability_dl"] >= reliability_th
        rows_usi["OK_ul"] = rows_usi["Reliability_ul"] >= reliability_th
        rows_usi["OK_vid"] = rows_usi["Reliability_vid"] >= reliability_th
        rows_usi["ALL_OK"] = rows_usi["OK_dl"] & rows_usi["OK_ul"] & rows_usi["OK_vid"]
        ok_rows = rows_usi.loc[rows_usi["ALL_OK"]]
        if not ok_rows.empty:
            # choose the largest MCS among ok rows (more frequent)
            chosen_row = ok_rows.sort_values(by="Bitrate").iloc[-1]
            curr_usi = usi
            curr_usi_index = usi_list.index(curr_usi)
            curr_MCS_index = bitrate_mcs_map[chosen_row["Bitrate"]]
            curr_MCS = mcs_indexes[curr_MCS_index]
            found_initial = True
            break
    if not found_initial:
        print("No initial (MCS,USI) pair can fulfill reliability at start. Abort MDP.")
        print(height, horizontal_dist[-1])
        return
    # -----------------------------------------------------------------------------------------

    # Do not record start state because it's from ground truth (consistent with other scripts)
    init_mcs_index = 7 # initial MCS to start from each time USI is changed. We make it restart from the top
    for j in range(1, len(horizontal_dist)):
        if end == 1:
            break

        # We'll attempt MCS first for current USI, then try next USI(s) one-by-one restarting MCS search
        # Prepare list of USI indexes to try starting from current_usi_index and then increasing
        usi_candidate_indexes = list(range(curr_usi_index, len(usi_list)))
        success_for_distance = False

        for usi_idx in usi_candidate_indexes:
            curr_usi = usi_list[usi_idx]
            # Get the order of next MCS to try, either by finding next closest value or by decreasing MCS index. UNCOMMENT the one to use
            # To get next closest MCS:
            # next_MCS_indexes = [x for _,x in sorted(zip(np.abs(np.arange(len(mcs_indexes))-curr_MCS_index), np.arange(len(mcs_indexes))))][1:]
            # To get next lower MCS:
            next_MCS_indexes = [i for i in range(curr_MCS_index-1, -1, -1)] # We exploit the fact that curr_MCS_index and curr_MCS are the same
            # Append current MCS index at start of candidate list to try it first
            next_MCS_indexes = [curr_MCS_index] + next_MCS_indexes

            # Try MCS candidates for this USI
            for mcs_idx in next_MCS_indexes:
                curr_MCS_index = mcs_idx
                curr_MCS = mcs_indexes[curr_MCS_index]

                m, s = sinr_lognormal_approx(horizontal_dist[j], height)
                mean_sinr = 2*(10*math.log10(m)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1
                std_dev_sinr = 2*(10*math.log10(s)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1

                dl_logits = dl_model.predict([[mean_sinr, std_dev_sinr, uav_send_int_norm[curr_usi], norm_MCS(curr_MCS)]], verbose=0)
                ul_logits = ul_model.predict([[mean_sinr, std_dev_sinr, uav_send_int_norm[curr_usi], norm_MCS(curr_MCS)]], verbose=0)
                vid_logits = vid_model.predict([[mean_sinr, std_dev_sinr, uav_send_int_norm[curr_usi], norm_MCS(curr_MCS)]], verbose=0)
                next_dl_rel = np.array([activations.softmax(K.constant([logits/dl_T]), axis=-1)[0][0].numpy() for logits in dl_logits])[0]
                next_ul_rel = np.array([activations.softmax(K.constant([logits/ul_T]), axis=-1)[0][0].numpy() for logits in ul_logits])[0]
                next_vid_rel = np.array([activations.softmax(K.constant([logits/vid_T]), axis=-1)[0][0].numpy() for logits in vid_logits])[0]
                # next_dl_rel = dl_model.predict([[mean_sinr, std_dev_sinr, uav_send_int_norm[curr_usi], norm_MCS(curr_MCS)]], verbose=0)[0][0]
                # next_ul_rel = ul_model.predict([[mean_sinr, std_dev_sinr, uav_send_int_norm[curr_usi], norm_MCS(curr_MCS)]], verbose=0)[0][0]
                # next_vid_rel = vid_model.predict([[mean_sinr, std_dev_sinr, uav_send_int_norm[curr_usi], norm_MCS(curr_MCS)]], verbose=0)[0][0]

                # Record every prediction
                state_record.append({"Horizontal_Distance": horizontal_dist[j], "Height": height, "UAV_Send_Interval": curr_usi, "MCS": curr_MCS,
                                     "Status": "Continue", "DL_Reliability": next_dl_rel, "UL_Reliability": next_ul_rel, "Vid_Reliability": next_vid_rel})

                ok = (next_dl_rel >= reliability_th) & (next_ul_rel >= reliability_th) & (next_vid_rel >= reliability_th)
                if ok:
                    # found a viable pair for this distance
                    # If this is final distance mark Dmax
                    if j == len(horizontal_dist)-1:
                        state_record[-1]["Status"] = "Dmax"
                    success_for_distance = True
                    # keep curr_MCS_index and curr_usi_index at these values for next distance
                    curr_usi_index = usi_idx
                    break  # break out of MCS loop; proceed to next distance
                else:
                    # if this MCS didn't work, and there are still MCS candidates, mark this row as Adapt when MCS is changed next loop
                    state_record[-1]["Status"] = "Adapt"
                    # continue to next MCS candidate
                    # If no more MCS candidates left for this USI, we will try next USI
                    continue

            if success_for_distance:
                break  # no need to try further USIs
            else:
                # exhausted MCS candidates for this USI -> if there is another USI to try, mark last record as Adapt to indicate USI change
                if usi_idx != usi_candidate_indexes[-1]:
                    # mark last appended record as Adapt to indicate we will change USI (if any record exists)
                    if len(state_record) > 0:
                        state_record[-1]["Status"] = "Adapt"
                    # continue to next USI (MCS search restarts)
                    curr_MCS_index = init_mcs_index  # reset MCS index to initial each time USI is changed
                    continue
                else:
                    # exhausted all USI (and MCS) -> Abort
                    end = 1
                    if len(state_record) > 0:
                        state_record[-1]["Status"] = "Abort"
                    break

    mdp_actions = pd.DataFrame(state_record)
    mdp_actions["Dmax"] = None
    mdp_actions.loc[0, "Dmax"] = horizontal_dist[-1]
    if save_file is not None:
        mdp_actions.to_csv(save_file, index=False)

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
        hdist = list(np.append(np.linspace(0, max_hdist-max_hdist%hdist_step_size, int(max_hdist//hdist_step_size)+1, endpoint=True), np.array([max_hdist])))
        save_file = os.path.join(save_path, 'height-{}_predictions.csv'.format(height))
        mdp_nn_both(dl_model, ul_model, vid_model, dl_T, ul_T, vid_T, height, reliability_th=reliability_th, horizontal_dist=hdist, save_file=save_file)
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