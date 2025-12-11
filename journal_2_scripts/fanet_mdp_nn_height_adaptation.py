"""
Date: 18/06/2024
Desc: Reliability-prediction-based USI adaptation using calibrated NN models.
      To write XML scenario scripts for test cases
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
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

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

def mdp_write_actions_2_xml(mdp_actions, nodes, uav_speed, xml_file_path,
                            adapt_usi=False, adapt_mcs=True):
    """
    mdp_actions: DF returned by mdp() function
    nodes: The list of nodes to apply parameter changes (includes GCS and UAVs)
    xml_file_path: scenario script file path
    adapt_height: Whether or not to adapt height
    adapt_usi: Whether or not to adapt UAV sending interval
    adapt_mcs: Whether or not to adapt MCS
    NOTE: Do not apply drop_duplicates to mdp_actions before passing it in function
    """
    # TODO: Complete for height adaptation
    uav_nodes = [node for node in nodes if "GCS" not in node] # To exclude GCS for USI adaptation
    mdp_actions_copy = mdp_actions.drop_duplicates(subset=["Height", "UAV_Send_Interval", "MCS"], keep='first')
    mdp_actions_diff = mdp_actions_copy.diff()
    # any_mcs_adapt = mdp_actions_diff["MCS"].any() # To check if there's any MCS adaptations
    # any_usi_adapt = mdp_actions_diff["UAV_Send_Interval"].any() # To check if there's any USI adaptations
    ''' Write the XML File for Scenario Scripting '''
    scenario = ET.Element('scenario')
    # Write the first row action
    first_mdp_actions = mdp_actions_copy.iloc[0]
    time = first_mdp_actions.Horizontal_Distance / uav_speed
    mcs = int(first_mdp_actions.MCS)
    usi = first_mdp_actions.UAV_Send_Interval
    at = ET.SubElement(scenario, 'at', t=str(time))
    if adapt_mcs:
        for node in nodes:
            set_param = ET.SubElement(at, 'set-param', attrib={"module":"{}.wlan[0].mac.dcf.rateControl".format(node), "par":"MCSIndex", "value":str(mcs)})
    if adapt_usi:
        for node in uav_nodes:
            set_param = ET.SubElement(at, 'set-param', attrib={"module":"{}.app[1]".format(node), "par":"sendInterval", "value":str(usi)+"ms"})
    
    # Write the rest of the actions
    for i in range(1,len(mdp_actions_diff)):
        row_mdp_actions = mdp_actions_copy.iloc[i]
        time = row_mdp_actions.Horizontal_Distance / uav_speed
        mcs = int(row_mdp_actions.MCS)
        usi = row_mdp_actions.UAV_Send_Interval
        flag_adapt_mcs = ((mdp_actions_diff.iloc[i].MCS != 0.0) and adapt_mcs) # Check whether to adapt MCS in this iter
        flag_adapt_usi = ((mdp_actions_diff.iloc[i].UAV_Send_Interval != 0.0) and adapt_usi) # Check whether to adapt USI in this iter 
        if flag_adapt_mcs or flag_adapt_usi:
            at = ET.SubElement(scenario, 'at', t=str(time))
            # If MCS changed, write new MCS to scenario script
            if flag_adapt_mcs:
                for node in nodes:
                    set_param = ET.SubElement(at, 'set-param', attrib={"module":"{}.wlan[0].mac.dcf.rateControl".format(node), "par":"MCSIndex", "value":str(mcs)})
            # If sendInterval changed, write new sendInterval to scenario script
            if flag_adapt_usi:
                for node in uav_nodes:
                    set_param = ET.SubElement(at, 'set-param', attrib={"module":"{}.app[1]".format(node), "par":"sendInterval", "value":str(usi)+"ms"})

    scenario_script = ET.ElementTree(scenario)
    scenario_script = BeautifulSoup(ET.tostring(scenario_script.getroot(), 'utf-8'), features="html.parser")
    with open(xml_file_path, 'w') as f:
        f.write(scenario_script.prettify())

def mdp_write_heights_bonn_motion(mdp_actions, uav_speed, max_hdist, file_path, hdist_step_size=1, x=0, y=0):
    """ 
    Write the Bonn Motion file for height adaptation. The motion file is intended for the gateway UAV. The other UAVs will use AttachedMobility
    mdp_actions: DF returned by mdp() function
    file_path: scenario script file path
    hdist_step_size: Horizontal Distance step size used in mdp
    x: Initial X-coord of UAV swarm, UAV swarm will move in this direction
    y: Y-coord of UAV swarm
    # NOTE: This should be integrated into mdp_write_actions_2_xml
    """
    mdp_actions_copy = mdp_actions.drop_duplicates(subset=["Height", "UAV_Send_Interval", "MCS"], keep='first')
    mdp_actions_diff = mdp_actions_copy.diff()
    bonn_motion = [] # To append values for the motion file
    # Write the starting height
    first_mdp_actions = mdp_actions_copy.iloc[0]
    hdist = first_mdp_actions.Horizontal_Distance
    assert hdist == 0, "Initial horizontal distance should be 0"
    height = first_mdp_actions.Height
    bonn_motion += [0, x, y, height]
    # Write for rest of heights
    curr_height = height
    curr_hdist = hdist
    curr_time = 0
    # If in the very first step (after 0) the height needs to change, I may need to adapt the following to take it into consideration
    for i in range(1,len(mdp_actions_diff)):
        if (mdp_actions_diff.iloc[i].Height != 0.0):
            row_mdp_actions = mdp_actions_copy.iloc[i]
            hdist = row_mdp_actions.Horizontal_Distance # This is the hdist by which the height must be updated
            hdist_trans = hdist - hdist_step_size # Change the height at this distance
            height = row_mdp_actions.Height
            # First adjust the hdist to reach the point where height needs to change (NOTE: This is one step size before the hdist)
            time = (hdist_trans - curr_hdist) / uav_speed # Time taken to travel to the hdist being considered
            bonn_motion += [curr_time + time, x + hdist_trans, y, curr_height]
            curr_time = curr_time + time
            # Then, adjust the height
            time = np.abs(height - curr_height) / uav_speed # Time taken to travel to the new height
            bonn_motion += [curr_time + time, x + hdist_trans, y, height]
            curr_time = curr_time + time
            curr_height = height
            curr_hdist = hdist_trans
    
    # Write the final motion to max_hdist (if curr_hdist is not max_hdist)
    if curr_hdist != max_hdist:
        time = (max_hdist - curr_hdist) / uav_speed
        bonn_motion += [curr_time + time, x + max_hdist, y, curr_height]
    
    # Write the motion file
    with open(file_path, 'w') as f:
        f.write(" ".join([str(x) for x in bonn_motion]))

def mdp_nn(dl_model, ul_model, vid_model, reliability_th=0.99, 
        horizontal_dist = None, heights=None, uav_send_ints=None, mcs_indexes=None,
        init_height=None, init_usi=None, init_mcs_index=7,
        save_file=None, end_at_max=False):
    '''
    Proposed communication reliability maintenance scheme
    Inputs:
    dl_model, ul_model, vid_model: The NN prediction models for DL, UL and VID links, respectively
    reliability_th: The required reliability level
    horizontal_dist: List of horizontal distances to travel through. Assumes ascending order. If None, defaults to 0-1200m, step size 10m
    heights: List of possible heights for adjustment. If None, defaults to [60, 90, 120, 150, 180, 210, 240, 270, 300]
    uav_send_ints: List of possible UAV sending intervals for adjustment. If None, defaults to [10, 20, 66.7, 100]
    mcs_indexes: List of possible MCS indexes for adjustment. If None, defaults to [7,6,5,4,3,2,1,0]
    init_height: The initial to start the algorithm from. If None, will be set to first height in heights
    init_usi: The initial uav send interval to start the algorithm from. If None, will be set to first usi in uav_send_ints
    init_mcs_index: The MCS index to start the algorithm from. If None, will be set to first mcs in mcs_indexes
    save_file: File to save results
    end_at_max: Flag to end the algorithm at max horizontal distance that fulfills reliability_th, 
                or continue for all horizontal_distance points while choosing the best MCS for overall reliability.
    '''

    # reliability_th = 0.99 # Target reliability 

    # Possible parameters:
    if type(heights) == type(None):
        heights = np.arange(60,301) # Step siz of 1m
    if type(uav_send_ints) == type(None):
        uav_send_ints = [10, 20, 66.7, 100]
    if type(horizontal_dist) == type(None):
        horizontal_dist = np.linspace(0, 700, 71, endpoint=True)
    if type(mcs_indexes) == type(None):
        mcs_indexes = [7,6,5,4,3,2,1,0]
    else:
        mcs_indexes = np.sort(mcs_indexes)[::-1] # MCS Indexes needs to be sorted in descending order for the algo to take highest MCS possible at each hdist step
    if type(init_height) == type(None):
        init_height = heights[0]
    if type(init_usi) == type(None):
        init_usi = uav_send_ints[0]
    if type(init_mcs_index) == type(None):
        init_mcs_index = mcs_indexes[0]
    
    max_mean_sinr = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
    max_std_dev_sinr = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
    min_mean_sinr = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
    min_std_dev_sinr = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)

    uav_send_int_norm = {10:-1, 20:-0.5, 40:0, 66.7: 0.5, 100:1, 1000:2}

    # MDP 
    assert init_height in heights, "init_height not in heights list/array"
    assert init_usi in uav_send_ints, "init_usi not in uav_send_ints list/array"
    assert init_mcs_index in mcs_indexes, "init_mcs_index not in mcs_indexes list/array"

    curr_uav_send_int_index = np.where(np.array(uav_send_ints) == init_usi)[0][0] # Index for UAV send int listed in variable uav_send_ints
    curr_height_index = np.where(np.array(heights) == init_height)[0][0]
    curr_MCS = np.where(np.array(mcs_indexes) == init_mcs_index)[0][0] # MCS Index
    state_record = []
    end = 0
    for j in range(0, len(horizontal_dist)): # Traverse the h_dist
        if end == 1:
            break
        # Get the order of closest heights and uav send ints based on current
        if len(heights) > 1:
            closest_height_indexes = [x for _,x in sorted(zip(np.abs(np.arange(len(heights))-curr_height_index), np.arange(len(heights))))][1:]
        else:
            closest_height_indexes = [] # Height is fixed in this case
        if len(uav_send_ints) > 1:
            closest_uav_send_int_indexes = [x for _,x in sorted(zip(np.abs(np.arange(len(uav_send_ints))-curr_uav_send_int_index), np.arange(len(uav_send_ints))))][1:]
        else:
            closest_uav_send_int_indexes = []
        record_all_actions = [] # To record all actions performed, to find best if end_at_max is True
        next_height_index = curr_height_index # Initialize next_height_index
        next_uav_send_int_index = curr_uav_send_int_index # Initialize next_uav_send_int_index
        # First, check if next state is reliable
        m, s = sinr_lognormal_approx(horizontal_dist[j], heights[curr_height_index])
        mean_sinr = 2*(10*math.log10(m)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1
        std_dev_sinr = 2*(10*math.log10(s)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1
        next_dl_rel = dl_model.predict([[mean_sinr, std_dev_sinr, uav_send_int_norm[uav_send_ints[curr_uav_send_int_index]], norm_MCS(curr_MCS)]], verbose=0)[0][0]
        next_ul_rel = ul_model.predict([[mean_sinr, std_dev_sinr, uav_send_int_norm[uav_send_ints[curr_uav_send_int_index]], norm_MCS(curr_MCS)]], verbose=0)[0][0]
        next_vid_rel = vid_model.predict([[mean_sinr, std_dev_sinr, uav_send_int_norm[uav_send_ints[curr_uav_send_int_index]], norm_MCS(curr_MCS)]], verbose=0)[0][0]
        record_all_actions.append(pd.DataFrame({"Height_Index": [curr_height_index], "UAV_Send_Interval_Index": [curr_uav_send_int_index], "MCS": [curr_MCS],
                                   "DL_Reliability": [next_dl_rel], "UL_Reliability": [next_ul_rel], "VID_Reliability": [next_vid_rel]}))
        # If next state is not reliable, first explore different MCS before trying to change sending rate
        while (next_dl_rel < reliability_th) or (next_ul_rel < reliability_th) or (next_vid_rel < reliability_th):
            inputs = np.hstack((np.array([mean_sinr]*len(mcs_indexes)).reshape(-1,1), np.array([std_dev_sinr]*len(mcs_indexes)).reshape(-1,1), 
                                np.array([uav_send_int_norm[uav_send_ints[next_uav_send_int_index]]]*len(mcs_indexes)).reshape(-1,1), np.array([norm_MCS(mcs) for mcs in mcs_indexes]).reshape(-1,1)))
            next_mcs_dl_results = dl_model.predict(inputs, verbose=0)
            next_mcs_dl_rel = [prob[0] for prob in next_mcs_dl_results]
            next_mcs_ul_results = ul_model.predict(inputs, verbose=0)
            next_mcs_ul_rel = [prob[0] for prob in next_mcs_ul_results]
            next_mcs_vid_results = vid_model.predict(inputs, verbose=0)
            next_mcs_vid_rel = [prob[0] for prob in next_mcs_vid_results]
            temp_df = pd.DataFrame({"DL_Reliability": next_mcs_dl_rel, "UL_Reliability": next_mcs_ul_rel, "VID_Reliability": next_mcs_vid_rel})
            temp_df["Height_Index"] = next_height_index
            temp_df["UAV_Send_Interval_Index"] = next_uav_send_int_index
            temp_df["MCS"] = mcs_indexes
            record_all_actions.append(temp_df)
            # Check if any MCS is able to maintain the reliability of all links
            next_mcs_rel_check = np.array([1 if rel >= reliability_th else 0 for rel in next_mcs_dl_rel]) & np.array([1 if rel >= reliability_th else 0 for rel in next_mcs_ul_rel]) & np.array([1 if rel >= reliability_th else 0 for rel in next_mcs_vid_rel]) 
            if not np.any(next_mcs_rel_check):
                if len(closest_height_indexes) == 0: # This means we've tried all heights
                    if len(closest_uav_send_int_indexes) == 0: # This means we've tried all UAV send ints
                        # If cannot increase the sending int further, and no MCS / height fulfills the requirements:
                        # Either just take best case for overall reliability ----------
                        if not end_at_max:
                            record_all_actions_df = pd.concat(record_all_actions)
                            dl_rel = record_all_actions_df["DL_Reliability"].values
                            ul_rel = record_all_actions_df["UL_Reliability"].values
                            vid_rel = record_all_actions_df["VID_Reliability"].values

                            # # To use max overall reliability, UNCOMMENT BELOW
                            # overall_reliabilty = np.array(dl_rel) * np.array(ul_rel) * np.array(vid_rel)
                            # index = np.argmax(overall_reliabilty)
                            # To use max minimum reliability, UNCOMMENT BELOW
                            min_reliabilties = np.min(np.vstack((dl_rel, ul_rel, vid_rel)), axis=0)

                            index = np.argmax(min_reliabilties)
                            best_action = record_all_actions_df.iloc[index]
                            curr_MCS = best_action.MCS
                            curr_height_index = int(best_action.Height_Index)
                            curr_uav_send_int_index = int(best_action.UAV_Send_Interval_Index)
                            next_dl_rel = best_action.DL_Reliability
                            next_ul_rel = best_action.UL_Reliability
                            next_vid_rel = best_action.VID_Reliability
                        # Or stop the algorithm ----------------------------------------
                        else:
                            end = 1 # Stop the MDP
                        break
                    else:
                        # If none of the heights were able to satisfy reliability, we try to next closest UAV send int
                        # Reset the closest_height_indexes (NOTE: curr_height_index should not have been changed at this point from previous value)
                        if len(heights) > 1:
                            closest_height_indexes = [x for _,x in sorted(zip(np.abs(np.arange(len(heights))-curr_height_index), np.arange(len(heights))))][1:]
                        else:
                            closest_height_indexes = []
                        next_height_index = curr_height_index # Reset next_height_index
                        next_uav_send_int_index = closest_uav_send_int_indexes.pop(0) # Straight change the UAV send int
                        # print("Updating current send int: {}".format(curr_uav_send_int_index))
                        # Update SINR param based on current height
                        m, s = sinr_lognormal_approx(horizontal_dist[j], heights[curr_height_index])
                        mean_sinr = 2*(10*math.log10(m)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1
                        std_dev_sinr = 2*(10*math.log10(s)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1
                else:
                    # Try next closest height, update SINR params
                    next_height_index = closest_height_indexes.pop(0)
                    # print("Setting next height: {}".format(next_height_index))
                    m, s = sinr_lognormal_approx(horizontal_dist[j], heights[next_height_index])
                    mean_sinr = 2*(10*math.log10(m)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1
                    std_dev_sinr = 2*(10*math.log10(s)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1
            else:
                # Take highest MCS index that fulfills requirement
                index = next_mcs_rel_check.tolist().index(1) # The index returned should correspond to the correct MCS index
                curr_MCS = mcs_indexes[index]
                curr_height_index = next_height_index
                curr_uav_send_int_index = next_uav_send_int_index
                # print("Updating current height: {}".format(curr_height_index))
                next_dl_rel = next_mcs_dl_rel[index]
                next_ul_rel = next_mcs_ul_rel[index]
                next_vid_rel = next_mcs_vid_rel[index]
        
        # To record the optimal acctions under each height:
        state_record.append({"Horizontal_Distance": horizontal_dist[j], "Height": heights[curr_height_index], "UAV_Send_Interval": uav_send_ints[curr_uav_send_int_index], "MCS": curr_MCS, 
                            "DL_Reliability": next_dl_rel, "UL_Reliability": next_ul_rel, "Vid_Reliability": next_vid_rel})

    mdp_actions = pd.DataFrame(state_record)

    if save_file is not None:
        mdp_actions.to_csv(save_file)

    return mdp_actions

def run_mdp(uav_speed, usi, mcs, init_height, heights, max_hdist_df, hdist_step_size, nodes, 
            dl_model_path, ul_model_path, vid_model_path, reliability_th, 
            scenario_script_path, bonn_motion_script_path, init_X_coord=0, Y_coord=0):
    # Load models
    dl_model = load_model(dl_model_path)
    ul_model = load_model(ul_model_path)
    vid_model = load_model(vid_model_path)
    # Run MDP
    print("UAV Speed:{}, USI: {}, MCS: {}".format(uav_speed, usi, mcs))
    max_hdist = max_hdist_df.loc[(max_hdist_df["UAV_Sending_Interval"]==usi) & (max_hdist_df["MCS"]==mcs)]["Max_Horizontal_Distance"].values[0]
    if (not pd.isna(max_hdist)) & (max_hdist > 0): # Exclude test cases with 0 max hdist
        hdist = list(np.append(np.linspace(0, max_hdist-max_hdist%hdist_step_size, int(max_hdist//hdist_step_size)+1, endpoint=True), np.array([max_hdist]))) # Evaluate for every 10m, includes last point (max_hdist)
        mdp_actions = mdp_nn(dl_model, ul_model, vid_model, reliability_th=reliability_th,
                            horizontal_dist=hdist, heights=heights, uav_send_ints=[usi], mcs_indexes=[mcs], 
                            init_height=init_height, init_usi=usi, init_mcs_index=mcs, save_file=None, end_at_max=False)
        # Write scenario script
        scenario_script_file = os.path.join(scenario_script_path, "initHeight-{}_usi-{}_mcs-{}_uavspeed-{}_scenario_script.xml".format(init_height, usi, mcs, uav_speed))
        mdp_write_actions_2_xml(mdp_actions, nodes, uav_speed, scenario_script_file, adapt_usi=True, adapt_mcs=False)
        boon_motion_file = os.path.join(bonn_motion_script_path, "initHeight-{}_usi-{}_mcs-{}_uavspeed-{}.movements".format(init_height, usi, mcs, uav_speed))
        mdp_write_heights_bonn_motion(mdp_actions, uav_speed, max_hdist, boon_motion_file, hdist_step_size=hdist_step_size, x=init_X_coord, y=Y_coord)
        print("Done - UAV Speed:{}, USI: {}, MCS: {}".format(uav_speed, usi, mcs))
    else:
        print("Test Case with No HDist - UAV Speed:{}, USI: {}, MCS: {}".format(uav_speed, usi, mcs))
    return

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    ''' Set parameters '''
    DL_NN_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Downlink.round-0_split-6_0.2024.h5"
    UL_NN_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Uplink.round-1_split-4_0.1346.h5"
    VID_NN_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Video.round-0_split-8_0.2655.h5"
    MAX_HDIST_DF_PATH = "/home/research-student/omnet-fanet/data-processing-scripts/mdp_max_hdist/MDP_Height_Max_HDist_fm_Sim_99.csv"
    SCENARIO_SCRIPT_PATH = "/home/research-student/omnet-fanet/scenario_scripts_height_adaptation/scenario_scripts_1m_nn_99"
    BONN_MOTION_PATH = "/home/research-student/omnet-fanet/bonn_motion_height_adaptation/bonn_motion_1m_nn_99"
    RELIABILITY_TH = 0.99
    INIT_X = 0 # The X Coord of the GCS in m
    Y_COORD = 0 # The Y Coord of the GCS in m
    NUM_WORKERS = 32
    HDIST_STEP_SIZE = 1 # In m
    MCS_LIST = np.arange(0, 8, 1)
    USI_LIST = [10, 20, 66.7, 100] # In ms 
    HEIGHTS = np.arange(60, 301)
    INIT_HEIGHTS = [180, 60, 300] # In m
    UAV_SPEEDS = [6]
    NODES = ["GCS", "gatewayNode", "adhocNode[0]", "adhocNode[1]", "adhocNode[2]", "adhocNode[3]", "adhocNode[4]", "adhocNode[5]", "adhocNode[6]"]

    """ Run MDP """
    max_hdist_df = pd.read_csv(MAX_HDIST_DF_PATH)
    var_params = [i for i in product(UAV_SPEEDS, USI_LIST, MCS_LIST, INIT_HEIGHTS)]
    with Pool(NUM_WORKERS) as pool:
        pool.starmap(run_mdp, zip(*zip(*var_params), repeat(HEIGHTS), repeat(max_hdist_df), repeat(HDIST_STEP_SIZE), repeat(NODES), 
                    repeat(DL_NN_PATH), repeat(UL_NN_PATH), repeat(VID_NN_PATH), 
                    repeat(RELIABILITY_TH), repeat(SCENARIO_SCRIPT_PATH), repeat(BONN_MOTION_PATH), repeat(INIT_X), repeat(Y_COORD)))