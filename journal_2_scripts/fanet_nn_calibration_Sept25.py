"""
Date: 15/09/2025
Desc: Edited fanet_nn_calibration.py to change definition of positive and negative prediction, and to calibrate for weighted Youden J
"""

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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from pickle import dump 


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
    df.loc[(df["Modulation"] == "QPSK") & (df["Bitrate"] == 13), "MCS"] = 1 # MCS Index 0
    df.loc[(df["Modulation"] == "QPSK") & (df["Bitrate"] == 19.5), "MCS"] = 2 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM16") & (df["Bitrate"] == 26), "MCS"] = 3 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM16") & (df["Bitrate"] == 39), "MCS"] = 4 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 52), "MCS"] = 5 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 58.5), "MCS"] = 6 # MCS Index 0
    df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 65), "MCS"] = 7 # MCS Index 0

    return df

def get_output_layer(model, layer_name):
    # From https://github.com/jacobgil/keras-cam/blob/master/model.py#L79
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

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

def objective_accuracy(T, X_train, y_train, reliability_th):
    """
    Objective function for temperature scaling optimization for accuracy
    T: temperature to scale the logits
    X_train: Logits from NN model evaluated on dataset (numpy arr)
    y_train: Boolean array of True/False for reliability state (reliability>=threshold) (numpy arr)
    reliability_th: Threshold value for reliability
    """
    calibrated_reliability_prediction = np.array([activations.softmax(K.constant([logits/T]), axis=-1)[0][0].numpy() for logits in X_train])
    calibrated_reliability_state = calibrated_reliability_prediction >= reliability_th
    accuracy = accuracy_score(y_train, calibrated_reliability_state)
    # Return the complement of accuracy since we are using a minimizing optimization
    return 1 - accuracy


def objective_f1score(T, X_train, y_train, reliability_th):
    """
    Objective function for temperature scaling optimization for F1 Score
    T: temperature to scale the logits
    X_train: Logits from NN model evaluated on dataset (numpy arr)
    y_train: Boolean array of True/False for reliability state (reliability>=threshold) (numpy arr)
    reliability_th: Threshold value for reliability
    """
    calibrated_reliability_prediction = np.array([activations.softmax(K.constant([logits/T]), axis=-1)[0][0].numpy() for logits in X_train])
    calibrated_reliability_state = calibrated_reliability_prediction >= reliability_th
    f1Score = f1_score(y_train, calibrated_reliability_state)
    # Return the complement of f1 score since we are using a minimizing optimization
    return 1 - f1Score

def objective_precision(T, X_train, y_train, reliability_th):
    """
    Objective function for temperature scaling optimization for precision
    T: temperature to scale the logits
    X_train: Logits from NN model evaluated on dataset (numpy arr)
    y_train: Boolean array of True/False for reliability state (reliability>=threshold) (numpy arr)
    reliability_th: Threshold value for reliability
    """
    calibrated_reliability_prediction = np.array([activations.softmax(K.constant([logits/T]), axis=-1)[0][0].numpy() for logits in X_train])
    calibrated_reliability_state = calibrated_reliability_prediction >= reliability_th
    precision = precision_score(y_train, calibrated_reliability_state)
    # Return the complement of precision since we are using a minimizing optimization
    return 1 - precision

def objective_specificity(T, X_train, y_train, reliability_th):
    """
    Objective function for temperature scaling optimization for specificity 
    NOTE: Setting pos_label=False in recall_score calculates specificity rather than recall
    T: temperature to scale the logits
    X_train: Logits from NN model evaluated on dataset (numpy arr)
    y_train: Boolean array of True/False for reliability state (reliability>=threshold) (numpy arr)
    reliability_th: Threshold value for reliability
    """
    calibrated_reliability_prediction = np.array([activations.softmax(K.constant([logits/T]), axis=-1)[0][0].numpy() for logits in X_train])
    calibrated_reliability_state = calibrated_reliability_prediction < reliability_th
    specificity = recall_score(y_train, calibrated_reliability_state, pos_label=False, average='binary')
    # Return the complement of precision since we are using a minimizing optimization
    return 1 - specificity

def objective_recall(T, X_train, y_train, reliability_th):
    """
    Objective function for temperature scaling optimization for specificity 
    NOTE: Setting pos_label=False in recall_score calculates specificity rather than recall
    T: temperature to scale the logits
    X_train: Logits from NN model evaluated on dataset (numpy arr)
    y_train: Boolean array of True/False for reliability state (reliability>=threshold) (numpy arr)
    reliability_th: Threshold value for reliability
    """
    calibrated_reliability_prediction = np.array([activations.softmax(K.constant([logits/T]), axis=-1)[0][0].numpy() for logits in X_train])
    calibrated_reliability_state = calibrated_reliability_prediction < reliability_th
    recall = recall_score(y_train, calibrated_reliability_state, pos_label=True, average='binary')
    # Return the complement of precision since we are using a minimizing optimization
    return 1 - recall

def objective_weighted_youden_j(T, X_train, y_train, reliability_th, w):
    """
    Objective function for temperature scaling optimization for weighted Youden J
    NOTE: Setting pos_label=False in recall_score calculates specificity rather than recall
    T: temperature to scale the logits
    X_train: Logits from NN model evaluated on dataset (numpy arr)
    y_train: Boolean array of True/False for reliability state (reliability>=threshold) (numpy arr)
    reliability_th: Threshold value for reliability
    w: The weight of sensitivity in weighted Youden J
    """
    calibrated_reliability_prediction = np.array([activations.softmax(K.constant([logits/T]), axis=-1)[0][0].numpy() for logits in X_train])
    calibrated_reliability_state = calibrated_reliability_prediction < reliability_th
    recall = recall_score(y_train, calibrated_reliability_state, pos_label=True, average='binary')
    specificity = recall_score(y_train, calibrated_reliability_state, pos_label=False, average='binary')
    weight_youden_j = 2 * (w * recall + (1-w) * specificity) - 1
    # Return 1 - weight_youden_j since we are using a minimizing optimization
    return 1 - weight_youden_j

if __name__ == "__main__":

    MODEL_PATHS = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Downlink.round-1_split-9_0.2027.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Uplink.round-1_split-9_0.1363.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Video.round-1_split-9_0.2720.h5"]
    # NOTE: Make sure DATASET_PATHS correspond to MODEL_PATHS
    DATASET_PATHS = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed/Downlink_Reliability.csv",
                     "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed/Uplink_Reliability.csv",
                     "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed/Video_Reliability.csv"]
    # NOTE: Make sure SAVE_PATHS correspond to MODEL_PATHS
    metric = "w75YoudenJ_999"
    SAVE_PATHS = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_calibration/djispark_dl_calibration_T_{}.pkl".format(metric),
                "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_calibration/djispark_ul_calibration_T_{}.pkl".format(metric),
                "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_calibration/djispark_vid_calibration_T_{}.pkl".format(metric)]
    SAVE_FILE = "nn_calibration/nn_calibration_T_{}.csv".format(metric)
    RELIABILITY_TH = 0.999
    BOUND_T = (0.3, 1.7)
    YOUDEN_J_W = 0.75 # Weight for Youden J
    NUM_WORKERS = 32
    SEED = 100
    DATA_DF_DTYPES = {"Horizontal_Distance": np.float64, "Height": np.int16,	"U2G_Distance": np.int32, "UAV_Sending_Interval": np.float64, "Mean_SINR": np.float64, "Std_Dev_SINR": np.float64,
                        "Num_Sent": np.int32, "Num_Reliable": np.int32, "Num_Delay_Excd": np.int32, "Num_Incr_Rcvd": np.int32, "Num_Q_Overflow": np.int32, "Modulation": str, "Bitrate": np.float64}
    calibration_T_list = []
    for i in range(len(MODEL_PATHS)):
        """ Load NN Model and Rebuild with No Output Activation Layer """
        model = tf.keras.models.load_model(MODEL_PATHS[i], compile=False)
        model.compile(optimizer='adam', 
                    loss={'packet_state': 'categorical_crossentropy'},
                    metrics={'packet_state': 'accuracy'})
        model_no_act = build_nn_model_v4_wobatchnorm_noactivation()
        model_no_act.set_weights(model.get_weights())
        
        """ Load train dataset """
        context = DATASET_PATHS[i].split("/")[-1].split(".")[0]
        dataset_details_df = pd.read_csv(DATASET_PATHS[i], 
                                        usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Bitrate", "Num_Sent", "Num_Reliable", "Num_Delay_Excd",
                                                    "Num_Incr_Rcvd", "Num_Q_Overflow"],
                                        dtype=DATA_DF_DTYPES)
        dataset_details_df = get_mcs_index(dataset_details_df)
        dataset_details_df = normalize_data(dataset_details_df, columns=["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"], save_details_path=None) 
        dataset_details_df["Reliability"] = (dataset_details_df["Num_Reliable"] / dataset_details_df["Num_Sent"]).values
        
        """ Get the appropriate y_train, depending on metric optimizing for"""
            # For reliability-level classificiation accuracy or F1 score or specificity or recall
        y_train = dataset_details_df["Reliability"] < RELIABILITY_TH # reliability requirement (Positive prediction means reliability below required level)
            
        """ Get logits from model """
        X_train = model_no_act.predict(dataset_details_df[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"]].values)

        """ Run Calibration """
        # calibrated_T = minimize(objective, init_T, args=(X_train, y_train, RELIABILITY_TH), method='BFGS', options={'gtol': 1e-3, 'eps': 0.1, 'maxiter': 100, 'disp': True})
        # calibrated_T = differential_evolution(objective_accuracy, bounds=[BOUND_T], args=(X_train, y_train, RELIABILITY_TH), workers=NUM_WORKERS, seed=SEED, disp=True)
        # calibrated_T = differential_evolution(objective_f1score, bounds=[BOUND_T], args=(X_train, y_train, RELIABILITY_TH), workers=NUM_WORKERS, seed=SEED, disp=True)
        # calibrated_T = differential_evolution(objective_specificity, bounds=[BOUND_T], args=(X_train, y_train, RELIABILITY_TH), workers=NUM_WORKERS, seed=SEED, disp=True)
        # calibrated_T = differential_evolution(objective_recall, bounds=[BOUND_T], args=(X_train, y_train, RELIABILITY_TH), workers=NUM_WORKERS, seed=SEED, disp=True)
        calibrated_T = differential_evolution(objective_weighted_youden_j, bounds=[BOUND_T], args=(X_train, y_train, RELIABILITY_TH, YOUDEN_J_W), workers=NUM_WORKERS, seed=SEED, disp=True)

        """ Save Results """
        dump(calibrated_T, open(SAVE_PATHS[i], 'wb'))
        model_name = MODEL_PATHS[i]
        calibration_T_list.append({"Model": model_name, "T": calibrated_T.x[0], "Success": calibrated_T.success, "Seed": SEED})

    calibration_T_df = pd.DataFrame(calibration_T_list)
    calibration_T_df.to_csv(SAVE_FILE, index=False)
