'''
Date: 11/09/2025
Desc: NN and calibrated NN prediction testing. Updated to test sensitivity of specificity-calibrated NN
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
    NN_MODELS = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Downlink.round-1_split-9_0.2027.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Uplink.round-1_split-9_0.1363.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts/model_Video.round-1_split-9_0.2720.h5"]
    NN_CALIBRATION_T_SPECIFICITY_90 = [1.69337168339217,1.66494184128457,1.62717176597833]
    NN_CALIBRATION_T_SPECIFICITY_99 = [1.69561830451873,1.67551651082116,1.62717176597833]
    NN_CALIBRATION_T_SPECIFICITY_999 = [1.69615238212838,1.68516226329027,1.62717176597833]
    TEST_DATASET = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/test_dataset_{}_processed/Downlink_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/test_dataset_{}_processed/Uplink_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/test_dataset_{}_processed/Video_Reliability.csv"]
    SAVE_PATH = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/Revised_Results_DJISpark_Downlink_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/Revised_Results_DJISpark_Uplink_Reliability.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/Revised_Results_DJISpark_Video_Reliability.csv"]
    METRIC_SAVE_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/Revised_Reliability_Prediction_Results_Obj2_final_model.csv"
    MAX_HDIST = 700 # Since the test datasets goes up to 1200m
    data_dtypes = {"Horizontal_Distance": np.float64, "Height":np.float64, "UAV_Sending_Interval": np.float64, "Modulation": 'str', "Bitrate": np.float64}
    # For BN Model
    HDIST_BIN = np.arange(0, 710, 10) # For associating each hdist to its nearest value in train dataset
    HEIGHT_BIN = np.arange(60, 330, 30) # For associating each height to its nearest value in train dataset
    metrics_list = [] # To store accuracy, F1, specificity, sensitivity
    for i in range(len(NN_MODELS)):
        ''' Load NN Model '''
        nn_model = tf.keras.models.load_model(NN_MODELS[i], compile=False)
        nn_model.compile(optimizer='adam', 
                    loss={'packet_state': 'categorical_crossentropy'},
                    metrics={'packet_state': 'accuracy'})
        nn_model_no_act = build_nn_model_v4_wobatchnorm_noactivation() # For testing calibrated NN
        nn_model_no_act.set_weights(nn_model.get_weights())
        ''' Load Test Dataset '''
        test_dataset_path = TEST_DATASET[i]
        test_data_df_1 = pd.read_csv(TEST_DATASET[i].format(1), dtype=data_dtypes)
        test_data_df_2 = pd.read_csv(TEST_DATASET[i].format(2), dtype=data_dtypes)
        test_data_df = pd.concat([test_data_df_1, test_data_df_2], ignore_index=True)
        test_data_df = test_data_df.loc[test_data_df["Horizontal_Distance"] <= MAX_HDIST]
        test_data_df = get_mcs_index(test_data_df)
        test_data_df["Reliability"] = (test_data_df["Num_Reliable"] / test_data_df["Num_Sent"]).values
        test_data_df["Reliable_State_90"] = test_data_df["Reliability"] >= 0.9
        test_data_df["Reliable_State_99"] = test_data_df["Reliability"] >= 0.99
        test_data_df["Reliable_State_999"] = test_data_df["Reliability"] >= 0.999
        
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
        T_SPECIFICITY_90 = NN_CALIBRATION_T_SPECIFICITY_90[i]
        T_SPECIFICITY_99 = NN_CALIBRATION_T_SPECIFICITY_99[i]
        T_SPECIFICITY_999 = NN_CALIBRATION_T_SPECIFICITY_999[i]
        nn_logits = nn_model_no_act.predict(test_data_df[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"]].values)
        test_data_df["CalibNN_Specificity_Predicted_Reliability_90"] = np.array([activations.softmax(K.constant([logits/T_SPECIFICITY_90]), axis=-1)[0][0].numpy() for logits in nn_logits])
        test_data_df["CalibNN_Specificity_Predicted_Reliability_99"] = np.array([activations.softmax(K.constant([logits/T_SPECIFICITY_99]), axis=-1)[0][0].numpy() for logits in nn_logits])
        test_data_df["CalibNN_Specificity_Predicted_Reliability_999"] = np.array([activations.softmax(K.constant([logits/T_SPECIFICITY_999]), axis=-1)[0][0].numpy() for logits in nn_logits])
        test_data_df["CalibNN_Specificity_Predicted_Reliable_State_90"] = test_data_df["CalibNN_Specificity_Predicted_Reliability_90"] >= 0.9
        test_data_df["CalibNN_Specificity_Predicted_Reliable_State_99"] = test_data_df["CalibNN_Specificity_Predicted_Reliability_99"] >= 0.99
        test_data_df["CalibNN_Specificity_Predicted_Reliable_State_999"] = test_data_df["CalibNN_Specificity_Predicted_Reliability_999"] >= 0.999
        
        ''' Save Results '''
        test_data_df.to_csv(SAVE_PATH[i], index=False)
        
        ''' Calulate metrics '''
        '''Reliabiliy State Specificity Score'''
        nn_reliability_state_specificity_90 = recall_score(test_data_df["Reliable_State_90"], test_data_df["NN_Predicted_Reliable_State_90"], pos_label=False, average='binary')
        nn_reliability_state_specificity_99 = recall_score(test_data_df["Reliable_State_99"], test_data_df["NN_Predicted_Reliable_State_99"], pos_label=False, average='binary')
        nn_reliability_state_specificity_999 = recall_score(test_data_df["Reliable_State_999"], test_data_df["NN_Predicted_Reliable_State_999"], pos_label=False, average='binary')
        calibnn_specificity_reliability_state_specificity_90 = recall_score(test_data_df["Reliable_State_90"], test_data_df["CalibNN_Specificity_Predicted_Reliable_State_90"], pos_label=False, average='binary')
        calibnn_specificity_reliability_state_specificity_99 = recall_score(test_data_df["Reliable_State_99"], test_data_df["CalibNN_Specificity_Predicted_Reliable_State_99"], pos_label=False, average='binary')
        calibnn_specificity_reliability_state_specificity_999 = recall_score(test_data_df["Reliable_State_999"], test_data_df["CalibNN_Specificity_Predicted_Reliable_State_999"], pos_label=False, average='binary')
        '''Reliabiliy State Recall (Sensitivity) Score'''
        nn_reliability_state_recall_90 = recall_score(test_data_df["Reliable_State_90"], test_data_df["NN_Predicted_Reliable_State_90"])
        nn_reliability_state_recall_99 = recall_score(test_data_df["Reliable_State_99"], test_data_df["NN_Predicted_Reliable_State_99"])
        nn_reliability_state_recall_999 = recall_score(test_data_df["Reliable_State_999"], test_data_df["NN_Predicted_Reliable_State_999"])
        calibnn_specificity_reliability_state_recall_90 = recall_score(test_data_df["Reliable_State_90"], test_data_df["CalibNN_Specificity_Predicted_Reliable_State_90"])
        calibnn_specificity_reliability_state_recall_99 = recall_score(test_data_df["Reliable_State_99"], test_data_df["CalibNN_Specificity_Predicted_Reliable_State_99"])
        calibnn_specificity_reliability_state_recall_999 = recall_score(test_data_df["Reliable_State_999"], test_data_df["CalibNN_Specificity_Predicted_Reliable_State_999"])
        
        ''' Record metrics to csv '''
        metrics_list.append({"Test_Dataset": os.path.join(test_dataset_path.split("/")[-3], test_dataset_path.split("/")[-2], test_dataset_path.split("/")[-1]), 
                             "NN_Model": NN_MODELS[i], "Cal_T_Specificity_90": T_SPECIFICITY_90, "Cal_T_Specificity_99": T_SPECIFICITY_99, "Cal_T_Specificity_999": T_SPECIFICITY_999, 
                             "nn_reliability_state_specificity_90": nn_reliability_state_specificity_90, "nn_reliability_state_specificity_99": nn_reliability_state_specificity_99, "nn_reliability_state_specificity_999": nn_reliability_state_specificity_999,
                             "calibnn_specificity_reliability_state_specificity_90": calibnn_specificity_reliability_state_specificity_90, "calibnn_specificity_reliability_state_specificity_99": calibnn_specificity_reliability_state_specificity_99, "calibnn_specificity_reliability_state_specificity_999": calibnn_specificity_reliability_state_specificity_999,
                             "nn_reliability_state_recall_90": nn_reliability_state_recall_90, "nn_reliability_state_recall_99": nn_reliability_state_recall_99, "nn_reliability_state_recall_999": nn_reliability_state_recall_999,
                             "calibnn_specificity_reliability_state_recall_90": calibnn_specificity_reliability_state_recall_90, "calibnn_specificity_reliability_state_recall_99": calibnn_specificity_reliability_state_recall_99, "calibnn_specificity_reliability_state_recall_999": calibnn_specificity_reliability_state_recall_999})
           
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(METRIC_SAVE_PATH)