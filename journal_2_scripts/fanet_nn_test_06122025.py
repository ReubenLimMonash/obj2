'''
Date: 06/12/2025
Desc: NN and calibrated NN prediction testing using complete test dataset.
      This script determines Reliability state based on all three links (Uplink, Downlink, Video).
      Predictions from all three NN models are needed.
      Overall system reliability is considered reliable only if ALL three links are reliable.
      Positive label is when one of the link reliability is below the required level, negative label when all link reliabilities above required level
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
    df.loc[(df["Bitrate"] == 6.5), "MCS"] = 0 # MCS Index 0
    df.loc[(df["Bitrate"] == 13.0), "MCS"] = 1 # MCS Index 0
    df.loc[(df["Bitrate"] == 19.5), "MCS"] = 2 # MCS Index 0
    df.loc[(df["Bitrate"] == 26.0), "MCS"] = 3 # MCS Index 0
    df.loc[(df["Bitrate"] == 39.0), "MCS"] = 4 # MCS Index 0
    df.loc[(df["Bitrate"] == 52.0), "MCS"] = 5 # MCS Index 0
    df.loc[(df["Bitrate"] == 58.5), "MCS"] = 6 # MCS Index 0
    df.loc[(df["Bitrate"] == 65.0), "MCS"] = 7 # MCS Index 0
    df["MCS"] = df["MCS"].astype(int)
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
    NN_CALIBRATION_T_SPECIFICITY_90 = [1.6933716833921753, # Downlink
                                       1.6649418412845718, # Uplink
                                       1.6271717659783327] # Video
    NN_CALIBRATION_T_SPECIFICITY_99 = [1.69561830451873, # Downlink
                                       1.67551651082116, # Uplink
                                       1.62717176597833] # Video
    NN_CALIBRATION_T_SPECIFICITY_999 = [1.69615238212838, # Downlink
                                        1.68516226329027, # Uplink
                                        1.62717176597833] # Video
    
    # Test datasets from Cell 3 of fanet_nn_test_08102025.ipynb
    TEST_DATASETS_GT = ["/media/research-student/DataDrive/FANET_Dataset/Dataset_NP100000_DJISpark/complete_testing_dmax_dataset/data_processed_complete/Downlink_Reliability.csv",
                        "/media/research-student/DataDrive/FANET_Dataset/Dataset_NP100000_DJISpark/complete_testing_dmax_dataset/data_processed_complete/Uplink_Reliability.csv",
                        "/media/research-student/DataDrive/FANET_Dataset/Dataset_NP100000_DJISpark/complete_testing_dmax_dataset/data_processed_complete/Video_Reliability.csv"]
    
    SAVE_PATH_COMBINED = "fanet_nn_predictions_results_06122025.csv"
    
    data_dtypes = {"Horizontal_Distance": np.float64, "Height":np.float64, "UAV_Sending_Interval": np.float64, "Modulation": 'str', "Bitrate": np.float64}

    test_data_dfs = []
    link_names = ["Downlink", "Uplink", "Video"]
    
    for i, dataset_path in enumerate(TEST_DATASETS_GT):
        print(f"Loading {link_names[i]} data...")
        df = pd.read_csv(dataset_path)
        df["Reliability"] = df["Num_Reliable"] / (df["Num_Delay_Excd"] + df["Num_Fail_Other"] + df["Num_Reliable"])
        df["Reliable_State_90"] = df["Reliability"] < 0.9
        df["Reliable_State_99"] = df["Reliability"] < 0.99
        df["Reliable_State_999"] = df["Reliability"] < 0.999
        df = get_mcs_index(df)
        test_data_dfs.append(df)
    
    # Load NN models
    print("Loading NN models...")
    nn_models = []
    nn_models_no_act = []
    for i, model_path in enumerate(NN_MODELS):
        print(f"Loading {link_names[i]} NN model...")
        nn_model = tf.keras.models.load_model(model_path, compile=False)
        nn_model.compile(optimizer='adam', 
                    loss={'packet_state': 'categorical_crossentropy'},
                    metrics={'packet_state': 'accuracy'})
        nn_models.append(nn_model)
        
        nn_model_no_act = build_nn_model_v4_wobatchnorm_noactivation()
        nn_model_no_act.set_weights(nn_model.get_weights())
        nn_models_no_act.append(nn_model_no_act)
    
    # Make predictions on all three links
    print("Making predictions...")
    predictions_list = []
    
    for i in range(len(NN_MODELS)):
        print(f"Predicting {link_names[i]}...")
        test_data_df = test_data_dfs[i].copy()
        nn_model = nn_models[i]
        nn_model_no_act = nn_models_no_act[i]
        
        # Normalize test data
        test_data_df = normalize_data(test_data_df, columns=["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"], save_details_path=None)
        
        # Get predictions from NN model
        nn_prediction = nn_model.predict(test_data_df[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"]].values)
        test_data_df[f'{link_names[i]}_NN_Predicted_Reliability'] = [prob[0] for prob in nn_prediction]
        test_data_df[f'{link_names[i]}_NN_Predicted_Reliable_State_90'] = test_data_df[f'{link_names[i]}_NN_Predicted_Reliability'] < 0.9
        test_data_df[f'{link_names[i]}_NN_Predicted_Reliable_State_99'] = test_data_df[f'{link_names[i]}_NN_Predicted_Reliability'] < 0.99
        test_data_df[f'{link_names[i]}_NN_Predicted_Reliable_State_999'] = test_data_df[f'{link_names[i]}_NN_Predicted_Reliability'] < 0.999
        
        # Get calibrated predictions
        T_SPECIFICITY_90 = NN_CALIBRATION_T_SPECIFICITY_90[i]
        T_SPECIFICITY_99 = NN_CALIBRATION_T_SPECIFICITY_99[i]
        T_SPECIFICITY_999 = NN_CALIBRATION_T_SPECIFICITY_999[i]
        
        nn_logits = nn_model_no_act.predict(test_data_df[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"]].values)
        test_data_df[f'{link_names[i]}_CalibNN_Specificity_Predicted_Reliability_90'] = np.array([activations.softmax(K.constant([logits/T_SPECIFICITY_90]), axis=-1)[0][0].numpy() for logits in nn_logits])
        test_data_df[f'{link_names[i]}_CalibNN_Specificity_Predicted_Reliability_99'] = np.array([activations.softmax(K.constant([logits/T_SPECIFICITY_99]), axis=-1)[0][0].numpy() for logits in nn_logits])
        test_data_df[f'{link_names[i]}_CalibNN_Specificity_Predicted_Reliability_999'] = np.array([activations.softmax(K.constant([logits/T_SPECIFICITY_999]), axis=-1)[0][0].numpy() for logits in nn_logits])
        test_data_df[f'{link_names[i]}_CalibNN_Specificity_Predicted_Reliable_State_90'] = test_data_df[f'{link_names[i]}_CalibNN_Specificity_Predicted_Reliability_90'] < 0.9
        test_data_df[f'{link_names[i]}_CalibNN_Specificity_Predicted_Reliable_State_99'] = test_data_df[f'{link_names[i]}_CalibNN_Specificity_Predicted_Reliability_99'] < 0.99
        test_data_df[f'{link_names[i]}_CalibNN_Specificity_Predicted_Reliable_State_999'] = test_data_df[f'{link_names[i]}_CalibNN_Specificity_Predicted_Reliability_999'] < 0.999
        
        predictions_list.append(test_data_df)
    
    # Combine predictions - system is reliable only if ALL three links are reliable
    print("Combining predictions from all three links...")
    combined_df = pd.DataFrame()

    # Include the Horizontal_Distance, Height, UAV_Sending_Interval, Bitrate columns from one of the dataframes
    combined_df["Horizontal_Distance"] = predictions_list[0]["Horizontal_Distance"].values
    combined_df["Height"] = predictions_list[0]["Height"].values
    combined_df["UAV_Sending_Interval"] = predictions_list[0]["UAV_Sending_Interval"].values
    combined_df["Bitrate"] = predictions_list[0]["Bitrate"].values
    
    # Add ground truth data - system is reliable if all three links are reliable
    combined_df['Gt_Reliable_State_90'] = predictions_list[0]["Reliable_State_90"].values | predictions_list[1]["Reliable_State_90"].values | predictions_list[2]["Reliable_State_90"].values
    combined_df['Gt_Reliable_State_99'] = predictions_list[0]["Reliable_State_99"].values | predictions_list[1]["Reliable_State_99"].values | predictions_list[2]["Reliable_State_99"].values
    combined_df['Gt_Reliable_State_999'] = predictions_list[0]["Reliable_State_999"].values | predictions_list[1]["Reliable_State_999"].values | predictions_list[2]["Reliable_State_999"].values
    
    # Add NN predictions - system is reliable if all three links are predicted as reliable
    combined_df['NN_Predicted_Reliable_State_90'] = predictions_list[0][f'Downlink_NN_Predicted_Reliable_State_90'].values | predictions_list[1][f'Uplink_NN_Predicted_Reliable_State_90'].values | predictions_list[2][f'Video_NN_Predicted_Reliable_State_90'].values
    combined_df['NN_Predicted_Reliable_State_99'] = predictions_list[0][f'Downlink_NN_Predicted_Reliable_State_99'].values | predictions_list[1][f'Uplink_NN_Predicted_Reliable_State_99'].values | predictions_list[2][f'Video_NN_Predicted_Reliable_State_99'].values
    combined_df['NN_Predicted_Reliable_State_999'] = predictions_list[0][f'Downlink_NN_Predicted_Reliable_State_999'].values | predictions_list[1][f'Uplink_NN_Predicted_Reliable_State_999'].values | predictions_list[2][f'Video_NN_Predicted_Reliable_State_999'].values
    
    # Add calibrated NN predictions
    combined_df['CalibNN_Specificity_Predicted_Reliable_State_90'] = predictions_list[0][f'Downlink_CalibNN_Specificity_Predicted_Reliable_State_90'].values | predictions_list[1][f'Uplink_CalibNN_Specificity_Predicted_Reliable_State_90'].values | predictions_list[2][f'Video_CalibNN_Specificity_Predicted_Reliable_State_90'].values
    combined_df['CalibNN_Specificity_Predicted_Reliable_State_99'] = predictions_list[0][f'Downlink_CalibNN_Specificity_Predicted_Reliable_State_99'].values | predictions_list[1][f'Uplink_CalibNN_Specificity_Predicted_Reliable_State_99'].values | predictions_list[2][f'Video_CalibNN_Specificity_Predicted_Reliable_State_99'].values
    combined_df['CalibNN_Specificity_Predicted_Reliable_State_999'] = predictions_list[0][f'Downlink_CalibNN_Specificity_Predicted_Reliable_State_999'].values | predictions_list[1][f'Uplink_CalibNN_Specificity_Predicted_Reliable_State_999'].values | predictions_list[2][f'Video_CalibNN_Specificity_Predicted_Reliable_State_999'].values
    
    # Add individual link predictions for reference
    for i, link_name in enumerate(link_names):
        combined_df[f'{link_name}_Gt_Reliable_State_90'] = predictions_list[i]["Reliable_State_90"].values
        combined_df[f'{link_name}_Gt_Reliable_State_99'] = predictions_list[i]["Reliable_State_99"].values
        combined_df[f'{link_name}_Gt_Reliable_State_999'] = predictions_list[i]["Reliable_State_999"].values
        combined_df[f'{link_name}_NN_Predicted_Reliability'] = predictions_list[i][f'{link_name}_NN_Predicted_Reliability'].values
        combined_df[f'{link_name}_NN_Predicted_Reliable_State_90'] = predictions_list[i][f'{link_name}_NN_Predicted_Reliable_State_90'].values
        combined_df[f'{link_name}_NN_Predicted_Reliable_State_99'] = predictions_list[i][f'{link_name}_NN_Predicted_Reliable_State_99'].values
        combined_df[f'{link_name}_NN_Predicted_Reliable_State_999'] = predictions_list[i][f'{link_name}_NN_Predicted_Reliable_State_999'].values
        combined_df[f'{link_name}_CalibNN_Specificity_Predicted_Reliability_90'] = predictions_list[i][f'{link_name}_CalibNN_Specificity_Predicted_Reliability_90'].values
        combined_df[f'{link_name}_CalibNN_Specificity_Predicted_Reliability_99'] = predictions_list[i][f'{link_name}_CalibNN_Specificity_Predicted_Reliability_99'].values
        combined_df[f'{link_name}_CalibNN_Specificity_Predicted_Reliability_999'] = predictions_list[i][f'{link_name}_CalibNN_Specificity_Predicted_Reliability_999'].values
        combined_df[f'{link_name}_CalibNN_Specificity_Predicted_Reliable_State_90'] = predictions_list[i][f'{link_name}_CalibNN_Specificity_Predicted_Reliable_State_90'].values
        combined_df[f'{link_name}_CalibNN_Specificity_Predicted_Reliable_State_99'] = predictions_list[i][f'{link_name}_CalibNN_Specificity_Predicted_Reliable_State_99'].values
        combined_df[f'{link_name}_CalibNN_Specificity_Predicted_Reliable_State_999'] = predictions_list[i][f'{link_name}_CalibNN_Specificity_Predicted_Reliable_State_999'].values
    
    # Save combined predictions
    print(f"Saving combined predictions to {SAVE_PATH_COMBINED}...")
    combined_df.to_csv(SAVE_PATH_COMBINED, index=False)
