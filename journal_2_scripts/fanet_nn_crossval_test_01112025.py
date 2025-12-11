'''
Date: 01/11/2025
Desc: To test the NN models trained under k-fold cross-validation
'''

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras import activations
from keras import backend as K
import pandas as pd  
import numpy as np
import math, os
from sklearn.metrics import accuracy_score, recall_score

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
    """EVERYTHING SHOULD CORRESPOND TO EACH OTHER"""
    # List the DL models here
    DL_NN_MODELS = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/kfold_nn_ckpts/Downlink_kfold_0_final_model.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/kfold_nn_ckpts/Downlink_kfold_1_final_model.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/kfold_nn_ckpts/Downlink_kfold_2_final_model.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/kfold_nn_ckpts/Downlink_kfold_3_final_model.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/kfold_nn_ckpts/Downlink_kfold_4_final_model.h5"]
    # List the UL models here
    UL_NN_MODELS = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/kfold_nn_ckpts/Uplink_kfold_0_final_model.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/kfold_nn_ckpts/Uplink_kfold_1_final_model.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/kfold_nn_ckpts/Uplink_kfold_2_final_model.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/kfold_nn_ckpts/Uplink_kfold_3_final_model.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/kfold_nn_ckpts/Uplink_kfold_4_final_model.h5"]
    # List the Video models here
    VID_NN_MODELS = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/kfold_nn_ckpts/Video_kfold_0_final_model.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/kfold_nn_ckpts/Video_kfold_1_final_model.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/kfold_nn_ckpts/Video_kfold_2_final_model.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/kfold_nn_ckpts/Video_kfold_3_final_model.h5",
                  "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/kfold_nn_ckpts/Video_kfold_4_final_model.h5"]
    # List the test dataset files
    DL_TEST_DATASET = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed_kfold/Downlink_Reliability_kfold_0_test.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed_kfold/Downlink_Reliability_kfold_1_test.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed_kfold/Downlink_Reliability_kfold_2_test.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed_kfold/Downlink_Reliability_kfold_3_test.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed_kfold/Downlink_Reliability_kfold_4_test.csv"]
    UL_TEST_DATASET = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed_kfold/Uplink_Reliability_kfold_0_test.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed_kfold/Uplink_Reliability_kfold_1_test.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed_kfold/Uplink_Reliability_kfold_2_test.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed_kfold/Uplink_Reliability_kfold_3_test.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed_kfold/Uplink_Reliability_kfold_4_test.csv"]
    VID_TEST_DATASET = ["/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed_kfold/Video_Reliability_kfold_0_test.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed_kfold/Video_Reliability_kfold_1_test.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed_kfold/Video_Reliability_kfold_2_test.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed_kfold/Video_Reliability_kfold_3_test.csv",
                      "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed_kfold/Video_Reliability_kfold_4_test.csv"]
    # List the calibration T values
    DL_NN_CALIBRATION_T_SPECIFICITY_90 = [1.6649418412845718, 1.6649418412845718, 1.6891207381494733, 1.6891854770981523, 1.6922842022803322]
    DL_NN_CALIBRATION_T_SPECIFICITY_99 = [1.6121261942083767, 1.6985870846559141, 1.6588825659260915, 1.682358458005504, 1.3258693105457207]
    DL_NN_CALIBRATION_T_SPECIFICITY_999 = [1.6790710032753904, 1.6829414364787487, 1.6973626533906057, 1.6649418412845718, 1.2482444519222895]
    UL_NN_CALIBRATION_T_SPECIFICITY_90 = [1.6829414364787487, 1.6589716013107143, 1.6649418412845718, 1.6275752983414973, 1.2482444519222895]
    UL_NN_CALIBRATION_T_SPECIFICITY_99 = [1.6664948076335737, 1.6939211159657614, 1.6829414364787487, 1.6649418412845718, 1.6649418412845718]
    UL_NN_CALIBRATION_T_SPECIFICITY_999 = [1.6956183045187352, 1.3258693105457207, 1.6987442711349452, 1.6732402146214733, 1.6938868113494374]
    VID_NN_CALIBRATION_T_SPECIFICITY_90 = [1.3258693105457207, 1.6909051625382538, 1.6638249603605595, 1.6981509139782545, 1.6649418412845718]
    VID_NN_CALIBRATION_T_SPECIFICITY_99 = [1.3258693105457207, 1.6323833009205209, 1.6649418412845718, 1.685758257582238, 1.6829414364787487]
    VID_NN_CALIBRATION_T_SPECIFICITY_999 = [1.6829414364787487, 1.6829414364787487, 1.6905990379455382, 1.6649418412845718, 0.999208567795641]
    SAVE_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/k_fold_results.csv"
    MAX_HDIST = 500 # We trained for h_dist less than 700 m
    metrics_list = [] # To store specificity, sensitivity
    for i in range(len(DL_NN_MODELS)):
        ''' Load NN Model '''
        # DL
        dl_nn_model = tf.keras.models.load_model(DL_NN_MODELS[i], compile=False)
        dl_nn_model.compile(optimizer='adam', 
                    loss={'packet_state': 'categorical_crossentropy'},
                    metrics={'packet_state': 'accuracy'})
        dl_nn_model_no_act = build_nn_model_v4_wobatchnorm_noactivation() # For testing calibrated NN
        dl_nn_model_no_act.set_weights(dl_nn_model.get_weights())

        ul_nn_model = tf.keras.models.load_model(UL_NN_MODELS[i], compile=False)
        ul_nn_model.compile(optimizer='adam', 
                    loss={'packet_state': 'categorical_crossentropy'},
                    metrics={'packet_state': 'accuracy'})
        ul_nn_model_no_act = build_nn_model_v4_wobatchnorm_noactivation() # For testing calibrated NN
        ul_nn_model_no_act.set_weights(ul_nn_model.get_weights())

        vid_nn_model = tf.keras.models.load_model(VID_NN_MODELS[i], compile=False)
        vid_nn_model.compile(optimizer='adam', 
                    loss={'packet_state': 'categorical_crossentropy'},
                    metrics={'packet_state': 'accuracy'})
        vid_nn_model_no_act = build_nn_model_v4_wobatchnorm_noactivation() # For testing calibrated NN
        vid_nn_model_no_act.set_weights(vid_nn_model.get_weights())

        """ Load Test Dataset """
        df_dtypes = {"Horizontal_Distance": np.int32, "Height": np.int16, "UAV_Sending_Interval": np.float64, "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
                    "Num_Sent": np.int32, "Num_Reliable": np.int32, "Num_Delay_Excd": np.int32, "Num_Incr_Rcvd": np.int32, "Num_Q_Overflow": np.int32, "Modulation": str, "Bitrate": np.float64}
        dl_df_test = pd.read_csv(DL_TEST_DATASET[i], 
                                usecols = ["Horizontal_Distance", "Height", "Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Bitrate", 
                                            "Num_Sent", "Num_Reliable", "Num_Delay_Excd", "Num_Incr_Rcvd", "Num_Q_Overflow"],
                                dtype=df_dtypes)
        dl_df_test = get_mcs_index(dl_df_test)
        # dl_df_test["UAV_Sending_Interval"] = dl_df_test["UAV_Sending_Interval"].astype(np.float16)
        ul_df_test = pd.read_csv(UL_TEST_DATASET[i], 
                                usecols = ["Horizontal_Distance", "Height", "Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Bitrate", 
                                            "Num_Sent", "Num_Reliable", "Num_Delay_Excd", "Num_Incr_Rcvd", "Num_Q_Overflow"],
                                dtype=df_dtypes)
        ul_df_test = get_mcs_index(ul_df_test)
        # ul_df_test["UAV_Sending_Interval"] = ul_df_test["UAV_Sending_Interval"].astype(np.float16)
        vid_df_test = pd.read_csv(VID_TEST_DATASET[i], 
                                usecols = ["Horizontal_Distance", "Height", "Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Bitrate", 
                                            "Num_Sent", "Num_Reliable", "Num_Delay_Excd", "Num_Incr_Rcvd", "Num_Q_Overflow"],
                                dtype=df_dtypes)
        vid_df_test = get_mcs_index(vid_df_test)
        # vid_df_test["UAV_Sending_Interval"] = vid_df_test["UAV_Sending_Interval"].astype(np.float16)
        dl_df_test["Reliability"] = dl_df_test["Num_Reliable"] / dl_df_test["Num_Sent"]
        ul_df_test["Reliability"] = ul_df_test["Num_Reliable"] / ul_df_test["Num_Sent"]
        vid_df_test["Reliability"] = vid_df_test["Num_Reliable"] / vid_df_test["Num_Sent"]
        df_test = dl_df_test.merge(ul_df_test, on=["UAV_Sending_Interval", "MCS", "Height", "Horizontal_Distance"], suffixes=('_dl', '_ul'))
        df_test = df_test.merge(vid_df_test, on=["UAV_Sending_Interval", "MCS", "Height", "Horizontal_Distance"])
        df_test = df_test.rename(columns={"Reliability": "Reliability_vid"})
        # Get reliability state: 1 if reliability of all links >= rel_th else 0
        df_test["Reliable_State_90"] = df_test.apply(lambda row: "Positive" if (row["Reliability_dl"] < 0.9 or row["Reliability_ul"] < 0.9 or row["Reliability_vid"] < 0.9) else "Negative", axis=1)
        df_test["Reliable_State_99"] = df_test.apply(lambda row: "Positive" if (row["Reliability_dl"] < 0.99 or row["Reliability_ul"] < 0.99 or row["Reliability_vid"] < 0.99) else "Negative", axis=1)
        df_test["Reliable_State_999"] = df_test.apply(lambda row: "Positive" if (row["Reliability_dl"] < 0.999 or row["Reliability_ul"] < 0.999 or row["Reliability_vid"] < 0.999) else "Negative", axis=1)

        ''' Test NN Model '''
        df_test = normalize_data(df_test, columns=["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"], save_details_path=None)
        dl_nn_prediction = dl_nn_model.predict(df_test[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"]].values)
        dl_nn_reliability_prediction = [prob[0] for prob in dl_nn_prediction]
        ul_nn_prediction = ul_nn_model.predict(df_test[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"]].values)
        ul_nn_reliability_prediction = [prob[0] for prob in ul_nn_prediction]
        vid_nn_prediction = vid_nn_model.predict(df_test[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"]].values)
        vid_nn_reliability_prediction = [prob[0] for prob in vid_nn_prediction]
        # Assign new column Predicted_Reliable_State_90 based on whether the values in dl_nn_reliability_prediction, ul_nn_reliability_prediction, vid_nn_reliability_prediction are all >= 0.9
        df_test["NN_Predicted_Reliable_State_90"] = ["Positive" if (dl_nn_reliability_prediction[j] < 0.9 or ul_nn_reliability_prediction[j] < 0.9 or vid_nn_reliability_prediction[j] < 0.9) else "Negative" for j in range(len(dl_nn_reliability_prediction))]
        df_test["NN_Predicted_Reliable_State_99"] = ["Positive" if (dl_nn_reliability_prediction[j] < 0.99 or ul_nn_reliability_prediction[j] < 0.99 or vid_nn_reliability_prediction[j] < 0.99) else "Negative" for j in range(len(dl_nn_reliability_prediction))]
        df_test["NN_Predicted_Reliable_State_999"] = ["Positive" if (dl_nn_reliability_prediction[j] < 0.999 or ul_nn_reliability_prediction[j] < 0.999 or vid_nn_reliability_prediction[j] < 0.999) else "Negative" for j in range(len(dl_nn_reliability_prediction))]

        ''' Test Calibrated NN Model '''
        dl_nn_logits = dl_nn_model_no_act.predict(df_test[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"]].values)
        dl_cal_nn_prediction_90 = np.array([activations.softmax(K.constant([logits/DL_NN_CALIBRATION_T_SPECIFICITY_90[i]]), axis=-1)[0][0].numpy() for logits in dl_nn_logits])
        dl_cal_nn_prediction_99 = np.array([activations.softmax(K.constant([logits/DL_NN_CALIBRATION_T_SPECIFICITY_99[i]]), axis=-1)[0][0].numpy() for logits in dl_nn_logits])
        dl_cal_nn_prediction_999 = np.array([activations.softmax(K.constant([logits/DL_NN_CALIBRATION_T_SPECIFICITY_999[i]]), axis=-1)[0][0].numpy() for logits in dl_nn_logits])
        ul_nn_logits = ul_nn_model_no_act.predict(df_test[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"]].values)
        ul_cal_nn_prediction_90 = np.array([activations.softmax(K.constant([logits/UL_NN_CALIBRATION_T_SPECIFICITY_90[i]]), axis=-1)[0][0].numpy() for logits in ul_nn_logits])
        ul_cal_nn_prediction_99 = np.array([activations.softmax(K.constant([logits/UL_NN_CALIBRATION_T_SPECIFICITY_99[i]]), axis=-1)[0][0].numpy() for logits in ul_nn_logits])
        ul_cal_nn_prediction_999 = np.array([activations.softmax(K.constant([logits/UL_NN_CALIBRATION_T_SPECIFICITY_999[i]]), axis=-1)[0][0].numpy() for logits in ul_nn_logits])
        vid_nn_logits = vid_nn_model_no_act.predict(df_test[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"]].values)
        vid_cal_nn_prediction_90 = np.array([activations.softmax(K.constant([logits/VID_NN_CALIBRATION_T_SPECIFICITY_90[i]]), axis=-1)[0][0].numpy() for logits in vid_nn_logits])
        vid_cal_nn_prediction_99 = np.array([activations.softmax(K.constant([logits/VID_NN_CALIBRATION_T_SPECIFICITY_99[i]]), axis=-1)[0][0].numpy() for logits in vid_nn_logits])
        vid_cal_nn_prediction_999 = np.array([activations.softmax(K.constant([logits/VID_NN_CALIBRATION_T_SPECIFICITY_999[i]]), axis=-1)[0][0].numpy() for logits in vid_nn_logits])
        df_test["Cal_NN_Predicted_Reliable_State_90"] = ["Positive" if (dl_cal_nn_prediction_90[j] < 0.9 or ul_cal_nn_prediction_90[j] < 0.9 or vid_cal_nn_prediction_90[j] < 0.9) else "Negative" for j in range(len(dl_cal_nn_prediction_90))]
        df_test["Cal_NN_Predicted_Reliable_State_99"] = ["Positive" if (dl_cal_nn_prediction_99[j] < 0.99 or ul_cal_nn_prediction_99[j] < 0.99 or vid_cal_nn_prediction_99[j] < 0.99) else "Negative" for j in range(len(dl_cal_nn_prediction_99))]
        df_test["Cal_NN_Predicted_Reliable_State_999"] = ["Positive" if (dl_cal_nn_prediction_999[j] < 0.999 or ul_cal_nn_prediction_999[j] < 0.999 or vid_cal_nn_prediction_999[j] < 0.999) else "Negative" for j in range(len(dl_cal_nn_prediction_999))]

        ''' Calulate metrics '''
        '''Reliabiliy State Specificity Score'''
        nn_reliability_state_specificity_90 = recall_score(df_test["Reliable_State_90"], df_test["NN_Predicted_Reliable_State_90"], pos_label="Negative", average='binary')
        nn_reliability_state_specificity_99 = recall_score(df_test["Reliable_State_99"], df_test["NN_Predicted_Reliable_State_99"], pos_label="Negative", average='binary')
        nn_reliability_state_specificity_999 = recall_score(df_test["Reliable_State_999"], df_test["NN_Predicted_Reliable_State_999"], pos_label="Negative", average='binary')
        cal_nn_reliability_state_specificity_90 = recall_score(df_test["Reliable_State_90"], df_test["Cal_NN_Predicted_Reliable_State_90"], pos_label="Negative", average='binary')
        cal_nn_reliability_state_specificity_99 = recall_score(df_test["Reliable_State_99"], df_test["Cal_NN_Predicted_Reliable_State_99"], pos_label="Negative", average='binary')
        cal_nn_reliability_state_specificity_999 = recall_score(df_test["Reliable_State_999"], df_test["Cal_NN_Predicted_Reliable_State_999"], pos_label="Negative", average='binary')
        '''Reliabiliy State Recall (Sensitivity) Score'''
        nn_reliability_state_recall_90 = recall_score(df_test["Reliable_State_90"], df_test["NN_Predicted_Reliable_State_90"], pos_label="Positive", average='binary')
        nn_reliability_state_recall_99 = recall_score(df_test["Reliable_State_99"], df_test["NN_Predicted_Reliable_State_99"], pos_label="Positive", average='binary')
        nn_reliability_state_recall_999 = recall_score(df_test["Reliable_State_999"], df_test["NN_Predicted_Reliable_State_999"], pos_label="Positive", average='binary')
        cal_nn_reliability_state_recall_90 = recall_score(df_test["Reliable_State_90"], df_test["Cal_NN_Predicted_Reliable_State_90"], pos_label="Positive", average='binary')
        cal_nn_reliability_state_recall_99 = recall_score(df_test["Reliable_State_99"], df_test["Cal_NN_Predicted_Reliable_State_99"], pos_label="Positive", average='binary')
        cal_nn_reliability_state_recall_999 = recall_score(df_test["Reliable_State_999"], df_test["Cal_NN_Predicted_Reliable_State_999"], pos_label="Positive", average='binary')

        ''' Record metrics to csv '''
        metrics_list.append({"Test_Dataset": os.path.join(DL_TEST_DATASET[0].split("/")[-3], DL_TEST_DATASET[0].split("/")[-2]), 
                             "nn_reliability_state_specificity_90": nn_reliability_state_specificity_90, "nn_reliability_state_specificity_99": nn_reliability_state_specificity_99, "nn_reliability_state_specificity_999": nn_reliability_state_specificity_999,
                             "cal_nn_reliability_state_specificity_90": cal_nn_reliability_state_specificity_90, "cal_nn_reliability_state_specificity_99": cal_nn_reliability_state_specificity_99, "cal_nn_reliability_state_specificity_999": cal_nn_reliability_state_specificity_999,
                             "nn_reliability_state_recall_90": nn_reliability_state_recall_90, "nn_reliability_state_recall_99": nn_reliability_state_recall_99, "nn_reliability_state_recall_999": nn_reliability_state_recall_999,
                             "cal_nn_reliability_state_recall_90": cal_nn_reliability_state_recall_90, "cal_nn_reliability_state_recall_99": cal_nn_reliability_state_recall_99, "cal_nn_reliability_state_recall_999": cal_nn_reliability_state_recall_999})
           
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(SAVE_PATH, index=False)