# Modified from: fanet_throughput_threshold_test_18102024.py
# Date: 24/10/2024
# Desc: To test threshold for nr, the no. reliable packets per time window
# Modified: To test each sample of time window rather than time series from simulation runs
# NOTE: Filter out data points before 1 s

import pandas as pd
import numpy as np 
import os
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from multiprocessing.pool import Pool
from itertools import repeat

def load_data(scenario_path):
    scenario_name = scenario_path.split("/")[-1]
    params = scenario_name.split("_")
    usi = [x for x in params if "UAVSendingInterval" in x][0].split('-')[-1]
    # Load DL Throughput
    dl_df_list = []
    dl_throughput_files = glob.glob(os.path.join(scenario_path, "Run-*_Downlink_Throughput.csv"))
    for file in dl_throughput_files:
        measured_df = pd.read_csv(file)
        dl_df_list.append(measured_df)
    dl_df = pd.concat(dl_df_list)
    # Load UL Throughput
    ul_df_list = []
    ul_throughput_files = glob.glob(os.path.join(scenario_path, "Run-*_Uplink_Throughput.csv"))
    for file in ul_throughput_files:
        measured_df = pd.read_csv(file)
        ul_df_list.append(measured_df)
    ul_df = pd.concat(ul_df_list)
    ul_df["USI"] = usi
    # Load Video Throughput
    vid_df_list = []
    vid_throughput_files = glob.glob(os.path.join(scenario_path, "Run-*_Video_Throughput.csv"))
    for file in vid_throughput_files:
        measured_df = pd.read_csv(file)
        vid_df_list.append(measured_df)
    vid_df = pd.concat(vid_df_list)

    return (dl_df, ul_df, vid_df)

if __name__ == "__main__":
    # pandarallel.initialize(progress_bar=False)
    ''' Define Paths Here'''
    # DATASET_NO_INT_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_no_int_test_processed"
    DATASET_INT_PATHS = ["/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_0_processed", 
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_1_processed",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_2_processed",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_0_processed",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_1_processed",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_2_processed"]
    SAVE_PATH = "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/throughput_failure_detection_time_window_results.csv"
    RELIABILITY_TH = 0.999 # Threshold to evaluate interference scenarios
    NUM_PROCS = 32
    # Create dataframe of n_r thresholds for uplink
    ul_th = {"10": 390558, "20": 182199, "66.7": 47379, "100": 28421}
    dl_th = 1027
    vid_th = 237800
    
    ''' INT SCENARIOS '''
    ''' Filter out runs where reliability is above the threshold '''
    print("Testing Int Scenarios")
    results = []
    for datasets in DATASET_INT_PATHS:
        scenario_paths = [x[0] for x in os.walk(datasets) if (os.path.isdir(x[0]) and x[0]!=datasets)]
        scenario_name = datasets.split("/")[-1]
        print(scenario_name)
        dl_throughput_df_list = []
        ul_throughput_df_list = []
        vid_throughput_df_list = []
        with Pool(NUM_PROCS) as pool:
            # for result in pool.starmap(load_throughput, zip(scenario_paths, max_throughput_dl, max_throughput_ul, max_throughput_vid)):
            # for result in pool.starmap(load_throughput, zip(scenario_paths, min_throughput_dl, min_throughput_ul, min_throughput_vid)):
            for result in pool.starmap(load_data, zip(scenario_paths)):
                dl_throughput_df_list.append(result[0])
                ul_throughput_df_list.append(result[1])
                vid_throughput_df_list.append(result[2])

        dl_throughput_df = pd.concat(dl_throughput_df_list)
        ul_throughput_df = pd.concat(ul_throughput_df_list)
        vid_throughput_df = pd.concat(vid_throughput_df_list)
        
        # Remove NaNs
        dl_throughput_df.dropna(subset=["Measured_Reliability_1", "Throughput"], inplace=True)
        ul_throughput_df.dropna(subset=["Measured_Reliability_1", "Throughput"], inplace=True)
        vid_throughput_df.dropna(subset=["Measured_Reliability_1", "Throughput"], inplace=True)

        # Filter samples with "Time" < 1
        dl_throughput_df = dl_throughput_df.loc[dl_throughput_df["Time"] >= 1]
        ul_throughput_df = ul_throughput_df.loc[ul_throughput_df["Time"] >= 1]
        vid_throughput_df = vid_throughput_df.loc[vid_throughput_df["Time"] >= 1]

        # Get the ground truths for each time window. 0 for no failure and 1 for failure
        dl_throughput_df["Failure_Occur"] = dl_throughput_df["Measured_Reliability_1"] < RELIABILITY_TH
        ul_throughput_df["Failure_Occur"] = ul_throughput_df["Measured_Reliability_1"] < RELIABILITY_TH
        vid_throughput_df["Failure_Occur"] = vid_throughput_df["Measured_Reliability_1"] < RELIABILITY_TH

        # Get prediction for DL and Vid
        dl_throughput_df["Failure_Predict"] = dl_throughput_df["Throughput"] < dl_th
        vid_throughput_df["Failure_Predict"] = vid_throughput_df["Throughput"] < vid_th
        ul_temp_list = []
        for usi in ["10", "20", "66.7", "100"]:
            ul_df = ul_throughput_df.loc[ul_throughput_df["USI"]==usi]
            ul_df["Failure_Predict"] = ul_df["Throughput"] < ul_th[usi]
            ul_temp_list.append(ul_df)
        ul_throughput_df = pd.concat(ul_temp_list)

        dl_accuracy = accuracy_score(dl_throughput_df["Failure_Occur"].to_numpy(), dl_throughput_df["Failure_Predict"].to_numpy())
        ul_accuracy = accuracy_score(ul_throughput_df["Failure_Occur"].to_numpy(), ul_throughput_df["Failure_Predict"].to_numpy())
        vid_accuracy = accuracy_score(vid_throughput_df["Failure_Occur"].to_numpy(), vid_throughput_df["Failure_Predict"].to_numpy())

        dl_tn, dl_fp, dl_fn, dl_tp = confusion_matrix(dl_throughput_df["Failure_Occur"].to_numpy(), dl_throughput_df["Failure_Predict"].to_numpy(), labels=[0,1]).ravel()
        ul_tn, ul_fp, ul_fn, ul_tp = confusion_matrix(ul_throughput_df["Failure_Occur"].to_numpy(), ul_throughput_df["Failure_Predict"].to_numpy(), labels=[0,1]).ravel()
        vid_tn, vid_fp, vid_fn, vid_tp = confusion_matrix(vid_throughput_df["Failure_Occur"].to_numpy(), vid_throughput_df["Failure_Predict"].to_numpy(), labels=[0,1]).ravel()
        
        dl_sensitivity = dl_tp / (dl_tp + dl_fn)
        dl_specificity = dl_tn / (dl_tn + dl_fp)
        ul_sensitivity = ul_tp / (ul_tp + ul_fn)
        ul_specificity = ul_tn / (ul_tn + ul_fp)
        vid_sensitivity = vid_tp / (vid_tp + vid_fn)
        vid_specificity = vid_tn / (vid_tn + vid_fp)

        results.append({"Scenario": scenario_name, "DL_Accuracy": dl_accuracy, "DL_Sensitivity": dl_sensitivity, "DL_Specificity": dl_specificity, "DL_TN": dl_tn, "DL_FP": dl_fp, "DL_FN": dl_fn, "DL_TP": dl_tp,
                        "UL_Accuracy": ul_accuracy, "UL_Sensitivity": ul_sensitivity, "UL_Specificity": ul_specificity, "UL_TN": ul_tn, "UL_FP": ul_fp, "UL_FN": ul_fn, "UL_TP": ul_tp,
                        "VID_Accuracy": vid_accuracy, "VID_Sensitivity": vid_sensitivity, "VID_Specificity": vid_specificity, "VID_TN": vid_tn, "VID_FP": vid_fp, "VID_FN": vid_fn, "VID_TP": vid_tp})
        temp_df = pd.DataFrame(results)
        temp_df.to_csv(SAVE_PATH)
    scenario_results_df = pd.DataFrame(results)
    scenario_results_df.to_csv(SAVE_PATH)
        