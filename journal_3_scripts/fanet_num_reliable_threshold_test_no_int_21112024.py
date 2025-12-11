# Modified from: fanet_num_reliable_threshold_test_overall_01112024.py
# Date: 21/11/2024
# Desc: To test threshold for nr, in no interference scenario

import pandas as pd
import numpy as np 
import os
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from multiprocessing.pool import Pool
from itertools import repeat

def evaluate_gamma_failure_prediction(row, ul_n_r, uav_0_n_r, uav_1_n_r, uav_2_n_r, uav_3_n_r, uav_4_n_r, uav_5_n_r, uav_6_n_r, uav_7_n_r):
    usi = row["USI"]
    ul_fail = row["UL_Num_Reliable"] < ul_n_r[usi]
    dl_0_fail = row["UAV_0_Num_Reliable"] < uav_0_n_r[usi]
    dl_1_fail = row["UAV_1_Num_Reliable"] < uav_1_n_r[usi]
    dl_2_fail = row["UAV_2_Num_Reliable"] < uav_2_n_r[usi]
    dl_3_fail = row["UAV_3_Num_Reliable"] < uav_3_n_r[usi]
    dl_4_fail = row["UAV_4_Num_Reliable"] < uav_4_n_r[usi]
    dl_5_fail = row["UAV_5_Num_Reliable"] < uav_5_n_r[usi]
    dl_6_fail = row["UAV_6_Num_Reliable"] < uav_6_n_r[usi]
    dl_7_fail = row["UAV_7_Num_Reliable"] < uav_7_n_r[usi]
    overall_fail = ul_fail | dl_0_fail | dl_1_fail | dl_2_fail | dl_3_fail | dl_4_fail | dl_5_fail | dl_6_fail | dl_7_fail
    return (overall_fail)

def evaluate_throughput_failure_prediction(row, ul_th, uav_0_th, uav_1_th, uav_2_th, uav_3_th, uav_4_th, uav_5_th, uav_6_th, uav_7_th):
    usi = row["USI"]
    ul_fail = row["UL_Throughput"] < ul_th[usi]
    dl_0_fail = row["UAV_0_Throughput"] < uav_0_th[usi]
    dl_1_fail = row["UAV_1_Throughput"] < uav_1_th[usi]
    dl_2_fail = row["UAV_2_Throughput"] < uav_2_th[usi]
    dl_3_fail = row["UAV_3_Throughput"] < uav_3_th[usi]
    dl_4_fail = row["UAV_4_Throughput"] < uav_4_th[usi]
    dl_5_fail = row["UAV_5_Throughput"] < uav_5_th[usi]
    dl_6_fail = row["UAV_6_Throughput"] < uav_6_th[usi]
    dl_7_fail = row["UAV_7_Throughput"] < uav_7_th[usi]
    overall_fail = ul_fail | dl_0_fail | dl_1_fail | dl_2_fail | dl_3_fail | dl_4_fail | dl_5_fail | dl_6_fail | dl_7_fail
    return (overall_fail)

if __name__ == "__main__":

    ''' Define Paths Here'''
    DATASET_PATHS = "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_no_int_test_{}_processed.csv"
    SAVE_PATH = "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/results_new/num_reliable_failure_detection_no_int_results_500_runs.csv"
    RELIABILITY_TH = 0.999 # Threshold to evaluate interference scenarios
    NUM_PROCS = 32
    NUM_UAV = 8
    NUM_RUNS = 499 # The max run number to consider (Run_Num starts at 0)
    ul_n_r = {10: 752, 20: 376, 66.7: 114, 100: 78} # For 99.9% reliability level
    uav_0_n_r = {10: 48, 20: 48, 66.7: 48, 100: 48}
    uav_1_n_r = {10: 48, 20: 48, 66.7: 48, 100: 48}
    uav_2_n_r = {10: 48, 20: 48, 66.7: 48, 100: 48}
    uav_3_n_r = {10: 48, 20: 48, 66.7: 48, 100: 48}
    uav_4_n_r = {10: 48, 20: 48, 66.7: 48, 100: 48}
    uav_5_n_r = {10: 48, 20: 48, 66.7: 48, 100: 48}
    uav_6_n_r = {10: 48, 20: 48, 66.7: 48, 100: 48}
    uav_7_n_r = {10: 48, 20: 48, 66.7: 48, 100: 48}
    ul_th = {10: 395913, 20: 188049, 66.7: 45094, 100: 27200}
    uav_0_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    uav_1_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    uav_2_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    uav_3_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    uav_4_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    uav_5_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    uav_6_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    uav_7_th = {10: 996, 20: 978, 66.7: 978, 100: 978}

    ul_num_reliable_df = pd.read_csv(DATASET_PATHS.format("ul"))
    ul_num_reliable_df = ul_num_reliable_df.loc[ul_num_reliable_df["Run_Num"] <= NUM_RUNS]
    ul_num_reliable_df.sort_values(by=["UAV_Speed", "UAV_Height", "Bit_Rate", "USI", "Time"], inplace=True)
    len_df = len(ul_num_reliable_df)
    dl_num_reliable_df_list = []
    for i in range(NUM_UAV):
        df = pd.read_csv(DATASET_PATHS.format("UAV_" + str(i)))
        df = df.loc[df["Run_Num"] <= NUM_RUNS]
        df.sort_values(by=["UAV_Speed", "UAV_Height", "Bit_Rate", "USI", "Time"], inplace=True)
        assert len(df) == len_df, "UAV-{} DF Not Same Length".format(i)
        dl_num_reliable_df_list.append(df)

    # Combine data to one DF
    ul_num_reliable_df.rename(columns={"Measured_Reliability_1": "UL_Measured_Reliability_1", "Num_Reliable": "UL_Num_Reliable", "Throughput": "UL_Throughput"}, inplace=True)
    for i in range(NUM_UAV):
        ul_num_reliable_df["UAV_{}_Measured_Reliability_1".format(i)] = dl_num_reliable_df_list[i]["Measured_Reliability_1"]
        ul_num_reliable_df["UAV_{}_Num_Reliable".format(i)] = dl_num_reliable_df_list[i]["Num_Reliable"]
        ul_num_reliable_df["UAV_{}_Throughput".format(i)] = dl_num_reliable_df_list[i]["Throughput"]
    
    # Remove NaNs, but combine them first
    ul_num_reliable_df.dropna(subset=["UL_Measured_Reliability_1", "UAV_0_Measured_Reliability_1", "UAV_1_Measured_Reliability_1", 
                                        "UAV_2_Measured_Reliability_1", "UAV_3_Measured_Reliability_1", "UAV_4_Measured_Reliability_1",
                                        "UAV_5_Measured_Reliability_1", "UAV_6_Measured_Reliability_1", "UAV_7_Measured_Reliability_1",
                                        "UL_Num_Reliable", "UAV_0_Num_Reliable", "UAV_1_Num_Reliable", "UAV_2_Num_Reliable", "UAV_3_Num_Reliable",
                                        "UAV_4_Num_Reliable", "UAV_5_Num_Reliable", "UAV_6_Num_Reliable", "UAV_7_Num_Reliable",
                                        "UL_Throughput", "UAV_0_Throughput", "UAV_1_Throughput", "UAV_2_Throughput", "UAV_3_Throughput",
                                        "UAV_4_Throughput", "UAV_5_Throughput", "UAV_6_Throughput", "UAV_7_Throughput"], inplace=True)

    # Filter samples with "Time" < 1
    ul_num_reliable_df = ul_num_reliable_df.loc[ul_num_reliable_df["Time"] >= 1]

    # Ground truths of UL and DL
    ul_num_reliable_df["UL_Failure_Occur"] = ul_num_reliable_df["UL_Measured_Reliability_1"] < RELIABILITY_TH
    ul_num_reliable_df["DL_Failure_Occur"] = (ul_num_reliable_df["UAV_0_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_num_reliable_df["UAV_7_Measured_Reliability_1"] < RELIABILITY_TH) | \
                                                (ul_num_reliable_df["UAV_1_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_num_reliable_df["UAV_2_Measured_Reliability_1"] < RELIABILITY_TH) | \
                                                (ul_num_reliable_df["UAV_3_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_num_reliable_df["UAV_4_Measured_Reliability_1"] < RELIABILITY_TH) | \
                                                (ul_num_reliable_df["UAV_5_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_num_reliable_df["UAV_6_Measured_Reliability_1"] < RELIABILITY_TH)
    ul_num_reliable_df["Failure_Occur"] = ul_num_reliable_df["UL_Failure_Occur"] | ul_num_reliable_df["DL_Failure_Occur"]

    # Get only the data points where no failure occur
    reliable_df = ul_num_reliable_df.loc[ul_num_reliable_df["Failure_Occur"] == False]

    # Get predictions (for both gamma and tau)
    reliable_df["Gamma_Failure_Predict"] = reliable_df.apply(lambda row: evaluate_gamma_failure_prediction(row, ul_n_r, uav_0_n_r, uav_1_n_r, uav_2_n_r, uav_3_n_r, uav_4_n_r, uav_5_n_r, uav_6_n_r, uav_7_n_r), axis=1)
    reliable_df["Throughput_Failure_Predict"] = reliable_df.apply(lambda row: evaluate_throughput_failure_prediction(row, ul_th, uav_0_th, uav_1_th, uav_2_th, uav_3_th, uav_4_th, uav_5_th, uav_6_th, uav_7_th), axis=1)

    # All the ground truth is negative, so just calculate accuracy
    gamma_accuracy = len(reliable_df.loc[reliable_df["Gamma_Failure_Predict"]==False]) / len(reliable_df)
    throughput_accuracy = len(reliable_df.loc[reliable_df["Throughput_Failure_Predict"]==False]) / len(reliable_df)

    results = pd.DataFrame({"Gamma_Accuracy": [gamma_accuracy], "Throughput_Accuracy": [throughput_accuracy]})
    results.to_csv(SAVE_PATH)
    



