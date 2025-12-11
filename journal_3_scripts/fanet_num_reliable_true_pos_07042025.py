# Modified from: fanet_num_reliable_false_neg_16032025.py
# Date: 07/04/2025
# Desc: To get the measured reliability (in time windows) of true positive predictions for modified throughput 

import pandas as pd
import numpy as np 
import os
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_failure_prediction(row, ul_n_r, uav_0_n_r, uav_1_n_r, uav_2_n_r, uav_3_n_r, uav_4_n_r, uav_5_n_r, uav_6_n_r, uav_7_n_r):
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
    dl_fail = dl_0_fail | dl_1_fail | dl_2_fail | dl_3_fail | dl_4_fail | dl_5_fail | dl_6_fail | dl_7_fail
    return (overall_fail, ul_fail, dl_fail)

if __name__ == "__main__":

    """ EVALUATE THE OVERALL SENSITIVITY AND SPECIFICITY OF ALL CASES AND OF EACH POSSIBILITY OF POSITIVE LABEL SEPARATELY, AND COMBINE USING PREVALENCE
        Possibility 1: Groundtruth of both UL and DL is positive
        Possibility 2: Groundtruth of both UL is positive, but DL is negative
        Possibility 3: Groundtruth of both DL is positive, but UL is negative
        Ground truth: If any link fail, label it positive, else negative
        Prediction: If any gamma below threshold, label it positive, else negative"""
    ''' Define Paths Here'''
    DATASET_INT_PATHS = [
                        # "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_0_{}_processed.csv",
                        #  "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_1_{}_processed.csv",
                        #  "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_2_{}_processed.csv"]
                         "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_1_{}_processed.csv",
                         "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_a_{}_processed.csv",
                         "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_2_{}_processed.csv"]
    SAVE_PATH = "/home/research-student/reuben_ws/omnet-fanet/results_new_Mar25/gamma_failure_detection_tp_999_manet.csv"
    
    RELIABILITY_TH = 0.999 # Threshold to evaluate interference scenarios
    NUM_PROCS = 32
    NUM_UAV = 8
    NUM_RUNS_START = 0 # Start range of run number to consider (inclusive)
    NUM_RUNS_END = 499 # Start range of run number to consider (inclusive)

    # Create dataframe of n_r thresholds for uplink
    ul_n_r = {10: 752, 20: 376, 66.7: 114, 100: 78} # For 99.9% reliability level
    uav_0_n_r = {10: 48, 20: 48, 66.7: 48, 100: 48}
    uav_1_n_r = {10: 48, 20: 48, 66.7: 48, 100: 48}
    uav_2_n_r = {10: 48, 20: 48, 66.7: 48, 100: 48}
    uav_3_n_r = {10: 48, 20: 48, 66.7: 48, 100: 48}
    uav_4_n_r = {10: 48, 20: 48, 66.7: 48, 100: 48}
    uav_5_n_r = {10: 48, 20: 48, 66.7: 48, 100: 48}
    uav_6_n_r = {10: 48, 20: 48, 66.7: 48, 100: 48}
    uav_7_n_r = {10: 48, 20: 48, 66.7: 48, 100: 48}
    
    tp_run_df_list = []

    for datasets in DATASET_INT_PATHS:
        scenario_name = datasets.split("/")[-1]
        print(scenario_name)
        print("LOADING AND PROCESSING DATA")
        ul_num_reliable_df = pd.read_csv(datasets.format("ul"))
        # ul_num_reliable_df = ul_num_reliable_df.loc[ul_num_reliable_df["Run_Num"] <= NUM_RUNS]
        ul_num_reliable_df = ul_num_reliable_df.loc[(ul_num_reliable_df["Run_Num"] >= NUM_RUNS_START) & (ul_num_reliable_df["Run_Num"] <= NUM_RUNS_END)]
        ul_num_reliable_df.sort_values(by=["UAV_Speed", "UAV_Height", "Bit_Rate", "USI", "Time"], inplace=True)
        len_df = len(ul_num_reliable_df)
        dl_num_reliable_df_list = []
        for i in range(NUM_UAV):
            df = pd.read_csv(datasets.format("UAV_" + str(i)))
            # df = df.loc[df["Run_Num"] <= NUM_RUNS]
            df = df.loc[(df["Run_Num"] >= NUM_RUNS_START) & (df["Run_Num"] <= NUM_RUNS_END)]
            df.sort_values(by=["UAV_Speed", "UAV_Height", "Bit_Rate", "USI", "Run_Num", "Time"], inplace=True)
            assert len(df) == len_df, "UAV-{} DF Not Same Length".format(i)
            dl_num_reliable_df_list.append(df)

        # Combine data to one DF
        ul_num_reliable_df.rename(columns={"Measured_Reliability_1": "UL_Measured_Reliability_1", "Num_Reliable": "UL_Num_Reliable"}, inplace=True)
        for i in range(NUM_UAV):
            ul_num_reliable_df["UAV_{}_Measured_Reliability_1".format(i)] = dl_num_reliable_df_list[i]["Measured_Reliability_1"]
            ul_num_reliable_df["UAV_{}_Num_Reliable".format(i)] = dl_num_reliable_df_list[i]["Num_Reliable"]
        
        # Remove NaNs, but combine them first
        ul_num_reliable_df.dropna(subset=["UL_Measured_Reliability_1", "UAV_0_Measured_Reliability_1", "UAV_1_Measured_Reliability_1", 
                                          "UAV_2_Measured_Reliability_1", "UAV_3_Measured_Reliability_1", "UAV_4_Measured_Reliability_1",
                                          "UAV_5_Measured_Reliability_1", "UAV_6_Measured_Reliability_1", "UAV_7_Measured_Reliability_1",
                                          "UL_Num_Reliable", "UAV_0_Num_Reliable", "UAV_1_Num_Reliable", "UAV_2_Num_Reliable", "UAV_3_Num_Reliable",
                                          "UAV_4_Num_Reliable", "UAV_5_Num_Reliable", "UAV_6_Num_Reliable", "UAV_7_Num_Reliable"], inplace=True)

        # Filter samples with "Time" < 1
        ul_num_reliable_df = ul_num_reliable_df.loc[ul_num_reliable_df["Time"] >= 1]

        # Ground truths 
        ul_num_reliable_df["Failure_Occur"] = (ul_num_reliable_df["UAV_0_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_num_reliable_df["UAV_7_Measured_Reliability_1"] < RELIABILITY_TH) | \
                                                 (ul_num_reliable_df["UAV_1_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_num_reliable_df["UAV_2_Measured_Reliability_1"] < RELIABILITY_TH) | \
                                                 (ul_num_reliable_df["UAV_3_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_num_reliable_df["UAV_4_Measured_Reliability_1"] < RELIABILITY_TH) | \
                                                 (ul_num_reliable_df["UAV_5_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_num_reliable_df["UAV_6_Measured_Reliability_1"] < RELIABILITY_TH) | \
                                                 (ul_num_reliable_df["UL_Measured_Reliability_1"] < RELIABILITY_TH)
        # Get predictions
        ul_num_reliable_df[["Failure_Predict", "UL_Failure_Predict", "DL_Failure_Predict"]] = ul_num_reliable_df.apply(lambda row: evaluate_failure_prediction(row, ul_n_r, uav_0_n_r, uav_1_n_r, uav_2_n_r, uav_3_n_r, uav_4_n_r, uav_5_n_r, uav_6_n_r, uav_7_n_r), 
                                                                                                                       axis=1, result_type='expand')

        # Get reliability of time windows with true positive predictions for each run of each combination, and concatenate them together
        df_groups = ul_num_reliable_df.groupby(['USI', 'Bit_Rate', 'UAV_Height', 'UAV_Speed'])
        print("EVALUATING EACH GROUP")
        for group in tqdm(set(df_groups.groups.keys())):
            group_df = df_groups.get_group(group)
            run_groups = group_df.groupby(["Run_Num"])
            for run_group in set(run_groups.groups.keys()):
                run_df = run_groups.get_group(run_group)
                tp_df = run_df.loc[(run_df["Failure_Occur"]==True) & (run_df["Failure_Predict"]==True)].copy()
                tp_df.drop(columns=['Time', 'Horizontal_Distance', 'USI', 'Bit_Rate', 'UAV_Height', 'UAV_Speed'], inplace=True)
                tp_df["Scenario"] = scenario_name
                tp_df["Combination"] = [group for i in range(len(tp_df))]
                tp_run_df_list.append(tp_df)

        df = pd.concat(tp_run_df_list)
        df.to_csv(SAVE_PATH)
        