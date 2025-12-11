# Modified from: fanet_measured_throughput_threshold_test_overall_03032025.py
# Date: 03/03/2025
# Desc: To test threshold for nr, the no. reliable packets per time window
# Modified: To test each sample of time window rather than time series from simulation runs
# NOTE: Filter out data points before 1 s
# Modified: Going back to analysing each run as a time series, and getting the sensitivity and specificity of each run, because Prof Lan did it again.
# Modified: Save the result and detail of each run, instead of the mean result of all runs per combo. Process the mean result later.
# Modified: Listing the number of negative / positive samples involved in the calculation of each metric, so that we can filter by no. samples.

import pandas as pd
import numpy as np 
import os
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from multiprocessing.pool import Pool
from itertools import repeat

def evaluate_failure_prediction(row, ul_th, uav_0_th, uav_1_th, uav_2_th, uav_3_th, uav_4_th, uav_5_th, uav_6_th, uav_7_th):
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
    dl_fail = dl_0_fail | dl_1_fail | dl_2_fail | dl_3_fail | dl_4_fail | dl_5_fail | dl_6_fail | dl_7_fail
    return (overall_fail, ul_fail, dl_fail)

def evaluate_failure_prediction_usi_mcs(row, tau_th_df):
    '''
    tau_th_df is the DataFrame of gamma threshold values for each link, for different combinations of USI and MCS
    '''
    usi = row["USI"]
    bitrate = row["Bit_Rate"]
    tau_th = tau_th_df.loc[(tau_th_df["USI"]==usi) & (tau_th_df["BitRate"]==bitrate)]
    ul_fail = row["UL_Throughput"] < tau_th["UL_Min_Tau"].values[0]
    dl_0_fail = row["UAV_0_Throughput"] < tau_th["UAV_0_Min_Tau"].values[0]
    dl_1_fail = row["UAV_1_Throughput"] < tau_th["UAV_1_Min_Tau"].values[0]
    dl_2_fail = row["UAV_2_Throughput"] < tau_th["UAV_2_Min_Tau"].values[0]
    dl_3_fail = row["UAV_3_Throughput"] < tau_th["UAV_3_Min_Tau"].values[0]
    dl_4_fail = row["UAV_4_Throughput"] < tau_th["UAV_4_Min_Tau"].values[0]
    dl_5_fail = row["UAV_5_Throughput"] < tau_th["UAV_5_Min_Tau"].values[0]
    dl_6_fail = row["UAV_6_Throughput"] < tau_th["UAV_6_Min_Tau"].values[0]
    dl_7_fail = row["UAV_7_Throughput"] < tau_th["UAV_7_Min_Tau"].values[0]
    overall_fail = ul_fail | dl_0_fail | dl_1_fail | dl_2_fail | dl_3_fail | dl_4_fail | dl_5_fail | dl_6_fail | dl_7_fail
    dl_fail = dl_0_fail | dl_1_fail | dl_2_fail | dl_3_fail | dl_4_fail | dl_5_fail | dl_6_fail | dl_7_fail
    return (overall_fail, ul_fail, dl_fail)

if __name__ == "__main__":

    """ EVALUATE THE OVERALL SENSITIVITY AND SPECIFICITY OF ALL CASES AND OF EACH POSSIBILITY OF POSITIVE LABEL SEPARATELY, AND COMBINE USING PREVALENCE
        Possibility 1: Groundtruth of both UL and DL is positive
        Possibility 2: Groundtruth of both UL is positive, but DL is negative
        Possibility 3: Groundtruth of both DL is positive, but UL is negative
        Ground truth: If any link fail, label it positive, else negative
        Prediction: If any tau below threshold, label it positive, else negative"""
    ''' Define Paths Here'''
    DATASET_INT_PATHS = ["/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_1_{}_processed.csv",
                         "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_0_{}_processed.csv", 
                         "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_1_{}_processed.csv",
                         "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_2_{}_processed.csv",
                         "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_a_{}_processed.csv",
                         "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_2_{}_processed.csv"]
    SAVE_PATH = "/home/research-student/reuben_ws/omnet-fanet/results_new_Mar25/tau_failure_detection_runs_results_usi_mcs_specific.csv"
    RELIABILITY_TH = 0.999 # Threshold to evaluate interference scenarios
    NUM_UAV = 8
    NUM_RUNS_START = 0 # Start range of run number to consider (inclusive)
    NUM_RUNS_END = 499 # Start range of run number to consider (inclusive)
    
    # Create dataframe of n_r thresholds for uplink
    # ul_th = {10: 395913, 20: 188049, 66.7: 45094, 100: 27200}
    # uav_0_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    # uav_1_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    # uav_2_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    # uav_3_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    # uav_4_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    # uav_5_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    # uav_6_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    # uav_7_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    # OR, if using threshold specific to USI and MCS combo:
    tau_th_df = pd.read_csv("/home/research-student/reuben_ws/omnet-fanet/results_new_Mar25/min_tau_usi_mcs_combos_Mar25.csv")

    run_results = []

    for datasets in DATASET_INT_PATHS:
        scenario_name = datasets.split("/")[-1]
        print(scenario_name)
        print("LOADING AND PROCESSING DATA")
        ul_throughput_df = pd.read_csv(datasets.format("ul"))
        # ul_throughput_df = ul_throughput_df.loc[ul_throughput_df["Run_Num"] <= NUM_RUNS]
        ul_throughput_df = ul_throughput_df.loc[(ul_throughput_df["Run_Num"] >= NUM_RUNS_START) & (ul_throughput_df["Run_Num"] <= NUM_RUNS_END)]
        ul_throughput_df.sort_values(by=["UAV_Speed", "UAV_Height", "Bit_Rate", "USI", "Time"], inplace=True)
        len_df = len(ul_throughput_df)
        dl_throughput_df_list = []
        for i in range(NUM_UAV):
            df = pd.read_csv(datasets.format("UAV_" + str(i)))
            # df = df.loc[df["Run_Num"] <= NUM_RUNS]
            df = df.loc[(df["Run_Num"] >= NUM_RUNS_START) & (df["Run_Num"] <= NUM_RUNS_END)]
            df.sort_values(by=["UAV_Speed", "UAV_Height", "Bit_Rate", "USI", "Time"], inplace=True)
            assert len(df) == len_df, "UAV-{} DF Not Same Length".format(i)
            dl_throughput_df_list.append(df)

        # Combine data to one DF
        ul_throughput_df.rename(columns={"Measured_Reliability_1": "UL_Measured_Reliability_1", "Throughput": "UL_Throughput"}, inplace=True)
        for i in range(NUM_UAV):
            ul_throughput_df["UAV_{}_Measured_Reliability_1".format(i)] = dl_throughput_df_list[i]["Measured_Reliability_1"]
            ul_throughput_df["UAV_{}_Throughput".format(i)] = dl_throughput_df_list[i]["Throughput"]
        
        # Remove NaNs, but combine them first
        ul_throughput_df.dropna(subset=["UL_Measured_Reliability_1", "UAV_0_Measured_Reliability_1", "UAV_1_Measured_Reliability_1", 
                                          "UAV_2_Measured_Reliability_1", "UAV_3_Measured_Reliability_1", "UAV_4_Measured_Reliability_1",
                                          "UAV_5_Measured_Reliability_1", "UAV_6_Measured_Reliability_1", "UAV_7_Measured_Reliability_1",
                                          "UL_Throughput", "UAV_0_Throughput", "UAV_1_Throughput", "UAV_2_Throughput", "UAV_3_Throughput",
                                          "UAV_4_Throughput", "UAV_5_Throughput", "UAV_6_Throughput", "UAV_7_Throughput"], inplace=True)

        # Filter samples with "Time" < 1
        ul_throughput_df = ul_throughput_df.loc[ul_throughput_df["Time"] >= 1]

        # Ground truths 
        ul_throughput_df["Failure_Occur"] = (ul_throughput_df["UAV_0_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_throughput_df["UAV_7_Measured_Reliability_1"] < RELIABILITY_TH) | \
                                                 (ul_throughput_df["UAV_1_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_throughput_df["UAV_2_Measured_Reliability_1"] < RELIABILITY_TH) | \
                                                 (ul_throughput_df["UAV_3_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_throughput_df["UAV_4_Measured_Reliability_1"] < RELIABILITY_TH) | \
                                                 (ul_throughput_df["UAV_5_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_throughput_df["UAV_6_Measured_Reliability_1"] < RELIABILITY_TH) | \
                                                 (ul_throughput_df["UL_Measured_Reliability_1"] < RELIABILITY_TH)
        # Get predictions                       
        # ul_throughput_df[["Failure_Predict", "UL_Failure_Predict", "DL_Failure_Predict"]] = ul_throughput_df.apply(lambda row: evaluate_failure_prediction(row, ul_th, uav_0_th, uav_1_th, uav_2_th, uav_3_th, uav_4_th, uav_5_th, uav_6_th, uav_7_th), 
        #                                                                                                            axis=1, result_type='expand')
        ul_throughput_df[["Failure_Predict", "UL_Failure_Predict", "DL_Failure_Predict"]] = ul_throughput_df.apply(lambda row: evaluate_failure_prediction_usi_mcs(row, tau_th_df), axis=1, result_type='expand')

        # Label whether any reliability is below 90% or 95%
        ul_throughput_df["Below_90"] = (ul_throughput_df["UAV_0_Measured_Reliability_1"] < 0.9) | (ul_throughput_df["UAV_7_Measured_Reliability_1"] < 0.9) | \
                                                 (ul_throughput_df["UAV_1_Measured_Reliability_1"] < 0.9) | (ul_throughput_df["UAV_2_Measured_Reliability_1"] < 0.9) | \
                                                 (ul_throughput_df["UAV_3_Measured_Reliability_1"] < 0.9) | (ul_throughput_df["UAV_4_Measured_Reliability_1"] < 0.9) | \
                                                 (ul_throughput_df["UAV_5_Measured_Reliability_1"] < 0.9) | (ul_throughput_df["UAV_6_Measured_Reliability_1"] < 0.9) | \
                                                 (ul_throughput_df["UL_Measured_Reliability_1"] < 0.9)
        ul_throughput_df["Below_95"] = (ul_throughput_df["UAV_0_Measured_Reliability_1"] < 0.95) | (ul_throughput_df["UAV_7_Measured_Reliability_1"] < 0.95) | \
                                                 (ul_throughput_df["UAV_1_Measured_Reliability_1"] < 0.95) | (ul_throughput_df["UAV_2_Measured_Reliability_1"] < 0.95) | \
                                                 (ul_throughput_df["UAV_3_Measured_Reliability_1"] < 0.95) | (ul_throughput_df["UAV_4_Measured_Reliability_1"] < 0.95) | \
                                                 (ul_throughput_df["UAV_5_Measured_Reliability_1"] < 0.95) | (ul_throughput_df["UAV_6_Measured_Reliability_1"] < 0.95) | \
                                                 (ul_throughput_df["UL_Measured_Reliability_1"] < 0.95)

        # Get sensitivity, specificity, and modified FNR for each run of each combination, and find averages over all runs for each combination
        df_groups = ul_throughput_df.groupby(['USI', 'Bit_Rate', 'UAV_Height', 'UAV_Speed'])
        print("EVALUATING EACH GROUP")
        for group in tqdm(set(df_groups.groups.keys())):
            group_df = df_groups.get_group(group)
            run_groups = group_df.groupby(["Run_Num"])
            for run_group in set(run_groups.groups.keys()):
                run_df = run_groups.get_group(run_group)
                num_samples = len(run_df)
                num_pos_labels = run_df["Failure_Occur"].sum()
                num_neg_labels = num_samples - num_pos_labels
                percent_pos = num_pos_labels / num_samples 
                percent_neg = num_neg_labels / num_samples
                fail_occur_overall = run_df["Failure_Occur"].values
                fail_predict_overall = run_df["Failure_Predict"].values
                run_accuracy = accuracy_score(fail_occur_overall, fail_predict_overall)
                run_tn, run_fp, run_fn, run_tp = confusion_matrix(fail_occur_overall, fail_predict_overall, labels=[0,1]).ravel()
                if ((run_tp + run_fn) > 0):
                    run_sensitivity = run_tp / (run_tp + run_fn)
                else:
                    run_sensitivity = np.nan
                if ((run_tn + run_fp) > 0):
                    run_specificity = run_tn / (run_tn + run_fp)
                else:
                    run_specificity = np.nan
                if ((run_tp + run_fp) > 0):
                    run_precision = run_tp / (run_tp + run_fp)
                else:
                    run_precision = np.nan
                # For mFNR for 90%
                run_df_90 = run_df.loc[run_df["Below_90"]==True]
                num_pos_90_labels = len(run_df_90)
                percent_pos_90 = num_pos_90_labels / num_samples
                percent_neg_90 = 1 - percent_pos_90
                if len(run_df_90) > 0:
                    fail_occur_overall = run_df_90["Failure_Occur"].values
                    fail_predict_overall = run_df_90["Failure_Predict"].values
                    tmp_tn, tmp_fp, tmp_fn, tmp_tp = confusion_matrix(fail_occur_overall, fail_predict_overall, labels=[0,1]).ravel()
                    run_mFNR_90 = tmp_fn / (tmp_fn+tmp_tp)
                else:
                    run_mFNR_90 = np.nan
                # For mFNR for 95%
                run_df_95 = run_df.loc[run_df["Below_95"]==True]
                num_pos_95_labels = len(run_df_95)
                percent_pos_95 = num_pos_95_labels / num_samples
                percent_neg_95 = 1 - percent_pos_95
                if len(run_df_90) > 0:
                    fail_occur_overall = run_df_95["Failure_Occur"].values
                    fail_predict_overall = run_df_95["Failure_Predict"].values
                    tmp_tn, tmp_fp, tmp_fn, tmp_tp = confusion_matrix(fail_occur_overall, fail_predict_overall, labels=[0,1]).ravel()
                    run_mFNR_95 = tmp_fn / (tmp_fn+tmp_tp)
                else:
                    run_mFNR_95 = np.nan

                run_results.append({"Scenario": scenario_name, "Combination": group, "Run_Num": run_group, "Num_Samples": num_samples,
                                    "Accuracy": run_accuracy, "Specificity": run_specificity, "Sensitivity": run_sensitivity, "mFNR_90": run_mFNR_90, "mFNR_95": run_mFNR_95, "Precision": run_precision,
                                    "Num_Pos_Label": num_pos_labels, "Num_Neg_Label": num_neg_labels, "Num_Pos_90_Label": num_pos_90_labels, "Num_Pos_95_Label": num_pos_95_labels, "Num_Pos_Prediction": run_tp + run_fp})

        scenario_results_df = pd.DataFrame(run_results)
        scenario_results_df.to_csv(SAVE_PATH)
    
    