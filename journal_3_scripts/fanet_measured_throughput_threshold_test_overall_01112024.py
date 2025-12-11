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

def load_throughput(scenario_path):
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

if __name__ == "__main__":

    """ EVALUATE THE OVERALL SENSITIVITY AND SPECIFICITY OF ALL CASES AND OF EACH POSSIBILITY OF POSITIVE LABEL SEPARATELY, AND COMBINE USING PREVALENCE
        Possibility 1: Groundtruth of both UL and DL is positive
        Possibility 2: Groundtruth of both UL is positive, but DL is negative
        Possibility 3: Groundtruth of both DL is positive, but UL is negative
        Ground truth: If any link fail, label it positive, else negative
        Prediction: If any gamma below threshold, label it positive, else negative"""
    ''' Define Paths Here'''
    DATASET_INT_PATHS = ["/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_0_{}_processed.csv", 
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_1_{}_processed.csv",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_2_{}_processed.csv",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_1_{}_processed.csv",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_a_{}_processed.csv",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_2_{}_processed.csv"]
    SAVE_PATH = "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/results_new/throughput_failure_detection_overall_results_100_runs.csv"
    CONF_MAT_PATH = "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/results_new/throughput_failure_detection_conf_matrix_100_runs.csv"
    FN_SAVE_PATH = ["/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/results_new/throughput_failure_detection_uav_0_fn_100_runs.csv",
                    "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/results_new/throughput_failure_detection_uav_1_fn_100_runs.csv",
                    "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/results_new/throughput_failure_detection_uav_2_fn_100_runs.csv",
                    "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/results_new/throughput_failure_detection_manet_1_fn_100_runs.csv",
                    "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/results_new/throughput_failure_detection_manet_a_fn_100_runs.csv",
                    "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/results_new/throughput_failure_detection_manet_2_fn_100_runs.csv"]
    RELIABILITY_TH = 0.999 # Threshold to evaluate interference scenarios
    NUM_PROCS = 32
    NUM_UAV = 8
    NUM_RUNS = 99 # The max run number to consider (Run_Num starts at 0)
    # Create dataframe of n_r thresholds for uplink
    ul_th = {10: 395913, 20: 188049, 66.7: 45094, 100: 27200}
    uav_0_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    uav_1_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    uav_2_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    uav_3_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    uav_4_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    uav_5_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    uav_6_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    uav_7_th = {10: 996, 20: 978, 66.7: 978, 100: 978}
    
    ''' INT SCENARIOS '''
    ''' Filter out runs where reliability is above the threshold '''
    print("Testing Int Scenarios")
    results = []
    conf_mat = [] # For TP, TN, FP, FN
    counter = 0
    for datasets in DATASET_INT_PATHS:
        scenario_name = datasets.split("/")[-1]
        print(scenario_name)

        ul_throughput_df = pd.read_csv(datasets.format("ul"))
        ul_throughput_df = ul_throughput_df.loc[ul_throughput_df["Run_Num"] <= NUM_RUNS]
        ul_throughput_df.sort_values(by=["UAV_Speed", "UAV_Height", "Bit_Rate", "USI", "Time"], inplace=True)
        len_df = len(ul_throughput_df)
        dl_throughput_df_list = []
        for i in range(NUM_UAV):
            df = pd.read_csv(datasets.format("UAV_" + str(i)))
            df = df.loc[df["Run_Num"] <= NUM_RUNS]
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

        # Ground truths of UL and DL
        ul_throughput_df["UL_Failure_Occur"] = ul_throughput_df["UL_Measured_Reliability_1"] < RELIABILITY_TH
        ul_throughput_df["DL_Failure_Occur"] = (ul_throughput_df["UAV_0_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_throughput_df["UAV_7_Measured_Reliability_1"] < RELIABILITY_TH) | \
                                                 (ul_throughput_df["UAV_1_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_throughput_df["UAV_2_Measured_Reliability_1"] < RELIABILITY_TH) | \
                                                 (ul_throughput_df["UAV_3_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_throughput_df["UAV_4_Measured_Reliability_1"] < RELIABILITY_TH) | \
                                                 (ul_throughput_df["UAV_5_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_throughput_df["UAV_6_Measured_Reliability_1"] < RELIABILITY_TH)
        # Get predictions
        ul_throughput_df[["Failure_Predict", "UL_Failure_Predict", "DL_Failure_Predict"]] = ul_throughput_df.apply(lambda row: evaluate_failure_prediction(row, ul_th, uav_0_th, uav_1_th, uav_2_th, uav_3_th, uav_4_th, uav_5_th, uav_6_th, uav_7_th), 
                                                                                                                   axis=1, result_type='expand')
        ul_throughput_df["Failure_Occur"] = ul_throughput_df["UL_Failure_Occur"] | ul_throughput_df["DL_Failure_Occur"]

        # # Do it for individual UAVs
        # ul_throughput_df["GW_Failure_Occur"] = ul_throughput_df["UAV_0_Measured_Reliability_1"] < RELIABILITY_TH
        # ul_throughput_df["GW_Failure_Predict"] = ul_throughput_df["UAV_0_Throughput"] < dl_th
        # ul_throughput_df["UAV_1_Failure_Occur"] = ul_throughput_df["UAV_1_Measured_Reliability_1"] < RELIABILITY_TH
        # ul_throughput_df["UAV_1_Failure_Predict"] = ul_throughput_df["UAV_1_Throughput"] < dl_th
        # ul_throughput_df["UAV_2_Failure_Occur"] = ul_throughput_df["UAV_2_Measured_Reliability_1"] < RELIABILITY_TH
        # ul_throughput_df["UAV_2_Failure_Predict"] = ul_throughput_df["UAV_2_Throughput"] < dl_th
        # ul_throughput_df["UAV_3_Failure_Occur"] = ul_throughput_df["UAV_3_Measured_Reliability_1"] < RELIABILITY_TH
        # ul_throughput_df["UAV_3_Failure_Predict"] = ul_throughput_df["UAV_3_Throughput"] < dl_th
        # ul_throughput_df["UAV_4_Failure_Occur"] = ul_throughput_df["UAV_4_Measured_Reliability_1"] < RELIABILITY_TH
        # ul_throughput_df["UAV_4_Failure_Predict"] = ul_throughput_df["UAV_4_Throughput"] < dl_th
        # ul_throughput_df["UAV_5_Failure_Occur"] = ul_throughput_df["UAV_5_Measured_Reliability_1"] < RELIABILITY_TH
        # ul_throughput_df["UAV_5_Failure_Predict"] = ul_throughput_df["UAV_5_Throughput"] < dl_th
        # ul_throughput_df["UAV_6_Failure_Occur"] = ul_throughput_df["UAV_6_Measured_Reliability_1"] < RELIABILITY_TH
        # ul_throughput_df["UAV_6_Failure_Predict"] = ul_throughput_df["UAV_6_Throughput"] < dl_th
        # ul_throughput_df["UAV_7_Failure_Occur"] = ul_throughput_df["UAV_7_Measured_Reliability_1"] < RELIABILITY_TH
        # ul_throughput_df["UAV_7_Failure_Predict"] = ul_throughput_df["UAV_7_Throughput"] < dl_th
        
        # ul_throughput_df["GW_UL_Failure_Predict"] = ul_throughput_df["UL_Failure_Predict"] | ul_throughput_df["GW_Failure_Predict"]
        # ul_throughput_df["GW_UL_Failure_Occur"] = ul_throughput_df["UL_Failure_Occur"] | ul_throughput_df["GW_Failure_Occur"]
        
        # # Process possibility 1: Groundtruth of both UL and DL is positive
        # possibility_1_df = ul_throughput_df.loc[(ul_throughput_df["UL_Failure_Occur"]) & (ul_throughput_df["DL_Failure_Occur"])]
        # fail_occur_1 = possibility_1_df["Failure_Occur"].values
        # fail_predict_1 = possibility_1_df["Failure_Predict"].values
        # overall_tn, overall_fp, overall_fn, overall_tp = confusion_matrix(fail_occur_1, fail_predict_1, labels=[0,1]).ravel()
        # assert (overall_tn == 0) | (overall_fp == 0), "Should not have Negative Labels here"
        # overall_sensitivity_1 = overall_tp / (overall_tp + overall_fn)

        # # Process possibility 2: Groundtruth of both UL is positive, but DL is negative
        # possibility_2_df = ul_throughput_df.loc[(ul_throughput_df["UL_Failure_Occur"]) & ~(ul_throughput_df["DL_Failure_Occur"])]
        # fail_occur_2 = possibility_2_df["Failure_Occur"].values
        # fail_predict_2 = possibility_2_df["Failure_Predict"].values
        # overall_tn, overall_fp, overall_fn, overall_tp = confusion_matrix(fail_occur_2, fail_predict_2, labels=[0,1]).ravel()
        # assert (overall_tn == 0) | (overall_fp == 0), "Should not have Negative Labels here"
        # overall_sensitivity_2 = overall_tp / (overall_tp + overall_fn)

        # # Process possibility 3: Groundtruth of both DL is positive, but UL is negative
        # possibility_3_df = ul_throughput_df.loc[~(ul_throughput_df["UL_Failure_Occur"]) & (ul_throughput_df["DL_Failure_Occur"])]
        # fail_occur_3 = possibility_3_df["Failure_Occur"].values
        # fail_predict_3 = possibility_3_df["Failure_Predict"].values
        # overall_tn, overall_fp, overall_fn, overall_tp = confusion_matrix(fail_occur_3, fail_predict_3, labels=[0,1]).ravel()
        # assert (overall_tn == 0) | (overall_fp == 0), "Should not have Negative Labels here"
        # overall_sensitivity_3 = overall_tp / (overall_tp + overall_fn)

        # prevalence_1 = len(possibility_1_df) / (len(possibility_1_df) + len(possibility_2_df) + len(possibility_3_df))
        # prevalence_2 = len(possibility_2_df) / (len(possibility_1_df) + len(possibility_2_df) + len(possibility_3_df))
        # prevalence_3 = len(possibility_3_df) / (len(possibility_1_df) + len(possibility_2_df) + len(possibility_3_df))

        fail_occur_overall = ul_throughput_df["Failure_Occur"].values
        fail_predict_overall = ul_throughput_df["Failure_Predict"].values
        overall_accuracy = accuracy_score(fail_occur_overall, fail_predict_overall)
        overall_tn, overall_fp, overall_fn, overall_tp = confusion_matrix(fail_occur_overall, fail_predict_overall, labels=[0,1]).ravel()
        overall_sensitivity = overall_tp / (overall_tp + overall_fn)
        overall_specificity = overall_tn / (overall_tn + overall_fp)

        # gw_ul_tn, gw_ul_fp, gw_ul_fn, gw_ul_tp = confusion_matrix(ul_throughput_df["GW_UL_Failure_Occur"].values, ul_throughput_df["GW_UL_Failure_Predict"].values, labels=[0,1]).ravel()
        # gw_ul_sensitivity = gw_ul_tp / (gw_ul_tp + gw_ul_fn)
        # gw_ul_specificity = gw_ul_tn / (gw_ul_tn + gw_ul_fp)

        # # Evaluate individual links specificity and sensitivity
        # ul_tn, ul_fp, ul_fn, ul_tp = confusion_matrix(ul_throughput_df["UL_Failure_Occur"].values, ul_throughput_df["UL_Failure_Predict"].values, labels=[0,1]).ravel()
        # ul_sensitivity = ul_tp / (ul_tp + ul_fn)
        # ul_specificity = ul_tn / (ul_tn + ul_fp)
        # ul_accuracy = accuracy_score(ul_throughput_df["UL_Failure_Occur"].values, ul_throughput_df["UL_Failure_Predict"].values)

        # dl_tn, dl_fp, dl_fn, dl_tp = confusion_matrix(ul_throughput_df["DL_Failure_Occur"].values, ul_throughput_df["DL_Failure_Predict"].values, labels=[0,1]).ravel()
        # dl_sensitivity = dl_tp / (dl_tp + dl_fn)
        # dl_specificity = dl_tn / (dl_tn + dl_fp)
        # dl_accuracy = accuracy_score(ul_throughput_df["DL_Failure_Occur"].values, ul_throughput_df["DL_Failure_Predict"].values)

        # gw_tn, gw_fp, gw_fn, gw_tp = confusion_matrix(ul_throughput_df["GW_Failure_Occur"].values, ul_throughput_df["GW_Failure_Predict"].values, labels=[0,1]).ravel()
        # gw_sensitivity = gw_tp / (gw_tp + gw_fn)
        # gw_specificity = gw_tn / (gw_tn + gw_fp)

        # uav_1_tn, uav_1_fp, uav_1_fn, uav_1_tp = confusion_matrix(ul_throughput_df["UAV_1_Failure_Occur"].values, ul_throughput_df["UAV_1_Failure_Predict"].values, labels=[0,1]).ravel()
        # uav_1_sensitivity = uav_1_tp / (uav_1_tp + uav_1_fn)
        # uav_1_specificity = uav_1_tn / (uav_1_tn + uav_1_fp)

        # uav_2_tn, uav_2_fp, uav_2_fn, uav_2_tp = confusion_matrix(ul_throughput_df["UAV_2_Failure_Occur"].values, ul_throughput_df["UAV_2_Failure_Predict"].values, labels=[0,1]).ravel()
        # uav_2_sensitivity = uav_2_tp / (uav_2_tp + uav_2_fn)
        # uav_2_specificity = uav_2_tn / (uav_2_tn + uav_2_fp)

        # uav_3_tn, uav_3_fp, uav_3_fn, uav_3_tp = confusion_matrix(ul_throughput_df["UAV_3_Failure_Occur"].values, ul_throughput_df["UAV_3_Failure_Predict"].values, labels=[0,1]).ravel()
        # uav_3_sensitivity = uav_3_tp / (uav_3_tp + uav_3_fn)
        # uav_3_specificity = uav_3_tn / (uav_3_tn + uav_3_fp)
        
        # uav_4_tn, uav_4_fp, uav_4_fn, uav_4_tp = confusion_matrix(ul_throughput_df["UAV_4_Failure_Occur"].values, ul_throughput_df["UAV_4_Failure_Predict"].values, labels=[0,1]).ravel()
        # uav_4_sensitivity = uav_4_tp / (uav_4_tp + uav_4_fn)
        # uav_4_specificity = uav_4_tn / (uav_4_tn + uav_4_fp)

        # uav_5_tn, uav_5_fp, uav_5_fn, uav_5_tp = confusion_matrix(ul_throughput_df["UAV_5_Failure_Occur"].values, ul_throughput_df["UAV_5_Failure_Predict"].values, labels=[0,1]).ravel()
        # uav_5_sensitivity = uav_5_tp / (uav_5_tp + uav_5_fn)
        # uav_5_specificity = uav_5_tn / (uav_5_tn + uav_5_fp)

        # uav_6_tn, uav_6_fp, uav_6_fn, uav_6_tp = confusion_matrix(ul_throughput_df["UAV_6_Failure_Occur"].values, ul_throughput_df["UAV_6_Failure_Predict"].values, labels=[0,1]).ravel()
        # uav_6_sensitivity = uav_6_tp / (uav_6_tp + uav_6_fn)
        # uav_6_specificity = uav_6_tn / (uav_6_tn + uav_6_fp)

        # uav_7_tn, uav_7_fp, uav_7_fn, uav_7_tp = confusion_matrix(ul_throughput_df["UAV_7_Failure_Occur"].values, ul_throughput_df["UAV_7_Failure_Predict"].values, labels=[0,1]).ravel()
        # uav_7_sensitivity = uav_7_tp / (uav_7_tp + uav_7_fn)
        # uav_7_specificity = uav_7_tn / (uav_7_tn + uav_7_fp)

        # Analysis for Different USI
        usi_10_df = ul_throughput_df.loc[ul_throughput_df["USI"]==10]
        overall_usi_10_accuracy = accuracy_score(usi_10_df["Failure_Occur"].values, usi_10_df["Failure_Predict"].values)
        overall_usi_10_tn, overall_usi_10_fp, overall_usi_10_fn, overall_usi_10_tp = confusion_matrix(usi_10_df["Failure_Occur"].values, usi_10_df["Failure_Predict"].values, labels=[0,1]).ravel()
        overall_usi_10_sensitivity = overall_usi_10_tp / (overall_usi_10_tp + overall_usi_10_fn)
        overall_usi_10_specificity = overall_usi_10_tn / (overall_usi_10_tn + overall_usi_10_fp)

        usi_20_df = ul_throughput_df.loc[ul_throughput_df["USI"]==20]
        overall_usi_20_accuracy = accuracy_score(usi_20_df["Failure_Occur"].values, usi_20_df["Failure_Predict"].values)
        overall_usi_20_tn, overall_usi_20_fp, overall_usi_20_fn, overall_usi_20_tp = confusion_matrix(usi_20_df["Failure_Occur"].values, usi_20_df["Failure_Predict"].values, labels=[0,1]).ravel()
        overall_usi_20_sensitivity = overall_usi_20_tp / (overall_usi_20_tp + overall_usi_20_fn)
        overall_usi_20_specificity = overall_usi_20_tn / (overall_usi_20_tn + overall_usi_20_fp)

        usi_667_df = ul_throughput_df.loc[ul_throughput_df["USI"]==66.7]
        overall_usi_667_accuracy = accuracy_score(usi_667_df["Failure_Occur"].values, usi_667_df["Failure_Predict"].values)
        overall_usi_667_tn, overall_usi_667_fp, overall_usi_667_fn, overall_usi_667_tp = confusion_matrix(usi_667_df["Failure_Occur"].values, usi_667_df["Failure_Predict"].values, labels=[0,1]).ravel()
        overall_usi_667_sensitivity = overall_usi_667_tp / (overall_usi_667_tp + overall_usi_667_fn)
        overall_usi_667_specificity = overall_usi_667_tn / (overall_usi_667_tn + overall_usi_667_fp)

        usi_100_df = ul_throughput_df.loc[ul_throughput_df["USI"]==100]
        overall_usi_100_accuracy = accuracy_score(usi_100_df["Failure_Occur"].values, usi_100_df["Failure_Predict"].values)
        overall_usi_100_tn, overall_usi_100_fp, overall_usi_100_fn, overall_usi_100_tp = confusion_matrix(usi_100_df["Failure_Occur"].values, usi_100_df["Failure_Predict"].values, labels=[0,1]).ravel()
        overall_usi_100_sensitivity = overall_usi_100_tp / (overall_usi_100_tp + overall_usi_100_fn)
        overall_usi_100_specificity = overall_usi_100_tn / (overall_usi_100_tn + overall_usi_100_fp)

        results.append({"Scenario": scenario_name, "Overall_Accuracy": overall_accuracy, "Overall_Sensitivity": overall_sensitivity, "Overall_Specificity": overall_specificity, 
                        # "UL_Accuracy": ul_accuracy, "UL_Sensitivity": ul_sensitivity, "UL_Specificity": ul_specificity, 
                        # "DL_Accuracy": dl_accuracy, "DL_Sensitivity": dl_sensitivity, "DL_Specificity": dl_specificity, 
                        # "GW_UL_Sensitivity": gw_ul_sensitivity, "GW_UL_Specificity": gw_ul_specificity,
                        # "GW_Sensitivity": gw_sensitivity, "GW_Specificity": gw_specificity,
                        # "UAV_1_Sensitivity": uav_1_sensitivity, "UAV_1_Specificity": uav_1_specificity,
                        # "UAV_2_Sensitivity": uav_2_sensitivity, "UAV_2_Specificity": uav_2_specificity,
                        # "UAV_3_Sensitivity": uav_3_sensitivity, "UAV_3_Specificity": uav_3_specificity,
                        # "UAV_4_Sensitivity": uav_4_sensitivity, "UAV_4_Specificity": uav_4_specificity,
                        # "UAV_5_Sensitivity": uav_5_sensitivity, "UAV_5_Specificity": uav_5_specificity,
                        # "UAV_6_Sensitivity": uav_6_sensitivity, "UAV_6_Specificity": uav_6_specificity,
                        # "UAV_7_Sensitivity": uav_7_sensitivity, "UAV_7_Specificity": uav_7_specificity,
                        "Overall_USI_10_Accuracy": overall_usi_10_accuracy, "Overall_USI_10_Sensitivity": overall_usi_10_sensitivity, "Overall_USI_10_Specificity": overall_usi_10_specificity,
                        "Overall_USI_20_Accuracy": overall_usi_20_accuracy, "Overall_USI_20_Sensitivity": overall_usi_20_sensitivity, "Overall_USI_20_Specificity": overall_usi_20_specificity, 
                        "Overall_USI_667_Accuracy": overall_usi_667_accuracy, "Overall_USI_667_Sensitivity": overall_usi_667_sensitivity, "Overall_USI_667_Specificity": overall_usi_667_specificity,
                        "Overall_USI_100_Accuracy": overall_usi_100_accuracy, "Overall_USI_100_Sensitivity": overall_usi_100_sensitivity, "Overall_USI_100_Specificity": overall_usi_100_specificity})
                        # "Pos_1_Sensitivity": overall_sensitivity_1, "Pos_2_Sensitivity": overall_sensitivity_2, "Pos_3_Sensitivity": overall_sensitivity_3,
                        # "Prevalence_Pos_1": prevalence_1, "Prevalence_Pos_2": prevalence_2, "Prevalence_Pos_3": prevalence_3})

        conf_mat.append({"Scenario": scenario_name, "Overall_TN": overall_tn, "Overall_FP": overall_fp, "Overall_FN": overall_fn, "Overall_TP": overall_tp,
                        #  "UL_TN": ul_tn, "UL_FP": ul_fp, "UL_FN": ul_fn, "UL_TP": ul_tp,
                        #  "DL_TN": dl_tn, "DL_FP": dl_fp, "DL_FN": dl_fn, "DL_TP": dl_tp,
                        #  "GW_UL_TN": gw_ul_tn, "GW_UL_FP": gw_ul_fp, "GW_UL_FN": gw_ul_fn, "GW_UL_TP": gw_ul_tp,
                        #  "GW_TN": gw_tn, "GW_FP": gw_fp, "GW_FN": gw_fn, "GW_TP": gw_tp,
                        #  "UAV_1_TN": uav_1_tn, "UAV_1_FP": uav_1_fp, "UAV_1_FN": uav_1_fn, "UAV_1_TP": uav_1_tp,
                        #  "UAV_2_TN": uav_2_tn, "UAV_2_FP": uav_2_fp, "UAV_2_FN": uav_2_fn, "UAV_2_TP": uav_2_tp,
                        #  "UAV_3_TN": uav_3_tn, "UAV_3_FP": uav_3_fp, "UAV_3_FN": uav_3_fn, "UAV_3_TP": uav_3_tp,
                        #  "UAV_4_TN": uav_4_tn, "UAV_4_FP": uav_4_fp, "UAV_4_FN": uav_4_fn, "UAV_4_TP": uav_4_tp,
                        #  "UAV_5_TN": uav_5_tn, "UAV_5_FP": uav_5_fp, "UAV_5_FN": uav_5_fn, "UAV_5_TP": uav_5_tp,
                        #  "UAV_6_TN": uav_6_tn, "UAV_6_FP": uav_6_fp, "UAV_6_FN": uav_6_fn, "UAV_6_TP": uav_6_tp,
                        #  "UAV_7_TN": uav_7_tn, "UAV_7_FP": uav_7_fp, "UAV_7_FN": uav_7_fn, "UAV_7_TP": uav_7_tp,
                         "Overall_USI_10_TN": overall_usi_10_tn, "Overall_USI_10_FP": overall_usi_10_fp, "Overall_USI_10_FN": overall_usi_10_fn, "Overall_USI_10_TP": overall_usi_10_tp,
                         "Overall_USI_20_TN": overall_usi_20_tn, "Overall_USI_20_FP": overall_usi_20_fp, "Overall_USI_20_FN": overall_usi_20_fn, "Overall_USI_20_TP": overall_usi_20_tp,
                         "Overall_USI_667_TN": overall_usi_667_tn, "Overall_USI_667_FP": overall_usi_667_fp, "Overall_USI_667_FN": overall_usi_667_fn, "Overall_USI_667_TP": overall_usi_667_tp,
                         "Overall_USI_100_TN": overall_usi_100_tn, "Overall_USI_100_FP": overall_usi_100_fp, "Overall_USI_100_FN": overall_usi_100_fn, "Overall_USI_100_TP": overall_usi_100_tp})
        temp_df = pd.DataFrame(results)
        temp_df.to_csv(SAVE_PATH)

        # Get the overall FN cases and store the results ul_num_reliable_df["Failure_Occur"]
        overall_fn_cases = ul_throughput_df.loc[(ul_throughput_df["Failure_Occur"] == 1) & (ul_throughput_df["Failure_Predict"] == 0)]
        overall_fn_cases = overall_fn_cases[["USI", "UL_Measured_Reliability_1", "UAV_0_Measured_Reliability_1", "UAV_1_Measured_Reliability_1", "UAV_2_Measured_Reliability_1",
                                             "UAV_3_Measured_Reliability_1", "UAV_4_Measured_Reliability_1", "UAV_5_Measured_Reliability_1", "UAV_6_Measured_Reliability_1", "UAV_7_Measured_Reliability_1"]]
        overall_fn_cases.to_csv(FN_SAVE_PATH[counter])
        counter += 1
        
    scenario_results_df = pd.DataFrame(results)
    scenario_results_df.to_csv(SAVE_PATH)

    conf_mat_df = pd.DataFrame(conf_mat)
    conf_mat_df.to_csv(CONF_MAT_PATH)
    
    