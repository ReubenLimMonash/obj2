# Modified from: fanet_num_reliable_threshold_test_overall_01112024.py
# Date: 20/11/2024
# Desc: To test threshold for nr obtained from ROC Youden Index

import pandas as pd
import numpy as np 
import os
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from multiprocessing.pool import Pool
from itertools import repeat

def load_num_reliable(scenario_path):
    scenario_name = scenario_path.split("/")[-1]
    params = scenario_name.split("_")
    usi = [x for x in params if "UAVSendingInterval" in x][0].split('-')[-1]
    # Load DL Throughput
    dl_df_list = []
    dl_num_reliable_files = glob.glob(os.path.join(scenario_path, "Run-*_Downlink_Throughput.csv"))
    for file in dl_num_reliable_files:
        measured_df = pd.read_csv(file)
        dl_df_list.append(measured_df)
    dl_df = pd.concat(dl_df_list)
    # Load UL Throughput
    ul_df_list = []
    ul_num_reliable_files = glob.glob(os.path.join(scenario_path, "Run-*_Uplink_Throughput.csv"))
    for file in ul_num_reliable_files:
        measured_df = pd.read_csv(file)
        ul_df_list.append(measured_df)
    ul_df = pd.concat(ul_df_list)
    ul_df["USI"] = usi
    # Load Video Throughput
    vid_df_list = []
    vid_num_reliable_files = glob.glob(os.path.join(scenario_path, "Run-*_Video_Throughput.csv"))
    for file in vid_num_reliable_files:
        measured_df = pd.read_csv(file)
        vid_df_list.append(measured_df)
    vid_df = pd.concat(vid_df_list)

    return (dl_df, ul_df, vid_df)

def evaluate_failure_prediction(row, youden_df):
    # youden_df is the rows from DF containing the Youden thresholds for this scenario, under different USI (in different rows)
    usi = row["USI"]
    youden_df_tmp = youden_df.loc[youden_df["USI"]==usi]
    ul_fail = row["UL_Num_Reliable"] < youden_df_tmp["UL"].values[0]
    dl_0_fail = row["UAV_0_Num_Reliable"] < youden_df_tmp["UAV_0"].values[0]
    dl_1_fail = row["UAV_1_Num_Reliable"] < youden_df_tmp["UAV_1"].values[0]
    dl_2_fail = row["UAV_2_Num_Reliable"] < youden_df_tmp["UAV_2"].values[0]
    dl_3_fail = row["UAV_3_Num_Reliable"] < youden_df_tmp["UAV_3"].values[0]
    dl_4_fail = row["UAV_4_Num_Reliable"] < youden_df_tmp["UAV_4"].values[0]
    dl_5_fail = row["UAV_5_Num_Reliable"] < youden_df_tmp["UAV_5"].values[0]
    dl_6_fail = row["UAV_6_Num_Reliable"] < youden_df_tmp["UAV_6"].values[0]
    dl_7_fail = row["UAV_7_Num_Reliable"] < youden_df_tmp["UAV_7"].values[0]
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
    DATASET_INT_PATHS = ["/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_0_{}_processed.csv", 
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_1_{}_processed.csv",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_2_{}_processed.csv",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_1_{}_processed.csv",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_a_{}_processed.csv",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_2_{}_processed.csv"]
    SAVE_PATH = "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/num_reliable_failure_detection_youden_overall_results_500_runs.csv"
    CONF_MAT_PATH = "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/num_reliable_failure_detection_youden_conf_matrix_500_runs.csv"
    YOUDEN_PATH = "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/roc_auc_analysis.csv"
    RELIABILITY_TH = 0.999 # Threshold to evaluate interference scenarios
    NUM_PROCS = 32
    NUM_UAV = 8
    NUM_RUNS = 499 # The max run number to consider (Run_Num starts at 0)

    youden_df = pd.read_csv(YOUDEN_PATH)
    
    ''' INT SCENARIOS '''
    ''' Filter out runs where reliability is above the threshold '''
    print("Testing Int Scenarios")
    results = []
    conf_mat = [] # For TP, TN, FP, FN
    for datasets in DATASET_INT_PATHS:
        scenario_name = datasets.split("/")[-1]
        print(scenario_name)

        # Get the rows in youden_df containing the relevant Youden Thresholds for this scenario under different USI
        youden_df_temp = youden_df.loc[(youden_df["Scenario"]==scenario_name) & (youden_df["Metric"]=="Gamma_Threshold")]
        assert not youden_df_temp.empty, "Youden Threshold for this scenario not found: {}".format(scenario_name)
        
        ul_num_reliable_df = pd.read_csv(datasets.format("ul"))
        ul_num_reliable_df = ul_num_reliable_df.loc[ul_num_reliable_df["Run_Num"] <= NUM_RUNS]
        ul_num_reliable_df.sort_values(by=["UAV_Speed", "UAV_Height", "Bit_Rate", "USI", "Time"], inplace=True)
        len_df = len(ul_num_reliable_df)
        dl_num_reliable_df_list = []
        for i in range(NUM_UAV):
            df = pd.read_csv(datasets.format("UAV_" + str(i)))
            df = df.loc[df["Run_Num"] <= NUM_RUNS]
            df.sort_values(by=["UAV_Speed", "UAV_Height", "Bit_Rate", "USI", "Time"], inplace=True)
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

        # Ground truths of UL and DL
        ul_num_reliable_df["UL_Failure_Occur"] = ul_num_reliable_df["UL_Measured_Reliability_1"] < RELIABILITY_TH
        ul_num_reliable_df["DL_Failure_Occur"] = (ul_num_reliable_df["UAV_0_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_num_reliable_df["UAV_7_Measured_Reliability_1"] < RELIABILITY_TH) | \
                                                 (ul_num_reliable_df["UAV_1_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_num_reliable_df["UAV_2_Measured_Reliability_1"] < RELIABILITY_TH) | \
                                                 (ul_num_reliable_df["UAV_3_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_num_reliable_df["UAV_4_Measured_Reliability_1"] < RELIABILITY_TH) | \
                                                 (ul_num_reliable_df["UAV_5_Measured_Reliability_1"] < RELIABILITY_TH) | (ul_num_reliable_df["UAV_6_Measured_Reliability_1"] < RELIABILITY_TH)
        
        

        # Get predictions
        ul_num_reliable_df[["Failure_Predict", "UL_Failure_Predict", "DL_Failure_Predict"]] = ul_num_reliable_df.apply(lambda row: evaluate_failure_prediction(row, youden_df_temp), axis=1, result_type='expand')
        ul_num_reliable_df["Failure_Occur"] = ul_num_reliable_df["UL_Failure_Occur"] | ul_num_reliable_df["DL_Failure_Occur"]

        fail_occur_overall = ul_num_reliable_df["Failure_Occur"].values
        fail_predict_overall = ul_num_reliable_df["Failure_Predict"].values
        overall_accuracy = accuracy_score(fail_occur_overall, fail_predict_overall)
        overall_tn, overall_fp, overall_fn, overall_tp = confusion_matrix(fail_occur_overall, fail_predict_overall, labels=[0,1]).ravel()
        overall_sensitivity = overall_tp / (overall_tp + overall_fn)
        overall_specificity = overall_tn / (overall_tn + overall_fp)

        # Analysis for Different USI
        usi_10_df = ul_num_reliable_df.loc[ul_num_reliable_df["USI"]==10]
        overall_usi_10_accuracy = accuracy_score(usi_10_df["Failure_Occur"].values, usi_10_df["Failure_Predict"].values)
        overall_usi_10_tn, overall_usi_10_fp, overall_usi_10_fn, overall_usi_10_tp = confusion_matrix(usi_10_df["Failure_Occur"].values, usi_10_df["Failure_Predict"].values, labels=[0,1]).ravel()
        overall_usi_10_sensitivity = overall_usi_10_tp / (overall_usi_10_tp + overall_usi_10_fn)
        overall_usi_10_specificity = overall_usi_10_tn / (overall_usi_10_tn + overall_usi_10_fp)

        usi_20_df = ul_num_reliable_df.loc[ul_num_reliable_df["USI"]==20]
        overall_usi_20_accuracy = accuracy_score(usi_20_df["Failure_Occur"].values, usi_20_df["Failure_Predict"].values)
        overall_usi_20_tn, overall_usi_20_fp, overall_usi_20_fn, overall_usi_20_tp = confusion_matrix(usi_20_df["Failure_Occur"].values, usi_20_df["Failure_Predict"].values, labels=[0,1]).ravel()
        overall_usi_20_sensitivity = overall_usi_20_tp / (overall_usi_20_tp + overall_usi_20_fn)
        overall_usi_20_specificity = overall_usi_20_tn / (overall_usi_20_tn + overall_usi_20_fp)

        usi_667_df = ul_num_reliable_df.loc[ul_num_reliable_df["USI"]==66.7]
        overall_usi_667_accuracy = accuracy_score(usi_667_df["Failure_Occur"].values, usi_667_df["Failure_Predict"].values)
        overall_usi_667_tn, overall_usi_667_fp, overall_usi_667_fn, overall_usi_667_tp = confusion_matrix(usi_667_df["Failure_Occur"].values, usi_667_df["Failure_Predict"].values, labels=[0,1]).ravel()
        overall_usi_667_sensitivity = overall_usi_667_tp / (overall_usi_667_tp + overall_usi_667_fn)
        overall_usi_667_specificity = overall_usi_667_tn / (overall_usi_667_tn + overall_usi_667_fp)

        usi_100_df = ul_num_reliable_df.loc[ul_num_reliable_df["USI"]==100]
        overall_usi_100_accuracy = accuracy_score(usi_100_df["Failure_Occur"].values, usi_100_df["Failure_Predict"].values)
        overall_usi_100_tn, overall_usi_100_fp, overall_usi_100_fn, overall_usi_100_tp = confusion_matrix(usi_100_df["Failure_Occur"].values, usi_100_df["Failure_Predict"].values, labels=[0,1]).ravel()
        overall_usi_100_sensitivity = overall_usi_100_tp / (overall_usi_100_tp + overall_usi_100_fn)
        overall_usi_100_specificity = overall_usi_100_tn / (overall_usi_100_tn + overall_usi_100_fp)

        results.append({"Scenario": scenario_name, "Overall_Accuracy": overall_accuracy, "Overall_Sensitivity": overall_sensitivity, "Overall_Specificity": overall_specificity, 
                        "Overall_USI_10_Accuracy": overall_usi_10_accuracy, "Overall_USI_10_Sensitivity": overall_usi_10_sensitivity, "Overall_USI_10_Specificity": overall_usi_10_specificity,
                        "Overall_USI_20_Accuracy": overall_usi_20_accuracy, "Overall_USI_20_Sensitivity": overall_usi_20_sensitivity, "Overall_USI_20_Specificity": overall_usi_20_specificity, 
                        "Overall_USI_667_Accuracy": overall_usi_667_accuracy, "Overall_USI_667_Sensitivity": overall_usi_667_sensitivity, "Overall_USI_667_Specificity": overall_usi_667_specificity,
                        "Overall_USI_100_Accuracy": overall_usi_100_accuracy, "Overall_USI_100_Sensitivity": overall_usi_100_sensitivity, "Overall_USI_100_Specificity": overall_usi_100_specificity})

        conf_mat.append({"Scenario": scenario_name, "Overall_TN": overall_tn, "Overall_FP": overall_fp, "Overall_FN": overall_fn, "Overall_TP": overall_tp,
                         "Overall_USI_10_TN": overall_usi_10_tn, "Overall_USI_10_FP": overall_usi_10_fp, "Overall_USI_10_FN": overall_usi_10_fn, "Overall_USI_10_TP": overall_usi_10_tp,
                         "Overall_USI_20_TN": overall_usi_20_tn, "Overall_USI_20_FP": overall_usi_20_fp, "Overall_USI_20_FN": overall_usi_20_fn, "Overall_USI_20_TP": overall_usi_20_tp,
                         "Overall_USI_667_TN": overall_usi_667_tn, "Overall_USI_667_FP": overall_usi_667_fp, "Overall_USI_667_FN": overall_usi_667_fn, "Overall_USI_667_TP": overall_usi_667_tp,
                         "Overall_USI_100_TN": overall_usi_100_tn, "Overall_USI_100_FP": overall_usi_100_fp, "Overall_USI_100_FN": overall_usi_100_fn, "Overall_USI_100_TP": overall_usi_100_tp})
        temp_df = pd.DataFrame(results)
        temp_df.to_csv(SAVE_PATH)
        
    scenario_results_df = pd.DataFrame(results)
    scenario_results_df.to_csv(SAVE_PATH)

    conf_mat_df = pd.DataFrame(conf_mat)
    conf_mat_df.to_csv(CONF_MAT_PATH)
    
    