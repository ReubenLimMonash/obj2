# Modified from: fanet_measured_throughput_false_neg_16032025.py
# Date: 07/04/2025
# Desc: To get the measured reliability (in time windows) of true positive predictions for throughput 

import pandas as pd
from tqdm import tqdm

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
        Prediction: If any tau below threshold, label it positive, else negative"""
    ''' Define Paths Here'''
    DATASET_INT_PATHS = ["/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_0_{}_processed.csv", 
                         "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_1_{}_processed.csv",
                         "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_2_{}_processed.csv",
                         "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_1_{}_processed.csv",
                         "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_a_{}_processed.csv",
                         "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_2_{}_processed.csv"]
    SAVE_PATH = "/home/research-student/reuben_ws/omnet-fanet/results_new_Mar25/tau_failure_detection_tp_999.csv"
    
    RELIABILITY_TH = 0.999 # Threshold to evaluate interference scenarios
    NUM_PROCS = 32
    NUM_UAV = 8
    NUM_RUNS_START = 0 # Start range of run number to consider (inclusive)
    NUM_RUNS_END = 499 # Start range of run number to consider (inclusive)

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
    
    tp_run_df_list = []

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
            df.sort_values(by=["UAV_Speed", "UAV_Height", "Bit_Rate", "USI", "Run_Num", "Time"], inplace=True)
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
        ul_throughput_df[["Failure_Predict", "UL_Failure_Predict", "DL_Failure_Predict"]] = ul_throughput_df.apply(lambda row: evaluate_failure_prediction(row, ul_th, uav_0_th, uav_1_th, uav_2_th, uav_3_th, uav_4_th, uav_5_th, uav_6_th, uav_7_th), 
                                                                                                                       axis=1, result_type='expand')
        # Get sensitivity, specificity, and modified FNR for each run of each combination, and find averages over all runs for each combination
        df_groups = ul_throughput_df.groupby(['USI', 'Bit_Rate', 'UAV_Height', 'UAV_Speed'])
        group_mean_results_list = []
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
        