# Date: 25/10/2024
# Desc: To analyse which link triggers detection first in each sim run

import pandas as pd
import numpy as np 
import os
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from multiprocessing.pool import Pool
from itertools import repeat

def sim_run_get_detection(run_num, scenario, reliability_th, dl_num_reliable_th, ul_num_reliable_th, vid_num_reliable_th):
    # Check the reliability of all links
    reliability_df = pd.read_csv(os.path.join(scenario, "Simulation_Results.csv"))
    run_reliability_df = reliability_df.loc[reliability_df["Run"] == run_num]
    reliable_check = (run_reliability_df["Total_Reliability_DL"].values[0] >= reliability_th) and (run_reliability_df["Total_Reliability_UL"].values[0] >= reliability_th) and (run_reliability_df["Total_Reliability_VID"].values[0] >= reliability_th)
    scenario_name = scenario.split("/")[-1]

    if reliable_check: # If the run is reliable, skip
        result = {"Scenario": scenario_name, "Run_Num": run_num, "DL_Fail_First": np.nan, "UL_Fail_First": np.nan, "VID_Fail_First": np.nan,
                "DL_Detect_First": np.nan, "UL_Detect_First": np.nan, "VID_Detect_First": np.nan,
            "DL_Fail_Time": np.nan, "UL_Fail_Time": np.nan, "Vid_Fail_Time": np.nan, 
            "DL_Detect_Time": np.nan, "UL_Detect_Time": np.nan, "Vid_Detect_Time": np.nan}
    
    else: # Get the link that failed first, and that detected first
        # Load throughput data
        ul_num_reliable_df = pd.read_csv(os.path.join(scenario, "Run-{}_Uplink_Throughput.csv".format(run_num)))
        vid_num_reliable_df = pd.read_csv(os.path.join(scenario, "Run-{}_Video_Throughput.csv".format(run_num)))
        dl_num_reliable_list = glob.glob(os.path.join(scenario, "Run-{}*_Downlink_Throughput.csv".format(run_num)))
        dl_df_list = []
        for file in dl_num_reliable_list:
            df = pd.read_csv(file)
            dl_df_list.append(df)
        dl_num_reliable_df = pd.concat(dl_df_list)

        # Filter data with time before 1
        ul_num_reliable_df = ul_num_reliable_df.loc[ul_num_reliable_df["Time"] >= 1]
        vid_num_reliable_df = vid_num_reliable_df.loc[vid_num_reliable_df["Time"] >= 1]
        dl_num_reliable_df = dl_num_reliable_df.loc[dl_num_reliable_df["Time"] >= 1]

        # Get the first occurence of failure (actual) and detected failure (if any)
        # Downlink
        dl_num_reliable_df["Failure"] = dl_num_reliable_df["Measured_Reliability_1"] < reliability_th 
        dl_num_reliable_df["Detection"] = dl_num_reliable_df["Num_Reliable"] < dl_num_reliable_th 
        dl_fail = dl_num_reliable_df.loc[dl_num_reliable_df["Failure"]]
        dl_fail.sort_values(by=["Time"], inplace=True)
        if len(dl_fail) > 0:
            dl_fail_time = dl_fail["Time"].values[0]
        else:
            dl_fail_time = np.inf
        dl_detected = dl_num_reliable_df.loc[dl_num_reliable_df["Detection"]] # Get detected rows
        dl_detected.sort_values(by=["Time"], inplace=True)
        if len(dl_detected) > 0:
            dl_detect_time = dl_detected["Time"].values[0]
        else:
            dl_detect_time = np.inf

        # Uplink
        ul_num_reliable_df["Failure"] = ul_num_reliable_df["Measured_Reliability_1"] < reliability_th 
        ul_num_reliable_df["Detection"] = ul_num_reliable_df["Num_Reliable"] < ul_num_reliable_th 
        ul_fail = ul_num_reliable_df.loc[ul_num_reliable_df["Failure"]]
        ul_fail.sort_values(by=["Time"], inplace=True)
        if len(ul_fail) > 0:
            ul_fail_time = ul_fail["Time"].values[0]
        else:
            ul_fail_time = np.inf
        ul_detected = ul_num_reliable_df.loc[ul_num_reliable_df["Detection"]] # Get detected rows
        ul_detected.sort_values(by=["Time"], inplace=True)
        if len(ul_detected) > 0:
            ul_detect_time = ul_detected["Time"].values[0]
        else:
            ul_detect_time = np.inf

        # Video
        vid_num_reliable_df["Failure"] = vid_num_reliable_df["Measured_Reliability_1"] < reliability_th 
        vid_num_reliable_df["Detection"] = vid_num_reliable_df["Num_Reliable"] < vid_num_reliable_th 
        vid_fail = vid_num_reliable_df.loc[vid_num_reliable_df["Failure"]]
        vid_fail.sort_values(by=["Time"], inplace=True)
        if len(vid_fail) > 0:
            vid_fail_time = vid_fail["Time"].values[0]
        else:
            vid_fail_time = np.inf
        vid_detected = vid_num_reliable_df.loc[vid_num_reliable_df["Detection"]] # Get detected rows
        vid_detected.sort_values(by=["Time"], inplace=True)
        if len(vid_detected) > 0:
            vid_detect_time = vid_detected["Time"].values[0]
        else:
            vid_detect_time = np.inf
        
        time_fail = np.min([dl_fail_time, ul_fail_time, vid_fail_time])
        time_detect = np.min([dl_detect_time, ul_detect_time, vid_detect_time])

        if dl_fail_time == time_fail:
            dl_fail_first = 1
        else:
            dl_fail_first = 0
        if ul_fail_time == time_fail:
            ul_fail_first = 1
        else:
            ul_fail_first = 0
        if vid_fail_time == time_fail:
            vid_fail_first = 1
        else:
            vid_fail_first = 0
        
        if dl_detect_time == time_detect:
            dl_detect_first = 1
        else:
            dl_detect_first = 0
        if ul_detect_time == time_detect:
            ul_detect_first = 1
        else:
            ul_detect_first = 0
        if vid_detect_time == time_detect:
            vid_detect_first = 1
        else:
            vid_detect_first = 0

        result = {"Scenario": scenario_name, "Run_Num": run_num, "DL_Fail_First": dl_fail_first, "UL_Fail_First": ul_fail_first, "VID_Fail_First": vid_fail_first,
                "DL_Detect_First": dl_detect_first, "UL_Detect_First": ul_detect_first, "VID_Detect_First": vid_detect_first,
                "DL_Fail_Time": dl_fail_time, "UL_Fail_Time": ul_fail_time, "Vid_Fail_Time": vid_fail_time, 
                "DL_Detect_Time": dl_detect_time, "UL_Detect_Time": ul_detect_time, "Vid_Detect_Time": vid_detect_time}
    return(result)

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    ''' Define Paths Here'''
    DATASET_INT_PATHS = ["/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_0_processed", 
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_1_processed",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_2_processed",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_0_processed",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_1_processed",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_2_processed"]
    SAVE_PATH = "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/first_fail_n_detection"
    SAVE_FILES = ["uav_interference_scenario_0_results.csv", "uav_interference_scenario_1_results.csv", "uav_interference_scenario_2_results.csv", 
                      "manet_interference_scenario_0_results.csv", "manet_interference_scenario_1_results.csv", "manet_interference_scenario_2_results.csv"]
    RELIABILITY_TH = 0.999 # Threshold to evaluate interference scenarios
    NUM_PROCS = 32
    n_r = {"USI": [10, 20, 66.7, 100], "DL_Min_Num_Reliable": [49, 49, 49, 49], "UL_Min_Num_Reliable": [744, 376, 115, 78], "VID_Min_Num_Reliable": [164, 164, 164, 164]}
    n_r_df = pd.DataFrame(n_r)
    counter = 0
    for datasets in DATASET_INT_PATHS:
        print(datasets.split("/")[-1])
        scenarios_int = [x[0] for x in os.walk(datasets) if (os.path.isdir(x[0]) and x[0]!=datasets)]
        scenario_results = []
        for scenario in tqdm(scenarios_int):
            # Get the parameters of the scenario
            scenario_name = scenario.split("/")[-1]
            params = scenario_name.split("_")
            usi = [x for x in params if "UAVSendingInterval" in x][0].split('-')[-1]
            
            # Get the min throughput of each link for the corresponding USI
            ul_num_reliable_th = n_r_df.loc[n_r_df["USI"] == float(usi)]["UL_Min_Num_Reliable"].values[0]
            dl_num_reliable_th = n_r_df.loc[n_r_df["USI"] == float(usi)]["DL_Min_Num_Reliable"].values[0]
            vid_num_reliable_th = n_r_df.loc[n_r_df["USI"] == float(usi)]["VID_Min_Num_Reliable"].values[0]
            
            # Each run should only have one uplink throughput, so use it to determine no. of runs
            uplink_num_reliable_files = glob.glob(os.path.join(scenario, "Run-*_Uplink_Throughput.csv"))
            num_runs = len(uplink_num_reliable_files)
            run_nums = [i for i in range(num_runs)]
            with Pool(NUM_PROCS) as pool:
                for result in pool.starmap(sim_run_get_detection, zip(run_nums, repeat(scenario), repeat(RELIABILITY_TH), repeat(dl_num_reliable_th), repeat(ul_num_reliable_th), repeat(vid_num_reliable_th))):
                    scenario_results.append(result)
        
        scenario_results_df = pd.DataFrame(scenario_results)
        scenario_results_df.to_csv(os.path.join(SAVE_PATH, SAVE_FILES[counter]))
        counter += 1