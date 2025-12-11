# Date: 14/08/2024
# Desc: To test OCSVM accuracy in no interference / UAV interference / MANET interference
#       Tests each run individually instead of combining all test throughput samples
# NOTE: Each run is simulated up to d_max. No filtering by hdist needed

import pandas as pd
import numpy as np 
import os
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from multiprocessing.pool import Pool
from itertools import repeat

def get_MCS_index(mcs_bitrate):
    mcs_index = {6.5: 0, 13: 1, 19.5: 2, 26: 3, 39: 4, 52: 5, 58.5: 6, 65: 7}
    return mcs_index[mcs_bitrate]

def test_no_int_run(run_num, scenario, reliability_th, dl_throughput_th, ul_throughput_th, vid_throughput_th):
    # print(os.path.join(scenario, "Run-{}_Uplink_Throughput.csv".format(run_num)).split("/")[-1])
    # Load throughput data
    ul_throughput_df = pd.read_csv(os.path.join(scenario, "Run-{}_Uplink_Throughput.csv".format(run_num)))
    vid_throughput_df = pd.read_csv(os.path.join(scenario, "Run-{}_Video_Throughput.csv".format(run_num)))
    dl_throughput_list = glob.glob(os.path.join(scenario, "Run-{}*_Downlink_Throughput.csv".format(run_num)))
    dl_df_list = []
    for file in dl_throughput_list:
        df = pd.read_csv(file)
        dl_df_list.append(df)
    dl_throughput_df = pd.concat(dl_df_list)

    # Filter data with time before 1
    ul_throughput_df = ul_throughput_df.loc[ul_throughput_df["Time"] >= 1]
    vid_throughput_df = vid_throughput_df.loc[vid_throughput_df["Time"] >= 1]
    dl_throughput_df = dl_throughput_df.loc[dl_throughput_df["Time"] >= 1]

    # Testing Throughput Threshold
    # For no interference case, filter throughput samples where measured_throughput < reliability threshold
    # Downlink
    dl_throughput_df_reliable = dl_throughput_df.loc[dl_throughput_df["Measured_Reliability"] >= reliability_th]
    # if len(dl_throughput_df_reliable) < len(dl_throughput_df):
    #     print("Samples with measured reliability < threshold detected in scenario: {}".format(scenario_name))
    #     print("Num samples with measured reliability < threshold: {}".format(len(dl_throughput_df) - len(dl_throughput_df_reliable)))
    dl_y_pred_outlier = dl_throughput_df_reliable["Throughput"] < dl_throughput_th 
    dl_num_outlier = dl_y_pred_outlier.sum() # This will sum the number of True elements
    dl_num_samples = len(dl_throughput_df_reliable)
    dl_accuracy_score = 1 - dl_num_outlier / dl_num_samples # Ground truth is all 1 for normal test data
    if dl_num_outlier > 0: 
        dl_int_detected = 1
    else:
        dl_int_detected = 0
    dl_results = {"run_num": run_num, "dl_accuracy_score": dl_accuracy_score, "dl_num_samples": dl_num_samples, "dl_int_detected": dl_int_detected}
    # Uplink
    ul_throughput_df_reliable = ul_throughput_df.loc[ul_throughput_df["Measured_Reliability"] >= reliability_th]
    # if len(ul_throughput_df_reliable) < len(ul_throughput_df):
    #     print("Samples with measured reliability < threshold detected in scenario: {}".format(scenario_name))
    ul_y_pred_outlier = ul_throughput_df_reliable["Throughput"] < ul_throughput_th 
    ul_num_outlier = ul_y_pred_outlier.sum() # This will sum the number of True elements
    ul_num_samples = len(ul_throughput_df_reliable)
    ul_accuracy_score = 1 - ul_num_outlier / ul_num_samples # Ground truth is all 1 for normal test data
    if ul_num_outlier > 0: 
        ul_int_detected = 1
    else:
        ul_int_detected = 0
    ul_results = {"run_num": run_num, "ul_accuracy_score": ul_accuracy_score, "ul_num_samples": ul_num_samples, "ul_int_detected": ul_int_detected}
    # Video
    vid_throughput_df_reliable = vid_throughput_df.loc[vid_throughput_df["Measured_Reliability"] >= reliability_th]
    # if len(vid_throughput_df_reliable) < len(vid_throughput_df):
    #     print("Samples with measured reliability < threshold detected in scenario: {}".format(scenario_name))
    vid_y_pred_outlier = vid_throughput_df_reliable["Throughput"] < vid_throughput_th 
    vid_num_outlier = vid_y_pred_outlier.sum() # This will sum the number of True elements
    vid_num_samples = len(vid_throughput_df_reliable)
    vid_accuracy_score = 1 - vid_num_outlier / vid_num_samples # Ground truth is all 1 for normal test data
    if vid_num_outlier > 0: 
        vid_int_detected = 1
    else:
        vid_int_detected = 0
    vid_results = {"run_num": run_num, "vid_accuracy_score": vid_accuracy_score, "vid_num_samples": vid_num_samples, "vid_int_detected": vid_int_detected}

    # Check for overall interference detection
    overall_int_detected = int(dl_int_detected == 1 or ul_int_detected == 1 or vid_int_detected == 1)
    overall_results = {"scenario": scenario, "run_num": run_num, "overall_int_detected": overall_int_detected}

    return(dl_results, ul_results, vid_results, overall_results)

def test_int_run(run_num, scenario, reliability_th, dl_throughput_th, ul_throughput_th, vid_throughput_th):
    # Check the reliability of all links
    reliability_df = pd.read_csv(os.path.join(scenario, "Simulation_Results.csv"))
    run_reliability_df = reliability_df.loc[reliability_df["Run"] == run_num]
    reliable_check = int((run_reliability_df["Total_Reliability_DL"].values[0] >= reliability_th) and (run_reliability_df["Total_Reliability_UL"].values[0] >= reliability_th) and (run_reliability_df["Total_Reliability_VID"].values[0] >= reliability_th))
    # Load throughput data
    ul_throughput_df = pd.read_csv(os.path.join(scenario, "Run-{}_Uplink_Throughput.csv".format(run_num)))
    vid_throughput_df = pd.read_csv(os.path.join(scenario, "Run-{}_Video_Throughput.csv".format(run_num)))
    dl_throughput_list = glob.glob(os.path.join(scenario, "Run-{}*_Downlink_Throughput.csv".format(run_num)))
    dl_df_list = []
    for file in dl_throughput_list:
        df = pd.read_csv(file)
        dl_df_list.append(df)
    dl_throughput_df = pd.concat(dl_df_list)

    # Filter data with time before 1
    ul_throughput_df = ul_throughput_df.loc[ul_throughput_df["Time"] >= 1]
    vid_throughput_df = vid_throughput_df.loc[vid_throughput_df["Time"] >= 1]
    dl_throughput_df = dl_throughput_df.loc[dl_throughput_df["Time"] >= 1]

    # Testing Throughput Threshold
    # For no interference case, filter throughput samples where measured_throughput < reliability threshold
    # Downlink
    dl_y_pred_outlier = dl_throughput_df["Throughput"] < dl_throughput_th 
    dl_num_outlier = dl_y_pred_outlier.sum() # This will sum the number of True elements
    dl_num_samples = len(dl_throughput_df)
    dl_percent_outlier = dl_num_outlier / dl_num_samples # Ground truth is all 1 for normal test data
    dl_int_detected = int(dl_num_outlier > 0)
    dl_results = {"run_num": run_num, "dl_percent_outlier": dl_percent_outlier, "dl_num_samples": dl_num_samples, "dl_int_detected": dl_int_detected}
    # Uplink
    ul_y_pred_outlier = ul_throughput_df["Throughput"] < ul_throughput_th 
    ul_num_outlier = ul_y_pred_outlier.sum() # This will sum the number of True elements
    ul_num_samples = len(ul_throughput_df)
    ul_percent_outlier = ul_num_outlier / ul_num_samples # Ground truth is all 1 for normal test data
    ul_int_detected = int(ul_num_outlier > 0)
    ul_results = {"run_num": run_num, "ul_percent_outlier": ul_percent_outlier, "ul_num_samples": ul_num_samples, "ul_int_detected": ul_int_detected}
    # Video
    vid_y_pred_outlier = vid_throughput_df["Throughput"] < vid_throughput_th 
    vid_num_outlier = vid_y_pred_outlier.sum() # This will sum the number of True elements
    vid_num_samples = len(vid_throughput_df)
    vid_percent_outlier = vid_num_outlier / vid_num_samples # Ground truth is all 1 for normal test data
    vid_int_detected = int(vid_num_outlier > 0)
    vid_results = {"run_num": run_num, "vid_percent_outlier": vid_percent_outlier, "vid_num_samples": vid_num_samples, "vid_int_detected": vid_int_detected}

    # Check for overall interference detection
    overall_int_detected = int(dl_int_detected == 1 or ul_int_detected == 1 or vid_int_detected == 1)
    dl_vid_int_detected = int(dl_int_detected == 1 or vid_int_detected == 1)
    ul_vid_int_detected = int(ul_int_detected == 1 or vid_int_detected == 1)
    dl_ul_int_detected = int(dl_int_detected == 1 or ul_int_detected == 1)
    overall_results = {"Scenario": scenario, "run_num": run_num, "Run_Reliability": reliable_check, "Overall_Int_Detected": overall_int_detected, "DL_VID_Int_Detected": dl_vid_int_detected, "UL_VID_Int_Detected": ul_vid_int_detected, "DL_UL_Int_Detected": dl_ul_int_detected, 
                "Total_Reliability_DL": run_reliability_df["Total_Reliability_DL"].values[0], "Total_Reliability_UL": run_reliability_df["Total_Reliability_UL"].values[0], "Total_Reliability_VID": run_reliability_df["Total_Reliability_VID"].values[0]}
    
    return(dl_results, ul_results, vid_results, overall_results)

if __name__ == "__main__":
    # pandarallel.initialize(progress_bar=False)
    ''' Define Paths Here'''
    MIN_THROUGHPUT_FILE = "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/data_manual_throughput_min_max_21102024.csv"
    DATASET_NO_INT_PATH = "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_no_int_test_processed"
    DATASET_INT_PATHS = ["/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_0_processed", 
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_1_processed",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_2_processed",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_0_processed",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_1_processed",
                         "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_2_processed"]
    SAVE_PATH = "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/throughput_failure_detection_results_v2"
    SAVE_FILE_NO_INT = "throughput_no_int_test_results.csv"
    SAVE_FILE_NO_INT_FP = "throughput_no_int_fp.csv"
    SAVE_FILES_INT = ["throughput_uav_interference_scenario_0_results.csv", "throughput_uav_interference_scenario_1_results.csv", "throughput_uav_interference_scenario_2_results.csv", 
                      "throughput_manet_interference_scenario_0_results.csv", "throughput_manet_interference_scenario_1_results.csv", "throughput_manet_interference_scenario_2_results.csv"]
    SAVE_FILES_CONF_MAT = ["throughput_uav_interference_scenario_0_conf_mat.csv", "throughput_uav_interference_scenario_1_conf_mat.csv", "throughput_uav_interference_scenario_2_conf_mat.csv", 
                      "throughput_manet_interference_scenario_0_conf_mat.csv", "throughput_manet_interference_scenario_1_conf_mat.csv", "throughput_manet_interference_scenario_2_conf_mat.csv"]
    SAVE_FILES_INT_FN = ["throughput_uav_interference_scenario_0_fn.csv", "throughput_uav_interference_scenario_1_fn.csv", "throughput_uav_interference_scenario_2_fn.csv", 
                         "throughput_manet_interference_scenario_0_fn.csv", "throughput_manet_interference_scenario_1_fn.csv", "throughput_manet_interference_scenario_2_fn.csv"]
    SAVE_FILES_INT_FP = ["throughput_uav_interference_scenario_0_fp.csv", "throughput_uav_interference_scenario_1_fp.csv", "throughput_uav_interference_scenario_2_fp.csv", 
                         "throughput_manet_interference_scenario_0_fp.csv", "throughput_manet_interference_scenario_1_fp.csv", "throughput_manet_interference_scenario_2_fp.csv"]
    RELIABILITY_TH = 0.999 # Threshold to evaluate interference scenarios
    NUM_PROCS = 32
    min_throughput_th_df = pd.read_csv(MIN_THROUGHPUT_FILE)
    # DL_THROUGHPUT_TH = 1007
    # VID_THROUGHPUT_TH = 174000

    # """ NO INT """
    print("Testing No Int Scenarios")
    scenarios_no_int = [x[0] for x in os.walk(DATASET_NO_INT_PATH) if (os.path.isdir(x[0]) and x[0]!=DATASET_NO_INT_PATH)]
    scenario_results = []
    false_pos_df_list = []
    for scenario in tqdm(scenarios_no_int):
        if scenario == DATASET_NO_INT_PATH: # This is the root path
            continue
        # Get the parameters of the scenario
        scenario_name = scenario.split("/")[-1]
        params = scenario_name.split("_")
        usi = [x for x in params if "UAVSendingInterval" in x][0].split('-')[-1]
        speed = int([x for x in params if "UAVSpeed" in x][0].split('-')[-1])
        hdist = int([x for x in params if "Distance" in x][0].split('-')[-1])
        if hdist < speed: # Distance too short for at least 1 s simulation time
            continue
        # Get the min throughput of each link for the corresponding USI
        ul_throughput_th = min_throughput_th_df.loc[min_throughput_th_df["USI"] == float(usi)]["UL_Min_Throughput"].values[0]
        dl_throughput_th = min_throughput_th_df.loc[min_throughput_th_df["USI"] == float(usi)]["DL_Min_Throughput"].values[0]
        vid_throughput_th = min_throughput_th_df.loc[min_throughput_th_df["USI"] == float(usi)]["VID_Min_Throughput"].values[0]
        # dl_throughput_th = DL_THROUGHPUT_TH
        # vid_throughput_th = VID_THROUGHPUT_TH

        # Each run should only have one uplink throughput, so use it to determine no. of runs
        uplink_throughput_files = glob.glob(os.path.join(scenario, "Run-*_Uplink_Throughput.csv"))
        num_runs = len(uplink_throughput_files)
        run_nums = [i for i in range(num_runs)]
        
        # Process each run
        ul_run_results = []
        dl_run_results = []
        vid_run_results = []
        runs_int_detected = [] # To record runs with false positive

        with Pool(NUM_PROCS) as pool:
            for result in pool.starmap(test_no_int_run, zip(run_nums, repeat(scenario), repeat(RELIABILITY_TH), 
                                                            repeat(dl_throughput_th), repeat(ul_throughput_th), repeat(vid_throughput_th))):
                dl_run_results.append(result[0])
                ul_run_results.append(result[1])
                vid_run_results.append(result[2])
                runs_int_detected.append(result[3])

        # Record run results
        dl_run_results_df = pd.DataFrame(dl_run_results)
        dl_acc_avg = dl_run_results_df["dl_accuracy_score"].mean()
        dl_acc_max = dl_run_results_df["dl_accuracy_score"].max()
        dl_acc_min = dl_run_results_df["dl_accuracy_score"].min()
        dl_acc_std_dev = dl_run_results_df["dl_accuracy_score"].std()
        dl_num_samples_avg = dl_run_results_df["dl_num_samples"].mean()
        dl_percent_int_detected = dl_run_results_df["dl_int_detected"].sum() / len(dl_run_results_df)
        dl_run_results_df["dl_num_accuracy_samples"] = dl_run_results_df["dl_accuracy_score"] * dl_run_results_df["dl_num_samples"]
        dl_overall_acc = dl_run_results_df["dl_num_accuracy_samples"].sum() / dl_run_results_df["dl_num_samples"].sum() # Overall accuracy considers all samples from all runs

        ul_run_results_df = pd.DataFrame(ul_run_results)
        ul_acc_avg = ul_run_results_df["ul_accuracy_score"].mean()
        ul_acc_max = ul_run_results_df["ul_accuracy_score"].max()
        ul_acc_min = ul_run_results_df["ul_accuracy_score"].min()
        ul_acc_std_dev = ul_run_results_df["ul_accuracy_score"].std()
        ul_num_samples_avg = ul_run_results_df["ul_num_samples"].mean()
        ul_percent_int_detected = ul_run_results_df["ul_int_detected"].sum() / len(ul_run_results_df)
        ul_run_results_df["ul_num_accuracy_samples"] = ul_run_results_df["ul_accuracy_score"] * ul_run_results_df["ul_num_samples"]
        ul_overall_acc = ul_run_results_df["ul_num_accuracy_samples"].sum() / ul_run_results_df["ul_num_samples"].sum() # Overall accuracy considers all samples from all runs

        vid_run_results_df = pd.DataFrame(vid_run_results)
        vid_acc_avg = vid_run_results_df["vid_accuracy_score"].mean()
        vid_acc_max = vid_run_results_df["vid_accuracy_score"].max()
        vid_acc_min = vid_run_results_df["vid_accuracy_score"].min()
        vid_acc_std_dev = vid_run_results_df["vid_accuracy_score"].std()
        vid_num_samples_avg = vid_run_results_df["vid_num_samples"].mean()
        vid_percent_int_detected = vid_run_results_df["vid_int_detected"].sum() / len(vid_run_results_df)
        vid_run_results_df["vid_num_accuracy_samples"] = vid_run_results_df["vid_accuracy_score"] * vid_run_results_df["vid_num_samples"]
        vid_overall_acc = vid_run_results_df["vid_num_accuracy_samples"].sum() / vid_run_results_df["vid_num_samples"].sum() # Overall accuracy considers all samples from all runs

        runs_int_detected_df = pd.DataFrame(runs_int_detected)
        false_pos_df = runs_int_detected_df.loc[runs_int_detected_df["overall_int_detected"] == 1]
        if not false_pos_df.empty:
            false_pos_df_list.append(false_pos_df)
        overall_percent_int_detected = len(false_pos_df) / len(uplink_throughput_files)

        # Record scenario results
        scenario_results.append({"Scenario": scenario_name, "Num_Runs": len(uplink_throughput_files), "Overall_Percent_Int_Detected": overall_percent_int_detected,
                                 "DL_Overall_Accuracy": dl_overall_acc, "DL_Accuracy_Avg": dl_acc_avg, "DL_Accuracy_Max": dl_acc_max, "DL_Accuracy_Min": dl_acc_min, "DL_Accuracy_Std_Dev": dl_acc_std_dev, "DL_Num_Samples_Avg": dl_num_samples_avg, "DL_Int_Detected_Percent": dl_percent_int_detected,
                                 "UL_Overall_Accuracy": ul_overall_acc, "UL_Accuracy_Avg": ul_acc_avg, "UL_Accuracy_Max": ul_acc_max, "UL_Accuracy_Min": ul_acc_min, "UL_Accuracy_Std_Dev": ul_acc_std_dev, "UL_Num_Samples_Avg": ul_num_samples_avg, "UL_Int_Detected_Percent": ul_percent_int_detected,
                                 "VID_Overall_Accuracy": vid_overall_acc, "VID_Accuracy_Avg": vid_acc_avg, "VID_Accuracy_Max": vid_acc_max, "VID_Accuracy_Min": vid_acc_min, "VID_Accuracy_Std_Dev": vid_acc_std_dev, "VID_Num_Samples_Avg": vid_num_samples_avg, "VID_Int_Detected_Percent": vid_percent_int_detected})

    # Save results
    scenario_results_df = pd.DataFrame(scenario_results)
    scenario_results_df.to_csv(os.path.join(SAVE_PATH, SAVE_FILE_NO_INT))
    if len(false_pos_df_list) > 0:
        false_pos_DF = pd.concat(false_pos_df_list)
        false_pos_DF.to_csv(os.path.join(SAVE_PATH, SAVE_FILE_NO_INT_FP))

    ''' INT SCENARIOS '''
    ''' Filter out runs where reliability is above the threshold '''
    # print("Testing Int Scenarios")
    # counter = 0
    # for datasets in DATASET_INT_PATHS:
    #     print(datasets.split("/")[-1])
    #     scenarios_int = [x[0] for x in os.walk(datasets) if (os.path.isdir(x[0]) and x[0]!=datasets)]
    #     scenario_results = []
    #     conf_mat_results = []
    #     false_neg_list = [] # To store details of runs with false negative detections
    #     false_pos_list = [] # To store details of runs with false positive detections
    #     for scenario in tqdm(scenarios_int):
    #         if scenario == datasets: # This is the root path
    #             continue
    #         # Get the parameters of the scenario
    #         scenario_name = scenario.split("/")[-1]
    #         params = scenario_name.split("_")
    #         usi = [x for x in params if "UAVSendingInterval" in x][0].split('-')[-1]
    #         speed = int([x for x in params if "UAVSpeed" in x][0].split('-')[-1])
    #         hdist = int([x for x in params if "Distance" in x][0].split('-')[-1])
    #         if hdist < speed: # Distance too short for at least 1 s simulation time
    #             continue
    #         # Get the min throughput of each link for the corresponding USI
    #         ul_throughput_th = min_throughput_th_df.loc[min_throughput_th_df["USI"] == float(usi)]["UL_Min_Throughput"].values[0]
    #         dl_throughput_th = min_throughput_th_df.loc[min_throughput_th_df["USI"] == float(usi)]["DL_Min_Throughput"].values[0]
    #         vid_throughput_th = min_throughput_th_df.loc[min_throughput_th_df["USI"] == float(usi)]["VID_Min_Throughput"].values[0]
    #         # dl_throughput_th = DL_THROUGHPUT_TH
    #         # vid_throughput_th = VID_THROUGHPUT_TH
            
    #         # Each run should only have one uplink throughput, so use it to determine no. of runs
    #         uplink_throughput_files = glob.glob(os.path.join(scenario, "Run-*_Uplink_Throughput.csv"))
    #         num_runs = len(uplink_throughput_files)
    #         run_nums = [i for i in range(num_runs)]

    #         # Process each run
    #         ul_run_results = []
    #         dl_run_results = []
    #         vid_run_results = []
    #         overall_run_results = []

    #         with Pool(NUM_PROCS) as pool:
    #             for result in pool.starmap(test_int_run, zip(run_nums, repeat(scenario), repeat(RELIABILITY_TH), repeat(dl_throughput_th), repeat(ul_throughput_th), repeat(vid_throughput_th))):
    #                 dl_run_results.append(result[0])
    #                 ul_run_results.append(result[1])
    #                 vid_run_results.append(result[2])
    #                 overall_run_results.append(result[3])  
            
    #         # Record run results
    #         overall_run_results_df = pd.DataFrame(overall_run_results)
    #         overall_percent_int_detected = np.sum(overall_run_results_df["Overall_Int_Detected"].sum()) / len(overall_run_results_df)
    #         overall_accuracy = accuracy_score(1-overall_run_results_df["Run_Reliability"].to_numpy(), overall_run_results_df["Overall_Int_Detected"].to_numpy())
    #         dl_vid_accuracy = accuracy_score(1-overall_run_results_df["Run_Reliability"].to_numpy(), overall_run_results_df["DL_VID_Int_Detected"].to_numpy())
    #         ul_vid_accuracy = accuracy_score(1-overall_run_results_df["Run_Reliability"].to_numpy(), overall_run_results_df["UL_VID_Int_Detected"].to_numpy())
    #         dl_ul_accuracy = accuracy_score(1-overall_run_results_df["Run_Reliability"].to_numpy(), overall_run_results_df["DL_UL_Int_Detected"].to_numpy())
    #         percent_reliable_runs = len(overall_run_results_df.loc[overall_run_results_df["Run_Reliability"] == 1]) / len(overall_run_results_df)

    #         dl_run_results_df = pd.DataFrame(dl_run_results)
    #         dl_percent_outlier_avg = dl_run_results_df["dl_percent_outlier"].mean()
    #         dl_num_samples_avg = dl_run_results_df["dl_num_samples"].mean()
    #         dl_num_int_detected = dl_run_results_df["dl_int_detected"].sum()
    #         dl_accuraccy = accuracy_score(overall_run_results_df["Run_Reliability"].to_numpy(), 1-dl_run_results_df["dl_int_detected"].to_numpy())

    #         ul_run_results_df = pd.DataFrame(ul_run_results)
    #         ul_percent_outlier_avg = ul_run_results_df["ul_percent_outlier"].mean()
    #         ul_num_samples_avg = ul_run_results_df["ul_num_samples"].mean()
    #         ul_num_int_detected = ul_run_results_df["ul_int_detected"].sum() / len(ul_run_results_df)
    #         ul_accuraccy = accuracy_score(overall_run_results_df["Run_Reliability"].to_numpy(), 1-ul_run_results_df["ul_int_detected"].to_numpy())

    #         vid_run_results_df = pd.DataFrame(vid_run_results)
    #         vid_percent_outlier_avg = vid_run_results_df["vid_percent_outlier"].mean()
    #         vid_num_samples_avg = vid_run_results_df["vid_num_samples"].mean()
    #         vid_num_int_detected = vid_run_results_df["vid_int_detected"].sum() 
    #         vid_accuraccy = accuracy_score(overall_run_results_df["Run_Reliability"].to_numpy(), 1-vid_run_results_df["vid_int_detected"].to_numpy())

    #         avg_dl_reliability = overall_run_results_df["Total_Reliability_DL"].mean()
    #         avg_ul_reliability = overall_run_results_df["Total_Reliability_UL"].mean()
    #         avg_vid_reliability = overall_run_results_df["Total_Reliability_VID"].mean()

    #         # Record scenario results
    #         scenario_results.append({"Scenario": scenario_name, "Num_Runs": len(uplink_throughput_files), "Overall_Accuracy": overall_accuracy, "Overall_Percent_Int_Detected": overall_percent_int_detected,
    #                                 "DL_VID_Accuracy": dl_vid_accuracy, "UL_VID_Accuracy": ul_vid_accuracy, "DL_UL_Accuracy": dl_ul_accuracy, "Percent_Runs_Reliable": percent_reliable_runs, 
    #                                 "DL_Percent_Outlier_Avg": dl_percent_outlier_avg, "DL_Num_Samples_Avg": dl_num_samples_avg, "DL_Num_Int_Detected": dl_num_int_detected, "DL_Accuracy": dl_accuraccy, "DL_Link_Reliability_Avg": avg_dl_reliability,
    #                                 "UL_Percent_Outlier_Avg": ul_percent_outlier_avg, "UL_Num_Samples_Avg": ul_num_samples_avg, "UL_Num_Int_Detected": ul_num_int_detected, "UL_Accuracy": ul_accuraccy, "UL_Link_Reliability_Avg": avg_ul_reliability,
    #                                 "VID_Percent_Outlier_Avg": vid_percent_outlier_avg, "VID_Num_Samples_Avg": vid_num_samples_avg, "VID_Num_Int_Detected": vid_num_int_detected, "VID_Accuracy": vid_accuraccy, "VID_Link_Reliability_Avg": avg_vid_reliability,})

    #         # Record TN, FP, FN, TP
    #         overall_tn, overall_fp, overall_fn, overall_tp = confusion_matrix(1-overall_run_results_df["Run_Reliability"].to_numpy(), overall_run_results_df["Overall_Int_Detected"].to_numpy(), labels=[0,1]).ravel()
    #         dl_vid_tn, dl_vid_fp, dl_vid_fn, dl_vid_tp = confusion_matrix(1-overall_run_results_df["Run_Reliability"].to_numpy(), overall_run_results_df["DL_VID_Int_Detected"].to_numpy(), labels=[0,1]).ravel()
    #         ul_vid_tn, ul_vid_fp, ul_vid_fn, ul_vid_tp = confusion_matrix(1-overall_run_results_df["Run_Reliability"].to_numpy(), overall_run_results_df["UL_VID_Int_Detected"].to_numpy(), labels=[0,1]).ravel()
    #         dl_ul_tn, dl_ul_fp, dl_ul_fn, dl_ul_tp = confusion_matrix(1-overall_run_results_df["Run_Reliability"].to_numpy(), overall_run_results_df["DL_UL_Int_Detected"].to_numpy(), labels=[0,1]).ravel()
    #         dl_tn, dl_fp, dl_fn, dl_tp = confusion_matrix(1-overall_run_results_df["Run_Reliability"].to_numpy(), dl_run_results_df["dl_int_detected"].to_numpy(), labels=[0,1]).ravel()
    #         ul_tn, ul_fp, ul_fn, ul_tp = confusion_matrix(1-overall_run_results_df["Run_Reliability"].to_numpy(), ul_run_results_df["ul_int_detected"].to_numpy(), labels=[0,1]).ravel()
    #         vid_tn, vid_fp, vid_fn, vid_tp = confusion_matrix(1-overall_run_results_df["Run_Reliability"].to_numpy(), vid_run_results_df["vid_int_detected"].to_numpy(), labels=[0,1]).ravel()
    #         conf_mat_results.append({"Scenario": scenario_name, "Num_Runs": len(uplink_throughput_files), 
    #                                 "Overall_TN": overall_tn, "Overall_FP": overall_fp, "Overall_FN": overall_fn, "Overall_TP": overall_tp,
    #                                 "DL_VID_TN": dl_vid_tn, "DL_VID_FP": dl_vid_fp, "DL_VID_FN": dl_vid_fn, "DL_VID_TP": dl_vid_tp,
    #                                 "UL_VID_TN": ul_vid_tn, "UL_VID_FP": ul_vid_fp, "UL_VID_FN": ul_vid_fn, "UL_VID_TP": ul_vid_tp,
    #                                 "DL_UL_TN": dl_ul_tn, "DL_UL_FP": dl_ul_fp, "DL_UL_FN": dl_ul_fn, "DL_UL_TP": dl_ul_tp,
    #                                 "DL_TN": dl_tn, "DL_FP": dl_fp, "DL_FN": dl_fn, "DL_TP": dl_tp,
    #                                 "UL_TN": ul_tn, "UL_FP": ul_fp, "UL_FN": ul_fn, "UL_TP": ul_tp,
    #                                 "VID_TN": vid_tn, "VID_FP": vid_fp, "VID_FN": vid_fn, "VID_TP": vid_tp})
    #         # Record false negative info
    #         false_neg_df = overall_run_results_df.loc[(overall_run_results_df["Run_Reliability"]==0) & (overall_run_results_df["Overall_Int_Detected"]==0)]
    #         if not false_neg_df.empty:
    #             false_neg_list.append(false_neg_df)

    #         # Record false positive info
    #         false_pos_df = overall_run_results_df.loc[(overall_run_results_df["Run_Reliability"]==1) & (overall_run_results_df["Overall_Int_Detected"]==1)]
    #         if not false_pos_df.empty:
    #             false_pos_list.append(false_pos_df)
        
    #     # Save results
    #     scenario_results_df = pd.DataFrame(scenario_results)
    #     scenario_results_df.to_csv(os.path.join(SAVE_PATH, SAVE_FILES_INT[counter]))
    #     conf_mat_results_df = pd.DataFrame(conf_mat_results)
    #     conf_mat_results_df.to_csv(os.path.join(SAVE_PATH, SAVE_FILES_CONF_MAT[counter]))
    #     if len(false_neg_list) > 0:
    #         fail_neg_df = pd.concat(false_neg_list)
    #         fail_neg_df.to_csv(os.path.join(SAVE_PATH, SAVE_FILES_INT_FN[counter]))
    #     if len(false_pos_list) > 0:
    #         fail_pos_df = pd.concat(false_pos_list)
    #         fail_pos_df.to_csv(os.path.join(SAVE_PATH, SAVE_FILES_INT_FP[counter]))
    #     counter += 1
