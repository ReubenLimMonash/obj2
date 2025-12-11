'''
Date: 20/01/2024
Desc: To check that the switching of all links (for those that switches) happen within 1 second (between min and max).
      We don't check that switching occurs in all links, since link(s) might not switch due to the sim time limit being reached.
      Modified after fanet_data_preprocessing_mitigation_rebroadcasting_08112024.py
'''

import pandas as pd # for data manipulation 
import numpy as np
from multiprocessing.pool import Pool
from itertools import repeat
import os, glob


def process_scenario(scenario_path, switch_time_path, num_processes, run_num):
    scenario = scenario_path.split("/")[-1]
    print(scenario)

    scenario_params = scenario.split('_')
    usi = float([x for x in scenario_params if "UAVSendingInterval" in x][0].split('-')[-1])
    uav_speed = float([x for x in scenario_params if "UAVSpeed" in x][0].split('-')[-1])
    height = float([x for x in scenario_params if "Height" in x][0].split('-')[-1])
    bitrate = float([x for x in scenario_params if "BitRate" in x][0].split('-')[-1])

    anomaly_list = [] # To store anomalous cases
    switch_time_df = pd.read_csv(switch_time_path)
    switch_time_df = switch_time_df.loc[(switch_time_df["Height"]==height)&(switch_time_df["USI"]==usi)&(switch_time_df["Bit_Rate"]==bitrate)&(switch_time_df["UAV_Speed"]==uav_speed)]
    switch_time_df = switch_time_df.loc[switch_time_df["Run"]<=run_num]
    # Get rows where not all switch times are nan
    run_df = switch_time_df[["Run"]]
    switch_time_df = switch_time_df[["GCS_Switch_Time", "GW_Switch_Time", "UAV_0_Switch_Time", "UAV_2_Switch_Time", "UAV_3_Switch_Time", "UAV_4_Switch_Time",
                                     "UAV_5_Switch_Time", "UAV_6_Switch_Time", "UAV_7_Switch_Time"]]
    run_df = run_df.loc[~switch_time_df.isnull().all(axis=1)]
    if not run_df.empty: # If the scenario has no runs where switching occured, no need to process
        run_number = run_df["Run"].values # Get the run_number based on switch time (we only process runs where a switch to mitigation occured)
        with Pool(num_processes) as pool:
            for result in pool.starmap(process_run, zip(run_number, repeat(scenario_path))):
                if (result[2] != None) or (result[3] != None):
                    anomaly_list.append({"Scenario_Param": result[0], "Run_Num": result[1], "Max_Diff_Switch_Time": result[2], "Num_Links_Switch": result[3]}) 
    
    # Save metrics of each run to file
    if len(anomaly_list) > 0:
        anomaly_df = pd.DataFrame(anomaly_list)
    else:
        anomaly_df = pd.DataFrame(columns=["Scenario_Param", "Run_Num", "Max_Diff_Switch_Time", "Num_Links_Switch"])
    
    return anomaly_df

def process_run(run_number, scenario_path):

    switch_time_df = pd.read_csv(os.path.join(scenario_path, "Run-{}_Switch_Time.csv".format(run_number)))

    min_switch_time = switch_time_df["Switch_Time"].min()
    max_switch_time = switch_time_df["Switch_Time"].max()

    if (max_switch_time - min_switch_time) > 1:
        switch_time_anomaly = max_switch_time - min_switch_time
    else:
        switch_time_anomaly = None

    if len(switch_time_df) != 9:
        switch_num_anomaly = len(switch_time_df)
    else:
        switch_num_anomaly = None
    
    scenario = scenario_path.split("/")[-1]
    return (scenario, run_number, switch_time_anomaly, switch_num_anomaly)

if __name__ == "__main__":
    '''
    MAKE SURE TO SET "GX_GCS" and "sending_interval_range" in function process_sim_data_v2
    '''
    # To suppress SettingWithCopyWarning
    pd.options.mode.chained_assignment = None  # default='warn'

    sim_root_paths = ["/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Waypoint_Only_UAV_Interference_new/uav_scenario_0", 
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Waypoint_Only_UAV_Interference_new/uav_scenario_1", 
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Waypoint_Only_UAV_Interference_new/uav_scenario_2", 
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Waypoint_Only_MANET_Interference_new/manet_scenario_1",
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Waypoint_Only_MANET_Interference_new/manet_scenario_a",
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Waypoint_Only_MANET_Interference_new/manet_scenario_2",
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Fixed_Prob_UAV_Interference_new/uav_scenario_0", 
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Fixed_Prob_UAV_Interference_new/uav_scenario_1", 
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Fixed_Prob_UAV_Interference_new/uav_scenario_2", 
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Fixed_Prob_MANET_Interference_new/manet_scenario_1",
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Fixed_Prob_MANET_Interference_new/manet_scenario_a",
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Fixed_Prob_MANET_Interference_new/manet_scenario_2",
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/DV_Based_500_UAV_Interference_new/uav_scenario_0", 
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/DV_Based_500_UAV_Interference_new/uav_scenario_1", 
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/DV_Based_500_UAV_Interference_new/uav_scenario_2", 
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/DV_Based_500_MANET_Interference_new/manet_scenario_1",
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/DV_Based_500_MANET_Interference_new/manet_scenario_a",
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/DV_Based_500_MANET_Interference_new/manet_scenario_2",
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/PM_Based_UAV_Interference_new/uav_scenario_0", 
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/PM_Based_UAV_Interference_new/uav_scenario_1", 
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/PM_Based_UAV_Interference_new/uav_scenario_2", 
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/PM_Based_MANET_Interference_new/manet_scenario_1",
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/PM_Based_MANET_Interference_new/manet_scenario_a",
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/PM_Based_MANET_Interference_new/manet_scenario_2",
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/PM_GCS_Based_UAV_Interference_new/uav_scenario_0", 
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/PM_GCS_Based_UAV_Interference_new/uav_scenario_1", 
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/PM_GCS_Based_UAV_Interference_new/uav_scenario_2", 
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/PM_GCS_Based_MANET_Interference_new/manet_scenario_1",
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/PM_GCS_Based_MANET_Interference_new/manet_scenario_a",
                      "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/PM_GCS_Based_MANET_Interference_new/manet_scenario_2"]
    switch_time_paths = ["/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/uav_scenario_0/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/uav_scenario_1/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/uav_scenario_2/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/manet_scenario_1/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/manet_scenario_a/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/manet_scenario_2/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/uav_scenario_0/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/uav_scenario_1/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/uav_scenario_2/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/manet_scenario_1/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/manet_scenario_a/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/manet_scenario_2/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/uav_scenario_0/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/uav_scenario_1/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/uav_scenario_2/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/manet_scenario_1/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/manet_scenario_a/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/manet_scenario_2/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/uav_scenario_0/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/uav_scenario_1/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/uav_scenario_2/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/manet_scenario_1/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/manet_scenario_a/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/manet_scenario_2/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/uav_scenario_0/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/uav_scenario_1/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/uav_scenario_2/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/manet_scenario_1/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/manet_scenario_a/Switch_Details.csv",
                         "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/obj3_scenario_script_new/manet_scenario_2/Switch_Details.csv"]
    save_path = "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/results_new"

    
    RUN_NUM = 499 # Max run number to consider, note it starts at 0
    results = []

    for sim_root_path, switch_time_path in zip(sim_root_paths, switch_time_paths):

        scenario_list = [f.path for f in os.scandir(sim_root_path) if f.is_dir()] # Get list of "unique" scenarios
        num_processes = 32
        # For each scenario, extract the UL and DL raw data
        for scenario_path in scenario_list:
            df = process_scenario(scenario_path, switch_time_path, num_processes, RUN_NUM)
            df["Method"] = sim_root_path.split("/")[-2]
            df["Scenario"] = sim_root_path.split("/")[-1]
            df = df.loc[:, ['Method', 'Scenario', "Scenario_Param", "Run_Num", "Max_Diff_Switch_Time", "Num_Links_Switch"]]
            results.append(df)

            df = pd.concat(results)
            df_time_anomaly = df.loc[df["Max_Diff_Switch_Time"] != None].drop(['Num_Links_Switch'], axis=1).copy()
            df_num_anomaly = df.loc[df["Num_Links_Switch"] != None].drop(['Max_Diff_Switch_Time'], axis=1).copy()
            
            df_time_anomaly.to_csv(os.path.join(save_path, "Switching_Time_Anomaly_Cases.csv"), index=False)
            df_num_anomaly.to_csv(os.path.join(save_path, "Switching_Num_Anomaly_Cases.csv"), index=False)