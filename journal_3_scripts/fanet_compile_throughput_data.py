import pandas as pd
import numpy as np 
import os
import glob
from tqdm import tqdm
from multiprocessing.pool import Pool

def load_num_reliable(scenario_path):
    '''Modified: Add the scenario parameters and run number so that DFs can be sorted for rows 
                 of each DF to correspond to the same time window of the same run
    '''
    scenario_name = scenario_path.split("/")[-1]
    params = scenario_name.split("_")
    usi = [x for x in params if "UAVSendingInterval" in x][0].split('-')[-1]
    speed = [x for x in params if "UAVSpeed" in x][0].split('-')[-1]
    height = [x for x in params if "Height" in x][0].split('-')[-1]
    bitrate = [x for x in params if "BitRate" in x][0].split('-')[-1]

    # Get Run Nums (Using UL files, since there's only one UL file per run)
    ul_num_reliable_files = glob.glob(os.path.join(scenario_path, "Run-*_Uplink_Throughput.csv"))
    run_num = [int(f.split("Run-")[-1].split("_")[0]) for f in ul_num_reliable_files]
    # Load DL Throughput
    uav_df_list = []
    for run in run_num:
        dl_df_0 = pd.read_csv(os.path.join(scenario_path, "Run-{}_UAV-0_Downlink_Throughput.csv".format(run)))
        dl_df_1 = pd.read_csv(os.path.join(scenario_path, "Run-{}_UAV-1_Downlink_Throughput.csv".format(run)))
        dl_df_2 = pd.read_csv(os.path.join(scenario_path, "Run-{}_UAV-2_Downlink_Throughput.csv".format(run)))
        dl_df_3 = pd.read_csv(os.path.join(scenario_path, "Run-{}_UAV-3_Downlink_Throughput.csv".format(run)))
        dl_df_4 = pd.read_csv(os.path.join(scenario_path, "Run-{}_UAV-4_Downlink_Throughput.csv".format(run)))
        dl_df_5 = pd.read_csv(os.path.join(scenario_path, "Run-{}_UAV-5_Downlink_Throughput.csv".format(run)))
        dl_df_6 = pd.read_csv(os.path.join(scenario_path, "Run-{}_UAV-6_Downlink_Throughput.csv".format(run)))
        dl_df_7 = pd.read_csv(os.path.join(scenario_path, "Run-{}_UAV-7_Downlink_Throughput.csv".format(run)))
        for df in [dl_df_0, dl_df_1, dl_df_2, dl_df_3, dl_df_4, dl_df_5, dl_df_6, dl_df_7]:
            df["Run_Num"] = run
            df["USI"] = usi
            df["UAV_Speed"] = speed
            df["UAV_Height"] = height
            df["Bit_Rate"] = bitrate
        uav_df_list.append([dl_df_0, dl_df_1, dl_df_2, dl_df_3, dl_df_4, dl_df_5, dl_df_6, dl_df_7])
    dl_df_list = [pd.concat([uav_df_list[i][j] for i in range(len(run_num))]) for j in range(len(uav_df_list[0]))]
    
    # Load UL Throughput
    ul_df_list = []
    for run in run_num:
        measured_df = pd.read_csv(os.path.join(scenario_path, "Run-{}_Uplink_Throughput.csv".format(run)))
        measured_df["Run_Num"] = run
        ul_df_list.append(measured_df)
    ul_df = pd.concat(ul_df_list)
    ul_df["USI"] = usi
    ul_df["UAV_Speed"] = speed
    ul_df["UAV_Height"] = height
    ul_df["Bit_Rate"] = bitrate
    # Load Video Throughput
    vid_df_list = []
    for run in run_num:
        measured_df = pd.read_csv(os.path.join(scenario_path, "Run-{}_Video_Throughput.csv".format(run)))
        measured_df["Run_Num"] = run
        vid_df_list.append(measured_df)
    vid_df = pd.concat(vid_df_list)
    vid_df["USI"] = usi
    vid_df["UAV_Speed"] = speed
    vid_df["UAV_Height"] = height
    vid_df["Bit_Rate"] = bitrate

    return (dl_df_list, ul_df, vid_df)

if __name__ == "__main__":
    DATASET_INT_PATHS = ["/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Manual_UAV_Interference_new/uav_scenario_2_processed", 
                        "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Manual_UAV_Interference_new/uav_scenario_1_processed",
                        "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Manual_UAV_Interference_new/uav_scenario_0_processed"]
    SAVE_PATH = ["/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Manual_UAV_Interference_new/uav_scenario_2_{}_processed.csv", 
                "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Manual_UAV_Interference_new/uav_scenario_1_{}_processed.csv",
                "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Manual_UAV_Interference_new/uav_scenario_0_{}_processed.csv"]
    # DATASET_INT_PATHS = ["/home/clow0003/Reuben_ws/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Manual_MANET_Interference_new/manet_scenario_1_processed", 
    #                     "/home/clow0003/Reuben_ws/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Manual_MANET_Interference_new/manet_scenario_a_processed",
    #                     "/home/clow0003/Reuben_ws/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Manual_MANET_Interference_new/manet_scenario_2_processed"]
    # SAVE_PATH = ["/home/clow0003/Reuben_ws/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Manual_MANET_Interference_new/manet_scenario_1_{}_processed.csv", 
    #             "/home/clow0003/Reuben_ws/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Manual_MANET_Interference_new/manet_scenario_a_{}_processed.csv",
    #             "/home/clow0003/Reuben_ws/FANET_Dataset/DJISpark_Obj3_Retransmission_Datasets/Manual_MANET_Interference_new/manet_scenario_2_{}_processed.csv"]
    
    NUM_PROCS = 64

    counter = 0
    for datasets in DATASET_INT_PATHS:
        scenario_paths = [x[0] for x in os.walk(datasets) if (os.path.isdir(x[0]) and x[0]!=datasets)]
        scenario_name = datasets.split("/")[-1]
        print(scenario_name)
        dl_num_reliable_df_list = []
        ul_num_reliable_df_list = []
        vid_num_reliable_df_list = []
        with Pool(NUM_PROCS) as pool:
            # for result in pool.starmap(load_num_reliable, zip(scenario_paths, max_num_reliable_dl, max_num_reliable_ul, max_num_reliable_vid)):
            # for result in pool.starmap(load_num_reliable, zip(scenario_paths, min_num_reliable_dl, min_num_reliable_ul, min_num_reliable_vid)):
            for result in tqdm(pool.starmap(load_num_reliable, zip(scenario_paths))):
                dl_num_reliable_df_list.append(result[0])
                ul_num_reliable_df_list.append(result[1])
                vid_num_reliable_df_list.append(result[2])

        uavs_num_reliable_df = [pd.concat([dl_num_reliable_df_list[i][j] for i in range(len(scenario_paths))]) for j in range(len(dl_num_reliable_df_list[0]))]           
        ul_num_reliable_df = pd.concat(ul_num_reliable_df_list)
        vid_num_reliable_df = pd.concat(vid_num_reliable_df_list)

        for i in range(len(uavs_num_reliable_df)):
            uavs_num_reliable_df[i].to_csv(SAVE_PATH[counter].format("UAV_" + str(i)))
        ul_num_reliable_df.to_csv(SAVE_PATH[counter].format("ul"))
        vid_num_reliable_df.to_csv(SAVE_PATH[counter].format("vid"))
        counter += 1

        # Free up mem
        del dl_num_reliable_df_list, ul_num_reliable_df_list, vid_num_reliable_df_list, uavs_num_reliable_df, ul_num_reliable_df, vid_num_reliable_df
