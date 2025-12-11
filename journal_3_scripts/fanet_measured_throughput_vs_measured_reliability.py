import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import glob, os
from scipy import stats
from multiprocessing.pool import Pool

# def load_throughput(scenario_path, min_throughput_dl, min_throughput_ul, min_throughput_vid):
#     # print(scenario_path.split("/")[-1])
#     # Load DL Throughput
#     dl_df_list = []
#     dl_throughput_files = glob.glob(os.path.join(scenario_path, "Run-*_Downlink_Throughput.csv"))
#     for file in dl_throughput_files:
#         measured_df = pd.read_csv(file)
#         dl_df_list.append(measured_df)
#     dl_df = pd.concat(dl_df_list)
#     # dl_df = dl_df.loc[dl_df["Throughput"] <= min_throughput_dl]
#     dl_df["Norm_Throughput"] = dl_df["Throughput"] / min_throughput_dl
#     # Load UL Throughput
#     ul_df_list = []
#     ul_throughput_files = glob.glob(os.path.join(scenario_path, "Run-*_Uplink_Throughput.csv"))
#     for file in ul_throughput_files:
#         measured_df = pd.read_csv(file)
#         ul_df_list.append(measured_df)
#     ul_df = pd.concat(ul_df_list)
#     # ul_df = ul_df.loc[ul_df["Throughput"] <= min_throughput_ul]
#     ul_df["Norm_Throughput"] = ul_df["Throughput"] / min_throughput_ul
#     # Load Video Throughput
#     vid_df_list = []
#     vid_throughput_files = glob.glob(os.path.join(scenario_path, "Run-*_Video_Throughput.csv"))
#     for file in vid_throughput_files:
#         measured_df = pd.read_csv(file)
#         vid_df_list.append(measured_df)
#     vid_df = pd.concat(vid_df_list)
#     # vid_df = vid_df.loc[vid_df["Throughput"] <= min_throughput_vid]
#     vid_df["Norm_Throughput"] = vid_df["Throughput"] / min_throughput_vid

#     return (dl_df, ul_df, vid_df)

def load_throughput(scenario_path):
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
    ul_throughput_files = glob.glob(os.path.join(scenario_path, "Run-*_Uplink_Throughput.csv"))
    run_num = [int(f.split("Run-")[-1].split("_")[0]) for f in ul_throughput_files]
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
            df["USI"] = float(usi)
            df["UAV_Speed"] = int(speed)
            df["UAV_Height"] = int(height)
            df["Bit_Rate"] = float(bitrate)
        uav_df_list.append([dl_df_0, dl_df_1, dl_df_2, dl_df_3, dl_df_4, dl_df_5, dl_df_6, dl_df_7])
    dl_df_list = [pd.concat([uav_df_list[i][j] for i in range(len(run_num))]) for j in range(len(uav_df_list[0]))]
    
    # Load UL Throughput
    ul_df_list = []
    for run in run_num:
        measured_df = pd.read_csv(os.path.join(scenario_path, "Run-{}_Uplink_Throughput.csv".format(run)))
        measured_df["Run_Num"] = run
        ul_df_list.append(measured_df)
    ul_df = pd.concat(ul_df_list)
    ul_df["USI"] = float(usi)
    ul_df["UAV_Speed"] = int(speed)
    ul_df["UAV_Height"] = int(height)
    ul_df["Bit_Rate"] = float(bitrate)
    # Load Video Throughput
    vid_df_list = []
    for run in run_num:
        measured_df = pd.read_csv(os.path.join(scenario_path, "Run-{}_Video_Throughput.csv".format(run)))
        measured_df["Run_Num"] = run
        vid_df_list.append(measured_df)
    vid_df = pd.concat(vid_df_list)
    vid_df["USI"] = float(usi)
    vid_df["UAV_Speed"] = int(speed)
    vid_df["UAV_Height"] = int(height)
    vid_df["Bit_Rate"] = float(bitrate)

    return (dl_df_list, ul_df, vid_df)

'''Load by compiling processed data'''
# NUM_PROCS = 32
# DATASET_PATH = ["/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_0_processed",
#                 "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_1_processed",
#                 "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_2_processed",
#                 "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_0_processed",
#                 "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_1_processed",
#                 "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_2_processed"] 
'''Load the compiled data from CSV files'''
NUM_UAV = 8
DATASET_PATH = ["/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_0_{}_processed.csv", 
                "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_1_{}_processed.csv",
                "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_2_{}_processed.csv",
                "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_1_{}_processed.csv",
                "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_a_{}_processed.csv",
                "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_2_{}_processed.csv"]

SAVE_CORRELATION_PEARSON = "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/results_new/Throughput_Pearson_99-299_runs.csv"
SAVE_CORRELATION_SPEARMAN = "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/results_new/Throughput_Spearman_400-499_runs.csv"
SAVE_CORRELATION_KENDALL = "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/results_new/Throughput_Kendall_99-299_runs.csv"
SAVE_SVG = "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/throughput_corr_svg"
SEED = 1
NUM_SAMPLES_PLOT = np.inf # 10000
IMG_FORMAT = "png" # 'png', 'svg'
NUM_RUNS_START = 400 # Start range of run number to consider (inclusive)
NUM_RUNS_END = 499 # Start range of run number to consider (inclusive)
UNIQUE = True # Whether to filter out duplicated data or not

# Throughput threshold for filtering
# usi_min_throughput_dl = {10: 600, 20: 600, 66.7: 600, 100: 600}
# usi_min_throughput_ul = {10: 183200, 20: 88850, 66.7: 22810, 100: 14390}
# usi_min_throughput_vid = {10:227166.66666666663, 20: 188500, 66.7: 188500, 100: 203000}

pearson_results = []
spearman_results = []
kendall_results = []

for dataset in DATASET_PATH:
    print(dataset)
    scenario_paths = [x[0] for x in os.walk(dataset) if (os.path.isdir(x[0]) and x[0]!=dataset)]
    scenario = dataset.split("/")[-1]

    '''Load by compiling processed data'''
    # dl_throughput_df_list = []
    # ul_throughput_df_list = []
    # vid_throughput_df_list = []
    # with Pool(NUM_PROCS) as pool:
    #     # for result in pool.starmap(load_throughput, zip(scenario_paths, max_throughput_dl, max_throughput_ul, max_throughput_vid)):
    #     # for result in pool.starmap(load_throughput, zip(scenario_paths, min_throughput_dl, min_throughput_ul, min_throughput_vid)):
    #     for result in pool.starmap(load_throughput, zip(scenario_paths)):
    #         dl_throughput_df_list.append(result[0])
    #         ul_throughput_df_list.append(result[1])
    #         vid_throughput_df_list.append(result[2])
    # ul_throughput_df = pd.concat(ul_throughput_df_list)
    # vid_throughput_df = pd.concat(vid_throughput_df_list)

    '''Load the compiled data from CSV files'''
    ul_throughput_df = pd.read_csv(dataset.format("ul"))
    ul_throughput_df = ul_throughput_df.loc[(ul_throughput_df["Run_Num"] >= NUM_RUNS_START) & (ul_throughput_df["Run_Num"] <= NUM_RUNS_END)]
    vid_throughput_df = pd.read_csv(dataset.format("vid"))
    vid_throughput_df = vid_throughput_df.loc[(vid_throughput_df["Run_Num"] >= NUM_RUNS_START) & (vid_throughput_df["Run_Num"] <= NUM_RUNS_END)]
    dl_throughput_df_list = []
    for i in range(NUM_UAV):
        df = pd.read_csv(dataset.format("UAV_" + str(i)))
        df = df.loc[(df["Run_Num"] >= NUM_RUNS_START) & (df["Run_Num"] <= NUM_RUNS_END)]
        dl_throughput_df_list.append(df)

    uav_0_throughput_df = dl_throughput_df_list[0]
    uav_1_throughput_df = dl_throughput_df_list[1]
    uav_2_throughput_df = dl_throughput_df_list[2]
    uav_3_throughput_df = dl_throughput_df_list[3]
    uav_4_throughput_df = dl_throughput_df_list[4]
    uav_5_throughput_df = dl_throughput_df_list[5]
    uav_6_throughput_df = dl_throughput_df_list[6]
    uav_7_throughput_df = dl_throughput_df_list[7]

    # There could be some datapoints with 0 throughput and nan measured reliability
    uav_0_throughput_df.dropna(subset=["Measured_Reliability_1", "Throughput"], inplace=True)
    uav_1_throughput_df.dropna(subset=["Measured_Reliability_1", "Throughput"], inplace=True)
    uav_2_throughput_df.dropna(subset=["Measured_Reliability_1", "Throughput"], inplace=True)
    uav_3_throughput_df.dropna(subset=["Measured_Reliability_1", "Throughput"], inplace=True)
    uav_4_throughput_df.dropna(subset=["Measured_Reliability_1", "Throughput"], inplace=True)
    uav_5_throughput_df.dropna(subset=["Measured_Reliability_1", "Throughput"], inplace=True)
    uav_6_throughput_df.dropna(subset=["Measured_Reliability_1", "Throughput"], inplace=True)
    uav_7_throughput_df.dropna(subset=["Measured_Reliability_1", "Throughput"], inplace=True)
    ul_throughput_df.dropna(subset=["Measured_Reliability_1", "Throughput"], inplace=True)
    vid_throughput_df.dropna(subset=["Measured_Reliability_1", "Throughput"], inplace=True)

    # Filter samples with "Time" < 1
    uav_0_throughput_df = uav_0_throughput_df.loc[uav_0_throughput_df["Time"] >= 1]
    uav_1_throughput_df = uav_1_throughput_df.loc[uav_1_throughput_df["Time"] >= 1]
    uav_2_throughput_df = uav_2_throughput_df.loc[uav_2_throughput_df["Time"] >= 1]
    uav_3_throughput_df = uav_3_throughput_df.loc[uav_3_throughput_df["Time"] >= 1]
    uav_4_throughput_df = uav_4_throughput_df.loc[uav_4_throughput_df["Time"] >= 1]
    uav_5_throughput_df = uav_5_throughput_df.loc[uav_5_throughput_df["Time"] >= 1]
    uav_6_throughput_df = uav_6_throughput_df.loc[uav_6_throughput_df["Time"] >= 1]
    uav_7_throughput_df = uav_7_throughput_df.loc[uav_7_throughput_df["Time"] >= 1]
    ul_throughput_df = ul_throughput_df.loc[ul_throughput_df["Time"] >= 1]
    vid_throughput_df = vid_throughput_df.loc[vid_throughput_df["Time"] >= 1]

    # Split the UL and DLs by USI
    # 10 ms
    ul_th_usi_10_df = ul_throughput_df.loc[ul_throughput_df["USI"] == 10]
    uav_0_th_usi_10_df = uav_0_throughput_df.loc[uav_0_throughput_df["USI"] == 10]
    uav_1_th_usi_10_df = uav_1_throughput_df.loc[uav_1_throughput_df["USI"] == 10]
    uav_2_th_usi_10_df = uav_2_throughput_df.loc[uav_2_throughput_df["USI"] == 10]
    uav_3_th_usi_10_df = uav_3_throughput_df.loc[uav_3_throughput_df["USI"] == 10]
    uav_4_th_usi_10_df = uav_4_throughput_df.loc[uav_4_throughput_df["USI"] == 10]
    uav_5_th_usi_10_df = uav_5_throughput_df.loc[uav_5_throughput_df["USI"] == 10]
    uav_6_th_usi_10_df = uav_6_throughput_df.loc[uav_6_throughput_df["USI"] == 10]
    uav_7_th_usi_10_df = uav_7_throughput_df.loc[uav_7_throughput_df["USI"] == 10]
    # 20 ms
    ul_th_usi_20_df = ul_throughput_df.loc[ul_throughput_df["USI"] == 20]
    uav_0_th_usi_20_df = uav_0_throughput_df.loc[uav_0_throughput_df["USI"] == 20]
    uav_1_th_usi_20_df = uav_1_throughput_df.loc[uav_1_throughput_df["USI"] == 20]
    uav_2_th_usi_20_df = uav_2_throughput_df.loc[uav_2_throughput_df["USI"] == 20]
    uav_3_th_usi_20_df = uav_3_throughput_df.loc[uav_3_throughput_df["USI"] == 20]
    uav_4_th_usi_20_df = uav_4_throughput_df.loc[uav_4_throughput_df["USI"] == 20]
    uav_5_th_usi_20_df = uav_5_throughput_df.loc[uav_5_throughput_df["USI"] == 20]
    uav_6_th_usi_20_df = uav_6_throughput_df.loc[uav_6_throughput_df["USI"] == 20]
    uav_7_th_usi_20_df = uav_7_throughput_df.loc[uav_7_throughput_df["USI"] == 20]
    # 66.7 ms
    ul_th_usi_667_df = ul_throughput_df.loc[ul_throughput_df["USI"] == 66.7]
    uav_0_th_usi_667_df = uav_0_throughput_df.loc[uav_0_throughput_df["USI"] == 66.7]
    uav_1_th_usi_667_df = uav_1_throughput_df.loc[uav_1_throughput_df["USI"] == 66.7]
    uav_2_th_usi_667_df = uav_2_throughput_df.loc[uav_2_throughput_df["USI"] == 66.7]
    uav_3_th_usi_667_df = uav_3_throughput_df.loc[uav_3_throughput_df["USI"] == 66.7]
    uav_4_th_usi_667_df = uav_4_throughput_df.loc[uav_4_throughput_df["USI"] == 66.7]
    uav_5_th_usi_667_df = uav_5_throughput_df.loc[uav_5_throughput_df["USI"] == 66.7]
    uav_6_th_usi_667_df = uav_6_throughput_df.loc[uav_6_throughput_df["USI"] == 66.7]
    uav_7_th_usi_667_df = uav_7_throughput_df.loc[uav_7_throughput_df["USI"] == 66.7]
    # 100 ms
    ul_th_usi_100_df = ul_throughput_df.loc[ul_throughput_df["USI"] == 100]
    uav_0_th_usi_100_df = uav_0_throughput_df.loc[uav_0_throughput_df["USI"] == 100]
    uav_1_th_usi_100_df = uav_1_throughput_df.loc[uav_1_throughput_df["USI"] == 100]
    uav_2_th_usi_100_df = uav_2_throughput_df.loc[uav_2_throughput_df["USI"] == 100]
    uav_3_th_usi_100_df = uav_3_throughput_df.loc[uav_3_throughput_df["USI"] == 100]
    uav_4_th_usi_100_df = uav_4_throughput_df.loc[uav_4_throughput_df["USI"] == 100]
    uav_5_th_usi_100_df = uav_5_throughput_df.loc[uav_5_throughput_df["USI"] == 100]
    uav_6_th_usi_100_df = uav_6_throughput_df.loc[uav_6_throughput_df["USI"] == 100]
    uav_7_th_usi_100_df = uav_7_throughput_df.loc[uav_7_throughput_df["USI"] == 100]

    # Drop duplicates before calculating Spearman and Kendall
    if UNIQUE:
        uav_0_throughput_df = uav_0_throughput_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_1_throughput_df = uav_1_throughput_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_2_throughput_df = uav_2_throughput_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_3_throughput_df = uav_3_throughput_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_4_throughput_df = uav_4_throughput_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_5_throughput_df = uav_5_throughput_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_6_throughput_df = uav_6_throughput_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_7_throughput_df = uav_7_throughput_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        ul_th_usi_10_df = ul_th_usi_10_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        ul_th_usi_20_df = ul_th_usi_20_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        ul_th_usi_667_df = ul_th_usi_667_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        ul_th_usi_100_df = ul_th_usi_100_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_0_th_usi_10_df = uav_0_th_usi_10_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_0_th_usi_20_df = uav_0_th_usi_20_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_0_th_usi_667_df = uav_0_th_usi_667_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_0_th_usi_100_df = uav_0_th_usi_100_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_1_th_usi_10_df = uav_1_th_usi_10_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_1_th_usi_20_df = uav_1_th_usi_20_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_1_th_usi_667_df = uav_1_th_usi_667_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_1_th_usi_100_df = uav_1_th_usi_100_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_2_th_usi_10_df = uav_2_th_usi_10_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_2_th_usi_20_df = uav_2_th_usi_20_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_2_th_usi_667_df = uav_2_th_usi_667_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_2_th_usi_100_df = uav_2_th_usi_100_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_3_th_usi_10_df = uav_3_th_usi_10_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_3_th_usi_20_df = uav_3_th_usi_20_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_3_th_usi_667_df = uav_3_th_usi_667_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_3_th_usi_100_df = uav_3_th_usi_100_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_4_th_usi_10_df = uav_4_th_usi_10_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_4_th_usi_20_df = uav_4_th_usi_20_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_4_th_usi_667_df = uav_4_th_usi_667_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_4_th_usi_100_df = uav_4_th_usi_100_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_5_th_usi_10_df = uav_5_th_usi_10_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_5_th_usi_20_df = uav_5_th_usi_20_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_5_th_usi_667_df = uav_5_th_usi_667_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_5_th_usi_100_df = uav_5_th_usi_100_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_6_th_usi_10_df = uav_6_th_usi_10_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_6_th_usi_20_df = uav_6_th_usi_20_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_6_th_usi_667_df = uav_6_th_usi_667_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_6_th_usi_100_df = uav_6_th_usi_100_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_7_th_usi_10_df = uav_7_th_usi_10_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_7_th_usi_20_df = uav_7_th_usi_20_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_7_th_usi_667_df = uav_7_th_usi_667_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        uav_7_th_usi_100_df = uav_7_th_usi_100_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
        vid_throughput_df = vid_throughput_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])

    # Calculate correlation
    # # Pearson Rho
    # pearsonr_uav_0 = stats.pearsonr(uav_0_throughput_df["Throughput"].values, uav_0_throughput_df["Measured_Reliability_1"].values)
    # pearsonr_uav_1 = stats.pearsonr(uav_1_throughput_df["Throughput"].values, uav_1_throughput_df["Measured_Reliability_1"].values)
    # pearsonr_uav_2 = stats.pearsonr(uav_2_throughput_df["Throughput"].values, uav_2_throughput_df["Measured_Reliability_1"].values)
    # pearsonr_uav_3 = stats.pearsonr(uav_3_throughput_df["Throughput"].values, uav_3_throughput_df["Measured_Reliability_1"].values)
    # pearsonr_uav_4 = stats.pearsonr(uav_4_throughput_df["Throughput"].values, uav_4_throughput_df["Measured_Reliability_1"].values)
    # pearsonr_uav_5 = stats.pearsonr(uav_5_throughput_df["Throughput"].values, uav_5_throughput_df["Measured_Reliability_1"].values)
    # pearsonr_uav_6 = stats.pearsonr(uav_6_throughput_df["Throughput"].values, uav_6_throughput_df["Measured_Reliability_1"].values)
    # pearsonr_uav_7 = stats.pearsonr(uav_7_throughput_df["Throughput"].values, uav_7_throughput_df["Measured_Reliability_1"].values)
    # pearsonr_ul_usi_10 = stats.pearsonr(ul_th_usi_10_df["Throughput"].values, ul_th_usi_10_df["Measured_Reliability_1"].values)
    # pearsonr_uav_0_usi_10 = stats.pearsonr(uav_0_th_usi_10_df["Throughput"].values, uav_0_th_usi_10_df["Measured_Reliability_1"].values)
    # pearsonr_uav_1_usi_10 = stats.pearsonr(uav_1_th_usi_10_df["Throughput"].values, uav_1_th_usi_10_df["Measured_Reliability_1"].values)
    # pearsonr_uav_2_usi_10 = stats.pearsonr(uav_2_th_usi_10_df["Throughput"].values, uav_2_th_usi_10_df["Measured_Reliability_1"].values)
    # pearsonr_uav_3_usi_10 = stats.pearsonr(uav_3_th_usi_10_df["Throughput"].values, uav_3_th_usi_10_df["Measured_Reliability_1"].values)
    # pearsonr_uav_4_usi_10 = stats.pearsonr(uav_4_th_usi_10_df["Throughput"].values, uav_4_th_usi_10_df["Measured_Reliability_1"].values)
    # pearsonr_uav_5_usi_10 = stats.pearsonr(uav_5_th_usi_10_df["Throughput"].values, uav_5_th_usi_10_df["Measured_Reliability_1"].values)
    # pearsonr_uav_6_usi_10 = stats.pearsonr(uav_6_th_usi_10_df["Throughput"].values, uav_6_th_usi_10_df["Measured_Reliability_1"].values)
    # pearsonr_uav_7_usi_10 = stats.pearsonr(uav_7_th_usi_10_df["Throughput"].values, uav_7_th_usi_10_df["Measured_Reliability_1"].values)
    # pearsonr_ul_usi_20 = stats.pearsonr(ul_th_usi_20_df["Throughput"].values, ul_th_usi_20_df["Measured_Reliability_1"].values)
    # pearsonr_uav_0_usi_20 = stats.pearsonr(uav_0_th_usi_20_df["Throughput"].values, uav_0_th_usi_20_df["Measured_Reliability_1"].values)
    # pearsonr_uav_1_usi_20 = stats.pearsonr(uav_1_th_usi_20_df["Throughput"].values, uav_1_th_usi_20_df["Measured_Reliability_1"].values)
    # pearsonr_uav_2_usi_20 = stats.pearsonr(uav_2_th_usi_20_df["Throughput"].values, uav_2_th_usi_20_df["Measured_Reliability_1"].values)
    # pearsonr_uav_3_usi_20 = stats.pearsonr(uav_3_th_usi_20_df["Throughput"].values, uav_3_th_usi_20_df["Measured_Reliability_1"].values)
    # pearsonr_uav_4_usi_20 = stats.pearsonr(uav_4_th_usi_20_df["Throughput"].values, uav_4_th_usi_20_df["Measured_Reliability_1"].values)
    # pearsonr_uav_5_usi_20 = stats.pearsonr(uav_5_th_usi_20_df["Throughput"].values, uav_5_th_usi_20_df["Measured_Reliability_1"].values)
    # pearsonr_uav_6_usi_20 = stats.pearsonr(uav_6_th_usi_20_df["Throughput"].values, uav_6_th_usi_20_df["Measured_Reliability_1"].values)
    # pearsonr_uav_7_usi_20 = stats.pearsonr(uav_7_th_usi_20_df["Throughput"].values, uav_7_th_usi_20_df["Measured_Reliability_1"].values)
    # pearsonr_ul_usi_667 = stats.pearsonr(ul_th_usi_667_df["Throughput"].values, ul_th_usi_667_df["Measured_Reliability_1"].values)
    # pearsonr_uav_0_usi_667 = stats.pearsonr(uav_0_th_usi_667_df["Throughput"].values, uav_0_th_usi_667_df["Measured_Reliability_1"].values)
    # pearsonr_uav_1_usi_667 = stats.pearsonr(uav_1_th_usi_667_df["Throughput"].values, uav_1_th_usi_667_df["Measured_Reliability_1"].values)
    # pearsonr_uav_2_usi_667 = stats.pearsonr(uav_2_th_usi_667_df["Throughput"].values, uav_2_th_usi_667_df["Measured_Reliability_1"].values)
    # pearsonr_uav_3_usi_667 = stats.pearsonr(uav_3_th_usi_667_df["Throughput"].values, uav_3_th_usi_667_df["Measured_Reliability_1"].values)
    # pearsonr_uav_4_usi_667 = stats.pearsonr(uav_4_th_usi_667_df["Throughput"].values, uav_4_th_usi_667_df["Measured_Reliability_1"].values)
    # pearsonr_uav_5_usi_667 = stats.pearsonr(uav_5_th_usi_667_df["Throughput"].values, uav_5_th_usi_667_df["Measured_Reliability_1"].values)
    # pearsonr_uav_6_usi_667 = stats.pearsonr(uav_6_th_usi_667_df["Throughput"].values, uav_6_th_usi_667_df["Measured_Reliability_1"].values)
    # pearsonr_uav_7_usi_667 = stats.pearsonr(uav_7_th_usi_667_df["Throughput"].values, uav_7_th_usi_667_df["Measured_Reliability_1"].values)
    # pearsonr_ul_usi_100 = stats.pearsonr(ul_th_usi_100_df["Throughput"].values, ul_th_usi_100_df["Measured_Reliability_1"].values)
    # pearsonr_uav_0_usi_100 = stats.pearsonr(uav_0_th_usi_100_df["Throughput"].values, uav_0_th_usi_100_df["Measured_Reliability_1"].values)
    # pearsonr_uav_1_usi_100 = stats.pearsonr(uav_1_th_usi_100_df["Throughput"].values, uav_1_th_usi_100_df["Measured_Reliability_1"].values)
    # pearsonr_uav_2_usi_100 = stats.pearsonr(uav_2_th_usi_100_df["Throughput"].values, uav_2_th_usi_100_df["Measured_Reliability_1"].values)
    # pearsonr_uav_3_usi_100 = stats.pearsonr(uav_3_th_usi_100_df["Throughput"].values, uav_3_th_usi_100_df["Measured_Reliability_1"].values)
    # pearsonr_uav_4_usi_100 = stats.pearsonr(uav_4_th_usi_100_df["Throughput"].values, uav_4_th_usi_100_df["Measured_Reliability_1"].values)
    # pearsonr_uav_5_usi_100 = stats.pearsonr(uav_5_th_usi_100_df["Throughput"].values, uav_5_th_usi_100_df["Measured_Reliability_1"].values)
    # pearsonr_uav_6_usi_100 = stats.pearsonr(uav_6_th_usi_100_df["Throughput"].values, uav_6_th_usi_100_df["Measured_Reliability_1"].values)
    # pearsonr_uav_7_usi_100 = stats.pearsonr(uav_7_th_usi_100_df["Throughput"].values, uav_7_th_usi_100_df["Measured_Reliability_1"].values)
    # pearsonr_vid = stats.pearsonr(vid_throughput_df["Throughput"].values, vid_throughput_df["Measured_Reliability_1"].values)

    # pearson_results.append({"Scenario": scenario, 
    #                      "Pearson_UAV_0_All_USI": pearsonr_uav_0[0], "Pearson_UAV_1_All_USI": pearsonr_uav_1[0],
    #                      "Pearson_UAV_2_All_USI": pearsonr_uav_2[0], "Pearson_UAV_3_All_USI": pearsonr_uav_3[0], "Pearson_UAV_4_All_USI": pearsonr_uav_4[0],
    #                      "Pearson_UAV_5_All_USI": pearsonr_uav_5[0], "Pearson_UAV_6_All_USI": pearsonr_uav_6[0], "Pearson_UAV_7_All_USI": pearsonr_uav_7[0],
    #                      "Pearson_UL_10": pearsonr_ul_usi_10[0], "Pearson_UL_20": pearsonr_ul_usi_20[0], "Pearson_UL_667": pearsonr_ul_usi_667[0], "Pearson_UL_100": pearsonr_ul_usi_100[0], 
    #                      "Pearson_UAV_0_10": pearsonr_uav_0_usi_10[0], "Pearson_UAV_0_20": pearsonr_uav_0_usi_20[0], "Pearson_UAV_0_667": pearsonr_uav_0_usi_667[0], "Pearson_UAV_0_100": pearsonr_uav_0_usi_100[0],
    #                      "Pearson_UAV_1_10": pearsonr_uav_1_usi_10[0], "Pearson_UAV_1_20": pearsonr_uav_1_usi_20[0], "Pearson_UAV_1_667": pearsonr_uav_1_usi_667[0], "Pearson_UAV_1_100": pearsonr_uav_1_usi_100[0],
    #                      "Pearson_UAV_2_10": pearsonr_uav_2_usi_10[0], "Pearson_UAV_2_20": pearsonr_uav_2_usi_20[0], "Pearson_UAV_2_667": pearsonr_uav_2_usi_667[0], "Pearson_UAV_2_100": pearsonr_uav_2_usi_100[0],
    #                      "Pearson_UAV_3_10": pearsonr_uav_3_usi_10[0], "Pearson_UAV_3_20": pearsonr_uav_3_usi_20[0], "Pearson_UAV_3_667": pearsonr_uav_3_usi_667[0], "Pearson_UAV_3_100": pearsonr_uav_3_usi_100[0],
    #                      "Pearson_UAV_4_10": pearsonr_uav_4_usi_10[0], "Pearson_UAV_4_20": pearsonr_uav_4_usi_20[0], "Pearson_UAV_4_667": pearsonr_uav_4_usi_667[0], "Pearson_UAV_4_100": pearsonr_uav_4_usi_100[0],
    #                      "Pearson_UAV_5_10": pearsonr_uav_5_usi_10[0], "Pearson_UAV_5_20": pearsonr_uav_5_usi_20[0], "Pearson_UAV_5_667": pearsonr_uav_5_usi_667[0], "Pearson_UAV_5_100": pearsonr_uav_5_usi_100[0],
    #                      "Pearson_UAV_6_10": pearsonr_uav_6_usi_10[0], "Pearson_UAV_6_20": pearsonr_uav_6_usi_20[0], "Pearson_UAV_6_667": pearsonr_uav_6_usi_667[0], "Pearson_UAV_6_100": pearsonr_uav_6_usi_100[0],
    #                      "Pearson_UAV_7_10": pearsonr_uav_7_usi_10[0], "Pearson_UAV_7_20": pearsonr_uav_7_usi_20[0], "Pearson_UAV_7_667": pearsonr_uav_7_usi_667[0], "Pearson_UAV_7_100": pearsonr_uav_7_usi_100[0],
    #                      "Pearson_VID": pearsonr_vid[0], 
    #                      "P_Value_UAV_0": pearsonr_uav_0[1], "P_Value_UAV_1": pearsonr_uav_1[1],
    #                      "P_Value_UAV_2": pearsonr_uav_2[1], "P_Value_UAV_3": pearsonr_uav_3[1], "P_Value_UAV_4": pearsonr_uav_4[1],
    #                      "P_Value_UAV_5": pearsonr_uav_5[1], "P_Value_UAV_6": pearsonr_uav_6[1], "P_Value_UAV_7": pearsonr_uav_7[1],
    #                      "P_Value_UL_10": pearsonr_ul_usi_10[1], "P_Value_UL_20": pearsonr_ul_usi_20[1], "P_Value_UL_667": pearsonr_ul_usi_667[1], "P_Value_UL_100": pearsonr_ul_usi_100[1],
    #                      "P_Value_UAV_0_10": pearsonr_uav_0_usi_10[1], "P_Value_UAV_0_20": pearsonr_uav_0_usi_20[1], "P_Value_UAV_0_667": pearsonr_uav_0_usi_667[1], "P_Value_UAV_0_100": pearsonr_uav_0_usi_100[1],
    #                      "P_Value_UAV_1_10": pearsonr_uav_1_usi_10[1], "P_Value_UAV_1_20": pearsonr_uav_1_usi_20[1], "P_Value_UAV_1_667": pearsonr_uav_1_usi_667[1], "P_Value_UAV_1_100": pearsonr_uav_1_usi_100[1],
    #                      "P_Value_UAV_2_10": pearsonr_uav_2_usi_10[1], "P_Value_UAV_2_20": pearsonr_uav_2_usi_20[1], "P_Value_UAV_2_667": pearsonr_uav_2_usi_667[1], "P_Value_UAV_2_100": pearsonr_uav_2_usi_100[1],
    #                      "P_Value_UAV_3_10": pearsonr_uav_3_usi_10[1], "P_Value_UAV_3_20": pearsonr_uav_3_usi_20[1], "P_Value_UAV_3_667": pearsonr_uav_3_usi_667[1], "P_Value_UAV_3_100": pearsonr_uav_3_usi_100[1],
    #                      "P_Value_UAV_4_10": pearsonr_uav_4_usi_10[1], "P_Value_UAV_4_20": pearsonr_uav_4_usi_20[1], "P_Value_UAV_4_667": pearsonr_uav_4_usi_667[1], "P_Value_UAV_4_100": pearsonr_uav_4_usi_100[1],
    #                      "P_Value_UAV_5_10": pearsonr_uav_5_usi_10[1], "P_Value_UAV_5_20": pearsonr_uav_5_usi_20[1], "P_Value_UAV_5_667": pearsonr_uav_5_usi_667[1], "P_Value_UAV_5_100": pearsonr_uav_5_usi_100[1],
    #                      "P_Value_UAV_6_10": pearsonr_uav_6_usi_10[1], "P_Value_UAV_6_20": pearsonr_uav_6_usi_20[1], "P_Value_UAV_6_667": pearsonr_uav_6_usi_667[1], "P_Value_UAV_6_100": pearsonr_uav_6_usi_100[1],
    #                      "P_Value_UAV_7_10": pearsonr_uav_7_usi_10[1], "P_Value_UAV_7_20": pearsonr_uav_7_usi_20[1], "P_Value_UAV_7_667": pearsonr_uav_7_usi_667[1], "P_Value_UAV_7_100": pearsonr_uav_7_usi_100[1], 
    #                      "P_Value_VID": pearsonr_vid[1]})
    
    # Spearman Rho
    spearmanr_uav_0 = stats.spearmanr(uav_0_throughput_df["Throughput"].values, uav_0_throughput_df["Measured_Reliability_1"].values)
    spearmanr_uav_1 = stats.spearmanr(uav_1_throughput_df["Throughput"].values, uav_1_throughput_df["Measured_Reliability_1"].values)
    spearmanr_uav_2 = stats.spearmanr(uav_2_throughput_df["Throughput"].values, uav_2_throughput_df["Measured_Reliability_1"].values)
    spearmanr_uav_3 = stats.spearmanr(uav_3_throughput_df["Throughput"].values, uav_3_throughput_df["Measured_Reliability_1"].values)
    spearmanr_uav_4 = stats.spearmanr(uav_4_throughput_df["Throughput"].values, uav_4_throughput_df["Measured_Reliability_1"].values)
    spearmanr_uav_5 = stats.spearmanr(uav_5_throughput_df["Throughput"].values, uav_5_throughput_df["Measured_Reliability_1"].values)
    spearmanr_uav_6 = stats.spearmanr(uav_6_throughput_df["Throughput"].values, uav_6_throughput_df["Measured_Reliability_1"].values)
    spearmanr_uav_7 = stats.spearmanr(uav_7_throughput_df["Throughput"].values, uav_7_throughput_df["Measured_Reliability_1"].values)
    spearmanr_ul_usi_10 = stats.spearmanr(ul_th_usi_10_df["Throughput"].values, ul_th_usi_10_df["Measured_Reliability_1"].values)
    spearmanr_uav_0_usi_10 = stats.spearmanr(uav_0_th_usi_10_df["Throughput"].values, uav_0_th_usi_10_df["Measured_Reliability_1"].values)
    spearmanr_uav_1_usi_10 = stats.spearmanr(uav_1_th_usi_10_df["Throughput"].values, uav_1_th_usi_10_df["Measured_Reliability_1"].values)
    spearmanr_uav_2_usi_10 = stats.spearmanr(uav_2_th_usi_10_df["Throughput"].values, uav_2_th_usi_10_df["Measured_Reliability_1"].values)
    spearmanr_uav_3_usi_10 = stats.spearmanr(uav_3_th_usi_10_df["Throughput"].values, uav_3_th_usi_10_df["Measured_Reliability_1"].values)
    spearmanr_uav_4_usi_10 = stats.spearmanr(uav_4_th_usi_10_df["Throughput"].values, uav_4_th_usi_10_df["Measured_Reliability_1"].values)
    spearmanr_uav_5_usi_10 = stats.spearmanr(uav_5_th_usi_10_df["Throughput"].values, uav_5_th_usi_10_df["Measured_Reliability_1"].values)
    spearmanr_uav_6_usi_10 = stats.spearmanr(uav_6_th_usi_10_df["Throughput"].values, uav_6_th_usi_10_df["Measured_Reliability_1"].values)
    spearmanr_uav_7_usi_10 = stats.spearmanr(uav_7_th_usi_10_df["Throughput"].values, uav_7_th_usi_10_df["Measured_Reliability_1"].values)
    spearmanr_ul_usi_20 = stats.spearmanr(ul_th_usi_20_df["Throughput"].values, ul_th_usi_20_df["Measured_Reliability_1"].values)
    spearmanr_uav_0_usi_20 = stats.spearmanr(uav_0_th_usi_20_df["Throughput"].values, uav_0_th_usi_20_df["Measured_Reliability_1"].values)
    spearmanr_uav_1_usi_20 = stats.spearmanr(uav_1_th_usi_20_df["Throughput"].values, uav_1_th_usi_20_df["Measured_Reliability_1"].values)
    spearmanr_uav_2_usi_20 = stats.spearmanr(uav_2_th_usi_20_df["Throughput"].values, uav_2_th_usi_20_df["Measured_Reliability_1"].values)
    spearmanr_uav_3_usi_20 = stats.spearmanr(uav_3_th_usi_20_df["Throughput"].values, uav_3_th_usi_20_df["Measured_Reliability_1"].values)
    spearmanr_uav_4_usi_20 = stats.spearmanr(uav_4_th_usi_20_df["Throughput"].values, uav_4_th_usi_20_df["Measured_Reliability_1"].values)
    spearmanr_uav_5_usi_20 = stats.spearmanr(uav_5_th_usi_20_df["Throughput"].values, uav_5_th_usi_20_df["Measured_Reliability_1"].values)
    spearmanr_uav_6_usi_20 = stats.spearmanr(uav_6_th_usi_20_df["Throughput"].values, uav_6_th_usi_20_df["Measured_Reliability_1"].values)
    spearmanr_uav_7_usi_20 = stats.spearmanr(uav_7_th_usi_20_df["Throughput"].values, uav_7_th_usi_20_df["Measured_Reliability_1"].values)
    spearmanr_ul_usi_667 = stats.spearmanr(ul_th_usi_667_df["Throughput"].values, ul_th_usi_667_df["Measured_Reliability_1"].values)
    spearmanr_uav_0_usi_667 = stats.spearmanr(uav_0_th_usi_667_df["Throughput"].values, uav_0_th_usi_667_df["Measured_Reliability_1"].values)
    spearmanr_uav_1_usi_667 = stats.spearmanr(uav_1_th_usi_667_df["Throughput"].values, uav_1_th_usi_667_df["Measured_Reliability_1"].values)
    spearmanr_uav_2_usi_667 = stats.spearmanr(uav_2_th_usi_667_df["Throughput"].values, uav_2_th_usi_667_df["Measured_Reliability_1"].values)
    spearmanr_uav_3_usi_667 = stats.spearmanr(uav_3_th_usi_667_df["Throughput"].values, uav_3_th_usi_667_df["Measured_Reliability_1"].values)
    spearmanr_uav_4_usi_667 = stats.spearmanr(uav_4_th_usi_667_df["Throughput"].values, uav_4_th_usi_667_df["Measured_Reliability_1"].values)
    spearmanr_uav_5_usi_667 = stats.spearmanr(uav_5_th_usi_667_df["Throughput"].values, uav_5_th_usi_667_df["Measured_Reliability_1"].values)
    spearmanr_uav_6_usi_667 = stats.spearmanr(uav_6_th_usi_667_df["Throughput"].values, uav_6_th_usi_667_df["Measured_Reliability_1"].values)
    spearmanr_uav_7_usi_667 = stats.spearmanr(uav_7_th_usi_667_df["Throughput"].values, uav_7_th_usi_667_df["Measured_Reliability_1"].values)
    spearmanr_ul_usi_100 = stats.spearmanr(ul_th_usi_100_df["Throughput"].values, ul_th_usi_100_df["Measured_Reliability_1"].values)
    spearmanr_uav_0_usi_100 = stats.spearmanr(uav_0_th_usi_100_df["Throughput"].values, uav_0_th_usi_100_df["Measured_Reliability_1"].values)
    spearmanr_uav_1_usi_100 = stats.spearmanr(uav_1_th_usi_100_df["Throughput"].values, uav_1_th_usi_100_df["Measured_Reliability_1"].values)
    spearmanr_uav_2_usi_100 = stats.spearmanr(uav_2_th_usi_100_df["Throughput"].values, uav_2_th_usi_100_df["Measured_Reliability_1"].values)
    spearmanr_uav_3_usi_100 = stats.spearmanr(uav_3_th_usi_100_df["Throughput"].values, uav_3_th_usi_100_df["Measured_Reliability_1"].values)
    spearmanr_uav_4_usi_100 = stats.spearmanr(uav_4_th_usi_100_df["Throughput"].values, uav_4_th_usi_100_df["Measured_Reliability_1"].values)
    spearmanr_uav_5_usi_100 = stats.spearmanr(uav_5_th_usi_100_df["Throughput"].values, uav_5_th_usi_100_df["Measured_Reliability_1"].values)
    spearmanr_uav_6_usi_100 = stats.spearmanr(uav_6_th_usi_100_df["Throughput"].values, uav_6_th_usi_100_df["Measured_Reliability_1"].values)
    spearmanr_uav_7_usi_100 = stats.spearmanr(uav_7_th_usi_100_df["Throughput"].values, uav_7_th_usi_100_df["Measured_Reliability_1"].values)
    spearmanr_vid = stats.spearmanr(vid_throughput_df["Throughput"].values, vid_throughput_df["Measured_Reliability_1"].values)

    spearman_results.append({"Scenario": scenario, 
                         "Spearman_UAV_0_All_USI": spearmanr_uav_0.correlation, "Spearman_UAV_1_All_USI": spearmanr_uav_1.correlation,
                         "Spearman_UAV_2_All_USI": spearmanr_uav_2.correlation, "Spearman_UAV_3_All_USI": spearmanr_uav_3.correlation, "Spearman_UAV_4_All_USI": spearmanr_uav_4.correlation,
                         "Spearman_UAV_5_All_USI": spearmanr_uav_5.correlation, "Spearman_UAV_6_All_USI": spearmanr_uav_6.correlation, "Spearman_UAV_7_All_USI": spearmanr_uav_7.correlation,
                         "Spearman_UL_10": spearmanr_ul_usi_10.correlation, "Spearman_UL_20": spearmanr_ul_usi_20.correlation, "Spearman_UL_667": spearmanr_ul_usi_667.correlation, "Spearman_UL_100": spearmanr_ul_usi_100.correlation, 
                         "Spearman_UAV_0_10": spearmanr_uav_0_usi_10.correlation, "Spearman_UAV_0_20": spearmanr_uav_0_usi_20.correlation, "Spearman_UAV_0_667": spearmanr_uav_0_usi_667.correlation, "Spearman_UAV_0_100": spearmanr_uav_0_usi_100.correlation,
                         "Spearman_UAV_1_10": spearmanr_uav_1_usi_10.correlation, "Spearman_UAV_1_20": spearmanr_uav_1_usi_20.correlation, "Spearman_UAV_1_667": spearmanr_uav_1_usi_667.correlation, "Spearman_UAV_1_100": spearmanr_uav_1_usi_100.correlation,
                         "Spearman_UAV_2_10": spearmanr_uav_2_usi_10.correlation, "Spearman_UAV_2_20": spearmanr_uav_2_usi_20.correlation, "Spearman_UAV_2_667": spearmanr_uav_2_usi_667.correlation, "Spearman_UAV_2_100": spearmanr_uav_2_usi_100.correlation,
                         "Spearman_UAV_3_10": spearmanr_uav_3_usi_10.correlation, "Spearman_UAV_3_20": spearmanr_uav_3_usi_20.correlation, "Spearman_UAV_3_667": spearmanr_uav_3_usi_667.correlation, "Spearman_UAV_3_100": spearmanr_uav_3_usi_100.correlation,
                         "Spearman_UAV_4_10": spearmanr_uav_4_usi_10.correlation, "Spearman_UAV_4_20": spearmanr_uav_4_usi_20.correlation, "Spearman_UAV_4_667": spearmanr_uav_4_usi_667.correlation, "Spearman_UAV_4_100": spearmanr_uav_4_usi_100.correlation,
                         "Spearman_UAV_5_10": spearmanr_uav_5_usi_10.correlation, "Spearman_UAV_5_20": spearmanr_uav_5_usi_20.correlation, "Spearman_UAV_5_667": spearmanr_uav_5_usi_667.correlation, "Spearman_UAV_5_100": spearmanr_uav_5_usi_100.correlation,
                         "Spearman_UAV_6_10": spearmanr_uav_6_usi_10.correlation, "Spearman_UAV_6_20": spearmanr_uav_6_usi_20.correlation, "Spearman_UAV_6_667": spearmanr_uav_6_usi_667.correlation, "Spearman_UAV_6_100": spearmanr_uav_6_usi_100.correlation,
                         "Spearman_UAV_7_10": spearmanr_uav_7_usi_10.correlation, "Spearman_UAV_7_20": spearmanr_uav_7_usi_20.correlation, "Spearman_UAV_7_667": spearmanr_uav_7_usi_667.correlation, "Spearman_UAV_7_100": spearmanr_uav_7_usi_100.correlation,
                         "Spearman_VID": spearmanr_vid.correlation, 
                         "P_Value_UAV_0": spearmanr_uav_0.pvalue, "P_Value_UAV_1": spearmanr_uav_1.pvalue,
                         "P_Value_UAV_2": spearmanr_uav_2.pvalue, "P_Value_UAV_3": spearmanr_uav_3.pvalue, "P_Value_UAV_4": spearmanr_uav_4.pvalue,
                         "P_Value_UAV_5": spearmanr_uav_5.pvalue, "P_Value_UAV_6": spearmanr_uav_6.pvalue, "P_Value_UAV_7": spearmanr_uav_7.pvalue,
                         "P_Value_UL_10": spearmanr_ul_usi_10.pvalue, "P_Value_UL_20": spearmanr_ul_usi_20.pvalue, "P_Value_UL_667": spearmanr_ul_usi_667.pvalue, "P_Value_UL_100": spearmanr_ul_usi_100.pvalue,
                         "P_Value_UAV_0_10": spearmanr_uav_0_usi_10.pvalue, "P_Value_UAV_0_20": spearmanr_uav_0_usi_20.pvalue, "P_Value_UAV_0_667": spearmanr_uav_0_usi_667.pvalue, "P_Value_UAV_0_100": spearmanr_uav_0_usi_100.pvalue,
                         "P_Value_UAV_1_10": spearmanr_uav_1_usi_10.pvalue, "P_Value_UAV_1_20": spearmanr_uav_1_usi_20.pvalue, "P_Value_UAV_1_667": spearmanr_uav_1_usi_667.pvalue, "P_Value_UAV_1_100": spearmanr_uav_1_usi_100.pvalue,
                         "P_Value_UAV_2_10": spearmanr_uav_2_usi_10.pvalue, "P_Value_UAV_2_20": spearmanr_uav_2_usi_20.pvalue, "P_Value_UAV_2_667": spearmanr_uav_2_usi_667.pvalue, "P_Value_UAV_2_100": spearmanr_uav_2_usi_100.pvalue,
                         "P_Value_UAV_3_10": spearmanr_uav_3_usi_10.pvalue, "P_Value_UAV_3_20": spearmanr_uav_3_usi_20.pvalue, "P_Value_UAV_3_667": spearmanr_uav_3_usi_667.pvalue, "P_Value_UAV_3_100": spearmanr_uav_3_usi_100.pvalue,
                         "P_Value_UAV_4_10": spearmanr_uav_4_usi_10.pvalue, "P_Value_UAV_4_20": spearmanr_uav_4_usi_20.pvalue, "P_Value_UAV_4_667": spearmanr_uav_4_usi_667.pvalue, "P_Value_UAV_4_100": spearmanr_uav_4_usi_100.pvalue,
                         "P_Value_UAV_5_10": spearmanr_uav_5_usi_10.pvalue, "P_Value_UAV_5_20": spearmanr_uav_5_usi_20.pvalue, "P_Value_UAV_5_667": spearmanr_uav_5_usi_667.pvalue, "P_Value_UAV_5_100": spearmanr_uav_5_usi_100.pvalue,
                         "P_Value_UAV_6_10": spearmanr_uav_6_usi_10.pvalue, "P_Value_UAV_6_20": spearmanr_uav_6_usi_20.pvalue, "P_Value_UAV_6_667": spearmanr_uav_6_usi_667.pvalue, "P_Value_UAV_6_100": spearmanr_uav_6_usi_100.pvalue,
                         "P_Value_UAV_7_10": spearmanr_uav_7_usi_10.pvalue, "P_Value_UAV_7_20": spearmanr_uav_7_usi_20.pvalue, "P_Value_UAV_7_667": spearmanr_uav_7_usi_667.pvalue, "P_Value_UAV_7_100": spearmanr_uav_7_usi_100.pvalue, 
                         "P_Value_VID": spearmanr_vid.pvalue})
    
    # # Kendall Tau
    # kendalltau_uav_0 = stats.kendalltau(uav_0_throughput_df["Throughput"].values, uav_0_throughput_df["Measured_Reliability_1"].values)
    # kendalltau_uav_1 = stats.kendalltau(uav_1_throughput_df["Throughput"].values, uav_1_throughput_df["Measured_Reliability_1"].values)
    # kendalltau_uav_2 = stats.kendalltau(uav_2_throughput_df["Throughput"].values, uav_2_throughput_df["Measured_Reliability_1"].values)
    # kendalltau_uav_3 = stats.kendalltau(uav_3_throughput_df["Throughput"].values, uav_3_throughput_df["Measured_Reliability_1"].values)
    # kendalltau_uav_4 = stats.kendalltau(uav_4_throughput_df["Throughput"].values, uav_4_throughput_df["Measured_Reliability_1"].values)
    # kendalltau_uav_5 = stats.kendalltau(uav_5_throughput_df["Throughput"].values, uav_5_throughput_df["Measured_Reliability_1"].values)
    # kendalltau_uav_6 = stats.kendalltau(uav_6_throughput_df["Throughput"].values, uav_6_throughput_df["Measured_Reliability_1"].values)
    # kendalltau_uav_7 = stats.kendalltau(uav_7_throughput_df["Throughput"].values, uav_7_throughput_df["Measured_Reliability_1"].values)
    # kendalltau_ul_usi_10 = stats.kendalltau(ul_th_usi_10_df["Throughput"].values, ul_th_usi_10_df["Measured_Reliability_1"].values)
    # kendalltau_uav_0_usi_10 = stats.kendalltau(uav_0_th_usi_10_df["Throughput"].values, uav_0_th_usi_10_df["Measured_Reliability_1"].values)
    # kendalltau_uav_1_usi_10 = stats.kendalltau(uav_1_th_usi_10_df["Throughput"].values, uav_1_th_usi_10_df["Measured_Reliability_1"].values)
    # kendalltau_uav_2_usi_10 = stats.kendalltau(uav_2_th_usi_10_df["Throughput"].values, uav_2_th_usi_10_df["Measured_Reliability_1"].values)
    # kendalltau_uav_3_usi_10 = stats.kendalltau(uav_3_th_usi_10_df["Throughput"].values, uav_3_th_usi_10_df["Measured_Reliability_1"].values)
    # kendalltau_uav_4_usi_10 = stats.kendalltau(uav_4_th_usi_10_df["Throughput"].values, uav_4_th_usi_10_df["Measured_Reliability_1"].values)
    # kendalltau_uav_5_usi_10 = stats.kendalltau(uav_5_th_usi_10_df["Throughput"].values, uav_5_th_usi_10_df["Measured_Reliability_1"].values)
    # kendalltau_uav_6_usi_10 = stats.kendalltau(uav_6_th_usi_10_df["Throughput"].values, uav_6_th_usi_10_df["Measured_Reliability_1"].values)
    # kendalltau_uav_7_usi_10 = stats.kendalltau(uav_7_th_usi_10_df["Throughput"].values, uav_7_th_usi_10_df["Measured_Reliability_1"].values)
    # kendalltau_ul_usi_20 = stats.kendalltau(ul_th_usi_20_df["Throughput"].values, ul_th_usi_20_df["Measured_Reliability_1"].values)
    # kendalltau_uav_0_usi_20 = stats.kendalltau(uav_0_th_usi_20_df["Throughput"].values, uav_0_th_usi_20_df["Measured_Reliability_1"].values)
    # kendalltau_uav_1_usi_20 = stats.kendalltau(uav_1_th_usi_20_df["Throughput"].values, uav_1_th_usi_20_df["Measured_Reliability_1"].values)
    # kendalltau_uav_2_usi_20 = stats.kendalltau(uav_2_th_usi_20_df["Throughput"].values, uav_2_th_usi_20_df["Measured_Reliability_1"].values)
    # kendalltau_uav_3_usi_20 = stats.kendalltau(uav_3_th_usi_20_df["Throughput"].values, uav_3_th_usi_20_df["Measured_Reliability_1"].values)
    # kendalltau_uav_4_usi_20 = stats.kendalltau(uav_4_th_usi_20_df["Throughput"].values, uav_4_th_usi_20_df["Measured_Reliability_1"].values)
    # kendalltau_uav_5_usi_20 = stats.kendalltau(uav_5_th_usi_20_df["Throughput"].values, uav_5_th_usi_20_df["Measured_Reliability_1"].values)
    # kendalltau_uav_6_usi_20 = stats.kendalltau(uav_6_th_usi_20_df["Throughput"].values, uav_6_th_usi_20_df["Measured_Reliability_1"].values)
    # kendalltau_uav_7_usi_20 = stats.kendalltau(uav_7_th_usi_20_df["Throughput"].values, uav_7_th_usi_20_df["Measured_Reliability_1"].values)
    # kendalltau_ul_usi_667 = stats.kendalltau(ul_th_usi_667_df["Throughput"].values, ul_th_usi_667_df["Measured_Reliability_1"].values)
    # kendalltau_uav_0_usi_667 = stats.kendalltau(uav_0_th_usi_667_df["Throughput"].values, uav_0_th_usi_667_df["Measured_Reliability_1"].values)
    # kendalltau_uav_1_usi_667 = stats.kendalltau(uav_1_th_usi_667_df["Throughput"].values, uav_1_th_usi_667_df["Measured_Reliability_1"].values)
    # kendalltau_uav_2_usi_667 = stats.kendalltau(uav_2_th_usi_667_df["Throughput"].values, uav_2_th_usi_667_df["Measured_Reliability_1"].values)
    # kendalltau_uav_3_usi_667 = stats.kendalltau(uav_3_th_usi_667_df["Throughput"].values, uav_3_th_usi_667_df["Measured_Reliability_1"].values)
    # kendalltau_uav_4_usi_667 = stats.kendalltau(uav_4_th_usi_667_df["Throughput"].values, uav_4_th_usi_667_df["Measured_Reliability_1"].values)
    # kendalltau_uav_5_usi_667 = stats.kendalltau(uav_5_th_usi_667_df["Throughput"].values, uav_5_th_usi_667_df["Measured_Reliability_1"].values)
    # kendalltau_uav_6_usi_667 = stats.kendalltau(uav_6_th_usi_667_df["Throughput"].values, uav_6_th_usi_667_df["Measured_Reliability_1"].values)
    # kendalltau_uav_7_usi_667 = stats.kendalltau(uav_7_th_usi_667_df["Throughput"].values, uav_7_th_usi_667_df["Measured_Reliability_1"].values)
    # kendalltau_ul_usi_100 = stats.kendalltau(ul_th_usi_100_df["Throughput"].values, ul_th_usi_100_df["Measured_Reliability_1"].values)
    # kendalltau_uav_0_usi_100 = stats.kendalltau(uav_0_th_usi_100_df["Throughput"].values, uav_0_th_usi_100_df["Measured_Reliability_1"].values)
    # kendalltau_uav_1_usi_100 = stats.kendalltau(uav_1_th_usi_100_df["Throughput"].values, uav_1_th_usi_100_df["Measured_Reliability_1"].values)
    # kendalltau_uav_2_usi_100 = stats.kendalltau(uav_2_th_usi_100_df["Throughput"].values, uav_2_th_usi_100_df["Measured_Reliability_1"].values)
    # kendalltau_uav_3_usi_100 = stats.kendalltau(uav_3_th_usi_100_df["Throughput"].values, uav_3_th_usi_100_df["Measured_Reliability_1"].values)
    # kendalltau_uav_4_usi_100 = stats.kendalltau(uav_4_th_usi_100_df["Throughput"].values, uav_4_th_usi_100_df["Measured_Reliability_1"].values)
    # kendalltau_uav_5_usi_100 = stats.kendalltau(uav_5_th_usi_100_df["Throughput"].values, uav_5_th_usi_100_df["Measured_Reliability_1"].values)
    # kendalltau_uav_6_usi_100 = stats.kendalltau(uav_6_th_usi_100_df["Throughput"].values, uav_6_th_usi_100_df["Measured_Reliability_1"].values)
    # kendalltau_uav_7_usi_100 = stats.kendalltau(uav_7_th_usi_100_df["Throughput"].values, uav_7_th_usi_100_df["Measured_Reliability_1"].values)
    # kendalltau_vid = stats.kendalltau(vid_throughput_df["Throughput"].values, vid_throughput_df["Measured_Reliability_1"].values)

    # kendall_results.append({"Scenario": scenario, 
    #                      "Kendall_UAV_0_All_USI": kendalltau_uav_0.correlation, "Kendall_UAV_1_All_USI": kendalltau_uav_1.correlation,
    #                      "Kendall_UAV_2_All_USI": kendalltau_uav_2.correlation, "Kendall_UAV_3_All_USI": kendalltau_uav_3.correlation, "Kendall_UAV_4_All_USI": kendalltau_uav_4.correlation,
    #                      "Kendall_UAV_5_All_USI": kendalltau_uav_5.correlation, "Kendall_UAV_6_All_USI": kendalltau_uav_6.correlation, "Kendall_UAV_7_All_USI": kendalltau_uav_7.correlation,
    #                      "Kendall_UL_10": kendalltau_ul_usi_10.correlation, "Kendall_UL_20": kendalltau_ul_usi_20.correlation, "Kendall_UL_667": kendalltau_ul_usi_667.correlation, "Kendall_UL_100": kendalltau_ul_usi_100.correlation, 
    #                      "Kendall_UAV_0_10": kendalltau_uav_0_usi_10.correlation, "Kendall_UAV_0_20": kendalltau_uav_0_usi_20.correlation, "Kendall_UAV_0_667": kendalltau_uav_0_usi_667.correlation, "Kendall_UAV_0_100": kendalltau_uav_0_usi_100.correlation,
    #                      "Kendall_UAV_1_10": kendalltau_uav_1_usi_10.correlation, "Kendall_UAV_1_20": kendalltau_uav_1_usi_20.correlation, "Kendall_UAV_1_667": kendalltau_uav_1_usi_667.correlation, "Kendall_UAV_1_100": kendalltau_uav_1_usi_100.correlation,
    #                      "Kendall_UAV_2_10": kendalltau_uav_2_usi_10.correlation, "Kendall_UAV_2_20": kendalltau_uav_2_usi_20.correlation, "Kendall_UAV_2_667": kendalltau_uav_2_usi_667.correlation, "Kendall_UAV_2_100": kendalltau_uav_2_usi_100.correlation,
    #                      "Kendall_UAV_3_10": kendalltau_uav_3_usi_10.correlation, "Kendall_UAV_3_20": kendalltau_uav_3_usi_20.correlation, "Kendall_UAV_3_667": kendalltau_uav_3_usi_667.correlation, "Kendall_UAV_3_100": kendalltau_uav_3_usi_100.correlation,
    #                      "Kendall_UAV_4_10": kendalltau_uav_4_usi_10.correlation, "Kendall_UAV_4_20": kendalltau_uav_4_usi_20.correlation, "Kendall_UAV_4_667": kendalltau_uav_4_usi_667.correlation, "Kendall_UAV_4_100": kendalltau_uav_4_usi_100.correlation,
    #                      "Kendall_UAV_5_10": kendalltau_uav_5_usi_10.correlation, "Kendall_UAV_5_20": kendalltau_uav_5_usi_20.correlation, "Kendall_UAV_5_667": kendalltau_uav_5_usi_667.correlation, "Kendall_UAV_5_100": kendalltau_uav_5_usi_100.correlation,
    #                      "Kendall_UAV_6_10": kendalltau_uav_6_usi_10.correlation, "Kendall_UAV_6_20": kendalltau_uav_6_usi_20.correlation, "Kendall_UAV_6_667": kendalltau_uav_6_usi_667.correlation, "Kendall_UAV_6_100": kendalltau_uav_6_usi_100.correlation,
    #                      "Kendall_UAV_7_10": kendalltau_uav_7_usi_10.correlation, "Kendall_UAV_7_20": kendalltau_uav_7_usi_20.correlation, "Kendall_UAV_7_667": kendalltau_uav_7_usi_667.correlation, "Kendall_UAV_7_100": kendalltau_uav_7_usi_100.correlation,
    #                      "Kendall_VID": kendalltau_vid.correlation, 
    #                      "P_Value_UAV_0": kendalltau_uav_0.pvalue, "P_Value_UAV_1": kendalltau_uav_1.pvalue,
    #                      "P_Value_UAV_2": kendalltau_uav_2.pvalue, "P_Value_UAV_3": kendalltau_uav_3.pvalue, "P_Value_UAV_4": kendalltau_uav_4.pvalue,
    #                      "P_Value_UAV_5": kendalltau_uav_5.pvalue, "P_Value_UAV_6": kendalltau_uav_6.pvalue, "P_Value_UAV_7": kendalltau_uav_7.pvalue,
    #                      "P_Value_UL_10": kendalltau_ul_usi_10.pvalue, "P_Value_UL_20": kendalltau_ul_usi_20.pvalue, "P_Value_UL_667": kendalltau_ul_usi_667.pvalue, "P_Value_UL_100": kendalltau_ul_usi_100.pvalue,
    #                      "P_Value_UAV_0_10": kendalltau_uav_0_usi_10.pvalue, "P_Value_UAV_0_20": kendalltau_uav_0_usi_20.pvalue, "P_Value_UAV_0_667": kendalltau_uav_0_usi_667.pvalue, "P_Value_UAV_0_100": kendalltau_uav_0_usi_100.pvalue,
    #                      "P_Value_UAV_1_10": kendalltau_uav_1_usi_10.pvalue, "P_Value_UAV_1_20": kendalltau_uav_1_usi_20.pvalue, "P_Value_UAV_1_667": kendalltau_uav_1_usi_667.pvalue, "P_Value_UAV_1_100": kendalltau_uav_1_usi_100.pvalue,
    #                      "P_Value_UAV_2_10": kendalltau_uav_2_usi_10.pvalue, "P_Value_UAV_2_20": kendalltau_uav_2_usi_20.pvalue, "P_Value_UAV_2_667": kendalltau_uav_2_usi_667.pvalue, "P_Value_UAV_2_100": kendalltau_uav_2_usi_100.pvalue,
    #                      "P_Value_UAV_3_10": kendalltau_uav_3_usi_10.pvalue, "P_Value_UAV_3_20": kendalltau_uav_3_usi_20.pvalue, "P_Value_UAV_3_667": kendalltau_uav_3_usi_667.pvalue, "P_Value_UAV_3_100": kendalltau_uav_3_usi_100.pvalue,
    #                      "P_Value_UAV_4_10": kendalltau_uav_4_usi_10.pvalue, "P_Value_UAV_4_20": kendalltau_uav_4_usi_20.pvalue, "P_Value_UAV_4_667": kendalltau_uav_4_usi_667.pvalue, "P_Value_UAV_4_100": kendalltau_uav_4_usi_100.pvalue,
    #                      "P_Value_UAV_5_10": kendalltau_uav_5_usi_10.pvalue, "P_Value_UAV_5_20": kendalltau_uav_5_usi_20.pvalue, "P_Value_UAV_5_667": kendalltau_uav_5_usi_667.pvalue, "P_Value_UAV_5_100": kendalltau_uav_5_usi_100.pvalue,
    #                      "P_Value_UAV_6_10": kendalltau_uav_6_usi_10.pvalue, "P_Value_UAV_6_20": kendalltau_uav_6_usi_20.pvalue, "P_Value_UAV_6_667": kendalltau_uav_6_usi_667.pvalue, "P_Value_UAV_6_100": kendalltau_uav_6_usi_100.pvalue,
    #                      "P_Value_UAV_7_10": kendalltau_uav_7_usi_10.pvalue, "P_Value_UAV_7_20": kendalltau_uav_7_usi_20.pvalue, "P_Value_UAV_7_667": kendalltau_uav_7_usi_667.pvalue, "P_Value_UAV_7_100": kendalltau_uav_7_usi_100.pvalue, 
    #                      "P_Value_VID": kendalltau_vid.pvalue})

    # Plot SVG
    # if (len(uav_0_throughput_df)>NUM_SAMPLES_PLOT):
    #     uav_0_sample_df = uav_0_throughput_df.sample(n=NUM_SAMPLES_PLOT, replace=False, random_state=SEED)
    # else:
    #     uav_0_sample_df = uav_0_throughput_df
    # if (len(uav_1_throughput_df)>NUM_SAMPLES_PLOT):
    #     uav_1_sample_df = uav_1_throughput_df.sample(n=NUM_SAMPLES_PLOT, replace=False, random_state=SEED)
    # else:
    #     uav_1_sample_df = uav_1_throughput_df
    # if (len(uav_2_throughput_df)>NUM_SAMPLES_PLOT):
    #     uav_2_sample_df = uav_2_throughput_df.sample(n=NUM_SAMPLES_PLOT, replace=False, random_state=SEED)
    # else:
    #     uav_2_sample_df = uav_2_throughput_df
    # if (len(uav_3_throughput_df)>NUM_SAMPLES_PLOT):
    #     uav_3_sample_df = uav_3_throughput_df.sample(n=NUM_SAMPLES_PLOT, replace=False, random_state=SEED)
    # else:
    #     uav_3_sample_df = uav_3_throughput_df
    # if (len(uav_4_throughput_df)>NUM_SAMPLES_PLOT):
    #     uav_4_sample_df = uav_4_throughput_df.sample(n=NUM_SAMPLES_PLOT, replace=False, random_state=SEED)
    # else:
    #     uav_4_sample_df = uav_4_throughput_df
    # if (len(uav_5_throughput_df)>NUM_SAMPLES_PLOT):
    #     uav_5_sample_df = uav_5_throughput_df.sample(n=NUM_SAMPLES_PLOT, replace=False, random_state=SEED)
    # else:
    #     uav_5_sample_df = uav_5_throughput_df
    # if (len(uav_6_throughput_df)>NUM_SAMPLES_PLOT):
    #     uav_6_sample_df = uav_6_throughput_df.sample(n=NUM_SAMPLES_PLOT, replace=False, random_state=SEED)
    # else:
    #     uav_6_sample_df = uav_6_throughput_df
    # if (len(uav_7_throughput_df)>NUM_SAMPLES_PLOT):
    #     uav_7_sample_df = uav_7_throughput_df.sample(n=NUM_SAMPLES_PLOT, replace=False, random_state=SEED)
    # else:
    #     uav_7_sample_df = uav_7_throughput_df

    # if (len(ul_th_usi_10_df)>NUM_SAMPLES_PLOT):
    #     ul_usi_10_sample_df = ul_th_usi_10_df.sample(n=NUM_SAMPLES_PLOT, replace=False, random_state=SEED)
    # else:
    #     ul_usi_10_sample_df = ul_th_usi_10_df
    # if (len(ul_th_usi_20_df)>NUM_SAMPLES_PLOT):
    #     ul_usi_20_sample_df = ul_th_usi_20_df.sample(n=NUM_SAMPLES_PLOT, replace=False, random_state=SEED)
    # else:
    #     ul_usi_20_sample_df = ul_th_usi_20_df
    # if (len(ul_th_usi_667_df)>NUM_SAMPLES_PLOT):
    #     ul_usi_667_sample_df = ul_th_usi_667_df.sample(n=NUM_SAMPLES_PLOT, replace=False, random_state=SEED)
    # else:
    #     ul_usi_667_sample_df = ul_th_usi_667_df
    # if (len(ul_th_usi_100_df)>NUM_SAMPLES_PLOT):
    #     ul_usi_100_sample_df = ul_th_usi_100_df.sample(n=NUM_SAMPLES_PLOT, replace=False, random_state=SEED)
    # else:
    #     ul_usi_100_sample_df = ul_th_usi_100_df

    # if (len(vid_throughput_df)>NUM_SAMPLES_PLOT):
    #     vid_sample_df = vid_throughput_df.sample(n=NUM_SAMPLES_PLOT, replace=False, random_state=SEED)
    # else:
    #     vid_sample_df = vid_throughput_df

    # plt.figure(figsize=(10,6))
    # plt.figure(figsize=(10,6))
    # plt.rcParams.update({'font.size': 16})
    # plt.scatter(uav_0_sample_df["Throughput"], uav_0_sample_df["Measured_Reliability_1"], c="tab:blue", alpha=0.006, linewidths=1, label="Gateway")
    # plt.scatter(uav_1_sample_df["Throughput"], uav_1_sample_df["Measured_Reliability_1"], c="tab:orange", alpha=0.006, linewidths=1, label="UAV 1")
    # plt.scatter(uav_2_sample_df["Throughput"], uav_2_sample_df["Measured_Reliability_1"], c="tab:green", alpha=0.006, linewidths=1, label="UAV 2")
    # plt.scatter(uav_3_sample_df["Throughput"], uav_3_sample_df["Measured_Reliability_1"], c="tab:red", alpha=0.006, linewidths=1, label="UAV 3")
    # plt.scatter(uav_4_sample_df["Throughput"], uav_4_sample_df["Measured_Reliability_1"], c="tab:purple", alpha=0.006, linewidths=1, label="UAV 4")
    # plt.scatter(uav_5_sample_df["Throughput"], uav_5_sample_df["Measured_Reliability_1"], c="tab:brown", alpha=0.006, linewidths=1, label="UAV 5")
    # plt.scatter(uav_6_sample_df["Throughput"], uav_6_sample_df["Measured_Reliability_1"], c="tab:pink", alpha=0.006, linewidths=1, label="UAV 6")
    # plt.scatter(uav_7_sample_df["Throughput"], uav_7_sample_df["Measured_Reliability_1"], c="tab:gray", alpha=0.006, linewidths=1, label="UAV 7")
    # plt.ylabel("Reliability")
    # plt.xlabel("Throughput (bytes/s)")
    # plt.xlim(left=0)
    # plt.ylim((0,1.05))
    # plt.legend(fontsize=14)
    # plt.savefig(os.path.join(SAVE_SVG, "{}_dl_corr_plot.{}".format(scenario, IMG_FORMAT)))

    # plt.figure(figsize=(10,6))
    # plt.rcParams.update({'font.size': 16})
    # plt.scatter(ul_usi_10_sample_df["Throughput"], ul_usi_10_sample_df["Measured_Reliability_1"], c="tab:blue", alpha=0.006, linewidths=1, label="USI: 10 ms")
    # plt.scatter(ul_usi_20_sample_df["Throughput"], ul_usi_20_sample_df["Measured_Reliability_1"], c="tab:orange", alpha=0.006, linewidths=1, label="USI: 20 ms")
    # plt.scatter(ul_usi_667_sample_df["Throughput"], ul_usi_667_sample_df["Measured_Reliability_1"], c="tab:green", alpha=0.006, linewidths=1, label="USI: 66.7 ms")
    # plt.scatter(ul_usi_100_sample_df["Throughput"], ul_usi_100_sample_df["Measured_Reliability_1"], c="tab:red", alpha=0.006, linewidths=1, label="USI: 100 ms")
    # plt.ylabel("Reliability")
    # plt.xlabel("Throughput (bytes/s)")
    # plt.xlim(left=0)
    # plt.ylim((0,1.05))
    # plt.legend(fontsize=14)
    # plt.savefig(os.path.join(SAVE_SVG, "{}_ul_corr_plot.{}".format(scenario, IMG_FORMAT)))

    # plt.figure(figsize=(10,6))
    # plt.rcParams.update({'font.size': 14})
    # plt.scatter(vid_sample_df["Throughput"], vid_sample_df["Measured_Reliability_1"], c="k", alpha=0.1, linewidths=1)
    # plt.ylabel("Reliability")
    # plt.xlabel("Throughput (bytes/s)")
    # plt.xlim(left=0)
    # plt.ylim((0,1.05))
    # plt.savefig(os.path.join(SAVE_SVG, "{}_vid_corr_plot.{}".format(scenario, IMG_FORMAT)))

# pearson_df = pd.DataFrame(pearson_results)
# pearson_df.to_csv(SAVE_CORRELATION_PEARSON)
spearman_df = pd.DataFrame(spearman_results)
spearman_df.to_csv(SAVE_CORRELATION_SPEARMAN)
# kendall_df = pd.DataFrame(kendall_results)
# kendall_df.to_csv(SAVE_CORRELATION_KENDALL)