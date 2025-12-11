# Date: 04/03/2025
# Desc: Calc and Plot Correlation Between Gamma and Measured Reliability 1 (version calculated from packets rcvd reliably and dropped in a time window)
# Modified from: fanet_measured_throughput_vs_measured_reliability.py
# Modified: To calculate correlation for each runs using data from that run only

import pandas as pd
import numpy as np 
import glob, os
from scipy import stats
from tqdm import tqdm

if __name__ == "__main__":
    '''Load the compiled data from CSV files'''
    NUM_UAV = 8
    DATASET_PATH = ["/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_0_{}_processed.csv", 
                    "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_1_{}_processed.csv",
                    "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_uav_interference/uav_scenario_2_{}_processed.csv",
                    "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_1_{}_processed.csv",
                    "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_a_{}_processed.csv",
                    "/media/research-student/KingstonSSD1/FANET_Dataset/DJISpark_Measured_Proxy/data_manual_throughput_manet_interference/manet_scenario_2_{}_processed.csv"]

    SAVE_CORRELATION_SPEARMAN = "/home/research-student/reuben_ws/omnet-fanet/results_new_Mar25/Throughput_Spearman_Runs_Correlation.csv"
    NUM_RUNS_START = 0 # Start range of run number to consider (inclusive)
    NUM_RUNS_END = 499 # Start range of run number to consider (inclusive)
    UNIQUE = True # Whether to filter out duplicated data or not

    spearman_results = []

    for dataset in DATASET_PATH:
        print(dataset)
        scenario_paths = [x[0] for x in os.walk(dataset) if (os.path.isdir(x[0]) and x[0]!=dataset)]
        scenario = dataset.split("/")[-1]

        '''Load the compiled data from CSV files'''
        ul_throughput_df = pd.read_csv(dataset.format("ul"))
        ul_throughput_df = ul_throughput_df.loc[(ul_throughput_df["Run_Num"] >= NUM_RUNS_START) & (ul_throughput_df["Run_Num"] <= NUM_RUNS_END)]
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

        '''Get the different combinations of USI, Bit_Rate, Speed, and Height using groupby'''
        gcs_groups = ul_throughput_df.groupby(['USI', 'Bit_Rate', 'UAV_Height', 'UAV_Speed'])
        uav0_groups = uav_0_throughput_df.groupby(['USI', 'Bit_Rate', 'UAV_Height', 'UAV_Speed'])
        uav1_groups = uav_1_throughput_df.groupby(['USI', 'Bit_Rate', 'UAV_Height', 'UAV_Speed'])
        uav2_groups = uav_2_throughput_df.groupby(['USI', 'Bit_Rate', 'UAV_Height', 'UAV_Speed'])
        uav3_groups = uav_3_throughput_df.groupby(['USI', 'Bit_Rate', 'UAV_Height', 'UAV_Speed'])
        uav4_groups = uav_4_throughput_df.groupby(['USI', 'Bit_Rate', 'UAV_Height', 'UAV_Speed'])
        uav5_groups = uav_5_throughput_df.groupby(['USI', 'Bit_Rate', 'UAV_Height', 'UAV_Speed'])
        uav6_groups = uav_6_throughput_df.groupby(['USI', 'Bit_Rate', 'UAV_Height', 'UAV_Speed'])
        uav7_groups = uav_7_throughput_df.groupby(['USI', 'Bit_Rate', 'UAV_Height', 'UAV_Speed'])

        '''Get the Spearman correlation over all data from all runs of each group (combinations)'''
        print("EVALUATING EACH GROUP")
        for group in tqdm(set(gcs_groups.groups.keys())):
            gcs_group_df = gcs_groups.get_group(group)
            uav0_group_df = uav0_groups.get_group(group)
            uav1_group_df = uav1_groups.get_group(group)
            uav2_group_df = uav2_groups.get_group(group)
            uav3_group_df = uav3_groups.get_group(group)
            uav4_group_df = uav4_groups.get_group(group)
            uav5_group_df = uav5_groups.get_group(group)
            uav6_group_df = uav6_groups.get_group(group)
            uav7_group_df = uav7_groups.get_group(group)

            gcs_run_groups = gcs_group_df.groupby(["Run_Num"])
            uav0_run_groups = uav0_group_df.groupby(["Run_Num"])
            uav1_run_groups = uav1_group_df.groupby(["Run_Num"])
            uav2_run_groups = uav2_group_df.groupby(["Run_Num"])
            uav3_run_groups = uav3_group_df.groupby(["Run_Num"])
            uav4_run_groups = uav4_group_df.groupby(["Run_Num"])
            uav5_run_groups = uav5_group_df.groupby(["Run_Num"])
            uav6_run_groups = uav6_group_df.groupby(["Run_Num"])
            uav7_run_groups = uav7_group_df.groupby(["Run_Num"])

            # Group by runs
            for run_group in set(gcs_run_groups.groups.keys()):
                gcs_run_df = gcs_run_groups.get_group(run_group)
                uav0_run_df = uav0_run_groups.get_group(run_group)
                uav1_run_df = uav1_run_groups.get_group(run_group)
                uav2_run_df = uav2_run_groups.get_group(run_group)
                uav3_run_df = uav3_run_groups.get_group(run_group)
                uav4_run_df = uav4_run_groups.get_group(run_group)
                uav5_run_df = uav5_run_groups.get_group(run_group)
                uav6_run_df = uav6_run_groups.get_group(run_group)
                uav7_run_df = uav7_run_groups.get_group(run_group)

                # Drop duplicates before calculating Spearman and Kendall
                if UNIQUE:
                    gcs_run_df = gcs_run_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
                    uav0_run_df = uav0_run_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
                    uav1_run_df = uav1_run_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
                    uav2_run_df = uav2_run_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
                    uav3_run_df = uav3_run_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
                    uav4_run_df = uav4_run_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
                    uav5_run_df = uav5_run_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
                    uav6_run_df = uav6_run_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
                    uav7_run_df = uav7_run_df.drop_duplicates(subset=["Throughput", "Measured_Reliability_1"])
            
                # Spearman Rho
                spearmanr_gcs = stats.spearmanr(gcs_run_df["Throughput"].values, gcs_run_df["Measured_Reliability_1"].values)
                spearmanr_uav_0 = stats.spearmanr(uav0_run_df["Throughput"].values, uav0_run_df["Measured_Reliability_1"].values)
                spearmanr_uav_1 = stats.spearmanr(uav1_run_df["Throughput"].values, uav1_run_df["Measured_Reliability_1"].values)
                spearmanr_uav_2 = stats.spearmanr(uav2_run_df["Throughput"].values, uav2_run_df["Measured_Reliability_1"].values)
                spearmanr_uav_3 = stats.spearmanr(uav3_run_df["Throughput"].values, uav3_run_df["Measured_Reliability_1"].values)
                spearmanr_uav_4 = stats.spearmanr(uav4_run_df["Throughput"].values, uav4_run_df["Measured_Reliability_1"].values)
                spearmanr_uav_5 = stats.spearmanr(uav5_run_df["Throughput"].values, uav5_run_df["Measured_Reliability_1"].values)
                spearmanr_uav_6 = stats.spearmanr(uav6_run_df["Throughput"].values, uav6_run_df["Measured_Reliability_1"].values)
                spearmanr_uav_7 = stats.spearmanr(uav7_run_df["Throughput"].values, uav7_run_df["Measured_Reliability_1"].values)
      
                spearman_results.append({"Scenario": scenario, "Run_Num": run_group, "Combination": group, 
                                        "Spearman_GCS": spearmanr_gcs.correlation, "Spearman_UAV_0": spearmanr_uav_0.correlation, "Spearman_UAV_1": spearmanr_uav_1.correlation,
                                        "Spearman_UAV_2": spearmanr_uav_2.correlation, "Spearman_UAV_3": spearmanr_uav_3.correlation, "Spearman_UAV_4": spearmanr_uav_4.correlation,
                                        "Spearman_UAV_5": spearmanr_uav_5.correlation, "Spearman_UAV_6": spearmanr_uav_6.correlation, "Spearman_UAV_7": spearmanr_uav_7.correlation,
                                        "P_Value_GCS": spearmanr_gcs.pvalue, "P_Value_UAV_0": spearmanr_uav_0.pvalue, "P_Value_UAV_1": spearmanr_uav_1.pvalue,
                                        "P_Value_UAV_2": spearmanr_uav_2.pvalue, "P_Value_UAV_3": spearmanr_uav_3.pvalue, "P_Value_UAV_4": spearmanr_uav_4.pvalue,
                                        "P_Value_UAV_5": spearmanr_uav_5.pvalue, "P_Value_UAV_6": spearmanr_uav_6.pvalue, "P_Value_UAV_7": spearmanr_uav_7.pvalue, 
                                        "GCS_DF_Len": len(gcs_run_df), "UAV_0_DF_Len": len(uav0_run_df), "UAV_1_DF_Len": len(uav1_run_df), "UAV_2_DF_Len": len(uav2_run_df),
                                        "UAV_3_DF_Len": len(uav3_run_df), "UAV_4_DF_Len": len(uav4_run_df), "UAV_5_DF_Len": len(uav5_run_df), "UAV_6_DF_Len": len(uav6_run_df), "UAV_7_DF_Len": len(uav7_run_df)})
        
        spearman_df = pd.DataFrame(spearman_results)
        spearman_df.to_csv(SAVE_CORRELATION_SPEARMAN)
