# To compute the Youden Index Threshold (Gamma and Throughput) for Different Scenario and USI
# Note: Since positive is when gamma or tau is below the threshold, we need to negate the parameter for it to work with sklearn roc_curve

import pandas as pd
import numpy as np 
import os
from sklearn import metrics

def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds,tpr, fpr))
    return j_ordered[-1]

if __name__ == "__main__":
    DATASET_INT_PATHS = ["/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_0_{}_processed.csv", 
                            "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_1_{}_processed.csv",
                            "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_uav_interference/uav_scenario_2_{}_processed.csv",
                            "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_0_{}_processed.csv",
                            "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_1_{}_processed.csv",
                            "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_a_{}_processed.csv",
                            "/media/research-student/KingstonSSD/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference/manet_scenario_2_{}_processed.csv"]
    SAVE_PATH = "/home/research-student/omnet-fanet/data-processing-scripts/journal_3_scripts/roc_auc_analysis.csv"
    REL_TH = 0.999
    results = []
    for datasets in DATASET_INT_PATHS:
        scenario_name = datasets.split("/")[-1]
        print(scenario_name)

        # Load dataset
        ul_df = pd.read_csv(datasets.format("ul"))
        uav_0_df = pd.read_csv(datasets.format("UAV_0"))
        uav_1_df = pd.read_csv(datasets.format("UAV_1"))
        uav_2_df = pd.read_csv(datasets.format("UAV_2"))
        uav_3_df = pd.read_csv(datasets.format("UAV_3"))
        uav_4_df = pd.read_csv(datasets.format("UAV_4"))
        uav_5_df = pd.read_csv(datasets.format("UAV_5"))
        uav_6_df = pd.read_csv(datasets.format("UAV_6"))
        uav_7_df = pd.read_csv(datasets.format("UAV_7"))

        # Filter
        ul_df = ul_df.loc[ul_df["Time"] >= 1]
        uav_0_df = uav_0_df.loc[uav_0_df["Time"] >= 1]
        uav_1_df = uav_1_df.loc[uav_1_df["Time"] >= 1]
        uav_2_df = uav_2_df.loc[uav_2_df["Time"] >= 1]
        uav_3_df = uav_3_df.loc[uav_3_df["Time"] >= 1]
        uav_4_df = uav_4_df.loc[uav_4_df["Time"] >= 1]
        uav_5_df = uav_5_df.loc[uav_5_df["Time"] >= 1]
        uav_6_df = uav_6_df.loc[uav_6_df["Time"] >= 1]
        uav_7_df = uav_7_df.loc[uav_7_df["Time"] >= 1]

        # Assign GT
        ul_df["Fail"] = ul_df["Measured_Reliability_1"] < REL_TH
        uav_0_df["Fail"] = uav_0_df["Measured_Reliability_1"] < REL_TH
        uav_1_df["Fail"] = uav_1_df["Measured_Reliability_1"] < REL_TH
        uav_2_df["Fail"] = uav_2_df["Measured_Reliability_1"] < REL_TH
        uav_3_df["Fail"] = uav_3_df["Measured_Reliability_1"] < REL_TH
        uav_4_df["Fail"] = uav_4_df["Measured_Reliability_1"] < REL_TH
        uav_5_df["Fail"] = uav_5_df["Measured_Reliability_1"] < REL_TH
        uav_6_df["Fail"] = uav_6_df["Measured_Reliability_1"] < REL_TH
        uav_7_df["Fail"] = uav_7_df["Measured_Reliability_1"] < REL_TH

        dl_df = pd.concat([uav_0_df, uav_1_df, uav_2_df, uav_3_df, uav_4_df, uav_5_df, uav_6_df, uav_7_df])

        ''' For Gamma '''
        gamma_yj_list = []
        gamma_threshold_list = []
        gamma_auc_list = []
        # Find Youden Indexes and ROC AUC for each USI 
        for usi in [10, 20, 66.7, 100]:

            # This is to handle the case where there is no negative cases
            if (usi == 10) and ((scenario_name == "uav_scenario_1_{}_processed.csv") or (scenario_name == "uav_scenario_2_{}_processed.csv")):
                ul_yj = np.nan
                ul_th = np.nan
                ul_auc = np.nan
            else: 
                tmp_ul_df = ul_df.loc[ul_df["USI"] == usi]
                fpr, tpr, thresholds = metrics.roc_curve(tmp_ul_df["Fail"].values, -tmp_ul_df["Num_Reliable"].values, pos_label=True)
                ul_yj, ul_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
                ul_auc = metrics.auc(fpr, tpr)

            tmp_uav_0_df = uav_0_df.loc[uav_0_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_uav_0_df["Fail"].values, -tmp_uav_0_df["Num_Reliable"].values, pos_label=True)
            uav_0_yj, uav_0_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                uav_0_auc = metrics.auc(fpr, tpr)
            except: # This is to handle the case where there is no negative cases
                uav_0_auc = np.nan

            tmp_uav_1_df = uav_1_df.loc[uav_1_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_uav_1_df["Fail"].values, -tmp_uav_1_df["Num_Reliable"].values, pos_label=True)
            uav_1_yj, uav_1_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                uav_1_auc = metrics.auc(fpr, tpr)
            except: # This is to handle the case where there is no negative cases
                uav_1_auc = np.nan

            tmp_uav_2_df = uav_2_df.loc[uav_2_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_uav_2_df["Fail"].values, -tmp_uav_2_df["Num_Reliable"].values, pos_label=True)
            uav_2_yj, uav_2_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                uav_2_auc = metrics.auc(fpr, tpr)
            except: # This is to handle the case where there is no negative cases
                uav_2_auc = np.nan

            tmp_uav_3_df = uav_3_df.loc[uav_3_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_uav_3_df["Fail"].values, -tmp_uav_3_df["Num_Reliable"].values, pos_label=True)
            uav_3_yj, uav_3_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                uav_3_auc = metrics.auc(fpr, tpr)
            except: # This is to handle the case where there is no negative cases
                uav_3_auc = np.nan

            tmp_uav_4_df = uav_4_df.loc[uav_4_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_uav_4_df["Fail"].values, -tmp_uav_4_df["Num_Reliable"].values, pos_label=True)
            uav_4_yj, uav_4_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                uav_4_auc = metrics.auc(fpr, tpr)
            except: # This is to handle the case where there is no negative cases
                uav_4_auc = np.nan

            tmp_uav_5_df = uav_5_df.loc[uav_5_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_uav_5_df["Fail"].values, -tmp_uav_5_df["Num_Reliable"].values, pos_label=True)
            uav_5_yj, uav_5_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                uav_5_auc = metrics.auc(fpr, tpr)
            except: # This is to handle the case where there is no negative cases
                uav_5_auc = np.nan

            tmp_uav_6_df = uav_6_df.loc[uav_6_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_uav_6_df["Fail"].values, -tmp_uav_6_df["Num_Reliable"].values, pos_label=True)
            uav_6_yj, uav_6_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                uav_6_auc = metrics.auc(fpr, tpr)
            except: # This is to handle the case where there is no negative cases
                uav_6_auc = np.nan

            tmp_uav_7_df = uav_7_df.loc[uav_7_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_uav_7_df["Fail"].values, -tmp_uav_7_df["Num_Reliable"].values, pos_label=True)
            uav_7_yj, uav_7_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                uav_7_auc = metrics.auc(fpr, tpr)
            except:
                uav_7_auc = np.nan

            tmp_dl_df = dl_df.loc[dl_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_dl_df["Fail"].values, -tmp_dl_df["Num_Reliable"].values, pos_label=True)
            dl_yj, dl_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                dl_auc = metrics.auc(fpr, tpr)
            except:
                dl_auc = np.nan

            gamma_yj_list.append({"USI": usi, "UL": ul_yj, "DL": dl_yj, "UAV_0": uav_0_yj, "UAV_1": uav_1_yj, "UAV_2": uav_2_yj,
                            "UAV_3": uav_3_yj, "UAV_4": uav_4_yj, "UAV_5": uav_5_yj, "UAV_6": uav_6_yj, "UAV_7": uav_7_yj}) 
            # Make sure to negate the thresholds               
            gamma_threshold_list.append({"USI": usi, "UL": -ul_th, "DL": -dl_th, "UAV_0": -uav_0_th, "UAV_1": -uav_1_th, "UAV_2": -uav_2_th,
                            "UAV_3": -uav_3_th, "UAV_4": -uav_4_th, "UAV_5": -uav_5_th, "UAV_6": -uav_6_th, "UAV_7": -uav_7_th})          
            gamma_auc_list.append({"USI": usi, "UL": ul_auc, "DL": dl_auc, "UAV_0": uav_0_auc, "UAV_1": uav_1_auc, "UAV_2": uav_2_auc,
                            "UAV_3": uav_3_auc, "UAV_4": uav_4_auc, "UAV_5": uav_5_auc, "UAV_6": uav_6_auc, "UAV_7": uav_7_auc})     
        
        ''' For Throughput '''
        throughput_yj_list = []
        throughput_threshold_list = []
        throughput_auc_list = []
        # Find Youden Indexes and ROC AUC for each USI 
        for usi in [10, 20, 66.7, 100]:
            
            if (usi == 10) and ((scenario_name == "uav_scenario_1_{}_processed.csv") or (scenario_name == "uav_scenario_2_{}_processed.csv")):
                ul_yj = np.nan
                ul_th = np.nan
                ul_auc = np.nan
            else:
                tmp_ul_df = ul_df.loc[ul_df["USI"] == usi]
                fpr, tpr, thresholds = metrics.roc_curve(tmp_ul_df["Fail"].values, -tmp_ul_df["Throughput"].values, pos_label=True)
                ul_yj, ul_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
                ul_auc = metrics.auc(fpr, tpr)

            tmp_uav_0_df = uav_0_df.loc[uav_0_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_uav_0_df["Fail"].values, -tmp_uav_0_df["Throughput"].values, pos_label=True)
            uav_0_yj, uav_0_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                uav_0_auc = metrics.auc(fpr, tpr)
            except: 
                uav_0_auc = np.nan

            tmp_uav_1_df = uav_1_df.loc[uav_1_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_uav_1_df["Fail"].values, -tmp_uav_1_df["Throughput"].values, pos_label=True)
            uav_1_yj, uav_1_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                uav_1_auc = metrics.auc(fpr, tpr)
            except: 
                uav_1_auc = np.nan

            tmp_uav_2_df = uav_2_df.loc[uav_2_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_uav_2_df["Fail"].values, -tmp_uav_2_df["Throughput"].values, pos_label=True)
            uav_2_yj, uav_2_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                uav_2_auc = metrics.auc(fpr, tpr)
            except:
                uav_2_auc = np.nan

            tmp_uav_3_df = uav_3_df.loc[uav_3_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_uav_3_df["Fail"].values, -tmp_uav_3_df["Throughput"].values, pos_label=True)
            uav_3_yj, uav_3_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                uav_3_auc = metrics.auc(fpr, tpr)
            except:
                uav_3_auc = np.nan

            tmp_uav_4_df = uav_4_df.loc[uav_4_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_uav_4_df["Fail"].values, -tmp_uav_4_df["Throughput"].values, pos_label=True)
            uav_4_yj, uav_4_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                uav_4_auc = metrics.auc(fpr, tpr)
            except:
                uav_4_auc = np.nan

            tmp_uav_5_df = uav_5_df.loc[uav_5_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_uav_5_df["Fail"].values, -tmp_uav_5_df["Throughput"].values, pos_label=True)
            uav_5_yj, uav_5_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                uav_5_auc = metrics.auc(fpr, tpr)
            except:
                uav_5_auc = np.nan

            tmp_uav_6_df = uav_6_df.loc[uav_6_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_uav_6_df["Fail"].values, -tmp_uav_6_df["Throughput"].values, pos_label=True)
            uav_6_yj, uav_6_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                uav_6_auc = metrics.auc(fpr, tpr)
            except:
                uav_6_auc = np.nan

            tmp_uav_7_df = uav_7_df.loc[uav_7_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_uav_7_df["Fail"].values, -tmp_uav_7_df["Throughput"].values, pos_label=True)
            uav_7_yj, uav_7_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                uav_7_auc = metrics.auc(fpr, tpr)
            except:
                uav_7_auc = np.nan

            tmp_dl_df = dl_df.loc[dl_df["USI"] == usi]
            fpr, tpr, thresholds = metrics.roc_curve(tmp_dl_df["Fail"].values, -tmp_dl_df["Throughput"].values, pos_label=True)
            dl_yj, dl_th, _, _ = cutoff_youdens_j(fpr,tpr,thresholds)
            try:
                dl_auc = metrics.auc(fpr, tpr)
            except:
                dl_auc = np.nan
                
            throughput_yj_list.append({"USI": usi, "UL": ul_yj, "DL": dl_yj, "UAV_0": uav_0_yj, "UAV_1": uav_1_yj, "UAV_2": uav_2_yj,
                            "UAV_3": uav_3_yj, "UAV_4": uav_4_yj, "UAV_5": uav_5_yj, "UAV_6": uav_6_yj, "UAV_7": uav_7_yj}) 
            throughput_threshold_list.append({"USI": usi, "UL": -ul_th, "DL": -dl_th, "UAV_0": -uav_0_th, "UAV_1": -uav_1_th, "UAV_2": -uav_2_th,
                            "UAV_3": -uav_3_th, "UAV_4": -uav_4_th, "UAV_5": -uav_5_th, "UAV_6": -uav_6_th, "UAV_7": -uav_7_th}) 
            throughput_auc_list.append({"USI": usi, "UL": ul_auc, "DL": dl_auc, "UAV_0": uav_0_auc, "UAV_1": uav_1_auc, "UAV_2": uav_2_auc,
                            "UAV_3": uav_3_auc, "UAV_4": uav_4_auc, "UAV_5": uav_5_auc, "UAV_6": uav_6_auc, "UAV_7": uav_7_auc})
            
        gamma_yj_df = pd.DataFrame(gamma_yj_list)
        gamma_yj_df["Metric"] = "Gamma_Youden_Index"
        gamma_yj_df["Scenario"] = scenario_name
        gamma_threshold_df = pd.DataFrame(gamma_threshold_list)
        gamma_threshold_df["Metric"] = "Gamma_Threshold"
        gamma_threshold_df["Scenario"] = scenario_name
        gamma_auc_df = pd.DataFrame(gamma_auc_list)
        gamma_auc_df["Metric"] = "Gamma_AUC"
        gamma_auc_df["Scenario"] = scenario_name
        throughput_yj_df = pd.DataFrame(throughput_yj_list)
        throughput_yj_df["Metric"] = "Throughput_Youden_Index"
        throughput_yj_df["Scenario"] = scenario_name
        throughput_threshold_df = pd.DataFrame(throughput_threshold_list)
        throughput_threshold_df["Metric"] = "Throughput_Threshold"
        throughput_threshold_df["Scenario"] = scenario_name
        throughput_auc_df = pd.DataFrame(throughput_auc_list)
        throughput_auc_df["Metric"] = "Throughput_AUC"
        throughput_auc_df["Scenario"] = scenario_name
        results.append(pd.concat([gamma_yj_df, gamma_threshold_df, gamma_auc_df, throughput_yj_df, throughput_threshold_df, throughput_auc_df]))

    results_df = pd.concat(results)
    results_df.to_csv(SAVE_PATH)