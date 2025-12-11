import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Date: 13/10/2023
# Desc: To get the max value in measured throughput dataset and plot its distribution

def get_measured_throughput(sim_root_path, link="Downlink"):
    '''
    Function to load the processed measured throughput data from CSV files stored in different subdirs in sim_root_path
    '''
    assert link in ["Downlink", "Uplink", "Video"], 'link must be one of "Downlink", "Uplink", "Video"'
    df_list = []
    scenario_list = [f.path for f in os.scandir(sim_root_path) if f.is_dir()] # Get list of "unique" scenarios
    for scenario in tqdm(scenario_list):
        # Get the measured throughput samples under this scenario
        measured_df = pd.read_csv(os.path.join(scenario, link + "_Throughput.csv"))
        df_list.append(measured_df)
    return pd.concat(df_list)

if __name__ == "__main__":
    DATASET_PATH = "/home/wlau0003/Reuben_ws/FANET_Dataset/DJISpark_Throughput/data_processed"
    links = ["Downlink", "Uplink", "Video"]
    for link in links:
        print("Loading {} Dataset".format(link))
        throughput_df = get_measured_throughput(DATASET_PATH, link)
        print("Max Measured Throughput: {}".format(throughput_df["Throughput"].max()))
        print("Min Measured Throughput: {}".format(throughput_df["Throughput"].min()))
        fig = plt.figure(figsize=(9.6,7.2))
        plt.hist(throughput_df["Throughput"], 300, density=True)
        plt.xlabel("Measured Throughput (bytes/sec)")
        plt.ylabel("PDF")
        plt.savefig("{}_Throughput_Histogram.jpg".format(link))