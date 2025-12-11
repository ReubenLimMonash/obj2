# Date: 20/05/24
# Desc: To compile the processed reliability dataset from multiple CSV files

import pandas as pd
import os

DATASET_PATHS = ["/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP100000_DJIMavicAir/data_height60_processed",
                 "/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP100000_DJIMavicAir/data_height90_processed",
                 "/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP100000_DJIMavicAir/data_height120_processed",
                 "/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP100000_DJIMavicAir/data_height150_processed",
                 "/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP100000_DJIMavicAir/data_height180_processed",
                 "/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP100000_DJIMavicAir/data_height210_processed",
                 "/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP100000_DJIMavicAir/data_height240_processed",
                 "/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP100000_DJIMavicAir/data_height270_processed",
                 "/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP100000_DJIMavicAir/data_height300_processed"]
SAVE_PATH = "/home/wlau0003/Reuben_ws/FANET_Dataset/Dataset_NP100000_DJIMavicAir/data_processed"

dl_df_list = []
ul_df_list = []
vid_df_list = []

for path in DATASET_PATHS:
    dl_df = pd.read_csv(os.path.join(path, "Downlink_Reliability.csv"))
    ul_df = pd.read_csv(os.path.join(path, "Uplink_Reliability.csv"))
    vid_df = pd.read_csv(os.path.join(path, "Video_Reliability.csv"))
    dl_df_list.append(dl_df)
    ul_df_list.append(ul_df)
    vid_df_list.append(vid_df)

dl_df_compiled = pd.concat(dl_df_list)
ul_df_compiled = pd.concat(ul_df_list)
vid_df_compiled = pd.concat(vid_df_list)

dl_df_compiled.to_csv(os.path.join(SAVE_PATH, "Downlink_Reliability.csv"))
ul_df_compiled.to_csv(os.path.join(SAVE_PATH, "Uplink_Reliability.csv"))
vid_df_compiled.to_csv(os.path.join(SAVE_PATH, "Video_Reliability.csv"))