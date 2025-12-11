'''
Date: 26/05/2022
Desc: FANET Simulation Data Pre-processing
'''

from microsim_postprocess import process_micro_sim
import os, sys
import pandas as pd
import math 

# Get the root path where simulation files are stored
sim_root_path = "/home/reuben/omnetpp_sim_results/FANET_Conv/Shorter"
delay_th = 1 # Delay threshold of 1ms
distance_offset = 0 # Swarm distance offset (GCS Y coordinate)
sending_interval = 0.1 # In seconds
# Create pd df to store micro-sim processed data
dl_df = pd.DataFrame(columns = ['Reliability', 'Avg_RSSI', 'Avg_SINR', 'Avg_Delay', 'Avg_Throughput', 'Swarm_Distance', 'Horizontal_Distance', 'Height', 'Inter_UAV_Distance', 'Num_Members', 'Sim_Time', 'Num_Packets']) # Downlink dataframe
ul_df = pd.DataFrame(columns = ['Reliability', 'Avg_RSSI', 'Avg_SINR', 'Avg_Delay', 'Avg_Throughput', 'Swarm_Distance', 'Horizontal_Distance', 'Height', 'Inter_UAV_Distance', 'Num_Members', 'Sim_Time', 'Num_Packets']) # Uplink dataframe
# The micro-sim data files are in the folder structures FANET -> NumMember-# -> Height-# -> Distance-#
# os.walk through the directory structure
for sim_time_dir in next(os.walk(sim_root_path))[1]:
    # print(sim_time_dir)
    num_member = 5
    sim_time = sim_time_dir.split('-')[-1]
    num_packets_sent = float(sim_time) * (1 / sending_interval)
    height = 200
    swarm_hor_distance = 250
    swarm_distance = math.sqrt(height**2 + swarm_hor_distance**2)
    dl_data, ul_data = process_micro_sim(os.path.join(sim_root_path, sim_time_dir), delay_th)
    if dl_data is not None:
        dl_data["Height"] = height
        dl_data["Swarm_Distance"] = swarm_distance
        dl_data["Horizontal_Distance"] = swarm_hor_distance
        dl_data["Inter_UAV_Distance"] = 4
        dl_data["Num_Members"] = num_member
        dl_data["Sim_Time"] = sim_time
        dl_data["Num_Packets"] = num_packets_sent
        dl_df = dl_df.append(dl_data, ignore_index=True)
    if ul_data is not None:
        ul_data["Height"] = height
        ul_data["Swarm_Distance"] = swarm_distance
        ul_data["Horizontal_Distance"] = swarm_hor_distance
        ul_data["Inter_UAV_Distance"] = 4
        ul_data["Num_Members"] = num_member
        ul_df = ul_df.append(ul_data, ignore_index=True)

dl_df.to_csv(os.path.join(sim_root_path,"FANET_downlink.csv"), index=False)
ul_df.to_csv(os.path.join(sim_root_path,"FANET_uplink.csv"), index=False)