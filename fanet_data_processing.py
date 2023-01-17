'''
Date: 26/05/2022
Desc: FANET Simulation Data Pre-processing
'''

from microsim_postprocess import process_micro_sim
import os, sys
import pandas as pd
import math 

# Get the root path where simulation files are stored
sim_root_path = "C:/Users/Joanne/Desktop/omnetpp-5.6.2/samples/Fanet/simulations/FANET"
delay_th = 1 # Delay threshold of 1ms
distance_offset = 400 # Swarm distance offset (GCS Y coordinate)
# Create pd df to store micro-sim processed data
dl_df = pd.DataFrame(columns = ['Reliability', 'Avg_RSSI', 'Avg_SINR', 'Avg_Delay', 'Avg_Throughput', 'Swarm_Distance', 'Horizontal_Distance', 'Height', 'Inter_UAV_Distance', 'Num_Members']) # Downlink dataframe
ul_df = pd.DataFrame(columns = ['Reliability', 'Avg_RSSI', 'Avg_SINR', 'Avg_Delay', 'Avg_Throughput', 'Swarm_Distance', 'Horizontal_Distance', 'Height', 'Inter_UAV_Distance', 'Num_Members']) # Uplink dataframe
# The micro-sim data files are in the folder structures FANET -> NumMember-# -> Height-# -> Distance-#
# os.walk through the directory structure
for num_members_dir in next(os.walk(sim_root_path))[1]:
    for inter_uav_distances_dir in next(os.walk(os.path.join(sim_root_path, num_members_dir)))[1]:
        for heights_dir in next(os.walk(os.path.join(sim_root_path, num_members_dir, inter_uav_distances_dir)))[1]:
            for distances_dir in next(os.walk(os.path.join(sim_root_path, num_members_dir, inter_uav_distances_dir, heights_dir)))[1]:
                num_member = num_members_dir.split('-')[-1]
                height = heights_dir.split('-')[-1]
                swarm_hor_distance = int(distances_dir.split('-')[-1]) - distance_offset # Horizontal Swarm Distance
                swarm_distance = math.sqrt(int(height)**2 + swarm_hor_distance**2)
                dl_data, ul_data = process_micro_sim(os.path.join(sim_root_path, num_members_dir, inter_uav_distances_dir, heights_dir, distances_dir), delay_th)
                if dl_data is not None:
                    dl_data["Height"] = height
                    dl_data["Swarm_Distance"] = swarm_distance
                    dl_data["Horizontal_Distance"] = swarm_hor_distance
                    dl_data["Inter_UAV_Distance"] = 2
                    dl_data["Num_Members"] = num_member
                    dl_df = dl_df.append(dl_data, ignore_index=True)
                if ul_data is not None:
                    ul_data["Height"] = height
                    ul_data["Swarm_Distance"] = swarm_distance
                    ul_data["Horizontal_Distance"] = swarm_hor_distance
                    ul_data["Inter_UAV_Distance"] = 2
                    ul_data["Num_Members"] = num_member
                    ul_df = ul_df.append(ul_data, ignore_index=True)

dl_df.to_csv(os.path.join(sim_root_path,"FANET_downlink.csv"), index=False)
ul_df.to_csv(os.path.join(sim_root_path,"FANET_uplink.csv"), index=False)