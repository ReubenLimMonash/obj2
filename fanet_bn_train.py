import pandas as pd # for data manipulation 
import networkx as nx # for drawing graphs
import matplotlib.pyplot as plt # for drawing graphs
import os, sys, glob, math, pickle
# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController

def process_micro_sim(file_list):
    '''
    Function to process the CSV files generated by each micro-simulation
    Input: file_list - List of simulation files belonging to a certain scenario (micro-sim)
    Output: concatenates the raw data to UL and DL dataframes
    '''
    uavs_rx_df_list = [] # List to store all df for UAVs Rx app
    uavs_tx_df_list = [] # List to store all df for UAVs Tx app
    uavs_mon_df_list = [] # List to store all df for UAVs monitor mode captures
    tx_dtype = {'TxTime': float, 'Packet_Name': str, 'Packet_Seq': int, 'Bytes': int, 'Dest_Addr': str, 'Dest_Port': int}
    rx_dtype = {'RxTime': float, 'TxTime': float, 'Packet_Name': str, 'Bytes': int, 'RSSI': str, 'SINR': float, 'Src_Addr': str, 'Src_Port': int, 'Dest_Addr': str, 'Dest_Port': int,	'Hop_Count': int, 'Delay': float, 'Distance': float}
    mon_dtype = {'RxTime': float, 'PkCreationTime': object, 'Packet_Name': str, 'Bytes': int, 'RSSI': str, 'SINR': float, 'Delay': float, 'Distance': float, 'HasError': int}
    for file in file_list:
        try:
            if ('_GCS-' in file) and ('-Tx' in file):
                # DOWNLINK
                # This is the GCS Tx file, recording the sent packets from GCS
                gcs_tx_df = pd.read_csv(file, dtype=tx_dtype)
            elif ('_GW-' in file) and ('-Rx' in file):
                # DOWNLINK
                # This is the gateway Rx file, let's get the information of packets received from GCS
                gw_rx_df = pd.read_csv(file, dtype=rx_dtype)
                uavs_rx_df_list.append(gw_rx_df)
            elif ('_UAV-' in file) and ('-Rx' in file):
                # DOWNLINK
                # This is a UAV Rx file. To concatenate all such files into a single df
                uav_rx_df = pd.read_csv(file, dtype=rx_dtype)
                # uav_cnc_data = uav_rx_df["CNCData" in uav_rx_df["Packet_Name"]] # Get the CNC Data received by this UAV
                # uav_cnc_reliable = uav_cnc_data[uav_cnc_data["Delay"] < delay_th] # Get the CNCData packets received reliably by this UAV (delay < 1ms)
                uavs_rx_df_list.append(uav_rx_df) # Append to list for concatenation later
            elif ('_GCS-' in file) and ('-Rx' in file):
                # UPLINK
                # This is a GCS Rx file, recording packets received from UAVs-
                gcs_rx_df = pd.read_csv(file, dtype=rx_dtype)
            elif ('_GW-' in file) and ('-Tx' in file):
                # UPLINK
                # This is the gateway Tx file, recording packet transmissions to GCS from gateway
                gw_tx_df = pd.read_csv(file, dtype=tx_dtype)
                uavs_tx_df_list.append(gw_tx_df) # Append to list for concatenation later
            elif ('_UAV-' in file) and ('-Tx' in file):
                # DOWNLINK
                # This is a UAV Rx file. To concatenate all such files into a single df
                uav_tx_df = pd.read_csv(file, dtype=tx_dtype)
                uavs_tx_df_list.append(uav_tx_df) # Append to list for concatenation later
            elif ('_GCS-' in file) and ('Wlan' in file):
                # Monitor mode file for GCS
                gcs_mon_df = pd.read_csv(file, dtype=mon_dtype)
                gcs_mon_df["Addr"] = "192.168.0.1"
            elif ('_GW-' in file) and ('Wlan' in file):
                # Monitor mode file for gateway
                gw_mon_df = pd.read_csv(file, dtype=mon_dtype)
                gw_mon_df["Addr"] = "192.168.0.2"
                uavs_mon_df_list.append(gw_mon_df)
            elif ('_UAV-' in file) and ('Wlan' in file):
                # Monitor mode file for GCS
                uav_mon_df = pd.read_csv(file, dtype=mon_dtype)
                uav_index = file.split("_")[-1].split("-")[1]
                uav_mon_df["Addr"] = "192.168.0.{}".format(int(uav_index) + 3)
                uavs_mon_df_list.append(uav_mon_df)
            else:
                # This file type is not handled, pass 
                pass
        except Exception as e:
            print(file)
            print(e)
        
    if uavs_rx_df_list:
        uavs_rx_df = pd.concat(uavs_rx_df_list, ignore_index = True)
    else:
        uavs_rx_df = None

    if uavs_tx_df_list:
        uavs_tx_df = pd.concat(uavs_tx_df_list, ignore_index = True)
    else:
        uavs_tx_df = None

    if uavs_mon_df_list:
        uavs_mon_df = pd.concat(uavs_mon_df_list, ignore_index = True)
    else:
        uavs_mon_df = None

    # The DL data is in uavs_rx_df
    dl_df = uavs_rx_df
    # The UL data is in gcs_rx_df
    ul_df = gcs_rx_df

    return dl_df, ul_df, gcs_tx_df, uavs_tx_df, gcs_mon_df, uavs_mon_df, gw_mon_df

def process_missing_data(tx_df, rx_df, mon_df, mode='downlink'):
    '''
    This function is to fill in missing data in rx_df with data from mon_df
    tx_df contains the list of all transmitted network packets (UL/DL)
    rx_df should only contain the captures of packets received successfully (regardless of delay)
    mon_df contains the monitor mode captures, and contains information of packets not received successfully
    DON'T MIX UL AND DL DATA TOGETHER IN THIS FUNCTION, EVALUATE THEM SEPARATELY.
    '''
    # Firstly, let's mark all the rows in rx_df as having been received correctly
    rx_df["Has_Error"] = 0
    for index, row in tx_df.iterrows():
        packetName = row["Packet_Name"] + "-" + str(row["Packet_Seq"])

        # Let's also use this function to do process_swarm_distance
        # swarm_distances = gw_mon_df.loc[(gw_mon_df["Packet_Name"] == packetName) & (gw_mon_df["Distance"] != inter_uav_dist), "Distance"].values
        # if swarm_distances.size > 0:
        #     swarm_distance = swarm_distances[0]
        # else:
        #     swarm_distance = rx_df.loc[(rx_df["Packet_Name"] == packetName), "Distance"].values # If packet not found in mon_df, just use back original Distance data

        # First, check if the packet is received successfully in rx_df
        if (packetName not in rx_df["Packet_Name"].values):
            dest_addr = row["Dest_Addr"]
            # If not received, find the data in mon_df and add it to rx_df
            # First choice: Try to find the packet from the intended node's monitor df, check if it failed at the last hop
            if mode == 'downlink':
                cap_pks = mon_df.loc[(mon_df["Packet_Name"] == packetName) & (mon_df["Addr"] == dest_addr) & (mon_df["HasError"] == 1) & (mon_df["Distance"] == '4')] # The magic number 4 here is the inter-UAV distance
            elif mode == 'uplink':
                cap_pks = mon_df.loc[(mon_df["Packet_Name"] == packetName) & (mon_df["Addr"] == dest_addr) & (mon_df["HasError"] == 1)] # TODO: Discriminate whether packet is from member or GW
            # If not there, check the gateway, maybe it failed there
            if cap_pks.empty:
                cap_pks = mon_df.loc[(mon_df["Packet_Name"] == packetName) & (mon_df["Addr"] == "192.168.0.2")]
            # If it fails at the intended UAV or the GW, only we record the failed packet
            if not cap_pks.empty:
                # Find the packet with the max SINR in cap_pks and use it to fill the missing data
                err_pk = cap_pks.loc[cap_pks["SINR"].idxmax()]
                err_pk_new_dict = {'RxTime': err_pk['RxTime'],'TxTime': err_pk['PkCreationTime'],'Packet_Name': err_pk['Packet_Name'],'Bytes': err_pk['Bytes'],'RSSI': err_pk['RSSI'],'SINR': err_pk['SINR'],'Src_Addr': "-",'Src_Port': "-",'Dest_Addr': row['Dest_Addr'],'Dest_Port': row['Dest_Port'],'Hop_Count': "-",'Delay': err_pk['Delay'],'Distance': err_pk['Distance'],'Has_Error': 1}
                err_pk_new_df = pd.DataFrame([err_pk_new_dict])
                rx_df = pd.concat([rx_df,err_pk_new_df], ignore_index = True)
        # else:
        #     rx_df.loc[(rx_df["Packet_Name"] == packetName), "Swarm_Distance"] = swarm_distance

    rx_df = rx_df.sort_values("RxTime")
    rx_df = rx_df.reset_index()
    return rx_df

def process_throughput(df, timeDiv):
    '''
    Function to calculate throughput data for a DataFrame
    timeDiv is the time division to use for calculating the throughput
    '''
    maxTime = math.ceil(float(df["RxTime"].max()))
    for i in range(math.ceil(maxTime / timeDiv)):
        df_in_range = df.loc[(df["RxTime"] >= (i*timeDiv)) & (df["RxTime"] < ((i+1)*timeDiv)) & (df["Has_Error"] == 0)]
        totalBytes = df_in_range["Bytes"].sum()
        throughput = totalBytes / timeDiv
        df.loc[(df["RxTime"] >= (i*timeDiv)) & (df["RxTime"] < ((i+1)*timeDiv)), "Throughput"] = throughput
    return df

def process_swarm_distance(df, mon_df):
    '''
    Function to fill in the swarm distance from mon_df
    For each packet in df, finds the packet with same name in mon_df but with distance != inter_uav_dist
    Use mon_df = gw_mon_df for CNCData, and gcs_mon_df for UAVData
    '''
    for index, row in df.iterrows():
        packetName = row["Packet_Name"]
        inter_uav_distance = row["Inter_UAV_Distance"]
        swarm_distances = mon_df.loc[(mon_df["Packet_Name"] == packetName) & (mon_df["Distance"] != inter_uav_distance), "Distance"].values
        if swarm_distances.size > 0:
            swarm_distance = swarm_distances[0]
        else:
            swarm_distance = row["Distance"] # If packet not found in mon_df, just use back original Distance data
        df.loc[index, "Swarm_Distance"] = swarm_distance
        height = row["Height"] - 1
        df.loc[index, "Horizontal_Distance"] = math.sqrt(swarm_distance**2 - height**2)
    return df

def process_sim_data(sim_root_path, delay_threshold):
    # Concatenates all UL & DL results from sim_root_path into a single df
    # Get list of "unique" scenarios
    scenario_list = [csv.split('/')[-1][0:-11] for csv in glob.glob(sim_root_path + "/*_GCS-Tx.csv")]

    # Dataframes to store UL & DL raw data
    # RxTime	TxTime	Packet_Name	Bytes	RSSI	SINR	Src_Addr	Src_Port	Dest_Addr	Dest_Port	Hop_Count	Delay	Throughput	Distance

    dl_df = pd.DataFrame(columns = ['RxTime','TxTime','Packet_Name','Bytes','RSSI','SINR','Src_Addr','Src_Port','Dest_Addr','Dest_Port','Hop_Count','Delay','Distance','Swarm_Distance','Horizontal_Distance','Height','Inter_UAV_Distance','Num_Members','Sending_Interval','Has_Error','Delay_Exceeded','Reliable']) # Downlink dataframe
    ul_df = pd.DataFrame(columns = ['RxTime','TxTime','Packet_Name','Bytes','RSSI','SINR','Src_Addr','Src_Port','Dest_Addr','Dest_Port','Hop_Count','Delay','Distance','Swarm_Distance','Horizontal_Distance','Height','Inter_UAV_Distance','Num_Members','Sending_Interval','Has_Error','Delay_Exceeded','Reliable']) # Uplink dataframe

    # For each scenario, extract the UL and DL raw data
    for scenario in scenario_list:
        scenario_files = glob.glob(sim_root_path + "/{}_*.csv".format(scenario)) # Get list of csv files belonging to this scenario
        scenario_params = scenario.split('_')
        num_member = int(scenario_params[0].split('-')[-1])
        inter_uav_distance = int(scenario_params[1].split('-')[-1])
        height = int(scenario_params[2].split('-')[-1])
        swarm_hor_distance = int(scenario_params[3].split('-')[-1]) # Horizontal Swarm Distance
        swarm_distance = math.sqrt(int(height)**2 + swarm_hor_distance**2)
        packet_size = int(scenario_params[4].split('-')[-1])
        sending_interval = int(scenario_params[5].split('-')[-1])
        try:
            dl_data, ul_data, dl_tx_df, ul_tx_df, gcs_mon_df, uavs_mon_df, gw_mon_df = process_micro_sim(scenario_files)
        except Exception as e:
            print(scenario)
            print(e)
        dl_data = process_missing_data(dl_tx_df, dl_data, uavs_mon_df, mode='downlink')
        # ul_data = process_missing_data(ul_tx_df, ul_data, gcs_mon_df)
        if dl_data is not None:
            try:
                dl_data["Height"] = height
                # dl_data["Swarm_Distance"] = swarm_distance
                dl_data["Horizontal_Distance"] = swarm_hor_distance
                dl_data["Inter_UAV_Distance"] = inter_uav_distance
                dl_data["Num_Members"] = num_member
                # dl_data["Packet_Size"] = packet_size
                dl_data["Sending_Interval"] = sending_interval
                # Fill in reliability data
                dl_data["Delay_Exceeded"] = 0
                dl_data.loc[dl_data["Delay"] > delay_threshold, "Delay_Exceeded"] = 1
                dl_data["Reliable"] = 0
                dl_data.loc[(dl_data["Delay_Exceeded"] == 0) & (dl_data["Has_Error"] == 0), "Reliable"] = 1
                dl_data = process_throughput(dl_data, 1)
                dl_data = process_swarm_distance(dl_data, gw_mon_df)
                dl_df = pd.concat([dl_df, dl_data], ignore_index=True)
            except Exception as e:
                print(scenario)
                print(e)
        if ul_data is not None:
            ul_data["Height"] = height
            ul_data["Swarm_Distance"] = swarm_distance
            ul_data["Horizontal_Distance"] = swarm_hor_distance
            ul_data["Inter_UAV_Distance"] = inter_uav_distance
            ul_data["Num_Members"] = num_member
            # ul_data["Packet_Size"] = packet_size
            ul_data["Sending_Interval"] = sending_interval
            # Fill in reliability data
            # ul_data["Delay_Exceeded"] = 0
            # ul_data.loc[ul_data["Delay"] > delay_threshold, "Delay_Exceeded"] = 1
            # ul_data["Reliable"] = 0
            # ul_data.loc[(ul_data["Delay_Exceeded"] == 0) & (ul_data["Has_Error"] == 0), "Reliable"] = 1
            ul_df = pd.concat([ul_df, ul_data], ignore_index=True)
    
    return dl_df, ul_df

# This function helps to calculate probability distribution, which goes into BBN (note, can handle up to 2 parents)
def cpt_probs(df, child, parents):
    try:
        # dependencies_arr = [pd.Categorical(df[parent],categories=df[parent].cat.categories.tolist()) for parent in parents]
        dependencies_arr = [df[parent] for parent in parents]
        cpt = pd.crosstab(dependencies_arr, df[child], rownames=parents, colnames=[child], margins=False, normalize='index', dropna=False).sort_index().to_numpy().reshape(-1).tolist()
        return cpt
    except Exception as err:
        print(err)
        return None 

# Define a function for printing marginal probabilities
def print_probs(join_tree):
    for node in join_tree.get_bbn_nodes():
        potential = join_tree.get_bbn_potential(node)
        print("Node:", node)
        print("Values:")
        print(potential)
        print('----------------')

# To add evidence of events that happened so probability distribution can be recalculated
def evidence(join_tree, nod, cat, val):
    ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name(nod)) \
    .with_evidence(cat, val) \
    .build()
    join_tree.set_observation(ev)

if __name__ == "__main__":
    sim_root_path = "/home/research-student/omnetpp_sim_results/FANET_Corr2"
    delay_threshold = 1
    dl_df, ul_df = process_sim_data(sim_root_path, delay_threshold=delay_threshold)
    dl_df.to_csv(os.path.join(sim_root_path,"FANET_downlink_raw.csv"), index=False)
    ul_df.to_csv(os.path.join(sim_root_path,"FANET_uplink_raw.csv"), index=False)

    # Let's train our BN

    # First, discretise the values to classes
    dl_df["H_Dist_Class"] = pd.cut(dl_df.Horizontal_Distance, bins=5, labels=['vs','s','m','l','vl'])
    dl_df["Height_Class"] = pd.cut(dl_df.Height, bins=3, labels=['s','m','l'])
    dl_df["Num_Members_Class"] = pd.cut(dl_df.Num_Members, bins=3, labels=['s','m','l'])
    dl_df["Sending_Interval_Class"] = pd.cut(dl_df.Sending_Interval, bins=3, labels=['s','m','l'])
    dl_df["Packet_Size_Class"] = pd.cut(dl_df.Bytes, bins=3, labels=['s','m','l'])

    dl_df["SINR_Class"] = pd.qcut(dl_df.SINR, q=3, labels=['s','m','l'])
    dl_df["Delay_Class"] = pd.qcut(dl_df.Delay, q=3, labels=['s','m','l'])
    dl_df["Throughput_Class"] = pd.qcut(dl_df.Throughgput, q=3, labels=['s','m','l'])

    # Calculate the conditional probabilities table for each second layer class
    parents_1 = ["H_Dist_Class", "Height_Class", "Num_Members_Class", "Sending_Interval_Class", "Packet_Size_Class"]
    sinr_cpt = cpt_probs(dl_df, child="SINR_Class", parents=parents_1)
    delay_cpt = cpt_probs(dl_df, child="Delay_Class", parents=parents_1)
    throughput_cpt = cpt_probs(dl_df, child="Throughput_Class", parents=parents_1)
    parents_2 = ["SINR_Class", "Delay_Class", "Throughput_Class"]
    reliability_cpt = cpt_probs(dl_df, child="Reliable", parents=parents_2)

    # Train the BBN
    H_Dist = BbnNode(Variable(0, 'H_Dist', ['vs']), [1])
    Height = BbnNode(Variable(1, 'Height', ['s']), [1] )
    Num_Members = BbnNode(Variable(2, 'Num_Members', ['s']), [1])
    Sending_Interval = BbnNode(Variable(3, 'Sending_Interval', ['s']), [1])
    Packet_Size = BbnNode(Variable(4, 'Packet_Size', ['s']), [1])
    SINR = BbnNode(Variable(5, 'SINR', ['s','m','l']), sinr_cpt)
    Delay = BbnNode(Variable(6, 'Delay', ['s','m','l']), delay_cpt)
    Throughput = BbnNode(Variable(7, 'Throughput', ['s','l']), throughput_cpt)
    Reliability = BbnNode(Variable(8, "Reliability", ['0', '1']), reliability_cpt)

    # Create Network
    bbn = Bbn() \
        .add_node(H_Dist) \
        .add_node(Height) \
        .add_node(Num_Members) \
        .add_node(Sending_Interval) \
        .add_node(Packet_Size) \
        .add_node(SINR) \
        .add_node(Delay) \
        .add_node(Throughput) \
        .add_node(Reliability) \
        .add_edge(Edge(H_Dist, SINR, EdgeType.DIRECTED)) \
        .add_edge(Edge(Height, SINR, EdgeType.DIRECTED)) \
        .add_edge(Edge(Num_Members, SINR, EdgeType.DIRECTED)) \
        .add_edge(Edge(Sending_Interval, SINR, EdgeType.DIRECTED)) \
        .add_edge(Edge(Packet_Size, SINR, EdgeType.DIRECTED)) \
        .add_edge(Edge(H_Dist, Delay, EdgeType.DIRECTED)) \
        .add_edge(Edge(Height, Delay, EdgeType.DIRECTED)) \
        .add_edge(Edge(Num_Members, Delay, EdgeType.DIRECTED)) \
        .add_edge(Edge(Sending_Interval, Delay, EdgeType.DIRECTED)) \
        .add_edge(Edge(Packet_Size, Delay, EdgeType.DIRECTED)) \
        .add_edge(Edge(H_Dist, Throughput, EdgeType.DIRECTED)) \
        .add_edge(Edge(Height, Throughput, EdgeType.DIRECTED)) \
        .add_edge(Edge(Num_Members, Throughput, EdgeType.DIRECTED)) \
        .add_edge(Edge(Sending_Interval, Throughput, EdgeType.DIRECTED)) \
        .add_edge(Edge(Packet_Size, Throughput, EdgeType.DIRECTED)) \
        .add_edge(Edge(SINR, Reliability, EdgeType.DIRECTED)) \
        .add_edge(Edge(Delay, Reliability, EdgeType.DIRECTED)) \
        .add_edge(Edge(Throughput, Reliability, EdgeType.DIRECTED)) \

    # Convert the BBN to a join tree
    join_tree = InferenceController.apply(bbn)

    # Save the trained BBN using pickle
    bbn_name = "FANET_BBN.pkl"
    with open(bbn_name, 'wb') as f:  
        pickle.dump([join_tree], f)


