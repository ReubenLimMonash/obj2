'''
Date: 20/05/2025
Desc: New NN Training Script with PyTorch GPU
Modified from: fanet_nn_train_05062024.py
NOTE: Output of model reduced to two packet state: Reliable or Fail
'''

import pandas as pd
import numpy as np 
import random
import math
import os
import pickle
import gc 
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def distribute_evenly(nr_variables, value):
    # Function to distribute value evenly (without decimal) into nr_variables no. of groups 
    base = value//nr_variables
    rem = int(value%nr_variables)

    return [(base+1)]*rem + [base]*(nr_variables-rem)

def dataset_details_train_test_split(dataset_details_df, num_split, max_pkts=100000):
    '''
    Splits the number of samples in dataset_details_df into num_split number of groups
    Produces num_split number of splitted dataset_details dataframes
    Before spolitting, we apply max number of packets - so if total pkts more than max, we divide it down proportionally
    '''
    dataset_details_splits = [[] for _ in range(num_split)] # To hold the splitted dataset_details dataframes
    for row in dataset_details_df.itertuples():
        mean_sinr = row.Mean_SINR
        std_dev_sinr = row.Std_Dev_SINR
        uav_send_int = row.UAV_Sending_Interval
        bitrate = row.Bitrate
        num_reliable = row.Num_Reliable
        num_delay_excd = row.Num_Delay_Excd
        num_fail_other = row.Num_Fail_Other
        
        if (num_reliable + num_delay_excd + num_fail_other) > max_pkts:
            # If the total number of packets is greater than max_pkts, we need to scale down
            num_reliable = round(num_reliable * (max_pkts / (num_reliable + num_delay_excd + num_fail_other)))
            num_delay_excd = round(num_delay_excd * (max_pkts / (num_reliable + num_delay_excd + num_fail_other)))
            num_fail_other = round(num_fail_other * (max_pkts / (num_reliable + num_delay_excd + num_fail_other)))

        num_reliable_splits = distribute_evenly(num_split, num_reliable)
        num_delay_excd_splits = distribute_evenly(num_split, num_delay_excd)
        num_fail_other_splits = distribute_evenly(num_split, num_fail_other)

        for i in range(num_split):
            dataset_details_splits[i].append({"Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr, "UAV_Sending_Interval": uav_send_int, "Bitrate": bitrate, 
                                  "Num_Reliable": num_reliable_splits[i], "Num_Delay_Excd": num_delay_excd_splits[i], "Num_Fail_Other": num_fail_other_splits[i]})
    dataset_details_splits_dfs = []
    for j in range(num_split):
        dataset_details_splits_dfs.append(pd.DataFrame(dataset_details_splits[j]))
    return dataset_details_splits_dfs

def generate_reliability_train_test_dataset(dataset_details_df, test_split=0.2):
    df_train_list = []
    df_test_list = []
    for row in dataset_details_df.itertuples():
        mean_sinr = row.Mean_SINR
        std_dev_sinr = row.Std_Dev_SINR
        uav_send_int = row.UAV_Sending_Interval
        mcs = row.MCS
        num_reliable = row.Num_Reliable
        num_fail = row.Num_Delay_Excd + row.Num_Fail_Other

        if num_reliable > 1:
            reliable_packets = pd.DataFrame({"Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr, "UAV_Sending_Interval": uav_send_int, "MCS": mcs, "Packet_State": 0}, index=[0])
            num_reliable_train = math.floor(num_reliable * (1-test_split))
            num_reliable_test = math.ceil(num_reliable * test_split)
            reliable_packets_train = reliable_packets.loc[reliable_packets.index.repeat(num_reliable_train)]
            reliable_packets_test = reliable_packets.loc[reliable_packets.index.repeat(num_reliable_test)]
        elif num_reliable == 1:
            reliable_packets_train = pd.DataFrame({"Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr, "UAV_Sending_Interval": uav_send_int, "MCS": mcs, "Packet_State": 0}, index=[0])
            reliable_packets_test = pd.DataFrame({})
        else:
            reliable_packets_train = pd.DataFrame({})
            reliable_packets_test = pd.DataFrame({})

        if num_fail > 1:
            fail_packets = pd.DataFrame({"Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr, "UAV_Sending_Interval": uav_send_int, "MCS": mcs, "Packet_State": 1}, index=[0])
            num_fail_train = math.floor(num_fail * (1-test_split))
            num_fail_test = math.ceil(num_fail * test_split)
            fail_packets_train = fail_packets.loc[fail_packets.index.repeat(num_fail_train)]
            fail_packets_test = fail_packets.loc[fail_packets.index.repeat(num_fail_test)]
        elif num_fail == 1:
            fail_packets_train = pd.DataFrame({"Mean_SINR": mean_sinr, "Std_Dev_SINR": std_dev_sinr, "UAV_Sending_Interval": uav_send_int, "MCS": mcs, "Packet_State": 1}, index=[0])
            fail_packets_test = pd.DataFrame({})
        else:
            fail_packets_train = pd.DataFrame({})
            fail_packets_test = pd.DataFrame({})

        df_train_concat = pd.concat([reliable_packets_train, fail_packets_train])
        df_train_concat["Mean_SINR"] = df_train_concat["Mean_SINR"].astype("float32")
        df_train_concat["Std_Dev_SINR"] = df_train_concat["Std_Dev_SINR"].astype("float32")
        df_train_concat["UAV_Sending_Interval"] = df_train_concat["UAV_Sending_Interval"].astype("category")
        df_train_concat["MCS"] = df_train_concat["MCS"].astype("category")
        df_train_concat["Packet_State"] = df_train_concat["Packet_State"].astype("int8")
        df_train_list.append(df_train_concat)
        
        df_test_concat = pd.concat([reliable_packets_test, fail_packets_test])
        df_test_concat["Mean_SINR"] = df_test_concat["Mean_SINR"].astype("float32")
        df_test_concat["Std_Dev_SINR"] = df_test_concat["Std_Dev_SINR"].astype("float32")
        df_test_concat["UAV_Sending_Interval"] = df_test_concat["UAV_Sending_Interval"].astype("category")
        df_test_concat["MCS"] = df_test_concat["MCS"].astype("category")
        df_test_concat["Packet_State"] = df_test_concat["Packet_State"].astype("int8")
        df_test_list.append(df_test_concat)

    df_train = pd.concat(df_train_list)
    df_train["Mean_SINR"] = df_train["Mean_SINR"].astype("float32")
    df_train["Std_Dev_SINR"] = df_train["Std_Dev_SINR"].astype("float32")
    df_train["UAV_Sending_Interval"] = df_train["UAV_Sending_Interval"].astype("float16")
    df_train["MCS"] = df_train["MCS"].astype("float16")
    df_train["Packet_State"] = df_train["Packet_State"].astype("int8")

    df_test = pd.concat(df_test_list)
    df_test["Mean_SINR"] = df_test["Mean_SINR"].astype("float32")
    df_test["Std_Dev_SINR"] = df_test["Std_Dev_SINR"].astype("float32")
    df_test["UAV_Sending_Interval"] = df_test["UAV_Sending_Interval"].astype("float16")
    df_test["MCS"] = df_test["MCS"].astype("float16")
    df_test["Packet_State"] = df_test["Packet_State"].astype("int8")

    return df_train, df_test

def normalize_data(df_in, columns=[]):
    '''
    columns: The pandas data columns to normalize, given as a list of column names
    '''
    df = df_in.copy()
    # Define the ranges of parametrers
    max_mean_sinr = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
    max_std_dev_sinr = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
    min_mean_sinr = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
    min_std_dev_sinr = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)
    max_height = 300
    min_height = 60
    max_h_dist = 1200
    min_h_dist = 0
    max_mcs = 7
    min_mcs = 0

    # Normalize data (Min Max Normalization between [-1,1])
    if "Height" in columns:
        df["Height"] = df["Height"].apply(lambda x: 2*(x-min_height)/(max_height-min_height) - 1)
    if "U2G_H_Dist" in columns:
        df["U2G_H_Dist"] = df["U2G_H_Dist"].apply(lambda x: 2*(x-min_h_dist)/(max_h_dist-min_h_dist) - 1)
    if "Mean_SINR" in columns:
        df["Mean_SINR"] = df["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
    if "Std_Dev_SINR" in columns:
        df["Std_Dev_SINR"] = df["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1) # Convert to dB space
    if "UAV_Sending_Interval" in columns:
        df["UAV_Sending_Interval"] = df["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 66.7: 0.5, 100:1, 1000:2})
    if "Modulation" in columns:
        df['Modulation'] = df['Modulation'].replace({"BPSK":1, "QPSK":0.3333, 16:-0.3333, "QAM-16":-0.3333, "QAM16":-0.3333, 64:-1, "QAM-64":-1, "QAM64":-1})
    if "MCS" in columns:
        df["MCS"] = df["MCS"].apply(lambda x: 2*(x-min_mcs)/(max_mcs-min_mcs) - 1)

    return df

def get_mcs_index(df_in):
    '''
    Gets the MCS index based on modulation and bitrate column of the df_in
    '''
    df = df_in.copy()
    df["MCS"] = ''
    df.loc[(df["Bitrate"] == 6.5), "MCS"] = 0 # MCS Index 0
    df.loc[(df["Bitrate"] == 13), "MCS"] = 1 # MCS Index 0
    df.loc[(df["Bitrate"] == 19.5), "MCS"] = 2 # MCS Index 0
    df.loc[(df["Bitrate"] == 26), "MCS"] = 3 # MCS Index 0
    df.loc[(df["Bitrate"] == 39), "MCS"] = 4 # MCS Index 0
    df.loc[(df["Bitrate"] == 52), "MCS"] = 5 # MCS Index 0
    df.loc[(df["Bitrate"] == 58.5), "MCS"] = 6 # MCS Index 0
    df.loc[(df["Bitrate"] == 65), "MCS"] = 7 # MCS Index 0

    return df

class PacketStateDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NNModelV4(nn.Module):
    def __init__(self):
        super(NNModelV4, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(4, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, 4)
        )
    def forward(self, x):
        return self.seq(x)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X, y in tqdm(loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in tqdm(loader):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            running_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return running_loss / total, correct / total

""" For Reproducibility """
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True) # To prevent compromise in performance

def seed_worker(worker_id):
    
    """
    Sets the random seed for a given worker, ensuring reproducibility in multiprocessing.
    
    Parameters:
    worker_id (int): The ID of the worker to set the seed for.
    """
    global SEED
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

SEED = 42
set_seed(SEED) # For initialization

if __name__ == "__main__":

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")
    NUM_WORKER = 16
    ROUNDS = 2
    BATCHSIZE = 64
    NUM_SPLIT = 10
    LR = 0.001
    LINK = "Downlink_UAV-4"
    # CHECKPOINT_FILEPATH = '/media/research-student/DataDrive/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts'
    # DATASET_PATH = "/media/research-student/DataDrive/FANET_Dataset/Dataset_NP100000_DJISpark/train_dataset_processed/{}_Reliability.csv"
    CHECKPOINT_FILEPATH = '/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP100000_DJISpark_May25/nn_ckpts'
    DATASET_PATH = "/home/clow0003/Reuben_ws/FANET_Dataset/Dataset_NP100000_DJISpark_May25/train_dataset_processed/{}_Reliability.csv"
    PRETRAINED_MODEL_PATH = os.path.join(CHECKPOINT_FILEPATH, "model_Downlink_UAV-0.round-1_split-2_valloss-0.0210.pt")  # Change to actual path
    LOAD_PRETRAINED = True  # Toggle this to enable/disable fine-tuning/resume training
    RESUME_ROUND = 0  # Round to resume training from
    RESUME_SPLIT = 3  # Split to resume training from

    if not os.path.isdir(CHECKPOINT_FILEPATH):
        os.mkdir(CHECKPOINT_FILEPATH)

    df_dtypes = {"Horizontal_Distance": np.float32, "Height": np.int16, "UAV_Sending_Interval": np.float16, "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
                "Num_Reliable": np.int32, "Num_Delay_Excd": np.int32, "Num_Fail_Other": np.int32, "Modulation": str, "Bitrate": np.float32}
    dataset_details_df = pd.read_csv(DATASET_PATH.format(LINK), 
                                usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Bitrate", 
                                           "Num_Reliable", "Num_Delay_Excd", "Num_Fail_Other"],
                                dtype=df_dtypes)
    dataset_details_df_splits = dataset_details_train_test_split(dataset_details_df, NUM_SPLIT, max_pkts=100000)

    model = NNModelV4().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    if LOAD_PRETRAINED and os.path.isfile(PRETRAINED_MODEL_PATH):
        print(f"Loading pretrained model from {PRETRAINED_MODEL_PATH}")
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state
        r = RESUME_ROUND
        n = RESUME_SPLIT
    else:
        r = 0
        n = 0

    for j in range(r, ROUNDS):
        for i in range(n, NUM_SPLIT):
            print(f"Round: {j}, Split: {i}")
            # To make it reproducible
            SEED = int(str(j) + str(i))
            set_seed(SEED)
            g = torch.Generator()
            g.manual_seed(SEED)

            print("Preparing data...")
            dataset_split = dataset_details_df_splits[i]
            # dataset_split = dataset_split.loc[dataset_split["Horizontal_Distance"] <= 700]
            dataset_split = get_mcs_index(dataset_split)
            dataset_split = normalize_data(dataset_split, columns=["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"]) 
            df_train, df_test = generate_reliability_train_test_dataset(dataset_split, test_split=0.2)                      
            X_train = df_train[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"]].to_numpy(dtype=np.float16)
            X_test = df_test[["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"]].to_numpy(dtype=np.float16)
            packet_state_train = df_train['Packet_State'].values.astype(np.int8)
            packet_state_test = df_test['Packet_State'].values.astype(np.int8)
            # Clean up to save memory (so that oom don't make me cry)
            del df_train, df_test
            gc.collect()

            # Create DataLoader
            train_dataset = PacketStateDataset(X_train, packet_state_train)
            test_dataset = PacketStateDataset(X_test, packet_state_test)
            train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, worker_init_fn=seed_worker, generator=g)
            test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False, worker_init_fn=seed_worker, generator=g)
            
            print("Training...")
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = eval_epoch(model, test_loader, criterion, device)

            # Save checkpoint
            checkpoint_path = os.path.join(CHECKPOINT_FILEPATH, f"model_{LINK}.round-{j}_split-{i}_valloss-{val_loss:.4f}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            # Save history (optional, here as a dict)
            history = {'Round': j, 'Epoch': i, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc}
            with open(os.path.join(CHECKPOINT_FILEPATH, f'{LINK}_trainHistoryDict.round-{j}_split-{i}.pkl'), 'wb') as file_pi:
                pickle.dump(history, file_pi)

    # Save final model
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_FILEPATH, f"{LINK}_final_model.pt"))

