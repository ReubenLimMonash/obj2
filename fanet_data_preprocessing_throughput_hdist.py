'''
Date: 10/07/2023
Desc: To calculate throughput and h_dist in 
'''

import pandas as pd # for data manipulation 
import numpy as np
# import matplotlib.pyplot as plt # for drawing graphs
import os, sys, glob, math
import time
from tqdm import tqdm

def process_throughput(df, timeDiv):
    '''
    Function to calculate throughput data for a DataFrame
    timeDiv is the time division to use for calculating the throughput
    '''
    maxTime = math.ceil(float(df["RxTime"].max()))
    for i in range(math.ceil(maxTime / timeDiv)):
        df_in_range = df.loc[(df["RxTime"] >= (i*timeDiv)) & (df["RxTime"] < ((i+1)*timeDiv)) & (df["Packet_State"].isin(["Reliable","Delay_Exceeded"]))]
        totalBytes = df_in_range["Bytes"].sum()
        throughput = totalBytes / timeDiv
        df.loc[(df["RxTime"] >= (i*timeDiv)) & (df["RxTime"] < ((i+1)*timeDiv)), "Throughput"] = throughput
    return df

def process_throughput_2(df, timeDiv):
    '''
    Function to calculate throughput data for a DataFrame
    timeDiv is the time division to use for calculating the throughput
    NOTE:
    ASSUMING THAT RxTime IN df IS SORTED AND START NEAR 0
    ONLY APPLY THIS FUNCTION TO INDIVIDUAL SCENARIOS
    '''
    df["Throughput"] = ''
    maxRoundTime = math.floor(float(df["RxTime"].max())) # Use floor to get the max time division in full seconds
    dfRowIndex = 0
    currTimeDiv = 1
    for index in range(len(df)): # Loop for all rows in df
        # print(index)
        if (df.iloc[index]["RxTime"] >= currTimeDiv): # If the current row belongs to the next time div, calculate throughput of time div
            df_in_range = df.iloc[dfRowIndex:index] # Take all rows from dfRowIndex to the row before index
            totalBytes = df_in_range["Bytes"].sum()
            throughput = totalBytes / timeDiv
            df.iloc[dfRowIndex:index, df.columns.get_loc('Throughput')] = throughput
            dfRowIndex = index # Update dfRowIndex to current index
            currTimeDiv += 1
        if currTimeDiv == maxRoundTime+1:
            df_in_range = df.iloc[dfRowIndex:] # Take all rows from dfRowIndex to the row before index
            totalBytes = df_in_range["Bytes"].sum()
            throughput = totalBytes / (df.iloc[-1]["RxTime"]-maxRoundTime)
            df.iloc[dfRowIndex:, df.columns.get_loc('Throughput')] = throughput
            break
        
    return df

if __name__ == "__main__":
    df_file ="/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_NoVideo/Test/Test_Dataset_1_10000_processed/NumMember-7_InterUAVDistance-5_Height-75_Distance-5_Modulation-QAM-64_UAVSendingInterval-1000_downlink.csv"
    df = pd.read_csv(df_file)
    timeDiv = 1
    start = time.time()
    df_processed = process_throughput_2(df, timeDiv)
    end = time.time()
    print(end-start)
    # df_processed.to_csv("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_NoVideo/Test/NumMember-7_InterUAVDistance-5_Height-75_Distance-5_Modulation-BPSK_UAVSendingInterval-10_uplink_throughput.csv")