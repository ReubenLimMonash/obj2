'''
Date: 08/05/2022
Desc: Network metric correlation study for FANET.
'''

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import scipy.stats as stats

def rssi_to_np(rssi):
    # Function to convert rssi data from string (e.g. "435 pW") to exp (435e-12)
    rssi_num = np.zeros(rssi.shape)
    index = 0
    for r in rssi:
        num = r[0:-3]
        expn = r[-2:]
        # print(num)
        # print(expn)
        if expn == "W":
            rssi_num[index] = float(num)
        elif expn == "mW":
            rssi_num[index] = float(num) * 1e-3
        elif expn == "uW":
            rssi_num[index] = float(num) * 1e-6
        elif expn == "nW":
            rssi_num[index] = float(num) * 1e-9
        elif expn == "pW":
            rssi_num[index] = float(num) * 1e-12
        else:
            raise ValueError("Unhandled unit prefix")
        index += 1
    return rssi_num
        

file_path = "C:/Users/Joanne/Desktop/omnetpp-5.6.2/samples/Fanet/simulations/FANET/Height-20-GW-App[0]-Rx.csv"
# Read in CSV data to Dataframe
df = pd.read_csv(file_path)
# Get the data for feature of interest
rssi = rssi_to_np(df["RSSI"])
sinr = df["SINR"].to_numpy()
distance = df["Distance"].to_numpy()
throughput = df["Throughput"].to_numpy()
delay = df["Delay"].to_numpy()
delay_cat = pd.qcut(df["Delay"],[0,.5,.75,1], labels=['0','1','2']) # categorize delay based on quantile
# print(rssi)
# Calculate correlations
# Mutual information
sinr_MI = mutual_info_classif(sinr.reshape((-1,1)), delay_cat)
rssi_MI = mutual_info_classif(rssi.reshape((-1,1)), delay_cat)
distance_MI = mutual_info_classif(distance.reshape(-1,1), delay_cat)
throughput_MI = mutual_info_classif(throughput.reshape(-1,1),delay_cat)
print(sinr_MI[0], rssi_MI[0], distance_MI[0], throughput_MI[0])

# Kendall's Tau
sinr_tau, sinr_p_value = stats.kendalltau(sinr, delay)
rssi_tau, rssi_p_value = stats.kendalltau(rssi, delay)
distance_tau, distance_p_value = stats.kendalltau(distance, delay)
throughput_tau, throughput_p_value = stats.kendalltau(throughput, delay)
print(sinr_tau, rssi_tau, distance_tau, throughput_tau)

# Spearman's Rho
sinr_rho, sinr_pval = stats.spearmanr(delay, sinr)
rssi_rho, rssi_pval = stats.spearmanr(delay, rssi)
distance_rho, distance_pval = stats.spearmanr(delay, distance)
throughput_rho, throughput_pval = stats.spearmanr(delay, throughput)
print(sinr_rho, rssi_rho, distance_rho, throughput_rho)