'''
Date: 08/05/2022
Desc: Plot FANET network metrics
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

file_path = "C:/Users/Joanne/Desktop/omnetpp-5.6.2/samples/Fanet/simulations/FANET/FANET_downlink.csv"
# Read in CSV data to Dataframe
df = pd.read_csv(file_path)
# Get the data for feature of interest
# rssi = rssi_to_np(df["Avg_RSSI"])
# print(df.head())
# rssi = df["GW_Avg_RSSI"].to_numpy() # For DOWNLINK
# sinr = df["GW_Avg_SINR"].to_numpy() # For DOWNLINK
rssi = df["Avg_RSSI"].to_numpy() # For UPLINK
sinr = df["Avg_SINR"].to_numpy() # For UPLINK
distance = df["Swarm_Distance"].to_numpy()
hor_distance = df["Horizontal_Distance"].to_numpy() # Horizontal Distance
height = df["Height"].to_numpy()
inter_uav_distance = df["Inter_UAV_Distance"].to_numpy()
throughput = df["Avg_Throughput"].to_numpy()
delay = df["Avg_Delay"].to_numpy()
reliability = df["Reliability"].to_numpy()

# Plot the data
# SINR
fig1, ax1 = plt.subplots()
ax1.scatter(sinr, reliability)
ax1.set_ylabel("Reliability (%)")
ax1.set_xlabel("Average GCS SINR")
ax1.set_yscale('log')
ax1.set_xscale('log')
# RSSI
fig2, ax2 = plt.subplots()
ax2.scatter(rssi, reliability)
ax2.set_ylabel("Reliability (%)")
ax2.set_xlabel("Average GCS RSSI (W)")
ax2.set_yscale('log')
ax2.set_xscale('log')
# Delay
fig3, ax3 = plt.subplots()
ax3.scatter(delay, reliability)
ax3.set_ylabel("Reliability (%)")
ax3.set_xlabel("Average Delay (s)")
ax3.set_yscale('log')
ax3.set_xscale('log')
# Throughput
fig4, ax4 = plt.subplots()
ax4.scatter(throughput, reliability)
ax4.set_ylabel("Reliability (%)")
ax4.set_xlabel("Average Throughput (bytes/s)")
ax4.set_yscale('log')
ax4.set_xscale('log')
# Distance
fig5, ax5 = plt.subplots()
ax5.scatter(distance, reliability)
ax5.set_ylabel("Reliability (%)")
ax5.set_xlabel("Gateway Distance (m)")
ax5.set_yscale('log')
ax5.set_xscale('log')
# Horizontal Distance
fig6, ax6 = plt.subplots()
ax6.scatter(hor_distance, reliability)
ax6.set_ylabel("Reliability (%)")
ax6.set_xlabel("Gateway Horizontal Distance (m)")
ax6.set_yscale('log')
ax6.set_xscale('log')
# Throughput
fig7, ax7 = plt.subplots()
ax7.scatter(height, reliability)
ax7.set_ylabel("Reliability (%)")
ax7.set_xlabel("Swarm Height (m)")
ax7.set_yscale('log')
ax7.set_xscale('log')

plt.show()