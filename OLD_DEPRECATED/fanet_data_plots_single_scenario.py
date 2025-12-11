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

file_path = "C:/Users/Joanne/Desktop/omnetpp-5.6.2/samples/Fanet/simulations/FANET/Height-20-GW-App[0]-Rx.csv"
# Read in CSV data to Dataframe
df = pd.read_csv(file_path)
# Get the data for feature of interest
rssi = rssi_to_np(df["RSSI"])
sinr = df["SINR"].to_numpy()
distance = df["Distance"].to_numpy()
throughput = df["Throughput"].to_numpy()
delay = df["Delay"].to_numpy()

# Plot the data
# SINR
fig1, ax1 = plt.subplots()
ax1.scatter(sinr, delay)
ax1.set_ylabel("Delay (s)")
ax1.set_xlabel("SINR")
ax1.set_yscale('log')
ax1.set_xscale('log')
# RSSI
fig2, ax2 = plt.subplots()
ax2.scatter(rssi, delay)
ax2.set_ylabel("Delay (s)")
ax2.set_xlabel("RSSI (W)")
ax2.set_yscale('log')
ax2.set_xscale('log')
# Distance
fig3, ax3 = plt.subplots()
ax3.scatter(distance, delay)
ax3.set_ylabel("Delay (s)")
ax3.set_xlabel("Distance (m)")
ax3.set_yscale('log')
ax3.set_xscale('log')
# Throughput
fig4, ax4 = plt.subplots()
ax4.scatter(throughput, delay)
ax4.set_ylabel("Delay (s)")
ax4.set_xlabel("Throughput (bytes/s)")
ax4.set_yscale('log')
ax4.set_xscale('log')

plt.show()