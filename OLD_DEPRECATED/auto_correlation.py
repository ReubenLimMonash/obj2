'''
Date: 09/03/2022
Desc: To calculate auto-correlation of time series data.
'''

import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
file_path = "C:/Users/Joanne/Desktop/omnetpp-5.6.2/samples/Fanet/simulations/M6/Circle/Low_Speed/GW--App[0]-Rx.csv"
# Read in CSV data to Dataframe
df = pd.read_csv(file_path)
#Get throughput data (dropping the first data)
throughput = df["Throughput (bps)"].iloc[1:]

# Use the Autocorrelation function
# from the statsmodel library passing
# our DataFrame object in as the data
# Note: Limiting Lags to 50
# plot_acf(throughput, lags=5)
plot_acf(throughput)
# Show the AR as a plot
plt.title("Low Speed")
plt.xlabel("Time lag")
plt.ylabel("Autocorrelation")
plt.show()