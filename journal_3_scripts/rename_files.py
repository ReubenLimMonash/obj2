import os
import glob

PATH = "/home/clow0003/Reuben_ws/FANET_Dataset/DJISpark_Measured_Throughput_100000_Samples/data_manual_throughput_manet_interference_99"

csv_files = glob.glob(os.path.join(PATH, "*.csv"))
for file in csv_files:
    names = file.split("/")[-1].split("_")
    names.insert(3, "99")
    os.rename(file, os.path.join(PATH,"_".join(names)))