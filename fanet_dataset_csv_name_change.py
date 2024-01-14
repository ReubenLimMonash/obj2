import glob, os

sim_root_path = "/home/rlim0005/FANET_Dataset/Dataset_NP10000_DJISpark/data"
csv_files = glob.glob(sim_root_path + "/*.csv")
for file in csv_files:
    segments = file.split("_")
    modulation = segments[-3].split("-")[-1]
    if modulation == '64':
        bitrate = 52
    elif modulation == '16':
        bitrate = 26
    elif modulation == 'QPSK':
        bitrate = 13
    elif modulation == 'BPSK':
        bitrate = 6.5
    segments[-6] = "BitRate-{}".format(bitrate)
    new_name = "_".join(segments)
    os.rename(file, new_name)