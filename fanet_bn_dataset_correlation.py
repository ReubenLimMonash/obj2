'''
Date: 28/02/2023
Desc: Fanet Dataset Correlation Study using MI and Spearman Rho
'''

import pandas as pd # for data manipulation 
import numpy as np
import os, sys, glob, math, pickle
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn import preprocessing

# This function helps to calculate probability distribution, which goes into BBN (note, can handle up to 2 parents)
def cpt_probs(df, child, parents):
    try:
        # dependencies_arr = [pd.Categorical(df[parent],categories=df[parent].cat.categories.tolist()) for parent in parents]
        dependencies_arr = [df[parent] for parent in parents]
        # cpt = pd.crosstab(dependencies_arr, df[child], rownames=parents, colnames=[child], margins=False, normalize='index', dropna=False).sort_index().to_numpy().reshape(-1).tolist()
        cpt = pd.crosstab(dependencies_arr, df[child], rownames=parents, colnames=[child], margins=False, normalize='index', dropna=False).sort_index()
        return cpt
    except Exception as err:
        print(err)
        return None 

def cpt_probs_freq(df, child, parents):
    try:
        # dependencies_arr = [pd.Categorical(df[parent],categories=df[parent].cat.categories.tolist()) for parent in parents]
        dependencies_arr = [df[parent] for parent in parents]
        # cpt = pd.crosstab(dependencies_arr, df[child], rownames=parents, colnames=[child], margins=False, normalize='index', dropna=False).sort_index().to_numpy().reshape(-1).tolist()
        cpt = pd.crosstab(dependencies_arr, df[child], rownames=parents, colnames=[child], margins=False, dropna=False).sort_index()
        return cpt
    except Exception as err:
        print(err)
        return None 

# Load classes_df for later parts (if previous part not run)
classes_df = pd.read_hdf("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_BPSK_6-5Mbps/classes_df_downlink.h5", 'Downlink')

import pandas as pd # for data manipulation 
import numpy as np
import os, sys, glob, math, pickle
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn import preprocessing

# This function helps to calculate probability distribution, which goes into BBN (note, can handle up to 2 parents)
def cpt_probs(df, child, parents):
    try:
        # dependencies_arr = [pd.Categorical(df[parent],categories=df[parent].cat.categories.tolist()) for parent in parents]
        dependencies_arr = [df[parent] for parent in parents]
        # cpt = pd.crosstab(dependencies_arr, df[child], rownames=parents, colnames=[child], margins=False, normalize='index', dropna=False).sort_index().to_numpy().reshape(-1).tolist()
        cpt = pd.crosstab(dependencies_arr, df[child], rownames=parents, colnames=[child], margins=False, normalize='index', dropna=False).sort_index()
        return cpt
    except Exception as err:
        print(err)
        return None 

def cpt_probs_freq(df, child, parents):
    try:
        # dependencies_arr = [pd.Categorical(df[parent],categories=df[parent].cat.categories.tolist()) for parent in parents]
        dependencies_arr = [df[parent] for parent in parents]
        # cpt = pd.crosstab(dependencies_arr, df[child], rownames=parents, colnames=[child], margins=False, normalize='index', dropna=False).sort_index().to_numpy().reshape(-1).tolist()
        cpt = pd.crosstab(dependencies_arr, df[child], rownames=parents, colnames=[child], margins=False, dropna=False).sort_index()
        return cpt
    except Exception as err:
        print(err)
        return None 

# Load classes_df for later parts (if previous part not run)
classes_df = pd.read_hdf("/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_BPSK_6-5Mbps/classes_df_downlink.h5", 'Downlink')

label_encoder = preprocessing.LabelEncoder()

sinr_label = label_encoder.fit_transform(classes_df["SINR_Class"])
h_dist_label = label_encoder.fit_transform(classes_df["H_Dist_Class"])
height_label = label_encoder.fit_transform(classes_df["Height_Class"])
num_members_label = label_encoder.fit_transform(classes_df["Num_Members_Class"])
sending_interval_label = label_encoder.fit_transform(classes_df["Sending_Interval_Class"])
packet_size_label = label_encoder.fit_transform(classes_df["Packet_Size_Class"])
ber_label = label_encoder.fit_transform(classes_df["BER_Class"])
delay_label = label_encoder.fit_transform(classes_df["Delay_Class"])
incr_rcvd_label = label_encoder.fit_transform(classes_df["Incorrectly_Received"])
delay_excd_label = label_encoder.fit_transform(classes_df["Delay_Exceeded"])
q_overflow_label = label_encoder.fit_transform(classes_df["Queue_Overflow"])
reliability_label = label_encoder.fit_transform(classes_df["Reliable"])

mi_hdist_sinr = mutual_info_classif(h_dist_label.reshape(-1,1), sinr_label.reshape(-1,1))
rho_hdist_sinr = stats.spearmanr(h_dist_label, sinr_label)
print("H_Dist to SINR - MI: {}, Spearman: {}".format(mi_hdist_sinr, rho_hdist_sinr))
mi_height_sinr = mutual_info_classif(height_label.reshape(-1,1), sinr_label.reshape(-1,1))
rho_height_sinr = stats.spearmanr(height_label, sinr_label)
print("Height to SINR - MI: {}, Spearman: {}".format(mi_height_sinr, rho_height_sinr))
mi_num_members_sinr = mutual_info_classif(num_members_label.reshape(-1,1), sinr_label.reshape(-1,1))
rho_num_members_sinr = stats.spearmanr(num_members_label, sinr_label)
print("num_members to SINR - MI: {}, Spearman: {}".format(mi_num_members_sinr, rho_num_members_sinr))
mi_sending_interval_sinr = mutual_info_classif(sending_interval_label.reshape(-1,1), sinr_label.reshape(-1,1))
rho_sending_interval_sinr = stats.spearmanr(sending_interval_label, sinr_label)
print("sending_interval to SINR - MI: {}, Spearman: {}".format(mi_sending_interval_sinr, rho_sending_interval_sinr))
mi_packet_size_sinr = mutual_info_classif(packet_size_label.reshape(-1,1), sinr_label.reshape(-1,1))
rho_packet_size_sinr = stats.spearmanr(packet_size_label, sinr_label)
print("packet_size to SINR - MI: {}, Spearman: {}".format(mi_packet_size_sinr, rho_packet_size_sinr))

mi_sinr_ber = mutual_info_classif(sinr_label.reshape(-1,1), ber_label.reshape(-1,1))
rho_sinr_ber = stats.spearmanr(sinr_label, ber_label)
print("SINR to BER - MI: {}, Spearman: {}".format(mi_sinr_ber, rho_sinr_ber))

mi_ber_delay = mutual_info_classif(ber_label.reshape(-1,1), delay_label.reshape(-1,1))
rho_ber_delay = stats.spearmanr(ber_label, delay_label)
print("BER to Delay - MI: {}, Spearman: {}".format(mi_ber_delay, rho_ber_delay))
mi_num_members_delay = mutual_info_classif(num_members_label.reshape(-1,1), delay_label.reshape(-1,1))
rho_num_members_delay = stats.spearmanr(num_members_label, delay_label)
print("No. UAVs to Delay - MI: {}, Spearman: {}".format(mi_num_members_delay, rho_num_members_delay))
mi_packet_size_delay = mutual_info_classif(packet_size_label.reshape(-1,1), delay_label.reshape(-1,1))
rho_packet_size_delay = stats.spearmanr(packet_size_label, delay_label)
print("Packet Size to Delay - MI: {}, Spearman: {}".format(mi_packet_size_delay, rho_packet_size_delay))
mi_sending_interval_delay = mutual_info_classif(sending_interval_label.reshape(-1,1), delay_label.reshape(-1,1))
rho_sending_interval_delay = stats.spearmanr(sending_interval_label, delay_label)
print("Sending Interval to Delay - MI: {}, Spearman: {}".format(mi_sending_interval_delay, rho_sending_interval_delay))

mi_ber_incr_rcvd = mutual_info_classif(ber_label.reshape(-1,1), incr_rcvd_label.reshape(-1,1))
rho_ber_incr_rcvd = stats.spearmanr(ber_label, incr_rcvd_label)
print("BER to Incorrect Rcvd - MI: {}, Spearman: {}".format(mi_ber_incr_rcvd, rho_ber_incr_rcvd))
mi_delay_incr_rcvd = mutual_info_classif(delay_label.reshape(-1,1), incr_rcvd_label.reshape(-1,1))
rho_delay_incr_rcvd = stats.spearmanr(delay_label, incr_rcvd_label)
print("Delay to Incorrect Rcvd - MI: {}, Spearman: {}".format(mi_delay_incr_rcvd, rho_delay_incr_rcvd))

mi_ber_delay_excd = mutual_info_classif(ber_label.reshape(-1,1), delay_excd_label.reshape(-1,1))
rho_ber_delay_excd = stats.spearmanr(ber_label, delay_excd_label)
print("BER to Delay Excd - MI: {}, Spearman: {}".format(mi_ber_delay_excd, rho_ber_delay_excd))
mi_delay_delay_excd = mutual_info_classif(delay_label.reshape(-1,1), delay_excd_label.reshape(-1,1))
rho_delay_delay_excd = stats.spearmanr(delay_label, delay_excd_label)
print("Delay to Delay Excd - MI: {}, Spearman: {}".format(mi_delay_delay_excd, rho_delay_delay_excd))

mi_ber_q_overflow = mutual_info_classif(ber_label.reshape(-1,1), q_overflow_label.reshape(-1,1))
rho_ber_q_overflow = stats.spearmanr(ber_label, q_overflow_label)
print("BER to Q Overflow - MI: {}, Spearman: {}".format(mi_ber_q_overflow, rho_ber_q_overflow))
mi_delay_q_overflow = mutual_info_classif(delay_label.reshape(-1,1), q_overflow_label.reshape(-1,1))
rho_delay_q_overflow = stats.spearmanr(delay_label, q_overflow_label)
print("Delay to Q Overflow - MI: {}, Spearman: {}".format(mi_delay_q_overflow, rho_delay_q_overflow))
mi_num_members_q_overflow = mutual_info_classif(num_members_label.reshape(-1,1), q_overflow_label.reshape(-1,1))
rho_num_members_q_overflow = stats.spearmanr(num_members_label, q_overflow_label)
print("No. UAVs to Q Overflow - MI: {}, Spearman: {}".format(mi_num_members_q_overflow, rho_num_members_q_overflow))
mi_packet_size_q_overflow = mutual_info_classif(packet_size_label.reshape(-1,1), q_overflow_label.reshape(-1,1))
rho_packet_size_q_overflow = stats.spearmanr(packet_size_label, q_overflow_label)
print("Packet Size to Q Overflow - MI: {}, Spearman: {}".format(mi_packet_size_q_overflow, rho_packet_size_q_overflow))
mi_sending_interval_q_overflow = mutual_info_classif(sending_interval_label.reshape(-1,1), q_overflow_label.reshape(-1,1))
rho_sending_interval_q_overflow = stats.spearmanr(sending_interval_label, q_overflow_label)
print("Sending Interval to Q Overflow - MI: {}, Spearman: {}".format(mi_sending_interval_q_overflow, rho_sending_interval_q_overflow))

