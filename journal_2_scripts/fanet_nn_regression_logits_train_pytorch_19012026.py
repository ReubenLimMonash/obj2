import argparse
import os
import random
from datetime import datetime
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

'''
NOTE: Edit load_dataset() function to choose different input features for the model.
This function trains a neural network regression model using PyTorch to predict reliability logits instead of the probability.

Usage:

python fanet_nn_regression_logits_train_pytorch_19012026.py \
  --data /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed/Downlink_Reliability.csv \
  --output_file Downlink_regression.h5 \
  --output_folder /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts_regression_distance_v3_logits_sgolay \
  --epochs 10000 \
  --batch-size 128

python fanet_nn_regression_logits_train_pytorch_19012026.py \
  --data /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed/Uplink_Reliability.csv \
  --output_file Uplink_regression.h5 \
  --output_folder /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts_regression_distance_v3_logits_sgolay \
  --epochs 10000 \
  --batch-size 128

python fanet_nn_regression_logits_train_pytorch_19012026.py \
  --data /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed/Video_Reliability.csv \
  --output_file Video_regression.h5 \
  --output_folder /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts_regression_distance_v3_logits_sgolay \
  --epochs 10000 \
  --batch-size 128
'''

# Define SINR_TH here if using lognormal_approx_reliability (in linear scale)
# SINR_TH = 0.00106802  # downlink (-29.7142 dB)
# SINR_TH = 0.0104514  # uplink (-19.8083 dB)
# SINR_TH = 0.0743382  # video (-11.2879 dB)

def lognormal_approx_reliability(mean_sinr, std_dev_sinr, rel_th):
	s_R = math.sqrt(math.log((std_dev_sinr**2)/(mean_sinr**2) + 1))
	m_R = math.log(mean_sinr) - 0.5 * s_R**2
	cdf = 0.5 + 0.5 * math.erf((math.log(rel_th) - m_R) / (s_R * math.sqrt(2)))
	return 1 - cdf

def set_seed(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

def get_mcs_index(df_in):
	'''
	Gets the MCS index based on modulation and bitrate column of the df_in
	'''
	df = df_in.copy()
	df["MCS"] = ''
	df.loc[(df["Modulation"] == "BPSK") & (df["Bitrate"] == 6.5), "MCS"] = 0 # MCS Index 0
	df.loc[(df["Modulation"] == "QPSK") & (df["Bitrate"] == 13), "MCS"] = 1 # MCS Index 0
	df.loc[(df["Modulation"] == "QPSK") & (df["Bitrate"] == 19.5), "MCS"] = 2 # MCS Index 0
	df.loc[(df["Modulation"] == "QAM16") & (df["Bitrate"] == 26), "MCS"] = 3 # MCS Index 0
	df.loc[(df["Modulation"] == "QAM16") & (df["Bitrate"] == 39), "MCS"] = 4 # MCS Index 0
	df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 52), "MCS"] = 5 # MCS Index 0
	df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 58.5), "MCS"] = 6 # MCS Index 0
	df.loc[(df["Modulation"] == "QAM64") & (df["Bitrate"] == 65), "MCS"] = 7 # MCS Index 0

	return df

def load_dataset(csv_path: str):
	df_dtypes = {"Horizontal_Distance": np.float32, "Height": np.int16, "UAV_Sending_Interval": np.float32, "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
				"Num_Sent": np.int32, "Num_Reliable": np.int32, "Num_Delay_Excd": np.int32, "Num_Incr_Rcvd": np.int32, "Num_Q_Overflow": np.int32, "Modulation": str, "Bitrate": np.float64}
	dataset_df = pd.read_csv(csv_path, 
				usecols = ["Horizontal_Distance", "Height", "Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Bitrate", 
							"Num_Sent", "Num_Reliable", "Num_Delay_Excd", "Num_Incr_Rcvd", "Num_Q_Overflow"],
				dtype=df_dtypes)
	dataset_df = dataset_df.loc[dataset_df["Horizontal_Distance"] <= 600]  # Apply max horizontal distance of 600 m
	dataset_df = get_mcs_index(dataset_df)

	"""For inputs: Horizontal distance, Height, MCS, UAV Sending Interval"""
	dataset_df = normalize_data(dataset_df, columns=["Horizontal_Distance", "Height", "UAV_Sending_Interval", "MCS"])
	features = dataset_df[["Horizontal_Distance", "Height", "MCS", "UAV_Sending_Interval"]].to_numpy(dtype=np.float32)
	""" For inputs: Lognormal Reliability, MCS, USI """
	# dataset_df["Lognormal_Reliability"] = dataset_df.apply(lambda x: lognormal_approx_reliability(x["Mean_SINR"], x["Std_Dev_SINR"], SINR_TH), axis=1)
	# dataset_df = normalize_data(dataset_df, columns=["UAV_Sending_Interval", "MCS"])
	# features = dataset_df[["Lognormal_Reliability", "MCS", "UAV_Sending_Interval"]].to_numpy(dtype=np.float32)
	""" For inputs: Mean_SINR, Std_Dev_SINR, MCS, USI """
	# dataset_df = normalize_data(dataset_df, columns=["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"])
	# features = dataset_df[["Mean_SINR", "Std_Dev_SINR", "MCS", "UAV_Sending_Interval"]].to_numpy(dtype=np.float32)
	
	dataset_df["Reliability"] = dataset_df["Num_Reliable"] / dataset_df["Num_Sent"]

	"""For target  = 'Reliability'"""
	# target = dataset_df["Reliability"].to_numpy(dtype=np.float32)

	'''For target = "Logit_Reliability"'''
	dataset_df = create_logits(dataset_df)
	# dataset_df = normalize_data(dataset_df, columns=["Logit_Reliability"])
	dataset_df = sgolay_filter_smooth(dataset_df, window_length=11, polyorder=2)
	target = dataset_df["Logit_Reliability"].to_numpy(dtype=np.float32)

	return features, target

def normalize_data(df_in, columns=[], save_details_path=None):
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
	max_h_dist = 600
	min_h_dist = 0
	max_mcs = 7
	min_mcs = 0

	# For logit normalization
	eps = 1e-6
	max_logit = math.log((1 - eps) / eps)
	min_logit = math.log(eps / (1 - eps))

	# Normalize data (Min Max Normalization between [-1,1])
	if "Height" in columns:
		df["Height"] = df["Height"].apply(lambda x: 2*(x-min_height)/(max_height-min_height) - 1)
	if "Horizontal_Distance" in columns:
		df["Horizontal_Distance"] = df["Horizontal_Distance"].apply(lambda x: 2*(x-min_h_dist)/(max_h_dist-min_h_dist) - 1)
	if "Mean_SINR" in columns:
		df["Mean_SINR"] = df["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1) # Convert to dB space
	if "Std_Dev_SINR" in columns:
		df["Std_Dev_SINR"] = df["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1) # Convert to dB space
	if "UAV_Sending_Interval" in columns:
		df["UAV_Sending_Interval"] = df["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 66.7: 0.5, 100:1, 1000:2})
	if "Packet_State" in columns:
		df['Packet_State'] = df['Packet_State'].replace({"Reliable":0, "QUEUE_OVERFLOW":1, "RETRY_LIMIT_REACHED":2, "Delay_Exceeded":3})
	if "Modulation" in columns:
		df['Modulation'] = df['Modulation'].replace({"BPSK":1, "QPSK":0.3333, 16:-0.3333, "QAM-16":-0.3333, "QAM16":-0.3333, 64:-1, "QAM-64":-1, "QAM64":-1})
	if "MCS" in columns:
		df["MCS"] = df["MCS"].apply(lambda x: 2*(x-min_mcs)/(max_mcs-min_mcs) - 1)
	if "Logit_Reliability" in columns:
		df["Logit_Reliability"] = df["Logit_Reliability"].apply(lambda x: 2*(x - min_logit)/(max_logit - min_logit) - 1)

	# Record details of inputs and output for model
	if save_details_path is not None:
		f = open(os.path.join(save_details_path,"model_details.txt"), "w")
		f.write("Max Height (m): {}\n".format(max_height))
		f.write("Min Height (m): {}\n".format(min_height))
		f.write("Max H_Dist (m): {}\n".format(max_h_dist))
		f.write("Min H_Dist (m): {}\n".format(min_h_dist))
		f.write("Max Mean SINR (dB): {}\n".format(max_mean_sinr))
		f.write("Min Mean SINR (dB): {}\n".format(min_mean_sinr))
		f.write("Max Std Dev SINR (dB): {}\n".format(max_std_dev_sinr))
		f.write("Min Std Dev SINR (dB): {}\n".format(min_std_dev_sinr))
		f.write("[BPSK: 1, QPSK: 0.3333, QAM16: -0.3333, QAM64: -1]\n")
		f.write("UAV Sending Interval: [10:-1, 20:-0.5, 40:0, 66.7: 0.5, 100:1, 1000:2]\n")
		f.write("Output: ['Reliable':0, 'QUEUE_OVERFLOW':1, 'RETRY_LIMIT_REACHED':2, 'Delay_Exceeded':3]\n")
		f.close()

	return df

def create_logits(df):
	'''
	Takes a df with the column 'Reliability' and creates logits for regression.
	Clips the logits to avoid inf values.
	'''
	df_out = df.copy()
	eps = 1e-6
	df_out['Reliability'] = df_out['Reliability'].apply(lambda x: min(max(x, eps), 1 - eps))
	df_out['Logit_Reliability'] = df_out['Reliability'].apply(lambda x: math.log(x / (1 - x)))
	return df_out

def sgolay_filter_smooth(data, window_length=11, polyorder=2):
	'''
	Applies Savitzky-Golay filter to smoothen the data.
	The smoothing is applied along horizontal distance, for each combination of height, usi and mcs.
	'''
	# Sort by USI, MCS, Height, Horizontal_Distance first to make sure that the data is in order
	data = data.sort_values(by=["UAV_Sending_Interval", "MCS", "Height", "Horizontal_Distance"])
	# Apply Savitzky-Golay filter
	heights = data["Height"].unique()
	usi_values = data["UAV_Sending_Interval"].unique()
	mcs_values = data["MCS"].unique()
	for h in heights:
		for usi in usi_values:
			for mcs in mcs_values:
				mask = (data["Height"] == h) & (data["UAV_Sending_Interval"] == usi) & (data["MCS"] == mcs)
				data.loc[mask, "Logit_Reliability"] = savgol_filter(data.loc[mask, "Logit_Reliability"], window_length=window_length, polyorder=polyorder)
	return data

# v3
class RegressionModel(nn.Module):
	def __init__(self, input_dim: int):
		super(RegressionModel, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(input_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 16),
			nn.ReLU(),
			nn.Linear(16, 4),
			nn.ReLU(),
			nn.Linear(4, 1),
			# nn.Sigmoid()
		)
	
	def forward(self, x):
		return self.model(x)

# v4
# class RegressionModel(nn.Module):
# 	def __init__(self, input_dim: int):
# 		super(RegressionModel, self).__init__()
# 		self.model = nn.Sequential(
# 			nn.Linear(input_dim, 256),
# 			nn.ReLU(),
# 			nn.Linear(256, 64),
# 			nn.ReLU(),
# 			nn.Linear(64, 16),
# 			nn.ReLU(),
# 			nn.Linear(16, 4),
# 			nn.ReLU(),
# 			nn.Linear(4, 1),
# 			# nn.Sigmoid()
# 		)
	
# 	def forward(self, x):
# 		return self.model(x)

# v1
# class RegressionModel(nn.Module):
# 	def __init__(self, input_dim: int):
# 		super(RegressionModel, self).__init__()
# 		self.model = nn.Sequential(
# 			nn.Linear(input_dim, 100),
# 			nn.ReLU(),
# 			nn.Linear(100, 50),
# 			nn.ReLU(),
# 			nn.Linear(50, 25),
# 			nn.ReLU(),
# 			nn.Linear(25, 10),
# 			nn.ReLU(),
# 			nn.Linear(10, 1),
# 			nn.Sigmoid()
# 		)
	
# 	def forward(self, x):
# 		return self.model(x)

# v2
# class RegressionModel(nn.Module):
# 	def __init__(self, input_dim: int):
# 		super(RegressionModel, self).__init__()
# 		self.model = nn.Sequential(
# 			nn.Linear(input_dim, 512),
# 			nn.ReLU(),
# 			nn.Linear(512, 256),
# 			nn.ReLU(),
# 			nn.Linear(256, 128),
# 			nn.ReLU(),
# 			nn.Linear(128, 64),
# 			nn.ReLU(),
# 			nn.Linear(64, 32),
# 			nn.ReLU(),
# 			nn.Linear(32, 16),
# 			nn.ReLU(),
# 			nn.Linear(16, 1),
# 			nn.Sigmoid()
# 		)
	
# 	def forward(self, x):
# 		return self.model(x)

def prepare_data(features: np.ndarray, target: np.ndarray, test_size: float, seed: int, batch_size: int):
	x_train, x_val, y_train, y_val = train_test_split(features, target, test_size=test_size, random_state=seed)
	
	# Convert to PyTorch tensors
	x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
	y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
	x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
	y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
	
	# Create DataLoaders
	train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
	val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	
	return train_loader, val_loader


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train NN regressor for reliability prediction")
	parser.add_argument("--data", required=True, help="Path to input CSV containing features and reliability")
	parser.add_argument("--output_folder", default="artifacts", help="Directory to store model and logs")
	parser.add_argument("--output_file", required=True, help="Filename to store the trained model")
	parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
	parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
	parser.add_argument("--val-size", type=float, default=0.2, help="Validation split size")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	set_seed(args.seed)

	# Setup device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	os.makedirs(args.output_folder, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_dir = os.path.join(args.output_folder, f"{args.output_file.split('.')[0]}_logs_{timestamp}")
	os.makedirs(log_dir, exist_ok=True)

	features, target = load_dataset(args.data)
	train_loader, val_loader = prepare_data(features, target, args.val_size, args.seed, args.batch_size)

	model = RegressionModel(input_dim=features.shape[1]).to(device)
	criterion = nn.MSELoss()
	# criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	# Training loop with early stopping
	best_val_loss = float('inf')
	patience_counter = 0
	patience = 100
	best_model_state = None
	
	training_log = []

	for epoch in range(args.epochs):
		# Training phase
		model.train()
		train_loss = 0.0
		train_mae = 0.0
		num_batches = 0

		for x_batch, y_batch in train_loader:
			x_batch, y_batch = x_batch.to(device), y_batch.to(device)
			
			optimizer.zero_grad()
			outputs = model(x_batch)
			loss = criterion(outputs, y_batch)
			loss.backward()
			optimizer.step()
			
			train_loss += loss.item()
			train_mae += torch.mean(torch.abs(outputs - y_batch)).item()
			num_batches += 1

		train_loss /= num_batches
		train_mae /= num_batches

		# Validation phase
		model.eval()
		val_loss = 0.0
		val_mae = 0.0
		num_val_batches = 0

		with torch.no_grad():
			for x_batch, y_batch in val_loader:
				x_batch, y_batch = x_batch.to(device), y_batch.to(device)
				outputs = model(x_batch)
				loss = criterion(outputs, y_batch)
				
				val_loss += loss.item()
				val_mae += torch.mean(torch.abs(outputs - y_batch)).item()
				num_val_batches += 1

		val_loss /= num_val_batches
		val_mae /= num_val_batches

		# Log results
		training_log.append({
			'epoch': epoch + 1,
			'loss': train_loss,
			'mae': train_mae,
			'val_loss': val_loss,
			'val_mae': val_mae
		})

		if (epoch + 1) % 10 == 0 or epoch == 0:
			print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {train_loss:.6f}, MAE: {train_mae:.6f}, Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")

		# Early stopping and model checkpointing
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_val_mae = val_mae
			patience_counter = 0
			best_model_state = model.state_dict().copy()
			# Save best model
			torch.save(best_model_state, os.path.join(args.output_folder, args.output_file))
		# UNCOMMENT BELOW TO ENABLE EARLY STOPPING
		# else:
		# 	patience_counter += 1
		# 	if patience_counter >= patience:
		# 		print(f"Early stopping at epoch {epoch+1}")
		# 		break

	# Save training log
	log_df = pd.DataFrame(training_log)
	log_df.to_csv(os.path.join(log_dir, "training_log.csv"), index=False)

	# Load best model and save final version
	if best_model_state is not None:
		model.load_state_dict(best_model_state)
	torch.save(model.state_dict(), os.path.join(args.output_folder, "final_" + args.output_file))

	# Save final metrics for quick inspection
	summary_path = os.path.join(args.output_folder, "metrics_summary.txt")
	with open(summary_path, "w", encoding="utf-8") as f:
		f.write(f"Best val_loss: {best_val_loss:.6f}\n")
		f.write(f"Best val_mae: {best_val_mae:.6f}\n")
		f.write(f"Logs: {log_dir}\n")


if __name__ == "__main__":
	main()
