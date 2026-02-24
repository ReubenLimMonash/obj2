import argparse
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, max_error

'''
NOTE: Edit load_dataset() function to choose different input features for the model.
NOTE: Edit SINR_TH variable to match the test dataset (downlink, uplink, video).

Usage:
python journal_2_scripts/fanet_nn_regression_logits_test_pytorch_19012026.py \
  --data /path/to/test_dataset.csv \
  --model /path/to/final_model.pth \
  --output_folder test_results \
  --link link

python fanet_nn_regression_logits_test_pytorch_19012026.py \
  --data /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/complete_testing_dmax_dataset/data_processed_complete/Downlink_Reliability.csv \
  --model /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts_regression_distance_v3_logits_sgolay/final_Downlink_regression.h5 \
  --output_folder /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/test_results_regression_distance_v3_logits_sgolay \
  --link downlink

python fanet_nn_regression_logits_test_pytorch_19012026.py \
  --data /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/complete_testing_dmax_dataset/data_processed_complete/Uplink_Reliability.csv \
  --model /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts_regression_distance_v3_logits_sgolay/final_Uplink_regression.h5 \
  --output_folder /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/test_results_regression_distance_v3_logits_sgolay \
  --link uplink

python fanet_nn_regression_logits_test_pytorch_19012026.py \
  --data /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/complete_testing_dmax_dataset/data_processed_complete/Video_Reliability.csv \
  --model /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts_regression_distance_v3_logits_sgolay/final_Video_regression.h5 \
  --output_folder /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/test_results_regression_distance_v3_logits_sgolay \
  --link video
'''

# Define SINR_TH here if using lognormal_approx_reliability (in linear scale) NOTE: UNCOMMENT THE RIGHT ONE BASED ON THE TEST DATASET
# SINR_TH = 0.00106802  # downlink (-29.7142 dB)
# SINR_TH = 0.0104514  # uplink (-19.8083 dB)
SINR_TH = 0.0743382  # video (-11.2879 dB)


def load_test_dataset(csv_path: str):
	'''
	Load and preprocess test dataset with the same transformations as training.
	'''
	df_dtypes = {"Horizontal_Distance": np.float32, "Height": np.int16, "UAV_Sending_Interval": np.float16, "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
				"Num_Reliable": np.int32, "Num_Delay_Excd": np.int32, "Num_Fail_Other": np.int32, "Modulation": str, "Bitrate": np.float64}
	dataset_df = pd.read_csv(csv_path, 
				usecols = ["Horizontal_Distance", "Height", "Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Bitrate", 
							"Num_Reliable", "Num_Delay_Excd", "Num_Fail_Other"],
				dtype=df_dtypes)
	
	dataset_df = dataset_df.loc[dataset_df["Horizontal_Distance"] <= 600]  # Apply max horizontal distance
	dataset_df = get_mcs_index(dataset_df)
	
	# """For inputs: Horizontal distance, Height, MCS, UAV Sending Interval"""
	dataset_df = normalize_data(dataset_df, columns=["Horizontal_Distance", "Height", "UAV_Sending_Interval", "MCS"])
	features = dataset_df[["Horizontal_Distance", "Height", "MCS", "UAV_Sending_Interval"]].to_numpy(dtype=np.float32)
	""" For inputs: Lognormal Reliability, MCS, USI """
	# dataset_df["Lognormal_Reliability"] = dataset_df.apply(lambda x: lognormal_approx_reliability(x["Mean_SINR"], x["Std_Dev_SINR"], SINR_TH), axis=1)
	# dataset_df = normalize_data(dataset_df, columns=["UAV_Sending_Interval", "MCS"])
	# features = dataset_df[["Lognormal_Reliability", "MCS", "UAV_Sending_Interval"]].to_numpy(dtype=np.float32)
	""" For inputs: Mean_SINR, Std_Dev_SINR, MCS, USI """
	# dataset_df = normalize_data(dataset_df, columns=["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "MCS"])
	# features = dataset_df[["Mean_SINR", "Std_Dev_SINR", "MCS", "UAV_Sending_Interval"]].to_numpy(dtype=np.float32)
	
	dataset_df["Reliability"] = dataset_df["Num_Reliable"] / (dataset_df["Num_Reliable"] + dataset_df["Num_Delay_Excd"] + dataset_df["Num_Fail_Other"])
	
	# Even though the model predicts logits, for evaluation we need the actual reliability values
	target = dataset_df["Reliability"].to_numpy(dtype=np.float32)
	
	return features, target, dataset_df

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
# 			nn.Linear(input_dim, 64),
# 			nn.ReLU(),
# 			nn.Linear(64, 32),
# 			nn.ReLU(),
# 			nn.Linear(32, 16),
# 			nn.ReLU(),
# 			nn.Linear(16, 4),
# 			nn.ReLU(),
# 			nn.Linear(4, 1),
# 			# nn.Sigmoid()
# 		)
	
# 	def forward(self, x):
# 		return self.model(x)
	
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

def lognormal_approx_reliability(mean_sinr, std_dev_sinr, rel_th):
	s_R = math.sqrt(math.log((std_dev_sinr**2)/(mean_sinr**2) + 1))
	m_R = math.log(mean_sinr) - 0.5 * s_R**2
	cdf = 0.5 + 0.5 * math.erf((math.log(rel_th) - m_R) / (s_R * math.sqrt(2)))
	return 1 - cdf


def get_mcs_index(df_in):
	'''
	Gets the MCS index based on modulation and bitrate column of the df_in
	'''
	df = df_in.copy()
	df["MCS"] = ''
	df.loc[(df["Bitrate"] == 6.5), "MCS"] = 0
	df.loc[(df["Bitrate"] == 13), "MCS"] = 1
	df.loc[(df["Bitrate"] == 19.5), "MCS"] = 2
	df.loc[(df["Bitrate"] == 26), "MCS"] = 3
	df.loc[(df["Bitrate"] == 39), "MCS"] = 4
	df.loc[(df["Bitrate"] == 52), "MCS"] = 5
	df.loc[(df["Bitrate"] == 58.5), "MCS"] = 6
	df.loc[(df["Bitrate"] == 65), "MCS"] = 7

	return df


def normalize_data(df_in, columns=[]):
	'''
	Normalize data using the same logic as training.
	columns: The pandas data columns to normalize, given as a list of column names
	'''
	df = df_in.copy()
	# Define the ranges of parameters
	max_mean_sinr = 10*math.log10(1123)
	max_std_dev_sinr = 10*math.log10(466)
	min_mean_sinr = 10*math.log10(0.2)
	min_std_dev_sinr = 10*math.log10(0.7)
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
		df["Mean_SINR"] = df["Mean_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_mean_sinr)/(max_mean_sinr-min_mean_sinr) - 1)
	if "Std_Dev_SINR" in columns:
		df["Std_Dev_SINR"] = df["Std_Dev_SINR"].apply(lambda x: 2*(10*math.log10(x)-min_std_dev_sinr)/(max_std_dev_sinr-min_std_dev_sinr) - 1)
	if "UAV_Sending_Interval" in columns:
		df["UAV_Sending_Interval"] = df["UAV_Sending_Interval"].replace({10:-1, 20:-0.5, 40:0, 66.7: 0.5, 100:1, 1000:2})
	if "MCS" in columns:
		df["MCS"] = df["MCS"].apply(lambda x: 2*(x-min_mcs)/(max_mcs-min_mcs) - 1)
	if "Logit_Reliability" in columns:
		df["Logit_Reliability"] = df["Logit_Reliability"].apply(lambda x: 2*(x - min_logit)/(max_logit - min_logit) - 1)

	return df


def unnormalize_data(df_in, columns=[]):
	'''
	Function to reverse normalization
	'''
	df = df_in.copy()
	# Define the ranges of parameters
	max_mean_sinr = 10*math.log10(1123)
	max_std_dev_sinr = 10*math.log10(466)
	min_mean_sinr = 10*math.log10(0.2)
	min_std_dev_sinr = 10*math.log10(0.7)
	max_height = 300
	min_height = 60
	max_h_dist = 600
	min_h_dist = 0
	max_mcs = 7
	min_mcs = 0
	# Reverse normalization (Min Max Normalization between [-1,1])
	if "Height" in columns:
		df["Height"] = df["Height"].apply(lambda x: ((x + 1)/2)*(max_height - min_height) + min_height)
	if "Horizontal_Distance" in columns:
		df["Horizontal_Distance"] = df["Horizontal_Distance"].apply(lambda x: ((x + 1)/2)*(max_h_dist - min_h_dist) + min_h_dist)
	if "Mean_SINR" in columns:
		df["Mean_SINR"] = df["Mean_SINR"].apply(lambda x: 10**((((x + 1)/2)*(max_mean_sinr - min_mean_sinr) + min_mean_sinr)/10))
	if "Std_Dev_SINR" in columns:
		df["Std_Dev_SINR"] = df["Std_Dev_SINR"].apply(lambda x: 10**((((x + 1)/2)*(max_std_dev_sinr - min_std_dev_sinr) + min_std_dev_sinr)/10))
	if "UAV_Sending_Interval" in columns:
		df["UAV_Sending_Interval"] = df["UAV_Sending_Interval"].replace({-1:10, -0.5:20, 0:40, 0.5:66.7, 1:100, 2:1000})
	if "MCS" in columns:
		df["MCS"] = df["MCS"].apply(lambda x: ((x + 1)/2)*(max_mcs - min_mcs) + min_mcs)

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

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Test PyTorch NN regressor for reliability prediction")
	parser.add_argument("--data", required=True, help="Path to test dataset CSV")
	parser.add_argument("--model", required=True, help="Path to trained PyTorch model (.pth or .pt)")
	parser.add_argument("--link", default="downlink", help="Used for filename purposes only")
	parser.add_argument("--output_folder", default="test_results", help="Directory to store test results")
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	# Setup device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	# Create output directory
	os.makedirs(args.output_folder, exist_ok=True)

	# Load test dataset
	print(f"Loading test dataset from {args.data}...")
	x_test, y_test, test_df = load_test_dataset(args.data)
	print(f"Loaded {len(x_test)} test samples with {x_test.shape[1]} features")

	# Load trained model
	print(f"Loading model from {args.model}...")
	model = RegressionModel(input_dim=x_test.shape[1]).to(device)
	model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
	model.eval()
	print("Model loaded successfully")

	# Make predictions
	print("Making predictions...")
	x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
	with torch.no_grad():
		y_pred_tensor = model(x_test_tensor)
	y_pred = y_pred_tensor.cpu().numpy().flatten()
	# Apply sigmoid function to predicted logits to get reliability in [0,1]
	y_pred = 1 / (1 + np.exp(-y_pred))

	# Calculate metrics (over all data)
	mse = mean_squared_error(y_test, y_pred)
	rmse = np.sqrt(mse)
	mae = mean_absolute_error(y_test, y_pred)
	mape = mean_absolute_percentage_error(y_test, y_pred)
	maxae = max_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)
	# Count how many error is above 0.05
	num_above_005 = np.sum(np.abs(y_test - y_pred) > 0.05)
	num_above_010 = np.sum(np.abs(y_test - y_pred) > 0.10)

	# Print results
	print("\n" + "="*60)
	print("EVALUATION METRICS (OVER ALL DATA)")
	print("="*60)
	print(f"Mean Squared Error (MSE):  {mse:.6f}")
	print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
	print(f"Mean Absolute Error (MAE): {mae:.6f}")
	print(f"Mean Absolute Percentage Error (MAPE): {mape:.6f}")
	print(f"Maximum Absolute Error (MaxAE): {maxae:.6f}")
	print(f"R² Score: {r2:.6f}")
	print(f"Percentage of samples with Absolute Error > 0.05: {num_above_005/len(y_test)}")
	print(f"Percentage of samples with Absolute Error > 0.10: {num_above_010/len(y_test)}")
	print("="*60 + "\n")

	# # Calculate metrics (for data where reliability is above 90%)
	# filter_indices = y_test >= 0.9
	# y_test_filtered = y_test[filter_indices]
	# y_pred_filtered = y_pred[filter_indices]
	# x_test_filtered = x_test[filter_indices]
	# mse_filtered = mean_squared_error(y_test_filtered, y_pred_filtered)
	# rmse_filtered = np.sqrt(mse_filtered)
	# mae_filtered = mean_absolute_error(y_test_filtered, y_pred_filtered)
	# mape_filtered = mean_absolute_percentage_error(y_test_filtered, y_pred_filtered)
	# maxae_filtered = max_error(y_test_filtered, y_pred_filtered)
	# r2_filtered = r2_score(y_test_filtered, y_pred_filtered)

	# # Print results
	# print("\n" + "="*60)
	# print("EVALUATION METRICS (RELIABILITY >= 90%)")
	# print("="*60)
	# print(f"Mean Squared Error (MSE):  {mse_filtered:.6f}")
	# print(f"Root Mean Squared Error (RMSE): {rmse_filtered:.6f}")
	# print(f"Mean Absolute Error (MAE): {mae_filtered:.6f}")
	# print(f"Mean Absolute Percentage Error (MAPE): {mape_filtered:.6f}")
	# print(f"Maximum Absolute Error (MaxAE): {maxae_filtered:.6f}")
	# print(f"R² Score: {r2_filtered:.6f}")
	# print("="*60 + "\n")

	# Save results to CSV
	results_df = test_df.copy()
	results_df = unnormalize_data(results_df, columns=["Horizontal_Distance", "Height", "UAV_Sending_Interval", "MCS"])
	results_df["Predicted_Reliability"] = y_pred
	results_df["Prediction_Error"] = y_test - y_pred
	results_df["Absolute_Error"] = np.abs(y_test - y_pred)

	results_csv = os.path.join(args.output_folder, "raw_results_" + args.link + ".csv")
	results_df.to_csv(results_csv, index=False)
	print(f"Predictions saved to {results_csv}")

	# Save evaluation metrics
	metrics_txt = os.path.join(args.output_folder, "evaluation_metrics_" + args.link + ".txt")
	with open(metrics_txt, "w", encoding="utf-8") as f:
		f.write("EVALUATION METRICS (OVER ALL DATA)\n")
		f.write("="*60 + "\n")
		f.write(f"Dataset: {args.data}\n")
		f.write(f"Model: {args.model}\n")
		f.write(f"Number of test samples: {len(x_test)}\n\n")
		f.write(f"Mean Squared Error (MSE):  {mse:.6f}\n")
		f.write(f"Root Mean Squared Error (RMSE): {rmse:.6f}\n")
		f.write(f"Mean Absolute Error (MAE): {mae:.6f}\n")
		f.write(f"Mean Absolute Percentage Error (MAPE): {mape:.6f}\n")
		f.write(f"Maximum Absolute Error (MaxAE): {maxae:.6f}\n")
		f.write(f"R² Score: {r2:.6f}\n")
		f.write("="*60 + "\n")
		# f.write("EVALUATION METRICS (RELIABILITY >= 90%)\n")
		# f.write("="*60 + "\n")
		# f.write(f"Dataset: {args.data}\n")
		# f.write(f"Model: {args.model}\n")
		# f.write(f"Number of test samples: {len(x_test_filtered)}\n\n")
		# f.write(f"Mean Squared Error (MSE):  {mse_filtered:.6f}\n")
		# f.write(f"Root Mean Squared Error (RMSE): {rmse_filtered:.6f}\n")
		# f.write(f"Mean Absolute Error (MAE): {mae_filtered:.6f}\n")
		# f.write(f"Mean Absolute Percentage Error (MAPE): {mape_filtered:.6f}\n")
		# f.write(f"Maximum Absolute Error (MaxAE): {maxae_filtered:.6f}\n")
		# f.write(f"R² Score: {r2_filtered:.6f}\n")
		# f.write("="*60 + "\n")

	print(f"Evaluation metrics saved to {metrics_txt}")


if __name__ == "__main__":
	main()
