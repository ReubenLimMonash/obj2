import argparse
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, max_error
from sklearn.preprocessing import StandardScaler

'''
Usage:
python journal_2_scripts/fanet_nn_regression_test_22122025.py \
  --data /path/to/test_dataset.csv \
  --model /path/to/final_model.h5 \
  --output_folder test_results \
  --output_file evaluation_metrics.csv

python journal_2_scripts/fanet_nn_regression_test_22122025.py \
  --data /media/research-student/DataDrive/FANET_Dataset/complete_testing_dmax_dataset/data_processed_complete/Downlink_Reliability.csv \
  --model /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts_regression/final_Downlink_regression.h5 \
  --output_folder /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/test_results_regression \
  --output_file downlink_evaluation_metrics.csv

python journal_2_scripts/fanet_nn_regression_test_22122025.py \
  --data /media/research-student/DataDrive/FANET_Dataset/complete_testing_dmax_dataset/data_processed_complete/Uplink_Reliability.csv \
  --model /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts_regression/final_Uplink_regression.h5 \
  --output_folder /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/test_results_regression \
  --output_file uplink_evaluation_metrics.csv

python journal_2_scripts/fanet_nn_regression_test_22122025.py \
  --data /media/research-student/DataDrive/FANET_Dataset/complete_testing_dmax_dataset/data_processed_complete/Video_Reliability.csv \
  --model /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts_regression/final_Video_regression.h5 \
  --output_folder /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/test_results_regression \
  --output_file video_evaluation_metrics.csv
'''


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

	return df

# Function to reverse normalization
def unnormalize_data(df_in, columns=[]):
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

def load_test_dataset(csv_path: str):
	'''
	Load and preprocess test dataset with the same transformations as training.
	'''
	df_dtypes = {
		"Horizontal_Distance": np.float32,
		"Height": np.int16,
		"UAV_Sending_Interval": np.float16,
		"Modulation": str,
		"Bitrate": np.float64,
		# "Num_Sent": np.int32,
		"Num_Reliable": np.int32,
		"Num_Delay_Excd": np.int32,
		"Num_Fail_Other": np.int32
	}
	dataset_df = pd.read_csv(
		csv_path,
		usecols=[
			"Horizontal_Distance", "Height", "UAV_Sending_Interval", "Modulation", "Bitrate",
			"Num_Reliable", "Num_Delay_Excd", "Num_Fail_Other"
		],
		dtype=df_dtypes
	)
	dataset_df = dataset_df.loc[dataset_df["Horizontal_Distance"] <= 600]  # Apply max horizontal distance
	dataset_df = get_mcs_index(dataset_df)
	dataset_df["Reliability"] = dataset_df["Num_Reliable"] / (dataset_df["Num_Reliable"] + dataset_df["Num_Delay_Excd"] + dataset_df["Num_Fail_Other"])
	dataset_df = normalize_data(dataset_df, columns=["Horizontal_Distance", "Height", "UAV_Sending_Interval", "MCS"])
	
	features = dataset_df[["Horizontal_Distance", "Height", "MCS", "UAV_Sending_Interval"]].to_numpy(dtype=np.float32)
	target = dataset_df["Reliability"].to_numpy(dtype=np.float32)
	
	return features, target, dataset_df

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Test NN regressor for reliability prediction")
	parser.add_argument("--data", required=True, help="Path to test dataset CSV")
	parser.add_argument("--model", required=True, help="Path to trained model (.h5 or .keras)")
	parser.add_argument("--output_folder", default="test_results", help="Directory to store test results")
	parser.add_argument("--output_file", default="evaluation_metrics.csv", help="File name to store test results")
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	# Create output directory
	os.makedirs(args.output_folder, exist_ok=True)

	# Load test dataset
	print(f"Loading test dataset from {args.data}...")
	x_test, y_test, test_df = load_test_dataset(args.data)
	print(f"Loaded {len(x_test)} test samples with {x_test.shape[1]} features")

	# Load trained model
	print(f"Loading model from {args.model}...")
	model = tf.keras.models.load_model(args.model)
	print("Model loaded successfully")

	# Make predictions
	print("Making predictions...")
	y_pred = model.predict(x_test, verbose=0)
	y_pred = y_pred.flatten()

	# Calculate metrics (over all data)
	mse = mean_squared_error(y_test, y_pred)
	rmse = np.sqrt(mse)
	mae = mean_absolute_error(y_test, y_pred)
	mape = mean_absolute_percentage_error(y_test, y_pred)
	maxae = max_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)

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
	print("="*60 + "\n")

	# Calculate metrics (for data where reliability is above 90%)
	filter_indices = y_test >= 0.9
	y_test_filtered = y_test[filter_indices]
	y_pred_filtered = y_pred[filter_indices]
	x_test_filtered = x_test[filter_indices]
	mse_filtered = mean_squared_error(y_test_filtered, y_pred_filtered)
	rmse_filtered = np.sqrt(mse_filtered)
	mae_filtered = mean_absolute_error(y_test_filtered, y_pred_filtered)
	mape_filtered = mean_absolute_percentage_error(y_test_filtered, y_pred_filtered)
	maxae_filtered = max_error(y_test_filtered, y_pred_filtered)
	r2_filtered = r2_score(y_test_filtered, y_pred_filtered)

	# Print results
	print("\n" + "="*60)
	print("EVALUATION METRICS (RELIABILITY >= 90%)")
	print("="*60)
	print(f"Mean Squared Error (MSE):  {mse_filtered:.6f}")
	print(f"Root Mean Squared Error (RMSE): {rmse_filtered:.6f}")
	print(f"Mean Absolute Error (MAE): {mae_filtered:.6f}")
	print(f"Mean Absolute Percentage Error (MAPE): {mape_filtered:.6f}")
	print(f"Maximum Absolute Error (MaxAE): {maxae_filtered:.6f}")
	print(f"R² Score: {r2_filtered:.6f}")
	print("="*60 + "\n")

	# Save results to CSV
	results_df = test_df.copy()
	results_df = unnormalize_data(results_df, columns=["Horizontal_Distance", "Height", "UAV_Sending_Interval", "MCS"])
	results_df["Predicted_Reliability"] = y_pred
	results_df["Prediction_Error"] = y_test - y_pred
	results_df["Absolute_Error"] = np.abs(y_test - y_pred)

	results_csv = os.path.join(args.output_folder, args.output_file)
	results_df.to_csv(results_csv, index=False)
	print(f"Predictions saved to {results_csv}")

	# Save evaluation metrics
	metrics_txt = os.path.join(args.output_folder, "evaluation_metrics_" + args.output_file)
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
		f.write("EVALUATION METRICS (RELIABILITY >= 90%)\n")
		f.write("="*60 + "\n")
		f.write(f"Dataset: {args.data}\n")
		f.write(f"Model: {args.model}\n")
		f.write(f"Number of test samples: {len(x_test_filtered)}\n\n")
		f.write(f"Mean Squared Error (MSE):  {mse_filtered:.6f}\n")
		f.write(f"Root Mean Squared Error (RMSE): {rmse_filtered:.6f}\n")
		f.write(f"Mean Absolute Error (MAE): {mae_filtered:.6f}\n")
		f.write(f"Mean Absolute Percentage Error (MAPE): {mape_filtered:.6f}\n")
		f.write(f"Maximum Absolute Error (MaxAE): {maxae_filtered:.6f}\n")
		f.write(f"R² Score: {r2_filtered:.6f}\n")
		f.write("="*60 + "\n")

	print(f"Evaluation metrics saved to {metrics_txt}")


if __name__ == "__main__":
	main()
