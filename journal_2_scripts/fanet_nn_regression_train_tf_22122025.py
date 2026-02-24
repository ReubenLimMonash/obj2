import argparse
import os
import random
from datetime import datetime
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
Usage:
python journal_2_scripts/fanet_nn_regression_train.py \
  --data /path/to/dataset.csv \
  --output_file Downlink_regression.h5 \
  --output_folder artifacts \
  --epochs 80 \
  --batch-size 128
tensorboard --logdir artifacts/logs_*

python journal_2_scripts/fanet_nn_regression_train.py \
  --data /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/data_processed/Downlink_Reliability.csv \
  --output_file Downlink_regression.h5 \
  --output_folder /media/research-student/KingstonSSD/FANET_Dataset/Dataset_NP100000_DJISpark/nn_ckpts_regression \
  --epochs 80 \
  --batch-size 128
'''

def set_seed(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)

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
	df_dtypes = {"Horizontal_Distance": np.float32, "Height": np.int16, "UAV_Sending_Interval": np.float16, "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
				"Num_Sent": np.int32, "Num_Reliable": np.int32, "Num_Delay_Excd": np.int32, "Num_Incr_Rcvd": np.int32, "Num_Q_Overflow": np.int32, "Modulation": str, "Bitrate": np.float64}
	dataset_df = pd.read_csv(csv_path, 
				usecols = ["Horizontal_Distance", "Height", "UAV_Sending_Interval", "Modulation", "Bitrate", 
							"Num_Sent", "Num_Reliable", "Num_Delay_Excd", "Num_Incr_Rcvd", "Num_Q_Overflow"],
				dtype=df_dtypes)
	dataset_df = dataset_df.loc[dataset_df["Horizontal_Distance"] <= 600]  # Apply max horizontal distance of 600 m
	dataset_df = get_mcs_index(dataset_df)
	dataset_df["Reliability"] = dataset_df["Num_Reliable"] / dataset_df["Num_Sent"]
	dataset_df = normalize_data(dataset_df, columns=["Horizontal_Distance", "Height", "UAV_Sending_Interval", "MCS"])
	features = dataset_df[["Horizontal_Distance", "Height", "MCS", "UAV_Sending_Interval"]].to_numpy(dtype=np.float32)
	target = dataset_df["Reliability"].to_numpy(dtype=np.float32)
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

def build_model(input_dim: int) -> tf.keras.Model:
	model = tf.keras.Sequential(
		[
			tf.keras.layers.Input(shape=(input_dim,)),
			tf.keras.layers.Dense(100, activation="relu"),
			tf.keras.layers.Dense(50, activation="relu"),
			tf.keras.layers.Dense(25, activation="relu"),
			tf.keras.layers.Dense(10, activation="relu"),
			tf.keras.layers.Dense(1, activation="linear"),
		]
	)
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae", "mse"])
	return model


# def prepare_data(features: np.ndarray, target: np.ndarray, test_size: float, seed: int):
# 	x_train, x_val, y_train, y_val = train_test_split(features, target, test_size=test_size, random_state=seed)
# 	scaler = StandardScaler()
# 	x_train_scaled = scaler.fit_transform(x_train)
# 	x_val_scaled = scaler.transform(x_val)
# 	return x_train_scaled, x_val_scaled, y_train, y_val, scaler


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

	os.makedirs(args.output_folder, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_dir = os.path.join(args.output_folder, f"logs_{timestamp}")
	os.makedirs(log_dir, exist_ok=True)

	features, target = load_dataset(args.data)
	x_train, x_val, y_train, y_val = train_test_split(features, target, test_size=args.val_size, random_state=args.seed)

	model = build_model(input_dim=x_train.shape[1])

	callbacks = [
		tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
		tf.keras.callbacks.ModelCheckpoint(os.path.join(args.output_folder, args.output_file), save_best_only=True, monitor="val_loss"),
		tf.keras.callbacks.CSVLogger(os.path.join(log_dir, "training_log.csv")),
		tf.keras.callbacks.TensorBoard(log_dir=log_dir),
	]

	history = model.fit(
		x_train,
		y_train,
		validation_data=(x_val, y_val),
		epochs=args.epochs,
		batch_size=args.batch_size,
		callbacks=callbacks,
		verbose=2,
	)

	# Save the final model in args.output_file but with prefix final
	model.save(os.path.join(args.output_folder, "final_" + args.output_file))

	# Save final metrics for quick inspection.
	final_val_loss = history.history["val_loss"][np.argmin(history.history["val_loss"])]
	final_val_mae = history.history["val_mae"][np.argmin(history.history["val_loss"])]
	summary_path = os.path.join(args.output_folder, "metrics_summary.txt")
	with open(summary_path, "w", encoding="utf-8") as f:
		f.write(f"Best val_loss: {final_val_loss:.6f}\n")
		f.write(f"Best val_mae: {final_val_mae:.6f}\n")
		f.write(f"Logs: {log_dir}\n")


if __name__ == "__main__":
	main()
