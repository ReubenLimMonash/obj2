'''
Date Modified: 19/07/2023
Desc: To train a NN classifier to predict FANET reliability and failure modes
Modified: To consolidate the different versions, training modes, models and dataset types into one script
'''

import pandas as pd
import numpy as np 
import math
import os
import pickle
import gc 
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split

# Keras specific
import keras
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Activation, Dropout
from keras.utils import to_categorical 
from keras.callbacks import Callback
from keras import initializers, regularizers, optimizers, backend

# Ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
'''
DataGenerator takes dataset_details_df of the following form
Horizontal_Distance,    Height, UAV_Sending_Interval,   Modulation, Mean_SINR,          Std_Dev_SINR,       Num_Sent,   Num_Reliable,   Num_Delay_Excd, Num_Incr_Rcvd,  Num_Q_Overflow
0.0,                    60,     66.7,                   QAM16,      1122.743643457063,  465.2159856885714,  70768,      70768,          0,              0,              0
Each packet in columns [Num_Sent,   Num_Reliable,   Num_Delay_Excd, Num_Incr_Rcvd,  Num_Q_Overflow] are treated as having a sequential index,
meaning in the first row example above, packet index 0 to 70767 are reliably received, with input data from the first row.
DataGenerator will create an index for each packet details in dataset_details_df and feeds them to the training pipeline, with/without shuffling.
'''
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_details_df, test_split=0.2, type='train', batch_size=32, input_dim=4, n_classes=4, shuffle=None):
        'Initialization'
        assert ((type == "train") or (type == "Train") or (type == "test") or (type == "Test")), "Type needs to be either 'train'/'test'"
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.dataset_details_df = dataset_details_df
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.type = type
        self.dataset_len = self.dataset_details_df["Num_Sent"].sum()
        self.cumulative_index = self.dataset_details_df["Num_Sent"].cumsum(axis=0).values # To help with finding the df row based on packet index
        self.indexes = np.array([]) # Initialize empty array, make sure this is done before calling self.on_epoch_end()
        if self.type == 'train':
            self.test_split = test_split
        elif self.type == 'test':
            self.test_split = 1 - test_split

        # FOR DEBUGGING
        # self.count = 0
        # self.X_rec = np.array([0,0,0,0]).reshape(-1,4)
        # self.y_rec = np.array([0,0,0,0]).reshape(-1,4)
        # if self.type == "train":
        #     df = pd.DataFrame(columns=["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Num_Reliable", "Num_Q_Overflow", "Num_Incr_Rcvd", "Num_Delay_Excd"])
        #     df.to_csv("/home/research-student/omnetpp_sim_results/PCAP_Test/ParrotAR2_ConstantSI/test_recon_training.csv")

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.indexes.size > 0:
            return int(np.ceil(len(self.indexes) / self.batch_size))
        else:
            if ((self.type == "train") or (self.type == "Train")):
                return int(np.ceil(self.dataset_len * (1 - self.test_split) / self.batch_size))
            elif ((self.type == "test") or (self.type == "Test")):
                return int(np.ceil(self.dataset_len * self.test_split / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        # FOR DEBUGGING
        # self.count += 1
        # print(self.count)
        # print(X.shape)
        # if self.type == "train":
        #     # self.X_rec = np.append(self.X_rec, X).reshape(-1,4)
        #     # self.y_rec = np.append(self.y_rec, y).reshape(-1,4)
        #     df = pd.DataFrame(np.hstack((X,y)))
        #     df.to_csv("/home/research-student/omnetpp_sim_results/PCAP_Test/ParrotAR2_ConstantSI/test_recon_training.csv", mode='a', header=False)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == 'all':
            if self.test_split == 0 or self.test_split == 1: # Don't split the dataset
                self.indexes = np.arange(self.dataset_len)
                np.random.shuffle(self.indexes)
            else:
                self.indexes, _ = train_test_split(np.arange(self.dataset_len), test_size=self.test_split, random_state=0, shuffle=True)
            
        elif self.shuffle == 'row':
            # Shuffle samples within each dataset row only
            tmp_indexes = np.arange(self.cumulative_index[0])
            if self.test_split == 0 or self.test_split == 1: # Don't split the dataset
                np.random.shuffle(tmp_indexes)
            else:
                tmp_indexes, _ = train_test_split(tmp_indexes, test_size=self.test_split, random_state=0, shuffle=True)
            self.indexes = tmp_indexes
            for i in range(1, len(self.cumulative_index)):
                tmp_indexes = np.arange(self.cumulative_index[i-1], self.cumulative_index[i])
                if self.test_split == 0 or self.test_split == 1: # Don't split the dataset
                    np.random.shuffle(tmp_indexes)
                else:
                    tmp_indexes, _ = train_test_split(tmp_indexes, test_size=self.test_split, random_state=0, shuffle=True)
                self.indexes = np.append(self.indexes, tmp_indexes)
        else:
            # Defaults to no shuffle
            if self.test_split == 0 or self.test_split == 1: # Don't split the dataset
                self.indexes = np.arange(self.dataset_len)
            else:
                self.indexes, _ = train_test_split(np.arange(self.dataset_len), test_size=self.test_split, random_state=0, shuffle=False)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(indexes), self.input_dim))
        y = np.empty((len(indexes)), dtype=int)
        # Generate data
        for i, index in enumerate(indexes):
            # NOTE: Index in indexes start at 0, bear this is mind when comparing index positions
            row_index = np.searchsorted(self.cumulative_index, index, side='right') # The df row this packet index points to (specifying the scenario)
            # Make sure to sort the columns in df_row based on the categorical order encoding for packet state: {"Reliable":0, "QUEUE_OVERFLOW":1, "RETRY_LIMIT_REACHED":2, "Delay_Exceeded":3}
            df_row = self.dataset_details_df.loc[row_index, ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", 
                                                             "Num_Reliable", "Num_Q_Overflow", "Num_Incr_Rcvd", "Num_Delay_Excd"]].values
            if row_index == 0:
                packet_state_index = index
            else:
                packet_state_index = index - self.cumulative_index[row_index-1]

            if packet_state_index < df_row[4]:
                # Case of reliable packet
                packet_state = 0
            elif packet_state_index < df_row[4] + df_row[5]:
                # Case of queue overflow packet
                packet_state = 1
            elif packet_state_index < df_row[4] + df_row[5] + df_row[6]:
                # Case of incr rcvd packet
                packet_state = 2
            else:
                # Case of delay excd packet
                packet_state = 3

            # Store sample
            X[i,] = df_row[0:4]
            # Store class
            y[i] = packet_state
        
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

def normalize_data(df, columns=[], save_details_path=None):
    '''
    columns: The pandas data columns to normalize, given as a list of column names
    '''
    # Define the ranges of parametrers
    max_mean_sinr = 10*math.log10(1123) # The max mean SINR calculated at (0,60) is 1122.743643457063 (linear)
    max_std_dev_sinr = 10*math.log10(466) # The max std dev SINR calculated at (0,60) is 465.2159856885714 (linear)
    min_mean_sinr = 10*math.log10(0.2) # The min mean SINR calculated at (1200,60) is 0.2251212887895188 (linear)
    min_std_dev_sinr = 10*math.log10(0.7) # The min std dev SINR calculated at (1200,300) is 0.7160093126585219 (linear)
    max_height = 300
    min_height = 60
    max_h_dist = 1200
    min_h_dist = 0

    # Normalize data (Min Max Normalization between [-1,1])
    if "Height" in columns:
        df["Height"] = df["Height"].apply(lambda x: 2*(x-min_height)/(max_height-min_height) - 1)
    if "U2G_H_Dist" in columns:
        df["U2G_H_Dist"] = df["U2G_H_Dist"].apply(lambda x: 2*(x-min_h_dist)/(max_h_dist-min_h_dist) - 1)
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
        f.write("UAV Sending Interval: [10:-1, 20:-0.5, 40:0, 100:0.5, 1000:1]\n")
        f.write("Output: ['Reliable':0, 'QUEUE_OVERFLOW':1, 'RETRY_LIMIT_REACHED':2, 'Delay_Exceeded':3]\n")
        f.close()

    return df

def build_nn_model_v4():
    # For multiple output model
    # Version 4: Having only a single output layer for packet state
    inputs = Input(shape=(4,))
    base = Dense(100, activation='relu')(inputs)
    base = BatchNormalization()(base)
    base = Dense(50, activation='relu')(base)
    base = BatchNormalization()(base)
    base = Dense(25, activation='relu')(base)
    base = BatchNormalization()(base)
    base = Dense(10, activation='relu')(base)
    base = BatchNormalization()(base)
    packet_state_out = Dense(4, activation='softmax', name='packet_state')(base)
    model = Model(inputs=inputs, outputs = packet_state_out)
    return model

def build_nn_model_v5(l2_reg=0.001):
    # For multiple output model
    # Version 5: Added regularizations 
    init_weight = initializers.GlorotUniform(seed=0)
    regu_weight = regularizers.l2(l2_reg)

    inputs = Input(shape=(4,))
    base = Dense(100, kernel_initializer=init_weight, kernel_regularizer=regu_weight)(inputs)
    base = BatchNormalization()(base)
    base = Activation('relu')(base)
    base = Dropout(0.2)(base)
    base = Dense(50, kernel_initializer=init_weight, kernel_regularizer=regu_weight)(base)
    base = BatchNormalization()(base)
    base = Activation('relu')(base)
    base = Dropout(0.2)(base)
    base = Dense(25, kernel_initializer=init_weight, kernel_regularizer=regu_weight)(base)
    base = BatchNormalization()(base)
    base = Activation('relu')(base)
    base = Dropout(0.2)(base)
    base = Dense(10, kernel_initializer=init_weight, kernel_regularizer=regu_weight)(base)
    base = BatchNormalization()(base)
    base = Activation('relu')(base)
    base = Dropout(0.2)(base)
    packet_state_out = Dense(4, activation='softmax', name='packet_state')(base)
    model = Model(inputs=inputs, outputs = packet_state_out)
    return model

# Custom callback to clear memory to reduce RAM usage after each epoch
class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        keras.backend.clear_session()

if __name__ == "__main__":
    # Training params
    EPOCHS = 10 # 5
    BATCHSIZE = 100
    WORKER = 20 # No. of CPU for generator workers
    LR = 0.001 # Adam Learning Rate
    L2_REG = 0.001 # L2 Norm Weight Regularization
    # CHECKPOINT_FILEPATH = '/home/research-student/omnetpp_sim_results/PCAP_Test/DJISpark_ConstantSI/dl_nn_ckpt2'
    CHECKPOINT_FILEPATH = '/home/research-student/omnetpp_sim_results/PCAP_Test/ParrotAR2_ConstantSI'
    # DATASET_PATH = "/home/research-student/omnetpp_sim_results/PCAP_Test/DJISpark_ConstantSI/DJISpark_ConstantSI_Downlink_Reliability.csv"
    DATASET_PATH = "/home/research-student/omnetpp_sim_results/PCAP_Test/ParrotAR2_ConstantSI/test2.csv"

    # Load dataset =================================================
    df_dtypes = {"Horizontal_Distance": np.float64, "Height": np.int16,	"U2G_Distance": np.int32, "UAV_Sending_Interval": np.float64, "Mean_SINR": np.float64, "Std_Dev_SINR": np.float64,
                 "Num_Sent": np.int32, "Num_Reliable": np.int32, "Num_Delay_Excd": np.int32, "Num_Incr_Rcvd": np.int32, "Num_Q_Overflow": np.int32}
    dataset_details = pd.read_csv(DATASET_PATH, 
                                usecols = ["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation", "Num_Sent", "Num_Reliable", "Num_Delay_Excd",
                                            "Num_Incr_Rcvd", "Num_Q_Overflow"],
                                dtype=df_dtypes)
    dataset_details = normalize_data(dataset_details, columns=["Mean_SINR", "Std_Dev_SINR", "UAV_Sending_Interval", "Modulation"], save_details_path=None)                         
    train_data_generator = DataGenerator(dataset_details, test_split=0.2, type='train', batch_size=BATCHSIZE, shuffle='row')
    val_data_generator = DataGenerator(dataset_details, test_split=0.2, type='test', batch_size=BATCHSIZE, shuffle='row')

    # Build model
    # model = build_nn_model_v5(l2_reg=L2_REG)
    model = build_nn_model_v4()

    # Load pre-trained model for finetuning
    # model = keras.models.load_model(os.path.join(CHECKPOINT_FILEPATH, "model.004-0.2158.h5"), compile=False)

    # Compile the model
    optmz = optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optmz, 
                loss={'packet_state': 'categorical_crossentropy'},
                metrics={'packet_state': 'accuracy'})

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_FILEPATH,"model.{epoch:03d}-{val_loss:.4f}.h5"),
        save_weights_only=False,
        monitor='val_loss',
        mode='auto',
        save_freq='epoch')

    date = datetime.now()
    date_str = date.strftime("%d%m%Y")
    # history = model.fit(X_train, packet_state_train, epochs=EPOCHS, callbacks=[model_checkpoint_callback, ClearMemory()], validation_data=(X_test, packet_state_test))
    history = model.fit(train_data_generator, 
                        epochs=EPOCHS,
                        # callbacks=[model_checkpoint_callback, ClearMemory()],
                        callbacks=[model_checkpoint_callback],
                        validation_data=val_data_generator,
                        steps_per_epoch=train_data_generator.__len__(),
                        workers=WORKER,
                        # workers=1,
                        # verbose=0,
                        use_multiprocessing=True)
    with open(os.path.join(CHECKPOINT_FILEPATH, 'trainHistoryDict_{}'.format(date_str)), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # Save final model
    model.save(os.path.join(CHECKPOINT_FILEPATH,"final_model.h5"))