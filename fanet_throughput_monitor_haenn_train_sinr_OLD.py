'''
Date: 27/06/2023
Desc: To train an NN classifier taking input from latent space of auto-encoder, for a hierarchical model
Modified: To load the train dataset from "<modulation>_processed_train_uplink.csv" and the test dataset from "<modulation>_processed_holdout_uplink.csv"
'''

import pandas as pd
import numpy as np 
import sklearn
import os
import math
import pickle
import gc
from datetime import datetime

# Keras specific
import keras
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization
from keras.utils import to_categorical 

# Training params
EPOCHS = 5
checkpoint_filepath = '/home/research-student/omnet-fanet/nn_checkpoints/throughput_haenn_multimodulation_novideo_sinr_ul'

df_dtypes = {"TxTime": np.float32, "U2G_Distance": np.float32, "Height": np.int16,	"Num_Members": np.int16, "UAV_Sending_Interval": np.int16, "Bytes": np.int16, 
            "U2G_SINR": np.float32, "U2G_BER": np.float32, "Delay": np.float32, "Throughput": np.float32, "Queueing_Time": np.float32, "Packet_State": 'category', 
            "Retry_Count": np.int8, "Incorrectly_Received": np.int8, "Queue_Overflow": np.int8, "Packet_Name": 'string', "Mean_SINR": np.float32, "Std_Dev_SINR": np.float32,
            "UAV_Sending_Interval": np.int16}

# Load training dataset ==========================================================================================================================
latent_features_file = "/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_MultiModulation_Hovering_NoVideo/throughput_ae_features_novideo_sinr_ul.npy"
latent_features = np.load(latent_features_file)
normal_out = np.ones((latent_features.shape[0], 1))
# Train model ==========================================================================================
inputs = Input(shape=(50,))
base = Dense(25, activation='relu')(inputs)
base = BatchNormalization()(base)
base = Dense(10, activation='relu')(base)
base = BatchNormalization()(base)
output = Dense(1, activation='linear', name='indicator')(base)
model = Model(inputs=inputs, outputs = output)

# Compile the model
model.compile(optimizer='adam', 
              loss={'indicator': 'mae'},
              metrics={'indicator': 'accuracy'})

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_filepath,"model.{epoch:03d}-{val_loss:.4f}.h5"),
    save_weights_only=False,
    monitor='val_loss',
    mode='auto',
    save_freq='epoch')

date = datetime.now()
date_str = date.strftime("%d%m%Y")

history = model.fit(latent_features, normal_out, epochs=EPOCHS, callbacks=[model_checkpoint_callback], validation_data=(latent_features, normal_out))
with open(os.path.join(checkpoint_filepath, 'trainHistoryDict_{}'.format(date_str)), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Save final model
model.save(os.path.join(checkpoint_filepath,"final_model.h5"))
