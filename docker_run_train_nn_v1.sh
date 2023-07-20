docker run --runtime=nvidia --gpus all -it --rm \
    -v "/media/research-student/One Touch/FANET Datasets/Dataset_NP10000_BPSK_6-5Mbps":"/tf/FANET Datasets/Dataset_NP10000_BPSK_6-5Mbps" \
    -v "/home/research-student/omnet-fanet/nn_checkpoints/nn_shared_hl-15032023":"/tf/nn_checkpoints" \
    -v "/home/research-student/omnet-fanet/data-processing-scripts/fanet_nn_train_v1.py":"/tf/fanet_nn_train_v1.py" \
    -w "/tf" \
    tensorflow/tensorflow:latest-gpu-jupyter-updated \
    python fanet_nn_train_v1.py