## RNNs and GNNs for Feature Extraction in Image Classification

We explore various methods of using RNNs and GNNs to improve the performance of a simple convolutional architecture (AlexNet).

Each file is self-contained, meaning it can be run on its own. The relevant models are
* `cnn.py` - File with base AlexNet inspired model along with various image preprocessing methods.
* `rnn_last.py` - Uses a quad-RNN in file layer to extract additional features from convolutional output.
* `rnn_all.py` - Extracts features from every convolutional layer using the quad-RNN construction. Randomly samples a direction and RNN to get features from to reduce model training time
* `gnn.py` - Runs a GNN layer on top of the AlexNet backbone. Includes options for Gaussian filter, Gabor filter, and predicted edges

Additional modules are described below:
* `hypercolumns.py` - Extracts the hypercolumns of a given convolutional output.
