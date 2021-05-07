## RNNs and GNNs for Feature Extraction in Image Classification

We explore various methods of using RNNs and GNNs to improve the performance of a simple convolutional architecture (AlexNet).

Each file is self-contained, meaning it can be run on its own. The relevant models are
* `cnn.py` - File with base AlexNet inspired model along with various image preprocessing methods.
* `rnn_last.py` - Uses a quad-RNN in file layer to extract additional features from convolutional output.

Additional modules are described below:
* `hypercolumns.py` - Extracts the hypercolumns of a given convolutional output.
