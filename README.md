3 files are contained:

* download_data.ipynb
* cub2rec.ipynb
* model_train.ipynb

Please run them with this sequence.

'download_data.ipynb' will download the dataset of CUB-200-2011 from the internet. This will create a folder named 'dataset' containing all the necessary training and testing images.

'cub2rec.ipynb' will first check wether mxnet is downloaded in current folder. If not, this will download the mxnet first.
Then this will generate the 'rec' file. The 'rec' file is a sequence of records which benefits from less storage size, continuous reading on disk and simplicity for partition. 
This will generate a folder named 'io_data' inside the 'dataset' folder which contains ‘rec’ file for training and testing with the image amount ratio of 9:1.

‘model_train_data.ipynb’ will do the model training. The current model is trained with the architecture from resnet18 and this may subject to change according to the performance in the following days.
With 55 epochs of training, the current accuracy is 95% in training and 70% in validation. Considering the class number of 200, this is not bad.
