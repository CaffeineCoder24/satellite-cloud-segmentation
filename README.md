# satellite-cloud-segmentation

Dataset URL: https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images

Cloud Segmentation using satellite images done and discussed using fast.ai approach and UNet+ResNet approach:

Fast.ai: 
The project is related to image segmentation and uses the fastai library for training a UNet model with a ResNet18 backbone.
Exploratory data analysis (EDA) is performed on the image dataset, including visualizing the class distribution and the number of labels per image. The code also includes functions for converting masks from/to run-length encoding (RLE) format. Next, I have defined the data pipeline for training the model. It loads the image and mask data, applies data augmentation, and creates a data bunch for training. The model is trained using the UNet architecture with a ResNet18 backbone and binary cross-entropy loss. The training process includes finding an optimal learning rate, fitting the model to the data, and saving the best model based on the dice metric. After the initial training, the code unfreezes the model and performs further fine-tuning with larger images. The process includes adjusting the learning rate and fitting the model to the data again. The best model is saved based on the dice metric.

UNet+ResNet:
In this project, image segmentation using a UNet model and ResNet34. The file includes importing necessary libraries, reading in data, preprocessing, modeling, training, and evaluation. The data is split into training and validation subsets, and a data loader is created for each subset. The UNet model is defined with a double convolution function and an expansive path. The ResNet34 model is modified to accept a custom number of channels. The model is trained using cross-entropy loss and Adam optimizer, and accuracy is used as a metric. The training and validation loss and accuracy are plotted over epochs. Finally, the model is evaluated using a batch of data, and the input, ground truth, and predicted masks are displayed.
