# Cloud Segmentation Using Satellite Images

This project focuses on cloud segmentation in satellite images and includes implementations using the fastai library for a UNet model with a ResNet18 backbone and a UNet model with a ResNet34 backbone.

---

## Dataset
- Dataset URL: [38-Cloud: Cloud Segmentation in Satellite Images](https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images)

---

## Fast.ai Approach
- The project uses the fastai library for training a UNet model with a ResNet18 backbone.
- Exploratory Data Analysis (EDA) is performed on the image dataset, including visualizing the class distribution and the number of labels per image.
- Functions for converting masks from/to run-length encoding (RLE) format are included.
- The data pipeline for training the model is defined, including data loading, augmentation, and creating a data bunch for training.
- The model is trained with binary cross-entropy loss, and the training process includes finding an optimal learning rate and saving the best model based on the dice metric.
- After initial training, the model is fine-tuned with larger images, adjusting the learning rate, and saving the best model.

---

## UNet+ResNet Approach
- This project uses a UNet model with a ResNet34 backbone for cloud segmentation.
- The code includes importing necessary libraries, data reading, preprocessing, modeling, training, and evaluation.
- Data is split into training and validation subsets, and data loaders are created for each subset.
- The UNet model is defined with a double convolution function and an expansive path, and ResNet34 is modified to accept a custom number of channels.
- Training is performed using cross-entropy loss and Adam optimizer, with accuracy as a metric.
- Training and validation loss and accuracy are plotted over epochs, and the model is evaluated using a batch of data.

---

## Usage
1. Download the dataset from the provided URL and extract it to a suitable location.
2. Install the required dependencies listed in `requirements.txt`.
3. Run the Jupyter notebooks or Python scripts provided in the repository to train and evaluate the models.
4. Optionally, modify the hyperparameters, model architectures, or data augmentation techniques to experiment with different configurations.

---

## Dependencies
- Python 3.x
- fastai
- PyTorch
- Pandas
- Matplotlib
- NumPy
- scikit-learn
- tqdm

---

