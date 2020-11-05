# Machine Learning Nanodegree
# Capstone Project
## Exoplanet-Classifier

The question of the century perhaps, is there another blue dot of life out there in this humungous universe? Are we alone? 
Is there any other ball of mass which could sustain life like our Earth? 

In this project my main focus would be to use the time-series data of flux received by the satellite for pattern recognition and identify the stars with potential exoplanets revolving around them. 

Libraries, Packages and Technology used:
Python3, Pytorch, AWS Sagemaker, Pandas, Numpy, Sci-kitLearn, Matplotlib, Jupyter Notebook, SciPy

### Data from Kaggle

https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data

### Preprocessed data used for training

https://drive.google.com/drive/folders/1B4ReAKbgUe2cN8JHQ_icYf_xMAv7SatN?usp=sharing

### Proposal

This contains the description of how I planned to approach the problem just after exploring the data. The skewness of the dataset proves to be a major roadblock in training the model.

### Capstone Project

This is the tentative project report describing the architecture used, workflow pipeline, and the obstacles confronted in the approach mentioned in the proposal. There is a wide scope in improving the performance of the model, which I would dig into in the near future.

### 1_data_preprocessing

The notebook contains the implementation of the preprocessing techniques.

### 2_training_and_prediction

The notebook contains the training and inference code used for training the model through AWS Sagemaker Pytorch estimators, models and endpoints.

### source_pytorch

Folder contains the python scripts to be used for defining the model, training the estimator and inferencing the predictor once the endpoint is deployed.

### train_data

Folder containing the running loss and accuracy during training. It is extracted from the output.tar.gz as shown in 2_training_and_prediction.ipynb.


