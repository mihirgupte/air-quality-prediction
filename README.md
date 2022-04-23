# Air Quality Prediction

  

In this project, we try to create a model for Air Quality Prediction using regression. The dataset can be found <a  href="https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india">here</a>. Based on the various parameters given to us, such as O3, NO2, Benzene, etc we implement a model to give us accurate predictions.

  

## What this repository contains

1.  `aqi_prediction.ipynb` - This jupyter notebook contains the EDA + preprocessing + model building part of the project. Here you can check the steps taken to arrive at the conclusions and the final pipeline of the model.

2.  `prediction.py` - A basic script which takes into account all the hyperparameters of the final model and serves as a basic script with a predict function for direct usage.

3.  `Regression model.pkl` - Contains the trained regression model which can be fit directly for predictions.

4.  `scaler.pkl` - Contains Standard Scaler which can be fit to scale the input data used for modelling.

  

## Results of the model

  

The final results given by the best model are as follows -

```py
train mse: 0.08160195893731888
train rmse: 0.2856605659472775
train r2: 0.8176628312441719

  

test mse: 0.08681634870905088
test rmse: 0.2946461415139368
test r2: 0.80849843120208
```