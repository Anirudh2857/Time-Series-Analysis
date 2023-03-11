This project aims to analyze the Bitcoin price data from 2013 to 2021 using various time series models. The models used in this project are naive model, model_1_dense_w7_h1, model_2_dense_w30_h1, model_3_dense_w30_h7, model_4_CONV1D, model_5_LSTM, model_6_multivariate, model_8_NBEATs, model_9_ensemble, and model_10_turkey. The output received from these models are mae, mse, rmse, mape, and mase.

**Models Used**

Naive Model
This model uses the previous day's price as the prediction for the next day.

Model_1_dense_w7_h1
This model uses a dense neural network with a window size of 7 and a horizon of 1.

Model_2_dense_w30_h1
This model uses a dense neural network with a window size of 30 and a horizon of 1.

Model_3_dense_w30_h7
This model uses a dense neural network with a window size of 30 and a horizon of 7.

Model_4_CONV1D
This model uses a 1D convolutional neural network.

Model_5_LSTM
This model uses a Long Short-Term Memory (LSTM) neural network.

Model_6_multivariate
This model uses a multivariate neural network with multiple features.

Model_8_NBEATs
This model uses the Neural basis expansion analysis for interpretable time series forecasting (N-BEATS) architecture.

Model_9_ensemble
This model is an ensemble of the above models.

Model_10_turkey
This model uses the turkey method for detecting outliers and is used as a benchmark.

**Ouput of the models are** 

![image](https://user-images.githubusercontent.com/55728354/223864255-6844e1fe-95f0-4c34-b533-eb082493e91d.png)


The turkey models giving the best MAE of any of the models 
