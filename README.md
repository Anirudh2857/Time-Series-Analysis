# Time Series Analysis of Bitcoin Prices

This notebook performs time series analysis on historical Bitcoin price data. It covers data loading, preprocessing, visualization, model building, and evaluation.

## Table of Contents

1.  **Data Loading and Preprocessing**
    * Downloading the Bitcoin historical data.
    * Importing and parsing the data using pandas.
    * Cleaning and preparing the data for analysis.
    * Visualizing the Bitcoin price over time.
2.  **Train-Test Split**
    * Demonstrating the incorrect and correct ways to split time series data.
    * Creating train and test sets for model evaluation.
    * Visualizing the train and test splits.
3.  **Naive Forecasting**
    * Implementing a naive forecast baseline.
    * Evaluating the naive forecast using MAE, MSE, RMSE, MAPE, and MASE.
    * Visualizing the naive forecast against the actual data.
4.  **Windowing the Data**
    * Creating windowed datasets for time series prediction.
    * Implementing functions to create labelled windows and train-test splits.
5.  **Model Building and Evaluation**
    * Building and training various machine learning models:
        * Dense neural networks (model\_1, model\_2, model\_3, model\_6).
        * 1D Convolutional Neural Network (Conv1D) (model\_4).
        * Long Short-Term Memory (LSTM) network (model\_5).
        * N-Beats Model (model_7)
    * Evaluating model performance using various metrics.
    * Visualizing model predictions against actual data.
    * Comparing the performance of different models.
6.  **Multivariate Time Series Analysis**
    * Adding block reward data to the Bitcoin price dataset.
    * Preparing the multivariate dataset for model training.
    * Training and evaluating a dense neural network on the multivariate data.
7.  **N-Beats Implementation**
    * Creating a custom N-Beats block layer.
    * Building and training an N-Beats model.
    * Evaluating the N-Beats model performance.
8.  **Ensemble Models (Conceptual)**
    * Function to create ensemble models (Not fully executed in this notebook).

## Getting Started

1.  **Dependencies:**
    * pandas
    * matplotlib
    * numpy
    * scikit-learn (sklearn)
    * tensorflow
    * wget (for downloading the dataset)

2.  **Installation:**
    ```bash
    pip install pandas matplotlib numpy scikit-learn tensorflow
    ```

3.  **Running the Notebook:**
    * Open the `Time_Series_Analysis.ipynb` notebook in Google Colab or Jupyter Notebook.
    * Run the cells sequentially to execute the analysis.

4.  **Data Source:**
    * The Bitcoin historical price data is downloaded from the following GitHub repository:
        `https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv`

## Usage

* The notebook can be used as a template for time series analysis on other datasets.
* The model architectures and hyperparameters can be adjusted to improve performance.
* The evaluation metrics can be extended to include other relevant metrics.
* The ensemble model section can be expanded upon.

## Key Concepts

* Time series data.
* Train-test split for time series.
* Naive forecasting.
* Windowing time series data.
* Neural network models for time series prediction.
* Multivariate time series.
* N-Beats Model.
* Model evaluation metrics (MAE, MSE, RMSE, MAPE, MASE).

## Notes

* The notebook uses Google Colab for execution, but it can also be run locally with Jupyter Notebook.
* The model training times may vary depending on the hardware and the number of epochs.
* The N-Beats model takes a substantial amount of time to run.

## Author

Anirudh Jeevan
