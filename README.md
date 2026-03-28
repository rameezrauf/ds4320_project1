# DS 4320 Project 1 - Predicting the next day movement in SPY 

### Executive Summary

Rameez Rauf

xqd7aq

DOI

[Link to Press Release](https://github.com/rameezrauf/ds4320_project1/blob/main/Docuements/press_release.md)

[Data](https://myuva-my.sharepoint.com/:f:/g/personal/xqd7aq_virginia_edu/IgARDf8LGEVVQ4r2StjgYvSoAWUb1tTRrgrdPxopX4r_vnQ?e=tjWGYD)

Pipeline files link

License 

## Problem Definition 

**Initial general problem**

Forecasting stock prices using historical financial data.

**Refined specific problem**

Predict whether the S&P 500 ETF (SPY) will increase or decrease in price on the next trading day using historical market data such as past returns, trading volume, and technical indicators derived from daily stock prices.

**Rationale for refinement**

The general problem of forecasting stock prices is extremely broad and difficult because markets are influenced by many unpredictable factors such as macroeconomic events, investor sentiment, and global news. Refining the problem to predicting next-day price direction for a specific index-based ETF (SPY) makes the problem more measurable and feasible. Instead of attempting to predict exact prices, which is unrealistic for a short class project, the refined problem focuses on a binary outcome (up or down), which can be modeled using standard classification techniques and evaluated clearly using accuracy metrics.

**Motivation**

Financial markets generate enormous amounts of publicly available data, making them a natural domain for data science and machine learning. Investors, analysts, and institutions constantly seek ways to better understand price movements and identify patterns in historical data. By analyzing past market behavior, it may be possible to detect signals that provide insights into short-term price movements. This project aims to demonstrate how historical financial data can be used to build predictive models that attempt to identify these patterns, while also highlighting the challenges and limitations of forecasting financial markets.

**Press release** 

[Link to Press Release](https://github.com/rameezrauf/ds4320_project1/blob/main/Docuements/press_release.md)

## Domain Exposition

**Terminology**

|Term | Definition |
| --- | --- |
| Stock price | The market price of a share of a publicly traded company or ETF at a given time.
| ETF (Exchange-Traded Fund) | A security that tracks an index, sector, or asset and trades on an exchange like a stock |
| SPY | The SPDR S&P 500 ETF, which tracks the performance of the S&P 500 index and is commonly used as a proxy for the overall U.S. stock market |
| Return | The percentage change in price over a specific period of time|
| Volatility | A measure of how much the price of a financial asset fluctuates over time |
| Trading volume | The number of shares traded during a given time period |
| Moving average | A technical indicator that smooths price data by averaging prices over a specific time window |
| Feature engineering | The process of creating new variables from raw data to improve model performance |
| Classification model | A predictive model that assigns observations into categories, such as predicting whether the market will go up or down|
| Market efficiency | The idea that asset prices reflect all available information, making prediction difficult |

**Background readings**

[Link to OneDrive Folder with readings](https://myuva-my.sharepoint.com/:f:/g/personal/xqd7aq_virginia_edu/IgCJNvNvYyOPR6IJ-EdxRcdeAb5kZyM_DypnSBBv8K_-2bI?e=0xRSfw)

**Reading Summary Table** 

| Title | Description | Link |
| --- | --- | --- |
| Efficient Capital Markets | Foundational paper explaining the efficient market hypothesis and challenges of predicting stock prices |https://www.jstor.org/stable/2325486 |
|Technical Analysis Basics | Overview of common indicators used by traders such as moving averages and price trends | https://www.investopedia.com/terms/t/technicalanalysis.asp |
| Machine Learning for Trading | Explains how machine learning techniques can be applied to financial market prediction | https://github.com/stefan-jansen/machine-learning-for-trading |
|  Understanding Moving Averages (Investopedia) | Provides intuition behind features like moving averages that are used in this project. | https://www.investopedia.com/terms/m/movingaverage.asp|
| Alpha Vantage API Documentation | Documentation for retrieving financial market data through an API | https://www.alphavantage.co/documentation/ |

## Data Creation 

**Data Acquisition** 

The dataset used in this project was obtained from Yahoo Finance using the yfinance Python library. Historical daily price data for the S&P 500 ETF (SPY) was downloaded programmatically, including variables such as open price, close price, high, low, and trading volume. The data was retrieved for a specified time range to ensure a sufficiently large sample for analysis.

Because the data was accessed through an API rather than manually collected, the acquisition process is reproducible and consistent. The main transformation applied after retrieval was the calculation of derived features such as daily returns, which measure the percentage change in price from one day to the next. This structured dataset serves as the foundation for analyzing patterns in market behavior and predicting next-day price direction.

**Code**

| File Name | Description | Link |
| --- | --- | --- |
| data_download.py | Uses yfinance to download historical SPY price data from Yahoo Finance | https://github.com/rameezrauf/ds4320_project1/blob/main/Pipeline/data_download.py |
| feature_engineering.py | Computes derived features such as daily returns and lagged variables for prediction | https://github.com/rameezrauf/ds4320_project1/blob/main/Pipeline/feature_engineering.py |

**Bias Identification**

Bias can be introduced into this dataset through the selection of SPY as the sole asset being analyzed, which may not represent the behavior of all stocks or financial markets. Additionally, the dataset reflects historical market conditions, meaning it may be influenced by specific time periods such as bull markets, crashes, or economic shocks, which can bias model predictions. Another source of bias is survivorship bias, as SPY represents a continuously updated index of successful companies, excluding firms that have failed or been removed from the index.

**Bias Mitigation**

Bias can be mitigated by expanding the dataset to include multiple assets, sectors, or indices to improve representativeness. Additionally, splitting the data into training and testing periods helps reduce overfitting to specific market conditions and ensures that models generalize better to unseen data. Techniques such as rolling window validation and evaluating performance across different time periods can also help account for temporal bias and changing market dynamics.

**Rationale**

Several key decisions were made to balance simplicity and analytical value in this project. First, SPY was chosen because it provides a broad representation of the U.S. stock market while offering high-quality and easily accessible data. Second, daily price data was used instead of higher-frequency data to reduce noise and simplify analysis. Finally, derived features such as daily returns were included to better capture meaningful patterns in price movement, as raw prices alone are less informative for predictive modeling. These decisions help ensure that the dataset is both manageable and suitable for building and evaluating forecasting models. While the dataset focuses on a single asset (SPY), which limits generalizability, it provides a clean and high-quality starting point for modeling financial time series and can be extended to multiple assets in future work.

## Metadata

**Schema**

|         SPY Feature Record           |
|--------------------------------------|
| PK: Date                             |
| Open                                 |
| High                                 |
| Low                                  |
| Close                                |
| Volume                               |
| Adj Close                            |
| daily_return                         |
| lag_return_1                         |
| lag_return_2                         |
| lag_return_3                         |
| lag_return_5                         |
| moving_avg_5                         |
| moving_avg_10                        |
| moving_avg_20                        |
| moving_avg_50                        |
| close_to_ma_5                        |
| close_to_ma_10                       |
| close_to_ma_20                       |
| close_to_ma_50                       |
| volatility_5                         |
| volatility_10                        |
| volatility_20                        |
| momentum_5                           |
| momentum_10                          |
| momentum_20                          |
| volume_change                        |
| volume_ma_5                          |
| volume_ma_20                         |
| volume_ratio_5                       |
| volume_ratio_20                      |
| intraday_return                      |
| high_low_range                       |
| open_close_range                     |
| target_direction                     |

The dataset contains one entity, SPY Market Record, where each row represents one trading day of historical market data for the SPY ETF. The primary key is Date, since each record corresponds to a unique trading day.

**Data**

| Table Name | Description | Link |
| --- | --- | --- |
| spy_features | Engineered daily SPY dataset including raw market data, returns, moving averages, volatility, momentum, and prediction target | https://github.com/rameezrauf/ds4320_project1/blob/main/Data/spy_features.csv |

**Data Dictionary Table**

| Name | Data Type | Description | Example |
| --- | --- | --- | --- |
| Date | Date | Trading day associated with the record | 2024-03-01 |
| Open | Float | Opening price of SPY on that trading day | 510.25 |
| High | Float | Highest price reached during the trading day | 512.10 |
| Low | Float | Lowest price reached during the trading day | 508.90 |
| Close | Float | Closing price of SPY on that trading day | 511.45 |
| Volume | Integer | Number of shares traded during the day | 75234120 |
| daily_return | Float | Percentage change in closing price from the previous trading day | 0.0042 |
| moving_avg_5 | Float | Five-day moving average of closing price | 509.87 |
| moving_avg_20 | Float | Twenty-day moving average of closing price | 504.33 |
| target_direction | Integer | Binary target indicating whether price went up (1) or down (0) the next day | 1 |

**Data Dictionary Quantification**

Uncertainty for numerical features was quantified using descriptive statistics such as mean, standard deviation, and quartiles. Price-based features such as Close and daily_return vary due to changing market conditions, while derived features like volatility and momentum capture short-term fluctuations. The standard deviation of returns and volatility measures highlights the level of uncertainty in market movements. These statistics provide a clear quantitative understanding of variability across features and help evaluate their stability for predictive modeling.
