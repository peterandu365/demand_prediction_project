# Time Series Prediction: Moon Phase Effects on Sales

This project aims to investigate the impact of moon phases on sales performance and create a time series prediction model to forecast future sales. The project uses SARIMA and LSTM models, along with Exploratory Data Analysis (EDA), to identify patterns and trends in sales data.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Cleaning](#data-cleaning)
3. [Exploratory Data Analysis (EDA)](#eda)
4. [Feature Engineering](#feature-engineering)
5. [Modeling](#modeling)
    - [Benchmark: Prophet](#benchmark-prophet)
    - [Dual-input LSTM Model](#dual-input-lstm-model)
    - [Basic LSTM Model](#basic-lstm-model)
    - [SARIMA + LSTM Model](#sarima-lstm-model)
    - [FFT + LSTM Model](#fft-lstm-model)
6. [Model Performance Comparison](#model-performance-comparison)
7. [Residual Analysis](#residual-analysis)
8. [Conclusions](#conclusions)
9. [Acknowledgements](#acknowledgements)

## Introduction <a name="introduction"></a>

The project uses time series prediction to explore the impact of moon phases on sales performance, including:
- Using SARIMA and Dual-input LSTM models for half-year or longer time frame predictions.
- Exploring single-dependent and multi-dependent LSTM setups.
- Incorporating long-term Trend Line to improve results.
- Conducting residual analysis to identify possible reasons for prediction performance.

## Data Cleaning <a name="data-cleaning"></a>

I used a "lookup-based imputation" technique to fill the missing values. And, I did notice that there was no data for June 9th, 2019, which could have been intentionally removed by the designer or due to a malfunction on the e-commerce website. As this was not a meaningful event that could be repeated, I used the forward fill (ffill) method to fill in the missing values.

## Exploratory Data Analysis (EDA) <a name="eda"></a>

An important observation is that there is a clear correlation between sales volume and the moon phase cycle. There are significant spikes in sales during the four special moon phases, corresponding to the new moon, half moon, and full moon. However, I have observed that the average "order size" remains unchanged across moon phases. This suggests that the increased sales are due to either an increase in the number of orders or an increase in the number of customers, rather than an increase in the number of items per order. Based on this observation, we can infer that the moon phase has an impact on the collective consumer behavior of humans, which could be a valuable reference for the company to allocate resources.

![SUM_and AVG_vs_Moon_Phase.png](output/images/SUM_and_AVG_vs_Moon_Phase.png)

The same pattern is observed for subcategories as well. Additionally, a T-test was used to confirm this observation.

![Sales_Performance_by_Group_2_vs._Moon_Phases.png](output/images/Sales_Performance_by_Group_2_vs._Moon_Phases.png)






## Feature Engineering <a name="feature-engineering"></a>

Although sales are related to weather, obtaining weather forecasts six months in advance is challenging. Therefore, I use calendar weeks as proxy variables for weather, based on the hypothesis that weather conditions during the same period each year do not exhibit significant differences. The Kruskal-Wallis test for numerical features and the Chi-square test for categorical features were used to confirm the reliability of this approach.

![Weekly_Sales_Quantity_by_Category_in_Group_2_Throughout_the_Year.png](output/images/Weekly_Sales_Quantity_by_Category_in_Group_2_Throughout_the_Year.png)


The moon phase special days are distributed across any day of the week. Although more items were indeed sold on the day with the name related to the moon, weekdays are insufficient to substitute the moon phase in terms of feature engineering. Since moon phases can be calculated rather than forecasted, they can be used as a future independent variable. Therefore, I will use both calendar and moon phase as features to train and predict.

![Moon_Phase_Distribution_over_Weekday_crop.png](output/images/Moon_Phase_Distribution_over_Weekday_crop.png)


![sum_vs_weekdays.gif](output/images/sum_vs_weekdays.gif)


## Modeling <a name="modeling"></a>

### Benchmark: Prophet <a name="benchmark-prophet"></a>

The Prophet model was used as a benchmark for comparison with other models.


![prophet_results_zoom_start_None_zoom_end_None_seasonality_prior_scale_10000.png](output/images/Prophet_benchmark/prophet_results_zoom_start_None_zoom_end_None_seasonality_prior_scale_10000.png)



![prophet_results_zoom_start_2021-09-28_zoom_end_2022-01-07_seasonality_prior_scale_10000.png](output/images/Prophet_benchmark/prophet_results_zoom_start_2021-09-28_zoom_end_2022-01-07_seasonality_prior_scale_10000.png)

![prophet_results_zoom_start_2021-06-01_zoom_end_2021-09-10_seasonality_prior_scale_10000.png](output/images/Prophet_benchmark/prophet_results_zoom_start_2021-06-01_zoom_end_2021-09-10_seasonality_prior_scale_10000.png)

| Model | Submodel | RMSE | MAE | R² |
| --- | --- | --- | --- | --- |
| Prophet | Benchmark 1, as | 1190.69 | 788.35 | 0.38 |

### Dual-input LSTM Model Infrastructure <a name="dual-input-lstm-model"></a>

A dual-input LSTM model was developed to include future independent variables (X1 and X2) and capture both long-term and short-term patterns.


![final_project_2X_model_LSTM_-_No_TrendLine.png](output/images/final_project_2X_model_LSTM_-_No_TrendLine.png)




### Basic LSTM Model <a name="basic-lstm-model"></a>

The first LSTM model was developed to capture high-frequency patterns.


![LSTM_noTrendLine_LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_None_zoom_end_None 2.png](output/images/LSTM_noTrendLine_LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_None_zoom_end_None.png)



### SARIMA + LSTM Model <a name="sarima-lstm-model"></a>

SARIMA model was used to capture long-term patterns and seasonality. The LSTM model was then combined with SARIMA to improve overall prediction performance.

![final_project_2X_model_LSTM_-_TrendLine.png](output/images/final_project_2X_model_LSTM_-_TrendLine.png)

![SARIMA_shift_15_days.gif](output/images/SARIMA_shift_15_days.gif)


![LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_None_zoom_end_None.png](output/images/LSTM_SARIMA/LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_None_zoom_end_None.png)


![LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_2021-09-28_zoom_end_2022-01-07.png](output/images/LSTM_SARIMA/LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_2021-09-28_zoom_end_2022-01-07.png)

![LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_2021-06-01_zoom_end_2021-09-10.png](output/images/LSTM_SARIMA/LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_2021-06-01_zoom_end_2021-09-10.png)


### FFT + LSTM Model <a name="fft-lstm-model"></a>

Fast Fourier Transform (FFT) was applied to extract a smooth trend line and preserve short-term moon phase patterns with the LSTM model.


![train_test_data_trend_Log-FFT Trend.png](output/images/LSTM_FFT/train_test_data_trend_Log-FFT_Trend.png)



![LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_None_zoom_end_None.png](output/images/LSTM_FFT/LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_None_zoom_end_None.png)






By contrast, we can see how well the FFT on recovering the long-term trend: 

![FFT_TrendLine_LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_None_zoom_end_None.gif](output/images/FFT_TrendLine_LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_None_zoom_end_None.gif)


And how accurately the LSTM in capturing the patterns related to moon phases:


![LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_2021-09-28_zoom_end_2022-01-07.png](output/images/LSTM_FFT/LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_2021-09-28_zoom_end_2022-01-07.png)



![LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_2021-06-01_zoom_end_2021-09-10.png](output/images/LSTM_FFT/LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_2021-06-01_zoom_end_2021-09-10.png)

This algorithm can be scaled to any number of dependents:


![4_dpendent_result.gif](output/images/4_dpendent_result.gif)


## Model Performance Comparison <a name="model-performance-comparison"></a>

Multiple model configurations were experimented with, and the combination of Calendar and Moonphase features resulted in the most significant improvement in prediction performance.

| Model | Submodel | RMSE | MAE | R² |
| --- | --- | --- | --- | --- |
| Prophet | Benchmark 1, as | 1190.69 | 788.35 | 0.38 |
| SARIMA+LSTM | - | 1126.23 | 780.51 | 0.46 |
| FFT Trend | Benchmark 2, as | 985.10 | 689.37 | 0.58 |
| FFT+LSTM | Benchmark 2 + Calendar | 887.41 | 639.60 | 0.67 |
| FFT+LSTM | Benchmark 2 + Calendar + Holiday | 870.64 | 617.96 | 0.68 |
| FFT+LSTM | Benchmark 2 + Calendar + Moonphase | 858.10 | 598.17 | 0.69 |
| FFT+LSTM | Benchmark 2 + Calendar + Moonphase + Holiday | 891.10 | 625.11 | 0.66 |

## Residual Analysis <a name="residual-analysis"></a>

Residual analysis was conducted to identify possible reasons for the prediction metric R² remaining around 0.69 and not improving further. The impact of actual weather conditions on sales was investigated, revealing that unusually high occurrence of windy and snowy weather in 2021 compared to previous years could be a contributing factor to the lower prediction accuracy.


![residua_analysis_LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_2021-09-28_zoom_end_2022-01-07.gif](output/images/residua_analysis_LSTM_lookback_365_forcast_183_epochs_100_batch_size_1_step_183_zoom_start_2021-09-28_zoom_end_2022-01-07.gif)


![residual_analysis_2021_weather_condition_distribution.gif](output/images/residual_analysis_2021_weather_condition_distribution.gif)

I came across news reports about the Record-Breaking extreme weather events in that time window. These confirm my conjecture.

[**Late November and early December 2021 coldwave in Europe (25.11.-10.12.2021) - National top minimum temperatures (below 1000 MASL) estimates**](https://mkweather.com/late-november-and-early-december-2021-coldwave-in-europe-25-11-10-12-2021-national-top-minimum-temperatures-berlow-1000-masl-estimates/)


[**Europe hit severe snowstorm, 10-30, rarely up to 50 cm of snow is reported from eastern France, Central Europe, Baltic region, and Balkan**](https://mkweather.com/europe-hit-severe-snowstorm-10-30-rarely-up-to-50-cm-of-snow-is-reported-from-eastern-france-central-europe-baltic-region-and-balkan/)

[**-20°C in the UK already this weekend? It's very close to 102-year all-time record!**](https://mkweather.com/20c-in-the-uk-already-this-weekend-its-very-close-to-102-year-all-time-record/)



## Conclusions <a name="conclusions"></a>

This project demonstrates the potential impact of moon phases on sales performance and the usefulness of time series prediction models in forecasting sales. The combination of SARIMA and LSTM models, along with incorporating moon phase features, significantly improved prediction performance compared to the benchmark Prophet model. However, further analysis and improvements can be made by considering additional external factors, such as weather conditions, which may impact sales performance.

## Acknowledgements <a name="acknowledgements"></a>

We would like to express our gratitude to the mentors and my cohort members who have supported and provided valuable feedback throughout the development of this project.

