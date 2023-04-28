from math import floor
import src.modules.project_settings as ps
import src.modules.data_cleaning as dc


# Function to calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100




def read_moonphase_data(file_path):
    """
    Reads moon phase data from a CSV file, processes the data, and returns a DataFrame.

    Args:
        file_path (str): The path to the input CSV file containing moon phase data.

    Returns:
        pd.DataFrame: A DataFrame containing the processed moon phase data, with the 'date' column as the index and the 'mp' column representing moon phases.

    Example usage:
        moonphase_df = read_moonphase_data('path/to/moonphase_data.csv')
    """    
    # Read the input file
    df = pd.read_csv(file_path)

    # Convert the 'datetime' column to datetime objects
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Rename the 'datetime' column to 'date' and set the index to 'date'
    df.rename(columns={'datetime': 'date'}, inplace=True)
    df = df.set_index('date')

    # Rename the 'moonphase' column to 'mp'
    df.rename(columns={'moonphase': 'mp'}, inplace=True)

    # Select only the 'mp' column and return the resulting DataFrame
    df = df[['mp']]
    
    # Return the resulting DataFrame
    return df

def auto_seasonality_TL3(df, TL=True, TL_type=1, NS=True, AS=False, num_AS=10, MP=False, moonphase_df=None, HL=False, truncate_date = '2021-07-07', predict_end_date='2022-01-14', trend_cycle_days=183, N=20, season=365):
    """
    Analyzes and visualizes time series data to identify auto-seasonality and trend lines.

    Args:
        df (pd.DataFrame): Input DataFrame containing the time series data.
        TL (bool): Whether to include Trend Line. Defaults to True.
        TL_type (int): Trend Line type to use. Defaults to 1.
        NS (bool): Whether to include Natural Seasonality. Defaults to True.
        AS (bool): Whether to include Auto Seasonality. Defaults to False.
        num_AS (int): Number of first auto seasonality to be used. Defaults to 10.
        MP (bool): Whether to include Moon Phase. Defaults to False.
        moonphase_df (pd.DataFrame): Moon phase data as DataFrame. Defaults to None.
        HL (bool): Whether to include holidays. Defaults to False.
        truncate_date (str): Date to truncate data, for split the training and testing. Defaults to '2021-07-07'.
        predict_end_date (str): Date to end predictions. Defaults to '2022-01-14'.
        trend_cycle_days (int): Trend cycle days for FFT. Defaults to 183.
        N (int):  The number of top peaks to select.. Defaults to 20.
        season (int): Seasonality period. Defaults to 365 (deprecated).

    Example usage:
        auto_seasonality_TL3(df)
    """
    n_dependent_variables = len(df.columns)
    
    
    trend_label_dict = {1: 'Fitted Trend', 
                    2: 'SARIMA Trend',
                    3: 'FFT Trend',
                    4: 'Log-FFT Trend'}
        
        
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    # Define a function to fit a polynomial
    def polynomial_func1(x, x0, a, b):
        return a * (x - x0) ** 1 + b

    # Define a function to fit a polynomial
    def polynomial_func2(x, x0, a, b, c):
        return a * (x - x0) ** 2 + b * (x - x0) + c

    # Define the exponential function y = a*e^(b*(x-x0))+c
    def exponential_func(x, x0, a, b, c):
        return a * np.exp(b * (x - x0)) + c


    # # Define the exponential function y = a*10^(b*(x-x0))+c
    # def exponential_func(x, x0, a, b, c):
    #     return a * np.power(10, b * (x - x0)) + c


    # Define a function to calculate MAE and MRE
    def calculate_errors(true_values, predicted_values):
        abs_errors = np.abs(true_values - predicted_values)
        mae = np.mean(abs_errors)
        mre = np.mean(abs_errors / true_values)
        return mae, mre

    # truncate_date = '2021-07-07'

    df_fft = df[df.index < truncate_date].copy()


    # Apply FFT to the signal
    fft_coefficients = np.fft.fft(df_fft.quantity.values)

    # Define the frequency cutoff for the low-pass filter (keep frequencies below the cutoff)
    cutoff = 1 / trend_cycle_days

    # Create a mask for frequencies below the cutoff
    frequencies = np.fft.fftfreq(len(df_fft))
    mask = np.abs(frequencies) <= cutoff

    # Filter out the high frequencies (only keep the low frequencies)
    filtered_fft_coefficients = fft_coefficients * mask

    # Apply the inverse FFT to obtain the trend line
    trend_line = np.fft.ifft(filtered_fft_coefficients).real


    # fitting the trend-line with polynomial and exponential functions

    # Fit the polynomial functions (n=1, 2) and the exponential function
    x_values = np.arange(len(trend_line))

    polynomial_params1, _ = curve_fit(polynomial_func1, x_values, trend_line)
    polynomial_fit1 = polynomial_func1(x_values, *polynomial_params1)

    try:
        polynomial_params2, _ = curve_fit(polynomial_func2, x_values, trend_line, maxfev=2000)
        polynomial_fit2 = polynomial_func2(x_values, *polynomial_params2)
    except RuntimeError:
        print("Unable to fit the 2nd-degree polynomial function.")
        polynomial_fit2 = None

    # e^ instead of 10^
    try:
        exponential_params, _ = curve_fit(exponential_func, x_values, trend_line, p0=[x_values[0], 1, 0.001, 0], maxfev=2000)
        exponential_fit = exponential_func(x_values, *exponential_params)
    except RuntimeError:
        print("Unable to fit the exponential function.")
        exponential_fit = None


    # Calculate errors for each fitted function
    errors = {}
    if polynomial_fit1 is not None:
        print("Polynomial (n=1) parameters:", polynomial_params1)
        errors['Polynomial (n=1)'] = calculate_errors(trend_line, polynomial_fit1)
    if polynomial_fit2 is not None:
        print("Polynomial (n=2) parameters:", polynomial_params2)
        errors['Polynomial (n=2)'] = calculate_errors(trend_line, polynomial_fit2)
    if exponential_fit is not None:
        print("Exponential parameters:", exponential_params)
        errors['Exponential'] = calculate_errors(trend_line, exponential_fit)

    # Find the function with the lowest MAE
    best_function = min(errors, key=lambda k: errors[k][0])



    # initialize the df_FE_diff variable
    df_FE_diff = df[df.index < truncate_date].copy()


    # Create df_trend DataFrame with the same index as df but only until predict_end_date
    df_trend = pd.DataFrame(index=pd.date_range(df_fft.index[0], predict_end_date), columns=['quantity'])

    # ===============================================
    # ========= trend line related: =================
    # ===============================================
    
    
    # ========= plot trend line: =================
    

    # Plot the best function
    _ = plt.figure(figsize=(12, 6))
    _ = plt.plot(df_fft.index, df_fft.quantity, color='grey', alpha=0.5, label='Original Data')
    _ = plt.plot(df_fft.index, trend_line, label='Trend Line (FFT)', color='green', alpha=0.5)

    if best_function == 'Polynomial (n=1)':  # label='Trend Line', color='orange', alpha=0.5, linestyle='-.', linewidth=5
        _ = plt.plot(df_fft.index, polynomial_fit1, label='Best Fit: Polynomial (n=1)', color='orange', alpha=0.5, linestyle='-.', linewidth=5)
    elif best_function == 'Polynomial (n=2)':
        _ = plt.plot(df_fft.index, polynomial_fit2, label='Best Fit: Polynomial (n=2)', color='orange', alpha=0.5, linestyle='-.', linewidth=5)
    else:
        _ = plt.plot(df_fft.index, exponential_fit, label='Best Fit: Exponential', color='orange', alpha=0.5, linestyle='-.', linewidth=5)
    # Add a vertical red dashed line on the first date of prediction_dates
    plt.axvline(x=pd.to_datetime(truncate_date), color='pink', alpha=0.9, linestyle='--')
    
    _ = plt.xlabel('Date')
    _ = plt.ylabel('Quantity')
    _ = plt.title('Original Data, Extracted Trend Line, and Best Fitted Function')
    _ = plt.legend()
    # save the plot as png dpi=300
    _ = plt.savefig(ps.LSTM1_OUT_PATH + 'original_data_trend_line_best_fitted_function.png', dpi=300)
    _ = plt.show()





    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Decompose the time series data into trend, seasonal, and residual components
    # The frequency parameter (freq) is set to 365, assuming daily data and a yearly pattern
    result = seasonal_decompose(df['quantity'], model='multiplicative', period=365)

    # Plot the decomposed time series components
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    result.trend.plot(ax=ax1)
    ax1.set_title('Trend')
    result.seasonal.plot(ax=ax2)
    ax2.set_title('Seasonal')
    result.resid.plot(ax=ax3)
    ax3.set_title('Residual')
    
    plt.tight_layout()
    # save the plot as png dpi=300
    _ = plt.savefig(ps.LSTM1_OUT_PATH + 'decomposed_time_series_components.png', dpi=300)
    plt.show()


    # print(f"FFT on FFT-Filtered Principal periods: {principal_periods}")


    print(f"Best function: {best_function}")
    print(f"MAE: {errors[best_function][0]}, MRE: {errors[best_function][1]}")

    # Get the extended x_values for the best_fit_extended
    extended_x_values = np.arange(len(df_trend.index))

    # Calculate the best_fit_extended using the best_function parameters
    if best_function == 'Polynomial (n=1)':
        best_fit_extended = polynomial_func1(extended_x_values, *polynomial_params1)
    elif best_function == 'Polynomial (n=2)':
        best_fit_extended = polynomial_func2(extended_x_values, *polynomial_params2)
    else:
        best_fit_extended = exponential_func(extended_x_values, *exponential_params)



    # Calculate the difference between the original data and the best_fit (up to the length of df_fft.index)
    difference = df_FE_diff.quantity.values - best_fit_extended[:len(df_fft.index)]
    
    _train_data = df[df.index < truncate_date]
    _test_data = df[(df.index >= truncate_date) & (df.index <= predict_end_date)]
    if TL_type == 1: 
        if TL is True:
            # Add the best_fit_extended as a new column in the df_trend DataFrame
            df_trend['quantity'] = best_fit_extended
            # Add the difference as a new column in the df_FE_diff DataFrame
            df_FE_diff['quantity'] = difference

        else:
            df_trend['quantity'] = 0
    elif TL_type == 2:
        
        if TL is True:

            trend_forecast = sarima_trend_MS_resample_3(_train_data, _test_data, 0, truncate_date, predict_end_date)
            
            # Convert daily_data (Series) to DataFrame
            trend_forecast_df = trend_forecast.to_frame(name='trend_forecast').ffill()
            trend_forecast_df.index.name = df_trend.index.name


            # Update the original DataFrame df_trend with the trend_forecast_df
            df_trend['quantity'] = trend_forecast_df['trend_forecast']
            
            df_trend['quantity'] = df_trend['quantity'].shift(15).bfill()
            
            # # ==== Find the best shift value for the trend line START ====
            # shift_range = range(1, 61)
            # df_trend, best_shift, min_rmse = align_dataframes(df_trend, df_FE_diff, shift_range)

            # print(f"Best shift: {best_shift}")
            # print(f"Minimum RMSE: {min_rmse}")
            # print(f"Aligned df_trend:\n{df_trend}")
            # # ==== Find the best shift value for the trend line END ====
                    
            df_FE_diff['quantity'] = df_FE_diff['quantity'] - df_trend['quantity']
            
        else:
            df_trend['quantity'] = 0
    elif TL_type == 3:
        
        if TL is True:
                                    
            best_fit_extended_fft = extend_fft_trend_5(df, truncate_date='2021-07-08', lag_years=2, lead_years=1, trend_cycle_days=trend_cycle_days, smooth_window=30)
            # Add the best_fit_extended as a new column in the df_trend DataFrame
            df_trend['quantity'] = best_fit_extended_fft

            # Add the difference as a new column in the df_FE_diff DataFrame
            df_FE_diff['quantity'] = df_FE_diff['quantity'] - df_trend['quantity']
            
            
        else:
            df_trend['quantity'] = 0
    
    elif TL_type == 4: # extract the FFT trandline in log space
        
        if TL is True:
            
            df_log = df.copy()
            df_log.iloc[:,0] = np.log10(df_log.iloc[:,0])            
            best_fit_extended_fft = extend_fft_trend_3(df_log, truncate_date='2021-07-08', lag_years=2, lead_years=1, trend_cycle_days=trend_cycle_days, smooth_window=30)
            # Add the best_fit_extended as a new column in the df_trend DataFrame
            df_trend['quantity'] = best_fit_extended_fft
            
            df_trend['quantity'] = 10**df_trend['quantity']

            # Add the difference as a new column in the df_FE_diff DataFrame
            df_FE_diff['quantity'] = df_FE_diff['quantity'] - df_trend['quantity']
            
            
        else:
            df_trend['quantity'] = 0        
        
    # ===============================================
    
    plt.figure(figsize=(12, 6))
    plt.plot(_train_data.index, _train_data.quantity, label='Training Data', color='blue', alpha=0.3)
    plt.plot(_test_data.index, _test_data.quantity, label='Test Data', color='green', alpha=0.3)

    # Split the trend data into train and test parts
    df_trend_train = df_trend[df_trend.index < truncate_date]
    df_trend_test = df_trend[(df_trend.index >= truncate_date) & (df_trend.index <= predict_end_date)]
    
    plt.plot(df_trend_train.index, df_trend_train.quantity, label='Trend - Train', color='red', alpha=0.6)
    plt.plot(df_trend_test.index, df_trend_test.quantity, label='Trend - Test', color='red', alpha=0.6, linestyle='--')

    # Add a vertical red dashed line on the first date of prediction_dates
    plt.axvline(x=pd.to_datetime(truncate_date), color='pink', alpha=0.9, linestyle='--')
    from dateutil.relativedelta import relativedelta


    plt.axvline(x=pd.to_datetime(truncate_date) - relativedelta(years=1), color='blue', alpha=0.3, linestyle='--')
    plt.axvline(x=pd.to_datetime(truncate_date) - relativedelta(years=2), color='blue', alpha=0.3, linestyle='--')

        
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.title('Training Data, Test Data, and Trend')
    plt.legend()

    # Save the plot as png dpi=300
    plt.savefig(ps.LSTM1_OUT_PATH + f'train_test_data_trend_{trend_label_dict[TL_type]}.png', dpi=300)
    plt.show()


    # ================ Trendline bench mark ================
    import math
    import os
    import pandas as pd
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from dateutil.relativedelta import relativedelta


    # Find the common range of test data and trend test data
    common_range_start = max(_test_data.index.min(), df_trend_test.index.min())
    common_range_end = min(_test_data.index.max(), df_trend_test.index.max())

    # Filter the test data and trend test data based on the common range
    test_data_common_range = _test_data[(common_range_start <= _test_data.index) & (_test_data.index <= common_range_end)]
    trend_test_data_common_range = df_trend_test[(common_range_start <= df_trend_test.index) & (df_trend_test.index <= common_range_end)]

    # Calculate evaluation metrics
    rmse = math.sqrt(mean_squared_error(test_data_common_range.quantity, trend_test_data_common_range.quantity))
    mae = mean_absolute_error(test_data_common_range.quantity, trend_test_data_common_range.quantity)
    mape = mean_absolute_percentage_error(test_data_common_range.quantity, trend_test_data_common_range.quantity)
    r2 = r2_score(test_data_common_range.quantity, trend_test_data_common_range.quantity)

    # Save evaluation metrics to a txt file
    output_path = ps.LSTM1_OUT_PATH + "FFT_trendline_benchmark_report.txt"
    with open(output_path, 'w') as f:
        f.write(f'RMSE: {rmse}\n')
        f.write(f'MAE: {mae}\n')
        f.write(f'MAPE: {mape}\n')
        f.write(f'R2: {r2}\n')    
    
    
    
    # ===============================================     
        
        
    # ===============================================
    # ========= sorted peak related: ================
    # ===============================================


        
    # convert quantity column to from time domain to frequency domain using FFT, and plot it. frequency range should be corresponding to 1 to 365.

    import numpy as np
    import matplotlib.pyplot as plt

    # Apply the Fast Fourier Transform (FFT) to the quantity column
    quantity_fft = np.fft.fft(difference)

    # Get the frequency corresponding to each FFT coefficient
    sample_frequency = np.fft.fftfreq(df_FE_diff.shape[0])

    # Calculate the magnitude of the FFT coefficients
    fft_magnitude = np.abs(quantity_fft)


    from scipy.signal import find_peaks

    # Find the peaks in the magnitude array
    peaks, _ = find_peaks(fft_magnitude)

    # Sort the peaks by magnitude in descending order and take the top N peaks
    # N = 20
    top_peaks = sorted(peaks, key=lambda i: fft_magnitude[i], reverse=True)[:N]

    # Get the central frequencies corresponding to the top peaks
    central_frequencies = sample_frequency[top_peaks]

    # Filter out the negative frequencies and calculate the corresponding periods
    positive_frequencies = central_frequencies[central_frequencies > 0]
    corresponding_periods = 1 / positive_frequencies

    # Sort periods by energy concentration in descending order
    sorted_periods = sorted(corresponding_periods, key=lambda p: np.mean(fft_magnitude[sample_frequency == 1 / p]), reverse=True)

    # Print the periods with the most energy concentration
    print("Cycles with most energy concentration:", sorted_periods)

    # Plot the FFT magnitudes with x-axis labeled as periods
    _ = plt.figure(figsize=(12, 8))
    _ = plt.plot(1 / sample_frequency[:len(sample_frequency) // 2], fft_magnitude[:len(fft_magnitude) // 2])
    _ = plt.xlabel("Period (days)")
    _ = plt.ylabel("Magnitude")
    _ = plt.title("Frequency Domain Representation of the Data (Periods)")
    _ = plt.xscale("log")

    # Highlight the periods with the most energy concentration and add vertical annotations
    for period in sorted_periods:
        _ = plt.axvline(x=period, color='r', alpha=0.3, linestyle='--')
        _ = plt.annotate(f"{period:.2f}", xy=(period, max(fft_magnitude[:len(fft_magnitude) // 2]) * 0.5), xytext=(period, max(fft_magnitude[:len(fft_magnitude) // 2])), rotation='vertical', fontsize=10, va='top', textcoords='data')

    _ = plt.xlim(1 / sample_frequency[len(sample_frequency) // 2], 1 / sample_frequency[1])  # Set the x-axis limits to exclude 0 frequency

    # save the "Frequency domain representation of the signal (Periods)" plot as png dpi=300
    _ = plt.savefig(ps.LSTM1_OUT_PATH + 'Frequency_domain_representation_of_the_signal_periods.png', dpi=300)

    _ = plt.show()


    # ===============================================
    # ========= sorted peak related: ================
    # ===============================================

    print('the most noticeable periods are:\n\n', [round(p, 2) for p in sorted_periods], '\n\n')



    # # Extend the index of df_FE_diff to include the future dates: predict_end_date='2022-01-14'
    # new_index = pd.date_range(df_FE_diff.index[0], predict_end_date)
    # df_FE_diff = df_FE_diff.reindex(new_index)

    
        
    # # ===============================================
    # # use the natural seasonality to create new features
    # # ===============================================    
    
    
    # if NS is True:

    #     # Add columns: month, week_of_year, day_of_month, day_of_week
    #     df_FE_diff['year'] = df_FE_diff.index.year
    #     df_FE_diff['month'] = df_FE_diff.index.month
    #     # df_FE_diff['week_of_year'] = df_FE_diff.index.isocalendar().week
    #     df_FE_diff['week_of_year'] = df_FE_diff.index.isocalendar().week.astype(int)
    #     df_FE_diff['day_of_month'] = df_FE_diff.index.day
    #     df_FE_diff['day_of_week'] = df_FE_diff.index.dayofweek

    import pandas as pd

    # Extend the index of df_FE_diff to include the future dates: predict_end_date='2022-01-14'
    new_index = pd.date_range(df_FE_diff.index[0], predict_end_date)
    df_FE_diff = df_FE_diff.reindex(new_index)
    
    # Use natural seasonality to create new features (if NS is True)
    if NS is True:
        # Add columns: year, month, week_of_year, day_of_month, day_of_week
        df_FE_diff['year'] = df_FE_diff.index.year.astype(int)
        df_FE_diff['month'] = df_FE_diff.index.month.astype(int)
        # df_FE_diff['week_of_year'] = df_FE_diff.index.isocalendar().week
        df_FE_diff['week_of_year'] = df_FE_diff.index.isocalendar().week.astype(int)
        df_FE_diff['day_of_month'] = df_FE_diff.index.day.astype(int)
        df_FE_diff['day_of_week'] = df_FE_diff.index.dayofweek.astype(int)



    # ===============================================
    # use auto_seasonality peaks to create new features
     # ===============================================
    if AS is True:
        # Create a column with the day number 
        # Calculate day_number for each row in the DataFrame
        df_FE_diff['day_number'] = (df_FE_diff.index - df_FE_diff.index[0]).days     
            
        # Create new features based on the sorted_periods
        for i, period in enumerate(sorted_periods[:num_AS]):
            feature_name = f"period_{period:.2f}"
            df_FE_diff[feature_name] = df_FE_diff['day_number'].apply(lambda x: int(round(x % period +1)))

        # drop the day_number column from df_FE_diff
        df_FE_diff.drop(columns=['day_number'], inplace=True)


    

    if MP is True: 
        ## categorize the moon phases
        _moonphase_df = moonphase_df.copy()
        import numpy as np

        moonphase_values = np.array([0, 0.25, 0.5, 0.75, 1.0])
        tolerance = 2.5e-2

        # Get the closest moonphase_value for each value in the 'moonphase' column
        closest_values = moonphase_values[np.abs(_moonphase_df['mp'].values[:, None] - moonphase_values).argmin(axis=1)]

        # Create a mask for values within the tolerance range of the closest moonphase_value
        within_tolerance_mask = np.abs(_moonphase_df['mp'].values - closest_values) <= tolerance

        # Assign labels for the values within the tolerance range
        moonphase_categories = pd.Series(['n'] * len(_moonphase_df['mp']), index=_moonphase_df['mp'].index)
        moonphase_categories[within_tolerance_mask] = np.select(
            condlist=[closest_values[within_tolerance_mask] == val for val in moonphase_values],
            choicelist=['0', '1', '2', '3', '0']
        )



        # Add moonphase_group column to moonphase_df
        _moonphase_df['mp'] = moonphase_categories

        # Merge moonphase_group column to df_FE_1a based on the index (date)
        df_FE_diff = df_FE_diff.merge(_moonphase_df[['mp']], left_index=True, right_index=True, how='left')

        # rename the moonphase_group column to mp
        # df_FE_diff.rename(columns={'moonphase_group': 'mp'}, inplace=True)


    if HL is True:
        import pandas as pd

        # Create a list of tuples with the holiday name and date range

        # Create a list of tuples with the holiday name and date range
        holidays = [('New Year', pd.date_range(start='2019-01-01', end='2019-01-02')),
                    ('Orthodox Christmas', pd.date_range(start='2019-01-07', end='2019-01-08')),
                    ('Easter', pd.date_range(start='2019-04-28', end='2019-04-29')),
                    ('Labour Day', pd.date_range(start='2019-05-01', end='2019-05-02')),
                    ('Whit Sunday', pd.date_range(start='2019-06-16', end='2019-06-17')),
                    ('Assumption Day', pd.date_range(start='2019-08-15', end='2019-08-16')),
                    ('Christmas', pd.date_range(start='2019-12-25', end='2019-12-26')),
                    ('New Year', pd.date_range(start='2020-01-01', end='2020-01-02')),
                    ('Orthodox Christmas', pd.date_range(start='2020-01-07', end='2020-01-08')),
                    ('Easter', pd.date_range(start='2020-04-19', end='2020-04-20')),
                    ('Labour Day', pd.date_range(start='2020-05-01', end='2020-05-02')),
                    ('Whit Sunday', pd.date_range(start='2020-06-07', end='2020-06-08')),
                    ('Assumption Day', pd.date_range(start='2020-08-15', end='2020-08-16')),
                    ('Christmas', pd.date_range(start='2020-12-25', end='2020-12-26')),
                    ('New Year', pd.date_range(start='2021-01-01', end='2021-01-02')),
                    ('Orthodox Christmas', pd.date_range(start='2021-01-07', end='2021-01-08')),
                    ('Easter', pd.date_range(start='2021-05-02', end='2021-05-03')),
                    ('Labour Day', pd.date_range(start='2021-05-01', end='2021-05-02')),
                    ('Whit Sunday', pd.date_range(start='2021-06-20', end='2021-06-21')),
                    ('Assumption Day', pd.date_range(start='2021-08-15', end='2021-08-16')),
                    ('Christmas', pd.date_range(start='2021-12-25', end='2021-12-26')),
                    ('New Year', pd.date_range(start='2022-01-01', end='2022-01-02')),
                    ('Orthodox Christmas', pd.date_range(start='2022-01-07', end='2022-01-08')),
                    ('Easter', pd.date_range(start='2022-04-24', end='2022-04-25')),
                    ('Labour Day', pd.date_range(start='2022-05-01', end='2022-05-02')),
                    ('Whit Sunday', pd.date_range(start='2022-06-12', end='2022-06-13')),
                    ('Assumption Day', pd.date_range(start='2022-08-15', end='2022-08-16')),
                    ('Christmas', pd.date_range(start='2022-12-25', end='2022-12-26'))]


        # Create an empty dataframe with the datetime index
        date_range = pd.date_range(start=df_FE_diff.index.min(), end=df_FE_diff.index.max(), freq='D')
        df_holiday = pd.DataFrame(index=date_range)

        # Fill in the holiday column with the corresponding holiday names
        for holiday in holidays:
            name = holiday[0]
            dates = holiday[1]
            for date in dates:
                df_holiday.loc[date, 'holiday'] = name

        # All non-NaN values in the column "holiday" to 'Yes'
        # df_holiday.loc[df_holiday['holiday'].notnull(), 'holiday'] = 'Yes'
        
        # Fill in missing values with "No"
        df_holiday['holiday'].fillna('No', inplace=True)
        
        df_FE_diff = df_FE_diff.merge(df_holiday[['holiday']], left_index=True, right_index=True, how='left')
        
        
        

    # set all coumns [1: ] 's type to string
    df_FE_diff[df_FE_diff.columns[n_dependent_variables:]] = df_FE_diff[df_FE_diff.columns[1:]].astype(str)

    

    # dc.print_df_with_dtypes(df_FE_diff, max_rows=200, print_on=True, data_generated_path=DATA_GENERATED_PATH, name_extension='EDA_quantity_sum_df')
    

    
    return df_FE_diff, df_trend






import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

def sarima_trend_MS(data_train, data_test, column_num, season, predict_start_date, predict_end_date):
    """
    Fits a SARIMA model to the training data, forecasts the trend, and appends the forecast to the original DataFrame.

    Args:
        data_train (pd.DataFrame): Training data as a DataFrame.
        data_test (pd.DataFrame): Test data as a DataFrame.
        column_num (int): Index of the target column in the DataFrame.
        season (int): Seasonality period to use for the SARIMA model.
        predict_start_date (str): Start date for forecasting in the format 'YYYY-MM-DD'.
        predict_end_date (str): End date for forecasting in the format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: The original DataFrame with the SARIMA forecast appended.
    """
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Set the target column
    target_col = data_train.columns[column_num]

    # Plot ACF and PACF to determine optimal SARIMA parameters
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(data_train[target_col], lags=30, ax=ax1)
    plot_pacf(data_train[target_col], lags=30, ax=ax2, method='ywm')
    plt.show()

    # Replace p, d, q, P, D, and Q with appropriate values based on ACF and PACF plots
    p, d, q = 1, 1, 1
    P, D, Q, s = 1, 1, 1, season  # Adjust the seasonal period based on the nature of the data

    # Fit the SARIMA model
    model = SARIMAX(data_train[target_col], order=(p, d, q), seasonal_order=(P, D, Q, s), freq='MS')
    results = model.fit()

    # Convert date strings to datetime objects
    predict_start_date = pd.to_datetime(predict_start_date)
    predict_end_date = pd.to_datetime(predict_end_date)

    # Calculate the number of steps for forecasting
    steps = ((predict_end_date.year - predict_start_date.year) * 12) + predict_end_date.month - predict_start_date.month + 1
    forecast = results.get_forecast(steps=steps)
    mean_forecast = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    plt.figure(figsize=(12, 6))
    plt.plot(data_train.index, data_train[target_col], label='Training Data')
    plt.plot(data_test.index, data_test[target_col], label='Test Data', color='g', alpha=0.4, linewidth=3.5)
    plt.plot(mean_forecast.index, mean_forecast, label='Forecast', color='r', alpha=0.8)
    plt.fill_between(confidence_intervals.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink')
    plt.legend(loc='best')
    plt.title('SARIMA Forecast')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.show()

    # Append forecast to the original DataFrame
    forecast_df = mean_forecast.to_frame(name=target_col)
    forecast_df.index.name = data_train.index.name
    final_df = pd.merge_ordered(data_train.reset_index(), forecast_df.reset_index(), on=data_train.index.name, how='outer', fill_method='ffill')
    final_df.set_index(data_train.index.name, inplace=True)
    return final_df.loc[:predict_end_date]



def sarima_trend_MS_resample_3(data_train, data_test, column_num, predict_start_date, predict_end_date):
    """
    Fits a SARIMA model to the resampled training data, forecasts the trend, and returns the forecasted values
    resampled back to the original frequency.

    Args:
        data_train (pd.DataFrame): Training data as a DataFrame.
        data_test (pd.DataFrame): Test data as a DataFrame.
        column_num (int): Index of the target column in the DataFrame.
        predict_start_date (str): Start date for forecasting in the format 'YYYY-MM-DD'.
        predict_end_date (str): End date for forecasting in the format 'YYYY-MM-DD'.

    Returns:
        pd.Series: The forecasted values resampled back to the original frequency.
    """
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Resample daily data to monthly averages
    data_train_monthly = data_train.resample('MS').mean()
    data_test_monthly = data_test.resample('MS').mean()
        
    # Set the target column
    target_col = data_train_monthly.columns[column_num]

    # # ========== Grid Search for the Best Parameters ==========
    # import itertools

    # p = d = q = range(0, 2)
    # P = D = Q = range(0, 2)
    # s = [12]  # seasonal frequency, 12 for monthly data

    # # Generate all combinations of p, d, q, P, D, Q, and s
    # pdq = list(itertools.product(p, d, q))
    # seasonal_pdq = list(itertools.product(P, D, Q, s))

    # best_aic = float("inf")
    # best_order = None
    # best_seasonal_order = None

    # for order in pdq:
    #     for seasonal_order in seasonal_pdq:
    #         try:
    #             model = SARIMAX(data_train_monthly[target_col], order=order, seasonal_order=seasonal_order, freq='MS')
    #             results = model.fit(method_kwargs={"disp": 0})

    #             if results.aic < best_aic:
    #                 best_aic = results.aic
    #                 best_order = order
    #                 best_seasonal_order = seasonal_order
    #         except:
    #             continue

    # print("Best AIC: ", best_aic)
    # print("Best order: ", best_order)
    # print("Best seasonal_order: ", best_seasonal_order)

    # ========== Grid Search End ==========

    # Fit the SARIMA model with the best parameters
    best_p, best_d, best_q = 0, 0, 0
    best_P, best_D, best_Q, best_s = 1, 1, 1, 12

    model = SARIMAX(data_train_monthly[target_col], order=(best_p, best_d, best_q), seasonal_order=(best_P, best_D, best_Q, best_s), freq='MS')
    results = model.fit()

    # Calculate the number of steps for forecasting
    predict_start_date = pd.to_datetime(predict_start_date)
    predict_end_date = pd.to_datetime(predict_end_date)
    steps = ((predict_end_date.year - predict_start_date.year) * 12) + predict_end_date.month - predict_start_date.month + 1
    forecast = results.get_forecast(steps=steps)
    mean_forecast = forecast.predicted_mean

    # Concatenate train data and forecast data
    combined_data = pd.concat([data_train_monthly[target_col], mean_forecast])

    # Fill missing values with interpolation
    combined_data = combined_data.interpolate()

    # Convert combined data back to daily interval
    daily_data = combined_data.resample('D').asfreq()
    daily_data = daily_data.interpolate()

    # Filter daily data to the desired time range
    daily_data = daily_data[(daily_data.index >= data_train.index.min()) & (daily_data.index <= max(data_test.index.max(), predict_end_date))]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(data_train_monthly.index, data_train_monthly[target_col], label='Training Data')
    plt.plot(data_test_monthly.index, data_test_monthly[target_col], label='Test Data', color='g', alpha=0.4, linewidth=3.5)
    plt.plot(mean_forecast.index, mean_forecast, label='Forecast', color='r', alpha=0.8)
    plt.legend(loc='best')
    plt.title('SARIMA Forecast')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.savefig('output/LSTM_1/SARIMA Forecast Yearly.png', dpi=300)
    plt.show()
    print('type of daily_data: ', type(daily_data))
    
    
    return daily_data


def sarima_trend_daily(data_train, data_test, column_num, predict_start_date, predict_end_date):
    """
    Fits a daily SARIMA model to the training data, forecasts the trend, and returns the forecasted values.

    Args:
        data_train (pd.DataFrame): Training data as a DataFrame.
        data_test (pd.DataFrame): Test data as a DataFrame.
        column_num (int): Index of the target column in the DataFrame.
        predict_start_date (str): Start date for forecasting in the format 'YYYY-MM-DD'.
        predict_end_date (str): End date for forecasting in the format 'YYYY-MM-DD'.

    Returns:
        pd.Series: The forecasted values.
    """
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Set the target column
    target_col = data_train.columns[column_num]

    # Fit the SARIMA model with the best parameters
    best_p, best_d, best_q = 0, 0, 0
    best_P, best_D, best_Q, best_s = 1, 1, 1, 365

    model = SARIMAX(data_train[target_col], order=(best_p, best_d, best_q), seasonal_order=(best_P, best_D, best_Q, best_s), freq='D')
    results = model.fit()

    # Calculate the number of steps for forecasting
    predict_start_date = pd.to_datetime(predict_start_date)
    predict_end_date = pd.to_datetime(predict_end_date)
    steps = (predict_end_date - predict_start_date).days + 1
    forecast = results.get_forecast(steps=steps)
    mean_forecast = forecast.predicted_mean

    # Concatenate train data and forecast data
    combined_data = pd.concat([data_train[target_col], mean_forecast])

    # Fill missing values with interpolation
    combined_data = combined_data.interpolate()

    # Filter combined data to the desired time range
    combined_data = combined_data[(combined_data.index >= data_train.index.min()) & (combined_data.index <= max(data_test.index.max(), predict_end_date))]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(data_train.index, data_train[target_col], label='Training Data')
    plt.plot(data_test.index, data_test[target_col], label='Test Data', color='g', alpha=0.4, linewidth=3.5)
    plt.plot(mean_forecast.index, mean_forecast, label='Forecast', color='r', alpha=0.8)
    plt.legend(loc='best')
    plt.title('SARIMA Forecast')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.savefig('output/LSTM_1/SARIMA Forecast Daily.png', dpi=300)
    plt.show()
    print('type of combined_data: ', type(combined_data))
    
    return combined_data



import numpy as np
from sklearn.metrics import mean_squared_error

def align_dataframes(df1, df2, shift_range):
    """
    Aligns two DataFrames by minimizing the RMSE between their 'quantity' columns based on a given range of shifts.

    Args:
        df1 (pd.DataFrame): First DataFrame to be aligned.
        df2 (pd.DataFrame): Second DataFrame to be aligned.
        shift_range (list): A list of integers representing the range of shifts to be tested.

    Returns:
        pd.DataFrame: The shifted and aligned first DataFrame.
        int: The best shift value that minimized the RMSE.
        float: The minimum RMSE obtained.
    """    
    best_shift = None
    min_rmse = np.inf

    # Find the common time range of both DataFrames
    common_start = max(df1.index.min(), df2.index.min())
    common_end = min(df1.index.max(), df2.index.max())
    common_range = df1.loc[common_start:common_end].index

    # Extract the common time range data from both DataFrames
    df1_common = df1.loc[common_range]
    df2_common = df2.loc[common_range]

    for shift in shift_range:
        shifted_df1 = df1_common.shift(-shift).ffill().bfill()

        rmse = np.sqrt(mean_squared_error(df2_common['quantity'], shifted_df1['quantity']))

        if rmse < min_rmse:
            min_rmse = rmse
            best_shift = shift

    aligned_df1 = df1.shift(-best_shift).bfill()

    # Writing results to a text file
    with open("SARIMA_best_shift_report.txt", "w") as file:
        file.write(f"Best shift: {best_shift}\n")
        file.write(f"Minimum RMSE: {min_rmse}\n")
        file.write(f"Aligned df_trend:\n{aligned_df1}\n")
    
    return aligned_df1, best_shift, min_rmse


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import pandas as pd
from scipy.signal import savgol_filter

def extend_fft_trend_1(df, truncate_date='2021-07-08', lag_years=2, lead_years=1, trend_cycle_days=60, smooth_window=15):
    """
    Extends the FFT trend line by scaling it based on the variance ratio between two years and applies smoothing around the truncate_date.

    Args:
        df (pd.DataFrame): Input DataFrame containing the time series data.
        truncate_date (str): Date in the format 'YYYY-MM-DD' used to truncate the DataFrame. Defaults to '2021-07-08'.
        lag_years (int): Number of years to go back for calculating the variance ratio. Defaults to 2.
        lead_years (int): Number of years to extend the trend line. Defaults to 1.
        trend_cycle_days (int): Number of days for the trend cycle. Defaults to 60.
        smooth_window (int): Window size for the Savitzky-Golay filter for smoothing. Defaults to 15.

    Returns:
        np.array: Truncated and extended FFT trend line.
    """
    df_fft = df[df.index < truncate_date]
    cutoff = 1 / trend_cycle_days

    # Use calendar date to slice/subset
    year1_start_date = pd.to_datetime(truncate_date) - pd.DateOffset(years=lag_years)
    year2_start_date = pd.to_datetime(truncate_date) - pd.DateOffset(years=lag_years - 1)
    
    year1_signal = df_fft.loc[year1_start_date:year2_start_date - pd.DateOffset(days=1)]['quantity'].values
    year2_signal = df_fft.loc[year2_start_date:truncate_date]['quantity'].values
    
    signal = np.concatenate((year1_signal, year2_signal))
    
    signal_fft = fft(signal)
    
    variance_year1 = np.var(year1_signal)
    variance_year2 = np.var(year2_signal)

    # Apply a low-pass filter to the FFT output to extract the trend line
    cutoff_frequency = int(len(signal) * cutoff)
    filtered_signal_fft = np.copy(signal_fft)
    filtered_signal_fft[cutoff_frequency:len(signal) - cutoff_frequency] = 0

    # Perform inverse FFT to get the trend line
    trend_line = np.real(ifft(filtered_signal_fft))

    # Compute the scaling factor for year 3 based on the variance ratio
    scaling_factor_year3 = variance_year2 / variance_year1

    # Scale the trend line for year 3 and concatenate it to the trend line for year 1 and year 2
    trend_line_year3 = trend_line[len(trend_line)//2:] * scaling_factor_year3
    extended_trend_line = np.concatenate((trend_line, trend_line_year3))

    # Truncate the extended trend line according to the date range of the original dataframe
    start_date = df.index.min()
    end_date = df.index.max()
    date_range = pd.date_range(start_date, end_date)
    truncated_trend_line = extended_trend_line[:len(date_range)]

    # Apply a smooth around the truncate_date, each side 15 days
    smooth_start = np.where(date_range == pd.to_datetime(truncate_date))[0][0] - smooth_window
    smooth_end = smooth_start + 2 * smooth_window + 1
    truncated_trend_line[smooth_start:smooth_end] = savgol_filter(truncated_trend_line[smooth_start:smooth_end], 2 * smooth_window + 1, 3)

    return truncated_trend_line




import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import pandas as pd
from scipy.signal import savgol_filter

def extend_fft_trend_2(df, truncate_date='2021-07-08', lag_years=2, lead_years=1, trend_cycle_days=60, smooth_window=15):
    """
    Extends the FFT trend line by scaling it based on the variance ratio between two years and applies smoothing around the truncate_date.

    Args:
        df (pd.DataFrame): Input DataFrame containing the time series data.
        truncate_date (str): Date in the format 'YYYY-MM-DD' used to truncate the DataFrame. Defaults to '2021-07-08'.
        lag_years (int): Number of years to go back for calculating the variance ratio. Defaults to 2.
        lead_years (int): Number of years to extend the trend line. Defaults to 1.
        trend_cycle_days (int): Number of days for the trend cycle. Defaults to 60.
        smooth_window (int): Window size for the Savitzky-Golay filter for smoothing. Defaults to 15.

    Returns:
        np.array: Truncated and extended FFT trend line.
    """
    df_fft = df[df.index < truncate_date]
    cutoff = 1 / trend_cycle_days

    # Use calendar date to slice/subset
    year1_start_date = pd.to_datetime(truncate_date) - pd.DateOffset(years=lag_years)
    year2_start_date = pd.to_datetime(truncate_date) - pd.DateOffset(years=lag_years - 1)
    
    year1_signal = df_fft.loc[year1_start_date:year2_start_date - pd.DateOffset(days=1)]['quantity'].values
    year2_signal = df_fft.loc[year2_start_date:truncate_date]['quantity'].values
    
    signal = np.concatenate((year1_signal, year2_signal))
    
    signal_fft = fft(signal)
    
    variance_year1 = np.var(year1_signal)
    variance_year2 = np.var(year2_signal)
    
    sum_year1 = year1_signal.sum()
    sum_year2 = year2_signal.sum()
    
    # Apply a low-pass filter to the FFT output to extract the trend line
    cutoff_frequency = int(len(signal) * cutoff)
    filtered_signal_fft = np.copy(signal_fft)
    filtered_signal_fft[cutoff_frequency:len(signal) - cutoff_frequency] = 0

    # Perform inverse FFT to get the trend line
    trend_line = np.real(ifft(filtered_signal_fft))

    # Compute the scaling factor for year 3 based on the variance ratio
    # scaling_factor_year3 = variance_year2 / variance_year1
    scaling_factor_year3 = sum_year2 / sum_year1
    # Scale the trend line for year 3 and concatenate it to the trend line for year 1 and year 2
    trend_line_year3 = trend_line[len(trend_line)//2:] * scaling_factor_year3
    extended_trend_line = np.concatenate((trend_line, trend_line_year3))


    # Truncate the extended trend line according to the date range of the original dataframe
    end_date = df.index.max()
    date_range = pd.date_range(year1_start_date, end_date)
    truncated_trend_line = extended_trend_line[:len(date_range)]

    # Apply a smooth around the truncate_date, each side 15 days
    smooth_start = np.where(date_range == pd.to_datetime(truncate_date))[0][0] - smooth_window
    smooth_end = smooth_start + 2 * smooth_window + 1
    truncated_trend_line[smooth_start:smooth_end] = savgol_filter(truncated_trend_line[smooth_start:smooth_end], 2 * smooth_window + 1, 3)

    # Create a DataFrame for truncated_trend_line
    truncated_trend_line_df = pd.DataFrame(truncated_trend_line, index=date_range, columns=['quantity'])
    truncated_trend_line_df.quantity.ffill(inplace=True)
    truncated_trend_line_df.quantity.bfill(inplace=True)
    return truncated_trend_line_df


def hanning_window_smooth(data, window_size):
    """
    Applies a Hanning window smoothing function to the input data.

    Args:
        data (array-like): Input data to be smoothed.
        window_size (int): The size of the Hanning window.

    Returns:
        np.ndarray: The smoothed data.
    """    
    window = np.hanning(window_size)
    return np.convolve(data, window, mode='same') / sum(window)

# V1
def extend_fft_trend_1(df, truncate_date='2021-07-08', lag_years=2, lead_years=1, trend_cycle_days=60, smooth_window=15):
    """
    Extends the FFT trend of a time series dataset.

    Args:
        df (pd.DataFrame): Input time series DataFrame with a DatetimeIndex and a 'quantity' column.
        truncate_date (str): Date to truncate the DataFrame for FFT analysis. Format: 'YYYY-MM-DD'.
        lag_years (int): Number of years of past data to use for FFT analysis.
        lead_years (int): Number of years of future data to forecast.
        trend_cycle_days (int): Number of days for a trend cycle.
        smooth_window (int): Size of the smoothing window for Hanning window function.

    Returns:
        pd.DataFrame: DataFrame with the extended FFT trend.
    """
    df_fft = df[df.index < truncate_date]
    cutoff = 1 / trend_cycle_days

    # Use calendar date to slice/subset
    year1_start_date = pd.to_datetime(truncate_date) - pd.DateOffset(years=lag_years)
    year2_start_date = pd.to_datetime(truncate_date) - pd.DateOffset(years=lag_years - 1)
    
    year1_signal = df_fft.loc[year1_start_date:year2_start_date - pd.DateOffset(days=1)]['quantity'].values
    year2_signal = df_fft.loc[year2_start_date:truncate_date]['quantity'].values
    
    signal = np.concatenate((year1_signal, year2_signal))
    
    signal_fft = fft(signal)
    
    variance_year1 = np.var(year1_signal)
    variance_year2 = np.var(year2_signal)
    
    sum_year1 = year1_signal.sum()
    sum_year2 = year2_signal.sum()

    # Apply a low-pass filter to the FFT output to extract the trend line
    cutoff_frequency = int(len(signal) * cutoff)
    filtered_signal_fft = np.copy(signal_fft)
    filtered_signal_fft[cutoff_frequency:len(signal) - cutoff_frequency] = 0

    # Perform inverse FFT to get the trend line
    trend_line = np.real(ifft(filtered_signal_fft))

    # Compute the scaling factor for year 3 based on the variance ratio
    scaling_factor_year3 = sum_year2 / sum_year1 

    # Scale the trend line for year 3 and concatenate it to the trend line for year 1 and year 2
    trend_line_year3 = trend_line[len(trend_line)//2:] * scaling_factor_year3

    # Compute the scaling factor for year 0 based on the variance ratio (assuming the variance ratio between year0 and year1 is the same as between year1 and year2)
    scaling_factor_year0 = sum_year1 / sum_year2

    # Scale the trend line for year 0
    trend_line_year0 = trend_line[:len(trend_line)//2] * scaling_factor_year0

    # Concatenate the trend lines for year 0, year 1, year 2, and year 3
    extended_trend_line = np.concatenate((trend_line_year0, trend_line, trend_line_year3))

    # Truncate the extended trend line according to the date range of the original dataframe
    end_date = df.index.max()
    date_range = pd.date_range(year1_start_date - pd.DateOffset(years=1), end_date)
    truncated_trend_line = extended_trend_line[:len(date_range)]


    
    
    
    from scipy.ndimage import gaussian_filter1d
    gaussian_sigma=3
    # Apply Gaussian smoothing around the year1_start_date, each side 15 days
    smooth_start_year1 = np.where(date_range == year1_start_date)[0][0] - smooth_window
    smooth_end_year1 = smooth_start_year1 + 2 * smooth_window + 1
    truncated_trend_line[smooth_start_year1:smooth_end_year1] = gaussian_filter1d(truncated_trend_line[smooth_start_year1:smooth_end_year1], gaussian_sigma)

    # Apply Gaussian smoothing around the truncate_date, each side 15 days
    smooth_start_truncate = np.where(date_range == pd.to_datetime(truncate_date))[0][0] - smooth_window
    smooth_end_truncate = smooth_start_truncate + 2 * smooth_window + 1
    truncated_trend_line[smooth_start_truncate:smooth_end_truncate] = gaussian_filter1d(truncated_trend_line[smooth_start_truncate:smooth_end_truncate], gaussian_sigma)

    # Create a DataFrame for truncated_trend_line
    truncated_trend_line_df = pd.DataFrame(truncated_trend_line, index=date_range, columns=['quantity'])
    truncated_trend_line_df.quantity.ffill(inplace=True)
    truncated_trend_line_df.quantity.bfill(inplace=True)
    return truncated_trend_line_df


    # # Apply smoothing around the year1_start_date, each side 15 days
    # smooth_start_year1 = np.where(date_range == year1_start_date)[0][0] - smooth_window
    # smooth_end_year1 = smooth_start_year1 + 2 * smooth_window + 1
    # truncated_trend_line[smooth_start_year1:smooth_end_year1] = savgol_filter(truncated_trend_line[smooth_start_year1:smooth_end_year1], 2 * smooth_window + 1, 3)

    # # Apply smoothing around the truncate_date, each side 15 days
    # smooth_start_truncate = np.where(date_range == pd.to_datetime(truncate_date))[0][0] - smooth_window
    # smooth_end_truncate = smooth_start_truncate + 2 * smooth_window + 1
    # truncated_trend_line[smooth_start_truncate:smooth_end_truncate] = savgol_filter(truncated_trend_line[smooth_start_truncate:smooth_end_truncate], 2 * smooth_window + 1, 3)

    # # Create a DataFrame for truncated_trend_line
    # truncated_trend_line_df = pd.DataFrame(truncated_trend_line, index=date_range, columns=['quantity'])
    # truncated_trend_line_df.quantity.ffill(inplace=True)
    # truncated_trend_line_df.quantity.bfill(inplace=True)
    # return truncated_trend_line_df
    
    
    
    


    # window_size=31
    # # Apply Hanning window smoothing around the year1_start_date, each side 15 days
    # smooth_start_year1 = np.where(date_range == year1_start_date)[0][0] - smooth_window
    # smooth_end_year1 = smooth_start_year1 + 2 * smooth_window + 1
    # truncated_trend_line[smooth_start_year1:smooth_end_year1] = hanning_window_smooth(truncated_trend_line[smooth_start_year1:smooth_end_year1], window_size)

    # # Apply Hanning window smoothing around the truncate_date, each side 15 days
    # smooth_start_truncate = np.where(date_range == pd.to_datetime(truncate_date))[0][0] - smooth_window
    # smooth_end_truncate = smooth_start_truncate + 2 * smooth_window + 1
    # truncated_trend_line[smooth_start_truncate:smooth_end_truncate] = hanning_window_smooth(truncated_trend_line[smooth_start_truncate:smooth_end_truncate], window_size)

    # # Create a DataFrame for truncated_trend_line
    # truncated_trend_line_df = pd.DataFrame(truncated_trend_line, index=date_range, columns=['quantity'])
    # truncated_trend_line_df.quantity.ffill(inplace=True)
    # truncated_trend_line_df.quantity.bfill(inplace=True)
    # return truncated_trend_line_df
    
    
import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft
from scipy.ndimage import gaussian_filter1d
from scipy.stats import iqr


# V2
def extend_fft_trend_2(df, truncate_date='2021-07-08', lag_years=2, lead_years=1, trend_cycle_days=60, smooth_window=15):
    """
    Extends the FFT trend of a time series dataset with sum and interquartile range scaling.

    Args:
        df (pd.DataFrame): Input time series DataFrame with a DatetimeIndex and a 'quantity' column.
        truncate_date (str): Date to truncate the DataFrame for FFT analysis. Format: 'YYYY-MM-DD'.
        lag_years (int): Number of years of past data to use for FFT analysis.
        lead_years (int): Number of years of future data to forecast.
        trend_cycle_days (int): Number of days for a trend cycle.
        smooth_window (int): Size of the smoothing window for Gaussian filter.

    Returns:
        pd.DataFrame: DataFrame with the extended FFT trend.
    """
    df_fft = df[df.index < truncate_date]
    cutoff = 1 / trend_cycle_days

    # Use calendar date to slice/subset
    year1_start_date = pd.to_datetime(truncate_date) - pd.DateOffset(years=lag_years)
    year2_start_date = pd.to_datetime(truncate_date) - pd.DateOffset(years=lag_years - 1)
    
    year1_signal = df_fft.loc[year1_start_date:year2_start_date - pd.DateOffset(days=1)]['quantity'].values
    year2_signal = df_fft.loc[year2_start_date:truncate_date]['quantity'].values
    
    signal = np.concatenate((year1_signal, year2_signal))
    
    signal_fft = fft(signal)
    
    variance_year1 = np.var(year1_signal)
    variance_year2 = np.var(year2_signal)
    
    sum_year1 = year1_signal.sum()
    sum_year2 = year2_signal.sum()


    median_year1 = np.median(year1_signal)
    median_year2 = np.median(year2_signal)
    
    
    # Apply a low-pass filter to the FFT output to extract the trend line
    cutoff_frequency = int(len(signal) * cutoff)
    filtered_signal_fft = np.copy(signal_fft)
    filtered_signal_fft[cutoff_frequency:len(signal) - cutoff_frequency] = 0

    # Perform inverse FFT to get the trend line
    trend_line = np.real(ifft(filtered_signal_fft))

    # Compute the scaling factor for year 3 based on the sum ratio
    scaling_factor_year3 = sum_year2 / sum_year1 

    # Scale the trend line for year 3 and concatenate it to the trend line for year 1 and year 2
    trend_line_year3 = trend_line[len(trend_line)//2:] * scaling_factor_year3

    # Compute the scaling factor for year 0 based on the sum ratio (assuming the sum ratio between year0 and year1 is the same as between year1 and year2)
    scaling_factor_year0 = sum_year1 / sum_year2

    # Scale the trend line for year 0
    trend_line_year0 = trend_line[:len(trend_line)//2] * scaling_factor_year0

    # Concatenate the trend lines for year 0, year 1, year 2, and year 3
    extended_trend_line = np.concatenate((trend_line_year0, trend_line, trend_line_year3))

    # Truncate the extended trend line according to the date range of the original dataframe
    end_date = df.index.max()
    date_range = pd.date_range(year1_start_date - pd.DateOffset(years=1), end_date)
    truncated_trend_line = extended_trend_line[:len(date_range)]

    # Calculate translation factors based on the interquartile range
    translation_factor_year3 = iqr(year2_signal) - iqr(year1_signal)
    translation_factor_year0 = iqr(year1_signal) - iqr(year2_signal)

    # Apply translation factors to year 3 and year 0
    trend_line_year3 += translation_factor_year3
    trend_line_year0 += translation_factor_year0

    # Smooth the trend line using Gaussian filtering
    gaussian_sigma = 3

    # Apply Gaussian smoothing around the year1_start_date, each side 15 days
    smooth_start_year1 = np.where(date_range == year1_start_date)[0][0] - smooth_window
    smooth_end_year1 = smooth_start_year1 + 2 * smooth_window + 1
    truncated_trend_line[smooth_start_year1:smooth_end_year1] = gaussian_filter1d(truncated_trend_line[smooth_start_year1:smooth_end_year1], gaussian_sigma)

    # Apply Gaussian smoothing around the truncate_date, each side 15 days
    smooth_start_truncate = np.where(date_range == pd.to_datetime(truncate_date))[0][0] - smooth_window
    smooth_end_truncate = smooth_start_truncate + 2 * smooth_window + 1
    truncated_trend_line[smooth_start_truncate:smooth_end_truncate] = gaussian_filter1d(truncated_trend_line[smooth_start_truncate:smooth_end_truncate], gaussian_sigma)

    # Create a DataFrame for truncated_trend_line
    truncated_trend_line_df = pd.DataFrame(truncated_trend_line, index=date_range, columns=['quantity'])
    truncated_trend_line_df.quantity.ffill(inplace=True)
    truncated_trend_line_df.quantity.bfill(inplace=True)
    return truncated_trend_line_df

    
    
# v3
def extend_fft_trend_3(df, truncate_date='2021-07-08', lag_years=2, lead_years=1, trend_cycle_days=60, smooth_window=15):
    """
    Extends the FFT trend of a time series dataset with median and interquartile range scaling.

    Args:
        df (pd.DataFrame): Input time series DataFrame with a DatetimeIndex and a 'quantity' column.
        truncate_date (str): Date to truncate the DataFrame for FFT analysis. Format: 'YYYY-MM-DD'.
        lag_years (int): Number of years of past data to use for FFT analysis.
        lead_years (int): Number of years of future data to forecast.
        trend_cycle_days (int): Number of days for a trend cycle.
        smooth_window (int): Size of the smoothing window for Gaussian filter.

    Returns:
        pd.DataFrame: DataFrame with the extended FFT trend.
    """
    df_fft = df[df.index < truncate_date]
    cutoff = 1 / trend_cycle_days

    # Use calendar date to slice/subset
    year1_start_date = pd.to_datetime(truncate_date) - pd.DateOffset(years=lag_years)
    year2_start_date = pd.to_datetime(truncate_date) - pd.DateOffset(years=lag_years - 1)
    
    year1_signal = df_fft.loc[year1_start_date:year2_start_date - pd.DateOffset(days=1)]['quantity'].values
    year2_signal = df_fft.loc[year2_start_date:truncate_date]['quantity'].values
    
    signal = np.concatenate((year1_signal, year2_signal))
    
    signal_fft = fft(signal)
    
    variance_year1 = np.var(year1_signal)
    variance_year2 = np.var(year2_signal)
    
    sum_year1 = year1_signal.sum()
    sum_year2 = year2_signal.sum()


    median_year1 = np.median(year1_signal)
    median_year2 = np.median(year2_signal)

    
    
    # Apply a low-pass filter to the FFT output to extract the trend line
    cutoff_frequency = int(len(signal) * cutoff)
    filtered_signal_fft = np.copy(signal_fft)
    filtered_signal_fft[cutoff_frequency:len(signal) - cutoff_frequency] = 0

    # Perform inverse FFT to get the trend line
    trend_line = np.real(ifft(filtered_signal_fft))

    # Compute the scaling factor for year 3 based on the sum ratio
    scaling_factor_year3 = median_year2 / median_year1 

    # Scale the trend line for year 3 and concatenate it to the trend line for year 1 and year 2
    trend_line_year3 = trend_line[len(trend_line)//2:] * scaling_factor_year3

    # Compute the scaling factor for year 0 based on the sum ratio (assuming the sum ratio between year0 and year1 is the same as between year1 and year2)
    scaling_factor_year0 = median_year1 / median_year2

    # Scale the trend line for year 0
    trend_line_year0 = trend_line[:len(trend_line)//2] * scaling_factor_year0

    # Concatenate the trend lines for year 0, year 1, year 2, and year 3
    extended_trend_line = np.concatenate((trend_line_year0, trend_line, trend_line_year3))

    # Truncate the extended trend line according to the date range of the original dataframe
    end_date = df.index.max()
    date_range = pd.date_range(year1_start_date - pd.DateOffset(years=1), end_date)
    truncated_trend_line = extended_trend_line[:len(date_range)]

    # Calculate translation factors based on the interquartile range
    translation_factor_year3 = iqr(year2_signal) - iqr(year1_signal)
    translation_factor_year0 = iqr(year1_signal) - iqr(year2_signal)

    # Apply translation factors to year 3 and year 0
    trend_line_year3 += translation_factor_year3
    trend_line_year0 += translation_factor_year0

    # Smooth the trend line using Gaussian filtering
    gaussian_sigma = 3

    # Apply Gaussian smoothing around the year1_start_date, each side 15 days
    smooth_start_year1 = np.where(date_range == year1_start_date)[0][0] - smooth_window
    smooth_end_year1 = smooth_start_year1 + 2 * smooth_window + 1
    truncated_trend_line[smooth_start_year1:smooth_end_year1] = gaussian_filter1d(truncated_trend_line[smooth_start_year1:smooth_end_year1], gaussian_sigma)

    # Apply Gaussian smoothing around the truncate_date, each side 15 days
    smooth_start_truncate = np.where(date_range == pd.to_datetime(truncate_date))[0][0] - smooth_window
    smooth_end_truncate = smooth_start_truncate + 2 * smooth_window + 1
    truncated_trend_line[smooth_start_truncate:smooth_end_truncate] = gaussian_filter1d(truncated_trend_line[smooth_start_truncate:smooth_end_truncate], gaussian_sigma)

    # Create a DataFrame for truncated_trend_line
    truncated_trend_line_df = pd.DataFrame(truncated_trend_line, index=date_range, columns=['quantity'])
    truncated_trend_line_df.quantity.ffill(inplace=True)
    truncated_trend_line_df.quantity.bfill(inplace=True)
    return truncated_trend_line_df

    
from scipy.stats import iqr
# v4
def extend_fft_trend_4(df, truncate_date='2021-07-08', lag_years=2, lead_years=1, trend_cycle_days=60, smooth_window=15):
    """
    Extends the FFT trend of a time series dataset with interquartile range scaling and sum-based translation.

    Args:
        df (pd.DataFrame): Input time series DataFrame with a DatetimeIndex and a 'quantity' column.
        truncate_date (str): Date to truncate the DataFrame for FFT analysis. Format: 'YYYY-MM-DD'.
        lag_years (int): Number of years of past data to use for FFT analysis.
        lead_years (int): Number of years of future data to forecast.
        trend_cycle_days (int): Number of days for a trend cycle.
        smooth_window (int): Size of the smoothing window for Gaussian filter.

    Returns:
        pd.DataFrame: DataFrame with the extended FFT trend.
    """    
    df_fft = df[df.index < truncate_date]
    cutoff = 1 / trend_cycle_days

    # Use calendar date to slice/subset
    year1_start_date = pd.to_datetime(truncate_date) - pd.DateOffset(years=lag_years)
    year2_start_date = pd.to_datetime(truncate_date) - pd.DateOffset(years=lag_years - 1)
    
    year1_signal = df_fft.loc[year1_start_date:year2_start_date - pd.DateOffset(days=1)]['quantity'].values
    year2_signal = df_fft.loc[year2_start_date:truncate_date]['quantity'].values
    
    signal = np.concatenate((year1_signal, year2_signal))
    
    signal_fft = fft(signal)
    
    sum_year1 = year1_signal.sum()
    sum_year2 = year2_signal.sum()

    iqr_year1 = iqr(year1_signal)
    iqr_year2 = iqr(year2_signal)

    # Apply a low-pass filter to the FFT output to extract the trend line
    cutoff_frequency = int(len(signal) * cutoff)
    filtered_signal_fft = np.copy(signal_fft)
    filtered_signal_fft[cutoff_frequency:len(signal) - cutoff_frequency] = 0

    # Perform inverse FFT to get the trend line
    trend_line = np.real(ifft(filtered_signal_fft))

    # Compute the scaling factor for year 3 based on the interquartile range ratio
    scaling_factor_year3 = iqr_year2 / iqr_year1

    # Scale the trend line for year 3 and concatenate it to the trend line for year 1 and year 2
    trend_line_year3 = trend_line[len(trend_line)//2:] * scaling_factor_year3

    # Compute the scaling factor for year 0 based on the interquartile range ratio (assuming the interquartile range ratio between year0 and year1 is the same as between year1 and year2)
    scaling_factor_year0 = iqr_year1 / iqr_year2

    # Scale the trend line for year 0
    trend_line_year0 = trend_line[:len(trend_line)//2] * scaling_factor_year0

    # Concatenate the trend lines for year 0, year 1, year 2, and year 3
    extended_trend_line = np.concatenate((trend_line_year0, trend_line, trend_line_year3))

    # Truncate the extended trend line according to the date range of the original dataframe
    end_date = df.index.max()
    date_range = pd.date_range(year1_start_date - pd.DateOffset(years=1), end_date)
    truncated_trend_line = extended_trend_line[:len(date_range)]

    # Calculate translation factors based on the sum
    translation_factor_year3 = sum_year2 - sum_year1
    translation_factor_year0 = sum_year1 - sum_year2

    # Apply translation factors to year 3 and year 0
    trend_line_year3 += translation_factor_year3
    trend_line_year0 += translation_factor_year0

    # Smooth the trend line using Gaussian filtering
    gaussian_sigma = 3

    # Apply Gaussian smoothing around the year1_start_date, each side 15 days
    smooth_start_year1 = np.where(date_range == year1_start_date)[0][0] - smooth_window
    smooth_end_year1 = smooth_start_year1 + 2 * smooth_window + 1
    truncated_trend_line[smooth_start_year1:smooth_end_year1] = gaussian_filter1d(truncated_trend_line[smooth_start_year1:smooth_end_year1], gaussian_sigma)

    # Apply Gaussian smoothing around the truncate_date, each side 15 days
    smooth_start_truncate = np.where(date_range == pd.to_datetime(truncate_date))[0][0] - smooth_window
    smooth_end_truncate = smooth_start_truncate + 2 * smooth_window + 1
    truncated_trend_line[smooth_start_truncate:smooth_end_truncate] = gaussian_filter1d(truncated_trend_line[smooth_start_truncate:smooth_end_truncate], gaussian_sigma)

    # Create a DataFrame for truncated_trend_line
    truncated_trend_line_df = pd.DataFrame(truncated_trend_line, index=date_range, columns=['quantity'])
    truncated_trend_line_df.quantity.ffill(inplace=True)
    truncated_trend_line_df.quantity.bfill(inplace=True)
    return truncated_trend_line_df



# V5
def extend_fft_trend_5(df, truncate_date='2021-07-08', lag_years=2, lead_years=1, trend_cycle_days=60, smooth_window=15):
    """
    Extends the FFT trend of a time series dataset with interquartile range scaling and median-based translation.

    Args:
        df (pd.DataFrame): Input time series DataFrame with a DatetimeIndex and a 'quantity' column.
        truncate_date (str): Date to truncate the DataFrame for FFT analysis. Format: 'YYYY-MM-DD'.
        lag_years (int): Number of years of past data to use for FFT analysis.
        lead_years (int): Number of years of future data to forecast.
        trend_cycle_days (int): Number of days for a trend cycle.
        smooth_window (int): Size of the smoothing window for Gaussian filter.

    Returns:
        pd.DataFrame: DataFrame with the extended FFT trend.
    """

    df_fft = df[df.index < truncate_date]
    cutoff = 1 / trend_cycle_days

    # Use calendar date to slice/subset
    year1_start_date = pd.to_datetime(truncate_date) - pd.DateOffset(years=lag_years)
    year2_start_date = pd.to_datetime(truncate_date) - pd.DateOffset(years=lag_years - 1)
    
    year1_signal = df_fft.loc[year1_start_date:year2_start_date - pd.DateOffset(days=1)]['quantity'].values
    year2_signal = df_fft.loc[year2_start_date:truncate_date]['quantity'].values
    
    signal = np.concatenate((year1_signal, year2_signal))
    
    signal_fft = fft(signal)
    
    median_year1 = np.median(year1_signal)
    median_year2 = np.median(year2_signal)

    iqr_year1 = iqr(year1_signal)
    iqr_year2 = iqr(year2_signal)

    # Apply a low-pass filter to the FFT output to extract the trend line
    cutoff_frequency = int(len(signal) * cutoff)
    filtered_signal_fft = np.copy(signal_fft)
    filtered_signal_fft[cutoff_frequency:len(signal) - cutoff_frequency] = 0

    # Perform inverse FFT to get the trend line
    trend_line = np.real(ifft(filtered_signal_fft))

    # Compute the scaling factor for year 3 based on the interquartile range ratio
    scaling_factor_year3 = iqr_year2 / iqr_year1

    # Scale the trend line for year 3 and concatenate it to the trend line for year 1 and year 2
    trend_line_year3 = trend_line[len(trend_line)//2:] * scaling_factor_year3

    # Compute the scaling factor for year 0 based on the interquartile range ratio (assuming the interquartile range ratio between year0 and year1 is the same as between year1 and year2)
    scaling_factor_year0 = iqr_year1 / iqr_year2

    # Scale the trend line for year 0
    trend_line_year0 = trend_line[:len(trend_line)//2] * scaling_factor_year0

    # Concatenate the trend lines for year 0, year 1, year 2, and year 3
    extended_trend_line = np.concatenate((trend_line_year0, trend_line, trend_line_year3))

    # Truncate the extended trend line according to the date range of the original dataframe
    end_date = df.index.max()
    date_range = pd.date_range(year1_start_date - pd.DateOffset(years=1), end_date)
    truncated_trend_line = extended_trend_line[:len(date_range)]

    # Calculate translation factors based on the median
    translation_factor_year3 = median_year2 - median_year1
    translation_factor_year0 = median_year1 - median_year2

    # Apply translation factors to year 3 and year 0
    trend_line_year3 += translation_factor_year3
    trend_line_year0 += translation_factor_year0

    # Smooth the trend line using Gaussian filtering
    gaussian_sigma = 3

    # Apply Gaussian smoothing around the year1_start_date, each side 15 days
    smooth_start_year1 = np.where(date_range == year1_start_date)[0][0] - smooth_window
    smooth_end_year1 = smooth_start_year1 + 2 * smooth_window + 1
    truncated_trend_line[smooth_start_year1:smooth_end_year1] = gaussian_filter1d(truncated_trend_line[smooth_start_year1:smooth_end_year1], gaussian_sigma)

    # Apply Gaussian smoothing around the truncate_date, each side 15 days
    smooth_start_truncate = np.where(date_range == pd.to_datetime(truncate_date))[0][0] - smooth_window
    smooth_end_truncate = smooth_start_truncate + 2 * smooth_window + 1
    truncated_trend_line[smooth_start_truncate:smooth_end_truncate] = gaussian_filter1d(truncated_trend_line[smooth_start_truncate:smooth_end_truncate], gaussian_sigma)

    # Create a DataFrame for truncated_trend_line
    truncated_trend_line_df = pd.DataFrame(truncated_trend_line, index=date_range, columns=['quantity'])
    truncated_trend_line_df.quantity.ffill(inplace=True)
    truncated_trend_line_df.quantity.bfill(inplace=True)
    return truncated_trend_line_df




