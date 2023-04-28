
# data_preprocessing.py

# immport libraries

from tabulate import tabulate

import itertools
import pandas as pd
from typing import List, Union


def fix_zero_value_by_cross_reference(df, col, key):
    """
    Replaces zero values in a DataFrame column with the median non-zero value for the corresponding key.

    Args:
        df (pandas.DataFrame): The DataFrame to be modified.
        col (str): The name of the column to be modified.
        key (str): The name of the column containing the key used to group the non-zero values.

    Returns:
        pandas.DataFrame: The modified DataFrame.

    Example:
        fixed_df = fix_zero_value_by_cross_reference(df, 'unit_rrp_vat_excl', 'item_code')
    """    
    
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Find non-zero values for each key
    non_zero_values = df[df[col] != 0].groupby(key)[col].median().to_dict()

    # Replace zero values with non-zero values for the same key
    df[col] = df.apply(lambda row: non_zero_values.get(row[key], row[col]) if row[col] == 0 else row[col], axis=1)

    return df



def evaluate_zero_value_fixability(df, col, key):
    """
    Evaluate the fixability of 0 values in a column based on values in another column with a common key.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        col (str): The name of the column to fix.
        key (str): The name of the column containing the common key.

    Returns:
        None. Prints the number of 0 values, the percentage of 0 values, the number of fixable rows,
        the percentage of fixable rows, and the percentage of unfixable rows as a percentage of all rows.

    """    
    zero_count = len(df[df[col] == 0])
    print(f'The number of 0 values in {col} is:')
    print(zero_count)
    print(f'The percentage of 0 values in {col} is:')
    print(zero_count / df.shape[0] * 100, '%')

    # Find non-zero rows for each key
    non_zero_values = df[df[col] != 0].groupby(key).size()

    # Filter the DataFrame for rows with zero values
    zero_values_df = df[df[col] == 0]

    # Get the key count for rows with zero values
    zero_values_counts = zero_values_df.groupby(key).size()

    # Find the intersection of keys with both zero and non-zero values
    common_keys = zero_values_counts.index.intersection(non_zero_values.index)

    # Calculate the number of fixable rows by summing the counts of common keys
    fixable_rows = zero_values_counts[common_keys].sum()

    print(f'By using the {col} values for other rows with the same {key}, we can fix:')
    print(fixable_rows)

    print('The percentage of fixable rows is:')
    print(fixable_rows / zero_count * 100, '%')

    print('The percentage of unfixable rows as a percentage of all rows is:')
    print((zero_count - fixable_rows) / df.shape[0] * 100, '%')

# Usage example
# evaluate_fixability(orders, 'unit_cogs', 'item_code')


def unique_combinations_counts(df, col1, col2):
    """
    Groups a pandas DataFrame by two columns, counts the occurrences of each unique combination, and returns the counts in a new DataFrame.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame to be grouped and counted.
    col1 : str
        The name of the first column to group by.
    col2 : str
        The name of the second column to group by.

    Returns:
    -------
    pandas.DataFrame
        A new DataFrame with three columns: col1, col2, and 'count', which represents the number of times the combination of col1 and col2 occurred in the original DataFrame. The DataFrame is sorted by the count column in descending order, with the most common combination appearing at the top.
    """
    
    

    # Group by col1 and col2 and count the occurrences of each unique combination
    unique_combinations = df.groupby([col1, col2]).size().reset_index().rename(columns={0:'count'})

    return unique_combinations

import itertools

def check_all_column_pairings(df):
    """
    Checks all possible pairs of columns in the input DataFrame to see if they are paired 1 to 1.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame to check for column pairing.

    Returns:
    -------
    None
        Prints a message for each pair of columns that are 1-to-1 paired.
    """

    def check_pairing(df, col1, col2):
        unique_combinations = unique_combinations_counts(df, col1, col2)

        if (unique_combinations.shape[0] == unique_combinations[col1].nunique()) and (unique_combinations.shape[0] == unique_combinations[col2].nunique()):
            print(f"{col1} and {col2} are paired 1-to-1")
        else:
            pass

    # Get all possible combinations of two columns
    column_combinations = list(itertools.combinations(df.columns, 2))

    # Check pairing for all combinations
    for col1, col2 in column_combinations:
        check_pairing(df, col1, col2)


def print_df_with_dtypes(df: pd.DataFrame, max_rows: int = 500, print_on: bool = True, data_generated_path: str = '', name_extension: str = '') -> None:
    """
    Prints a pandas DataFrame with data types and NaN values for each column.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame to be printed.
    max_rows : int, optional (default=500)
        The maximum number of rows to print. If the input DataFrame has more rows than max_rows, only the first max_rows will be printed.
    print_on : bool, optional (default=True)
        A boolean parameter indicating whether to print the DataFrame or not. If print_on is False, the function does not print anything.
    name_extension : str, optional (default='')
        A string parameter representing the extension to be added to the file name. If name_extension is an empty string, the file name will not be modified.
    data_generated_path : str, optional (default='')
        A string parameter representing the path where the generated file will be saved. If not provided, uses the global variable DATA_GENERATED_PATH defined in the ETL notebook.

    Returns:
    -------
    None
    """
    if data_generated_path == '':
        data_generated_path = DATA_GENERATED_PATH
    
    # Create a new DataFrame with the same data but with updated column names
    new_column_names = [f"{col}\n{dtype}\nNaN: {df[col].isna().sum()}" for col, dtype in zip(df.columns, df.dtypes)]
    df_with_dtypes = df.head(max_rows).copy()
    df_with_dtypes.columns = new_column_names

    # Print the updated DataFrame using the 'psql' table format if print_on is True
    if print_on:
        print(tabulate(df_with_dtypes, headers='keys', tablefmt='psql'))

    # Save the updated DataFrame to a txt file with an optional name extension
    file_name = 'df_head_' + str(max_rows) + '_' + name_extension + '.txt'
    with open(data_generated_path + file_name, 'w') as f:
        f.write(tabulate(df_with_dtypes, headers='keys', tablefmt='psql'))


import pandas as pd

def data_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input pandas DataFrame by dropping duplicate rows, converting data types, and filling missing values.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame to be cleaned.

    Returns:
    -------
    pandas.DataFrame
        A cleaned DataFrame with updated data types and filled missing values.
    """
    # Drop duplicate rows considering only the columns, not the index, and apply the changes directly to the original DataFrame
    rows_before_drop_duplicates = df.shape[0]
    df.drop_duplicates(inplace=True)
    rows_after_drop_duplicates = df.shape[0]

    # Print some statistics on the number of rows dropped
    print('The number of rows in the merged dataframe is: ', rows_before_drop_duplicates)
    print('The number of rows in the merged dataframe after dropping duplicates is: ', rows_after_drop_duplicates)
    print('The number of rows dropped is: ', rows_before_drop_duplicates - rows_after_drop_duplicates)
    print('The percentage of rows dropped is: ', (rows_before_drop_duplicates - rows_after_drop_duplicates) / rows_before_drop_duplicates * 100, '%')

    # Convert the 'date' and 'CreatedAt' columns to datetime format
    df['date'] = pd.to_datetime(df['date'])
    df['CreatedAt'] = pd.to_datetime(df['CreatedAt'])
    # Print the progress message
    print('The data types of the date and CreatedAt columns have been converted to datetime.')
    
    # Set the 'date' column as the index
    df.set_index('date', inplace=True)
    # Print the progress message
    print('The date column has been set as the index.')
    
    # Sort the DataFrame by the index
    df.sort_index(inplace=True)
    # Print the progress message
    print('The DataFrame has been sorted by the index.')
    
    # Replace NaN values with 0 in gift_quantity column
    df['gift_quantity'] = df['gift_quantity'].fillna(0)
    # Print the progress message
    print('The NaN values in the gift_quantity column have been replaced with 0.')

    # Change the datatype of gift_quantity to int64
    df['gift_quantity'] = df['gift_quantity'].astype('int64')
    # Print the progress message
    print('The datatype of the gift_quantity column has been changed to int64.')

    # Replace "- žádný výrobce -" with "no_manufacturer" in the name column
    df['name'] = df['name'].replace('- žádný výrobce -', 'no_manufacturer')
    # Print the progress message
    print('The "- žádný výrobce -" values in the name column have been replaced with "no_manufacturer".')
    
    # Fill missing values with 'unspecified' and convert to string for selected columns
    df['payment'] = df['payment'].fillna('unspecified').astype(str)
    df['group0_id'] = df['group0_id'].fillna(0).astype(int).astype(str)
    df['group1_id'] = df['group1_id'].fillna(0).astype(int).astype(str)
    df['group2_id'] = df['group2_id'].fillna(0).astype(int).astype(str)
    # Print the progress message
    print('The missing values in the payment, group0_id, group1_id, and group2_id columns have been filled with "unspecified" and converted to string.')
    
    # Rename 'name' to 'brand_name'
    df.rename(columns={'name': 'brand_name'}, inplace=True)
    # Print the progress message
    print('The name column has been renamed to brand_name.')
    
    # Convert selected columns to object
    df['order_id'] = df['order_id'].astype('string')
    df['brand_id'] = df['brand_id'].astype('string')
    df['group0_id'] = df['group0_id'].astype('string')
    df['group1_id'] = df['group1_id'].astype('string')
    df['group2_id'] = df['group2_id'].astype('string')
    # Print the progress message
    print('The order_id, brand_id, group0_id, group1_id, and group2_id columns have been converted to object.')
    
    # Replace the NaN in unit_rrp_vat_excl column to 0
    df['unit_rrp_vat_excl'] = df['unit_rrp_vat_excl'].fillna(0)
    
    # Fill missing values with 'unspecified' and convert to string for remaining columns with object data type
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('unspecified').astype(str)

    return df
    # Print the progress message
    print('The missing values in the remaining columns with object data type have been filled with "unspecified" and converted to string.')


import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from calendar import month_abbr

def plot_seasonality_boxplot(df, x, y, hue, cycle, measure):
    """
    Plots a boxplot of the distribution of the given variable (y) by cycle (month, quarter, etc.) and a categorical
    variable (hue). The plot allows comparing the distribution of the variable across the different cycles, as well
    as across the categories of the hue variable.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to be plotted.
    x : str
        The name of the column in the dataframe containing the date values.
    y : str
        The name of the column in the dataframe containing the variable to be plotted.
    hue : str
        The name of the column in the dataframe containing the categorical variable to be used for grouping the data.
    cycle : str
        The name of the cycle to group the data by. Supported values are: monthly, quarterly, seasonly, weekly,
        day_of_year, day_of_month, week_of_year.
    measure : str
        The aggregation method to use for grouping the data. Supported values are: mean, sum.

    Returns:
    --------
    None
    """
        
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Convert the index to a datetime format
    df[x] = pd.to_datetime(df.index)

    _marker_size = 18
    # Aggregate data based on the given cycle
    if cycle == 'monthly':
        df['cycle'] = df[x].dt.month
        month_names = [month_abbr[i] for i in range(1, 13)]
        df['cycle'] = pd.Categorical(df['cycle'], categories=range(1, 13), ordered=True)
        df['cycle'] = df['cycle'].apply(lambda m: month_names[m-1])
    elif cycle == 'quarterly':
        df['cycle'] = df[x].dt.to_period('Q').dt.strftime('Q%q')
    elif cycle == 'seasonly':
        df['cycle'] = (df[x].dt.month % 12 + 3) // 3
        seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
        df['cycle'] = df['cycle'].map(seasons)
        season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
        df['cycle'] = pd.Categorical(df['cycle'], categories=season_order, ordered=True)
    elif cycle == 'weekly':
        df['cycle'] = df[x].dt.day_name()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['cycle'] = pd.Categorical(df['cycle'], categories=weekday_order, ordered=True)
    elif cycle == 'day_of_year':
        df['cycle'] = df[x].dt.dayofyear
        _marker_size = 8
    elif cycle == 'day_of_month':
        df['cycle'] = df[x].dt.day
    elif cycle == 'week_of_year':
        df['cycle'] = df[x].dt.isocalendar().week
        _marker_size = 12        
    else:
        raise ValueError('Invalid cycle value. Supported values are: monthly, quarterly, seasonly, weekly, daily, week_of_year.')

    # Aggregate data based on the given measure
    if measure == 'mean':
        df_agg = df.groupby([hue, 'cycle'])[y].mean().reset_index()
    elif measure == 'sum':
        df_agg = df.groupby([hue, 'cycle'])[y].sum().reset_index()
    else:
        raise ValueError('Invalid measure value. Supported values are: mean, sum.')

    # Sort the legend based on the average value
    df_avg = df_agg.groupby(hue)[y].mean().sort_values(ascending=False).reset_index()
    hue_order = df_avg[hue].tolist()

    cycles = df_agg['cycle'].unique()
    box_color = 'rgba(128, 128, 128, 0.5)'  # Transparent grey
    marker_colors = px.colors.qualitative.Plotly

    fig = go.Figure()

    for idx, _cycle in enumerate(cycles):
        cycle_data = df_agg[df_agg['cycle'] == _cycle]

        # Sort the hue values based on the average value in the cycle_data
        hue_data_mean = cycle_data.groupby(hue)[y].mean().reset_index()
        hue_data_mean = hue_data_mean.sort_values(by=y, ascending=False)
        hue_order_cycle = hue_data_mean[hue].tolist()

        fig.add_trace(go.Box(
            x=cycle_data['cycle'],
            y=cycle_data[y],
            name=str(_cycle),  # Convert the cycle name to a string
            marker_color=box_color,
            showlegend=False
        ))
        
        for hue_idx, hue_value in enumerate(hue_order_cycle):
            hue_data = cycle_data[cycle_data[hue] == hue_value]
            fig.add_trace(go.Scatter(
                x=hue_data['cycle'],
                y=hue_data[y],
                mode='markers',
                marker_color=marker_colors[hue_idx % len(marker_colors)],
                marker_size=_marker_size,
                name=f'{hue_value}',
                legendgroup=hue_value,
                showlegend=(idx == 0)  # Show legend only for the first cycle to avoid duplicate entries
            ))

    # Customize the appearance of the plot
    fig.update_layout(
        xaxis=dict(type='category'),
        boxmode='overlay',
        plot_bgcolor='white',
        font=dict(size=12, color='black', family='Arial'),
        title={
            'text': f'{measure.capitalize()} "{y}" Distribution by {cycle.capitalize()} and "{hue}"',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            title=dict(text=f"{hue}", font=dict(size=14)),
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=1.1,
            bgcolor='rgba(0.1,0.1,0.1,0.1)',
            bordercolor='black',
            borderwidth=1,
            traceorder='reversed',
            font=dict(size=12),
            itemsizing='constant',
            itemwidth=70,  # Reduce this value to control the width of the legend box
            itemclick="toggleothers"
        ),
        width=1600,
        height=900,
        yaxis=dict(gridcolor='lightgrey')
    )

    # Reverse the order of legend items
    fig.update_layout(legend=dict(traceorder="reversed+grouped"))

    fig.show()



