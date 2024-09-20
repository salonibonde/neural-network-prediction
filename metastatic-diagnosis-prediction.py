import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
%matplotlib inline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

def analyze_temperature_data(df):
    """
    This function performs data analysis on a DataFrame containing temperature data. It follows these steps:
    1. Checks and displays the count of null values in each column.
    2. Explores and processes the average temperature columns.
    3. Reshapes the DataFrame for easier analysis.
    4. Analyzes and plots temperature data for each unique patient_zip3.
    5. Decomposes the time series data and checks for stationarity using the Augmented Dickey-Fuller test.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing temperature data.

    Returns:
    pd.DataFrame: A reshaped DataFrame with month_year in string format.
    """
    
    # Step 1: Check and display null values
    null_counts = df.isnull().sum()
    null_counts_df = pd.DataFrame({'Null Values Count': null_counts, 'Data Type': df.dtypes})
    print("Total count of Null Values:", null_counts.sum())
    display(null_counts_df)

    # Step 2: Explore the Average Temperature Column
    ts_cols = [col for col in df.columns if col.startswith('Average')]
    temp_cols = ['patient_zip3'] + ts_cols
    temp_df = df[temp_cols].drop_duplicates()
    temp_df = temp_df[temp_df.drop(columns=['patient_zip3']).isnull().any(axis=1)].reset_index(drop=True)
    
    # Step 3: Reshape the DataFrame
    melted_df = pd.melt(temp_df, id_vars=['patient_zip3'], var_name='month_year', value_name='avg_temp')
    melted_df['month_year'] = melted_df['month_year'].str.extract('Average of (.*)')
    
    # Convert 'month_year' to datetime format
    melted_df['month_year'] = pd.to_datetime(melted_df['month_year'], format='%b-%y')
    melted_df = melted_df.sort_values(by=['patient_zip3', 'month_year']).reset_index(drop=True)
    
    # Step 4: Analyze and plot the data for each unique patient_zip3
    unique_zips = melted_df['patient_zip3'].unique()
    
    for zip_code in unique_zips:
        # Filter the DataFrame for the current zip code
        zip_df = melted_df[melted_df['patient_zip3'] == zip_code].copy()

        # Set 'month_year' as the index and sort by index
        zip_df.set_index('month_year', inplace=True)
        zip_df.sort_index(inplace=True)

        # Plot the average temperature over time
        plt.figure(figsize=(16, 7))
        plt.plot(zip_df.index, zip_df['avg_temp'], marker='s')
        plt.xlabel("Time")
        plt.ylabel("Average Temperature")
        plt.title(f"Average Temperature over time - zip {zip_code}")
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.show()

        # Dropping rows with missing values
        df_no_null = zip_df.dropna()

        # Decomposing the series
        decomposition = seasonal_decompose(df_no_null['avg_temp'], model='additive', period=12)  # Yearly seasonality

        # Plotting the decomposition
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
        decomposition.trend.plot(ax=ax1, color='blue')
        ax1.set_ylabel('Trend')
        ax1.set_title('Seasonal Decomposition of Time Series')

        decomposition.seasonal.plot(ax=ax2, color='green')
        ax2.set_ylabel('Seasonality')

        decomposition.resid.plot(ax=ax3, color='orange')
        ax3.set_ylabel('Residuals')

        zip_df['avg_temp'].plot(ax=ax4, color='red')
        ax4.set_ylabel('Observed')
        ax4.set_xlabel('Date')

        plt.tight_layout()
        plt.show()

        # Checking stationarity with the Augmented Dickey-Fuller test
        adf_result = adfuller(df_no_null['avg_temp'])
        adf_statistic, p_value = adf_result[0], adf_result[1]
        print(f"ADF Statistic for zip {zip_code}: {adf_statistic}, p-value: {p_value}")

def categorical_eda(train_df):
    """
    Perform exploratory data analysis (EDA) on the categorical columns of the given DataFrame and generate various visualizations and grouping summaries.

    Parameters:
    train_df (pandas.DataFrame): The input DataFrame containing the data to be analyzed.

    The function performs the following tasks:
    1. Generates count plots for all categorical columns.
    2. Identifies and prints zip codes that are incorrectly mapped to multiple states.
    3. Identifies and prints states mapped to multiple regions.
    4. Identifies and prints states mapped to multiple divisions.
    5. Groups the data by 'patient_zip3', 'patient_race', and 'patient_state' to find the most frequent payer type for each group 
       and generates a count plot showing the distribution of the most frequent payer types across different races.
    6. Generates batch plots to show the distribution of 'patient_race' for each unique state.
    7. Groups the data by 'breast_cancer_diagnosis_code' and 'breast_cancer_diagnosis_desc' and 
       displays the count of 'patient_id' for each group.

    The function does not return any value. It displays the plots and prints the grouping summaries directly.
    """
    
    categorical_cols = train_df.select_dtypes(include=['object']).columns

    plt.figure(figsize=(20, len(categorical_cols) * 5))

    for i, col in enumerate(categorical_cols):
        plt.subplot(len(categorical_cols) // 2 + len(categorical_cols) % 2, 2, i + 1)
        ax = sns.countplot(y=col, data=train_df, palette='viridis')
        plt.title(col)
        plt.xlabel('Count')
        plt.ylabel(col)
        
        for p in ax.patches:
            width = p.get_width()
            plt.text(width * 1.01,
                     p.get_y() + p.get_height() / 2,
                     '{:1.0f}'.format(width), 
                     ha='left', 
                     va='center')

    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.tight_layout()
    plt.show()

    # Group by 'patient_zip3' and count unique 'patient_state' values
    zip_state_counts = train_df.groupby('patient_zip3')['patient_state'].nunique()

    # Identify zip codes with multiple states
    multiple_states_per_zip = zip_state_counts[zip_state_counts > 1]
    print("Incorrectly mapped zip codes: ", multiple_states_per_zip)

    # Check if one state is mapped to more than one region
    state_region_counts = train_df.groupby('patient_state')['Region'].nunique()
    multiple_regions_per_state = state_region_counts[state_region_counts > 1]
    print("States mapped to multiple regions: ", multiple_regions_per_state)

    # Check if one state is mapped to more than one division
    state_division_counts = train_df.groupby('patient_state')['Division'].nunique()
    multiple_divisions_per_state = state_division_counts[state_division_counts > 1]
    print("States mapped to multiple divisions: ", multiple_divisions_per_state)

    # Define a function to get the most frequent payer type
    def most_frequent(series):
        if series.empty:
            return None
        return series.mode().iloc[0] if not series.mode().empty else None

    # Group by patient_zip3, patient_race, and patient_state and get the most frequent payer_type
    most_frequent_payer = train_df.groupby(['patient_zip3', 'patient_race', 'patient_state'])['payer_type'].agg(most_frequent).reset_index()
    most_frequent_payer = most_frequent_payer.dropna(subset=['payer_type'])

    # Create a bar plot to show the distribution of the most frequent payer types across different races
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.countplot(data=most_frequent_payer, x='payer_type', hue='patient_race', palette='viridis')
    plt.title('Distribution of Most Frequent Payer Types Across Different Races')
    plt.xlabel('Payer Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Patient Race', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    df_copy = train_df.dropna(subset=['patient_race'])
    unique_zip_codes = df_copy['patient_state'].unique()
    cols_per_row = 4
    batch_size = 4
    num_batches = (len(unique_zip_codes) + batch_size - 1) // batch_size

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min(start_idx + batch_size, len(unique_zip_codes))
        fig, axes = plt.subplots(1, cols_per_row, figsize=(20, 6))

        for i in range(start_idx, end_idx):
            zip_code = unique_zip_codes[i]
            subset = df_copy[df_copy['patient_state'] == zip_code]
            ax = axes[i - start_idx]
            sns.countplot(data=subset, x='patient_race', palette='viridis', ax=ax)
            ax.set_title(f'Zip Code - {zip_code}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Patient Race', fontsize=14, labelpad=10)
            ax.set_ylabel('Count', fontsize=14, labelpad=10)
            ax.tick_params(axis='x', rotation=45, labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', label_type='center', fontsize=12, color='white', weight='bold')

        if end_idx - start_idx < cols_per_row:
            for j in range(end_idx - start_idx, cols_per_row):
                fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
    
    
    group_df = train_df.groupby(['breast_cancer_diagnosis_code', 'breast_cancer_diagnosis_desc'], as_index=False)['patient_id'].count()
    display(group_df)


    def melt_temperature_data(df):
        """
    Transforms a given DataFrame by melting temperature columns and handling missing values.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing temperature data and patient zip codes.

    Returns:
    pd.DataFrame: The transformed DataFrame in long format with columns 'patient_zip3', 'month_year', and 'avg_temp'.

    """
    ts_cols = [col for col in df.columns if col.startswith('Average')]
    temp_cols = ['patient_zip3'] + ts_cols
    temp_df = df[temp_cols].drop_duplicates()
    temp_df = temp_df[temp_df.drop(columns=['patient_zip3']).isnull().any(axis=1)].reset_index(drop=True)

    # Reshape the DataFrame
    melted_df = pd.melt(temp_df, id_vars=['patient_zip3'], var_name='month_year', value_name='avg_temp')
    melted_df['month_year'] = melted_df['month_year'].str.extract('Average of (.*)')

    # Convert 'month_year' to datetime format
    melted_df['month_year'] = pd.to_datetime(melted_df['month_year'], format='%b-%y')
    melted_df = melted_df.sort_values(by=['patient_zip3', 'month_year']).reset_index(drop=True)
    melted_df['month_year'] = melted_df['month_year'].dt.strftime('%b-%y')
    
    return melted_df


def predict_missing_values(df, start_index):
    """
    This function takes a DataFrame as input, identifies if there are missing values
    at the end of the series in the 'avg_temp' column, and uses ARIMA to predict these
    missing values. The graph of the series with predicted values is then displayed.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with 'avg_temp' column
    """
    # Check positions of missing values in the avg_temp column
    missing_values = df['avg_temp'].isnull()
    
    # Identify the positions of missing values
    missing_values_positions = missing_values[missing_values].index

    data_before_missing = df[:start_index]['avg_temp']
    
    # Define and fit the ARIMA model
    model = ARIMA(data_before_missing, order=(2, 0, 2))
    model_fit = model.fit()
    
    # Calculate the number of missing values to predict
    num_missing_values = len(missing_values_positions)
    
    # Predict the missing values
    predicted_values = model_fit.forecast(steps=num_missing_values)
    
    # Fill the missing values in the original DataFrame
    df.loc[missing_values_positions, 'avg_temp'] = predicted_values
    
    # Plot the avg_temp column, highlighting the predicted values with red 'X' markers
    plt.figure(figsize=(16, 5))
    plt.plot(df['month_year'], df['avg_temp'], label='Temperature', marker='o', color='b')
    plt.axvline(x=missing_values_positions[0] - 1, color='r', linestyle='--', label='Prediction Start')
    plt.plot(df['month_year'][missing_values_positions], df['avg_temp'][missing_values_positions], 'rx', markersize=10, label='Predicted Values', linestyle='None')
    plt.title('Arima Prediction - ' + str(df.loc[0,'patient_zip3']))
    plt.xlabel('Month-Year')
    plt.ylabel('Average Temperature')
    plt.xticks(rotation = 90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df

def impute_missing_values(df):
    """
    This function takes a DataFrame as input, interpolates missing values in the 'avg_temp' column using linear interpolation, 
    and plots the original and interpolated data, highlighting the imputed points.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with 'avg_temp' column
    
    Returns:
    pd.DataFrame: DataFrame with interpolated 'avg_temp' values and a new column 'orig_avg_temp' containing the original values
    """
    # Interpolating linearly
    df['orig_avg_temp'] = df['avg_temp']
    #df['avg_temp'] = df['avg_temp'].interpolate(method='linear')
    df['avg_temp'] = df['avg_temp'].interpolate(method='linear', limit_direction='both')
    missing_value_indices = df[df['orig_avg_temp'].isna()].index

    # Plotting the interpolated data
    plt.figure(figsize=(16, 5))
    plt.plot(df['month_year'], df['orig_avg_temp'], label='Original Data', marker='o', linestyle='-', color='blue')
    plt.plot(df['month_year'], df['avg_temp'], label='Linearly Interpolated Data', linestyle=':', color='green')
    plt.scatter(missing_value_indices, df.loc[missing_value_indices, 'avg_temp'], color='red', marker='x', s=100, label='Imputed Points', zorder=5)
    plt.title('Linear Interpolation - ' + str(df.loc[0,'patient_zip3']))
    plt.xlabel('Month-Year')
    plt.ylabel('Average Temperature')
    plt.xticks(rotation = 90)
    plt.legend()
    plt.grid(True)
    plt.show()

    return df

def find_start_index_of_last_null_sequence(df):
    """
    This function takes a DataFrame as input and identifies the start index of the last sequence of null values 
    in the 'avg_temp' column. If the last element is not null or no null groups exist, it returns None.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with 'avg_temp' column
    
    Returns:
    int or None: The start index of the last sequence of null values, or None if not applicable
    """
    # Check if the last element is NaN
    if pd.isna(df['avg_temp'].iloc[-1]):
        is_null = df['avg_temp'].isnull()
        
        # Mark the changes between null and non-null
        change = is_null.ne(is_null.shift())
        
        # Assign groups by cumulative sum of changes
        groups = change.cumsum()
        null_groups = groups[is_null]
        
        # Find the last null group
        if not null_groups.empty:
            last_group = null_groups.iloc[-1]
            start_index = null_groups[null_groups == last_group].index[0]
            return start_index
        
    return None  # Return None if the last element is not NaN or no null groups exist


def process_data(df, val):
    """
    The `process_data` function takes a DataFrame and a string value as inputs and processes the data to handle missing values based on the patient's zip code. 
    The processing involves melting the data, separating it by zip codes, handling missing values through interpolation and ARIMA prediction, and finally concatenating the processed data back into a single DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data to be processed.
    - val (str): A string value (e.g., "train" or "test") used for logging purposes during processing.

    Returns:
    - final_melted_df (pd.DataFrame): The concatenated DataFrame after processing and imputing missing values for each zip code.
    """
    melted_df = melt_temperature_data(df)
    
    # Initialize a dictionary to hold the updated DataFrames
    updated_melted_dfs = {}

    patient_zip_df = {f"{zip_code}df": melted_df[melted_df['patient_zip3'] == zip_code].reset_index(drop=True) for zip_code in melted_df['patient_zip3'].unique()}
    patient_zip_df_keys = list(patient_zip_df.keys())

    for key, value in patient_zip_df.items():
        globals()[f"df_{key[:-2]}"] = value 

    count = 0

    for dataF in patient_zip_df_keys:
        count = count+1
        dummy = dataF.rstrip('df')
        print(f"{count}. {val} : Zip - {dummy}")
        start_index = find_start_index_of_last_null_sequence(patient_zip_df[dataF])
        if start_index is not None:
            interpol_df = patient_zip_df[dataF][:start_index]
            final_df = impute_missing_values(interpol_df)
            final_df = final_df.iloc[:, :-1]
            null_row_df = patient_zip_df[dataF][start_index:]
            arima_df = pd.concat([final_df, null_row_df], ignore_index=True)
            final_df = predict_missing_values(arima_df, start_index)
        else: 
            interpol_df = patient_zip_df[dataF]
            final_df = impute_missing_values(interpol_df)
        
        # Store the updated DataFrame in the dictionary
        updated_melted_dfs[dummy] = final_df   

    final_melted_df = pd.concat(updated_melted_dfs.values(), ignore_index=True)
    
    return final_melted_df

def process_temperature_data(df):
    """
    Process a DataFrame containing temperature data to pivot it by 'patient_zip3' and 'month_year'.
    
    This function converts the 'month_year' column to datetime, formats it as a string for pivoting,
    and rearranges the columns in the desired order. The final DataFrame has the average temperature 
    for each 'patient_zip3' for each month and year.

    Parameters:
    df (pd.DataFrame): The input DataFrame with columns 'patient_zip3', 'month_year', and 'avg_temp'.

    Returns:
    pd.DataFrame: The processed DataFrame with pivoted and formatted temperature data.
    """
    df['month_year'] = pd.to_datetime(df['month_year'], format='%b-%y')
    df['month_year'] = df['month_year'].dt.strftime('%b-%y')
    df['month_year'] = 'Average of ' + df['month_year']
    
    # Pivot the DataFrame
    pivot_df = df.pivot(index='patient_zip3', columns='month_year', values='avg_temp')
    pivot_df.reset_index(inplace=True)
    
    # Rearrange columns in the desired order
    desired_order = [f"Average of {month}-{year}" for year in range(13, 19) for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
    pivot_df = pivot_df[['patient_zip3'] + desired_order]
    
    # Round the float columns to 2 decimal places
    float_columns = pivot_df.select_dtypes(include=['float64']).columns
    pivot_df[float_columns] = pivot_df[float_columns].round(2)
    
    # Remove the column names (columns.name)
    pivot_df.columns.name = None
    
    return pivot_df

def process_zip_data(df, pivot_df,flag):
    """
    Processes patient data to merge average columns and pivot data into the original dataframe.

    Parameters:
    df (pd.DataFrame): Original DataFrame containing patient data.
    pivot_df (pd.DataFrame): Pivot DataFrame containing transformed data to merge.

    Returns:
    pd.DataFrame: The processed DataFrame with merged data.
    """
    
    # Extract columns that start with 'Average'
    ts_cols = [col for col in df.columns if col.startswith('Average')]
    zip_cols = ['patient_zip3'] + ts_cols
    zip_df = df[zip_cols].drop_duplicates()
    zip_df.reset_index(drop=True, inplace=True)
    
    # Concatenate pivot_df and zip_df, then drop duplicates
    final_zip_df = pd.concat([pivot_df, zip_df], ignore_index=True)
    final_zip_df.drop_duplicates(inplace=True)

    print(f"Unique Zip Count in {flag}:", final_zip_df['patient_zip3'].nunique())
    
    # Drop NaN values from final_zip_df
    final_zip_df.dropna(inplace=True)
    
    # Drop columns that start with 'Average' from original df
    columns_to_drop = [col for col in df.columns if col.startswith('Average')]
    df.drop(columns=columns_to_drop, inplace=True)
    
    # Merge final_zip_df with train_df on 'patient_zip3'
    df_copy = df.merge(final_zip_df, on='patient_zip3', how='left')
    df = df_copy.copy()

    return df

def categorize_bmi(bmi):
    if pd.isna(bmi):
        return 0  # Missing
    elif bmi < 18.5:
        return 1  # Underweight
    elif 18.5 <= bmi < 25:
        return 2  # Normal
    elif 25 <= bmi < 30:
        return 3  # Overweight
    else:
        return 4
    

    def correct_outliers(df):
        """
    This function identifies and corrects outliers in the DataFrame based on the 'patient_zip3' and 'population' columns. 
    It returns a DataFrame with the corrected values and a list of ZIP codes where outliers were detected and corrected.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing patient data, including 'patient_zip3' and 'population' columns.

    Returns:
    tuple: A tuple containing two elements:
        - pandas.DataFrame: The DataFrame with outliers corrected.
        - list: A list of ZIP codes where outliers were detected and corrected.

    The function performs the following steps:
    1. Identifies ZIP codes where the population varies within the same 'patient_zip3' group.
    2. For each identified ZIP code, it replaces the values in columns 13 to 76 with the values from a sample row where 
       the population equals the mode (most frequent) population value for that ZIP code.
    3. Returns the corrected DataFrame and the list of ZIP codes with outliers.
    
    """
    zip_lst = []
    for zip_code in set(list(df['patient_zip3'].values)):
        if len(set(list(df[df['patient_zip3'] == zip_code]['population'].values))) != 1:
            zip_lst.append(zip_code)

    def replace_outlier_columns(group):
        # Find the most frequent population value
        mode_value = group['population'].mode()[0]
        # Find a sample row where the population equals the mode value
        sample_row = group[group['population'] == mode_value].iloc[0]

        for idx, row in group.iterrows():
            if row['population'] != mode_value:
                group.loc[idx, group.columns[13:77]] = sample_row[group.columns[13:77]]

        return group

    df = df.groupby('patient_zip3').apply(replace_outlier_columns)
    df = df.reset_index(drop=True)

    return df, zip_lst

def preprocess_categorical_data(df):
    """
    This function preprocesses the given  dataframe with the following steps:
    1. Drops column and rows which have either missing values or same values
    2. Creates a state division dataframe and merges it with the dataframe.
    3. Updates 'patient_state' based on specific 'patient_zip3' values.
    4. Recalculates and prints the zip codes mapped to multiple states.
    5. Imputes missing 'payer_type' values based on the mode within each 'patient_zip3' and 'patient_state' groups.
    6. Fills missing 'patient_race' values with 'Missing'.
    7. Updates 'breast_cancer_diagnosis_code' based on the first digit of the code.
    
    Parameters:
    train_df (pd.DataFrame): The input training dataframe.

    Returns:
    pd.DataFrame: The processed training dataframe.
    """
    df = df[df['patient_zip3'] != 772]
    df.drop(columns=['patient_gender', 'breast_cancer_diagnosis_desc', 'metastatic_cancer_diagnosis_code', 'metastatic_first_novel_treatment', 'metastatic_first_novel_treatment_type'], inplace=True)
    
    # Define state division data
    state_division_data = {
        'patient_state': ['WA', 'OR', 'CA', 'HI', 'AK', 'MT', 'ID', 'WY', 'NV', 'UT', 'CO', 'AZ', 'NM', 
                          'ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'OK', 'TX', 'AR', 'LA', 
                          'WI', 'MI', 'IL', 'IN', 'OH', 
                          'KY', 'TN', 'MS', 'AL', 
                          'ME', 'NH', 'VT', 'MA', 'RI', 'CT', 
                          'NY', 'NJ', 'PA', 
                          'DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL'],
        'Division': ['Pacific', 'Pacific', 'Pacific', 'Pacific', 'Pacific', 
                     'Mountain', 'Mountain', 'Mountain', 'Mountain', 'Mountain', 'Mountain', 'Mountain', 'Mountain', 
                     'West North Central', 'West North Central', 'West North Central', 'West North Central', 'West North Central', 'West North Central', 'West North Central', 
                     'West South Central', 'West South Central', 'West South Central', 'West South Central', 
                     'East North Central', 'East North Central', 'East North Central', 'East North Central', 'East North Central', 
                     'East South Central', 'East South Central', 'East South Central', 'East South Central', 
                     'New England', 'New England', 'New England', 'New England', 'New England', 'New England', 
                     'Middle Atlantic', 'Middle Atlantic', 'Middle Atlantic', 
                     'South Atlantic', 'South Atlantic', 'South Atlantic', 'South Atlantic', 'South Atlantic', 'South Atlantic', 'South Atlantic', 'South Atlantic', 'South Atlantic']
    }

    # Create and preprocess state division dataframe
    state_division_df = pd.DataFrame(state_division_data)
    state_division_df = state_division_df.sort_values(by='patient_state').reset_index(drop=True)
    state_division_df = state_division_df[state_division_df['Division'] != 'New England'].reset_index(drop=True)

    # Merge state division dataframe with training dataframe
    df = pd.merge(df.drop(columns=['Division']), state_division_df, on='patient_state', how='left')

    # Update 'patient_state' based on 'patient_zip3' values
    df['patient_state'] = np.where(df['patient_zip3'] == 630, 'MO',
                                         np.where(df['patient_zip3'] == 864, 'AZ',
                                                  df['patient_state']))

    # Recalculate zip_state_counts after updating 'patient_state'
    zip_state_counts_after = df.groupby('patient_zip3')['patient_state'].nunique()
    multiple_states_per_zip_after = zip_state_counts_after[zip_state_counts_after > 1]

    print("Incorrectly mapped zip codes after update: ", multiple_states_per_zip_after)

    # Impute missing 'payer_type' based on 'patient_zip3'
    df_intermediate = df[['patient_zip3', 'patient_state', 'payer_type']].copy()

    # Mode imputation for categorical columns within each 'patient_zip3' group
    for column in df_intermediate.columns:
        if column != 'patient_zip3':  # Exclude the group column
            if df_intermediate[column].dtype not in [np.dtype('float_'), np.dtype('int_')]:
                mode_impute = df_intermediate.groupby('patient_zip3')[column].apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
                df_intermediate[column] = df_intermediate[column].fillna(df_intermediate['patient_zip3'].map(mode_impute))

    # Second step: Impute remaining missing 'payer_type' based on 'patient_state'
    df_final = df_intermediate.copy()

    # Mode imputation for categorical columns within each 'patient_state' group
    for column in df_final.columns:
        if column != 'patient_state':  # Exclude the group column
            if df_final[column].dtype not in [np.dtype('float_'), np.dtype('int_')]:
                mode_impute = df_final.groupby('patient_state')[column].apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
                df_final[column] = df_final[column].fillna(df_final['patient_state'].map(mode_impute))

    # Assign the imputed 'payer_type' back to the original dataframe
    df['payer_type'] = df_final['payer_type'].values

    # Fill missing 'patient_race' values with 'Missing'
    df['patient_race'] = df['patient_race'].fillna('Missing')

    # Update 'breast_cancer_diagnosis_code' based on the first digit of the code
    df['code'] = df['breast_cancer_diagnosis_code'].apply(lambda x: 0 if str(x).startswith('1') else 1)
    df['breast_cancer_diagnosis_code'] = df['code']
    df = df.drop(columns=['code'])

    return df

def encode_categorical_columns(train_df, test_df):
    """
    Encodes categorical columns in the training and testing dataframes using OrdinalEncoder.
    
    Parameters:
    train_df (pd.DataFrame): The training dataframe.
    test_df (pd.DataFrame): The testing dataframe.
    
    Returns:
    final_train_df (pd.DataFrame): The training dataframe with encoded categorical columns.
    final_test_df (pd.DataFrame): The testing dataframe with encoded categorical columns.
    """
    train_cat_cols = train_df.select_dtypes(include=['object', 'category']).columns
    test_cat_cols = test_df.select_dtypes(include=['object', 'category']).columns

    encoder = OrdinalEncoder()

    # Apply Ordinal Encoding to categorical columns
    final_train_df = train_df.copy()
    final_train_df[train_cat_cols] = encoder.fit_transform(train_df[train_cat_cols])
    
    final_test_df = test_df.copy()
    final_test_df[test_cat_cols] = encoder.transform(test_df[test_cat_cols])
    
    return final_train_df, final_test_df


def train_lstm_model(train_df, test_df, target_column='metastatic_diagnosis_period', epochs = 136, batch_size = 32):
    """
    Trains an LSTM model on the given training DataFrame and evaluates its performance on the validation set.
    The model is then used to predict the target variable on the test DataFrame.

    Parameters:
    train_df (pandas.DataFrame): The training DataFrame containing the features and target variable.
    test_df (pandas.DataFrame): The test DataFrame containing the features.
    target_column (str): The name of the target column to predict. Default is 'metastatic_diagnosis_period'.
    epochs (int): The number of epochs to train the model. Default is 136.
    batch_size (int): The batch size for training the model. Default is 32.

    Returns:
    tuple: A tuple containing two elements:
        - float: The root mean squared error (RMSE) on the validation set.
        - numpy.ndarray: The predicted values for the target variable on the test DataFrame.
    """

    X = train_df.drop(columns=[target_column, 'patient_id'])
    y = train_df[target_column]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling the data
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(test_df.drop(columns=['patient_id']))

    # Reshaping the data for LSTM
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_val_scaled = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Defining the LSTM Model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Training the Model
    history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_scaled, y_val), verbose=1, shuffle=True)

    y_val_pred = model.predict(X_val_scaled)

    lstm_mse = mean_squared_error(y_val, y_val_pred)
    rmse = round(np.sqrt(lstm_mse), 3)
    print(f'Root Mean Squared Error: {rmse}')

    y_test_pred = model.predict(X_test_scaled)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return rmse, y_test_pred

def process_data_pipeline(train_df, test_df):

    pd.set_option('display.max_rows', 160)
    pd.set_option('display.max.columns', None)
    
    print("Train Data:")
    display(train_df)
    print("Test Data:")
    display(test_df)
    
    final_melted_train_df = process_data(train_df, "train")
    final_melted_test_df = process_data(test_df, "test")
    
    print("Train Data:")
    display(final_melted_train_df)
    print("Test Data:")
    display(final_melted_test_df)
    
    final_melted_train_df = process_temperature_data(final_melted_train_df)
    final_melted_test_df = process_temperature_data(final_melted_test_df)
    
    print("Train Data:")
    display(final_melted_train_df)
    print("Test Data:")
    display(final_melted_test_df)
    
    train_df_processed = process_zip_data(train_df, final_melted_train_df, "train")
    test_df_processed = process_zip_data(test_df, final_melted_test_df, "test")
    
    print("Processed Train DataFrame:")
    display(train_df_processed)
    
    print("\nProcessed Test DataFrame:")
    display(test_df_processed)
    
    train_df_processed['bmi'] = train_df_processed['bmi'].apply(categorize_bmi)
    test_df_processed['bmi'] = test_df_processed['bmi'].apply(categorize_bmi)
    
    train_df_processed, zip_lst_train = correct_outliers(train_df_processed)
    
    train_df_processed = preprocess_categorical_data(train_df_processed)
    test_df_processed = preprocess_categorical_data(test_df_processed)
    
    final_train_df, final_test_df = encode_categorical_columns(train_df_processed, test_df_processed)
    
    print("Final Train Data: ")
    display(final_train_df)
    print("\nFinal Test Data:")
    display(final_test_df)
    
    return final_train_df, final_test_df

train_df = pd.read_csv(r"train.csv")
test_df = pd.read_csv(r"test.csv")

analyze_temperature_data(train_df)
categorical_eda(train_df)

rmse, final_test_df['metastatic_diagnosis_period'] = train_lstm_model(final_train_df, final_test_df)
final_test_df = final_test_df[['patient_id', 'metastatic_diagnosis_period']]
final_test_df['metastatic_diagnosis_period'] = final_test_df['metastatic_diagnosis_period'].round(0).astype(int)
display(final_test_df)
final_test_df.to_csv("submission.csv")
print("Validation RMSE:", rmse)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'iterations': [100, 200],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid_search = GridSearchCV(estimator=catboost, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best RMSE found: ", np.sqrt(-grid_search.best_score_))


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assume the following functions are defined as needed:
# process_data, process_temperature_data, process_zip_data, categorize_bmi, correct_outliers,
# preprocess_categorical_data, encode_categorical_columns, train_lstm_model

def process_data_pipeline(train_df, test_df):

    pd.set_option('display.max_rows', 160)
    pd.set_option('display.max.columns', None)
    
    # Initial Data Display
    print("Initial Train Data:")
    display(train_df)
    print("Initial Test Data:")
    display(test_df)
    
    final_melted_train_df = process_data(train_df, "train")
    final_melted_test_df = process_data(test_df, "test")
    
    final_melted_train_df = process_temperature_data(final_melted_train_df)
    final_melted_test_df = process_temperature_data(final_melted_test_df)
    
    train_df_processed = process_zip_data(train_df, final_melted_train_df, "train")
    test_df_processed = process_zip_data(test_df, final_melted_test_df, "test")
    
    train_df_processed['bmi'] = train_df_processed['bmi'].apply(categorize_bmi)
    test_df_processed['bmi'] = test_df_processed['bmi'].apply(categorize_bmi)
    
    train_df_processed, _ = correct_outliers(train_df_processed)
    
    train_df_processed = preprocess_categorical_data(train_df_processed)
    test_df_processed = preprocess_categorical_data(test_df_processed)
    
    final_train_df, final_test_df = encode_categorical_columns(train_df_processed, test_df_processed)
    
    # Ensure target column is present in both train and test sets
    if 'metastatic_diagnosis_period' not in final_test_df.columns:
        final_test_df['metastatic_diagnosis_period'] = np.nan  # Placeholder if it's missing

    # Feature Scaling
    scaler = StandardScaler()
    numeric_features = final_train_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'metastatic_diagnosis_period' in numeric_features:
        numeric_features.remove('metastatic_diagnosis_period')
    
    scaled_train_df = final_train_df.copy()
    scaled_test_df = final_test_df.copy()
    
    scaled_train_df[numeric_features] = scaler.fit_transform(final_train_df[numeric_features])
    scaled_test_df[numeric_features] = scaler.transform(final_test_df[numeric_features])
    
    # Feature Engineering: Polynomial Features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_train_features = poly.fit_transform(scaled_train_df[numeric_features])
    poly_test_features = poly.transform(scaled_test_df[numeric_features])
    
    poly_feature_names = poly.get_feature_names_out(numeric_features)
    poly_train_df = pd.DataFrame(poly_train_features, columns=poly_feature_names, index=scaled_train_df.index)
    poly_test_df = pd.DataFrame(poly_test_features, columns=poly_feature_names, index=scaled_test_df.index)
    
    scaled_train_df = pd.concat([scaled_train_df.reset_index(drop=True), poly_train_df.reset_index(drop=True)], axis=1)
    scaled_test_df = pd.concat([scaled_test_df.reset_index(drop=True), poly_test_df.reset_index(drop=True)], axis=1)
    
    # Feature Selection
    selector = SelectKBest(score_func=f_regression, k=50)
    X_train = scaled_train_df.drop(columns=['metastatic_diagnosis_period', 'patient_id'])
    y_train = scaled_train_df['metastatic_diagnosis_period']
    selector.fit(X_train, y_train)
    
    selected_features = selector.get_support(indices=True)
    selected_feature_names = X_train.columns[selected_features]
    
    # Correct the feature names for selected features
    selected_feature_names = [feature for feature in selected_feature_names if feature in scaled_train_df.columns]
    
    # Retain the original 'metastatic_diagnosis_period' and 'patient_id' columns
    scaled_train_df = pd.concat([scaled_train_df[['metastatic_diagnosis_period', 'patient_id']], scaled_train_df[selected_feature_names]], axis=1)
    scaled_test_df = pd.concat([scaled_test_df[['patient_id']], scaled_test_df[selected_feature_names]], axis=1)
    
    # Verify column names after feature selection
    print("Columns after feature selection:")
    print("Train:", scaled_train_df.columns)
    print("Test:", scaled_test_df.columns)
    
    print("Final Train Data: ")
    display(scaled_train_df)
    print("\nFinal Test Data:")
    display(scaled_test_df)
    
    return scaled_train_df, scaled_test_df

# Define evaluation metrics
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def print_evaluation(y_true, y_pred):
    mae, rmse, r2 = evaluate_model(y_true, y_pred)
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-Squared: {r2}")

# Load your data
train_df = pd.read_csv(r"train.csv")
test_df = pd.read_csv(r"test.csv")

# Process the data
final_train_df, final_test_df = process_data_pipeline(train_df, test_df)

# Split the training data into a new training set and a validation set
X_train_full = final_train_df.drop(columns=['metastatic_diagnosis_period'])
y_train_full = final_train_df['metastatic_diagnosis_period']

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Support Vector Regression': SVR()
}

# Hyperparameter tuning and model evaluation
for model_name, model in models.items():
    if model_name in ['Ridge Regression', 'Lasso Regression']:
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    elif model_name == 'Random Forest':
        param_grid = {'n_estimators': [100, 200], 'max_depth': [4, 6, 8, 10]}
    elif model_name == 'Gradient Boosting':
        param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 4, 5], 'learning_rate': [0.01, 0.1]}
    elif model_name == 'Support Vector Regression':
        param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'epsilon': [0.1, 0.2, 0.5]}
    else:
        param_grid = {}

    if param_grid:
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model.fit(X_train, y_train)

    y_pred = best_model.predict(X_val)
    print(f"\n{model_name} Evaluation Metrics:")
    print_evaluation(y_val, y_pred)

# Train the LSTM model with the final_train_df and final_test_df
rmse, final_test_df['metastatic_diagnosis_period'] = train_lstm_model(final_train_df, final_test_df)
final_test_df = final_test_df[['patient_id', 'metastatic_diagnosis_period']]
final_test_df['metastatic_diagnosis_period'] = final_test_df['metastatic_diagnosis_period'].round(0).astype(int)
display(final_test_df)
final_test_df.to_csv("submission.csv")
print("Validation RMSE:", rmse)

