import pandas as pd

def state_mapping(df):
    all_state_mapping = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
    }
    # Add the Abbreviation column to the DataFrame
    df['Abbreviation'] = df['State'].map(all_state_mapping)
    return df

def load_data(path):
    return pd.read_csv(path)

def date_time_processing_pipeline(df):
    # Convert the "Order Date" column to a datetime format with "dd/mm/yyyy" format
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    
    # Extract day, month, and year into separate columns
    df['Day'] = df['Order Date'].dt.day
    df['Month'] = df['Order Date'].dt.month
    df['Year'] = df['Order Date'].dt.year
    
    # Combine "Month" and "Year" into a datetime column
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str), format='%Y-%m')
    
    # Convert "Order Date" to a datetime column
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')
    
    # Generate a date range from the minimum to the maximum date
    date_range = pd.date_range(start=df['Order Date'].min(), end=df['Order Date'].max())
    
    # Find missing dates by comparing the date range with the unique dates in the "Order Date" column
    missing_dates = date_range[~date_range.isin(df['Order Date'])]
    
    # Count the total number of missing dates
    total_missing_dates = len(missing_dates) #228

    # Convert "Order Date" to a datetime column
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')

    # Define the date range
    date_range = pd.date_range(start=df['Order Date'].min(), end=df['Order Date'].max())

    # Find missing dates
    missing_dates = date_range.difference(df['Order Date'])

    # Add missing dates to the DataFrame with Sales number of 0
    missing_data = {
        "Order Date": missing_dates,
        "Sales": [0] * len(missing_dates)
    }
    missing_df = pd.DataFrame(missing_data)

    # Concatenate the missing data with the original DataFrame
    df = pd.concat([df, missing_df], ignore_index=True)

    # Sort the DataFrame by "Order Date"
    df = df.sort_values(by='Order Date')

    # Convert "Order Date" to a datetime column
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')

    # Extract the "Day," "Month," and "Year" from the "Order Date" column
    df['Day'] = df['Order Date'].dt.day
    df['Month'] = df['Order Date'].dt.month
    df['Year'] = df['Order Date'].dt.year
    
    return df

def preprocessing():
    #Load Data
    df = load_data('Data/train.csv')
    
    # Create a mapping for all 50 states 
    df = state_mapping(df)
    
    # Sort the DataFrame by 'Sales' in descending order
    df = df.sort_values(by='Sales', ascending=False)
    
    df = date_time_processing_pipeline(df)
    
    return df