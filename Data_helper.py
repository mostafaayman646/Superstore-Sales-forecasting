import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class Preprocessing_Pipeline:
    def __init__(self,df):
        self.df = df
    
    def remove_features(self,features):
        return self.df.drop(columns = features)
        
    def change_to_obj(self,feature):
        if feature in self.df.columns:
            if self.df[feature].dtype != 'object':
                self.df[feature] = self.df[feature].astype('object')
            return self.df
        
        return 'This feature not exist'
    
    def change_to_date(self,feature):
        if feature in self.df.columns:
            self.df[feature] = pd.to_datetime(self.df[feature], format='%d/%m/%Y')
            return self.df
        
        return 'This feature not exist'
    
    def handle_na(self,feature,remove=False,value_to_add = None):
        if feature in self.df.columns:
            if not remove:
                self.df[feature] = self.df[feature].fillna(value_to_add)
            
            return self.df
        return 'This feature not exist'
    
    def handle_outliers(self,feature,log_transform = False):
        if feature in self.df.columns:
            Q1 = self.df[feature].quantile(0.25)
            Q3 = self.df[feature].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            self.df = self.df.loc[(self.df[feature] >= lower_bound) & (self.df[feature] <= upper_bound)]

            if log_transform:
                self.df[f'log_scaled_{feature}'] = np.log1p(self.df[feature])
            
            return self.df
        
        return 'This feature not exist'
    
    def create_Time_Based_features(self):
        #Day of Week
        self.df["dayofweek"] = self.df["Order Date"].dt.dayofweek

        self.df["dayofweek"] = self.df["dayofweek"].map(
            lambda x: "Weekend" if x >= 5 else "Job_days"
        )

        # Day of month and day of year
        self.df["day"]= self.df["Order Date"].dt.day

        doy_bins   = [0, 90, 180, 270, 365]  
        doy_labels = ["Q1", "Q2", "Q3", "Q4"]  
        self.df["dayofyear_bin"] = pd.cut(
            self.df["Order Date"].dt.day_of_year,
            bins=doy_bins,
            labels=doy_labels
        )

        # Month and # Seasons
        self.df["month"] = self.df["Order Date"].dt.month

        season_bins   = [0, 3, 6, 9, 11]   # Winter(Dec–Feb), Spring, Summer, Autumn
        season_labels = ['Winter', 'Spring', 'Summer', 'Autumn']
        self.df["season"] = pd.cut(
            self.df["month"],
            bins=season_bins,
            labels=season_labels,
            right=False
        )
        
        #Years
        self.df['year'] = self.df['Order Date'].dt.year
    
        return self.df

    def scaling_encoding(self):
        temp_features = [
            'Row ID','Order ID','Ship Date','Order Date','Customer ID',
            'Customer Name','Product ID','Product Name','Postal Code'
        ]
        
        temp_df = self.df[temp_features].copy()

        self.df = self.remove_features(features=temp_features)

        numeric_features = self.df.select_dtypes(include=["float64"]).columns.tolist()
        categorical_features = self.df.select_dtypes(exclude=["float64"]).columns.tolist()

        scaler = MinMaxScaler()
        numeric_pipeline = Pipeline([("scaler", scaler)])

        clf = ColumnTransformer([
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('numeric', numeric_pipeline, numeric_features)
        ], remainder='passthrough')

        trf = clf.fit_transform(self.df)

        # Get transformed column names
        all_features = clf.get_feature_names_out()

        transformed_df = pd.DataFrame(trf, columns=all_features, index=self.df.index)

        self.df = pd.concat([temp_df, transformed_df], axis=1)

        return self.df