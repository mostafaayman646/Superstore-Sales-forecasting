import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_engine.timeseries.forecasting import LagFeatures

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
        self.df['Day']   =  self.df['Order Date'].dt.day
        self.df['Month'] =  self.df['Order Date'].dt.month
        self.df['Year']  =  self.df['Order Date'].dt.year
    
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

        transformed_df = pd.self.dfFrame(trf, columns=all_features, index=self.df.index)

        self.df = pd.concat([temp_df, transformed_df], axis=1)

        return self.df
    
    def add_lag_features(self):
        Lag_tranformer = LagFeatures(variables = ['Sales'], periods=[1,7,30,60])
        
        return Lag_tranformer.fit_transform(self.df)