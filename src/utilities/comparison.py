#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Source code for ARIMA_Proph_comp.ipynb
"""

import pickle
import pandas as pd
from fbprophet import Prophet
import numpy as np
import matplotlib.pyplot as plt

from src.utilities import ARIMA as ar 

def import_pickled_prophet_forecasts():
    """
    Imports pickle files and returns a DF containing the Prophet Model forecasts
    for all zipcodes. 
    """
    with open('../../data/processed/prophet_lists_wyatt2', 'rb') as p:
        wyatt = pickle.load(p)
    with open('../../data/processed/prophet_lists_brent', 'rb') as fp:
        brent = pickle.load(fp)
    with open('../../data/processed/prophet_lists_kw', 'rb') as kp:
        karen = pickle.load(kp)
    with open('../../data/processed/prophet_lists_forgotten', 'rb') as bp:
        forgotten = pickle.load(bp)
        
    wyatt.extend(brent)
    wyatt.extend(karen)
    wyatt.extend(forgotten)
    df_summary = pd.DataFrame(wyatt)
    return df_summary

def find_top_5(prophet_df, df_original):
    """
    returns the top 5 zipcodes that have the highest projected 5 year return as 
    as percent of investment AND has at least 20 years worth of historic data
    
    prophet_df : df of prophet forecasts for all zipcodes
    df_original : original df from .csv import

    returns : a list of top 5 zipcodes that match the criteria 
    
    """
    df_summary = prophet_df

    #Find the zipcodes with at least 20 years of data:
    twentyyear = list(df_original[df_original['1998-04'].isna() != True]['RegionName'])

    #select the top 5 with highest forecst growth
    topfive = list(df_summary[df_summary['Zipcode'].isin(twentyyear)].nlargest(5, 'pct_change_5year')['Zipcode'])

    return topfive 

def one_zipcode(df, zipcode, index = None):
    """
    This function pulls the data for one zipcode at a time and retuns a DataFrame for using in Prophet.
    """

    if index != None:
        series = df.iloc[index]
    else:
        series = df.loc[df['RegionName'] == zipcode]
        series = series.iloc[0]
    series_data = series.iloc[7:]
    df_series = pd.DataFrame(series_data.values, index = series_data.index, columns = ['y'])
    df_series.index = pd.to_datetime(df_series.index, yearfirst = True, format = '%Y-%m')
    df_series['ds'] = df_series.index
    df_series.reset_index(drop = True, inplace = True)
    df_series['y'] = df_series['y'].astype('int64', errors='ignore')
    return df_series

def Prophet_analysis(df, zipcode, index = None):
    """
    This function instantiates a Prophet model, fits it to the DataFrame, and predicts values which are returned in 
    a forecast Dataframe.

    inputs:
    df : original df from .csv import
    zipcode: zipcode of interest
    index: optinal way to identify zipcode using index reference number

    outputs: model, forecast_dataframe
    """
    df_series = one_zipcode(df, zipcode, index)

    m = Prophet(seasonality_mode='multiplicative', interval_width=0.95)
    m.fit(df_series)
    future = m.make_future_dataframe(60, freq = 'M')
    forecast = m.predict(future)
    return m, forecast

def comparison_plot(df, zipcode):
    """
    Produce a plot comparing the prophet forecast to the ARIMA forecast
    """
    #Get the prophet forecast for 34982
    proph_model, proph_forecast = Prophet_analysis(df, zipcode)
    #set the proph forecast to Datetime series for plotting
    proph_forecast.set_index('ds', inplace=True)

    #Fit the ARIMA model the best p,q,d found in the ARIMA_Analysis.ipynp or by 
    #grid searching for new parameters if zipcode is an alternativs zipcode
    arima_df = ar.prep_zip_for_ARIMA(df, zipcode)
    
    if zipcode == 15201:
        params = (4,2,5)
        
    elif zipcode == 37209:
        params = (2,2,5)
        
    elif zipcode == 34951 or 33982:
        params = (3,3,5)
        
    elif zipcode == 34982:
        params = (2,3,5)
        
    else:
        params = ar.ARIMA_param_gridsearch(df_zip, max_range = 3, verbose=True)
        
    ARIMA_model = ar.ARIMA_MODEL(arima_df, params, verbose = False)

    #create a series with the ARIMA model future forecast
    ARIMA_forecast = ARIMA_model.get_forecast(steps=60, dynamic = False).predicted_mean
    
    #create a series of the historic observed values to be plot
    historical = one_zipcode(df, zipcode=zipcode)
    historical.set_index('ds', inplace=True)
    historical.rename(columns={'y':'Observed'}, inplace=True)

    #Plot the Prophet and Arima Forecasts and save the figure 
    figsize=(10, 6)
    fig = plt.figure(figsize= figsize , facecolor='w')
    ax = fig.add_subplot(111)

    #Plot prophet values
    proph_forecast['yhat']['2018':].plot(ax=ax, label='Prophet Forecast', ls='dotted', c='green')

    #Plot arima forecast values
    ARIMA_forecast.plot(ax=ax, label='ARIMA Forecast', color='#0072B2', ls = '--', alpha=.8)

    #Plot historic observed 
    historical.plot(ax=ax, label='Observed', color='black', alpha=.8)


    #Set axes labels
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Median Home Price')
    plt.xlim(right=proph_forecast.index[-1])
    plt.ylim(bottom = 0,
            top = proph_forecast.yhat.max()+100000)

    #get info for title
    city = list(df[df['RegionName'] == zipcode]['City'])[0]
    state = list(df[df['RegionName'] == zipcode]['State'])[0]
    plt.legend()

    #save figure
    plt.title(f"ARIMA/Prophet Model Comparison \n Median Home Sale Price for {city}, {state}, {zipcode}")
    fig.tight_layout()
    plt.savefig(f'../../figures/comparison/{city}_{zipcode}_ARIMA_model_validation.png')
    plt.show()
