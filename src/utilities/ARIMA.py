#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Source code for ARIMA_Analysis.ipynb
"""

import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm 
import matplotlib.pyplot as plt

def one_zipcode(df, zipcode, index = None):
    """
    This function pulls the data for one zipcode at a time and retuns a DataFrame for using in Prophet.
    
    Inputs:
    df: dataframe of zipcodes and time series in wide format
    zipcode: A particular zipcode of interest. 
    index: index of a zipcode of interest. Defaults to none.
    
    Output:
    A pandas dataframe time series of the zipcode in long format. 
    """
    if index != None: #if input is index number and not zipcode
        series = df.iloc[index]
    else:
        series = df.loc[df['RegionName'] == zipcode] #if input is zipcode and not index
        series = series.iloc[0]
        
    series_data = series.iloc[7:]
    df_series = pd.DataFrame(series_data.values, index = series_data.index, columns = ['y'])
    df_series.index = pd.to_datetime(df_series.index, yearfirst = True, format = '%Y-%m')
    df_series['ds'] = df_series.index
    df_series.reset_index(drop = True, inplace = True)
    df_series['y'] = df_series['y'].astype('int64', errors='ignore')
    return df_series

def prep_zip_for_ARIMA(df, zipcode):
    """
    Prepares a dataframe to be put into an ARIMA Model by providing the large dataframe of all the zipcodes.
    
    input:
    df : a pandas dataframe with time series and median home value for all of the zipcodes. 
    
    output:
    df_zip : a dataframe with na's dropped, the datetime column set to the index, and the median home value cast to int64
    """
    
    df_zip = one_zipcode(df, zipcode= zipcode)
    df_zip.set_index('ds', inplace=True)
    df_zip.dropna(inplace=True)
    df_zip['y'] = df_zip['y'].astype({'y': 'int64'})
    return df_zip

def get_param_combos(max_range=None, p_max=None ,d_max=None, q_max=None):
    """
    Creates the p,d,q combos for an ARIMA model grid search
    
    inputs:
    max_range: input max range if you want all three parameters to have the same number of possible combos, otherwise
    specify with the other KWARGS
        
    """
    
    if max_range != None:
        p_max = d_max = q_max = range(0, max_range)
        pdq = list(itertools.product(p_max,d_max,q_max))
        return pdq
    else:
        p_max = range(0,p_max)
        q_max = range(0,q_max)
        d_max = range(0,d_max)
        pdq = list(itertools.product(p_max,d_max,q_max))
        return pdq

def ARIMA_param_gridsearch(df, max_range=None, p_max=None ,d_max=None, q_max=None, verbose = True):
        
    """
    Finds the optimum p,d,q parameters for an ARIMA model using AIC as the metric
    
    inputs:
    df: the dataframe to be modeled
    pdq_combo_list: list of possible pdq combinations
    
    returns: prints optimum combo with corresponding AIC and returns a tuple of the ideal combo
    """
    
    pdq_combo_list = get_param_combos(max_range , p_max ,d_max, q_max)

    ans = []
    for comb in pdq_combo_list:
        
        mod = sm.tsa.statespace.SARIMAX(df,
                                    order=(comb),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)                                      
        output = mod.fit()
        ans.append([comb, output.aic])
        if verbose == True:
            print('ARIMA {} : AIC Calculated ={}'.format(comb, output.aic))

    ans_df = pd.DataFrame(ans, columns=['pdq', 'aic'])
    
    print(ans_df.loc[ans_df['aic'].idxmin()])
    return ans_df.loc[ans_df['aic'].idxmin()][0]

def ARIMA_MODEL(df, params, verbose = True):
    """
    fits an ARIMA model to the input df using the params (tuple) to define p,d,q
    """
    
    ARIMA_MODEL = sm.tsa.statespace.SARIMAX(df,
                                order=params,
                                enforce_stationarity=False,
                                enforce_invertibility=False)

    output = ARIMA_MODEL.fit()




    # Fit the model and print results
    # print(output.summary().tables[1])

    if verbose == True:
        print(output.summary().tables[1])
        output.plot_diagnostics(figsize=(15, 18))
        plt.show()
    
    return output
    
def get_forecast_startdate(df):
    """
    helper function to get the first date of observed data from the zip code dataframe
    """
    return df.index[5]
    
def ARIMA_predictions(df, estimator):
    """
    create ARIMA model predictions and confidence intervals
    
    input: time series df used to fit the estimator
    estimator: ARIMA model used to create the predictions
    
    """
    start_date = get_forecast_startdate(df)
    
    pred = estimator.get_prediction(start=pd.to_datetime(start_date),  dynamic=False)
    
    pred_conf = pred.conf_int()
    
    return pred, pred_conf 

def ARIMA_rmse(predicted_values, observed_values_df):
    """
    calculates the rmse for the ARIMA model
    """
    # Get the Real and predicted values
    zip_forecasted_array = predicted_values.predicted_mean
    zip_truth = observed_values_df[get_forecast_startdate(observed_values_df):]['y']

    # Compute the mean square error
    rmse = np.sqrt(((zip_forecasted_array - zip_truth)**2).mean())
    print('The Root Mean Squared Error of our forecasts is {}'.format(round(rmse, 2)))
    return rmse
    

def plot_model_validation(predictions, prediction_confidence_intervals, observed_values_df, df_original, zipcode):
    
    """
    Plot real vs predicted values along with confidence interval
    """

    figsize=(10, 6)
    
    
    fig = plt.figure(figsize= figsize , facecolor='w')
    ax = fig.add_subplot(111)
    
    #Plot observed values
    observed_values_df[get_forecast_startdate(observed_values_df):].plot(ax=ax, label='observed', ls='dotted', c='black')

    #Plot predicted values
    predictions.predicted_mean.plot(ax=ax, label='Forecast', color='#0072B2', alpha=.8)

    # Plot the range for confidence intervals
    ax.fill_between(prediction_confidence_intervals.index,
                    prediction_confidence_intervals.iloc[:, 0],
                    prediction_confidence_intervals.iloc[:, 1], color='#0072B2', alpha=.2)

    #Set axes labels
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Median Home Price')
    plt.xlim(right=observed_values_df.index[-1])
    plt.ylim(bottom = 0,
            top = predictions.predicted_mean.max()+100000)
    
    #get info for title
    city = list(df_original[df_original['RegionName'] == zipcode]['City'])[0]
    state = list(df_original[df_original['RegionName'] == zipcode]['State'])[0]
    RMSE = ARIMA_rmse(predictions, observed_values_df)
    
    
    plt.legend()
    plt.title(f"ARIMA Model validation on median home sale price for {city}, {state}, {zipcode} \n  RMSE: {round(RMSE)}")
    fig.tight_layout()
    plt.savefig(f'../../figures/{city}_ARIMA_model_validation.png')
    plt.show()


def ARIMA_forecast(observed_df, df_original, ARIMA_model, steps, zipcode):
    """
    Plot real vs predicted values with confidence interval, and extend the predictions into the future for
    number of  `steps` 
    """

    figsize=(10, 6)
    
    fig = plt.figure(figsize= figsize , facecolor='w')
    ax = fig.add_subplot(111)
    
    prediction = ARIMA_model.get_forecast(steps=steps, dynamic=False)

    # Get confidence intervals of forecasts
    pred_conf = prediction.conf_int()
    
    #get confidence intervals for historic forecast
    pred, pred_conf_hist = ARIMA_predictions(observed_df, ARIMA_model)
    
   
    
    #plot forecast
    
    observed_df.plot(ax = ax, label='Observed', ls='dotted', c='black')
    prediction.predicted_mean.plot(ax=ax, label='Forecast', c ='#0072B2' )
    ax.fill_between(pred_conf.index,
                    pred_conf.iloc[:, 0],
                    pred_conf.iloc[:, 1], color='#0072B2', alpha=.25)
    
     # Plot the range for confidence intervals
    ax.fill_between(pred_conf_hist.index,
                    pred_conf_hist.iloc[:, 0],
                    pred_conf_hist.iloc[:, 1], color='#0072B2', alpha=.2)
    
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Median Home Value')
    plt.xlim(right=prediction.predicted_mean.index[-1])
    plt.ylim(bottom = 0,
            top = prediction.predicted_mean.max()+100000)

    ax.legend()
    
    #get values for Title
    city = list(df_original[df_original['RegionName'] == zipcode]['City'])[0]
    state = list(df_original[df_original['RegionName'] == zipcode]['State'])[0]
    
    
    plt.title(f"ARIMA Model Forecast \n Median Home sale price for {city}, {state}, {zipcode})")
    fig.tight_layout()
    plt.savefig(f'../../figures/{city}_ARIMA_model_forecast.png')
    
    plt.show()

def ARIMA_Analysis(df, zipcode, forecast_steps, param_max_range=None, p_max=None ,d_max=None, q_max=None  ):
    
    df_zip = prep_zip_for_ARIMA(df, zipcode)
    
    combo = ARIMA_param_gridsearch(df_zip, max_range = param_max_range, p_max = p_max, d_max = d_max, q_max= q_max)
    
    output = ARIMA_MODEL(df_zip, combo)
    
    pred, pred_conf = ARIMA_predictions(df_zip, output)
    
    plot_model_validation(pred, pred_conf, df_zip, df, zipcode)
    
    ARIMA_rmse(pred, df_zip)
    
    ARIMA_forecast(df_zip, df, output, forecast_steps, zipcode)