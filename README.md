# National Home Value Analysis and Modeling
Using time series modeling to analyze and forecast nation wide zipcode single family median home values using Facebook Prophet and ARIMA time series models.   

#### This README.pdf file will serve as a roadmap to this repository. The repository is open and available to the public.

### Directories and files to be aware of:

### • `zipcode` conda environment

This project relies on you using the [`environment.yml`](environment.yml) file to recreate the `zipcode` conda environment. To do so, please run the following commands:

```bash
# create the zipcode conda environment
conda env create -f environment.yml

# activate the zipcode conda environment
conda activate zipcode

# make zipcode available to you as a kernel in jupyter
python -m ipykernel install --user --name zipcode --display-name "Python (zipcode)"
```

### • `.src` source code:

This project contains several .py modules in the `src/utilities` directory. Please use the following bash command to install the .src module:

``` bash
#install the .src modules
pip install -e .
```

### • A notebooks directory that contains multiple Jupyter Notebooks:
    1. `notebooks/exploratory/EDA.ipynb`

         This notebook performs basic data munging and preperation for initial Prophet Modeling. 

    2. `notebooks/exploratory/model_batch.ipynb`

        This streamlines the EDA notebook's process into a pipline implementation to perform batch modeling on the nearly 15000 zipcodes
        
    3. `notebooks/report/Prophet_Analysis.ipynb`

        This notebook builds the selection criteria and selects the top 5 zipcodes. We then perform model crossvalidation and analyze the forecasted median home values. 
    
    4. `notebooks/report/ARIMA_Analysis.ipynb`

        This notebook builds a pipeline for ARIMA modeling. We then feed the top 5 zipcodes through this pipeline and analyze 
    
    5. `notebooks/report/ARIMA_Proph_comp.ipynb`

        This notebook produces visualizations to compare the ARIMA and Prophet model forecasts. 

### • The raw data is included in this public repo in `/data/raw/zillow.csv`
This data was originally accessed via Zillow's data portal [here](https://www.zillow.com/research/data/)

### • '.pickle' files
If you wish to explore our results, you can avoid the long processing time of modeling by utilizing our pickled dictionaries of modeling outputs. These can be found in `/data/processed/` 

### • A one-page .pdf memo summarizing our project written for non technical stakeholders can be found in the `/reports/memo.md`

### • A side deck summarizing our project can be found `/reports/presentation.pdf`




## Methodology 
Using approximately 20 years of published data from Zillow on the monthly median home sale price for nearly 15,000 Zip Codes across the U.S., we examine the projected growth in each Zip Code to see how that market is expected to change in five years. We first performed a five year forecast using the Facebook Profit model for each Zip Code, and then identified 5 candidate Zip Codes with at least 20 years of data and the largest five year percent growth. We performed cross-validation on the 5 candidate Zip Code models and reforecasted the five year period with an ARIMA model to provide secondary evidence.



## Results 

From our analysis, the five best zip codes to invest in are as follows: 
  * 34982 - Fort Pierce, FL - projected 71% 5-year appreciation 
  * 33982 - Punta Gorda, FL - projected 66% 5-year appreciation 
  * 34951 - Fort Pierce, FL - projected 65% 5-year appreciation 
  * 37209 - Nashville-Davidson, TN - projected 65% 5-year appreciation 
  * 15201 - Pittsburgh, PA - projected 64% 5-year appreciation

The top three Zip Codes are located in coastal Florida communities. While these three Zip Codes have the largest percent growth, they have much larger confidence intervals than the last two, suggesting that Nashville and Pittsburgh may be “safer” investment opportunities. The large confidence intervals for the Florida communities may be a result of lingering effects from the 2008 recession.



