To: Real estate firm stakeholders
From: Data Scientists: Brent Butler, Wyatt Sharber, Karen Warmbein
RE: Best 5 locations in the U.S. for real estate investment!


### Objective
Investment in real estate can lead to quick profit turnarounds, however, knowing where to invest is critical. Our business aims to help customers invest in real estate for resale within markets that will appreciate quickly. Smart investment strategies for our customers will consider many factors, and here we present an integral piece of information by forecasting median sales prices for homes across the U.S. over a five year window. We identify the top five housing markets by U.S. Zip Code that are projected to grow rapidly within the next five years.

### Data and Methods
Using approximately 20 years of published data from Zillow on the monthly median home sale price for nearly 15,000 Zip Codes across the U.S., we examine the projected growth in each Zip Code to see how that market is expected to change in five years. We first performed a five year forecast using the Facebook Profit model for each  Zip Code, and then identified 5 candidate Zip Codes with at least 20 years of data and the largest five year percent growth. We performed cross-validation on the 5 candidate Zip Code models and reforecasted the five year period with an ARIMA model to provide secondary evidence. 

### Key Results
From our analysis, the five best zip codes to invest in are as follows:
34982 - Fort Pierce, FL - projected 71% 5-year appreciation
33982 - Punta Gorda, FL - projected 66% 5-year appreciation
34951 - Fort Pierce, FL - projected 65% 5-year appreciation
37209 - Nashville-Davidson, TN - projected 65% 5-year appreciation
15201 - Pittsburgh, PA - projected 64% 5-year appreciation

The top three Zip Codes are located in coastal Florida communities. While these three Zip Codes have the largest percent growth, they have much larger confidence intervals than the last two, suggesting that Nashville and Pittsburgh may be “safer” investment opportunities. The large confidence intervals for the Florida communities may be a result of lingering effects from the 2008 recession.

### Future Work
This work is preliminary in nature, and in order to corroborate our predictions, we plan to run similar forecasts with other time-series analyses, such as ARMA and SARIMA models. In order to better identify investment opportunities for our customers, we would like to consider other factors in our recommendations, such as population growth, unemployment rate, job growth, and cost of living. 
