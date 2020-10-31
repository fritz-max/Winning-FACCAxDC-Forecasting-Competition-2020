# energy-forecasting
This repo contains the code used by *Team Anaconda* to win the Energy forecasting competition 2020 hosted by Finance and Consulting Club Aarhus (FACCA) and Danske Commodities. 

[Event description:](https://facca.dk/events/facca-x-dc-forecasting-competition?fbclid=IwAR1egmMOBNAQKFcqitFw4oWuQKZ58bZE7b6Z2Vwenzez_lLgMJgGhmIn4Gc)
> In collaboration with Danske Commodities, FACCA is inviting you to a virtual Forecasting Competition centered around data and value-generation. DC is a data-driven energy trading company, executing more than 5700 trades a day. Electricity demand is a central feature used specifically to optimise these trading decisions, and any improvements in forecasting accuracy directly generate more value in DC. This Forecasting Competition will make it possible for you to explore your data science skills by building a forecasting model of your own choice.

15 teams of students from Aarhus University participated in the competition that ran from 27th to 29th of October 2020. 

**The Team consisted of:**
- Osvald Frisk
- Mikołaj Plotecki
- Hans-Hendrik Karro
- Friedrich Dörmann 

## The Case
The challenge was about forecasting hourly Energy Consumption in Spain based on a dataset containing weather data for 6 largest cities in spain: Madrid, Barcelona, Valencia, Sevilla, Zaragoza and Malaga.

<img src="https://github.com/fritz-max/energy-forecasting/blob/master/images/spain.png" width="400">

The given features were:
- Timestamps in UTC time
- Weather Features ([More Info](https://apps.ecmwf.int/codes/grib/param-db)):
  - d2m: 2 metre dewpoint temperature
  - t2m: 2 metre temperature
  - i10fg: 10 metre wind gust (wind speeds) 
  - sp: surface pressure 
  - tcc: total cloud coverage 
  - tp: total precipitation
  
Therefore, the dataset consisted of 37 individual features (Time + 6x6 Weather Feature per City).
Training Data was provided for 2015-2018 and the results were evaluated on the data from 2019.

## Data Exploration and Feature Engineering
We started out with data exploration. The Dataset was provided in an already clean and formatted way. As can be expected, we identified strong seasonality in the data (hourly, weekly, monthly, etc.). 

The phase of feature engineering included adding and transforming features in several ways, e.g.:
- Engineering time-related features from the time stamp 
  - This included hour, day of the week/month/year, quarter, business hours, weekend, holidays
- Transforming cloud coverage feature to binary 
  - (Most of the samples were of value 0 or 1 already, as can be seen in the image below) 
  - <img src="https://github.com/fritz-max/energy-forecasting/blob/master/images/binary-cloud-coverage.png" width="500">
- Min-Max-Scaling the weather features accross the 6 cities
- Weighting the weather features by the population size of the respective cities to reflect the impact on energy consumption
  - (i.e. one can expect Madrid to have vastly more influence on energy demand than Malaga, since it has 10x the population)

After feature engineering, our dataset included 48 features.

## Model and Feature Selection
The next step was to select a model and a subset of features. 
We tried different machine learning approaches including *Linear Regression*, *Convolutional neural nets*, *LSTMs*, *Random Forest* and *XGBoost* and ended up choosing **XGBoost** based on initial performance on the complete dataset. 

XGBoosts built-in `feature_importance` function gave insights on the vastly different influences of features on the predicitions. 

<img src="https://github.com/fritz-max/energy-forecasting/blob/master/images/feature-importance.png" width="500">

Therefore we chose to do feature selection to prevent overfitting and reduce training time. This was done using *Forward Stepwise Selection*. 

<img src="https://github.com/fritz-max/energy-forecasting/blob/master/images/feature-selection.png" width="500">

The best validation accuracy was achieved with 25 features. However, the main increase in accuracy already happened by including the 6 most important features, as can be seen in the graph. To prevent overfitting we therefore chose the 6 most important features for our final model:
```python
features = [
    "hour",
    "dayofweek",
    "dayofyear",
    "Madrid_t2m",
    "holidays",
    "Malaga_t2m"
]
```

## Model Training and Hyperparameter Tuning
We then went on to tune the models hyperparameters using Randomized Search and Time-Series Cross Validation. This included search for `max_depth`, `learning-rate`, `min_child_weight` and `gamma` to further improve the models ability to generalize on the data.
The final hyperparameters were:
```python 
n_estimators = 389
params = {
    'learning_rate': 0.0562,
    'max_depth': 7,
    'min_child_weight': 1,
    'gamma': 0.5,
    'objective': 'reg:squarederror'
}
```
Finally the model was trained on the complete training set and the predicitions for the testset were generated.

## Final Results
During Validation, we achieved following results:

| Metric        | Results       | 
| :------------ | :------------ | 
| MSE           |  683543 | 
| MAE           | 644.21 |  
| MAPE          | 2.24 %      | 
| R²            | 0.97      | 

Following plot also shows an example of predictions on the training data:
<img src="https://github.com/fritz-max/energy-forecasting/blob/master/images/prediction.png" width="700">

Finally, the model achieved a MAE of 725.45 on the test set.
