# energy-forecasting
This repo contains the code used by *Team Anaconda* in the Energy forecasting competition 2020 hosted by Finance and Consulting Club Aarhus (FACCA) and Danske Commodities. 

[Event description:](https://facca.dk/events/facca-x-dc-forecasting-competition?fbclid=IwAR1egmMOBNAQKFcqitFw4oWuQKZ58bZE7b6Z2Vwenzez_lLgMJgGhmIn4Gc)
> In collaboration with Danske Commodities, FACCA is inviting you to a virtual Forecasting Competition centered around data and value-generation. DC is a data-driven energy trading company, executing more than 5700 trades a day. Electricity demand is a central feature used specifically to optimise these trading decisions, and any improvements in forecasting accuracy directly generate more value in DC. This Forecasting Competition will make it possible for you to explore your data science skills by building a forecasting model of your own choice.

15 teams of students from Aarhus University participated in the competition that ran from 27th to 29th of October 2020. 
We won the competition with the approach outlined in this repo.

More information: 
- Linkedin Post: ?

## The Case
The challenge was about forecasting hourly Energy Consumption in Spain based on a dataset containing weather data for 6 largest cities in spain. 

<img src="https://github.com/fritz-max/energy-forecasting/blob/master/images/spain.png" width="300">

The given features were:
- Timestamps in UTC time
- Weather Features ([More Info](https://apps.ecmwf.int/codes/grib/param-db)):
  - d2m: 2 metre dewpoint temperature
  - t2m: 2 metre temperature
  - i10fg: 10 metre wind gust (wind speeds) 
  - sp: surface pressure 
  - tcc: total cloud coverage 
  - tp: total precipitation
  
Training Data was provided for 2015-2018 and the results were evaluated on the data from 2019.

## Data Exploration and Feature Engineering
<img src="https://github.com/fritz-max/energy-forecasting/blob/master/images/binary-cloud-coverage.png" width="400">
<img src="https://github.com/fritz-max/energy-forecasting/blob/master/images/feature-importance.png" width="400">


## Model and Feature Selection

<img src="https://github.com/fritz-max/energy-forecasting/blob/master/images/feature-selection.png" width="400">


## Model Training and Hyperparameter Tuning

## Final Results
<img src="https://github.com/fritz-max/energy-forecasting/blob/master/images/prediction.png" width="500">



## The Team
- Osvald Frisk
- Mikołaj Plotecki
- Hans-Hendrik Karro
- Friedrich Dörmann

