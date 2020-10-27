Welcome to the Virtual Forecasting Competition co-hosted by Danske Commodities and FACCA.

Here are some tips to get you started:

1) Open the "example_model.py" or "example_model.r" to see how you can create a simple model in either Python or R.
2) Find more info on the weather parameters on https://apps.ecmwf.int/codes/grib/param-db
3) Investigate the "preds_template.csv" (or "preds_template.excel" file if you use excel, but DO NOT SUPPLY PREDICTIONS IN EXCEL FORMAT)
4) Make sure your prediction looks EXACTLY similar - i.e., no index columns, no timestamps and 8760 predicted values/rows.
5) Label your predictions file "predictions.csv"

Extra note: If you scale the y-variable - remember to rescale your predictions!


(OPTIONAL) We have provided a "evaluate_preds_format.py" script to ensure your predictions have the correct format. 

Do the following:
1) Install Python and the pandas package
2) Put the "evaluate_preds_format.py" script in the same folder as your "predictions.csv" file (predictions folder)
3a) If you are unfamiliar with python, run the "run_evalate.bat" file.
3b) Otherwise, run the script as you would any python script and see that no error arises.


If you do not run the above OR your model predictions are in a different format - we cannot guarantee that your results 
will be evaluated correctly!




FINAL SUBMISSION

1) Once you have ensured correct format of your model put the "predictions.csv" file and the one_pager into the "submissions" folder.
2) Zip the folder and upload it for evaluation!


Congratulations - you are now done with the Forecasting Competition!