### tasks
- [ ] [download country data](https://climateknowledgeportal.worldbank.org/download-data)
- [ ] write script to merge it with our [datasheet](https://docs.google.com/spreadsheets/d/1qPP5FzFYa5JpMEu8WhVvxF2-kvKqf_0Qg7zEU55pdf0/edit#gid=441730022) with that country - Temp column after 2nd column
<!-- - [ ] for any request check first firebase with the respected code (countryCode + )  -->

### choosing ml model
- Lets understand our requirement, we have two variables as our input that is temperature and year.
- we need to predict the crop production for the next 10 years. 
- so it boils down to a __bi/ multi-variate regression problem__.
- __but!!__
- here one of the input taken as x is nothing but time and hence this becomes a time series problem.
- one of the most commonly used methods for multivariate time series forecasting – __Vector Auto Regression (VAR)__.


### installation
```sh
sudo apt-get python3
sudo apt install python3-pip
pip install pandas numpy tensorflow sklearn
pip3 install -U scikit-learn scipy matplotlib seaborn
```