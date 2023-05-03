# Rent-Info-Prediction 

## Installation and Useage 
1. Install the necessary dependencies and libaries such as numpy, matplotlib, pandas and sklearn using command: `pip install numpy matplotlib pandas sklearn scikit-learn`
2. Clone or download this repository to your local system.
3. Open the Python script in your preferred text editor or IDE.
4. Run the script and input your desired unit type and date 

You can Open and view the project using the .zip file provided or at my [Github Repo](https://github.com/DialecticalJuche1912/Rental-prediction)

## Functionalities
This program analyzes renting data using various regression and ML models, including Supervised Learning with random forest and linear regression. The data consists of rent prices, the type of unit, and the reference date. 
The code takes user input for unit type and reference date and returns the predicted price using the trained Linear Regression model. If you want to use the Random Forest model instead, replace `lr` with `rf` in the `predict_price` function call. 

### Libraries and dependencies used: 
The following libraries are used in this program:
1. pandas (imported as pd): Work with data frames. It is used to read the CSV file, clean the data, and prepare it for modeling.
2. numpy (imported as np): Is a pandas dependecy that is used for the various numerical operations in this program. U
3. scikit-learn (imported as sklearn): Various functions used in the data manipulation and supervised learning process such as splitting the data as well as training the differentiated sets.

## Acknowledgement
This renting info is from [Government of Canada](https://open.canada.ca/data/en/dataset/13425ff1-aa23-495f-a80d-7178af53bc84) 
