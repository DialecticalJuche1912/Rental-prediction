import pandas as pd 
import numpy as np 

df = pd.read_csv('https://raw.githubusercontent.com/DialecticalJuche1912/renting-info.csv/main/renting-data.csv')
df 

# Data cleaning 
# dropping incomplete data 
complete_df = df.dropna(subset=['Type of unit', 'VALUE'])
# dropping unrelated columns 
complete_df_1 = complete_df[['REF_DATE', 'Type of unit', 'VALUE']]
# Replacing string row value under column Type of unit with integers 
replaced_df = complete_df_1.replace(['Bachelor units','One bedroom units',
                                     'Two bedroom units','Three bedroom units']
                                    , [1,1,2,3])
replaced_df
# rename unit column 
clean_df = replaced_df.rename(columns={"Type of unit":"Number of units"})
clean_df 

# Variable Sorting 
y = clean_df['VALUE']
y
x = clean_df.drop('VALUE', axis=1)
x

# Data Splitting 
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, 
                                                    random_state=100)
x_train
x_test
y_train
y_test 

# Model Bulding 
# Linear Regression 

from sklearn.linear_model import LinearRegression 
# training the model on the following dataset 
lr = LinearRegression()
lr.fit(x_train, y_train)

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

# Here goes 75% of the data into training set
y_lr_train_pred
# Here goes 25% of the data into testing set 
y_lr_test_pred

# Model performance comparison
    #comparing the actual value vs the predicted value, 
    # to see the degree of dispersion they demonstrate once they are lined up
from sklearn.metrics import mean_squared_error, r2_score

# for the training set 
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

# for the testing set 
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# outputting results
print('The Mean Sqaured Error of the training set is: ', lr_train_mse)
print('The R2 score of the training set is:', lr_test_r2)
print('The Mean Sqaured Error of the testing set is: ', lr_test_mse)
print('The R2 score of the testing set is:', lr_test_r2)
lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Testing MSE', 'Testing R2']
lr_results 

# ML supervised learning with Random Forest 
from sklearn.ensemble import RandomForestRegressor 

rf = RandomForestRegressor(max_depth =2, random_state = 100)
rf.fit(x_train, y_train)
   #: Model Application + Prediction making
y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test) 
   #: Model Performance Evaluation 
from sklearn.metrics import mean_squared_error, r2_score

# for the training set 
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

# for the testing set 
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)
# displaying results: 
rf_results = pd.DataFrame(['Random forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Testing MSE', 'Testing R2']
rf_results 

# now we have two two tables: Linear Regression and Random Forest 

# Combine the end results 
df_combined = pd.concat([lr_results, rf_results], 
                        axis=0).reset_index(drop=True)

df_combined
# could also use other regression models from sklearn package 

# Visualization of prediction results
import matplotlib.pyplot as plt 

plt.figure(figsize=(6,7))
plt.scatter (x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.35)

# labelling x and y axies 
line1 = np.polyfit(y_train, y_lr_train_pred, 1)
line2 = np.poly1d(line1)
plt.plot(y_train,line2(y_train), '#E71')
plt.ylabel('Predicted Price of Housing')
plt.xlabel('Actual Price of Housing')

