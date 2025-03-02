import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("Student_Marks.csv")   #### To Read the Input CSV file
print(df.head())     #### print the first 5 rows of each columns
print(df.describe())    #####

Null_Columns =(df.isna())
print("Null Columns", Null_Columns)

x = df[["number_courses","time_study"]]   ##### Input dependent variables
y = df["Marks"]    ### Marks is my target variable Y

model = LinearRegression()   ##### Create an Instance for Linear Regression Model
x_train, x_test, Y_train,y_test = train_test_split(x, y, train_size=0.8, random_state=42)    ##### Train the dataset
model.fit(x,y)   ############# Fit the Linear Regression model of Input Dependent variables and Target Variables

print(model.predict([[3,8]]))    ########## Predict the Marks who taken 3 coureses and studied for 8 Hours
print ("Maximum Marks in a Marks Column", df["Marks"].max())        ######### Print the maximum number of marks scored

print("Model Score is ", model.score(x,y))
