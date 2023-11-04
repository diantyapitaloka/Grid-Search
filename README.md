## Grid Search

## Prepare Dataset
First, download the file "Salary_Data.csv"
- import pandas as pd
 
## Reading Dataset
Turn it into a dataframe
- data = pd.read_csv('Salary_Data.csv')

## Separating Attributes and Labels
Then separate the attributes and labels in the dataset. Do you still remember that if there is only 1 attribute in the dataset, we need to change its shape so that it can be used in model training.
- import numpy as np
- X = data['YearsExperience']
- Y = data['Salary']

## Changing Attributes
Change the shape of the attribute
- X = np.array(X)
- X = X.reshape(-1,1)

## GridSearch
Next, to use grid search, we import the GridSearchCV library from sklearn.model_selection. Then we create the model that we want to test with grid search, in this case the SVR model. Then we create a python dictionary containing the names of the parameters to be tested, as well as their values. Next, we create a grid search object and fill in the parameters. The first parameter is the model we will test. The second parameter is a dictionary which contains a collection of parameters from the model to be tested. Finally, we call the fit() function on the grid search object that has been created.
- from sklearn.model_selection import GridSearchCV
- from sklearn.svm import SVR
 
## Model Parameters
Build a model with C, gamma, and kernel parameters
model = SVR()
- parameters = {'kernel': ['rbf'], 'C': [1000, 10000, 100000], 'gamma': [0.5, 0.05,0.005]}
- grid_search = GridSearchCV(model, parameters)
 
## Function Input
Training a model with a fit function
- grid_search.fit(X,y)

## Calling Attributes
After grid search looks for the best parameters in the model, we can display the best parameters by calling attributes
- best_params_ of the grid search object.

## Displays Parameters
Returns the best parameters of the grid_search object
- print(grid_search.best_params_)

## New model
Next, you can try to create a new SVM model with the grid search results parameters and train it on the data.
Create a new SVM model with the best parameters from the grid search results
- new_model = SVR(C=100000, gamma=0.005, kernel='rbf')
- new_model.fit(X,y)

## Grid Search Visualization
Finally, we can visualize the SVR with the grid search results parameters. It can be seen from the plot results that the grid search succeeded in finding better parameters thereby improving the performance of the model.
- import matplotlib.pyplot as plt
- plt.scatter(X, y)
- plt.plot(X, new_model.predict(X))

![image](https://github.com/diantyapitaloka/Grid-Search/assets/147487436/f96a617a-6d48-4844-bef3-3ce8b8d1cc89)

Grid search will save a lot of time in finding the best parameters of the machine learning model.
