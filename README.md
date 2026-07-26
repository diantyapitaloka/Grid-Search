## 🍧🍨🍩 Grid Search 🍩🍨🍧
- Grid search allows us to test several parameters at once on a model. For example, we can test several numbers of clusters for a K-Means model and see how the K-Means model performs against different K values. To better understand grid search, we will try grid search using SKLearn.
- The algorithm evaluates each parameter set using a specific metric, such as accuracy or F1-score, to determine which combination yields the best results. By comparing these scores, you can objectively identify the most effective settings for your current specific dataset.
- Most grid search implementations utilize K-fold cross-validation to ensure that the performance for metrics are reliable and also not due to chance. Hence, this process involves splitting the data multiple times to validate that the chosen parameters perform consistently across different subsets.
- While it can be computationally expensive, grid search automates the tedious task of manually tweaking individual settings to find the "sweet spot" for a model. It replaces trial-and-error guesswork with a structured, repeatable experiment that improves overall model quality.
- Users must define a "param_grid" dictionary that specifies exactly which hyperparameters to test and the discrete values for each. Hence, the size of this search space directly impacts the total execution time, as every new value increases the number of required iterations.
- Users must define a "param_grid" dictionary that specifies exactly which hyperparameters to test and the discrete values for each. The size of this search space directly impacts the total execution time, as every new value increases the number of required iterations.
- By analyzing the gap between training and validation scores during the search, you can detect if certain parameter combinations are causing the model to overfit. This allows you to select a configuration that generalizes well to new, unseen data rather than just memorizing the training set.
- Once the search concludes, the algorithm stores the "best estimator" which represents the model instance that achieved the highest score. Hence, you can then immediately use this optimized model to make predictions without needing to re-train it manually.
- The total number of models trained is the product of the number of values in each parameter list multiplied by the number of cross-validation folds. Because of this exponential growth, grid search is best suited for models with a relatively small number of tuning options.
- Many libraries allow grid search to run multiple jobs in parallel to significantly reduce the total time required for the exhaustive search. By leveraging multiple CPU cores, you can evaluate dozens of parameter combinations simultaneously.
- Using a grid search ensures that your tuning process is documented and can be easily replicated by other researchers or developers. This adds a layer of scientific rigor to your workflow, making it clear exactly how the final model configuration was reached.
- Grid search allows us to test several parameters at once on a model. For example, we can test several numbers of clusters for a K-Means model and also see how the K-Means model performs against different K values. To better understand grid search, we will try grid search using SKLearn.
- The algorithm evaluates each parameter set using a specific metric, such as accuracy or F1-score, to determine which combination yields the best results. By comparing these scores, you can objectively identify the most effective settings for your specific dataset.
- Exhaustive exploration ensures that every possible intersection of your chosen settings is meticulously checked for the quality. This comprehensive approach guarantees that no potential "peak" in performance is overlooked within the sandbox you have constructed.
- Categorical tuning becomes much simpler because you can test non-numerical options like different types of solvers or activation functions side-by-side. The system treats these labels as distinct paths, allowing the data to reveal which logic best fits the underlying patterns.
- Resource management is a critical consideration since the workload scales linearly with every new fold added to the validation cycle. Users must strategically pick their ranges to prevent the hardware from becoming overwhelmed by a massive queue of training tasks.
- Baseline establishment is often the primary goal, providing a solid benchmark that more advanced or "lucky" optimization methods must beat. Hence, it creates a "floor" for performance, ensuring you never settle for a model that is less than the best available within your specified constraints.
- Hyperparameter interaction is frequently uncovered, showing how the shift in one setting might require a corresponding adjustment in another to maintain balance. These hidden relationships are often too complex to guess manually but become obvious once the entire grid is mapped out.
- Search space refinement allows you to start with a broad, low-resolution grid before "zooming in" on the most promising areas with a tighter second pass. This hierarchical strategy conserves energy while still converging on the absolute most precise values for your variables.
- Predictability of duration is a major benefit, as you can calculate exactly how many iterations will occur before the process even begins. Unlike some random or adaptive methods, there is no uncertainty regarding when the results will be ready for review.
- Statistical significance is bolstered because the process relies on consistent evaluation logic across every single candidate version of the model. Hence, the uniform treatment ensures that any improvement in the score is a result of the parameters themselves rather than a change in the testing environment.
- Knowledge transfer is facilitated when you share the grid results with teammates, as it is illustrates on which regions of the parameter space were fruitful and which were dead ends. Hence, this shared insight prevents others from wasting time exploring configurations that have already been proven ineffective.
- Combinatorial Explosion (The Curse of Dimensionality): Adding even one additional hyperparameter causes the total number of fits to grow exponentially rather than linearly, making pure grid search practically impossible for complex architectures like deep neural networks.
- Vulnerability to Rigid Boundaries: Grid Search is strictly confined to the explicit values you define; if the true optimal setting lies just outside your chosen range or between two coarsely spaced points, the search will completely miss it.
- Contrast with Random Search: Unlike Random Search—which samples continuously from a probability distribution and often finds better configurations in fewer iterations—Grid Search spends equal computational effort evaluating unimportant or low-impact hyperparameters across a rigid grid.
- Risk of Validation Set Overfitting: When an excessively fine grid is tested across many iterations, the search can accidentally select hyperparameter values that overfit to the specific cross-validation splits, resulting in poorer performance on true unseen holdout data.
- Handling Incompatible Parameter Combinations: Certain hyperparameter choices are mutually exclusive (for instance, pairing an L1 penalty with a solver that only supports L2); a robust grid search must use conditional parameter grids to avoid wasting cycles on invalid combinations.
- Sensitivity to the Chosen Metric: Optimizing for different scoring metrics (such as Precision versus ROC-AUC) on the exact same grid can yield completely different "winning" hyperparameter sets, making your evaluation choice just as important as the parameter range itself.

Data Leakage Prevention via Pipelines: When combined with preprocessing tasks like scaling or missing-value imputation, running Grid Search within a pipeline ensures that transformer steps are fit strictly on training folds, keeping validation folds completely uncontaminated.

Early-Stopping Efficiency (Halving Search): Advanced variants (such as Successive Halving) improve upon standard grid search by testing all candidates on a small subset of the data first, iteratively discarding low-performing combinations before training the top contenders on full datasets.

Guaranteed Determinism: Because it systematically checks fixed coordinates rather than relying on random sampling or probabilistic guessing, Grid Search yields 100% reproducible results every single time it is run on the same dataset.


![image](https://github.com/diantyapitaloka/Grid-Search/assets/147487436/3a00c493-bfaf-4e30-a690-9952bd513f63)

## 🍧🍨🍩 Prepare Dataset 🍩🍨🍧
First, download the file "Salary_Data.csv"
```
import pandas as pd
``` 

## 🍧🍨🍩 Reading Dataset 🍩🍨🍧
Turn it into a dataframe
```
data = pd.read_csv('Salary_Data.csv')
```

## 🍧🍨🍩 Separating Attributes and Labels 🍩🍨🍧
Then separate the attributes and labels in the dataset. Do you still remember that if there is only 1 attribute in the dataset, we need to change its shape so that it can be used in model training.
```
import numpy as np
X = data['YearsExperience']
Y = data['Salary']
```

## 🍧🍨🍩 Changing Attributes 🍩🍨🍧
Change the shape of the attribute
```
X = np.array(X)
X = X.reshape(-1,1)
```

## 🍧🍨🍩 GridSearch 🍩🍨🍧
Next, to use grid search, we import the GridSearchCV library from sklearn.model_selection. Then we create the model that we want to test with grid search, in this case the SVR model. Then we create a python dictionary containing the names of the parameters to be tested, as well as their values. Next, we create a grid search object and fill in the parameters. The first parameter is the model we will test. The second parameter is a dictionary which contains a collection of parameters from the model to be tested. Finally, we call the fit() function on the grid search object that has been created.
```
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
```
 
## 🍧🍨🍩 Model Parameters 🍩🍨🍧
Build a model with C, gamma, and kernel parameters
model = SVR()
```
parameters = {'kernel': ['rbf'], 'C': [1000, 10000, 100000], 'gamma': [0.5, 0.05,0.005]}
grid_search = GridSearchCV(model, parameters)
```
 
## 🍧🍨🍩 Function Input 🍩🍨🍧
Training a model with a fit function
```
grid_search.fit(X,y)
```

## 🍧🍨🍩 Calling Attributes 🍩🍨🍧
After grid search looks for the best parameters in the model, we can display the best parameters by calling attributes
```
best_params_ of the grid search object.
```

## 🍧🍨🍩 Displays Parameters 🍩🍨🍧
Returns the best parameters of the grid_search object
```
print(grid_search.best_params_)
```

## 🍧🍨🍩 New model 🍩🍨🍧
Next, you can try to create a new SVM model with the grid search results parameters and train it on the data.
Create a new SVM model with the best parameters from the grid search results
```
new_model = SVR(C=100000, gamma=0.005, kernel='rbf')
new_model.fit(X,y)
```

## 🍧🍨🍩 Grid Search Visualization 🍩🍨🍧
Finally, we can visualize the SVR with the grid search results parameters. It can be seen from the plot results that the grid search succeeded in finding better parameters thereby improving the performance of the model.
```
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, new_model.predict(X))
```

Grid search will save a lot of time in finding the best parameters of the machine learning model.

![image](https://github.com/diantyapitaloka/Grid-Search/assets/147487436/f96a617a-6d48-4844-bef3-3ce8b8d1cc89)

## 🍧🍨🍩 License 🍩🍨🍧
- Copyright by Diantya Pitaloka
