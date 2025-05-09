## **Fraudulent Claim Detection**
## Problem Statement
Global Insure, a leading insurance company, processes thousands of claims annually. However, a significant percentage of these claims turn out to be fraudulent, resulting in considerable financial losses. The company’s current process for identifying fraudulent claims involves manual inspections, which is time-consuming and inefficient. Fraudulent claims are often detected too late in the process, after the company has already paid out significant amounts. Global Insure wants to improve its fraud detection process using data-driven insights to classify claims as fraudulent or legitimate early in the approval process. This would minimise financial losses and optimise the overall claims handling process.

## Business Objective
Global Insure wants to build a model to classify insurance claims as either fraudulent or legitimate based on historical claim details and customer profiles. By using features like claim amounts, customer profiles and claim types, the company aims to predict which claims are likely to be fraudulent before they are approved.


Based on this assignment, you have to answer the following questions:<br>

● How can we analyse historical claim data to detect patterns that indicate fraudulent claims?<br>
● Which features are most predictive of fraudulent behaviour?<br>
● Can we predict the likelihood of fraud for an incoming claim, based on past data?<br>
● What insights can be drawn from the model that can help in improving the fraud detection process?<br>
## Assignment Tasks
You need to perform the following steps for successfully completing this assignment:
1. Data Preparation
2. Data Cleaning
3. Train Validation Split 70-30
4. EDA on Training Data
5. EDA on Validation Data (optional)
6. Feature Engineering
7. Model Building
8. Predicting and Model Evaluation
## Data Dictionary
The insurance claims data has 40 Columns and 1000 Rows. Following data dictionary provides the description for each column present in dataset:<br>

<table>
  <thead>
    <tr>
      <th>Column Name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>months_as_customer</td>
      <td>Represents the duration in months that a customer has been associated with the insurance company.</td>
    </tr>
    <tr>
      <td>age</td>
      <td>Represents the age of the insured person.</td>
    </tr>
    <tr>
      <td>policy_number</td>
      <td>Represents a unique identifier for each insurance policy.</td>
    </tr>
    <tr>
      <td>policy_bind_date</td>
      <td>Represents the date when the insurance policy was initiated.</td>
    </tr>
    <tr>
      <td>policy_state</td>
      <td>Represents the state where the insurance policy is applicable.</td>
    </tr>
    <tr>
      <td>policy_csl</td>
      <td>Represents the combined single limit for the insurance policy.</td>
    </tr>
    <tr>
      <td>policy_deductable</td>
      <td>Represents the amount that the insured person needs to pay before the insurance coverage kicks in.</td>
    </tr>
    <tr>
      <td>policy_annual_premium</td>
      <td>Represents the yearly cost of the insurance policy.</td>
    </tr>
    <tr>
      <td>umbrella_limit</td>
      <td>Represents an additional layer of liability coverage provided beyond the limits of the primary insurance policy.</td>
    </tr>
    <tr>
      <td>insured_zip</td>
      <td>Represents the zip code of the insured person.</td>
    </tr>
    <tr>
      <td>insured_sex</td>
      <td>Represents the gender of the insured person.</td>
    </tr>
    <tr>
      <td>insured_education_level</td>
      <td>Represents the highest educational qualification of the insured person.</td>
    </tr>
    <tr>
      <td>insured_occupation</td>
      <td>Represents the profession or job of the insured person.</td>
    </tr>
    <tr>
      <td>insured_hobbies</td>
      <td>Represents the hobbies or leisure activities of the insured person.</td>
    </tr>
    <tr>
      <td>insured_relationship</td>
      <td>Represents the relationship of the insured person to the policyholder.</td>
    </tr>
    <tr>
      <td>capital-gains</td>
      <td>Represents the profit earned from the sale of assets such as stocks, bonds, or real estate.</td>
    </tr>
    <tr>
      <td>capital-loss</td>
      <td>Represents the loss incurred from the sale of assets such as stocks, bonds, or real estate.</td>
    </tr>
    <tr>
      <td>incident_date</td>
      <td>Represents the date when the incident or accident occurred.</td>
    </tr>
    <tr>
      <td>incident_type</td>
      <td>Represents the category or type of incident that led to the claim.</td>
    </tr>
    <tr>
      <td>collision_type</td>
      <td>Represents the type of collision that occurred in an accident.</td>
    </tr>
    <tr>
      <td>incident_severity</td>
      <td>Represents the extent of damage or injury caused by the incident.</td>
    </tr>
    <tr>
      <td>authorities_contacted</td>
      <td>Represents the authorities or agencies that were contacted after the incident.</td>
    </tr>
    <tr>
      <td>incident_state</td>
      <td>Represents the state where the incident occurred.</td>
    </tr>
    <tr>
      <td>incident_city</td>
      <td>Represents the city where the incident occurred.</td>
    </tr>
    <tr>
      <td>incident_location</td>
      <td>Represents the specific location or address where the incident occurred.</td>
    </tr>
    <tr>
      <td>incident_hour_of_the_day</td>
      <td>Represents the hour of the day when the incident occurred.</td>
    </tr>
    <tr>
      <td>number_of_vehicles_involved</td>
      <td>Represents the total number of vehicles involved in the incident.</td>
    </tr>
    <tr>
      <td>property_damage</td>
      <td>Represents whether there was any damage to property in the incident.</td>
    </tr>
    <tr>
      <td>bodily_injuries</td>
      <td>Represents the number of bodily injuries resulting from the incident.</td>
    </tr>
    <tr>
      <td>witnesses</td>
      <td>Represents the number of witnesses present at the scene of the incident.</td>
    </tr>
    <tr>
      <td>police_report_available</td>
      <td>Represents whether a police report is available for the incident.</td>
    </tr>
    <tr>
      <td>total_claim_amount</td>
      <td>Represents the total amount claimed by the insured person for the incident.</td>
    </tr>
    <tr>
      <td>injury_claim</td>
      <td>Represents the amount claimed for injuries sustained in the incident.</td>
    </tr>
    <tr>
      <td>property_claim</td>
      <td>Represents the amount claimed for property damage in the incident.</td>
    </tr>
    <tr>
      <td>vehicle_claim</td>
      <td>Represents the amount claimed for vehicle damage in the incident.</td>
    </tr>
    <tr>
      <td>auto_make</td>
      <td>Represents the manufacturer of the insured vehicle.</td>
    </tr>
    <tr>
      <td>auto_model</td>
      <td>Represents the specific model of the insured vehicle.</td>
    </tr>
    <tr>
      <td>auto_year</td>
      <td>Represents the year of manufacture of the insured vehicle.</td>
    </tr>
    <tr>
      <td>fraud_reported</td>
      <td>Represents whether the claim was reported as fraudulent or not.</td>
    </tr>
    <tr>
      <td>_c39</td>
      <td>Represents an unknown or unspecified variable.</td>
    </tr>
  </tbody>
</table>
## **1. Data Preparation**
In this step, read the dataset provided in CSV format and look at basic statistics of the data, including preview of data, dimension of data, column descriptions and data types.
### **1.0 Import Libraries**
# Supress unnecessary warnings
import warnings
warnings.filterwarnings("ignore")
# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
### **1.1 Load the Data**
# Load the dataset

# Check at the first few entries

# Inspect the shape of the dataset

# Inspect the features in the dataset

## **2. Data Cleaning** <font color = red>[10 marks]</font>
### **2.1 Handle null values** <font color = red>[2 marks]</font>
#### **2.1.1** Examine the columns to determine if any value or column needs to be treated <font color="red">[1 Mark]</font>
# Check the number of missing values in each column

#### **2.1.2** Handle rows containing null values <font color="red">[1 Mark]</font>
# Handle the rows containing null values

### **2.2 Identify and handle redundant values and columns** <font color = red>[5 marks]</font>
#### **2.2.1** Examine the columns to determine if any value or column needs to be treated <font color="red">[2 Mark]</font>
# Write code to display all the columns with their unique values and counts and check for redundant values

#### **2.2.2** Identify and drop any columns that are completely empty <font color="red">[1 Mark]</font>
# Identify and drop any columns that are completely empty

#### **2.2.3** Identify and drop rows where features have illogical or invalid values, such as negative values for features that should only have positive values <font color="red">[1 Mark]</font>
# Identify and drop rows where features have illogical or invalid values, such as negative values for features that should only have positive values

#### **2.2.4** Identify and remove columns where a large proportion of the values are unique or near-unique, as these columns are likely to be identifiers or have very limited predictive power <font color="red">[1 Mark]</font>
# Identify and remove columns that are likely to be identifiers or have very limited predictive power

# Check the dataset

### **2.3 Fix Data Types** <font color = red>[3 marks]</font>
Carefully examine the dataset and identify columns that contain date or time information but are not stored as the appropriate data type. Convert these columns to the correct datetime data type to enable proper analysis and manipulation of temporal information.
# Fix the data types of the columns with incorrect data types

# Check the features of the data again

## **3. Train-Validation Split** <font color = red>[5 marks]</font>
### **3.1 Import required libraries**  
# Import train-test-split
from sklearn.model_selection import train_test_split
### **3.2 Define feature and target variables** <font color = red>[2 Marks]</font>
# Put all the feature variables in X

# Put the target variable in y

### **3.3 Split the data** <font color="red">[3 Marks]</font>
# Split the dataset into 70% train and 30% validation and use stratification on the target variable

# Reset index for all train and test sets

## **4. EDA on training data** <font color = red>[20 marks]</font>
### **4.1 Perform univariate analysis** <font color = red>[5 marks]</font>
#### **4.1.1** Identify and select numerical columns from training data for univariate analysis <font color = "red">[1 Mark]</font>
# Select numerical columns

#### **4.1.2** Visualise the distribution of selected numerical features using appropriate plots to understand their characteristics <font color = "red">[4 Marks]</font>
# Plot all the numerical columns to understand their distribution

### **4.2 Perform correlation analysis** <font color="red">[3 Marks]</font>
 Investigate the relationships between numerical features to identify potential multicollinearity or dependencies. Visualise the correlation structure using an appropriate method to gain insights into feature relationships.
# Create correlation matrix for numerical columns

# Plot Heatmap of the correlation matrix

### **4.3 Check class balance** <font color="red">[2 Marks]</font>
Examine the distribution of the target variable to identify potential class imbalances using visualisation for better understanding.
# Plot a bar chart to check class balance

### **4.4 Perform bivariate analysis** <font color="red">[10 Marks]</font>
#### **4.4.1** Target likelihood analysis for categorical variables. <font color="red">[5 Marks]</font>
Investigate the relationships between categorical features and the target variable by analysing the target event likelihood (for the `'Y'` event) for each level of every relevant categorical feature. Through this analysis, identify categorical features that do not contribute much in explaining the variation in the target variable.
# Write a function to calculate and analyse the target variable likelihood for categorical features

#### **4.4.2** Explore the relationships between numerical features and the target variable to understand their impact on the target outcome using appropriate visualisation techniques to identify trends and potential interactions. <font color="red">[5 Marks]</font>
# Visualise the relationship between numerical features and the target variable to understand their impact on the target outcome

## **5. EDA on validation data** <font color = red>[OPTIONAL]</font>
### **5.1 Perform univariate analysis**
#### **5.1.1** Identify and select numerical columns from training data for univariate analysis.
# Select numerical columns

#### **5.1.2** Visualise the distribution of selected numerical features using appropriate plots to understand their characteristics.
# Plot all the numerical columns to understand their distribution

### **5.2 Perform correlation analysis**
 Investigate the relationships between numerical features to identify potential multicollinearity or dependencies. Visualise the correlation structure using an appropriate method to gain insights into feature relationships.
# Create correlation matrix for numerical columns

# Plot Heatmap of the correlation matrix

### **5.3 Check class balance**
Examine the distribution of the target variable to identify potential class imbalances. Visualise the distribution for better understanding.
# Plot a bar chart to check class balance

### **5.4 Perform bivariate analysis**
#### **5.4.1** Target likelihood analysis for categorical variables.
Investigate the relationships between categorical features and the target variable by analysing the target event likelihood (for the `'Y'` event) for each level of every relevant categorical feature. Through this analysis, identify categorical features that do not contribute much in explaining the variation in the target variable.
# Write a function to calculate and analyse the target variable likelihood for categorical features

#### **5.4.2** Explore the relationships between numerical features and the target variable to understand their impact on the target outcome. Utilise appropriate visualisation techniques to identify trends and potential interactions.
# Visualise the relationship between numerical features and the target variable to understand their impact on the target outcome

## **6. Feature Engineering** <font color = red>[25 marks]</font>
### **6.1 Perform resampling** <font color="red">[3 Marks]</font>
Handle class imbalance in the training data by applying resampling technique.

Use the **RandomOverSampler** technique to balance the data and handle class imbalance. This method increases the number of samples in the minority class by randomly duplicating them, creating synthetic data points with similar characteristics. This helps prevent the model from being biased toward the majority class and improves its ability to predict the minority class more accurately.

**Note:** You can try other resampling techniques to handle class imbalance
# Import RandomOverSampler from imblearn library
from imblearn.over_sampling import RandomOverSampler

# Perform resampling on training data

### **6.2 Feature Creation** <font color="red">[4 marks]</font>
Create new features from existing ones to enhance the model's ability to capture patterns in the data. This may involve deriving features from date/time columns, combining features, or creating interaction terms.
# Create new features based on your understanding for both training and validation data

### **6.3 Handle redundant columns** <font color="red">[3 marks]</font>
Analyse the data to identify features that may be redundant or contribute minimal information toward predicting the target variable and drop them.

- You can consider features that exhibit high correlation with other variables, which you may have observed during the EDA phase.
- Features that don't strongly influence the prediction, which you may have observed during the EDA phase.
- Categorical columns with low value counts for some levels can be remapped to reduce number of unique levels, and features with very high counts for just one level may be removed, as they resemble unique identifier columns and do not provide substantial predictive value.
- Additionally, eliminate any columns from which the necessary features have already been extracted in the preceding step.
# Drop redundant columns from training and validation data

# Check the data

### **6.4 Combine values in Categorical Columns** <font color="red">[6 Marks]</font>
During the EDA process, categorical columns with multiple unique values may be identified. To enhance model performance, it is essential to refine these categorical features by grouping values that have low frequency or provide limited predictive information.

Combine categories that occur infrequently or exhibit similar behavior to reduce sparsity and improve model generalisation.
# Combine categories that have low frequency or provide limited predictive information

### **6.5 Dummy variable creation** <font color="red">[6 Marks]</font>
Transform categorical variables into numerical representations using dummy variables. Ensure consistent encoding between training and validation data.
#### **6.5.1** Identify categorical columns for dummy variable creation <font color="red">[1 Mark]</font>
# Identify the categorical columns for creating dummy variables

#### **6.5.2** Create dummy variables for categorical columns in training data <font color="red">[2 Marks]</font>
# Create dummy variables using the 'get_dummies' for categorical columns in training data

#### **6.5.3** Create dummy variables for categorical columns in validation data <font color="red">[2 Marks]</font>
# Create dummy variables using the 'get_dummies' for categorical columns in validation data

#### **6.5.4** Create dummy variable for dependent feature in training and validation data <font color = "red">[1 Mark]</font>
# Create dummy variable for dependent feature in training data

# Create dummy variable for dependent feature in validation data

### **6.6 Feature scaling** <font color = red>[3 marks]</font>
Scale numerical features to a common range to prevent features with larger values from dominating the model.  Choose a scaling method appropriate for the data and the chosen model. Apply the same scaling to both training and validation data.
# Import the necessary scaling tool from scikit-learn

# Scale the numeric features present in the training data

# Scale the numeric features present in the validation data

## **7. Model Building** <font color = red>[50 marks]</font>
In this task, you will be building two machine learning models: Logistic Regression and Random Forest. Each model will go through a structured process to ensure optimal performance. The key steps for each model are outlined below:

**Logistic Regression Model**
- Feature Selection using RFECV – Identify the most relevant features using Recursive Feature Elimination with Cross-Validation.
- Model Building and Multicollinearity Assessment – Build the logistic regression model and analyse statistical aspects such as p-values and VIFs to detect multicollinearity.
- Model Training and Evaluation on Training Data – Fit the model on the training data and assess initial performance.
- Finding the Optimal Cutoff – Determine the best probability threshold by analysing the sensitivity-specificity tradeoff and precision-recall tradeoff.
- FInal Prediction and Evaluation on Training Data using the Optimal Cutoff – Generate final predictions using the selected cutoff and evaluate model performance.

**Random Forest Model**
- Get Feature Importances - Obtain the importance scores for each feature and select the important features to train the model.
- Model Evaluation on Training Data – Assess performance metrics on the training data.
- Check Model Overfitting using Cross-Validation – Evaluate generalisation by performing cross-validation.
- Hyperparameter Tuning using Grid Search – Optimise model performance by fine-tuning hyperparameters.
- Final Model and Evaluation on Training Data – Train the final model using the best parameters and assess its performance.
### **7.1 Feature selection** <font color = red>[4 marks]</font>
Identify and select the most relevant features for building a logistic regression model using Recursive Feature Elimination with Cross-Validation (RFECV).
#### **7.1.1** Import necessary libraries <font color="red">[1 Mark]</font>
# Import necessary libraries

#### **7.1.2** Perform feature selection <font color="red">[2 Mark]</font>
# Apply RFECV to identify the most relevant features

# Display the features ranking by RFECV in a DataFrame

#### **7.1.2** Retain the selected features <font color="red">[1 Mark]</font>
# Put columns selected by RFECV into variable 'col'

### **7.2 Build Logistic Regression Model** <font color = red>[12 marks]</font>
After selecting the optimal features using RFECV, utilise these features to build a logistic regression model with Statsmodels. This approach enables a detailed statistical analysis of the model, including the assessment of p-values and Variance Inflation Factors (VIFs). Evaluating these metrics is crucial for detecting multicollinearity and ensuring that the selected predictors are not highly correlated.
#### **7.2.1** Select relevant features and add constant in training data <font color="red">[1 Mark]</font>
# Select only the columns selected by RFECV

# Import statsmodels and add constant

# Check the data

#### **7.2.2** Fit logistic regression model <font color="red">[2 Marks]</font>
# Fit a logistic Regression model on X_train after adding a constant and output the summary

**Model Interpretation**

The output summary table will provide the features used for building model along with coefficient of each of the feature and their p-value. The p-value in a logistic regression model is used to assess the statistical significance of each coefficient. Lesser the p-value, more significant the feature is in the model.

A positive coefficient will indicate that an increase in the value of feature would increase the odds of the event occurring. On the other hand, a negative coefficient means the opposite, i.e, an increase in the value of feature would decrease the odds of the event occurring.
Now check VIFs for presence of multicollinearity in the model.
#### **7.2.3** Evaluate VIF of features to assess multicollinearity <font color="red">[2 Marks]</font>
# Import 'variance_inflation_factor'
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Make a VIF DataFrame for all the variables present

Proceed to the next step if p-values and VIFs are within acceptable ranges. If you observe high p-values or VIFs, drop the features and retrain the model. <font color="red">[THIS IS OPTIONAL]</font>
#### **7.2.4** Make predictions on training data <font color = "red">[1 Mark]</font>
# Predict the probabilities on the training data

# Reshape it into an array

#### **7.2.5** Create a DataFrame that includes actual fraud reported flags, predicted probabilities, and a column indicating predicted classifications based on a cutoff value of 0.5 <font color="red">[1 Mark]</font>


# Create a new DataFrame containing the actual fraud reported flag and the probabilities predicted by the model

# Create new column indicating predicted classifications based on a cutoff value of 0.5

**Model performance evaluation**

Evaluate the performance of the model based on predictions made on the training data.
#### **7.2.6** Check the accuracy of the model <font color = "red">[1 Mark]</font>
# Import metrics from sklearn for evaluation
from sklearn import metrics

# Check the accuracy of the model

#### **7.2.7** Create a confusion matrix based on the predictions made on the training data <font color="red">[1 Mark]</font>
# Create confusion matrix

#### **7.2.8** Create variables for true positive, true negative, false positive and false negative <font color="red">[1 Mark]</font>
# Create variables for true positive, true negative, false positive and false negative

#### **7.2.9** Calculate sensitivity, specificity, precision, recall and F1-score <font color="red">[2 Marks]</font>
# Calculate the sensitivity


# Calculate the specificity


# Calculate Precision


# Calculate Recall


# Calculate F1 Score

### **7.3 Find the Optimal Cutoff** <font color = red>[12 marks]</font>
Find the optimal cutoff to improve model performance by evaluating various cutoff values and their impact on relevant metrics.
#### **7.3.1** Plot ROC Curve  to visualise the trade-off between true positive rate and false positive rate across different classification thresholds <font color="red">[2 Marks]</font>
# Import libraries or function to plot the ROC curve


# Define ROC function

# Call the ROC function

**Sensitivity and Specificity tradeoff**

After analysing the area under the curve of the ROC, check the sensitivity and specificity tradeoff to find the optimal cutoff point.
#### **7.3.2** Predict on training data at various probability cutoffs <font color="red">[1 Mark]</font>
# Create columns with different probability cutoffs to explore the impact of cutoff on model performance

#### **7.3.3** Plot accuracy, sensitivity, specificity at different values of probability cutoffs <font color="red">[2 Marks]</font>
# Create a DataFrame to see the values of accuracy, sensitivity, and specificity at different values of probability cutoffs

# Plot accuracy, sensitivity, and specificity at different values of probability cutoffs

#### **7.3.4** Create a column for final prediction based on optimal cutoff <font color="red">[1 Mark]</font>
# Create a column for final prediction based on the optimal cutoff

#### **7.3.5** Calculate the accuracy <font color="red">[1 Mark]</font>
# Check the accuracy now

#### **7.3.6** Create confusion matrix <font color="red">[1 Mark]</font>
# Create the confusion matrix once again

#### **7.3.7** Create variables for true positive, true negative, false positive and false negative <font color="red">[1 Mark]</font>
# Create variables for true positive, true negative, false positive and false negative

#### **7.3.8** Calculate sensitivity, specificity, precision, recall and F1-score of the model <font color="red">[2 Mark]</font>
# Calculate the sensitivity


# Calculate the specificity


# Calculate Precision


# Calculate Recall


# Calculate F1 Score

**Precision and Recall tradeoff**

Check optimal cutoff value by plotting precision-recall curve, and adjust the cutoff based on precision and recall tradeoff if required.
# Import precision-recall curve function
from sklearn.metrics import precision_recall_curve
#### **7.3.9** Plot precision-recall curve <font color="red">[1 Mark]</font>
# Plot precision-recall curve

### **7.4 Build Random Forest Model** <font color = red>[12 marks]</font>
Now that you have built a logistic regression model, let's move on to building a random forest model.
#### **7.4.1** Import necessary libraries
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
#### **7.4.2** Build the random forest model <font color="red">[1 Mark]</font>
# Build a base random forest model

#### **7.4.3** Get feature importance scores and select important features <font color="red">[2 Marks]</font>
# Get feature importance scores from the trained model

# Create a DataFrame to visualise the importance scores

# Select features with high importance scores

# Create a new training data with only the selected features

#### **7.4.4** Train the model with selected features <font color="red">[1 Mark]</font>
# Fit the model on the training data with selected features

#### **7.4.5** Generate predictions on the training data <font color="red">[1 Mark]</font>
# Generate predictions on training data

#### **7.4.6** Check accuracy of the model <font color="red">[1 Mark]</font>
# Check accuracy of the model

#### **7.4.7** Create confusion matrix <font color="red">[1 Mark]</font>
# Create the confusion matrix to visualise the performance

#### **7.4.8** Create variables for true positive, true negative, false positive and false negative <font color="red">[1 Mark]</font>
# Create variables for true positive, true negative, false positive and false negative

#### **7.4.9** Calculate sensitivity, specificity, precision, recall and F1-score of the model <font color="red">[2 Marks]</font>
# Calculate the sensitivity


# Calculate the specificity


# Calculate Precision


# Calculate Recall


# Calculate F1 Score

#### **7.4.10** Check if the model is overfitting training data using cross validation <font color = "red">[2 marks]</font>
# Use cross validation to check if the model is overfitting

### **7.5 Hyperparameter Tuning** <font color = red>[10 Marks]</font>
 Enhance the performance of the random forest model by systematically exploring and selecting optimal hyperparameter values using grid search.
#### **7.5.1** Use grid search to find the best hyperparameter values <font color = red>[2 Marks]</font>
# Use grid search to find the best hyperparamter values

# Best Hyperparameters

#### **7.5.2** Build a random forest model based on hyperparameter tuning results <font color = red>[1 Mark]</font>
# Building random forest model based on results of hyperparameter tuning

#### **7.5.3** Make predictions on training data <font color = red>[1 Mark]</font>
# Make predictions on training data

#### **7.5.4** Check accuracy of Random Forest Model <font color = red>[1 Mark]</font>
# Check the accuracy

#### **7.5.5** Create confusion matrix <font color = red>[1 Mark]</font>
# Create the confusion matrix

#### **7.5.6** Create variables for true positive, true negative, false positive and false negative <font color = red>[1 Mark]</font>
# Create variables for true positive, true negative, false positive and false negative

#### **7.5.7** Calculate sensitivity, specificity, precision, recall and F1-score of the model <font color = red>[3 Marks]</font>
# Calculate the sensitivity


# Calculate the specificity


# Calculate Precision


# Calculate Recall


# Calculate F1-Score

## **8. Prediction and Model Evaluation** <font color = red>[20 marks]</font>
Use the model from the previous step to make predictions on the validation data with the optimal cutoff. Then evaluate the model's performance using metrics such as accuracy, sensitivity, specificity, precision, and recall.
### **8.1 Make predictions over validation data using logistic regression model** <font color = red>[10 marks]</font>
#### **8.1.1** Select relevant features for validation data and add constant <font color="red">[1 Mark]</font>
# Select the relevant features for validation data

# Add constant to X_validation

#### **8.1.2** Make predictions over validation data <font color="red">[1 Mark]</font>
# Make predictions on the validation data and store it in the variable 'y_validation_pred'

#### **8.1.3** Create DataFrame with actual values and predicted values for validation data <font color="red">[2 Marks]</font>
#  Create DataFrame with actual values and predicted values for validation data

#### **8.1.4** Make final prediction based on cutoff value <font color="red">[1 Mark]</font>
# Make final predictions on the validation data using the optimal cutoff

#### **8.1.5** Check the accuracy of logistic regression model on validation data <font color="red">[1 Mark]</font>
# Check the accuracy

#### **8.1.6** Create confusion matrix <font color="red">[1 Mark]</font>
# Create the confusion matrix

#### **8.1.7** Create variables for true positive, true negative, false positive and false negative <font color="red">[1 Mark]</font>
# Create variables for true positive, true negative, false positive and false negative

#### **8.1.8** Calculate sensitivity, specificity, precision, recall and f1 score of the model <font color="red">[2 Marks]</font>
# Calculate the sensitivity


# Calculate the specificity


# Calculate Precision


# Calculate Recall


# Calculate F1 Score

### **8.2 Make predictions over validation data using random forest model** <font color = red>[10 marks]</font>
#### **8.2.1** Select the important features and make predictions over validation data <font color="red">[2 Marks]</font>
# Select the relevant features for validation data

# Make predictions on the validation data


#### **8.2.2** Check accuracy of random forest model <font color="red">[1 Mark]</font>
# Check accuracy

#### **8.2.3** Create confusion matrix <font color="red">[1 Mark]</font>
# Create the confusion matrix

#### **8.2.4** Create variables for true positive, true negative, false positive and false negative <font color="red">[1 Mark]</font>
# Create variables for true positive, true negative, false positive and false negative

#### **8.2.5** Calculate sensitivity, specificity, precision, recall and F1-score of the model <font color="red">[5 Marks]</font>
# Calculate Sensitivity


# Calculate Specificity


# Calculate Precision


# Calculate Recall


# Calculate F1-score

## **Evaluation and Conclusion**
Write the conclusion.
