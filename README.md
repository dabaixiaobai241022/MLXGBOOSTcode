# README


## Project Overview

This repository contains the code for the study "Targeted metabolomics identified novel metabolites, predominantly phosphatidylcholines and docosahexaenoic acid-containing lipids, predictive of incident chronic kidney disease in middle-to-elderly-aged Chinese adults." The study aims to combine predictive models to identify novel biomarkers for CKD risk prior to disease onset. The original data is currently not shared due to subsequent research needs; therefore, please use the code with your own data.

## Predictor Variables

We measured baseline levels of metabolites, including amino acids, acyl-carnitines, lipids, and other metabolites. Subsequently, we further selected metabolites as candidate predictors using methods such as LASSO.

## Outcome Measures

CKD was defined as having an eGFR < 60 ml/min per 1.73 m², with CKD labeled as 1 and other diagnoses labeled as 0.

## Model Development and Evaluation

The machine learning models currently applied include Random Forest and XGBoost. 

The logistic regression model and Random Forest model were analyzed using default parameters, while XGBoost employed grid search to explore all possible parameter combinations in order to find the optimal parameters.

A total of 1,000 repetitions of 10-fold cross-validation were performed for each of the three models.

## Repository Structure

- `01_model_logistic_code: Code for logistic regression model with 1,000 repetitions of 10-fold cross-validation using default parameters.

- `02_mode_Randomforest_code: Code for random forest model with 1,000 repetitions of 10-fold cross-validation using default parameters.

- `03_model_XGBOOST_code: The code implemented a grid search for the XGBoost model and performed 1,000 repetitions of 10-fold cross-validation.
