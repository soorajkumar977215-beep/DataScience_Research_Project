Impact of Missing Data on Machine Learning Prediction Performance
Overview

This independent research project investigates how different levels of missing data affect the prediction performance of basic machine learning models.

The study compares three machine learning models under different levels of artificially introduced missing data and examines whether simple imputation methods can reduce the impact on model performance.

Research Questions
How does model performance change as the percentage of missing data increases?
Do different machine learning models respond differently to missing data?
How do mean and median imputation compare under different missing-data levels?
Dataset

The study uses the Banknote Authentication dataset from the UCI Machine Learning Repository.

Observations: 1,372
Input features: 4
Prediction task: Binary classification
Features: Variance, skewness, curtosis, and entropy

Dataset source: https://archive.ics.uci.edu/dataset/267/banknote+authentication

Models
Logistic Regression
Decision Tree
Random Forest
Experimental Setup
Missing-data levels: 0%, 5%, 10%, 20%, 30%, and 40%
Training/testing split: 80:20
Five random seeds for repeated missing-data experiments
Missing values introduced into the training data
Test data kept complete
Mean and median imputation compared
Evaluation metrics: Accuracy and F1 score
Main Findings

In this experimental setup, increasing missing data generally reduced model performance. Logistic Regression showed the largest overall decrease in mean accuracy, while Random Forest showed the smallest overall decrease among the three models.

The comparison of mean and median imputation showed that neither method was consistently better at every missing-data level.

Project Structure
01_Research/ — Research questions, hypotheses, literature review, and research notes
02_Data/ — Dataset files
03_Notebooks/ — Jupyter notebooks
04_Experiments/ — Experiment code
05_Results/ — Experimental results
06_Figures/ — Research figures
07_Paper/ — Research manuscript
Limitations

This study uses one dataset, three machine learning models, randomly introduced missing values, and two simple imputation methods. The results should not be generalized to all datasets or real-world missing-data situations.

Status

Independent research project. The manuscript has not been represented as a published or accepted paper.

Reproducibility

Code and experimental instructions will be organized in this repository so that the experiments can be reviewed and reproduced.

Author

Sooraj Kumar
B.S. Computer Science
Shah Abdul Latif University, Khairpur, Pakistan