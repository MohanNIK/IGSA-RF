# IGSA-RF
IGSA-RF
# IGSA-RF: Optimizing Random Forest with Improved Gravitational Search Algorithm

This repository contains the Python implementation of the **IGSA-RF algorithm** for evaluating the impact of Digital Inclusive Finance on Rural Labor Economics (RLE). It is developed for the research article:

> **Modeling Rural Labor Responses to Digital Finance: A Hybrid IGSA-Random Forest Approach**  
> Lin, Z, *Mathematics*, 2025.

## Overview

This project uses an Improved Gravitational Search Algorithm (IGSA) to optimize key hyperparameters of a Random Forest model. It then applies the optimized model to estimate Gini and OOB-based feature importance values in a rural labor indicator system.

Key features:
- IGSA implementation for optimizing RF parameters: `n_estimators`, `max_features`, `min_samples_split`
- Gini coefficient and permutation-based OOB importance
- Model performance evaluation (R², MSE, MAE, MAPE)

## File Structure

- `igsa_rf_model.py`: Main script for data processing, optimization, training and evaluation
- `训练数据.xlsx`: Input dataset (you must add this file)
- `README.md`: Project description and instructions

## ⚙Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- rfpimp (for permutation importances)

Install dependencies via:

```bash
pip install pandas numpy scikit-learn rfpimp
