import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter
import matplotlib.pyplot as plt

# Load data from SQLite database
dataset = "dataset_2012-24_new"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort data by Date so the most recent games are at the end
data = data.sort_values(by='Date').reset_index(drop=True)

# Extract target variable
margin = data['Home-Team-Win']

# Add winning percentage for home and away teams
data['Winning-Percentage-Home'] = data['W'] / (data['W'] + data['L'])
data['Winning-Percentage-Away'] = data['W.1'] / (data['W.1'] + data['L.1'])

# Replace NaN values with 0 for teams with no games played yet (e.g., early in the season)
data.fillna(0, inplace=True)

# Drop unnecessary columns before training
data.drop(
    ['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'],
    axis=1, inplace=True)

data = data.values.astype(float)

tscv = TimeSeriesSplit(n_splits=8)


def objective(config):
    cv_scores = []

    for train_index, val_index in tscv.split(data):
        x_train, x_val = data[train_index], data[val_index]
        y_train, y_val = margin.iloc[train_index], margin.iloc[val_index]

        # Add sample weights to focus on recent games
        sample_weights = np.linspace(0.5, 1, len(x_train))

        train = xgb.DMatrix(x_train, label=y_train, weight=sample_weights)
        val = xgb.DMatrix(x_val, label=y_val)

        param = {
            'max_depth': int(config['max_depth']),
            'eta': config['eta'],
            'subsample': config['subsample'],
            'colsample_bytree': config['colsample_bytree'],
            'gamma': config['gamma'],
            'min_child_weight': int(config['min_child_weight']),
            'reg_alpha': config['reg_alpha'],
            'reg_lambda': config['reg_lambda'],
            'objective': 'binary:logistic',
            'tree_method': 'hist'
        }

        model = xgb.train(
            param,
            train,
            num_boost_round=1000,
            evals=[(val, "eval")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        predictions = model.predict(val)
        y_pred = [1 if p > 0.5 else 0 for p in predictions]

        acc = accuracy_score(y_val, y_pred)
        cv_scores.append(acc)

    mean_acc = np.mean(cv_scores)
    print(f"Mean CV Accuracy: {mean_acc:.4f} | Params: {config}")
    tune.report({"accuracy": mean_acc})


# Define search space for hyperparameter tuning
def create_search_space():
    config_space = ConfigurationSpace()
    config_space.add_hyperparameters([
        UniformIntegerHyperparameter('max_depth', lower=3, upper=7),
        UniformFloatHyperparameter('eta', lower=0.0001, upper=0.1, log=True),
        UniformFloatHyperparameter('subsample', lower=0.6, upper=0.9),
        UniformFloatHyperparameter('colsample_bytree', lower=0.7, upper=1.0),
        UniformFloatHyperparameter('gamma', lower=0.01, upper=0.9),
        UniformIntegerHyperparameter('min_child_weight', lower=3, upper=10),
        UniformFloatHyperparameter('reg_alpha', lower=1, upper=4),
        UniformFloatHyperparameter('reg_lambda', lower=100, upper=150)
    ])
    return config_space


# Run Bayesian optimization
bohb_hyperband = HyperBandForBOHB(
    time_attr="training_iteration",
    max_t=100,
    reduction_factor=3,
    stop_last_trials=False
)

bohb_search = TuneBOHB(
    space=create_search_space(),
    metric="accuracy",
    mode="max"
)

tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="accuracy",
        mode="max",
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        num_samples=15,
    )
)

results = tuner.fit()

# Get the best tuning results
best_result = results.get_best_result("accuracy", "max")
print("Best hyperparameters found were:", best_result.config)
print("Best mean CV accuracy achieved:", best_result.metrics["accuracy"])

# Train final model using the best hyperparameters
best_params = best_result.config
best_params['objective'] = 'binary:logistic'
best_params['tree_method'] = 'hist'

# Hold out the most recent season for testing
test_size = len(data) // 8  # Assuming 12 seasons of data
x_train, x_test = data[:-test_size], data[-test_size:]
y_train, y_test = margin.iloc[:-test_size], margin.iloc[-test_size:]

# Train final model on all but the most recent season
train = xgb.DMatrix(x_train, label=y_train)
final_model = xgb.train(best_params, train, num_boost_round=1000)

# Evaluate on the held-out test set
test = xgb.DMatrix(x_test, label=y_test)
predictions = final_model.predict(test)
y_pred = [1 if p > 0.5 else 0 for p in predictions]
test_acc = accuracy_score(y_test, y_pred) * 100
print(f"Test set accuracy: {test_acc:.2f}%")

final_model.save_model(f'../../Models/XGBoost_Models/XGBoost_{test_acc:.2f}%_ML-4.json')
