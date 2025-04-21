import pandas as pd
from datetime import datetime as dt
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np
import glob

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# def load_data_with_features(filename):
#     df = pd.read_csv(filename)

#     columns = df.columns

#     X = df[columns[1:]]
#     Y = df[columns[1:]]

#     return X, Y

def load_data_with_features(filename):

    # Procedure to process dataset into a useful form (multi-output prediction)
    csvs = []
    for file in glob.glob(filename):
        csvs.append(file)

    specific_tags = ["mom","ROC","EMA"]

    dfX = pd.DataFrame()
    dfy = []
    dfy_cols = []

    #Has to mirror the cleaning below
    common_cols = set.intersection(*[set(pd.read_csv(csv).set_index("Date").iloc[200:].dropna(axis=1).drop(columns=["Close","Name"]).columns)
                                            for csv in csvs])

    for csv in csvs:

        ticker = csv.split("_")[-1].split(".")[0]
        data = pd.read_csv(csv)
        data = data.set_index("Date")
        data = data.iloc[200:]
        data = data.dropna(axis=1)
        data = data.drop(columns=["Close","Name"])

        if dfX.empty:


            stock_specific_cols = set([col for col in common_cols if any([i in col for i in specific_tags])])
            shared_cols = list(common_cols - stock_specific_cols)

            dfX = data.loc[:,shared_cols]


        for col in stock_specific_cols:
            if col != "mom":
                dfX.loc[:,col+f"_{ticker}"] = data.loc[:,col]
            else:
                dfy.append(data.loc[:,col])
                dfy_cols.append(f"ret_{ticker}")

    dfy = pd.DataFrame(dfy,index=dfy_cols).T
    X = dfX
    Y = dfy

    cb = int(0.1 * X.shape[0])
    X, Y = X.iloc[:-cb, :], Y.iloc[:-cb]

    return X, Y

def get_train_holdout_test(hyperparameters):


    test_fraction = hyperparameters['test_fraction']
    validation_fraction = hyperparameters['validation_fraction']

    X, Y = load_data_with_features("./data/portfolio_optimisation/CNNpred/Processed*.csv")

    hyperparameters['forwards_variables'] = []
    hyperparameters['backwards_variables'] = list(X.columns)
    hyperparameters['static_variables'] = []
    hyperparameters['n_context'] = int(len(hyperparameters['forwards_variables']) * hyperparameters['f']
                                    + len(hyperparameters['backwards_variables']) * hyperparameters['l']
                                    + len(hyperparameters['static_variables']))

    hyperparameters['n_channels'] = Y.shape[-1]



    test_boundary = int(X.shape[0] * test_fraction)

    X_test, Y_test = X.iloc[-test_boundary:, :], Y.iloc[-test_boundary:]

    arrays = {'X_test': X_test, 'Y_test': Y_test}

    if validation_fraction:

        validation_boundary = int(X.shape[0] * (test_fraction + validation_fraction))

        X_validation, Y_validation = X.iloc[-validation_boundary:-test_boundary, :], Y.iloc[-validation_boundary:-test_boundary]
        arrays['X_validation'] = X_validation
        arrays['Y_validation'] = Y_validation

        X_train, Y_train = X.iloc[:-validation_boundary, :], Y.iloc[:-validation_boundary]

    else:
        X_train, Y_train = X.iloc[:-test_boundary, :], Y.iloc[:-test_boundary]

    arrays['X_train'] = X_train
    arrays['Y_train'] = Y_train

    # Scale X variables:
    scaler = StandardScaler()

    scaler.fit(arrays['X_train'])

    arrays['X_train'].loc[:,arrays['X_train'].columns] = scaler.transform(arrays['X_train'][arrays['X_train'].columns])
    if validation_fraction:
        arrays['X_validation'].loc[:,arrays['X_validation'].columns] = scaler.transform(arrays['X_validation'][arrays['X_validation'].columns])
    arrays['X_test'].loc[:,arrays['X_test'].columns] = scaler.transform(arrays['X_test'][arrays['X_test'].columns])


    hyperparameters['smigma'] = torch.tensor(Y_train.cov().to_numpy()).float()

    return arrays


class MultistageOptimisationDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, hyperparameters):
        self.X = X
        self.Y = Y
        self.hyperparameters = hyperparameters

    def __len__(self):
        return self.Y.shape[0] - (self.hyperparameters['f']*2 - 1) - self.hyperparameters['l']

    def __getitem__(self, idx):

        Z = []
        for t in range(self.hyperparameters['f']):

            context_t = []

            context_t.append(self.X[self.hyperparameters['backwards_variables']].iloc[
                idx + t: self.hyperparameters['l'] + idx + t].stack())

            context_t.append(self.X[self.hyperparameters['forwards_variables']].iloc[
                self.hyperparameters['l'] + idx + t: self.hyperparameters['l'] + idx + self.hyperparameters['f'] + t].stack())

            context_t.append(self.X[self.hyperparameters['static_variables']].iloc[self.hyperparameters['l'] + idx + t])

            context_t = pd.concat(context_t)
            Z.append(torch.tensor(context_t.values).float())

        Z = torch.stack(Z, dim = 0)

        initial_state = torch.ones((self.hyperparameters['n_channels'],1)) / self.hyperparameters['n_channels']

        context = {
            'Z' : Z,
            'initial_state' : initial_state,
        }

        theta = self.Y.iloc[self.hyperparameters['l']+idx:self.hyperparameters['l'] + idx + self.hyperparameters['f']].to_numpy().reshape([self.hyperparameters['f']*self.hyperparameters['n_channels'],1])
        theta = torch.tensor(theta).float() 
    
        oracle_thetas = []
        for t in range(self.hyperparameters['f']):
                o_theta = self.Y.iloc[self.hyperparameters['l'] + idx + t: self.hyperparameters['l'] + idx + self.hyperparameters['f'] + t].to_numpy().reshape([self.hyperparameters['f']*self.hyperparameters['n_channels'],1])
                oracle_thetas.append(torch.tensor(o_theta).float())
        oracle_thetas = torch.stack(oracle_thetas, dim = 0)

        true_parameters = {
            'theta' : theta,
            'oracle_thetas' : oracle_thetas, # [Batch, Time at which forecast is performed, Stage for which the forecast is made, 1 - to make it a vector]
        }

        return context, true_parameters
    

class MultistageOptimisationDatasetEvaluation(torch.utils.data.Dataset):
    def __init__(self, X, Y, hyperparameters):
        self.X = X
        self.Y = Y
        self.hyperparameters = hyperparameters
        self.length_full = self.Y.shape[0] - (self.hyperparameters['f']*2 - 1) - self.hyperparameters['l']
    def __len__(self):
        return 1

    def __getitem__(self, idx):

        Z = []
        for t in range(self.length_full):

            context_t = []

            context_t.append(self.X[self.hyperparameters['backwards_variables']].iloc[
                idx + t: self.hyperparameters['l'] + idx + t].stack())

            context_t.append(self.X[self.hyperparameters['forwards_variables']].iloc[
                self.hyperparameters['l'] + idx + t: self.hyperparameters['l'] + idx + self.hyperparameters['f'] + t].stack())

            context_t.append(self.X[self.hyperparameters['static_variables']].iloc[self.hyperparameters['l'] + idx + t])

            context_t = pd.concat(context_t)
            Z.append(torch.tensor(context_t.values).float())
        Z = torch.stack(Z, dim = 0)

        initial_state = torch.ones((self.hyperparameters['n_channels'],1)) / self.hyperparameters['n_channels']

        context = {
            'Z' : Z,
            'initial_state' : initial_state,
        }

        theta = self.Y.iloc[self.hyperparameters['l']+idx:self.hyperparameters['l'] + idx + self.length_full].to_numpy().reshape([-1,1])
        theta = torch.tensor(theta).float() 
    
        oracle_thetas = []
        for t in range(self.length_full):
                o_theta = self.Y.iloc[self.hyperparameters['l'] + idx + t: self.hyperparameters['l'] + idx + self.hyperparameters['f'] + t].to_numpy().reshape([self.hyperparameters['f']*self.hyperparameters['n_channels'],1])
                oracle_thetas.append(torch.tensor(o_theta).float())
        oracle_thetas = torch.stack(oracle_thetas, dim = 0)

        true_parameters = {
            'theta' : theta,
            'oracle_thetas' : oracle_thetas
        }

        return context, true_parameters
