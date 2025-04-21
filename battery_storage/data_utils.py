import pandas as pd
from datetime import datetime as dt
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def get_features_labels(path = 'Data/battery_storage/storage_data.csv'):
    tz = pytz.timezone('America/New_York')
    df = pd.read_csv(path, parse_dates=[0])
    df['date'] = df['datetime'].apply(lambda x: x.date())
    df['hour'] = df['datetime'].apply(lambda x: x.hour)
    # Prices
    df_prices = df[['da_price']].apply(lambda x: pd.to_numeric(x), axis=1)
    df_prices = df_prices.fillna(method='backfill').transpose()
    df_prices = df_prices.fillna(method='ffill').transpose()
    df_prices_log = np.log(df_prices)
    # Load forecasts
    df_load = df[['load_forecast']].apply(lambda x: pd.to_numeric(x), axis=1)
    df_load = df_load.fillna(method='backfill').transpose()
    df_load = df_load.fillna(method='ffill').transpose()

    # Temperatures
    df_temp = df[['temp_dca']].apply(lambda x: pd.to_numeric(x), axis=1)
    df_temp = df_temp.fillna(method='backfill').transpose()
    df_temp = df_temp.fillna(method='ffill').transpose()

    holidays = USFederalHolidayCalendar().holidays(
        start='2011-01-01', end='2017-01-01').to_pydatetime()
    holiday_dates = set([h.date() for h in holidays])

    s = df['datetime']

    data={"weekend": s.apply(lambda x: x.isoweekday() >= 6).values,
          "holiday": s.apply(lambda x: x.date() in holiday_dates).values,
          "cos_doy": s.apply(lambda x: np.cos(
            float(x.timetuple().tm_yday)/365*2*np.pi)).values,
          "sin_doy": s.apply(lambda x: np.sin(
            float(x.timetuple().tm_yday)/365*2*np.pi)).values,
          "cos_hour": s.apply(lambda x: np.cos(
            float(x.hour)/365*2*np.pi)).values,
          "sin_hour": s.apply(lambda x: np.sin(
            float(x.hour)/365*2*np.pi)).values,
            }

    df_feat = pd.DataFrame(data=data, index=df_prices_log.index)

    time_series = pd.concat([df_prices_log,
                             df_load,
                             df_temp,
                             df_feat]
                            , axis = 1)

    time_series.index = df['datetime']
    # time_series = time_series[:500] # Delete in prod

    X = time_series.copy().astype('float')
    # Add features
    for col in ['temp_dca']:
        X[f'{col}_squared'] = X[col]**2
        X[f'{col}_cubed'] = X[col]**3

    X = (X - X.mean(axis=0))/ X.std(axis = 0) # Normalise

    Y = time_series['da_price'].copy()

    return X,Y

def get_train_holdout_test(hyperparameters,
                           path = 'Data/battery_storage/storage_data.csv'):

    test_fraction = hyperparameters['test_fraction']
    validation_fraction = hyperparameters['validation_fraction']

    X, Y = get_features_labels(path = path)

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

        initial_state = torch.tensor([self.hyperparameters['B']/2]).float()

        context = {
            'Z' : Z,
            'initial_state' : initial_state,
        }
        theta = self.Y.iloc[self.hyperparameters['l']+idx:self.hyperparameters['l'] + idx + self.hyperparameters['f']]
        theta = theta.to_numpy()
        theta = torch.tensor(theta).float().unsqueeze(-1) # Turn into a vector with unsqueeze
    
        oracle_thetas = []
        for t in range(self.hyperparameters['f']):
                o_theta = self.Y.iloc[self.hyperparameters['l'] + idx + t: self.hyperparameters['l'] + idx + self.hyperparameters['f'] + t]
                o_theta = o_theta.to_numpy()
                oracle_thetas.append(torch.tensor(o_theta).float().unsqueeze(-1))
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

        initial_state = torch.tensor([self.hyperparameters['B']/2]).float()

        context = {
            'Z' : Z,
            'initial_state' : initial_state,
        }

        theta = self.Y.iloc[self.hyperparameters['l']+idx:self.hyperparameters['l'] + idx + self.length_full]
        theta = theta.to_numpy()
        theta = torch.tensor(theta).float().unsqueeze(-1) # Turn into a vector with unsqueeze
    
        oracle_thetas = []
        for t in range(self.length_full):
                o_theta = self.Y.iloc[self.hyperparameters['l'] + idx + t: self.hyperparameters['l'] + idx + self.hyperparameters['f'] + t]
                o_theta = o_theta.to_numpy()
                oracle_thetas.append(torch.tensor(o_theta).float().unsqueeze(-1))
        oracle_thetas = torch.stack(oracle_thetas, dim = 0)

        true_parameters = {
            'theta' : theta,
            'oracle_thetas' : oracle_thetas
        }

        return context, true_parameters
