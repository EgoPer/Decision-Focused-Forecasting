import torch
from torch import nn
from cvxpylayers.torch import CvxpyLayer
from optimisation_utils import define_parameterised_optimisation_model

class SimpleNN(nn.Module):
    def __init__(self, n_context, hidden_size, n_out, dropout = 0.1):
        super(SimpleNN, self).__init__()
        self.latent_1 = nn.Linear(n_context, hidden_size)
        self.latent_2 = nn.Linear(hidden_size, hidden_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout)
        self.y_predict = nn.Linear(hidden_size, n_out)

    def forward(self, z):
        """
        z - vector of context at time t [batch, variables]
        """
        # hidden layer + nonlinearity
        z = self.latent_1(z)
        z = self.relu(z)
        z = self.dropout(z)

        z = self.latent_2(z)
        z = self.relu(z)
        z = self.dropout(z)

        # predict conditional price vector
        y = self.y_predict(z).unsqueeze(-1)

        return {"theta" : y}

class DFF(nn.Module):
    def __init__(self, hyperparameters):
        super(DFF, self).__init__()

        self.hyperparameters = hyperparameters

        self.prediction_module = SimpleNN(n_context = hyperparameters['n_context'],
                                          hidden_size = hyperparameters['latent_size'],
                                          n_out = hyperparameters['f'])

        problem, decisions, uncertain_parameters = define_parameterised_optimisation_model(hyperparameters)

        opt_layer = CvxpyLayer(problem,
                   parameters = list(uncertain_parameters.values()),
                   variables= list(decisions.values()))

        self.optimisation_module = opt_layer
        self.opt_problem = problem
        self.decisions_dct = decisions
        self.uncertain_parameters_dct = uncertain_parameters


    def forward(self, Z, initial_state):
        """
        Z - matrix of context at times t:t+f [batch, time, variables]
        initial_state - scalar or vector of initial state [batch, variables]
        """
        Theta = []
        decisions_now = {}

        # Initial prediction and decision
        theta_1 = self.prediction_module(Z[:,0,...])['theta']

        Theta.append(theta_1)
        decisions_optimised_1 = self.optimisation_module(*[theta_1,initial_state[:,[0],...]])
        decisions_optimised_1_dct = dict(zip(self.decisions_dct.keys(),decisions_optimised_1))

        for name in self.decisions_dct.keys():
            # Add the first two states and the first set of in and out decisions
            decisions_now[name] = decisions_optimised_1_dct[name][:,:-(self.hyperparameters['f']-1)]

        for t in range(1,(self.hyperparameters['f'])):
            theta_t = self.prediction_module(Z[:,t,...])['theta']
            Theta.append(theta_t)

            previous_state = decisions_now['z_state'][:,t,...] # Get latest state

            decisions_optimised_t = self.optimisation_module(*[theta_t,previous_state])
            decisions_optimised_t_dct = dict(zip(self.decisions_dct.keys(),decisions_optimised_t))

            for name in self.decisions_dct.keys():
                # Add the here and now decisions for the rest of the stages
                decisions_now[name] = torch.concat([decisions_now[name],decisions_optimised_t_dct[name][:,[-self.hyperparameters['f']]]],dim = 1) # Only keep here an now (decision at relative time t=1)

        return decisions_now


class DFL(nn.Module):
    def __init__(self, hyperparameters):
        super(DFL, self).__init__()
        """
        Classic DFL - one prediction, assume there is only one stage
        """

        self.dff = DFF(hyperparameters=hyperparameters)

        self.prediction_module = self.dff.prediction_module

    def forward(self, Z, initial_state):
        """
        Z - matrix of context at times t:t+f [batch, time, variables]
        initial_state - scalar or vector of initial state [batch, variables]
        """
        Theta = []
        decisions_now = {}

        # Initial prediction and decision
        theta_1 = self.dff.prediction_module(Z[:,0,...])['theta']

        Theta.append(theta_1)

        decisions_optimised_1 = self.dff.optimisation_module(*[theta_1,initial_state[:,[0],...]])
        decisions_optimised_1_dct = dict(zip(self.dff.decisions_dct.keys(),decisions_optimised_1))

        for name in self.dff.decisions_dct.keys():
            # Add all decisions
            decisions_now[name] = decisions_optimised_1_dct[name]

        return decisions_now

class two_stage(nn.Module):
    def __init__(self, hyperparameters):
        super(two_stage, self).__init__()

        self.hyperparameters = hyperparameters

        self.prediction_module = SimpleNN(n_context = hyperparameters['n_context'],
                                          hidden_size = hyperparameters['latent_size'],
                                          n_out = hyperparameters['f'])

    def forward(self, Z, initial_state):
        """
        Z - matrix of context at times t:t+f [batch, time, variables]
        initial_state - scalar or vector of initial state [batch, variables]

        returns predictions
        """

        theta = self.prediction_module(Z[:,0,...])

        return theta

