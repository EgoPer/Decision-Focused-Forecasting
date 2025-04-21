import torch
from torch import nn
from cvxpylayers.torch import CvxpyLayer
from optimisation_utils import define_parameterised_optimisation_model

class MLPBlock(nn.Module):
    def __init__(self, hidden_size, dropout = 0.2):
        super(MLPBlock, self).__init__()

        self.latent_1 = nn.Linear(hidden_size, hidden_size)

        self.latent_2 = nn.Linear(hidden_size, hidden_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout)
        self.norm = nn.LayerNorm([hidden_size])

    def forward(self,z):
        z = self.latent_1(z)
        z = self.relu(z)
        z = self.dropout(z)

        z = self.latent_2(z)
        z = self.norm(z)
        z = self.relu(z)

        return z

class SimpleNN(nn.Module):
    def __init__(self, n_context, hidden_size, n_out, n_channels, n_blocks, dropout = 0.2):
        super(SimpleNN, self).__init__()
        self.input_layer = nn.Linear(n_context, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout)
        self.out_layer = nn.Linear(hidden_size, n_out * n_channels)

        self.blocks = [MLPBlock(hidden_size=hidden_size,dropout=dropout) for _ in range(n_blocks)]
        self.blocks_layer = nn.Sequential(*self.blocks)

    def forward(self, z):
        """
        z - vector of context at time t [batch, time, variables]
        """
        # hidden layer + nonlinearity
        z = self.input_layer(z)
        z = self.relu(z)
        z = self.dropout(z)

        z = self.blocks_layer(z)

        # predict conditional price vector
        y = self.out_layer(z).unsqueeze(-1)
        return {"theta" : y}


class TSkip(nn.Module):
    def __init__(self, n_context, hidden_size, n_out, n_channels, n_blocks, skip_c, dropout = 0.2):
        super(TSkip, self).__init__()
        
        self.input_layer = nn.Linear(n_context, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout)
        self.out_layer = nn.Linear(hidden_size, n_out * n_channels)

        self.blocks = nn.ModuleList([MLPBlock(hidden_size=hidden_size,dropout=dropout) 
                                     for _ in range(n_blocks)])
        
        self.skip_c = skip_c
        self.skip_bias = torch.nn.Parameter(torch.rand([1,1,hidden_size])) # [B,T,L]

    def forward(self, z):
        """
        z - vector of context at time t [batch, time, variables]
        """
        bs = z.shape[0]
        # hidden layer + nonlinearity
        z = self.input_layer(z)
        z = self.relu(z)
        z = self.dropout(z)
        
        for i, block in enumerate(self.blocks):
            z = block(z)
            
            if self.skip_c:
    #             if i + 1 <= len(self.blocks):
                skip = torch.cat([self.skip_bias.expand(bs,-1,-1),z[...,1:,:]], dim = -2)
                z = (z + skip)/2
        

        # predict conditional price vector
        y = self.out_layer(z).unsqueeze(-1)
        return {"theta" : y}

class DFF(nn.Module):
    def __init__(self, hyperparameters, skip_c = True):
        super(DFF, self).__init__()

        self.hyperparameters = hyperparameters

        # self.prediction_module = SimpleNN(n_context = hyperparameters['n_context'],
        #                                   hidden_size = hyperparameters['latent_size'],
        #                                   n_out = hyperparameters['f'],
        #                                   n_channels = hyperparameters['n_channels'],
        #                                   n_blocks = hyperparameters['n_blocks']
        #                                   )
        

        self.prediction_module = TSkip(n_context = hyperparameters['n_context'],
                                          hidden_size = hyperparameters['latent_size'],
                                          n_out = hyperparameters['f'],
                                          n_channels = hyperparameters['n_channels'],
                                          n_blocks = hyperparameters['n_blocks'],
                                          skip_c = skip_c,
                                          )
        
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
        thetas = self.prediction_module(Z)
        theta_1 = thetas['theta'][:,0,...]

        Theta.append(theta_1)
        decisions_optimised_1 = self.optimisation_module(*[theta_1,initial_state])
        decisions_optimised_1_dct = dict(zip(self.decisions_dct.keys(),decisions_optimised_1))

        for name in self.decisions_dct.keys():
            # Add the first two states and the first set of in and out decisions
            decisions_now[name] = decisions_optimised_1_dct[name][:,:-(self.hyperparameters['f']-1)]
            decisions_optimised_1_dct[name][:,:-(self.hyperparameters['f']-1)]
            
        for t in range(1,(self.hyperparameters['f'])):
            theta_t = thetas['theta'][:,t,...]
            Theta.append(theta_t)

            previous_state = decisions_now['z_state'][:,t,...] # Get latest state
            # previous_state = torch.clamp(previous_state, min = 0, max =1) # Avoiding feasibility issues due to approximate outputs

            decisions_optimised_t = self.optimisation_module(*[theta_t,previous_state.unsqueeze(-1)])
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

        self.dff = DFF(hyperparameters=hyperparameters, skip_c= False)

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

        decisions_optimised_1 = self.dff.optimisation_module(*[theta_1,initial_state])
        decisions_optimised_1_dct = dict(zip(self.dff.decisions_dct.keys(),decisions_optimised_1))

        for name in self.dff.decisions_dct.keys():
            # Add all decisions
            decisions_now[name] = decisions_optimised_1_dct[name]

        return decisions_now

class two_stage(nn.Module):
    def __init__(self, hyperparameters):
        super(two_stage, self).__init__()

        self.hyperparameters = hyperparameters

        # self.prediction_module = SimpleNN(n_context = hyperparameters['n_context'],
        #                                   hidden_size = hyperparameters['latent_size'],
        #                                   n_out = hyperparameters['f'],
        #                                   n_channels = hyperparameters['n_channels'],
        #                                   n_blocks = hyperparameters['n_blocks']
        #                                   )
        

        self.prediction_module = TSkip(n_context = hyperparameters['n_context'],
                                          hidden_size = hyperparameters['latent_size'],
                                          n_out = hyperparameters['f'],
                                          n_channels = hyperparameters['n_channels'],
                                          n_blocks = hyperparameters['n_blocks'],
                                          skip_c = False
                                          )

    def forward(self, Z, initial_state):
        """
        Z - matrix of context at times t:t+f [batch, time, variables]
        initial_state - scalar or vector of initial state [batch, variables]

        returns predictions
        """

        theta = self.prediction_module(Z[:,0,...])
        
        return theta

