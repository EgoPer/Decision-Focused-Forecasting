import cvxpy as cp
import torch

def objective_function(decisions,uncertain_parameters,hyperparameters):
    """
    decisions - cvxpy variable: the vector or container of decisions
        {z_state: portfolio composition,
        z_delta: trades in each stage,
        }

    uncertain_parameters - cvxpy parameter: matrix of returns [time, asset]

    hyperparameters - dict: dictionary of fixed parameters in the model
    hyperparams = {"tau": 0.01, "gamma": 1/2, "smigma" =  smigma which is the train empirical covariance}
    """



    size = uncertain_parameters['theta'].size()
    r = uncertain_parameters['theta'].reshape((size[0],-1,hyperparameters['n_channels'],1)) # [batch,time,channel,1] vector form 
    d = decisions['z_state'].unsqueeze(-1) # [batch,time + 0 state,channel,1]
    t = decisions['z_delta'].unsqueeze(-1) # [batch, time,channel,1]
    
    returns = (d[:,1:,...] * r).sum(dim = - 2)
    
    # [batch, time, 1, weights]
    variance = (d[:,1:,...].permute([0,1,3,2]) @ hyperparameters['smigma'] @ d[:,1:,...]).squeeze(-1)
    # print(variance)
    # variance = torch.zeros([1])

    transaction_costs = hyperparameters['tau'] * torch.abs(t).sum(dim = -2)
    # transaction_costs = hyperparameters['tau'] * torch.linalg.vector_norm(t,dim=-2)

    objective_value = (
        - returns 
        + hyperparameters['gamma'] * variance 
        + transaction_costs
    )

    # print((objective_value > 0).sum())
    # print(objective_value.min())

    objective_value = objective_value.sum(dim = -2) # sum across time terms in objective

    # print(objective_value)

    return torch.mean(objective_value,dim = 0) # sum across sample

def get_cvxpy_objective(decisions,uncertain_parameters,hyperparameters):
    """
    decisions - cvxpy variable: the vector or container of decisions
        {z_state: portfolio composition,
        z_delta: trades in each stage,
        }

    uncertain_parameters - cvxpy parameter: matrix of returns [time, asset]

    hyperparameters - dict: dictionary of fixed parameters in the model
    hyperparams = {"tau": 0.01, "gamma": 1/2, "smigma" = smigma which is the train empirical covariance}
    """


    r = uncertain_parameters['theta'].reshape((hyperparameters['f'],hyperparameters['n_channels']), order = 'C') # [time,channel] vector form 
    d = decisions['z_state'] # [time + 0 state,channel]
    t = decisions['z_delta'] # [time,channel]
    
    returns = cp.sum(cp.multiply(r,d[1:]))
    
    variance = cp.sum([cp.quad_form(d[i,:],hyperparameters['smigma']) for i in range(1,d.shape[0])])
    # variance = 0

    transaction_costs = hyperparameters['tau'] * cp.norm(t.reshape((-1)), p = 1)
    # transaction_costs = hyperparameters['tau'] * cp.norm(t.reshape((-1)), p = 2)

    objective = (
        - returns + hyperparameters['gamma'] * variance + transaction_costs
    )

    return cp.Minimize(objective)

def get_cvxpy_constraints(decisions,uncertain_parameters,hyperparameters):
    """
    decisions - cvxpy variable: the vector or container of decisions
        {z_state: portfolio composition,
        z_delta: trades in each stage,
        }

    uncertain_parameters - cvxpy parameter: matrix of returns [time * asset]

    hyperparameters - dict: dictionary of fixed parameters in the model
    hyperparams = {"tau": 0.01, "gamma": 1/2, "smigma" = smigma which is the train empirical covariance}
    """
    constraints = []
    d = decisions['z_state'] # [time + 0 state,channel]
    t = decisions['z_delta'] # [time,channel]
    # initial
    constraints += [uncertain_parameters['initial_state'].reshape(-1) == d[0,:]]

    # subsequent
    constraints += [d[1:,:] == d[:-1,:] + t]

    # allocation constraints
    constraints += [cp.sum(d[1:,:],axis =1) <= 1] # no leverage
    # constraints += [cp.sum(d[1:,:],axis =1) == 1] # no leverage, no cash positions

    constraints += [d[1:].reshape((-1)) >= 0] # no shorting

    # constraints += [d[-1,:] == 1/hyperparameters['n_channels']] # same terminal position as starting position
    # constraints += [d[-1,:] == 0] # sell all in the future

    constraints += [cp.norm(t[i,:], p = 2).reshape(-1) <= 0.2 for i in range(hyperparameters['f'])] # at most 0.25 movement in portfolio position each stage

    return constraints

def define_parameterised_optimisation_model(hyperparameters):

    # Variables for decision problem
    portfolio = cp.Variable(((hyperparameters['f'] + 1),hyperparameters['n_channels']),name = "state of portfolio")
    trades = cp.Variable((hyperparameters['f'],hyperparameters['n_channels']),name = "trading decision at the start of every period")

    decisions = {
        "z_state" : portfolio,
        "z_delta" : trades,
    }
    # Parameters for decision problem
    returns = cp.Parameter((hyperparameters['f']*hyperparameters['n_channels'],1),name = "returns in periods 1,...,f")
    init_state = cp.Parameter((hyperparameters['n_channels'],1),name = 'initial generation')
    uncertain_parameters = {
        "theta": returns,
        "initial_state": init_state
    }

    objective = get_cvxpy_objective(decisions = decisions,
                                    uncertain_parameters=uncertain_parameters,
                                    hyperparameters=hyperparameters)

    constraints = get_cvxpy_constraints(decisions = decisions,
                                        uncertain_parameters=uncertain_parameters,
                                        hyperparameters=hyperparameters)
    problem = cp.Problem(objective,constraints)


    return problem, decisions, uncertain_parameters
