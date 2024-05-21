import cvxpy as cp
import torch

def battery_storage_objective_function(decisions,uncertain_parameters,hyperparameters):
    """
    for torch calculations
    """

    objective_value = ((uncertain_parameters['theta'] * (decisions['z_in'] - decisions['z_out']))
                       + hyperparameters['lambda']*(decisions['z_state'][...,1:,:]- 0.5 * hyperparameters['B'])**2
                       + hyperparameters['epsilon']*(decisions['z_in'])**2
                       + hyperparameters['epsilon']*(decisions['z_out'])**2) # By term in objective
    
    objective_value = objective_value.sum(dim = -2) # sum across terms in objective
    terminal_value = decisions['z_state'][...,-1,:] * uncertain_parameters['theta'][...,-1,:] # Add estimate of terminal value

    objective_value -= terminal_value

    return torch.mean(objective_value,dim = 0) # sum across sample

def get_cvxpy_objective_battery_storage(decisions,uncertain_parameters,hyperparameters):
    """
    decisions - cvxpy variable: the vector or container of decisions
        in this case a container of three vectors
        {z_in: power in,
        z_out: power out,
        z_state: auxiliary variable of power state (one longer than the in and out to account for initial state),
        }

    uncertain_parameters - cvxpy parameter: vector of electricity prices
        in this case a vector (\mu_i in above)

    hyperparameters - dict: dictionary of fixed parameters in the model
        in this case
        {B:battery capacity,
        \gamma_{eff}:charging efficiency,
        c_{in}:max hourly charge,
        c_{out}:max hourly discharge,
        \lambda:penalty for not being flexible (close to at half charged),
        \epsilon:penalty for too much activity (damaging to battery health),
        F: forecast horizon,
        }
    """
    objective = (uncertain_parameters['prices'].T @ (decisions['z_in'] - decisions['z_out'])
                 + hyperparameters['lambda']*cp.sum_squares(decisions['z_state']- 0.5 * hyperparameters['B'])
                 + hyperparameters['epsilon']*cp.sum_squares(decisions['z_in'])
                 + hyperparameters['epsilon']*cp.sum_squares(decisions['z_out'])
                 - cp.multiply(uncertain_parameters['prices'][-1],decisions['z_state'][-1])
                 )

    return cp.Minimize(objective)

def get_cvxpy_constraints_battery_storage(decisions,uncertain_parameters,hyperparameters):
    """
    decisions - cvxpy variable: the vector or container of decisions
        in this case a container of three vectors
        {z_in: power in,
        z_out: power out,
        z_state: auxiliary variable of power state (one longer than the in and out to account for initial state),
        }

    uncertain_parameters - cvxpy parameter: vector of electricity prices
        in this case a vector (\mu_i in above) and
        init_state: initial state: B/2 default - functions as an uncertain parameter in case of multistage,

    hyperparameters - dict: dictionary of fixed parameters in the model
        in this case
        {B:battery capacity,
        \gamma_{eff}:charging efficiency,
        c_{in}:max hourly charge,
        c_{out}:max hourly discharge,
        \lambda:penalty for not being flexible (close to at half charged),
        \epsilon:penalty for too much activity (damaging to battery health),
        F: forecast horizon,
        }
    """
    constraints = []
    # Linking constraints for T-1 stages
    constraints += [decisions['z_state'][1:,:] == decisions['z_state'][:-1,:] - decisions['z_out'][:,:] + hyperparameters['eff']*decisions['z_in'][:,:]]


    # Initialisation of state 1
    constraints += [decisions['z_state'][0,:] == uncertain_parameters['init_state']]


    # Feasility constraints for T-1 stages
    constraints += [decisions['z_state'][1:,:]>=0, decisions['z_state'][1:,:]<=hyperparameters['B']]


    # # Final state reverts to B/2? this should be changed in continuous model
    # constraints += [decisions['z_state'][-1,:] == 0.5*hyperparameters['B']]

    # Constraints on outflows and inflows
    constraints += [decisions['z_in']>=0, decisions['z_in']<=hyperparameters['in_max']]
    constraints += [decisions['z_out']>=0, decisions['z_out']<=hyperparameters['out_max']]


    return constraints

def define_parameterised_optimisation_model(hyperparameters):

    # Variables for decision problem
    z_in = cp.Variable((hyperparameters['f'],1),name = "power in")
    z_out = cp.Variable((hyperparameters['f'],1),name = "power out")
    z_state = cp.Variable((hyperparameters['f'] + 1,1),name = "state of battery")

    decisions = {
        "z_in" : z_in,
        "z_out" : z_out,
        "z_state" : z_state,
    }
    # Parameters for decision problem
    prices = cp.Parameter((hyperparameters['f'],1),name = "prices predicted at current time")
    init_state = cp.Parameter((1),name = 'current battery power')
    uncertain_parameters = {
        "prices": prices,
        "init_state": init_state
    }

    objective = get_cvxpy_objective_battery_storage(decisions = decisions,
                                            uncertain_parameters=uncertain_parameters,
                                            hyperparameters=hyperparameters)

    constraints = get_cvxpy_constraints_battery_storage(decisions = decisions,
                                                    uncertain_parameters=uncertain_parameters,
                                                    hyperparameters=hyperparameters)
    problem = cp.Problem(objective,constraints)


    return problem, decisions, uncertain_parameters
