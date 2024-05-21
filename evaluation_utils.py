from cvxpylayers.torch import CvxpyLayer
from optimisation_utils import define_parameterised_optimisation_model
import torch
from tqdm import tqdm
    
def predict_then_optimise_follow_policy(dataloader_evaluation, prediction_model_trained, hyperparameters):
    
    decisions_now = {}
    true_parameters = {}
    prediction_model_trained.eval()

    problem, decisions, uncertain_parameters = define_parameterised_optimisation_model(hyperparameters)
    opt_layer = CvxpyLayer(problem, 
               parameters = list(uncertain_parameters.values()), 
               variables= list(decisions.values()))
    
    decisions_now = {}
    
    for context, true_parameters in dataloader_evaluation:
        break
    
    with torch.no_grad():
        
        # Initial prediction and decision
        theta_1 = prediction_model_trained(context['Z'][:,0,...])['theta']

        decisions_optimised_1 = opt_layer(*[theta_1,context['initial_state'][:,[0],...]])
        decisions_optimised_1_dct = dict(zip(decisions.keys(),decisions_optimised_1))

        for name in decisions.keys():
            # Add the first two states and the first set of in and out decisions
            decisions_now[name] = decisions_optimised_1_dct[name][:,:-(hyperparameters['f']-1)]

        for t in tqdm(range(1,context['Z'].shape[1])):
            theta_t = prediction_model_trained(context['Z'][:,t,...])['theta']

            previous_state = decisions_now['z_state'][:,t,...] # Get latest state

            decisions_optimised_t = opt_layer(*[theta_t,previous_state])
            decisions_optimised_t_dct = dict(zip(decisions.keys(),decisions_optimised_t))

            for name in decisions.keys():
                # Add the here and now decisions for the rest of the stages
                decisions_now[name] = torch.concat([decisions_now[name],decisions_optimised_t_dct[name][:,[-hyperparameters['f']]]],dim = 1) # Only keep here and now (decision at relative time t=1)

            
                
    return decisions_now, true_parameters



def predict_then_optimise_follow_policy(dataloader_evaluation, prediction_model_trained, hyperparameters):
    
    decisions_now = {}
    true_parameters = {}
    prediction_model_trained.eval()

    problem, decisions, uncertain_parameters = define_parameterised_optimisation_model(hyperparameters)
    opt_layer = CvxpyLayer(problem, 
               parameters = list(uncertain_parameters.values()), 
               variables= list(decisions.values()))
    
    decisions_now = {}
    
    for context, true_parameters in dataloader_evaluation:
        break
    
    with torch.no_grad():
        
        # Initial prediction and decision
        theta_1 = prediction_model_trained(context['Z'][:,0,...])['theta']

        decisions_optimised_1 = opt_layer(*[theta_1,context['initial_state'][:,[0],...]])
        decisions_optimised_1_dct = dict(zip(decisions.keys(),decisions_optimised_1))

        for name in decisions.keys():
            # Add the first two states and the first set of in and out decisions
            decisions_now[name] = decisions_optimised_1_dct[name][:,:-(hyperparameters['f']-1)]

        for t in tqdm(range(1,context['Z'].shape[1]), leave = False, desc= 'evaluation'):
            theta_t = prediction_model_trained(context['Z'][:,t,...])['theta']

            previous_state = decisions_now['z_state'][:,t,...] # Get latest state

            decisions_optimised_t = opt_layer(*[theta_t,previous_state])
            decisions_optimised_t_dct = dict(zip(decisions.keys(),decisions_optimised_t))

            for name in decisions.keys():
                # Add the here and now decisions for the rest of the stages
                decisions_now[name] = torch.concat([decisions_now[name],decisions_optimised_t_dct[name][:,[-hyperparameters['f']]]],dim = 1) # Only keep here and now (decision at relative time t=1)

            
                
    return decisions_now, true_parameters


def oracle_follow_policy(dataloader_evaluation, hyperparameters):
    
    decisions_now = {}
    true_parameters = {}

    problem, decisions, uncertain_parameters = define_parameterised_optimisation_model(hyperparameters)
    opt_layer = CvxpyLayer(problem, 
               parameters = list(uncertain_parameters.values()), 
               variables= list(decisions.values()))
    
    decisions_now = {}
    
    for context, true_parameters in dataloader_evaluation:
        break
    
    with torch.no_grad():
        
        # Initial prediction and decision
        theta_1 = true_parameters['oracle_thetas'][:,0,...]

        decisions_optimised_1 = opt_layer(*[theta_1,context['initial_state'][:,[0],...]])
        decisions_optimised_1_dct = dict(zip(decisions.keys(),decisions_optimised_1))

        for name in decisions.keys():
            # Add the first two states and the first set of in and out decisions
            decisions_now[name] = decisions_optimised_1_dct[name][:,:-(hyperparameters['f']-1)]

        for t in tqdm(range(1,context['Z'].shape[1]), leave = False, desc= 'evaluation'):
            theta_t = true_parameters['oracle_thetas'][:,t,...]

            previous_state = decisions_now['z_state'][:,t,...] # Get latest state

            decisions_optimised_t = opt_layer(*[theta_t,previous_state])
            decisions_optimised_t_dct = dict(zip(decisions.keys(),decisions_optimised_t))

            for name in decisions.keys():
                # Add the here and now decisions for the rest of the stages
                decisions_now[name] = torch.concat([decisions_now[name],decisions_optimised_t_dct[name][:,[-hyperparameters['f']]]],dim = 1) # Only keep here and now (decision at relative time t=1)

            
                
    return decisions_now, true_parameters


def predict_then_optimise_follow_policy_singlestage(dataloader_evaluation, prediction_model_trained, hyperparameters):
    
    decisions_now = {}
    true_parameters = {}
    prediction_model_trained.eval()

    problem, decisions, uncertain_parameters = define_parameterised_optimisation_model(hyperparameters)
    opt_layer = CvxpyLayer(problem, 
               parameters = list(uncertain_parameters.values()), 
               variables= list(decisions.values()))
    
    decisions_now = {}
    
    for context, true_parameters in dataloader_evaluation:
        break
    
    with torch.no_grad():
        
        # Initial prediction and decision
        theta_1 = prediction_model_trained(context['Z'][:,0,...])['theta']

        decisions_optimised_1 = opt_layer(*[theta_1,context['initial_state'][:,[0],...]])
        decisions_optimised_1_dct = dict(zip(decisions.keys(),decisions_optimised_1))

        for name in decisions.keys():
            # Add the first two states and the first set of in and out decisions
            decisions_now[name] = decisions_optimised_1_dct[name][:,:]
        for t in tqdm(range(1,context['Z'].shape[1]// hyperparameters['f']+1), leave = False, desc= 'evaluation'):
            theta_t = prediction_model_trained(context['Z'][:,int(t*hyperparameters['f']),...])['theta']

            previous_state = decisions_now['z_state'][:,int(t*hyperparameters['f']),...] # Get latest state

            decisions_optimised_t = opt_layer(*[theta_t,previous_state])
            decisions_optimised_t_dct = dict(zip(decisions.keys(),decisions_optimised_t))

            for name in decisions.keys():
                # Add the here and now decisions for the rest of the stages
                decisions_now[name] = torch.concat([decisions_now[name],decisions_optimised_t_dct[name][:,-hyperparameters['f']:]],dim = 1) # Only keep here and now (decision at relative time t=1)

        for name in decisions.keys():
                # cut final few if the forecast horizon does not divide the series - 1
                if 'state' in name:
                    decisions_now[name] = decisions_now[name][:,:context['Z'].shape[1]+1,:]
                else:
                    decisions_now[name] = decisions_now[name][:,:context['Z'].shape[1],:]

                
    return decisions_now, true_parameters