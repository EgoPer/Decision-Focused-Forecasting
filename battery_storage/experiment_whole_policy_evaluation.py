from data_utils import get_train_holdout_test, MultistageOptimisationDataset, MultistageOptimisationDatasetEvaluation
from models import two_stage, DFL, DFF
from evaluation_utils import ( 
predict_then_optimise_follow_policy,
oracle_follow_policy,
predict_then_optimise_follow_policy_singlestage,
)
from optimisation_utils import battery_storage_objective_function
from hyperparameters import init_hyperparams_battery_storage

import torch
import pickle 
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

def experiment_whole_policy_evaluation(setting):
    hyperparameters = init_hyperparams_battery_storage(setting)

    hyperparameters_for_path = {key:value for key, value in hyperparameters.items() if "variable" not in key}
    base_path = os.path.join(".","results","policy",str(hyperparameters_for_path))

    # Get and split data
    data = get_train_holdout_test(hyperparameters = hyperparameters)

    # Input layer size in network
    hyperparameters['n_context'] = int(len(hyperparameters['forwards_variables']) * hyperparameters['f']
                                    + len(hyperparameters['backwards_variables']) * hyperparameters['l']
                                    + len(hyperparameters['static_variables']))

    ## Set models and corresponding optimisers
    # Decision focused forecasting
    dff= DFF(hyperparameters = hyperparameters)
    optimiser = torch.optim.Adam(dff.parameters(), lr = hyperparameters['learning_rate'])

    # Two stage, separate predict and optimise
    just_forecasting = two_stage(hyperparameters = hyperparameters)
    optimiser_just_forecasting = torch.optim.Adam(just_forecasting.prediction_module.parameters(), 
                                                lr = hyperparameters['learning_rate'])
    # DFL but single stage
    dfl = DFL(hyperparameters = hyperparameters)
    optimiser_dfl = torch.optim.Adam(dfl.parameters(),
                                    lr = hyperparameters['learning_rate'])

    # Set torch datasets and dataloaders
    set_types = set([name.split("_")[-1] for name in data])

    torch_datasets = {}
    for name in set_types:
        torch_datasets[name] = MultistageOptimisationDataset(X = data[f'X_{name}'], Y = data[f'Y_{name}'], 
                                                            hyperparameters = hyperparameters)
        
    dataloaders = {}
    for name in set_types:
        shuffle = True if name == "train" else False
        dataloaders[name] = torch.utils.data.DataLoader(dataset = torch_datasets[name],
                                                        batch_size = hyperparameters['batch_size'], 
                                                        shuffle = shuffle)
    # Evaluation dataloader
    dataset_evaluation = MultistageOptimisationDatasetEvaluation(X = data[f'X_test'], Y = data[f'Y_test'], 
                                                            hyperparameters = hyperparameters)
        
        

    dataloader_evaluation = torch.utils.data.DataLoader(dataset = dataset_evaluation,
                                                    batch_size = hyperparameters['batch_size'], 
                                                    shuffle = False)

    # Set useful dictionaries for iteration over different models in training and evaluation
    model_names = ['dff','two_stage', 'dfl']

    # model_names = ['two_stage', 'dfl']


    # Dictionary of models
    models_dct = dict(zip(model_names,[dff,just_forecasting, dfl]))

    model_names = ['two_stage', 'dfl']
    models_dct = dict(zip(model_names,[just_forecasting, dfl]))



    # Dictionary of optimisers
    optimisers_dct = {}
    for name in model_names:
        optimisers_dct[name] = torch.optim.Adam(models_dct[name].parameters(), 
                                                lr = hyperparameters['learning_rate'])
        
    # Result tracking objects {{[]}} - list within dict within dict
    models_losses = {}
    for name in set_types:
        models_losses[name] = {}
        for model_name in model_names:
            models_losses[name][f"{model_name}_multi"] = []
            if model_name != 'dff':
                models_losses[name][f"{model_name}_single"] = []

    # Loss at initialisation
    # Loop across models
    # Oracle case
    decisions_opt, true_parameters_test = oracle_follow_policy(dataloader_evaluation = dataloader_evaluation, hyperparameters = hyperparameters)
    oracle_loss = battery_storage_objective_function(decisions = decisions_opt, 
                                                    uncertain_parameters = true_parameters_test, 
                                                    hyperparameters = hyperparameters)
    oracle_loss = float(oracle_loss.detach().numpy())
    loss_string = f""
    oracle_loss_string = f"_oracle={float(oracle_loss):.3f}"
    loss_string += oracle_loss_string

    # for model_name, model in models_dct.items():

    #     decisions_test, true_parameters_test = predict_then_optimise_follow_policy(dataloader_evaluation = dataloader_evaluation, 
    #                                             prediction_model_trained = model.prediction_module, 
    #                                             hyperparameters = hyperparameters)

    #     test_loss = battery_storage_objective_function(decisions = decisions_test, 
    #                                                 uncertain_parameters = true_parameters_test, 
    #                                                 hyperparameters = hyperparameters)
        
    #     models_losses['test'][f"{model_name}_multi"].append(float(test_loss.detach().numpy()))
    #     loss_string += f"_{model_name}_multi={float(test_loss.detach().numpy()):.3f}"

    #     if model_name != 'dff':
    #         decisions_test, true_parameters_test = predict_then_optimise_follow_policy_singlestage(dataloader_evaluation = dataloader_evaluation, 
    #                                                 prediction_model_trained = model.prediction_module, 
    #                                                 hyperparameters = hyperparameters)
            
    #         test_loss = battery_storage_objective_function(decisions = decisions_test, 
    #                                                 uncertain_parameters = true_parameters_test, 
    #                                                 hyperparameters = hyperparameters)

    #         models_losses['test'][f"{model_name}_single"].append(float(test_loss.detach().numpy()))
    #         loss_string += f"_{model_name}_single={float(test_loss.detach().numpy()):.3f}"

    # models_losses['test']['oracle'] = [oracle_loss]

    # print(loss_string)

    # Training
    # Epoch loop
    for i in (pbar := tqdm(range(hyperparameters['epochs']), desc = f"Epoch=0{loss_string}", leave = False)):
        loss_string = ""
        loss_string += oracle_loss_string
        # Dataloader loop
        for context, true_parameters in tqdm(dataloaders["train"], leave = False):
            # Model loop
            for model_name, model in models_dct.items():
                optimiser = optimisers_dct[model_name]

                optimiser.zero_grad()
                
                if 'two_stage' in model_name:
                    predictions = model(**context)
                    
                    loss = torch.nn.MSELoss()(true_parameters['theta'], predictions['theta'])
                    
                else:
                    decisions_MS = model(**context)

                    loss = battery_storage_objective_function(decisions = decisions_MS,
                                                            uncertain_parameters = true_parameters,
                                                            hyperparameters = dff.hyperparameters)


                loss.backward()

                optimiser.step()
                    
        # Loss on test set
        for model_name, model in models_dct.items():

            decisions_test, true_parameters_test = predict_then_optimise_follow_policy(dataloader_evaluation = dataloader_evaluation, 
                                                    prediction_model_trained = model.prediction_module, 
                                                    hyperparameters = hyperparameters)

            test_loss = battery_storage_objective_function(decisions = decisions_test, 
                                                        uncertain_parameters = true_parameters_test, 
                                                        hyperparameters = hyperparameters)
            
            models_losses['test'][f"{model_name}_multi"].append(float(test_loss.detach().numpy()))
            loss_string += f"_{model_name}_multi={float(test_loss.detach().numpy()):.3f}"


            if model_name != 'dff':
                decisions_test, true_parameters_test = predict_then_optimise_follow_policy_singlestage(dataloader_evaluation = dataloader_evaluation, 
                                                        prediction_model_trained = model.prediction_module, 
                                                        hyperparameters = hyperparameters)
                
                test_loss = battery_storage_objective_function(decisions = decisions_test, 
                                                        uncertain_parameters = true_parameters_test, 
                                                        hyperparameters = hyperparameters)

                models_losses['test'][f"{model_name}_single"].append(float(test_loss.detach().numpy()))
                loss_string += f"_{model_name}_single={float(test_loss.detach().numpy()):.3f}"

        # models_losses['test']['oracle'].append(oracle_loss)

        pbar.set_description(f"Epoch={i+1}{loss_string}")

    ## Recording experiment outcomes
    # Save models
    hyperparameters_for_path = {key:value for key, value in hyperparameters.items() if "variable" not in key}
    base_path = os.path.join(".","results","policy",str(hyperparameters_for_path))

    # For replicating experiments
    if not os.path.exists(base_path): # Make folders if they don't exist already
        os.makedirs(base_path)
            
    directory = os.listdir(base_path)
    n_exp = 1 if not directory else np.max(([1 if not 'exp_' in folder else int(folder.split('_')[-1]) for folder in directory]))
    
    base_path_old = os.path.join(base_path,f'exp_{n_exp}')
    hyperparameters_path = os.path.join(base_path_old, 'hyperparams.pkl')
    if os.path.exists(hyperparameters_path):
        n_exp += 1

    base_path = os.path.join(base_path,f'exp_{n_exp}')
    for model_name, model in models_dct.items():
        model_path = os.path.join(base_path,model_name,"model.pt")
        if not os.path.exists(os.path.dirname(model_path)): # Make folders if they don't exist already
            os.makedirs(os.path.dirname(model_path))
        torch.save(model, model_path)
    # Save full hyperparameter information
    hyperparameters_path = os.path.join(base_path, 'hyperparams.pkl')
    with open(hyperparameters_path, 'wb') as f:
        pickle.dump(hyperparameters, f)
    # Save results 
    df_losses = pd.DataFrame(models_losses['test'])
    results_path = os.path.join(base_path, 'results.csv')
    df_losses.to_csv(results_path, index = False)

