def init_hyperparams_battery_storage(param_set):

    # potential (lambda, epsilon) pairs for experiment
    settings = [[0.1, 0.05], [1, 0.5], [10, 5], [35, 15]]

    hyperparams = {}

    # Battery capacity
    hyperparams['B'] = 1

    # Battery efficiency
    hyperparams['eff'] = 0.9

    # Battery max power in
    hyperparams['in_max'] = 0.5
    
    # Battery max power out
    hyperparams['out_max'] = 0.2
    
    # Forecast horizon
    hyperparams['f'] = 48
    
    # Lookback horizon
    hyperparams['l'] = 48
    
    # Preference for battery staying in middle of range
    hyperparams['lambda'] = settings[param_set][0]

    # Regularize z_in and z_out
    hyperparams['epsilon'] = settings[param_set][1]
    
    # Problem specific variables for prediction
    hyperparams['forwards_variables'] = ['load_forecast','temp_dca','temp_dca_squared','temp_dca_cubed']
    hyperparams['backwards_variables'] = ['da_price','temp_dca','temp_dca_squared']
    hyperparams['static_variables'] = ['weekend', 'holiday', 'cos_doy', 'sin_doy', 'cos_hour', 'sin_hour']
    hyperparams['n_context'] = int(len(hyperparams['forwards_variables']) * hyperparams['f']
                                    + len(hyperparams['backwards_variables']) * hyperparams['l']
                                    + len(hyperparams['static_variables']))

    # Network latent size
    hyperparams['latent_size'] = hyperparams['f'] * 2
    # Network learning rate
    hyperparams['learning_rate'] = 1e-3
    # Training batch size
    hyperparams['batch_size'] = 64
    # Number of training epochs
    hyperparams['epochs'] = 50
    # Proportion of set dedicated to testing
    hyperparams['test_fraction'] = 0.2
    # Proportion of set dedicated to validation
    hyperparams['validation_fraction'] = 0.0

    return hyperparams