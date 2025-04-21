def init_hyperparams(param_set):


    hyperparams = {}

    # Decision problem specific hyperparameters

    hyperparams = {"tau": 0.0, "gamma": 0.5}

    # Forecast horizon
    hyperparams['f'] = 7
    # Lookback horizon
    hyperparams['l'] = 21
    
    # Problem specific variables for prediction
    # defined in data utils here 

    # hyperparams['n_channels'] = 5

    # Network latent size
    hyperparams['latent_size'] = 1000 #(hyperparams['f'] + hyperparams['l']) * 10
    # Number of mlp blocks in model
    hyperparams['n_blocks'] = 4

    # Network learning rate
    hyperparams['learning_rate'] = 1e-4
    # Training batch size
    hyperparams['batch_size'] = 64
    # Number of training epochs
    hyperparams['epochs'] = 25
    # Training batch size
    # hyperparams['es'] = 10

    # hyperparams['early_stopping'] = hyperparams['es']
    # Proportion of set dedicated to testing
    hyperparams['tf'] = 0.1
    hyperparams['test_fraction'] = hyperparams['tf']
    # Proportion of set dedicated to validation
    hyperparams['vf'] = 0
    hyperparams['validation_fraction'] = hyperparams['vf']

    return hyperparams