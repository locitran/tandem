import copy
import ml_collections
import logging 

def model_config() -> ml_collections.ConfigDict:
    cfg = copy.deepcopy(CONFIG_IMPROVE)
    return cfg

# Try the architecture of hinden layers can be 
# 33-node by 33-node by 33-node by 33-node by 10 node and by 2-node of DNN first
CONFIG_IMPROVE = ml_collections.ConfigDict({
    'model': {
        # Input layer
        'input': {
            'n_neurons': 33,
            'dropout_rate': 0.0,
        },

        # Hidden layers
        'hidden': {
            'hidden_0': {
                'n_neurons': 33,
                'activation': 'gelu',
                'initializer': 'glorot_uniform',
                'batch_norm': False,
                'dropout_rate': 0,
                'l1': 0,
                'l2': 1e-4,
            },
            'hidden_1': {
                'n_neurons': 33,
                'activation': 'gelu',
                'initializer': 'glorot_uniform',
                'batch_norm': False,
                'dropout_rate': 0,
                'l1': 0,
                'l2': 1e-4,
            },
            'hidden_2': {
                'n_neurons': 33,
                'activation': 'gelu',
                'initializer': 'glorot_uniform',
                'batch_norm': False,
                'dropout_rate': 0,
                'l1': 0,
                'l2': 1e-4,
            },
            'hidden_3': {
                'n_neurons': 33,
                'activation': 'gelu',
                'initializer': 'glorot_uniform',
                'batch_norm': False,
                'dropout_rate': 0,
                'l1': 0,
                'l2': 1e-4,
            },
            'hidden_4': {
                'n_neurons': 10,
                'activation': 'gelu',
                'initializer': 'glorot_uniform',
                'batch_norm': False,
                'dropout_rate': 0,
                'l1': 0,
                'l2': 1e-4,
            },
        },

        # Output layer
        'output': {
            'n_neurons': 2,
            'activation': 'softmax',
        },
    },
    'optimizer': {
        'name': 'Nadam',
        'learning_rate': 5e-5,
        # Drop to 1e-5 after 300 epochs
        # 'schedule': {
        #     'start_epoch': 50,
        #     'learning_rate': 1e-5,
        # },

    },
    'training': {
        'batch_size': 300,
        'n_epochs': 300,
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy', 
                    'AUC', 
                    # 'AUCPR',
                    # 'F1Score',
                    # 'Precision', 
                    # 'Recall', 
                    # 'TruePositives', 
                    # 'TrueNegatives', 
                    # 'FalsePositives', 
                    # 'FalseNegatives'
                    ],
        'callbacks': {
            'EarlyStopping': {
                'monitor': 'val_loss',
                'patience': 30,
                'mode': 'min',
                'restore_best_weights': True,
                'start_from_epoch': 50,
                'verbose': 1,
            },
            'ModelCheckpoint': {
                'monitor': 'val_loss',
                'save_best_only': True,
                'mode': 'min',
            },
            # 'ReduceLROnPlateau': {
            #     'monitor': 'val_loss',
            #     'factor': 0.1,
            #     'patience': 10,
            #     'mode': 'min',
            # },
        },
        'class_weight': None, # 'auto' or None; 'auto': adjust weights inversely proportional to class frequencies
    },
})


def get_config(input_shape=33, n_hidden=5, patience=50, dropout_rate=0., 
               n_neuron_per_hidden=None, n_neuron_last_hidden=None, verbose=True):
    cfg = model_config()
    cfg.model.input.n_neurons = input_shape
    logging.error("Input Layer: %d", cfg.model.input.n_neurons) # Write to log
    cfg['model']['input']['dropout_rate'] = dropout_rate

    # No. of neurons in the output layer
    n_neuron_per_hidden = input_shape if n_neuron_per_hidden is None else n_neuron_per_hidden
    n_neuron_last_hidden = 10 if n_neuron_last_hidden is None else n_neuron_last_hidden
    
    weight_initialization = ['glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform']
    initializer = 'glorot_uniform'
    
    for item in cfg.model.hidden:
        del cfg['model']['hidden'][item]
    for i in range(n_hidden):
        n_neuron = n_neuron_last_hidden if i == n_hidden - 1 else n_neuron_per_hidden
        hidden_name = f'hidden_{i:02d}'
        cfg['model']['hidden'][hidden_name] = {
            'activation': 'gelu',
            'batch_norm': False,
            'dropout_rate': dropout_rate,
            'initializer': initializer,
            'l1': 0,
            'l2': 0.0001,
            'n_neurons': n_neuron
        }
    cfg['training']['callbacks']['EarlyStopping']['patience'] = patience

    if verbose:
        from prettytable import PrettyTable
        # print cfg as table
        tb = PrettyTable()
        tb.field_names = ["Layer", "Activation", "Batch Norm", "Dropout Rate", "Initializer", "L1", "L2", "N Neurons"]
        tb.add_row(["Input", "-", "-", cfg.model.input.dropout_rate, "-", "-", "-", cfg.model.input.n_neurons])
        for layer in cfg.model.hidden:
            tb.add_row([layer, cfg.model.hidden[layer].activation, cfg.model.hidden[layer].batch_norm, cfg.model.hidden[layer].dropout_rate, cfg.model.hidden[layer].initializer, cfg.model.hidden[layer].l1, cfg.model.hidden[layer].l2, cfg.model.hidden[layer].n_neurons])
        tb.add_row(["Output", cfg.model.output.activation, "-", "-", "-", "-", "-", cfg.model.output.n_neurons])
        logging.error("Model Configuration: \n%s", tb) # Write to log

        # print training and optimizer
        tb = PrettyTable()
        tb.field_names = ["Training", "Batch Size", "N Epochs", "Loss", "Metrics"]
        tb.add_row(["Training", cfg.training.batch_size, cfg.training.n_epochs, cfg.training.loss, cfg.training.metrics])
        tb.add_row(["Optimizer", cfg.optimizer.learning_rate, cfg.optimizer.name, "-", "-"])
        logging.error("Training Configuration: \n%s", tb) # Write to log
    return cfg