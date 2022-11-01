params = {
    # Train
    "n_epochs": 200,
    "learning_rate": 1e-4,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-8,
    "clip_grad_value": 5.0,
    "evaluate_span": 50,
    "checkpoint_span": 50,

    # Early-stopping
    "no_improve_epochs": 50,

    # Model
    "model": "model-mod-8",
    "n_rnn_layers": 1,
    "n_rnn_units": 128,
    "sampling_rate": 25.0,
    "input_size": 750,
    "n_classes": 2,
    "l2_weight_decay": 1e-3,

    # Dataset
    "dataset": "resteaze",
    "data_dir": "./prof_data/resteaze/ppg_ledgreen",
    "n_folds": 5,
    "n_subjects": 20,

    # Data Augmentation
    "augment_seq": True,
    "augment_signal_full": True,
    "weighted_cross_ent": False,
}

train = params.copy()
train.update({
    "seq_length": 20,
    "batch_size": 15,
})

predict = params.copy()
predict.update({
    "batch_size": 1,
    "seq_length": 1,
})
