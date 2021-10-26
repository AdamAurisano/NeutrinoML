from train import init
import os, sherpa

def train_sherpa():

  config, trainer, train_loader, valid_loader = init()

  parameters = [sherpa.Continuous('learning_rate', [1e-6, 1e-4]),
                sherpa.Discrete('n_iters', [2,6])]

  alg = sherpa.algorithms.GPyOpt(max_num_trials=50)

  study = sherpa.Study(parameters=parameters,
                       algorithm=alg,
                       lower_is_better=True,
                       dashboard_port=os.environ['SHERPA_PORT'])

  for trial in study:
    config['model']['optimizer_params']['AdamW']['lr'] = trial.parameters['learning_rate']
    config['model']['n_iters'] = trial.parameters['n_iters']
      
    # Build model
    trainer.build_model(**config['model'])

    # Train!
    train_summary = trainer.train(train_loader, config['trainer']['n_epochs'], valid_data_loader=valid_loader)
    print(train_summary)

if __name__ == "__main__":
  train_sherpa()
