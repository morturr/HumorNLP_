import os
import wandb
import logging
from Data.DataPreprocessing import DataPreprocessing
from Model.HumorPredictor import HumorPredictor
from Model.HumorTrainer import HumorTrainer
from Utils.utils import print_str, my_parse_args
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# wandb.login(key='94ee7285d2d25226f2c969e28645475f9adffbce')
logger = logging.getLogger(__name__)


def choose_hyperparam_train():
    # train_ons = ['amazon', 'headlines', 'igg', 'twss']
    train_ons = ['headlines']
    BATCH_SIZES = [4, 8]
    LR_S = [3e-6, 1e-5]
    EPOCHS = [3, 4]
    seeds = [3]

    for train_on in train_ons:
        for seed in seeds:
            for epoch in EPOCHS:
                for lr in LR_S:
                    for bs in BATCH_SIZES:
                        print_str(f'STARTED RUN ON SEED {seed} TRAINED ON {train_on}')
                        # train model
                        model_name = 'bert'
                        model_params = {
                            'model': model_name,
                            'model_dir': models[model_name],
                            'train_on_dataset': train_on,
                            'seed': seed,
                            'epoch': epoch,
                            'learning_rate': lr,
                            'batch_size': bs
                        }

                        time = datetime.now()
                        run_name = '{0}_on_{4}_seed={5}_{1}_{2}:{3}'.format(model_params['model'],
                                                                            time.date(), time.hour, time.minute,
                                                                            model_params['train_on_dataset'],
                                                                            model_params['seed'])

                        output_path = 'Model/SavedModels/{0}'.format(run_name)
                        model_path = output_path + '/model'
                        predictions_path = output_path + '/predictions'

                        # train model
                        wandb.init(project='HumorNLP', name=run_name)
                        h_trainer = HumorTrainer(model_params, args, datasets, run_name)
                        trained_model = h_trainer.train()
                        wandb.finish()

                        # save model
                        h_trainer.save_model(model_path)

                        # predict labels
                        h_predictor = HumorPredictor(trained_model, h_trainer.get_test_datasets(),
                                                     h_trainer.get_tokenizer())

                        # predict only on the dataset that used for train
                        h_predictor.predict([train_on])
                        h_predictor.write_dataset_predictions(predictions_path, [train_on])

                        print_str(f'FINISHED RUN ON SEED {seed} TRAINED ON {train_on}')


def train_on_params(train_params):
    for curr_train_params in train_params:
        train_on = curr_train_params['dataset']
        seeds = curr_train_params['seeds']

        for seed in seeds:
            print_str(f'STARTED RUN ON SEED {seed} TRAINED ON {train_on}')
            # train model
            model_name = 'bert'
            model_params = {
                'model': model_name,
                'model_dir': models[model_name],
                'train_on_dataset': train_on,
                'seed': seed,
                'epoch': curr_train_params['epoch'],
                'learning_rate': curr_train_params['lr'],
                'batch_size': curr_train_params['batch_size']
            }

            time = datetime.now()
            run_name = '{0}_on_{4}_seed={5}_{1}_{2}:{3}'.format(model_params['model'],
                                                                time.date(), time.hour, time.minute,
                                                                model_params['train_on_dataset'], model_params['seed'])

            output_path = 'Model/SavedModels/{0}'.format(run_name)
            model_path = output_path + '/model'
            predictions_path = output_path + '/predictions'

            # train model
            wandb.init(project='HumorNLP', name=run_name)
            h_trainer = HumorTrainer(model_params, args, datasets, run_name)
            trained_model = h_trainer.train()
            wandb.finish()

            # save model
            h_trainer.save_model(model_path)

            # predict labels
            h_predictor = HumorPredictor(trained_model, h_trainer.get_test_datasets(), h_trainer.get_tokenizer())

            # predict on all datasets
            h_predictor.predict(dataset_names)
            h_predictor.write_dataset_predictions(predictions_path, dataset_names)

            print_str(f'FINISHED RUN ON SEED {seed} TRAINED ON {train_on}')


if __name__ == '__main__':
    # load datasets
    print_str('STARTED RUN')

    args = my_parse_args()

    models = {'bert': 'bert-base-uncased',
              'gpt2': 'gpt2'}

    dpp = DataPreprocessing()
    dataset_names = ['amazon', 'headlines', 'twss', 'igg']
    data_path = 'Data/humor_datasets/'
    for name in dataset_names:
        dpp.load_data(data_path + name, name)
    datasets = dpp.get_datasets()

    if args.task == 'hyperparams':
        choose_hyperparam_train()

    elif args.task == 'train':
        DEFAULT_LR = 5e-5
        DEFAULT_EPOCH = 3
        _train_params = [{'dataset': 'amazon', 'seeds': [0], 'epoch': DEFAULT_EPOCH, 'lr': DEFAULT_LR, 'batch_size': 8},
                         {'dataset': 'headlines', 'seeds': [3], 'epoch': DEFAULT_EPOCH, 'lr': 1e-5, 'batch_size': 8},
                         {'dataset': 'igg', 'seeds': [28], 'epoch': DEFAULT_EPOCH, 'lr': DEFAULT_LR, 'batch_size': 4},
                         {'dataset': 'twss', 'seeds': [42], 'epoch': DEFAULT_EPOCH, 'lr': DEFAULT_LR, 'batch_size': 8}]

        train_on_params(_train_params)

    print_str('FINISHED RUN')
