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

if __name__ == '__main__':
    # load datasets
    print_str('STARTED RUN')
    dpp = DataPreprocessing()
    dataset_names = ['amazon', 'headlines', 'twss', 'igg']
    data_path = 'Data/humor_datasets/'
    for name in dataset_names:
        dpp.load_data(data_path + name, name)

    datasets = dpp.get_datasets()

    seeds = list(range(4))
    # seeds = [0, 1, 3, 42, 27]
    # seeds = [3, 4]
    train_ons = ['headlines', 'igg', 'twss']
    # train_ons = ['amazon']
    models = {'bert': 'bert-base-uncased',
              'gpt2': 'gpt2'}
    for train_on in train_ons:
        for seed in seeds:
            print_str(f'STARTED RUN ON SEED {seed} TRAINED ON {train_on}')
            # train model
            model_name = 'bert'
            model_params = {
                'model': model_name,
                'model_dir': models[model_name],
                'train_on_dataset': train_on,
                'seed': seed,
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
            h_trainer = HumorTrainer(model_params, my_parse_args(), datasets, run_name)
            trained_model = h_trainer.train()
            wandb.finish()

            # save model
            h_trainer.save_model(model_path)

            # predict labels
            h_predictor = HumorPredictor(trained_model, h_trainer.get_test_datasets(), h_trainer.get_tokenizer())
            # predict on all datasets
            # for name in dataset_names:
            #     h_predictor.predict(name)
            #     h_predictor.write_dataset_predictions(predictions_path, name)

            # predict only on the dataset that used for train
            h_predictor.predict(train_on)
            h_predictor.write_dataset_predictions(predictions_path, train_on)

            print_str(f'FINISHED RUN ON SEED {seed} TRAINED ON {train_on}')

    print_str('FINISHED RUN')
