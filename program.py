import torch as th
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
from data_module import PlDataModule
from plnn_module import PlDNNModule
from result_writer import ResultWriter


def parse_arguments():
    parser = ArgumentParser()

    # program arguments
    parser.add_argument('--checkpoint_name', type=str, default='./checkpoint/final.ckpt')
    parser.add_argument('--data_folder', type=str, default='./data/test_1')
    parser.add_argument('--train_input', type=str, default='train.csv')
    parser.add_argument('--test_input', type=str, default='test.csv')
    parser.add_argument('--test_output', type=str, default='test_pred.csv')

    # trainer arguments
    parser.add_argument('--default_root_dir', type=str, default='./logs')
    parser.add_argument('--max_epochs', type=int, default=450)
    parser.add_argument('--gpus', type=int, default=(-1 if th.cuda.is_available() else 0))
    parser.add_argument('--auto_select_gpus', type=bool, default=True)
    parser.add_argument('--deterministic', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.02)

    # debug parameters - enable fast_dev_run for quick sanity check of the training loop
    parser.add_argument('--fast_dev_run', type=bool, default=False)

    args = parser.parse_args()
    return args


early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=True, mode='min')


def train_model(hparams):

    seed_everything(77)

    ml_module = PlDNNModule(hparams=hparams)
    data_module = PlDataModule(hparams=hparams)

    #model_trainer = pl.Trainer.from_argparse_args(hparams)
    model_trainer = pl.Trainer.from_argparse_args(hparams, callbacks=[early_stop_callback])

    model_trainer.logger = pl.loggers.TensorBoardLogger('logs/', name='exp')

    model_trainer.fit(ml_module, data_module)
    model_trainer.save_checkpoint(hparams.checkpoint_name)


def test_model(hparams):
    model = PlDNNModule.load_from_checkpoint(hparams.checkpoint_name)
    output_writer = ResultWriter(model, hparams=hparams)
    output_writer.test_model()


def main(args):
    train_model(hparams=args)
    test_model(hparams=args)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
