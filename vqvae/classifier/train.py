# Code adapted from: https://github.com/huyvnphan/PyTorch_CIFAR10

import os
import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from module import TinyImagenet_Generated_Module

def main(hparams):
    
    seed_everything(0)
    
    # If only train on 1 GPU. Must set_device otherwise PyTorch always store model on GPU 0 first
    if type(hparams.gpus) == str and len(hparams.gpus) == 2: # GPU number and comma e.g. '0,' or '1,'
        torch.cuda.set_device(int(hparams.gpus[0]))
    
    # Model
    if hparams.dataset == 'tinyimagenet':
        classifier = TinyImagenet_Generated_Module(hparams)

    # Trainer
    logdir = f"{hparams.classifier}_lr{hparams.learning_rate}"
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("classifier/logs", name=logdir)
    trainer = Trainer(callbacks=[lr_logger], gpus=hparams.gpus, max_epochs=hparams.max_epochs,
                      deterministic=True, logger=logger)
    trainer.fit(classifier)

    # Load best checkpoint
    checkpoint_path = os.path.join(os.getcwd(), "classifier/logs", logdir, hparams.classifier, 'version_' + str(classifier.logger.version), 'checkpoints')
    classifier = TinyImagenet_Generated_Module.load_from_checkpoint(os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0]))
    
    # Save weights from checkpoint
    statedict_path = os.path.join(os.getcwd(), 'classifier/models', hparams.classifier + '.pt')
    torch.save(classifier.model.state_dict(), statedict_path)
    
    # Test model
    trainer.test(classifier)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier', type=str, default='wide_resnet50_2')
    parser.add_argument('--dataset', type=str, default='tinyimagenet')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--gpus', default='0,') # use None to train on CPU
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    args = parser.parse_args()
    main(args)
