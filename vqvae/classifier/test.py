# Code adapted from: https://github.com/huyvnphan/PyTorch_CIFAR10

import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from module import TinyImagenet_Generated_Module

def main(hparams):
    seed_everything(0)    
    # If only train on 1 GPU. Must set_device otherwise PyTorch always store model on GPU 0 first
    if type(hparams.gpus) == str and len(hparams.gpus) == 2: # GPU number and comma e.g. '0,' or '1,'
        torch.cuda.set_device(int(hparams.gpus[0]))
    
    model = TinyImagenet_Generated_Module.load_from_checkpoint(hparams.ckpt, hparams=hparams)
    
    trainer = Trainer(gpus=hparams.gpus)
    trainer.test(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tinyimagenet')
    parser.add_argument('--classifier', type=str, default='wide_resnet50_2')
    parser.add_argument('--ckpt', type=str, default='classifier/logs/wide_resnet50_2_lr0.0001/version_9/checkpoints/epoch=3-step=1559.ckpt')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--gpus', default='0,')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    args = parser.parse_args()
    main(args)
