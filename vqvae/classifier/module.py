# Code adapted from: https://github.com/huyvnphan/PyTorch_CIFAR10

import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models

def get_classifier(classifier, pretrained):
    if classifier == 'vgg16_bn':
        return models.vgg16(pretrained=pretrained)
    elif classifier == 'alexnet':
        return models.alexnet(pretrained=pretrained)
    elif classifier == 'mnasnet1':
        return models.mnasnet1_0(pretrained=pretrained)
    elif classifier == 'resnet18':
        return models.resnet18(pretrained=pretrained)
    elif classifier == 'shufflenet':
        return models.shufflenet_v2_x1_0(pretrained=pretrained)
    elif classifier == 'resnet50':
        return models.resnext50_32x4d(pretrained=pretrained)
    elif classifier == 'squeezenet':
        return models.squeezenet1_0(pretrained=pretrained)
    elif classifier == 'densenet161':
        return models.densenet161(pretrained=pretrained)
    elif classifier == 'wide_resnet50_2':
        return models.wide_resnet50_2(pretrained=pretrained)
    elif classifier == 'mobilenet_v2':
        return models.mobilenet_v2(pretrained=pretrained)
    elif classifier == 'googlenet':
        return models.googlenet(pretrained=pretrained)
    elif classifier == 'inception_v3':
        return models.inception_v3(pretrained=pretrained)
    else:
        raise NameError('Please enter a valid classifier')
 
class TinyImagenet_Generated_Module(pl.LightningModule):
    def __init__(self, hparams, pretrained=True):
        super().__init__()
        self._hparams = hparams
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = get_classifier(hparams.classifier, pretrained)
        self.train_size = len(self.train_dataloader().dataset)
        print("training set size:", self.train_size)
        self.val_size = len(self.val_dataloader().dataset)
        print("validation set size:", self.val_size)
 
    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        argmax = torch.max(predictions, 1)[1]
        accuracy = torch.sum(argmax == labels.data).float() / batch[0].size(0)

        # https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
        pred_5 = predictions.topk(5,1,largest=True,sorted=True)[1]
        pred_5 = pred_5.t()
        correct = pred_5.eq(labels.view(1, -1).expand_as(pred_5))
        correct_k = correct[:5].reshape(-1).float().sum(0, keepdim=True)
        top_5_accuracy = correct_k.mul_(100.0 / batch[0].size(0))

        return loss, accuracy, top_5_accuracy
 
    def training_step(self, batch, batch_nb):
        loss, accuracy, top_5_accuracy= self.forward(batch)
        logs = {'loss/train': loss, 'accuracy/train': accuracy, 'top_5_accuracy/train': top_5_accuracy}
        return {'loss': loss, 'log': logs}
 
    def validation_step(self, batch, batch_nb):
        avg_loss, accuracy, top_5_accuracy = self.forward(batch)
        loss = avg_loss * batch[0].size(0)
        corrects = accuracy * batch[0].size(0)
        top_5_corrects = top_5_accuracy * batch[0].size(0)
        logs = {'loss/val': loss, 'corrects': corrects, 'top_5_corrects/val': top_5_corrects}
        return logs

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss/val'] for x in outputs]).sum() / self.val_size * 200 / 10
        accuracy = torch.stack([x['corrects'] for x in outputs]).sum() / self.val_size * 200 / 10
        top_5_accuracy = torch.stack([x['top_5_corrects/val']  for x in outputs]).sum() / self.val_size * 200 / 10
        logs = {'loss/val': loss, 'accuracy/val': accuracy, 'top_5_accuracy/val': top_5_accuracy}
        return {'val_loss': loss, 'log': logs}
    
    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)
    
    def test_epoch_end(self, outputs):
        accuracy = self.validation_epoch_end(outputs)['log']['accuracy/val']
        accuracy = round((100 * accuracy).item(), 3)
        top_5_accuracy = round(self.validation_epoch_end(outputs)['log']['top_5_accuracy/val'].item(), 3)
        result = {'Accuracy': accuracy, 'Top 5 Accuracy': top_5_accuracy}
        return {'progress_bar': result}
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._hparams.learning_rate,
                                     weight_decay=self._hparams.weight_decay)
            
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
        scheduler = {'scheduler': scheduler,
                     'interval': 'step', 'name': 'learning_rate'}
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        transform_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((64, 64), (0.5, 1)),
            ],
        ) # transforms.Normalize(self.mean, self.std)])
        dataset = ImageFolder(self._hparams.data_dir, transform_val)
        dataloader = DataLoader(dataset, batch_size=self._hparams.batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader
    
    def val_dataloader(self):
        transform_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # transforms.Normalize(self.mean, self.std)])
        dataset = ImageFolder(self._hparams.data_dir, transform_val)
        dataloader = DataLoader(dataset, batch_size=self._hparams.batch_size, num_workers=4, pin_memory=True, shuffle=False)
        return dataloader
    
    def test_dataloader(self):
        return self.val_dataloader()