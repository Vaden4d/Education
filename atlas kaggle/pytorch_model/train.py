import numpy as np
import argparse
import os
import json

import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from utils import FLWithLogits, macro_f1
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from Model import DynamicUnet, Encoder
from Trainer import Trainer
from Data import Data, LabeledData, Splitter
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--trainimages', type=str, default='data/train/images')
parser.add_argument('--masksdir', type=str, default='data/train/masks/')
parser.add_argument('--testimages', type=str, default='data/test/images')
parser.add_argument('--logdir', type=str, default='data/logs')
parser.add_argument('--chkpdir', type=str, default='data/chkp')
parser.add_argument('--chkpname', type=str, default='none')
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--train_batch_size', type=int, default=200)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.25)
parser.add_argument('--clip_norm', type=float, default=0.1)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# creating dataloaders
dataset = LabeledData()
dataset.download_metadata()
splitter = Splitter(dataset)
train_dataset, test_dataset = splitter.train_test_split(test_size=0.2)

train_dataset.batch_size = args.train_batch_size
test_dataset.batch_size = args.test_batch_size

#train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

print('Number of samples for train - {}'.format(len(train_dataset)))
print('Number of samples for test - {}'.format(len(test_dataset)))
print('Train batch size - {}'.format(args.train_batch_size))
print('Test batch size - {}'.format(args.test_batch_size))

# optimizer
lr = args.lr

model = Model()
optimizer = Adam(model.parameters(), lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, eps=1e4)
clip_norm = args.clip_norm
criterion = FLWithLogits()
metrics = macro_f1

writer = SummaryWriter(args.logdir)
trainer = Trainer(model, optimizer, criterion, metric, clip_norm, writer, device)

for epoch in range(args.num_epochs):
    for train_batch, test_batch in zip(train_loader, test_loader):

        train_images, train_labels = train_batch
        test_images, test_labels = test_batch

        train_loss, train_f1 = trainer.train_step((train_images, train_labels))
        test_loss, test_f1 = trainer.test_step((test_images, test_labels))

        print('Iteration - {}'.format(trainer.num_updates))
        print('Train loss - {x:.5f}, test loss - {y:.5f}'.format(x=train_loss, y=test_loss))
        print('Train MacroF1 - {x:.5f}, test MacroF1 - {y:.5f}'.format(x=train_f1, y=test_f1))
        print('LR - {}'.format(lr))

        trainer.writer.add_scalars('Loss', {'train' : train_loss, 'test' : test_loss}, trainer.num_updates)
        trainer.writer.add_scalars('MacroF1', {'train' : train_iou, 'test' : test_iou}, trainer.num_updates)

        #if trainer.num_updates % args.save_every == 0:
        #    chkp_name = 'iter-{i}-train_loss-{tr_l:.5f}-test_loss-{te_l:.5f}.chkp'.format(i=trainer.num_updates,
        #                                                                                tr_l=train_loss,
        #                                                                                te_l=test_loss)
        #    save_model(trainer.model, args.chkpdir, chkp_name)

    scheduler.step(test_loss)
    lr = [group['lr'] for group in optimizer.param_groups][0]
