import timeit

import torch
import pandas as pd
import numpy as np
import os
import glob
import tqdm
from cnn_dataset_one import CnnDataset
from torch.utils.data import DataLoader

from network import CNN_1D

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device being used:", device)

nEpochs = 100
resume_epoch = 0
nTestInterval = 20
useTest = False
snapshot = 50
lr = 1e-3

dataset = 'elas_hand'

if dataset == 'elas_hand':
    num_classes = 10
elif dataset == 'open_hand':
    num_classes = 15
else:
    print('We only implemented elas_hand and open_hand datasets.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'elas1D'
saveName = modelName + '-' + dataset

def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):

    if modelName == 'elas1D':
        model = CNN_1D.CNN1D(num_classes=num_classes, pretrained=True)
        # train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
        #                 {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    else:
        print('We only implemented elas_hand models.')
        raise NotImplementedError

    weights = '../model/net.pth'
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights))
        print("loading successful")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs
    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
            map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(CnnDataset(dataset=dataset, split='train', clip_len=16), batch_size=20,
                                  shuffle=True, num_workers=4)
    val_dataloader = DataLoader(CnnDataset(dataset=dataset, split='val', clip_len=16), batch_size=20, num_workers=4)
    test_dataloader = DataLoader(CnnDataset(dataset=dataset, split='test', clip_len=16), batch_size=20, num_workers=4)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    for epoch in range(resume_epoch, num_epochs):
        for phase in ['train', 'val']:
            start_time = timeit.default_timer() #记录起始时间

            running_loss = 0.0
            running_corrects = 0.0

            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            for i, (inputs, labels) in enumerate(trainval_loaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                # print(i, inputs.shape,labels.shape)
                # 梯度清零
                optimizer.zero_grad()


                if phase == 'train':
                    # 前向计算，计算loss
                    out = model(inputs)
                else:
                    with torch.no_grad():
                        out = model(inputs)

                probs = torch.nn.Softmax(dim=1)(out)
                preds = torch.max(probs, 1)[1]
                loss = criterion(out, labels)

                #梯度下降
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

            if epoch % save_epoch == (save_epoch - 1):
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
                print("Save model at {}\n".format(
                    os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

            if useTest and epoch % test_interval == (test_interval - 1):
                model.eval()
                start_time = timeit.default_timer()

                running_loss = 0.0
                running_corrects = 0.0

                for inputs, labels in tqdm(test_dataloader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.no_grad():
                        outputs = model(inputs)
                    probs = torch.nn.Softmax(dim=1)(outputs)
                    preds = torch.max(probs, 1)[1]
                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / test_size
                epoch_acc = running_corrects.double() / test_size

                print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch + 1, nEpochs, epoch_loss, epoch_acc))
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")


if __name__ == '__main__':
    train_model()
