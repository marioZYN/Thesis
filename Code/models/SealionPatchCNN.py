# -*- coding: utf-8 -*-
# @Author: marioZYN
# @Date:   2018-11-26 18:35:21
# @Last Modified by:   marioZYN
# @Last Modified time: 2018-11-26 19:13:08
import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import pandas as pd
import os
import glob
import time
import sys
import getopt

def gen_test_patches(img):
    res = []
    s = 96
    for i in range(img.shape[0]//s):
        for j in range(img.shape[1]//s):
            patch = img[s*i:min(img.shape[0], s*(i+1)), s*j:min(img.shape[1], s*(j+1)), :]
            if patch.shape != (s, s, 3):
                continue
            else:
                res.append(patch)
    return res

class SealionPatchCNN(torch.nn.Module):

    def __init__(self):

        super(SealionPatchCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 5, kernel_size=3, padding=0),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(5, 5, kernel_size=3, padding=0),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Linear(20*20*10, 6)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

class CustomDataset(Dataset):

    def __init__(self, dirs):

        self.dirs = dirs
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        adult_male = self.transform(cv2.cvtColor(cv2.imread(self.dirs[0][index]), cv2.COLOR_BGR2RGB))
        subadult_male = self.transform(cv2.cvtColor(cv2.imread(self.dirs[1][index]), cv2.COLOR_BGR2RGB))
        adult_female = self.transform(cv2.cvtColor(cv2.imread(self.dirs[2][index]), cv2.COLOR_BGR2RGB))
        juvenile = self.transform(cv2.cvtColor(cv2.imread(self.dirs[3][index]), cv2.COLOR_BGR2RGB))
        pup = self.transform(cv2.cvtColor(cv2.imread(self.dirs[4][index]), cv2.COLOR_BGR2RGB))
        background = self.transform(cv2.cvtColor(cv2.imread(self.dirs[5][index]), cv2.COLOR_BGR2RGB))

        return torch.stack([adult_male, subadult_male, adult_female, juvenile, pup, background]), \
               torch.from_numpy(np.asarray([0, 1, 2, 3, 4, 5]))

    def __len__(self):

        return len(self.dirs[0])

def do_train(learning_rate, n_epochs, batch_sizem, model):
    # set up log files
    files = glob.glob("../logs/cnn/train/*")
    for f in files:
        os.remove(f)
    processlog = open('../logs/cnn/process.log', 'a')
    trainlog = open('../logs/cnn/train.log', 'a')
    testlog = open('../logs/cnn/test.log', 'a')
    print('epoch\ttrain_loss\tvalid_loss\ttrain_acc\tvalid_acc', file=trainlog)
    print('tid\tread\tpred', file=testlog)

    # start training process
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train, valid
    sealion_types = ['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups', 'backgrounds']
    dirs = []
    for t in sealion_types:
        dirs.append(sorted(glob.glob('../data/cnn/{}/*.png'.format(t))))

    n_train = int(len(dirs[0]) * 0.85)
    train_img_dirs = [x[:n_train] for x in dirs]
    valid_img_dirs = [x[n_train:] for x in dirs]

    train_dataset = CustomDataset(train_img_dirs)
    valid_dataset = CustomDataset(valid_img_dirs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=b,
                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=1,
                                               shuffle=False)

    print('there are totally {} train images, {} valid images'.format(len(train_dataset), len(valid_dataset)))

    best_valid_loss = float('inf')
    for epoch in range(n_epochs):
        start = time.time()
        train_loss = 0
        train_correct = 0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device).reshape(-1, 3, 96, 96)
            y = y.to(device).reshape(-1)

            output = model.forward(x)

            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss += loss.item()
                pred_labels = [output[x, :].argmax().item() for x in range(y.shape[0])]
                for k, v in enumerate(pred_labels):
                    if v == k % 6:
                        train_correct += 1

            print('\rEpoch [{}/{}]. Step [{}/{}], loss {:.3f}'
                  .format(epoch+1, n_epochs, i+1, len(train_loader), loss.item()), end='')
            print('Epoch [{}/{}]. Step [{}/{}], loss {:.3f}'
                  .format(epoch+1, n_epochs, i+1, len(train_loader), loss.item()), file=processlog)

        print()

        with torch.no_grad():
            valid_loss = 0
            valid_correct = 0
            for i, (x, y) in enumerate(valid_loader):
                x = x.to(device).reshape(-1, 3, 96, 96)
                y = y.to(device).reshape(-1)

                output = model.forward(x)
                loss = criterion(output, y)

                valid_loss += loss.item()
                pred_labels = [output[x, :].argmax().item() for x in range(y.shape[0])]
                for k, v in enumerate(pred_labels):
                    if v == k % 6:
                        valid_correct += 1
            
            valid_loss /= (len(valid_loader)*6)
            train_loss /= (len(train_loader)*6)
            valid_acc = valid_correct / (len(valid_loader)*6)
            train_acc = train_correct / (len(train_loader)*6*b)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), '../saved/cnn/SealionPatchCNN_best.ckpt')
                print('\nbetter model find and saved, valid loss = {:.3f}'.format(valid_loss))
                print('\nbetter model find and saved, valid loss = {:.3f}'.format(valid_loss), file=processlog)
            
            print()
            print('Epoch [{}/{}], train_loss {:.2f}, valid_loss {:.2f}, train_acc {:.2f}, valid_acc {:.2f}, time {:.2f}'
                  .format(epoch+1, n_epochs, train_loss, valid_loss, train_acc, valid_acc, (time.time()-start)/60))
            print('{}\t{}\t{}\t{}\t{}'.format(epoch+1, train_loss, valid_loss, train_acc, valid_acc), file=trainlog)

            torch.save(model.state_dict(), '../saved/cnn/SealionPatchCNN_each.ckpt')

def do_test():
    # set up log file
    files = glob.glob("../logs/cnn/test/*")
    for f in files:
        os.remove(f)
    testlog = open('../logs/cnn/test.log', 'a')

    with torch.no_grad():
        coords_df = pd.read_csv('../data/inception/original/coords.csv')
        train_df = pd.read_csv('../data/inception/original/train.csv')
        tids = pd.unique(coords_df.tid)
        tids = tids[tids >= 750]
        print('there are total {} test images'.format(len(tids)))
        errors = [0]*5

        for i, tid in enumerate(tids):
            path = '../data/inception/original/{}.jpg'.format(tid)
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            patches = gen_test_patches(img)

            # get real counts
            real = [0]*6
            real[0] = train_df.loc[train_df.train_id == tid]['adult_males'].values[0]
            real[1] = train_df.loc[train_df.train_id == tid]['adult_females'].values[0]
            real[2] = train_df.loc[train_df.train_id == tid]['subadult_males'].values[0]
            real[3] = train_df.loc[train_df.train_id == tid]['juveniles'].values[0]
            real[4] = train_df.loc[train_df.train_id == tid]['pups'].values[0]

            # make predictions
            pred = [0]*6
            for patch in patches:
                patch_tensor = transforms.ToTensor()(patch).unsqueeze(0).to(device)
                res = model.forward(patch_tensor)
                label = res.argmax().item()
                pred[label] += 1

            print('{}, [{}/{}], {}, {}'.format(tid, i+1, len(tids), real, pred))
            print('{}, [{}/{}], {}, {}'.format(tid, i+1, len(tids), real, pred), file=testlog)

            # calculate error
            for j in range(5):
                errors[j] += abs(real[j] - pred[j])

        print('avg error {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'
              .format(errors[0]/len(tids),
                      errors[1] / len(tids),
                      errors[2] / len(tids),
                      errors[3] / len(tids),
                      errors[4] / len(tids)))

        print('avg error {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'
              .format(errors[0] / len(tids),
                      errors[1] / len(tids),
                      errors[2] / len(tids),
                      errors[3] / len(tids),
                      errors[4] / len(tids)), file=testlog)

def usage():
    print('Usage: ./SealionPatchCNN.py [-hlnbm][--traintest]')
    print('   -h  print this message')
    print('   -l  specify learning rate, default value 1e-5' )
    print('   -m  specify the model to be loaded')
    print('   -n  specify the number of epochs to train the model, default value is 100')
    print('   -b  specify the batch_size, default values is 5')
    print('   --train  set to train mode')
    print('   --test  set to test mode')
    print('example: ./SealionPatchCNN.py --train')

if __name__ == '__main__':

    # set up gpu device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device != 'cpu':
        print(torch.cuda.get_device_name(0))
    else:
        print(device)

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'l:n:b:m:h', ['train', 'test'])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    train, test, load_from_past = False, False, False
    l = 1e-5
    n = 100
    b = 5
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        if opt == '-l':
            l = float(arg)
        elif opt == '-n':
            n = int(arg)
        elif opt == '-b':
            b = int(arg)
        elif opt == '-m':
            load_from_past = True
            model_name = arg
        elif opt == '--train':
            train = True
        elif opt == '--test':
            test = True
        else:
            print(opt)
            print('Error, unhandled options')
            sys.exit(2)

    # model
    model = SealionPatchCNN().to(device)
    if load_from_past:
        model.load_state_dict('../saved/cnn/{}'.format(model_name))
        print('model loaded')
        model.to(device)

    if train:
        do_train(l, n, b, model)
    if test:
        do_test()



