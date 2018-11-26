# -*- coding: utf-8 -*-
# @Author: marioZYN
# @Date:   2018-11-26 17:01:47
# @Last Modified by:   marioZYN
# @Last Modified time: 2018-11-26 18:14:30

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
import cv2
from torchvision import transforms
import glob
import time
import os
import sys
import datetime
import getopt

class SealionCountInception(torch.nn.Module):

    def __init__(self):

        super(SealionCountInception, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=19, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.layer2_1 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        self.layer2_2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        self.layer3_1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.layer3_2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size=14, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        self.layer5_1 = nn.Sequential(
            nn.Conv2d(16, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU()
        )
        self.layer5_2 = nn.Sequential(
            nn.Conv2d(16, 112, kernel_size=1, padding=0),
            nn.BatchNorm2d(112),
            nn.LeakyReLU()
        )
        self.layer6_1 = nn.Sequential(
            nn.Conv2d(160, 40, kernel_size=3, padding=1),
            nn.BatchNorm2d(40),
            nn.LeakyReLU()
        )
        self.layer6_2 = nn.Sequential(
            nn.Conv2d(160, 40, kernel_size=1, padding=0),
            nn.BatchNorm2d(40),
            nn.LeakyReLU()
        )
        self.layer7_1 = nn.Sequential(
            nn.Conv2d(80, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
        )
        self.layer7_2 = nn.Sequential(
            nn.Conv2d(80, 32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=17, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 5, kernel_size=1, padding=0),
            nn.BatchNorm2d(5),
            nn.ReLU()
        )

    def forward(self, x):

        x = self.layer1(x)

        x1 = self.layer2_1(x)
        x2 = self.layer2_2(x)
        x = torch.cat((x1, x2), 1)

        x1 = self.layer3_1(x)
        x2 = self.layer3_2(x)
        x = torch.cat((x1, x2), 1)

        x = self.layer_4(x)

        x1 = self.layer5_1(x)
        x2 = self.layer5_2(x)
        x = torch.cat((x1, x2), 1)

        x1 = self.layer6_1(x)
        x2 = self.layer6_2(x)
        x = torch.cat((x1, x2), 1)

        x1 = self.layer7_1(x)
        x2 = self.layer7_2(x)
        x = torch.cat((x1, x2), 1)

        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)

        return x

class TargetConstructNet(nn.Module):

    def __init__(self):

        super(TargetConstructNet, self).__init__()

        self.layer = nn.Conv2d(5, 5, kernel_size=48, padding=0)

        # assign weights
        w = torch.zeros((5, 5, 48, 48), dtype=torch.float)

        for i in range(5):
            w[i, i, :, :] = 1
        self.layer.weight = nn.Parameter(w)
        self.layer.bias = nn.Parameter(torch.zeros(5, dtype=torch.float))

        # freeze weights
        for param in self.layer.parameters():
            param.require_grad = False

    def forward(self, x):

        x = self.layer(x)

        return x

class CustomDataset(Dataset):

    def __init__(self, img_dirs, img_dotted_dirs):

        self.img_dirs = img_dirs
        self.img_dotted_dirs = img_dotted_dirs
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):

        img = np.load(self.img_dirs[index])
        tar = np.load(self.img_dotted_dirs[index])

        tar = np.rollaxis(tar, 2, 0)

        return self.transform(img), torch.from_numpy(tar).float()

    def __len__(self):

        return len(self.img_dirs)

def gen_test(tid, n_sep, r):

    train_df = pd.read_csv('../data/inception/original/train.csv')
    img = cv2.cvtColor(cv2.imread('../data/inception/original/{}.jpg'.format(tid)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    img = np.pad(img, ((47, 47), (47, 47), (0, 0)), mode='constant', constant_values=0)

    # divide image into pieces
    patches = []
    w, h = img.shape[1], img.shape[0]
    dw, dh = w // n_sep, h // n_sep
    for j in range(n_sep):
        h_end = min((j + 1) * dh, h)
        for k in range(n_sep):
            w_end = min((k + 1) * dw, w)
            piece = img[j * dh:h_end, k * dw:w_end, :]
            piece = transforms.ToTensor()(piece)
            patches.append(piece)

    # get real count
    real_adult_males = train_df.loc[train_df.train_id == tid]['adult_males'].values[0]
    real_adult_females = train_df.loc[train_df.train_id == tid]['adult_females'].values[0]
    real_subadult_males = train_df.loc[train_df.train_id == tid]['subadult_males'].values[0]
    real_juveniles = train_df.loc[train_df.train_id == tid]['juveniles'].values[0]
    real_pups = train_df.loc[train_df.train_id == tid]['pups'].values[0]

    return patches, (real_adult_males, real_subadult_males, real_adult_females, real_juveniles, real_pups)

def do_train(n_epoch, lr, input_model, target_model):
    # create log file
    files = glob.glob("../logs/inception/train/*")
    for f in files:
        os.remove(f)
    losslog = open('../logs/inception/train/loss.log', 'a')
    countlog = open('../logs/inception/train/count.log', 'a')
    processlog = open('../logs/inception/train/process.log', 'a')
    print('epoch\tavg_trainloss\tavg_validloss', file=losslog)
    print('epoch\tavg_train_cerr\tavg_valid_cerr', file=countlog)

    # loss
    criterion = torch.nn.L1Loss(size_average=True)
    optimizer = torch.optim.Adam(input_model.parameters(), lr=lr)

     # train, valid
    img_dirs = sorted(glob.glob('../data/inception/patches/*cell.npy'))
    img_dotted_dirs = sorted(glob.glob('../data/inception/patches/*dot.npy'))

    n_train = int(len(img_dirs) * 0.85)
    train_img_dirs = img_dirs[:n_train]
    valid_img_dirs = img_dirs[n_train:]
    train_dotted_dirs = img_dotted_dirs[:n_train]
    valid_dotted_dirs = img_dotted_dirs[n_train:]

    train_dataset = CustomDataset(train_img_dirs, train_dotted_dirs)
    valid_dataset = CustomDataset(valid_img_dirs, valid_dotted_dirs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=1,
                                               shuffle=False)

    print('there are totally {} train images, {} valid images'.format(len(train_dataset), len(valid_dataset)))

    # start training process
    best_valid_loss = float('inf')
    for epoch in range(n_epoch):
        input_model.train()
        train_count_error = [0]*5
        train_loss = 0
        start = time.time()

        for i, (img, img_dotted) in enumerate(train_loader):
            img = img.to(device)
            img_dotted = img_dotted.to(device)

            output = input_model.forward(img)
            with torch.no_grad():
                target = target_model.forward(img_dotted)

                real = target.sum(2).sum(2) / (48*48)
                pred = output.sum(2).sum(2) / (48*48)

                predictions = [int(pred.squeeze()[x].item()) for x in range(5)]
                reals = [int(real.squeeze()[x].item()) for x in range(5)]

                for j in range(5):
                    train_count_error[j] += abs(reals[j] - predictions[j])

                # create mask
                mask = torch.ones_like(target)
                total_pixels = mask.shape[2] * mask.shape[3]
                tmp = target.clone()
                tmp[tmp >= 1] = 1
                for j in range(5):
                    if j == 2:
                        continue
                    mask[0, j, :, :] += 9 * tmp[0, j, :, :]

            loss = criterion(output*mask, target*mask)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('\rEpoch [{}/{}], Step [{}/{}], Loss: {:.3f}, '
                  '{}/{}, {}/{}, {}/{}, {}/{}, {}/{}'
                  .format(epoch+1, n_epochs, i+1, len(train_loader), loss,
                          predictions[0], reals[0],
                          predictions[1], reals[1],
                          predictions[2], reals[2],
                          predictions[3], reals[3],
                          predictions[4], reals[4]), end='')

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}, '
                  '{}/{}, {}/{}, {}/{}, {}/{}, {}/{}'
                  .format(epoch + 1, n_epochs, i + 1, len(train_loader), loss,
                          predictions[0], reals[0],
                          predictions[1], reals[1],
                          predictions[2], reals[2],
                          predictions[3], reals[3],
                          predictions[4], reals[4]), file=processlog)

            processlog.flush()

        print()

        with torch.no_grad():
            input_model.eval()
            valid_count_error = [0]*5
            valid_loss = 0

            for i, (img, img_dotted) in enumerate(valid_loader):
                img = img.to(device)
                img_dotted = img_dotted.to(device)

                output = input_model.forward(img)
                target = target_model.forward(img_dotted)

                real = target.sum(2).sum(2) / (48 * 48)
                pred = output.sum(2).sum(2) / (48 * 48)

                predictions = [int(pred.squeeze()[x].item()) for x in range(5)]
                reals = [real.squeeze()[x].item() for x in range(5)]

                for j in range(5):
                    valid_count_error[j] += abs(reals[j] - predictions[j])
                    valid_loss += criterion(output, target).item()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(input_model.state_dict(), '../saved/inception/SealionCountInception_best.ckpt')
                print('\nbetter model found and saved, best valid loss {:.2f}'.format(best_valid_loss))
                print('\nbetter model found and saved, best valid loss {:.2f}'.format(best_valid_loss), file=processlog)
                processlog.flush()

        print()
        print('*** Epoch [{}/{}] {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, time {:.1f} min\n'
              .format(epoch+1, n_epochs,
                      valid_count_error[0] / len(valid_loader),
                      valid_count_error[1] / len(valid_loader),
                      valid_count_error[2] / len(valid_loader),
                      valid_count_error[3] / len(valid_loader),
                      valid_count_error[4] / len(valid_loader),
                      (time.time()-start)/60))

        print('*** Epoch [{}/{}] {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, time {:.1f} min\n'
              .format(epoch + 1, n_epochs,
                      valid_count_error[0] / len(valid_loader),
                      valid_count_error[1] / len(valid_loader),
                      valid_count_error[2] / len(valid_loader),
                      valid_count_error[3] / len(valid_loader),
                      valid_count_error[4] / len(valid_loader),
                      (time.time() - start) / 60), file=processlog)
        processlog.flush()

        print('{}\t{}\t{}'.format(epoch+1, train_loss/len(train_loader), valid_loss/len(valid_loader)), file=losslog)
        print('{}\t[{}/{}/{}/{}/{}]\t[{}/{}/{}/{}/{}]'.format(epoch+1,
                                                              train_count_error[0] / len(train_loader),
                                                              train_count_error[1] / len(train_loader),
                                                              train_count_error[2] / len(train_loader),
                                                              train_count_error[3] / len(train_loader),
                                                              train_count_error[4] / len(train_loader),

                                                              valid_count_error[0] / len(valid_loader),
                                                              valid_count_error[1] / len(valid_loader),
                                                              valid_count_error[2] / len(valid_loader),
                                                              valid_count_error[3] / len(valid_loader),
                                                              valid_count_error[4] / len(valid_loader)), file=countlog)
        losslog.flush()
        countlog.flush()

        torch.save(input_model.state_dict(), '../saved/inception/SealionCountInception_each.ckpt')

def do_test(r, s, input_model):
    # create log file
    files = glob.glob("../logs/inception/test/*")
    for f in files:
        os.remove(f)
    testlog = open('../logs/inception/test/test.log', 'a')

    # start testing process
    input_model.eval()
    with torch.no_grad():
        test_count_error = [0]*5

        coords_df = pd.read_csv('../data/inception/original/coords.csv')
        tids = pd.unique(coords_df.tid)
        tids = tids[tids >= 750]
        print('there are total {} test images'.format(len(tids)))

        for i, tid in enumerate(tids):
            patches, real_counts_tuple = gen_test(tid, s, r)

            start = time.time()
            predictions = [0]*5

            for patch in patches:
                patch = patch.to(device)
                output = input_model.forward(patch.unsqueeze(0))
                pred = output.sum(2).sum(2) / (48*48)
                for j in range(5):
                    predictions[j] += int(max(pred.squeeze()[j].item(), 0))

            for j in range(5):
                test_count_error[j] += abs(predictions[j] - real_counts_tuple[j])

            print('[{}, {}/{}] -- {}/{}, {}/{}, {}/{}, {}/{}, {}/{} -- time {:.2f} min'
                  .format(tid, i+1, len(tids),
                          predictions[0], real_counts_tuple[0],
                          predictions[1], real_counts_tuple[1],
                          predictions[2], real_counts_tuple[2],
                          predictions[3], real_counts_tuple[3],
                          predictions[4], real_counts_tuple[4], (time.time()-start)/60))

            print('[{}, {}/{}] -- {}/{}, {}/{}, {}/{}, {}/{}, {}/{} -- time {:.2f} min'
                  .format(tid, i + 1, len(tids),
                          predictions[0], real_counts_tuple[0],
                          predictions[1], real_counts_tuple[1],
                          predictions[2], real_counts_tuple[2],
                          predictions[3], real_counts_tuple[3],
                          predictions[4], real_counts_tuple[4], (time.time() - start) / 60), file=testlog)

        print()
        print('result: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}'
              .format(test_count_error[0] / len(tids),
                          test_count_error[1] / len(tids),
                          test_count_error[2] / len(tids),
                          test_count_error[3] / len(tids),
                          test_count_error[4] / len(tids)))

        print('result: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}'
              .format(test_count_error[0] / len(tids),
                      test_count_error[1] / len(tids),
                      test_count_error[2] / len(tids),
                      test_count_error[3] / len(tids),
                      test_count_error[4] / len(tids)), file=testlog)

def usage():
    print('Usage: ./SealionCountInception.py [-hrdlmny][--traintest]')
    print('   -h  print this message')
    print('   -r  specify resizing factor, max 1.0')
    print('   -s  specify number of patches(=s*s) to extract from each image')
    print('   -l  specify learning rate, default value 5e-5' )
    print('   -m  specify the model to be loaded')
    print('   -n  specify the number of epochs to train the model, default value is 50')
    print('   --train  set to train mode')
    print('   --test  set to test mode')
    print('example: ./SealionCountInception.py -r 1.0 -n 50 -s 5 --train')
    print('**Caution**, r and s values are a must for test mode, and they should be the same to the values used in genPatch.py')

if __name__ == '__main__':

    # set up gpu device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device != 'cpu':
        print(torch.cuda.get_device_name(0))
    else:
        print(device)

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'r:n:s:l:m:h', ['train', 'test'])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    load_from_past, train, test = False, False, False
    n_epoch = 50
    lr = 5e-5
    for opt, arg in opts:
        if opt == '-r':
            r = float(arg)
        elif opt == '-s':
            s = int(arg)  
        elif opt == '-n':
            n_epoch = int(arg)
        elif opt == '-l':
            lr = float(arg)
        elif opt == '-m':
            load_from_past = True
            model_name = arg
        elif opt == '-h':
            usage()
            sys.exit()
        elif opt == '--train':
            train = True
        elif opt == '--test':
            test = True
        else:
            print(opt)
            print('Error, unhandled options')
            sys.exit(2)

    # model
    input_model = SealionCountInception().to(device)
    target_model = TargetConstructNet().to(device)
    if load_from_past:
        input_model.load_state_dict(torch.load('../saved/inception/{}'.format(model_name)))
        print('model loaded')
        input_model.to(device)

    # train
    if train:
        do_train(n_epoch, lr, input_model, target_model)
    if test:
        do_test(r, s, input_model)
