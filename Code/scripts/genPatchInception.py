# -*- coding: utf-8 -*-
# @Author: marioZYN
# @Date:   2018-11-26 12:54:20
# @Last Modified by:   marioZYN
# @Last Modified time: 2018-11-26 18:15:09

import getopt
import sys
import pandas as pd
import numpy as np
import glob
import os
import cv2

def usage():
    print('Usage: ./genPatchInception.py [-rnsh]')
    print('   -h  print this message')
    print('   -r  specify resizing factor, max 1.0')
    print('   -n  sepcify number of images to use')
    print('   -s  specify number of patches(=s*s) to extract from each image')
    print('example: ./genPatchInception.py -r 1.0 -n 500 -s 5 ')
    print('the example above will generate 25 patches from each image with full resolution'+
          ', and we use 500 images in total')

def gen_patch(n_sep, n_images, r):
    """
    generate training dataset for five sea lion class counting problem. Each patch is under dimension MxNxD, where D equals the number of sea lion types. Sea lion type order in the patch is: adult_males, subadult_males, adult_females, juveniles, pups

    n_sep: each row, col will be divided into n_sep parts, thus total patches per image is n_sep^2
    n_image: number of images used to extract patches
    r: resizing factor
    """
    coords_df = pd.read_csv('../data/inception/original/coords.csv')
    train_df = pd.read_csv('../data/inception/original/train.csv')

    # delete previous generated images
    files = glob.glob('../data/inception/patches/*.npy')
    for f in files:
        os.remove(f)

    # sort by sea lion counts
    features = train_df.columns != 'train_id'
    train_df['sum'] = train_df[list(train_df.columns[features])].sum(axis=1)
    train_df = train_df.sort_values('sum', ascending=False)

    # get tids
    tids = list(train_df.train_id)
    legal_tids = pd.unique(coords_df.tid)
    legal_tids = [x for x in legal_tids if x < 750]
    tids = list(filter(lambda x: x in legal_tids, tids))[:n_images]
    print('there are total {} images'.format(len(tids)))

    # generate img and img_dotted
    cnt = 0
    for i, tid in enumerate(tids):
        img = img = cv2.cvtColor(cv2.imread('../data/inception/original/'+ str(tid) + '.jpg'), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))

        y = np.zeros((img.shape[0], img.shape[1], 5))
        for j in range(5):
            rows = (coords_df[(coords_df['tid'] == tid) & (coords_df['cls'] == j)].row * r).astype(int)
            cols = (coords_df[(coords_df['tid'] == tid) & (coords_df['cls'] == j)].col * r).astype(int)
            y[rows, cols, j] = 1

        # divide the image into pieces
        w, h = img.shape[1], img.shape[0]
        dw, dh = w // n_sep, h // n_sep
        for j in range(n_sep):
            h_end = min((j + 1) * dh, h)
            for k in range(n_sep):
                w_end = min((k + 1) * dw, w)
                piece = img[j * dh:h_end, k * dw:w_end, :]
                piece_y = y[j * dh:h_end, k * dw:w_end, :]

                # filter out empty patches
                if piece_y[:,:,0].sum() == 0 or piece_y[:,:,1].sum() == 0 \
                        or piece_y[:,:,2].sum() == 0 or piece_y[:,:,3].sum() == 0 or piece_y[:,:,4].sum() == 0:
                    continue

                piece = np.pad(piece, ((47, 47), (47, 47), (0, 0)), mode='constant', constant_values=0)
                piece_y = np.pad(piece_y, ((47, 47), (47, 47), (0, 0)), mode='constant', constant_values=0)

                np.save('../data/inception/patches/{}_cell'.format(cnt), piece)
                np.save('../data/inception/patches/{}_dot'.format(cnt), piece_y)

                cnt += 1
                print('\r{} patch generates...'.format(cnt), end='')

    print('\npatch generation completes')

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'r:n:s:h')
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    if not opts:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt == '-r':
            r = float(arg)
        elif opt == '-n':
            n = int(arg)
        elif opt == '-s':
            s = int(arg)
        else:
            print(opt)
            print('Error, unhandled options')
            sys.exit(2)

    gen_patch(n, s, r)




    