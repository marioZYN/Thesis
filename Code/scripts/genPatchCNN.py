# -*- coding: utf-8 -*-
# @Author: marioZYN
# @Date:   2018-11-26 18:12:10
# @Last Modified by:   marioZYN
# @Last Modified time: 2018-11-26 18:32:33

import getopt
import sys
import pandas as pd
import numpy as np
import glob
import os
import cv2

def usage():
    print('Usage: ./genPatchCNN.py [-n]')
    print('   -h  print this message')
    print('   -n  sepcify number of patches to extract from each sea lion type')
    print('example: ./genPatchCNN.py -n 2000')

def inside(ul, ur, dl, dr, c):
    if ul[0] <= c[0] <= dl[0] and ul[1] <= c[1] <= ur[1]:
        return True
    else:
        return False

def gen_patch(number):
    coords_df = pd.read_csv('../data/inception/original/coords.csv')

    # delete previous generated images
    files = glob.glob('../data/cnn/patches/*/*.png')
    for f in files:
        os.remove(f)

    data = {
        0: [], 1: [], 2: [], 3: [], 4: [], 5: []
    }

    # get sea lion patches
    for tid in pd.unique(coords_df.tid):
        # check terminating condiction
        total = 0
        for k, v in data.items():
            total += len(v)
        if total == 5*number:
            break

        stat = coords_df[coords_df.tid == tid]
        img = cv2.imread('../data/inception/original/{}.jpg'.format(tid))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(stat)):
            sealion_class = stat.iloc[i].cls
            if len(data[sealion_class]) == number:
                continue
            row = stat.iloc[i].row
            col = stat.iloc[i].col

            patch = img[max(0, row - 48):min(img.shape[0], row + 48), max(0, col - 48):min(img.shape[1], col + 48), :]
            if patch.shape != (96, 96, 3):
                continue
            else:
                data[sealion_class].append(patch)

    print('sea lions done')
    # get background patches
    for tid in pd.unique(coords_df.tid):
        stat = coords_df[coords_df.tid == tid]
        img = cv2.imread('../data/inception/original/{}.jpg'.format(tid))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for r in range(img.shape[0] // 96):
            for c in range(img.shape[1] // 96):
                patch = img[96 * r:min(img.shape[0], 96 * (r + 1)), 96 * c:min(img.shape[1], 96 * (c + 1)), :]
                ul, ur = (r * 96, c * 96), (r * 96, (c + 1) * 96)
                dl, dr = ((r + 1) * 96, c * 96), ((r + 1) * 96, (c + 1) * 96)
                if patch.shape != (96, 96, 3):
                    continue
                ok_flag = True
                for i in range(len(stat)):
                    row = stat.iloc[i].row
                    col = stat.iloc[i].col
                    if inside(ul, ur, dl, dr, (row, col)):
                        ok_flag = False
                        break
                if ok_flag:
                    data[5].append(patch)
                    if len(data[5]) == number:
                        print('backgrounds done')
                        return data

if __name__ == '__main__':

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'n:')
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-n':
            n = int(arg)
        else:
            print(opt)
            print('Error, unhandled options')
            sys.exit(2)
    
    data = gen_patch(n)
    sealion_types = ['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups', 'backgrounds']
    for k in data:
        path = '../data/cnn/{}/'.format(sealion_types[k])
        for i, p in enumerate(data[k]):
            cv2.imwrite(path + '{}.png'.format(i), cv2.cvtColor(p, cv2.COLOR_BGR2RGB))
        print(sealion_types[k], 'saved to images')


