import numpy as np
import torch
import scipy.io as io
import os
import csv


def txt2struct(filename):
    lines = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    comment = False
    fieldnames = []
    fieldvalues = []
    for line in lines:
        if line.startswith('%{'):  # comment open
            comment = True
        if line.startswith('%}'):  # comment close
            comment = False
        if comment or len(line) == 0:
            continue
        if line.startswith('#'):  # field name
            fieldnames.append(line[1:])
        else:
            if line.count(' ') > 0:
                newval = line
            else:
                if line.isnumeric():
                    newval = float(line)
                else:
                    newval = line
            fieldvalues.append(newval)
    return [fieldnames, fieldvalues]


def textread(filename, headerlines):
    ids = []
    labels = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        ctr = 0
        for row in reader:
            if ctr < headerlines:
                ctr += 1
                continue
            ids.append(row[0])
            labels.append(row[1])
    return [ids, labels]


def shownotimplemented():
    print("NOT IMPLEMENTED")


def diva_targets_readcsvfile(filename):
    if not os.path.exists(filename):
        print("warning: file" + filename + " does not exist: initializing")
        production_id = []
        production_label = [];
    else:
        [production_id, production_label] = textread(filename, 1)
    return [production_id, production_label]


def read_file_list(filename, parameter):
    data = io.loadmat(filename)
    ret_list = []
    num_arr = data[parameter]
    for arr in num_arr[0]:
        asArray = np.array(arr)
        asArray = asArray.astype(np.float)
        ret_list.append(torch.from_numpy(asArray).to(torch.float64))
    return ret_list


def write_file(filename, mdict):
    if os.path.exists(filename):
        os.remove(filename)
    io.savemat(filename, mdict=mdict)


def read_file_parameter(filename, parameter):
    data = io.loadmat(filename)
    return torch.from_numpy(data[parameter].astype(np.double))


def read_file_parameter_alternate(filename, parameter):
    data = io.loadmat(filename, mdict=None, appendmat=False, struct_as_record=False)
    return data[parameter]


def read_input_file(filename):
    return read_file_parameter(filename, 'x')


def read_output_file(filename):
    return read_file_parameter(filename, 't')


def read_weight_file(filename):
    return read_file_parameter(filename, 'W')


def read_bias_file(filename):
    return read_file_parameter(filename, 'W0')


def read_file(filename):
    # read .mat file using scipy
    data = io.loadmat(filename)
    # put W and W0 into numpy arrays
    # print(data)
    # return torch tensor from numpy array
    W = torch.from_numpy(data['W'].astype(np.double)).to(torch.float64)
    W0 = None
    if "W0" in data:
        W0 = torch.from_numpy(data['W0'].astype(np.double)).to(torch.float64)

    return W, W0


"""
# Reshape a given 1D input tensor such that it is divided in half and the halfway point becomes col:2
# This is an equivalent method to the Matlab reshape call used in diva_weightsadaptive.m
"""


def tensor_reshape(tens):
    tSize = tens.size()[0]
    half = tSize / 2
    col1 = tens[0:int(half)]
    col2 = tens[int(half):int(tSize)]
    return torch.stack([col1, col2], dim=1)
