import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn

from video_features_generator.opts import parse_opts
from video_features_generator.model import generate_model
from video_features_generator.mean import get_mean
from video_features_generator.classify import classify_video

def load_model(model, opt):
    if not model:
        model = generate_model(opt)
        print('loading model {}'.format(opt.model))
        model_data = torch.load(opt.model)
        assert opt.arch == model_data['arch']
        model.load_state_dict(model_data['state_dict'])
        model.eval()
        if opt.verbose:
            print(model)

def extract_features(model, opt, video_path):
    if not model:
        load_model(model, opt)
        print("Model is null")
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 400

    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        print(model)

    class_names = []
    with open('video_features_generator/class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'

    # if os.path.exists('/home/ubuntu/temp_files/'):
    #    subprocess.call('rm -rf /home/ubuntu/temp_files/', shell=True)

    outputs = []
    if os.path.exists(video_path):
        print(video_path)
        subprocess.call('mkdir /home/ubuntu/temp_files/', shell=True)
        subprocess.call('ffmpeg -i {} /home/ubuntu/temp_files/image_%05d.jpg'.format(video_path),
                        shell=True)

        result = classify_video('/home/ubuntu/temp_files/', video_path, class_names, model, opt)
        outputs.append(result)

        # subprocess.call('rm -rf /home/ubuntu/temp_files/', shell=True)
    else:
        print('{} does not exist'.format(video_path))

    # if os.path.exists('/home/ubuntu/temp_files/'):
    #    subprocess.call('rm -rf /home/ubuntu/temp_files/', shell=True)

    # with open(opt.output, 'w') as f:
    #    json.dump(outputs, f)
    return outputs


if __name__ == "__main__":
    opt = parse_opts()
    load_model(opt.model, opt)
    extract_features(opt.model, opt)

