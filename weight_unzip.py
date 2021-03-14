import argparse
import json
import os
from os.path import join
from attgan import AttGAN

def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', type=str, default='06-36AM on March 09, 2021')
    parser.add_argument('--load_epoch', dest='load_epoch', type=int, default=0)
    parser.add_argument('--gpu', dest='gpu', type=bool, default=False)
    return parser.parse_args(args)

args_ = parse()
with open(join('output', args_.experiment_name, 'setting.txt'), 'r') as f:
    args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

args.gpu = args_.gpu
args.experiment_name = args_.experiment_name
args.load_epoch = args_.load_epoch
args.betas = (args.beta1, args.beta2)

model = AttGAN(args)
model.load(os.path.join('output', args.experiment_name, 'checkpoint','weights.'+str(args.load_epoch)+'.pth'))
model.saveG_D(os.path.join(
                'output', args.experiment_name, 'checkpoint', 'weights_unzip.{:d}.pth'.format(args.load_epoch)
            ), flag='unzip')