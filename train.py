import data
import argparse
from model import EDSR
from DIV2K import *
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data/General-100")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=10,type=int)
parser.add_argument("--savedir",default='saved_models')
parser.add_argument("--iterations",default=1000,type=int)
args = parser.parse_args()
# data.load_dataset(args.dataset)
down_size = args.imgsize//args.scale
network = EDSR(down_size,args.layers,args.featuresize,args.scale)
network.test_data = DIV2K().get_test()
# network.set_data_fn(data.get_batch,(args.batchsize,args.imgsize,down_size),data.get_test_set,(args.imgsize,down_size))
network.train(args.iterations,args.savedir)
