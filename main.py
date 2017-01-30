import sys
from iro.download import download
from iro.preload import preload
from iro.train import train
from iro.predictor import predict
from iro.validator import validate

if __name__ == '__main__':
    args = sys.argv
    if args[1] == 'download':
        download()
    elif args[1] == 'preload':
        preload()
    elif args[1] == 'train':
        train()
    elif args[1] == 'predict':
        predict(args[2], args[3])
    elif args[1] == 'validate':
        validate()
    else:
        print('start with args: download, preload, train, predict, validate')
    exit()
