from iro.download import download
from iro.preload import preload
from iro.train import train
from iro.predictor import predict
from iro.validator import validate

method = input("1. Download\n2. Preload\n3. Train\n4. Predict\n5. Validate\nEnter Your Input: ")
if method == '1':
    download()
elif method == '2':
    preload()
elif method == '3':
    train()
elif method == '4':
    predict()
elif method == '5':
    validate()
else:
    print('No method matched')

exit()
