from iro.network import GAN


def train():
    print('Building Network')
    gan = GAN()

    print('Rescuing Previous Weights')
    try:
        gan.load_weights()
    except IOError:
        print('Cannot Rescue, Initializing New Weights')

    print('Caching Data')
    gan.load_data()

    print('Training')
    gan.train()
