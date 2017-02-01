from iro.network import GAN


def train():
    print('Building Network')
    gan = GAN()

    print('Rescue Previous Network')
    try:
        gan.load_weights()
    except IOError:
        print('Cannot Rescue, Build New Network')

    print('Caching Data')
    gan.load_data()

    print('Training')
    gan.train()
