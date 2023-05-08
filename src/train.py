from src.trainer import Trainer


def train(config):
    trainer = Trainer(config)
    trainer.train()
    print('training complete')


def test(config):
    trainer = Trainer(config)
    trainer.test()
    print('testing complete')
