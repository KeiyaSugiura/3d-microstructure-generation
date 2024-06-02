from .gan import Critic, Generator
from .srgan import Critic_SRGAN, Generator_SRGAN


# def get_generator(config):
#     _C = config["generator"]
#     return Generator(**_C["params"])


# def get_critic(config):
#     _C = config["critic"]
#     return Critic(**_C["params"])


def get_generator(config):
    _C = config["generator"]
    return Generator_SRGAN(**_C["params"])


def get_critic(config):
    _C = config["critic"]
    return Critic_SRGAN(**_C["params"])