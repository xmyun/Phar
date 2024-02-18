import argparse


class SSL_Argument(object):
    def __init__(self, name, type, default, help=''):
        """
        Model specific arguments should be added via this class.
        """
        self.name = name
        self.type = type
        self.default = default
        self.help = help


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')