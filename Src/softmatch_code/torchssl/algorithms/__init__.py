from queue import Full
from .fixmatch import FixMatch
from .flexmatch import FlexMatch
from .pimodel import PiModel
from .meanteacher import MeanTeacher
from .pseudolabel import PseudoLabel
from .uda import UDA
from .mixmatch import MixMatch
from .vat import VAT
from .remixmatch import ReMixMatch
from .dash import Dash
from .mpl import MPL
from .fullysupervised import FullySupervised
from .softmatch import SoftMatch
from .freematch import FreeMatch

# if any new alg., please append the dict
name2alg = {
    'fullysupervised': FullySupervised,
    'softmatch': SoftMatch,
    'fixmatch': FixMatch,
    'flexmatch': FlexMatch,
    'freematch': FreeMatch,
    'pimodel': PiModel,
    'meanteacher': MeanTeacher,
    'pseudolabel': PseudoLabel,
    'uda': UDA,
    'vat': VAT,
    'mixmatch': MixMatch,
    'remixmatch': ReMixMatch,
    'dash': Dash,
    'mpl': MPL
}


def get_algorithm(args, net_builder, tb_log, logger):
    try:
        model = name2alg[args.algorithm](
            args=args,
            net_builder=net_builder,
            num_classes=args.num_classes,
            ema_m=args.ema_m,
            lambda_u=args.ulb_loss_ratio,
            num_eval_iter=args.num_eval_iter,
            tb_log=tb_log,
            logger=logger
        )
        return model
    except KeyError as e:
        print(f'Unknown algorithm: {str(e)}')



