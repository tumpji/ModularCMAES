from argparse import ArgumentParser

from .configurablecmaes import ConfigurableCMAES, evaluate


parser = ArgumentParser(
    description='Run single function CMAES')
parser.add_argument(
    '-f', "--fid", type=int,
    help="bbob function id", required=False, default=5
)
parser.add_argument(
    '-d', "--dim", type=int,
    help="dimension", required=False, default=5
)
parser.add_argument(
    '-i', "--iterations", type=int,
    help="number of iterations per agent",
    required=False, default=50
)
parser.add_argument(
    '-l', '--logging', required=False,
    action='store_true', default=False
)
parser.add_argument(
    '-L', '--label', type=str, required=False,
    default=""
)
parser.add_argument(
    "-s", "--seed", type=int, required=False,
    default=42
)
parser.add_argument(
    "-p", "--data_folder", type=str, required=False
)
parser.add_argument(
    "-a", "--arguments", nargs='+', required=False
)

args = vars(parser.parse_args())
for arg in (args.pop("arguments") or []):
    exec(arg, None, args)

evaluate(**args)
