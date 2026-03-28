import atexit
import faulthandler
import os, sys, time
import argparse
import signal
import torch
import wandb
import matplotlib.pyplot as plt
import logging
import torch.distributed as dist
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.trainer import Trainer


def install_runtime_debug_handlers():
    logging.warning(
        "Runtime debug logging enabled: pid=%s cwd=%s TMPDIR=%s",
        os.getpid(),
        os.getcwd(),
        os.environ.get("TMPDIR", "<unset>"),
    )

    try:
        faulthandler.enable(all_threads=True)
        logging.warning("Runtime debug: faulthandler enabled")
    except Exception:
        logging.exception("Runtime debug: failed to enable faulthandler")

    for sig_name in ("SIGUSR1", "SIGUSR2"):
        if hasattr(signal, sig_name):
            signum = getattr(signal, sig_name)
            try:
                faulthandler.register(signum, file=sys.stderr, all_threads=True, chain=True)
                logging.warning("Runtime debug: registered faulthandler for %s", sig_name)
            except Exception:
                logging.exception("Runtime debug: failed to register faulthandler for %s", sig_name)

    def _terminating_signal_handler(signum, _frame):
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        logging.error("Runtime debug: received signal %s (%s)", signame, signum)
        try:
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        except Exception:
            logging.exception("Runtime debug: failed to dump traceback for signal %s", signame)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig_name in ("SIGTERM", "SIGINT"):
        if hasattr(signal, sig_name):
            signum = getattr(signal, sig_name)
            try:
                signal.signal(signum, _terminating_signal_handler)
                logging.warning("Runtime debug: installed handler for %s", sig_name)
            except Exception:
                logging.exception("Runtime debug: failed to install handler for %s", sig_name)

    def _log_debug_atexit():
        logging.warning("Runtime debug: atexit handler reached")

    atexit.register(_log_debug_atexit)

if __name__ == '__main__':
    # parsers
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/operators.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument("--root_dir", default='./', type=str, help='root dir to store results')
    parser.add_argument("--run_num", default='0', type=str, help='sub run config')
    parser.add_argument("--sweep_id", default=None, type=str, help='sweep config from ./configs/sweeps.yaml')
    parser.add_argument("--seed", default=None, type=int, help='override YAML seed for this run')
    parser.add_argument(
        "--train_shuffle",
        action='store_true',
        help='opt-in shuffle for training dataloader; leaves legacy behavior unchanged unless set'
    )
    parser.add_argument(
        "--random_train_subset",
        action='store_true',
        help='opt-in seeded random subset selection for train data when subsample > 1'
    )
    parser.add_argument(
        "--subset_seed",
        default=None,
        type=int,
        help='seed for random train subset selection; defaults to --seed / config seed'
    )
    parser.add_argument(
        "--debug_runtime_logging",
        action='store_true',
        help='enable extra runtime diagnostics for silent crashes and abrupt exits'
    )
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    if args.seed is not None:
        params['seed'] = args.seed
    if args.train_shuffle:
        params['train_shuffle'] = True
    if args.random_train_subset:
        params['random_train_subset'] = True
    if args.subset_seed is not None:
        params['subset_seed'] = args.subset_seed
    if args.debug_runtime_logging:
        params['debug_runtime_logging'] = True
        install_runtime_debug_handlers()
    trainer = Trainer(params, args)

    if args.sweep_id and trainer.world_rank==0:
        logging.disable(logging.CRITICAL)
        wandb.agent(args.sweep_id, function=trainer.launch, count=1, entity=trainer.params.entity, project=trainer.params.project) 
    else:
        trainer.launch()

    if dist.is_initialized():
        dist.barrier()

    logging.info('DONE')
