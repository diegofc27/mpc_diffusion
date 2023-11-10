if __name__ == '__main__':
    from ml_logger import logger, instr, needs_relaunch
    from analysis import RUN
    import jaynes
    from scripts.train import main
    from config.locomotion_config import Config
    from params_proto.hyper import Sweep
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--exp', type=str)
    #add args
    args = parser.parse_args()
    sweep = Sweep(RUN, Config).load(f"/home/fernandi/projects/decision-diffuser/code/analysis/safe_grid/{args.exp}.jsonl")

    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        jaynes.config("local")
        thunk = instr(main, **kwargs)
        jaynes.run(thunk)

    # jaynes.listen()
 