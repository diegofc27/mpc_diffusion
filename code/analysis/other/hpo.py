if __name__ == '__main__':
    from ml_logger import logger, instr, needs_relaunch
    from analysis import RUN
    import jaynes
    #from scripts.evaluate_inv_parallel import evaluate
    #from scripts.evaluate_skills import evaluate
    
    #from scripts.evaluate_skills_parallel import evaluate
    #from scripts.evaluate_panda_parallel_script import evaluate
    #from scripts.eval_point import evaluate
    #from scripts.evaluate_safe_parallel import evaluate
    from scripts.hpo_safe_parallel_eval_multiple import hpo
    from scripts.hpo_safe_factor import hpo_w
    #from scripts.find_composition_w import evaluate
    from config.locomotion_config import Config
    from params_proto.hyper import Sweep

    sweep = Sweep(RUN, Config).load("/home/fernandi/projects/decision-diffuser/code/analysis/omni/combine_skills/basic_3.jsonl")

    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        jaynes.config("local")
        thunk = instr(hpo_w, **kwargs)
        jaynes.run(thunk)

    # jaynes.listen()