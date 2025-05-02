from models import Generator, Discriminator, SymSimulator
from datasets import Dataset, load_dataset
from utils import *
import gc
import sys
import argparse
from functools import partial
import re
from pathlib import Path



parser = argparse.ArgumentParser()



parser.add_argument('--data') # data path

parser.add_argument('--model', type=str, default='gpt-4.1')  # the model name
parser.add_argument('--visualize', action='store_true')  # whether visualize the language model output
parser.add_argument('--batch_process', action='store_true') # whether use batch processing for inference

parser.add_argument('--output_dir')  # the output directory for inference results

parser.add_argument('--max_steps', type=int, default=20)  # the maximum number of steps for reasoning
parser.add_argument('--num_rollouts', type=int, default=8)  # the number of rollouts for each problem
parser.add_argument('--num_generations', type=int, default=5)  # the number of generations for each step
parser.add_argument('--cmp_per_opt', type=int, default=1)  # the number of comparisons per option
parser.add_argument('--group_size', type=int, default=3) # the group size for single-time comparison (recommend: 2 or 3)
parser.add_argument('--beam_width', type=int, default=3)

parser.add_argument('--enable_symbolic', action='store_true')  # whether enable symbolic reasoning
parser.add_argument('--iterative_correction', action='store_true')  # whether enable iterative correction
parser.add_argument('--contrastive_ranking', action='store_true')  # whether enable contrastive ranking


args = parser.parse_args()





def SymPlanner(sample, output_dir, max_steps=20, num_rollouts=8, num_generations=3, 
                cmp_per_opt=1, group_size=3, beam_width=3, enable_symbolic=True, num_future_steps=1,
                visualize=False, model='gpt-4o', iterative_correction=True, contrastive_ranking=True, 
                use_API=True):
    '''
    Run the workflow of SymPlanner.

    Args:
        output_dir (str): The output directory.
        max_steps (int): The maximum number of steps for reasoning.
        num_rollouts (int): The number of rollouts for each problem.
        num_generations (int): The number of generations for each step.
        cmp_per_opt (int): The number of comparisons per option.
        group_size (int): The group size for single-time comparison.
        enable_symbolic (bool): Whether to enable symbolic reasoning.
        visualize (bool): Whether visualize the language model output.

    Returns:
        None
    '''
    # Initialize Generator and Discriminator
    simulator = SymSimulator(use_API=use_API, parse_model='gpt-4.1')
    agent_gen = Generator(use_API=use_API, model=model, enable_symbolic=enable_symbolic, simulator=simulator)
    agent_disc = Discriminator(use_API=use_API, model=model, simulator=simulator)

    cnt = 0
    while cnt < max_steps:
        # Generator perform inference
        flag_finish = agent_gen.inference([sample], output_dir, num_rollouts, num_generations, num_future_steps, beam_width,
                                            visualize=visualize, iterative_correction=iterative_correction)
        if flag_finish:
            break
        # Discriminator perform inference
        agent_disc.inference([sample], output_dir, cmp_per_opt, group_size, beam_width, num_generations, visualize=visualize, 
                             contrastive_ranking=contrastive_ranking)
        cnt += 1

    # Perform final aggregation for all rollouts.
    agent_disc.inference([sample], output_dir, cmp_per_opt, group_size, 1, num_generations, visualize=visualize, final_agg=True,
                         contrastive_ranking=contrastive_ranking)
    return


def build_dataset(args):
    '''
    Build the test dataset.

    Args:
        args (argparse.Namespace): The arguments.

    Returns:
        dataset_test: The test dataset.
    '''
    sim = SymSimulator()   
    dataset_test = []
    for file in os.listdir(args.data):
        if file.endswith('json'):
            with open(os.path.join(args.data, file), 'r') as f:
                data = json.load(f)
            for sample in data:
                # Read the PDDL file
                pddl = Path(os.path.join('../dataset', sample[0])).read_text().strip()
                init_predicates = extract_predicates(pddl, "init")
                goal_predicates = extract_predicates(pddl, "goal")
                data = {'goal': sim.describe_state(goal_predicates), 'initial_state': sim.describe_state(init_predicates), 'ref_plan': sample[1], 
                        'split': 'v1', 'num_steps': sample[2], 'id': f"{sample[0].split('/')[-1].split('.')[0]}_{sample[2]}"}
                # if os.path.exists(os.path.join(args.output_dir, data['id'] + '.json')):
                #     with open(os.path.join(args.output_dir, data['id'] + '.json'), 'r') as f:
                #         res = json.load(f)
                #     if res['num_steps'] == data['num_steps']:
                #         continue
                # data['id'] += f"_{data['num_steps']}"
                dataset_test.append(data)
    return dataset_test



def model_wrapper(model, question, extra):
    # Here, question is one element of dataset_test
    return SymPlanner(
        question,
        extra['output_dir'],
        max_steps=extra['max_steps'],
        num_rollouts=extra['num_rollouts'],
        num_generations=extra['num_generations'],
        cmp_per_opt=extra['cmp_per_opt'],
        group_size=extra['group_size'],
        beam_width=extra['beam_width'],
        enable_symbolic=extra['enable_symbolic'],
        iterative_correction=extra['iterative_correction'],
        contrastive_ranking=extra['contrastive_ranking'],
        visualize=extra['visualize'],
        model=model
    )




if __name__ == '__main__':
    if args.data is not None:
        dataset_test = build_dataset(args)
        print(f"Loaded {len(dataset_test)} samples from {args.data}.")

        # Create the extra parameters dictionary
        extra_params = {
            'output_dir': args.output_dir,
            'max_steps': args.max_steps,
            'num_rollouts': args.num_rollouts,
            'num_generations': args.num_generations,
            'cmp_per_opt': args.cmp_per_opt,
            'group_size': args.group_size,
            'beam_width': args.beam_width,
            'enable_symbolic': args.enable_symbolic,
            'iterative_correction': args.iterative_correction,
            'contrastive_ranking': args.contrastive_ranking,
            'visualize': args.visualize,
        }


        if args.batch_process:
            # Use partial to bind the extra_data argument if desired (or simply pass it via the wrapper)
            model_wrapper = partial(model_wrapper, extra=extra_params)
            # Now call batch_processing with model_wrapper. Note that batch_processing will pass in the model and question.
            batch_processing(
                model=args.model,
                fun=model_wrapper,
                question_list=dataset_test,
                num_workers=10,
                timeout_duration=3600
            )
        else:
            # We call model_wrapper for each sample
            for sample in dataset_test:
                model_wrapper(args.model, sample, extra_params)