from __future__ import annotations

import sys
import json
import os
import random
import itertools
from math import ceil
from collections import defaultdict
from tqdm import tqdm
from utils import *

import re
from typing import List, Set, Tuple
import ast


class APIModelConfig:
    def __init__(self, model_name='gpt-4o-mini', timeout=120, max_tokens=2048, wait_time=0, api_key=None, temperature=0.7, source='openai'):
        self.model_name = model_name if model_name else 'gpt-4o-mini'
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.wait_time = wait_time
        self.api_key = api_key
        self.temperature = temperature
        self.source = source



class Generator:
    def __init__(self, use_API=False, model=None, enable_symbolic=False, simulator=None):
        '''
        Initialize the Generator object.

        Args:
            show_prompt_only (bool): Whether to only show the prompts without generating completions.
            use_API (bool): Whether to use the API.
            API_model (str): The name of the API model to use.
            fast_mode (bool): Whether to use the fast mode.
            mixed_act_type (bool): Whether to use mixed action types.
            allow_assumption (bool): Whether to allow assumptions.

        Returns:
            None
        '''
        self.use_API = use_API
        self.enable_symbolic = enable_symbolic
        self._build_model(model)
        self.simulator = simulator


    def _build_model(self, model):
        '''
        Build the generator model.
        '''
        if self.use_API:
            self.model = APIModelConfig(model_name=model, timeout=120, max_tokens=4096, wait_time=0, api_key=None, temperature=0.7, source='openai')


    def _run_one_batch(self, samples, num_future_steps, output_dir, visualize=True, max_num_retry=10, iterative_correction=False):
        '''
        Run one batch of samples through the generator model.
        
        Args:
            samples (List[Dict]): The list of samples to process.
            force_termination (bool): Whether to force termination.
            output_dir (str): The output directory to save the results.
            visualize (bool): Whether to visualize the results.

        Returns:
            None
        '''
        for sample in samples:
            prompts = []
            rolloutID = []
            for rollout_id in sample['rollout']:
                rollout = sample['rollout'][rollout_id]
                if rollout['active'] and '"Goal Achieved":' not in sample['rollout'][rollout_id]['prompt']:
                    prompts.extend([rollout['prompt']] * rollout['num_gen'])
                    rolloutID.extend([rollout_id] * rollout['num_gen']) 
            
            results = []
            futures = []
            with open(f'../prompt/prompt_generation.txt', 'r') as file:
                instruction = file.read()

            if iterative_correction:
                with open(f'../prompt/prompt_iterative_correction.txt', 'r') as file:
                    instruction_IC = file.read()
                err_act = []

            for (idx_prompt, prompt) in enumerate(prompts):
                cnt_gen = 0
                future = 'ukn'
                # retry if generation fails
                while cnt_gen < max_num_retry:
                    input = f'{instruction}\n\n\nTest:\n{prompt}'
                    if future is None:
                        input = f'{instruction_IC}\n\n\nTest:\n{prepare_prompt_for_iterative_correction(prompt, err_act)}\n\n### Output:\n'
                    messages = [{"role": "user", "content": input}]
                    response = my_gpt_completion(self.model.model_name, messages, self.model.timeout, max_tokens=self.model.max_tokens, wait_time=self.model.wait_time, api_key=self.model.api_key, temperature=self.model.temperature, source=self.model.source)
                    cnt_gen += 1
                    
                    # print(response)
                    # print('-' *20)

                    # post-process the response
                    if '### Output:' in response:
                        response = response.split('### Output:')[1]
                    response = response.strip().split('\n')
                    response = [step.strip() for step in response if len(step.strip()) > 0]
                    response = [step.replace('**', '"') if step.startswith('**') else step for step in response]
                    response = [step for step in response if step.startswith('"')]
                    response = [step for step in response if step.split(':')[0] not in prompt.split("### Output:")[-1]]

                    if len(response) == 0:
                        continue
                
                    result = response[0]
                    stepName = result.split(':')[0].strip()
                    rollout_id = rolloutID[idx_prompt]
                    if 'action' in stepName.lower():
                        future = '\n'.join(response[1 : 1 + num_future_steps]) if len(response) > 1 else ''
                        action = self.simulator.parse_action(result.split(':')[1].strip()[1:-1])
                        sample['rollout'][rollout_id]['sym_action'] = action
                        # apply the action to the state (we need to check the preconditions)
                        try:
                            state = set(tuple(item) for item in sample['rollout'][rollout_id]['sym_state'])
                            next_state = self.simulator.apply_action(state, action)
                            sample['rollout'][rollout_id]['sym_state'] = list(next_state)
                            if self.enable_symbolic:
                                future = self.simulator.describe_state(next_state)
                                NextstepName = f'"State {stepName.split(" ")[-1]}'
                                future = f'{NextstepName}: "{future}"'
                        except:
                            if self.enable_symbolic:
                                future = None
                                if iterative_correction:
                                    err_act.append(result)
                            else:
                                sample['rollout'][rollout_id]['sym_state'] = None

                    else:
                        # 'state' or 'goal achieved'
                        if 'state' in stepName.lower() and self.enable_symbolic:
                            try:
                                state = set(tuple(item) for item in sample['rollout'][rollout_id]['sym_state'])
                                action = sample['rollout'][rollout_id]['sym_action']
                                next_state = self.simulator.apply_action(state, action)
                            except:
                                next_state = self.simulator.parse_state(result)
                            sample['rollout'][rollout_id]['sym_state'] = list(next_state)
                            result = self.simulator.describe_state(next_state)
                            result = f'{stepName}: "{result}"'
                        future = ''

                    if future is not None:
                        break
                
                if future is None:
                    future = f'"State {stepName.split(" ")[-1]}: "invalid action"'
                results.append(result)
                futures.append(future)
                
            if visualize:
                print('Generation results:')
            for idx_prompt in range(len(prompts)):
                if visualize:
                    print('Prompt:')
                    print(prompts[idx_prompt])
                    print('-------------------')
                    print('Result:')
                    print(results[idx_prompt])
                    print('-------------------')
                    print('Future:')
                    print(futures[idx_prompt])
                    print('-------------------')
                
                rollout_id = rolloutID[idx_prompt]
                
                sample['rollout'][rollout_id]['responses'].append(results[idx_prompt])
                sample['rollout'][rollout_id]['futures'].append(futures[idx_prompt])

            if visualize:
                print('===================')

            with open(f'{output_dir}/{sample["id"]}.json', 'w') as f:
                json.dump(sample, f)
            

    def _rollout_init(self, prompt, state=None):
        '''
        Initialize the rollout.

        Args:
            prompt (str): The prompt to initialize the rollout with.

        Returns:
            rollout (Dict): The initialized rollout.
        '''
        rollout = {}
        rollout['active'] = True
        rollout['prompt'] = prompt
        rollout['num_gen'] = 1
        rollout['responses'] = []
        rollout['futures'] = []
        rollout['state_search_history'] = []
        rollout['sym_state'] = state

        return rollout


    def inference(self, dataset, output_dir, num_rollouts, num_generations, num_future_steps, beam_width, visualize=False, 
                  iterative_correction=False):
        '''
        Perform inference on the given dataset.

        Args:
            dataset (List[Dict]): The dataset to perform inference on.
            output_dir (str): The output directory to save the results.
            visualize (bool): Whether to visualize the results.

        Returns:
            flag_finish (bool): Whether the inference is finished.
        '''
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        flag_finish = False
        
        num_processed_samples = 0
        samples = []
        # for sample in tqdm(dataset, total=len(dataset)):
        for sample in dataset:
            if os.path.exists(f'{output_dir}/{sample["id"]}.json'):
                with open(f'{output_dir}/{sample["id"]}.json', 'r') as f:
                    sample = json.load(f)
                
                if 'flag_correct' in sample:
                    continue

            if 'rollout' not in sample:
                sample['sym_goal'] = list(self.simulator.parse_state(sample['goal'])) 
                sample['rollout'] = {}

            if len(sample['rollout']) == 0:
                sample['flag_done'] = False
                goal = sample['goal']
                initial_state = sample['initial_state']
                prompt = f'### Input:\n"Goal": "{goal}"\n"Initial state": "{initial_state}"\n\n### Output:\n'
                state = self.simulator.parse_state(initial_state)
                state = list(state)
                sample['rollout']['0'] = self._rollout_init(prompt, state=state)

                
            num_active_rollout = 0
            for rollout_id in range(len(sample['rollout'])):
                rollout_id = str(rollout_id)
                if not sample['rollout'][rollout_id]['active']:
                    continue
                sample['rollout'][rollout_id]['prompt'] = f"{sample['rollout'][rollout_id]['prompt'].strip()}\n"
                
                if '"Goal Achieved":' in sample['rollout'][rollout_id]['prompt']:
                    continue
                
                # determine the number of generations
                lastStep = sample['rollout'][rollout_id]['prompt'].strip().split('\n')[-1].strip()
                lastStepName = lastStep.split(':')[0].strip()

                sample['rollout'][rollout_id]['num_gen'] = 1
                if 'action' not in lastStepName.lower():
                    # action search
                    for _ in range(num_generations-1):
                        next_rollout_id = len(sample['rollout'])
                        if next_rollout_id < num_rollouts:
                            state = sample['rollout'][rollout_id]['sym_state']
                            sample['rollout'][str(next_rollout_id)] = self._rollout_init(sample['rollout'][rollout_id]['prompt'], 
                                                                                         state=state)
                num_active_rollout += 1

            # if no active rollout or more than beam_width rollouts (we should first perform discrimination), skip the sample
            if num_active_rollout == 0 or num_active_rollout > beam_width:
                continue

            samples.append(sample)
            num_processed_samples += 1


        if len(samples) > 0:
            self._run_one_batch(samples, num_future_steps, output_dir, visualize=visualize, iterative_correction=iterative_correction)
        
        if num_processed_samples == 0:
            flag_finish = True
 
        return flag_finish




class Discriminator():
    def __init__(self, use_API=False, model=None, simulator=None):
        '''
        Initialize the Discriminator object.

        Args:
            use_meta_knwoledge (bool): Whether to use meta-knowledge.
            show_prompt_only (bool): Whether to only show the prompts without generating completions.
            use_API (bool): Whether to use the API.
            API_model (str): The name of the API model to use.
            fast_mode (bool): Whether to use the fast mode.
            mixed_act_type (bool): Whether to use mixed action types.
            allow_assumption (bool): Whether to allow assumptions.

        Returns:
            None
        '''
        self.use_API = use_API
        self._build_model(model)
        self.simulator = simulator


    def _build_model(self, model):
        '''
        Build the discriminator model.
        '''
        if self.use_API:
            self.model = APIModelConfig(model_name=model, timeout=120, max_tokens=4096, wait_time=0, api_key=None, temperature=0, source='openai')
            return


    def _schedule_all_comparisons(self, options, group_size=3):
        """
        Schedules all possible comparisons with up to 3 options.
        Each comparison is between 2 or 3 options.

        Args:
            options (List[Option]): List of options to compare.

        Returns:
            List[List[Option]]: List of comparisons.
        """
        comparisons = []
        
        # Schedule all possible 2-option comparisons
        comparisons.extend(list(itertools.combinations(options, 2)))
        
        if group_size > 2:
            # Schedule all possible 3-option comparisons
            comparisons.extend(list(itertools.combinations(options, 3)))
        
        return [list(comparison) for comparison in comparisons]


    def _schedule_random_comparisons(self, options, cmp_per_opt=3, group_size=3):
        """
        Schedules a random subset of comparisons ensuring each option participates
        in approximately 'cmp_per_opt' comparisons.
        Useful for larger N to limit the number of comparisons.

        Args:
            options (List[Option]): List of options to compare.
            cmp_per_opt (int): Number of comparisons each option should participate in.
            group_size (int): Number of options in each comparison.

        Returns:
            List[List[Option]]: List of comparisons.
        """
        N = len(options)
        target_total_comparisons = ceil((cmp_per_opt * N) / group_size)
        
        if len(options) < group_size:
            # Generate combinations of all available elements
            all_comparisons = list(itertools.combinations(options, len(options)))
        else:
            # Generate all possible group_size-opt comparisons
            all_comparisons = list(itertools.combinations(options, group_size))
        random.shuffle(all_comparisons)
        
        comparisons = []
        participation_count = defaultdict(int)
        
        for comparison in all_comparisons:
            if all(participation_count[option.id] < cmp_per_opt for option in comparison):
                comparisons.append(list(comparison))
                for option in comparison:
                    participation_count[option.id] += 1
                if len(comparisons) >= target_total_comparisons:
                    break
        
        return comparisons


    def _rank_options(self, options):
        """
        Ranks options based on their scores.
        Returns the list of options sorted by score descending.

        Args:
            options (List[Option]): List of options to rank.

        Returns:
            List[Option]: List of options sorted by score descending.
        """
        return sorted(options, key=lambda x: x.score, reverse=True)


    def _post_process(self, data, selected_rollout_ids, selected_option, disc_data, output_dir, filename=None, mode='action_plan', 
                      final_agg=False):
        '''
        Post-process the generated results.

        Args:
            data (Dict): The data dictionary.
            disc_data (List): The discrimination data.
            output_dir (str): The output directory to save the results.
            filename (str): The filename to save the results.

        Returns:
            data (Dict): The updated data dictionary.
        '''
        if mode == 'action_plan':
            for rollout_id in data['rollout']:
                cur_rollout = data['rollout'][rollout_id]
                if rollout_id not in selected_rollout_ids:
                    cur_rollout['active'] = False
                    continue
                
                if not final_agg:
                    if len(cur_rollout['responses']) > 0:
                        cur_rollout['prompt'] = f"{cur_rollout['prompt'].strip()}\n{cur_rollout['responses'][0]}"
                        if isinstance(cur_rollout['futures'][0], str):
                            cur_rollout['prompt'] += f"\n{cur_rollout['futures'][0]}"
                    cur_rollout['responses'] = []
                    cur_rollout['futures'] = []

        elif mode == 'state_pred':
            rollout_id = selected_rollout_ids[0]
            cur_rollout = data['rollout'][rollout_id]
            cur_rollout['prompt'] = f"{cur_rollout['prompt'].strip()}\n{selected_option}"
            cur_rollout['responses'] = []
            cur_rollout['futures'] = []
            cur_rollout['state_search_history'].append(disc_data) 

        if filename is not None:
            with open(f'{output_dir}/{filename}', 'w') as f:
                json.dump(data, f)

        return data


    def _reshape_res(self, prompts_ls, result):
        '''
        Reshape the results to the original list.

        Args:
            prompts_ls (List[List[str]]): The list of prompts.
            result (List[str]): The list of results.

        Returns:
            original_dist (List[List[str]]): The reshaped list of results.
        '''
        original_dist = []
        index = 0
        for sublist in prompts_ls:
            length = len(sublist)
            original_dist.append(result[index:index + length])
            index += length
        return original_dist


    def _extract_plan(self, result):
        '''
        Extract the plan from the result.

        Args:
            result (str): The result string.

        Returns:
            plan (str): The extracted plan.
        '''
        if '### Output:' in result:
            result = result.split('### Output:')[1]
        result = result.strip().split('\n')
        result = [step.split(':')[1].strip()[1:-1] for step in result if len(step.strip()) > 0 and 'action' in step.split(':')[0].lower()]
        return '\n'.join(result)


    def _run_one_batch(self, output_dir, samples, prompts_ls, options_ls, comparisons_ls, filenames, beam_width, visualize, final_agg, 
                       mode='action_plan', contrastive_ranking=False):
        '''
        Run one batch of samples through the discriminator model.

        Args:
            output_dir (str): The output directory to save the results.
            samples (List[Dict]): The list of samples to process.
            prompts_ls (List[List[str]]): The list of prompts.
            options_ls (List[List[Option]]): The list of options.
            comparisons_ls (List[List[List[Option]]]): The list of comparisons.
            filenames (List[str]): The list of filenames.
            visualize (bool): Whether to visualize the results.

        Returns:
            None
        '''
        flat_prompts_ls = [item for sublist in prompts_ls for item in sublist]

        if self.use_API:
            flat_results = []
            for prompt in flat_prompts_ls:
                # print(prompt)
                # print('-------------------')
                messages = [{"role": "user", "content": prompt}]
                response = my_gpt_completion(self.model.model_name, messages, self.model.timeout, max_tokens=self.model.max_tokens if contrastive_ranking else 1, wait_time=self.model.wait_time, api_key=self.model.api_key, temperature=self.model.temperature, source=self.model.source)
                # print(response)
                # print('===================')
                flat_results.append(response)

        flat_results = [f'{prompt.strip()}\n{res}' for prompt, res in zip(flat_prompts_ls, flat_results)]
        recovered_res = self._reshape_res(prompts_ls, flat_results)

        for i in range(len(samples)):
            disc_res = recovered_res[i]
            comparisons = comparisons_ls[i]
            
            if visualize:
                print('Discrimination result:')
            for idx_res in range(len(disc_res)):
                cur_res = disc_res[idx_res]
                if visualize:
                    print(cur_res)
                    print('------------------------------------------')
                if contrastive_ranking:
                    if '"Conclusion":' in cur_res:
                        cur_res = cur_res.split('"Conclusion":')[-1].strip()
                    cur_res = cur_res.lower()
                    winner = None
                    if 'option 1' in cur_res:
                        winner = comparisons[idx_res][0]
                    elif 'option 2' in cur_res and len(comparisons[idx_res]) > 1:
                        winner = comparisons[idx_res][1]
                    elif 'option 3' in cur_res and len(comparisons[idx_res]) > 2:
                        winner = comparisons[idx_res][2]
                    
                    if winner is not None:
                        winner.score += 1
                else:
                    score = cur_res.split('### Rating:')[-1].strip()
                    if not score.isdigit():
                        score = 5
                    # print(score)
                    comparisons[idx_res][0].score = int(score)

            ranked_options = self._rank_options(options_ls[i])
            final_winners = ranked_options[:beam_width]

            if final_agg:
                samples[i]['flag_done'] = True
                samples[i]['generated_plan'] = self._extract_plan(final_winners[0].description)
                if 'sym_state' in samples[i]['rollout'][final_winners[0].rollout_id]:
                    final_symState = samples[i]['rollout'][final_winners[0].rollout_id]['sym_state']
                else:
                    final_symState = final_winners[0].description.split('"State')[-1].split('"Goal Achieved"')[0].split(':')[1].strip()[1:-1]
                    final_symState = self.simulator.parse_state(final_symState)
                try:
                    samples[i]['correct'] = all(predicate in final_symState for predicate in samples[i]['sym_goal'])
                except:
                    samples[i]['correct'] = False

            disc_data = [f'{option.description}\n\n{option.future}' for option in options_ls[i]] if mode == 'state_pred' else None
            self._post_process(samples[i], [winner.rollout_id for winner in final_winners], final_winners[0].description, disc_data, output_dir, filenames[i], mode=mode, final_agg=final_agg)
        if visualize:
            print('=============================================')


    def inference(self, dataset, output_dir, cmp_per_opt, group_size, beam_width, num_generations, deduplicate=False, visualize=False, 
                  final_agg=False, contrastive_ranking=False):
        '''
        Perform inference on the given dataset.

        Args:
            output_dir (str): The output directory to save the results.
            cmp_per_opt (int): The number of comparisons per option.
            group_size (int): The number of options in each comparison.
            deduplicate (bool): Whether to deduplicate the responses.
            visualize (bool): Whether to visualize the results.
            final_agg (bool): Whether to perform final aggregation.

        Returns:
            None
        '''        
        if self.use_API:
            with open('../prompt/prompt_contrastive_ranking.txt', 'r') as file:
                instruction_CR = file.read().strip()
            
            with open('../prompt/prompt_rating.txt', 'r') as file:
                instruction_rating = file.read().strip()

        samples = []
        prompts_ls = []
        options_ls = []
        comparisons_ls = []
        filenames = []

        for data in dataset:
            filename = f'{data["id"]}.json'
            
            if not os.path.exists(f'{output_dir}/{filename}'):
                continue
            with open(f'{output_dir}/{filename}', 'r') as f:
                sample = json.load(f)
            
            if sample['flag_done']:
                continue

            goal, initial_state = sample['goal'], sample['initial_state']
            
            # Either we compare different rollouts (planning or aggregating) or we compare different responses from the same rollout (state prediction)
            context = ''
            responses = []
            futures = []
            rollout_ids = []
            for rollout_id in sample['rollout']:
                cur_rollout = sample['rollout'][rollout_id]
                if not cur_rollout['active']:
                    continue
                if not final_agg:
                    cur_context = cur_rollout['prompt'].split('### Output:')[1].strip()
                    response = cur_rollout['responses'][0] if len(cur_rollout['responses']) > 0 else ''
                    response = f'{cur_context},\n{response}' if len(response) > 0 else cur_context
                    future = cur_rollout['futures'][0] if len(cur_rollout['futures']) > 0 else ''
                    responses.append(response)
                    futures.append(future)
                else:
                    responses.append(cur_rollout['prompt'].split('### Output:')[1].strip())
                    futures.append([])
                rollout_ids.append(rollout_id)

            if not final_agg:
                if deduplicate:
                    # Initialize a dictionary to maintain unique responses and corresponding futures
                    unique_responses = {}
                    for response, future, rollout_id in zip(responses, futures, rollout_ids):
                        if response not in unique_responses:
                            unique_responses[response] = (future, rollout_id)

                    # Extract the deduplicated responses and their corresponding futures
                    responses = list(unique_responses.keys())
                    futures = [unique_responses[response][0] for response in responses]
                    rollout_ids = [unique_responses[response][1] for response in responses]
                
            responses = responses[:beam_width*num_generations]
            futures = futures[:beam_width*num_generations]
            rollout_ids = rollout_ids[:beam_width*num_generations] 

            options = [Option(i, responses[i], futures[i], rollout_ids[i]) for i in range(len(responses))]
            random.shuffle(options)
            num_options = len(responses)
            
            if num_options == 0:
                # Empty invalid responses and futures
                if not final_agg:
                    for rollout_id in sample['rollout']:
                        cur_rollout = sample['rollout'][rollout_id]
                        if not cur_rollout['active']:
                            continue
                        cur_rollout['responses'] = []
                        cur_rollout['futures'] = []
                continue
            elif num_options <= beam_width:
                if final_agg:
                    sample['flag_done'] = True
                    sample['generated_plan'] = self._extract_plan(responses[0])
                    if 'sym_state' in sample['rollout'][rollout_ids[0]]:
                        final_symState = sample['rollout'][rollout_ids[0]]['sym_state']
                    else:
                        final_symState = responses[0].split('"State')[-1].split('"Goal Achieved"')[0].split(':')[1].strip()[1:-1]
                        final_symState = self.simulator.parse_state(final_symState)
                    try:
                        sample['correct'] = all(predicate in final_symState for predicate in sample['sym_goal'])
                    except:
                        sample['correct'] = False

                self._post_process(sample, rollout_ids, None, None, output_dir, filename=filename)
                continue
            
            if contrastive_ranking:
                comparisons = self._schedule_random_comparisons(options, cmp_per_opt, group_size)
                if self.use_API:
                    prompts = [f'{instruction_CR}\n\n\nTest:\n ### Input: \n{prepare_prompt_for_disciminator(goal, initial_state, context, [option.description for option in cur_batch], [option.future for option in cur_batch])}\n\n ### Output: \n' for cur_batch in comparisons]
            else:
                input = prepare_task_prompt(goal, initial_state)
                comparisons = [[option] for option in options]
                prompts = []
                for cur_batch in comparisons:
                    prompt = f'{instruction_rating}\n\n\n## Test:\n### Input:\n{input}\n\n### Output:\n{cur_batch[0].description}'
                    if isinstance(cur_batch[0].future, str):
                        prompt += f'\n{cur_batch[0].future}'
                    prompt += '\n\n### Rating:\n'
                    prompts.append(prompt)

            samples.append(sample)
            prompts_ls.append(prompts)
            options_ls.append(options)
            comparisons_ls.append(comparisons)
            filenames.append(filename)
                
        if len(samples) > 0:
            self._run_one_batch(output_dir, samples, prompts_ls, options_ls, comparisons_ls, filenames, beam_width, visualize, final_agg,
                                contrastive_ranking=contrastive_ranking)



class Option:
    '''
    Class to represent an option in the comparison task for the discriminator model.
    '''
    def __init__(self, option_id: int, description: str, future: str, rollout_id: str):
        '''
        Initialize the Option object.
        
        Args:
            option_id (int): The ID of the option.
            description (str): The description of the option.
            future (str): The future of the option.

        Returns:
            None
        '''
        self.id = option_id
        self.description = description
        self.future = future
        self.rollout_id = rollout_id
        self.score = 0



class SymSimulator:
    """
    Symbolic simulator for the classic Blocks‐World domain.
    """
    # Type aliases
    Predicate = Tuple[str, ...]         # e.g. ('on', 'a', 'b')
    State     = Set[Predicate]
    Action    = Predicate

    # --------------------------------------------------------------------
    # Natural‑language templates
    # --------------------------------------------------------------------
    PREDICATE_TEMPLATES = {
        "on"       : "the {0} block is on top of the {1} block",
        "ontable"  : "the {0} block is on the table",
        "clear"    : "the {0} block is clear",
        "holding"  : "the hand is holding the {0} block",
        "handempty": "the hand is empty",
    }

    ACTION_TEMPLATES = {
        "pickup" : "pick up the {0} block",
        "putdown": "put down the {0} block",
        "stack"  : "stack the {0} block on top of the {1} block",
        "unstack": "unstack the {0} block from on top of the {1} block",
    }

    # ---------------------------------------------------------------------------
    # Regular expressions that correspond 1‑to‑1 to the English templates
    # ---------------------------------------------------------------------------
    PATTERNS = [
        # the red block is on top of the blue block   --> ('on','red','blue')
        (
            re.compile(r"^the (\w+) block is on top of the (\w+) block$", re.I),
            lambda m: ("on", m.group(1), m.group(2)),
        ),
        # the red block is on the table              --> ('ontable','red')
        (
            re.compile(r"^the (\w+) block is on the table$", re.I),
            lambda m: ("ontable", m.group(1)),
        ),
        # the red block is clear                     --> ('clear','red')
        (
            re.compile(r"^the (\w+) block is clear$", re.I),
            lambda m: ("clear", m.group(1)),
        ),
        # the hand is holding the red block          --> ('holding','red')
        (
            re.compile(r"^the hand is holding the (\w+) block$", re.I),
            lambda m: ("holding", m.group(1)),
        ),
        # the hand is empty                          --> ('handempty',)
        (
            re.compile(r"^the hand is empty$", re.I),
            lambda _: ("handempty",),
        ),
    ]


    def __init__(self, use_API: bool = False, parse_model: str | None = None) -> None:
        self.use_API = bool(use_API)
        self._build_parser(parse_model)


    def _build_parser(self, model: str | None) -> None:
        """
        Initialise the LLM model wrapper if the user asked for it.
        """
        if self.use_API:
            self.model = APIModelConfig(
                model_name=model or "gpt‑4.1",
                timeout=120,
                max_tokens=2048,
                wait_time=0,
                api_key=None,
                temperature=0,
                source="openai",
            )


    def apply_action(self, state: State, action: Action) -> State:
        """
        Return the successor state that results from applying ACTION to STATE.
        Preconditions are checked; if violated a ValueError is raised.
        """
        op = action[0].lower()
        new_state: SymSimulator.State = set(state)          # copy

        # ---------- PICK‑UP (x) ----------
        if op == "pickup":
            _, x = action
            pre = {("ontable", x), ("clear", x), ("handempty",)}
            if not pre <= state:
                raise ValueError(f"Pre‑conditions for {action} not satisfied")
            new_state.difference_update(pre)
            new_state.add(("holding", x))
            return new_state

        # ---------- UNSTACK (x,y) ----------
        if op == "unstack":
            _, x, y = action
            pre = {("on", x, y), ("clear", x), ("handempty",)}
            if not pre <= state:
                raise ValueError(f"Pre‑conditions for {action} not satisfied")
            new_state.difference_update(pre)
            new_state.update({("holding", x), ("clear", y)})
            return new_state

        # ---------- PUT‑DOWN (x) ----------
        if op == "putdown":
            _, x = action
            pre = {("holding", x)}
            if not pre <= state:
                raise ValueError(f"Pre‑conditions for {action} not satisfied")
            new_state.difference_update(pre)
            new_state.update({("ontable", x), ("clear", x), ("handempty",)})
            return new_state

        # ---------- STACK (x,y) ----------
        if op == "stack":
            _, x, y = action
            pre = {("holding", x), ("clear", y)}
            if not pre <= state:
                raise ValueError(f"Pre‑conditions for {action} not satisfied")
            new_state.difference_update(pre)
            new_state.update({("on", x, y), ("clear", x), ("handempty",)})
            return new_state

        # ---------- UNKNOWN ----------
        raise ValueError(f"Unknown action: {action}")


    def parse_action(self, action: str) -> Action:
        """
        Use an LLM to translate a natural‑language action description
        into its PDDL form.
        Requires `use_API=True` and a working `APIModelConfig`.
        """
        if not self.use_API:
            raise RuntimeError("parse_action requires use_API=True")

        # Read the prompt that instructs the LLM how to behave
        with open("../prompt/prompt_parse_action.txt", "r") as fh:
            instruction = fh.read()

        prompt = f"{instruction}\n\n\nTest:\n### Input:\n{action}\n\n### Output:\n"
        messages = [{"role": "user", "content": prompt}]

        response = my_gpt_completion(
            self.model.model_name,
            messages,
            self.model.timeout,
            max_tokens=self.model.max_tokens,
            wait_time=self.model.wait_time,
            api_key=self.model.api_key,
            temperature=self.model.temperature,
            source=self.model.source,
        )

        # Post‑process helper prompt format if necessary
        if "### Output:" in response:
            response = response.split("### Output:", 1)[1]
        response = response.strip().split("\n")[0]
        return tuple(ast.literal_eval(response))


    def parse_state(self, text: str) -> State:
        """
        Convert a Blocks‑World English description into a State set.

        The input must use ONLY the sentences produced by the templates:

            • the X block is clear
            • the X block is on the table
            • the X block is on top of the Y block
            • the hand is holding the X block
            • the hand is empty

        Phrases may be separated by commas and a single final 'and'.
        Capitalisation and optional trailing full stop are ignored.
        """
        # Normalise separators: “…, A and B”  → “…, A, B”
        text = text.strip().rstrip(".")
        text = text.replace(" and ", ", ")

        # Split on commas, discard empty fragments
        phrases = [p.strip() for p in text.split(",") if p.strip()]

        state = set()
        for phrase in phrases:
            for pattern, builder in SymSimulator.PATTERNS:
                m = pattern.match(phrase)
                if m:
                    state.add(builder(m))
                    break
            else:  # no break ⇒ nothing matched
                raise ValueError(f"Unrecognised fragment: {phrase!r}")

        return state


    def describe_state(self, state: State) -> str:
        """
        Convert a collection of predicates into a list of English sentences.
        Sentences are sorted for deterministic output.
        """
        lines = []
        for pred in sorted(state, key=str):
            functor, *args = pred
            template = self.PREDICATE_TEMPLATES.get(functor)
            if template:
                lines.append(template.format(*args))
            else:
                # Unknown predicate: just show the tuple textually
                lines.append(str(pred))
        return ', '.join(lines)


    def describe_action(self, action: Action) -> str:
        """
        Turn an operator tuple such as ('stack','blue','orange')
        into a single English sentence.
        """
        name, *args = action
        template = self.ACTION_TEMPLATES.get(name.lower())
        if template is None:
            raise ValueError(f"Unknown action: {action}")
        return template.format(*args)