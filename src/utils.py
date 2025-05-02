import json
from itertools import islice
import re
from concurrent.futures import ThreadPoolExecutor, wait
import traceback
from openai import OpenAI
import time
import torch
from tqdm import tqdm
import os
import sys
import re
from typing import List, Tuple

Predicate = Tuple[str, ...] 


openai_API_default = None
replicate_API_default = None
deepseek_API_default = None
togetherAI_API_default = None
deepinfra_API_default = None




def convert_element_format(key, value, convert_json=False):
    '''
    Convert the key-value pair into a string format.

    Args:
        key (str): The key.
        value (str): The value.
        convert_json (bool): Whether to convert the value into JSON format.

    Returns:
        str: The formatted key-value pair.
    '''
    if convert_json:
        return f'"{key}": {json.dumps(value)}'
    return f'"{key}": "{value}"'


def convert_dict_format(Dict, use_graph=True):
    '''
    Convert the dictionary into a string format.

    Args:
        Dict (dict): The dictionary.
        use_graph (bool): Whether to use the graph.

    Returns:
        str: The formatted dictionary.
    '''
    if Dict is None:
        return None
    op = ''
    for key in Dict:
        if (not use_graph) and ('graph' in key.lower()):
            continue
        content = str(Dict[key])
        op += f'"{key}": "{content}"\n'
    return op.strip()


def convert_list_into_dict(ls):
    '''
    Convert the list into a dictionary.
    '''
    return '{' + ', '.join(ls) + '}'


def merge_dicts(dict1, dict2):
    '''
    Merge two dictionaries.
    '''
    for key in dict1:
        dict1[key] += dict2[key]
    return dict1


def create_subset(dataset, size=None, indices=None, shuffle=False, seed=None):
    '''
    Create a subset of the dataset.
    '''
    if size == -1:
        return dataset
    
    # Define the indices for the subset
    if indices is None:
        indices = list(range(len(dataset)))[:size]

    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    
    subset = dataset.select(indices)
    return subset


def obtain_data_dict(file_ls):
    '''
    Obtain the data dictionary.
    '''
    data_dict = {'problem': [], 'label': [], 'filename': []}
    for file in file_ls:
        with open(file, 'r') as f:
            data = json.load(f)
        data_dict['problem'] += data['problems']
        data_dict['label'] += data['labels']
        data_dict['filename'] += [file.split('/')[-1].split('.')[0] + f'_idx_{i}' for i in range(len(data['problems']))]
        # print(data_dict['filename'][-2:])
    return data_dict


def obtain_data_dict2(file_ls):
    '''
    Obtain the data dictionary.
    '''
    data_dict = {'problem': [], 'solution': [], 'idx': []}
    for file in file_ls:
        with open(file, 'r') as f:
            data = json.load(f)
        data_dict['problem'].append(data['problem'])
        data_dict['solution'].append(data['solution'])
        data_dict['idx'].append(file.split('/')[-1].split('.')[0])

    return data_dict


def chunked(iterable, chunk_size):
    '''
    Chunk the iterable into chunks of size chunk_size.
    '''
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk



def compact_list(cur_list, mask=None):
    '''
    Compact the list based on the mask or remove None values.
    '''
    if mask is None:
        return [x for x in cur_list if x is not None]
    return [x for x, m in zip(cur_list, mask) if m]


def convert_mask_into_idx(mask):
    '''
    Convert the mask into indices.
    '''
    return [idx for idx, m in enumerate(mask) if m]




def remove_tag(sentence):
    '''
    Remove the tag from the sentence.
    '''
    if sentence is None:
        return None
    if ':' in sentence:
        sentence = sentence.split(':')[-1].strip()
    if sentence[0] == '"':
        sentence = sentence[1:]
    if sentence[-1] == '"':
        sentence = sentence[:-1]
    return sentence.strip()




def create_subset(dataset, size=None, indices=None, shuffle=False, seed=None):
    '''
    Create a subset of the dataset.
    '''
    if size == -1:
        return dataset
    
    # Define the indices for the subset
    if indices is None:
        indices = list(range(len(dataset)))[:size]

    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    
    subset = dataset.select(indices)
    return subset




def my_gpt_completion(model_name, messages, timeout, max_tokens=128, wait_time=0, temperature=0.7, api_key=None, local_model_ls=None,
                      source=None):
    if source == 'local':
        model, tokenizer = local_model_ls
        
        input_prompts = [messages[0]["content"].strip() + '\n```json']  # prompt the model to directly start with the json output
        tokenized_inputs = tokenizer(input_prompts, padding='longest', return_tensors="pt")
        input_tokens = tokenized_inputs["input_ids"].to("cuda")
        attention_mask = tokenized_inputs['attention_mask'].to("cuda")
        
        with torch.cuda.amp.autocast():
            generation_output = model.generate(
                input_ids=input_tokens,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_k=10,
                top_p=0.9,
                temperature=temperature,
                repetition_penalty=1.15,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=attention_mask
                )

        response = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        response = response[len(messages[0]["content"].strip()):]
        time.sleep(wait_time)
        return response

    elif source == 'replicate':
        api_key = api_key if api_key is not None else replicate_API_default
        os.environ['REPLICATE_API_TOKEN'] = api_key
        input={"prompt": messages[0]["content"].strip(), "max_tokens": max_tokens, "temperature": temperature}
        
        output = replicate.run(
            model_name,
            input=input
        )
        response = "".join(output)
        time.sleep(wait_time)
        return response

    elif source == 'deepseek':
        base_api_url = 'https://api.deepseek.com'
        api_key = api_key if api_key is not None else deepseek_API_default
    elif source == 'togetherAI':
        base_api_url = 'https://api.together.xyz/v1/'
        api_key = api_key if api_key is not None else togetherAI_API_default
    elif source == 'openai':
        api_key = api_key if api_key is not None else openai_API_default
    elif source == 'deepinfra':
        base_api_url = 'https://api.deepinfra.com/v1/openai'
        api_key = api_key if api_key is not None else deepinfra_API_default
    else:
        raise ValueError(f"Invalid source: {source}")

    client = OpenAI(api_key=api_key)
    if 'gpt' not in model_name:
        client.base_url = base_api_url

    completion =client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
    response = completion.choices[0].message.content

    time.sleep(wait_time)
    return response


def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def batch_processing(model, fun, question_list, timeout_duration=600, retry_attempts=0, num_workers=20):
    indexed_list = [data for data in question_list]
    num_workers = min(num_workers, len(question_list))
    new_list = []
    
    executor = ThreadPoolExecutor(max_workers=num_workers)
    message_chunks = list(chunk_list(indexed_list, num_workers))
        
    try:
        for chunk in tqdm(message_chunks, desc="Processing messages"):
            # Submit tasks for the current chunk
            future_to_message = {executor.submit(fun, model, message): message for message in chunk}
            for _ in range(retry_attempts + 1):  # initial attempt plus retries
                done, not_done = wait(future_to_message.keys(), timeout=timeout_duration)
                # Cancel any futures that haven't completed within the timeout
                for future in not_done:
                    future.cancel()
                for future in done:
                    try:
                        result = future.result()  # This will raise if the task failed.
                        new_list.append(result)
                    except Exception as e:
                        message = future_to_message.get(future, "Unknown message")
                        print(f"Error processing message {message}: {e}")
                        traceback.print_exc()
                        # Re-raise the exception so that it isn’t silently swallowed.
                        raise
                # If all futures completed, break out of the retry loop.
                if len(not_done) == 0:
                    break
                # Resubmit the not_done futures if you want to retry them.
                future_to_message = {executor.submit(fun, model, future_to_message[future]): future for future in not_done}
    except Exception as e:
        print(f"Error occurred in batch_processing: {e}")
        traceback.print_exc()
        raise  # Re-raise to propagate the error.
    finally:
        executor.shutdown(wait=False)
    return new_list



def prepare_task_prompt(goal, initial_state, add_comma=False):
    g = convert_element_format("Goal", goal, convert_json=True)
    init_s = convert_element_format("Initial state", initial_state, convert_json=True)
    return f'{g},\n{init_s}' if add_comma else f'{g}\n{init_s}'



def prepare_prompt_for_disciminator(goal, initial_state, context, options, futures, future_range=None):
    '''
    Prepare the prompt for the discriminator during inference.

    Args:
        problem (str): The problem.
        context (str): The context.
        options (list): The list of options.
        futures (list): The list of futures.
        future_range (list): The range of the future.

    Returns:
        prompt (str): The prompt.
    '''    
    # g is goal, init_s is init state, context is the known steps before the search steps
    merged_context = prepare_task_prompt(goal, initial_state, add_comma=True)
    if len(context) > 0:
        merged_context += f',\n{context}'
    context = merged_context.strip()
    if context.endswith(','):
        context = context[:-1]

    options = [option.replace('\\\n', '\\n') for option in options]
    option1 = options[0].strip().split('\n')
    if len(options) >= 2:
        option2 = options[1].strip().split('\n')
    if len(options) >= 3:
        option3 = options[2].strip().split('\n')

    futures = [future.replace('\\\n', '\\n') if len(future) > 0 else [] for future in futures]
    future1 = futures[0].strip().split('\n') if len(futures[0]) > 0 else []
    if len(futures) >= 2:
        future2 = futures[1].strip().split('\n') if len(futures[1]) > 0 else []
    if len(futures) >= 3:
        future3 = futures[2].strip().split('\n') if len(futures[2]) > 0 else []

    answer1 = convert_list_into_dict(option1)
    future1 = convert_list_into_dict(future1) if future_range is None else \
                convert_list_into_dict([future1[idx] for idx in future_range if idx < len(future1)])
    if len(options) >= 2:
        answer2 = convert_list_into_dict(option2)
        future2 = convert_list_into_dict(future2) if future_range is None else \
                convert_list_into_dict([future2[idx] for idx in future_range if idx < len(future2)])
    if len(options) >= 3:
        answer3 = convert_list_into_dict(option3)
        future3 = convert_list_into_dict(future3) if future_range is None else \
                convert_list_into_dict([future3[idx] for idx in future_range if idx < len(future3)])

    key = 'Search steps'

    prompt = f'{context},\n"{key}":\n{{\n"Option 1": {answer1},\n'
    if len(options) >= 2:
        prompt += f'"Option 2": {answer2},\n'
    if len(options) >= 3:
        prompt += f'"Option 3": {answer3}\n'
    prompt += '}'

    prompt_future = f',\n"Futures":\n{{\n"Future 1": {future1},\n'
    if len(options) >= 2:
        prompt_future += f'"Future 2": {future2},\n'
    if len(options) >= 3:
        prompt_future += f'"Future 3": {future3}\n'
    prompt_future += '}'
    prompt += prompt_future

    return prompt



def prepare_prompt_for_iterative_correction(prompt, err_act):
    items = prompt.split('### Output:')
    context = items[0].strip() + '\n' + items[1].strip() + '\n"Previous invalid actions": ' + convert_list_into_dict(err_act)
    return context.strip()


def extract_predicates(pddl: str, keyword: str) -> List[Predicate]:
    """
    Return a list of (functor, arg1, …) tuples that appear inside the
    section that starts with `(:keyword … )`.

    Works even when that section contains nested parentheses (e.g. (:goal (and …))).
    """
    # 1. find the *start* of the block `(:keyword`
    start_match = re.search(rf"\(\s*:{keyword}\b", pddl, re.I)
    if not start_match:
        raise ValueError(f"No :{keyword} section found")

    i = start_match.end()            # index just after "(:keyword"
    depth = 1                        # we are already inside one '('
    predicates, token = [], []
    while i < len(pddl):
        ch = pddl[i]

        if ch == '(':
            depth += 1
            token = []
        elif ch == ')':
            depth -= 1
            if token:                         # close a predicate
                predicates.append(tuple(token))
                token = []
            if depth == 0:                   # finished the whole block
                break
        elif ch.isspace():
            if token and token[-1] != '':    # finish current word
                token.append('')
        else:
            if not token:
                token.append('')
            token[-1] += ch.lower()          # accumulate word

        i += 1

    # strip possible empty strings left by whitespace splitting
    preds = [tuple(word for word in pred if word) for pred in predicates]
    return preds