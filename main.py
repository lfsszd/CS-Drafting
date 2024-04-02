import torch
import json
import time
from tqdm import tqdm
from pprint import pprint
from torch.multiprocessing import Pool, Process, set_start_method, current_process, freeze_support
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from csd import csd
from model import CSDraftingEncoderDecoderModel, CSDraftingMaGModel, \
CountedCSDraftingEncoderDecoderModel, CountedCSDraftingCachedEncoderDecoderModel, \
get_mag_model, DummyModel, CSDraftingDecoderModel, CountedCSDraftingDecoderModel, CountedCSDraftingCachedDecoderModel
from csd_datasets import get_test_set, format_initial_input


usable_devices = [0, 1, 2] * 2
n_process = len(usable_devices)


def start_from_config(config):
    draft_model_list = []
    for draft_name in config['draft_names']:
        draft_model_list.append(AutoModelForSeq2SeqLM.from_pretrained(draft_name))



def get_total_time_cost(draft_list, target_model, print_count=False):
    if print_count:
        print('Target model call: {}'.format(target_model.forward_count))
        for i, draft_model in enumerate(draft_list):
            try:
                print('Draft model {} call: {}'.format(draft_model.name, draft_model.forward_count))
            except:
                pass
    total_cost = 0
    for model in draft_list:
        if not isinstance(model, CSDraftingMaGModel):
            total_cost += model.calculate_time_cost()
    total_cost += target_model.calculate_time_cost()
    return total_cost


def true_work(package):
    global tokenizer, dataset_name, draft_list, target_model
    item, k_matrix, leniency = package
    k_matrix = torch.tensor(k_matrix)
    text_input = format_initial_input(item, dataset_name)
    initial_input = tokenizer(text_input, truncation=True, padding=False, return_tensors="pt")['input_ids'].to(target_model.device)
    key = str(item)
    if isinstance(target_model, CountedCSDraftingCachedEncoderDecoderModel) \
        or isinstance(target_model, CountedCSDraftingEncoderDecoderModel):
        input_ids = torch.full((1, 1),
            target_model.first_decode_id,
            dtype=torch.long)
    else:
        input_ids = initial_input
    input_ids = input_ids.to(target_model.device)
    result_ids = csd(draft_list, target_model, initial_input, input_ids, k_matrix, leniency=leniency)
    key = (k_matrix, leniency)
    return str(key), get_total_time_cost(draft_list, target_model)



def work(package):
    global draft_list
    cp = current_process()
    start_time = time.time()
    target_device = usable_devices[cp._identity[0] % n_process]
    for draft_model in draft_list:
        if draft_model.device != torch.device('cuda:{}'.format(target_device)):
            draft_model.cuda(target_device)
    if target_model.device != torch.device('cuda:{}'.format(target_device)):
        target_model.cuda(target_device)
    pbar.update(1)
    return true_work(package)



def recursive_parameter_construction(all_params_set):
    if len(all_params_set) == 0:
        return [[]]
    param1 = all_params_set[0]
    res = []
    prev = recursive_parameter_construction(all_params_set[1:])
    for p in param1:
        for remain in prev:
            res.append([p] + remain)
    return res



# The parameter here is slightly different from the one on the paper as 
# k_{12} on the paper is actually k[1, 1] + k[1, 2] here
config = {
    'draft_names': ['google/flan-t5-base', 'google/flan-t5-small'],
    'target_name': 'google/flan-t5-xxl',
    'is_decoder_only': False,
    'use_mag': True,
    'k_matrix': [[5, 14, 10], [0, 1, 10], [0, 0, 10]],
    'lenience': 2,
    'dataset': 'mmlu',
    'counter_version': 'previous_work',
    'sample': False
}


draft_list = []
target_model = None
tokenizer = None
dataset_name = config['dataset']



def aggregate_result(generation_result_for_all_params):
    total_time_cost = 0
    for time_cost in generation_result_for_all_params:
        # print(time_cost)
        total_time_cost += time_cost
    return total_time_cost / len(generation_result_for_all_params)


def aggregate_result_and_print(generation_result_for_all_params):
    total_time_cost = {}
    for key, time_cost in generation_result_for_all_params:
        if key not in total_time_cost:
            total_time_cost[key] = []
        total_time_cost[key].append(time_cost)
    res = {}
    for key, time_cost_list in total_time_cost.items():
        res[key] = sum(time_cost_list) / len(time_cost_list)
    return res


pbar = tqdm(total=100000)
_CACHE_DIR = './cache/'

if config['is_decoder_only']:
    for draft_name in config['draft_names']:
        hf_model = AutoModelForCausalLM.from_pretrained(draft_name)
        model = CountedCSDraftingDecoderModel(hf_model, sample=config['sample'], name=draft_name, vocab_size=32000, counter_version=config['counter_version'])
        draft_list.append(model)
    cache_dir = _CACHE_DIR + 'LLAMA_7B_cache.json'
    target_model = CountedCSDraftingCachedDecoderModel(DummyModel(vocab_size=32000), sample=config['sample'], name=config['target_name'], cache_dir=cache_dir, counter_version=config['counter_version'])
    tokenizer = AutoTokenizer.from_pretrained(draft_name)
    dataset_name = config['dataset']
else:
    for draft_name in config['draft_names']:
        hf_model = AutoModelForSeq2SeqLM.from_pretrained(draft_name)
        model = CountedCSDraftingEncoderDecoderModel(hf_model, sample=config['sample'], name=draft_name, counter_version=config['counter_version'])
        draft_list.append(model)
    cache_dir = _CACHE_DIR + 'FLAN_T5_xxl_cache.json'
    target_model = CountedCSDraftingCachedEncoderDecoderModel(DummyModel(), sample=config['sample'], name=config['target_name'], cache_dir=cache_dir, counter_version=config['counter_version'])
    tokenizer = AutoTokenizer.from_pretrained(config['target_name'])
    dataset_name = config['dataset']


if config['use_mag']:
    _BIGRAM_DIR = './bigram_models/'
    if 't5' in config['target_name'].lower():
        bi_gram_path = _BIGRAM_DIR + 'wiki_bigram_naive_bayers_greedy_next_token.json'
    else:
        bi_gram_path = _BIGRAM_DIR + 'wiki_bigram_naive_bayers_greedy_llama_next_token.json'
    mag_model = get_mag_model(bi_gram_path, config['is_decoder_only'])
    draft_list.append(mag_model)


if __name__ == '__main__':
    freeze_support()
    test = get_test_set(dataset_name)
    torch.multiprocessing.set_start_method('spawn', force=True)
    k_matrix = config['k_matrix']
    cur_params = [[k_matrix, config['lenience']]]
    to_go = recursive_parameter_construction([test, cur_params])
    to_go = [[p[0]] + p[1] for p in to_go]
    generation_result_for_all_params = []
    with Pool(n_process) as p:
        generation_result_for_all_params = p.map(work, to_go)
    p.join()
    all_time_cost = aggregate_result_and_print(generation_result_for_all_params)
    print('Config')
    pprint(config)
    print('All time cost: ')
    pprint(all_time_cost)



