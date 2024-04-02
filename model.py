import json
import torch
import time
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from mag import draft_sample_k_bn_gram


TIME_COST = {
    'model_parameters': {
        "xxl": 11300,
        "small": 60, 
        "base": 220, 
        "large": 770,
        "xl": 3000,
        'JackFram/llama-68m': 68,
        'JackFram/llama-160m': 160,
        '7b': 70000,
        'llama': 70000,
    },
    'previous_work': {
        "xxl": 1,
        "small": 0.02, 
        "base": 0.04, 
        "large": 0.11,
    }
}


def crop_past_key_values(past_key_values, maximum_length):
    new_past = []
    for idx in range(len(past_key_values)):
        new_past.append(
            (
                past_key_values[idx][0][:, :, :maximum_length, :],
                past_key_values[idx][1][:, :, :maximum_length, :],
            )
        )
    past_key_values = tuple(new_past)
    return past_key_values



_T5_DECODER_START_TOKEN_ID = 0
def tokens_to_new_key(tokens):
    return '_'.join([str(token) for token in tokens.tolist()[0]])


def key_to_tokens(key):
    return torch.tensor([int(token) for token in key.split('_')]).unsqueeze(0)


def load_cache_model(cache_dir):
    with open(cache_dir) as f:
        target_cache = json.load(f)
    return target_cache



def get_mag_model(bi_gram_path, is_decoder_only=True):
    with open(bi_gram_path) as f:
        bi_gram_model = json.load(f)
    bi_gram_model = torch.tensor(bi_gram_model)
    res = CSDraftingMaGModel(bi_gram_model, name='mag')
    if is_decoder_only:
        res.vocab_size = 32000
    return res


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=32128):
        super().__init__()
        self.device = torch.device('cpu')
        self.vocab_size = vocab_size
    def cuda(self, device):
        self.device = torch.device(device)
    def to(self, device):
        self.device = torch.device(device)
    def cpu(self):
        self.device = torch.device('cpu')


class CSDraftingModel(torch.nn.Module):
    def __init__(self, model, sample=False, name='', vocab_size=32128):
        super().__init__()
        self.model = model
        self.sample = sample
        try:
            self.device = model.device
        except:
            self.device = torch.device('cpu')
        self.name = name
        try:
            self.vocab_size = model.config.vocab_size
        except:
            self.vocab_size = vocab_size

    def cuda(self, device):
        self.model.cuda(device)
        self.device = self.model.device
        # return self
    def to(self, device):
        self.model.to(device)
        self.device = self.model.device
    def cpu(self):
        self.model.cpu()
        self.device = self.model.device


class CSDraftingMaGModel(CSDraftingModel):
    def propose(self, initial_input, input_ids, k):
        initial_input = initial_input.to(self.model.device)
        input_ids = input_ids.to(self.model.device)
        res = draft_sample_k_bn_gram(self.model, initial_input, input_ids, k)
        return res
    def calculate_time_cost(self):
        return 0
    def cuda(self, device):
        self.model = self.model.cuda(device)
        self.device = self.model.device
    def to(self, device):
        self.model = self.model.to(device)
        self.device = self.model.device
    def cpu(self):
        self.model.cpu()
        self.device = self.model.device




class CSDraftingEncoderDecoderModel(CSDraftingModel):
    def __init__(self, model, sample=False, name='', vocab_size=32128):
        super().__init__(model, sample, name, vocab_size=vocab_size)
        self.first_decode_id = _T5_DECODER_START_TOKEN_ID
    def review(self, initial_input, input_ids, probs, review_index, leniency=1):
        target_logits = self.model(initial_input, decoder_input_ids=input_ids).logits
        prefix_input_ids = input_ids[:, :review_index]
        target_probs = torch.nn.functional.softmax(target_logits, dim=-1)
        probs_new = probs[:, review_index - 1:, :]
        if not self.sample:
            target_ids = torch.argmax(target_probs, dim=-1)
            first_decode_id = torch.full((1, 1),
                self.first_decode_id,
                dtype=torch.long,
                device=self.model.device)
            target_ids = torch.concat([first_decode_id, target_ids], dim=-1)
            target_ids = target_ids[:, review_index:]
            target_probs = target_probs[:, review_index - 1:, :]
            input_ids = input_ids[:, review_index:]
            target_ids = target_ids.to(input_ids.device)
            match_ct = 0
            for i in range(probs_new.shape[1]):
                if target_ids[0, i] == input_ids[0, i]:
                    match_ct += 1
                    continue
                else:
                    if leniency > 1 and target_probs[0, i, input_ids[0, i]] > probs_new[0, i, input_ids[0, i]] / leniency:
                        match_ct += 1
                        continue
                    else:
                        i = i - 1
                        break
            input_ids = torch.cat([input_ids[:, :i + 1], target_ids[:, i + 1:i + 2]], dim=-1)
            id_res = torch.concat([prefix_input_ids, input_ids], dim=-1)
            prob_res = torch.concat([probs[:, :review_index, :], target_probs[:, :i + 1, :]], dim=-2)
            return id_res, prob_res





class CountedCSDraftingEncoderDecoderModel(CSDraftingEncoderDecoderModel):
    def __init__(self, model, sample=False, name='', counter_version='model_parameters'):
        super().__init__(model, sample, name)
        self.forward_count = 0
        self.counter_version = counter_version
        time_cost_dict = TIME_COST[counter_version]
        for model_abbr in time_cost_dict:
            if model_abbr in name:
                self.time_cost = time_cost_dict[model_abbr]
                break
    def review(self, initial_input, input_ids, probs, review_index, leniency=1):
        self.forward_count += 1
        res = super().review(initial_input, input_ids, probs, review_index, leniency)
        return res
    def calculate_time_cost(self):
        res = self.forward_count * self.time_cost
        self.forward_count = 0
        return res  





def torch_index(t, value):
    temp = t == value
    match = temp.nonzero(as_tuple=False)[0]
    res = match[1]
    return res


def torch_index(t, value):
    all_start = time.time()
    start = time.time()
    temp = t == value
    match = temp.nonzero(as_tuple=False)[0]
    res = match[1]
    return res



class CSDraftingDecoderModel(CSDraftingModel):
    def __init__(self, model, sample=False, name='', vocab_size=32000):
        super().__init__(model, sample, name, vocab_size=vocab_size)
    def propose(self, initial_input, input_ids, k):
        input_ids = input_ids.to(self.model.device)
        for i in range(k):
            res = self.model(input_ids, use_cache=False)
            new_token = torch.argmax(res.logits[0, -1, :])
            input_ids = torch.cat([input_ids, new_token.unsqueeze(0).unsqueeze(0)], dim=1)
        return input_ids
    def review(self, initial_input, input_ids, probs, review_index, leniency=1):
        start = time.time()
        target_logits = self.model(input_ids).logits
        start = time.time()
        prefix_input_ids = input_ids[:, :review_index]
        target_probs = torch.nn.functional.softmax(target_logits, dim=-1)
        probs_new = probs[:, review_index - 1:, :]
        if not self.sample:
            target_ids = torch.argmax(target_probs, dim=-1)
            target_ids = torch.concat([input_ids[:, :1], target_ids], dim=-1)
            target_ids = target_ids[:, review_index:]
            target_probs = target_probs[:, review_index - 1:, :]
            input_ids = input_ids[:, review_index:]
            target_ids = target_ids.to(input_ids.device)
            match_ct = 0
            for i in range(probs_new.shape[1]):
                if target_ids[0, i] == input_ids[0, i]:
                    match_ct += 1
                    continue
                else:
                    if leniency > 1 and target_probs[0, i, input_ids[0, i]] > probs_new[0, i, input_ids[0, i]] / leniency:
                        match_ct += 1
                        continue
                    else:
                        i = i - 1
                        break
            input_ids = torch.cat([input_ids[:, :i + 1], target_ids[:, i + 1:i + 2]], dim=-1)
            id_res = torch.concat([prefix_input_ids, input_ids], dim=-1)
            prob_res = torch.concat([probs[:, :review_index, :], target_probs[:, :i + 1, :]], dim=-2)
            return id_res, prob_res




class CSDraftingDecoderModelKVCache(CSDraftingModel):
    def __init__(self, model, sample=False, name='', vocab_size=32000):
        super().__init__(model, sample, name, vocab_size=vocab_size)
        self.past_key_values = None
        self.past_ids = None
    @classmethod
    def longest_common_prefix(cls, a, b):
        match = a[:, :b.shape[-1]] == b[:, :a.shape[-1]]
        match_ct = torch_index(torch.cat([match, torch.full((1, 1), False, device=match.device)], dim=-1), False)
        return match_ct
    def prepare_input(self, input_ids, review_index):
        if self.past_key_values is None:
            return input_ids, None
        else:
            longest_common_prefix = self.longest_common_prefix(self.past_ids, input_ids)
            longest_common_prefix = min(longest_common_prefix, review_index - 1)
            if longest_common_prefix < 10:
                self.past_key_values = None
                self.past_ids = None
                return input_ids, None
            new_token_ct = input_ids.shape[-1] - longest_common_prefix
            need_crop = self.past_ids.shape[-1] - longest_common_prefix > 0
            if need_crop:
                new_past_key_values = crop_past_key_values(self.past_key_values, longest_common_prefix)
                new_past_ids = self.past_ids[:, :longest_common_prefix]
                self.past_key_values = new_past_key_values
                self.past_ids = new_past_ids
            return input_ids[:, longest_common_prefix:], self.past_key_values
    def post_forward_cache(self, out, whole_input_ids):
        self.past_key_values = out.past_key_values
        self.past_ids = whole_input_ids
        assert self.past_ids.shape[-1] == self.past_key_values[0][0].shape[-2]
    def review(self, initial_input, input_ids, probs, review_index, leniency=1):
        start = time.time()
        cut_input_ids, past_key_values = self.prepare_input(input_ids, review_index)
        cache_len = 0
        if past_key_values is not None:
            cache_len = self.past_ids.shape[-1]
        out = self.model(cut_input_ids, past_key_values=self.past_key_values, use_cache=True)
        target_logits = out.logits
        self.post_forward_cache(out, input_ids)
        prefix_input_ids = input_ids[:, :review_index]
        target_probs = torch.nn.functional.softmax(target_logits, dim=-1)
        probs_new = probs[:, review_index - 1:, :]
        input_ids = input_ids[:, review_index:]
        target_index = review_index - 1 - cache_len + 1
        target_ids = torch.argmax(target_probs, dim=-1)
        target_ids = torch.concat([input_ids[:, :1], target_ids], dim=-1)
        target_ids = target_ids[:, target_index:]
        target_probs = target_probs[:, target_index:, :]
        target_ids = target_ids.to(input_ids.device)
        match_ct = 0
        start = time.time()
        for i in range(probs_new.shape[1]):
            if target_ids[0, i] == input_ids[0, i]:
                match_ct += 1
                continue
            else:
                if leniency > 1 and target_probs[0, i, input_ids[0, i]] > probs_new[0, i, input_ids[0, i]] / leniency:
                    match_ct += 1
                    continue
                else:
                    i = i - 1
                    break
        start = time.time()
        input_ids = torch.cat([input_ids[:, :i + 1], target_ids[:, i + 1:i + 2]], dim=-1)
        id_res = torch.concat([prefix_input_ids, input_ids], dim=-1)
        prob_res = torch.concat([probs[:, :review_index, :], target_probs[:, :i + 1, :]], dim=-2)
        return id_res, prob_res



class CountedCSDraftingDecoderModel(CSDraftingDecoderModel):
    def __init__(self, model, sample=False, name='', counter_version='model_parameters', vocab_size=32000):
        super().__init__(model, sample, name)
        self.forward_count = 0
        self.counter_version = counter_version
        time_cost_dict = TIME_COST[counter_version]
        self.time_cost = 0
        for model_abbr in time_cost_dict:
            if model_abbr in name:
                self.time_cost = time_cost_dict[model_abbr]
                break
        self.wall_time = []
    def review(self, initial_input, input_ids, probs, review_index, leniency=1):
        self.forward_count += 1
        start = time.time()
        res = super().review(initial_input, input_ids, probs, review_index, leniency)
        self.wall_time.append(time.time() - start)
        return res
    def calculate_time_cost(self):
        res = self.forward_count * self.time_cost
        self.forward_count = 0
        return res  




class CountedCSDraftingDecoderModelKVCache(CSDraftingDecoderModelKVCache):
    def __init__(self, model, sample=False, name='', counter_version='model_parameters', vocab_size=32000):
        super().__init__(model, sample, name)
        self.forward_count = 0
        self.counter_version = counter_version
        time_cost_dict = TIME_COST[counter_version]
        self.time_cost = 0
        for model_abbr in time_cost_dict:
            if model_abbr in name:
                self.time_cost = time_cost_dict[model_abbr]
                break
        self.wall_time = []
    def review(self, initial_input, input_ids, probs, review_index, leniency=1):
        self.forward_count += 1
        start = time.time()
        res = super().review(initial_input, input_ids, probs, review_index, leniency)
        self.wall_time.append(time.time() - start)
        return res
    def calculate_time_cost(self):
        res = self.forward_count * self.time_cost
        self.forward_count = 0
        # print('updated!')
        # print('Name of model: {}'.format(self.name))
        # print('Wall time: {}'.format(sum(self.wall_time) / len(self.wall_time)))
        return res  


class CountedCSDraftingCachedEncoderDecoderModel(CountedCSDraftingEncoderDecoderModel):
    def __init__(self, model, sample=False, name='', counter_version='model_parameters', cache_dir=''):
        super().__init__(model, sample, name, counter_version=counter_version,)
        self.cache = load_cache_model(cache_dir)
        if 't5' in name:
            self.first_decode_id = _T5_DECODER_START_TOKEN_ID
    def review(self, initial_input, input_ids, probs, review_index, leniency=1):
        self.forward_count += 1
        key = tokens_to_new_key(initial_input)
        cached_out = self.cache[key]
        decoded_tokens = key_to_tokens(cached_out)
        input_id_len = input_ids.shape[-1]
        target_ids = decoded_tokens[:, :input_id_len + 1]
        max_len = min(target_ids.shape[-1], input_ids.shape[-1])
        target_ids = target_ids.to(input_ids.device)
        target_ids_for_review = target_ids[:, review_index:max_len]
        input_ids_for_review = input_ids[:, review_index:max_len]
        matches = (target_ids_for_review[0, :] == input_ids_for_review[0, :]).int().detach().tolist()
        if 0 not in matches: 
            res_ids = target_ids[:, :len(matches) + 1 + review_index]
        else:
            res_ids = target_ids[:, :matches.index(0) + 1 + review_index]
        return res_ids, None

tokenizer = AutoTokenizer.from_pretrained('JackFram/llama-160m')

class CountedCSDraftingCachedDecoderModel(CountedCSDraftingDecoderModel):
    def __init__(self, model, sample=False, name='', counter_version='model_parameters', cache_dir=''):
        super().__init__(model, sample, name, counter_version='model_parameters')
        self.cache = load_cache_model(cache_dir)
    def review(self, initial_input, input_ids, probs, review_index, leniency=1):
        self.forward_count += 1
        key = tokens_to_new_key(initial_input)
        cached_out = self.cache[key]
        decoded_tokens = key_to_tokens(cached_out)
        input_id_len = input_ids.shape[-1]
        target_ids = decoded_tokens[:, :input_id_len + 1]
        max_len = min(target_ids.shape[-1], input_ids.shape[-1])
        target_ids = target_ids.to(input_ids.device)
        target_ids_for_review = target_ids[:, review_index:max_len]
        input_ids_for_review = input_ids[:, review_index:max_len]
        matches = (target_ids_for_review[0, :] == input_ids_for_review[0, :]).int().detach().tolist()
        if 0 not in matches: 
            res_ids = target_ids[:, :len(matches) + 1 + review_index]
        else:
            res_ids = target_ids[:, :matches.index(0) + 1 + review_index]
        return res_ids, None

