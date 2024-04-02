import time
import torch
from csd import csd
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import CSDraftingDecoderModel, get_mag_model

device = 0
draft_list = []
draft_names = ['JackFram/llama-160m']
for draft_name in draft_names:
    hf_model = AutoModelForCausalLM.from_pretrained(draft_name)
    model = CSDraftingDecoderModel(hf_model, name=draft_name)
    model.cuda(device)
    draft_list.append(model)

_BIGRAM_DIR = './bigram_models/'
bi_gram_path = _BIGRAM_DIR + 'wiki_bigram_naive_bayers_greedy_llama_next_token.json'
mag_model = get_mag_model(bi_gram_path, True)
mag_model.cuda(device)
draft_list.append(mag_model)


LLAMA_PATH = '/scratch/your_dir/llama/llama/'

k_matrix = torch.tensor([[5, 10], [0, 10]])
LLAMA_HF_PATH = LLAMA_PATH + 'hf_7b_chat'
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained(LLAMA_HF_PATH)
hf_model = LlamaForCausalLM.from_pretrained(LLAMA_HF_PATH)

target_model = CSDraftingDecoderModel(hf_model, name='llama', vocab_size=32000)
target_model.cuda(device)


question = '<Your inputs>'
initial_input = tokenizer(question, truncation=True, padding=False, return_tensors="pt")['input_ids'].to(target_model.device)
input_ids = initial_input
res = csd(draft_list, target_model, initial_input, input_ids, k_matrix, max_length=200)
generated_text = tokenizer.batch_decode(res, skip_special_tokens=True)
