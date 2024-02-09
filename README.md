# Cascade Speculative Drafting

The official implementation for "[Cascade Speculative Drafting for Even Faster LLM Inference](https://arxiv.org/abs/2312.11462)"

Cascade Speculative Drafting (CS Drafting) is an algorithm that improves upon speculative decoding by further speeding up LLM inference through cascades without sacrificing generation quality.


## Setup

It's likely that our code is competable with your local environment, so you are welcome to skip to usage section.

Our version of pip package can be found in requirements.txt.
We run our experiments with python3.9. 
You can install our environment by using anaconda

```
conda create --name csd python=3.9
conda activate csd
pip install requirements.txt
```


## Recreating Our Experiments

The starting point of the report is main.py which can be run without args for maximum hackability.
All experiment setting can be adjusted in the config diction in main.py.
GPU usage can be adjusted by changing the following line in main.py

```
usable_devices = [0, 1, 2] * 2
```

Each index in the list representing a single process on gpu of the index.
Note that target model is cached in ./cache, so running each process will cost less than 8GB of memory.
We recommend using 2 process for each GPU with at least 16gb of memory for higher GPU utiliization.

To run experiments with FLAN-T5 on mmlu for SWI (model size) setup, change the config to the following:

```
config = {
    'draft_names': ['google/flan-t5-base', 'google/flan-t5-small'],
    'target_name': 'google/flan-t5-xxl',
    'is_decoder_only': False,
    'use_mag': True,
    'k_matrix': [[5, 14, 10], [0, 1, 10], [0, 0, 10]],
    'lenience': 2,
    'dataset': 'mmlu',
    'counter_version': 'model_parameters',
    'sample': False
}
```

For SWI (previous work)

```
config = {
    'draft_names': ['google/flan-t5-base', 'google/flan-t5-small'],
    'target_name': 'google/flan-t5-xxl',
    'is_decoder_only': False,
    'use_mag': True,
    'k_matrix': [[5, 8, 10], [0, 1, 10], [0, 0, 10]],
    'lenience': 5,
    'dataset': 'sampled_mmlu',
    'counter_version': 'previous_work',
    'sample': False
}
```

To run LLAMA-7B on mmlu

```
config = {
    'draft_names': ['JackFram/llama-160m'],
    'target_name': 'llama_7b',
    'is_decoder_only': True,
    'use_mag': True,
    'k_matrix': [[13, 10], [0, 10]],
    'lenience': 3,
    'dataset': 'mmlu',
    'counter_version': 'model_parameters',
    'sample': False
}
```

To run gsm8k, you can change the dataset field in the config to 

```
'dataset': 'gsm8k'
```
Note that when using two draft models other than mag, the parameter
in k_matrix is different from the one in the paper. Their relations are the following:
```
k_matrix[0][0] = k<sub>11</sub>
k_matrix[0][1] = k<sub>12</sub> - k_matrix[0][0]
```



## Using CS Drafting for Inference

To run csd on your own inputs

```
from csd import csd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from model import CSDraftingDecoderModel, get_mag_model


draft_list = []
draft_names = ['JackFram/llama-160m']
for draft_name in draft_names:
    hf_model = AutoModelForSeq2SeqLM.from_pretrained(draft_name)
    model = CSDraftingDecoderModel(hf_model, name=draft_name, counter_version=config['counter_version'])
    draft_list.append(model)

_BIGRAM_DIR = './bigram_models/'
bi_gram_path = _BIGRAM_DIR + 'wiki_bigram_naive_bayers_greedy_llama_next_token.json'
mag_model = get_mag_model(bi_gram_path, config['is_decoder_only'])
draft_list.append(mag_model)

LLAMA_HF_PATH = LLAMA_PATH + 'hf_7b_chat'
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = <your hugginface llama tokenizer>
hf_model = <your hugginface llama model>

target_model = CSDraftingDecoderModel(hf_model, name='llama', vocab_size=32000)
target_model.cuda(device)

question = '<Your inputs>'
initial_input = tokenizer(question, truncation=True, padding=False, return_tensors="pt")['input_ids'].to(target_model.device)
input_ids = initial_input
res = csd(draft_list, target_model, initial_input, input_ids, k_matrix, max_length=200)
generated_text = tokenizer.batch_decode(res, skip_special_tokens=True)
```

 

## Citation

The details of this repo are described in the following paper:

```
@article{chen2023cascade,
  title={Cascade Speculative Drafting for Even Faster LLM Inference},
  author={Chen, Ziyi and Yang, Xiaocong and Lin, Jiacheng and Sun, Chenkai and Chen, Yangyi and Chang, Kevin Chen-Chuan and Huang, Jie},
  journal={arXiv preprint arXiv:2312.11462},
  year={2023}
}
```

