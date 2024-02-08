import json
from datasets import load_dataset

mmlu_features = ['input', 'A', 'B', 'C', 'D', 'target']
template_mmlu = 'Question: {}\n(A) {} (B) {} (C) {} (D) {}\n Answer: \n Let\'s think step by step. '
template_gsm8k = 'Reason the math question below step by step. Question: {}.\n Answer: '
mmlu_sub_categories = ['high_school_european_history', 'business_ethics', 'clinical_knowledge', 'medical_genetics', 'high_school_us_history', 'high_school_physics', 'high_school_world_history', 'virology', 'high_school_microeconomics', 'econometrics', 'college_computer_science', 'high_school_biology', 'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine', 'nutrition', 'global_facts', 'machine_learning', 'security_studies', 'public_relations', 'professional_psychology', 'prehistory', 'anatomy', 'human_sexuality', 'college_medicine', 'high_school_government_and_politics', 'college_chemistry', 'logical_fallacies', 'high_school_geography', 'elementary_mathematics', 'human_aging', 'college_mathematics', 'high_school_psychology', 'formal_logic', 'high_school_statistics', 'international_law', 'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 'miscellaneous', 'high_school_chemistry', 'marketing', 'professional_law', 'management', 'college_physics', 'jurisprudence', 'world_religions', 'sociology', 'us_foreign_policy', 'high_school_macroeconomics', 'computer_security', 'moral_scenarios', 'moral_disputes', 'electrical_engineering', 'astronomy', 'college_biology']


def format_initial_input(item, dataset_name):
    if dataset_name == 'gsm8k':
        initial_input = template_gsm8k.format(item['question'])
    elif dataset_name == 'mmlu' or 'mmlu' in dataset_name:
        initial_input = template_mmlu.format(*[item[f] for f in mmlu_features[:-1]])
    return initial_input


def get_test_set(dataset_name):
    if dataset_name == 'mmlu':
        mmlu = []
        for cat in mmlu_sub_categories:
            cur_set = load_dataset('lukaemon/mmlu', cat)
            mmlu += list(cur_set['test'])
        return mmlu
    elif dataset_name == 'gsm8k':
        # full_gsm8k_dir = '/home/your_dir/gsm8k_dataset.json'
        # with open(full_gsm8k_dir) as f:
        #     test = json.load(f)
        # return test['test']
        gsm8k = load_dataset('gsm8k', 'main')
        return gsm8k['test']
