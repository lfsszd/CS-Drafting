import torch


_MAG_GENERATION = 10


def _csd_iteration(draft_list, target_model, initial_input, input_ids, k_matrix, leniency=1, is_first=False):
    if len(draft_list) == 0:
        input_id = target_model.propose(initial_input, input_ids, _MAG_GENERATION)
        probs = torch.nn.functional.one_hot(input_id, num_classes=target_model.vocab_size).float()
        # First input id doesn't have a probability
        probs = probs[:, 1:, :]
        return input_id, probs
    ks = k_matrix[0]
    n = len(draft_list)
    cur_input_ids = input_ids
    cur_probs = torch.zeros([1, 0, target_model.vocab_size], device=target_model.device)
    prev_ids_len = input_ids.shape[1]
    review_index = input_ids.shape[-1]
    for i in range(n):
        while cur_input_ids.shape[1] - prev_ids_len < sum(ks[:i + 1]):
            cur_target = draft_list[i]
            cur_draft_list = draft_list[i+1:]
            cur_k_matrix = k_matrix[i+1:, i+1:]
            new_input_ids, new_probs = _csd_iteration(cur_draft_list, cur_target, initial_input, cur_input_ids, cur_k_matrix, leniency=leniency)
            cur_input_ids = new_input_ids
            cur_probs = torch.cat([cur_probs, new_probs[:,cur_probs.shape[1]:, :]], dim=1)
    if is_first:
        leniency = 1
    accepted_tokens, target_probs = target_model.review(initial_input, cur_input_ids, new_probs, review_index, leniency=leniency)
    return accepted_tokens, target_probs




def csd(draft_list, target_model, initial_input, input_ids, k_matrix, max_length=200, leniency=1):
    with torch.no_grad():
        initial_len = input_ids.shape[-1]
        input_ids = input_ids.to(target_model.device)
        initial_input = initial_input.to(target_model.device)
        if 't5' in target_model.name:
            _EOS = 1
        else:
            _EOS = 2
        while input_ids.shape[-1] - initial_len < max_length:
            previous_len = input_ids.shape[-1]
            input_ids, _ = _csd_iteration(draft_list, target_model, initial_input, input_ids, k_matrix, leniency=leniency, is_first=True)
            ## check if <EOS> in the newly added token 
            if torch.any((input_ids[0, previous_len:]) == _EOS):
                break
            # Return if target cache is fiinished
            if input_ids.shape[-1] <= previous_len:
                break
            previous_len = input_ids.shape[-1]
        return input_ids
