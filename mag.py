import torch
def _bigram_sampling(input_id, bi_gram_model):
    res = bi_gram_model[input_id]
    for i in range(2 - len(res.shape)):
        res = res.unsqueeze(0)
    return res


def torch_index(t, value):
    return (t == value).nonzero(as_tuple=True)[0][0]

# def _fast_n_gram_search_index_legacy(input_ids, encoder_ids, n=1):
#     matches = (encoder_ids[0] == input_ids[0, -1]).int()
#     if matches.sum() < 1:
#         return None
#     for i in range(2, input_ids.shape[-1] + 1):
#         new_matches = (encoder_ids[0, :(-1 * (i - 1))] == input_ids[0, -1 * i]).int()
#         combined_matches = (2 - new_matches == matches[1:]).int()
#         if combined_matches.sum() < 1:
#             index = torch_index(torch.cat((torch.tensor([0] * (i - 1), device=torch.device(encoder_ids.device)), matches), dim=-1), 1)
#             return encoder_ids[:, index:index + n]
#         else:
#             matches = combined_matches
#     index = torch_index(torch.cat((torch.tensor([0] * (encoder_ids.shape[-1] - matches.shape[-1]), device=matches.device), matches), dim=-1), 1)
#     return encoder_ids[:, index+1:index + n+1]


def _fast_n_gram_search_index(input_ids, encoder_ids, n=1):
    encoder_ids = torch.cat([encoder_ids, input_ids[0, :-1].unsqueeze(0)], dim=-1)
    matches = (encoder_ids[0] == input_ids[0, -1]).int()
    if matches.sum() < 1:
        return None
    for i in range(2, input_ids.shape[-1] + 1):
        new_matches = (encoder_ids[0, :(-1 * (i - 1))] == input_ids[0, -1 * i]).int()
        combined_matches = (2 - new_matches == matches[1:]).int()
        if combined_matches.sum() < 1:
            index = torch_index(torch.cat((torch.tensor([0] * (i - 1), device=torch.device(encoder_ids.device)), matches), dim=-1), 1)
            return encoder_ids[:, index:index + n]
        else:
            matches = combined_matches
    index = torch_index(torch.cat((torch.tensor([0] * (encoder_ids.shape[-1] - matches.shape[-1]), device=matches.device), matches), dim=-1), 1)
    return encoder_ids[:, index+1:index + n+1]

def draft_sample_k_bn_gram(bigram_list, initial_input, input_ids, k):
    """_summary_

    Args:
        model (_type_): smallest model used for verification
        initial_input (_type_): question
        input_ids (_type_): partial answer
        k (_type_): number of tokens to sample at the smallest model level
        n (_type_): n-gram model to exact match
        bigram_list (_type_): bigram model to sample
    """
    t0 = input_ids.shape[-1]
    inputs_plus_k = input_ids
    candidate_chunk = _fast_n_gram_search_index(inputs_plus_k, initial_input, k)
    if candidate_chunk is not None:
        candidate_chunk = candidate_chunk.to(inputs_plus_k.device)
        inputs_plus_k = torch.cat(
            [inputs_plus_k, candidate_chunk],
            dim=-1)
        if inputs_plus_k.shape[-1] >= t0 + k:
            # print('_fast_n_gram_search_index Newly proposed tokens len')
            # print(inputs_plus_k.shape[-1] - input_ids.shape[-1])
            return inputs_plus_k
    # Try to do max gram match in generated input ids
    candidate_chunk = _fast_n_gram_search_index(inputs_plus_k, inputs_plus_k[:,:-1], k)
    if candidate_chunk is not None:
        candidate_chunk = candidate_chunk.to(inputs_plus_k.device)
        inputs_plus_k = torch.cat(
            [inputs_plus_k, candidate_chunk],
            dim=-1)
        if inputs_plus_k.shape[-1] >= t0 + k:
            return inputs_plus_k
    next_tokens = []
    cur_token = inputs_plus_k[0][-1]
    for i in range(t0 + k - inputs_plus_k.shape[-1]):
        ## do drafting from exact match or bigram sampling
        next_token = _bigram_sampling(cur_token, bigram_list).to(inputs_plus_k.device)
        next_tokens.append(next_token)
        cur_token = next_token
    inputs_plus_k = torch.cat(
            [inputs_plus_k] + next_tokens,
            dim=-1)
    return inputs_plus_k  ## no logits can be returned anymore


