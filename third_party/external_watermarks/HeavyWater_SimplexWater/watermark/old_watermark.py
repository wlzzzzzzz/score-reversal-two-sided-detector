# Modified: retain HeavyWater processor dependencies; unused evaluation helpers removed.
from __future__ import annotations

from typing import List, Optional
import random
import torch
from transformers import LogitsProcessor

def top_p_indices(matrix, p):
    """
    Given:
    - a 1D tensor of shape (num_cols,) representing a single probability distribution, OR
    - a 2D tensor of shape (batch_size, num_cols) representing multiple distributions,

    return:
    - if 1D input: a 1D tensor containing the smallest set of indices whose sum is >= p.
    - if 2D input: a list of length batch_size, each element is a 1D tensor of column indices
        for the smallest set of entries in that row whose sum is >= p.

    :param matrix: Probability distribution(s) with shape either (num_cols,) or (batch_size, num_cols).
    :param p: Threshold for cumulative probability (0 < p <= 1).
    :return: A 1D tensor of INDICES (for 1D input) or a list of 1D tensors (for 2D input).
    """
    if matrix.dim() not in (1, 2):
        raise ValueError("Input must be either a 1D or 2D tensor.")

    if matrix.dim() == 1:
        # Single distribution: shape (num_cols,).
        # Convert to shape (1, num_cols) for unified processing,
        # then we'll extract the single row result at the end.
        matrix = matrix.unsqueeze(0)
        single_distribution = True
    else:
        single_distribution = False

    # 1) Sort probabilities (descending) along the last dimension and keep track of original indices
    sorted_probs, sorted_indices = torch.sort(matrix, descending=True, dim=-1)

    # 2) Compute cumulative sums along the sorted probabilities
    cumsum_probs = sorted_probs.cumsum(dim=-1)

    # 3) For each row, find how many entries are needed to reach or exceed p
    #    We do this by checking where cumsum_probs < p, counting those positions,
    #    then taking the next index to ensure >= p.
    keep_lens = (cumsum_probs < p).sum(dim=-1) + 1  # Convert count of "still below p" to index of >= p

    # 4) Gather top-p indices
    results = []
    for row_idx, length in enumerate(keep_lens):
        results.append(sorted_indices[row_idx, :length])

    # For a single distribution, return the single row's result, otherwise return list of results
    if single_distribution:
        return results[0]
    else:
        return results


class BlacklistLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that enforces that specified sequences will never be sampled.

    Args:
        bad_words_ids (`List[List[int]]`):
            List of list of token ids that are not allowed to be generated. In order to get the token ids of the words
            that should not appear in the generated text, use `tokenizer(bad_words, add_prefix_space=True,
            add_special_tokens=False).input_ids`.
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
    """

    def __init__(self, 
                bad_words_ids: List[List[int]], 
                eos_token_id: int,
                vocab: list[int], 
                vocab_size: int,
                bl_proportion: float=0.5,
                bl_logit_bias: float=1.0,
                bl_type: str = "hard", # "soft"
                initial_seed: int=None, 
                dynamic_seed: str=None, # "initial", "markov_1", None
                store_bl_ids: bool=False,
                store_spike_ents: bool = False,
                noop_blacklist: bool = False,
                top_p: float = 0.999,
                temperature: float = 1.0,
                context=1,
                hashing='min',
                ):
        
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.bl_proportion = bl_proportion
        self.bl_logit_bias = bl_logit_bias
        self.bl_type = bl_type
        self.seed_increment = 0
        self.saved_distributions = []
        self.top_p = top_p
        self.temperature = temperature
        self.context = context
        self.hashing = hashing

        if initial_seed is None: 
            self.initial_seed = None
            assert dynamic_seed != "initial"
        else:
            random.seed(initial_seed)
            self.initial_seed = initial_seed
            
        self.dynamic_seed = dynamic_seed

        self.eos_token_id = eos_token_id
        # self.bad_words_id_length_1 = self._prepare_bad_words(bad_words_ids)

        self.bad_words_mask: Optional[torch.LongTensor] = None

        self.store_bl_ids = store_bl_ids
        self.bl_ids = None

        self.store_spike_ents = store_spike_ents
        self.spike_entropies = None

        # hack to replace this with an approximation of infinite bias
        # so that the expectation coefficient will come out to 1.0
        if self.bl_type == "hard":
            self.bl_logit_bias = 10000 # FIXME to a value that is actually close to the largest soft watermark used

        alpha = torch.exp(torch.tensor(self.bl_logit_bias)).item()
        # gamma = self.bl_proportion
        gamma = 1.0-self.bl_proportion
        self.alpha = alpha
        self.gamma = gamma

        self.z_value = ((1-gamma)*(alpha-1))/(1-gamma+(alpha*gamma))
        self.expected_wl_coef = (gamma*alpha)/(1-gamma+(alpha*gamma))

        # catch for overflow when bias is "infinite"
        if self.alpha == torch.inf:
            self.z_value = 1.0
            self.expected_wl_coef = 1.0

        self.noop_blacklist = noop_blacklist
        if self.noop_blacklist: print("Blacklist processor for accounting only, no rescoring of logits")

        self.g_cuda = None
        self.large_prime = 15485863

    @property
    def blacklisted_ids(self):
        assert self.store_bl_ids, "Need to instantiate processor with `store_bl_ids` to be able to retrieve them later"
        # flatten the each indexes blacklist
        l_of_bl_ids = [[] for _ in range(len(self.bl_ids))]
        for b_idx, batch in enumerate(self.bl_ids):
            for l_of_l, seed in batch:
                bl_ids = [l[0] for l in l_of_l] # this was the main line, maybe unnecessary now?
                l_of_bl_ids[b_idx].append((bl_ids,seed))
        return l_of_bl_ids

    def get_and_clear_stored_bl_ids(self):
        old_bl_ids = self.bl_ids
        self.bl_ids = None
        return old_bl_ids

    def get_spike_entropies(self):
        spike_ents = [[] for _ in range(len(self.spike_entropies))]
        for b_idx, ent_tensor_list in enumerate(self.spike_entropies):
            for ent_tensor in ent_tensor_list:
                spike_ents[b_idx].append(ent_tensor.item())
        return spike_ents

    def get_and_clear_stored_spike_ents(self):
        spike_ents = self.get_spike_entropies()
        self.spike_entropies = None
        return spike_ents

   
    def compute_spike_entropy(self, scores):
        # precomputed z value in init
        probs = scores.softmax(dim=-1)
        denoms = 1+(self.z_value*probs)
        renormed_probs = probs / denoms
        sum_renormed_probs = renormed_probs.sum()
        return sum_renormed_probs

    def gen_seed(self, token_ids):
        token_ids = token_ids.tolist()
        if self.hashing == 'min':
            agg = min(token_ids)
        elif self.hashing == 'sum':
            agg = sum(token_ids)
        elif self.hashing == 'prod':
            agg = 1
            for i in token_ids:
                agg *= i
        elif self.hashing == 'repeat':
            token_ids = tuple(token_ids)
            if token_ids in self.hash_dict:
                pass
            else:
                self.hash_dict[token_ids] = self.seed_increment
                self.seed_increment += 1
                agg = 1
                for i in token_ids:
                    agg *= i
        return agg
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.bad_words_id_length_1 = [None for _ in range(input_ids.shape[0])]

        if self.g_cuda is None:
            self.g_cuda = torch.Generator(device=input_ids.device)

        for b_idx in range(input_ids.shape[0]):
            if self.dynamic_seed == "initial":
                self.g_cuda.manual_seed(self.large_prime*self.initial_seed)
            elif self.dynamic_seed == "markov_1":
                self.g_cuda.manual_seed(self.large_prime*input_ids[b_idx][-1].item())
            elif self.dynamic_seed is None:
                # let the rng evolve naturally - this is not a realistic setting
                pass
            elif self.dynamic_seed == 'fresh':
                self.seed_increment += 1
                seed = self.large_prime + self.seed_increment
                self.g_cuda.manual_seed(seed)
                # print(f'seed {seed}, prev_token {input_ids[b_idx][-1].item()}')
            elif self.dynamic_seed == 'agg_hash':
                #TD - create vrious sliding window hashes.
                seed = self.gen_seed(input_ids[b_idx][-self.context:])
                self.g_cuda.manual_seed(seed)
            #
            # apply temperature
            scores[b_idx] = scores[b_idx] / self.temperature
            p = torch.softmax(scores[b_idx], dim=-1)
            self.saved_distributions.append(p.detach().cpu().clone())
            filter_indices = top_p_indices(p, self.top_p)
            ####
            p_new = torch.zeros(size=(self.vocab_size,), device=p.device)
            p_new[filter_indices] = p[filter_indices]
            p_new = p_new / p_new.sum()
            ####

            scores[b_idx] = torch.log(p_new+ 1e-10)  # add small value to avoid log(0)
            # print(f"now tok is {input_ids[b_idx][-1].item()}")
            bl_ct = int(self.vocab_size*self.bl_proportion)
            blacklist_ids = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.g_cuda)[:bl_ct] # ty Yuxin :]
            # print(f"blacklist_ids[0] is {blacklist_ids[0]}")

            if self.store_bl_ids: 
                if self.bl_ids is None: self.bl_ids = [[] for _ in range(input_ids.shape[0])]
                self.bl_ids[b_idx].append((blacklist_ids,input_ids.tolist()[b_idx][-1]))

            if self.store_spike_ents: 
                if self.spike_entropies is None: self.spike_entropies = [[] for _ in range(input_ids.shape[0])]
                self.spike_entropies[b_idx].append(self.compute_spike_entropy(scores[b_idx]))
            
            # self.bad_words_id_length_1[b_idx] = self._prepare_bad_words(blacklist_ids)
            # this logic may not really be necessary for our usecase
            self.bad_words_id_length_1[b_idx] = blacklist_ids
        
        if not self.noop_blacklist:
            self.bad_words_mask = self._calc_curr_bad_word_mask(scores)
            scores = self._set_scores_to_inf_for_banned_tokens(scores)
            
        # print("scores shape is", scores.shape)
        
        return scores

    def _prepare_bad_words(self, bad_words_ids: List[List[int]]) -> list[int]:
        bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [self.eos_token_id], bad_words_ids))
        return bad_words_ids
        # used to have more logic, not used now

    def _calc_curr_bad_word_mask(self, scores: torch.FloatTensor) -> torch.BoolTensor:
        bad_words_mask = torch.zeros_like(scores)
        for b_idx in range(len(self.bad_words_id_length_1)):
            bad_words_mask[b_idx][self.bad_words_id_length_1[b_idx]] = 1
        final_mask = bad_words_mask.bool()
        return final_mask

    def _set_scores_to_inf_for_banned_tokens(
        self, scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be a
        list of list of banned tokens to ban in the format [[batch index, vocabulary position],...

        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
            # NOTE^^ Omitted logic for dynamic mask based on multi-token ban words
        """
        if self.bl_type == "hard":
            scores = scores.masked_fill(self.bad_words_mask, -float("inf"))
        elif self.bl_type == "soft":
            whitelist_mask = torch.logical_not(self.bad_words_mask)
            scores[whitelist_mask] = scores[whitelist_mask] + self.bl_logit_bias
        else:
            raise NotImplementedError(f"unrecognized bl type {self.bl_type}!")          
        return scores


