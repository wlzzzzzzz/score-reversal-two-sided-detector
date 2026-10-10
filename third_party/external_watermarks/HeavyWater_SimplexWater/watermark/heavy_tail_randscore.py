# Modified: retain the generation kernel; unused watermark detector removed.
from __future__ import annotations

import torch
import ot
from watermark.old_watermark import BlacklistLogitsProcessor, top_p_indices
import numpy as np


def generate_normalized_lognormal_cost_matrix(n_tokens, S_size=1024, mean=0.0, sigma=1.0, seed = 15485863, device = None, dist='lognormal'):
    if seed is not None:
        np.random.seed(seed)

    # Draw samples from a lognormal distribution
    if dist == 'gamma':
        print('chose gamma')
        k=0.2
        scale = 1.0
        raw_C = np.random.gamma(shape=k, scale=scale, size=(n_tokens, S_size))
    elif dist =='pareto':
        print('applying pareto')
        alpha = 2.5
        x_m   = 1.0
        raw_C = (np.random.pareto(alpha, size=(n_tokens, S_size)) + 1) * x_m
    else:
        #lognormal
        raw_C = np.random.lognormal(mean=mean, sigma=sigma, size=(n_tokens, S_size))
    
    
    # Normalize each row to have zero mean and unit variance
    row_means = np.mean(raw_C, axis=1, keepdims=True)
    row_stds = np.std(raw_C, axis=1, keepdims=True)
    normalized_C = (raw_C - row_means) / row_stds
    return torch.tensor(normalized_C, dtype=torch.float32, device=device)


class HeavyTailLogitsProcessor(BlacklistLogitsProcessor):
    def __init__(self, 
                bad_words_ids, 
                eos_token_id,
                vocab, 
                vocab_size,
                bl_proportion=0.5,
                bl_logit_bias=1.0,
                bl_type= "hard", # "soft"
                initial_seed=None, 
                dynamic_seed=None, # "initial", "markov_1", None
                store_bl_ids=False,
                store_spike_ents= False,
                noop_blacklist= False,
                tilt=False,
                tilting_delta=0.1,
                top_p = 0.999,
                context=1,
                hashing='min',
                hash_key=15485863, # large prime
                k=1024, # size of the side info space
                dist = 'lognormal',
                temperature=1.0,
                sinkhorn_reg = 0.05,
                sinkhorn_thresh = 1e-5
                ):
        super().__init__(bad_words_ids, 
                eos_token_id,
                vocab, 
                vocab_size,
                bl_proportion,
                bl_logit_bias,
                bl_type, # "soft"
                initial_seed, 
                dynamic_seed, # "initial", "markov_1", None
                store_bl_ids,
                store_spike_ents,
                noop_blacklist)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tilt = tilt
        self.tilting_delta = tilting_delta
        self.seed_increment = 0
        self.saved_distributions = []
        self.top_p = top_p
        self.context = context
        self.hashing = hashing
        self.hash_key = hash_key
        self.k = k
        self.temperature = temperature
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_thresh = sinkhorn_thresh
        self.cost_matrix = generate_normalized_lognormal_cost_matrix(n_tokens = vocab_size, \
            S_size=self.k, mean=0.0, sigma=1.0, seed = self.hash_key, device=self.device, dist=dist)
    

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
            
    def __call__(self, input_ids, scores):
        """
        Heavy Tail WM logitprocessor - currently lognormal.
        Steps:
        1. sample s uniformly over {1, (self.k)}
        2. gen p = distr via softmax
        3. p_new - tilt_q_HT(p, s, self.cost_matrix)
        4. p_new into logits
        """

        if self.g_cuda is None:
            self.g_cuda = torch.Generator(device=input_ids.device)

        for b_idx in range(input_ids.shape[0]):
            # seed generation:
            if self.dynamic_seed == "initial":
                seed = self.large_prime*self.initial_seed
            elif self.dynamic_seed == "markov_1":
                seed = self.large_prime*input_ids[b_idx][-1].item()
            elif self.dynamic_seed is None:
                # let the rng evolve naturally - this is not a realistic setting
                pass
            elif self.dynamic_seed == 'fresh':
                self.seed_increment += 1
                seed = self.large_prime + self.seed_increment
            elif self.dynamic_seed == 'agg_hash':
                #TD - create vrious sliding window hashes.
                seed = self.gen_seed(input_ids[b_idx][-self.context:])
            # set seed and generate s
            self.g_cuda.manual_seed(seed)
            s = torch.randint(
                low=1,
                high=self.k + 1,  # S_size is inclusive
                size=(1,),
                generator=self.g_cuda,
                device=input_ids.device
            ).item()  

        if not self.noop_blacklist:
            # we choose to watermark
            scores_new = [None for _ in range(input_ids.shape[0])]
            for b_idx in range(input_ids.shape[0]):
                # apply temperature
                scores[b_idx] = scores[b_idx] / self.temperature
                p = torch.softmax(scores[b_idx], dim=-1)
                # Save a clone so future changes don't affect it
                self.saved_distributions.append(p.detach().cpu().clone())
                p_new =  self.tilt_q_HT(distribution=p, side_info=s)
                scores_new[b_idx] = torch.log(p_new)
            scores_new = torch.stack(scores_new,axis=0)
        return scores_new
    
    def tilt_q_HT(self, distribution, side_info, reg_const=0.05, num_iter=2000, top_k=100):
        # reg_const = 0.5
        # num_iter=20
        # top-p filtering:
        # apply top_p filtering to the scores
        filter_indices = top_p_indices(distribution, self.top_p)
        if len(filter_indices) == 1:
            # then there's nothing to watermark.
            return distribution
        distribution = distribution[filter_indices]
        ## normalize
        distribution = distribution / distribution.sum()
        ##
        ps = torch.ones(self.k, device=distribution.device) / self.k  # uniform over S

        # convert to cost minimization problem
        C = - self.cost_matrix[filter_indices, :]
        #prev:
        # C = (C - C.min())/10  # make sure all values are non-negative and small enough for Sinkhorn
        # new normalization:
        C = (C - C.min())  # make sure all values are non-negative and small enough for Sinkhorn
        C = C/C.max()

        # sinkhorn on reduced cost
        P = ot.sinkhorn(
            a=distribution,
            b=ps,
            M= C,
            reg=self.sinkhorn_reg,
            numItermax=num_iter,
            # stopThr=1e-7
            stopThr=self.sinkhorn_thresh
        )
        # numpy version
        # P = ot.sinkhorn(
        #     a=distribution.cpu().numpy(),
        #     b=ps.cpu().numpy(),
        #     M= self.cost_matrix[filter_indices, :],
        #     reg=reg_const,
        #     numItermax=num_iter,
        #     stopThr=1e-5
        # )
        P_wm = P[:, side_info - 1] # map side_info [1,...m-1] to [0,...,m-2] inclusive.
        if self.tilt:
            #linear tilt:
            c = self.cost_matrix[ filter_indices,side_info - 1]
            # index_1s = torch.where(c >=0)[0]
            # index_0s = torch.where(c <0)[0]
            # P_wm[index_1s] = P_wm[index_1s]*(1+self.tilting_delta)
            # P_wm[index_0s] = P_wm[index_0s]*(1-self.tilting_delta)
            # exponential tilt:
            # c = self.cost_matrix[ filter_indices,side_info - 1]
            P_wm = P_wm*torch.exp(self.tilting_delta*c) # null mean is 0
        # reconstruct conditional
        # P_wm = torch.zeros(m, device=distribution.device)
        # merged_col = reduced['inverse'][side_info - 1]
        # P_wm[reduced['keep_indices']] = P_reduced[:, merged_col]
        P_wm = P_wm / P_wm.sum()
        out_p = torch.zeros(size=(self.vocab_size,), device=distribution.device)
        out_p[filter_indices] = P_wm
        return out_p
