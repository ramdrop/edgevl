import torch
import numpy as np
import torch.utils.data as data
import itertools
from dataset.base import Base

class BaseCTRS:

    def __init__(self, margin, n_neg, is_embedding_set, mining_method, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.n_neg = n_neg
        self.is_embedding_set = is_embedding_set
        self.mining_method = mining_method
        print(f"==> BaseCTRS: margin={margin}, n_neg={n_neg}, is_embedding_set={is_embedding_set}, mining_method={mining_method}")

        self._build_relations()
        if self.is_embedding_set == False:
            self.embedding_cache = None          # torch.rand((self.__len__(), 512)) # ([5000, 512]) # how much memory does it take? 5000 * 512 * 4 = 10MB


    def _build_relations(self):
        self.positives = []                                            # 5000
        for i, class_id in enumerate(self.class_id_stack):
            positive = np.where(self.class_id_stack == class_id)[0] # find same-label samples
            positive = np.delete(positive, np.where(positive == i)[0]) # delete self
            self.positives.append(positive)

        self.negatives = []
        for i, class_id in enumerate(self.class_id_stack):
            negative = np.where(self.class_id_stack != class_id)[0] # find different-label samples
            self.negatives.append(negative)

        return


    def _load_transform(self, index):
        rgb_np, depth_np = self._load_sample(index)
        rgb_np, depth_np = self._apply_transform(rgb_np, depth_np)
        return torch.from_numpy(rgb_np), torch.from_numpy(depth_np)

    def mining_legacy(self, index):
        p_indices = self.positives[index]
        if len(p_indices) < 1:
            return None
        a_emb = self.embedding_cache[index]
        p_emb = self.embedding_cache[p_indices]
        dist_ap = torch.norm(a_emb - p_emb, dim=1, p=None) # Np NOTE
        d_p, inds_p = dist_ap.topk(1, largest=False)       # e.g., d_p:tensor([8.4966, 8.7970, 8.8521]), NOTE introcude some randomness here?
        index_p = p_indices[inds_p].item()

        n_indices = self.negatives[index]
        # n_indices = np.random.choice(self.negatives[index], self.n_negative_subset)                # randomly choose potential_negatives
        n_emb = self.embedding_cache[n_indices]                                                                              # (Np, D)
        dist_an = torch.norm(a_emb - n_emb, dim=1, p=None)                                                                   # Np
        d_n, inds_n = dist_an.topk(self.n_neg * 100 if len(n_indices) > self.n_neg * 100 else len(n_indices), largest=False) # e.g., d_p:tensor([8.4966, 8.7970, 8.8521]),
        loss_positive_indices = d_n < d_p + self.margin                                                                      # [True, True, ...] tensor
        if torch.sum(loss_positive_indices) < 1:
            return None
        n_indices_valid = inds_n[loss_positive_indices][:self.n_neg].numpy()                                                 # tensor -> numpy: a[tensor(5)] = 1, a[numpy(5)]=array([1])
        index_n = n_indices[n_indices_valid]

        a_rgb, a_depth = self._load_transform(index)                                # (C, H, W), (C, H, W)
        p_rgb, p_depth = self._load_transform(index_p)                              # (C, H, W), (C, H, W)
        n_rgb = torch.stack([self._load_transform(ind)[0] for ind in index_n])      # (n_neg, C, H, W])
        n_depth = torch.stack([self._load_transform(ind)[1] for ind in index_n], 0) # (n_neg, C, H, W])

        return a_rgb, p_rgb, n_rgb, a_depth, p_depth, n_depth, [self.class_id_stack[index], self.class_id_stack[index_p]] + [self.class_id_stack[x] for x in index_n]

    def mining(self, index):
        p_indices = self.positives[index]
        if len(p_indices) < 1:
            return None
        a_emb = self.embedding_cache[index]
        p_emb = self.embedding_cache[p_indices]
        dist_ap = torch.norm(a_emb - p_emb, dim=1, p=None) # Np NOTE
        if self.mining_method['POSITIVE'] == 'easy':
            d_p, inds_p = dist_ap.topk(1, largest=False)   # e.g., d_p:tensor([8.4966, 8.7970, 8.8521]), NOTE introcude some randomness here?
            index_p = p_indices[inds_p].item()
        elif self.mining_method['POSITIVE'] == 'hard':
            d_p, inds_p = dist_ap.topk(1, largest=True)
            index_p = p_indices[inds_p].item()
        elif self.mining_method['POSITIVE'] == 'random':
            index_tmp = np.random.choice(len(p_indices), 1)[0]
            d_p = dist_ap[index_tmp]
            index_p = p_indices[index_tmp]
        else:
            raise ValueError(f"positive_level {self.positive_level} not supported")

        n_indices = self.negatives[index]
        n_emb = self.embedding_cache[n_indices]  # self.embedding_cache: (3912, 1024)
        dist_an = torch.norm(a_emb - n_emb, dim=1, p=None)           # Np
        if self.mining_method['NEGATIVE'] == 'easy':
            d_n, inds_n = dist_an.topk(len(n_indices), largest=True) # e.g., d_p:tensor([8.4966, 8.7970, 8.8521]),
        elif self.mining_method['NEGATIVE'] == 'hard':
            d_n, inds_n = dist_an.topk(len(n_indices), largest=False)
        elif self.mining_method['NEGATIVE'] == 'random':
            index_tmp = np.random.choice(len(n_indices), len(n_indices), replace=False)
            d_n = dist_an[index_tmp]
            inds_n = torch.from_numpy(n_indices[index_tmp])
        else:
            raise ValueError(f"negative_level {self.hard_level} not supported")

        if self.mining_method['TOTAL'] == 'all':
            loss_positive_indices = d_n < d_p + self.margin                  # [True, True, ...] tensor
        elif self.mining_method['TOTAL'] == 'hard':
            loss_positive_indices = d_n < d_p
        elif self.mining_method['TOTAL'] == 'semihard':
            loss_positive_indices = (d_n < d_p + self.margin) & (d_n > d_p)
        else:
            raise ValueError(f"total_level {self.total_level} not supported")
        if torch.sum(loss_positive_indices) < 1:
            return None
        index_n = inds_n[loss_positive_indices][:self.n_neg].numpy() # tensor -> numpy: a[tensor(5)] = 1, a[numpy(5)]=array([1])

        a_rgb, a_depth = self._load_transform(index)                                # (C, H, W), (C, H, W)
        p_rgb, p_depth = self._load_transform(index_p)                              # (C, H, W), (C, H, W)
        n_rgb = torch.stack([self._load_transform(ind)[0] for ind in index_n])      # (n_neg, C, H, W])
        n_depth = torch.stack([self._load_transform(ind)[1] for ind in index_n], 0) # (n_neg, C, H, W])

        return a_rgb, p_rgb, n_rgb, a_depth, p_depth, n_depth, [self.class_id_stack[index], self.class_id_stack[index_p]] + [self.class_id_stack[x] for x in index_n]


    def __getitem__(self, index):
        if self.is_embedding_set == True:
            rgb_np, depth_np = self._load_transform(index)
            return rgb_np, depth_np, index

        else:
            if self.mining_method == {}:
                return self.mining_legacy(index)
            else:
                return self.mining(index)


    def collate_fn(self, batch_input):
        batch = list(filter(lambda x: x is not None, batch_input))
        if len(batch) == 0:
            return None, None, None

        a_rgb, p_rgb, n_rgb, a_depth, p_depth, n_depth, labels = zip(*batch)
        num_negs = data.dataloader.default_collate([x.shape[0] for x in n_rgb]) # ([B, C, H, W]) = ([C, H, W])

        a_rgb = data.dataloader.default_collate(a_rgb)     # ([B, C, H, W]) = ([C, H, W]) + ...
        p_rgb = data.dataloader.default_collate(p_rgb)     # ([B, C, H, W]) = ([C, H, W]) + ...
        n_rgb = torch.cat(n_rgb, 0)                        # ([B, C, H, W]) = ([C, H, W]) + ...
        a_depth = data.dataloader.default_collate(a_depth) # ([B, C, H, W]) = ([C, H, W]) + ...
        p_depth = data.dataloader.default_collate(p_depth) # ([B, C, H, W]) = ([C, H, W]) + ...
        n_depth = torch.cat(n_depth, 0)                    # ([B, C, H, W]) = ([C, H, W]) + ...

        labels = list(itertools.chain(*labels))

        return torch.cat((a_rgb, p_rgb, n_rgb, a_depth, p_depth, n_depth)), num_negs, labels
