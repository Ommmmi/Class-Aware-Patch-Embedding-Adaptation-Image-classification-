import torch
import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CPEA(nn.Module):
    def __init__(self, in_dim=384):
        super().__init__()
        # Class-aware patch adaptation MLP
        self.fc1 = Mlp(in_features=in_dim, hidden_features=int(in_dim / 4), out_features=in_dim)
        self.fc_norm1 = nn.LayerNorm(in_dim)
        # Patch similarity scoring MLP
        self.fc2 = Mlp(in_features=196 ** 2, hidden_features=256, out_features=1)
        

    def forward(self, feat_query, feat_shot, args):
        # feat_query: [Q, n, C], feat_shot: [KS, n, C]
        _, n, c = feat_query.size()

        # Patchwise class adaptation: add global patch mean (like CLS token) to each patch
        feat_query = self.fc1(torch.mean(feat_query, dim=1, keepdim=True)) + feat_query  # [Q, n, C]
        feat_shot  = self.fc1(torch.mean(feat_shot,  dim=1, keepdim=True)) + feat_shot   # [KS, n, C]
        feat_query = self.fc_norm1(feat_query)
        feat_shot  = self.fc_norm1(feat_shot)

        # Split into class token and patch tokens
        query_class   = feat_query[:, 0, :].unsqueeze(1)   # [Q, 1, C]
        query_image   = feat_query[:, 1:, :]               # [Q, L, C]
        support_class = feat_shot[:, 0, :].unsqueeze(1)    # [KS, 1, C]
        support_image = feat_shot[:, 1:, :]                # [KS, L, C]

        # Fuse class token with image tokens
        feat_query = query_image + 2.0 * query_class       # [Q, L, C]
        feat_shot  = support_image + 2.0 * support_class   # [KS, L, C]

        # Normalize and center patch features
        feat_query = F.normalize(feat_query, p=2, dim=2)
        feat_query = feat_query - torch.mean(feat_query, dim=2, keepdim=True)
        # Support shot aggregation/normalization
        feat_shot = feat_shot.contiguous().reshape(args.shot, -1, n-1, c) # [K, S, L, C]
        feat_shot = feat_shot.mean(dim=0)                                # [S, L, C]
        feat_shot = F.normalize(feat_shot, p=2, dim=2)
        feat_shot = feat_shot - torch.mean(feat_shot, dim=2, keepdim=True)

        # Dense patch-patch similarity scoring
        results = []
        for idx in range(feat_query.size(0)):
            tmp_query = feat_query[idx].unsqueeze(0)              # [1, L, C]
            out = torch.matmul(feat_shot, tmp_query.transpose(1, 2)) # [S, L, L]
            out = out.flatten(1)                                     # [S, L*L]
            out = self.fc2(out.pow(2))                               # [S, 1]
            out = out.transpose(0, 1)                                # [1, S]
            results.append(out)

        return results, None
