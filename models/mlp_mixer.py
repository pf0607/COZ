import torch.nn as nn

from models import register
import torch
from einops.layers.torch import Rearrange

    
class FeedForward_sr64(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 2*hidden_dim),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.GELU(),
            # nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class MixerBlock_noLnorm_sr64(nn.Module):

    def __init__(self, dim, num_patch,  dropout = 0.):
        super().__init__()

        self.rearrange = Rearrange('b n d -> b d n')
        self.token_mix = nn.Sequential(
            # nn.LayerNorm(dim),
            # Rearrange('b n d -> b d n'),
            FeedForward_sr64(num_patch+16, num_patch, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            # nn.LayerNorm(dim),
            FeedForward_sr64(dim+6, dim, dropout),
        )

    def forward(self, x, coord, rel_cell,scale):
        tmp = self.rearrange(x)
        tmp = torch.cat([tmp,rel_cell],dim=-1)
        x = x + self.token_mix(tmp)

        tmp = torch.cat([x,coord,scale],dim=-1)
        x = x + self.channel_mix(tmp)

        return x

    
    
@register('MLP-mixer-all-no_norm-sr64-localE')
class MLPMixer_NoNorm_all_sr64_localE(nn.Module):

    def __init__(self, dim, num_patch, out_dim, hidden_list,
                 depth=3, token_dim=2, channel_dim=256):
        super().__init__()

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock_noLnorm_sr64(dim, num_patch))
        
        # self.mlp_head = MLP(in_dim=dim, out_dim=channel_dim//2, hidden_list = hidden_list)


        self.out = nn.Linear(dim,out_dim)

    def forward(self, x,coord,rel_cell,scale):

        res = x
        for mixer_block in self.mixer_blocks:
            x =res+ mixer_block(x,coord ,rel_cell,scale)

        # x += res
        # x = self.layer_norm(x)
        # x = x.mean(dim=1)
        return self.out(x)
    
    
    



    
    
    

    
    