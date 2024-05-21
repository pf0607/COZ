import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

    
@register('lmi')
class LMI(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True, with_area=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.with_area = with_area

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            # if self.feat_unfold:
            #     imnet_in_dim *= 9
            # imnet_in_dim += 2 # attach coord
            # if self.cell_decode:
            #     imnet_in_dim += 2
            # if with_area:
            #     imnet_in_dim += 1
            self.imnet = models.make(imnet_spec, args={'dim': imnet_in_dim, 'num_patch': 16})
        else:
            self.imnet = None

        metanet_spec = {
            'name': 'mlp',
            'args': {
                'in_dim': 33,
                'out_dim': imnet_in_dim*16,
                'hidden_list': [imnet_in_dim]
            }
        }
        self.metanet = models.make(metanet_spec)


    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        feat_rgb = self.inp        

        # if self.feat_unfold:    # [B,9*64,48,48]
        #     feat = F.unfold(feat, 3, padding=1).view(
        #         feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-3, -1, 1, 3]
            vy_lst = [-3, -1, 1, 3]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        # [B,2,48,48]

        # preds = []
        areas = []
        inps = []
        rel_coords = []
        inps_rgb=[]
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                q_feat_rgb = F.grid_sample(
                    feat_rgb, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                inps_rgb.append(q_feat_rgb.unsqueeze(2))

                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                # inp = torch.cat([q_feat, rel_coord], dim=-1)

                inps.append(q_feat.unsqueeze(2))
                rel_coords.append(rel_coord.unsqueeze(2))

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)
            
        bs, q = coord.shape[:2]
        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat.shape[-2]
        rel_cell = rel_cell[:,:,0:1].unsqueeze(2)
        rel_coords.append(rel_cell)
        meta_inp = torch.cat(rel_coords,dim=-1)
        meta_mix = self.metanet(meta_inp.view(bs*q,-1))

        inp = torch.cat(inps, dim=2)
        # print(inp.shape)
        # for inp, area in zip(inps, areas):
        #     area /= tot_area

        rel_cell = rel_cell.view(bs*q,1,-1).repeat(1,16,1)
        rel_coord = torch.cat(rel_coords[0:16],dim=2)
        inp_rgb = torch.cat(inps_rgb,dim=2)
        rel_coord = torch.cat([inp_rgb,rel_coord],dim=-1)
        inp = inp.contiguous()
        preds= self.imnet(inp.view(bs*q, len(inps),-1), rel_coord.view(bs*q,len(inps),-1), meta_mix.view(bs*q,-1,16), rel_cell).view(bs, q, 16,-1)
        preds = torch.chunk(preds,16,dim=2)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            for i in range(8):
                t = areas[i]; areas[i] = areas[15-i]; areas[15-i] = t
            # t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred.squeeze(2) * (area / tot_area).unsqueeze(-1)

        return ret

    def forward(self, inp, coord, cell):
        self.inp = inp
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)