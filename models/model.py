# from xformers.factory.model_factory import xFormer, xFormerConfig
from xformers.factory.model_factory import xFormer, xFormerConfig,xFormerDecoderConfig,xFormerDecoderBlock
# from xformers.factory.block_factory import xFormerEncoderBlock,xFormerEncoderConfig
from xformers.components.positional_embedding import RotaryEmbedding
import torch
from torch.nn import Embedding,Linear,Sequential
import pandas as pd
from typing import Optional,List,Tuple
from torch import nn

class IndoorLocModel(torch.nn.Module):
    
    def __init__(self,xformerConfig:xFormerDecoderConfig,len_floors:int,wirelessDF:pd.DataFrame)-> None:
        super().__init__()
        self.rope = RotaryEmbedding(dim_model=xformerConfig.dim_model)

        self.decoder_wifi = xFormerDecoderBlock(xformerConfig)
        self.decoder_beac = xFormerDecoderBlock(xformerConfig)
        
        self.wifiEmbeddings = torch.nn.ModuleDict([[k,Embedding(len(v),xformerConfig.dim_model)] for k,v in wirelessDF[["BuildingID","BSSID"]].values])
        self.beaconEmbeddings = torch.nn.ModuleDict([[k,Embedding(len(v),xformerConfig.dim_model)] for k,v in wirelessDF[["BuildingID","UUID"]].values])
        self.wifiProj =torch.nn.ModuleDict([[k,Linear(len(v),xformerConfig.dim_model//3)] for k,v in wirelessDF[["BuildingID","BSSID"]].values])
        self.beaconProj =torch.nn.ModuleDict([[k,Linear(len(v),xformerConfig.dim_model//3)] for k,v in wirelessDF[["BuildingID","UUID"]].values])

        self.locProj = nn.Linear(xformerConfig.dim_model,2)
        
        
        self.imuProj = Linear(12,xformerConfig.dim_model//3)
        self.locPred = Linear(xformerConfig.dim_model,2)
        self.floorPred = Linear(xformerConfig.dim_model,len_floors)
        
    def getLenMask(self,len_mask,BATCH,SEQ)->torch.Tensor:
        if not isinstance(len_mask,type(None)):
            len_mask_ = torch.zeros((BATCH,SEQ),dtype=torch.bool).to(len_mask.device)
            for i,l in enumerate(len_mask):
                len_mask_[i,:l] = 1
        return len_mask_
    
    def maskBeforeReturn(self,x,len_mask):
        if not isinstance(len_mask,type(None)):
            for i,l in enumerate(len_mask):
                x[i,l:] = 0
                
        return x
    
    def padSequence(self,x:List[torch.Tensor])->Tuple[torch.Tensor,torch.Tensor]:
        lens = torch.as_tensor([i.shape[1] for i in x],dtype=torch.int64,device=x[0].device)
        dim = x[0].shape[-1]
        padded = torch.zeros((len(x),max(lens),dim)).to(x[0].device)

        for i,arr in enumerate(x):
            padded[i,:arr.shape[1],:] = arr
        return padded,lens

    
    def forward(self,buildingIDs:List[str],\
                wifiIDX:List[torch.Tensor],\
                beacIDX:List[torch.Tensor],\
                wifiData:List[torch.Tensor],\
                beacData:List[torch.Tensor],\
                imuData:torch.Tensor,\
                len_mask:Optional[torch.Tensor] = None)->Tuple[torch.Tensor,torch.Tensor]:
        
        wifiEmb,lenWifi = self.padSequence([self.wifiEmbeddings[bID](wID)[None,...] for bID,wID in zip(buildingIDs,wifiIDX)]) #Batch,n_wifi,dim
        beacEmb,lenBeac = self.padSequence([self.beaconEmbeddings[bID](beacID)[None,...] for bID,beacID in zip(buildingIDs,beacIDX)])
        
        wifiFeat = torch.cat([self.wifiProj[bID](wID)[None,...] for bID,wID in zip(buildingIDs,wifiData)],dim=0)
        beacFeat = torch.cat([self.beaconProj[bID](beacID)[None,...] for bID,beacID in zip(buildingIDs,beacData)],dim=0)
        
        
        BATCH,SEQ,_ = imuData.shape
                
        x = self.imuProj(imuData)
        
        x = torch.cat([x,wifiFeat,beacFeat],dim=-1)
        # x = torch.cat([wifiEmb_,beacEmb_,x],dim=1)
        
        len_mask_ = self.getLenMask(len_mask,BATCH,SEQ)
                
        x = self.decoder_wifi(target=x,memory=wifiEmb,input_mask=len_mask_)
        x = self.decoder_beac(target=x,memory=beacEmb,input_mask=len_mask_)

        locPred = self.locPred(x)

        floorPred = self.floorPred(x)

        locPred = self.maskBeforeReturn(locPred ,len_mask)
        return locPred,floorPred
    
my_config ={
            "reversible": True,  # Optionally make these layers reversible, to save memory
            "block_type": "decoder",
            "num_layers": 3,  # Optional, this means that this config will repeat N times
            "dim_model": 64*3,
            "residual_norm_style": "post",  # Optional, pre/post
            "position_encoding_config": {
                "name": "sine",  # whatever position encodinhg makes sense
            },
            "multi_head_config_masked": {
                "num_heads": 8,
                "residual_dropout": 0.1,
                "attention": {
                    "name": "nystrom",  # whatever attention mechanism
                    "dropout": 0.1,
                    "causal": True,
                },
            },
            "multi_head_config_cross": {
                "num_heads": 8,
                "residual_dropout": 0,
                "attention": {
                    "name": "favor",  # whatever attention mechanism
                    "dropout": 0.1,
                    "causal": False,
                },
            },
            "feedforward_config": {
                "name": "MLP",
                "dropout": 0.1,
                "activation": "relu",
                "hidden_layer_multiplier": 4,
            }}

config = xFormerDecoderConfig(**my_config)