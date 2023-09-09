from xformers.factory.model_factory import xFormer, xFormerConfig
from xformers.factory.block_factory import xFormerEncoderBlock,xFormerEncoderConfig
from xformers.components.positional_embedding import RotaryEmbedding
import torch
from torch.nn import Embedding,Linear,Sequential
import pandas as pd
from typing import Optional,List,Tuple
from torch import nn

class IndoorLocModel(torch.nn.Module):
    
    def __init__(self,xformerConfig:xFormerEncoderConfig,len_floors:int,wirelessDF:pd.DataFrame)-> None:
        super().__init__()
        # self.rope = RotaryEmbedding(dim_model=xformerConfig.dim_model)
        self.encoder = xFormerEncoderBlock(xformerConfig)
        
        self.wifiEmbeddings = torch.nn.ModuleDict([[k,Embedding(len(v),xformerConfig.dim_model)] for k,v in wirelessDF[["BuildingID","BSSID"]].values])
        self.beaconEmbeddings = torch.nn.ModuleDict([[k,Embedding(len(v),xformerConfig.dim_model)] for k,v in wirelessDF[["BuildingID","UUID"]].values])
        self.wifiProj =torch.nn.ModuleDict([[k,Linear(len(v),xformerConfig.dim_model//3)] for k,v in wirelessDF[["BuildingID","BSSID"]].values])
        self.beaconProj =torch.nn.ModuleDict([[k,Linear(len(v),xformerConfig.dim_model//3)] for k,v in wirelessDF[["BuildingID","UUID"]].values])
        
        
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
    
    def padSequence(self,x:List[torch.Tensor],len_mask:Optional[torch.Tensor]= None)->torch.Tensor:
        if isinstance(len_mask,torch.Tensor):
            dim = x[0].shape[-1]
            padded = torch.zeros((len(x),len_mask.max(),dim)).to(len_mask.device)

            for i,arr in enumerate(x):
                padded[i,:arr.shape[1],:] = arr
            return padded
        else:
            assert len(x)==1, "If not using len_mask, batch must be 0, "
            return x[0]
        
    
    def forward(self,buildingIDs:List[str],\
                wifiIDX:List[torch.Tensor],\
                beacIDX:List[torch.Tensor],\
                wifiData:List[torch.Tensor],\
                beacData:List[torch.Tensor],\
                imuData:torch.Tensor,\
                len_mask:Optional[torch.Tensor] = None)->Tuple[torch.Tensor,torch.Tensor]:
        
        wifiEmb = torch.cat([self.wifiEmbeddings[bID](wID)[None,...].mean(1,keepdim=True) for bID,wID in zip(buildingIDs,wifiIDX)],dim=0) #Batch,n_wifi,dim
        beacEmb = torch.cat([self.beaconEmbeddings[bID](beacID)[None,...].mean(1,keepdim=True) for bID,beacID in zip(buildingIDs,beacIDX)],dim=0)

        wifiFeat = torch.cat([self.wifiProj[bID](wID)[None,...] for bID,wID in zip(buildingIDs,wifiData)],dim=0)
        beacFeat = torch.cat([self.beaconProj[bID](beacID)[None,...] for bID,beacID in zip(buildingIDs,beacData)],dim=0)
        
        
        BATCH,SEQ,_ = imuData.shape
                
        x = self.imuProj(imuData)
        
        x = torch.cat([x,wifiFeat,beacFeat],dim=-1)
        x = torch.cat([wifiEmb,beacEmb,x],dim=1)
        
        len_mask_ = self.getLenMask(len_mask+2,BATCH,SEQ+2)
                
        x = self.encoder(x,input_mask=len_mask_)[:,2:,:]
        locPred = self.locPred(x)

        floorPred = self.floorPred(x).max(1).values

        locPred = self.maskBeforeReturn(locPred ,len_mask)
        return locPred,floorPred
    

my_config ={
        "reversible": True,  # Optionally make these layers reversible, to save memory
        "block_type": "encoder",
        "num_layers": 6,  # Optional, this means that this config will repeat N times
        "dim_model": 128*3,
        "residual_norm_style": "pre",  # Optional, pre/post
        "position_encoding_config": {
                                    "name": "sine",  # whatever position encodinhg makes sense
                                    "dim_model": 128*3,
                                    },
        "multi_head_config": {
            "num_heads": 8,
            "residual_dropout": 0.1,
            "attention": {
                "name": "nystrom",  # whatever attention mechanism
                "dropout": 0.1,
                "causal": True
            },
        },
        "feedforward_config": {
            "name": "MLP",
            "dropout": 0.1,
            "activation": "relu",
            "hidden_layer_multiplier": 4,
        },
    }

config = xFormerEncoderConfig(**my_config)