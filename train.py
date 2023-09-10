import torch
from models.dataset import IndoorLocDataset,collate_fn
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model import IndoorLocModel,config
from torch.optim import AdamW
from tensorboardX import SummaryWriter
from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch.nn.functional as F
from joblib import load

EPOCHS = 30
N_BATCH = 20

if __name__=="__main__":
    
    
    # root = "../data/"
    root="/home/lilong/4TB/AI6128_Project_1/data/"
    dataset = IndoorLocDataset(root=root,norm="./dataset_details/std_scaler.bin")
    train_dataloader = DataLoader(dataset, batch_size=N_BATCH, shuffle=True,collate_fn = collate_fn,num_workers=4)
    
    val_dataset = IndoorLocDataset(root=root,split="val",norm="./dataset_details/std_scaler.bin")
    val_dataloader = DataLoader(val_dataset, batch_size=N_BATCH,collate_fn = collate_fn,num_workers=4)

    print(f"n_training:{len(dataset)} | n_val:{len(val_dataset)}")

    n_floors = dataset.nFloors

    model = IndoorLocModel(config,n_floors,dataset.wirelessDF).cuda()

    opt = AdamW(model.parameters(),lr=1e-04,weight_decay=0.01)

    writer = SummaryWriter(logdir = f"./tensorboard_logs/train_cross")

    itr = 0
    for e in range(EPOCHS):

        #=================[TRAINING CODE]============================#
        training_loss = 0
        training_acc = 0
        training_samples = 0
        kaggle_score = 0
        model.train()

        for d in tqdm(train_dataloader):
            buildingID = d["buildingID"]
            timeLen = d["timeLen"].cuda()
            imuData = d["imuData"].cuda()
            target = d["target"].cuda()
            floorID = d["floor"].cuda()
            wifiData = [x.cuda() for x in d["wifiData"]]
            beacData = [x.cuda() for x in d["beacData"]]
            wifiIDX = [x.cuda() for x in d["wifiIDX"]]
            beacIDX = [x.cuda() for x in d["beacIDX"]]

            B,T,_ = imuData.shape
            locPred,floorPred = model(  buildingIDs = buildingID,\
                                        wifiIDX = wifiIDX,\
                                        beacIDX = beacIDX,\
                                        wifiData = wifiData,\
                                        beacData = beacData,\
                                        imuData = imuData,\
                                        len_mask = timeLen)
            
            floorPred = floorPred.flatten(0,1)
            y_oh = F.one_hot(floorID,n_floors).unsqueeze(1).repeat(1,T,1).flatten(0,1)
            floorLoss = sigmoid_focal_loss(floorPred,y_oh.float()).mean()

            locLoss = torch.square(locPred-target).mean()

            loss = floorLoss+locLoss
            # break
            opt.zero_grad()
            loss.backward()
            opt.step()
            itr+=1
            
            with torch.no_grad():
                floorClss = floorPred.sigmoid().argmax(-1)
                floorID_ = floorID.unsqueeze(1).repeat(1,T).flatten()
                equals = (floorClss==floorID_).reshape(-1,1).detach().cpu()
                training_acc += torch.sum(equals.type(torch.FloatTensor)).item()/T
                floorDiff = torch.abs(floorClss-floorID_).float()
                kaggle_score += B*((locPred-target).norm(dim=-1).mean()+ 15*(floorDiff).mean()).item()
                training_loss += B * loss.item()
                training_samples += B

            if itr%50:
                kaggle_scores = kaggle_score/training_samples
                train_loss = (training_loss/training_samples)
                train_acc = (training_acc/training_samples)
                writer.add_scalar("train/acc", train_acc, itr)
                writer.add_scalar("train/kaggle", kaggle_scores, itr)
                writer.add_scalar("train/loss", train_loss, itr)
        # break
        train_loss = (training_loss/training_samples)
        train_acc = (training_acc/training_samples)  
        kaggle_scores =   kaggle_score/training_samples
        writer.add_scalar("train/kaggle_e", kaggle_scores, itr)
        writer.add_scalar("train/loss_e", train_loss, itr)   
        writer.add_scalar("train/acc_e", train_acc, itr)   


        #=================[VALIDATION CODE]============================#
        val_loss = 0
        val_acc = 0
        val_samples = 0
        kaggle_score = 0
        ys = []
        preds = []
        model.eval()
        for d in tqdm(val_dataloader):
            buildingID = d["buildingID"]
            timeLen = d["timeLen"].cuda()
            imuData = d["imuData"].cuda()
            target = d["target"].cuda()
            floorID = d["floor"].cuda()
            wifiData = [x.cuda() for x in d["wifiData"]]
            beacData = [x.cuda() for x in d["beacData"]]
            wifiIDX = [x.cuda() for x in d["wifiIDX"]]
            beacIDX = [x.cuda() for x in d["beacIDX"]]

            with torch.no_grad():
                B,T,_ = imuData.shape
                locPred,floorPred = model(  buildingIDs = buildingID,\
                                            wifiIDX = wifiIDX,\
                                            beacIDX = beacIDX,\
                                            wifiData = wifiData,\
                                            beacData = beacData,\
                                            imuData = imuData,\
                                            len_mask = timeLen)
                floorPred = floorPred.flatten(0,1)
                y_oh = F.one_hot(floorID,n_floors).unsqueeze(1).repeat(1,T,1).flatten(0,1)
                floorLoss = sigmoid_focal_loss(floorPred,y_oh.float()).mean()

                locLoss = torch.square(locPred-target).mean()

                loss = floorLoss+locLoss

                
                floorClss = floorPred.sigmoid().argmax(-1)
                floorID_ = floorID.unsqueeze(1).repeat(1,T).flatten()
                equals = (floorClss==floorID_).reshape(-1,1).detach().cpu()
                val_acc += torch.sum(equals.type(torch.FloatTensor)).item()/T
                floorDiff = torch.abs(floorClss-floorID_).float()
                kaggle_score += B*((locPred-target).norm(dim=-1).mean()+ 15*(floorDiff).mean()).item()
                val_loss += B * loss.item()
                val_samples += B

        # break
        val_loss = (val_loss/val_samples)
        val_acc = (val_acc/val_samples)  
        kaggle_scores =   kaggle_score/val_samples
        writer.add_scalar("val/kaggle_e", kaggle_scores, itr)
        writer.add_scalar("val/loss_e", val_loss, itr)   
        writer.add_scalar("val/acc_e", val_acc, itr)   
    
        model_dict = {"params":model.state_dict(),"itr":itr,"epoch":e}
        torch.save(model_dict,"./weights/cross_model.pth")