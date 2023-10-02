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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
EPOCHS = 20
N_BATCH = 16

class OnlineVariance(object):
    """
    Welford's algorithm computes the sample variance incrementally.
    """

    def __init__(self, iterable=None, ddof=1):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, datum: np.ndarray)->None:
        self.n += datum.shape[0]
        self.delta = (datum - self.mean).sum()
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean).sum()

    @property
    def variance(self):
        return self.M2 / (self.n - self.ddof+1e-05)

    @property
    def std(self):
        return np.sqrt(self.variance)
    
if __name__=="__main__":
    
    def unnorm(Original,Cx,Cy,Xmax,Ymax,Xmin,Ymin):
        return_arr = Original.clone()
        Cx,Cy,Xmax,Ymax,Xmin,Ymin = Cx.view(-1,1),Cy.view(-1,1),Xmax.view(-1,1),Ymax.view(-1,1),Xmin.view(-1,1),Ymin.view(-1,1)
        Cx,Cy,Xmax,Ymax,Xmin,Ymin = Cx.to(Original.device),Cy.to(Original.device),Xmax.to(Original.device),Ymax.to(Original.device),Xmin.to(Original.device),Ymin.to(Original.device)

        return_arr[...,1] = return_arr[...,1]*(Ymax-Ymin+1e-05)+Cy
        return_arr[...,0] = return_arr[...,0]*(Xmax-Xmin+1e-05)+Cx
        return return_arr
    

    def plot(pred,wireless,targets):
        IDX = np.random.randint(0,high=pred.shape[0])
        
        Original = targets[IDX].detach().cpu().numpy()
        Pred = pred[IDX].detach().cpu().numpy()
        PredW = wireless[IDX].detach().cpu().numpy()
        
        
        fig,ax = plt.subplots(1,1,figsize=(5,5))
        ax.plot(Original[:,0],Original[:,1],"r")
        ax.plot(Pred[:,0],Pred[:,1],"b--",alpha=0.5)
        
        fig2,ax2 = plt.subplots(1,1,figsize=(5,5))
        ax2.plot(Original[:,0],Original[:,1],"r")
        ax2.plot(PredW[:,0],PredW[:,1],"xb",alpha=0.5)
        
        err = np.linalg.norm(Pred-Original,axis=-1).flatten()
        err = err[err!=0]
        efig = sns.displot(err)
        plt.title(f"STD{err.std()} | Mean{err.mean()}")

        return fig,fig2,efig.fig

    def criterion(timeLen,floorPred,floorID,n_floors,locPred,wirelessLocPred,locGT):
        clss_loss = 0
        loc_loss = 0
        y_oh = F.one_hot(floorID,n_floors).unsqueeze(1)
        
        for t_len,flrPred,flrOH,locgt,locpred,wlocpred in zip(timeLen,floorPred,y_oh,locGT,locPred,wirelessLocPred):
            clss_loss += sigmoid_focal_loss(flrPred[:t_len],flrOH.repeat(t_len,1).float()).mean()
            loc_loss += 3*loss_func(locpred[:t_len],locgt[:t_len]) + loss_func(wlocpred[:t_len],locgt[:t_len])

        return (clss_loss+loc_loss)/timeLen.shape[0]
    def classEquals(timeLen,floorPred,floorID):
        floorClss = floorPred.sigmoid()
        # floorID_ = floorID.unsqueeze(1).repeat(1,T).flatten()
        total_equals = 0
        floorDiff = 0
        for t,f_pred,f_true in zip(timeLen,floorPred,floorID):
            # print(t,f_pred[:t].shape,f_true)
            f_pred = f_pred[:t].argmax(-1)
            f_true = f_true.repeat(t).flatten()
            equals = (f_pred==f_true).detach().sum().cpu()
            total_equals += equals/t
            
            floorDiff_ = torch.abs(f_pred-f_true).detach().sum().cpu()
            floorDiff += floorDiff_/t
        return total_equals,floorDiff
        
    # root = "../data/"
    root="/home/lilong/4TB/AI6128_Project_1/data/"
    dataset = IndoorLocDataset(root=root,norm="./dataset_details/std_scaler.bin")
    train_dataloader = DataLoader(dataset, batch_size=N_BATCH, shuffle=True,collate_fn = collate_fn,num_workers=4)
    
    val_dataset = IndoorLocDataset(root=root,split="val",norm="./dataset_details/std_scaler.bin")
    val_dataloader = DataLoader(val_dataset, batch_size=N_BATCH,collate_fn = collate_fn,num_workers=4)

    print(f"n_training:{len(dataset)} | n_val:{len(val_dataset)}")

    n_floors = dataset.nFloors
    
    loss_func = torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)
    model = IndoorLocModel(config,n_floors,dataset.wirelessDF).cuda()

    # modelD = torch.load("./weights/model_CP2.pth")
    # model.load_state_dict(modelD["params"])
    
    opt = AdamW(model.parameters(),lr=1e-04,weight_decay=0.01)

    writer = SummaryWriter(logdir = f"./tensorboard_logs/train")
    
    itr = 0
    # itr = modelD["itr"]
    for e in tqdm(range(EPOCHS)):

        #=================[TRAINING CODE]============================#
        training_loss = 0
        training_acc = 0
        training_samples = 0
        training_grad = 0
        kaggle_score = 0
        model.train()
        tracker = OnlineVariance()
        for d in train_dataloader:
            buildingID = d["buildingID"]
            timeLen = d["timeLen"].cuda()
            imuData = d["imuData"].cuda()
            target = d["target"].cuda()
            floorID = d["floor"].cuda()
            wifiData = [x.cuda() for x in d["wifiData"]]
            beacData = [x.cuda() for x in d["beacData"]]
            wifiIDX = [x.cuda() for x in d["wifiIDX"]]
            beacIDX = [x.cuda() for x in d["beacIDX"]]
            floorStats = d["floorStats"]

            B,T,_ = imuData.shape
            # T=T-1
            locPred,floorPred,wirelessLocPred= model(  buildingIDs = buildingID,\
                                                        wifiIDX = wifiIDX,\
                                                        beacIDX = beacIDX,\
                                                        wifiData = wifiData,\
                                                        beacData = beacData,\
                                                        imuData = imuData,\
                                                        len_mask = timeLen)
            # floorPred = floorPred.flatten(0,1)
            # y_oh = F.one_hot(floorID,n_floors).unsqueeze(1).repeat(1,T,1).flatten(0,1)
            # floorLoss = sigmoid_focal_loss(floorPred,y_oh.float()).mean()
            # floorLoss = classLoss(timeLen,floorPred,floorID,n_floors)
            # locLoss = 3*loss_func(locPred,target) + loss_func(wirelessLocPred,target)
            loc_clss_loss = criterion(timeLen=timeLen,\
                                      floorPred=floorPred,\
                                      floorID=floorID,\
                                      n_floors=n_floors,\
                                      locPred=locPred,\
                                      wirelessLocPred=wirelessLocPred,\
                                      locGT=target)
            grad_loss = 0.1*torch.square((locPred[:,1:]-locPred[:,:-1]).norm(dim=-1)).mean()

            loss = loc_clss_loss + grad_loss
            # break
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            

            
            with torch.no_grad():

                Cx,Cy,Xmax,Ymax,Xmin,Ymin = floorStats.T
                a,b = unnorm(locPred,Cx,Cy,Xmax,Ymax,Xmin,Ymin),unnorm(target,Cx,Cy,Xmax,Ymax,Xmin,Ymin)
                err = (a-b).norm(p='fro',dim=2).flatten().cpu().numpy()
                err = err[err!=0]
                tracker.include(err.flatten())
                
                equals,floorDiff = classEquals(timeLen,floorPred,floorID)
                training_acc += equals.item()#/timeLen.detach().cpu().sum().numpy()
                kaggle_score += B*(err.mean()+ 15*(floorDiff).mean()).item()
                training_loss += (B * loss.item())
                training_samples += B
                training_grad += B*grad_loss.item()

            if itr%50 :
                train_grad = training_grad/training_samples
                kaggle_scores = kaggle_score/training_samples
                train_loss = (training_loss/training_samples)
                train_acc = (training_acc/training_samples)
                
                writer.add_scalar("train/mean_error_meters", tracker.mean, itr)
                writer.add_scalar("train/std_error_meters", tracker.std, itr)
                writer.add_scalar("train/grad", train_grad, itr)
                writer.add_scalar("train/acc", train_acc, itr)
                writer.add_scalar("train/kaggle", kaggle_scores, itr)
                writer.add_scalar("train/loss", train_loss, itr)

            if itr%150==0 :
                locpred,target_ = unnorm(locPred,Cx,Cy,Xmax,Ymax,Xmin,Ymin),unnorm(target,Cx,Cy,Xmax,Ymax,Xmin,Ymin)
                wirelesspred = unnorm(wirelessLocPred,Cx,Cy,Xmax,Ymax,Xmin,Ymin)
                fig,fig2,efig = plot(locpred,wirelesspred,target_)
                writer.add_figure("train/trajectory",fig,itr)
                writer.add_figure("train/fingerprint",fig2,itr)
                writer.add_figure("train/error_dist",efig,itr)
            itr+=1
        # break
        train_loss = (training_loss/training_samples)
        train_acc = (training_acc/training_samples)  
        kaggle_scores =   kaggle_score/training_samples
        train_grad = training_grad/training_samples
        writer.add_scalar("train/grad", train_grad, itr)
        writer.add_scalar("train/kaggle_e", kaggle_scores, itr)
        writer.add_scalar("train/loss_e", train_loss, itr)   
        writer.add_scalar("train/acc_e", train_acc, itr)   


        #=================[VALIDATION CODE]============================#
        val_loss = 0
        val_acc = 0
        val_samples = 0
        kaggle_score = 0
        val_grad = 0
        ys = []
        preds = []
        model.eval()
        tracker = OnlineVariance()
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
            floorStats = d["floorStats"]

            with torch.no_grad():
                B,T,_ = imuData.shape
                # T = T-1
                locPred,floorPred,wirelessLocPred = model(  buildingIDs = buildingID,\
                                                            wifiIDX = wifiIDX,\
                                                            beacIDX = beacIDX,\
                                                            wifiData = wifiData,\
                                                            beacData = beacData,\
                                                            imuData = imuData,\
                                                            len_mask = timeLen)
                loc_clss_loss = criterion(timeLen=timeLen,\
                                          floorPred=floorPred,\
                                          floorID=floorID,\
                                          n_floors=n_floors,\
                                          locPred=locPred,\
                                          wirelessLocPred=wirelessLocPred,\
                                          locGT=target)
                grad_loss = 0.1*torch.square((locPred[:,1:]-locPred[:,:-1]).norm(dim=-1)).mean()

                loss = loc_clss_loss + grad_loss
                

                
                # floorClss = floorPred.sigmoid().argmax(-1)
                # floorID_ = floorID.unsqueeze(1).repeat(1,T).flatten()
                # equals = (floorClss==floorID_).reshape(-1,1).detach().cpu()
                Cx,Cy,Xmax,Ymax,Xmin,Ymin = floorStats.T
                err = (unnorm(locPred,Cx,Cy,Xmax,Ymax,Xmin,Ymin)-unnorm(target,Cx,Cy,Xmax,Ymax,Xmin,Ymin)).norm(p='fro',dim=2).flatten().cpu().numpy()
                err = err[err!=0]
                tracker.include(err.flatten())

                equals,floorDiff = classEquals(timeLen,floorPred,floorID)
                val_acc += equals.item()#/timeLen.detach().cpu().sum().numpy()
                kaggle_score += B*(err.mean()+ 15*(floorDiff).mean()).item()
                val_loss += (B * loss.item())
                val_samples += B
                val_grad += B*grad_loss.item()

        # break
        val_loss = (val_loss/val_samples)
        val_acc = (val_acc/val_samples)  
        kaggle_scores =   kaggle_score/val_samples
        val_grad = val_grad/val_samples
        writer.add_scalar("val/mean_error_meters", tracker.mean, itr)
        writer.add_scalar("val/std_error_meters", tracker.std, itr)
        writer.add_scalar("val/grad", val_grad, itr)
        writer.add_scalar("val/kaggle_e", kaggle_scores, itr)
        writer.add_scalar("val/loss_e", val_loss, itr)   
        writer.add_scalar("val/acc_e", val_acc, itr)   
    
        model_dict = {"params":model.state_dict(),"itr":itr,"epoch":e}
        torch.save(model_dict,"./weights/model.pth")
        
    model_dict = {"params":model.state_dict(),"itr":itr,"epoch":e}
    torch.save(model_dict,"./weights/model.pth")