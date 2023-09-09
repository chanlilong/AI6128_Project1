from scipy.interpolate import interp1d
from pathlib import Path
from read_utils.io_f import read_data_file
import ast
import numpy as np
import pandas as pd
from read_utils.io_f import read_data_file
from glob import glob
from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import List,Dict,Optional
from joblib import load
NoneType = type(None)

class IndoorLocDataset(Dataset):
    '''
    Returns each .txt from the indoor loc dataset into variables that aims to predict a set of targets.
    Variables : [IMU sensors, Wifi Data, iBeacon Data]
    Target: [X,Y, Floor]
    '''    
    def __init__(self,root:str="/home/lilong/4TB/AI6128_Project_1/data",\
                 buildingStats_loc :str= "./dataset_details/floorStats.csv",\
                 wirelessDF_loc :str= "./dataset_details/wirelessStats.csv",\
                 split = "train",\
                 split_size = 0.95,
                 norm :Optional[str]=None):
        

        # self.allFiles = glob(f"{root}\\train\\**\\**\\*.txt")
        self.return_files(root,split,split_size)

        if norm:
            self.norm = load(norm)
        else:
            self.norm = norm
        
        buildingStats = pd.read_csv(buildingStats_loc)
        buildingStats[["Cx","Cy","Xmax","Ymax","Xmin","Ymin"]] = buildingStats[["Cx","Cy","Xmax","Ymax","Xmin","Ymin"]].astype(np.float32)
        buildingStats.index = pd.MultiIndex.from_tuples(buildingStats.apply(lambda x: (x.ID,x.floor),axis=1), names=["ID", "floor"])
        buildingStats = buildingStats.drop(["ID","floor"],axis=1)


        wirelessDF = pd.read_csv(wirelessDF_loc)
        wirelessDF["BSSID"] = wirelessDF["BSSID"].apply(lambda x: list(set(ast.literal_eval(x))))
        wirelessDF["UUID"] = wirelessDF["UUID"].apply(lambda x: list(set(ast.literal_eval(x))))
        wirelessDF.index = wirelessDF["BuildingID"]
        
        self.uniqueBuilding = buildingStats.index.get_level_values("ID")
        self.uniqueFloors = buildingStats.index.get_level_values("floor").unique().sort_values()

        floor_map = {'1F':0,'2F':1,'3F':2,'4F':3,'5F':4,'6F':5,'7F':6,'8F':7,'9F':8,'B':-1,'B1':-2,'B2':-3,'B3':-4,'BF':-1,'BM':-2,'F1':0,'F10':9,'F2':1,'F3':2,'F4':3,'F5':4,\
        'F6':5,'F7':6,'F8':7,'F9':8,'G':0,'L1':0,'L10':9,'L11':10,'L2':1,'L3':2,'L4':3,'L5':4,'L6':5,'L7':6,'L8':7,'L9':8,'LG1':0,'LG2':2,'LM':1,'M':2,'P1':-1,'P2':-2}
        self.minFloor = min(floor_map.values())
        self.floor2int = {k:v+abs(self.minFloor) for k,v in floor_map.items()}
        self.nFloors = len(set(self.floor2int.values()))


        self.buildingStats = buildingStats
        self.wirelessDF = wirelessDF
        self.build2int = {b:i for i,b in enumerate(self.uniqueBuilding)}
        # self.floor2int = {f:i for i,f in enumerate(self.uniqueFloors)}
        # self.wifi2int = self.makeIDX(self.wirelessDF["BSSID"])
        # self.beac2int = self.makeIDX(self.wirelessDF["UUID"])

    def return_files(self,root:str,split:str,split_size:float):
        allFiles = np.array(glob(f"{root}/train/**/**/*.txt"))
        # print(len(allFiles))
        allidx = np.arange(len(allFiles))

        np.random.seed(42)
        train_idx = np.random.choice(allidx,size=int(split_size*(len(allFiles))),replace=False)
        val_idx = np.array(list(set(allidx) - set(train_idx)))
        np.random.seed()

        if split=="train":
            self.allFiles = allFiles[train_idx]
        elif split=="val":
            self.allFiles = allFiles[val_idx]
            
    @staticmethod    
    def makeIDX(arr):
        wifi2int = {}

        for wifiList in arr:
            for i,wifi in enumerate(wifiList): 
                wifi2int[wifi] = i
        return wifi2int
    def __len__(self)->int:
        return len(self.allFiles)
    
    def interpolate(self,acce_datas:np.ndarray,b:np.ndarray) -> np.ndarray:
        if acce_datas.shape[0]==b.shape[0]:
            return b
        else:
            ret = np.zeros((acce_datas.shape[0],b.shape[1]),dtype=np.float32)
            interp = interp1d(b[:,0],b[:,1:],axis=0,fill_value="extrapolate")
            ret[:,1:] = interp(acce_datas[:,0])
            ret[:,0] = acce_datas[:,0]
            
            return ret
    def nanFilter(self,x:np.ndarray)->np.ndarray:
        x[np.isnan(x)] = 0
        return x
    
    def normIMU(self,x:np.ndarray)->np.ndarray:
        # self.norm(x.reshape(-1,12))
        if self.norm:
            shape = x.shape
            x = self.norm.transform(x.reshape(-1,12)).reshape(shape)

            return x
        else:
            return x

    def __getitem__(self, idx) -> dict:

        try:
            d = self.get_data(idx)
        except:
            rnd = np.random.randint(low=0,high=len(self))
            d= self.get_data(rnd)

        return d
        
    def get_data(self,idx: int ) -> dict:
        
        path_filename = self.allFiles[idx]
        path = Path(path_filename)
        
        try:
            path_datas = read_data_file(path_filename)
        except:
            rndPath = np.random.choice(self.allFiles)
            path_datas = read_data_file(rndPath)
        acce_datas = path_datas.acce
        magn_datas = path_datas.magn
        ahrs_datas = path_datas.ahrs
        gyro_datas = path_datas.gyro
        wifi_datas = path_datas.wifi
        ibeacon_datas = path_datas.ibeacon
        posi_datas = path_datas.waypoint
        
        gyro_datas,magn_datas,ahrs_datas = [self.interpolate(acce_datas,x) for x in [gyro_datas,magn_datas,ahrs_datas]]
        IMU_data = np.concatenate([acce_datas,gyro_datas[:,1:],magn_datas[:,1:],ahrs_datas[:,1:]],axis=1).astype(np.float32)
        

        buildingID,floor = path.parent.parent.stem,path.parent.stem
        flooridx = self.floor2int[floor]
        # buildingidx = self.build2int[buildingID]
        
        Cx,Cy,Xmax,Ymax,Xmin,Ymin = self.buildingStats.loc[buildingID,floor].values

        interp = interp1d(posi_datas[:,0],posi_datas[:,1:],axis=0,fill_value="extrapolate")
        posInterp = interp(IMU_data[:,0])
        posInterp[:,0] = (posInterp[:,0]-Cx)/(Xmax-Xmin+1e-05)
        posInterp[:,1] = (posInterp[:,1]-Cy)/(Ymax-Ymin+1e-05)
        
        # IMUsensors = np.concatenate([IMU_data,posInterp],axis=1)
        
        
        #===================[Wifi Processing]=====================#
        # try:
        if wifi_datas.shape[0]>0:
            wifiDF = pd.DataFrame(wifi_datas[:,:-1],columns=["time","SSID","BSSID","RSSI"])
            wifiDF["time"]= wifiDF["time"].astype(np.int64)
            wifiDF["RSSI"]= wifiDF["RSSI"].astype(np.float32).fillna(-110)
            wifiTime = wifiDF.pivot_table(values="RSSI",columns="time",index="BSSID").fillna(-110).T
            xInterp = wifiTime.index.values.astype(np.float32)
            # if len(xInterp)>2:
            # minT,maxT = xInterp.min(),xInterp.max()
            interp = interp1d(xInterp,wifiTime.values,axis=0,fill_value="extrapolate",kind="nearest")#,kind="nearest")
            wifiData_ = interp(IMU_data[:,0])
            wifiBSSID = wifiTime.columns.values


            wifi2int = {x:i for i,x in enumerate(self.wirelessDF.loc[buildingID].BSSID)}
            n_wifi = len(wifi2int)

            wifiIDX = np.array([wifi2int[x] if x in wifi2int.keys() else 0 for x in wifiBSSID ])
            wifiData = np.ones((IMU_data.shape[0],n_wifi),dtype=np.float32)*-110
            wifiData[:,wifiIDX] = self.nanFilter(wifiData_)
            # else:
            #     wifi2int = {x:i for i,x in enumerate(self.wirelessDF.loc[buildingID].BSSID)}
            #     n_wifi = len(wifi2int)
            #     wifiData = np.ones((IMU_data.shape[0],n_wifi),dtype=np.float32)*-110
            #     wifiIDX = np.array([0],dtype=np.int64) 
            
        else:
            wifi2int = {x:i for i,x in enumerate(self.wirelessDF.loc[buildingID].BSSID)}
            n_wifi = len(wifi2int)
            wifiData = np.ones((IMU_data.shape[0],n_wifi),dtype=np.float32)*-110
            wifiIDX = np.array([0],dtype=np.int64)


        #===================[iBeacon Processing]=====================#
        if ibeacon_datas.shape[0]>0:
            ibeaconDF = pd.DataFrame(ibeacon_datas,columns=["time","UUID","RSSI"])
            ibeaconDF["time"]= ibeaconDF["time"].astype(np.int64)
            ibeaconDF["RSSI"]= ibeaconDF["RSSI"].astype(np.float32).fillna(-110)
            ibeaconTime = ibeaconDF.pivot_table(values="RSSI",columns="time",index="UUID").fillna(-110).T
            xInterp = ibeaconTime.index.values.astype(np.float32)
            # if len(xInterp)>2:
            interp = interp1d(xInterp,ibeaconTime.values,axis=0,fill_value="extrapolate",kind="nearest")#,kind="nearest")
            ibeaconData_ = interp(IMU_data[:,0])
            ibeaconUUID = ibeaconTime.columns.values


            beac2int = {x:i for i,x in enumerate(self.wirelessDF.loc[buildingID].UUID)}
            n_beac = len(beac2int)
            ibeaconUUID = [x for x in ibeaconUUID if x in beac2int.keys()]
            validIDX =  [i for i,x in enumerate(ibeaconUUID) if x in beac2int.keys()]
            beacIDX = np.array([beac2int[x] for x in ibeaconUUID])
            ibeaconData = np.ones((IMU_data.shape[0],n_beac),dtype=np.float32)*-110
            ibeaconData[:,beacIDX] = self.nanFilter(ibeaconData_[:,validIDX])
            # else:
            #     beac2int = {x:i for i,x in enumerate(self.wirelessDF.loc[buildingID].UUID)}
            #     n_beac = len(beac2int)
            #     ibeaconData = np.ones((IMU_data.shape[0],n_beac),dtype=np.float32)*-110
            #     beacIDX = np.array([0],dtype=np.int64)
        else:
            beac2int = {x:i for i,x in enumerate(self.wirelessDF.loc[buildingID].UUID)}
            n_beac = len(beac2int)
            ibeaconData = np.ones((IMU_data.shape[0],n_beac),dtype=np.float32)*-110
            beacIDX = np.array([0],dtype=np.int64)
                
        # print(self.normIMU(IMU_data[:,1:]))
        return {"buildingID":buildingID,\
                "floor":flooridx,\
                "imuData":self.nanFilter(self.normIMU(IMU_data[:,1:])),\
                "target":self.nanFilter(posInterp),\
                "wifiData":wifiData,\
                "beacData":ibeaconData,\
                "wifiIDX":wifiIDX,\
                "beacIDX":beacIDX}
    

def collate_fn(batch : List[dict], maxlen :int = 4096):
    '''
    Collates batch from pytorch dataloader into tensor batches with a maximum temporal dimension
    This function also normlize RSSI values from wifi and ibeacon by dividing them by -110.
    '''
    # print(batch)
    
    B = len(batch)
    lenTime = torch.as_tensor([x["imuData"].shape[0] for x in batch],dtype=torch.int64)
    max_t = np.minimum(max(lenTime) ,maxlen)
    # keys =['buildingID', 'floor', 'imuData', 'target', 'wifiData', 'beacData', 'wifiIDX', 'beacIDX']
    # buildingID = torch.zeros((B,1),dtype=torch.int64)
    buildingID = []
    floorID = torch.zeros((B,),dtype=torch.int64)
    imuData = torch.zeros((B,max_t,12),dtype=torch.float32)
    target = torch.zeros((B,max_t,2),dtype=torch.float32)
    wifiIDX = []
    beacIDX = []
    wifiData= []
    beacData = []
    
    for i,b in enumerate(batch):
        
        imu_ = b["imuData"][:max_t]
        target_ = b["target"][:max_t]
        
        # buildingID[i] = b["buildingID"]
        buildingID.append(b["buildingID"])
        floorID[i] = b["floor"]
        imuData[i,:imu_.shape[0]] = torch.as_tensor(imu_,dtype=torch.float32)
        target[i,:imu_.shape[0]] = torch.as_tensor(target_,dtype=torch.float32)
        wifiIDX.append(torch.as_tensor(b["wifiIDX"],dtype=torch.int64))
        beacIDX.append(torch.as_tensor(b["beacIDX"],dtype=torch.int64))
        
        w= torch.as_tensor(b["wifiData"],dtype=torch.float32)[:max_t,:]
        w_ = torch.ones(max_t,w.shape[1],dtype=torch.float32)*-110
        w_[:w.shape[0]] = w
        
        wifiData.append(w_/-110.0)
        
        b= torch.as_tensor(b["beacData"],dtype=torch.float32)[:max_t,:]
        b_ = torch.ones(max_t,b.shape[1],dtype=torch.float32)*-110
        b_[:b.shape[0]] = b
        
        beacData.append(b_/-110.0)
            
    returnData = {'buildingID':buildingID,\
                  'floor':floorID,\
                  'timeLen' :torch.minimum(lenTime,max_t),
                  'imuData':imuData,\
                  'target':target,\
                  'wifiData':wifiData,\
                  'beacData':beacData,\
                  'wifiIDX':wifiIDX,\
                  'beacIDX':beacIDX}
    return returnData
