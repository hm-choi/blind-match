import torch
import numpy as np
from imgaug import augmenters as iaa
import random
import PIL.Image as Image
import warnings
import argparse, sys, os
from tqdm import tqdm
from blindMuth import *
from utils import *
warnings.filterwarnings('ignore')

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from os import path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import Grayscale
from sklearn.metrics import roc_curve, auc, make_scorer
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd

##############################################################
def _save_snapshot(self, epoch, name):
    path=f"Write down your model path",
    snapshot = {
        "MODEL_STATE": self.model.state_dict(), #module
        "EPOCHS_RUN": epoch,
        "MIN_LOSS": self.min_valid_loss,
        "LEARNING_RATE": self.learning_rate,
        "MAX_ACC": self.max_valid_accuracy,
    }
    torch.save(snapshot, path)
    print("Test model: ", name.upper())
##############################################################
    
    
##############################################################
def main(args, out_results, dataloader: DataLoader, dataloader_val: DataLoader, dataloader1N: DataLoader):
    args.pos_lbl = 1.0
    gpu_id = 'cuda'
    print("initialize model .... >>", args.train)
    if args.train == "1N":
        dataloader = dataloader1N
    else: dataloader = dataloader
    if (args.model == 'authST'):
        net = FingerNet(128)
        fc = FingerCentroids(300, 128)
    elif (args.model == 'auth'):
        net = FingerSTNNet(128)
        fc = FingerCentroids(300, 128)
    net = net.to(gpu_id)
    fc = fc.to(gpu_id)
    if torch.cuda.device_count() > 1: 
        print('==============Mutliple CUDA', torch.cuda.device_count(), gpu_id)
        net = nn.DataParallel(net)
        fc = nn.DataParallel(fc)
    ##############################################################
    loss_arcface = ArcFace(m=0.2).to(gpu_id)
    loss_crossentropy = nn.CrossEntropyLoss().to(gpu_id)
    loss_contrastive = ContrastiveLoss().to(gpu_id)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    '''scheduler = optim.lr_scheduler.LambdaLR(optimizer=optim,
                                    lr_lambda=lambda epoch: 0.99 ** epoch,
                                    last_epoch=-1,
                                    verbose=False)'''
    ##############################################################
    # Agumentation setting of training dataset 
    train_seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0, 0.3)),
        iaa.Dropout((0.01, 0.15), per_channel=0.5),
        iaa.Affine(
            translate_percent={"x": (-0.15, 0.2), "y": (-0.15, 0.2)},
            order=[0, 1],
            cval=1
        )
    ], random_order=True) 
    ##############################################################
    best_metric = [0]
    for epoch in range(0, args.epochs):
        net.train()
        loss_accum=[]
        all_labels = []
        all_distances = []
        tq = tqdm(enumerate(dataloader))
        if args.train == "1N":
            tq.set_description(f"1N == [GPU {gpu_id} - TRAIN] Epoch {epoch} |")
            for iter, (source, targets)  in tq:
                source, lbl = source.to(gpu_id), targets.to(gpu_id)
                feat1 = net(source)
                logit1 = fc(feat1)
                logit = loss_arcface(logit1, lbl)
                loss = loss_crossentropy(logit, lbl)  # torch.nn.functional.cross_entropy(logit, lbl)
                loss_accum.append(loss.item())

                loss.backward()
                optim.step()
                optim.zero_grad()
                tq.set_postfix( loss='{:.3f}'.format(sum(loss_accum)/len(loss_accum))  , iter = f"{iter}", total=f"{len(dataloader)}")
        else:         #  "11":
            tq.set_description(f"11 == [GPU {gpu_id} - TRAIN] Epoch {epoch} |")
            for iter, (source, source2, targets)  in tq:
                targets = torch.tensor(targets)
                source, source2, lbl = source.to(gpu_id), source2.to(gpu_id), targets.to(gpu_id)
        
                feat1 = net(source)
                feat2 = net(source2)
                if args.score_type == "cosine":
                    distances = F.cosine_similarity(feat1, feat2, dim=1)
                    logit     = distances
                if args.score_type == "eucli":
                    distances = F.pairwise_distance(feat1, feat2, keepdim = True)
                    logit     = distances
                else: 
                    distances = torch.norm(feat1 - feat2, dim=1)
                    distances = torch.square(distances)
                    logit = 1-F.sigmoid(distances)
                loss = contrastive_loss(logit, lbl)  # torch.nn.functional.cross_entropy(logit, lbl)
                loss_accum.append(loss.item())

                loss.backward()
                optim.step()
                optim.zero_grad()
                
                all_labels.extend(lbl.cpu().detach().numpy())
                all_distances.extend(logit.cpu().detach().numpy())
                tq.set_postfix( loss='{:.3f}'.format(loss.item()),  losses = '{:.3f}'.format(sum(loss_accum)/len(loss_accum)), iter = f"{iter}", total=f"{len(dataloader)}")
            if args.score_type == "cosine":
                thresholds, th, auc_score, eer, acc, metrics, fpr, tpr = auc_loop(all_distances, all_labels, pos_label=args.pos_lbl)
            else: 
                thresholds, th, auc_score, eer, acc, metrics, fpr, tpr = auc_loop(all_distances, all_labels, pos_label=args.pos_lbl)
            print(f"TRAIN AUC: {auc_score:.04f}, EER: {eer:.04f}, Acc: {acc:.04f}, Precision: {metrics[0]:.04f}, Recall: {metrics[1]:.04f}, F1s: {metrics[2]:.04f} :{th} \n")
        tq.close()
        ##############################################################
        all_labels = []
        all_distances = []
        loss_accum = []
        net.eval()
        with torch.no_grad():
            tq = tqdm(enumerate(dataloader_val))
            tq.set_description(f"11 ** [GPU {gpu_id} - EVAL ] Epoch {epoch} |")
            for iter, (source, source2, targets)  in tq:
                targets = torch.tensor(targets)
                source, source2, lbl = source.to(gpu_id), source2.to(gpu_id), targets.to(gpu_id)
            
                feat1 = net(source)
                feat2 = net(source2)

                if args.score_type == "cosine":
                    distances = F.cosine_similarity(feat1, feat2, dim=1)
                    # logit = F.sigmoid(distances)
                    logit     = distances
                    # logit = F.softmax(distances)
                if args.score_type == "eucli":
                    distances = F.pairwise_distance(feat1, feat2, keepdim = True)
                    logit     = distances
                else: 
                    distances = torch.norm(feat1 - feat2, dim=1)
                    distances = torch.square(distances)
                    logit = 1-F.sigmoid(distances)

                loss = contrastive_loss(logit, lbl)  # torch.nn.functional.cross_entropy(logit, lbl)
                loss_accum.append(loss.item())
                
                all_labels.extend(lbl.cpu().numpy())
                all_distances.extend(logit.cpu().numpy())
                tq.set_postfix( loss='{:.3f}'.format(sum(loss_accum)/len(loss_accum))  , iter = f"{iter}", total=f"{len(dataloader)}")
            tq.close()
            # if args.score_type == "cosine":
            #     thresholds, th, auc_score, eer, acc, metrics, fpr, tpr = auc_loop_pred(all_distances, all_labels, pos_label=args.pos_lbl)
            #     print(f"EVAL AUC: {auc_score:.04f}, EER: {eer:.04f}, TPR: {tpr:.04f}, FPR: {fpr:.04f}, Acc: {acc:.04f}, Precision: {metrics[0]:.04f}, Recall: {metrics[1]:.04f}, F1s: {metrics[2]:.04f} :{th} \n")
            # else: 
            thresholds, th, auc_score, eer, acc, metrics, fpr, tpr = auc_loop(all_distances, all_labels, pos_label=args.pos_lbl)
            print(f"EVAL AUC: {auc_score:.04f}, EER: {eer:.04f}, Acc: {acc:.04f}, Precision: {metrics[0]:.04f}, Recall: {metrics[1]:.04f}, F1s: {metrics[2]:.04f} :{th} \n\n")

        
        ##############################################################
        if auc_score*100 > best_metric[0]:
            np.save(f'/media/data2/jiwon/pred_tmp{epoch}_{auc_score:.04f}_{eer:.04f}.npy', all_distances)
            np.save(f'/media/data2/jiwon/lb_tmp{epoch}_{auc_score:.04f}_{eer:.04f}.npy',   all_labels)
            print(f"===BEST=== AUC: {auc_score:.04f}, EER: {eer:.04f}, TPR: {tpr}, FPR: {fpr} Acc: {acc:.04f}, Precision: {metrics[0]:.04f}, Recall: {metrics[1]:.04f}, F1s: {metrics[2]:.04f} TH:{th} \n")
            best_metric = [auc_score*100, eer*100, acc*100, metrics[0]*100, metrics[1]*100, metrics[2]*100, fpr, tpr, th]
        ACC, EER, AUC, PRE, F1S, TAR, FAR, THR = best_metric[2], best_metric[1], best_metric[0], best_metric[3], best_metric[5], best_metric[6], best_metric[7] , best_metric[8]
        out_results = pd.concat([out_results, pd.DataFrame({
                                            "Epoch":epoch, 
                                            "Acc": [np.round(ACC,2)], 
                                            "EER": [np.round(EER,2)], 
                                            "AUC": [np.round(AUC,2)],
                                            "PRE": [np.round(PRE,2)],
                                            "F1S": [np.round(F1S,2)],
                                            "TAR": [TAR],
                                            "FAR": [FAR],
                                            "THR": [THR],
                                            })])
        ##############################################################
    return ACC, EER, AUC, PRE, F1S, TAR, FAR, THR, out_results
    ##############################################################

          
          
def test_folders(args):
    """
    Test all the generated data and export them into csv file with 4 columns
    - Dataset
    - ACC
    - ACC @best
    - AUC
    """
    print("TRAIN UPON SETTING", args.name)
    data_path1 = 'The path of the first session of polyU dataset'
    data_path2 = 'The path of the second session of polyU dataset'
    # R, L = args.resize, args.resize
    ##############################################################
    session1 = []
    session2 = [] 
    for i in range(len(os.listdir(data_path1))):
        for j in range(6):
            tmp_path = data_path1 + os.listdir(data_path1)[i] + '/p' + str(j+1) + args.expansion
            if j == 0:
                session1.append([])
            tmp_list = session1[i]
            tmp_list.append(tmp_path)
            session1[i] = tmp_list
            
    for file in os.listdir(data_path2):
        tmp_list = []
        for sub in os.listdir(data_path2 + file):
            tmp_path = data_path2 + file + '/'+ sub
            tmp_list.append(tmp_path) 
        session2.append(tmp_list)

    print("==============first session:", len(session1), ", second session:", len(session2))
    ##############################################################
    DB_LIST_T_G = []
    DB_LIST_T_I = []
    DB_LIST_TRAIN = []
    ## first_set 에서 136개 추출
    ## 6C2 X 136 = 2040
    id = 0
    for i in range(0, 136):
        for j in range(0, 5):
            for k in range(j+1, 6):
                photo1 = session1[i][j]
                photo2 = session1[i][k]
                DB_LIST_T_G.append([photo1, photo2, 1.0])
        for j in range(0, 6):
            photo1 = session1[i][j]
            DB_LIST_TRAIN.append([photo1, id+0.0])
        id+=1
    for i in range(0, 136):
        for j in range(0, 5):
            for k in range(j+1, 6):
                photo1 = session1[i][j]
                photo2 = session1[i][k]
                DB_LIST_T_G.append([photo2, photo1, 1.0])     
    ## 104
    ## 6C2 X 104 = 1560
    for i in range(0, 160):
        for j in range(1, 5):
            for k in range(j+1, 6):
                photo1 = session2[i][j]
                photo2 = session2[i][k]
                DB_LIST_T_G.append([photo1, photo2, 1.0])
        for j in range(0, 6):
            photo1 = session2[i][j]
            DB_LIST_TRAIN.append([photo1, id+0.0])
        id+=1
    for i in range(0, 160):
        for j in range(1, 5):
            for k in range(j+1, 6):
                photo1 = session2[i][j]
                photo2 = session2[i][k]
                DB_LIST_T_G.append([photo2, photo1, 1.0])
    for i in range(0, 136):
        for j in range(i+1, 137):
            k1, k2 = random.randint(0,5), random.randint(0,5)
            photo1 = session1[i][k1]
            photo2 = session1[j][k2]
            DB_LIST_T_I.append([photo1, photo2, 0.0])
    for i in range(0, 159):
        for j in range(i+1, 160):
            k1, k2 = random.randint(0,5), random.randint(0,5)
            photo1 = session2[i][k1]
            photo2 = session2[j][k2]
            DB_LIST_T_I.append([photo1, photo2, 0.0])
    DB_LIST_T_G = np.array(DB_LIST_T_G)
    DB_LIST_T_I = np.array(DB_LIST_T_I)
    np.random.shuffle(DB_LIST_T_G) 
    np.random.shuffle(DB_LIST_T_I) 
    ##############################################################
    DB_LIST_TEST = []
    ## first_set 에서 200개 추출
    for i in range(136, 336):
        for j in range(0, 5):
            for k in range(j+1, 6):
                photo1 = session1[i][j]
                photo2 = session1[i][k]
                DB_LIST_TEST.append([photo1, photo2, 1.0])

    for i in range(136, 335):
        for j in range(i+1, 336):
            k1, k2 = random.randint(0,5), random.randint(0,5)
            photo1 = session1[i][k1]
            photo2 = session1[j][k2]
            DB_LIST_TEST.append([photo1, photo2, 0.0])
            
    DB_LIST_TEST = np.array(DB_LIST_TEST)                
    np.random.shuffle(DB_LIST_TEST)

    print("==============", len(DB_LIST_T_G), len(DB_LIST_T_I), len(DB_LIST_TEST), f"IDs: {id}")
    ##############################################################
    trainset = TrainPairDataset(pospairlist=DB_LIST_T_G, negpairlist=DB_LIST_T_I, degree=0, size=args.resize)
    train1Nset = FingerprintDataset(DB_LIST_TRAIN, degree=0, size=args.resize)
    testset  = TestPairDataset(DB_LIST_TEST, 0, args.resize)

    dataloader     = DataLoader(trainset,   batch_size=args.batch, shuffle=True, drop_last=True, num_workers=5)
    dataloader1N   = DataLoader(train1Nset, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=5)
    dataloader_val = DataLoader(testset,    batch_size=args.batch, shuffle=False, drop_last=False, num_workers=5)
    ##############################################################
    out_results = pd.DataFrame({
        "Epoch":[], "Acc": [], "EER": [], "AUC": [], "PRE":[], "F1S":[], "TAR":[], "FAR":[], "THR":[]
    })#          
    return args, out_results, dataloader, dataloader_val, dataloader1N
    ##############################################################
       
      
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process to test a model with a folder of images')
    parser.add_argument('--expansion', default='.bmp', type=str)
    parser.add_argument('--score_type', default='cosine', type=str)
    parser.add_argument('--model', default='auth', type=str)
    parser.add_argument('--resize', default=224, type=int)
    parser.add_argument('--batch', default=256, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--train', default='11', type=str)
    args = parser.parse_args()
    
    args.name = f'{args.model}-{args.train}-{args.score_type}_{args.epochs}-{args.lr}-{args.batch}'
    args, out_results, dataloader, dataloader_val, dataloader1N = test_folders(args)
    ACC, EER, AUC, PRE, F1S, TAR, FAR, THR, out_results = main(args, out_results, dataloader, dataloader_val, dataloader1N)
    
    out_results.to_csv(f"../{args.name}__{args.score_type}_{AUC:.04f}_{EER:.04f}.csv", index=False)
