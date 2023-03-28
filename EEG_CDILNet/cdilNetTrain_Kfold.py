from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.preprocessing import create_windows_from_events

#modle import
import torch
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, Deep4Net ,EEGNetv4,TCN,EEGResNet
from model.TCNet import TCNet
from model.AMSINet import AMSINet
from model.MIEEGNet import MIEEGNet

from model.EEG_CDILNet_tiny import EEG_CDILNet

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier

from braindecode.datasets import HGD
import pandas as pd
import joblib  
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, precision_score, recall_score, f1_score   #构建混淆矩阵 kappa 等
from braindecode.visualization import plot_confusion_matrix 
import csv
channelDrop = ['AF3', 'AF4', 'AF7', 'AF8', 'AFF1', 'AFF2', 'AFF5h', 'AFF6h', 'AFp3h', 'AFp4h', 'AFz', 'Cz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FFT7h', 'FFT8h', 'FT10', 'FT7', 'FT8', 'FT9', 'FTT10h', 'FTT7h', 'FTT8h', 'FTT9h', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'I1', 'I2', 'Iz', 'M1', 'M2', 'O1', 'O2', 'OI1h', 'OI2h', 'Oz', 'P1', 'P10', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'PO10', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'PO9', 'POO10h', 'POO3h', 'POO4h', 'POO9h', 'POz', 'PPO1', 'PPO10h', 'PPO2', 'PPO5h', 'PPO6h', 'PPO9h', 'Pz', 'T7', 'T8', 'TP7', 'TP8', 'TPP10h', 'TPP7h', 'TPP8h', 'TPP9h', 'TTP7h', 'TTP8h'] 

K_fold = 10 # K fold

def KFtrain(name, Parms, datasChoose = 0, records = True):
    subNum = 9 if datasChoose == 0 else 14
    subject_all = [i+1 for i in range(subNum)]
    maxList = []
    for subject_id in subject_all:
        #loading
        if datasChoose == 0:
            dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id]) 
        else:
            dataset = HGD(subject_id)
        #Preprocessing
        low_cut_hz = 4.  # low cut frequency for filtering
        high_cut_hz = 38.  # high cut frequency for filtering
        # Parameters for exponential moving standardization
        factor_new = 1e-3
        init_block_size = 1000
        if datasChoose == 0:
            preprocessors = [
            Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
            Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
            Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
            Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                        factor_new=factor_new, init_block_size=init_block_size)
            ]
        else:
            preprocessors = [
            Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
            Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
            Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
            Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                        factor_new=factor_new, init_block_size=init_block_size),
            Preprocessor('drop_channels', ch_names=channelDrop), # Remove the extra 84 channels of HGD
            Preprocessor('resample', sfreq=250 )  # downshample 250hz
            ]


        # Transform the data
        preprocess(dataset, preprocessors)
        #Cut Compute Windows
        trial_start_offset_seconds = -0.5
        # Extract sampling frequency, check that they are same in all datasets
        sfreq = dataset.datasets[0].raw.info['sfreq']
        assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
        # Calculate the trial start offset in samples.
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
        # Create windows using braindecode function for this. It needs parameters to define how
        # trials should be used.
        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            preload=True,
        )


        #K-fold cross-training has no test set, and the validation set is divided in it
        train_set = windows_dataset  

        from skorch.helper import predefined_split, SliceDataset
        from sklearn.model_selection import train_test_split
        from torch.utils.data import Subset
        import numpy as np
        X_train = SliceDataset(train_set, idx=0)
        y_train = np.array([y for y in SliceDataset(train_set, idx=1)])

        #Create model
        cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
        device = 'cuda' if cuda else 'cpu'
        if cuda:
            torch.backends.cudnn.benchmark = True

        seed = 20220820
        set_random_seeds(seed=seed, cuda=cuda)

        n_classes = 4
        # Extract number of chans and time steps from dataset
        n_chans = train_set[0][0].shape[0]
        input_window_samples = train_set[0][0].shape[1]

        step = int(len(X_train.indices_) / K_fold) + 1 
        testAcc = []
        knum =0
        for fold in range(K_fold):
            inchannel = 22 if dataType == 0 else 44
            model = EEG_CDILNet(inChannel=inchannel, F1=Parms[0], KE = Parms[1], D = Parms[2], pe = Parms[3], hiden=int(Parms[4]), layer = int(Parms[5]), ks =int(Parms[6]))
            # Send model to GPU
            if cuda:
                model.cuda()
            
            lr = 0.005
            weight_decay = 0.5 * 0.001

            batch_size = 32
            n_epochs = 200
            if fold == K_fold-1:
                val_indices = X_train.indices_[fold*step: -1]
            else:
                val_indices = X_train.indices_[fold*step: fold*step + step]
            train_indices = list(set(X_train.indices_) - set(val_indices))
            train_indices = np.array(train_indices)

            train_subset = Subset(train_set, train_indices)
            val_subset = Subset(train_set, val_indices)
            clf = EEGClassifier(
                model,
                criterion=torch.nn.NLLLoss,
                optimizer=torch.optim.AdamW,
                train_split=predefined_split(val_subset),
                optimizer__lr=lr,
                optimizer__weight_decay=weight_decay,
                batch_size=batch_size,
                warm_start=False,#warm boot
                callbacks=[
                    "accuracy", 
                    ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
                ],
                device=device,
            )
            clf.fit(train_subset, y=None, epochs=n_epochs)


            # Extract loss and accuracy values for plotting from history object
            results_columns = ['epoch','train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
            df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,index=clf.history[:, 'epoch'])

            Max = df.loc[[df['train_accuracy'].idxmax(),df['valid_accuracy'].idxmax()]]
            indexMax = Max['valid_accuracy'].idxmax()
            try:
                temMax = float(Max['valid_accuracy'][indexMax])
            except:
                temMax = 0


            import joblib  
            kfoldBest = joblib.load(r'bestKfold.pkl') #Take out the historical best for comparison    
            if kfoldBest[subject_id-1][fold] < temMax:
                kfoldBest[subject_id-1][fold] = temMax
            joblib.dump(kfoldBest,r'bestKfold.pkl')

            testAcc.append(round(temMax,3))
            if temMax != 0:
                knum += 1

            Max['subjectId'] = [subject_id]*2
            Max['time'] = [int(name.split('_')[0])]*2
            if records :
                recordNameMax = "trainDataMax.csv" if datasChoose == 0 else  "trainDataMax_HGD.csv"
                recodeName = "trainData.csv" if datasChoose == 0 else "trainData_HGD.csv"
                Max.round(decimals = 3).to_csv(recordNameMax, mode='a',index=False, sep=',')
                df.round(decimals = 3).to_csv(recodeName, mode='a', index=False, sep=',')  

        maxList.append(round(sum(testAcc)/knum,3))

    maxList.insert(0,name)
    recordValid(maxList,select = datasChoose)



def recordValid(listData,select = 0):
    import csv
    if select == 0:
        filename = r'contrast_BCIC_Kfold.csv'
    else:
        filename = r'contrast_HGD_Kfold.csv'
    f = open(filename, 'a',newline='')
    writer = csv.writer(f)
    writer.writerow(listData)
    f.close()


def gridsearch(totalParamenter, temp, start, dataType):
    if len(temp) == len(totalParamenter) and start == len(totalParamenter):
        import time #
        Time = time.strftime("%m%d%H",time.localtime())
        Time += '_BCIC' if dataType == 0 else '_HGD' 
        t = [str(i) for i in temp]
        Time += '_F1:' + t[0] +'_Ke:' + t[1] + '_D:' + t[2] + '_pe:'+ t[3] + '_H:' + t[4] + '_L:' + t[5] + '_Ks:' + t[6]
        KFtrain(Time,Parms=temp, datasChoose=dataType)
        #print(Time)
        return 
    for param in totalParamenter[start]:
        temp.append(param)
        gridsearch(totalParamenter, temp, start+1, dataType)
        temp.pop()

'''
To evaluate the predictive performance and generalization ability of our proposed model, we performed experiments using a 10x cross-validation method.
We combine the training and test sets of BCIIV2a. There were 576 trials per participant, and they were then randomly divided into 10 equal parts.
Each run uses 9 subsets as the training set and one subset as the test set, i.e. 518 and 58 experimental trials for training and testing. 
The final accuracy is obtained by averaging 10 times the optimal value.
'''

if __name__ == "__main__":
    #(F1=12, KE = 32, D = 2, pe = 0.2, hiden=20, layer = 2, ks =5)
    #                   F1       KE    D    pe     hiden  layer   ks
    #
    dataType = 0 #0:BCIC 1:HGD
    KFTrain = True
    if KFTrain:
        parma = [24, 48, 2, 0.2, 24, 2, 3]
        for i in range(10):
            import time 
            Time = time.strftime("%m%d%H",time.localtime())+'_Kfold=' + str(K_fold) + '_'
            Time += '_BCIC' if dataType == 0 else '_HGD' 
            # print(int(Time.split('_')[0]))
            KFtrain(Time, parma, dataType)
    
    #Read the best K-fold crossover results
    import joblib  
    kfoldBest = joblib.load(r'bestKfold.pkl') #Take out the historical best for comparison 
    ret = []
    for subj in kfoldBest:
        ret.append(sum(subj)/K_fold)
    print(ret)