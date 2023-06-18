# Import the braindecode 
from braindecode.datasets import MOABBDataset
from braindecode.datasets import HGD
from braindecode.preprocessing import (exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.preprocessing import create_windows_from_events
import torch
from braindecode.util import set_random_seeds

from braindecode import EEGClassifier


# Import the skorch 
from skorch.callbacks import EarlyStopping # 
from skorch.callbacks import Checkpoint
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from skorch.helper import predefined_split, SliceDataset
from skorch.callbacks import Checkpoint  # Record the training optimal value 
from skorch.callbacks import Freezer # freeze

# Import model for baseline 
from model.TCNet import TCNet
from model.AMSINet import AMSINet
from model.MIEEGNet import MIEEGNet
from braindecode.models import ShallowFBCSPNet, Deep4Net ,EEGNetv4,TCN,EEGResNet
from model.EEG_CDILNet_tiny import EEG_CDILNet
#from model.EEG_CDILNet_covd1_v2 import EEG_CDILNet
#from model.EEG_CDILNet_tiny_flatten import EEG_CDILNet

# EEG channels that are not needed in the HGD dataset
channelDrop = ['AF3', 'AF4', 'AF7', 'AF8', 'AFF1', 'AFF2', 'AFF5h', 'AFF6h', 'AFp3h', 'AFp4h', \
               'AFz', 'Cz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FFT7h', 'FFT8h',  \
               'FT10', 'FT7', 'FT8', 'FT9', 'FTT10h', 'FTT7h', 'FTT8h', 'FTT9h', 'Fp1', 'Fp2',  \
               'Fpz', 'Fz', 'I1', 'I2', 'Iz', 'M1', 'M2', 'O1', 'O2', 'OI1h', 'OI2h', 'Oz', 'P1', \
                'P10', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'PO10', 'PO3', 'PO4', 'PO5', \
                'PO6', 'PO7', 'PO8', 'PO9', 'POO10h', 'POO3h', 'POO4h', 'POO9h', 'POz', 'PPO1', 'PPO10h', \
                'PPO2', 'PPO5h', 'PPO6h', 'PPO9h', 'Pz', 'T7', 'T8', 'TP7', 'TP8', 'TPP10h', 'TPP7h', 'TPP8h',\
                  'TPP9h', 'TTP7h', 'TTP8h'] 

import pandas as pd
import joblib  # 
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, precision_score, recall_score, f1_score   
from braindecode.visualization import plot_confusion_matrix 
import csv
import numpy as np


def loadDatasets(type = 0,subject_id = 3):
    if type == 0:
        dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id]) 
    elif type == 1:
        dataset = HGD(subject_id)
    elif type == 2:
        dataset = MOABBDataset(dataset_name="BNCI2014004", subject_ids=[subject_id]) 
    return dataset

def preProcess(type = 0, low_cut_hz = 4, high_cut_hz = 38, factor_new = 1e-3, init_block_size = 1000):
    if type == 0 or type == 2:
            preprocessors = [
            Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
            Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
            Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter [4,38] or [0,38]
            Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                        factor_new=factor_new, init_block_size=init_block_size),
            #Preprocessor('resample', sfreq=150 )  # 150Hz or 180hz or 250hz
            ]
    else:
        preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
        Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
        Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                    factor_new=factor_new, init_block_size=init_block_size),
        Preprocessor('drop_channels', ch_names=channelDrop), # Remove the extra 84 channels of HGD
        Preprocessor('resample', sfreq=250 )  # Resampling is 250hz
        ]
    return preprocessors

def splitDataset(type, windows_dataset):
    """
    Divide the training and test sets
    For different researchers, the BCIIV2a dataset has two ways to divide the
        training and test sets : 5:5 and 8:2
    The other two datasets are divided by default.
    """
    if type == 0: # BCIIV2a
        spliType = 1   #spliType == 1 in the paper
        if spliType == 1:  
        # training : test =  5 : 5  by default
            splitted = windows_dataset.split('session')
            train_set = splitted['session_T']
            test_set = splitted['session_E']
        elif spliType == 2:
        # training : test =  8 : 2   test
            import random
            index_train = random.sample([i for i in range(12)],9)
            index_valid = list(set([i for i in range(12)]) - set(index_train))
            train_set = windows_dataset.split(index_train)['0']
            test_set = windows_dataset.split(index_valid)['0']
    elif type == 1: # HGD
        splitted = windows_dataset.split('run')
        train_set = splitted['train']
        test_set = splitted['test']
    elif type == 2: # # BCIIV2b
        index_train = [2,3,4]
        index_valid = [0,1]
        train_set = windows_dataset.split(index_train)['0']
        test_set = windows_dataset.split(index_valid)['0']
    return train_set, test_set

def trainPrarms():
    # try lr = 0.001 0.005 0.01  
    if dataSetType == 0:
        inchannel = 22
        outclass = 4
        lr = 0.1 *0.05
        n_epochs = 500
    elif dataSetType == 1:
        inchannel = 44
        outclass = 4
        lr = 0.01 *0.1
        n_epochs = 500
    elif dataSetType == 2:
        inchannel = 3
        outclass = 2
        lr = 0.9 *0.01
        n_epochs = 500

    weight_decay = 0.5 * 0.001
    batch_size = 64
    
    return inchannel, outclass, lr, n_epochs, weight_decay, batch_size

def parFilename(subject_id):
    if dataSetType == 0:
        fname = r'./bestParams/best_model_BCICA_'
    elif dataSetType == 1:
        fname = r'./bestParams/best_model_HGD_'
    elif dataSetType == 2:
        fname = r'./bestParams/best_model_BCICB_'
    f_params_name = fname + str(subject_id) + '.pt'
    return f_params_name
    
def train(name, Parms, dataSetType = 0, records = True, optimalPra = False,solo = False, soloSub = 14):
    """
    This is the function used to train and test the model.
    Parameters:
    name - The file name where the training results are stored
    Parms - The parameters used in this training
    dataSetType - The dataset used for this training
    records - Whether to record the results of this training
    optimalPra - Whether to save the best model
    solo - Whether to train on a single subject, and if false, train on all subjects
    soloSub - Choose which one subject to train on individually, ask solo = true

    """
    subNum = 9 if dataSetType == 0 or dataSetType == 2 else 14 # number of subjects
    subject_all = [i+1 for i in range(subNum)]
    maxList = [0 for i in range(subNum)]
    for subject_id in subject_all:
        if solo: # Only one subject is trained
            if subject_id != soloSub:
                continue
        #loading
        dataset = loadDatasets(type = dataSetType,subject_id = subject_id)
        #Preprocessing
        low_cut_hz = 4.  # low cut frequency for filtering    
        high_cut_hz = 38.  # high cut frequency for filtering

        # Parameters for exponential moving standardization
        factor_new = 1e-3
        init_block_size = 1000

        # preprocess
        preprocessors = preProcess(type = dataSetType,low_cut_hz=low_cut_hz, high_cut_hz=high_cut_hz, \
                                    factor_new=factor_new, init_block_size=init_block_size)

        # Transform the data
        preprocess(dataset, preprocessors)

        #Cut Compute Windows  [-0.5, 4]s
        trial_start_offset_seconds = -0.5    
        trial_end_offset_seconds = 0
        # Extract sampling frequency, check that they are same in all datasets
        sfreq = dataset.datasets[0].raw.info['sfreq']
        assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
        # Calculate the trial start offset in samples.
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
        trial_end_offset_samples = int(trial_end_offset_seconds * sfreq)
        # Create windows using braindecode function for this. It needs parameters to define how
        # trials should be used.
        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=trial_end_offset_samples,
            preload=True,
        )

        #Split dataset into train and valid
        train_set, test_set = splitDataset(type = dataSetType, windows_dataset = windows_dataset)
        
        X_train = SliceDataset(train_set, idx=0)
        y_train = np.array([y for y in SliceDataset(train_set, idx=1)])
        train_indices, val_indices = train_test_split(
            X_train.indices_, test_size=0.2, shuffle=False
        )
        train_subset = Subset(train_set, train_indices)
        val_subset = Subset(train_set, val_indices)

        #Create model
        cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
        device = 'cuda' if cuda else 'cpu'
        if cuda:
            torch.backends.cudnn.benchmark = True

        seed = 202201010
        set_random_seeds(seed=seed, cuda=cuda)

        # Extract number of chans and time steps from dataset
        n_chans = train_set[0][0].shape[0]
        input_window_samples = train_set[0][0].shape[1]
        
        # training-time parameter
        inchannel, outclass, lr, n_epochs, weight_decay, batch_size = trainPrarms()
        # Use your own model and correspond to your own parameters
        model = EEG_CDILNet(inChannel=inchannel, outClass=outclass,F1=Parms[0], KE = Parms[1], D = Parms[2], pe = Parms[3], hiden=int(Parms[4]), layer = int(Parms[5]), ks =int(Parms[6]), pool = int(Parms[7]))
        if cuda:
            model.cuda()
        
        # Record the best parameters for training
        f_params_name = parFilename(subject_id)
        
        checkpoint = Checkpoint( f_params=f_params_name, monitor='valid_loss_best',load_best=True)
        early_stop = EarlyStopping(monitor='valid_accuracy',lower_is_better=False,patience=50)
        # train model
        clf = EEGClassifier(
            model,
            criterion=torch.nn.NLLLoss,
            optimizer=torch.optim.AdamW,
            train_split=predefined_split(val_set),  # using valid_set for validation     
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay,
            batch_size=batch_size,
            callbacks=[
                "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
                early_stop,
                checkpoint,  
            ],
            device=device,
        )
        clf.fit(train_subset, y=None, epochs=n_epochs)

        # score the Model after training (optional)
        y_test = test_set.get_metadata().target
        test_acc = clf.score(test_set, y=y_test)

        #Write the training results to a file
        # Extract loss and accuracy values for plotting from history object
        results_columns = ['epoch','train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
        df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,index=clf.history[:, 'epoch'])
        
        #Extract the two largest rows of the test and training sets
        Max = df.loc[[df['train_accuracy'].idxmax(),df['valid_accuracy'].idxmax()]]
        indexMax = Max['valid_accuracy'].idxmax()
        try:
            temMax = float(Max['valid_accuracy'][indexMax])
        except:
            temMax = 0
        maxList[subject_id - 1] = round(temMax,3)
        Max['subjectId'] = [subject_id]*2
        Max['time'] = [int(name.split('_')[0])]*2


        flname = "BCIIVa.csv" if dataSetType == 0 else "HGD.csv" if dataSetType == 1 else "BCIIVb.csv"  
        if records :
            recordNameMax = "recordMax_" + flname
            recodeName = "record_" + flname
            Max.round(decimals = 3).to_csv(recordNameMax, mode='a',index=False, sep=',')
            df.round(decimals = 3).to_csv(recodeName, mode='a', index=False, sep=',')  

        if optimalPra:
            import joblib  
            (bestBcica, bestHgd,bestBcicb) = joblib.load(r'bestVaild.pkl') # Take out the historical best for comparison
            # get the targets
            y_true = test_set.get_metadata().target
            y_pred = clf.predict(test_set)
            if dataSetType == 0:
                filename = r'./confusionMat/CM' + '_BCIC_' + str(subject_id) + '.pkl'
                if bestBcica[subject_id-1] < test_acc:
                    bestBcica[subject_id-1] = test_acc
                    joblib.dump((y_true,y_pred), filename)  
            elif dataSetType == 1:
                filename = r'./confusionMat/CM' + '_HGD_' + str(subject_id) + '.pkl'
                if bestHgd[subject_id-1] < test_acc:
                    bestHgd[subject_id-1] = test_acc
                    joblib.dump((y_true,y_pred), filename)  
            elif dataSetType == 2:
                filename = r'./confusionMat/CM' + '_BCICb_' + str(subject_id) + '.pkl'
                if bestBcicb[subject_id-1] < test_acc:
                    bestBcicb[subject_id-1] = test_acc
                    joblib.dump((y_true,y_pred), filename)  
            # Save the best parameters that have changed
            joblib.dump((bestBcica,bestHgd,bestBcicb), r'bestVaild.pkl')  

            
    # Save the training results of this round to CSV for  statistics
    maxList.insert(0,name)
    recordValid(maxList,select = dataSetType)

def confusionMatPlot(filename, type = 0):
    #type = 0 BCIC2a  select = 1 HGD type = 0 BCIC2b
    if type == 0:
        labels = ['feet', 'left_hand', 'right_hand', 'tongue']
    elif type == 1:
        labels = ['feet', 'left_hand', 'rest', 'right_hand']
    elif type == 2:
        labels = ['left_hand', 'right_hand']
    (y_true, y_pred) = joblib.load(filename) 
    # generating confusion matrix
    confusion_mat = confusion_matrix(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average= None)
    recall = recall_score(y_true, y_pred, average= None)
    f1 = f1_score(y_true, y_pred, average= 'weighted')
    #print(out)
    # 
    #plot_confusion_matrix(confusion_mat, class_names=labels)
    return confusion_mat, kappa, precision, recall, f1

def PrReF1(dataSetType = 0):
    count = 9 if dataSetType == 0 else 14
    kappas = []
    precisions = []
    recalls = []
    f1s = []
    for i in range(count):
        if dataSetType == 0:
            filename = r'./confusionMat/CM' + '_BCIC_' + str(i+1) + '.pkl' 
        else: 
            filename = r'./confusionMat/CM' + '_HGD_' + str(i+1) + '.pkl'
        confusion_mat, kappa, precision, recall, f1 = confusionMatPlot(filename, dataSetType)
        kappas.append(round(kappa, 3))
        f1s.append(round(f1, 3))
        precisions.append(precision)
        recalls.append(recall)
    kappas.insert(0, 'k-value:')
    f1s.insert(0, 'f1:')

    precisions = list(map(list, zip(*precisions)))
    recalls = list(map(list, zip(*recalls)))

    out_filename = r'out_kprf.csv'
    f = open(out_filename, 'a',newline='')
    writer = csv.writer(f)
    writer.writerow(kappas)
    writer.writerow(f1s)
    for i in range(4):
        writer.writerow(precisions[i])
    for i in range(4):
        writer.writerow(recalls[i])
    f.close()

def recordValid(listData,select = 0):
    import csv
    if select == 0:
        filename = r'contrast_BCIC.csv'
    elif select == 1:
        filename = r'contrast_HGD.csv'
    elif select == 2:
        filename = r'contrast_BCICB.csv'
    f = open(filename, 'a',newline='')
    writer = csv.writer(f)
    writer.writerow(listData)
    f.close()



def conf(dataSetType = 0):
    #Calculates the average confusion matrix
    count = 9 if dataSetType == 0 or dataSetType == 2 else 14
    confusion_mats = []
    for i in range(count):
        if dataSetType == 0:
            type =  '_BCIC_'  
        elif dataSetType == 1: 
            type = '_HGD_' + str(i+1) 
        elif dataSetType == 2:
            type = '_BCICb_' + str(i+1)
        filename = r'./confusionMat/CM' + + str(i+1) + '.pkl'
        confusion_mat, kappa, precision, recall, f1 = confusionMatPlot(filename, dataSetType) 
        confusion_mats.append(confusion_mat)
    outClass = 2 if dataSetType == 2 else 4
    rets =[[0 for i in range(outClass)] for i in range(outClass)]
    for i in range(outClass):
        for j in range(outClass):
            for comf in confusion_mats:
            #    rets[i][j] += comf[i][j] / sum(comf[i])
                rets[i][j] += comf[i][j]
    #ret = [[rets[i][j] / count for j in range(4)] for i in range(4)]
    #print(ret)
    out_filename = r'out_conf.csv'
    f = open(out_filename, 'a',newline='')
    writer = csv.writer(f)
    for row in rets:
        writer.writerow(row)
    f.close()

def gridsearch(totalParamenter, temp, start, dataSetType):
    """
    This function uses the backtracking algorithm to implement grid search of parameters
    totalParamenter - All parameters to be searched
    temp - 
    start - 
    dataSetType - Select the dataset
    """
    if len(temp) == len(totalParamenter) and start == len(totalParamenter):
        import time #
        Time = time.strftime("%m%d%H",time.localtime()) 
        Time += 'Grid_datasets_' + str(dataSetType)  
        t = [str(i) for i in temp]
        Time += '_F1:' + t[0] +'_Ke:' + t[1] + '_D:' + t[2] + '_pe:'+ t[3] + '_H:' + t[4] + '_L:' + t[5] + '_Ks:' + t[6] + '_pool:' + t[7]
        train(Time,Parms=temp, dataSetType=dataSetType)
        return 
    for param in totalParamenter[start]:
        temp.append(param)
        gridsearch(totalParamenter, temp, start+1, dataSetType)
        temp.pop()

if __name__ == "__main__":
    """
    dataSetType - select dataset: 0:BCICa 1:HGD 2BCICb
    There are two ways: grid search parameters, multiple repeated training
        GridCV - grid search parameters
        Train - multiple repeated training
    """
    dataSetType = 0
    GridCV = False
    Train = True

    #conf(dataSetType)
    #PrReF1(dataSetType)
    if Train:

        count = 20
        if dataSetType == 0:
            parma = [24,64, 2, 0.5, 24, 2, 3, 8]
        elif dataSetType == 1:
            parma = [24, 64, 2, 0.5, 24, 2, 3, 8]
        elif dataSetType == 2:
            parma = [12, 64, 2, 0.5, 24, 2, 3, 8]
        for i in range(count):
            import time #
            Time = time.strftime("%m%d%H",time.localtime()) 
            Time += '_datasets_' + str(dataSetType) 
            train(Time, parma, dataSetType)
    
    if GridCV:
        #                    F1       KE      D    pe     hiden  layer   ks  pool
        totalParamenter1 = [[24,32,48], [48,64], [2,4], [0.2,0.5], [24,32], [2,3], [3],[8]]
        temp = []
        gridsearch(totalParamenter1, temp, 0, dataSetType)
