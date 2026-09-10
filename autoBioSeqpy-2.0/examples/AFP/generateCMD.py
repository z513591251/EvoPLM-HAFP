# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 15:35:51 2022

@author: tmp
"""

import os, sys
from itertools import combinations
positiveSeqFile = './examples/AFP/data/potr.txt'
negativeSeqFile = './examples/AFP/data/netr.txt'

# positiveFeaFiles = []
# negativeFeaFiles = []
# dataPath = './data'
# for f in os.listdir(dataPath):
#     if f.startswith('po'):
#         positiveFeaFiles.append('%s/%s' %(dataPath,f))
#     elif f.startswith('ne'):
#         negativeFeaFiles.append('%s/%s' %(dataPath,f))

    
positiveFeaFiles = ['./examples/AFP/data/potr_CTriad.txt', './examples/AFP/data/potr_DC.txt', './examples/AFP/data/potr_modlamp.txt', './examples/AFP/data/potr_PAAComp.txt', './examples/AFP/data/potr_peptide.txt', './examples/AFP/data/potr_peptidy.txt', './examples/AFP/data/potr_ESM2.txt']

negativeFeaFiles = ['./examples/AFP/data/netr_CTriad.txt', './examples/AFP/data/netr_DC.txt', './examples/AFP/data/netr_modlamp.txt', './examples/AFP/data/netr_PAAComp.txt', './examples/AFP/data/netr_peptide.txt', './examples/AFP/data/netr_peptidy.txt', './examples/AFP/data/netr_ESM2.txt']

# spcLenList = []

feaDict = {}
for i in range(len(positiveFeaFiles)):
    posFile = positiveFeaFiles[i]
    negFile = negativeFeaFiles[i]
    feaName = os.path.split(posFile)[-1].split('_')[-1].split('.')[0]
    modelName = './examples/AFP/model/%s.py' %feaName
    # assert os.path.exists(modelName)
    feaDict[feaName] = (posFile,negFile,modelName,200)

repeatTime = 5

cmdTemp = 'python running.py --dataType protein %s --dataEncodingType dict %s --dataTrainFilePaths %s --dataTrainLabel 1 0%s --dataSplitScale 0.8 --modelLoadFile examples/AFP/model/transformer.py %s --verbose 1 --showFig 0 --outSaveFolderPath %s --savePrediction 1 --saveFig 1 --batch_size 256 --epochs 20 --shuffleDataTrain 1 --spcLen 50 %s --noGPU 1 --paraSaveName parameters.txt --optimizer optimizers.Adam(lr=0.001,amsgrad=False,decay=False) --dataTrainModelInd 0 0%s'
errCMDList = []
feaNames = list(feaDict.keys())

for repeatNum in range(repeatTime):
    for combNum in range(11):
        combIterObj = combinations(feaNames, combNum + 1)
        
        for combIter in combIterObj:
            dataType = ''
            dataEncodingType = ''
            dataTrainFilePaths = positiveSeqFile+' '+negativeSeqFile #oriFile needed
            dataTrainLabel = ''
            modelLoadFile = ''
            outSaveFolderPath = 'outs%d/' %repeatNum
            spcLen = ''
            dataTrainModelInd = ''
            modelCount = 1
            
            for feaName in combIter:
                _posFile,_negFile,_modelName,_spcLen = feaDict[feaName]
                
                dataType += ' other'
                dataEncodingType += ' other'
                dataTrainFilePaths += ' ' + _posFile + ' ' + _negFile
                dataTrainLabel += ' 1 0'
                modelLoadFile += ' ' + _modelName
                outSaveFolderPath += feaName + '__'
                spcLen += ' %d' %_spcLen
                dataTrainModelInd += ' %d %d' %(modelCount,modelCount)
                modelCount += 1
            outSaveFolderPath = outSaveFolderPath[:-2]
            cmd = cmdTemp %(dataType, dataEncodingType, dataTrainFilePaths, dataTrainLabel, modelLoadFile, outSaveFolderPath, spcLen, dataTrainModelInd)
            print('#' * 10)
            print(cmd)
            isErr = os.system(cmd)
            if isErr:
                errCMDList.append('%d::%s' %(repeatNum,cmd))
print('#'*50)
print('err:')
for cmd in errCMDList:
    print('*'*10)
    print(cmd)
