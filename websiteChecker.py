#from torch_model import no_def_model
from torch_model_modified import no_def_model
import torch
import numpy as np


class WebsiteIndex:
    # number: test class number, data: xdata as a list for each website, classNum: classifier target classes number
    # trueTarg: true target num
    def __init__(self, number, data, classNum, trueTarg):
        self.number = number
        self.data = data
        self.classNum = classNum
        self.target = trueTarg

    def classify(self):
        model = no_def_model(self.classNum)
        # model.load_state_dict(torch.load('torchData/No_def_replacedFilteredOld.pkl'))
        model.load_state_dict(torch.load('torchData/No_def_replacedFiltered1Mymodelmodified.pkl'))
        model.eval()

        pred_dict = {}

        #for i in range(self.number):
            #predictions = []
            #x = self.data[i]
            #for j in range(x.shape[0]):

                #d = torch.unsqueeze(x[j], 0)
                #logits = model(d)
                #scores = np.array(logits[0].data.view(-1))
                #predictions.append(np.argmax(scores))
            #pred_dict[self.target[i]] = predictions
        #return pred_dict

        for i,k in enumerate(self.target):
            predictions = []
            x = self.data[i]
            for j in range(x.shape[0]):
            #for j in range(40):
                d = torch.unsqueeze(x[j], 0)
                logits = model(d)
                scores = np.array(logits[0].data.view(-1))
                predictions.append(np.argmax(scores))
            pred_dict[k] = predictions
        return pred_dict

