# import utility_torch as ut
# import torch
# from torch_model import no_def_model
import lrp_func as lp
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pyshark


def generate_data(sourcefilename):
    """This method takes a text file with packet directions and converts the direction data into an array format that
    the trained model can take as input"""
    valueArray = []
    with open(sourcefilename, 'r') as f:
        line = f.readline()
        valueArray.append(int(line))
        while line:
            line = f.readline()
            if not line:
                break
            valueArray.append(int(line))
    length = len(valueArray)
    if length <= 5000:
        lengthDiff = 5000 - length
        for j in range(lengthDiff):
            valueArray.append(0)
    else:
        valueArray = valueArray[0:5000]
    #lengthDiff = 5000-length
    #for i in range(lengthDiff):
        #valueArray.append(0)

    valueArray = np.array(valueArray)
    valueArray = valueArray[np.newaxis, np.newaxis, :]
    valueArray = torch.Tensor(valueArray)
    return valueArray


class WebsiteInstance:
    """This class creates a Webpage Instance which has to be initialized with x_data(generate the data format using generate_data function),
    webList(A dcitionary where key is the class and value is the websiteName), model(the trained model)"""
    cellDict = {0: "PADDING", 1: "CREATE", 2: "CREATED", 3: "RELAY", 4: "DESTROY", 5: "CREATE_FAST", 6: "CREATED_FAST",
                7: "VERSIONS", 8: "NETINFO", 9: "RELAY_EARLY", 10: "CREATE2", 11: "CREATED2", 12: "PADDING_NEGOTIATE"}

    def __init__(self, x_data, webList, model):
        self.x_data = x_data
        self.webList = webList
        self.relevance, self.prediction = lp.lrp(model, self.x_data, len(self.webList))

    def relevance_score(self, packetrange=500):
        """This method returns the relevance score of the packets in the given range (default=500)."""
        dff = pd.DataFrame(self.relevance[0:packetrange]).T
        dff = dff[dff > 0]
        dff = dff.fillna(0)

        # Scaling the values between 0-1
        dff = dff.T
        q = dff.max() - dff.min()
    
        dff = dff - dff.min()
       
        dff = dff.div(q)
        
        dff = dff.T
       

        dff = dff.to_numpy()
        dffR = [i for i in dff[0]]
        return dffR

    def score_indices_percentile(self, percentile, pckRange=500):
        """This method finds the relevance scores which are greater than a given percentile value
        and return the indices(packet number) of those scores"""
        score = np.array(self.relevance_score(pckRange))
        val = np.percentile(score, percentile)
        new_indices = []
        for i in range(len(score)):
            if score[i] >= val:
                new_indices.append(i)
                continue
            else:
                score[i] == 0

        return new_indices
