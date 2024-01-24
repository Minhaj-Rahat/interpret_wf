import WebsiteInstance as wb
from WebsiteInstance import WebsiteInstance as WbI
import utility_torch as ut
import torch
from torch_model import no_def_model

import numpy as np
import matplotlib.pyplot as plt

"""A sample for testing WebsiteInstance Class"""
# x_test, y_test = ut.load_replaced_test()
# x1 = torch.unsqueeze(x_test[10], 0)
x1 = wb.generate_data("1w1.txt")
model = no_def_model(95)
model.load_state_dict(torch.load('torchData/No_def_replaced.pkl'))
model.eval()

wb1 = WbI(x1, "Google", model)
# indices = wb1.score_indices(0.45)
# print(indices)
# print(wb1.prediction)
# wb1.heat_map()

# analyze packet
wb1.analyze_packets("1w1.pcap","keyS1w1.txt")
'''
(score = wb1.relevance_score()
score_arr = np.array(score)
print("The Relevance Scores are: ")
print(score)

avg = np.average(score)
print(avg)
std_dev = np.std(score)
print(std_dev)
min_deviation = 0.45
distance_from_mean = abs(score - avg)
not_outlier = distance_from_mean > min_deviation * std_dev

print("Scores within the selected range of standard deviation: ")
new_score = score_arr[not_outlier]
print(new_score)

new_indices = []

for i in new_score:
    index = np.where(score_arr == i)[0][0]
    new_indices.append(index)
print(new_score.shape)
print(len(score))
print(min_deviation * std_dev)

print("Indices of the score with that range")
print(new_indices)
# count = 0
# for i in range(0,len(score)):
    # if score[i]>=p:
        # count +=1
# print(count)
# plt.hist(score)
# plt.show() )'''
