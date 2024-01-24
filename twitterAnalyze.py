"""Print the relevant packet information for a given sample. The relevant packet will be above the given
percentile values."""

import packetAnalyzerClass as pck
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt


# website list in the classifier
myList = {0: 'Google.com', 1: 'Youtube.com', 2: 'qq.com', 3: 'Facebook.com', 4: 'Taobao.com', 5: 'Amazon.com',
          6: 'Zhihu.com',
          7: 'Wikipedia.org', 8: 'Zoom.us', 9: 'Bilibili.com', 10: 'Microsoft.com', 11: 'Office.com', 12: 'Weibo.com',
          13: 'Sohu.com', 14: 'Reddit.com', 15: 'Netflix.com', 16: 'Tmall.com', 17: 'Stackoverflow.com',
          18: '360.cn', 19: 'Twitter.com'}



"""load Twitter Sample"""
percentile = 50

# Sample 1
file_name = 'testing/twitter/46/directions/46/7.txt'
pcap = 'testing/twitter/46/46_7.pcap'
key_log_file = 'testing/twitter/46/twitter_1.txt'


twtpck = pck.PacketAnalyzer(file_name, pcap, key_log_file, percentile)


print(f'Sample is Twitter. The predicted Class is {myList[twtpck.w.prediction]}')

# See Relevant Packets
packetInfo1, timestamps, times, lengths, directions, rt, rl, rd = twtpck.packet_analyze()  # some extra variables are for later work


for i in packetInfo1:
    print(i)