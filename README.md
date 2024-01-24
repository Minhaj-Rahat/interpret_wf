# Interpreting Website Fingerprinting
This is an implementation of interpreting Deep Fingerprinting. We are using the Layerwise Relevance Propagation algorithm to
understand which part of traffic data was relevant in classifying an specific website.

# Documentation

1. Use torch_Train.py to train the model. 
2. DataSet directory 'dataSet/NoDef/'  for no defense data
3. Saved trained model in 'torchData/'
4. Use Analyze2.py for average heatmap generation
5. Use twitterAnalyze.py to analyze twitter packets
6. Use webSiteCheckertest.py to test the predictions of the trained calssifer

# Create Dataset

1. Use the [crawler](https://github.com/webfp/tor-browser-crawler) to crawl websites and dump pcap files
2. Use create_direction.py to create the packet direction features for the dumped pcap files
3. Use ProcessAllv2.py to create train-test-validation dataset
4. Utility_torch.py has some functions to process the created dataset as per the requirement of the classifier input
