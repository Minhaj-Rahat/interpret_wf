#from torch_model import no_def_model
from torch_model_modified import no_def_model
import helperDataTorch as ht
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
import torch.nn as nn
import utility_torch as ut
import numpy as np

# Defining parameter
extra_class = 0
NB_CLASSES = 20 + extra_class
EPOCH = 30
BATCH_SIZE = 128
LR = 0.001
#LR = 0.0001

# Loading Dataset
dataset_train, dataset_validation, dataset_test = ht.create_Dataset()

# datasetSizes
train_size = len(dataset_train)
test_size = len(dataset_test)
val_size = len(dataset_validation)

# train data shuffling
shuffle_dataset = True
random_seed = 16

train_indices = list(range(train_size))
val_indices = list(range(val_size))

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# Create data pipeline for training
train_data = DataLoader(dataset_train, batch_size=BATCH_SIZE, sampler=train_sampler)
test_data = DataLoader(dataset_test, batch_size=BATCH_SIZE)
validation_data = DataLoader(dataset_validation, batch_size=BATCH_SIZE, sampler=val_sampler)

# Load CNN model
model = no_def_model(NB_CLASSES).float()
# weight initialization
model.apply(ut.torch_initializer)

# Setting optimizer and loss function
optimizer = torch.optim.Adamax(model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
#loss_func = nn.NLLLoss()

# Scheduler for learning rate decay
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# Training and Validation
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_data):
        output = model(b_x.float())[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        if step % 50 == 0:
            corrects = 0
            avg_loss = 0
            for _, (b_xx, b_yy) in enumerate(validation_data):
                logit = model(b_xx.float())[0]
                loss = loss_func(logit, b_yy)
                avg_loss += loss.item()
                corrects += (torch.max(logit, 1)
                             [1].view(b_yy.size()).data == b_yy.data).sum()

            size = val_size
            avg_loss /= size
            accuracy = 100.0 * corrects / size
            print('Epoch: {:2d}({:6d}/{}) Evaluation - loss: {:.6f}  acc: {:3.4f}%({}/{})'.format(
                epoch,
                step * 128,
                len(train_data),
                avg_loss,
                accuracy,
                corrects,
                size))

    #scheduler.step()

torch.save(model.state_dict(), 'torchData/No_def_replacedFiltered1Mymodelmodified.pkl')

corrects = 0
avg_loss = 0
model.eval()
for _, (b_x, b_y) in enumerate(test_data):
    logit = model(b_x.float())[0]
    loss = loss_func(logit, b_y)
    avg_loss += loss.item()
    corrects += (torch.max(logit, 1)
                 [1].view(b_y.size()).data == b_y.data).sum()

size = test_size
accuracy = 100.0 * corrects / size
print("accuracy: {:3.4f}%".format(accuracy))
