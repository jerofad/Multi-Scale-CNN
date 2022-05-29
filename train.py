from csv import writer
import torch.utils.data as Data
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from fusion_models.early_fusion import *
from data import *
import random
from sklearn.metrics import classification_report
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from polyloss import PolyLoss

torch.manual_seed(2021)  # cpu
torch.cuda.manual_seed(2021)  # gpu
np.random.seed(2021)  # numpy
random.seed(2021)  # random and transforms
torch.backends.cudnn.deterministic = True  

writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss = 'polyloss'  # cross_entropy
def worker_init_fn(worker_id):
    np.random.seed(2021 + worker_id)


batch_size = 16
num_epochs = 5
target_names = ['Abnormal', 'normal']

train_data_loader, test_data_loader, num_train_instances, num_test_instances = data_load()

msresnet = MSResNet(input_channel=1, layers=[1, 1, 1], num_classes=2)
msresnet = msresnet.to(device)
if loss != 'polyloss':
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
else:
    criterion = PolyLoss(reduction='sum').to(device)

optimizer = torch.optim.Adam(msresnet.parameters(), lr=0.0001, weight_decay=0.000002)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)
train_loss = np.zeros([num_epochs, 1])
test_loss = np.zeros([num_epochs, 1])
train_acc = np.zeros([num_epochs, 1])
test_acc = np.zeros([num_epochs, 1])


if __name__ == '__main__':
    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        msresnet.train()

        loss_x = 0
        correct_train = 0
        for i, (samples, labels) in enumerate(train_data_loader):
                samplesV = Variable(samples.to(device))
                labelsV = Variable(labels.to(device))
                optimizer.zero_grad()
                predict_label = msresnet(samplesV)
                _, prediction = torch.max(predict_label, 1)
                correct_train += prediction.eq(labelsV.data.long()).sum()

                loss = criterion(predict_label, labelsV)
                loss_x += loss.item()
                loss.backward()
                optimizer.step()
                
        scheduler.step()

        accuracy = 100*float(correct_train)/num_train_instances
        
        print("Training accuracy:", accuracy)
        writer.add_scalar("Loss/train", loss_x / num_train_instances, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)

        train_loss[epoch] = loss_x / num_train_instances 
        train_acc[epoch] = 100*float(correct_train)/num_train_instances

        trainacc = str(100*float(correct_train)/num_train_instances)[0:6]

        loss_x = 0
        correct_test = 0
        y_true = []
        y_pre = []
        temp_true = []
        temp_pre = []

        msresnet.eval()
        for i, (samples, labels) in enumerate(test_data_loader):
            with torch.no_grad():
                samplesV = Variable(samples.to(device))
                labelsV = Variable(labels.to(device))
                predict_label = msresnet(samplesV)
                _, prediction = torch.max(predict_label, 1)

                correct_test += prediction.eq(labelsV.data.long()).sum()
                y_true.extend(labels.cpu().numpy())
                y_pre.extend(prediction.cpu().numpy())
                loss = criterion(predict_label, labelsV)
                loss_x += loss.item()

        acc_test = (100 * float(correct_test) / num_test_instances)
        print("Test accuracy:", acc_test )

        writer.add_scalar("Loss/test", loss_x / num_test_instances, epoch)
        writer.add_scalar("Accuracy/test", acc_test, epoch)
        test_loss[epoch] = loss_x / num_test_instances
        test_acc[epoch] = acc_test

        testacc = str(acc_test)[0:6]

        if epoch == 0:
            temp_test = correct_test
            temp_train = correct_train
            temp_true = y_true
            temp_pre = y_pre
        elif correct_test >= temp_test:
            temp_test = correct_test
            temp_train = correct_train
            temp_epoch = 0
            temp_true = y_true
            temp_pre = y_pre
            temp_epoch = epoch
            print('“\nThis is the  classification report:...\n')
            print(classification_report(y_true, y_pre, digits=4, target_names=target_names), '\n')

    print(str(100 * float(temp_test) / num_test_instances)[0:6])