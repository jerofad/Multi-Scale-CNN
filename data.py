import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable


def data_load(batch_size):
    # load data
    data_transform = transforms.Compose([
                transforms.Grayscale(1),
                transforms.Resize(256),
                transforms.ToTensor(),
                
        ])



    # B-mode data loader
    train_data_Bmode = torchvision.datasets.ImageFolder(root='dataset/Bmode/train', transform=data_transform)
    train_data_Bmode_loader = torch.utils.data.DataLoader(train_data_Bmode, batch_size=batch_size,  shuffle=False, num_workers=0)
    num_train_instances = len(train_data_Bmode)

    test_data_Bmode = torchvision.datasets.ImageFolder(root='dataset/Bmode/test', transform=data_transform)
    test_data_Bmode_loader = torch.utils.data.DataLoader(test_data_Bmode, batch_size=batch_size, shuffle=False, num_workers=0)
    num_test_instances = len(test_data_Bmode)

    return train_data_Bmode_loader, test_data_Bmode_loader, num_train_instances, num_test_instances