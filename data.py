import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


img_size = 256
# Change this path to where you have the images
COVID_US_path = '../input/covid19ultrasound/covid_us'
covid_19_path = '../input/covid19ultrasound/image_dataset/image_dataset'


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


def data_load(batch_size, path):
    # load data
    data_transform = transforms.Compose([
                transforms.Grayscale(1),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                
        ])



    # Data Loader
    dataset = torchvision.datasets.ImageFolder(root=path,
                                               transform=data_transform)
    # split the dataset into train and validation set
    print(len(dataset))
    datasets = train_val_dataset(dataset)
    print(len(datasets['train']))
    print(len(datasets['val']))
    # The original dataset is available in the Subset class
    print(datasets['train'].dataset)


    train_data_loader = torch.utils.data.DataLoader(datasets['train'], 
                                                    batch_size=batch_size,  
                                                    shuffle=False, 
                                                    num_workers=0)
    num_train_instances = len(datasets['train'])
    x,y = next(iter(train_data_loader))
    print(x.shape, y.shape)

    val_data_loader = torch.utils.data.DataLoader(datasets['val'], 
                                                  batch_size=batch_size, 
                                                  shuffle=False, 
                                                  num_workers=0)
    num_val_instances = len(datasets['val'])

    return train_data_loader, val_data_loader, num_train_instances, num_val_instances