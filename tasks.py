import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import random
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
from models import mnist_model,cifar_model,a9a_model,mimic_model
from partymodel import ServeParty,ClientParty
from src.torchmimic.data import IHMDataset
from src.torchmimic.utils import pad_colalte

device = torch.device("cpu")

class A9ADataset(Dataset):
    def __init__(self,data,labels):     
        self.data = data
        self.labels = labels
        self.size = data.shape[0]

    def __getitem__(self, index):
        return self.data[index],self.labels[index]

    def __len__(self):
        return self.size
    
def set_seed(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)

def load_a9a_data(path):
    data = []
    labels = []
    file = open(path,'r')
    file_data = file.readlines()
    for row in file_data:
        tmp_list = row.split(' ')
        labels.append(1 if int(tmp_list[0])==1 else 0)
        one_row = [0]*123
        for val in tmp_list[1:-1]:
            one_row[int(val.split(':')[0])-1] = 1
        data.append(one_row)
    
    data = torch.Tensor(data)
    labels = torch.Tensor(labels).long()
    return data,labels

def get_task_data(task_name,id=0,is_asyn=True,use_gpu=False,estimation=False,search=False):
    global device
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    set_seed()
    
    if task_name == 'mnist':
        return get_mnist_task_data(id,is_asyn,estimation,search)
    elif task_name == 'cifar':
        return get_cifar_task_data(id,is_asyn,estimation,search)
    elif task_name == 'a9a':
        return get_a9a_task_data(id,is_asyn,estimation,search)
    elif task_name == 'mimic':
        return get_mimic_task_data(id,is_asyn,estimation,search)
    return -1

def reset_ps_party(task_name,id=0,lr=0.001):
    task_info = json.load(open('task_info.json','r',encoding='utf-8'))[task_name]
    n_local = task_info['n_local']
    div = task_info['div']

    if task_name == 'mnist':
        if id == 0:
            model = mnist_model.ServerNet().to(device)
            loss_func = nn.CrossEntropyLoss().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ServeParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id])
        else:
            model = mnist_model.ClientNet().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])
    elif task_name == 'cifar':
        if id == 0:
            model = cifar_model.ServerNet().to(device)
            loss_func = nn.CrossEntropyLoss().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ServeParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id])
        else:
            model = cifar_model.ClientNet().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])
    elif task_name == 'a9a':
        if id == 0:
            model = a9a_model.ServerNet().to(device)
            loss_func = nn.CrossEntropyLoss().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ServeParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id])
        else:
            model = a9a_model.ClientNet(div[id]-div[id-1]).to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])
    elif task_name == 'mimic':
        if id == 0:
            model = mimic_model.ServerNet(num_layers=2).to(device)
            loss_func = nn.BCELoss().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ServeParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id])
        else:
            model = mimic_model.ClientNet(48*(div[id]-div[id-1])).to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ClientParty(model=model, optimizer=optimizer, n_iter=n_local[id])
    
    return party

def get_mnist_task_data(id,is_asyn,estimation,search):
    task_info = json.load(open('task_info.json','r',encoding='utf-8'))['mnist']
    data_dir = task_info['data_dir']
    client_num = task_info['client_num']
    n_local = task_info['n_local']
    bounds = task_info['bounds'] if is_asyn else [0]*(client_num+1)
    train_batch_size = task_info['train_batch_size']
    test_batch_size = task_info['test_batch_size']
    epochs = task_info['epochs']
    div = task_info['div']
    lr = task_info['lr']
    delta_T = task_info['delta_T']
    CT = task_info['CT']
    Tw = task_info['Tw']
    c0 = task_info['c0']
    estimation_D = task_info['estimation_D']
    search_CT = task_info['search_CT']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    g = torch.Generator()

    task_data = []


    if id == 0:
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)
        model = mnist_model.ServerNet().to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ServeParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])
        task_data.append(lr)
        task_data.append(delta_T)
        task_data.append(CT)

        if estimation:
            task_data.append(Tw)
            task_data.append(c0)
            task_data.append(estimation_D)
        elif search:
            task_data.append(search_CT)
            task_data.append(c0)
    else:
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
        train_dataset.data = train_dataset.data[:,:,div[id-1]:div[id]]
        test_dataset.data = test_dataset.data[:,:,div[id-1]:div[id]]
        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)
        model = mnist_model.ClientNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])

        if estimation:
            task_data.append(Tw)

    return task_data

def get_cifar_task_data(id,is_asyn,estimation,search):
    task_info = json.load(open('task_info.json','r',encoding='utf-8'))['cifar']
    data_dir = task_info['data_dir']
    client_num = task_info['client_num']
    n_local = task_info['n_local']
    bounds = task_info['bounds'] if is_asyn else [0]*(client_num+1)
    train_batch_size = task_info['train_batch_size']
    test_batch_size = task_info['test_batch_size']
    epochs = task_info['epochs']
    div = task_info['div']
    lr = task_info['lr']
    delta_T = task_info['delta_T']
    CT = task_info['CT']
    Tw = task_info['Tw']
    c0 = task_info['c0']
    estimation_D = task_info['estimation_D']
    search_CT = task_info['search_CT']

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    g = torch.Generator()

    task_data = []


    if id == 0:
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform_test)
        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)
        model = cifar_model.ServerNet().to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ServeParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id])
        
        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])
        task_data.append(lr)
        task_data.append(delta_T)
        task_data.append(CT)

        if estimation:
            task_data.append(Tw)
            task_data.append(c0)
            task_data.append(estimation_D)
        elif search:
            task_data.append(search_CT)
            task_data.append(c0)
    else:
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform_test)
        train_dataset.data = train_dataset.data[:,:,div[id-1]:div[id]]
        test_dataset.data = test_dataset.data[:,:,div[id-1]:div[id]]
        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)
        model = cifar_model.ClientNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])

        if estimation:
            task_data.append(Tw)

    return task_data

def get_a9a_task_data(id,is_asyn,estimation,search):
    task_info = json.load(open('task_info.json','r',encoding='utf-8'))['a9a']
    data_dir = task_info['data_dir']
    client_num = task_info['client_num']
    n_local = task_info['n_local']
    bounds = task_info['bounds'] if is_asyn else [0]*(client_num+1)
    train_batch_size = task_info['train_batch_size']
    test_batch_size = task_info['test_batch_size']
    epochs = task_info['epochs']
    div = task_info['div']
    lr = task_info['lr']
    delta_T = task_info['delta_T']
    CT = task_info['CT']
    Tw = task_info['Tw']
    c0 = task_info['c0']
    estimation_D = task_info['estimation_D']
    search_CT = task_info['search_CT']

    train_data,train_labels = load_a9a_data(data_dir+'/train.txt')
    test_data,test_labels = load_a9a_data(data_dir+'/test.txt')

    g = torch.Generator()

    task_data = []


    if id == 0:
        train_dataset = A9ADataset(train_data,train_labels)
        test_dataset = A9ADataset(test_data,test_labels)
        train_loader = DataLoader(train_dataset,train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,test_batch_size,drop_last=True)
        model = a9a_model.ServerNet().to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ServeParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id])
        
        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])
        task_data.append(lr)
        task_data.append(delta_T)
        task_data.append(CT)

        if estimation:
            task_data.append(Tw)
            task_data.append(c0)
            task_data.append(estimation_D)
        elif search:
            task_data.append(search_CT)
            task_data.append(c0)
    else:
        train_dataset = A9ADataset(train_data[:,div[id-1]:div[id]],train_labels)
        test_dataset = A9ADataset(test_data[:,div[id-1]:div[id]],test_labels)
        train_loader = DataLoader(train_dataset,train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,test_batch_size,drop_last=True)
        model = a9a_model.ClientNet(div[id]-div[id-1]).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])
    
        if estimation:
            task_data.append(Tw)

    return task_data


def get_mimic_task_data(id,is_asyn,estimation,search):
    task_info = json.load(open('task_info.json', 'r', encoding='utf-8'))['mimic']
    data_dir = task_info['data_dir']
    client_num = task_info['client_num']
    n_local = task_info['n_local']
    bounds = task_info['bounds'] if is_asyn else [0]*(client_num+1)
    train_batch_size = task_info['train_batch_size']
    test_batch_size = task_info['test_batch_size']
    epochs = task_info['epochs']
    div = task_info['div']
    lr = task_info['lr']
    delta_T = task_info['delta_T']
    CT = task_info['CT']
    Tw = task_info['Tw']
    c0 = task_info['c0']
    estimation_D = task_info['estimation_D']
    search_CT = task_info['search_CT']

    train_dataset = IHMDataset(data_dir, train=True, n_samples=None)
    test_dataset = IHMDataset(data_dir, train=False, n_samples=None)
    # train_dataset.labels = train_dataset.labels[:, None]
    # test_dataset.labels = test_dataset.labels[:, None]
    train_dataset.labels = torch.zeros(len(train_dataset), 2).scatter_(1, train_dataset.labels[:, None].to(torch.int64), 1)
    test_dataset.labels = torch.zeros(len(test_dataset), 2).scatter_(1, test_dataset.labels[:, None].to(torch.int64), 1)
    kwargs = {"num_workers": 0, "pin_memory": True} if device else {}

    g = torch.Generator()

    task_data = []

    if id == 0:
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size,shuffle=True,drop_last=True,generator=g,collate_fn=pad_colalte,**kwargs)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size,shuffle=False,drop_last=True,collate_fn=pad_colalte,**kwargs)
        model = mimic_model.ServerNet(num_layers=2).to(device)
        loss_func = nn.BCELoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ServeParty(model=model, loss_func=loss_func, optimizer=optimizer, n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])
        task_data.append(lr)
        task_data.append(delta_T)
        task_data.append(CT)

        if estimation:
            task_data.append(Tw)
            task_data.append(c0)
            task_data.append(estimation_D)
        elif search:
            task_data.append(search_CT)
            task_data.append(c0)
    else:
        train_dataset.data = [t[:, div[id - 1]:div[id]] for t in train_dataset.data]
        test_dataset.data = [t[:, div[id - 1]:div[id]] for t in test_dataset.data]
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,drop_last=True,generator=g,collate_fn=pad_colalte,**kwargs)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,drop_last=True,collate_fn=pad_colalte,**kwargs)
        model = mimic_model.ClientNet(48*(div[id]-div[id-1])).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ClientParty(model=model, optimizer=optimizer, n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])

        if estimation:
            task_data.append(Tw)

    return task_data