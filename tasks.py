import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import random
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
from models import agnews_model,cifar_model,mimic_model,imagenet_model
from partymodel import ServerParty,ClientParty
from MLclf import MLclf
from src.torchmimic.data import IHMDataset
from src.torchmimic.utils import pad_colalte
from torchtext.datasets import AG_NEWS
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from transformers import BertTokenizer

device = torch.device("cpu")
    
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

def get_task_data(task_name,id=0,is_asyn=True,use_gpu=False,use_concat=False,estimation=False,search=False):
    global device
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    set_seed()
    
    if task_name == 'cifar':
        return get_cifar_task_data(id,is_asyn,False,estimation,search)
    elif task_name == 'mimic':
        return get_mimic_task_data(id,is_asyn,False,estimation,search)
    elif task_name == 'imagenet':
        return get_imagenet_task_data(id,is_asyn,True,estimation,search)
    elif task_name == 'agnews':
        return get_agnews_task_data(id,is_asyn,False,estimation,search)
    return -1

def reset_ps_party(task_name,id=0,lr=0.001):
    task_info = json.load(open('task_info.json','r',encoding='utf-8'))[task_name]
    n_local = task_info['n_local']
    div = task_info['div']

    if task_name == 'cifar':
        if id == 0:
            model = cifar_model.ServerNet().to(device)
            loss_func = nn.CrossEntropyLoss().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr)
            party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id])
        else:
            model = cifar_model.ClientNet().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr)
            party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])
    elif task_name == 'mimic':
        if id == 0:
            model = mimic_model.ServerNet().to(device)
            loss_func = nn.BCELoss().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id])
        else:
            model = mimic_model.ClientNet(div[id]-div[id-1]).to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ClientParty(model=model, optimizer=optimizer, n_iter=n_local[id])
    elif task_name == 'imagenet':
        if id == 0:
            model = imagenet_model.ServerNet().to(device)
            loss_func = nn.CrossEntropyLoss().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id])
        else:
            model = imagenet_model.ClientNet().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])
    elif task_name == 'agnews':
        use_concat = False
        if id == 0:
            model = agnews_model.ServerNet(party_num=task_info['client_num'] if use_concat else 1,mask_dim=50,output_dim=4).to(device)
            loss_func = nn.CrossEntropyLoss().to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id],use_concat=use_concat)
        else:
            model = agnews_model.ClientNet().to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])
    
    return party

def get_cifar_task_data(id,is_asyn,use_concat,estimation,search):
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

        # train_dataset.data = np.pad(train_dataset.data, ((0,0),(32,32),(32,32),(0,0)), mode='constant', constant_values=0)
        # test_dataset.data = np.pad(test_dataset.data, ((0,0),(32,32),(32,32),(0,0)), mode='constant', constant_values=0)

        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)
        model = cifar_model.ServerNet().to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id],use_concat=use_concat)
        
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

        # padding = transforms.Pad(padding=32, fill=0)
        # train_dataset.data = np.pad(train_dataset.data, ((0,0),(32,32),(32,32),(0,0)), mode='constant', constant_values=0)
        # test_dataset.data = np.pad(test_dataset.data, ((0,0),(32,32),(32,32),(0,0)), mode='constant', constant_values=0)

        train_dataset.data = train_dataset.data[:,:,div[id][0]:div[id][1]]
        test_dataset.data = test_dataset.data[:,:,div[id][0]:div[id][1]]

        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)
        # model = cifar_model.ClientNet(n_dim=32*3*(div[id]-div[id-1])).to(device) # n_dim
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

def get_mimic_task_data(id,is_asyn,use_concat,estimation,search):
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
    # train_dataset.labels = torch.zeros(len(train_dataset), 2).scatter_(1, train_dataset.labels[:, None].to(torch.int64), 1)
    # test_dataset.labels = torch.zeros(len(test_dataset), 2).scatter_(1, test_dataset.labels[:, None].to(torch.int64), 1)
    kwargs = {"num_workers": 5, "pin_memory": True} if device else {}

    g = torch.Generator()

    task_data = []

    if id == 0:
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size,shuffle=True,drop_last=True,generator=g,collate_fn=pad_colalte,**kwargs)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size,shuffle=False,drop_last=True,collate_fn=pad_colalte,**kwargs)
        model = mimic_model.ServerNet().to(device)
        loss_func = nn.BCELoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
        party = ServerParty(model=model, loss_func=loss_func, optimizer=optimizer, n_iter=n_local[id],use_concat=use_concat)

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
        model = mimic_model.ClientNet(div[id]-div[id-1]).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
        party = ClientParty(model=model, optimizer=optimizer, n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])

        if estimation:
            task_data.append(Tw)

    return task_data

def get_imagenet_task_data(id,is_asyn,use_concat,estimation,search):
    task_info = json.load(open('task_info.json','r',encoding='utf-8'))['imagenet']
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

    class ImageNetDataset(Dataset):
        def __init__(self,data,labels):     
            self.data = data
            self.labels = labels
            self.size = data.shape[0]

        def __getitem__(self, index):
            return self.data[index],self.labels[index]

        def __len__(self):
            return self.size

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    g = torch.Generator()

    task_data = []


    if id == 0:
        train_dataset, validation_dataset, test_dataset = MLclf.miniimagenet_clf_dataset(data_dir=data_dir,ratio_train=0.6, ratio_val=0.2, seed_value=None, shuffle=True, transform=transform, save_clf_data=True)

        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g,num_workers=2)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True,num_workers=2)
        model = imagenet_model.ServerNet().to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id],use_concat=use_concat,evaluate_func='top5')
        
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
        train_dataset, validation_dataset, test_dataset = MLclf.miniimagenet_clf_dataset(data_dir=data_dir,ratio_train=0.6, ratio_val=0.2, seed_value=None, shuffle=True, transform=transform, save_clf_data=True)

        train_dataset = ImageNetDataset(train_dataset.tensors[0][:,:,div[id][0]:div[id][1]],train_dataset.tensors[1])
        test_dataset = ImageNetDataset(test_dataset.tensors[0][:,:,div[id][0]:div[id][1]],test_dataset.tensors[1])

        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g,num_workers=2)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True,num_workers=2)
        model = imagenet_model.ClientNet().to(device)
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

def get_agnews_task_data(id,is_asyn,use_concat,estimation,search):
    task_info = json.load(open('task_info.json','r',encoding='utf-8'))['agnews']
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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_pipeline = lambda x: tokenizer(
                            x,                      
                            add_special_tokens = True,  # 添加special tokens， 也就是CLS和SEP
                            max_length = 50,           # 设定最大文本长度
                            padding = 'max_length',   # pad到最大的长度  
                            return_tensors = 'pt',       # 返回的类型为pytorch tensor
                            truncation = True
                    )
    label_pipeline = lambda x: int(x) - 1

    def collate_batch(batch):
        label_list, text_list = [], []
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            text_list.append(text_pipeline(_text))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.cat(
            [torch.cat([text['input_ids'] for text in text_list]).unsqueeze(1),
             torch.cat([text['attention_mask'] for text in text_list]).unsqueeze(1)],
             dim=1
        )

        return text_list, label_list
    
    g = torch.Generator()

    task_data = []

    if id == 0:
        train_iter, test_iter = AG_NEWS(root=data_dir)
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)
        test_dataset, _ = random_split(test_dataset, [0.05, 0.95], generator=g)

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True, shuffle=True, collate_fn=collate_batch,generator=g)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, drop_last=True, collate_fn=collate_batch)

        model = agnews_model.ServerNet(party_num=client_num if use_concat else 1,mask_dim=50,output_dim=4).to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id],use_concat=use_concat)
        
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
        train_iter, test_iter = AG_NEWS(root=data_dir)
        train_iter = [(label,text[int(len(text) * div[id-1]):int(len(text) * div[id])]) for label,text in train_iter]
        test_iter = [(label,text[int(len(text) * div[id-1]):int(len(text) * div[id])]) for label,text in test_iter]
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)
        test_dataset, _ = random_split(test_dataset, [0.05, 0.95], generator=g)

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True, shuffle=True, collate_fn=collate_batch,generator=g)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, drop_last=True, collate_fn=collate_batch)

        model = agnews_model.ClientNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])

        if estimation:
            task_data.append(Tw)

    return task_data