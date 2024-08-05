import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

class ServerParty(object):
    def __init__(self, model, loss_func, optimizer, n_iter=1, use_concat=False, evaluate_func='default'):
        super(ServerParty, self).__init__()
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        # self.val_loader = val_loader
        self.n_iter = n_iter
        self.use_concat = use_concat
        
        self.parties_grad_list = []
        self.y = None
        self.batch_size = None
        self.h_input = None
        self.loss = None
        
        self.h_weight_list = None
        # self.h_weight_list = torch.tensor([1, 1, 1]) / 3

        self.evaluate_func = evaluate_func

    def set_batch(self, y):
        self.y = y
        self.batch_size = y.shape[0]
    
    def get_loss(self):
        return self.loss

    def compute_parties_grad(self):
        output = self.model(self.h_input)
        loss = self.loss_func(output,self.y)
        self.optimizer.zero_grad()
        loss.backward()
        self.loss = loss
        parties_grad = self.h_input.grad

        self.parties_grad_list = []
        if self.use_concat:
            start = 0
            for dim in self.h_dim_list:
                self.parties_grad_list.append(parties_grad[:,start:start+dim])
                start += dim
        else:
            self.parties_grad_list = [parties_grad[:, :h_dim] * weight for h_dim, weight in zip(self.h_dim_list, self.h_weight_list)] # h grad的维度和每个party相同
            
        correct,accuracy = self.evaluate(output,self.y)
        print(f'loss={loss} correct={correct} accuracy={accuracy}\n')
    
    def local_update(self):
        self.optimizer.step()
    
    def local_iterations(self):
        self.h_input.requires_grad = False
        for i in range(self.n_iter-1):
            self.compute_parties_grad()
            self.local_update()

    def pull_parties_h(self, h_list):
        h_input = None
        self.h_dim_list = [h.shape[1] for h in h_list]
        if self.use_concat:

            for h in h_list:
                if h_input is None:
                    h_input = h
                else:
                    h_input = torch.cat([h_input,h],1)
        else:
            if self.h_weight_list is None:
                self.h_weight_list = [1/len(h_list) for _ in h_list]
            max_h = max(h_list,key=lambda t:t.shape[1])
            h_input = torch.zeros(max_h.shape).to(max_h.device)
            for h, weight in zip(h_list,self.h_weight_list):
                h_input[:,:h.shape[1]] += h * weight
        # h_input = h_input.detach()
        h_input.requires_grad = True
        self.h_input = h_input

    def send_parties_grad(self):
        return self.parties_grad_list
    
    def predict(self,h_list,y):
        self.pull_parties_h(h_list)
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.h_input)
            loss = self.loss_func(output,y)
            correct,accuracy = self.evaluate(output,y)
            return loss,correct,accuracy
    
    def evaluate(self,output,y):
        if self.evaluate_func == 'top5':
            _,pred = torch.topk(output,5,1)
            correct = torch.sum(pred == y.unsqueeze(1)).item()
            accuracy = correct / (y.shape[0])
        elif isinstance(self.loss_func, nn.BCELoss):
            true = y.cpu().numpy()
            pred = output.cpu().detach().numpy()
            correct = -1
            accuracy = roc_auc_score(true,pred)
        else:
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(y.view_as(pred)).sum().item()
            accuracy = correct / (y.shape[0])

        return correct,accuracy


class ClientParty(object):
    def __init__(self, model, optimizer, n_iter=1):
        super(ClientParty, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.n_iter = n_iter

        self.x = None
        self.h = None
        self.partial_grad = None
        self.batch_size = None

    def set_batch(self, x):
        self.x = x
        self.batch_size = x.shape[0]
    
    def get_h(self):
        return self.h

    def compute_h(self):
        self.h = self.model(self.x)

    def local_update(self):
        self.optimizer.zero_grad()
        self.h.backward(self.partial_grad)
        self.optimizer.step()

    def send_h(self):
        return self.h

    def pull_grad(self, grad):
        self.partial_grad = grad

    def local_iterations(self):
        for i in range(self.n_iter - 1):
            self.compute_h()
            self.local_update()

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            predict_h = self.model(x)
        return predict_h