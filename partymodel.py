import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

class ServeParty(object):
    def __init__(self, model, loss_func, optimizer, n_iter=1):
        super(ServeParty, self).__init__()
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.n_iter = n_iter
        
        self.h_dim_list = []
        self.parties_grad_list = []
        self.y = None
        self.batch_size = None
        self.h_input = None
        self.loss = None

    def set_batch(self, y):
        self.y = y
        self.batch_size = y.shape[0]

    def get_theta(self):
        theta = None
        for para in self.model.parameters():
            theta_tmp = torch.flatten(para, 0)
            if theta is None:
                theta = theta_tmp
            else:
                theta = torch.cat([theta,theta_tmp],0)
        return theta
    
    def get_theta_grad(self):
        theta_grad = None
        for para in self.model.parameters():
            theta_grad_tmp = torch.flatten(para.grad, 0)
            if theta_grad is None:
                theta_grad = theta_grad_tmp
            else:
                theta_grad = torch.cat([theta_grad,theta_grad_tmp],0)
        return theta_grad
    
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
        start = 0
        for dim in self.h_dim_list:
            self.parties_grad_list.append(parties_grad[:,start:start+dim])
            start += dim

        if isinstance(self.loss_func, nn.BCELoss):
            y_classes = self.y.nonzero()[:, 1].cpu().numpy()  # one-hot label to classes
            softmax = nn.Softmax(dim=1)
            output_prob = softmax(output).detach().cpu().numpy()
            loss = self.loss_func(output, self.y)  # one-hot label
            correct = -1
            try:
                accuracy = roc_auc_score(y_classes, output_prob[:, 1])  # auc
            except ValueError:
                accuracy = -1
        else:
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(self.y.view_as(pred)).sum().item()
            accuracy = correct/self.batch_size
        print(f'loss={loss} correct={correct} accuracy={accuracy}\n')
    
    def local_update(self):
        self.optimizer.step()
    
    def local_iterations(self):
        self.h_input.requires_grad = False
        for i in range(self.n_iter-1):
            self.compute_parties_grad()
            self.local_update()

    def pull_parties_h(self, h_list):
        self.h_dim_list = [h.shape[1] for h in h_list]
        h_input = None
        for h in h_list:
            if h_input is None:
                h_input = h
            else:
                h_input = torch.cat([h_input,h],1)
        h_input.requires_grad = True
        self.h_input = h_input

    def send_parties_grad(self):
        return self.parties_grad_list

    def predict(self,h_list,y):
        batch_size = y.shape[0]
        self.pull_parties_h(h_list)
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.h_input)
            if isinstance(self.loss_func, nn.BCELoss):
                y_classes = y.nonzero()[:, 1].cpu().numpy()  # one-hot label to classes
                softmax = nn.Softmax(dim=1)
                output_prob = softmax(output).cpu().numpy()
                loss = self.loss_func(output, y)  # one-hot label
                correct = -1
                accuracy = roc_auc_score(y_classes, output_prob[:, 1])  # auc
            else:
                loss = self.loss_func(output,y)
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(y.view_as(pred)).sum().item()
                accuracy = correct/batch_size

        return loss,correct,accuracy



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

    def set_h(self, h):
        self.h = h

    def get_h(self):
        return self.h

    def get_theta(self):
        theta = None
        for para in self.model.parameters():
            theta_tmp = torch.flatten(para, 0)
            if theta is None:
                theta = theta_tmp
            else:
                theta = torch.cat([theta, theta_tmp], 0)
        return theta

    def get_theta_grad(self):
        theta_grad = None
        for para in self.model.parameters():
            theta_grad_tmp = torch.flatten(para.grad, 0)
            if theta_grad is None:
                theta_grad = theta_grad_tmp
            else:
                theta_grad = torch.cat([theta_grad, theta_grad_tmp], 0)
        return theta_grad

    def get_h_theta_grad(self):
        self.optimizer.zero_grad()
        self.h.backward(torch.ones_like(self.h))
        h_theta_grad = None
        for para in self.model.parameters():
            h_theta_grad_tmp = torch.flatten(para.grad, 0)
            if h_theta_grad is None:
                h_theta_grad = h_theta_grad_tmp
            else:
                h_theta_grad = torch.cat([h_theta_grad, h_theta_grad_tmp], 0)
        return h_theta_grad

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
        predict_h = self.model(x)
        return predict_h