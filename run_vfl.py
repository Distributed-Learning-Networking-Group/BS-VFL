import os
import sys
import argparse
import torch
import torch.distributed as dist
import math
import time
import numpy as np
import json
import random
from sympy import *
from decimal import Decimal
from queue import Queue
from threading import Thread
from tasks import get_task_data,reset_ps_party
from itertools import combinations
from torch.utils.tensorboard import SummaryWriter

# device = torch.device("cpu")
gr = (math.sqrt(5) + 1) / 2
bandwidth_mbps = 300

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default=0, type=int, help='rank')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    parser.add_argument('--ps_ip', default='localhost', type=str, help='ip of ps')
    parser.add_argument('--ps_port', default='8888', type=str, help='port of ps')
    parser.add_argument('--task_name', default='mnist', type=str, help='task name')
    parser.add_argument('--use_gpu', action='store_true', help='use gpu or not')
    parser.add_argument('--is_asyn', action='store_true', help='asynchronous training or not')
    parser.add_argument('--use_reweight', action='store_true', help='reweight or not')
    parser.add_argument('--search', action='store_true', help='search Q_star or not')
    args = parser.parse_args()
    print(args)

    task_info = json.load(open('task_info.json','r',encoding='utf-8'))[args.task_name]
    global div
    div = task_info['div']

    global device
    device = torch.device('cuda:0' if args.use_gpu else 'cpu')
    print(device)
    print('reweight:', args.use_reweight)

    backend = 'gloo'
    os.environ['MASTER_ADDR'] = args.ps_ip
    os.environ['MASTER_PORT'] = args.ps_port
    if sys.platform == 'linux':
        network_card_list = os.listdir('/sys/class/net/')
        if "eth0" in network_card_list:
            os.environ['TP_SOCKET_IFNAME'] = "eth0"
            os.environ['GLOO_SOCKET_IFNAME'] = "eth0"
        # if "enp5s0f1" in network_card_list:
        #     os.environ['TP_SOCKET_IFNAME'] = "enp5s0f1"
        #     os.environ['GLOO_SOCKET_IFNAME'] = "enp5s0f1"
    dist.init_process_group(backend=backend, world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        if args.search:
            search_ps(args)
        else:
            run_ps(args)
    else:
        if args.search:
            search_cl(args)
        else:
            run_client(args)

def gss(f, a, b, tol=1e-5):
    """Golden-section search
    to find the minimum of f on [a,b]
    f: a strictly unimodal function on [a,b]

    Example:
    >>> f = lambda x: (x-2)**2
    >>> x = gss(f, 1, 5)
    >>> print("%.15f" % x)
    2.000009644875678
    """
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol:
        if f(c) < f(d):  # f(c) > f(d) to find the maximum
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2


def run_ps(args):
    rank = args.rank
    rank_list = [i+1 for i in range(args.world_size-1)]
    party,train_loader,test_loader,epochs,bound,lr,delta_T,CT = get_task_data(task_name=args.task_name,id=0,is_asyn=args.is_asyn,use_gpu=args.use_gpu)
    train_batches = len(train_loader)
    test_batches = len(test_loader)
    recording_period = 100

    global_step = 0
    running_time = 0
    recv_wait_time = 0
    t2_total = 0
    t2_first = 0
    t2_last = 0

    t2 = 0
    last_t2 = 0
    T_step = 0
    local_step = 0
    last_running_time = 0
    gap_time = 0
    gap_scale = 0

    Q = party.n_iter
    Q_l = Q
    Q_last = Q_l
    D = bound if bound > 0 else 1
    T = train_batches * epochs
    N = T / D
    N_prime = delta_T / D
    Cl = None
    c0 = lr * (T**0.5)
    loss_l = None
    E = loss_l
    G = None
    
    shape_list = []
    predict_shape_list = []
    theta_shape_list = []

    is_finish = False

    # h_list_queue = Queue()
    h_queue_list = [Queue() for _ in rank_list]
    grad_list_queue = Queue()
    predict_h_list_queue = Queue()

    send_thread = Thread(target=process_communicate,daemon=True,args=('send',grad_list_queue,rank_list,0))
    send_thread.start()

    log_dir = os.path.join('summary_pic',args.task_name,time.strftime("%Y%m%d-%H%M"))
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_scalar("Q_l&step", Q_l, global_step)
    writer.add_scalar("Q_l&time", Q_l, running_time*1000)

    log_data = {
        'Q':Q_l,
        'D':D,
        'accuracy&step':{'x':[],'y':[]},
        'accuracy&time':{'x':[],'y':[]},
        'loss':{'x':[],'y':[]},
        'running_time':{'x':[],'y':[]},
        'CT':{'x':[],'y':[]},
        'commucation_time':0,
        'computation_time':0,
        }

    print(f'server start with batches={len(train_loader)}')

    if True:
        if not predict_shape_list:
            tmp = torch.zeros(2).long()
            predict_shape_list = [torch.zeros_like(tmp) for _ in range(args.world_size)]
            print('gather predict shape..')
            dist.gather(tensor=tmp,gather_list=predict_shape_list)
            print('gather shape ok')

            predict_shape_list.pop(0)
            for i,shape in enumerate(predict_shape_list):
                predict_shape_list[i] = shape.tolist()
            print(predict_shape_list)

            predict_thread = Thread(target=process_communicate,daemon=True,args=('pull',predict_h_list_queue,rank_list,1,predict_shape_list))
            predict_thread.start()

        loss_list = []
        correct_list = []
        acc_list = []

        for _, test_target in test_loader:
            predict_h_list = predict_h_list_queue.get()
            for i,h in enumerate(predict_h_list):
                predict_h_list[i] = h.to(device)
            
            predict_y = test_target.to(device)
            loss,correct,accuracy = party.predict(predict_h_list,predict_y)
            loss_list.append(loss)
            correct_list.append(correct)
            acc_list.append(accuracy)
        loss = sum(loss_list) / test_batches
        correct = sum(correct_list) / test_batches
        accuracy = sum(acc_list) / test_batches

        writer.add_scalar("accuracy&step", accuracy, global_step)
        writer.add_scalar("accuracy&time", accuracy, running_time*1000)
        log_data["accuracy&step"]['x'].append(global_step)
        log_data["accuracy&step"]['y'].append(accuracy)
        log_data["accuracy&time"]['x'].append(running_time*1000)
        log_data["accuracy&time"]['y'].append(accuracy)
        print(f'server figure out loss={loss} correct={correct} accuracy={accuracy}\n')

    for ep in range(epochs):
        print(f'server start epoch {ep}')

        for batch_idx, (_, target) in enumerate(train_loader):
            if running_time >= CT:
                send_data([torch.tensor(-1,dtype=torch.float32) for _ in rank_list],rank_list,tag=1)
                grad_list_queue.put(parties_grad_list)
                is_finish = True
                break
            
            if Cl is None and global_step >= delta_T:
                Cl = (t2 / (Q_l * T_step)) * N_prime * D * Q_l

            if Cl is not None and running_time > last_running_time + Cl:
                t2 = t2 / (Q * T_step)
                gap_time = gap_time / (Q * T_step)
                E = loss_l
                if G is None:
                    G = symbols('G')
                    solution = solve(
                        (-E * ((N * t2)**0.5) / (c0 * ((Cl * N_prime)**0.5) * ((Q_l**3)**0.5))
                         + 2*G * ((t2 * Q_last)**2) * Q_l / (Cl**2)
                         + 9*G * t2 * (D+1) * (Q_l**2) / (Cl * (2*(D**2) + D))
                         ),
                        G
                    )
                    G = solution[0]
                else:
                    Q_last = Q_l
                    f = lambda Q_l:abs(-E * ((N * t2)**0.5) / (c0 * ((Cl * N_prime)**0.5) * ((Q_l**3)**0.5))
                         + 2*G * ((t2 * Q_last)**2) * Q_l / (Cl**2)
                         + 9*G * t2 * (D+1) * (Q_l**2) / (Cl * (2*(D**2) + D))
                         )
                    Q_l = gss(f,0,Q_last*2)
                    # print("Q_l:",Q_l)
                
                # if Q_l > 1:
                #     # Q = math.ceil(Q_l) if Q_l - int(Q_l) > random.uniform(0,1) else int(Q_l)
                #     Q = round(Q_l)
                # else:
                #     Q = 1
                # party.n_iter = Q
                send_data([torch.tensor(Q,dtype=torch.float32) for _ in rank_list],rank_list,tag=1)

                Cl = t2 * N_prime * D * Q_l
                last_t2 = t2 - gap_time
                writer.add_scalar("t2", t2, global_step)
                gap_time = 0
                t2 = 0
                T_step = 0
                last_running_time = running_time
                gap_scale = np.random.rand()*10

                writer.add_scalar("Q_l&step", Q_l, global_step)
                writer.add_scalar("Q_l&time", Q_l, running_time*1000)
                writer.add_scalar("Q&time", Q, running_time*1000)
                writer.add_scalar("Cl", Cl, global_step)

            party.model.train()
            start_time = time.time()

            target = target.to(device)
            party.set_batch(target)
            print(f'server set batch {batch_idx}\n')

            if not shape_list:
                tmp = torch.zeros(2).long()
                shape_list = [torch.zeros_like(tmp) for _ in range(args.world_size)]
                print('gather shape..')
                dist.gather(tensor=tmp,gather_list=shape_list)
                print('gather shape ok')

                shape_list.pop(0)
                for i,shape in enumerate(shape_list):
                    shape_list[i] = shape.tolist()
                print(shape_list)

                # pull_thread = Thread(target=process_communicate,daemon=True,args=('pull',h_list_queue,rank_list,0,shape_list))
                # pull_thread.start()
                pull_thread_list = []
                for rank in rank_list:
                    pull_thread = Thread(target=process_communicate,daemon=True,args=('pull',h_queue_list[rank-1],[rank],0,[shape_list[rank-1]]))
                    pull_thread.start()
                    pull_thread_list.append(pull_thread)
                
            timestamp1 = time.time()
            recv_start_time = time.time()
            # print("h_list_queue",h_list_queue.qsize(),recv_start_time)
            h_list = []
            for h_queue in h_queue_list:
                h_list.append(h_queue.get()[0])
                print("h_queue",h_queue.qsize())

            recv_end_time = time.time()
            recv_spend_time = recv_end_time-recv_start_time
            recv_wait_time += recv_spend_time
            print('recv spend time: ',recv_spend_time)
            timestamp2 = time.time()

            for i,h in enumerate(h_list):
                h_list[i] = h.to(device)

            party.pull_parties_h(h_list) # concat / avg 得到server的输入h
            party.compute_parties_grad() # 得到返回给每个client的grad
            parties_grad_list = party.send_parties_grad()

            for i,grad in enumerate(parties_grad_list):
                parties_grad_list[i] = grad.contiguous().cpu()
            grad_list_queue.put(parties_grad_list)

            # for _ in range(Q):
            #     time.sleep(0.01)
            
            timestamp3 = time.time()
            # gap = np.random.poisson(1) * last_t2 * gap_scale
            # print("gap:",gap)
            # time.sleep(gap)
            timestamp4 = time.time()
            gap_time += timestamp4 - timestamp3

            party.local_update()
            loss = party.get_loss()
            if loss_l is None:
                loss_l = loss
            party.local_iterations()

            end_time = time.time()
            spend_time = end_time - start_time
            running_time += spend_time
            # print(f"spend_time={spend_time} running_time={running_time}")
            t2_total += spend_time - (timestamp2 - timestamp1)
            # t2_first += timestamp1 - start_time + (timestamp3 - timestamp2)
            # t2_last += end_time - timestamp3
            t2 += spend_time - (timestamp2 - timestamp1)
            # print("t2_total",t2_total)

            global_step += 1
            local_step += Q
            T_step += 1

            writer.add_scalar("running_time", running_time, global_step)
            writer.add_scalar("recv_wait_time", recv_wait_time, global_step)
            writer.add_scalar("loss", loss.detach(), global_step)
            log_data["running_time"]['x'].append(global_step)
            log_data["running_time"]['y'].append(running_time)
            log_data["loss"]['x'].append(global_step)
            log_data["loss"]['y'].append(float(loss.detach()))

            if global_step % recording_period == 0:
                if not predict_shape_list:
                    tmp = torch.zeros(2).long()
                    predict_shape_list = [torch.zeros_like(tmp) for _ in range(args.world_size)]
                    print('gather shape..',tmp.shape)
                    dist.gather(tensor=tmp,gather_list=predict_shape_list)
                    print('gather shape ok')

                    predict_shape_list.pop(0)
                    for i,shape in enumerate(predict_shape_list):
                        predict_shape_list[i] = shape.tolist()
                    print(predict_shape_list)

                    predict_thread = Thread(target=process_communicate,daemon=True,args=('pull',predict_h_list_queue,rank_list,1,predict_shape_list))
                    predict_thread.start()

                loss_list = []
                correct_list = []
                acc_list = []

                for _, test_target in test_loader:
                    predict_h_list = predict_h_list_queue.get()
                    for i,h in enumerate(predict_h_list):
                        predict_h_list[i] = h.to(device)
                    
                    predict_y = test_target.to(device)
                    loss,correct,accuracy = party.predict(predict_h_list,predict_y)
                    loss_list.append(loss)
                    correct_list.append(correct)
                    acc_list.append(accuracy)
                loss = sum(loss_list) / test_batches
                correct = sum(correct_list) / test_batches
                accuracy = sum(acc_list) / test_batches

                writer.add_scalar("accuracy&step", accuracy, global_step)
                writer.add_scalar("accuracy&time", accuracy, running_time*1000)
                log_data["accuracy&step"]['x'].append(global_step)
                log_data["accuracy&step"]['y'].append(accuracy)
                log_data["accuracy&time"]['x'].append(running_time*1000)
                log_data["accuracy&time"]['y'].append(accuracy)
                print(f'server figure out loss={loss} correct={correct} accuracy={accuracy}\n')

        if is_finish:
            break

    log_data['commucation_time'] = recv_wait_time
    log_data['computation_time'] = t2_total
    
    t2_total = t2_total / local_step
    # print("t2_total",t2_total)
    # t2_first = t2_first / global_step
    # print("t2_first",t2_first)
    # t2_last = t2_last / (local_step - global_step)
    # print("t2_last",t2_last)
        
    timestamp1 = time.time()
    tmp = torch.zeros(1)
    timestamp_list = [torch.zeros_like(tmp) for _ in range(args.world_size)]
    dist.gather(tensor=tmp,gather_list=timestamp_list)
    # print('gather timestamp ok')
    max_timestamp = 0
    for timestamp in timestamp_list:
        max_timestamp = max(max_timestamp,timestamp)
    running_time += max_timestamp - timestamp1
    writer.add_scalar("running_time", running_time, global_step+1)
    # print("running_time",running_time)

    parties_t0_list = [torch.zeros_like(tmp) for _ in range(args.world_size)]
    dist.gather(tensor=tmp,gather_list=parties_t0_list)
    # print('gather t0 ok')

    parties_t3_list = [torch.zeros_like(tmp) for _ in range(args.world_size)]
    dist.gather(tensor=tmp,gather_list=parties_t3_list)
    # print('gather t3 ok')

    res_list = [torch.zeros(shape,dtype=torch.double) for shape in shape_list]
    recv_data(res_list,rank_list,tag=2)
    timestamp2 = time.time()
    parties_t1_list = [timestamp2 - timestamp[0][0] for timestamp in res_list]
    # print("parties_t1_list",parties_t1_list)

    max_t0_t1 = 0
    for t0,t1 in zip(parties_t0_list,parties_t1_list):
        max_t0_t1 = max(max_t0_t1,t0 + t1)
    # print("max_t0_t1",max_t0_t1)

    max_t1_Qt3 = 0
    for t1,t3 in zip(parties_t1_list,parties_t3_list):
        max_t1_Qt3 = max(max_t1_Qt3,t1 + party.n_iter * t3)
    # print("max_t1_Qt3",max_t1_Qt3)
    
    for T in range(1,global_step):
        CT = max(max_t0_t1 + T * party.n_iter * t2_total, max_t0_t1 + (T-1) * party.n_iter * t2_total + t2_total + max_t1_Qt3)
        writer.add_scalar("CT", CT, T)
        log_data["CT"]['x'].append(T)
        log_data["CT"]['y'].append(float(CT))
    # print("CT",CT)


    dump_data = json.dumps(log_data)
    with open(os.path.join(log_dir,"log_data.json"), 'w') as file_object:
        file_object.write(dump_data)
    
    writer.close()

def run_client(args):
    rank = args.rank
    ps_rank = 0
    party,train_loader,test_loader,epochs,bound = get_task_data(task_name=args.task_name,id=rank,is_asyn=args.is_asyn,use_gpu=args.use_gpu)
    print('bound',bound)
    train_batches = len(train_loader)
    test_batches = len(test_loader)
    recording_period = 100
    global_step = 0
    waiting_grad_num = 0
    t0 = 0
    t1 = 0
    t3 = 0

    shape = None
    predict_shape = None
    theta_shape = None
    last_theta = None

    is_finish = False

    batch_cache = Queue()
    h_queue = Queue()
    grad_queue = Queue()
    predict_h_queue = Queue()
    Ql_queue = Queue()

    send_thread = Thread(target=process_communicate,daemon=True,args=('send',h_queue,[ps_rank],0,None))
    predict_thread = Thread(target=process_communicate,daemon=False,args=('send',predict_h_queue,[ps_rank],1))
    pull_Ql_thread = Thread(target=process_communicate,daemon=True,args=('pull',Ql_queue,[ps_rank],1,[[]]))
    send_thread.start()
    predict_thread.start()
    pull_Ql_thread.start()

    # print(f'client start with batches={len(train_loader)}')

    if True:
        for test_data, _ in test_loader:
            predict_x = test_data.to(device)
            predict_h = party.predict(predict_x)
            if predict_shape is None:
                predict_shape = torch.tensor(predict_h.shape)
                print('gather predict shape..', predict_shape)
                dist.gather(tensor=predict_shape)
                print('gather shape ok')
            predict_h_queue.put(predict_h.cpu())

    for ep in range(epochs):
        print(f'client start epoch {ep}\n')

        # validation_period = 30 if ep < 10 else 10

        for batch_idx, (data, _) in enumerate(train_loader):
            if not Ql_queue.empty():
                Q_l = int(Ql_queue.get()[0])
                if Q_l == -1:
                    is_finish = True
                    break

                party.n_iter = Q_l
                # print("Q_l:",Q_l)
            
            party.model.train()

            timestamp1 = time.time()

            data = data.to(device)
            party.set_batch(data)
            batch_cache.put([global_step,data])
            # print(f'client set batch {batch_idx}\n')
            
            party.compute_h()
            timestamp2 = time.time()
            t0 += timestamp2 - timestamp1
            h = party.get_h()

            timestamp3 = time.time()
            h_queue.put(h.cpu())
            # print('h_queue:',h_queue.qsize())

            if shape is None:
                shape = torch.tensor(h.shape)
                print('gather shape..')
                dist.gather(tensor=shape)
                print('gather shape ok')

                pull_thread = Thread(target=process_communicate,daemon=True,args=('pull',grad_queue,[ps_rank],0,[shape.tolist()]))
                pull_thread.start()

            waiting_grad_num += 1

            timestamp4 = time.time()
            t3 += timestamp4 - timestamp3

            while waiting_grad_num > bound or not grad_queue.empty() or (ep == epochs - 1 and batch_idx == train_batches - 1 and waiting_grad_num > 0):
                grad = grad_queue.get()[0].to(device)
                timestamp5 = time.time()
                cache_idx, batch_x = batch_cache.get()
                # print(f'client local update with batch {cache_idx}\n')
                if not cache_idx == global_step:
                    party.set_batch(batch_x)
                    party.compute_h()
                party.pull_grad(grad)
                party.local_update()
                party.local_iterations()

                timestamp6 = time.time()
                t3 += timestamp6 - timestamp5

                waiting_grad_num -= 1
                global_step += 1

                if global_step % recording_period == 0:
                    for test_data, _ in test_loader:
                        predict_x = test_data.to(device)
                        predict_h = party.predict(predict_x)
                        if predict_shape is None:
                            predict_shape = torch.tensor(predict_h.shape)
                            print('gather predict shape..', predict_shape)
                            dist.gather(tensor=predict_shape)
                            print('gather predict shape ok')
                        predict_h_queue.put(predict_h.cpu())                  

        if is_finish:
            break
    
    dist.gather(tensor=torch.tensor(time.time()))
    # print('gather timestamp ok')

    t0 = t0 / global_step
    # print("t0",t0)
    t3 = t3 / (global_step * party.n_iter)
    # print("t3",t3)
    dist.gather(tensor=torch.tensor(t0))
    # print('gather t0 ok')
    dist.gather(tensor=torch.tensor(t3))
    # print('gather t3 ok')

    dist.send(tensor=torch.full(shape.tolist(),time.time(),dtype=torch.double),dst=0,tag=2)
    
    predict_h_queue.put(-1)

def search_ps(args):
    rank = args.rank
    rank_list = [i+1 for i in range(args.world_size-1)]
    party,train_loader,test_loader,epochs,bound,lr,delta_T,CT,search_CT,c0, = get_task_data(task_name=args.task_name,id=0,is_asyn=args.is_asyn,use_gpu=args.use_gpu,search=args.search)
    train_batches = len(train_loader)
    test_batches = len(test_loader)
    recording_period = 10
    global_step = 0
    running_time = 0
    recv_wait_time = 0
    l0 = None
       
    min_loss = None
    Q_star = 1
    current_T = 0
    search_T = 0
    current_t2 = 0
    last_t2 = 0.135
    search_t2 = 0
    search_c0 = 0
    search_l0 = 0
    search_lr = 0


    flag_list = [torch.tensor(1.) for _ in rank_list]
    
    shape_list = []

    h_list_queue = Queue()
    grad_list_queue = Queue()

    send_thread = Thread(target=process_communicate,daemon=True,args=('send',grad_list_queue,rank_list,0))
    send_thread.start()

    save_dir = os.path.join('search_results',args.task_name)
    save_name = time.strftime("%Y%m%d-%H%M")+'.txt'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir,save_name), 'w') as file_object:
        file_object.write(f"task_name: {args.task_name}\n")
        file_object.write(f"search_CT: {search_CT}\n")
        file_object.write(f"CT: {CT}\n")
        file_object.write(f"c0: {c0}\n")
        # file_object.write(f"lr: {lr}\n")
    
    for i in range(10):
        Q = i+1

        lr = c0 / ((search_CT / (Q * last_t2))**0.5)
        send_data([torch.tensor(lr,dtype=torch.float32) for _ in rank_list],rank_list,tag=1)
        print("lr:",lr)
        party = reset_ps_party(task_name=args.task_name,id=0,lr=lr)
        global_step = 0
        running_time = 0
        recv_wait_time = 0
        current_t2 = 0

        is_finish = False
        recv_h_count = 0
        max_h_count = 0
        loss = None
        party.n_iter = Q

        # recv_data([torch.tensor(1.) for _ in rank_list],rank_list,tag=3)
        send_data(flag_list,rank_list,tag=3)

        print(f'server start with batches={len(train_loader)}')

        for ep in range(epochs):
            print(f'server start epoch {ep}')
            for batch_idx, (_, target) in enumerate(train_loader):
                if running_time >= search_CT:
                    grad_list_queue.join()
                    send_data(flag_list,rank_list,tag=3)
                    grad_list_queue.put(parties_grad_list)
                    grad_list_queue.join()

                    h_count_list = [torch.tensor(1.) for _ in rank_list]
                    recv_data(h_count_list,rank_list,tag=3)
                    print('h_count_list',h_count_list)
                    max_h_count = 0
                    for h_count in h_count_list:
                        max_h_count = max(max_h_count,h_count)
                    send_data([max_h_count for _ in rank_list],rank_list,tag=4)

                    current_T = global_step
                    is_finish = True
                    break

                party.model.train()
                start_time = time.time()

                target = target.to(device)
                party.set_batch(target)
                print(f'server set batch {batch_idx}\n')

                if not shape_list:
                    tmp = torch.zeros(2).long()
                    shape_list = [torch.zeros_like(tmp) for _ in range(args.world_size)]
                    print('gather shape..',tmp.shape)
                    dist.gather(tensor=tmp,gather_list=shape_list)
                    print('gather shape ok')

                    shape_list.pop(0)
                    for i,shape in enumerate(shape_list):
                        shape_list[i] = shape.tolist()
                    print(shape_list)

                    pull_thread = Thread(target=process_communicate,daemon=True,args=('pull',h_list_queue,rank_list,0,shape_list))
                    pull_thread.start()
                timestamp1 = time.time()
                recv_start_time = time.time()
                print("h_list_queue",h_list_queue.qsize(),recv_start_time)
                h_list = h_list_queue.get()
                recv_end_time = time.time()
                recv_spend_time = recv_end_time-recv_start_time
                recv_wait_time += recv_spend_time
                print('recv spend time: ',recv_spend_time)
                timestamp2 = time.time()
                
                for i,h in enumerate(h_list):
                    h_list[i] = h.to(device)
                recv_h_count += 1

                party.pull_parties_h(h_list)
                party.compute_parties_grad()
                parties_grad_list = party.send_parties_grad()

                for i,grad in enumerate(parties_grad_list):
                    parties_grad_list[i] = grad.contiguous().cpu()
                grad_list_queue.put(parties_grad_list)
                timestamp3 = time.time()
                
                party.local_update()
                loss = party.get_loss()
                if l0 is None:
                    l0 = loss

                party.local_iterations()

                end_time = time.time()
                spend_time = end_time - start_time
                running_time += spend_time
                print(f"spend_time={spend_time} running_time={running_time}")
                current_t2 += spend_time - (timestamp2 - timestamp1)
                print("current_t2",current_t2)

                global_step += 1
            
            if is_finish:
                while not (h_list_queue.empty() and recv_h_count >= max_h_count):
                    h_list_queue.get()
                    recv_h_count += 1
                break
        
        current_t2 = current_t2 / (current_T * Q)
        last_t2 = current_t2

        if min_loss is None or loss < min_loss:
            min_loss = loss
            Q_star = Q
            search_T = current_T
            search_t2 = current_t2
            search_lr = lr
            search_l0 = l0
        print(f"Q: {Q}\tQ_star: {Q_star}\tloss: {loss}\tmin_loss: {min_loss}\tcurrent_T: {current_T}\tsearch_T: {search_T}\tcurrent_t2: {current_t2}\tsearch_t2: {search_t2}\tlr: {lr}")
        with open(os.path.join(save_dir,save_name), 'a') as file_object:
            file_object.write(f"Q: {Q}\tQ_star: {Q_star}\tloss: {loss}\tmin_loss: {min_loss}\tcurrent_T: {current_T}\tsearch_T: {search_T}\tcurrent_t2: {current_t2}\tsearch_t2: {search_t2}\tlr: {lr}\n")

    search_c0 = search_lr * (search_T**0.5)
    print("search_c0",search_c0)
    E_square = search_l0**2
    print("E_square",E_square)
    A = ((E_square * search_CT) / (9*(c0**6) * search_t2 * (Q_star**7)))**0.5
    print("A",A)
    Q_star = ((E_square * CT) / (9*(c0**6) * search_t2 * A**2))**(1/7)
    print("Q_star",Q_star)

    with open(os.path.join(save_dir,save_name), 'a') as file_object:
        file_object.write(f"search_t2: {search_t2}\n")
        file_object.write(f"search_c0: {search_c0}\n")
        file_object.write(f"E_square: {E_square}\n")
        file_object.write(f"A: {A}\n")
        file_object.write(f"Q_star: {Q_star}\n")

def search_cl(args):
    rank = args.rank
    ps_rank = 0
    party,train_loader,test_loader,epochs,bound = get_task_data(task_name=args.task_name,id=rank,is_asyn=args.is_asyn,use_gpu=args.use_gpu)
    print('bound',bound)
    train_batches = len(train_loader)
    test_batches = len(test_loader)
    recording_period = 10
    global_step = 0
    waiting_grad_num = 0
    t0 = 0
    t1 = 0
    t3 = 0

    shape = None

    batch_cache = Queue()
    h_queue = Queue()
    grad_queue = Queue()
    predict_h_queue = Queue()
    flag_queue = Queue()

    send_thread = Thread(target=process_communicate,daemon=True,args=('send',h_queue,[ps_rank],0))
    flag_thread = Thread(target=process_communicate,daemon=True,args=('pull',flag_queue,[ps_rank],3,[[]]))
    send_thread.start()
    flag_thread.start()

    for i in range(10):
        Q = i+1

        lr = torch.tensor(0.)
        recv_data([lr],[ps_rank],tag=1)
        lr = float(lr)
        print("lr:",lr)
        party = reset_ps_party(task_name=args.task_name,id=rank,lr=lr)
        global_step = 0
        
        is_finish = False
        h_count = 0
        party.n_iter = Q

        # send_data(torch.tensor(1.),[ps_rank],tag=3)
        flag_queue.get()

        print(f'client start with batches={len(train_loader)}')

        for ep in range(epochs):
            print(f'client start epoch {ep}\n')
            for batch_idx, (data, _) in enumerate(train_loader):
                if not flag_queue.empty():
                    flag_queue.get()

                    send_data([torch.tensor(h_count,dtype=torch.float32)],[ps_rank],tag=3)
                    max_h_count = torch.tensor(1.)
                    recv_data([max_h_count],[ps_rank],tag=4)
                    print("max_h_count",max_h_count)
                    print("h_count:",h_count)
                    while h_count < max_h_count:
                        h_queue.put(h.cpu())
                        h_count += 1
                        print("h_count:",h_count)
                    
                    h_queue.join()
                    is_finish = True
                    break

                party.model.train()

                data = data.to(device)
                party.set_batch(data)
                batch_cache.put([batch_idx,data])
                print(f'client set batch {batch_idx}\n')
                
                party.compute_h()
                h = party.get_h()
                h_queue.put(h.cpu())
                h_count += 1
                if shape is None:
                    shape = torch.tensor(h.shape)
                    print('gather shape..')
                    dist.gather(tensor=shape)
                    print('gather shape ok')

                    pull_thread = Thread(target=process_communicate,daemon=True,args=('pull',grad_queue,[ps_rank],0,[shape.tolist()]))
                    pull_thread.start()

                waiting_grad_num += 1

                while waiting_grad_num > bound or not grad_queue.empty() or (ep == epochs - 1 and batch_idx == train_batches - 1 and waiting_grad_num > 0):
                    grad = grad_queue.get()[0].to(device)
                    cache_idx, batch_x = batch_cache.get()
                    party.set_batch(batch_x)

                    print(f'client local update with batch {cache_idx}\n')
                    party.compute_h()
                    party.pull_grad(grad)
                    party.local_update()
                    party.local_iterations()

                    waiting_grad_num -= 1
                    global_step += 1

            if is_finish:
                while not grad_queue.empty(): grad_queue.get()
                break

def process_communicate(task_name,dq,ranks,tag,shape_list=None):
    print(f'{task_name} thread start\n')

    if type(ranks) is not list:
        ranks = [ranks]
    if type(shape_list) is not list:
        shape_list = [shape_list]
        
    while(True):
        if task_name == 'send':
            data_list = dq.get()
            if type(data_list) is int and data_list==-1:
                break
            if type(data_list) is not list:
                data_list = [data_list]
            send_data(data_list,ranks,tag)
            dq.task_done()
        elif task_name == 'pull':
            res_list = [torch.zeros(shape) for shape in shape_list]
            recv_data(res_list,ranks,tag)
            dq.put(res_list)
        

def send_data(data_list,dst_list,tag=0):
    if type(data_list) is not list:
        data_list = [data_list]
    if type(dst_list) is not list:
        dst_list = [dst_list]
    req_list = []
    # print('sending..')
    for i,rank in enumerate(dst_list):
        req_list.append(dist.isend(tensor=data_list[i],dst=rank,tag=tag))            
    for req in req_list:
        req.wait()
    # print('send ok')

def recv_data(res_list,src_list,tag=0):
    if type(res_list) is not list:
        res_list = [res_list]
    if type(src_list) is not list:
        src_list = [src_list]
    req_list = []
    # print('pulling..')
    for i,rank in enumerate(src_list):
        req_list.append(dist.irecv(tensor=res_list[i],src=rank,tag=tag))
    for req in req_list:
        req.wait()

    x = res_list[0]
    size_in_mb = x.element_size() * x.numel() * len(res_list) *8/1024/1024
    comm_time = size_in_mb / bandwidth_mbps
    if not tag == 1:
        time.sleep(comm_time)
    # print('pull ok')


if __name__ == '__main__':
    main()