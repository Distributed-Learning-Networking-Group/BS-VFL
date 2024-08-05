import os
import sys
import argparse
import torch
import torch.distributed as dist
import math
import time
import numpy as np
import json
import random as r
from sympy import *
from queue import Queue
from threading import Thread
from tasks import get_task_data
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cpu")
bandwidth_mbps = 300

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default=0, type=int, help='rank')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    parser.add_argument('--ps_ip', default='localhost', type=str, help='ip of ps')
    parser.add_argument('--ps_port', default='8888', type=str, help='port of ps')
    parser.add_argument('--task_name', default='mnist', type=str, help='task name')
    parser.add_argument('--use_gpu', action='store_true', help='use gpu or not')
    parser.add_argument('--is_asyn', action='store_true', help='asynchronous training or not')
    args = parser.parse_args()
    print(args)

    global device
    device = torch.device('cuda:0' if args.use_gpu else 'cpu')
    print(device)

    backend = 'gloo'
    os.environ['MASTER_ADDR'] = args.ps_ip
    os.environ['MASTER_PORT'] = args.ps_port
    if sys.platform == 'linux':
        network_card_list = os.listdir('/sys/class/net/')
        if "enp5s0f1" in network_card_list:
            os.environ['GLOO_SOCKET_IFNAME'] = "enp5s0f1"
    dist.init_process_group(backend=backend, world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        run_ps(args)
    else:
        run_client(args)

def run_ps(args):
    rank = args.rank
    world_size = args.world_size
    rank_list = [i+1 for i in range(world_size-1)]
    party,train_loader,test_loader,epochs,bound,lr,delta_T,CT = get_task_data(task_name=args.task_name,id=0,use_gpu=args.use_gpu)
    batch_size = train_loader.batch_size
    train_batches = len(train_loader)
    test_batches = len(test_loader)
    num_train_samples = train_batches * batch_size
    recording_period = 100
    global_step = 0
    running_time = 0
    recv_wait_time = 0
    t2_total = 0

    last_time = 0

    local_step = 0
    Q = party.n_iter
    D = bound
    
    shape_list = []
    predict_shape_list = []
    div = []
    VAFL_h_cache = None
    samples_cache = None
    
    parties_counter_list = [0]*(world_size-1)

    predict_h_list_queue = Queue()
    VAFL_h_queue = Queue()
    VAFL_grad_queue_list = [Queue() for _ in rank_list]

    VAFL_send_threads = []
    for VAFL_grad_queue,rank in zip(VAFL_grad_queue_list,rank_list):
        send_thread = Thread(target=process_communicate,daemon=True,args=('send',VAFL_grad_queue,[rank],rank-1))
        VAFL_send_threads.append(send_thread)
        send_thread.start()

    log_dir = os.path.join('summary_pic',args.task_name,time.strftime("%Y%m%d-%H%M-VAFL"))
    writer = SummaryWriter(log_dir=log_dir)

    log_data = {
        'Q':Q,
        'D':D,
        'accuracy&step':{'x':[],'y':[]},
        'accuracy&time':{'x':[],'y':[]},
        'loss':{'x':[],'y':[]},
        'running_time':{'x':[],'y':[]},
        'CT':{'x':[],'y':[]},
        'commucation_time':0,
        'computation_time':0,
        }

    for batch_idx, (_, target) in enumerate(train_loader):
        if samples_cache is None:
            tmp = list(target.shape)
            tmp[0] = num_train_samples
            samples_cache = torch.zeros(tmp)
        samples_cache[batch_idx*batch_size:(batch_idx+1)*batch_size] = target
    
    print("server set samples cache ok")

    if not shape_list:
        tmp = torch.zeros(2).long()
        shape_list = [torch.zeros_like(tmp) for _ in range(world_size)]
        print('gather shape..',tmp.shape)
        dist.gather(tensor=tmp,gather_list=shape_list)
        print('gather shape ok')

        shape_list.pop(0)
        num_features = 0
        div.append(0)
        for i,shape in enumerate(shape_list):
            shape_list[i] = shape.tolist()
            num_features += shape_list[i][1]-2
            div.append(num_features)
        print(shape_list)

        VAFL_h_cache = torch.zeros([num_train_samples,num_features],dtype=torch.float32)

        VAFL_pull_threads = []
        for rank,shape in zip(rank_list,shape_list):
            pull_thread = Thread(target=process_communicate,daemon=True,args=('pull',VAFL_h_queue,[rank],rank-1,[shape]))
            VAFL_pull_threads.append(pull_thread)
            pull_thread.start()
    
    # if not predict_shape_list:
    #     tmp = torch.zeros(2).long()
    #     predict_shape_list = [torch.zeros_like(tmp) for _ in range(world_size)]
    #     print('gather shape..',tmp.shape)
    #     dist.gather(tensor=tmp,gather_list=predict_shape_list)
    #     print('gather shape ok')

    #     predict_shape_list.pop(0)
    #     for i,shape in enumerate(predict_shape_list):
    #         predict_shape_list[i] = shape.tolist()
    #     print(predict_shape_list)

    #     predict_thread = Thread(target=process_communicate,daemon=True,args=('pull',predict_h_list_queue,rank_list,world_size-1,predict_shape_list))
    #     predict_thread.start()
    
    print(f'server start with batches={len(train_loader)}')
    
    while running_time < CT:
        if running_time - last_time > 20:
            send_data([torch.tensor(torch.tensor(1,dtype=torch.float32),dtype=torch.float32) for _ in rank_list],rank_list,tag=world_size-1)
            last_time = running_time
        
        party.model.train()
        start_time = time.time()

        timestamp1 = time.time()
        recv_start_time = time.time()
        h = VAFL_h_queue.get()[0]
        ids = np.array(h[:,0],dtype=np.int64)
        party_rank = int(h[0,1])
        h = h[:,2:]
        VAFL_h_cache[ids,div[party_rank-1]:div[party_rank]] = h
        h_list = [VAFL_h_cache[ids,div[i]:div[i+1]] for i in range(world_size-1)]
        recv_end_time = time.time()
        recv_spend_time = recv_end_time-recv_start_time
        recv_wait_time += recv_spend_time
        print('recv spend time: ',recv_spend_time)
        timestamp2 = time.time()
        
        for i,h in enumerate(h_list):
            h_list[i] = h.to(device)
        
        party.pull_parties_h(h_list)
        party.set_batch(samples_cache[ids].to(device))
        party.compute_parties_grad()
        parties_grad_list = party.send_parties_grad()

        for i,grad in enumerate(parties_grad_list):
            parties_grad_list[i] = grad.contiguous().cpu()
        VAFL_grad_queue_list[party_rank-1].put(parties_grad_list[party_rank-1])

        # for _ in range(Q):
        #     time.sleep(0.01)

        party.local_update()
        loss = party.get_loss()
        party.local_iterations()

        end_time = time.time()
        spend_time = end_time - start_time
        running_time += spend_time
        print(f"spend_time={spend_time} running_time={running_time}")
        t2_total += spend_time - (timestamp2 - timestamp1)
        print("t2_total",t2_total)

        global_step += 1
        local_step += Q
        parties_counter_list[party_rank-1] += 1
        print("parties_counter_list",parties_counter_list)

        writer.add_scalar("running_time", running_time, global_step)
        writer.add_scalar("recv_wait_time", recv_wait_time, global_step)
        writer.add_scalar("loss", loss.detach(), global_step)
        log_data["running_time"]['x'].append(global_step)
        log_data["running_time"]['y'].append(running_time)
        log_data["loss"]['x'].append(global_step)
        log_data["loss"]['y'].append(float(loss.detach()))

        if min(parties_counter_list) >= recording_period:
            print("server start predict")
            if not predict_shape_list:
                tmp = torch.zeros(2).long()
                predict_shape_list = [torch.zeros_like(tmp) for _ in range(world_size)]
                print('gather shape..',tmp.shape)
                dist.gather(tensor=tmp,gather_list=predict_shape_list)
                print('gather shape ok')

                predict_shape_list.pop(0)
                for i,shape in enumerate(predict_shape_list):
                    predict_shape_list[i] = shape.tolist()
                print(predict_shape_list)

                predict_thread = Thread(target=process_communicate,daemon=True,args=('pull',predict_h_list_queue,rank_list,world_size-1,predict_shape_list))
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

            for i,_ in enumerate(parties_counter_list):
                parties_counter_list[i] -= recording_period
            print("server finish predict")

    send_data([torch.tensor(-1,dtype=torch.float32) for _ in rank_list],rank_list,tag=world_size-1)
    for rank in rank_list:
        VAFL_grad_queue_list[rank-1].put(parties_grad_list[rank-1])
    print("server finish")

    log_data['commucation_time'] = recv_wait_time
    log_data['computation_time'] = t2_total
    
    t2_total = t2_total / local_step
    print("t2_total",t2_total)
        
    timestamp1 = time.time()
    tmp = torch.zeros(1)
    timestamp_list = [torch.zeros_like(tmp) for _ in range(world_size)]
    dist.gather(tensor=tmp,gather_list=timestamp_list)
    print('gather timestamp ok')
    max_timestamp = 0
    for timestamp in timestamp_list:
        max_timestamp = max(max_timestamp,timestamp)
    running_time += max_timestamp - timestamp1
    writer.add_scalar("running_time", running_time, global_step+1)
    print("running_time",running_time)

    parties_t0_list = [torch.zeros_like(tmp) for _ in range(world_size)]
    dist.gather(tensor=tmp,gather_list=parties_t0_list)
    print('gather t0 ok')

    parties_t3_list = [torch.zeros_like(tmp) for _ in range(world_size)]
    dist.gather(tensor=tmp,gather_list=parties_t3_list)
    print('gather t3 ok')

    res_list = [torch.zeros(shape,dtype=torch.double) for shape in shape_list]
    recv_data(res_list,rank_list,tag=world_size)
    timestamp2 = time.time()
    parties_t1_list = [timestamp2 - timestamp[0][0] for timestamp in res_list]
    print("parties_t1_list",parties_t1_list)

    max_t0_t1 = 0
    for t0,t1 in zip(parties_t0_list,parties_t1_list):
        max_t0_t1 = max(max_t0_t1,t0 + t1)
    print("max_t0_t1",max_t0_t1)

    max_t1_Qt3 = 0
    for t1,t3 in zip(parties_t1_list,parties_t3_list):
        max_t1_Qt3 = max(max_t1_Qt3,t1 + party.n_iter * t3)
    print("max_t1_Qt3",max_t1_Qt3)
    
    for T in range(1,global_step):
        CT = max(max_t0_t1 + T * party.n_iter * t2_total, max_t0_t1 + (T-1) * party.n_iter * t2_total + t2_total + max_t1_Qt3)
        writer.add_scalar("CT", CT, T)
        log_data["CT"]['x'].append(T)
        log_data["CT"]['y'].append(float(CT))
    print("CT",CT)


    dump_data = json.dumps(log_data)
    with open(os.path.join(log_dir,"log_data.json"), 'w') as file_object:
        file_object.write(dump_data)
    
    writer.close()

def run_client(args):
    rank = args.rank
    world_size = args.world_size
    ps_rank = 0
    party,train_loader,test_loader,epochs,bound = get_task_data(task_name=args.task_name,id=rank,use_gpu=args.use_gpu)
    print('bound',bound)
    batch_size = train_loader.batch_size
    train_batches = len(train_loader)
    test_batches = len(test_loader)
    num_train_samples = train_batches * batch_size
    recording_period = 100
    global_step = 0
    waiting_grad_num = 0
    ep = 0
    t0 = 0
    t1 = 0
    t3 = 0

    samples_cache = None

    shape = None
    predict_shape = None

    is_finish = False

    batch_cache = Queue()
    h_queue = Queue()
    grad_queue = Queue()
    predict_h_queue = Queue()
    flag_queue = Queue()

    send_thread = Thread(target=process_communicate,daemon=True,args=('send',h_queue,[ps_rank],rank-1))
    predict_thread = Thread(target=process_communicate,daemon=False,args=('send',predict_h_queue,[ps_rank],world_size-1))
    flag_thread = Thread(target=process_communicate,daemon=True,args=('pull',flag_queue,[ps_rank],world_size-1,[[]]))
    send_thread.start()
    predict_thread.start()
    flag_thread.start()

    for batch_idx, (data, _) in enumerate(train_loader):
        if samples_cache is None:
            tmp = list(data.shape)
            tmp[0] = num_train_samples
            samples_cache = torch.zeros(tmp)
        samples_cache[batch_idx*batch_size:(batch_idx+1)*batch_size] = data
    
    print("client set samples cache ok")

    # if shape is None:
    #     party.set_batch(samples_cache[:batch_size].to(device))
    #     party.compute_h()
    #     h = party.get_h()
    #     shape = list(h.shape)
    #     shape[1] += 2
    #     shape = torch.tensor(shape)
    #     print('gather shape..')
    #     dist.gather(tensor=shape)
    #     print('gather shape ok')

    #     pull_shape = list(h.shape)
    #     pull_thread = Thread(target=process_communicate,daemon=True,args=('pull',grad_queue,[ps_rank],rank-1,[pull_shape]))
    #     pull_thread.start()
    
    # if predict_shape is None:
    #     predict_h = None
    #     for test_data, _ in test_loader:
    #         predict_x = test_data.to(device)
    #         predict_h = party.predict(predict_x)
    #         break
    #     predict_shape = torch.tensor(predict_h.shape)
    #     print('gather shape..', predict_shape)
    #     dist.gather(tensor=predict_shape)
    #     print('gather shape ok')

    print(f'client start with batches={len(train_loader)}')

    while True:
        idlist = list(range(num_train_samples))
        r.shuffle(idlist)
        print(f'client start epoch {ep}\n')
        for batch_idx in range(train_batches):
            if not flag_queue.empty():
                flag = flag_queue.get()[0]
                if flag == -1:
                    is_finish = True
                    break

            party.model.train()

            timestamp1 = time.time()

            ids = idlist[batch_idx*batch_size:(batch_idx+1)*batch_size]
            party.set_batch(samples_cache[ids].to(device))
            # batch_cache.put([batch_idx,ids])
            print(f'client set batch {batch_idx}\n')
            
            party.compute_h()
            timestamp2 = time.time()
            t0 += timestamp2 - timestamp1
            h = party.get_h().cpu()
            timestamp3 = time.time()
            tmp = torch.zeros([batch_size,2],dtype=torch.float32)
            tmp[:,0] = torch.tensor(ids)
            tmp[:,1] = rank
            h = torch.cat([tmp,h],1)
            h_queue.put([h])

            if shape is None:
                shape = torch.tensor(h.shape)
                print('gather shape..')
                dist.gather(tensor=shape)
                print('gather shape ok')

                pull_shape = shape.tolist()
                pull_shape[1] -= 2
                pull_thread = Thread(target=process_communicate,daemon=True,args=('pull',grad_queue,[ps_rank],rank-1,[pull_shape]))
                pull_thread.start()

            waiting_grad_num += 1

            timestamp4 = time.time()
            t3 += timestamp4 - timestamp3

            # while not grad_queue.empty():
            grad = grad_queue.get()[0].to(device)
            timestamp5 = time.time()
            # cache_idx, ids = batch_cache.get()
            # party.set_batch(samples_cache[ids].to(device))

            print(f'client local update with batch {batch_idx}\n')
            party.pull_grad(grad)
            party.local_update()
            party.local_iterations()
            timestamp6 = time.time()
            t3 += timestamp6 - timestamp5

            waiting_grad_num -= 1
            global_step += 1

            if global_step % recording_period == 0:
                print("client start predict")
                for test_data, _ in test_loader:
                    predict_x = test_data.to(device)
                    predict_h = party.predict(predict_x)
                    if predict_shape is None:
                        predict_shape = torch.tensor(predict_h.shape)
                        print('gather shape..', predict_shape)
                        dist.gather(tensor=predict_shape)
                        print('gather shape ok')
                    predict_h_queue.put(predict_h.cpu())
                print("client finish predict")
                        
        ep += 1
        if is_finish:
            break
    
    print("client finish")
    
    dist.gather(tensor=torch.tensor(time.time()))
    print('gather timestamp ok')

    t0 = t0 / global_step
    print("t0",t0)
    t3 = t3 / (global_step * party.n_iter)
    print("t3",t3)
    dist.gather(tensor=torch.tensor(t0))
    print('gather t0 ok')
    dist.gather(tensor=torch.tensor(t3))
    print('gather t3 ok')

    dist.send(tensor=torch.full(shape.tolist(),time.time(),dtype=torch.double),dst=0,tag=world_size)
    
    predict_h_queue.put(-1)

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
    print('sending..')
    for i,rank in enumerate(dst_list):
        req_list.append(dist.isend(tensor=data_list[i],dst=rank,tag=tag))            
    for req in req_list:
        req.wait()
    print('send ok')

def recv_data(res_list,src_list,tag=0):
    if type(res_list) is not list:
        res_list = [res_list]
    if type(src_list) is not list:
        src_list = [src_list]
    req_list = []
    print('pulling..')
    for i,rank in enumerate(src_list):
        req_list.append(dist.irecv(tensor=res_list[i],src=rank,tag=tag))
    for req in req_list:
        req.wait()

    x = res_list[0]
    size_in_bits = x.element_size() * x.numel() * len(res_list) *8/1024/1024
    comm_time = size_in_bits / bandwidth_mbps
    if not tag == 7: 
        time.sleep(comm_time)
    print('pull ok')  


if __name__ == '__main__':
    main()