import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm


def loss_func(pred, label):

    CE = nn.CrossEntropyLoss()
    loss = 0

    Ndigits = label.shape[1]
    for d in range(Ndigits):
        pred_d = pred[:,d*10:(d+1)*10]
        label_d = label[:,d]
        loss += CE(pred_d,label_d)

    return loss

def acc_func(pred, label):

    correct = []

    Ndigits = label.shape[1]

    for d in range(Ndigits):
        pred_d = pred[:,d*10:(d+1)*10]
        ans_d = torch.argmax(pred_d,dim=1)
        correct.append(ans_d == label[:,d])

    Ncorrect = torch.sum(torch.concat(correct,dim=0)).item()
    Ntotal   = label.shape[0]*Ndigits

    return Ncorrect, Ntotal


def train_valid_loop(net, train_dl, valid_dl, Nepochs, learning_rate=0.001, output_file='saved_model.pt'):

    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []

    ### Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # optimizer = torch.optim.Adam(net.parameters())
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4)

    ### Check for GPU
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print('Found GPU!')
        device = torch.device("cuda:0")

    net.to(device)

    for epoch in tqdm(range(Nepochs)):

        ### Counts
        train_total, train_correct = 0, 0
        valid_total, valid_correct = 0, 0

        ### Training
        net.train()

        train_loss_epoch = []
        for xb,yb in tqdm(train_dl):
            xb = xb.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            pred = net(xb)

            ### Loss
            loss = loss_func(pred,yb)
            loss.backward()
            train_loss_epoch.append(loss.item())
            optimizer.step()

            ### Accuracy
            Ncorr, Ntot = acc_func(pred,yb)
            train_correct +=Ncorr
            train_total   +=Ntot

        train_loss.append(np.mean(train_loss_epoch))
        train_acc.append(train_correct/train_total)
        #scheduler.step()

        ### Validation
        net.eval()

        valid_loss_epoch = []
        for xb,yb in tqdm(valid_dl):
            xb = xb.to(device)
            yb = yb.to(device)

            pred = net(xb)

            ### Loss
            loss = loss_func(pred,yb)
            valid_loss_epoch.append(loss.item())

            ### Accuracy
            Ncorr, Ntot = acc_func(pred,yb)
            valid_correct +=Ncorr
            valid_total   +=Ntot

        valid_loss.append(np.mean(valid_loss_epoch))
        valid_acc.append(valid_correct/valid_total)

        ### Model checkpointing
        if epoch > 0:
            if valid_loss[-1] < min(valid_loss[:-1]):
                torch.save(net.state_dict(), output_file)

        print(f'Epoch: {epoch}, Train loss (acc): {train_loss[-1]} ({train_acc[-1]}), Valid loss (acc): {valid_loss[-1]} ({valid_acc[-1]})')

    #Bring net back to CPU
    net.cpu()

    return {'train_loss': train_loss, 'valid_loss': valid_loss, 'train_acc': train_acc, 'valid_acc': valid_acc}
