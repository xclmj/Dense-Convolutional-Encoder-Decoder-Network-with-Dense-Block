"""
Convolutional Encoder-Decoder Networks for Image-to-Image Regression

Surrogate: y = f(x, t)

Adapted from Yinhao Zhu (yzhu10@nd.edu), Shaoxing Mo (smo@smail.nju.edu.cn)

Aug 15, 2019

modified by Cong Xiao (C.Xiao@tudelft.nl)

"""

import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.utils.data import Dataset
from dense_ed import DenseEDT  # DenseEDTI
from data import load_data, load_data_dynamic, load_tensor,load_tensor_expanded
from args import Parser
import h5py
import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from time import time
import seaborn as sns
import torch
from sklearn.metrics import mean_squared_error
plt.switch_backend('agg')


args = Parser().parse()
device = th.device("cuda" if th.cuda.is_available() else "cpu")
device = th.device("cpu" if th.cuda.is_available() else "cuda")
print(device)

# enters time in latent
model = DenseEDT(1, 2, blocks=args.blocks, times=args.times,
                 growth_rate=args.growth_rate, drop_rate=args.drop_rate,
                 bn_size=args.bn_size, num_init_features=args.init_features,
                 bottleneck=args.bottleneck, time_channels=args.zt).to(device)
print(model)

# load data
h5py_dir = args.data_dir 
train_data_dir = h5py_dir + '/lhs{}_t{}_expanded.hdf5'.format(args.n_train, len(args.times))
test_data_dir = h5py_dir + '/lhs{}_t{}_expanded.hdf5'.format(args.n_test, len(args.times))
interp_data_dir = h5py_dir + '/lhs{}_t{}.hdf5'.format(400, 3)   # data used to plot the results at t= 5, 50, and 100 days
train_loader, train_stats = load_data_dynamic(train_data_dir, args.batch_size)
test_loader, test_stats = load_data_dynamic(test_data_dir, args.test_batch_size)
interp_x, interp_y, interp_times = load_tensor(interp_data_dir)
print('Loaded data!')


if args.loss_fn == "mse":
    loss_fn = nn.MSELoss(reduction='sum')
elif args.loss_fn == "l1":
    loss_fn = nn.L1Loss()
elif args.loss_fn == 'huber':
    loss_fn = nn.SmoothL1Loss()

optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.1, patience=10,
                    verbose=True, threshold=0.0001, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-08)

n_out_pixels_train = len(train_loader.dataset) * train_loader.dataset[0][1].numel()
n_out_pixels_test = len(test_loader.dataset) * test_loader.dataset[0][1].numel()

def test(epoch, plot_intv):
    model.eval()
    loss = 0.
    for batch_idx, (input, target, times) in enumerate(test_loader):
        input, target, times = input.to(device), target[:, :2].to(device), \
                               times.to(device)

        with th.no_grad():
            output = model(input, times)
        loss += loss_fn(output, target).item()

        # plot predictions
        if epoch % plot_intv == 0 and batch_idx == len(test_loader) - 1:
            np.random.seed(1)
            if epoch == args.n_epochs:
                n_samples = 4
            else:
                n_samples = 2
            idx = th.LongTensor(np.random.choice(interp_x.size(0), n_samples, replace=False))
            x = interp_x[idx]
            y = interp_y[idx]
            print("Index of data: {}".format(idx))
            print(x.shape)
            print(y.shape)

            n_grid = 100
            xx = np.linspace(0, 1, n_grid)
            yy = np.linspace(0, 1, n_grid)
            X, Y = np.meshgrid(xx, yy)

            for i in range(n_samples):
                # x[i]: (iC, iH, iW), y[i]: (T * oC, oH, oW)
                # prepare input, total time steps
                # (iC, iH, iW) --> (T, iC, iH, iW)
                x_input = x[i].unsqueeze(0).expand(len(interp_times), *x.size()[1:])
                y_output = model(x_input.to(device), interp_times.to(device))
                print(y_output.size())
                # (T, oC, oH, oW) --> (T * oC, oH, oW)
                y_c1, y_c2 = [], []
                for j, _ in enumerate(interp_times):
                    y_c1.append(y_output[j, 0])
                    y_c2.append(y_output[j, 1])
                y_output = th.cat((th.stack(y_c1), th.stack(y_c2))).data.cpu()
                print(y_output.size())
                print(y[i].size())
                # (3 * T * oC, oH, oW)
                samples = th.cat((y[i], y_output, y[i] - y_output)).numpy()
                print(samples.shape)
                
                S_bounds = np.full( (1,2), 0.0)                # upper and lower bounds of saturation
                P_bounds = np.full( (1,2), 0.0)                # upper and lower bounds of pressure
                
                x1=y_output[:len(interp_times),:,:]
                y1=y[i]
                y1=y1[:len(interp_times),:,:]
                samples1 = th.cat((x1, y1)).numpy()
                P_bounds[0,0]=samples1.max()
                P_bounds[0,1]=samples1.min()
                
                x1=y_output[len(interp_times):len(interp_times)*2,:,:]
                y1=y[i]
                y1=y1[len(interp_times):len(interp_times)*2,:,:]
                samples1 = th.cat((x1, y1)).numpy()
                S_bounds[0,0]=samples1.max()
                S_bounds[0,1]=samples1.min()
                

                fig, axes = plt.subplots(3, len(interp_times) * 2, figsize=(3 * len(interp_times) * 2, 7))
                for j, ax in enumerate(fig.axes):
                    ax.set_aspect('equal')
                    ax.set_axis_off()
                    
                    if j<len(interp_times):
                        v = np.linspace(P_bounds[0,1], P_bounds[0,0], 10, endpoint=True)
                        cax = ax.contourf(X, Y, samples[j], 50, cmap='jet')
                    # cax = ax.imshow(samples[j], cmap='jet', origin='lower')
                        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True),ticks=v)
                        cbar.formatter.set_powerlimits((0, 0))
                        cbar.ax.yaxis.set_offset_position('left')
                        cbar.update_ticks()
                    elif j>=len(interp_times) and j<2*len(interp_times):  
                        v = np.linspace(S_bounds[0,1], S_bounds[0,0], 10, endpoint=True)
                        cax = ax.contourf(X, Y, samples[j], 50, cmap='jet')
                    # cax = ax.imshow(samples[j], cmap='jet', origin='lower')
                        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True),ticks=v)
                        cbar.formatter.set_powerlimits((0, 0))
                        cbar.ax.yaxis.set_offset_position('left')
                        cbar.update_ticks()
                    elif j>=2*len(interp_times) and j<3*len(interp_times): 
                        v = np.linspace(P_bounds[0,1], P_bounds[0,0], 10, endpoint=True)
                        cax = ax.contourf(X, Y, samples[j], 50, cmap='jet')
                    # cax = ax.imshow(samples[j], cmap='jet', origin='lower')
                        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True),ticks=v)
                        cbar.formatter.set_powerlimits((0, 0))
                        cbar.ax.yaxis.set_offset_position('left')
                        cbar.update_ticks()
                    elif j>=3*len(interp_times) and j<4*len(interp_times):     
                        v = np.linspace(S_bounds[0,1], S_bounds[0,0], 10, endpoint=True)
                        cax = ax.contourf(X, Y, samples[j], 50, cmap='jet')
                    # cax = ax.imshow(samples[j], cmap='jet', origin='lower')
                        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True),ticks=v)
                        cbar.formatter.set_powerlimits((0, 0))
                        cbar.ax.yaxis.set_offset_position('left')
                        cbar.update_ticks()
                    elif j>=4*len(interp_times):
                        cax = ax.contourf(X, Y, samples[j], 50, cmap='jet')
                    # cax = ax.imshow(samples[j], cmap='jet', origin='lower')
                        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True))
                        cbar.formatter.set_powerlimits((0, 0))
                        cbar.ax.yaxis.set_offset_position('left')
                        cbar.update_ticks()
                     
                plt.savefig(args.output_dir + '/epoch_{}_output_{}.png'.format(epoch, idx[i]),
                            bbox_inches='tight',dpi=350)
                plt.close(fig)
                print("epoch {}, done with printing sample output {}".format(epoch, idx[i]))

    rmse_test = np.sqrt(loss / n_out_pixels_test)
    r2_score = 1 - loss / test_stats['y_var']
    loss_test = loss
    print("epoch: {}, test r2-score:  {:.6f}".format(epoch, r2_score))
    print("epoch: {}, test rmse: {:.6f}".format(epoch, rmse_test))
    print("epoch: {}, test loss: {:.6f}".format(epoch, loss_test))
    return r2_score, rmse_test, loss_test


tic = time()
r2_train, r2_test = [], []
rmse_train, rmse_test = [], []
loss_train, loss_test = [], []
exp_dir = args.exp_dir
# network training
for epoch in range(1, args.n_epochs + 1):
    model.train()
    mse = 0.
    for batch_idx, (input, target, times) in enumerate(train_loader):
        input, target, times = input.to(device), target.to(device), \
                               times.to(device)
        model.zero_grad()
        
        output = model(input, times)
        
        """
        print(output.shape)
        print(target.shape)
        """
        
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        mse += loss.item()

    rmse = np.sqrt(mse / n_out_pixels_train)
    if epoch % args.log_interval == 0:
        r2_score = 1 - mse / train_stats['y_var']
        print("epoch: {}, training r2-score: {:.6f}".format(epoch, r2_score))
        print("epoch: {}, training rmse: {:.6f}".format(epoch, rmse))
        print("epoch: {}, training loss: {:.6f}".format(epoch, mse))
        r2_train.append(r2_score)
        rmse_train.append(rmse)
        loss_train.append(mse)
        r2_t, rmse_t, loss_t = test(epoch, plot_intv=args.plot_interval)
        r2_test.append(r2_t)
        rmse_test.append(rmse_t)
        loss_test.append(loss_t)

    scheduler.step(rmse)

    # save model
    if epoch == args.n_epochs:
        th.save(model.state_dict(), args.exp_dir + "/model_epoch{}.pth".format(epoch))
        
tic2 = time()
print("Done training {} epochs with {} data using {} seconds"
      .format(args.n_epochs, args.n_train, tic2 - tic))

# plot the rmse, r2-score curve
x = np.arange(args.log_interval, args.n_epochs + args.log_interval,
                args.log_interval)
plt.figure()
plt.plot(x, r2_train, label="train: {:.3f}".format(np.mean(r2_train[-5: -1])))
plt.plot(x, r2_test, label="test: {:.3f}".format(np.mean(r2_test[-5: -1])))
plt.xlabel('Epoch')
plt.ylabel(r'$R^2$-score')
plt.legend(loc='lower right')
plt.savefig(exp_dir + "/r2.pdf", format='pdf', dpi=1000)
plt.close()
np.savetxt(exp_dir + "/r2_train.txt", r2_train)
np.savetxt(exp_dir + "/r2_test.txt", r2_test)

plt.figure()
plt.plot(x, rmse_train, label="train: {:.3f}".format(np.mean(rmse_train[-5: -1])))
plt.plot(x, rmse_test, label="test: {:.3f}".format(np.mean(rmse_test[-5: -1])))
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend(loc='upper right')
plt.savefig(exp_dir + "/rmse.pdf", dpi=600)
plt.close()
np.savetxt(exp_dir + "/rmse_train.txt", rmse_train)
np.savetxt(exp_dir + "/rmse_test.txt", rmse_test)

plt.figure()
plt.plot(x, rmse_train, label="train: {:.3f}".format(np.mean(loss_train[-5: -1])))
plt.plot(x, rmse_test, label="test: {:.3f}".format(np.mean(loss_test[-5: -1])))
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.savefig(exp_dir + "/loss.pdf", dpi=600)
plt.close()
np.savetxt(exp_dir + "/loss_train.txt", loss_train)
np.savetxt(exp_dir + "/loss_test.txt", loss_test)

# save args and time taken
args.times = args.times.tolist()
args_dict = {}
for arg in vars(args):
    args_dict[arg] = getattr(args, arg)
args_dict['time'] = tic2 - tic
n_params, n_layers = model._num_parameters_convlayers()
args_dict['num_layers'] = n_layers
args_dict['num_params'] = n_params
with open(exp_dir + "/args.txt", 'w') as file:
    file.write(json.dumps(args_dict))

