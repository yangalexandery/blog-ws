from __future__ import print_function

import argparse
import os
from multiprocessing import Pool

import utils
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from networks import Generator, Discriminator, Downsampler
from data import ImageLoader

import time
import random

def reconstruct_image(im):
    im = im.numpy()
    im = np.transpose(im, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    im = 256 * (im * std + mean)
    return im


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.3 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def random_latent_vectors(batch_size, latent_dim):
    return np.random.randn(batch_size, latent_dim, 1, 1)

def logify(val):
    if -1 <= val <= 1:
        return val
    return np.log(val) if val > 0 else -np.log(-val)

CHANNELS = [256, 128, 64, 32, 16, 16]
LATENT_DIM = 100

NAME_TO_MODEL = {
    'gen': Generator(len(CHANNELS), LATENT_DIM, CHANNELS),
    'disc': Discriminator(len(CHANNELS), CHANNELS)
}


if __name__ == '__main__':
    default_path = '../data'
    loss_fn = CrossEntropyLoss()

    # set up argument parser
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--exp', default = 'default')
    parser.add_argument('--resume', default = None, type = str)
    parser.add_argument('--clean', action = 'store_true')

    # dataset
    parser.add_argument('--data_path', default = default_path)
    parser.add_argument('--file_path', default = 'data.npy')

    # training
    parser.add_argument('--epochs', default = 500, type = int)
    parser.add_argument('--batch', default = 16, type = int)
    parser.add_argument('--snapshot', default = 10, type = int)
    parser.add_argument('--workers', default = 8, type = int)
    parser.add_argument('--gpu', default = '0')
    parser.add_argument('--name', default = 'fc512')

    # Training Parameters
    parser.add_argument('--lr', default = 0.1, type = float)
    parser.add_argument('--momentum', default = 0.9, type = float)
    parser.add_argument('--weight_decay', default = 1e-5, type = float)

    # parse arguments
    args = parser.parse_args()
    print('==> arguments parsed')
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))

    # set up gpus for training
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # set up datasets and loaders
    data, loaders = {}, {}
    for split in ['train']:
        data[split] = ImageLoader(data_path = args.data_path, split = split, file_path = args.file_path)
        loaders[split] = DataLoader(data[split], batch_size = args.batch, shuffle = True, num_workers = args.workers)
    print('==> dataset loaded')
    print('[size] = {0}'.format(len(data['train'])))

    # set up model and convert into cuda
    gs = [Generator(i+1, LATENT_DIM, CHANNELS[:i+1]).cuda() for i in range(6)]
    ds = [Discriminator(i+1, CHANNELS[:i+1]).cuda() for i in range(6)]

    # g = NAME_TO_MODEL['gen'].cuda()
    # d = NAME_TO_MODEL['disc'].cuda()
    print('==> model loaded')

    # set up optimizer for training

    # set up experiment path
    exp_path = os.path.join('exp', args.exp)
    utils.shell.mkdir(exp_path, clean = args.clean)
    logger = utils.Logger(exp_path)
    print('==> save logs to {0}'.format(exp_path))


    best_top_5 = 0

    step = 0 #epoch * len(data['train'])
    for layer in range(len(CHANNELS)):
        optimizer_g = torch.optim.Adam(gs[layer].params, lr=1e-4, weight_decay=1e-5)
        optimizer_d = torch.optim.Adam(ds[layer].params, lr=1e-4, weight_decay=1e-5)
        # optimizer_g = torch.optim.SGD(gs[layer].params, lr=1e-4, weight_decay=1e-5)
        # optimizer_d = torch.optim.SGD(ds[layer].params, lr=1e-4, weight_decay=1e-5)

        step_d_layer = 0
        step_g_layer = 0

        if args.resume is not None:
            if os.path.isfile(args.resume):
                layer = len(CHANNELS) - 1
                snapshot = torch.load(args.resume)
                epoch = snapshot['epoch']
                ds[layer].load_state_dict(snapshot['model_d'])
                gs[layer].load_state_dict(snapshot['model_g'])
                optimizer_d = torch.optim.Adam(ds[layer].params, lr=1e-4, weight_decay=1e-5)
                optimizer_g = torch.optim.Adam(gs[layer].params, lr=1e-4, weight_decay=1e-5)
                # optimizer_g = torch.optim.SGD(gs[layer].params, lr=1e-4, weight_decay=1e-5)
                # optimizer_d = torch.optim.SGD(ds[layer].params, lr=1e-4, weight_decay=1e-5)
                optimizer_d.load_state_dict(snapshot['optimizer_d'])
                optimizer_g.load_state_dict(snapshot['optimizer_g'])
                step = snapshot['step'] if 'step' in snapshot.keys() else 0
                step_d_layer = 2 * len(data['train'])
                step_g_layer = 2 * len(data['train'])
                ds[layer].alpha = 1
                gs[layer].alpha = 1
                print('==> snapshot "{0}" loaded (epoch {1})'.format(args.resume, epoch))
            else:
                raise FileNotFoundError('no snapshot found at "{0}"'.format(args.resume))
        else:
            if layer != 0:
                ds[layer].load_prev_model(ds[layer - 1])
                gs[layer].load_prev_model(gs[layer - 1])
            epoch = 0
        print('==> optimizer loaded')

        args.epochs = 30


        tot_d_loss = 0.0
        tot_g_loss = 0.0
        # layer_epochs = [5, 1000]
        layer_epochs = [60, 120, 180, 180, 300, 600]

        for epoch in range(epoch, layer_epochs[layer]):

            # training the model
            gs[layer].eval()
            ds[layer].train()

            downs = Downsampler(128, 4 * (2 ** layer)).cuda()

            tqdm_d = tqdm(loaders['train'], desc = 'epoch %d (D) layer %d' % (epoch + 1, layer + 1))

            if True or tot_g_loss <= 0.0:
                tot_d_loss = 0.0
                for images in tqdm_d:
                    # convert images and labels into cuda tensor
                    latents = random_latent_vectors(args.batch, LATENT_DIM)
                    latents = Variable(torch.FloatTensor(latents).cuda())

                    fake_images = gs[layer].forward(latents)
                    Dz = ds[layer].forward(fake_images)

                    images = downs.forward(Variable(images.cuda()).float())
                    Dx = ds[layer].forward(images)
                    # print(fake_images.size(), images.size())

                    ### Wasserstein Distance *** look this up ***
                    # https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
                    wd = Dz - Dx

                    # # uncomment this for gradient penalty
                    mix = np.tile(np.random.rand(args.batch, 1, 1, 1), (1, images.size(1), images.size(2), images.size(3)))
                    mix = torch.FloatTensor(mix)
                    x_hat = torch.mul(fake_images.data.cpu(), mix) + torch.mul(images.data.cpu(), 1 - mix)
                    x_hat = Variable(x_hat.cuda(), requires_grad=True)
                    Dx_hat = ds[layer].forward(x_hat)
                    # print(Dx_hat)
                    grads = torch.autograd.grad(Dx_hat, x_hat, grad_outputs=torch.ones(Dx_hat.size()).cuda(),
                                                create_graph=True, retain_graph=True)[0]
                    gp = torch.pow(grads, 2)
                    for i in range(3):
                        gp = torch.sum(gp, dim=1)
                    gp = torch.sqrt(gp)
                    ### hyperparameters
                    w_gamma = 1.0
                    epsilon = 0.001

                    gp = torch.pow((gp - w_gamma) / w_gamma, 2)
                    gp_scaled = gp * w_gamma

                    epsilon_cost = epsilon * torch.pow(Dx, 2)
                    # loss = gp_scaled
                    # print(grads[0].cpu().data.numpy())
                    # print(grads.size())
                    loss = torch.mean(wd + gp_scaled + epsilon_cost)

                    # images = Variable(images.cuda()).float()
                    # labels = Variable(labels.cuda())
                    # initialize optimizer
                    # optimizer.zero_grad()

                    # forward pass
                    # outputs = model.forward(images)
                    # loss = loss_fn(outputs, labels.squeeze())
                    # add summary to logger
                    logger.scalar_summary('loss (D)', (loss.data[0]), step)
                    wd_log = torch.mean(wd)
                    logger.scalar_summary('Wasserstein Distance', (wd_log.data[0]), step)
                    gp_log = torch.mean(gp_scaled)
                    logger.scalar_summary('Gradient Penalty (scaled)', (gp_log.data[0]), step)
                    epsilon_log = torch.mean(epsilon_cost)
                    logger.scalar_summary('Epsilon Cost', (epsilon_log.data[0]), step)

                    step += args.batch
                    # tot_d_loss += wd_log.data[0]
                    # backward pass
                    loss.backward()


                    # Clip gradient norms
                    # clip_grad_norm(model.parameters(), 10.0)

                    optimizer_d.step()
                    optimizer_d.zero_grad()

                    step_d_layer += args.batch
                    ds[layer].alpha = min(1, step_d_layer / (10 * len(data['train'])))

            del tqdm_d

            gs[layer].train()
            ds[layer].eval()

            tqdm_g = tqdm(loaders['train'], desc = 'epoch %d (G) layer %d' % (epoch + 1, layer + 1))

            if True or tot_d_loss <= 0.0:
                tot_g_loss = 0.0
                for images in tqdm_g:
                    latents = random_latent_vectors(args.batch, LATENT_DIM)
                    latents = Variable(torch.FloatTensor(latents).cuda())

                    fake_images = gs[layer].forward(latents)
                    Dz = ds[layer].forward(fake_images)

                    loss = torch.mean(-Dz)

                    logger.scalar_summary('loss (G)', (loss.data[0]), step)

                    step += args.batch
                    tot_g_loss += loss.data[0]

                    loss.backward()

                    optimizer_g.step()
                    optimizer_g.zero_grad()

                    step_g_layer += args.batch
                    gs[layer].alpha = min(1, step_g_layer / (10 * len(data['train'])))

            del tqdm_g
    ##### LOGGING STUFF #####
            gs[layer].eval()
            if layer == len(CHANNELS) - 1 and args.snapshot != 0 and (epoch + 1) % args.snapshot == 0:
                snapshot = {
                    'epoch': epoch + 1,
                    'model_g': gs[-1].state_dict(),
                    'model_d': ds[-1].state_dict(),
                    'optimizer_g': optimizer_g.state_dict(),
                    'optimizer_d': optimizer_d.state_dict(),
                    'step': step
                }
                torch.save(snapshot, os.path.join(exp_path, 'epoch_%d.pth' % (epoch + 1)))
                print('==> saved snapshot to "{0}"'.format(os.path.join(exp_path, 'epoch_%d.pth' % (epoch + 1))))

            if args.snapshot != 0 and (epoch + 1) % args.snapshot == 0:
                num_imgs = 8
                latents = random_latent_vectors(num_imgs, LATENT_DIM)
                latents = Variable(torch.FloatTensor(latents).cuda())

                fake_images = gs[layer].forward(latents)

                img_data = fake_images.cpu().data.numpy()
                img_data = np.transpose(img_data, (0, 2, 3, 1))
                logger.image_summary('preview images', img_data, step)

                img_index = random.randint(0, 9000)
                images = data['train'].images[img_index:img_index + 8]
                images = torch.FloatTensor(np.transpose(images, (0, 3, 1, 2)))
                # print(images.size())
                images = downs.forward(Variable(images.cuda()).float())
                img_data = images.cpu().data.numpy()
                logger.image_summary('actual_images', img_data, step)
            #     model.eval()
            #     top1 = AverageMeter()
            #     top5 = AverageMeter()

            #     for images, labels in tqdm(loaders['val'], desc = 'epoch %d' % (epoch + 1)):
            #         outputs = model.forward(Variable(images.cuda())).cpu()

            #         prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            #         top1.update(prec1[0], images.size(0))
            #         top5.update(prec5[0], images.size(0))

            #     if top5.avg > best_top_5:
            #         best_top_5 = top5.avg

            #         # snapshot model and optimizer
            #         print('==> saved snapshot to "{0}"'.format(os.path.join(exp_path, 'best.pth')))

                    # torch.save(snapshot, os.path.join(exp_path, 'best.pth'))
            #     print('Val:      * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} in validation'.format(top1=top1, top5=top5))
            #     logger.scalar_summary('Top 1', top1.avg, epoch)
            #     logger.scalar_summary('Top 5', top5.avg, epoch)

            #     if args.file_path == 'data.npz':
            #         top1_test = AverageMeter()
            #         top5_test = AverageMeter()

            #         for images, labels in tqdm(loaders['test'], desc = 'epoch %d' % (epoch + 1)):
            #             outputs = model.forward(Variable(images.cuda())).cpu()

            #             prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            #             top1_test.update(prec1[0], images.size(0))
            #             top5_test.update(prec5[0], images.size(0))

            #         # if top5.avg > best_top_5:
            #         #     best_top_5 = top5.avg

            #         #     # snapshot model and optimizer
            #         #     snapshot = {
            #         #         'epoch': epoch + 1,
            #         #         'model': model.state_dict(),
            #         #         'optimizer': optimizer.state_dict()
            #         #     }
            #         #     torch.save(snapshot, os.path.join(exp_path, 'best.pth'))
            #         #     print('==> saved snapshot to "{0}"'.format(os.path.join(exp_path, 'best.pth')))

            #         print('Test: * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} in validation'.format(top1=top1_test, top5=top5_test))
            #         logger.scalar_summary('Top 1 Test', top1_test.avg, epoch)
            #         logger.scalar_summary('Top 5 Test', top5_test.avg, epoch)
