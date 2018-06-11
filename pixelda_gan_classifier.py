from __future__ import print_function
import os.path as path
import matplotlib
matplotlib.use('agg')
import os, sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.models as models
from torch.autograd import Variable
import logging
import json

## Import GANs ##
from GANs import *
## Import Classifiers ##
from classifiers import *
## Import utility functions ##
from utils import progress_bar, init_params, weights_init
from params import get_params, check_dirs
from dataset import get_dataset
from plotter import Plotter

## Hyper parameters ##
opt = get_params()
opt.ngpu = torch.cuda.device_count()
opt.outf = os.path.join(opt.outf, "{}_images".format(opt.sourceDataset))
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

## Saving Dirs ##
save_dir = os.path.join("saved", opt.uid)
chkpt_dir = os.path.join(save_dir, opt.chkpt)
imgs_dir = os.path.join(save_dir, opt.outf)
plots_dir = os.path.join(save_dir, opt.plotdir)
# Create these dirs
check_dirs([chkpt_dir, imgs_dir, plots_dir])
with open(os.path.join(save_dir, 'params.json'), 'w') as f:
    json.dump(opt.__dict__, f)
    
## Logger ##
logger = logging.getLogger()
file_log_handler = logging.FileHandler(opt.logfile)
logger.addHandler(file_log_handler)

stderr_log_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stderr_log_handler)

logger.setLevel('INFO')
formatter = logging.Formatter()
file_log_handler.setFormatter(formatter)
stderr_log_handler.setFormatter(formatter)

logger.info(opt)
logger.info("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

## Data Loaders ##
source_train_loader, source_test_loader = get_dataset(dataset=opt.sourceDataset, root_dir=opt.sourceroot,
                          imageSize=opt.imageSize, batchSize=opt.batchSize, workers=opt.workers)

target_train_loader, target_test_loader = get_dataset(dataset=opt.targetDataset, root_dir=opt.targetroot,
                          imageSize=opt.imageSize, batchSize=opt.batchSize, workers=opt.workers)

nz = int(opt.nz) # 10 in PixelDA
imageSize = int(opt.imageSize)
source_channels = 1 if opt.sourceDataset in ['mnist', 'usps'] else 3
target_channels = 1 if opt.targetDataset in ['mnist', 'usps'] else 3
num_classes = 10

##### Generator #####
netG = pixelda_G(out_channels=target_channels, image_size=[imageSize, imageSize, source_channels], opt=opt)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
logger.info(netG)

##### Discrminator #####
netD = pixelda_D(in_channels=target_channels, image_size=[imageSize, imageSize, target_channels], opt=opt)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
logger.info(netD)

##### Classifier #####
if opt.netT != '':
    chk = torch.load(opt.netT)
    netT = chk['netT']
    best_acc = chk['acc']
    netT_epoch = chk['epoch']
else:
    #TODO: Change ResNet18() to use 224 input size. or make
    #torchvision's resnet work with data.data_parallel
    #What's the difference between dataparallel and data_parallel???
    #netT = models.resnet18(pretrained=True)
    #netT = torch.nn.DataParallel(netT)
    # netT = ResNet18()
    netT = MnistClassifier(source_channels, target_channels, num_classes, opt.ngpu)
    init_params(netT)
    best_acc = 0
    netT_epoch = 0
logger.info(netT)

criterion_D = nn.BCEWithLogitsLoss()
criterion_G = nn.BCEWithLogitsLoss()
criterion_T = nn.CrossEntropyLoss()

inputs = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netT.cuda()
    criterion_D.cuda()
    criterion_G.cuda()
    criterion_T.cuda()
    inputs, label = inputs.cuda(), label.cuda()
    noise = noise.cuda()

# Plotters
plot_gan_loss = Plotter("%s/gan_loss.jpeg" % (plots_dir), num_lines=2, legends=["g_loss", "d_loss"],
    xlabel="Number of iterations", ylabel="Loss", title="GAN Loss vs Iterations(%s->%s)" %(opt.sourceDataset, opt.targetDataset))

plot_clf_loss = Plotter("%s/%s_clf_loss.jpeg" % (plots_dir, opt.sourceDataset), num_lines=1, legends=[""],
    xlabel="Number of iterations", ylabel="Loss", title="Classifier loss vs Iterations(%s->%s)" %(opt.sourceDataset, opt.targetDataset))

plot_source_acc = Plotter("%s/%s_clf_acc.jpeg" % (plots_dir, opt.sourceDataset), num_lines=1, legends=[""],
    xlabel="Epochs", ylabel="Accuracy", title="Accuracy vs Epochs(%s->%s)" %(opt.sourceDataset, opt.targetDataset))

plot_target_acc = Plotter("%s/%s_clf_acc.jpeg" % (plots_dir, opt.targetDataset), num_lines=1, legends=[""],
    xlabel="Epochs", ylabel="Accuracy", title="Accuracy vs Epochs(%s->%s)" %(opt.sourceDataset, opt.targetDataset))

plotters = [plot_gan_loss, plot_clf_loss, plot_source_acc, plot_target_acc]

# setup optimizer
optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=opt.lr_gan, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_gan, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
optimizerT = optim.Adam(netT.parameters(), lr=opt.lr_clf, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
# optimizerT = optim.SGD(netT.parameters(), lr=opt.lr_clf, momentum=0.9, weight_decay=5e-4)

lr_scheduler_D = optim.lr_scheduler.StepLR(optimizerD, step_size=opt.lr_decay_step, gamma=opt.lr_decay_rate)
lr_scheduler_G = optim.lr_scheduler.StepLR(optimizerG, step_size=opt.lr_decay_step, gamma=opt.lr_decay_rate)
lr_scheduler_T = optim.lr_scheduler.StepLR(optimizerT, step_size=opt.lr_decay_step, gamma=opt.lr_decay_rate)

lr_schedulers = [lr_scheduler_D, lr_scheduler_G, lr_scheduler_T]

def test(epoch, test_loader, save=True, dataset="target", is_plot=False):
    global best_acc
    epoch += netT_epoch + 1
    netT.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if opt.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = netT(inputs)

        loss = criterion_T(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        # print(targets.size(0))
        correct += predicted.eq(targets.data).cpu().sum()
        # progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    logger.info('======================================================')
    logger.info('Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (epoch, test_loss/len(test_loader), acc, correct, total))
    logger.info('======================================================')
    if is_plot:
        p = plot_target_acc if dataset == "target" else plot_source_acc
        p((epoch, acc))
    if save and (acc > best_acc):
        logger.info('Saving..')
        logger.info("Epoch: %d Accuracy: %.3f%%" % (epoch, acc))
        state = {
            'net': netT, #.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(chkpt_dir, "netT_epoch_{}.pth".format(epoch)))
        best_acc = acc

# print("Testing on MNIST dataset")
# test(-1, source_train_loader, save=False)
# print("Testing on USPS dataset")
# test(-1, target_test_loader)

target_data_iter = iter(target_train_loader)
iterations = 0

for epoch in range(opt.niter):
    netT.train()
    netD.train()
    netG.train()

    for i, source_data in enumerate(source_train_loader, 0):
        map(lambda scheduler: scheduler.step(), lr_schedulers)

        # Idefinitely loop over target data loader #
        try:
            target_data = target_data_iter.next()
        except StopIteration:
            target_data_iter = iter(target_train_loader)
            target_data = target_data_iter.next()

        # BatchNorm needs all batches to be of the same size to get
        # "similar" statistics
        if source_data[0].size(0) != opt.batchSize or target_data[0].size(0) != opt.batchSize:
            continue

        ### Discriminator Step ###
        ################################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #
        ################################################################
        netD.zero_grad()

        ## Target Batch ##
        target_cpu, _ = target_data
        batch_size = target_cpu.size(0)
        if opt.cuda:
            target_cpu = target_cpu.cuda()

        ## Train with target data ##
        # inputs.resize_as_(target_cpu).copy_(target_cpu)
        inputv = Variable(target_cpu)
        output = netD(inputs=inputv)
        label.resize_(output.size()).fill_(real_label)
        labelv = Variable(label)
        errD_target = criterion_D(output, labelv) * opt.domain_loss_wt
        errD_target.backward(retain_graph=True)
        D_x = output.data.mean()

        ## Source Batch ##
        source_cpu, source_label = source_data
        batch_size = source_cpu.size(0)
        if opt.cuda:
            source_cpu, source_label = source_cpu.cuda(), source_label.cuda()

        ## Train with fake ##
        # Sampling from Uniform Distribution as per the paper #
        noise.resize_(batch_size, nz).uniform_(-1, 1)
        noisev = Variable(noise)
        inputv = Variable(source_cpu)
        fake = netG(inputs=inputv, noise_vector=noisev)
        output = netD(inputs=fake.detach())
        label.resize_(output.size()).fill_(fake_label)
        labelv = Variable(label)
        errD_fake = criterion_D(output, labelv) * opt.domain_loss_wt
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_target + errD_fake

        ## Update D's params ##
        optimizerD.step()

        #############################
        # (2) Update T network #
        #############################
        netT.zero_grad()

        ## Train with source ##
        inputv = Variable(source_cpu)
        labelv = Variable(source_label)
        output = netT(inputv)
        errT_source = criterion_T(output, labelv) * opt.task_loss_wt
        errT_source.backward(retain_graph=True)
        # T_x = output.data.mean()

        ## Train with fake ##
        output = netT(fake.detach())
        errT_fake = criterion_T(output, labelv) * opt.task_loss_wt     # Same Label as of source images
        errT_fake.backward()
        errT = errT_source + errT_fake

        ## Update T's params ##
        optimizerT.step()

        ### Generator Step ###
        ################################################
        # (3) Update G network: maximize log(D(G(z))) #
        ################################################
        netG.zero_grad()
        netT.zero_grad()
        netD.zero_grad()

        ## Generator loss due to discriminator ##
        # Sampling from Uniform Distribution as per the paper #
        noise.resize_(batch_size, nz).uniform_(-1, 1)
        noisev = Variable(noise)
        inputv = Variable(source_cpu)
        fake = netG(inputs=inputv, noise_vector=noisev)
        output = netD(inputs=fake)
        label.resize_(output.size()).fill_(real_label)
        labelv = Variable(label)  # fake labels are real for generator cost
        errG_d = criterion_G(output, labelv) * opt.style_transfer_loss_wt
        errG_d.backward(retain_graph=True)
        D_G_z2 = output.data.mean()

        ## Generator loss due to task specific loss ##
        output = netT(fake)
        labelv = Variable(source_label)
        errG_t = criterion_T(output, labelv) * opt.G_task_loss_wt     # Same Label as of source images
        errG_t.backward()
        errG = errG_d + errG_t

        ## Update G's params ##
        optimizerG.step()

        plot_gan_loss((iterations, errG.item()), (iterations, errD.item()))
        plot_clf_loss((iterations, errT.item()))

        logger.info('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_T: %.4f D(x): %.4f D(G(z)): %.4f / %.4f | Best Acc: %3f%%'
              % (epoch, opt.niter, i, len(source_train_loader),
                 errD.item(), errG.item(), errT.item(), D_x, D_G_z1, D_G_z2, best_acc))
        
        iterations += 1
        if (i % 100 == 0) or ((i+1) == len(source_train_loader)):
            vutils.save_image(target_cpu, path.join(imgs_dir, "real_samples_target_epoch_{:03d}.jpeg".format(epoch + netT_epoch + 1)), normalize=True)
            vutils.save_image(source_cpu, path.join(imgs_dir, "real_samples_source_epoch_{:03d}.jpeg".format(epoch + netT_epoch + 1)), normalize=True)

            # Sampling from Uniform Distribution as per the paper
            noise.resize_(batch_size, nz).uniform_(-1, 1)
            noisev = Variable(noise)
            inputv = Variable(source_cpu)
            fake = netG(inputs=inputv, noise_vector=noisev)
            vutils.save_image(fake.data, path.join(imgs_dir, "fake_samples_epoch_{:03d}.jpeg".format(epoch + netT_epoch + 1)), normalize=True)
            
    logger.info("Testing on %s training dataset" % (opt.sourceDataset))
    test(epoch, source_train_loader, save=False, dataset="source", is_plot=True)
    logger.info("Testing on %s test dataset" % (opt.targetDataset))
    test(epoch, target_test_loader, save=False, dataset="target")
    logger.info("Testing on %s train dataset" % (opt.targetDataset))
    test(epoch, target_train_loader, dataset="target", is_plot=True)   # Use train dataset for validation on classifer

    # do checkpointing
    torch.save(netG.state_dict(), os.path.join(chkpt_dir, "netG_epoch_{}.pth".format(epoch + netT_epoch + 1)))
    torch.save(netD.state_dict(), os.path.join(chkpt_dir, "netD_epoch_{}.pth".format(epoch + netT_epoch + 1)))
logger.info("TRAINING DONE!")
logger.info("Testing on %s train dataset" % (opt.sourceDataset))
test(-1, source_train_loader, save=False, dataset="source")
logger.info("Testing on %s test dataset" % (opt.sourceDataset))
test(-1, source_test_loader, save=False, dataset="source")
logger.info("Testing on %s train dataset" % (opt.targetDataset))
test(-1, target_train_loader, save=False, dataset="target")
logger.info("Testing on %s test dataset" % (opt.targetDataset))
test(-1, target_test_loader, save=False, dataset="target")

map(lambda plots: plots.queue.put(None), plotters)
map(lambda plots: plots.queue.join(), plotters)
map(lambda plots: plots.clean_up(), plotters)
