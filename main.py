import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
from config import paths
from criterion import MyLoss
from datasets import OwnDatasets
from tqdm import tqdm
from net import TransPoseNet

parser = argparse.ArgumentParser(description="This is a FDIP of %(prog)s", epilog="This is a epilog of %(prog)s", prefix_chars="-+", fromfile_prefix_chars="@", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-b", "--batch_size",metavar="批次数量", type=int, required=True)
parser.add_argument("-m", "--model", choices=['RNN', 'TCN', 'GCN'], required=True, metavar="模型类型")
parser.add_argument("-f", "--fineturning", action="store_true", help="isFineTurning")
parser.add_argument("-c", "--cuda", action="store_true", help="isCuda")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_first', action="store_false", help="isFirst")

def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    losses = AverageMeter()

    # switch to train mode
    model.train()

    bar = tqdm(enumerate(train_loader), total = len(train_loader))
    for i, (imu, nn_pose,leaf_jtr, full_jtr, stable, velocity_local, root_ori) in bar:

        if args.batch_first:
            imu = imu.transpose(0, 1)
            nn_pose = nn_pose.transpose(0, 1)
            leaf_jtr = leaf_jtr.transpose(0, 1)
            full_jtr = full_jtr.transpose(0, 1)
            stable = stable.transpose(0, 1)
            velocity_local = velocity_local.transpose(0, 1)
            root_ori = root_ori.transpose(0, 1)
        
        if args.cuda:
            imu = imu.cuda()
            nn_pose = nn_pose.cuda()
            leaf_jtr = leaf_jtr.cuda()
            full_jtr = full_jtr.cuda()
            stable = stable.cuda()
            velocity_local = velocity_local.cuda()
            root_ori = root_ori.cuda()
        if args.half:
            imu = imu.half()
            nn_pose = nn_pose.half()
            leaf_jtr = leaf_jtr.half()
            full_jtr = full_jtr.half()
            stable = stable.half()
            velocity_local = velocity_local.half()
            root_ori = root_ori.half()


        # compute output
        output = model(imu)            
        target = (leaf_jtr, full_jtr, nn_pose, stable, velocity_local)
        loss_dict, totalLoss = criterion(output, target)

        bar.set_description(
                f"Train[{epoch}/{args.epochs}] lr={optimizer.param_groups[0]['lr']}")
        bar.set_postfix(**{k:v for k,v in loss_dict.items()})
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        totalLoss.backward()
        optimizer.step()

        losses.update(totalLoss.item(), imu.size(1))

        # measure elapsed time
        # if i % args.print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
        #               epoch, i, len(train_loader), batch_time=batch_time,
        #               data_time=data_time, loss=losses))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    bar = tqdm(enumerate(val_loader), total = len(val_loader))
    for i, (imu, nn_pose,leaf_jtr, full_jtr, stable, velocity_local, root_ori) in bar:
        if args.batch_first:
            imu = imu.transpose(0, 1)
            nn_pose = nn_pose.transpose(0, 1)
            leaf_jtr = leaf_jtr.transpose(0, 1)
            full_jtr = full_jtr.transpose(0, 1)
            stable = stable.transpose(0, 1)
            velocity_local = velocity_local.transpose(0, 1)
            root_ori = root_ori.transpose(0, 1)
        
        if args.cuda:
            imu = imu.cuda()
            nn_pose = nn_pose.cuda()
            leaf_jtr = leaf_jtr.cuda()
            full_jtr = full_jtr.cuda()
            stable = stable.cuda()
            velocity_local = velocity_local.cuda()
            root_ori = root_ori.cuda()

        if args.half:
            imu = imu.half()
            nn_pose = nn_pose.half()
            leaf_jtr = leaf_jtr.half()
            full_jtr = full_jtr.half()
            stable = stable.half()
            velocity_local = velocity_local.half()
            root_ori = root_ori.half()

        # compute output
        with torch.no_grad():
            output = model(imu)
            target = (leaf_jtr, full_jtr, nn_pose, stable, velocity_local)
            loss_dict, totalLoss = criterion(output, target)

        bar.set_description("Val")
        bar.set_postfix(**{k:v for k,v in loss_dict.items()})

        # measure accuracy and record loss
        losses.update(totalLoss.item(), imu.size(1))

        # measure elapsed time

        # if i % args.print_freq == 0:
        #     print('Test: [{0}/{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
        #               i, len(val_loader), batch_time=batch_time, loss=losses))

    return

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

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
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    global args
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    model = TransPoseNet().to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark = True

    train_dataset = OwnDatasets(os.path.join(paths.amass_dir, "veri.pt"))
    val_dataset = OwnDatasets(os.path.join(paths.amass_dir, "veri.pt"))
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # 
    criterion = MyLoss()
    if args.cuda:
        criterion = criterion.cuda()
    else:
        criterion = criterion.cpu()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = True
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}_{}.tar'.format("fineturning" if args.fineturning else "pretrain", epoch)))


if __name__ == '__main__':
    main()