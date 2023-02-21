# -*- coding: utf-8 -*-
# @Author : Yu Liu
# @Email : 15001737229@163.com
# @File : train_full.py
# @Project : cp-vton-experiment


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from datasets import ADataset, ADataLoader
from modules import LPM, IGMM, TOFM
from torch.utils.tensorboard import SummaryWriter
from utils import board_add_images, load_checkpoint, save_checkpoint, VGGLoss
import argparse


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('-b', '--batch-size', type=int, default=8)

    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument('--result_dir', type=str,
                        default='results', help='save result infos')
    parser.add_argument("--stage", default="LPM")
    parser.add_argument("--data_list", default="train.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str,
                        default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='models', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=50000)
    parser.add_argument("--keep_step", type=int, default=400000)
    parser.add_argument("--decay_step", type=int, default=400000)
    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')

    opt = parser.parse_args()
    return opt


def train_lpm(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 1.0 - max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        img = inputs["image"].cuda()
        im_pose = inputs['densepose'].cuda()
        agnostic = inputs['agnostic'].cuda()
        im_other = inputs['im_other'].cuda()
        c = inputs['cloth'].cuda()
        im_other_skin = inputs["im_other_skin"].cuda()

        output = model(agnostic, c)
        output = torch.tanh(output)

        loss_l1 = criterionL1(output, im_other_skin)
        loss_vgg = criterionVGG(output, im_other_skin)
        loss = loss_l1 + loss_vgg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        visuals = [[img, im_other, im_pose],
                   [c, im_other_skin, output]]

        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'limbs', visuals, step + 1)
            board.add_scalar('metric', loss.item(), step + 1)
            board.add_scalar('l1', loss_l1.item(), step + 1)
            board.add_scalar('vgg', loss_vgg.item(), step + 1)

            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, l1: %.4f, vgg: %.4f'
                  % (step + 1, t, loss_l1.item(), loss_vgg.item()), flush=True)

        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(
                opt.checkpoint_dir, opt.stage, 'step_%06d.pth' % (step + 1)))


def train_igmm(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lr_lambda=lambda step: 1.0 - max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im_pose = inputs['densepose'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        im_c = inputs['original_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
        im_other = inputs['im_other'].cuda()

        grid = model(agnostic, c)

        warped_cloth = F.grid_sample(
            c, grid, padding_mode='border', align_corners=False)
        warped_grid = F.grid_sample(
            im_g, grid, padding_mode='zeros', align_corners=False)

        loss_l1 = criterionL1(warped_cloth, im_c)
        loss = loss_l1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        visuals = [[im_other, im_pose, c],
                   [im_c, warped_grid, warped_cloth]]

        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'warped clothes', visuals, step + 1)
            board.add_scalar('metric', loss.item(), step + 1)
            board.add_scalar('l1', loss_l1.item(), step + 1)
            t = time.time() - iter_start_time
            print(
                'step: %8d, time: %.3f, l1: %.4f'
                % (step + 1, t, loss_l1.item()), flush=True)

        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(
                opt.checkpoint_dir, opt.stage, 'step_%06d.pth' % (step + 1)))


def train_tofm(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['densepose'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        c_mask = inputs['c_mask'].cuda()
        im_other = inputs['im_other'].cuda()
        im_hend_detail = inputs["hand_detail"].cuda()

        outputs = model(agnostic, c)
        p_rendered, m_composite = torch.split(outputs, [3, 1], dim=1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, c_mask)
        loss_l1 = criterionL1(p_tryon, im)
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        visuals = [[im_other, im_pose, c],
                   [im_hend_detail, c_mask * 2 - 1, m_composite * 2 - 1],
                   [im, p_rendered, p_tryon]]

        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step + 1)
            board.add_scalar('metric', loss.item(), step + 1)
            board.add_scalar('l1', loss_l1.item(), step + 1)
            board.add_scalar('vgg', loss_vgg.item(), step + 1)
            board.add_scalar('mask', loss_mask.item(), step + 1)
            t = time.time() - iter_start_time

            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f'
                  % (step + 1, t, loss.item(), loss_l1.item(),
                     loss_vgg.item(), loss_mask.item()), flush=True)

        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(
                opt.checkpoint_dir, opt.stage, 'step_%06d.pth' % (step + 1)))


def train(opt):
    # create dataset
    train_dataset = ADataset(opt)

    # create dataloader
    train_loader = ADataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.stage))

    if opt.stage == 'LPM':
        model = LPM()
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
            print("load model from {}".format(opt.checkpoint))
        train_lpm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(
            opt.checkpoint_dir, opt.stage, 'lpm_final.pth'))
    elif opt.stage == 'IGMM':
        model = IGMM(opt)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
            print("load model from {}".format(opt.checkpoint))
        train_igmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(
            opt.checkpoint_dir, opt.stage, 'igmm_final.pth'))
    elif opt.stage == 'TOFM':
        model = TOFM()
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
            print("load model from {}".format(opt.checkpoint))
        train_tofm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(
            opt.checkpoint_dir, opt.stage, 'tofm_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished training %s' % (opt.stage))


def main():
    opt = get_opt()
    print("Start to train stage: %s" % (opt.stage))

    train(opt)


if __name__ == "__main__":
    main()
