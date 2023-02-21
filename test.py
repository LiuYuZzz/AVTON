import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from datasets import ADataset, ADataLoader
from modules import LPM, IGMM, TOFM
from utils import save_images, load_checkpoint
import argparse
from pytorch_msssim import ssim
import lpips
from torchvision.models.inception import inception_v3


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--tensorboard_dir', type=str,
                        default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--result_dir', type=str,
                        default='results', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='model checkpoint for test')
    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--shuffle", action='store_true',
                        default=False, help='shuffle input data')

    opt = parser.parse_args()
    return opt


def test_lpm(opt, test_loader, model):
    model.cuda()
    model.eval()
    Lpips = lpips.LPIPS(net="alex").cuda()
    Lpips.eval()
    ssim_val = []
    lpips_val = []
    types = []

    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    save_dir = os.path.join(opt.result_dir, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_dir = os.path.join(save_dir, 'limbs')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    for _, inputs in enumerate(test_loader.data_loader):
        type = inputs['type']
        im_names = inputs['im_name']
        im_names = [im_name + ".png" for im_name in im_names]
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        im_other_skin = inputs["im_other_skin"].cuda()
        output = model(agnostic, c)
        output = torch.tanh(output)

        save_images(output, im_names, result_dir)

        output = (output + 1) / 2
        im_other_skin = (im_other_skin + 1) / 2
        ssim_val_tmp = ssim(output, im_other_skin,
                            data_range=1, size_average=False)
        ssim_val.append(ssim_val_tmp)
        types += type
        lpips_val_tmp = Lpips(output, im_other_skin, normalize=True).detach()
        lpips_val.append(lpips_val_tmp)


    ssim_val = torch.cat(ssim_val, 0)
    top_ssim_val = torch.tensor(
        [ssim_val[i] for i, t in enumerate(types) if t == "top"])
    bottom_ssim_val = torch.tensor(
        [ssim_val[i] for i, t in enumerate(types) if t == "bottom"])
    whole_ssim_val = torch.tensor(
        [ssim_val[i] for i, t in enumerate(types) if t == "whole"])

    lpips_val = torch.cat(lpips_val, 0)
    top_lpips_val = torch.tensor(
        [lpips_val[i] for i, t in enumerate(types) if t == "top"])
    bottom_lpips_val = torch.tensor(
        [lpips_val[i] for i, t in enumerate(types) if t == "bottom"])
    whole_lpips_val = torch.tensor(
        [lpips_val[i] for i, t in enumerate(types) if t == "whole"])

    print("Mean: %.3f, Top: %.3f, Bottom: %.3f, Whole: %.3f" % (
        torch.mean(ssim_val), torch.mean(top_ssim_val), torch.mean(bottom_ssim_val), torch.mean(whole_ssim_val)))
    print("Mean: %.3f, Top: %.3f, Bottom: %.3f, Whole: %.3f" % (
        torch.mean(lpips_val), torch.mean(top_lpips_val), torch.mean(bottom_lpips_val), torch.mean(whole_lpips_val)))


def test_igmm(opt, test_loader, model):
    model.cuda()
    model.eval()
    Lpips = lpips.LPIPS(net="alex").cuda()
    Lpips.eval()
    ssim_val = []
    lpips_val = []
    types = []

    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    save_dir = os.path.join(opt.result_dir, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warped_cloth_dir = os.path.join(save_dir, 'warped-clothes')
    if not os.path.exists(warped_cloth_dir):
        os.makedirs(warped_cloth_dir)
    warped_grid_dir = os.path.join(save_dir, 'warped-grid')
    if not os.path.exists(warped_grid_dir):
        os.makedirs(warped_grid_dir)

    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    for _, inputs in enumerate(test_loader.data_loader):
        type = inputs['type']
        c_names = inputs['c_name']
        c_names = [c_name + ".jpg" for c_name in c_names]
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
        im_c = inputs['original_cloth'].cuda()

        grid = model(agnostic, c)
        warped_cloth = F.grid_sample(
            c, grid, padding_mode='border', align_corners=False)
        warped_grid = F.grid_sample(
            im_g, grid, padding_mode='zeros', align_corners=False)
        
        save_images(warped_cloth, c_names, warped_cloth_dir)
        save_images(warped_grid, c_names, warped_grid_dir)

        warped_cloth = (warped_cloth + 1) / 2
        im_c = (im_c + 1) / 2
        ssim_val_tmp = ssim(warped_cloth, im_c,
                            data_range=1, size_average=False)
        ssim_val.append(ssim_val_tmp)
        types += type
        lpips_val_tmp = Lpips(warped_cloth, im_c, normalize=True).detach()
        lpips_val.append(lpips_val_tmp)

    
    ssim_val = torch.cat(ssim_val, 0)
    top_ssim_val = torch.tensor(
        [ssim_val[i] for i, t in enumerate(types) if t == "top"])
    bottom_ssim_val = torch.tensor(
        [ssim_val[i] for i, t in enumerate(types) if t == "bottom"])
    whole_ssim_val = torch.tensor(
        [ssim_val[i] for i, t in enumerate(types) if t == "whole"])

    lpips_val = torch.cat(lpips_val, 0)
    top_lpips_val = torch.tensor(
        [lpips_val[i] for i, t in enumerate(types) if t == "top"])
    bottom_lpips_val = torch.tensor(
        [lpips_val[i] for i, t in enumerate(types) if t == "bottom"])
    whole_lpips_val = torch.tensor(
        [lpips_val[i] for i, t in enumerate(types) if t == "whole"])

    print("Mean: %.3f, Top: %.3f, Bottom: %.3f, Whole: %.3f" % (
        torch.mean(ssim_val), torch.mean(top_ssim_val), torch.mean(bottom_ssim_val), torch.mean(whole_ssim_val)))
    print("Mean: %.3f, Top: %.3f, Bottom: %.3f, Whole: %.3f" % (
        torch.mean(lpips_val), torch.mean(top_lpips_val), torch.mean(bottom_lpips_val), torch.mean(whole_lpips_val)))


def test_tofm(opt, test_loader, model):
    model.cuda()
    model.eval()
    Lpips = lpips.LPIPS(net="alex").cuda()
    Lpips.eval()
    ssim_val = []
    lpips_val = []
    types = []

    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    save_dir = os.path.join(opt.result_dir, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)

    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    for _, inputs in enumerate(test_loader.data_loader):
        type = inputs['type']
        im_names = inputs['im_name']
        im_names = [im_name + ".png" for im_name in im_names]
        im = inputs['image'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()

        outputs = model(agnostic, c)
        p_rendered, m_composite = torch.split(outputs, [3, 1], dim=1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        save_images(p_tryon, im_names, try_on_dir)

        p_tryon = (p_tryon + 1) / 2
        im = (im + 1) / 2
        ssim_val_tmp = ssim(p_tryon, im,
                            data_range=1, size_average=False)
        ssim_val.append(ssim_val_tmp)
        types += type
        lpips_val_tmp = Lpips(p_tryon, im, normalize=True).detach()
        lpips_val.append(lpips_val_tmp)

    
    ssim_val = torch.cat(ssim_val, 0)
    top_ssim_val = torch.tensor(
        [ssim_val[i] for i, t in enumerate(types) if t == "top"])
    bottom_ssim_val = torch.tensor(
        [ssim_val[i] for i, t in enumerate(types) if t == "bottom"])
    whole_ssim_val = torch.tensor(
        [ssim_val[i] for i, t in enumerate(types) if t == "whole"])

    lpips_val = torch.cat(lpips_val, 0)
    top_lpips_val = torch.tensor(
        [lpips_val[i] for i, t in enumerate(types) if t == "top"])
    bottom_lpips_val = torch.tensor(
        [lpips_val[i] for i, t in enumerate(types) if t == "bottom"])
    whole_lpips_val = torch.tensor(
        [lpips_val[i] for i, t in enumerate(types) if t == "whole"])

    print("Mean: %.3f, Top: %.3f, Bottom: %.3f, Whole: %.3f" % (
        torch.mean(ssim_val), torch.mean(top_ssim_val), torch.mean(bottom_ssim_val), torch.mean(whole_ssim_val)))
    print("Mean: %.3f, Top: %.3f, Bottom: %.3f, Whole: %.3f" % (
        torch.mean(lpips_val), torch.mean(top_lpips_val), torch.mean(bottom_lpips_val), torch.mean(whole_lpips_val)))


def test(opt):
    # create dataset
    test_dataset = ADataset(opt)

    # create dataloader
    test_loader = ADataLoader(opt, test_dataset)

    # visualization

    if opt.stage == 'LPM':
        model = LPM()
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_lpm(opt, test_loader, model)
    elif opt.stage == 'IGMM':
        model = IGMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_igmm(opt, test_loader, model)
    elif opt.stage == 'TOFM':
        model = TOFM()
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tofm(opt, test_loader, model)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished test %s' % (opt.stage))


def main():
    opt = get_opt()
    print("Start to test stage: %s" % (opt.stage))

    test(opt)


if __name__ == "__main__":
    main()
