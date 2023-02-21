import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

import os.path as osp
import numpy as np


class ADataset(data.Dataset):
    def __init__(self, opt):
        super(ADataset, self).__init__()
        # base setting
        self.stage = opt.stage  # LPM, IGMM or TOFM
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.result_path = osp.join(opt.result_dir, opt.datamode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.resize = transforms.Compose([
            transforms.Resize(min(opt.fine_height, opt.fine_width)),
            transforms.CenterCrop([opt.fine_height, opt.fine_width])
        ])

        # load data list
        self.types = []
        self.im_names = []
        self.c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                type, im_name, c_name = line.strip().split()
                self.types.append(type)
                self.im_names.append(im_name)
                self.c_names.append(c_name)

    def name(self):
        return "ADataset"

    def __getitem__(self, index):
        type = self.types[index]
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # target person image
        im = Image.open(osp.join(self.data_path, 'people', im_name + ".jpg"))
        im = self.resize(im)
        im = self.transform(im)  # [-1,1]

        # human parsing image
        im_parse = Image.open(
            osp.join(self.data_path, 'parse', im_name + ".png"))
        im_parse = self.resize(im_parse)
        parse_array = np.array(im_parse)
        parse_head = (parse_array == 1).astype(np.float32) + \
                     (parse_array == 2).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32)
        parse_top = (parse_array == 5).astype(np.float32) + \
            (parse_array == 6).astype(np.float32) + \
            (parse_array == 7).astype(np.float32) + \
            (parse_array == 11).astype(np.float32)
        parse_bottom = (parse_array == 9).astype(np.float32) + \
            (parse_array == 10).astype(np.float32) + \
            (parse_array == 12).astype(np.float32)
        parse_whole = parse_top + parse_bottom
        parse_hand = (parse_array == 3).astype(np.float32) + \
                     (parse_array == 14).astype(np.float32) + \
                     (parse_array == 15).astype(np.float32)
        parse_foot = (parse_array == 16).astype(np.float32) + \
                     (parse_array == 17).astype(np.float32)
        parse_shoes = (parse_array == 8).astype(np.float32) + \
                      (parse_array == 18).astype(np.float32) + \
                      (parse_array == 19).astype(np.float32)
        phead = torch.from_numpy(parse_head)
        ptop = torch.from_numpy(parse_top)
        pbottom = torch.from_numpy(parse_bottom)
        pwhole = torch.from_numpy(parse_whole)
        phand = torch.from_numpy(parse_hand)
        pfoot = torch.from_numpy(parse_foot)
        pshoes = torch.from_numpy(parse_shoes)

        # target clothing image / warped clothing image
        if self.stage == 'LPM':
            c = Image.open(
                osp.join(self.data_path, 'clothes', c_name + ".jpg"))
            c = self.resize(c)
            c = self.transform(c)  # [-1,1]
        elif self.stage == 'IGMM':
            c = Image.open(
                osp.join(self.data_path, 'clothes', c_name + ".jpg"))
            c = self.resize(c)
            c = self.transform(c)  # [-1,1]
        else:
            c = Image.open(
                osp.join(self.result_path, 'warped-clothes', c_name + ".jpg"))
            c = self.resize(c)
            c = self.transform(c)  # [-1,1]

        # densepose
        img_densepose = Image.open(
            osp.join(self.data_path, 'densepose', im_name + ".png"))
        img_densepose = self.resize(img_densepose)
        densepose_arr = np.array(img_densepose)
        i = np.eye(25)[densepose_arr[:, :, 2]].astype(np.float32)
        uv = (densepose_arr[:, :, 0:2].astype(np.float32) - 0.0) / 255.0
        densepose = np.concatenate((uv, i), axis=-1)
        densepose = torch.from_numpy(densepose).permute(2, 0, 1)
        img_densepose = self.transform(img_densepose)  # for visualization

        # hand detail
        hand_detail = (((densepose_arr[:, :, 2] == 3) | (
            densepose_arr[:, :, 2] == 4)) & (parse_hand == 1)).astype(np.float32)
        phand_detail = torch.from_numpy(hand_detail)
        if self.stage == "LPM":
            phand_detail_ = 0
        else:
            phand_detail_ = phand_detail

        # original clothes
        if type == "top":
            parse_skin = parse_hand
            pskin = torch.from_numpy(parse_skin)
            im_hand_detail = im * phand_detail - (1 - phand_detail)
            im_c = im * ptop + (1 - ptop)
            c_mask = ptop.unsqueeze(0)
            im_other = im * (pbottom + pshoes + phead + pfoot + phand_detail_) - \
                (1 - pbottom - pshoes - phead - pfoot - phand_detail_)
            im_other_skin = im * (pbottom + pshoes + phead + pfoot + pskin) - \
                (1 - pbottom - pshoes - phead - pfoot - pskin)
        elif type == "bottom":
            parse_skin = parse_foot
            pskin = torch.from_numpy(parse_skin)
            im_hand_detail = im * 0 - 1
            im_c = im * pbottom + (1 - pbottom)
            c_mask = pbottom.unsqueeze(0)
            im_other = im * (ptop + pshoes + phead + phand) - \
                (1 - ptop - pshoes - phead - phand)
            im_other_skin = im * (ptop + pshoes + phead + phand + pskin) - \
                (1 - ptop - pshoes - phead - phand - pskin)
        else:
            parse_skin = parse_hand + parse_foot
            pskin = torch.from_numpy(parse_skin)
            im_c = im * pwhole + (1 - pwhole)
            im_hand_detail = im * phand_detail - (1 - phand_detail)
            c_mask = pwhole.unsqueeze(0)
            im_other = im * (pshoes + phead + phand_detail_) - \
                (1 - pshoes - phead - phand_detail_)
            im_other_skin = im * (pshoes + phead + pskin) - \
                (1 - pshoes - phead - pskin)

        # cloth-agnostic representation
        if self.stage == "LPM":
            agnostic = torch.cat([densepose, im_other], 0)
        elif self.stage == 'IGMM':
            im_other = Image.open(
                osp.join(self.result_path, 'limbs', im_name + ".png"))
            im_other = self.resize(im_other)
            im_other = self.transform(im_other)
            agnostic = torch.cat([densepose, im_other], 0)
        else:
            im_other = Image.open(
                osp.join(self.result_path, 'limbs', im_name + ".png"))
            im_other = self.resize(im_other)
            im_other = self.transform(im_other)
            agnostic = torch.cat([densepose, im_other, im_hand_detail], 0)

        im_g = Image.open('./assets/grid.png')
        im_g = self.transform(im_g)

        result = {
            "type": type,
            'c_name': c_name,
            'im_name': im_name,
            'cloth': c,  # for input
            'agnostic': agnostic,  # for input
            'hand_detail': im_hand_detail,  # for input
            "im_other_skin": im_other_skin,  # for ground truth
            'original_cloth': im_c,  # for ground truth
            'image': im,  # for ground truth
            'c_mask': c_mask,  # for ground truth
            'densepose': img_densepose,  # for visualization
            'im_other': im_other,  # for visualization
            'grid_image': im_g,  # for visualization
        }

        return result

    def __len__(self):
        return len(self.im_names)


class ADataLoader(object):
    def __init__(self, opt, dataset):
        super(ADataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=(
            train_sampler is None), num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
