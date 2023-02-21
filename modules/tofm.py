import torch
from torch import nn


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type="zero", norm_layer=nn.InstanceNorm2d, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class TOFM(nn.Module):
    def __init__(self):
        super(TOFM, self).__init__()

        # encode the reference person and the output of LPM and IGMM
        unet_1_0 = [
            nn.Conv2d(33, 64, kernel_size=4, stride=2,
                      padding=1, bias=True),
            nn.InstanceNorm2d(64)
        ]
        unet_2_0 = [
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4,
                      stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128)
        ]
        unet_3_0 = [
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4,
                      stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256)
        ]
        unet_4_0 = [
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4,
                      stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(512)
        ]

        # encode the warped clothes
        encoder_1_0 = [
            nn.Conv2d(3, 64, kernel_size=4, stride=2,
                      padding=1, bias=True),
            nn.InstanceNorm2d(64)
        ]
        encoder_2_0 = [
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4,
                      stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128)
        ]
        encoder_3_0 = [
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4,
                      stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256)
        ]
        encoder_4_0 = [
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4,
                      stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(512)
        ]

        # resnet blocks
        resnet = []
        for i in range(9):
            resnet += [
                nn.ReLU(),
                ResnetBlock(512)
            ]

        # decode and synthesis the rendered person
        unet_4_1 = [
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1024, 256, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(256)
        ]
        unet_3_1 = [
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 128, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(128)
        ]
        unet_2_1 = [
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(64)
        ]
        unet_1_1 = [
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 3, kernel_size=3,
                      stride=1, padding=1, bias=True)
        ]

        # decode and synthesis the composition mask
        decoder_4_1 = [
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1024, 256, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(64)
        ]
        decoder_3_1 = [
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 128, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(64)
        ]
        decoder_2_1 = [
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(64)
        ]
        decoder_1_1 = [
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 1, kernel_size=3,
                      stride=1, padding=1, bias=True),
        ]

        self.unet_1_0 = nn.Sequential(*unet_1_0)
        self.unet_2_0 = nn.Sequential(*unet_2_0)
        self.unet_3_0 = nn.Sequential(*unet_3_0)
        self.unet_4_0 = nn.Sequential(*unet_4_0)
        self.unet_1_1 = nn.Sequential(*unet_1_1)
        self.unet_2_1 = nn.Sequential(*unet_2_1)
        self.unet_3_1 = nn.Sequential(*unet_3_1)
        self.unet_4_1 = nn.Sequential(*unet_4_1)
        self.resnet = nn.Sequential(*resnet)
        self.encoder_4_0 = nn.Sequential(*encoder_4_0)
        self.encoder_3_0 = nn.Sequential(*encoder_3_0)
        self.encoder_2_0 = nn.Sequential(*encoder_2_0)
        self.encoder_1_0 = nn.Sequential(*encoder_1_0)
        self.decoder_4_1 = nn.Sequential(*decoder_4_1)
        self.decoder_3_1 = nn.Sequential(*decoder_3_1)
        self.decoder_2_1 = nn.Sequential(*decoder_2_1)
        self.decoder_1_1 = nn.Sequential(*decoder_1_1)

    def forward(self, input_1, input_2):
        x_1_0 = self.unet_1_0(input_1)
        x_2_0 = self.unet_2_0(x_1_0)
        x_3_0 = self.unet_3_0(x_2_0)
        x_4_0 = self.unet_4_0(x_3_0)

        e_1_0 = self.encoder_1_0(input_2)
        e_2_0 = self.encoder_2_0(e_1_0)
        e_3_0 = self.encoder_3_0(e_2_0)
        e_4_0 = self.encoder_4_0(e_3_0)

        f = self.resnet(x_4_0 + e_4_0)

        x_3_1 = self.unet_4_1(torch.cat([x_4_0, f], 1))
        x_2_1 = self.unet_3_1(torch.cat([x_3_0, x_3_1], 1))
        x_1_1 = self.unet_2_1(torch.cat([x_2_0, x_2_1], 1))
        output_1 = self.unet_1_1(torch.cat([x_1_0, x_1_1], 1))
        d_3_1 = self.decoder_4_1(torch.cat([e_4_0, f], 1))
        d_2_1 = self.decoder_3_1(torch.cat([e_3_0, d_3_1], 1))
        d_1_1 = self.decoder_2_1(torch.cat([e_2_0, d_2_1], 1))
        output_2 = self.decoder_1_1(torch.cat([e_1_0, d_1_1], 1))

        output = torch.cat([output_1, output_2], dim=1)

        return output
