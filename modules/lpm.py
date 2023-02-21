import torch
from torch import nn


class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) +
                         epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = feature_mul.view(
            b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor


class LPM(nn.Module):
    def __init__(self):
        super(LPM, self).__init__()
        # encode the reference person
        unet_1_0 = [
            nn.Conv2d(30, 64, kernel_size=4, stride=2,
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
        unet_conv = [
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.ReLU(),
            nn.InstanceNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ]

        # encode the target clothes
        encoder_1_0 = [
            nn.Conv2d(3, 64, kernel_size=4, stride=2,
                      padding=1, bias=True),
            nn.InstanceNorm2d(64)
        ]
        encoder_2_0 = [
            nn.ReLU(),
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
        encoder_conv = [
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.ReLU(),
            nn.InstanceNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ]

        # feature l2norm
        self.l2norm = FeatureL2Norm()

        # feature corrlelation matching
        self.correlation = FeatureCorrelation()

        # decode and synthesis image
        conv = [
            nn.Conv2d(192, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(512)
        ]
        unet_4_1 = [
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64 * 16, 256, kernel_size=3,
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
                      stride=1, padding=1)
        ]

        self.unet_1_0 = nn.Sequential(*unet_1_0)
        self.unet_2_0 = nn.Sequential(*unet_2_0)
        self.unet_3_0 = nn.Sequential(*unet_3_0)
        self.unet_4_0 = nn.Sequential(*unet_4_0)
        self.unet_1_1 = nn.Sequential(*unet_1_1)
        self.unet_2_1 = nn.Sequential(*unet_2_1)
        self.unet_3_1 = nn.Sequential(*unet_3_1)
        self.unet_4_1 = nn.Sequential(*unet_4_1)
        self.unet_conv = nn.Sequential(*unet_conv)
        self.encoder_4_0 = nn.Sequential(*encoder_4_0)
        self.encoder_3_0 = nn.Sequential(*encoder_3_0)
        self.encoder_2_0 = nn.Sequential(*encoder_2_0)
        self.encoder_1_0 = nn.Sequential(*encoder_1_0)
        self.encoder_conv = nn.Sequential(*encoder_conv)
        self.conv = nn.Sequential(*conv)

    def forward(self, input_1, input_2):
        x_1_0 = self.unet_1_0(input_1)
        x_2_0 = self.unet_2_0(x_1_0)
        x_3_0 = self.unet_3_0(x_2_0)
        x_4_0 = self.unet_4_0(x_3_0)
        x = self.unet_conv(x_4_0)
        x = self.l2norm(x)

        e_1_0 = self.encoder_1_0(input_2)
        e_2_0 = self.encoder_2_0(e_1_0)
        e_3_0 = self.encoder_3_0(e_2_0)
        e_4_0 = self.encoder_4_0(e_3_0)
        e = self.encoder_conv(e_4_0)
        e = self.l2norm(e)

        f = self.correlation(x, e)
        f = self.conv(f)

        x_3_1 = self.unet_4_1(torch.cat([x_4_0, f], 1))
        x_2_1 = self.unet_3_1(torch.cat([x_3_0, x_3_1], 1))
        x_1_1 = self.unet_2_1(torch.cat([x_2_0, x_2_1], 1))
        output = self.unet_1_1(torch.cat([x_1_0, x_1_1], 1))

        return output
