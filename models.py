from torch import nn
from utils import q, zero_padding
from torchvision.models.detection import maskrcnn_resnet50_fpn

MaskRCNN = maskrcnn_resnet50_fpn(pretrained=True)

class Zhang16(nn.Module):
    def __init__(self, q=q):
        super(Zhang16, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
            ) # à trous
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
            ) # à trous
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
            )
        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, q, kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, img_l):
        x = (img_l-50.)/100. # normalize L* input
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        z = self.softmax(self.conv8(x)) # a*b* probability distribution
        return z.transpose(-3, -2).transpose(-2, -1)

class Su20_feature_fusion(nn.Module):
    def __init__(self, c):
        super(Su20_feature_fusion, self).__init__()
        self.background_conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLu(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            )
        self.instance_conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLu(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, background, instance, box):
        background_weight = self.background_conv(background)
        instance_weight = [self.instance_conv(i) for i in instance]
        weight_map = zero_padding(background_weight, instance_weight, box)
        fused = self.softmax(background_weight)*background
        for n in range(fused.shape[0]):
            fused[n] += self.softmax(weight_map[n])*instance[n]
        return fused
