import torch
from torch import nn
from utils import q, zero_padding, extract, collage
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn

maskRCNN = maskrcnn_resnet50_fpn(pretrained=True)
if torch.cuda.is_available():
    maskRCNN.cuda()
for param in maskRCNN.parameters():
    param.requires_grad = False
maskRCNN.eval()

def model_init(model, mode='xavier'):
    '''
    Initialise the weights of a network with Xavier/constant initialisation
    for convolutional and linear layers. Batch norm layers are initialised with
    a normal distribution. All biases are set to 0.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network.
    mode : str, optional
        Type of initialisation (Xavier / constant). The default is 'xavier'.

    Raises
    ------
    NameError
        If the mode is not recognised

    Returns
    -------
    None.

    '''
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if mode == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=0.02)
            elif mode == 'constant':
                nn.init.constant_(m.weight.data, 1e-2)
            else:
                raise NameError(mode)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1., 0.02)
            nn.init.constant_(m.bias.data, 0.)
    model.apply(init_func)

class Zhang16(nn.Module):
    def __init__(self, q=q, weights=None, fine_tune=False, freeze=False, init_weights=True):
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
            nn.ReLU(inplace=True)
            )
        self.conv_out = nn.Conv2d(256, q, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)
        if init_weights:
            model_init(self)
        if weights: # allows incomplete state dict
            new_weights = self.state_dict()
            new_weights.update(torch.load(weights))
            self.load_state_dict(new_weights)
        if fine_tune:
            for param in self.parameters():
                param.requires_grad = False
            for param in self.conv_out.parameters():
                param.requires_grad = True
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, img_l):
        x = (img_l-50.)/100. # normalize L* input
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        z = self.softmax(self.conv_out(x)) # a*b* probability distribution
        return z.transpose(-3, -2).transpose(-2, -1)

class Su20Fusion(nn.Module):
    def __init__(self, c, weights=None, init_weights=True):
        super(Su20Fusion, self).__init__()
        self.background_conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            )
        self.instance_conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            )
        self.softmax = nn.Softmax(dim=1)
        if init_weights:
            model_init(self)
        if weights: # allows incomplete state dict
            new_weights = self.state_dict()
            new_weights.update(torch.load(weights))
            self.load_state_dict(new_weights)

    def forward(self, background, instance, box):
        background_weight = self.background_conv(background)
        instance_weight = [self.instance_conv(i) if i is not None else None for i in instance]
        instance_padded, weight_map = zero_padding(background, instance, instance_weight, box)
        fused = self.softmax(background_weight)*background
        for n in range(fused.shape[0]):
            if instance_padded[n] is None:
                fused[n] = background[n]
            else:
                fused[n] += torch.sum(self.softmax(weight_map[n])*instance_padded[n], dim=0)
        return fused

class Su20Zhang16Instance(nn.Module):
    def __init__(self, q=q, return_features=False, min_score=0.5, max_box=8, weights=None, fine_tune=False, freeze=False, init_weights=True):
        super(Su20Zhang16Instance, self).__init__()
        self.return_features = return_features
        self.min_score = min_score
        self.max_box = max_box
        self.resize = transforms.Resize((256, 256))
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
            nn.ReLU(inplace=True)
            )
        self.conv_out = nn.Conv2d(256, q, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)
        if init_weights:
            model_init(self)
        if weights: # allows incomplete state dict
            new_weights = self.state_dict()
            new_weights.update(torch.load(weights))
            self.load_state_dict(new_weights)
        if fine_tune:
            for param in self.parameters():
                param.requires_grad = False
            for param in self.conv_out.parameters():
                param.requires_grad = True
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, img_l):
        x = img_l/100. # normalize L* input (1/2)
        mask = maskRCNN(x)
        box = [m['boxes'][m['scores'] > self.min_score] for m in mask]
        box = [b[(b[:, 2]-b[:, 0] >= 8.) & (b[:, 3]-b[:, 1] >= 8.)][:self.max_box].round().int() for b in box] # keep only boxes that are at least 8x8
        x -= 0.5 # normalize L* input (2/2)
        instance = extract(x, box, self.resize)
        feature = [list() for _ in range(8)]
        color = list()
        for x0 in instance:
            if x0 is None:
                x1, x2, x3, x4, x5, x6, x7, x8 = None, None, None, None, None, None, None, None
                z = None
            else:
                x1 = self.conv1(x0)
                x2 = self.conv2(x1)
                x3 = self.conv3(x2)
                x4 = self.conv4(x3)
                x5 = self.conv5(x4)
                x6 = self.conv6(x5)
                x7 = self.conv7(x6)
                x8 = self.conv8(x7)
                z = self.softmax(self.conv_out(x8)) # a*b* probability distribution
                z = z.transpose(-3, -2).transpose(-2, -1)
            feature[0].append(x1)
            feature[1].append(x2)
            feature[2].append(x3)
            feature[3].append(x4)
            feature[4].append(x5)
            feature[5].append(x6)
            feature[6].append(x7)
            feature[7].append(x8)
            color.append(z)
        if self.return_features:
            return feature, box
        else:
            return color, box

class Su20Zhang16Background(nn.Module):
    def __init__(self, q=q, weights=None, freeze=False, init_weights=True):
        super(Su20Zhang16Background, self).__init__()
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
            nn.ReLU(inplace=True)
            )
        self.conv_out = nn.Conv2d(256, q, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        if init_weights:
            model_init(self)
        self.fusion1 = Su20Fusion(64, init_weights=init_weights)
        self.fusion2 = Su20Fusion(128, init_weights=init_weights)
        self.fusion3 = Su20Fusion(256, init_weights=init_weights)
        self.fusion4 = Su20Fusion(512, init_weights=init_weights)
        self.fusion5 = Su20Fusion(512, init_weights=init_weights)
        self.fusion6 = Su20Fusion(512, init_weights=init_weights)
        self.fusion7 = Su20Fusion(512, init_weights=init_weights)
        self.fusion8 = Su20Fusion(256, init_weights=init_weights)
        if weights: # allows incomplete state dict
            new_weights = self.state_dict()
            new_weights.update(torch.load(weights))
            self.load_state_dict(new_weights)
    
    def forward(self, img_l, feature, box):
        x = (img_l-50.)/100. # normalize L* input
        box2 = [b//2 for b in box]
        box4 = [b//4 for b in box]
        box8 = [b//8 for b in box]
        # x = self.conv1(x)
        x = self.fusion1(self.conv1(x), feature[0], box2)
        # x = self.conv2(x)
        x = self.fusion2(self.conv2(x), feature[1], box4)
        # x = self.conv3(x)
        x = self.fusion3(self.conv3(x), feature[2], box8)
        # x = self.conv4(x)
        x = self.fusion4(self.conv4(x), feature[3], box8)
        # x = self.conv5(x)
        x = self.fusion5(self.conv5(x), feature[4], box8)
        # x = self.conv6(x)
        x = self.fusion6(self.conv6(x), feature[5], box8)
        # x = self.conv7(x)
        x = self.fusion7(self.conv7(x), feature[6], box8)
        # x = self.conv8(x)
        x = self.fusion8(self.conv8(x), feature[7], box4)
        z = self.softmax(self.conv_out(x)) # a*b* probability distribution
        return z.transpose(-3, -2).transpose(-2, -1)

class Su20(nn.Module):
    def __init__(self, q=q, weights=None, init_weights=True):
        super(Su20, self).__init__()
        self.instance_colorization = Su20Zhang16Instance(q=q, return_features=True, freeze=True, init_weights=False)
        self.background_colorization = Su20Zhang16Background(q=q, freeze=True, init_weights=False)
        if init_weights:
            model_init(self)
        if weights: # allows incomplete state dict
            new_weights = self.state_dict()
            new_weights.update(torch.load(weights))
            self.load_state_dict(new_weights)
    
    def forward(self, img_l):
        feature, box = self.instance_colorization(img_l)
        z = self.background_colorization(img_l, feature, box)
        return z

class Collage(nn.Module):
    def __init__(self, q=q, weights=None, init_weights=True):
        super(Collage, self).__init__()
        self.instance_colorization = Su20Zhang16Instance(q=q, return_features=False, freeze=True, init_weights=False)
        self.background_colorization = Zhang16(q=q, freeze=True, init_weights=False)
        self.resize = transforms.Resize((256, 256))
        if init_weights:
            model_init(self)
        if weights: # allows incomplete state dict
            new_weights = self.state_dict()
            new_weights.update(torch.load(weights))
            self.load_state_dict(new_weights)
    
    def forward(self, img_l):
        z_instance, box = self.instance_colorization(img_l)
        z_background = self.background_colorization(img_l)
        return collage(z_background.cpu(), [z.cpu() if z is not None else None for z in z_instance], [b.cpu() for b in box], self.resize).to(img_l.device)
