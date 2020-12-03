import torch
from torchvision import transforms

def rgb2lab(img_rgb):
    '''
    RGB to L*a*b* conversion.

    Parameters
    ----------
    img_rgb : torch.tensor
        (..., 3, H, W) RGB image(s).

    Returns
    -------
    torch.tensor
        (..., 3, H, W) image(s) in the L*a*b* color space.

    '''
    img = img_rgb.detach().clone()
    # RGB > XYZ
    mask = img > 0.04045
    img[mask] = ((img[mask]+0.055)/1.055)**2.4
    img[~mask] /= 12.92
    X = img[..., 0, :, :]*0.412453+img[..., 1, :, :]*0.357580+img[..., 2, :, :]*0.180423
    Y = img[..., 0, :, :]*0.212671+img[..., 1, :, :]*0.715160+img[..., 2, :, :]*0.072169
    Z = img[..., 0, :, :]*0.019334+img[..., 1, :, :]*0.119193+img[..., 2, :, :]*0.950227
    # XYZ > L*a*b*
    X, Y, Z = X/0.95047, Y/1.00000, Z/1.08883 # CIE Standard Illuminant D65, 2° observer
    img_xyz = torch.stack([X, Y, Z], dim=-3)
    mask = img_xyz > 0.008856
    img_xyz[mask] = img_xyz[mask]**(1./3.)
    img_xyz[~mask] = 7.787*img_xyz[~mask]+16./116.
    L = 116.*img_xyz[..., 1, :, :]-16.
    a = 500.*(img_xyz[..., 0, :, :]-img_xyz[..., 1, :, :])
    b = 200.*(img_xyz[..., 1, :, :]-img_xyz[..., 2, :, :])
    return torch.stack([L, a, b], dim=-3)

def lab2rgb(img_lab):
    '''
    L*a*b* to RGB conversion.

    Parameters
    ----------
    img_lab : torch.tensor
        (..., 3, H, W) L*a*b* image(s).

    Returns
    -------
    img_rgb : torch.tensor
        (..., 3, H, W) image(s) in RGB.

    '''
    img = img_lab.detach().clone()
    # L*a*b* > XYZ
    Y = (img[..., 0, :, :]+16.)/116.
    X, Z = img[..., 1, :, :]/500.+Y, Y-img[..., 2, :, :]/200.
    img_xyz = torch.stack([X, Y, Z], dim=-3)
    mask = img_xyz > 0.2068966
    img_xyz[mask] = img_xyz[mask]**3
    img_xyz[~mask] = (img_xyz[~mask]-16./116.)/7.787
    # CIE Standard Illuminant D65, 2° observer
    img_xyz[..., 0, :, :] *= 0.95047
    img_xyz[..., 1, :, :] *= 1.00000
    img_xyz[..., 2, :, :] *= 1.08883
    # XYZ > RGB
    R = img_xyz[..., 0, :, :]*3.240481+img_xyz[..., 1, :, :]*-1.537152+img_xyz[..., 2, :, :]*-0.498536
    G = img_xyz[..., 0, :, :]*-0.969255+img_xyz[..., 1, :, :]*1.87599+img_xyz[..., 2, :, :]*0.041556
    B = img_xyz[..., 0, :, :]*0.055647+img_xyz[..., 1, :, :]*-0.204041+img_xyz[..., 2, :, :]*1.057311
    img_rgb = torch.stack([R, G, B], dim=-3)
    mask = img_rgb > 0.0031308
    img_rgb[mask] = 1.055*(img_rgb[mask]**(1./2.4))-0.055
    img_rgb[~mask] *= 12.92
    return img_rgb

def rgb2l(img_rgb):
    '''
    RGB to L* convsersion.

    Parameters
    ----------
    img_rgb : torch.tensor
        (..., 3, H, W) RGB image(s).

    Returns
    -------
    torch.tensor
        (..., 1, H, W) lightness (L*) image(s).

    '''
    img = img_rgb.detach().clone()
    # RGB > Y
    mask = img > 0.04045
    img[mask] = ((img[mask]+0.055)/1.055)**2.4
    img[~mask] /= 12.92
    Y = img[..., 0, :, :]*0.212671+img[..., 1, :, :]*0.715160+img[..., 2, :, :]*0.072169
    # Y > L*
    Y = Y/1.00000 # CIE Standard Illuminant D65, 2° observer
    mask = Y > 0.008856
    Y[mask] = Y[mask]**(1./3.)
    Y[~mask] = 7.787*Y[~mask]+16./116.
    L = 116.*Y-16.
    return L.unsqueeze(dim=-3)

def data_transform(img_pil):
    img = transforms.ToTensor()(img_pil.convert('RGB'))
    return (rgb2l(img), rgb2lab(img))
