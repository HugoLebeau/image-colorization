import torch
from torchvision import transforms

mat_rgb2lab = torch.tensor([[0.412453, 0.357580, 0.180423],
                            [0.212671, 0.715160, 0.072169],
                            [0.019334, 0.119193, 0.950227]])
mat_lab2rgb = torch.inverse(mat_rgb2lab)

d65_2 = torch.tensor([0.95047, 1.00000, 1.08883]) # CIE Standard Illuminant D65, 2° observer

def rgb2lab(img_rgb):
    '''
    RGB to L*a*b* conversion.

    Parameters
    ----------
    img_rgb : torch.tensor, shape (..., 3, H, W)
        RGB image(s).

    Returns
    -------
    torch.tensor, shape (..., 3, H, W)
        Image(s) in the L*a*b* color space.

    '''
    img = img_rgb.detach().clone()
    # RGB > XYZ
    mask = img > 0.04045
    img[mask] = ((img[mask]+0.055)/1.055)**2.4
    img[~mask] /= 12.92
    img_xyz = torch.matmul(mat_rgb2lab, img.transpose(-3, -2).transpose(-2, -1).unsqueeze(dim=-1)).squeeze(dim=-1)
    # XYZ > L*a*b*
    img_xyz /= d65_2
    img_xyz = img_xyz.transpose(-1, -2).transpose(-2, -3)
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
    img_lab : torch.tensor, shape (..., 3, H, W)
        L*a*b* image(s).

    Returns
    -------
    img_rgb : torch.tensor, shape (..., 3, H, W)
        Image(s) in RGB.

    '''
    img = img_lab.detach().clone()
    # L*a*b* > XYZ
    Y = (img[..., 0, :, :]+16.)/116.
    X, Z = img[..., 1, :, :]/500.+Y, Y-img[..., 2, :, :]/200.
    img_xyz = torch.stack([X, Y, Z], dim=-3)
    mask = img_xyz > 0.2068966
    img_xyz[mask] = img_xyz[mask]**3
    img_xyz[~mask] = (img_xyz[~mask]-16./116.)/7.787
    img_xyz = img_xyz.transpose(-3, -2).transpose(-2, -1)
    img_xyz *= d65_2
    # XYZ > RGB
    img_rgb = torch.matmul(mat_lab2rgb, img_xyz.unsqueeze(dim=-1)).squeeze(dim=-1).transpose(-1, -2).transpose(-2, -3)
    mask = img_rgb > 0.0031308
    img_rgb[mask] = 1.055*(img_rgb[mask]**(1./2.4))-0.055
    img_rgb[~mask] *= 12.92
    return img_rgb

def rgb2l(img_rgb):
    '''
    RGB to L* convsersion.

    Parameters
    ----------
    img_rgb : torch.tensor, shape (..., 3, H, W)
        RGB image(s).

    Returns
    -------
    torch.tensor, shape (..., 1, H, W)
        Lightness (L*) image(s).

    '''
    img = img_rgb.detach().clone()
    # RGB > Y
    mask = img > 0.04045
    img[mask] = ((img[mask]+0.055)/1.055)**2.4
    img[~mask] /= 12.92
    Y = torch.matmul(mat_rgb2lab[1], img.transpose(-3, -2).transpose(-2, -1).unsqueeze(dim=-1)).squeeze(dim=-1)
    # Y > L*
    Y = Y/d65_2[1]
    mask = Y > 0.008856
    Y[mask] = Y[mask]**(1./3.)
    Y[~mask] = 7.787*Y[~mask]+16./116.
    L = 116.*Y-16.
    return L.unsqueeze(dim=-3)

def rgb2ab(img_rgb):
    '''
    RGB to a*b* conversion.

    Parameters
    ----------
    img_rgb : torch.tensor, shape (..., 3, H, W)
        RGB image(s).

    Returns
    -------
    torch.tensor, shape (..., 2, H, W)
        Image(s) in the a*b* color space.

    '''
    img = img_rgb.detach().clone()
    # RGB > XYZ
    mask = img > 0.04045
    img[mask] = ((img[mask]+0.055)/1.055)**2.4
    img[~mask] /= 12.92
    img_xyz = torch.matmul(mat_rgb2lab, img.transpose(-3, -2).transpose(-2, -1).unsqueeze(dim=-1)).squeeze(dim=-1)
    # XYZ > L*a*b*
    img_xyz /= d65_2
    img_xyz = img_xyz.transpose(-1, -2).transpose(-2, -3)
    mask = img_xyz > 0.008856
    img_xyz[mask] = img_xyz[mask]**(1./3.)
    img_xyz[~mask] = 7.787*img_xyz[~mask]+16./116.
    a = 500.*(img_xyz[..., 0, :, :]-img_xyz[..., 1, :, :])
    b = 200.*(img_xyz[..., 1, :, :]-img_xyz[..., 2, :, :])
    return torch.stack([a, b], dim=-3)

def data_transform(img_pil):
    '''
    Convert a PIL image to its torch.tensor L* and L*a*b* representation.

    Parameters
    ----------
    img_pil : PIL image
        Image to be converted.

    Returns
    -------
    torch.tensor
        L* representation.
    torch.tensor
        L*a*b* representation.

    '''
    img = transforms.ToTensor()(img_pil.convert('RGB'))
    return (rgb2l(img), rgb2lab(img))
