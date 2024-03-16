import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from skimage import io, color
from utils import *
import torch.nn.functional as F
from NTIRE2022Util import *
from differential_color_functions import ciede2000_diff, rgb2lab_diff
from guided_filter import GuidedFilter
from einops import rearrange, repeat
from torch.autograd import Variable

rgbfilterpath = 'resources/RGB_Camera_QE.csv'
camera_filter, filterbands = load_rgb_filter(rgbfilterpath)
cube_bands = np.linspace(400,700,31)
index = np.linspace(40, 340, 31)
cam_filter = np.zeros([31, 3])
count = 0
for i in index:
    i = int(i)
    cam_filter[count,:] = camera_filter[i,:] 
    count += 1
cam_filter = cam_filter.astype(np.float32)
cam_filter = torch.from_numpy(cam_filter).to(device)

xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]], dtype=np.float32)

_illuminants = \
    {"A": {'2': (1.098466069456375, 1, 0.3558228003436005),
           '10': (1.111420406956693, 1, 0.3519978321919493),
           'R': (1.098466069456375, 1, 0.3558228003436005)},
     "B": {'2': (0.9909274480248003, 1, 0.8531327322886154),
           '10': (0.9917777147717607, 1, 0.8434930535866175),
           'R': (0.9909274480248003, 1, 0.8531327322886154)},
     "C": {'2': (0.980705971659919, 1, 1.1822494939271255),
           '10': (0.9728569189782166, 1, 1.1614480488951577),
           'R': (0.980705971659919, 1, 1.1822494939271255)},
     "D50": {'2': (0.9642119944211994, 1, 0.8251882845188288),
             '10': (0.9672062750333777, 1, 0.8142801513128616),
             'R': (0.9639501491621826, 1, 0.8241280285499208)},
     "D55": {'2': (0.956797052643698, 1, 0.9214805860173273),
             '10': (0.9579665682254781, 1, 0.9092525159847462),
             'R': (0.9565317453467969, 1, 0.9202554587037198)},
     "D65": {'2': (0.95047, 1., 1.08883),   # This was: `lab_ref_white`
             '10': (0.94809667673716, 1, 1.0730513595166162),
             'R': (0.9532057125493769, 1, 1.0853843816469158)},
     "D75": {'2': (0.9497220898840717, 1, 1.226393520724154),
             '10': (0.9441713925645873, 1, 1.2064272211720228),
             'R': (0.9497220898840717, 1, 1.226393520724154)},
     "E": {'2': (1.0, 1.0, 1.0),
           '10': (1.0, 1.0, 1.0),
           'R': (1.0, 1.0, 1.0)}}

def rgb2xyz(rgb):
    arr = Normalize(rgb)
    mask = arr > 0.04045
    arr[mask] = torch.pow((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] /= 12.92
    return arr.float() @ torch.from_numpy(xyz_from_rgb).to(device).T

def xyz_tristimulus_values(*, illuminant, observer, dtype=float):
    illuminant = illuminant.upper()
    observer = observer.upper()
    try:
        return torch.from_numpy(np.asarray(_illuminants[illuminant][observer], dtype=dtype)).to(device)
    except KeyError:
        raise ValueError(f'Unknown illuminant/observer combination '
                         f'(`{illuminant}`, `{observer}`)')

def odd_pow(input, exponent):
    return input.sign() * input.abs().pow(exponent)

def xyz2lab(xyz, illuminant="D65", observer="2", *, channel_axis=-1):
    arr = xyz

    xyz_ref_white = xyz_tristimulus_values(
        illuminant=illuminant, observer=observer, dtype=np.float32
    )

    # scale by CIE XYZ tristimulus values of the reference white point
    arr = arr / xyz_ref_white

    # Nonlinear distortion and linear transformation
    mask = arr > 0.008856
    arr[mask] = odd_pow(arr[mask], 1.0/3.0)
    arr[~mask] = 7.787 * arr[~mask] + 16. / 116.

    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]

    # Vector scaling
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    return torch.concat([torch.unsqueeze(L, dim=-1), torch.unsqueeze(a, dim=-1), torch.unsqueeze(b, dim=-1)], dim=-1)

def cal_deltaE_2000(lab1, lab2):
    L1 = lab1[:,:,0]
    a1 = lab1[:,:,1]
    b1 = lab1[:,:,2]
    L2 = lab2[:,:,0]
    a2 = lab2[:,:,1]
    b2 = lab2[:,:,2]
    L_p = (L1 + L2) / 2.0
    C1 = torch.sqrt(a1 ** 2 + b1 ** 2)
    C2 = torch.sqrt(a2 ** 2 + b2 ** 2)
    Cmean = (C1 + C2) / 2.0
    G = 0.5 * ( 1 - torch.sqrt(Cmean ** 7 / ( Cmean ** 7 + 25.0 ** 7)))
    a1_ = a1 * (1+G)
    a2_ = a2 * (1+G)
    C1_ = torch.sqrt(a1_ ** 2 + b1 ** 2)
    C2_ = torch.sqrt(a2_ ** 2 + b2 ** 2)
    Cmean_ = (C1_ + C2_) / 2.0
    h1_ = torch.atan2(b1, a1_) + 2 * torch.pi * (torch.atan2(b1, a1_) < 0)
    h2_ = torch.atan2(b2, a2_) + 2 * torch.pi * (torch.atan2(b2, a2_) < 0)
    H__ = (h1_ + h2_ + torch.pi * 2.0 * (torch.abs(h1_ - h2_) > torch.pi)) / 2.0
    T = 1 - 0.17 * torch.cos(H__ - torch.pi / 6.0) + 0.24 * torch.cos(2 * H__) + 0.32 * torch.cos(3 * H__ + 6.0 / 180.0 * torch.pi) - 0.2 * torch.cos(4 * H__ - 63.0/180*torch.pi)
    dh_ = torch.where(torch.abs(h1_ - h2_) <= torch.pi, h2_ - h1_, torch.where(h2_ < h1_, h2_ - h1_ + 2.0 * torch.pi, h2_ - h1_ - 2.0 * torch.pi))

    delta_L = L2 - L1
    delta_C = C2_ - C1_
    delta_h = 2.0 * torch.sqrt(C1_ * C2_) * torch.sin(dh_ / 2.0)
    Sl = 1 + 0.015 * ((L_p - 50) ** 2) / torch.sqrt(20.0 + (L_p - 50) ** 2)
    Sc = 1 + 0.045 * Cmean_
    Sh = 1 + 0.015 * Cmean_ * T
    delta_theta = 30 * torch.exp(-((H__ - 275.0 / 180.0 * torch.pi)/25.0) ** 2)
    Rc = 2 * torch.sqrt(Cmean_ ** 7 / (Cmean_ ** 7 + 25 ** 7))
    Rt = -Rc * torch.sin(2.0 * delta_theta)
    Kl = 1
    Kc = 1
    Kh = 1
    deltaE = torch.sqrt((delta_L / Kl / Sl) ** 2 + (delta_C / Kc / Sc) ** 2 + (delta_h / Kh / Sh) ** 2 + Rt + delta_C * delta_h / (Kc * Sc * Kh * Sh))
    return deltaE

def cal_deltaE_1976(lab1, lab2):
    return F.mse_loss(lab1, lab2)

def projectCube_(pixels, filters, clipNegative=False):
    """
    Project multispectral pixels to low dimension using a filter function (such as a camera response)

    :param pixels: numpy array of multispectral pixels, shape [..., num_hs_bands]
    :param filters: filter response, [num_hs_bands, num_mc_chans]
    :param clipNegative: whether to clip negative values

    :return: a numpy array of the projected pixels, shape [..., num_mc_chans]

    :raise: RuntimeError if `pixels` or `filters` are passed transposed

    """

    # assume the number of spectral channels match (will crash inside if not)
    if np.shape(pixels)[-1] != np.shape(filters)[0]:
        raise RuntimeError(f'{__file__}: projectCube - incompatible dimensions! got {np.shape(pixels)} and {np.shape(filters)}')

    filters = torch.from_numpy(filters).to(device)
    filters = filters.float()
    projected = torch.matmul(pixels, filters.detach())

    if clipNegative:
        projected = projected.clip(0, None)

    return projected

def projectHS_(cube, cube_bands, qes, qe_bands, clipNegative, interp_mode='linear'):
    """
    Project a spectral array

    :param cube: Input hyperspectral cube
    :param cube_bands: bands of hyperspectral cube
    :param qes: filter response to use for projection
    :param qe_bands: bands of filter response
    :param clipNegative: clip values below 0
    :param interp_mode: interpolation mode for missing values
    :return:
    :return: numpy array of projected data, shape [..., num_channels ]
    """
    # if not np.all(qe_bands == cube_bands):  # then sample the qes on the data bands
    dx_qes = qe_bands[1] - qe_bands[0]
    dx_hs = cube_bands[1] - cube_bands[0]
    if np.any(np.diff(qe_bands) != dx_qes) or np.any(np.diff(cube_bands) != dx_hs):
        raise ValueError(f'V81Filter.projectHS - can only interpolate from uniformly sampled bands\n'
                            f'got hs bands: {cube_bands}\n'
                            f'filter bands: {qe_bands}')

    if dx_qes < 0:
        # we assume the qe_bands are sorted ascending inside resampleHSPicked, reverse them
        qes = qes[::-1]
        qe_bands = qe_bands[::-1]

    # find the limits of the interpolation, WE DON'T WANT TO EXTRAPOLATE!
    # the limits must be defined by the data bands so the interpolated qe matches
    min_band = cube_bands[
        np.argwhere(cube_bands >= qe_bands.min()).min()]  # the first data band which has a respective qe value
    max_band = cube_bands[
        np.argwhere(cube_bands <= qe_bands.max()).max()]  # the last data band which has a respective qe value
    # TODO is there a minimal overlap we want to enforce?

    # cube = cube[..., np.logical_and(cube_bands >= min_band, cube_bands <= max_band)]
    shared_bands = make_spectral_bands(min_band, max_band,
                                        dx_hs)  # shared domain with the spectral resolution of the spectral data
    qes = resampleHSPicked(qes.T, bands=qe_bands, newBands=shared_bands, interpMode=interp_mode,
                            fill_value=np.nan).T

    return projectCube_(cube, qes, clipNegative=clipNegative)

def hsi2rgb(labels, camera_filter, filterbands):
    B, C, H, W = labels.shape
    labels = rearrange(labels, 'b c h w -> b h w c')
    cube_bands = np.linspace(400,700,31)
    # rgb = projectHS_(labels, cube_bands, camera_filter, filterbands, clipNegative=False)
    rgb = torch.matmul(labels, cam_filter)
    rgb = normalization_image(rgb)
    rgb = rearrange(rgb, 'b h w c -> b c h w')
    return rgb

def deltaELoss(gt, fake, filter = GuidedFilter(radius=5, eps = 0.1)):
    # rgb = gt
    rgb = hsi2rgb(gt, camera_filter, filterbands)
    fake_rgb = hsi2rgb(fake, camera_filter, filterbands)
    fake_rgb = filter(fake_rgb, fake_rgb)
    rgb = filter(rgb, rgb)
    loss = ciede2000_diff(rgb2lab_diff(rgb, device), rgb2lab_diff(fake_rgb, device), device)
    color_loss=loss.mean()
    return color_loss
    # color_dis=torch.norm(loss.view(1,-1),dim=1)
    # color_loss=color_dis.mean()
    # return color_loss
    # return F.mse_loss(rgb, fake_rgb)

# def deltaELoss(label, rgb, filter = GuidedFilter(radius=5, eps = 0.1)):
#     fake_rgb = hsi2rgb(label, camera_filter, filterbands)
#     rgb = Normalize(rgb)
#     # rgb = rearrange(rgb, 'b c h w -> (b h) w c')
#     # lab1 = xyz2lab(rgb2xyz(rgb))
#     # lab2 = xyz2lab(rgb2xyz(fake_rgb))
#     # color_loss = cal_deltaE_1976(lab1, lab2)
#     return F.l1_loss(fake_rgb, rgb)

if __name__ == '__main__':
    model = nn.Conv2d(31, 31, 3, 1, 1).to(device)
    # optimizer = torch.optim.RMSprop(model.parameters(), 4e-5)
    optimizer = torch.optim.Adam(model.parameters(), 4e-4)
    gradients = []
    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach().cpu().numpy())
    handle = model.register_backward_hook(save_gradients)

    guided_filter = GuidedFilter(radius=5, eps = 0.1)
    path = '/work3/s212645/Spectral_Reconstruction/clean/ARAD/006.mat'
    mat = scipy.io.loadmat(path)

    hyper = mat['cube']
    rgb = mat['rgb']
    rgb = Normalize(rgb)
    rgb = np.transpose(rgb, [2, 0, 1])
    rgb = torch.from_numpy(rgb).to(device)
    rgb = rgb.float()
    rgb = rgb.unsqueeze(dim=0)

    hyper = np.transpose(hyper, [2, 0, 1])
    hyper = torch.from_numpy(hyper).to(device)
    hyper = hyper.unsqueeze(dim=0)
    hyper = hyper.float()
    fakergb = hsi2rgb(hyper, camera_filter, filterbands)

    targets = hyper
    inputs = torch.randn([1, 31, 482, 512])

    for i in range(10):
        a = model(inputs)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss1 = F.mse_loss(outputs, targets)
        loss1.backward()

        gradients1 = [param.grad.clone() for param in model.parameters()]
        optimizer.step()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss2 = deltaELoss(outputs, targets)
        loss2.backward()

        mask_gradients = []
        for grad1, grad2 in zip(gradients1, model.parameters()):
            mask1 = torch.abs(torch.sign(grad1) + torch.sign(grad2.grad)) / 2.0
            mask2 = - torch.abs(torch.sign(grad1) - torch.sign(grad2.grad)) / 2.0
            mask = mask1 + mask2
            mask_gradients.append(grad2.grad * mask)

        for param, grad in zip(model.parameters(), mask_gradients):
            param.grad = grad

        optimizer.step()
        print(loss1.item())

    
    # loss_deltae = deltaELoss(hyper, a)
    # loss_deltae.backward(retain_graph = True)
    # direct = torch.sign(model.weight.grad)

    # loss_mse = F.mse_loss(a, hyper)
    # loss_mse.backward()
    # mse_grad = model.weight.grad.clone()
    # direct2 = torch.sign(model.weight.grad)
    # mask = torch.abs(direct + direct2) / 2.0
    # model.weight.grad *= mask
    # model.weight.grad += mse_grad
    # for i in range(200):
    #     optimizer.zero_grad()
    #     output = model(hyper)
    #     deltaELoss(output, hyper).backward()
    #     optimizer.step()
    #     print(deltaELoss(output, hyper).item())
    # loss = deltaELoss(output, rgb)
    # color_dis=torch.norm(loss.view(1,-1),dim=1)
    # color_loss=color_dis.sum()
    # loss.backward()
    # print(loss)
    # print(deltaELoss(output, hyper))
    # rgb = rgb.permute(0, 2, 3, 1)
    # rgb = rgb.reshape([512,512, 3])
    # rgb = rgb.float()
    # print(F.l1_loss(rgb, rgb_recon))