import random

try:
    from torchvision.transforms import functional as TF
    HAS_TV = True
except ImportError:
    HAS_TV = False


def _apply_brightness_contrast(img, brightness_factor, contrast_factor):
    # img: [C, H, W] float tensor in [0, 1]
    img = img * brightness_factor
    # Per-channel contrast around mean
    mean = img.mean(dim=(1, 2), keepdim=True)
    img = (img - mean) * contrast_factor + mean
    return img


def augment_stage1(global_image, global_mask, brightness=0.1, contrast=0.1, saturation=0.1, ice_prob=0.0):
    """
    Stage 1 轻量增强：ColorJitter（仅 image）+ 随机水平翻转（image+mask 同步）。
    ice_prob>0 时以该概率施加结冰模拟（增亮+略降对比度），利于识别全结冰湖泊。
    global_image [1,C,H,W], global_mask [1,1,H,W]。
    """
    if not HAS_TV:
        return global_image, global_mask
    img = global_image.clone()
    msk = global_mask.clone()
    if img.dim() == 4:
        img = img.squeeze(0)
    if msk.dim() == 4:
        msk = msk.squeeze(0)
    if img.max() > 1.0 + 1e-6:
        img = img / (img.max() + 1e-6)
    img = img.clamp(0.0, 1.0)
    brightness_factor = 1.0 + random.uniform(-brightness, brightness)
    contrast_factor = 1.0 + random.uniform(-contrast, contrast)
    saturation_factor = 1.0 + random.uniform(-saturation, saturation)
    c = img.shape[0]
    if c in (1, 3):
        img = TF.adjust_brightness(img, brightness_factor)
        img = TF.adjust_contrast(img, contrast_factor)
        img = TF.adjust_saturation(img, saturation_factor)
    else:
        # torchvision only supports 1/3 channel; fallback for multi-spectral (e.g. 4ch)
        img = _apply_brightness_contrast(img, brightness_factor, contrast_factor)
    img = img.clamp(0.0, 1.0)
    if random.random() > 0.5:
        img = TF.hflip(img)
        msk = TF.hflip(msk)
    if ice_prob > 0 and random.random() < ice_prob:
        ice_b = 1.0 + random.uniform(0.2, 0.35)
        ice_c = random.uniform(0.88, 0.94)
        if c in (1, 3):
            img = TF.adjust_brightness(img, ice_b)
            img = TF.adjust_contrast(img, ice_c)
        else:
            img = _apply_brightness_contrast(img, ice_b, ice_c)
        img = img.clamp(0.0, 1.0)
    img = img.unsqueeze(0)
    msk = msk.unsqueeze(0)
    return img, msk
