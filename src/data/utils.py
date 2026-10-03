import torch


MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def denormalize_image(image):
    """
    Convert a normalized image tensor back to displayable RGB values.
    """

    mean = MEAN[:, None, None]
    std = STD[:, None, None]

    image = image * std + mean

    return torch.clamp(image, 0, 1)