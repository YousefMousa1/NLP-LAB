import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

unloader = transforms.ToPILImage()


def image_loader(image_path):
    image = Image.open(image_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def save_tensor_as_image(tensor, path):
    image = tensor.clone().cpu().squeeze(0)
    image = unloader(image)
    image.save(path)


def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # content
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())


def style_transfer(content_img, style_img, steps=20, style_weight=1e6, content_weight=1):
    vgg = models.vgg19(pretrained=True).features.to(device).eval()

    content_features = get_features(content_img, vgg)
    style_features = get_features(style_img, vgg)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    target = content_img.clone().requires_grad_(True).to(device)
    optimizer = torch.optim.Adam([target], lr=0.003)

    for i in range(steps):
        target_features = get_features(target, vgg)
        content_loss = F.mse_loss(target_features['conv4_2'], content_features['conv4_2'])

        style_loss = 0
        for layer in style_grams:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            _, d, h, w = target_feature.shape
            style_loss += F.mse_loss(target_gram, style_gram) / (d * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

    return target