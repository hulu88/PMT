from model.vision_transformer import ViT
import torch
import torch.nn as nn

# L2 norm
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class build_vision_transformer(nn.Module):
    def __init__(self, num_classes, cfg):
        super(build_vision_transformer, self).__init__()
        self.in_planes = 768

        self.base = ViT(img_size=[cfg.H,cfg.W],
                        stride_size=cfg.STRIDE_SIZE,
                        drop_path_rate=cfg.DROP_PATH,
                        drop_rate=cfg.DROP_OUT,
                        attn_drop_rate=cfg.ATT_DROP_RATE)

        self.base.load_param(cfg.PRETRAIN_PATH)

        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)


    def forward(self, x):
        features = self.base(x)
        feat = self.bottleneck(features)

        if self.training:
            cls_score = self.classifier(feat)

            return cls_score, features

        else:
            return self.l2norm(feat)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))