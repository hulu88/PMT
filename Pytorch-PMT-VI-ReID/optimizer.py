import torch

def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr = cfg.BASE_LR
        weight_decay = cfg.WEIGHT_DECAY

        if "bias" in key:
            lr = cfg.BASE_LR * cfg.BIAS_LR_FACTOR
            weight_decay = cfg.WEIGHT_DECAY_BIAS

        if "base.patch_embed.proj" in key:
            params +=[{"params": [value], "lr": lr * cfg.LR_PRETRAIN, "weight_decay": weight_decay}]
            continue

        if "base.blocks." in key:
            params +=[{"params": [value], "lr": lr * cfg.LR_PRETRAIN, "weight_decay": weight_decay}]
            continue

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.OPTIMIZER_NAME)(params, momentum=cfg.MOMENTUM)

    elif cfg.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.BASE_LR, weight_decay=cfg.WEIGHT_DECAY)

    else:
        optimizer = getattr(torch.optim, cfg.OPTIMIZER_NAME)(params)

    return optimizer