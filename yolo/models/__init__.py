from .yolo import YOLO


def get_model(config):
    model_hub = {'yolo':YOLO}

    if config.model in model_hub.keys():
        model = model_hub[config.model](num_class=config.num_class, backbone_type=config.backbone_type, 
                            label_assignment_method=config.label_assignment_method, anchor_boxes=config.anchor_boxes,
                            channel_sparsity=config.channel_sparsity, p2=config.p2, downsample_rate=config.downsample_rate,
                            heads=config.heads)
    else:
        raise NotImplementedError(f"Unsupport model type: {config.model}")

    return model