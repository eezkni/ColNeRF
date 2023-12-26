from .models import ColNeRFNet

def make_model(conf, *args, **kwargs):
    """ Placeholder to allow more model types """
    model_type = conf.get_string("type", "colnerf")  # single
    if model_type == "colnerf":
        net = ColNeRFNet(conf, *args, **kwargs)
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    return net
