import torch


def save_checkpoint(path,
                    model,
                    args):
    """
    Saves a model checkpoint.

    :param model: A MoleculeModel.
    :param args: Arguments namespace.
    :param path: Path where checkpoint will be saved.
    """
    state = {
        'args': args,
        'state_dict': model.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(model,
                    path,
                    cuda):
    """
    Loads a model checkpoint.

    :param model:
    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :return: The loaded MoleculeModel.
    """

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']

    model_state_dict = model.state_dict()

    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():
        pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if cuda:
        model = model.cuda()

    return model
