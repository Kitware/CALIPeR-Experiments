"""Load inference weights and check that the checkpoint covers the backbone."""


def load_inference_weights(model, filename, unused=()):
    import torch
    saved = torch.load(filename, map_location='cpu', weights_only=True)
    for key in ('state_dict', 'model', 'model_state_dict'):
        if isinstance(saved, dict) and isinstance(saved.get(key), dict):
            saved = saved[key]
    if not isinstance(saved, dict):
        raise ValueError('checkpoint must contain a tensor state dictionary')
    state = {k.removeprefix('module.') if hasattr(k, 'removeprefix') else
             (k[7:] if k.startswith('module.') else k): v for k, v in saved.items()}
    own = model.state_dict()
    def ignored(key):
        return any(key == prefix or key.startswith(prefix + '.') or key.startswith(prefix + '_')
                   for prefix in unused)
    required = {k for k in own if not ignored(k)}
    missing = sorted(required - set(state))
    unexpected = sorted(k for k in state if k not in own and not ignored(k))
    mismatched = sorted(k for k in required & set(state)
                        if not isinstance(state[k], torch.Tensor) or state[k].shape != own[k].shape)
    if not required or missing or unexpected or mismatched:
        raise ValueError('incomplete inference backbone: missing={} unexpected={} mismatched={}'.format(
            missing[:12], unexpected[:12], mismatched[:12]))
    # Validate the entire state before changing a single model tensor.
    with torch.no_grad():
        for key in required:
            own[key].copy_(state[key])
    print('loaded {} inference tensors from {}'.format(len(required), filename))
    return len(required)
