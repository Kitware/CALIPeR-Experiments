# ImageNet ResNet-50 weights

For the centroids workflow, download the official PyTorch weights into this directory:

```bash
curl --fail --location https://download.pytorch.org/models/resnet50-19c8e357.pth --output resnet50-19c8e357.pth.part
printf '%s  %s\n' '19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097' 'resnet50-19c8e357.pth.part' | sha256sum --check && mv resnet50-19c8e357.pth.part resnet50-19c8e357.pth
```

Expected size: 102,502,400 bytes. Do not use a partial or checksum-mismatched
file. This is an ImageNet backbone, not a CALIPeR experiment checkpoint; CAL
extraction uses the separate `--weight` argument. Upstream centroids commands
may set `MODEL.PRETRAIN_PATH` to the absolute location of this file.
