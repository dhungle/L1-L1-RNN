import torch

def psnr(ref, reconstructed):
    if ref.max() == 1:
        ref *= 255
        reconstructed *= 255
    reconstructed[reconstructed < 0] = 0
    reconstructed[reconstructed > 255] = 255
    reconstructed = reconstructed.int().float()
    ref = ref.int().float()
    mse = torch.mean((ref - reconstructed) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(255 / torch.sqrt(mse))
