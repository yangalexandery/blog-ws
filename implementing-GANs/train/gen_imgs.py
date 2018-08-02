import numpy as np
import torch

from networks import Generator, Discriminator, Downsampler

CHANNELS = [16, 16, 32, 64, 128, 256]

resume = 'exp/resume_official_train_5/epoch_44.pth'

def random_latent_vectors(batch_size, latent_dim):
    return np.random.randn(batch_size, latent_dim, 1, 1)

g = Generator(6, 32, CHANNELS).cuda()

snapshot = torch.load(args.resume)
g.load_state_dict(snapshot['model_g'])

num_imgs = 32

latents = random_latent_vectors(num_imgs, 32)
latents = Variable(torch.FloatTensor(latents).cuda())

fake_images = gs[layer].forward(latents)
