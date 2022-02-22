import torch

from network import MetaNet
from sparse_net import SparseNetwork


if __name__ == '__main__':
    dense_metanet = MetaNet(30, depth=5, width=6, out=10, residual_skip=True,
                            weight_hidden_size=3, )
    sparse_metanet = SparseNetwork(30, depth=5, width=6, out=10, residual_skip=True,
                                   weight_hidden_size=3,)

    particles = [torch.cat([x.view(-1) for x in x.parameters()]) for x in dense_metanet.particles]

    # Transfer weights
    sparse_metanet = sparse_metanet.replace_weights_by_particles(dense_metanet.particles)

    # Transfer weights
    dense_metanet = dense_metanet.replace_particles(sparse_metanet.particle_weights)
    new_particles = [torch.cat([x.view(-1) for x in x.parameters()]) for x in dense_metanet.particles]

    print(f' Particles are same: {all([(x==y).all() for x,y in zip(particles, new_particles) ])}')
