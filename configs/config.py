class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

vae_config = dotdict({
    'latent_dims': 2,
    'capacity': 64,
    'variational_beta': 1.0
})

capsnet_config = dotdict({
    'cnn_in_channels': 1,
    'cnn_out_channels': 256,
    'cnn_kernel_size': 9,
    'pc_num_capsules': 8,
    'pc_in_channels': 256,
    'pc_out_channels': 32,
    'pc_kernel_size': 9,
    'pc_num_routes': 32 * 6 * 6,
    'dc_num_capsules': 13, # 13 writing systems
    'dc_num_routes': 32 * 6 * 6,
    'dc_in_channels': 8,
    'dc_out_channels': 16,
    'input_width': 28,
    'input_height': 28,
    'reconstruction_coeff': 100
})