import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# (1x28x28 tensor input)
def get_digit_caps(model, image):
    input_ = image.unsqueeze(0).to(torch.float32)
    digit_caps, probs = model.capsnet(input_)
    return digit_caps

# takes digit_caps output and target label
def get_reconstruction(model, digit_caps, label):
    target = torch.LongTensor([label])
    reconstruction = model.reconstruction_net(digit_caps, target)
    return reconstruction[0].cpu().detach().numpy().reshape(28, 28)

# create reconstructions with perturbed digit capsule
def dimension_perturbation_reconstructions(model, digit_caps, label, dimension, dim_values):
    reconstructions = []
    for dim_value in dim_values:
        digit_caps_perturbed = digit_caps.clone()
        digit_caps_perturbed[0, label, dimension] = dim_value
        reconstruction = get_reconstruction(model, digit_caps_perturbed, label)
        reconstructions.append(reconstruction)
    return reconstructions

def get_latent_representation(model, dataloader, device, override_y=None):
    images = []
    reconstructions = []
    latent_representations = []
    labels = []
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            for tensor in x:
                # Remove the batch dimension and convert the tensor to a numpy array
                image_np = tensor.squeeze().cpu().numpy()
                # Convert numpy array to PIL image
                pil_image = Image.fromarray((image_np * 255).astype('uint8'), mode='L')
                images.append(pil_image)

            x = x.to(device)
            digit_caps, probs = model.capsnet(x)
            if override_y is None:
                recon = model.reconstruction_net(digit_caps, y)
            else:
                recon = model.reconstruction_net(digit_caps, torch.ones_like(y) * override_y)
            latent_representations.append(digit_caps)
            reconstructions.append(recon.data.view(-1, 1, 28, 28))
            labels.append(y)

    latent_representations = torch.cat(latent_representations).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    reconstructions = torch.cat(reconstructions).cpu().numpy()

    pil_reconstructions = []
    for img_np in reconstructions:
        img_np = img_np.squeeze() * 255
        pil_img = Image.fromarray(img_np.astype('uint8'), mode='L')
        pil_reconstructions.append(pil_img)

    return images, labels, latent_representations, pil_reconstructions

def caps_interpolation(t, model, img1, img2, y):
    y = torch.tensor([y])

    with torch.no_grad():
        img1 = img1.to(device)
        latent_1, _ = model.capsnet(img1)
        recon = model.reconstruction_net(latent_1, y)

        img2 = img2.to(device)
        latent_2, _ = model.capsnet(img2)
        recon_2 = model.reconstruction_net(latent_2, y)

        inter_latent = t * latent_1 + (1 - t) * latent_2
        inter_image = model.reconstruction_net(inter_latent, y)
        inter_image = inter_image.data.view(-1, 1, 28, 28).cpu()

        return inter_image

def vae_interpolation(t, model, img1, img2):
    with torch.no_grad():
        img1 = img1.to(device)
        latent_1, _ = model.encoder(img1)

        img2 = img2.to(device)
        latent_2, _ = model.encoder(img2)

        inter_latent = t * latent_1 + (1 - t) * latent_2
        inter_image = model.decoder(inter_latent)
        inter_image = inter_image.cpu()

        return inter_image

def load_image(letter, char, writing_system, font, variation, size):
    image_path = f"datasets/base_dataset/{letter}/{writing_system}/{letter}_{writing_system}_{font}-{variation}_{char}_{size}_0.0.png"
    img = load_image_as_tensor(image_path)
    return img

def generate_animation(letter, char1, char2, writing_system1, writing_system2, font1, font2, variation, size):
    img1 = load_image(letter, char1, writing_system1, font1, variation, size)
    img2 = load_image(letter, char2, writing_system2, font2, variation, size)

    letter_idx = letter_to_label[letter]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_axis_off()

    num_frames = 30

    def update(frame):
        if frame < num_frames / 2:
            t = frame / (num_frames / 2)
            t = np.sin(t * np.pi / 2)
        else:
            t = (frame - num_frames / 2) / (num_frames / 2)
            t = 1 - np.sin(t * np.pi / 2)

        inter_image = caps_interpolation(t, model, img1, img2, letter_idx)
        ax.imshow(inter_image[0, 0], cmap='gray')
        return ax

    ani = FuncAnimation(fig, update, frames=num_frames, interval=100)
    return ani

def generate_alphabet_animations(df, ws1, ws2, font1, font2, variation, size):
    animations = {}
    for _, row in df.iterrows():
        letter = row['letter']
        char1 = row['char1']
        char2 = row['char2']
        
        ani = generate_animation(letter, char1, char2, ws1, ws2, font1, font2, variation, size)
        animations[letter] = ani
    return animations
