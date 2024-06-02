import torch


def calculate_gradient_penalty(config, netC, real_2d, fake_2d, device):
    alpha = torch.rand((config["batch_size"], config["num_phases"], config["crop_size"], config["crop_size"]), device=device)
    interpolates = (alpha * real_2d + ((1 - alpha) * fake_2d)).requires_grad_(True)
    critic_interpolates = netC(interpolates)
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size(), device=device),
        create_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return config["lambda"] * ((gradients.norm(2, dim=1) - 1) ** 2).mean()