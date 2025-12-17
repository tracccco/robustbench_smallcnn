import torch

def fgsm_attack(model, images, labels, loss_fn, epsilon):
    images = images.clone().detach()
    images.requires_grad = True

    logits = model(images)
    loss = loss_fn(logits, labels)
    model.zero_grad()
    loss.backward()

    adv_images = images + epsilon * images.grad.sign()
    adv_images = torch.clamp(adv_images, 0, 1)
    return adv_images.detach()
