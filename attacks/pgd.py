import torch

def pgd_attack(model, images, labels, loss_fn,
               epsilon=8/255, alpha=2/255, steps=10):
    ori_images = images.clone().detach()
    adv_images = images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        logits = model(adv_images)
        loss = loss_fn(logits, labels)
        model.zero_grad()
        loss.backward()

        adv_images = adv_images + alpha * adv_images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, -epsilon, epsilon)
        adv_images = torch.clamp(ori_images + eta, 0, 1).detach()

    return adv_images
