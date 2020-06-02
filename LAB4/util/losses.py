import torch


def get_vae_loss():
    bcewll = torch.nn.BCEWithLogitsLoss(reduction="none")

    def _loss(y_real, y_pred, mu, logvar, kl_beta: int = 1.):
        ce = torch.sum(bcewll(y_pred, y_real), dim=1)
        kl_div = torch.sum(torch.square(mu) + torch.exp(logvar) - logvar - 1,
                           dim=1) / 2

        return torch.mean(ce + kl_div * kl_beta)

    return _loss


def get_gan_loss():
    bcel = torch.nn.BCELoss()

    def _loss(y_pred, y_real):
        return bcel(y_pred, y_real)

    return _loss
