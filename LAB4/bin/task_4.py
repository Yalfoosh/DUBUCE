from sys import stdout
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm

import seaborn as sns

from util.losses import get_vae_loss


class VAE(torch.nn.Module):
    def __init__(self,
                 encoder_units: List[int] or Tuple[int] = (200, 200),
                 bottleneck_size: int = 20,
                 decoder_units: List[int] or Tuple[int] = (200, 200),
                 data_units: int = 784,
                 loss: Callable = None):
        super(VAE, self).__init__()

        self._bottleneck_size = bottleneck_size
        self._data_units = data_units
        self._loss = get_vae_loss() if loss is None else loss

        self._encoder = torch.nn.ModuleList()
        self._bottleneck = torch.nn.ModuleDict()
        self._decoder = torch.nn.ModuleList()

        encoder_units = (self.data_units, *encoder_units)
        decoder_units = (bottleneck_size, *decoder_units)

        for i in range(1, len(encoder_units)):
            self.encoder.append(torch.nn.Linear(encoder_units[i - 1],
                                                encoder_units[i]))

        for key in ["mu", "logvar"]:
            self.bottleneck[key] = torch.nn.Linear(encoder_units[-1],
                                                   bottleneck_size)

        for i in range(1, len(decoder_units)):
            self.decoder.append(torch.nn.Linear(decoder_units[i - 1],
                                                decoder_units[i]))

        self._reconstructor = torch.nn.Linear(decoder_units[-1],
                                              self.data_units)

        self._softplus = torch.nn.Softplus()

        self.reset_parameters()

    # region Properties
    @property
    def bottleneck_size(self):
        return self._bottleneck_size

    @property
    def data_units(self):
        return self._data_units

    @property
    def loss(self):
        return self._loss

    @property
    def encoder(self):
        return self._encoder

    @property
    def bottleneck(self):
        return self._bottleneck

    @property
    def decoder(self):
        return self._decoder

    @property
    def reconstructor(self):
        return self._reconstructor

    @property
    def softplus(self):
        return self._softplus

    # endregion

    def reset_parameters(self):
        for fc in self.encoder:
            torch.nn.init.kaiming_normal_(fc.weight, nonlinearity="relu")
            torch.nn.init.normal_(fc.bias, 0., 1e-6 / 3)

        for fc in self.bottleneck.values():
            torch.nn.init.xavier_normal_(fc.weight)
            torch.nn.init.normal_(fc.bias, 0., 1e-6 / 3)

        for fc in self.decoder:
            torch.nn.init.kaiming_normal_(fc.weight, nonlinearity="relu")
            torch.nn.init.normal_(fc.bias, 0., 1e-6 / 3)

        torch.nn.init.xavier_normal_(self.reconstructor.weight)
        torch.nn.init.constant_(self.reconstructor.bias, 0.)

    @staticmethod
    def get_z(mu: torch.Tensor,
              logvar: torch.Tensor,
              noise: torch.Tensor = None):
        if noise is None:
            noise = 1.

        return mu + torch.sqrt(torch.exp(logvar)) * noise

    def decode(self, x):
        for fc in self.decoder:
            x = fc(x)
            x = self.softplus(x)

        return self.reconstructor(x)

    def forward(self, x):
        # Flatten
        y = x.view(-1, self.data_units)

        # Encode
        for fc in self.encoder:
            y = fc(y)
            y = self.softplus(y)

        # Reparametrize
        mu = self.bottleneck["mu"](y)
        logvar = self.bottleneck["logvar"](y)
        noise = torch.normal(0., 1., size=logvar.shape, device=logvar.device)

        y = self.get_z(mu, logvar, noise)

        # Decode
        for fc in self.decoder:
            y = fc(y)
            y = self.softplus(y)

        # Reconstruct (without sigmoid because it's in the loss)
        y = self.reconstructor(y)

        return y, mu, logvar

    def fit(self,
            dataset: torch.utils.data.Dataset,
            n_epochs: int = 1,
            batch_size: int = 1,
            learning_rate: float = 3e-4,
            lr_gamma: float = 0.95,
            kl_beta_sine_multiplier: float = 0.3,
            device: str = "cpu",
            verbose: int = 1):
        self.train()
        self.to(device)

        tr_loss = list()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=lr_gamma)

        tr_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

        for epoch in range(n_epochs):
            iterator = tqdm(tr_loader, file=stdout) \
                       if verbose > 0 \
                       else tr_loader
            kl_beta = 1. \
                      if kl_beta_sine_multiplier is None \
                      else abs(np.sin(epoch * kl_beta_sine_multiplier))

            for i, (x, _) in enumerate(iterator):
                x = x.to(device)

                optimizer.zero_grad()

                y, mu, logvar = self.forward(x)
                loss = self.loss(y_real=x.view(y.shape), y_pred=y,
                                 mu=mu, logvar=logvar,
                                 kl_beta=kl_beta)
                tr_loss.append(float(loss))

                if verbose > 0:
                    iterator.set_description(f"Epoch {epoch + 1}   "
                                             f"Loss: {np.mean(tr_loss):.04f}")

                loss.backward()
                optimizer.step()

            scheduler.step()
            tr_loss.clear()

    def generate_mu_and_stddev_dataframes(self,
                                          dataset: torch.utils.data.Dataset,
                                          device: str = "cpu",
                                          verbose: int = 1):
        self.to(device)
        self.eval()

        loader = torch.utils.data.DataLoader(dataset)

        mus = list()
        logvars = list()
        labels = list()

        with torch.no_grad():
            iterator = tqdm(loader, file=stdout) if verbose > 0 else loader

            for x, y in iterator:
                _, mu, logvar = self.forward(x.view(-1, self.data_units)
                                              .to(device))

                mus.extend(mu.data.cpu().numpy())
                logvars.extend(logvar.data.cpu().numpy())
                labels.extend(y.data.cpu().numpy())

            mus = np.array(mus)
            logvars = np.array(logvars)

            to_return = list()

            for c_list, c_name in zip((mus, logvars), ("μ", "σ")):
                t_dict = {"label": labels}

                for i in range(c_list.shape[1]):
                    t_dict[f"{c_name} {i:02d}"] = c_list[..., i]

                to_return.append(pd.DataFrame(t_dict))
                to_return[-1]["label"] = \
                    to_return[-1]["label"].astype("category")

            return to_return

    def plot_io(self,
                dataset: torch.utils.data.Dataset,
                n_samples: int,
                device: str = "cpu"):
        self.to(device)
        self.eval()

        n_samples = min(n_samples, len(dataset))

        loader = torch.utils.data.DataLoader(dataset, batch_size=n_samples)
        batch, _ = next(iter(loader))

        fig, ax = plt.subplots(n_samples, 2,
                               figsize=(4, n_samples * 2))

        with torch.no_grad():
            output, mu, logvar = self.forward(batch.to(device))

            for i in range(n_samples):
                x = batch[i, ...].view(28, 28).data.cpu()
                y = torch.sigmoid(output)[i, ...].view(28, 28).cpu()

                for j, plot_subject in enumerate((x, y)):
                    ax[i][j].axis("off")
                    ax[i][j].imshow(plot_subject, vmin=-4, vmax=4)

        fig.tight_layout()

        return fig, ax

    @staticmethod
    def plot_distribution(mu_dataframe: pd.DataFrame,
                          stddev_dataframe: pd.DataFrame):
        fig, ax = plt.subplots(3, 1, figsize=(16, 30))

        sns.scatterplot(x="μ 00", y="μ 01", hue="label", s=50,
                        data=mu_dataframe, ax=ax[-1])

        mu_dataframe = mu_dataframe.melt(["label"])
        stddev_dataframe = stddev_dataframe.melt(["label"])

        for i, (df, name) in enumerate(zip((mu_dataframe, stddev_dataframe),
                                           ("μ", "σ"))):
            df.boxplot(ax=ax[i], column="value", by="variable")

            ax[i].title.set_text(f"Distribucija {name}")
            ax[i].set_xlabel("Varijabla")
            ax[i].set_ylabel("Vrijednost")

        return fig, ax

    def plot_latent_space(self,
                          n_samples: int = 20,
                          latent_space_limit: int = 3,
                          device: str = "cpu"):
        canvas = np.zeros((n_samples * 28, n_samples * 28))

        d1 = np.linspace(-latent_space_limit, latent_space_limit,
                         num=n_samples)
        d2 = np.linspace(-latent_space_limit, latent_space_limit,
                         num=n_samples)

        _d1, _d2 = np.meshgrid(d1, d2)
        synth_reps = np.array([_d1.flatten(), _d2.flatten()]).T
        synth_reps_pt = torch.from_numpy(synth_reps).float().to(device)

        recons = self.decode(synth_reps_pt)

        for idx in range(0, n_samples * n_samples):
            x, y = np.unravel_index(idx, (n_samples, n_samples))

            sample_offset = n_samples - x - 1

            first_from = 28 * sample_offset
            first_to = 28 * (sample_offset + 1)

            second_from = 28 * y
            second_to = 28 * (y + 1)

            canvas[first_from: first_to, second_from:second_to] = \
                recons[idx, ...].view(28, 28).data.cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        ax.imshow(canvas)

        return fig, ax

    def plot_generations(self,
                         dataset: torch.utils.data.Dataset,
                         n_samples: int = 4,
                         shape: Tuple[int, int] = None,
                         base_size: Tuple[int, int] = (1.6, 1.6),
                         device: str = "cpu"):
        self.eval()
        self.to(device)

        if n_samples is None or n_samples < 3:
            n_samples = 4

        if shape is None or shape[0] * shape[1] < n_samples:
            height = min(int((n_samples ** 0.5) + 1e-6), 10)
            width = (n_samples + height - 1) // height

            shape = (height, width)

        loader = torch.utils.data.DataLoader(dataset, batch_size=n_samples)
        batch, _ = next(iter(loader))

        fig, ax = plt.subplots(*shape, figsize=(base_size[0] * shape[0],
                                                base_size[1] * shape[1]))

        with torch.no_grad():
            output, _, _ = self.forward(batch.to(device))

            for i in range(n_samples):
                curr_axis = ax[i // shape[0]][i % shape[0]]
                y = torch.sigmoid(output)[i, ...].view(28, 28).cpu()

                curr_axis.axis("off")
                curr_axis.imshow(y, vmin=0, vmax=1)

        fig.tight_layout()

        return fig, ax

