from copy import deepcopy
from sys import stdout
from time import sleep
from typing import Callable, List, Tuple

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from util.losses import get_gan_loss


class Discriminator(torch.nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 channels: List[int] or Tuple[int] = (64, 128, 256, 512, 1),
                 kernels: List[int] or Tuple[int] = (4, 4, 4, 4, 4),
                 strides: List[int] or Tuple[int] = (2, 2, 2, 2, 1),
                 padding: List[int] or Tuple[int] = (1, 1, 1, 1, 0),
                 leaky_relu_slope: float = 0.2,
                 use_batch_norm: bool = True):
        super().__init__()

        self._conv = torch.nn.ModuleList()
        self._batch_norm = torch.nn.ModuleList() if use_batch_norm else None
        self._leaky_relu = torch.nn.LeakyReLU(leaky_relu_slope)

        last_out = in_channels

        for i, (chan, kern, stri, padd) in enumerate(zip(channels,
                                                         kernels,
                                                         strides,
                                                         padding)):
            self.conv.append(torch.nn.Conv2d(in_channels=last_out,
                                             out_channels=chan,
                                             kernel_size=kern,
                                             stride=stri,
                                             padding=padd))

            if use_batch_norm and i != 0 and i != (len(channels) - 1):
                self.batch_norm.append(torch.nn.BatchNorm2d(num_features=chan))

            last_out = chan

        self.reset_parameters()

    # region Properties
    @property
    def conv(self) -> List[torch.nn.Conv2d]:
        return self._conv

    @property
    def batch_norm(self) -> List[torch.nn.BatchNorm2d]:
        return self._batch_norm

    @property
    def leaky_relu(self) -> Callable:
        return self._leaky_relu

    # endregion

    def reset_parameters(self):
        for conv in self.conv[:-1]:
            torch.nn.init.kaiming_normal_(conv.weight,
                                          nonlinearity="leaky_relu")
            torch.nn.init.normal_(conv.bias, 0, 1e-6 / 3)

        torch.nn.init.xavier_normal_(self.conv[-1].weight)
        torch.nn.init.constant_(self.conv[-1].bias, 0.)

    def forward(self, x):
        y = self.conv[0](x)
        y = self.leaky_relu(y)

        for i, conv in enumerate(self.conv[1:-1]):
            y = conv(y)
            y = self.leaky_relu(y)

            if self.batch_norm is not None:
                y = self.batch_norm[i](y)

        y = self.conv[-1](y)
        y = y.view(-1)

        return torch.sigmoid(y)


class Generator(torch.nn.Module):
    def __init__(self,
                 input_size: int = 100,
                 channels: List[int] or Tuple[int] = (512, 256, 128, 64, 1),
                 kernels: List[int] or Tuple[int] = (4, 4, 4, 4, 4),
                 strides: List[int] or Tuple[int] = (1, 2, 2, 2, 2),
                 padding: List[int] or Tuple[int] = (0, 1, 1, 1, 1),
                 leaky_relu_slope: float = 0.2,
                 use_batch_norm: bool = True):
        super().__init__()

        self._input_size = input_size
        self._conv = torch.nn.ModuleList()
        self._batch_norm = torch.nn.ModuleList() if use_batch_norm else None
        self._leaky_relu = torch.nn.LeakyReLU(leaky_relu_slope)

        last_out = input_size

        for i, (chan, kern, stri, padd) in enumerate(zip(channels,
                                                         kernels,
                                                         strides,
                                                         padding)):
            self.conv.append(torch.nn.ConvTranspose2d(in_channels=last_out,
                                                      out_channels=chan,
                                                      kernel_size=kern,
                                                      stride=stri,
                                                      padding=padd))

            if use_batch_norm and i != (len(channels) - 1):
                self.batch_norm.append(torch.nn.BatchNorm2d(num_features=chan))

            last_out = chan

        self.reset_parameters()

    # region Properties
    @property
    def input_size(self):
        return self._input_size

    @property
    def conv(self) -> List[torch.nn.Conv2d]:
        return self._conv

    @property
    def batch_norm(self) -> List[torch.nn.BatchNorm2d]:
        return self._batch_norm

    @property
    def leaky_relu(self) -> Callable:
        return self._leaky_relu

    # endregion

    def reset_parameters(self):
        for conv in self.conv[:-1]:
            torch.nn.init.kaiming_normal_(conv.weight,
                                          nonlinearity="leaky_relu")
            torch.nn.init.normal_(conv.bias, 0, 1e-6 / 3)

        torch.nn.init.xavier_normal_(self.conv[-1].weight)
        torch.nn.init.constant_(self.conv[-1].bias, 0.)

    def forward(self, x):
        for i in range(len(self.conv) - 1):
            x = self.conv[i](x)
            x = self.leaky_relu(x)

            if self.batch_norm is not None:
                x = self.batch_norm[i](x)

        x = self.conv[-1](x)

        return torch.tanh(x)


class DCGAN(torch.nn.Module):
    def __init__(self,
                 discriminator: Discriminator,
                 generator: Generator,
                 loss: Callable = None):
        super().__init__()

        self._discriminator = deepcopy(discriminator)
        self._generator = deepcopy(generator)
        self._loss = get_gan_loss() if loss is None else loss

        self._component_names = ("discriminator", "generator")

    # region Properties
    @property
    def discriminator(self):
        return self._discriminator

    @property
    def generator(self):
        return self._generator

    @property
    def loss(self):
        return self._loss

    @property
    def component_names(self):
        return self._component_names

    # endregion

    def fit(self,
            dataset: torch.utils.data.Dataset,
            n_epochs: int = 1,
            batch_size: int = 1,
            learning_rate: float or Tuple[float, float] or List[float] = 3e-4,
            lr_gamma: float or Tuple[float, float] or List[float] = 0.95,
            device: str = "cpu",
            discriminator_batches_till_step: int = 1,
            generator_batches_till_step: int = 1,
            n_epochs_till_chill: int = None,
            n_seconds_to_chill: int = 90,
            verbose: int = 1):
        self.train()
        self.to(device)

        if isinstance(learning_rate, int) or isinstance(learning_rate, float):
            learning_rate = tuple([learning_rate] * 2)

        if isinstance(lr_gamma, int) or isinstance(lr_gamma, float):
            lr_gamma = tuple([lr_gamma] * 2)

        loss = dict()
        losses = dict()
        optimizer = dict()
        scheduler = dict()

        for key, component, lr, gamma in zip(self.component_names,
                                             [self.discriminator,
                                              self.generator],
                                             learning_rate,
                                             lr_gamma):
            loss[key] = self.loss
            losses[key] = list()
            optimizer[key] = torch.optim.Adam(component.parameters(),
                                              lr=lr)
            scheduler[key] = torch.optim \
                                  .lr_scheduler \
                                  .ExponentialLR(optimizer[key], gamma=gamma)

        tr_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

        for epoch in range(n_epochs):
            iterator = tqdm(tr_loader, file=stdout)\
                       if verbose > 0\
                       else tr_loader

            for i, (x, _) in enumerate(iterator):
                curr_batch_size = x.shape[0]

                noise = torch.randn((curr_batch_size, self.generator.input_size,
                                    1, 1),
                                    device=device)

                x_real = x.to(device)
                x_fake = self.generator.forward(noise)

                y_real = torch.ones(curr_batch_size, device=device).float()
                y_fake = torch.zeros(curr_batch_size, device=device).float()
                y_dis_real = self.discriminator.forward(x_real)
                y_dis_fake = self.discriminator.forward(x_fake.detach())

                loss_dis = loss[self.component_names[0]](y_dis_real, y_real) +\
                           loss[self.component_names[0]](y_dis_fake, y_fake)
                loss_dis.backward()
                losses[self.component_names[0]].append(float(loss_dis))

                if (i + 1) % discriminator_batches_till_step == 0:
                    optimizer[self.component_names[0]].step()

                # --------------------------------------------------------------

                y_dis_fake = self.discriminator.forward(x_fake)

                loss_gen = loss[self.component_names[1]](y_dis_fake, y_real)
                loss_gen.backward()
                losses[self.component_names[1]].append(float(loss_gen))

                if (i + 1) % generator_batches_till_step == 0:
                    optimizer[self.component_names[1]].step()

                if verbose > 0:
                    iterator.set_description(
                        f"Epoch {epoch + 1}    "
                        f"DisLoss: "
                        f"{np.mean(losses[self.component_names[0]]):.04f}  "
                        f"GenLoss: "
                        f"{np.mean(losses[self.component_names[1]]):.04f}")

                for component_name in self.component_names:
                    optimizer[component_name].zero_grad()

            if n_epochs_till_chill is not None \
                    and (epoch + 1) % n_epochs_till_chill == 0:
                if verbose > 0:
                    print(f"\nChilling for {n_seconds_to_chill} seconds to "
                          f"prevent overheating...\n")
                sleep(n_seconds_to_chill)

            for component_name in self.component_names:
                scheduler[component_name].step()
                losses[component_name].clear()

    def plot_generations(self,
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

        fig, ax = plt.subplots(*shape, figsize=(base_size[0] * shape[0],
                                                base_size[1] * shape[1]))

        with torch.no_grad():
            samples = self.generator.forward(
                torch.randn(n_samples, 100, 1, 1, device=device))\
                     .view(n_samples, 64, 64)\
                     .data\
                     .cpu()\
                     .numpy()

            for i in range(n_samples):
                curr_axis = ax[i // shape[0]][i % shape[0]]

                curr_axis.axis("off")
                curr_axis.imshow(samples[i], vmin=0, vmax=1)

        return fig, ax
