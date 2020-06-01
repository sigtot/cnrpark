#!/usr/bin/env python
from statistics import NormalDist
from typing import List, Tuple

import torch
from torchvision import transforms
from torchvision.models import AlexNet
import torchvision.models

from dataset import CNRParkDataset
from malexnet_torch import mAlexNet
import torch.optim as optim
import torch.nn as nn
import numpy as np


def cost(time_series_list: torch.tensor, pi_ft: float, pi_tf: float, sigma_ft: float, sigma_tf: float) -> torch.tensor:
    pi_ft_hats, pi_tf_hats = calc_pi_list(time_series_list)
    ovl_ft = calc_ovl(pi_ft_hats, pi_ft, sigma_ft)
    ovl_tf = calc_ovl(pi_tf_hats, pi_tf, sigma_tf)
    return ovl_ft + ovl_tf


def calc_ovl(pi_list: torch.tensor, pi: float, sigma: float, ) -> torch.tensor:
    tol = 10e-7
    if len(pi_list) == 0:
        return 0
    pi_hat = torch.mean(pi_list)
    sigma_hat = torch.var(pi_list)
    if sigma_hat > tol:
        return kl_div_gaussian(pi_hat, sigma_hat, pi, sigma)
    else:
        return torch.tensor(100, dtype=torch.float32, requires_grad=True)


def kl_div_gaussian(mu_0: torch.tensor, sigma_0: torch.tensor, mu_1: torch.tensor, sigma_1: torch.tensor) -> torch.tensor:
    return 0.5*(sigma_0 / sigma_1 + ((mu_1 - mu_0)**2)/sigma_1 - 1 + torch.log(sigma_1/sigma_0))


def calc_pi_list(time_series_list: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    num_series = time_series_list.shape[0]
    pi_fts = np.nan*torch.ones(num_series, requires_grad=True, dtype=torch.float32)
    pi_tfs = np.nan*torch.ones(num_series, requires_grad=True, dtype=torch.float32)

    for i, time_series in enumerate(time_series_list):
        pi_ft, pi_tf = calc_pi(time_series)
        pi_fts[i] = pi_ft
        pi_tfs[i] = pi_tf
    return pi_fts, pi_tfs


def calc_pi(time_series: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    n_ff, n_ft, n_tf, n_tt = 0, 0, 0, 0
    for i in range(time_series.shape[0] - 1):
        v_0 = time_series[i]
        v_1 = time_series[i+1]
        if (v_0, v_1) == (0, 0):
            n_ff += 1
        elif (v_0, v_1) == (0, 1):
            n_ft += 1
        elif (v_0, v_1) == (1, 0):
            n_tf += 1
        elif (v_0, v_1) == (1, 1):
            n_tt += 1
    try:
        pi_ft = n_ft / (n_ft + n_ff)
    except ZeroDivisionError:
        pi_ft = torch.tensor(0)  # TODO: make this undefined behavior
    try:
        pi_tf = n_tf / (n_tf + n_tt)
    except ZeroDivisionError:
        pi_tf = torch.tensor(0)
    return pi_ft, pi_tf


def main():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CNRParkDataset("./PATCHES", "./LABELS", transform=preprocess)
    model = torchvision.models.alexnet(pretrained=True)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 10
    batch_size = 5
    for e in range(epochs):
        for i in range(0, len(dataset), batch_size):
            outputs = []
            while len(outputs) < batch_size:
                sample = dataset[i]
                images = sample["images"]
                images_unsqueezed = [image.unsqueeze(dim=0) for image in images]
                image_batch = torch.cat(images_unsqueezed)
                output = model.forward(image_batch).softmax(dim=1)
                output_int = output.max(dim=1)[1].float()
                all_one = all([o == 1 for o in output_int])
                all_zero = all([o == 0 for o in output_int])
                if not all_one and not all_zero:
                    outputs.append(output_int.unsqueeze(dim=0))
            outputs_tensor = torch.cat(outputs)
            print("output shape: ", outputs_tensor.shape)
            c = cost(outputs_tensor, 0.2, 0.044, 0.036, 0.0046)
            print("cost: ", c)
            loss = criterion(c, torch.tensor(0, dtype=torch.float32))
            print(f"{e}.{i}: {''.join([str(o) for o in outputs])}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #c = cost(outputs_int, 0.044, 0.2, 0.036, 0.0046)
            #print(f"cost: {c}")


if __name__ == "__main__":
    main()
