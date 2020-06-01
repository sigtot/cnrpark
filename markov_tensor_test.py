from markov_training_test import cost as cost_tensor
from calc_markov import cost as cost_list
import torch

pi_ft, pi_tf, sigma_ft, sigma_tf = 0.2, 0.044, 0.036, 0.0046

fake_series = [
     [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
     [1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
     [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
     [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1],
]
print(f"fake cost list: {cost_list(fake_series, pi_ft, pi_tf, sigma_ft, sigma_tf)}")
print(f"fake cost tensor: {cost_tensor(torch.tensor(fake_series), pi_ft, pi_tf, sigma_ft, sigma_tf)}")

zero_series = [[0] * 12, [0] * 12, [0] * 12, [0] * 12, [0] * 12]
print(f"zero cost list: {cost_list(zero_series, pi_ft, pi_tf, sigma_ft, sigma_tf)}")
print(f"zero cost tensor: {cost_tensor(torch.tensor(zero_series), pi_ft, pi_tf, sigma_ft, sigma_tf)}")

likely_series = [
     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
     [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
     [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
]
print(f"likely cost list: {cost_list(likely_series, pi_ft, pi_tf, sigma_ft, sigma_tf)}")
print(f"likely cost tensor: {cost_tensor(torch.tensor(likely_series), pi_ft, pi_tf, sigma_ft, sigma_tf)}")
